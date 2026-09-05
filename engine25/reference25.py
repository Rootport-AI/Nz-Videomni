"""IC-LoRA reference-video conditioning for LTX 2.5 (§3-102 third stage).

An IC-LoRA (In-Context LoRA) does not work from its weights alone. It was
trained on a sequence where the reference material's tokens sit alongside the
target's, and inference has to reproduce that layout: decode the reference,
VAE-encode it, and APPEND the resulting tokens to the latent sequence as clean
latents. This module is the whole of that second half. The weights half lives in
:mod:`engine25.gguf_transformer` (``Ltx25DiffusionStage.set_loras``).

Three callers, one set of rules
-------------------------------
* **Single** (:mod:`engine25.pipeline25`) installs :func:`reference_patch` around
  ``DistilledPipeline.__call__``. The official pipeline builds its conditioning
  list through ``combined_image_conditionings``, a MODULE GLOBAL of
  ``ltx_pipelines.distilled`` that it looks up once per stage, so rebinding that
  name is what lets a reference be appended without forking the pipeline. The
  same seam and the same stage discriminator LTX 2.3 uses
  (``engine/pipeline/fast_video_pipeline.py:1526-1545``); ``ltxcore_compat.verify``
  pins both.
* **Chained** (:mod:`engine25.chain25`) has no ``DistilledPipeline.__call__`` to
  patch -- it drives the blocks itself -- so it calls
  :func:`reference_conditioning_from_pixels` directly, once per stage-1 segment,
  with that segment's window of one long reference (:func:`iter_reference_windows`).
* Both resolve the reference's decode size through :func:`reference_pixel_dims`
  and its downscale factor through :func:`resolve_reference_downscale_factor`, so
  the two cannot drift apart.

STAGE 1 ONLY, in both. Stage 2 never gets a reference: 2.3 does not add one
either, and experiment 2C measured that stage-1 injection alone carries both the
composition and the sharpening (``outputs/ltx25-iclora-compat/C_RESULTS.md``).

Why not ``append_ic_lora_reference_video_conditionings``
-------------------------------------------------------
The official helper (``ltx_pipelines/iclora_utils.py``) does the same three
steps, and experiment 2C drove it directly. It is not used in the product for two
reasons, both about the shape of the work rather than the numbers:

* it decodes with ``video_preprocess``, which ``torch.cat``s the growing
  ``(1,C,F,H,W)`` tensor ON THE GPU -- allocating a fresh full-size buffer per
  frame while the previous one is still live, so the allocated volume grows with
  the SQUARE of the frame count. A chain's reference can be thousands of frames.
  :func:`iter_reference_frames_cpu` runs the identical per-frame maths on the
  identical device and moves each finished frame to CPU immediately;
* it takes a PATH, so it cannot be handed the per-segment pixel WINDOW a long
  chain needs (the same file, cut into overlapping runs, decoded once).

The reference decoder is also deliberately NOT ``chain25._load_video_frames_cpu``:
that one raises on a frame-rate mismatch and on a short file, which are the right
answers for a V2V tail (a speed change exactly at the junction would otherwise be
silent) and the wrong ones for a reference. A reference is control material, not
continuity material -- the owner's rule is "a missing reference means generate
without one", never an error -- and its frame rate need not match the output's.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

from chain_math import REFERENCE_ENCODE_TILE_TOKEN_BUDGET, reference_encode_tokens
from engine25.ltxcore_compat import (
    ConditioningItem,
    ConditioningItemAttentionStrengthWrapper,
    VideoConditionByReferenceLatent,
    cleanup_memory,
    combined_image_conditionings,
    decode_video_by_frame,
    ltx_distilled,
    normalize_images,
    read_lora_reference_downscale_factor,
    resize_and_center_crop,
)

logger = logging.getLogger(__name__)

#: The dtype every conditioning latent in this engine is built at. Same constant
#: as ``chain25.DTYPE``; duplicated rather than imported to keep this module free
#: of a chain25 import (chain25 imports THIS one).
DTYPE = torch.bfloat16

#: The VRAM phase the Single path records the reference encode under.
REFERENCE_ENCODE_PHASE = "11_reference_encode"


# ---------------------------------------------------------------------------
# The downscale factor: a VOTE over the job's IC-LoRAs
# ---------------------------------------------------------------------------


def resolve_reference_downscale_factor(
    ic_loras: list[tuple] | None,
    ic_reference: tuple[str, float] | None,
) -> int | None:
    """The reference's spatial downscale factor, read from the LoRA metadata.

    ``None`` when no reference is requested. Otherwise an ``int >= 1``: the
    target/reference resolution ratio the adapter was TRAINED with, which is
    stored in the LoRA's safetensors header and nowhere else (union-control
    declares 2, deblur declares 1).

    Transcribed from 2.3's ``FastVideoPipeline._set_ic_job``
    (``engine/pipeline/fast_video_pipeline.py:425-493``), including the part that
    is easy to get wrong: the official reader returns 1 BOTH for "declared 1" and
    for "key absent", so a Style LoRA (which has no such key) would otherwise
    cast a vote for 1 and quietly override a union-control adapter's 2. Key
    presence is therefore tested separately with ``safe_open``, and only LoRAs
    that actually declare the key may vote. A header that cannot be read does not
    vote -- the official reader degrades the same way, and every registered
    adapter has already been header-validated by ``services/lora_registry.py``.
    """
    if ic_reference is None:
        return None
    entries = list(ic_loras or [])
    if not entries:
        raise ValueError(
            "ic_reference set but no ic_loras -- the reference downscale factor is "
            "read from the LoRA metadata; supply the IC-LoRA."
        )

    from safetensors import safe_open  # noqa: PLC0415 -- only the reference path needs it

    declared: dict[str, int] = {}
    for entry in entries:
        path = str(entry[0])
        try:
            with safe_open(path, framework="pt") as handle:
                metadata = handle.metadata() or {}
        except Exception:  # noqa: BLE001 -- an unreadable header simply does not vote
            continue
        if "reference_downscale_factor" in metadata:
            declared[path] = read_lora_reference_downscale_factor(path)

    if not declared:
        raise ValueError(
            f"ic_reference set but none of the IC-LoRAs {[str(e[0]) for e in entries]} "
            "declares reference_downscale_factor (metadata missing?). Refusing to run "
            "reference conditioning without a declared factor."
        )
    # ONE reference video is loaded at ONE resolution, so every declaring LoRA
    # has to agree -- including a declared 1 against a declared 2, which would
    # otherwise feed one of the two adapters a reference at a scale it was never
    # trained on, and look like a weak adapter rather than a wiring mistake.
    factors = sorted(set(declared.values()))
    if len(factors) > 1:
        raise ValueError(
            f"Conflicting reference_downscale_factor values in IC-LoRAs: {declared}. "
            "Cannot combine LoRAs that expect different reference resolutions."
        )
    factor = factors[0]
    if factor < 1:
        raise ValueError(
            f"IC-LoRA metadata reports reference_downscale_factor={factor} ({declared}); "
            "expected >= 1."
        )
    return factor


def reference_pixel_dims(scale: int | None, cond_height: int, cond_width: int) -> tuple[int, int, int]:
    """``(scale, ref_height, ref_width)``: the size the reference is decoded at.

    ``cond_height`` / ``cond_width`` are the dimensions of the CONDITIONING being
    built -- stage 1's half-resolution pair on every path that uses this, which is
    every path (stage 2 gets no reference).

    Transcribed from 2.3's ``_reference_pixel_dims``
    (``engine/pipeline/fast_video_pipeline.py:1164-1186``). The divisibility guard
    is the same one the official ``append_ic_lora_reference_video_conditionings``
    applies, and it is why every reference-carrying request is held to
    ``width/height % 128 == 0`` at the API: a factor-2 adapter at an odd
    half-resolution has no integral reference size.
    """
    if scale is None or scale < 1:
        raise ValueError(
            f"reference downscale factor not initialised (got {scale}); "
            "resolve_reference_downscale_factor must run before the conditioning is built."
        )
    if cond_height % scale != 0 or cond_width % scale != 0:
        raise ValueError(
            f"Stage-1 dims ({cond_height}x{cond_width}) must be divisible by "
            f"reference_downscale_factor ({scale})"
        )
    return scale, cond_height // scale, cond_width // scale


# ---------------------------------------------------------------------------
# Decode: one CPU frame at a time, no contract checks
# ---------------------------------------------------------------------------


def iter_reference_frames_cpu(
    path: str,
    *,
    height: int,
    width: int,
    frame_cap: int,
    device: torch.device,
) -> Iterator[torch.Tensor]:
    """Yield ONE ``(1,C,1,H,W)`` CPU tensor per decoded reference frame, in order.

    The same three per-frame operations, in the same order, on the same device as
    ``chain25._load_video_frames_cpu`` and as the official ``video_preprocess``:
    ``resize_and_center_crop`` on float32 -> ``normalize_images`` (``x/127.5-1``)
    to :data:`DTYPE` -> ``.to("cpu")``. Only the ASSEMBLY differs, and only to
    keep the GPU holding one frame at a time (see the module docstring).

    What it deliberately does NOT do is check anything. It stops when the file
    stops -- a short reference yields fewer frames, and the callers treat that as
    "generate the rest without one" rather than as an error. There is no
    frame-rate check either: a reference carries composition, not continuity, so
    a 30fps control video over a 24fps generation is a legitimate request.
    """
    for raw in decode_video_by_frame(path=str(path), device=device, frame_cap=int(frame_cap)):
        frame = resize_and_center_crop(raw.to(torch.float32), height, width)
        out = normalize_images(frame, device, DTYPE).to("cpu")
        del raw, frame
        yield out


def load_reference_pixels_cpu(
    path: str,
    *,
    height: int,
    width: int,
    frame_cap: int,
    device: torch.device,
) -> torch.Tensor | None:
    """The whole reference as ONE ``(1,C,F,H,W)`` CPU tensor, or ``None`` if empty.

    ``torch.cat`` over :func:`iter_reference_frames_cpu`; ``None`` rather than a
    ``torch.cat`` on an empty list, because a 0-frame reference is the "generate
    without one" case, not a failure.
    """
    frames = list(iter_reference_frames_cpu(
        path, height=height, width=width, frame_cap=frame_cap, device=device
    ))
    if not frames:
        return None
    out = torch.cat(frames, dim=2)
    del frames
    return out


def iter_reference_windows(
    frame_iter: Iterable[torch.Tensor],
    windows: list[tuple[int, int]],
) -> Iterator[torch.Tensor | None]:
    """Cut ONE long reference into the per-segment pixel windows a chain needs.

    Yields EXACTLY ``len(windows)`` items, one per stage-1 segment, in order: a
    ``(1,C,F,H,W)`` CPU tensor built by ``torch.cat``-ing that window's frames, or
    ``None`` when the reference ran out before the window began.

    ``windows`` comes from ``chain_math.video_segment_windows`` and is
    ``(start_px, len_px)`` on the GLOBAL reference timeline, monotonically
    increasing in ``start_px``, with adjacent windows OVERLAPPING by the ``K_v``
    carry band. This walks the stream ONCE -- skip to the next window's start,
    buffer its length, yield, keep the overlap tail for the next window -- so peak
    buffering is ``max(clip_frames)`` frames rather than the whole file.

    Transcribed from 2.3's ``chain_pipeline._iter_reference_windows``
    (``engine/pipeline/chain_pipeline.py:326-385``), short-reference policy
    included: a partially covered window yields the frames it DOES have (the VAE
    crops a non-8n+1 pixel run itself, so there is no need to round down), and
    only the zero-frame case yields ``None``.
    """
    it = iter(frame_iter)
    buf: list[torch.Tensor] = []   # consecutive frames; buf[0] is global frame ``base``
    base = 0                       # global index of buf[0]; == frames consumed when buf is empty
    exhausted = False

    for start, length in windows:
        # 1. drop what this window has left behind (the previous window's
        #    non-overlapping head), then skip forward if it starts beyond that.
        if start > base:
            drop = min(start - base, len(buf))
            if drop:
                del buf[:drop]
                base += drop
        while base < start and not exhausted:   # buf is empty here by construction
            if next(it, None) is None:
                exhausted = True
                break
            base += 1
        # 2. top the buffer up to this window's length.
        while len(buf) < length and not exhausted:
            nxt = next(it, None)
            if nxt is None:
                exhausted = True
                break
            buf.append(nxt)
        # 3. hand out what we have (never a 0-frame tensor).
        if not buf:
            yield None
            continue
        yield torch.cat(buf[: min(length, len(buf))], dim=2)


# ---------------------------------------------------------------------------
# Encode: pixels -> [VideoConditionByReferenceLatent]
# ---------------------------------------------------------------------------


def set_conv3d_memory_format(module: Any, memory_format: torch.memory_format) -> int:
    """Re-lay-out every ``Conv3d`` weight inside *module* in place; return the count.

    Transcribed from 2.3's ``_set_conv3d_memory_format``
    (``engine/pipeline/fast_video_pipeline.py:47``). ``Conv3d`` ONLY, never a
    whole-module ``.to(memory_format=channels_last_3d)``: the video encoder also
    holds rank-4 ``Conv2d`` weights, which raise "required rank 5 tensor".
    """
    touched = 0
    for child in module.modules():
        if isinstance(child, torch.nn.Conv3d):
            child.to(memory_format=memory_format)
            touched += 1
    return touched


def reference_conditioning_from_pixels(
    video: torch.Tensor,
    *,
    video_encoder: Any,
    device: torch.device,
    tiling_config: Any,
    scale: int,
    strength: float,
    attention_strength: float = 1.0,
    vram: Any = None,
    phase: str | None = None,
) -> list[ConditioningItem]:
    """VAE-encode already-decoded reference PIXELS into ``[conditioning]``.

    ``video`` is a CPU ``(1,C,F,H,W)`` tensor at :func:`reference_pixel_dims`
    resolution. The returned latent stays on ``device`` --
    ``VideoConditionByReferenceLatent.apply_to`` indexes it against the latent
    state's device, so a CPU copy would break rather than save.

    NO STAGE DISCRIMINATOR HERE, on purpose: that check belongs to the callers
    (the Single patch tests ``height == full_height // 2``; the chain only ever
    calls this from its stage-1 loop). Moving it inside would make the chain path
    either always empty or, worse, attach a reference to stage 2.

    Two encode paths -- 2.3's, kept in step with it because both halves of it
    were measured there. The adapter's declared factor settles the first case
    outright; in the second the reference's own LENGTH settles it:

    * **factor 1** (deblur) feeds the reference at 4x the pixel count of a
      factor-2 one, and the untiled encode's intermediates OOM a 16GB card, so it
      goes through ``tiled_encode`` with the DECODE's tiling config (there is no
      separate encode config). It also re-lays the encoder's ``Conv3d`` weights to
      ``channels_last_3d`` for the duration: torch 2.9's bf16 ``Conv3d`` takes an
      im2col fallback for contiguous weights and materialises an
      ``in_ch*27 x output-volume`` bf16 matrix per convolution, and THAT
      intermediate -- not the activations -- is what breaks the factor-1 encode
      (2.3 measured peak 6265 -> 1198 MiB). The restore in ``finally`` is a
      CORRECTNESS requirement, not hygiene: stage 2 of the same job re-encodes its
      images through this very encoder and those latents do change under
      ``channels_last``.
    * **factor >= 2** is encoded in ONE SHOT up to
      :data:`chain_math.REFERENCE_ENCODE_TILE_TOKEN_BUDGET` reference tokens,
      and switches to the SAME tiled + ``channels_last_3d`` path above it
      (§3-76). Below the line nothing moved: every reference short enough for
      the Chained screen's stage-1 comfort banner to stay silent takes the
      untiled branch it always took, so already-shipped outputs stay
      byte-identical. Above it the two mechanisms hand over together --
      tiling and the memory-layout switch are the one combination this
      repository has measured as safe, and tiling with contiguous weights was
      measured OOM. The reference's OWN LENGTH is the entire switch: there is
      no config key, no API field and no UI control for it.
      ``VideoEncoder.forward`` expects its input already on the compute device
      (``tiled_encode`` moves tiles itself; the plain call does not), which is
      why only the untiled branch does the ``.to(device)``.

    The attention-strength wrapper is applied ONLY below 1.0. At 1.0 the bare
    ``VideoConditionByReferenceLatent`` is returned, which is both what the
    official ``iclora_utils`` does and what keeps the ordinary case structurally
    identical to an unpatched run.
    """
    # (1,C,F,H,W) -> the reference's own latent token count, the only switch.
    ref_tokens = reference_encode_tokens(
        int(video.shape[4]), int(video.shape[3]), int(video.shape[2])
    )
    tiled = scale == 1 or ref_tokens > REFERENCE_ENCODE_TILE_TOKEN_BUDGET
    # channels_last_3d rides WITH tiling, never without it: tiled + contiguous
    # was measured OOM, tiled + channels_last_3d is the combination the
    # factor-1 path has shipped on all along.
    relayout = torch.device(device).type == "cuda" and tiled
    converted = 0
    started = time.perf_counter()
    if relayout:
        converted = set_conv3d_memory_format(video_encoder, torch.channels_last_3d)
    try:
        # cleanup AFTER the switch, so the contiguous weight storages it just
        # dropped are reclaimed by this same pass.
        cleanup_memory()
        if vram is not None:
            vram.reset()
        if tiled:
            encoded = video_encoder.tiled_encode(video, tiling_config)
        else:
            encoded = video_encoder(video.to(device))
        del video
    finally:
        if relayout:
            try:
                # Unconditional force back to contiguous: the encoder always
                # enters here contiguous, so "restore" and "normalise" are the
                # same operation.
                set_conv3d_memory_format(video_encoder, torch.contiguous_format)
                cleanup_memory()
            except Exception:  # noqa: BLE001 -- must never mask an in-flight encode failure
                logger.exception(
                    "IC-LoRA reference encode: failed to restore the video encoder's "
                    "contiguous layout"
                )

    seconds = time.perf_counter() - started
    if vram is not None and phase:
        vram.record(phase, seconds)
    logger.info(
        "IC-LoRA reference encode (scale=%d tiled=%s channels_last_3d convs=%d "
        "ref_tokens=%d): latent %s in %.2fs",
        scale, tiled, converted, ref_tokens, tuple(encoded.shape), seconds,
    )

    cond: ConditioningItem = VideoConditionByReferenceLatent(
        latent=encoded,
        downscale_factor=scale,
        strength=float(strength),
    )
    if float(attention_strength) < 1.0:
        cond = ConditioningItemAttentionStrengthWrapper(
            cond, attention_mask=float(attention_strength)
        )
    return [cond]


# ---------------------------------------------------------------------------
# The Single path's monkeypatch
# ---------------------------------------------------------------------------


@contextmanager
def reference_patch(
    *,
    ic_reference: tuple[str, float] | None,
    factor: int | None,
    attention_strength: float,
    full_height: int,
    num_frames: int,
    tiling_config: Any,
    vram: Any = None,
) -> Iterator[list[dict]]:
    """Append the reference to STAGE 1 of ``DistilledPipeline.__call__``.

    Yields a list of per-stage receipt dicts (what was appended, where, how big),
    which is what makes "did the reference actually reach the model" answerable
    from a run rather than from reading the code.

    ``ic_reference is None`` installs NOTHING -- not an inert patch, no patch at
    all -- so a job without a reference runs the unmodified official code path.

    ``tiling_config`` must already be RESOLVED (``ensure_tiling_config``): the
    patched function receives only ``images/height/width/video_encoder/dtype/
    device/color_space``, so neither the tiling config nor ``num_frames`` reaches
    it. Both are closed over instead. ``num_frames`` is the caller's requested
    count, which is the pipeline's resolved count too -- ``ltxcore_compat.verify``
    pins the short-circuit that makes those the same number.
    """
    receipts: list[dict] = []
    if ic_reference is None:
        yield receipts
        return

    ref_path, ref_strength = ic_reference
    original = ltx_distilled.combined_image_conditionings
    if original is not combined_image_conditionings:
        # A patch is already installed (a previous job leaked one, or two jobs
        # are running in one process). Both are bugs this engine's one-job-at-a-
        # time design makes impossible; refusing is cheaper than stacking.
        raise RuntimeError(
            "ltx_pipelines.distilled.combined_image_conditionings is already patched; "
            "refusing to stack a second reference conditioning"
        )

    def patched(
        *,
        images: Any,
        height: int,
        width: int,
        video_encoder: Any,
        dtype: torch.dtype,
        device: torch.device,
        color_space: Any = None,
    ) -> list[ConditioningItem]:
        conds = original(
            images=images, height=height, width=width, video_encoder=video_encoder,
            dtype=dtype, device=device, color_space=color_space,
        )
        # STAGE 1 ONLY. `height` is the sole per-stage-differing argument the
        # official pipeline passes here (verify pins `stage_1_h = height // 2`).
        if height != full_height // 2:
            receipts.append({"stage": 2, "height": height, "action": "skipped"})
            return conds

        scale, ref_h, ref_w = reference_pixel_dims(factor, height, width)
        pixels = load_reference_pixels_cpu(
            ref_path, height=ref_h, width=ref_w, frame_cap=int(num_frames), device=device,
        )
        if pixels is None:
            # "A missing reference means generate without one", never an error.
            logger.warning(
                "IC-LoRA reference %s yielded no frames; generating without a reference",
                ref_path,
            )
            receipts.append({"stage": 1, "height": height, "action": "no_frames"})
            return conds

        frames = int(pixels.shape[2])
        added = reference_conditioning_from_pixels(
            pixels,
            video_encoder=video_encoder,
            device=device,
            tiling_config=tiling_config,
            scale=scale,
            strength=float(ref_strength),
            attention_strength=attention_strength,
            vram=vram,
            phase=REFERENCE_ENCODE_PHASE,
        )
        # IMAGES FIRST, REFERENCE LAST -- the official `ic_lora` pipeline's own
        # order, and the one the I2V + reference combination is trained against.
        conds = conds + added
        inner = getattr(added[-1], "conditioning", added[-1])
        latent = getattr(inner, "latent", None)
        receipt = {
            "stage": 1,
            "height": height,
            "width": width,
            "action": "appended",
            "reference": str(ref_path),
            "reference_frames": frames,
            "reference_pixels": [ref_h, ref_w],
            "downscale_factor": scale,
            "strength": float(ref_strength),
            "attention_strength": float(attention_strength),
            "wrapper": type(added[-1]).__name__,
            "reference_tokens": (
                None if latent is None
                else int(latent.shape[2] * latent.shape[3] * latent.shape[4])
            ),
        }
        receipts.append(receipt)
        logger.info("IC-LoRA reference conditioning appended: %s", receipt)
        return conds

    ltx_distilled.combined_image_conditionings = patched
    logger.info(
        "IC-LoRA reference patch installed: %s strength=%.3f factor=%s attn=%.3f",
        Path(str(ref_path)).name, float(ref_strength), factor, float(attention_strength),
    )
    try:
        yield receipts
    finally:
        ltx_distilled.combined_image_conditionings = original
        logger.info("IC-LoRA reference patch removed")
