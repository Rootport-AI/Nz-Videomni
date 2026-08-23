"""Masked AV-latent clip chaining for LTX 2.5 (§3-102 C1).

What this is
------------
The 2.5 half of "Chained": several clips generated as ONE continuous latent
timeline and decoded ONCE, so every clip boundary lives inside the decode
instead of being an ffmpeg cut between independently decoded videos.

    per-clip STAGE 1 (half-res) with a frozen video+audio carry band at each
    clip's head
      -> assemble ONE stage-1 timeline (linear crossfade over the overlaps)
      -> ONE spatial upsample (temporally chunked, CPU-parked)
      -> STAGE 2 refine in fixed temporal TILES, each tile's leading overlap
         hard-frozen from the previous tile's output
      -> reassemble -> ONE VAE decode (video + audio) -> ONE mp4.

The PROCEDURE is 2.3's, which is the validated one
(``engine/pipeline/chain_pipeline.py``: ``_crossfade_concat``,
``_denoise_av_with_carry``, ``_tile_images``, ``_chunked_upsample_cpu``, and the
order of ``run_chain``'s phases). The CODE is not: 2.3 reaches into a wheel that
no longer exists in this shape -- ``ModelLedger``, ``encode_text``,
``simple_denoising_func``, ``euler_denoising_loop(denoise_fn=)`` are all gone in
LTX-2 v1.2.0 -- so every step here is written against the 1.2.0 blocks
(``prompt_encoder`` / ``image_conditioner`` / ``stage`` / ``upsampler`` /
``video_decoder`` / ``audio_decoder``), reached through
:mod:`engine25.ltxcore_compat` as engine25's rules require.

The geometry is NOT re-derived. :mod:`chain_math` is the torch-free module the
API validator, the mock backend and the 2.3 engine already share; this module
imports the same functions and calls ``compute_chain_layout`` with the same
arguments the app does, so a junction index computed here and one shown in the
UI cannot disagree. There is deliberately no reduced copy of any of it.

The one genuinely new mechanism: the frozen band
------------------------------------------------
2.3 froze a carry band by hand -- create the initial state, overwrite the first
``K`` latent frames' entries in ``denoise_mask`` with ``1 - overlap_strength``,
THEN apply conditionings, THEN noise. 1.2.0 ships that as a first-class
conditioning item, :class:`VideoConditionByMask`, whose ``apply_to`` computes

    clean_latent = clean * (1-m) + tokens * m
    denoise_mask = denoise_mask * (1-m) + (1 - strength) * m

and whose call site, ``create_noised_state``, runs initial-state ->
conditionings -> noiser in exactly 2.3's order. Handed the same tensor as
``initial_latent`` and as the item's ``latent``, with ``m`` = 1 over the head
band, the two routes agree bit for bit (gate G1(a)), so the band is the OFFICIAL
mechanism rather than a re-implementation of the wheel's internals.

There is no audio equivalent upstream, so :class:`AudioHeadBandMask` below is a
line-for-line audio twin of those two expressions. It is the only conditioning
maths this file owns, and ``ltxcore_compat.verify()`` pins the video original it
mirrors so a divergence upstream is reported by name.

Two 2.5-only facts the band has to answer for:

* **The ancestral sampler.** 2.5 distilled samples stage 1 with
  ``euler_ancestral_denoising_loop``, which injects fresh noise every step. It
  re-pins the conditioned tokens after that injection ONLY on the branch it
  takes when ``stepper.eta > 0`` -- so at ``eta == 0`` an "ancestral" run would
  quietly let the frozen band drift. Every ancestral call site here asserts
  ``stepper.eta > 0``, and :func:`ltxcore_compat.verify` pins the condition.
* **``keyframes_mask``** (new in 1.2.0). ``create_initial_state`` marks the
  first latent frame of every state as a single-pixel-frame keyframe --
  unconditionally, because for a fresh generation it is one. For clip/tile
  ``i >= 1`` that frame is carried-over content from the previous clip, so the
  mark is wrong; :class:`ClearKeyframesMask` removes it there.

Scope (§3-102 first stage)
--------------------------
Multi-clip T2V/I2V only: per-clip prompts, clip 0's conditioning images, the
overlap knobs, ``chunked_upsample`` and ``stage2_window``. V2V, A2V, retake, end
source, reference video, LoRA, NAG/VSF and the acceleration knobs are NOT here
and are refused at the API. ``chain_math`` still computes their geometry; this
file simply never passes those arguments, which is why every layout number it
reads is the plain-chain one.
"""

from __future__ import annotations

import copy
import gc
import logging
import re
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch

from chain_math import (
    VIDEO_TIME_FACTOR,
    ChainLayout,
    compute_chain_layout,
    plan_upsample_chunks,
    resolve_stage2_window,
)
from engine25.ltxcore_compat import (
    AUTO_TILING,
    DISTILLED_SIGMAS,
    STAGE_2_DISTILLED_SIGMAS,
    AllocatorTrimStrategy,
    AudioLatentShape,
    ConditioningItem,
    EulerAncestralDiffusionStep,
    GaussianNoiser,
    ImageConditioningInput,
    LatentState,
    LatentTools,
    ModalitySpec,
    SimpleDenoiser,
    VideoConditionByMask,
    VideoLatentShape,
    VideoPixelShape,
    cleanup_memory,
    encode_video,
    ensure_tiling_config,
    euler_ancestral_denoising_loop,
    get_video_chunks_number,
    gpu_model,
    image_conditionings_by_adding_guiding_latent,
    image_conditionings_by_replacing_latent,
    tiling_scale_factors_for_vae,
    upsample_video,
    upsampler_builders,
)

logger = logging.getLogger(__name__)

DTYPE = torch.bfloat16

#: Stage-1 sampler for a chain. See :func:`run_chain`'s ``stage1_sampler``.
#: The value is a MEASUREMENT, not a preference -- gate G1(b) compared the three
#: candidates on seam MAD ratio and same-seed reproducibility; the result and its
#: numbers live in ``outputs/ltx25-chain-g1/G1_RESULTS.md``.
STAGE1_SAMPLER = "ancestral"

#: ``eta`` for the ancestral stage-1 sampler. 1.0 is what ``DistilledPipeline``
#: uses for a single generation, i.e. what the checkpoint was distilled for.
STAGE1_ANCESTRAL_ETA = 1.0
STAGE1_ANCESTRAL_S_NOISE = 1.0

#: Seed offset of the ancestral loop's own noise generator. Same constant the
#: official pipeline uses (``ANCESTRAL_NOISE_SEED_OFFSET``): without it the
#: loop's first draw would be bit-identical to the initial latent noise, since
#: both are ``torch.randn`` at the same shape from a freshly seeded generator.
ANCESTRAL_NOISE_SEED_OFFSET = 10000

#: Clear the wrongly-inherited first-frame keyframe marker on every clip and
#: tile that starts on carried-over content. See the module docstring; the A/B
#: behind the default is gate G1(c).
CLEAR_KEYFRAMES_ON_CARRY = True

#: Stage-2 seed offset, mirroring 2.3: stage-1 segment ``i`` uses ``seed + i``,
#: stage-2 tile ``i`` uses ``seed + 100 + i``, the decode uses ``seed``.
STAGE2_SEED_OFFSET = 100

# progress(stage, index, total, *, outer_index=None, outer_total=None).
# Coarse stages are 2.3's own names -- "encode" / "stage1" / "upsample" /
# "tile" / "decode" -- because the app's receipt loop already maps them.
# "encode" is not in this list because it is not emitted here: the prompt
# encoder emits it itself (``Ltx25PromptEncoder.__call__``), which is also what
# makes it fire around the actual Gemma load rather than around this function's
# idea of when the encode happens.
ProgressFn = Callable[..., None]

STAGE_1 = "stage1"
STAGE_UPSAMPLE = "upsample"
STAGE_TILE = "tile"
STAGE_DECODE = "decode"
STAGE_1_DENOISE = "stage1_denoise"
STAGE_2_DENOISE = "stage2_denoise"


class ChainError(RuntimeError):
    """A chain request the LTX 2.5 engine cannot run."""


# ---------------------------------------------------------------------------
# Request / result
# ---------------------------------------------------------------------------


@dataclass
class ChainClipSpec:
    """One clip of the chain (prompt already resolved to its effective value)."""

    prompt: str
    num_frames: int
    images: list[ImageConditioningInput] = field(default_factory=list)


@dataclass
class ChainSpec:
    """One chain job: exactly the body keys of the ``generate_chain`` payload.

    Deliberately nothing else. The 422'd features (V2V, A2V, retake, end source,
    reference video, LoRA, NAG/VSF, the acceleration knobs) have no field here
    at all, so "this engine does not do that" is visible in the type rather than
    in a runtime branch -- and adding one later is a deliberate act.

    ``num_steps`` is carried and reported but never acted on, exactly as in 2.3:
    the distilled schedule is fixed at 8 + 3 sigmas.
    """

    clips: list[ChainClipSpec]
    width: int
    height: int
    frame_rate: float
    seed: int
    overlap_frames: int
    overlap_strength: float
    output_path: str
    num_steps: int = 0
    chunked_upsample: bool = False
    stage2_window: str | None = None


@dataclass
class ChainResult:
    """What one :func:`run_chain` produced."""

    output_path: str
    metadata: dict


# ---------------------------------------------------------------------------
# Conditioning items
# ---------------------------------------------------------------------------


class AudioHeadBandMask(ConditioningItem):
    """Freeze the leading ``K`` audio latent frames -- the audio twin of
    :class:`VideoConditionByMask`.

    Upstream has a masked conditioning item for video and none for audio, so
    this is the one piece of conditioning arithmetic engine25 owns. It is
    written to be the same two expressions, not merely an equivalent freeze:

        clean_latent = clean * (1-m) + tokens * m
        denoise_mask = denoise_mask * (1-m) + (1 - strength) * m

    ``ltxcore_compat.verify()`` asserts those two lines are still what
    ``VideoConditionByMask.apply_to`` computes, so if upstream ever changes the
    video formula this twin is reported as stale by name instead of silently
    diverging.

    The audio token domain is one token per latent frame (``AudioPatchifier`` at
    ``patch_size=1``), so ``mask`` is ``(B, T)``: entry ``t`` is 1 where latent
    frame ``t`` is carried over. It is patchified through the SAME call the
    video item uses -- ``patchify(mask[:, None, :, None])`` -- rather than being
    reshaped by hand, so the token order is the patchifier's, whatever that is.

    MUST be applied BEFORE any conditioning item that APPENDS tokens (a
    ``frame_idx > 0`` keyframe image): the arithmetic is elementwise over the
    whole token axis, so a state that has already grown extra tokens would not
    broadcast. :func:`run_chain` builds the list in that order; the assertion
    below is what turns a future reordering into an error rather than a wrong
    freeze.
    """

    def __init__(self, latent: torch.Tensor, mask: torch.Tensor, strength: float = 1.0) -> None:
        self.latent = latent
        self.mask = mask
        self.strength = strength

    def apply_to(self, latent_state: LatentState, latent_tools: LatentTools) -> LatentState:
        tokens = latent_tools.patchifier.patchify(self.latent)
        mask = latent_tools.patchifier.patchify(self.mask[:, None, :, None])

        assert tokens.shape[1] == latent_state.denoise_mask.shape[1], (
            f"AudioHeadBandMask covers {tokens.shape[1]} tokens but the state has "
            f"{latent_state.denoise_mask.shape[1]}; the band must be applied before any "
            f"conditioning item that appends tokens."
        )

        m = mask.to(dtype=latent_state.latent.dtype)
        inv = 1 - m

        return replace(
            latent_state,
            clean_latent=latent_state.clean_latent * inv + tokens * m,
            denoise_mask=latent_state.denoise_mask * inv + (1.0 - self.strength) * m,
        )


class ClearKeyframesMask(ConditioningItem):
    """Drop the first-frame keyframe marker on a clip/tile that starts on carry.

    ``VideoLatentTools.create_initial_state`` marks the first latent frame of
    every state -- unconditionally, and correctly, for a fresh generation: the
    causal video VAE makes latent 0 cover exactly one pixel frame, the same
    token class as a keyframe slot, and the checkpoint carries a learned
    embedding for it (``use_keyframes_abs_pos_embedding``).

    In a chain, clip/tile ``i >= 1`` starts on the PREVIOUS clip's tail, where
    latent 0 is ordinary carried-over content covering eight pixel frames. Left
    marked, it would get an embedding meant for a standalone frame.

    Zeroes rather than sets ``None``: the marker is consumed as
    ``(keyframes_mask > 0) * embedding``, so an all-zero mask is the same
    addition of zero that ``None`` short-circuits to, and keeping a tensor means
    items that append tokens (``extend_keyframes_mask``) stay on one code path.
    """

    def apply_to(self, latent_state: LatentState, latent_tools: LatentTools) -> LatentState:  # noqa: ARG002
        if latent_state.keyframes_mask is None:
            return latent_state
        return replace(latent_state, keyframes_mask=torch.zeros_like(latent_state.keyframes_mask))


# ---------------------------------------------------------------------------
# Primitives (ported from 2.3, which validated them)
# ---------------------------------------------------------------------------


def _crossfade_concat(seg0: torch.Tensor, seg1: torch.Tensor, k: int) -> torch.Tensor:
    """Temporal (dim=2) crossfade concat; total = L0 + L1 - K.

    Verbatim from ``engine/pipeline/chain_pipeline.py`` -- the fade weights and
    the fp32 intermediate are load-bearing, not stylistic: over a hard-frozen
    overlap both sides hold the SAME latents, and ``a*(1-x) + a*x == a`` in fp32
    for these linspace weights, so the fade is the identity there. Works for 5D
    video ``(b,c,f,h,w)`` and 4D audio ``(b,c,t,f)`` alike (time is dim 2).
    """
    l0, l1 = seg0.shape[2], seg1.shape[2]
    total = l0 + l1 - k
    out_shape = list(seg0.shape)
    out_shape[2] = total
    out = torch.zeros(out_shape, dtype=seg0.dtype, device=seg0.device)
    out[:, :, :l0] = seg0
    if k > 0:
        alpha = torch.linspace(0.0, 1.0, k + 2, device=seg0.device, dtype=torch.float32)[1:-1]
        for i in range(k):
            a = float(alpha[i])
            g = l0 - k + i
            out[:, :, g] = (
                seg0[:, :, l0 - k + i].float() * (1.0 - a) + seg1[:, :, i].float() * a
            ).to(seg0.dtype)
    out[:, :, l0:] = seg1[:, :, k:]
    return out


def _video_conditionings(
    images: Sequence[ImageConditioningInput],
    *,
    height: int,
    width: int,
    video_encoder: Any,
    device: torch.device,
) -> list[ConditioningItem]:
    """Hybrid keyframe routing, 2.3's rule on 1.2.0's helpers.

    ``frame_idx == 0`` -> latent REPLACE, ``frame_idx > 0`` -> keyframe/guide
    APPEND with ``frame_idx`` as a raw pixel RoPE offset. This is the pair the
    2.3 chain uses and both helpers survive unchanged in 1.2.0.

    Deliberately NOT ``combined_image_conditionings`` (what
    ``DistilledPipeline.__call__`` calls): that one pins a ``frame_idx == 0``
    image to ``latent_idx=0`` and is a different routing. Keeping 2.3's pair is
    what makes a 2.5 chain's conditioning the same operation a 2.3 chain's was.

    Two 1.2.0 differences ride along and are accepted as 2.5's behaviour:
    ``VideoConditionByKeyframeIndex`` now defaults ``num_pixel_frames=1`` (which
    narrows a keyframe's temporal span to one frame) and applies ``causal_fix``
    only at ``frame_idx == 0``.
    """
    if not images:
        return []
    replace_imgs = [im for im in images if im.frame_idx == 0]
    guide_imgs = [im for im in images if im.frame_idx > 0]
    conds: list[ConditioningItem] = []
    if replace_imgs:
        conds += image_conditionings_by_replacing_latent(
            images=replace_imgs, height=height, width=width,
            video_encoder=video_encoder, dtype=DTYPE, device=device,
        )
    if guide_imgs:
        conds += image_conditionings_by_adding_guiding_latent(
            images=guide_imgs, height=height, width=width,
            video_encoder=video_encoder, dtype=DTYPE, device=device,
        )
    return conds


def _tile_images(
    images: Sequence[ImageConditioningInput], vs: int, vlen: int
) -> list[ImageConditioningInput]:
    """Filter + remap clip-0's conditioning images to the stage-2 tile ``[vs, vs+vlen)``.

    Verbatim from 2.3. A conditioning belongs to the tile whose global latent
    range contains its latent frame, and ``frame_idx`` is remapped to the
    tile-local pixel offset because tiles restart their RoPE positions. For the
    common case (a start frame at index 0) everything lands in tile 0 unchanged.
    """
    out: list[ImageConditioningInput] = []
    for im in images:
        g_lat = 0 if im.frame_idx == 0 else (im.frame_idx - 1) // VIDEO_TIME_FACTOR + 1
        if vs <= g_lat < vs + vlen:
            local = 0 if im.frame_idx == 0 else im.frame_idx - vs * VIDEO_TIME_FACTOR
            out.append(im._replace(frame_idx=max(0, local)))
    return out


class _StatsOnlyEncoder:
    """Just the ``per_channel_statistics`` of a video encoder, outliving it.

    ``upsample_video`` uses the encoder for exactly two things -- un-normalize
    before the upsampler and normalize after -- and nothing else. The chunked
    path builds the encoder once, takes a DEEP COPY of that submodule and frees
    the ~1 GB encoder before the first chunk, so it is not resident alongside
    the upsampler for the whole loop.

    The copy is mandatory, not tidiness: ``Disposable.dispose`` moves every
    persistent buffer to the meta device, and the two statistics tensors are
    persistent buffers. A reference to the live submodule would come back as
    meta tensors on the first chunk.
    """

    def __init__(self, encoder: Any) -> None:
        self.per_channel_statistics = copy.deepcopy(encoder.per_channel_statistics)


def _chunked_upsample_cpu(
    assembled_v: torch.Tensor,
    *,
    upsampler_block: Any,
    device: torch.device,
    dtype: torch.dtype,
    progress: ProgressFn | None = None,
) -> torch.Tensor:
    """Temporal-chunked spatial upsample; returns the upscaled latent on CPU.

    The memory-bounded twin of one ``upsampler(latent)`` over the whole
    timeline. 2.3's recipe, unchanged, because it is the one that was measured:

    * the assembled stage-1 latent is parked on CPU and each
      :func:`chain_math.plan_upsample_chunks` chunk (core + halo) is streamed to
      the GPU, upsampled, its halo dropped, and evicted back to CPU with an
      ``empty_cache`` -- the per-chunk cache release is mandatory; without it
      the reserved pool fragments and the job spills to shared memory;
    * ``channels_last_3d`` is set on the ``Conv3d`` modules ONLY. The upsampler
      also holds ``Conv2d`` (the spatial 2x is ``Conv2d`` + ``PixelShuffleND``),
      and a whole-module ``.to(memory_format=channels_last_3d)`` raises
      "required rank 5 tensor" on those;
    * ``halo = 18`` is not a guess: ``LatentUpsampler`` at
      ``num_blocks_per_stage=4`` has 1 + 2*4 + 2*4 + 1 = 18 ``Conv3d(k=3)``
      layers and no temporal resampling, so its temporal receptive field is 18
      frames each side and a chunk padded by 18 reproduces the one-shot
      convolution's interior. (Gate G1(d) checks the count against the shipped
      checkpoint's own config and measures the residual difference, which is not
      zero: ``GroupNorm`` statistics are computed per chunk. That is the same
      known non-exactness 2.3 accepted.)

    Both models are built ONCE. The official ``VideoUpsampler.__call__`` builds
    and frees them per call, on a registry constructed with
    ``cache_weights=False``, so driving it per chunk would re-read both
    checkpoints from disk every time.
    """
    plan = plan_upsample_chunks(int(assembled_v.shape[2]))
    src_cpu = assembled_v[:1].to("cpu")

    encoder_builder, upsampler_builder = upsampler_builders(upsampler_block)
    with gpu_model(
        encoder_builder.build(device=device, dtype=dtype).eval(),
        alloc_trim_strategy=AllocatorTrimStrategy.TRIM,
    ) as encoder:
        stats = _StatsOnlyEncoder(encoder)

    out_parts: list[torch.Tensor] = []
    with gpu_model(
        upsampler_builder.build(device=device, dtype=dtype).eval(),
        alloc_trim_strategy=AllocatorTrimStrategy.TRIM,
    ) as upsampler:
        for module in upsampler.modules():
            if isinstance(module, torch.nn.Conv3d):
                module.to(memory_format=torch.channels_last_3d)
        for index, chunk in enumerate(plan):
            xin = src_cpu[:, :, chunk.in_start:chunk.in_start + chunk.in_len]
            xin = xin.to(device).contiguous(memory_format=torch.channels_last_3d)
            yout = upsample_video(latent=xin, video_encoder=stats, upsampler=upsampler)
            out_parts.append(
                yout[:, :, chunk.keep_lo:chunk.keep_lo + chunk.keep_len].to("cpu")
            )
            del xin, yout
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if progress is not None:
                progress(STAGE_UPSAMPLE, index, len(plan))

    result = torch.cat(out_parts, dim=2)
    del out_parts, src_cpu
    return result


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _stage1_sampler_kwargs(sampler: str, eta: float, seed: int, dtype: torch.dtype) -> dict[str, Any]:
    """``stepper`` / ``loop`` for one stage-1 segment.

    ``"euler"`` returns an EMPTY dict on purpose: ``DiffusionStage`` then applies
    its own ``EulerDiffusionStep`` + ``euler_denoising_loop`` defaults, so the
    deterministic arm is the stage's own default path rather than a restatement
    of it. That is also the documented fallback if the ancestral arm ever has to
    be backed out.
    """
    if sampler == "euler":
        return {}
    if sampler != "ancestral":
        raise ChainError(f"unknown stage-1 sampler {sampler!r} (expected 'ancestral' or 'euler')")

    stepper = EulerAncestralDiffusionStep(eta=eta, s_noise=STAGE1_ANCESTRAL_S_NOISE)
    # THE assertion the frozen band rests on. The ancestral driver re-pins the
    # conditioned tokens only on the branch it takes when it draws noise, and it
    # draws noise iff eta > 0 -- so an eta of 0 here would not be "the same
    # sampler, less noise", it would be a chain whose carry bands drift with
    # nothing reporting it.
    assert stepper.eta > 0, (
        "an ancestral stage-1 needs eta > 0: at eta == 0 the loop stops re-pinning "
        "the frozen carry band after each step (see ltxcore_compat.verify)"
    )

    def loop(**kwargs: Any) -> tuple[LatentState | None, LatentState | None]:
        return euler_ancestral_denoising_loop(
            noise_seed=seed + ANCESTRAL_NOISE_SEED_OFFSET,
            model_dtype=dtype,
            **kwargs,
        )

    return {"stepper": stepper, "loop": loop}


def _band_conditionings(
    *,
    video_latent: torch.Tensor | None,
    video_frames_frozen: int,
    audio_latent: torch.Tensor | None,
    audio_frames_frozen: int,
    strength: float,
    clear_keyframes: bool,
) -> tuple[list[ConditioningItem], list[ConditioningItem]]:
    """The ``(video, audio)`` conditioning prefixes for one segment or tile.

    Both lists are EMPTY when nothing is frozen, which is what makes a free head
    (clip 0 at stage 1, tile 0 at stage 2) literally the official code path with
    no band item in it at all.
    """
    video: list[ConditioningItem] = []
    audio: list[ConditioningItem] = []

    if video_latent is not None and video_frames_frozen > 0:
        if clear_keyframes:
            video.append(ClearKeyframesMask())
        b, _, _, h, w = video_latent.shape
        mask = torch.zeros((b, video_latent.shape[2], h, w), dtype=torch.float32, device=video_latent.device)
        mask[:, :video_frames_frozen] = 1.0
        video.append(VideoConditionByMask(latent=video_latent, mask=mask, strength=strength))

    if audio_latent is not None and audio_frames_frozen > 0:
        mask_a = torch.zeros(
            (audio_latent.shape[0], audio_latent.shape[2]),
            dtype=torch.float32,
            device=audio_latent.device,
        )
        mask_a[:, :audio_frames_frozen] = 1.0
        audio.append(AudioHeadBandMask(latent=audio_latent, mask=mask_a, strength=strength))

    return video, audio


def _vram_summary(phases: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Reduce a chain's per-phase peaks to what belongs on ``done``.

    A chain records one phase per clip and per tile, so the full table grows
    with the request and would put dozens of entries on an event the app parses
    for two numbers. The maxima are what a gate reads; ``count`` is what says
    how many phases they were taken over, so a truncated run is visible. The
    full table is kept alongside it under ``metadata["ltx25"]["phases"]``, which
    is where a post-mortem looks and where the app does not.
    """
    if not phases:
        return {"count": 0}

    def _peak(key: str) -> float | None:
        values = [entry.get(key) for entry in phases.values() if entry.get(key) is not None]
        return max(values) if values else None

    by_kind: dict[str, dict[str, Any]] = {}
    for name, entry in phases.items():
        # "21_stage1_denoise_c3" -> "stage1_denoise": drop the ordering prefix
        # and the per-clip/per-tile index, which is exactly what a summary
        # should collapse. The index is matched as a TRAILING "_c<n>"/"_t<n>"
        # rather than by splitting on the substring -- "11_image_conditioning"
        # contains "_c" in the middle of a word and would otherwise be reported
        # as a phase called "image".
        kind = name.split("_", 1)[1] if "_" in name else name
        kind = re.sub(r"_[ct]\d+$", "", kind)
        bucket = by_kind.setdefault(kind, {"count": 0, "seconds": 0.0, "peak_allocated_gib": None})
        bucket["count"] += 1
        bucket["seconds"] = round(bucket["seconds"] + float(entry.get("seconds") or 0.0), 2)
        peak = entry.get("peak_allocated_gib")
        if peak is not None:
            current = bucket["peak_allocated_gib"]
            bucket["peak_allocated_gib"] = peak if current is None else max(current, peak)

    return {
        "count": len(phases),
        "peak_allocated_gib": _peak("peak_allocated_gib"),
        "peak_reserved_gib": _peak("peak_reserved_gib"),
        "rss_peak_gib": _peak("rss_gib"),
        "seconds_total": round(sum(float(e.get("seconds") or 0.0) for e in phases.values()), 2),
        "by_kind": by_kind,
    }


def _decode_progress(video: Iterator[torch.Tensor], chunks: int, progress: ProgressFn | None) -> Iterator[torch.Tensor]:
    """Report decode progress as the mp4 encoder pulls chunks off the iterator.

    The decode is lazy, so counting here counts the decode itself. Same shape as
    the single-generation path's counter in :mod:`engine25.pipeline25`.
    """
    if progress is None:
        return video

    def counted() -> Iterator[torch.Tensor]:
        done = 0
        for chunk in video:
            yield chunk
            done += 1
            progress(STAGE_DECODE, done - 1, max(1, chunks))

    return counted()


# ---------------------------------------------------------------------------
# The chain
# ---------------------------------------------------------------------------


def run_chain(  # noqa: PLR0915 -- one linear procedure; splitting it would hide the order
    pipeline: Any,
    spec: ChainSpec,
    progress: ProgressFn | None = None,
    *,
    stage1_sampler: str | None = None,
    stage1_eta: float | None = None,
    clear_keyframes: bool | None = None,
) -> ChainResult:
    """Run one masked AV-latent chain to ONE mp4.

    ``pipeline`` is an :class:`engine25.pipeline25.Ltx25Pipeline` -- the loaded
    model, whose official blocks this function drives directly instead of going
    through ``DistilledPipeline.__call__``.

    The three keyword-only arguments are ENGINE-INTERNAL experiment knobs, not
    request fields: they exist so gate G1's A/B arms can be run against the
    shipped code path rather than a copy of it, and they default to the module
    constants above, which are what every real job uses. Nothing in the
    ``generate_chain`` payload maps to them.
    """
    sampler = STAGE1_SAMPLER if stage1_sampler is None else stage1_sampler
    eta = STAGE1_ANCESTRAL_ETA if stage1_eta is None else float(stage1_eta)
    clear_kf = CLEAR_KEYFRAMES_ON_CARRY if clear_keyframes is None else bool(clear_keyframes)

    dp = pipeline.pipeline            # the official DistilledPipeline
    stage = pipeline.stage            # Ltx25ProgressStage
    device: torch.device = pipeline.device
    vram = pipeline.vram

    clips = list(spec.clips)
    if not clips:
        raise ChainError("a chain needs at least one clip")
    n = len(clips)
    kv = int(spec.overlap_frames)
    width, height = int(spec.width), int(spec.height)
    frame_rate = float(spec.frame_rate)
    base_seed = int(spec.seed)

    # ── Geometry: the SHARED pure module, called the way the app calls it ─────
    # Same positional pair, same ``kv``, same resolved (v_tile, v_adv). The
    # feature keywords the app also passes (source_context_px / retake_glue_px /
    # end_context_px) are all None on every request this engine accepts, so
    # omitting them is the same call -- gate G1(e) pins that against the app's
    # own call sites rather than leaving it as a comment.
    v_tile, v_adv = resolve_stage2_window(spec.stage2_window)
    layout: ChainLayout = compute_chain_layout(
        [c.num_frames for c in clips], frame_rate,
        kv=kv,
        v_tile=v_tile, v_adv=v_adv,
    )
    seg_frames = layout.seg_frames
    n_seg = len(seg_frames)
    assert n_seg == n, (n_seg, n)   # no feature here appends a segment
    ka_list = layout.ka_list
    v_tiles, a_tiles = layout.v_tiles, layout.a_tiles
    kt_v, kt_a = layout.kt_v, layout.kt_a
    n_tiles = layout.n_tiles
    total_px = layout.total_px

    seeds = [base_seed + i for i in range(n_seg)]

    out_path = Path(spec.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "chain: %d clips %s -> %d px frames, %dx%d @ %.3f fps, kv=%d strength=%.3f, "
        "%d stage-2 tiles (window %s = %d/%d), sampler=%s eta=%s clear_keyframes=%s, chunked_upsample=%s",
        n, [c.num_frames for c in clips], total_px, width, height, frame_rate, kv,
        spec.overlap_strength, n_tiles, spec.stage2_window or "standard", v_tile, v_adv,
        sampler, eta if sampler == "ancestral" else "-", clear_kf, spec.chunked_upsample,
    )

    stage.begin_job()
    stage.progress = progress
    pipeline.prompt_encoder.progress = progress
    vram.phases.clear()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    started = time.time()

    with torch.no_grad():
        # ── Text encode: every DISTINCT prompt, in ONE call ───────────────────
        # ``PromptEncoder.__call__`` takes a list and loads Gemma once for the
        # whole list, so the dedup is about the transformer-side encode cost,
        # not about the load. Order-preserving so a two-clip chain with the same
        # prompt twice is one encode and the mapping back is by value.
        distinct = list(dict.fromkeys(c.prompt for c in clips))
        encoded = pipeline.prompt_encoder(distinct)
        ctx_by_prompt = {
            prompt: (out.video_encoding, out.audio_encoding)
            for prompt, out in zip(distinct, encoded, strict=True)
        }
        seg_ctx = [ctx_by_prompt[c.prompt] for c in clips]

        # ── Image conditioning: ONE encoder build for every resolution ────────
        # ``ImageConditioner.__call__(fn)`` builds the video encoder, calls
        # ``fn(encoder)`` once and frees it, so the half-resolution stage-1 items
        # and the full-resolution per-tile items are all made inside a single
        # call. ``resolve_crf`` first, always: the checkpoint's own CRF is what
        # the images must be re-compressed at, and ``crf=None`` reaches
        # ``load_image_and_preprocess`` as a hard failure.
        clip0_images: list[ImageConditioningInput] = list(clips[0].images)
        stage1_conds: list[ConditioningItem] = []
        tile_conds: list[list[ConditioningItem]] = [[] for _ in range(n_tiles)]
        if clip0_images:
            clip0_images = dp.image_conditioner.resolve_crf(clip0_images)

            def _build_all(encoder: Any) -> tuple[list, list[list]]:
                half = _video_conditionings(
                    clip0_images, height=height // 2, width=width // 2,
                    video_encoder=encoder, device=device,
                )
                full = [
                    _video_conditionings(
                        _tile_images(clip0_images, vs, vlen), height=height, width=width,
                        video_encoder=encoder, device=device,
                    )
                    for vs, vlen in v_tiles
                ]
                return half, full

            vram.reset()
            conditioning_started = time.perf_counter()
            stage1_conds, tile_conds = dp.image_conditioner(_build_all)
            vram.record("11_image_conditioning", time.perf_counter() - conditioning_started)

        # ── STAGE 1: per-clip at half resolution, with the carry band ─────────
        stage_1_sigmas = DISTILLED_SIGMAS.to(dtype=torch.float32, device=device)
        seg_v: list[torch.Tensor] = []
        seg_a: list[torch.Tensor] = []
        for i in range(n_seg):
            seg_shape = VideoPixelShape(1, seg_frames[i], height // 2, width // 2, frame_rate)
            v_shape = VideoLatentShape.from_pixel_shape(
                seg_shape, scale_factors=stage.video_scale_factors
            )
            a_shape = AudioLatentShape.from_video_pixel_shape(seg_shape)

            init_v = init_a = None
            fkv = fka = 0
            if i > 0:
                # The carry: the previous clip's tail, in tensors sized from THIS
                # clip's shapes (clip lengths may differ, so ``zeros_like(prev)``
                # would be the wrong shape).
                prev_v, prev_a = seg_v[i - 1], seg_a[i - 1]
                ka_i = ka_list[i - 1]
                init_v = torch.zeros(tuple(v_shape.to_torch_shape()), dtype=DTYPE, device=device)
                init_v[:, :, :kv] = prev_v[:, :, prev_v.shape[2] - kv:]
                init_a = torch.zeros(tuple(a_shape.to_torch_shape()), dtype=DTYPE, device=device)
                init_a[:, :, :ka_i] = prev_a[:, :, prev_a.shape[2] - ka_i:]
                fkv, fka = kv, ka_i

            band_v, band_a = _band_conditionings(
                video_latent=init_v, video_frames_frozen=fkv,
                audio_latent=init_a, audio_frames_frozen=fka,
                strength=float(spec.overlap_strength),
                clear_keyframes=clear_kf,
            )
            # Clip 0's images are the TIMELINE's opening keyframes, so they go on
            # clip 0 only -- and AFTER the band items, which must see an
            # un-extended token axis (see AudioHeadBandMask).
            conds_v = band_v + (stage1_conds if i == 0 else [])

            noiser = GaussianNoiser(generator=torch.Generator(device=device).manual_seed(seeds[i]))
            video_context, audio_context = seg_ctx[i]
            stage.announce(
                STAGE_1_DENOISE,
                vram_phase=f"21_stage1_denoise_c{i}",
                outer_index=i,
                outer_total=n_seg,
            )
            vstate, astate = stage(
                denoiser=SimpleDenoiser(video_context, audio_context),
                sigmas=stage_1_sigmas,
                noiser=noiser,
                width=width // 2,
                height=height // 2,
                frames=seg_frames[i],
                fps=frame_rate,
                video=ModalitySpec(
                    context=video_context, conditionings=conds_v, initial_latent=init_v
                ),
                audio=ModalitySpec(
                    context=audio_context, conditionings=band_a, initial_latent=init_a
                ),
                **_stage1_sampler_kwargs(sampler, eta, seeds[i], pipeline.dtype),
            )
            seg_v.append(vstate.latent.detach().clone())
            seg_a.append(astate.latent.detach().clone())
            del vstate, astate, init_v, init_a
            if progress is not None:
                progress(STAGE_1, i, n_seg)

        # ── Assemble ONE continuous stage-1 timeline ──────────────────────────
        assembled_v = seg_v[0]
        for i in range(1, n_seg):
            assembled_v = _crossfade_concat(assembled_v, seg_v[i], kv)
        assembled_a = seg_a[0]
        for i in range(1, n_seg):
            assembled_a = _crossfade_concat(assembled_a, seg_a[i], ka_list[i - 1])
        del seg_v, seg_a

        exp_v = VideoLatentShape.from_pixel_shape(
            VideoPixelShape(1, total_px, height // 2, width // 2, frame_rate),
            scale_factors=stage.video_scale_factors,
        ).to_torch_shape()
        exp_a = AudioLatentShape.from_video_pixel_shape(
            VideoPixelShape(1, total_px, height, width, frame_rate)
        ).to_torch_shape()
        assert tuple(assembled_v.shape) == tuple(exp_v), (tuple(assembled_v.shape), tuple(exp_v))
        assert tuple(assembled_a.shape) == tuple(exp_a), (tuple(assembled_a.shape), tuple(exp_a))

        # ── ONE upsample over the whole timeline ──────────────────────────────
        vram.reset()
        upsample_started = time.perf_counter()
        if spec.chunked_upsample:
            upscaled_v = _chunked_upsample_cpu(
                assembled_v,
                upsampler_block=dp.upsampler,
                device=device,
                dtype=pipeline.dtype,
                progress=progress,
            )
        else:
            upscaled_v = dp.upsampler(assembled_v[:1])
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        vram.record("23_upsample", time.perf_counter() - upsample_started)
        del assembled_v
        cleanup_memory()

        # ── STAGE 2: always-tiled refine, video + audio jointly ───────────────
        # ONE context for the WHOLE refine -- clip 0's. Per-clip prompt variation
        # lives in stage 1, where the carry + crossfade absorbs it; switching the
        # AUDIO context mid-tile-overlap injects a click at the frozen tile seam
        # (2.3's measured finding, and the reason its S2 spike used one prompt).
        stage2_vctx, stage2_actx = seg_ctx[0]
        stage_2_sigmas = STAGE_2_DISTILLED_SIGMAS.to(dtype=torch.float32, device=device)
        noise_scale2 = float(stage_2_sigmas[0].item())
        refined_v: list[torch.Tensor] = []
        refined_a: list[torch.Tensor] = []
        for i in range(n_tiles):
            vs, vlen = v_tiles[i]
            as_, alen = a_tiles[i]
            tile_px = (vlen - 1) * VIDEO_TIME_FACTOR + 1

            init_v = upscaled_v[:, :, vs:vs + vlen].contiguous().clone()
            if init_v.device.type != device.type:
                init_v = init_v.to(device)
            init_a = assembled_a[:, :, as_:as_ + alen].contiguous().clone()

            fkv = fka = 0
            if i > 0:
                # HARD freeze (strength 1.0 -> denoise mask 0.0) of the leading
                # overlap, overwritten from the previous tile's OUTPUT rather
                # than from the upsampled stage-1 approximation. Re-freezing is
                # mandatory: stage 2 re-noises at sigma[0] ~= 0.909, which would
                # destroy a stage-1-only freeze outright.
                init_v[:, :, :kt_v] = refined_v[i - 1][:, :, refined_v[i - 1].shape[2] - kt_v:]
                init_a[:, :, :kt_a] = refined_a[i - 1][:, :, refined_a[i - 1].shape[2] - kt_a:]
                fkv, fka = kt_v, kt_a

            band_v, band_a = _band_conditionings(
                video_latent=init_v, video_frames_frozen=fkv,
                audio_latent=init_a, audio_frames_frozen=fka,
                strength=1.0,
                clear_keyframes=clear_kf,
            )
            conds_v = band_v + tile_conds[i]

            noiser2 = GaussianNoiser(
                generator=torch.Generator(device=device).manual_seed(base_seed + STAGE2_SEED_OFFSET + i)
            )
            stage.announce(
                STAGE_2_DENOISE,
                vram_phase=f"22_stage2_denoise_t{i}",
                outer_index=i,
                outer_total=n_tiles,
            )
            # Stage 2 is DETERMINISTIC Euler, in both engines and upstream: its
            # three-step refine is too short to remove freshly injected noise.
            # Passing neither ``stepper`` nor ``loop`` is how that is said --
            # they are ``DiffusionStage``'s own defaults.
            vstate2, astate2 = stage(
                denoiser=SimpleDenoiser(stage2_vctx, stage2_actx),
                sigmas=stage_2_sigmas,
                noiser=noiser2,
                width=width,
                height=height,
                frames=tile_px,
                fps=frame_rate,
                video=ModalitySpec(
                    context=stage2_vctx, conditionings=conds_v,
                    noise_scale=noise_scale2, initial_latent=init_v,
                ),
                audio=ModalitySpec(
                    context=stage2_actx, conditionings=band_a,
                    noise_scale=noise_scale2, initial_latent=init_a,
                ),
            )
            refined_v.append(vstate2.latent.detach().clone())
            refined_a.append(astate2.latent.detach().clone())
            del vstate2, astate2, init_v, init_a
            if progress is not None:
                progress(STAGE_TILE, i, n_tiles)

        del upscaled_v, assembled_a
        cleanup_memory()

        # ── Reassemble the refined tiles (crossfade over hard-frozen overlaps) ─
        final_v = refined_v[0]
        for i in range(1, n_tiles):
            final_v = _crossfade_concat(final_v, refined_v[i], kt_v)
        final_a = refined_a[0]
        for i in range(1, n_tiles):
            final_a = _crossfade_concat(final_a, refined_a[i], kt_a)
        del refined_v, refined_a
        assert final_v.shape[2] == layout.f_total, (final_v.shape[2], layout.f_total)
        assert final_a.shape[2] == layout.a_total, (final_a.shape[2], layout.a_total)

        # ── ONE VAE decode -> ONE mp4 ─────────────────────────────────────────
        # The tiling config is resolved for the WHOLE timeline (``total_px``),
        # not per tile: this is the decode's own chunking, unrelated to the
        # stage-2 tiles, and sizing it from a tile would under-tile the decode.
        if progress is not None:
            progress(STAGE_DECODE, 0, 1)
        tiling_config = ensure_tiling_config(
            AUTO_TILING,
            scale_factors=tiling_scale_factors_for_vae(dp.video_decoder.checkpoint_path),
            video_shape=VideoPixelShape(1, total_px, height, width, frame_rate),
            vae_checkpoint_path=dp.video_decoder.checkpoint_path,
            diffvae_optimization=dp.video_decoder.diffvae_optimization,
            device=device,
        )
        chunks = get_video_chunks_number(total_px, tiling_config)
        generator = torch.Generator(device=device).manual_seed(base_seed)

        vram.reset()
        decode_started = time.perf_counter()
        decoded_video = dp.video_decoder(final_v, tiling_config, generator)
        decoded_audio = dp.audio_decoder(final_a)
        encode_fps = int(round(frame_rate))
        encode_video(
            video=_decode_progress(decoded_video, chunks, progress),
            fps=encode_fps,
            audio=decoded_audio,
            output_path=str(out_path),
            video_chunks_number=chunks,
        )
        vram.record("30_decode_encode", time.perf_counter() - decode_started)
        del decoded_video, decoded_audio, final_v, final_a

    wall = time.time() - started

    if not out_path.is_file() or out_path.stat().st_size <= 0:
        raise ChainError(f"the chain produced no/empty output: {out_path}")

    gc.collect()
    cleanup_memory()

    # ``vram_peak_mb`` is the highest PER-PHASE peak, not ``max_memory_allocated``
    # read at the end. The per-phase recorder resets CUDA's peak counters around
    # every phase -- which is the point of having phases at all -- so a global
    # read here would report only the decode's peak and quietly understate the
    # job by several GiB. Same decimal-MB unit 2.3's ``vram_peak_mb`` uses, so a
    # client comparing the two engines is comparing the same number.
    summary = _vram_summary(vram.phases)
    peak_gib = summary.get("peak_allocated_gib")
    peak_mb = round(float(peak_gib) * 2 ** 30 / 1e6, 1) if peak_gib is not None else 0.0
    metadata = layout.to_dict()
    metadata.update({
        "n_clips": n,
        "seed": base_seed,
        "seeds": seeds,
        "width": width,
        "height": height,
        "num_steps": int(spec.num_steps),
        "overlap_frames": kv,
        "overlap_strength": float(spec.overlap_strength),
        "wall_s": round(wall, 2),
        "vram_peak_mb": peak_mb,
        "vram_within_16gb": peak_mb < 16000,
        "output_mp4": str(out_path),
        # engine25-only facts, in their own sub-dict so the 2.3-shaped keys above
        # stay exactly the 2.3 set a client already parses.
        "ltx25": {
            "stage1_sampler": sampler,
            "stage1_eta": eta if sampler == "ancestral" else None,
            "stage2_sampler": "euler",
            "clear_keyframes_on_carry": clear_kf,
            "chunked_upsample": bool(spec.chunked_upsample),
            "stage2_window": spec.stage2_window or "standard",
            "encode_fps": encode_fps,
            "video_chunks": chunks,
            "tiling": None if tiling_config is None else repr(tiling_config),
            "size_bytes": out_path.stat().st_size,
            "vram": summary,
            "phases": dict(vram.phases),
        },
    })
    logger.info(
        "CHAIN_OK %.1fs peak=%sMB %d clips / %d tiles -> %s",
        wall, peak_mb, n, n_tiles, out_path,
    )
    return ChainResult(output_path=str(out_path), metadata=metadata)
