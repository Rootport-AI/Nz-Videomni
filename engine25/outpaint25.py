"""Outpainting for LTX 2.5 (§3-102 Outpainting increment, commit C1) -- INACTIVE.

What this is
------------
The 2.5 half of "Outpainting": one clip generated at a canvas larger than the
material, with the four green pad bands invented by the model and the kept
rectangle blended back in pixel space between the two stages.

    text encode
      -> freeze the SOURCE video's audio latent (official note, node 5392:
         "Frozen audio helps guiding outpainting to be consistent with the
         sounds in the video")
      -> encode the GREEN CANVAS as the IC-LoRA reference conditioning
      -> STAGE 1 at half resolution, with that reference attached
      -> decode stage 1 to pixels (half res)
      -> BLEND 1: Laplacian pyramid, dilation 5, against the green canvas
      -> 2x PIXEL upscale
      -> tiled VAE re-encode at full resolution
      -> STAGE 2 at full resolution, no reference conditioning
      -> decode stage 2 to pixels (full res)
      -> BLEND 2: Laplacian pyramid, dilation 2, against the green canvas
      -> mux the source's ORIGINAL waveform -> one mp4.

NOTHING CALLS THIS YET. Commit C1 lands the driver with zero call sites: the
adapter still refuses ``outpaint`` at the API and the worker has no branch for
it, so the shipped behaviour of every existing job is unchanged by
construction, not merely by inspection. The opening is commit C2.

Why a separate module rather than a flag on the single-generate path
--------------------------------------------------------------------
:meth:`engine25.pipeline25.Ltx25Pipeline.generate` delegates into the official
``DistilledPipeline.__call__``, which runs stage 1 and stage 2 back to back
with a LATENT-space upsampler in between. Outpainting has to interrupt exactly
there -- decode to pixels, blend the generated frame with the green canvas,
upscale in PIXEL space, re-encode -- so it needs its own two-stage driver.
:func:`engine25.chain25.run_chain` is the precedent for writing one, and 2.3's
``engine/pipeline/outpaint_pipeline.py`` is the same argument made on the older
wheel. THE PROCEDURE HERE IS 2.3's, which the W2 gate validated; the CODE is
not, because 2.5 drives official 1.2.0 blocks (``prompt_encoder`` /
``audio_conditioner`` / ``image_conditioner`` / ``stage`` / ``video_decoder`` /
``audio_decoder``) instead of a ``ModelLedger``.

WHAT IS BORROWED, AND WHY EACH IS NOT COPIED
--------------------------------------------
Every name below is imported, never restated. The rule is engine25's standing
one -- one definition per fact -- and each line says what would go wrong with a
second copy:

* ``engine.outpaint.canvas.OutpaintGeometry`` / ``build_blend_mask`` /
  ``CANVAS_MULTIPLE`` (via the geometry) -- the API validator, the mock backend
  and the 2.3 engine already share this module, so a copy would let the canvas
  the app validated and the canvas this engine blends disagree.
* ``engine.outpaint.pyramid_blend.blend_video_u8`` -- the blend IS the feature;
  a second implementation would be a second set of seams to gate.
* ``engine25.chain25.DTYPE`` -- the dtype every conditioning latent in this
  engine is built at; a local ``torch.bfloat16`` would drift the day the engine
  moves off bf16.
* ``chain25.AudioBandMask`` -- the ONE piece of conditioning arithmetic
  engine25 owns, pinned against upstream's video twin by
  ``ltxcore_compat.verify``; a copy would not be pinned.
* ``chain25._load_audio_stereo`` / ``_encode_audio_latent`` -- the mono->stereo
  duplication and the "cast the WAVEFORM, not the encoder input" rule are two
  facts a copy would get subtly wrong (the STFT would run in float32).
* ``chain25._stage1_sampler_kwargs`` -- carries the ``eta > 0`` assertion the
  frozen band rests on; a copy could be written without it.
* ``chain25.STAGE1_SAMPLER`` / ``STAGE1_ANCESTRAL_ETA`` / ``STAGE2_SEED_OFFSET``
  -- the same three constants the chain samples with, so "which sampler did 2.5
  use" has one answer per engine rather than one per feature.
* ``chain25._vram_summary`` -- the reduction the app parses on ``done``; a copy
  would be a second definition of ``seconds_total``.
* ``reference25.reference_conditioning_from_pixels`` /
  ``load_reference_pixels_cpu`` / ``reference_pixel_dims`` /
  ``resolve_reference_downscale_factor`` / ``set_conv3d_memory_format`` -- the
  IC-LoRA reference path, whole. Its factor VOTE and its channels_last_3d
  re-layout are both measurements; a copy would be an unmeasured second path.
* ``pipeline25.validate_geometry`` / ``_log_ignored`` / ``_peaks`` /
  ``STAGE_1_DENOISE`` / ``STAGE_2_DENOISE`` / ``STAGE_DECODE`` -- the geometry
  backstop, the ignored-field report and the progress vocabulary the app's
  receipt loop already maps. A second vocabulary would cost a branch app-side.
* ``ltxcore_compat`` (everything from the wheel) -- engine25's ONE entry point
  rule; a direct ``ltx_core`` import here would be a second door for upstream
  to change behind.

Deliberate differences from 2.3, all recorded rather than hidden
---------------------------------------------------------------
* **SAMPLER (M-7).** 2.3 runs plain euler in BOTH stages. This engine runs the
  ANCESTRAL euler in stage 1 -- what the 2.5 distilled checkpoint was distilled
  for, what ``DistilledPipeline`` selects for a single generation, and what
  :data:`chain25.STAGE1_SAMPLER` already uses. It is also nearer the official
  workflow (``euler_ancestral_cfg_pp``). Determinism is unaffected: the
  ancestral loop's own noise generator is seeded at
  ``seed + ANCESTRAL_NOISE_SEED_OFFSET``. Stage 2 is deterministic euler in
  both engines (neither ``stepper`` nor ``loop`` is passed, which is how
  ``DiffusionStage``'s own default is said).
* **STAGE-2 AUDIO INIT.** 2.3 rebuilds stage 2's audio latent from zeros with
  the frozen head copied in. This engine CARRIES stage 1's audio output and
  writes the frozen head back over it, then re-freezes -- ``chain25``'s rule
  (stage 2 always starts from stage 1's audio) applied here. With a fully
  frozen head the two are identical; they can only differ over a PARTIAL
  freeze, where 2.3 hands stage 2 zeros for the un-frozen tail and this hands
  it stage 1's generated audio. Recorded as
  ``stage2_audio_init_policy: "stage1_carry"`` so gate O7's judgement is made
  against this policy and NOT against 2.3's absolute numbers (M-2).
* **RE-ENCODE TILING (G0-d).** Both engines hand the full-resolution re-encode
  the DECODE's resolved tiling config. 2.3 additionally narrows the spatial
  tiles for its retake-class encodes; this engine does not, and instead wraps
  the encode in ``set_conv3d_memory_format`` -- measured at 27188 MB -> 6174 MB
  reserved for -0.002 dB, where the tile budget cost -0.378 dB. See
  ``outputs/ltx25-outpaint-prep/RESULTS.md`` §1.
* **PIXEL DOMAIN.** 2.3's video VAE decode already yielded uint8; 2.5's yields
  float ``[0, 1]`` in ``(F, H, W, C)``, and its mp4 encoder WANTS float
  ``[0, 1]`` where 2.3's wanted a materialised uint8 tensor. The four
  conversion points are named in :func:`run_outpaint`'s body (C-1).

What this module deliberately does NOT do
-----------------------------------------
* **It never clears the keyframe marker**, in either stage. ``ClearKeyframesMask``
  exists for a clip that starts on CARRIED-OVER content, where latent 0 covers
  eight pixel frames and the "this is a single pixel frame" marker is a lie.
  Neither stage here is such a clip: stage 1 generates from noise with no
  frozen video head at all (``freeze_kv == 0``), and stage 2 starts from a
  FRESH causal encode of the blended pixels at timeline position 0 -- the same
  two facts that make :data:`chain25.CLEAR_KEYFRAMES_ON_RETAKE_HEAD` ``False``.
  Clearing it would throw away a true fact about the tensor.
* **It adds no progress phase.** The vocabulary is the four names the app
  already maps (``encode`` / ``stage1_denoise`` / ``stage2_denoise`` /
  ``decode``) and no chain position is announced, so a receiver that has only
  ever seen a single generation sees exactly the calls it has always seen.
  KNOWN CONSEQUENCE (M-3): the app's SINGLE-generate fraction maps
  ``stage1_denoise`` to 0.06..0.50 and ``stage2_denoise`` to 0.50..0.85 and
  ignores ``decode`` entirely, so the bar STANDS STILL at 0.50 for the whole of
  decode 1 + blend 1 + upscale + re-encode. That is a long, silent stretch at
  production resolution. It is documented rather than fixed: a new phase name
  would need an app-side branch, which this theme does not touch.
* **It does not carry its own metadata into ``metadata.json``.** ``done``'s
  additive keys and the worker log are where the outpaint facts land; putting
  them in ``metadata.json`` needs an app-layer change and is out of scope.
  :class:`OutpaintResult` is a superset of
  :class:`~engine25.pipeline25.GenerationResult` precisely so the numbers the
  app DOES store (``vram_optimization.peak_vram_mb``) still get filled in.

Run notes
---------
One job at a time, like everything else in this engine. The two
``ImageConditioner`` calls (the reference encode and the re-encode) each build
and free the video encoder; G0-d measured the second build at 0.34-0.47 s and
proved the two calls bit-identical, so the lifecycle is left alone.
"""

from __future__ import annotations

import gc
import logging
import math
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Callable

import torch

from engine.outpaint.canvas import OutpaintGeometry, build_blend_mask
from engine.outpaint.pyramid_blend import blend_video_u8
from engine25.chain25 import (
    DTYPE,
    STAGE1_ANCESTRAL_ETA,
    STAGE1_SAMPLER,
    STAGE2_SEED_OFFSET,
    AudioBandMask,
    _encode_audio_latent,
    _load_audio_stereo,
    _stage1_sampler_kwargs,
    _vram_summary,
)
from engine25.ltxcore_compat import (
    AUTO_TILING,
    DISTILLED_SIGMAS,
    STAGE_2_DISTILLED_SIGMAS,
    Audio,
    AudioLatentShape,
    ConditioningItem,
    GaussianNoiser,
    ModalitySpec,
    SimpleDenoiser,
    VideoPixelShape,
    cleanup_memory,
    decode_video_by_frame,
    encode_video,
    ensure_tiling_config,
    get_video_chunks_number,
    resize_and_center_crop,
    tiling_scale_factors_for_vae,
)
from engine25.pipeline25 import (
    STAGE_1_DENOISE,
    STAGE_2_DENOISE,
    STAGE_DECODE,
    GenerationResult,
    _log_ignored,
    _peaks,
    validate_geometry,
)
from engine25.reference25 import (
    REFERENCE_ENCODE_PHASE,
    load_reference_pixels_cpu,
    reference_conditioning_from_pixels,
    reference_pixel_dims,
    resolve_reference_downscale_factor,
    set_conv3d_memory_format,
)

logger = logging.getLogger(__name__)

#: ``progress(stage, index, total)``. Deliberately the THREE-POSITIONAL form:
#: outpainting is one clip, so there is no chain position to announce, and a
#: receiver written for a single generation must keep seeing the call it has
#: always seen (see the module docstring).
ProgressFn = Callable[..., None]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Stage-2 sigma schedule: the wheel's own stage-2 ladder WITH ITS FIRST RUNG
#: REMOVED.
#:
#: DERIVED, NOT TRANSCRIBED, and that is the whole point of this line. 2.3
#: writes ``[0.725, 0.421875, 0.0]`` as a literal; here the same three numbers
#: are a SLICE of the constant the wheel ships, so an upstream that re-tunes the
#: distilled schedule moves this with it instead of leaving a stale literal
#: behind. The two engines' ``constants.py`` files are NOT identical, but the
#: two schedule lines in them are, which is what makes the slice legitimate
#: rather than a coincidence:
#:
#:     DISTILLED_SIGMA_VALUES[-4:] == STAGE_2_DISTILLED_SIGMA_VALUES
#:                                 == [0.909375, 0.725, 0.421875, 0.0]
#:
#: WHY DROP THE FIRST RUNG. The official outpaint workflow's ManualSigmas node
#: (5211) starts one step LOWER than the stock stage 2, at 0.725, and the
#: difference is load-bearing rather than cosmetic: stage 2's initial latent is
#: the RE-ENCODE of the BLENDED pixels, so the noise level it starts from
#: decides how much of the blend survives. Starting where the stock schedule
#: does (0.909375) would partly re-generate the very seam blend 1 just fixed.
#:
#: ``.tolist()`` IS MANDATORY, not stylistic: ``STAGE_2_DISTILLED_SIGMAS`` is a
#: ``torch.Tensor`` and ``[1:]`` on a tensor is a VIEW. Holding a module-global
#: view would keep the wheel's constant tensor alive, would put a tensor in the
#: job metadata, and -- worst -- would let anything that mutated the wheel's
#: tensor in place change this module's schedule. A plain ``list[float]`` cannot.
OUTPAINT_STAGE2_SIGMAS: list[float] = STAGE_2_DISTILLED_SIGMAS[1:].tolist()

#: The first rung, asserted rather than assumed. The slice above is only
#: correct while the ladder it slices still starts where the workflow says; an
#: upstream re-tune that moved 0.725 would otherwise change every outpaint job
#: silently. ``math.isclose`` rather than ``==`` because the value arrives
#: through a float32 tensor.
assert math.isclose(OUTPAINT_STAGE2_SIGMAS[0], 0.725, abs_tol=1e-6), OUTPAINT_STAGE2_SIGMAS
assert OUTPAINT_STAGE2_SIGMAS[-1] == 0.0, OUTPAINT_STAGE2_SIGMAS

#: Frames per Laplacian-pyramid blend chunk, and per upscale/encode chunk.
#: Blending pads each frame out to the next power of two per side (960x544 ->
#: 1024x1024, 1920x1152 -> 2048x2048), so VRAM scales with this directly.
#: Overridable for the GPU gate without an API field, because it changes nothing
#: about the result. DELIBERATELY THE SAME VARIABLE NAME 2.3 uses, so an
#: operator who learned it on one engine does not have to learn it twice.
_BLEND_CHUNK_ENV = "LTX_OUTPAINT_BLEND_CHUNK"

#: Default frames per pixel-space chunk (upscale, and the mp4 encoder's feed).
#: 2.3's value. Small on purpose: at 1920x1152 a chunk of 8 frames is 212 MB as
#: float32, and the whole point of the uint8 timelines is that nothing bigger
#: than a chunk is ever float.
_PIXEL_CHUNK_FRAMES = 8

#: What stage 2's audio latent is initialised FROM. Recorded in the metadata
#: because it is an INTENTIONAL DIFFERENCE from 2.3 (see the module docstring),
#: and gate O7's partial-freeze arm must be judged against this policy rather
#: than against 2.3's absolute numbers.
STAGE2_AUDIO_INIT_POLICY = "stage1_carry"


class OutpaintError(RuntimeError):
    """An outpaint request the LTX 2.5 engine cannot run."""


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OutpaintResult:
    """What one :func:`run_outpaint` produced.

    A STRICT SUPERSET of :class:`engine25.pipeline25.GenerationResult` -- same
    field names, same meanings, same :meth:`as_dict` keys -- plus
    :attr:`metadata`. That is a contract, not a convenience, and
    ``tests/test_ltx25_outpaint.py`` derives the field list from
    ``GenerationResult`` itself rather than restating it.

    WHY IT HAS TO BE. ``engine25/worker.py``'s ``done`` event reads eight
    attributes off the generate result by name (``peak_allocated_gib``,
    ``peak_reserved_gib``, ``rss_peak_gib``, ``seconds``, ``num_frames``,
    ``encode_fps``, ``size_bytes``, ``phases``) and logs a ninth thing,
    ``result.as_dict()``, as the ``GENERATE_REPORT`` line. The app then stores
    ``peak_vram_mb`` into ``metadata.json``'s ``vram_optimization`` block, which
    is what the job-VRAM gate reads. An outpaint result that answered to fewer
    names would leave that number empty and the gate blind -- so the C2 worker
    branch can hand this object to exactly the same ``done`` builder.

    :attr:`metadata` is the 2.3-shaped job metadata dict (the ``outpaint``
    sub-dict, the flat geometry/timing keys, and the ``ltx25`` sub-dict), the
    same shape :class:`engine25.chain25.ChainResult` carries. It rides on
    ``done`` as an additive key and in the worker log; it does NOT reach
    ``metadata.json``, which would need an app-layer change (out of scope --
    see the module docstring).
    """

    output_path: str
    seed: int
    width: int
    height: int
    num_frames: int
    frame_rate: float
    encode_fps: int
    num_images: int
    size_bytes: int
    seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)
    phases: dict[str, dict[str, Any]] = field(default_factory=dict)
    peak_allocated_gib: float | None = None
    peak_reserved_gib: float | None = None
    rss_peak_gib: float | None = None
    video_chunks: int = 1
    tiling: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """``GenerationResult.as_dict()``'s keys, in its order, plus ``outpaint``.

        The order is copied deliberately: the ``GENERATE_REPORT`` line is read
        by eye as often as by machine, and a report whose first twelve keys sit
        where a plain generation's do is one a reader can compare across job
        kinds without re-learning it.
        """
        return {
            "output_path": self.output_path,
            "seed": self.seed,
            "width": self.width,
            "height": self.height,
            "num_frames": self.num_frames,
            "frame_rate": self.frame_rate,
            "encode_fps": self.encode_fps,
            "num_images": self.num_images,
            "size_bytes": self.size_bytes,
            "seconds": round(self.seconds, 2),
            "video_chunks": self.video_chunks,
            "tiling": self.tiling,
            "peak_allocated_gib": self.peak_allocated_gib,
            "peak_reserved_gib": self.peak_reserved_gib,
            "rss_peak_gib": self.rss_peak_gib,
            "phases": self.phases,
            # The one ADDITIVE key. Named for the feature rather than folded in
            # flat, so a reader (and a JSON schema) can tell the generation
            # contract from this job kind's own facts.
            "outpaint": self.metadata,
        }


# The superset claim, checked at import time as well as in the unit test: a
# field added to GenerationResult and forgotten here would otherwise be found
# only by the worker, at run time, on a real job.
assert {f.name for f in fields(GenerationResult)} <= {f.name for f in fields(OutpaintResult)}, (
    sorted({f.name for f in fields(GenerationResult)} - {f.name for f in fields(OutpaintResult)})
)


# ---------------------------------------------------------------------------
# Pixels: the four uint8 <-> float boundaries (C-1)
# ---------------------------------------------------------------------------
#
# 2.5's pixel domain is NOT 2.3's, and every one of the four crossings below was
# measured in C0-prep before a line of this module existed
# (``outputs/ltx25-outpaint-prep/RESULTS.md`` §4):
#
#   (1) the video VAE DECODE yields float ``[0, 1]`` chunks shaped
#       ``(F, H, W, C)`` -- 2.3's yielded uint8, so a straight port would have
#       delivered a video 255x too dark;
#   (2) the video VAE ENCODE wants ``(1, C, F, H, W)`` in ``[-1, 1]``
#       (``x / 127.5 - 1``), which is 2.3's own expression;
#   (3) the mp4 ENCODER wants float ``[0, 1]`` ``(F, H, W, C)`` -- 2.3's wanted
#       a materialised uint8 tensor;
#   (4) a video FILE decodes to ``(1, H, W, C)`` uint8 through
#       ``decode_video_by_frame``, where 2.3's decoder gave ``(1, C, 1, H, W)``
#       float ``[0, 255]``.
#
# Everything BETWEEN those crossings is uint8, and that is a memory decision,
# not a style one: a 1920x1152x241 timeline is 1.6 GB as uint8 against 6.4 GB as
# float32, and the blend needs three such buffers at once.


def _resolve_stage2_sigmas(stage2_sigmas: list[float] | None) -> list[float]:
    """The stage-2 schedule this job runs: the override, or the module default.

    A function rather than an inline ``or`` because it is the ONE place the
    override arm (gate O4's three-step schedule) and the shipped default meet,
    and because the values are re-read downstream in TWO places -- the sigma
    tensor handed to the stage, and ``noise_scale`` (C-4). Both read this list,
    so an override cannot move one without the other.

    Copied to ``float`` rather than passed through: a caller's list must not be
    able to change under the job, and the metadata must carry plain floats
    rather than whatever numeric type arrived.
    """
    if stage2_sigmas:
        return [float(v) for v in stage2_sigmas]
    return list(OUTPAINT_STAGE2_SIGMAS)


def _blend_chunk_size() -> int | None:
    """``LTX_OUTPAINT_BLEND_CHUNK`` as a positive int, or ``None`` for the default."""
    raw = os.environ.get(_BLEND_CHUNK_ENV, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        logger.warning("%s=%r is not an integer; using the default", _BLEND_CHUNK_ENV, raw)
        return None
    return value if value > 0 else None


def _require_frames(what: str, got: int, need: int, *, source: str) -> None:
    """The frame-shortfall detector, said once for both operands.

    NAMED AND LOUD on purpose. The app layer already rejects a source shorter
    than ``num_frames`` and ``services.video_io.pad_green_mp4`` clone-pads the
    tail as a backstop, so a shortfall here means one of those two failed. The
    alternative to raising is worse than a failed job: stage 2's
    ``create_initial_state`` would assert instead, several minutes later,
    naming a latent shape rather than the file that was short.
    """
    if int(got) < int(need):
        raise ValueError(
            f"outpaint frame shortfall: {what} yielded {int(got)} frames but the job needs "
            f"{int(need)} ({source}). The canvas must be built with pad_green_mp4's "
            f"exact-frame-count guarantee."
        )


def _canvas_u8(
    *, video_path: str, height: int, width: int, frame_cap: int, device: torch.device
) -> torch.Tensor:
    """Decode ``video_path`` to ``(F, H, W, 3)`` uint8 on the CPU. Boundary (4).

    A uint8 twin of :func:`engine25.chain25._load_video_frames_cpu`, running the
    SAME per-frame op on the same device (``resize_and_center_crop`` on float32)
    but storing the result as uint8 instead of normalising to ``[-1, 1]``. It is
    one of the two blend operands and the blend is uint8-only, so converting
    here would be converting twice.

    NO FRAME-RATE CHECK, unlike the V2V/retake loader it is otherwise a twin of.
    The canvas is written by ``pad_green_mp4`` from the very source this job was
    validated against, at the rate the app resolved; a rate check here would be
    checking the app against itself. What IS checked is the frame COUNT, which
    is the one thing that can silently go wrong (see :func:`_require_frames`).
    """
    frames: list[torch.Tensor] = []
    for raw in decode_video_by_frame(path=str(video_path), device=device, frame_cap=int(frame_cap)):
        # (1, H, W, C) uint8 -> (1, C, 1, H, W) float32 -> (H, W, C) uint8
        frame = resize_and_center_crop(raw.to(torch.float32), int(height), int(width))
        frame = frame.round().clamp(0, 255).to(torch.uint8)
        frames.append(frame[0].permute(1, 2, 3, 0)[0].cpu())
        del raw, frame
    if not frames:
        raise ValueError(f"outpaint canvas decoded to 0 frames: {video_path}")
    _require_frames(
        "the green canvas", len(frames), int(frame_cap), source=str(video_path)
    )
    return torch.stack(frames, dim=0)


def _decoded_to_u8(
    decoded: Any, *, progress: ProgressFn | None = None, chunks: int = 1
) -> torch.Tensor:
    """Materialise a video decode as ``(F, H, W, 3)`` uint8 on the CPU. Boundary (1).

    ``decoded`` is the LAZY iterator ``VideoDecoder.__call__`` returns: one
    temporally-tiled ``(f, H, W, C)`` float ``[0, 1]`` chunk at a time, in the
    decoder's own dtype (bf16 on this engine). Each chunk is quantised AS IT
    ARRIVES and moved to the CPU, so the GPU never holds more than one chunk and
    the host never holds a float copy of the timeline.

    THE PROMOTION TO float32 BEFORE THE MULTIPLY IS DELIBERATE. bf16 carries 8
    mantissa bits, so ``x * 255`` in bf16 rounds to a grid 0.5 wide near the top
    of the range; every integer is still representable, so ``.round()`` would
    survive it, but the rounding would happen TWICE (once into bf16, once into
    the integer) and the two can disagree by one code. One promotion per chunk
    costs a temporary the size of a chunk and removes the question.

    Progress is reported per chunk, which is progress about the DECODE itself
    rather than a re-walk of finished work -- the same seam
    ``Ltx25Pipeline._with_decode_progress`` uses.
    """
    if torch.is_tensor(decoded):
        decoded = [decoded]
    out: list[torch.Tensor] = []
    done = 0
    for chunk in decoded:
        out.append(
            chunk.detach()
            .to(torch.float32)
            .mul(255.0)
            .round()
            .clamp(0.0, 255.0)
            .to(torch.uint8)
            .cpu()
        )
        del chunk
        done += 1
        if progress is not None:
            progress(STAGE_DECODE, done - 1, max(1, int(chunks)))
    if not out:
        raise ValueError("the video decode yielded no chunks")
    return torch.cat(out, dim=0)


def _upscale_u8(
    pixels: torch.Tensor, *, height: int, width: int, device: torch.device, step: int
) -> torch.Tensor:
    """2x pixel upscale of a ``(F, H, W, 3)`` uint8 timeline, chunk by chunk.

    Bicubic, where the official workflow uses lanczos: torch has no lanczos
    kernel and bicubic is the closest windowed-sinc-like resampler available.
    Recorded as an intentional difference and A/B'd in the GPU gate, exactly as
    in 2.3.

    CHUNK-INVARIANT BY CONSTRUCTION, and the unit test pins it: the resample is
    SPATIAL only, so frames do not see each other and the chunk size is a
    memory knob with no effect on the bytes produced. That matters because the
    knob is reachable from the environment -- an operator narrowing it for a
    16 GB card must not be changing the deliverable.
    """
    frames = int(pixels.shape[0])
    out = torch.empty((frames, int(height), int(width), 3), dtype=torch.uint8, device="cpu")
    step = max(1, int(step))
    for start in range(0, frames, step):
        end = min(start + step, frames)
        block = pixels[start:end].to(device).permute(0, 3, 1, 2).to(torch.float32)
        block = torch.nn.functional.interpolate(
            block, size=(int(height), int(width)), mode="bicubic", align_corners=False
        )
        block = block.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
        out[start:end] = block.cpu()
        del block
    return out


def _encode_input_from_u8(pixels: torch.Tensor) -> torch.Tensor:
    """``(F, H, W, 3)`` uint8 -> ``(1, C, F, H, W)`` :data:`DTYPE` in ``[-1, 1]``. Boundary (2).

    2.3's expression, character for character, because it is the same VAE input
    convention (``normalize_images``: ``x / 127.5 - 1``) and a second spelling
    of one formula is a second chance to get a sign wrong.

    STAYS ON THE CPU. ``VideoEncoder.tiled_encode`` moves each tile to the
    compute device itself, so materialising the whole timeline on the GPU would
    allocate the very gigabytes the tiling exists to avoid.
    """
    return (pixels.permute(3, 0, 1, 2).unsqueeze(0).to(torch.float32) / 127.5 - 1.0).to(DTYPE)


def _encode_chunk_count(frames: int, step: int) -> int:
    """How many chunks :func:`_iter_encode_chunks` will yield for ``frames``."""
    step = max(1, int(step))
    return max(1, -(-int(frames) // step))


def _iter_encode_chunks(pixels: torch.Tensor, step: int) -> Iterator[torch.Tensor]:
    """Feed the mp4 encoder float ``[0, 1]`` chunks off a uint8 timeline. Boundary (3).

    A GENERATOR, not a converted tensor, and that is the whole reason this
    function exists: ``encode_video`` accepts either, and handing it one
    full-resolution float32 timeline would be a 6.4 GB host allocation at
    1920x1152x241 -- on top of the uint8 original, which is still live because
    the generator is reading from it.

    ``.to(torch.float32)`` on a uint8 tensor always COPIES, which is what makes
    the in-place divide safe; the caller's uint8 timeline is never touched.

    The chunk COUNT must equal the ``video_chunks_number`` handed alongside it
    (the encoder drives its progress bar off that number), which is why the
    count has its own function -- :func:`_encode_chunk_count` -- rather than
    being counted by a caller that could disagree.
    """
    frames = int(pixels.shape[0])
    step = max(1, int(step))
    for start in range(0, frames, step):
        yield pixels[start : min(start + step, frames)].to(torch.float32).div_(255.0)


# ---------------------------------------------------------------------------
# Audio: one freeze mechanism, and its proof
# ---------------------------------------------------------------------------


def _audio_freeze(
    audio_latent: torch.Tensor | None, frames_frozen: int, *, strength: float = 1.0
) -> list[ConditioningItem]:
    """``[AudioBandMask]`` over the LEADING ``frames_frozen`` latent frames, or ``[]``.

    ONE MECHANISM FOR BOTH STAGES, and deliberately the band rather than
    ``ModalitySpec(frozen=True)``. ``frozen=True`` freezes the WHOLE modality
    and additionally zeroes its scalar sigma -- the right statement for A2V,
    where the audio IS the subject. Here the audio is GUIDANCE for a video that
    is already fully specified, and a source whose audio ends early must freeze
    what it has and generate the rest. A band expresses both the full and the
    partial case in one expression, and at ``strength=1.0`` the band's
    ``denoise_mask = denoise_mask * (1-m) + (1 - strength) * m`` is exactly
    2.3's ``mask_value=0.0``, so the two engines freeze the same way.

    THE LATENT IS THE FULL-LENGTH ONE, never the frozen head alone:
    :class:`~engine25.chain25.AudioBandMask` patchifies ``latent`` and ``mask``
    against the state's whole token axis, so a short tensor would not broadcast.
    The MASK is what selects the head.

    Returns a list because that is what a ``ModalitySpec`` takes, and because
    "nothing is frozen" is then the empty list -- inert, with no branch at the
    call site.
    """
    if audio_latent is None or int(frames_frozen) <= 0:
        return []
    total = int(audio_latent.shape[2])
    n = min(int(frames_frozen), total)
    mask = torch.zeros(
        (int(audio_latent.shape[0]), total),
        dtype=torch.float32,
        device=audio_latent.device,
    )
    mask[:, :n] = 1.0
    return [AudioBandMask(latent=audio_latent, mask=mask, strength=float(strength))]


def _mad(a: torch.Tensor, b: torch.Tensor) -> float:
    """Max absolute difference, in float32, as a plain float."""
    return float((a.float() - b.float()).abs().max().item())


def _freeze_proof(
    *,
    frozen_audio: torch.Tensor | None,
    s1_audio_head: torch.Tensor | None,
    s2_audio_head: torch.Tensor | None,
) -> dict[str, float | bool | None]:
    """Did the audio band actually hold? Two numbers and a verdict.

    ``s1_audio_head`` / ``s2_audio_head`` are what came OUT of each denoise over
    the frozen span; ``frozen_audio`` is what went IN. Both differences MUST be
    exactly 0.0, and exactly rather than approximately is attainable for the
    reason the retake proof gives: every write is a bf16 copy of the encoder's
    own output and a denoise mask of 0.0 returns those tokens untouched.

    ``None`` -- NEVER 0.0 -- when nothing was frozen. There is nothing to
    compare, and an invented zero would fake a passing proof of exactly the
    thing being proven. ``pass`` is then ``None`` too: "no verdict", not "yes".

    OBSERVATION ONLY; this never raises. A non-zero number means degraded audio
    guidance, not a corrupt job, and turning a metadata probe into a new crash
    path would be the worse trade.
    """
    if frozen_audio is None or s1_audio_head is None or s2_audio_head is None:
        return {"s1_audio_head": None, "s2_audio_head": None, "pass": None}
    checks: dict[str, float | bool | None] = {
        "s1_audio_head": _mad(s1_audio_head, frozen_audio),
        "s2_audio_head": _mad(s2_audio_head, frozen_audio),
    }
    checks["pass"] = checks["s1_audio_head"] == 0.0 and checks["s2_audio_head"] == 0.0
    return checks


def _mux_plan(
    *,
    freeze_source_audio: bool,
    source_waveform: torch.Tensor | None,
    audio_frozen_frames: int,
) -> dict[str, Any]:
    """Which of the THREE audio outcomes this job has, as metadata.

    Three branches, two outcomes, and the branch is worth recording even where
    the outcome is shared:

    * ``frozen_source_waveform`` -- the source's own recording is muxed VERBATIM
      and the vocoder never runs. Not an optimisation: the delivered track IS
      the user's sound, so rendering the audio latent would spend a vocoder pass
      on a waveform that is then discarded AND would leave the output one
      round-trip away from the input.
    * ``generated_freeze_disabled`` -- the caller asked for ``freeze=False``, so
      the model invented audio for the widened frame and the vocoder renders it.
    * ``generated_no_source_audio`` -- the caller asked to freeze but the source
      has no decodable audio stream. Same outcome as above, DIFFERENT cause, and
      a gate reading "vocoder ran" needs to know which.
    """
    if source_waveform is not None and int(audio_frozen_frames) > 0:
        return {
            "audio_branch": "frozen_source_waveform",
            "muxed_original_waveform": True,
            "vocoder_skipped": True,
        }
    if not freeze_source_audio:
        return {
            "audio_branch": "generated_freeze_disabled",
            "muxed_original_waveform": False,
            "vocoder_skipped": False,
        }
    return {
        "audio_branch": "generated_no_source_audio",
        "muxed_original_waveform": False,
        "vocoder_skipped": False,
    }


def _phase_peak_mb(phases: dict[str, dict[str, Any]], name: str) -> float | None:
    """One phase's peak allocation in decimal MB, or ``None`` if it was not recorded.

    2.3 reports ``blend1_peak_vram_mb`` / ``blend2_peak_vram_mb`` by reading
    ``torch.cuda.max_memory_allocated`` at the two blend sites. This engine
    already records a per-phase peak around each of them, so the same two
    numbers are a lookup rather than a second measurement -- and they carry the
    same unit (decimal MB) so a client comparing the engines compares the same
    number.
    """
    entry = phases.get(name)
    if not entry:
        return None
    gib = entry.get("peak_allocated_gib")
    return None if gib is None else round(float(gib) * 2**30 / 1e6, 1)


# ---------------------------------------------------------------------------
# The job
# ---------------------------------------------------------------------------


@torch.inference_mode()
def run_outpaint(  # noqa: PLR0913, PLR0915 -- one linear procedure; splitting it would hide the order
    pipeline: Any,
    *,
    prompt: str,
    canvas_path: str,
    source_path: str | None,
    geometry: OutpaintGeometry,
    num_frames: int,
    frame_rate: float,
    num_steps: int,
    seed: int,
    output_path: str,
    ic_loras: list[tuple] | None = None,
    ic_reference: tuple[str, float] | None = None,
    ic_attention_strength: float = 1.0,
    blend_dilation_stage1: int = 5,
    blend_dilation_stage2: int = 2,
    freeze_source_audio: bool = True,
    stage2_sigmas: list[float] | None = None,
    ignored: dict[str, Any] | None = None,
    progress: ProgressFn | None = None,
) -> OutpaintResult:
    """Extend ``canvas_path``'s footage into its green pad band -> one mp4.

    ``pipeline`` is an :class:`engine25.pipeline25.Ltx25Pipeline` -- the loaded
    model, whose official blocks this function drives directly instead of going
    through ``DistilledPipeline.__call__``.

    ``canvas_path`` is the lossless, video-only green canvas built by
    ``services.video_io.pad_green_mp4``; ``ic_reference`` points at that SAME
    file (the app substitutes it for the uploaded reference, so the IC-LoRA
    plumbing needs no changes at all). ``source_path`` is the ORIGINAL upload
    and is read only for its audio, because the canvas is written without an
    audio stream on purpose.

    ``geometry`` describes the canvas and its four pads, and is the ONLY source
    of the output resolution -- there is no ``width``/``height`` argument, so
    the canvas the app validated and the canvas this generates cannot disagree.
    ``num_frames`` / ``frame_rate`` are the generation timeline, which the app
    has already checked the source is long enough for.

    ``num_steps`` is carried and reported but never acted on, exactly as in the
    chain: the distilled schedule is fixed at 8 + 3 sigmas, and stage 2's is
    further fixed at :data:`OUTPAINT_STAGE2_SIGMAS`.

    ``stage2_sigmas`` is an ENGINE-INTERNAL experiment knob, not a request
    field: gate O4's three-step arm runs against the shipped code path rather
    than a copy of it. ``None`` -- every real job -- uses the module constant.

    ``ignored`` is the subset of the request this engine drops; it is reported
    through the same :func:`~engine25.pipeline25._log_ignored` a plain
    generation uses, so a job's log names every knob that had no effect.

    Returns an :class:`OutpaintResult` -- a superset of the plain generation's
    result, so the worker's ``done`` builder needs no branch.
    """
    # ── Geometry: the app's rules, restated by the two owners of them ─────────
    # ``validate_geometry`` is the two-stage pipeline's own backstop (multiples
    # of 64, 8n+1 frames) and ``geometry.validate()`` is the canvas module's
    # (multiples of 128, a non-empty pad, a kept rectangle the blend cannot
    # eat). BOTH, because they check different things and each is the last line
    # of defence for its own: a payload that reached the engine another way --
    # the selftest CLI, a future MCP tool -- has passed neither.
    width = int(geometry.canvas_width)
    height = int(geometry.canvas_height)
    validate_geometry(width, height, int(num_frames))
    geometry.validate()
    _log_ignored(ignored)

    dp = pipeline.pipeline            # the official DistilledPipeline
    stage = pipeline.stage            # Ltx25ProgressStage
    device: torch.device = pipeline.device
    vram = pipeline.vram

    encode_fps = int(round(float(frame_rate)))
    if encode_fps < 1:
        raise OutpaintError(f"frame_rate={frame_rate} rounds to {encode_fps} fps")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lora_entries = list(ic_loras or [])
    # Resolved BEFORE any model is built: it is a header read, and discovering
    # "this LoRA declares no reference factor" two minutes into a job would be a
    # poor trade. It also RAISES when a reference arrives with no adapters,
    # which is the one combination the outpaint contract cannot run -- the
    # official workflow has no LoRA-free outpainting path at all.
    reference_factor = resolve_reference_downscale_factor(lora_entries, ic_reference)

    stage2_values = _resolve_stage2_sigmas(stage2_sigmas)
    chunk_size = _blend_chunk_size()
    pixel_step = chunk_size or _PIXEL_CHUNK_FRAMES

    full_shape = VideoPixelShape(1, int(num_frames), height, width, float(frame_rate))
    a_total = int(AudioLatentShape.from_video_pixel_shape(full_shape).to_torch_shape()[2])

    logger.info(
        "outpaint %dx%d canvas (inner %dx%d at %d,%d) / %d frames @ %.3f fps "
        "(encode %d fps) seed=%d loras=%d reference=%s dilation=%d/%d "
        "freeze_audio=%s stage2_sigmas=%s -> %s",
        width, height, geometry.inner_width, geometry.inner_height,
        geometry.inner_x, geometry.inner_y, int(num_frames), float(frame_rate),
        encode_fps, int(seed), len(lora_entries),
        "no" if ic_reference is None
        else f"{Path(str(ic_reference[0])).name} strength={float(ic_reference[1]):.3f} "
             f"factor={reference_factor} attn={float(ic_attention_strength):.3f}",
        int(blend_dilation_stage1), int(blend_dilation_stage2),
        bool(freeze_source_audio), stage2_values, out_path,
    )

    # set_loras BEFORE begin_job, and unconditionally: begin_job releases the
    # PREVIOUS job's attachment, and a job that asks for no adapter must clear
    # rather than inherit one. Same two lines, same order, as the single and
    # chain paths -- the stage owns that state and this function keeps no second
    # copy of it.
    stage.set_loras(lora_entries)
    stage.begin_job()
    stage.progress = progress
    pipeline.prompt_encoder.progress = progress
    vram.phases.clear()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    started = time.time()

    with torch.no_grad():
        # ── 10_prompt_encode ──────────────────────────────────────────────────
        # ONE prompt, ONE call. The phase, its two sub-phases and the ``encode``
        # progress events are all emitted by ``Ltx25PromptEncoder`` itself,
        # which is why nothing is recorded here: they fire around the actual
        # Gemma load rather than around this function's idea of when it happens.
        (encoded,) = pipeline.prompt_encoder([str(prompt)])
        video_ctx, audio_ctx = encoded.video_encoding, encoded.audio_encoding
        del encoded

        # ── 12_audio_conditioning: freeze the SOURCE video's audio ────────────
        # BEFORE the video encoder and the transformer, because the audio
        # encoder is the cheapest model in the job (~46 MB) and doing it here
        # means it is never resident alongside either of them --
        # ``chain25``'s ordering, for its reason.
        #
        # UNDERRUN IS NOT FATAL. The chain's A2V path errors out because there
        # the audio IS the subject; here it is guidance for a video that is
        # already fully specified, so a source whose audio ends early freezes
        # what it has and generates the rest. 2.3 ruled the same way.
        frozen_audio: torch.Tensor | None = None
        source_waveform: torch.Tensor | None = None
        audio_sr = 0
        audio_available = 0
        n_frozen = 0
        source_had_audio = False
        if freeze_source_audio and source_path:
            loaded = _load_audio_stereo(str(source_path), device)
            source_had_audio = loaded is not None
            if loaded is None:
                logger.warning(
                    "outpaint: %s has no decodable audio stream; the model will generate "
                    "audio for the widened frame instead of following it",
                    source_path,
                )
            else:
                waveform, audio_sr = loaded
                # The ORIGINAL waveform, kept on the CPU for the mux. Never a
                # vocoder render: the delivered track has to BE the source's.
                source_waveform = (
                    waveform.squeeze(0).detach().to(torch.float32).cpu().contiguous()
                )
                vram.reset()
                audio_started = time.perf_counter()
                encoded_a = pipeline.audio_conditioner(
                    lambda enc: _encode_audio_latent(enc, waveform, audio_sr)
                )
                vram.record("12_audio_conditioning", time.perf_counter() - audio_started)
                audio_available = int(encoded_a.shape[2])
                n_frozen = min(a_total, audio_available)
                frozen_audio = encoded_a[:, :, :n_frozen].detach().clone().to(DTYPE)
                if n_frozen < a_total:
                    logger.warning(
                        "outpaint: source audio covers %d of %d audio latent frames; "
                        "the tail will be generated",
                        n_frozen, a_total,
                    )
                del encoded_a, waveform
                cleanup_memory()
            del loaded

        # ── 11_reference_encode: the green canvas, at HALF resolution ─────────
        # ``ImageConditioner.__call__(fn)`` builds the video encoder, calls
        # ``fn(encoder)`` once and frees it again, so the closure is where the
        # encoder exists. This is the FIRST of the job's two builds; the
        # re-encode below is the second, and G0-d measured the extra build at
        # 0.34-0.47 s with bit-identical output, which is why the lifecycle is
        # left exactly as upstream wrote it.
        #
        # THE TILING CONFIG IS THE DECODE'S, HANDED OVER RESOLVED. Resolving it
        # here, with no models built, is also where the (unused, conv-VAE)
        # free-memory branch would be at its most optimistic -- the same hoist
        # ``run_chain`` makes and for the same reason.
        tiling_config = ensure_tiling_config(
            AUTO_TILING,
            scale_factors=tiling_scale_factors_for_vae(dp.video_decoder.checkpoint_path),
            video_shape=full_shape,
            vae_checkpoint_path=dp.video_decoder.checkpoint_path,
            diffvae_optimization=dp.video_decoder.diffvae_optimization,
            device=device,
        )

        conds_ref: list[ConditioningItem] = []
        reference_frames = 0
        if ic_reference is not None:
            ref_path, ref_strength = ic_reference
            # The reference is read at the STAGE-1 dimensions divided by the
            # adapter's declared factor. The in/outpainting IC-LoRA declares 1,
            # so the green canvas is read at exactly half the canvas -- which is
            # what makes the model see the pad bands where it will generate them.
            scale, ref_h, ref_w = reference_pixel_dims(
                reference_factor, height // 2, width // 2
            )

            def _encode_reference(encoder: Any) -> tuple[list[ConditioningItem], int]:
                pixels = load_reference_pixels_cpu(
                    str(ref_path), height=ref_h, width=ref_w,
                    frame_cap=int(num_frames), device=device,
                )
                if pixels is None:
                    # "A missing reference means generate without one", never an
                    # error -- the owner's standing rule. It would be a very bad
                    # outpaint, and the metadata says so rather than the job
                    # failing at the end of the encode.
                    logger.warning(
                        "outpaint: the canvas reference %s yielded no frames; "
                        "generating without a reference", ref_path,
                    )
                    return [], 0
                frames = int(pixels.shape[2])
                items = reference_conditioning_from_pixels(
                    pixels,
                    video_encoder=encoder,
                    device=device,
                    tiling_config=tiling_config,
                    scale=scale,
                    strength=float(ref_strength),
                    attention_strength=float(ic_attention_strength),
                    vram=vram,
                    phase=REFERENCE_ENCODE_PHASE,
                )
                return items, frames

            conds_ref, reference_frames = dp.image_conditioner(_encode_reference)
            cleanup_memory()

        # ── 21_stage1_denoise: half resolution, green canvas attached ─────────
        # NO INITIAL VIDEO LATENT and no video band: stage 1 generates the whole
        # canvas from noise, guided by the reference. The audio is the only
        # modality with anything frozen, which is why ``freeze_kv`` has no
        # counterpart here at all.
        stage_1_sigmas = DISTILLED_SIGMAS.to(dtype=torch.float32, device=device)
        init_a1: torch.Tensor | None = None
        if frozen_audio is not None:
            init_a1 = torch.zeros(
                tuple(AudioLatentShape.from_video_pixel_shape(full_shape).to_torch_shape()),
                dtype=DTYPE,
                device=device,
            )
            init_a1[:, :, :n_frozen] = frozen_audio
        band_a1 = _audio_freeze(init_a1, n_frozen)

        # NO ``outer_index`` / ``outer_total``: outpainting is one clip, so the
        # per-step events go out as THREE POSITIONAL ARGUMENTS and a receiver
        # written for a single generation is unchanged.
        stage.announce(STAGE_1_DENOISE, vram_phase="21_stage1_denoise")
        vstate, astate = stage(
            denoiser=SimpleDenoiser(video_ctx, audio_ctx),
            sigmas=stage_1_sigmas,
            noiser=GaussianNoiser(
                generator=torch.Generator(device=device).manual_seed(int(seed))
            ),
            width=width // 2,
            height=height // 2,
            frames=int(num_frames),
            fps=float(frame_rate),
            video=ModalitySpec(
                context=video_ctx, conditionings=conds_ref, initial_latent=None
            ),
            audio=ModalitySpec(
                context=audio_ctx, conditionings=band_a1, initial_latent=init_a1
            ),
            # The ancestral stage-1 sampler, with the ``eta > 0`` assertion the
            # frozen band rests on carried inside it. See the module docstring
            # for why this differs from 2.3.
            **_stage1_sampler_kwargs(
                STAGE1_SAMPLER, STAGE1_ANCESTRAL_ETA, int(seed), pipeline.dtype
            ),
        )
        stage1_latent = vstate.latent.detach().clone()
        stage1_audio = astate.latent.detach().clone()
        s1_audio_head = (
            stage1_audio[:, :, :n_frozen].detach().clone() if frozen_audio is not None else None
        )
        del vstate, astate, conds_ref, band_a1, init_a1
        cleanup_memory()

        # ── 24_stage1_decode ─────────────────────────────────────────────────
        # THIS IS THE WHOLE REASON outpainting cannot ride the official
        # two-stage call: that call upsamples in LATENT space between the
        # stages, and the blend has to happen on PIXELS.
        vram.reset()
        decode1_started = time.perf_counter()
        gen1 = torch.Generator(device=device).manual_seed(int(seed))
        stage1_pixels = _decoded_to_u8(
            dp.video_decoder(stage1_latent, tiling_config, gen1),
            progress=progress,
            chunks=get_video_chunks_number(int(num_frames), tiling_config),
        )
        vram.record("24_stage1_decode", time.perf_counter() - decode1_started)
        del stage1_latent, gen1
        cleanup_memory()

        canvas_half = _canvas_u8(
            video_path=str(canvas_path),
            height=height // 2,
            width=width // 2,
            frame_cap=int(num_frames),
            device=device,
        )
        _require_frames(
            "the stage-1 decode", int(stage1_pixels.shape[0]), int(num_frames),
            source="the half-resolution VAE decode",
        )
        canvas_half = canvas_half[: int(num_frames)]
        stage1_pixels = stage1_pixels[: int(num_frames)]

        # ── 25_blend1: Laplacian pyramid against the green canvas ────────────
        vram.reset()
        blend1_started = time.perf_counter()
        mask_half = build_blend_mask(geometry, height=height // 2, width=width // 2)
        blended_half = blend_video_u8(
            stage1_pixels,
            canvas_half,
            mask_half,
            mask_low_res_dilation=int(blend_dilation_stage1),
            chunk_size=chunk_size,
            device=device,
        )
        vram.record("25_blend1", time.perf_counter() - blend1_started)
        del stage1_pixels, canvas_half, mask_half
        cleanup_memory()

        # ── 26_pixel_upscale ─────────────────────────────────────────────────
        vram.reset()
        upscale_started = time.perf_counter()
        upscaled = _upscale_u8(
            blended_half, height=height, width=width, device=device, step=pixel_step
        )
        vram.record("26_pixel_upscale", time.perf_counter() - upscale_started)
        del blended_half
        cleanup_memory()

        # ── 27_stage2_encode: the SECOND ImageConditioner build ──────────────
        # DECODE TILING, DIRECT, plus ``channels_last_3d`` for the duration.
        # G0-d measured both halves of that sentence: with the decode config
        # alone a 1280x768 x 241f encode reserves 27188 MB and takes 52.3 s (a
        # WDDM spill on a 16 GB card); with the Conv3d weights re-laid out it is
        # 6174 MB and 19.0 s for a 0.002 dB difference. The alternative --
        # 2.3's spatial tile budget -- costs 0.378 dB and is NOT taken.
        #
        # THE RESTORE IN ``finally`` IS A CORRECTNESS REQUIREMENT, not hygiene:
        # torch's bf16 Conv3d produces different latents under the two layouts,
        # and this same encoder object is reachable again within the process.
        vram.reset()
        encode2_started = time.perf_counter()

        def _reencode(encoder: Any) -> torch.Tensor:
            relayout = torch.device(device).type == "cuda"
            converted = 0
            if relayout:
                converted = set_conv3d_memory_format(encoder, torch.channels_last_3d)
            try:
                # cleanup AFTER the switch, so the contiguous weight storages it
                # just dropped are reclaimed by this same pass.
                cleanup_memory()
                encode_input = _encode_input_from_u8(upscaled)
                latent = encoder.tiled_encode(encode_input, tiling_config)
                del encode_input
                logger.info(
                    "outpaint stage-2 re-encode (channels_last_3d convs=%d): latent %s",
                    converted, tuple(latent.shape),
                )
                return latent.detach().clone()
            finally:
                if relayout:
                    try:
                        set_conv3d_memory_format(encoder, torch.contiguous_format)
                        cleanup_memory()
                    except Exception:  # noqa: BLE001 -- must never mask an in-flight failure
                        logger.exception(
                            "outpaint: failed to restore the video encoder's contiguous layout"
                        )

        stage2_init = dp.image_conditioner(_reencode)
        vram.record("27_stage2_encode", time.perf_counter() - encode2_started)
        del upscaled
        cleanup_memory()

        # ── 22_stage2_denoise: full resolution, NO reference conditioning ────
        # The official graph strips the guide latents with ``LTXVCropGuides``
        # before stage 2; here the conditioning list is simply empty, which says
        # the same thing with nothing to strip. Experiment 2C measured that
        # stage-1 injection alone carries both the composition and the
        # sharpening, so this is also what every other reference path in this
        # engine does.
        stage2_sigma_tensor = torch.tensor(
            stage2_values, dtype=torch.float32, device=device
        )
        # C-4: computed ONCE and handed to BOTH modalities. Stage 2 starts from
        # a re-encode rather than from noise, so its noise scale must be the
        # schedule's own first sigma -- and if a caller overrides the schedule
        # (gate O4), the scale has to follow it. Two separate reads is how that
        # link gets broken; a partial audio freeze is the only arm that would
        # expose the break (gate O7), which is why O7 is mandatory.
        noise_scale2 = float(stage2_sigma_tensor[0].item())

        # STAGE 1's AUDIO, CARRIED, with the frozen head written back over it
        # and re-frozen. Re-freezing is mandatory rather than belt-and-braces:
        # stage 2 re-noises at ``noise_scale2``, which would destroy a
        # stage-1-only freeze outright. See the module docstring for why the
        # CARRY (rather than 2.3's zeros) is an intentional difference.
        init_a2 = stage1_audio.detach().clone()
        if frozen_audio is not None:
            init_a2[:, :, :n_frozen] = frozen_audio
        band_a2 = _audio_freeze(init_a2, n_frozen)

        stage.announce(STAGE_2_DENOISE, vram_phase="22_stage2_denoise")
        vstate2, astate2 = stage(
            denoiser=SimpleDenoiser(video_ctx, audio_ctx),
            sigmas=stage2_sigma_tensor,
            noiser=GaussianNoiser(
                generator=torch.Generator(device=device).manual_seed(
                    int(seed) + STAGE2_SEED_OFFSET
                )
            ),
            width=width,
            height=height,
            frames=int(num_frames),
            fps=float(frame_rate),
            video=ModalitySpec(
                context=video_ctx,
                conditionings=[],
                noise_scale=noise_scale2,
                initial_latent=stage2_init.to(DTYPE),
            ),
            audio=ModalitySpec(
                context=audio_ctx,
                conditionings=band_a2,
                noise_scale=noise_scale2,
                initial_latent=init_a2,
            ),
            # Neither ``stepper`` nor ``loop``: stage 2 is DETERMINISTIC Euler in
            # both engines and upstream, and passing nothing is how
            # ``DiffusionStage``'s own default is said rather than restated.
        )
        final_v = vstate2.latent.detach().clone()
        final_a = astate2.latent.detach().clone()
        s2_audio_head = (
            final_a[:, :, :n_frozen].detach().clone() if frozen_audio is not None else None
        )
        del vstate2, astate2, stage2_init, init_a2, band_a2, stage1_audio
        cleanup_memory()

        # ── 28_stage2_decode ─────────────────────────────────────────────────
        vram.reset()
        decode2_started = time.perf_counter()
        gen2 = torch.Generator(device=device).manual_seed(int(seed))
        stage2_pixels = _decoded_to_u8(
            dp.video_decoder(final_v, tiling_config, gen2),
            progress=progress,
            chunks=get_video_chunks_number(int(num_frames), tiling_config),
        )[: int(num_frames)]
        vram.record("28_stage2_decode", time.perf_counter() - decode2_started)
        del final_v, gen2
        cleanup_memory()

        # ── 29_blend2: the final blend, at full resolution ───────────────────
        vram.reset()
        blend2_started = time.perf_counter()
        canvas_full = _canvas_u8(
            video_path=str(canvas_path),
            height=height,
            width=width,
            frame_cap=int(num_frames),
            device=device,
        )[: int(num_frames)]
        _require_frames(
            "the stage-2 decode", int(stage2_pixels.shape[0]), int(num_frames),
            source="the full-resolution VAE decode",
        )
        mask_full = build_blend_mask(geometry, height=height, width=width)
        final_pixels = blend_video_u8(
            stage2_pixels,
            canvas_full,
            mask_full,
            mask_low_res_dilation=int(blend_dilation_stage2),
            chunk_size=chunk_size,
            device=device,
        )
        vram.record("29_blend2", time.perf_counter() - blend2_started)
        del stage2_pixels, canvas_full, mask_full
        cleanup_memory()

        # ── 30_decode_encode: the freeze proof, the mux, the mp4 ─────────────
        freeze_proof = _freeze_proof(
            frozen_audio=frozen_audio,
            s1_audio_head=s1_audio_head,
            s2_audio_head=s2_audio_head,
        )
        logger.info(
            "outpaint: audio freeze proof %s %s",
            {None: "N/A", True: "PASS", False: "FAIL"}[freeze_proof["pass"]],
            {k: v for k, v in freeze_proof.items() if k != "pass"},
        )
        del s1_audio_head, s2_audio_head

        mux_meta = _mux_plan(
            freeze_source_audio=bool(freeze_source_audio),
            source_waveform=source_waveform,
            audio_frozen_frames=n_frozen,
        )
        vram.reset()
        encode_started = time.perf_counter()
        muxed_samples = 0
        if mux_meta["muxed_original_waveform"]:
            assert source_waveform is not None  # the branch's own precondition
            # The ORIGINAL waveform, trimmed to the VIDEO's duration. The same
            # ``round(px / fps * sr)`` the chain's A2V and retake mux paths use,
            # because it answers the same question: how many samples is this
            # many frames.
            n_mux = int(round(int(num_frames) / float(frame_rate) * audio_sr))
            mux_wf = source_waveform[:, :n_mux].contiguous()
            mux_audio: Any = Audio(
                waveform=mux_wf.to(torch.float32), sampling_rate=int(audio_sr)
            )
            muxed_samples = int(mux_wf.shape[-1])
            logger.info(
                "outpaint: muxing the source's ORIGINAL waveform (%d samples, %d channels "
                "@ %d Hz); the vocoder is NOT run",
                muxed_samples, int(mux_wf.shape[0]), int(audio_sr),
            )
            del mux_wf
        else:
            mux_audio = dp.audio_decoder(final_a)
        del final_a
        cleanup_memory()

        chunks = _encode_chunk_count(int(num_frames), pixel_step)
        encode_video(
            video=_iter_encode_chunks(final_pixels, pixel_step),
            fps=encode_fps,
            audio=mux_audio,
            output_path=str(out_path),
            video_chunks_number=chunks,
        )
        vram.record("30_decode_encode", time.perf_counter() - encode_started)
        del final_pixels, mux_audio

    wall = time.time() - started

    if not out_path.is_file() or out_path.stat().st_size <= 0:
        raise OutpaintError(f"the outpaint produced no/empty output: {out_path}")

    gc.collect()
    cleanup_memory()

    # The job's RESTING footprint after the collector has run -- the same marker
    # the single and chain paths record, so all three are read the same way.
    vram.record("40_job_end", 0.0)

    summary = _vram_summary(vram.phases)
    peak_gib = summary.get("peak_allocated_gib")
    peak_mb = round(float(peak_gib) * 2**30 / 1e6, 1) if peak_gib is not None else 0.0

    metadata: dict[str, Any] = {
        # 2.3's ``outpaint`` sub-dict, key for key, so a client that already
        # parses a 2.3 outpaint job parses this one.
        "outpaint": {
            **geometry.as_dict(),
            "blend_dilation_stage1": int(blend_dilation_stage1),
            "blend_dilation_stage2": int(blend_dilation_stage2),
            "blend_chunk_size": chunk_size,
            "blend1_peak_vram_mb": _phase_peak_mb(vram.phases, "25_blend1"),
            "blend2_peak_vram_mb": _phase_peak_mb(vram.phases, "29_blend2"),
            "stage2_sigmas": list(stage2_values),
            "freeze_source_audio": bool(freeze_source_audio),
            "audio_frozen_latent_frames": int(n_frozen),
            "audio_latent_frames_required": int(a_total),
            "muxed_original_waveform": bool(mux_meta["muxed_original_waveform"]),
            "muxed_audio_samples": int(muxed_samples),
            "audio_sampling_rate": int(audio_sr),
            "canvas_path": str(canvas_path),
            # ── 2.5-only additions ───────────────────────────────────────────
            # The proof that the band actually held. None (never 0.0) when
            # nothing was frozen -- see ``_freeze_proof``.
            "freeze_proof": freeze_proof,
            # Which of the three audio outcomes ran, and whether the vocoder
            # did. ``source_had_audio`` is DISTINCT from
            # ``muxed_original_waveform``: a source can have a track and still
            # freeze nothing (freeze disabled, or a zero-length encode).
            "audio_branch": mux_meta["audio_branch"],
            "vocoder_skipped": bool(mux_meta["vocoder_skipped"]),
            "source_had_audio": bool(source_had_audio),
            "encoded_audio_frames_available": int(audio_available),
            "source_path": None if source_path is None else str(source_path),
        },
        # The flat 2.3 keys.
        "seed": int(seed),
        "width": width,
        "height": height,
        "num_frames": int(num_frames),
        "num_steps": int(num_steps),
        "wall_s": round(wall, 2),
        "vram_peak_mb": peak_mb,
        "vram_within_16gb": peak_mb < 16000,
        "output_mp4": str(out_path),
        # engine25-only facts, in their own sub-dict so the 2.3-shaped keys
        # above stay exactly the 2.3 set a client already parses.
        "ltx25": {
            "stage1_sampler": STAGE1_SAMPLER,
            "stage1_eta": STAGE1_ANCESTRAL_ETA if STAGE1_SAMPLER == "ancestral" else None,
            "stage2_sampler": "euler",
            "stage2_noise_scale": noise_scale2,
            "stage2_seed": int(seed) + STAGE2_SEED_OFFSET,
            "stage2_audio_init_policy": STAGE2_AUDIO_INIT_POLICY,
            "reference_frames": int(reference_frames),
            "reference_downscale_factor": reference_factor,
            "reference_attention_strength": float(ic_attention_strength),
            "encode_fps": encode_fps,
            "video_chunks": chunks,
            "pixel_chunk_frames": int(pixel_step),
            "tiling": None if tiling_config is None else repr(tiling_config),
            "size_bytes": out_path.stat().st_size,
            "vram": summary,
            "phases": dict(vram.phases),
            # WRITTEN DOWN RATHER THAN INFERRED, because a gate that compares
            # this engine's numbers with 2.3's must know which differences are
            # decisions. See the module docstring for each one's reasoning.
            "intentional_differences": {
                "stage1_sampler": "2.3 runs euler in both stages; 2.5 runs ancestral in stage 1",
                "stage2_audio_init": STAGE2_AUDIO_INIT_POLICY,
                "stage2_encode_tiling": "decode config + channels_last_3d (no tile-area budget)",
            },
        },
    }

    result = OutpaintResult(
        output_path=str(out_path),
        seed=int(seed),
        width=width,
        height=height,
        num_frames=int(num_frames),
        frame_rate=float(frame_rate),
        encode_fps=encode_fps,
        # Outpainting takes no conditioning images: the canvas arrives as the
        # IC-LoRA REFERENCE, not as a keyframe. Reported as 0 rather than
        # omitted, because the field is part of the generation contract.
        num_images=0,
        size_bytes=out_path.stat().st_size,
        seconds=wall,
        metadata=metadata,
        phases=dict(vram.phases),
        video_chunks=chunks,
        tiling=None if tiling_config is None else repr(tiling_config),
        **_peaks(vram, device),
    )
    # ``GENERATED_OK`` DELIBERATELY, not a new marker: the worker's own
    # machine-readable line stays ``GENERATE_REPORT`` (which is what the
    # existing evidence collectors grep for) and this human summary keeps the
    # single path's marker, so an outpaint job reads like the generation it is.
    logger.info(
        "GENERATED_OK %.1fs peak_allocated=%sGiB peak_reserved=%sGiB rss_peak=%sGiB "
        "outpaint %dx%d -> %s",
        result.seconds, result.peak_allocated_gib, result.peak_reserved_gib,
        result.rss_peak_gib, width, height, result.output_path,
    )
    return result
