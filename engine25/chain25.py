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

There is no audio equivalent upstream, so :class:`AudioBandMask` below is a
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

Scope
-----
Multi-clip T2V/I2V, plus V2V and A2V (§3-102 second stage): per-clip prompts,
clip 0's conditioning images, the overlap knobs, ``chunked_upsample``,
``stage2_window``, ``source`` and ``audio_source``. Retake, end source,
reference video, LoRA, NAG/VSF and the acceleration knobs are NOT here and are
refused at the API. ``chain_math`` still computes their geometry; this file
simply never passes those arguments.

The two ways material gets frozen
---------------------------------
V2V and A2V both start from an uploaded file and both end up freezing latents,
but they use DIFFERENT mechanisms, and the difference is not stylistic:

* **V2V** freezes a HEAD BAND -- the same one an inter-clip carry uses, via
  ``VideoConditionByMask`` / :class:`AudioBandMask`. Stage 1 holds it at
  ``1 - overlap_strength`` so the continuation can still bend toward it; stage 2
  pins it outright, from a fresh FULL-resolution encode rather than from the
  upsampled stage-1 latent. Only the head is affected; the rest of the timeline
  is generated normally, and the frozen context is then trimmed off the decode
  so the delivered mp4 is the new part alone.
* **A2V** freezes the WHOLE AUDIO MODALITY, via the official
  ``ModalitySpec(frozen=True, noise_scale=0.0, initial_latent=...)``. That is a
  stronger statement than a full-length band would be: ``frozen`` also zeroes
  the modality's scalar ``sigma``, so the prompt AdaLN and the cross-modality
  gates see finished audio rather than audio at timestep zero. The video half
  keeps its ordinary ``overlap_strength`` seams -- welding those shut is exactly
  the long-A2V failure 2.3 recorded. The vocoder never runs: the ORIGINAL
  waveform is muxed, so the delivered audio track is the upload by construction.

With ``source`` and ``audio_source`` both ``None`` -- every plain T2V/I2V chain
-- none of the branches either feature opens is taken, and the output is
byte-identical to the chain that shipped before them (gate G1(a)).
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
    audio_segment_windows,
    compute_chain_layout,
    freeze_mask_values,
    plan_upsample_chunks,
    resolve_stage2_window,
    video_segment_windows,
)
from engine25.ltxcore_compat import (
    AUTO_TILING,
    DISTILLED_SIGMAS,
    STAGE_2_DISTILLED_SIGMAS,
    AllocatorTrimStrategy,
    Audio,
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
    decode_audio_from_file,
    decode_video_by_frame,
    encode_video,
    ensure_tiling_config,
    euler_ancestral_denoising_loop,
    get_video_chunks_number,
    get_videostream_fps,
    gpu_model,
    image_conditionings_by_adding_guiding_latent,
    image_conditionings_by_replacing_latent,
    normalize_images,
    resize_and_center_crop,
    tiling_scale_factors_for_vae,
    upsample_video,
    upsampler_builders,
    vae_encode_audio,
)
from engine25.reference25 import (
    iter_reference_frames_cpu,
    iter_reference_windows,
    reference_conditioning_from_pixels,
    reference_pixel_dims,
    resolve_reference_downscale_factor,
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

#: The SAME question for a video-to-video head, whose answer is the OPPOSITE --
#: which is why it is a separate constant rather than a reuse of the one above.
#:
#: An inter-clip carry starts on latent 0 of the previous clip's TAIL: ordinary
#: content covering eight pixel frames, so the "this is a single pixel frame"
#: marker is wrong and is cleared. A V2V head starts on latent 0 of a FRESH
#: encode of the uploaded tail file, and the video VAE is causal -- its latent 0
#: covers exactly one pixel frame. The marker is therefore CORRECT there, and
#: clearing it would throw away a true fact about the tensor.
#:
#: ``False`` is the reasoned default and gate G1(e) is the measurement behind it.
CLEAR_KEYFRAMES_ON_V2V_HEAD = False

#: Tolerance on the source video's frame rate, in fps. The app re-encodes the
#: tail to the requested rate before the engine ever sees it
#: (``cut_tail_mp4``), so a mismatch here is a broken contract rather than a
#: rounding artefact -- but ``average_rate`` is a ``Fraction`` and 23.976 is
#: ``24000/1001``, so an exact comparison would reject a correct file. Anything
#: larger than this means the frozen head and the generated continuation
#: disagree about how long a frame is, which is a silent time-warp at the
#: junction; hence a hard failure, not a warning.
FPS_TOLERANCE = 1e-3

#: Linear fade-in on the delivered V2V audio head, in seconds. 2.3's value,
#: measured there: the vocoder render and the AAC noise floor meet at the
#: client-side join and the step between them is audible as a click. The video
#: needs no equivalent.
V2V_AUDIO_FADE_IN_SECONDS = 0.030

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
class SourceSpec:
    """Video-to-video continuation source (the uploaded video's tail).

    Same two fields, same meanings, as 2.3's ``SourceSpec`` -- the app layer that
    fills them in is engine-independent and already shipped.

    * ``path``: an mp4 that is ALREADY the tail, cut at the requested frame rate.
      The app guarantees that (``cut_tail_mp4`` runs before the engine is
      called); this engine does not resample, and checks the rate rather than
      trusting it. Its first ``context_frames`` pixel frames are VAE-encoded and
      frozen as the head of clip 0's timeline, and the rest of clip 0 is the
      generated continuation.
    * ``context_frames``: the ``8n + 1`` pixel-frame context span, i.e.
      ``source_context_px`` in :func:`chain_math.compute_chain_layout`.
    """

    path: str
    context_frames: int


@dataclass
class AudioSourceSpec:
    """Audio-to-video source (an uploaded audio file, or an A/V file's track).

    ``path``: any media file with a decodable audio stream. Its waveform is
    VAE-encoded once and the resulting latent is HARD-frozen over the WHOLE
    timeline while only the video is denoised, so the model's cross-modal
    attention drives the picture toward the given audio. The ORIGINAL waveform
    -- never a vocoder render -- is what gets muxed into the delivered mp4,
    truncated to the video's duration, so the output audio track IS the upload
    by construction.

    Mutually exclusive with :class:`SourceSpec`: one chain is either a
    video-to-video continuation or an audio-to-video generation. A separate type
    rather than a flag on ``SourceSpec`` because the two carry unrelated
    payloads -- this one has no context span to speak of.
    """

    path: str


@dataclass
class ChainSpec:
    """One chain job: exactly the body keys of the ``generate_chain`` payload.

    Deliberately nothing else. The features this engine still refuses (retake,
    end source, reference video, LoRA, NAG/VSF, the acceleration knobs) have no
    field here at all, so "this engine does not do that" is visible in the type
    rather than in a runtime branch -- and adding one later is a deliberate act.

    ``source`` and ``audio_source`` are BOTH ``None`` on a plain multi-clip
    T2V/I2V chain, and that case is byte-identical to the chain that shipped
    before they existed (gate G1(a)): every branch they open is guarded on them
    being present, and none of them touches the RNG.

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
    source: SourceSpec | None = None
    audio_source: AudioSourceSpec | None = None


@dataclass
class ChainResult:
    """What one :func:`run_chain` produced."""

    output_path: str
    metadata: dict


# ---------------------------------------------------------------------------
# Conditioning items
# ---------------------------------------------------------------------------


class AudioBandMask(ConditioningItem):
    """Freeze a masked band of audio latent frames -- the audio twin of
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
    frame ``t`` is frozen -- the LEADING frames for a carry, the trailing ones
    for a tail band, and this class has no opinion about which (hence the name:
    it was ``AudioHeadBandMask`` while the head was the only band there was).
    It is patchified through the SAME call the video item uses --
    ``patchify(mask[:, None, :, None])`` -- rather than being reshaped by hand,
    so the token order is the patchifier's, whatever that is.

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
            f"AudioBandMask covers {tokens.shape[1]} tokens but the state has "
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
# Material ingest (V2V / A2V): uploaded file -> frozen latent
# ---------------------------------------------------------------------------
#
# The four functions below are the whole of "read what the user uploaded". They
# live here rather than in a module of their own because each is a handful of
# lines whose ONLY caller is :func:`run_chain`, and because what they must be
# checked against -- the shapes the stage builds, the layout's context counts --
# is in this file.


def _load_video_frames_cpu(
    path: str,
    *,
    frame_cap: int,
    height: int,
    width: int,
    device: torch.device,
    expected_fps: float,
) -> torch.Tensor:
    """The leading ``frame_cap`` frames of ``path`` as ONE CPU ``(1,C,F,H,W)``.

    The low-VRAM twin of the official ``media_io.video_preprocess``, and 2.3's
    ``load_video_conditioning_cpu`` re-expressed on the 1.2.0 API. Same three
    per-frame operations in the same order on the same device --
    ``resize_and_center_crop`` on float32 -> ``normalize_images`` (``x/127.5-1``)
    to ``DTYPE`` -- so the per-frame VALUES are the official ones; only the
    assembly differs. ``video_preprocess`` grows its result with a ``torch.cat``
    per frame ON THE GPU, allocating a fresh full-size buffer while the previous
    one is still live, so the total allocated volume goes with the SQUARE of the
    frame count; this moves each finished frame to CPU immediately and cats once,
    holding at most one frame on the device. Gate G1(d) runs one frame generator
    through both and compares with ``torch.equal``.

    ``decode_video_by_frame`` is the decoder rather than ``decode_video_from_file``
    for one reason: it is the only one in 1.2.0 that takes ``frame_cap``. A tail
    context is 25..145 frames of a file that may be minutes long.

    The frame-rate check is the SECOND line of defence. The app re-encodes the
    tail to the requested rate before calling the engine, so a mismatch means
    that contract is broken -- and the failure it would otherwise produce is a
    silent one: a frozen head whose frames are a different duration from the
    continuation's, i.e. a speed change exactly at the junction. Loud is the
    only safe setting.
    """
    actual_fps = float(get_videostream_fps(path))
    if abs(actual_fps - float(expected_fps)) > FPS_TOLERANCE:
        raise ValueError(
            f"source video frame rate mismatch: {path} is {actual_fps:.6f} fps but the chain "
            f"runs at {float(expected_fps):.6f} fps (tolerance {FPS_TOLERANCE}). The tail must be "
            f"re-encoded to the chain's rate before it reaches the engine."
        )

    frames: list[torch.Tensor] = []
    for raw in decode_video_by_frame(path=path, device=device, frame_cap=int(frame_cap)):
        frame = resize_and_center_crop(raw.to(torch.float32), height, width)
        frames.append(normalize_images(frame, device, DTYPE).to("cpu"))
        del raw, frame

    if len(frames) != int(frame_cap):
        raise ValueError(
            f"source video too short: {path} yielded {len(frames)} frames but the requested "
            f"context needs {int(frame_cap)}."
        )
    out = torch.cat(frames, dim=2)
    del frames
    return out


def _encode_source_heads(
    video_encoder: Any,
    *,
    source: SourceSpec,
    layout: ChainLayout,
    width: int,
    height: int,
    frame_rate: float,
    tiling_config: Any,
    scale_factors: Any,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """VAE-encode the source tail TWICE: ``(half_res_head, full_res_head)``.

    Two encodes, not one plus a resize, because the two are used at different
    stages and stage 2 must not inherit stage 1's approximation:

    * the HALF-res head is stage 1's partially-frozen carry, and has to match the
      half-resolution latent stage 1 works in;
    * the FULL-res head is stage 2's HARD freeze, and comes from a fresh
      full-resolution encode of the original file rather than from the upsampled
      stage-1 latent. That is 2.3's "variant B", and the reason for it is
      measured: variant A (freezing the upsampled approximation) drifts in colour
      and tone across the junction.

    Both go through ``tiled_encode`` -- mandatory, not tidiness: 2.3 measured the
    untiled full-res head at ~+1GB, which OOMs at 720p. The tiling config is the
    one already resolved for the DECODE, reused exactly as 2.3 reuses it, so the
    encode and the decode chunk the same way.

    ``tiled_encode`` CROPS a frame count that is not ``8k + 1`` with a warning and
    carries on, which would silently shorten the context; the assertion below
    fires first, and ``ltxcore_compat.verify`` pins that crop as the behaviour
    being guarded against.
    """
    ctx_px = int(source.context_frames)
    n_ctx_v = int(layout.n_ctx_v)
    if (ctx_px - 1) % VIDEO_TIME_FACTOR != 0:
        raise ValueError(
            f"context_frames={ctx_px} is not 8n+1; the causal video VAE would silently "
            f"crop it to {ctx_px - (ctx_px - 1) % VIDEO_TIME_FACTOR}."
        )

    heads: list[torch.Tensor] = []
    for label, h, w in (("half", height // 2, width // 2), ("full", height, width)):
        pixels = _load_video_frames_cpu(
            source.path, frame_cap=ctx_px, height=h, width=w,
            device=device, expected_fps=frame_rate,
        )
        encoded = video_encoder.tiled_encode(pixels, tiling_config)
        del pixels
        if encoded.shape[2] < n_ctx_v:
            raise ChainError(
                f"the {label}-res source encode produced {encoded.shape[2]} latent frames but the "
                f"layout needs n_ctx_v={n_ctx_v}"
            )
        head = encoded[:, :, :n_ctx_v].detach().clone()
        del encoded
        cleanup_memory()

        expected = VideoLatentShape.from_pixel_shape(
            VideoPixelShape(1, ctx_px, h, w, frame_rate), scale_factors=scale_factors
        ).to_torch_shape()
        want = (expected[0], expected[1], n_ctx_v, expected[3], expected[4])
        assert tuple(head.shape) == want, (label, tuple(head.shape), want)
        heads.append(head)

    src_half, src_full = heads
    return src_half, src_full


def _build_reference_conditionings(
    video_encoder: Any,
    *,
    ic_reference: tuple[str, float],
    factor: int | None,
    attention_strength: float,
    windows: list[tuple[int, int]],
    height: int,
    width: int,
    tiling_config: Any,
    device: torch.device,
) -> list[list[ConditioningItem]]:
    """ONE long IC-LoRA reference -> one conditioning list per stage-1 segment.

    The long-IC-LoRA half of §3-102's third stage, and 2.3's §3-78 re-expressed
    on the 2.5 encoder lifecycle. One decode of the reference file is cut into
    the ``windows`` ``chain_math.video_segment_windows`` computed -- the SAME
    windows the app republishes as ``reference_segment_windows`` in the job
    metadata -- and each window is VAE-encoded into its segment's conditioning.

    Returns EXACTLY ``len(windows)`` lists. An empty one means that segment gets
    no reference, which happens when the file ran out before the window began:
    the owner's rule is "a missing reference means generate without one", never
    an error, and the alternative (refusing a reference shorter than the
    timeline) would make a 24-clip chain need a 24-clip control video.

    Called from inside ``ImageConditioner.__call__``'s closure because that is
    the only place the video encoder exists. Everything it returns is a latent,
    which is why holding all of them is affordable while holding all the PIXELS
    would not be.

    The half-resolution pair is what the reference is sized against
    (``reference_pixel_dims(factor, height // 2, width // 2)``), because stage 1
    is the only stage a reference is added to.
    """
    ref_path, ref_strength = ic_reference
    scale, ref_h, ref_w = reference_pixel_dims(factor, height // 2, width // 2)
    # Everything past the last window's end is waste: the chain cannot consume it.
    frame_cap = windows[-1][0] + windows[-1][1]
    logger.info(
        "chain reference: %s -> %dx%d, factor=%d, %d window(s), frame_cap=%d",
        ref_path, ref_w, ref_h, scale, len(windows), frame_cap,
    )

    out: list[list[ConditioningItem]] = []
    frames = iter_reference_frames_cpu(
        str(ref_path), height=ref_h, width=ref_w, frame_cap=frame_cap, device=device,
    )
    for index, pixels in enumerate(iter_reference_windows(frames, windows)):
        if pixels is None:
            logger.warning(
                "chain reference: exhausted before segment %d; that clip generates "
                "without a reference", index,
            )
            out.append([])
            continue
        out.append(
            reference_conditioning_from_pixels(
                pixels,
                video_encoder=video_encoder,
                device=device,
                tiling_config=tiling_config,
                scale=scale,
                strength=float(ref_strength),
                attention_strength=attention_strength,
            )
        )
        del pixels
    assert len(out) == len(windows), (len(out), len(windows))
    return out


def _load_audio_stereo(path: str, device: torch.device) -> tuple[torch.Tensor, int] | None:
    """``(waveform (1,2,N), sampling_rate)`` from any media file, or ``None``.

    STEREO always. The audio VAE's ``conv_in`` has a two-channel weight and the
    mp4 mux refuses anything that is not ``(2, N)``, so a mono upload has to be
    duplicated -- and this is the one place that happens, for BOTH V2V and A2V.

    That unification is a deliberate 2.5 change: 2.3 duplicated on its A2V path
    and not on its V2V one, so a mono source video reached the encoder with one
    channel. Nothing downstream distinguishes the two cases, so having two
    behaviours was an accident of where the code grew, not a decision.
    """
    audio = decode_audio_from_file(path, device)
    if audio is None:
        return None
    waveform = audio.waveform
    if waveform.dim() == 2:                     # defensive; the loader returns 3-D
        waveform = waveform.unsqueeze(0)
    if waveform.shape[1] == 1:
        waveform = waveform.repeat(1, 2, 1)
    return waveform, int(audio.sampling_rate)


def _encode_audio_latent(audio_encoder: Any, waveform: torch.Tensor, sampling_rate: int) -> torch.Tensor:
    """Waveform ``(1,C,N)`` -> audio latent ``(1,8,T,16)``.

    The ``.to(DTYPE)`` is on the WAVEFORM and happens here, before the call, not
    inside the encoder: ``encode_audio`` computes its mel spectrogram in the
    waveform's own dtype and only casts on the way into the network, so handing
    it float32 would run the whole STFT in float32 and produce a different latent
    from the one every other path in this engine produces. Resampling to the
    encoder's 16kHz is the encoder's own business and is left to it.

    Cloned because the caller outlives the encoder: ``AudioConditioner.__call__``
    frees the model as soon as ``fn`` returns, and a view into its output buffer
    would be a view into freed storage.
    """
    return vae_encode_audio(
        Audio(waveform=waveform.to(DTYPE), sampling_rate=int(sampling_rate)),
        audio_encoder,
        None,
    ).detach().clone()


def _write_audio_handle(out_path: Path, waveform: torch.Tensor, sampling_rate: int) -> str | None:
    """Write ``<stem>_audio_handle.wav`` beside the mp4. Returns its name, or ``None``.

    FLOAT32, via ``scipy.io.wavfile``, and both halves of that matter. The handle
    exists so a client join can crossfade over the context region, which means
    mixing it with the ORIGINAL recording -- and 16-bit PCM (what the official
    ``media_io.encode_audio`` writes, and what stdlib ``wave`` can write) would
    quantise the material before that mix. 2.3 made the same choice for the same
    reason, and Join reads the file 2.3 wrote.

    Best-effort by design: the deliverable is the mp4, which is byte-identical
    whether or not this succeeds. A missing sidecar costs a client the true
    crossfade, not the clip -- so a failure here is logged and the job continues,
    and ``audio_handle_filename`` is ``None`` in the metadata, which is exactly
    what Join checks.
    """
    try:
        import numpy as np                                    # noqa: PLC0415
        from scipy.io import wavfile                          # noqa: PLC0415

        handle_path = out_path.with_name(out_path.stem + "_audio_handle.wav")
        samples = waveform.detach().to(torch.float32).cpu().numpy()    # (channels, samples)
        wavfile.write(str(handle_path), int(sampling_rate), np.ascontiguousarray(samples.T))
        return handle_path.name
    except Exception as exc:  # noqa: BLE001 -- sidecar must never fail the job
        logger.warning("chain: audio handle sidecar not written: %s", exc)
        return None


def _drop_leading_frames(
    chunks: Iterator[torch.Tensor], trim_px: int, counters: dict[str, int]
) -> Iterator[torch.Tensor]:
    """Drop the first ``trim_px`` decoded frames, STREAMING.

    V2V delivers the new part only, so the frozen context has to come off the
    front of the decode. 2.3 did that by materialising the whole timeline and
    slicing it; here the decode stays lazy and the trim is spliced into the
    chunk stream instead -- the same WDDM discipline the rest of this file keeps,
    since a full-length ``(F,H,W,3)`` tensor at production resolution is
    gigabytes.

    The splice has to handle a trim that lands INSIDE a chunk, and it always
    will: decode chunks advance 56 frames at a time (tile 80, overlap 24) while
    ``trim_px`` is 25..145, so the boundary essentially never coincides. Chunks
    the trim consumes entirely are skipped rather than yielded empty -- the mp4
    encoder reads the frame size off the FIRST chunk it receives, so a zero-frame
    first chunk would take the width and height from nothing.

    ``counters`` is filled in as the stream is consumed, so ``decoded_px`` and
    ``new_px`` in the metadata are measurements of what was actually written
    rather than a restatement of the layout's prediction. It is read after
    ``encode_video`` returns, by which time the iterator is exhausted.
    """
    remaining = int(trim_px)
    for chunk in chunks:
        length = int(chunk.shape[0])
        counters["decoded_px"] += length
        if remaining >= length:
            remaining -= length
            continue
        out = chunk[remaining:] if remaining > 0 else chunk
        remaining = 0
        counters["new_px"] += int(out.shape[0])
        yield out
    if remaining > 0:
        raise ChainError(
            f"the decode ended {remaining} frames before the {int(trim_px)}-frame V2V context "
            f"was consumed (decoded {counters['decoded_px']} frames in total)"
        )


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


def _freeze_strengths(
    mask_value: float,
    tail_mask_value: float | None = None,
    audio_mask_value: float | None = None,
    audio_tail_mask_value: float | None = None,
) -> tuple[float, float, float, float]:
    """``chain_math.freeze_mask_values`` read from the other end.

    2.3 writes DENOISE-MASK VALUES by hand; 2.5's band items take the
    COMPLEMENT (``denoise_mask = denoise_mask * inv + (1 - strength) * m``), so
    the four numbers a two-sided freeze needs are the same four the app already
    validated, subtracted from one. Calling the shared resolver rather than
    restating its three-override precedence (tail splits head from tail, audio
    splits the modalities, audio-tail splits the tail BY modality) is what keeps
    a retake or an end-source freeze in this engine the one 2.3 measured.

    Returns ``(video_head, video_tail, audio_head, audio_tail)`` as STRENGTHS,
    ready to hand straight to :func:`_band_conditionings`' four strength
    arguments in that order.

    FOR THE RETAKE / END SOURCE BANDS ONLY -- never for the ordinary carry seam.
    The seam's strength is ``float(spec.overlap_strength)`` and it is passed
    through verbatim, because a round trip through this function is a DOUBLE
    COMPLEMENT and floating point does not survive one: ``1.0 - (1.0 - 0.3)`` is
    ``0.30000000000000004``, not ``0.3``. Every existing chain would keep its
    digest at the default ``overlap_strength`` of 0.5 (which round-trips
    exactly) and quietly change at every other value -- the class of regression
    no gate that runs only the defaults can see. ``tests/test_ltx25_band.py``
    pins both halves: the resolver's table, and the fact that a head band's
    strength arrives as the float it was given.
    """
    v_head, v_tail, a_head, a_tail = freeze_mask_values(
        mask_value, tail_mask_value, audio_mask_value, audio_tail_mask_value
    )
    return 1.0 - v_head, 1.0 - v_tail, 1.0 - a_head, 1.0 - a_tail


def _band_conditionings(
    *,
    video_latent: torch.Tensor | None,
    video_frames_frozen: int,
    audio_latent: torch.Tensor | None,
    audio_frames_frozen: int,
    strength: float,
    clear_keyframes: bool,
    video_tail_frozen: int = 0,
    audio_tail_frozen: int = 0,
    tail_strength: float | None = None,
    audio_strength: float | None = None,
    audio_tail_strength: float | None = None,
) -> tuple[list[ConditioningItem], list[ConditioningItem]]:
    """The ``(video, audio)`` conditioning prefixes for one segment or tile.

    Both lists are EMPTY when nothing is frozen and no marker has to be cleared,
    which is what makes a free head (clip 0 at stage 1, tile 0 at stage 2)
    literally the official code path with no band item in it at all.

    A HEAD band and a TAIL band are separate items over the SAME latent tensor,
    with masks that are disjoint by construction: the head covers
    ``[0, video_frames_frozen)``, the tail covers the last
    ``video_tail_frozen`` frames, and the assertion below refuses the pairing
    that would make them meet. Two items rather than one two-lobed mask because
    the two halves carry DIFFERENT strengths (a retake pins both hard; an end
    source holds its tail at the user's strength while a carried-in head stays
    at ``overlap_strength``), and a single item has one.

    The tail arguments default to zero width, so every call that does not ask
    for one builds exactly the items it built before they existed -- gate G1(a).

    ORDER. Every item here is elementwise over the whole token axis, so all of
    them must be applied BEFORE any conditioning item that APPENDS tokens (a
    ``frame_idx > 0`` keyframe image). That is carried by CONSTRUCTION ORDER at
    the call sites (``conds_v = band_v + ...``), not by a type: there is
    deliberately no ``VideoBandMask`` subclass to sort on, because the sort
    would be a second mechanism to keep true and the one that already exists is
    one line long. ``tests/test_ltx25_band.py`` pins the construction order
    instead, which is the same bargain the unused ``TemporalRegionMask`` got.

    ``clear_keyframes`` is INDEPENDENT of everything else here. It used to be
    reachable only from inside the "a video head is frozen" branch, which tied
    the marker's correctness -- a question about POSITION on the timeline -- to
    a question about freezing; the two come apart the moment a segment can be
    generated in reverse order (§3-102 C3).
    """
    video: list[ConditioningItem] = []
    audio: list[ConditioningItem] = []

    # The same resolution order ``chain_math.freeze_mask_values`` uses, in
    # strength space: an omitted override means "same as the one above it", so a
    # caller that passes only ``strength`` gets the single-value behaviour the
    # ordinary carry seam has always had.
    v_head_strength = float(strength)
    a_head_strength = v_head_strength if audio_strength is None else float(audio_strength)
    if tail_strength is None:
        v_tail_strength, a_tail_strength = v_head_strength, a_head_strength
    else:
        v_tail_strength = a_tail_strength = float(tail_strength)
    if audio_tail_strength is not None:
        a_tail_strength = float(audio_tail_strength)

    if clear_keyframes:
        video.append(ClearKeyframesMask())

    if video_latent is not None and video_frames_frozen > 0:
        b, _, _, h, w = video_latent.shape
        mask = torch.zeros((b, video_latent.shape[2], h, w), dtype=torch.float32, device=video_latent.device)
        mask[:, :video_frames_frozen] = 1.0
        video.append(VideoConditionByMask(latent=video_latent, mask=mask, strength=v_head_strength))

    if video_latent is not None and video_tail_frozen > 0:
        b, _, f, h, w = video_latent.shape
        assert video_frames_frozen + video_tail_frozen <= f, (
            f"a {video_frames_frozen}-frame head and a {video_tail_frozen}-frame tail do not "
            f"fit in {f} video latent frames without overlapping"
        )
        mask_t = torch.zeros((b, f, h, w), dtype=torch.float32, device=video_latent.device)
        mask_t[:, f - video_tail_frozen:] = 1.0
        video.append(VideoConditionByMask(latent=video_latent, mask=mask_t, strength=v_tail_strength))

    if audio_latent is not None and audio_frames_frozen > 0:
        mask_a = torch.zeros(
            (audio_latent.shape[0], audio_latent.shape[2]),
            dtype=torch.float32,
            device=audio_latent.device,
        )
        mask_a[:, :audio_frames_frozen] = 1.0
        audio.append(AudioBandMask(latent=audio_latent, mask=mask_a, strength=a_head_strength))

    if audio_latent is not None and audio_tail_frozen > 0:
        t = audio_latent.shape[2]
        assert audio_frames_frozen + audio_tail_frozen <= t, (
            f"a {audio_frames_frozen}-frame head and a {audio_tail_frozen}-frame tail do not "
            f"fit in {t} audio latent frames without overlapping"
        )
        mask_at = torch.zeros(
            (audio_latent.shape[0], t),
            dtype=torch.float32,
            device=audio_latent.device,
        )
        mask_at[:, t - audio_tail_frozen:] = 1.0
        audio.append(AudioBandMask(latent=audio_latent, mask=mask_at, strength=a_tail_strength))

    return video, audio


#: Phases whose ``seconds`` must NOT be added into ``seconds_total``.
#:
#: Two shapes, one expression:
#:
#: * ``<digits><lowercase letters>_`` -- a SUB-phase (``10a_te_build``,
#:   ``10b_ep_build``). Its time is already inside its parent (``10_``), so
#:   adding it would double-count. The letter suffix on the ordering prefix IS
#:   the convention: a bare ``NN_`` prefix is a phase in its own right, an
#:   ``NNx_`` prefix is a slice of the ``NN_`` above it.
#: * ``40_job_end`` -- a marker, not an interval. Its seconds are 0.0 by
#:   construction, so excluding it changes no arithmetic; it is listed here so
#:   the rule reads as "everything summed is an elapsed interval" rather than
#:   "everything summed is an interval, except one entry that happens to be
#:   zero".
#:
#: ``by_kind`` is deliberately NOT filtered: it is a per-kind table, and a
#: reader looking up "how long did the text encoder build take" wants to find
#: it there. The extra buckets (``te_build`` / ``ep_build`` / ``job_end``) are
#: additions to that table, and ``count`` still counts every phase recorded.
_NOT_IN_SECONDS_TOTAL = re.compile(r"^\d+[a-z]+_|^40_job_end$")


def _vram_summary(phases: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Reduce a chain's per-phase peaks to what belongs on ``done``.

    A chain records one phase per clip and per tile, so the full table grows
    with the request and would put dozens of entries on an event the app parses
    for two numbers. The maxima are what a gate reads; ``count`` is what says
    how many phases they were taken over, so a truncated run is visible. The
    full table is kept alongside it under ``metadata["ltx25"]["phases"]``, which
    is where a post-mortem looks and where the app does not.

    ``seconds_total`` sums the phases that ARE elapsed intervals; sub-phases and
    markers are left out of it (see ``_NOT_IN_SECONDS_TOTAL``).
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
        # Elapsed intervals only -- see ``_NOT_IN_SECONDS_TOTAL``.
        "seconds_total": round(
            sum(
                float(entry.get("seconds") or 0.0)
                for name, entry in phases.items()
                if not _NOT_IN_SECONDS_TOTAL.match(name)
            ),
            2,
        ),
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
    ic_loras: list[tuple] | None = None,
    ic_reference: tuple[str, float] | None = None,
    ic_attention_strength: float = 1.0,
) -> ChainResult:
    """Run one masked AV-latent chain to ONE mp4.

    ``pipeline`` is an :class:`engine25.pipeline25.Ltx25Pipeline` -- the loaded
    model, whose official blocks this function drives directly instead of going
    through ``DistilledPipeline.__call__``.

    ``stage1_sampler`` / ``stage1_eta`` / ``clear_keyframes`` are ENGINE-INTERNAL
    experiment knobs, not request fields: they exist so gate G1's A/B arms can be
    run against the shipped code path rather than a copy of it, and they default
    to the module constants above, which are what every real job uses. Nothing in
    the ``generate_chain`` payload maps to them.

    ``ic_loras`` / ``ic_reference`` / ``ic_attention_strength`` ARE request
    fields (§3-102 third stage). ``ic_loras`` is the job's adapters as
    ``(path, strength[, audio_strength])``; ``ic_reference`` is ONE long control
    video ``(path, strength)`` laid over the whole assembled timeline and cut
    into per-clip windows -- long IC-LoRA, 2.3's §3-78. Defaults of ``None``
    leave the chain byte-identical to a chain without them.
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

    source = spec.source
    audio_source = spec.audio_source
    # Enforced at the API too (422), and by ``chain_math`` implicitly -- but the
    # two write the SAME tensors with different intents (V2V freezes a head band,
    # A2V freezes the whole audio modality), so a request that reached here with
    # both would silently get one of them. An assertion is the cheapest way for
    # that to be impossible rather than merely unlikely.
    assert not (source is not None and audio_source is not None), (
        "V2V (source) and A2V (audio_source) are mutually exclusive; one chain is a "
        "video continuation or an audio-driven generation, never both."
    )
    # An IC-LoRA reference is API-exclusive with V2V (and, in 2.3, with retake and
    # end source -- neither of which exists on this engine, so the assertion is
    # 2.3's list minus the two modes that cannot reach it). The injection below
    # LEANS on that: it sits after the head branch, so a reference reaching a V2V
    # chain would append conditioning to a segment whose head is a frozen carry
    # from a file. A2V is deliberately NOT in the list -- 2.3 allows the pair, and
    # adding a constraint the older engine does not have would be a new
    # restriction dressed up as parity.
    assert ic_reference is None or source is None, (
        "run_chain: ic_reference is mutually exclusive with source (V2V)"
    )

    # ── Geometry: the SHARED pure module, called the way the app calls it ─────
    # Same positional pair, same ``kv``, same resolved (v_tile, v_adv), and now
    # the same ``source_context_px`` -- which is None on every request without a
    # source video, i.e. the same call the plain chain always made. The two
    # remaining feature keywords (retake_glue_px / end_context_px) are still
    # refused at the API, so omitting them is still the same call; gate G1(e)
    # pins that against the app's own call sites rather than leaving it as a
    # comment.
    v_tile, v_adv = resolve_stage2_window(spec.stage2_window)
    source_context_px = None if source is None else int(source.context_frames)
    layout: ChainLayout = compute_chain_layout(
        [c.num_frames for c in clips], frame_rate,
        kv=kv,
        v_tile=v_tile, v_adv=v_adv,
        source_context_px=source_context_px,
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

    # ── Decode tiling, resolved ONCE for the whole timeline ───────────────────
    # Moved ahead of everything else because V2V's source ENCODE reuses it (2.3
    # does the same): the tail must be chunked the way the timeline is, and
    # resolving it twice would be two chances to disagree. Sizing it from
    # ``total_px`` and the FULL resolution is what makes it the decode's own
    # chunking rather than a stage-2 tile's.
    #
    # Safe to hoist because it is deterministic for this checkpoint: the shipped
    # 2.5 video VAE is a CONV VAE, so ``AUTO_TILING`` resolves through the
    # aspect-only branch (768/64 spatial, 80/24 temporal) and reads no free-VRAM
    # figure. A future DIFFUSION VAE would take the memory-aware branch, and
    # THEN this position would matter -- it is called here with no models built,
    # where the free-memory reading is at its most optimistic.
    tiling_config = ensure_tiling_config(
        AUTO_TILING,
        scale_factors=tiling_scale_factors_for_vae(dp.video_decoder.checkpoint_path),
        video_shape=VideoPixelShape(1, total_px, height, width, frame_rate),
        vae_checkpoint_path=dp.video_decoder.checkpoint_path,
        diffvae_optimization=dp.video_decoder.diffvae_optimization,
        device=device,
    )

    # The IC-LoRA reference's downscale factor, resolved from the LoRA headers
    # BEFORE any model is built (it is a header read; discovering "this LoRA
    # declares no reference factor" mid-job would be a poor trade). The
    # per-segment pixel windows come from the SAME shared arithmetic the app uses
    # to publish ``reference_segment_windows`` in the job metadata.
    reference_factor = resolve_reference_downscale_factor(list(ic_loras or []), ic_reference)
    ref_px_windows: list[tuple[int, int]] = (
        [] if ic_reference is None else video_segment_windows(layout)
    )

    logger.info(
        "chain: %d clips %s -> %d px frames, %dx%d @ %.3f fps, kv=%d strength=%.3f, "
        "%d stage-2 tiles (window %s = %d/%d), sampler=%s eta=%s clear_keyframes=%s, "
        "chunked_upsample=%s, source=%s, audio_source=%s, loras=%d, reference=%s",
        n, [c.num_frames for c in clips], total_px, width, height, frame_rate, kv,
        spec.overlap_strength, n_tiles, spec.stage2_window or "standard", v_tile, v_adv,
        sampler, eta if sampler == "ancestral" else "-", clear_kf, spec.chunked_upsample,
        "-" if source is None else f"{source.path} ctx={source.context_frames}",
        "-" if audio_source is None else audio_source.path,
        len(list(ic_loras or [])),
        "-" if ic_reference is None
        else f"{ic_reference[0]} strength={ic_reference[1]:.3f} factor={reference_factor} "
             f"attn={float(ic_attention_strength):.3f} windows={ref_px_windows}",
    )

    # set_loras BEFORE begin_job, and unconditionally: begin_job releases the
    # PREVIOUS job's attachment, and a chain that asks for no adapter must clear
    # rather than inherit one. The stage owns the state; run_chain does not keep
    # a second copy of it.
    stage.set_loras(list(ic_loras or []))
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

        # ── Material ingest: the uploaded audio, if any ───────────────────────
        # BEFORE the video encoder and the transformer, because it is the
        # cheapest model in the job (~46MB) and doing it here means it is never
        # resident alongside either of them. ``AudioConditioner.__call__`` builds
        # the encoder, runs the closure and frees it -- the audio twin of
        # ``ImageConditioner`` and the same lifecycle the official A2V pipeline
        # uses.
        #
        # V2V and A2V both land here, and take DIFFERENT amounts of the result:
        # V2V keeps a head band (``n_ctx_a`` frames, short) and A2V keeps the
        # whole timeline. They also differ on what a shortfall means, which is
        # the interesting part -- see each branch.
        a2v_a: torch.Tensor | None = None          # (1,8,a_total,16) frozen audio
        a2v_orig_wf: torch.Tensor | None = None    # (2,N) CPU float32 -- the mux
        a2v_sr = 0
        a2v_avail = 0
        a_seg_windows: list[tuple[int, int]] = []
        src_a: torch.Tensor | None = None          # (1,8,freeze_ka,16) V2V head
        freeze_ka = 0
        source_had_audio = False

        if audio_source is not None:
            loaded = _load_audio_stereo(audio_source.path, device)
            if loaded is None:
                raise ValueError(
                    f"audio_source has no decodable audio stream: {audio_source.path}"
                )
            wf, a2v_sr = loaded
            a2v_orig_wf = wf.squeeze(0).detach().to(torch.float32).cpu().contiguous()

            vram.reset()
            audio_started = time.perf_counter()
            encoded_a = pipeline.audio_conditioner(
                lambda encoder: _encode_audio_latent(encoder, wf, a2v_sr)
            )
            vram.record("12_audio_conditioning", time.perf_counter() - audio_started)

            a_total = int(layout.a_total)
            a2v_avail = int(encoded_a.shape[2])
            # A HARD failure, unlike the V2V underrun below, and the asymmetry is
            # the point: here the audio is what the whole video is being
            # generated FROM, so a timeline longer than the upload would have
            # stretches with nothing driving them. The video length is
            # authoritative and the audio is truncated to it, never padded --
            # so "too short" has no sensible answer and is refused.
            if a2v_avail < a_total:
                raise ValueError(
                    f"audio_source too short: encoded {a2v_avail} audio-latent frames < the "
                    f"a_total={a_total} this {total_px}-pixel-frame timeline needs. The video "
                    f"length is authoritative; audio is truncated, never padded."
                )
            a2v_a = encoded_a[:, :, :a_total].detach().clone()
            del encoded_a, wf
            cleanup_memory()
            a_seg_windows = audio_segment_windows(layout)

        elif source is not None:
            loaded = _load_audio_stereo(source.path, device)
            source_had_audio = loaded is not None
            if loaded is not None:
                wf, src_sr = loaded
                vram.reset()
                audio_started = time.perf_counter()
                encoded_a = pipeline.audio_conditioner(
                    lambda encoder: _encode_audio_latent(encoder, wf, src_sr)
                )
                vram.record("12_audio_conditioning", time.perf_counter() - audio_started)

                n_ctx_a = int(layout.n_ctx_a)
                avail = int(encoded_a.shape[2])
                freeze_ka = min(n_ctx_a, avail)
                # A WARNING, not an error -- the opposite of A2V above. Here the
                # audio is only continuity material for the junction: freezing
                # fewer frames than asked for makes the join less smooth, it does
                # not make the job meaningless. 2.3 chose the same, and
                # ``audio_head_frozen`` in the metadata reports what happened.
                if avail < n_ctx_a:
                    logger.warning(
                        "chain: source audio underrun -- only %d encoded audio-latent frames "
                        "available but the context needs n_ctx_a=%d; freezing %d.",
                        avail, n_ctx_a, freeze_ka,
                    )
                src_a = encoded_a[:, :, :freeze_ka].detach().clone() if freeze_ka > 0 else None
                del encoded_a, wf
                cleanup_memory()

        # ── Image conditioning: ONE encoder build for every resolution ────────
        # ``ImageConditioner.__call__(fn)`` builds the video encoder, calls
        # ``fn(encoder)`` once and frees it, so the half-resolution stage-1 items,
        # the full-resolution per-tile items AND the two V2V source heads are all
        # made inside a single call -- one build of the ~1GB encoder for the whole
        # job. ``resolve_crf`` first, always: the checkpoint's own CRF is what
        # the images must be re-compressed at, and ``crf=None`` reaches
        # ``load_image_and_preprocess`` as a hard failure.
        #
        # A V2V request never carries clip-0 images (the API refuses the pair) and
        # an image request never carries a source, so in practice each call does
        # one of the two jobs; the closure handles both because the block that
        # OWNS the encoder should not have to know which.
        clip0_images: list[ImageConditioningInput] = list(clips[0].images)
        stage1_conds: list[ConditioningItem] = []
        tile_conds: list[list[ConditioningItem]] = [[] for _ in range(n_tiles)]
        src_half: torch.Tensor | None = None
        src_full: torch.Tensor | None = None
        # ONE list per stage-1 segment, empty when no reference was asked for --
        # which is what keeps the segment loop's ``conds_v`` byte-identical on a
        # chain without one.
        ref_conds: list[list[ConditioningItem]] = [[] for _ in range(n_seg)]
        if clip0_images or source is not None or ic_reference is not None:
            if clip0_images:
                clip0_images = dp.image_conditioner.resolve_crf(clip0_images)

            def _build_all(encoder: Any) -> tuple[list, list[list], Any, Any, list[list]]:
                half: list[ConditioningItem] = []
                full: list[list[ConditioningItem]] = [[] for _ in v_tiles]
                if clip0_images:
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
                head_half = head_full = None
                if source is not None:
                    head_half, head_full = _encode_source_heads(
                        encoder,
                        source=source, layout=layout, width=width, height=height,
                        frame_rate=frame_rate, tiling_config=tiling_config,
                        scale_factors=stage.video_scale_factors, device=device,
                    )
                # LAST, and all n_seg of them in one pass. 2.3 could encode each
                # window lazily just before its segment because its video encoder
                # was a long-lived object it could carry into the loop; 2.5's
                # ``ImageConditioner`` builds the encoder, calls this closure and
                # frees it again, so the encoder only exists HERE. The stream is
                # still walked once and lazily (``iter_reference_windows`` buffers
                # at most one window), and what is retained is the LATENTS, which
                # are small: 512x320/49f at factor 1 is ~70KB per segment.
                refs: list[list[ConditioningItem]] = [[] for _ in range(n_seg)]
                if ic_reference is not None:
                    refs = _build_reference_conditionings(
                        encoder,
                        ic_reference=ic_reference,
                        factor=reference_factor,
                        attention_strength=float(ic_attention_strength),
                        windows=ref_px_windows,
                        height=height, width=width,
                        tiling_config=tiling_config,
                        device=device,
                    )
                return half, full, head_half, head_full, refs

            vram.reset()
            conditioning_started = time.perf_counter()
            stage1_conds, tile_conds, src_half, src_full, ref_conds = dp.image_conditioner(_build_all)
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
            seg_clear_kf = clear_kf
            if i == 0 and source is not None:
                # ── video-to-video: the source tail IS clip 0's head ──────────
                # Exactly the inter-clip carry mechanism below, fed from a file
                # instead of from a previous segment: same partial freeze
                # (1 - overlap_strength), same band item, same order. The only
                # difference is the keyframe marker, and it differs because the
                # tensor genuinely differs -- see CLEAR_KEYFRAMES_ON_V2V_HEAD.
                init_v = torch.zeros(tuple(v_shape.to_torch_shape()), dtype=DTYPE, device=device)
                init_v[:, :, :layout.n_ctx_v] = src_half.to(DTYPE)
                fkv = int(layout.n_ctx_v)
                if freeze_ka > 0:
                    init_a = torch.zeros(tuple(a_shape.to_torch_shape()), dtype=DTYPE, device=device)
                    init_a[:, :, :freeze_ka] = src_a.to(DTYPE)
                    fka = freeze_ka
                seg_clear_kf = CLEAR_KEYFRAMES_ON_V2V_HEAD
            elif i > 0:
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
                # VERBATIM, never through ``_freeze_strengths``: see that
                # function's docstring for the 1-ULP reason.
                strength=float(spec.overlap_strength),
                # The head-freeze test is spelled out HERE because that is where
                # it used to live implicitly -- ``_band_conditionings`` appended
                # the marker clear only from inside its "a video head is frozen"
                # branch, so this expression is today's effective value written
                # down, to the letter (``fkv > 0`` never holds without
                # ``init_v``). §3-102 C3 replaces it with the TIMELINE INDEX,
                # which is what the marker's correctness actually depends on.
                clear_keyframes=seg_clear_kf and fkv > 0,
            )
            # Clip 0's images are the TIMELINE's opening keyframes, so they go on
            # clip 0 only -- and AFTER the band items, which must see an
            # un-extended token axis (see AudioBandMask).
            # THIS segment's window of the long reference (§3-78's long IC-LoRA),
            # appended last. An empty list is both "no reference asked for" and
            # "the reference ran out before this segment" -- the owner's rule is
            # that a missing reference means generate without one, never an error.
            # STAGE 1 ONLY; the stage-2 tiles below never get one.
            conds_v = band_v + (stage1_conds if i == 0 else []) + ref_conds[i]

            noiser = GaussianNoiser(generator=torch.Generator(device=device).manual_seed(seeds[i]))
            video_context, audio_context = seg_ctx[i]

            # ── audio-to-video: the WHOLE audio modality is frozen ────────────
            # A different mechanism from the band above, deliberately. The band
            # is a partial freeze over a few leading frames and leaves the
            # modality's scalar ``sigma`` alone; ``frozen=True`` zeroes the
            # denoise mask AND that scalar, so the prompt AdaLN and the
            # cross-modality gates all see "this audio is finished". That is what
            # the official A2V pipeline does, and A2V's contract -- the picture is
            # driven by EXACTLY the uploaded audio -- is a statement about the
            # whole timeline, not about a head.
            #
            # The VIDEO side keeps its ordinary ``overlap_strength`` band: the
            # clip seams still have to crossfade. Welding them shut here is
            # precisely the long-A2V bug 2.3 recorded, so ``band_a`` is discarded
            # while ``band_v`` is not.
            if audio_source is not None:
                ws, wl = a_seg_windows[i]
                assert wl == a_shape.frames, (i, wl, a_shape.frames)
                audio_spec = ModalitySpec(
                    context=audio_context,
                    frozen=True,
                    noise_scale=0.0,
                    initial_latent=a2v_a[:, :, ws:ws + wl].contiguous().clone().to(DTYPE),
                )
            else:
                audio_spec = ModalitySpec(
                    context=audio_context, conditionings=band_a, initial_latent=init_a
                )

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
                audio=audio_spec,
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
            tile_clear_kf = clear_kf
            if i == 0 and source is not None:
                # ── video-to-video, "variant B": HARD-freeze the source head ──
                # Same hard freeze the i>=1 tile joins below use, but fed from a
                # FULL-RESOLUTION re-encode of the original file rather than from
                # the upsampled stage-1 latent. Freezing the upsampled
                # approximation ("variant A") was measured in 2.3 and produced a
                # visible colour/tone drift across the junction; this is the fix,
                # and it is why ``_encode_source_heads`` encodes twice.
                #
                # Note the strength: stage 1 held this head at
                # ``1 - overlap_strength`` so the continuation could still bend
                # toward it, while stage 2 pins it outright. Stage 2 re-noises at
                # sigma[0] ~= 0.909, so anything less would simply be erased.
                init_v[:, :, :layout.n_ctx_v] = src_full.to(DTYPE)
                fkv = int(layout.n_ctx_v)
                if freeze_ka > 0:
                    init_a[:, :, :freeze_ka] = src_a.to(DTYPE)
                    fka = freeze_ka
                tile_clear_kf = CLEAR_KEYFRAMES_ON_V2V_HEAD
            elif i > 0:
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
                # Today's effective value, written down -- see the stage-1 call.
                clear_keyframes=tile_clear_kf and fkv > 0,
            )
            conds_v = band_v + tile_conds[i]

            # A2V again: this tile's window of the uploaded audio, frozen whole.
            # It REPLACES ``init_a`` rather than being written into it -- there is
            # no carry to preserve, because every tile's audio comes from the same
            # source latent and the overlaps therefore already agree.
            if audio_source is not None:
                audio_spec2 = ModalitySpec(
                    context=stage2_actx,
                    frozen=True,
                    noise_scale=0.0,
                    initial_latent=a2v_a[:, :, as_:as_ + alen].contiguous().clone().to(DTYPE),
                )
            else:
                audio_spec2 = ModalitySpec(
                    context=stage2_actx, conditionings=band_a,
                    noise_scale=noise_scale2, initial_latent=init_a,
                )

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
                audio=audio_spec2,
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
        # ``tiling_config`` was resolved for the WHOLE timeline (``total_px``) at
        # the top of this function -- the decode's own chunking, unrelated to the
        # stage-2 tiles, and shared with the V2V source encode.
        if progress is not None:
            progress(STAGE_DECODE, 0, 1)
        chunks = get_video_chunks_number(total_px, tiling_config)
        generator = torch.Generator(device=device).manual_seed(base_seed)

        vram.reset()
        decode_started = time.perf_counter()
        decoded_video = dp.video_decoder(final_v, tiling_config, generator)
        encode_fps = int(round(frame_rate))
        v2v_meta: dict[str, Any] | None = None
        a2v_meta: dict[str, Any] | None = None

        if audio_source is not None:
            # ── A2V: mux the ORIGINAL waveform; the vocoder never runs ────────
            # Not an optimisation -- it is the contract. The delivered audio track
            # IS the upload (truncated to the video's duration), so rendering the
            # audio latent would produce a waveform that is then thrown away, and
            # would leave the output one vocoder round-trip away from the file the
            # user handed in. The video side still streams chunk by chunk.
            n_mux = int(round(total_px / frame_rate * a2v_sr))
            mux_wf = a2v_orig_wf[:, :n_mux].contiguous()
            encode_video(
                video=_decode_progress(decoded_video, chunks, progress),
                fps=encode_fps,
                audio=Audio(waveform=mux_wf.to(torch.float32), sampling_rate=a2v_sr),
                output_path=str(out_path),
                video_chunks_number=chunks,
            )
            a2v_meta = {
                "source_audio_path": str(audio_source.path),
                "a_total": int(layout.a_total),
                "encoded_audio_frames_available": int(a2v_avail),
                "muxed_original_waveform": True,
                "vocoder_skipped": True,
                "muxed_audio_samples": int(mux_wf.shape[-1]),
                "audio_sampling_rate": int(a2v_sr),
                "audio_channels": int(mux_wf.shape[0]),
            }
            del decoded_video, mux_wf

        elif source is not None:
            # ── V2V: deliver the NEW part only ───────────────────────────────
            # The frozen context was generated so the continuation would have
            # something to continue FROM; it is not part of what the user asked
            # for, and the app lays this clip after the original on its timeline.
            decoded_audio = dp.audio_decoder(final_a)
            trim_px = int(layout.trim_px)
            sr = int(decoded_audio.sampling_rate)
            waveform = decoded_audio.waveform
            if waveform.dim() == 3:
                waveform = waveform.squeeze(0)          # (channels, samples)
            n_trim_a = int(round(trim_px / frame_rate * sr))

            # The audio HANDLE sidecar: the FULL, untrimmed, unfaded timeline
            # audio next to the mp4. Both the source recording and this render
            # depict the context region, so a client join that owns both can do a
            # true overlapped equal-power crossfade there instead of a
            # no-overlap fade pair, which leaves an energy valley. Purely
            # additive -- the mp4 is byte-identical with or without it -- so a
            # failure to write it is a warning, never a failed job.
            audio_handle_filename = _write_audio_handle(out_path, waveform, sr)

            new_wf = waveform[:, n_trim_a:].contiguous()
            fade_n = min(int(round(V2V_AUDIO_FADE_IN_SECONDS * sr)), int(new_wf.shape[1]))
            if fade_n > 1:
                ramp = torch.linspace(0.0, 1.0, fade_n, device=new_wf.device, dtype=new_wf.dtype)
                new_wf[:, :fade_n] = new_wf[:, :fade_n] * ramp

            # The encoder's chunk total is taken from the DELIVERED length, not
            # the decoded one, because that is what it will actually receive.
            counters = {"decoded_px": 0, "new_px": 0}
            new_chunks = get_video_chunks_number(int(layout.new_frames_px), tiling_config)
            encode_video(
                video=_decode_progress(
                    _drop_leading_frames(decoded_video, trim_px, counters), new_chunks, progress
                ),
                fps=encode_fps,
                audio=Audio(waveform=new_wf, sampling_rate=sr),
                output_path=str(out_path),
                video_chunks_number=new_chunks,
            )
            assert counters["decoded_px"] == total_px, (counters["decoded_px"], total_px)

            v2v_meta = {
                "context_frames": int(source.context_frames),
                "n_ctx_v": int(layout.n_ctx_v),
                "n_ctx_a": int(layout.n_ctx_a),
                "freeze_ka": int(freeze_ka),
                "trimmed_px": trim_px,
                "trimmed_audio_samples": n_trim_a,
                "audio_fade_in_samples": int(fade_n if fade_n > 1 else 0),
                "source_had_audio": bool(source_had_audio),
                # Distinct from source_had_audio: the file can HAVE an audio
                # stream and still end up with a zero-frame frozen head, if the
                # encoded source audio ran out before n_ctx_a (see the underrun
                # warning). This key answers "did the head actually carry audio
                # continuity" without ambiguity.
                "audio_head_frozen": bool(freeze_ka > 0),
                "new_frames_px": int(counters["new_px"]),
                "decoded_frames_px": int(counters["decoded_px"]),
                "v2v_context_junction_px": layout.v2v_context_junction_px,
                "audio_handle_filename": audio_handle_filename,
                "handle_context_seconds": round(trim_px / frame_rate, 6),
            }
            del decoded_video, decoded_audio, waveform, new_wf

        else:
            # ── the plain chain: unchanged, and byte-identical (gate G1(a)) ───
            decoded_audio = dp.audio_decoder(final_a)
            encode_video(
                video=_decode_progress(decoded_video, chunks, progress),
                fps=encode_fps,
                audio=decoded_audio,
                output_path=str(out_path),
                video_chunks_number=chunks,
            )
            del decoded_video, decoded_audio

        vram.record("30_decode_encode", time.perf_counter() - decode_started)
        del final_v, final_a

    wall = time.time() - started

    if not out_path.is_file() or out_path.stat().st_size <= 0:
        raise ChainError(f"the chain produced no/empty output: {out_path}")

    gc.collect()
    cleanup_memory()

    # The chain's resting footprint after the collector has run -- the same
    # marker the single-generation path records, so the two are read the same
    # way. Seconds are 0.0: nothing is being timed, the entry exists for its
    # ``rss_gib``.
    vram.record("40_job_end", 0.0)

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
    # The runtime V2V facts MERGE into the geometry sub-dict ``ChainLayout``
    # already put there, rather than replacing it: the app reads ONE
    # ``chain.v2v`` block and both halves belong in it. A2V has no geometry half
    # -- the layout is the same one a plain chain gets -- so it is a plain
    # assignment. Both key sets are 2.3's, name for name, because the app that
    # reads them is engine-independent.
    if v2v_meta is not None:
        metadata.setdefault("v2v", {}).update(v2v_meta)
    if a2v_meta is not None:
        metadata["a2v"] = a2v_meta
    logger.info(
        "CHAIN_OK %.1fs peak=%sMB %d clips / %d tiles -> %s",
        wall, peak_mb, n, n_tiles, out_path,
    )
    return ChainResult(output_path=str(out_path), metadata=metadata)
