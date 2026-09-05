"""Masked AV-latent clip chaining (Phase 3 slice-2, WP4 rewrite).

Production port of the VALIDATED spikes
``outputs/phase3_clip_concat_spike/s1_chain_spike.py`` (2-segment AV carry +
single-decode) and ``s2_tiled_spike.py`` (always-tiled stage-2 for long
timelines). Replaces the old per-clip generate + ffmpeg-concat path (which cut
hard at every boundary because each clip was decoded independently). Here EVERY
boundary lives INSIDE one continuous latent timeline decoded ONCE.

Flow (one worker invocation, latents resident across segments):
  per-segment STAGE 1 (half-res) with video+audio latent tail carry+freeze
    -> assemble ONE continuous stage-1 AV latent (linear crossfade at overlaps)
    -> ONE upsample over the whole timeline
    -> STAGE 2 refine in TEMPORAL TILES (always tiled; a short chain degenerates
       to a single tile), tile i>=1 leading overlap hard-frozen then blended
    -> ONE VAE decode (video+audio) -> ONE mp4.

Reuses the engine's low-VRAM ``LTXFastVideoPipeline`` EXACTLY: model_ledger /
pipeline_components / block-swap / dit-cpu-load / te-offload / tiling_config are
all inherited by reaching into ``pipe.pipeline`` (DistilledPipeline) and calling
the same ledger builders the wheel uses. No wheel edits; no monkeypatches.

Geometry is delegated to the pure-Python :mod:`chain_math` (shared with the app
so junction indices agree byte-for-byte).
"""

from __future__ import annotations

import dataclasses
import gc
import logging
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from chain_math import (
    STAGE2_V_ADV,
    STAGE2_V_TILE,
    VIDEO_TIME_FACTOR,
    ChainLayout,
    audio_segment_windows,
    compute_chain_layout,
    freeze_mask_values,
    plan_upsample_chunks,
    retake_tail_token_range,
    tail_tile_bands,
    video_segment_windows,
)
from engine import progress_shim
from engine.api_types import ImageConditioningInput
from engine.pipeline.common import (
    default_tiling_config,
    encode_video_output,
    iter_video_conditioning_cpu,
    load_video_conditioning_cpu,
    video_chunks_number,
)
from engine.transformer.nag_service import NagParams, encode_negative
from engine.transformer.vsf_service import VsfParams

if TYPE_CHECKING:
    from engine.gguf.ic_lora_common import IcLoraEntry

DTYPE = torch.bfloat16

# Stage-1 denoise-mask value for a retake's two glue bands. 0.0 == HARD freeze.
# Both 0.0 and 0.5 passed the spike; 0.0 is the owner-chosen default because it
# is the value that makes the stage-1 bands bit-exact and therefore machine-
# provable by ``freeze_proof`` (VERIFICATION_LOG §55.3). Named rather than
# inlined so the freeze-proof's "which checks must be zero" rule below reads off
# the same constant the denoise used.
RETAKE_STAGE1_MASK_VALUE = 0.0

# Whether an end source's OWN AUDIO is encoded and frozen along with its video.
# The rollback switch for the whole feature: at ``False`` the audio block in
# :func:`_encode_end_source` is skipped, ``end_a`` stays None, and every audio
# write below is guarded by ``if end_a is not None:`` — never by ``if end_source
# is not None:`` — so the job is byte-identical to the video-only design. Named
# rather than inlined so that guarantee is one grep, not a reading exercise.
END_SOURCE_FREEZE_AUDIO = True

logger = logging.getLogger(__name__)

# progress(stage, index, total) — stage in {"encode","stage1","tile","decode"}.
# "encode" (F2, additive) fires BEFORE the text encode (receivers use dict.get
# and ignore unknown stages, so older consumers are unaffected). Per-step
# denoise events ("stage1_denoise"/"stage2_denoise") do NOT go through this
# callback — they are emitted by the tqdm shim (engine/progress_shim.py),
# phase-tagged via set_phase() below.
ProgressFn = Callable[[str, int, int], None]


@dataclass
class ChainClipSpec:
    """One clip in the chain (prompt already resolved to its effective value)."""

    prompt: str
    num_frames: int
    images: list[ImageConditioningInput] = field(default_factory=list)


@dataclass
class SourceSpec:
    """Video-to-video continuation source (the uploaded video's tail).

    * ``path``: an mp4 that is ALREADY the tail cut at the correct fps (the app
      layer guarantees this in S2 — the engine does not resample). Its first
      ``context_frames`` pixel frames are VAE-encoded and frozen as the head of
      clip-0's timeline; the rest of clip-0 is generated as the continuation.
    * ``context_frames``: 8n+1 pixel-frame context span (== ``source_context_px``
      in :func:`chain_math.compute_chain_layout`).
    """

    path: str
    context_frames: int


@dataclass
class AudioSourceSpec:
    """Audio-to-video source (an uploaded audio-only or A/V file's audio track).

    * ``path``: a media file with a decodable audio stream (wav/mp3/m4a/…). Its
      waveform is VAE-encoded to an audio latent that is HARD-FROZEN (mask 0.0)
      over the FULL timeline while ONLY the video is denoised, so the model's
      cross-modal attention drives lip-sync toward the given audio. The ORIGINAL
      waveform (never the vocoder) is muxed into the delivered mp4, trimmed to
      the video duration — the output audio track == the upload by construction.

    Mutually exclusive with :class:`SourceSpec` (V2V): a single chain is either a
    video-to-video continuation or an audio-to-video generation, never both
    (asserted in :func:`run_chain`; also 422 at the API layer). A distinct type
    from ``SourceSpec`` on purpose — the two carry unrelated payloads.
    """

    path: str


@dataclass
class RetakeSpec:
    """Retake (temporal inpainting) — regenerate the MIDDLE of an existing clip.

    * ``path``: an mp4 that is ALREADY the exact window the app cut out of the
      user's material — 8n+1 frames, CFR, at the request fps. THE ENGINE NEVER
      CUTS OR RESAMPLES. That division of labour is deliberate and mirrors
      :class:`SourceSpec`: the app owns "which pixels", the engine owns "what
      happens to them". ``services/video_io.cut_window_mp4`` is the cutter, and
      it verifies the frame count it wrote by re-probing the file.
    * ``head_px`` / ``tail_px``: the GLUE bands kept frozen at the two ends of
      the window. Their defaults (25 / 24) are the calibrated recommendation
      from VERIFICATION_LOG §55.5 — 9/8 leaves too little material to hold the
      ends, and 49/48 weakens the audio seam, so wider is NOT monotonically
      better. The two are ASYMMETRIC because the video VAE is causal: a head
      band must be 8n+1 pixels and a tail band a multiple of 8
      (``chain_math.v_tail_latents``).
    * ``regenerate_audio``: True (v1 default) regenerates the audio inside the
      free middle along with the video. False keeps the ORIGINAL window
      waveform and muxes it back verbatim — in that mode the vocoder is skipped
      entirely (there is no point rendering audio that is about to be thrown
      away).

    Mutually exclusive with :class:`SourceSpec` (V2V) AND :class:`AudioSourceSpec`
    (A2V) — all three want to own the same latent ends. Asserted in
    :func:`run_chain`; also 422 at the API layer, together with the
    reference-video and clip-0 conditioning-image exclusions.
    """

    path: str
    head_px: int = 25
    tail_px: int = 24
    regenerate_audio: bool = True


@dataclass
class EndSourceSpec:
    """End source — the chain must END on this material (the mirror of V2V).

    * ``path``: an mp4 the app has ALREADY cut to ``context_frames + 1`` frames
      at the request fps (a still image was turned into a video app-side, so the
      engine only ever sees a video, one code path). Same division of labour as
      :class:`SourceSpec` / :class:`RetakeSpec`: the app owns "which pixels", the
      engine owns "what happens to them" — nothing here cuts or resamples.
    * ``context_frames``: the frozen TAIL band in pixel frames, a MULTIPLE OF 8
      (``chain_math.v_tail_latents``); a tail is counted back from the end in
      whole groups of 8 and never reaches the lone keyframe latent, so the head
      grid's 8n+1 does not apply.

    WHERE THE BAND LIVES DEPENDS ON THE CLIP COUNT — ``compute_chain_layout``
    decides it once and reports it as ``layout.end_source_mode``:

      * ONE CLIP -> ``"in_window"``. The band is that clip's OWN last ``n_end_v``
        latents; nothing is appended and the delivered length is the clip length,
        unchanged. Stage 1 denoises a one-clip chain as a SINGLE window, so the
        frozen band is inside the window every generated latent attends over — the
        clip can steer towards the material rather than being crossfaded onto it.
        The band therefore eats into the clip's own free latents, and chain_math
        rejects (422) a clip that has none left once a V2V head is also frozen.
      * TWO OR MORE CLIPS -> ``"reverse"``. The same promise on a chain: the band
        is the LAST clip's own tail, nothing is appended and the delivered length
        is the clips' own total. What differs is the stage-1 SCHEDULE — the
        segments are generated last-to-first and each freezes the next one's head
        as its own tail, so the chain is generated TOWARDS the material. The
        three tables ``layout.seg_generation_order`` / ``seg_head_source`` /
        ``seg_tail_source`` are what the stage-1 loop reads; it never tests the
        mode name.
      * ``"internal_segment"`` is the historical two-or-more-clips design, in
        which THE BAND WAS APPENDED as a segment of its own and the delivered
        length grew by the band. Nothing attends across that segment boundary,
        which is why it reliably crossfaded into the material rather than
        arriving at it. It is no longer reachable from the API; the code is kept
        whole and stays under test through ``compute_chain_layout``'s
        ``end_source_mode_override``.

    In EVERY mode nothing is trimmed (unlike a V2V head, which is cut off) and
    ``config.limits.end_context_frames_max`` (136) is an OPERATIONAL ceiling on the
    region the real-hardware gate has looked at, NOT a geometric one: the band may
    span as many stage-2 tiles as it needs (``layout.end_tile_bands`` is the
    per-tile freeze plan), and a tile it swallows whole is simply fully frozen.

    WHY THE FILE CARRIES ONE EXTRA FRAME (the "+1 primer"): the video VAE is
    causal. Latent 0 is a keyframe built from pixel frame 0 alone, and latent
    ``k >= 1`` covers the group of 8 pixels ``8k-7 .. 8k``. Encoding the last 8n
    pixels on their own would therefore make ITS latent 0 a fresh keyframe —
    both the wrong index mapping and the wrong content. Handing the encoder one
    primer frame in front and dropping latent 0 (``[1:]``) yields exactly the
    ``n_end_v`` tail latents that belong at the end of the timeline. Same
    argument :func:`_encode_retake_window` makes for encoding a whole window
    instead of its two bands. The consequence for the user is that the primer
    frame never appears: the delivered tail is the material's frames
    1..context_frames.

    * ``strength``: 1.0 (default) is a HARD freeze, byte-identical to before
      this field existed. Below 1.0 the stage-1 tail mask value becomes
      ``1.0 - strength`` instead of ``0.0``, softening only how hard stage 1
      is pinned to the material. STAGE 2 ALWAYS HARD-FREEZES (mask 0.0)
      REGARDLESS OF THIS VALUE, so the delivered last frame is the material
      either way; ``strength`` only changes how much stage 1 is allowed to
      drift from it before stage 2 re-pins it. IT IS A VIDEO-ONLY KNOB: the
      material's audio band is hard-frozen in BOTH stages at every strength
      (see :func:`_encode_end_source`), so a soft-strength job still ends on
      the material's own sound.

    THE MATERIAL'S AUDIO IS TAKEN WHENEVER THERE IS ANY — not a mode, not a
    field: a video end source with an audio track has that track's latents
    frozen over the band alongside the video, and a still image (or a video
    with no audio) simply has none and the tail's audio is generated freely.
    The frozen region is LATENTS through the audio VAE and vocoder, not a
    waveform mux, which is what lets the freely generated part attend to it.

    Mutually exclusive with :class:`RetakeSpec` (which already owns both ends of
    its single window) and :class:`AudioSourceSpec` (which owns the whole audio
    timeline). COMBINABLE with :class:`SourceSpec`: a start source plus an end
    source is an interpolation between the two. Asserted in :func:`run_chain`;
    also 422 at the API layer.
    """

    path: str
    context_frames: int
    strength: float = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Re-orchestration primitives (faithful to s1/s2 spikes).
# ─────────────────────────────────────────────────────────────────────────────
def _crossfade_concat(seg0: torch.Tensor, seg1: torch.Tensor, k: int) -> torch.Tensor:
    """Temporal (dim=2) crossfade concat; total = L0 + L1 - K. (spike-identical).

    Works for 5D video (b,c,f,h,w) and 4D audio (b,c,t,f) latents (time = dim 2).
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


def _build_video_conditionings(
    images: list[ImageConditioningInput],
    *,
    height: int,
    width: int,
    video_encoder,
    device: torch.device,
):
    """Hybrid keyframe routing (matches LTXFastVideoPipeline._run_inference):

    * frame_idx == 0 -> latent REPLACE (VideoConditionByLatentIndex),
    * frame_idx  > 0 -> keyframe/guide APPEND (VideoConditionByKeyframeIndex),
                        frame_idx passed as a raw PIXEL RoPE offset.
    ``images`` frame_idx values are already snapped to the 0-or-8n+1 grid by the
    API validator.
    """
    from ltx_pipelines.utils.args import ImageConditioningInput as _LtxImageInput
    from ltx_pipelines.utils.helpers import (
        image_conditionings_by_adding_guiding_latent as _add_guide,
        image_conditionings_by_replacing_latent as _replace,
    )

    if not images:
        return []
    replace_imgs = [_LtxImageInput(im.path, im.frame_idx, im.strength) for im in images if im.frame_idx == 0]
    guide_imgs = [_LtxImageInput(im.path, im.frame_idx, im.strength) for im in images if im.frame_idx > 0]
    conds: list = []
    if replace_imgs:
        conds += _replace(images=replace_imgs, height=height, width=width,
                          video_encoder=video_encoder, dtype=DTYPE, device=device)
    if guide_imgs:
        conds += _add_guide(images=guide_imgs, height=height, width=width,
                            video_encoder=video_encoder, dtype=DTYPE, device=device)
    return conds


def _iter_reference_windows(
    frame_iter: Iterable[torch.Tensor],
    windows: list[tuple[int, int]],
) -> Iterator[torch.Tensor | None]:
    """Cut ONE long reference video into the per-segment pixel windows §1-15 needs.

    Yields EXACTLY ``len(windows)`` items, one per stage-1 segment, in order: a
    (1,C,F,H,W) CPU tensor built by ``torch.cat``-ing that window's frames, or
    ``None`` when the reference ran out before the window began. ``frame_iter``
    supplies one (1,C,1,H,W) CPU frame per PIXEL frame in decode order —
    :func:`engine.pipeline.common.iter_video_conditioning_cpu` in production, a
    fake iterator in the tests (which is the whole point of the split: it lets a
    unit test assert the window OFFSETS, not just the window arithmetic).

    ``windows`` comes from ``chain_math.video_segment_windows`` and is
    ``(start_px, len_px)`` on the GLOBAL reference timeline, monotonically
    increasing in ``start_px``, with adjacent windows OVERLAPPING by ``8*kv - 7``
    frames (the K_v carry band). This walks the stream once: skip up to the next
    window's start, buffer its length, yield, then keep the overlap tail for the
    next window. Peak buffer is therefore ``max(clip_frames)`` frames — not the
    whole (up to 11544-frame) reference.

    Short-reference policy (owner: "missing reference -> generate without one",
    never an error): a window the stream only partially covers yields the frames
    it DOES have — the wheel's VAE crops the tail of a non-8n+1 pixel run itself
    (video_vae.py), so there is no need to round the partial length down; only
    the zero-frame case has to be avoided, and that yields ``None``. Every window
    after the stream is drained yields ``None`` too.
    """
    it = iter(frame_iter)
    buf: list[torch.Tensor] = []   # consecutive frames, buf[0] is global frame ``base``
    base = 0                       # global index of buf[0]; == frames consumed when buf is empty
    exhausted = False

    for start, length in windows:
        # 1. drop the frames this window has left behind (the previous window's
        #    non-overlapping head), then skip forward if it starts beyond them.
        if start > base:
            drop = min(start - base, len(buf))
            if drop:
                del buf[:drop]
                base += drop
        while base < start and not exhausted:   # buf is empty here by construction
            nxt = next(it, None)
            if nxt is None:
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


def _denoise_av_with_carry(
    *,
    output_shape,
    components,
    transformer,
    video_context,
    audio_context,
    video_conditionings,
    noiser,
    stepper,
    sigmas: torch.Tensor,
    noise_scale: float,
    initial_video_latent: torch.Tensor | None,
    initial_audio_latent: torch.Tensor | None,
    freeze_kv: int,
    freeze_ka: int,
    mask_value: float,
    device: torch.device,
    freeze_tail_v: int = 0,
    freeze_tail_a: int = 0,
    tail_mask_value: float | None = None,
    audio_mask_value: float | None = None,
    audio_tail_mask_value: float | None = None,
):
    """Reimpl of helpers.denoise_audio_video with an overlap-freeze mask.

    Builds the video + audio initial states, overrides the denoise_mask on the
    first freeze_k{v,a} latent frames to ``mask_value`` (= 1 - overlap_strength),
    applies video conditionings, noises (mask-respecting), runs the euler loop
    (re-pins mask<1 tokens each step), and unpatchifies. Verbatim mechanics from
    the s1/s2 spikes (which faithfully reimplement the wheel helpers) with an
    added optional ``video_conditionings`` (empty for T2V segments/tiles).

    ``freeze_tail_v`` / ``freeze_tail_a`` (retake, additive): ALSO freeze the
    trailing N latent frames, so the middle is regenerated between two frozen
    ends. Both default to 0 -> every pre-retake call site keeps the exact
    head-only behaviour it had (the two ``if`` conditions below degenerate to
    the original ``if freeze_k? > 0``). ``tail_mask_value`` defaults to
    ``mask_value``; it exists so a future caller can hold the two ends at
    different strengths, which is what the spike's stage-1 0.5 arm exercised.

    ``audio_mask_value`` (long A2V, additive) splits the freeze STRENGTH per
    MODALITY the way ``tail_mask_value`` splits it per END: the video mask keeps
    ``mask_value`` while the AUDIO mask uses this value. Defaults to ``None`` ->
    ``mask_value``, so every pre-A2V call site is bit-identical. It exists
    because a multi-clip A2V segment must HARD-freeze its uploaded audio window
    (0.0) while the video seam still carries over at the user's
    ``overlap_strength``; passing 0.0 as the single ``mask_value`` would freeze
    the video seam solid and silently discard ``overlap_strength``.

    ``audio_tail_mask_value`` (end source, additive) splits the TAIL by MODALITY:
    the material's audio band is always hard-frozen (0.0) while the video tail
    honours ``end_source.strength`` as ``1 - strength``. Defaults to ``None`` ->
    whatever the tail resolves to otherwise, so every pre-end-source-audio call
    site is bit-identical. The three overrides are resolved into the four band
    strengths by the pure :func:`chain_math.freeze_mask_values`.

    The tail range is ABSOLUTE (``chain_math.retake_tail_token_range``), never
    ``m[:, -k*hw:]`` — see that function for why negative indexing silently
    under-freezes once conditioning tokens have been appended. The two-sided
    freeze itself was validated in VERIFICATION_LOG §55.3, with three ablation
    arms proving the causality.
    """
    from ltx_core.tools import AudioLatentTools, VideoLatentTools
    from ltx_core.types import AudioLatentShape, VideoLatentShape
    from ltx_pipelines.utils.helpers import simple_denoising_func, state_with_conditionings
    from ltx_pipelines.utils.samplers import euler_denoising_loop

    # The four frozen bands' strengths, resolved by the pure (torch-free) helper
    # so the app venv can regression-test the overrides. All three overrides
    # default to None -> all four are ``mask_value`` -> bit-identical to the
    # pre-override behaviour.
    v_head_mv, v_tail_mv, a_head_mv, a_tail_mv = freeze_mask_values(
        mask_value, tail_mask_value, audio_mask_value, audio_tail_mask_value
    )

    # ── VIDEO ──
    vshape = VideoLatentShape.from_pixel_shape(
        shape=output_shape,
        latent_channels=components.video_latent_channels,
        scale_factors=components.video_scale_factors,
    )
    vtools = VideoLatentTools(components.video_patchifier, vshape, output_shape.fps)
    vstate = vtools.create_initial_state(device, DTYPE, initial_video_latent)
    # hw / v_tokens are hoisted OUT of the freeze branch (they are plain geometry
    # and the tail block needs them too); the bound that matters is the VIDEO
    # token span, not ``m.shape[1]``, which conditioning items may already have
    # extended.
    hw = vshape.height * vshape.width
    v_tokens = vshape.frames * hw
    if freeze_kv > 0 or freeze_tail_v > 0:
        m = vstate.denoise_mask.clone()
        assert m.shape[1] >= v_tokens, (m.shape[1], v_tokens)
        if freeze_kv > 0:
            m[:, : freeze_kv * hw, ...] = v_head_mv
        if freeze_tail_v > 0:
            lo, hi = retake_tail_token_range(vshape.frames, hw, freeze_tail_v)
            assert 0 <= lo < hi <= v_tokens, (lo, hi, v_tokens)
            m[:, lo:hi, ...] = v_tail_mv
        vstate = dataclasses.replace(vstate, denoise_mask=m)
    # ORDERING (load-bearing): every mask edit above happens BEFORE this call,
    # exactly where the head-only freeze already sat. state_with_conditionings
    # appends conditioning tokens, so editing after it would address the wrong
    # tokens on the tail side.
    vstate = state_with_conditionings(vstate, video_conditionings or [], vtools)
    vstate = noiser(vstate, noise_scale)

    # ── AUDIO ── (patchified audio token i == audio latent frame i; mask (b,T,1))
    ashape = AudioLatentShape.from_video_pixel_shape(output_shape)
    atools = AudioLatentTools(components.audio_patchifier, ashape)
    astate = atools.create_initial_state(device, DTYPE, initial_audio_latent)
    if freeze_ka > 0 or freeze_tail_a > 0:
        m = astate.denoise_mask.clone()
        assert m.shape[1] >= ashape.frames, (m.shape[1], ashape.frames)
        if freeze_ka > 0:
            m[:, :freeze_ka, ...] = a_head_mv
        if freeze_tail_a > 0:
            # hw == 1 in the audio token domain (one token per latent frame).
            lo, hi = retake_tail_token_range(ashape.frames, 1, freeze_tail_a)
            assert 0 <= lo < hi <= ashape.frames, (lo, hi, ashape.frames)
            m[:, lo:hi, ...] = a_tail_mv
        astate = dataclasses.replace(astate, denoise_mask=m)
    astate = state_with_conditionings(astate, [], atools)
    astate = noiser(astate, noise_scale)

    # ── DENOISE LOOP ──
    vstate, astate = euler_denoising_loop(
        sigmas=sigmas, video_state=vstate, audio_state=astate, stepper=stepper,
        denoise_fn=simple_denoising_func(
            video_context=video_context, audio_context=audio_context, transformer=transformer,
        ),
    )

    vstate = vtools.clear_conditioning(vstate)
    vstate = vtools.unpatchify(vstate)
    astate = atools.clear_conditioning(astate)
    astate = atools.unpatchify(astate)
    return vstate, astate


def _tile_images(images: list[ImageConditioningInput], vs: int, vlen: int) -> list[ImageConditioningInput]:
    """Filter+remap clip-0 conditioning images to a stage-2 tile [vs, vs+vlen).

    A conditioning applies to the tile whose global latent range contains its
    latent frame; frame_idx is remapped to the tile-local pixel offset (tiles
    restart RoPE positions locally). For the common case (start-frame idx==0 and
    low keyframes on clip 0) everything lands in tile 0 (vs==0) unchanged.
    """
    out: list[ImageConditioningInput] = []
    for im in images:
        if im.frame_idx == 0:
            g_lat = 0
        else:
            g_lat = (im.frame_idx - 1) // VIDEO_TIME_FACTOR + 1
        if vs <= g_lat < vs + vlen:
            local = 0 if im.frame_idx == 0 else im.frame_idx - vs * VIDEO_TIME_FACTOR
            out.append(im._replace(frame_idx=max(0, local)))
    return out


def _encode_source_heads(
    *,
    source: "SourceSpec",
    layout: ChainLayout,
    width: int,
    height: int,
    video_encoder,
    tiling_cfg,
    ledger,
    device: torch.device,
):
    """VAE-encode the source tail into frozen HEAD latents (video-to-video).

    Returns ``(src_head_v_half, src_head_v_full, src_head_a, freeze_ka,
    had_audio)``:
      * ``src_head_v_half`` — half-res head for stage-1 carry (n_ctx_v frames),
      * ``src_head_v_full`` — FULL-res head for stage-2 variant-B hard-freeze,
      * ``src_head_a`` / ``freeze_ka`` — audio head latents (None/0 if the source
        has no audio -> free audio generation).

    BOTH video encodes use ``VideoEncoder.tiled_encode`` (mandatory: the spike's
    untiled full-res head encode cost ~+1GB and would OOM at 720p).
    """
    from ltx_core.model.audio_vae import encode_audio as vae_encode_audio
    from ltx_core.types import Audio
    from ltx_pipelines.utils.helpers import cleanup_memory
    from ltx_pipelines.utils.media_io import decode_audio_from_file

    ctx_px = source.context_frames
    n_ctx_v = layout.n_ctx_v
    n_ctx_a = layout.n_ctx_a

    # ── half-res head (stage-1 carry, matches the half-res stage-1 latent) ──
    # CPU-assembled source pixels (numerics-identical; keeps the full context
    # tensor off the GPU while tiled_encode streams tiles back one at a time).
    src_half = load_video_conditioning_cpu(
        video_path=source.path, height=height // 2, width=width // 2,
        frame_cap=ctx_px, dtype=DTYPE, device=device,
    )
    src_head_v_half = video_encoder.tiled_encode(src_half, tiling_cfg)[:, :, :n_ctx_v].detach().clone()
    del src_half
    cleanup_memory()
    assert src_head_v_half.shape[2] == n_ctx_v, (src_head_v_half.shape[2], n_ctx_v)

    # ── full-res head (stage-2 variant-B hard-freeze) — TILED (720p-critical) ──
    src_full = load_video_conditioning_cpu(
        video_path=source.path, height=height, width=width,
        frame_cap=ctx_px, dtype=DTYPE, device=device,
    )
    src_head_v_full = video_encoder.tiled_encode(src_full, tiling_cfg)[:, :, :n_ctx_v].detach().clone()
    del src_full
    cleanup_memory()
    assert src_head_v_full.shape[2] == n_ctx_v, (src_head_v_full.shape[2], n_ctx_v)

    # ── audio head (first-ever audio_encoder load; cheap, ~46MB) ──
    src_head_a = None
    freeze_ka = 0
    src_audio = decode_audio_from_file(source.path, device)
    had_audio = src_audio is not None
    if had_audio:
        audio_encoder = ledger.audio_encoder()
        # decode_audio_from_file always returns Audio(waveform=(1,channels,samples))
        # (ltx_pipelines.utils.media_io.decode_audio_from_file docstring + impl:
        # the final `.unsqueeze(0)` always yields a 3-D tensor) — no 2-D case to
        # normalise here.
        enc = vae_encode_audio(
            Audio(waveform=src_audio.waveform.to(DTYPE), sampling_rate=src_audio.sampling_rate),
            audio_encoder, None,
        )
        avail = enc.shape[2]
        freeze_ka = min(n_ctx_a, avail)
        if avail < n_ctx_a:
            logger.warning(
                "chain: source audio underrun — only %d encoded audio-latent "
                "frames available but the requested context needs n_ctx_a=%d; "
                "freezing %d frames only (silent under-freeze, no error).",
                avail, n_ctx_a, freeze_ka,
            )
        src_head_a = enc[:, :, :freeze_ka, :].detach().clone()
        del audio_encoder, enc
        cleanup_memory()
    return src_head_v_half, src_head_v_full, src_head_a, freeze_ka, had_audio


def _encode_end_source(
    *,
    end_source: "EndSourceSpec",
    layout: ChainLayout,
    width: int,
    height: int,
    video_encoder,
    tiling_cfg,
    ledger,
    device: torch.device,
):
    """VAE-encode the end material into the frozen TAIL latents (end source).

    Returns ``(end_v_half, end_v_full, end_a, n_end_a_eff, audio_status)`` — the
    half-res band stage 1 freezes at the end of its LAST stage-1 segment, the
    FULL-res band stage 2 re-writes and re-freezes in EVERY tile the band
    reaches (one slice per tile out of ``layout.end_tile_bands``, variant B — the
    same choice the V2V head makes in :func:`_encode_source_heads`), and the
    material's OWN AUDIO band, which both stages freeze the same way. The three
    tensors are each the WHOLE band; the stages index into them. Stage 1 writes
    its two in one piece because the last segment holds all of both in EVERY
    end-source mode — the user's last clip under ``"in_window"`` / ``"reverse"``,
    the appended internal band segment under the API-unreachable
    ``"internal_segment"``. NOTHING HERE DEPENDS ON THE MODE: the encode is of
    the material, not of the timeline.

    Both video encodes consume the WHOLE app-cut file — ``context_frames + 1``
    pixel frames — and then drop latent 0. That primer frame is what makes the
    remaining ``n_end_v`` latents land on the timeline's own grid; see
    :class:`EndSourceSpec` for the causal-VAE argument. The ``+ 1`` assert is
    therefore the real contract check between the app's cutter and this encode:
    a file of the wrong length would silently freeze the wrong latents.

    THE AUDIO IS TAKEN FROM THE SAME FILE, and it is there already: the app's
    cutter (``services/video_io.cut_window_mp4``) resamples and cuts the upload
    with its audio track attached, so the fps the request asked for and the
    audio the engine reads are the same cut. A still image simply has none.

    WHAT IS FROZEN, AND HOW HARD. The audio band is ALWAYS a hard freeze (mask
    0.0) in both stages: ``end_source.strength`` is a VIDEO-only knob. And it is
    LATENTS, not a waveform mux — unlike A2V and a non-regenerating retake, which
    deliver the original samples. Freezing latents is what puts the material's
    audio inside the attention the free region is generated against; a mux would
    arrive after the denoise and could not do that. The cost is a VAE + vocoder
    round trip on the frozen region, i.e. the same music, slightly duller.

    SLICED FROM THE TAIL, ``enc[:, :, avail - n_end_a_eff : avail]``, because the
    file's LAST audio sample is what lines up with the timeline's last frame.
    The material's own latent grid and the timeline's are not in phase in
    general, so up to ONE audio latent (40 ms) of offset remains — accepted,
    and finer than the 1-3 frame uncertainty §55.6 already records for the
    encoder's time support. (A future tightening would pad the waveform at the
    FRONT so the grids align; nothing else here would change.)

    THE WAVEFORM IS TRIMMED TO THE VIDEO'S OWN LENGTH BEFORE ENCODING. The
    cutter writes PCM where it can and falls back to AAC, and AAC pads its last
    frame (up to ~21 ms) at the STREAM'S END, which is exactly the end this
    function slices from. Cutting the waveform to ``cut_px / fps`` seconds first
    restores "the last sample is the last frame" — the same ``round(px / fps *
    sr)`` the A2V and retake mux paths use for the same reason.

    FALLBACKS ARE SILENT, NEVER ERRORS (owner adjudication): no audio track or an
    undecodable one -> ``audio_status="no_audio"`` and the tail's audio is
    generated freely, exactly as before this existed. A SHORT encode freezes what
    there is (``n_end_a_eff = min(n_end_a, avail)``, ``"partial"``) rather than
    failing the job — the same "quiet under-freeze" :func:`_encode_source_heads`
    settled on. Digital silence is NOT detected: silence is frozen as silence.
    ``END_SOURCE_FREEZE_AUDIO = False`` skips the block entirely
    (``"disabled"``), which is the feature's rollback.

    Both video encodes go through ``tiled_encode`` for the same reason the two
    functions around it do: an untiled full-res encode would OOM at 720p.
    """
    from ltx_core.model.audio_vae import encode_audio as vae_encode_audio
    from ltx_core.types import Audio
    from ltx_pipelines.utils.helpers import cleanup_memory
    from ltx_pipelines.utils.media_io import decode_audio_from_file

    n_end_v = layout.n_end_v
    n_end_a = layout.n_end_a
    cut_px = int(end_source.context_frames) + 1

    # ── half-res tail band (stage-1, matches the half-res stage-1 latent) ──
    end_half = load_video_conditioning_cpu(
        video_path=end_source.path, height=height // 2, width=width // 2,
        frame_cap=cut_px, dtype=DTYPE, device=device,
    )
    enc = video_encoder.tiled_encode(end_half, tiling_cfg)
    assert enc.shape[2] == n_end_v + 1, (enc.shape[2], n_end_v + 1)
    end_v_half = enc[:, :, 1:].detach().clone()
    del end_half, enc
    cleanup_memory()

    # ── full-res tail band (stage-2 variant-B hard-freeze) — TILED ──
    end_full = load_video_conditioning_cpu(
        video_path=end_source.path, height=height, width=width,
        frame_cap=cut_px, dtype=DTYPE, device=device,
    )
    enc = video_encoder.tiled_encode(end_full, tiling_cfg)
    assert enc.shape[2] == n_end_v + 1, (enc.shape[2], n_end_v + 1)
    end_v_full = enc[:, :, 1:].detach().clone()
    del end_full, enc
    cleanup_memory()

    # ── the material's own audio band (one encode; both stages share it) ──
    # There is no half/full split here: an audio latent has no resolution —
    # ``AudioLatentShape.from_video_pixel_shape`` reads only the frame count and
    # the fps — so the single encode is what stage 1 and stage 2 both freeze.
    end_a = None
    n_end_a_eff = 0
    audio_status = "disabled"
    if END_SOURCE_FREEZE_AUDIO:
        audio_status = "no_audio"
        src_audio = decode_audio_from_file(end_source.path, device)
        if src_audio is not None:
            wf = src_audio.waveform                    # (1, channels, samples)
            if wf.dim() == 2:                          # defensive; loader returns 3-D
                wf = wf.unsqueeze(0)
            # The audio VAE encoder's conv_in is stereo-only (weight [128,2,3,3]),
            # so a mono track must be duplicated — the same normalisation the A2V
            # and retake paths do.
            if wf.shape[1] == 1:
                wf = wf.repeat(1, 2, 1)
            sr = int(src_audio.sampling_rate)
            # HEAD-BASED TRIM TO THE VIDEO'S LENGTH, BEFORE THE ENCODE — see the
            # docstring: the tail slice below assumes the last sample is the last
            # frame, and a codec's end padding is what would break that.
            wf = wf[..., : round(cut_px / float(layout.fps) * sr)]
            audio_encoder = ledger.audio_encoder()
            enc = vae_encode_audio(
                Audio(waveform=wf.to(DTYPE), sampling_rate=sr), audio_encoder, None,
            )
            avail = int(enc.shape[2])
            n_end_a_eff = min(n_end_a, avail)
            if n_end_a_eff > 0:
                # Kept in DTYPE, not fp32: the freeze proof asks for an EXACT 0.0
                # and gets it only because every write is a bf16 copy of the
                # encoder's own output. Holding fp32 here and rounding at the
                # write site would leave the proof comparing two different
                # roundings of the same number.
                end_a = (
                    enc[:, :, avail - n_end_a_eff: avail, :]
                    .detach().clone().to(DTYPE)
                )
                audio_status = "frozen" if n_end_a_eff == n_end_a else "partial"
            if n_end_a_eff < n_end_a:
                logger.warning(
                    "chain: end source audio underrun — only %d encoded "
                    "audio-latent frames available but the band needs "
                    "n_end_a=%d; freezing %d frames only (silent under-freeze, "
                    "no error).",
                    avail, n_end_a, n_end_a_eff,
                )
            del audio_encoder, enc, src_audio, wf
            cleanup_memory()

    return end_v_half, end_v_full, end_a, n_end_a_eff, audio_status


def _encode_retake_window(
    *,
    retake: "RetakeSpec",
    layout: ChainLayout,
    width: int,
    height: int,
    video_encoder,
    tiling_cfg,
    ledger,
    device: torch.device,
):
    """VAE-encode the WHOLE retake window at both resolutions (+ its audio).

    Returns ``(rt_v_half, rt_v_full, rt_a, had_audio, orig_wf, sampling_rate)``.

    WHY THE WHOLE WINDOW AND NOT JUST THE TWO GLUE BANDS: the video VAE is
    causal and temporally strided, so a standalone encode of the last 25 pixels
    would make ITS latent 0 a fresh keyframe — both the wrong index mapping and
    the wrong content. One full-window encode yields correctly-aligned latents
    for both ends at once (spike ``d1_both_side_freeze.py`` lines 561-566). The
    same argument applies to the audio encode.

    Both video encodes go through ``tiled_encode`` for the same reason
    :func:`_encode_source_heads` does: an untiled full-res encode would OOM at
    720p.

    AUDIO ADJUDICATION (owner decision, VERIFICATION_LOG §55.6):
      * window HAS audio but encodes to fewer than ``a_total`` latent frames ->
        HARD FAIL. A short encode puts the tail glue at the wrong index and
        would silently invalidate the freeze; that is worse than a failed job.
      * window has NO audio -> continue with no audio freeze at all
        (``had_audio=False``), recorded in the metadata rather than raised.
      * ``regenerate_audio=False`` -> the underrun hard-fail does NOT apply: the
        delivered audio is the original waveform, so a short/absent encode can
        only under-freeze latents that are about to be discarded.
    """
    from ltx_core.model.audio_vae import encode_audio as vae_encode_audio
    from ltx_core.types import Audio
    from ltx_pipelines.utils.helpers import cleanup_memory
    from ltx_pipelines.utils.media_io import decode_audio_from_file

    window_px = int(layout.retake_window_px or 0)
    f_total = layout.f_total
    a_win = layout.a_total

    # ── half-res window (stage-1) ──
    win_half = load_video_conditioning_cpu(
        video_path=retake.path, height=height // 2, width=width // 2,
        frame_cap=window_px, dtype=DTYPE, device=device,
    )
    rt_v_half = video_encoder.tiled_encode(win_half, tiling_cfg).detach().clone()
    del win_half
    cleanup_memory()
    assert rt_v_half.shape[2] == f_total, (rt_v_half.shape[2], f_total)

    # ── full-res window (stage-2 variant-B hard freeze) ──
    win_full = load_video_conditioning_cpu(
        video_path=retake.path, height=height, width=width,
        frame_cap=window_px, dtype=DTYPE, device=device,
    )
    rt_v_full = video_encoder.tiled_encode(win_full, tiling_cfg).detach().clone()
    del win_full
    cleanup_memory()
    assert rt_v_full.shape[2] == f_total, (rt_v_full.shape[2], f_total)

    # ── window audio ──
    rt_a = None
    orig_wf = None
    sampling_rate = 0
    src_audio = decode_audio_from_file(retake.path, device)
    had_audio = src_audio is not None
    if had_audio:
        wf = src_audio.waveform                    # (1, channels, samples)
        if wf.dim() == 2:                          # defensive; loader returns 3-D
            wf = wf.unsqueeze(0)
        # The audio VAE encoder's conv_in is stereo-only (weight [128,2,3,3]),
        # so a mono window must be duplicated — same normalisation the A2V path
        # does, and the mux writer needs stereo too.
        if wf.shape[1] == 1:
            wf = wf.repeat(1, 2, 1)
        sampling_rate = int(src_audio.sampling_rate)
        # Kept on CPU for a possible verbatim re-mux (regenerate_audio=False).
        orig_wf = wf.squeeze(0).detach().to(torch.float32).cpu().contiguous()
        audio_encoder = ledger.audio_encoder()
        enc = vae_encode_audio(
            Audio(waveform=wf.to(DTYPE), sampling_rate=sampling_rate), audio_encoder, None,
        )
        avail = int(enc.shape[2])
        if avail < a_win:
            if retake.regenerate_audio:
                raise ValueError(
                    f"retake window audio encoded to {avail} audio-latent frames "
                    f"< a_total={a_win} required by the {window_px}-frame window; "
                    "the tail glue would land on the wrong latent index"
                )
            logger.warning(
                "chain retake: window audio encoded to %d < a_total=%d, but "
                "regenerate_audio=False — continuing with NO audio freeze "
                "(the delivered audio is the original waveform).",
                avail, a_win,
            )
            had_audio = False
        else:
            rt_a = enc[:, :, :a_win, :].detach().clone()
        del audio_encoder, enc
        del src_audio, wf
        cleanup_memory()
    return rt_v_half, rt_v_full, rt_a, had_audio, orig_wf, sampling_rate


def _chunked_upsample_cpu(assembled_v, video_encoder, upsampler, upsample_video_fn, device, progress=None):
    """Temporal-chunked spatial upsample. Returns the upscaled latent on CPU.

    Memory-bounded twin of the one-shot ``upsample_video`` over the whole
    timeline: the assembled stage-1 latent is parked on CPU and each
    :func:`chain_math.plan_upsample_chunks` chunk (core + halo) is streamed to the
    GPU in ``channels_last_3d``, upsampled, its halo dropped, and immediately
    evicted back to CPU with an ``empty_cache`` so only one chunk's working set is
    resident at a time (chunk-wise cache release is mandatory — skipping it
    fragments the reserved pool and spills). Cores tile the timeline with no
    gap/overlap, so the CPU ``cat`` reassembles the identical latent (halo=18
    matches the one-shot convolution interior).
    """
    plan = plan_upsample_chunks(int(assembled_v.shape[2]))
    src_cpu = assembled_v[:1].to("cpu")
    # channels_last_3d ONLY on the Conv3d layers (in-place): the upsampler also
    # holds Conv2d modules (spatial 2x, rank-4 weights), and a whole-module
    # .to(channels_last_3d) raises "required rank 5 tensor" on those.
    for m in upsampler.modules():
        if isinstance(m, torch.nn.Conv3d):
            m.to(memory_format=torch.channels_last_3d)
    out_parts = []
    for k, ch in enumerate(plan):
        xin = src_cpu[:, :, ch.in_start:ch.in_start + ch.in_len]
        xin = xin.to(device).contiguous(memory_format=torch.channels_last_3d)
        yout = upsample_video_fn(xin, video_encoder, upsampler)
        out_parts.append(yout[:, :, ch.keep_lo:ch.keep_lo + ch.keep_len].to("cpu"))
        del xin, yout
        torch.cuda.empty_cache()
        if progress:
            progress("upsample", k, len(plan))
    result = torch.cat(out_parts, dim=2)
    del out_parts, src_cpu
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main chain orchestration.
# ─────────────────────────────────────────────────────────────────────────────
@torch.inference_mode()
def run_chain(
    pipe,
    *,
    clips: list[ChainClipSpec],
    width: int,
    height: int,
    frame_rate: float,
    num_steps: int,
    seed: int,
    overlap_frames: int,
    overlap_strength: float,
    output_path: str,
    progress: ProgressFn | None = None,
    source: "SourceSpec | None" = None,
    audio_source: "AudioSourceSpec | None" = None,
    retake: "RetakeSpec | None" = None,
    end_source: "EndSourceSpec | None" = None,
    ic_loras: list[IcLoraEntry] | None = None,
    ic_reference: tuple[str, float] | None = None,
    ic_attention_strength: float = 1.0,
    chunked_upsample: bool = False,
    stage2_v_tile: int | None = None,
    stage2_v_adv: int | None = None,
    nag: NagParams | VsfParams | None = None,
) -> dict:
    """Run a masked AV-latent chain to ONE mp4. Returns metadata incl. junctions.

    All heavy weights (transformer/video_encoder) are built ONCE and reused for
    every stage-1 segment and stage-2 tile; the text encoder is loaded once,
    used to encode all distinct clip prompts, then freed (mirrors the spikes +
    DistilledPipeline ordering).

    ``audio_source`` (audio-to-video, additive to the ``audio_source=None`` path,
    which stays byte-identical) freezes an uploaded audio latent over the whole
    timeline and drives the video off it; mutually exclusive with ``source``.

    ``retake`` (temporal inpainting, additive — ``None`` keeps every other path
    byte-identical) takes an app-cut window (ONE clip, 8n+1 frames, <= one
    stage-2 tile) and regenerates only its MIDDLE, holding the head and tail glue
    bands frozen through BOTH stages. The delivered mp4 is the WHOLE window,
    untrimmed. Mutually exclusive with ``source`` and ``audio_source``.

    ``end_source`` (end source, additive — ``None`` keeps every other path
    byte-identical) freezes app-cut material as the TAIL of the timeline, so the
    chain ENDS on it. TWO MODES, chosen by chain_math from the clip count and
    reported as ``layout.end_source_mode`` (see :class:`EndSourceSpec`):

      * ONE clip -> ``"in_window"``: no extra segment, ``n_seg == len(clips) == 1``
        and the band is that one segment's own tail, so it sits inside the single
        stage-1 denoise window. The output length is the clip length, unchanged.
      * TWO OR MORE clips -> ``"internal_segment"``: chain_math APPENDS a segment
        for the band, so stage 1 runs ``n_seg == len(clips) + 1`` and the band is
        the appended segment's tail. The output is longer than the clips by
        exactly the band.

    Either way stage 2 then re-writes and re-freezes the band in EVERY tile it
    reaches, driven by ``layout.end_tile_bands``, and the output is NOT trimmed
    (unlike ``source``, whose frozen head is cut off). Mutually exclusive with
    ``retake`` and ``audio_source``, combinable with ``source`` — start + end is an
    interpolation between two given ends.

    ``ic_loras`` (style/character IC-LoRA, additive): ``(path, strength,
    audio_strength)`` adapters applied via the forward-time weight patch across
    the whole chain (the single transformer is reused for every stage-1 segment
    + stage-2 tile, so the LoRA effects the entire timeline). Set EXPLICITLY
    before the transformer is built below — an empty list clears any stale
    ``_ic_loras`` left by a prior single ``generate()`` on the resident
    pipeline, so ``ic_loras=None/[]`` is a genuine "no LoRA" (byte-identical to
    before) rather than a leak of the last job's.

    ``ic_reference`` / ``ic_attention_strength`` (additive): a control-adapter
    reference video ``(path, strength)`` plus its conditioning_attention_strength
    knob, for ANY clip count 1..24 (§1-15; the old "clips=1 only" alpha scope was
    lifted 2026-08-11). ONE long reference is laid over the assembled timeline and
    auto-sliced per stage-1 segment: segment ``i`` gets pixel window
    ``chain_math.video_segment_windows(layout)[i]`` == ``(8*s_i, clip_frames[i])``,
    decoded LAZILY (``_iter_reference_windows`` over
    ``iter_video_conditioning_cpu``) and VAE-encoded right before that segment
    denoises, via ``pipe._reference_conditioning_from_pixels`` — the same encoder
    path the single ``generate()`` uses on its stage-1 pass. Adjacent windows
    overlap by exactly the K_v carry band, so neighbours see the same footage
    across a seam. STAGE 1 ONLY; stage-2 tiles never get a reference. A reference
    shorter than the timeline is NOT an error: the windows it no longer covers are
    simply generated without one (partial windows keep the frames they do have).
    For clips=1 this reduces to the historical single window ``(0,
    clip_frames[0])``. ``ic_reference=None`` -> the chain is byte-identical to
    before: the ``_set_ic_job`` below is called with ``(loras, None, 1.0)``
    (stale-clear semantics preserved), no decode is opened and no reference latent
    is injected.

    ``stage2_v_tile`` / ``stage2_v_adv`` (stage-2 window, additive): the tile
    geometry ``compute_chain_layout`` lays the stage-2 pass out with. ``None``
    (both) -> ``chain_math.STAGE2_V_TILE`` / ``STAGE2_V_ADV``, i.e. byte-identical
    to before this knob existed. The caller resolves the preset NAME
    (``chain_math.resolve_stage2_window``); this function only ever sees numbers.

    ``nag`` (NAG negative-prompt guidance, additive): always set explicitly
    (``None`` included — the same stale-clear discipline as ``ic_loras`` above),
    via ``pipe._set_nag_job`` BEFORE the positive-prompt text encode below. This
    is EARLIER than ``_set_ic_job``'s call site further down, which can wait
    until just before the transformer build because IC-LoRA is a forward-time
    weight patch with no encode step of its own. NAG's negative prompt, by
    contrast, MUST be encoded together with the positive prompts while
    ``text_encoder`` is still alive — encoding it after ``del text_encoder``
    below would require a second Gemma load, which is exactly what the single-
    generate path's encode_text patch avoids (see fast_video_pipeline.py's D2
    ordering comment).
    """
    assert not (source is not None and audio_source is not None), (
        "run_chain: source (V2V) and audio_source (A2V) are mutually exclusive"
    )
    # Retake owns BOTH ends of the one and only clip, so it can share the
    # timeline with neither of the other two (the API layer 422s this first).
    assert sum(x is not None for x in (source, audio_source, retake)) <= 1, (
        "run_chain: source (V2V), audio_source (A2V) and retake are mutually exclusive"
    )
    # An end source freezes the timeline's TAIL. Retake already owns both ends of
    # its single window, and A2V owns the whole audio timeline while its video is
    # driven off it — neither can share the timeline with an end source (the API
    # layer 422s both first). ``source`` is the one that CAN combine: it owns the
    # opposite end, which is exactly the start+end interpolation case.
    assert not (end_source is not None and retake is not None), (
        "run_chain: end_source and retake are mutually exclusive"
    )
    assert not (end_source is not None and audio_source is not None), (
        "run_chain: end_source and audio_source (A2V) are mutually exclusive"
    )
    # A reference is API-exclusive with V2V, retake AND end source (api/models.py),
    # and the per-segment injection below LEANS on that TWICE: it sits after the
    # head-branch chain, so a reference reaching such a chain would append
    # conditioning to branches whose comments (and behaviour) say they have none —
    # and its window generator is consumed strictly in ascending segment order,
    # which the end source's ``reverse`` schedule does not follow (asserted again
    # at the generator's construction site). Assert rather than branch — the
    # exclusivity is the invariant, not a case to handle.
    assert ic_reference is None or (
        source is None and retake is None and end_source is None
    ), (
        "run_chain: ic_reference is mutually exclusive with source (V2V), retake "
        "and end_source"
    )
    from ltx_core.components.diffusion_steps import EulerDiffusionStep
    from ltx_core.components.noisers import GaussianNoiser
    from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
    from ltx_core.model.upsampler import upsample_video
    from ltx_core.model.video_vae import decode_video as vae_decode_video
    from ltx_core.text_encoders.gemma import encode_text
    from ltx_core.types import AudioLatentShape, VideoLatentShape, VideoPixelShape
    from ltx_pipelines.utils.constants import (
        DISTILLED_SIGMA_VALUES,
        STAGE_2_DISTILLED_SIGMA_VALUES,
    )
    from ltx_pipelines.utils.helpers import cleanup_memory

    dp = pipe.pipeline                # DistilledPipeline
    device = dp.device
    ledger = dp.model_ledger
    components = dp.pipeline_components
    n = len(clips)
    kv = int(overlap_frames)
    stage1_mask_value = 1.0 - max(0.0, min(1.0, float(overlap_strength)))

    tiling_cfg = default_tiling_config(
        spatial_tile_size=pipe._vae_spatial_tile_size,
        temporal_tile_size=pipe._vae_temporal_tile_size,
    )

    clip_frames = [c.num_frames for c in clips]
    src_ctx_px = int(source.context_frames) if source is not None else None
    # Stage-2 window geometry: ``None`` -> the module defaults, so the layout
    # (and therefore every pixel downstream) is byte-identical to before this
    # knob existed. The caller resolved the preset NAME; only the resolved
    # numbers reach here, keeping this function preset-agnostic.
    v_tile = STAGE2_V_TILE if stage2_v_tile is None else int(stage2_v_tile)
    v_adv = STAGE2_V_ADV if stage2_v_adv is None else int(stage2_v_adv)
    layout: ChainLayout = compute_chain_layout(
        clip_frames, frame_rate, kv=kv,
        v_tile=v_tile, v_adv=v_adv,
        source_context_px=src_ctx_px,
        retake_glue_px=(
            None if retake is None else (int(retake.head_px), int(retake.tail_px))
        ),
        end_context_px=(
            None if end_source is None else int(end_source.context_frames)
        ),
    )
    n_ctx_v = layout.n_ctx_v  # 0 when source is None
    # Frozen TAIL latents of the end source — 0 when end_source is None, which is
    # what keeps every write/slice below inert on the ordinary path.
    n_end_v = layout.n_end_v
    # STAGE-1 SEGMENT COUNT, which is NOT the clip count in the end source's
    # ``internal_segment`` mode: there chain_math appends an internal band segment
    # to ``seg_frames`` (kv carry + the band), so ``n_seg == n + 1``. In its
    # ``in_window`` and ``reverse`` modes (the band is the LAST clip's own tail)
    # and on every chain without an end source, ``n_seg == n``. The loop, the
    # seeds and the progress denominator all run on ``n_seg``; ``n`` survives only
    # as the number of USER clips (metadata ``n_clips``, clip-0 conditioning),
    # which is what a client counts.
    seg_frames = layout.seg_frames
    n_seg = len(seg_frames)
    assert n_seg == n + (
        1 if layout.end_source_mode == "internal_segment" else 0
    ), (n_seg, n, layout.end_source_mode)
    # Retake glue sizes — the SINGLE source of truth is the layout, so the app
    # validator, the mock and this engine cannot disagree about which latents
    # are frozen. All zero on every non-retake path.
    rt_geo = layout.to_dict().get("retake") or {}
    n_head_v = int(rt_geo.get("n_head_v", 0))
    n_tail_v = int(rt_geo.get("n_tail_v", 0))
    n_head_a = int(rt_geo.get("n_head_a", 0))
    n_tail_a = int(rt_geo.get("n_tail_a", 0))
    ka_list = layout.ka_list
    v_tiles = layout.v_tiles
    a_tiles = layout.a_tiles
    kt_v, kt_a = layout.kt_v, layout.kt_a
    n_tiles = layout.n_tiles
    total_px = layout.total_px

    base_seed = int(seed)
    # One seed per SEGMENT (the band segment gets the next one in the run). With no
    # end source n_seg == n, so the list — and therefore the metadata — is
    # unchanged.
    seeds = [base_seed + i for i in range(n_seg)]

    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()

    # ── NAG job state — set BEFORE encoding (see this function's docstring for
    # why this is earlier than _set_ic_job below): the negative prompt must be
    # encoded together with the positives while text_encoder is still alive.
    pipe._set_nag_job(nag)

    # ── Text encode ONCE for all DISTINCT prompts, then free the encoder. ─────
    # F2: announce the encode phase (fires BEFORE the encode so the app's job
    # status can show "encoding" during the wait; the first stage1_denoise step
    # event implicitly ends it).
    if progress:
        progress("encode", 0, 1)
    text_encoder = ledger.text_encoder()
    distinct = list(dict.fromkeys(c.prompt for c in clips))  # preserves order, dedups
    ctx_by_prompt: dict[str, tuple] = {}
    for p in distinct:
        vctx, actx = encode_text(text_encoder, prompts=[p])[0]
        ctx_by_prompt[p] = (vctx, actx)
    # NAG/VSF negative encode: one extra call against the SAME live text_encoder
    # (avoids a second Gemma load), stashed in NagState before it's freed below.
    # Inert when this chain didn't request either (pipe._nag.requested is False).
    # VSF additionally trims the encoding to the prompt's real tokens — its
    # single shared softmax must not sign-flip the connector's learned register
    # embeddings (see encode_negative's docstring); NAG passes False and stays
    # byte-identical.
    if pipe._nag.requested:
        nvc, nac = encode_negative(
            text_encoder,
            pipe._nag.params.negative_prompt,
            slice_to_real_tokens=isinstance(pipe._nag.params, VsfParams),
        )
        pipe._nag.set_contexts(nvc, nac)
    torch.cuda.synchronize()
    del text_encoder
    cleanup_memory()

    seg_ctx = [ctx_by_prompt[c.prompt] for c in clips]
    # The internal band segment REUSES the last clip's encoding — it is the tail of
    # that same scene, and its audio (the one modality still generated freely there)
    # should read as a continuation of it, not as a scene change. Reusing the tuple
    # costs ZERO extra text encodes: this appends a reference, and the encoder was
    # already freed above. ONLY ``internal_segment`` mode has such a segment — this
    # list must stay one entry per stage-1 segment, so appending in ``in_window``
    # mode (where seg_frames == clip_frames) would leave it one too long.
    if layout.end_source_mode == "internal_segment":
        seg_ctx.append(seg_ctx[-1])

    # ── audio-to-video: encode the uploaded waveform to a frozen audio latent ──
    # ONE encode (mirrors _encode_source_heads' audio path), then free the tiny
    # ~46MB audio_encoder. Done here — before the big video_encoder/transformer
    # build — to keep VRAM lowest (G0 spike ordering). The ORIGINAL waveform is
    # kept on CPU for the final mux; the vocoder is skipped entirely so the
    # delivered audio track == the upload. a2v_a is the a_total-length latent
    # frozen across every stage-1 segment + stage-2 tile.
    a2v_a = None                 # (1, C, a_total, F) frozen audio latent
    a2v_orig_wf = None           # (channels, samples) stereo CPU float32 — mux
    a2v_sr = 0
    a2v_avail = 0
    a_seg_windows: list[tuple[int, int]] | None = None
    if audio_source is not None:
        from ltx_core.model.audio_vae import encode_audio as vae_encode_audio
        from ltx_core.types import Audio
        from ltx_pipelines.utils.media_io import decode_audio_from_file

        a_total = layout.a_total
        src_audio = decode_audio_from_file(audio_source.path, device)
        if src_audio is None:
            raise ValueError(
                f"audio_source has no decodable audio stream: {audio_source.path}"
            )
        wf = src_audio.waveform                    # (1, channels, samples)
        if wf.dim() == 2:                          # defensive; loader returns 3-D
            wf = wf.unsqueeze(0)
        # The audio VAE encoder's conv_in expects STEREO (weight [128,2,3,3]); a
        # mono upload must be duplicated to 2 channels — both for the encode and
        # for the stereo-only mux writer (G0 finding).
        if wf.shape[1] == 1:
            wf = wf.repeat(1, 2, 1)
        a2v_sr = int(src_audio.sampling_rate)
        a2v_orig_wf = wf.squeeze(0).detach().to(torch.float32).cpu().contiguous()  # (2, S)

        audio_encoder = ledger.audio_encoder()
        enc = vae_encode_audio(
            Audio(waveform=wf.to(DTYPE), sampling_rate=a2v_sr), audio_encoder, None,
        )
        a2v_avail = int(enc.shape[2])
        if a2v_avail < a_total:
            raise ValueError(
                f"audio_source too short: encoded {a2v_avail} audio-latent frames "
                f"< required a_total={a_total} for the {total_px}-pixel-frame "
                "timeline (video length is authoritative; audio is truncated, "
                "never padded)."
            )
        a2v_a = enc[:, :, :a_total, :].detach().clone()
        del audio_encoder, enc, src_audio, wf
        cleanup_memory()
        a_seg_windows = audio_segment_windows(layout)

    # ── IC-LoRA state MUST be set before the transformer is built/fetched ──────
    # The forward-time weight patch reads pipe._ic_loras via the provider wired at
    # transformer build; set it here (explicitly, empty list = clear) so THIS
    # chain's style adapters — and ONLY this chain's — apply. Clearing on the empty
    # path is the stale-detach that keeps a prior single generate()'s LoRA from
    # bleeding into the chain denoise (mirrors generate()'s _set_ic_job call). A
    # control-adapter reference (any clip count — §1-15) is forwarded here too so
    # _set_ic_job resolves its downscale factor + attention wrapper exactly as the
    # single path; ic_reference=None keeps the historical (None, 1.0) stale-clear.
    pipe._set_ic_job(list(ic_loras or []), ic_reference, ic_attention_strength)

    # ── Build video_encoder + transformer ONCE (reuse for stage1 + stage2). ───
    video_encoder = ledger.video_encoder()
    transformer = ledger.transformer()
    # Drop the Windows-stranded reserved pool from the block-swap load-then-evict
    # (Phase 5B fix; the chain path bypasses the worker's denoise_audio_video
    # wrapper, so release explicitly here before the first heavy denoise).
    gc.collect()
    torch.cuda.empty_cache()

    # ── video-to-video: encode the source tail into frozen HEAD latents ───────
    src_head_v_half = src_head_v_full = src_head_a = None
    freeze_ka = 0
    source_had_audio = False
    if source is not None:
        (src_head_v_half, src_head_v_full, src_head_a,
         freeze_ka, source_had_audio) = _encode_source_heads(
            source=source, layout=layout, width=width, height=height,
            video_encoder=video_encoder, tiling_cfg=tiling_cfg,
            ledger=ledger, device=device,
        )
        gc.collect()
        torch.cuda.empty_cache()

    # ── end source: encode the tail material (both resolutions + its audio) ───
    end_v_half = end_v_full = end_a = None
    n_end_a_eff = 0
    end_audio_status = "disabled"
    # The AUDIO band's per-tile stage-2 plan. Normally the layout's own, but a
    # short encode (``"partial"``) freezes fewer latents than the geometry says,
    # and the tiles must be re-cut to THAT width — one call to the same pure
    # function chain_math used. Left empty when no audio is frozen, which is
    # exactly when the stage-2 write below is skipped.
    end_bands_a: list[tuple[int, int]] = []
    if end_source is not None:
        (end_v_half, end_v_full, end_a, n_end_a_eff,
         end_audio_status) = _encode_end_source(
            end_source=end_source, layout=layout, width=width, height=height,
            video_encoder=video_encoder, tiling_cfg=tiling_cfg,
            ledger=ledger, device=device,
        )
        if end_a is not None:
            end_bands_a = (
                layout.end_tile_bands_a if n_end_a_eff == layout.n_end_a
                else tail_tile_bands(layout.a_tiles, layout.a_total, n_end_a_eff)
            )
        gc.collect()
        torch.cuda.empty_cache()

    # ── retake: encode the WHOLE window (both resolutions + audio) once ───────
    rt_v_half = rt_v_full = rt_a = None
    rt_orig_wf = None
    rt_sr = 0
    retake_had_audio = False
    if retake is not None:
        (rt_v_half, rt_v_full, rt_a, retake_had_audio,
         rt_orig_wf, rt_sr) = _encode_retake_window(
            retake=retake, layout=layout, width=width, height=height,
            video_encoder=video_encoder, tiling_cfg=tiling_cfg,
            ledger=ledger, device=device,
        )
        if not retake_had_audio:
            # No audio to freeze -> the glue bands are video-only (recorded in
            # the metadata; NOT an error — owner adjudication §55.6).
            n_head_a = n_tail_a = 0
        gc.collect()
        torch.cuda.empty_cache()

    # ── clip-wise IC-LoRA reference (§1-15): ONE lazy decode of the long
    # reference, cut into the per-stage-1-segment windows chain_math computed.
    # Constructed here (the video_encoder exists; the loop is next) but NOT
    # consumed: both generators are lazy, so the first frame is decoded only when
    # segment 0 asks for its window and each later window is decoded+encoded just
    # before the segment that needs it — no silent multi-minute block up front, and
    # at most max(clip_frames) reference frames buffered at a time.
    ref_windows: Iterator[torch.Tensor | None] | None = None
    ref_scale = 1
    ref_strength = 1.0
    ref_cond_kwargs: dict = {}
    if ic_reference is not None:
        # THE GENERATOR IS CONSUMED ONCE, IN ASCENDING WINDOW ORDER: it walks the
        # decode stream forwards and cannot go back, so a stage-1 schedule that
        # visits the segments out of order would hand each one the WRONG window.
        # A reference is API-exclusive with the end source (the only mode with
        # such a schedule), so this cannot happen — asserted rather than assumed,
        # because the failure would be a silently mismatched reference.
        assert layout.seg_generation_order == list(range(n_seg)), (
            layout.seg_generation_order
        )
        ref_path, ref_strength = ic_reference
        ref_px_windows = video_segment_windows(layout)
        # Same guards + same arithmetic the single generate() path uses, reached
        # through the shared helper so the two cannot drift.
        ref_scale, ref_h, ref_w = pipe._reference_pixel_dims(height // 2, width // 2)
        ref_cond_kwargs = {
            "height": height // 2,
            "width": width // 2,
            "video_encoder": video_encoder,
            "dtype": DTYPE,
            "device": device,
            # Tiled-encoded there: factor-1 references (deblur) always, plus any
            # reference above REFERENCE_ENCODE_TILE_TOKEN_BUDGET tokens (§3-76).
            "tiling_config": tiling_cfg,
        }
        ref_windows = _iter_reference_windows(
            iter_video_conditioning_cpu(
                video_path=ref_path,
                height=ref_h,
                width=ref_w,
                # The last window's end == everything the chain can consume
                # (== layout.total_px for 8n+1 clips); decoding beyond it is waste.
                frame_cap=ref_px_windows[-1][0] + ref_px_windows[-1][1],
                dtype=DTYPE,
                device=device,
            ),
            ref_px_windows,
        )

    stepper = EulerDiffusionStep()
    stage1_sigmas = torch.Tensor(DISTILLED_SIGMA_VALUES).to(device)

    # ── STAGE 1: per-segment (half-res) with carry+freeze. ────────────────────
    # THE LOOP RUNS IN GENERATION ORDER, NOT TIMELINE ORDER. ``i`` is always the
    # TIMELINE index (it picks the seed, the prompt, the clip and the slot the
    # result is stored in); ``order_idx`` is only how far through the schedule we
    # are. On every mode but the end source's ``reverse`` the two coincide and
    # ``seg_generation_order`` is ``[0..n_seg)``, so this is the same loop it has
    # always been. The slots are PRE-ALLOCATED because a reverse schedule fills
    # them out of order — the assembly below asserts none stayed empty.
    seg_v: list[torch.Tensor | None] = [None] * n_seg
    seg_a: list[torch.Tensor | None] = [None] * n_seg
    # What stage 1 ACTUALLY did, recorded as it goes: the order it visited the
    # segments in and the four freeze widths each one ran with. Every other
    # end-source number in the metadata is a copy of a ``chain_math`` value, so
    # this is the only record that can show the engine really ran the schedule
    # the layout declared (published under ``end_source`` in the meta below).
    stage1_order: list[int] = []
    stage1_freezes: list[dict] = []
    for order_idx, i in enumerate(layout.seg_generation_order):
        # F2: tag the upcoming wheel denoising loop with its chain position so
        # the per-step tqdm shim events carry segment context (observation only).
        # The TIMELINE index is the useful tag here — the observer wants to know
        # which part of the video is being made, not where we are in the queue.
        progress_shim.set_phase("stage1_denoise", outer_index=i, outer_total=n_seg)
        # SHAPE COMES FROM ``layout.seg_frames``, NOT ``clip_frames``: in the end
        # source's legacy ``internal_segment`` mode the last entry is the appended
        # band segment, which has no clip of its own. Everywhere else the two
        # lists are equal element-for-element, so this is byte-identical there.
        seg_shape = VideoPixelShape(1, seg_frames[i], height // 2, width // 2, frame_rate)
        noiser = GaussianNoiser(generator=torch.Generator(device=device).manual_seed(seeds[i]))
        # Tail freeze is retake / end source / the reverse のり代 only; 0 everywhere
        # else keeps every other branch below exactly as it was.
        # ``seg_tail_mask_value`` is initialised here (not next to seg_mask_value
        # further down) because the two blocks that set it sit earlier — None
        # means "hold the tail at the same strength as the head", which is what
        # every pre-end-source caller did AND what the reverse carry wants.
        ftv = fta = 0
        seg_tail_mask_value: float | None = None
        # Initialised alongside its video twin for the same reason: the end
        # source block that sets it sits earlier than the mask-value block
        # further down. None means "the audio tail is held at whatever the video
        # tail resolves to", which is what every caller but the end source wants.
        seg_audio_tail_mask_value: float | None = None
        if i == 0 and retake is not None:
            # ── retake: freeze BOTH ends of the single window segment ────────
            # The init tensor carries the ORIGINAL window's stage-1-res latents
            # at the two glue bands and zeros in the middle; the mask (built in
            # _denoise_av_with_carry from fkv/ftv/fka/fta) is what actually
            # holds them. Stage-1 mask value is 0.0 — a HARD freeze. 0.5 also
            # passed the spike (the band drifts 0.62-0.93 at stage 1 but
            # reconverges at stage 2); 0.0 is the owner-chosen default because
            # it makes the stage-1 bands bit-exact and therefore PROVABLE
            # (VERIFICATION_LOG §55.3).
            from ltx_core.types import AudioLatentShape as _ALShape
            from ltx_core.types import VideoLatentShape as _VLShape
            v_half_shape = _VLShape.from_pixel_shape(
                seg_shape,
                latent_channels=components.video_latent_channels,
                scale_factors=components.video_scale_factors,
            ).to_torch_shape()
            init_v = torch.zeros(tuple(v_half_shape), dtype=DTYPE, device=device)
            init_v[:, :, :n_head_v] = rt_v_half[:, :, :n_head_v].to(DTYPE)
            init_v[:, :, layout.f_total - n_tail_v:] = (
                rt_v_half[:, :, layout.f_total - n_tail_v:].to(DTYPE)
            )
            if n_head_a > 0 or n_tail_a > 0:
                a_shape = _ALShape.from_video_pixel_shape(seg_shape).to_torch_shape()
                init_a = torch.zeros(tuple(a_shape), dtype=DTYPE, device=device)
                init_a[:, :, :n_head_a] = rt_a[:, :, :n_head_a].to(DTYPE)
                init_a[:, :, layout.a_total - n_tail_a:] = (
                    rt_a[:, :, layout.a_total - n_tail_a:].to(DTYPE)
                )
            else:
                init_a = None
            fkv, fka = n_head_v, n_head_a
            ftv, fta = n_tail_v, n_tail_a
            conds = []  # retake & conditioning_images are mutually exclusive (app-enforced)
        elif i == 0 and source is not None:
            # video-to-video: freeze the source tail as clip-0's head (same
            # carry mechanism as an inter-clip join; mask = 1-overlap_strength).
            from ltx_core.types import AudioLatentShape as _ALShape
            from ltx_core.types import VideoLatentShape as _VLShape
            v_half_shape = _VLShape.from_pixel_shape(
                seg_shape,
                latent_channels=components.video_latent_channels,
                scale_factors=components.video_scale_factors,
            ).to_torch_shape()
            init_v = torch.zeros(tuple(v_half_shape), dtype=DTYPE, device=device)
            init_v[:, :, :n_ctx_v] = src_head_v_half.to(DTYPE)
            if freeze_ka > 0:
                a_shape = _ALShape.from_video_pixel_shape(seg_shape).to_torch_shape()
                init_a = torch.zeros(tuple(a_shape), dtype=DTYPE, device=device)
                init_a[:, :, :freeze_ka] = src_head_a.to(DTYPE)
            else:
                init_a = None
            fkv, fka = n_ctx_v, freeze_ka
            conds = []  # source & conditioning_images are mutually exclusive (app-enforced)
        elif layout.seg_head_source[i] is not None:
            # ── forward carry: freeze the PREVIOUS segment's tail as this head ─
            # ``h`` is that previous segment, read from the layout's table rather
            # than assumed to be ``i - 1``. On every schedule that existed before
            # the table it IS ``i - 1``, so this branch is unchanged there; in the
            # end source's ``reverse`` mode the table is all-None and the branch
            # never fires (every head is free).
            h = layout.seg_head_source[i]
            ka_i = ka_list[h]      # the join between h and i
            prev_v, prev_a = seg_v[h], seg_a[h]
            assert prev_v is not None and prev_a is not None, (i, h)
            # Size the init tensors from the CURRENT segment's latent shapes —
            # NOT zeros_like(prev_*). The previous segment may have a different
            # num_frames (unequal clip lengths are legal), so its latent shape
            # need not match this segment's create_initial_state target. Mirror
            # the i==0/source branch: build the current segment's shape, then
            # copy the K_v / K_a tail of the previous segment into the frozen
            # head. dtype/device are preserved from the previous segment.
            v_shape = VideoLatentShape.from_pixel_shape(
                seg_shape,
                latent_channels=components.video_latent_channels,
                scale_factors=components.video_scale_factors,
            ).to_torch_shape()
            init_v = torch.zeros(tuple(v_shape), dtype=prev_v.dtype, device=prev_v.device)
            init_v[:, :, :kv] = prev_v[:, :, prev_v.shape[2] - kv:]
            a_shape = AudioLatentShape.from_video_pixel_shape(seg_shape).to_torch_shape()
            init_a = torch.zeros(tuple(a_shape), dtype=prev_a.dtype, device=prev_a.device)
            init_a[:, :, :ka_i] = prev_a[:, :, prev_a.shape[2] - ka_i:]
            fkv, fka = kv, ka_i
            conds = []
        else:
            # ── free head: nothing is frozen at the front of this segment ─────
            # The historical ``i == 0`` case, now stated as "no head source". In
            # the end source's ``reverse`` mode EVERY segment lands here, which is
            # the mode's premise: each one invents its own opening and is steered
            # only by what it must END on.
            init_v = init_a = None
            fkv = fka = 0
            # Clip-0 conditioning at HALF resolution (stage-1), and ONLY on the
            # TIMELINE's first clip — the keyframes belong to the start of the
            # video, not to whichever segment happens to be generated first. In
            # ``reverse`` mode that segment is generated LAST, which is exactly
            # what the API contract already says (``clips[0].conditioning_images``).
            conds = (
                _build_video_conditionings(
                    clips[0].images, height=height // 2, width=width // 2,
                    video_encoder=video_encoder, device=device,
                )
                if i == 0 else []
            )
        # ── reverse carry (additive): freeze the NEXT segment's HEAD as this ──
        #    segment's TAIL — the mirror image of the forward carry above.
        # Another INDEPENDENT if for the same reason the end-source block below
        # is one: it does not replace a branch, it adds a frozen band at the far
        # end of whatever that branch built. ``seg_tail_source`` is all-None on
        # every mode but ``reverse``, so this is inert everywhere else.
        #
        # The strength is deliberately NOT set here: leaving ``seg_tail_mask_value``
        # at None makes ``chain_math.freeze_mask_values`` hold the tail at the same
        # ``1 - overlap_strength`` as a forward seam, for video AND audio alike.
        # That is the mirror: the user's one "seam blend" knob governs both
        # directions. IT HOLDS ONLY WHILE ``audio_mask_value`` AND
        # ``audio_tail_mask_value`` ARE ALSO None — ``freeze_mask_values`` returns
        # all four bands as ``mask_value`` only then. The first is A2V's, which is
        # API-exclusive with an end source; the second is the end source's own
        # audio band, and it is set ONLY on the LAST segment, whose tail carries
        # the band instead of a reverse のり代 (``seg_tail_source[-1] is None``,
        # asserted below). So no segment ever reaches this line with either set,
        # and unlocking either pairing means revisiting it.
        t_src = layout.seg_tail_source[i]
        if t_src is not None:
            next_v, next_a = seg_v[t_src], seg_a[t_src]
            # The schedule's whole promise, checked rather than trusted: the
            # segment we are borrowing from has already been generated.
            assert next_v is not None and next_a is not None, (i, t_src)
            # The join on THIS segment's tail side is ``ka_list[i]`` — the
            # forward carry's ``ka_list[h]`` is the one on its head side. Getting
            # these two the wrong way round would misalign the audio crossfade.
            ka_i = ka_list[i]
            if init_v is None:
                v_shape = VideoLatentShape.from_pixel_shape(
                    seg_shape,
                    latent_channels=components.video_latent_channels,
                    scale_factors=components.video_scale_factors,
                ).to_torch_shape()
                init_v = torch.zeros(
                    tuple(v_shape), dtype=next_v.dtype, device=next_v.device
                )
            if init_a is None:
                a_shape = AudioLatentShape.from_video_pixel_shape(
                    seg_shape).to_torch_shape()
                init_a = torch.zeros(
                    tuple(a_shape), dtype=next_a.dtype, device=next_a.device
                )
            init_v[:, :, init_v.shape[2] - kv:] = next_v[:, :, :kv]
            init_a[:, :, init_a.shape[2] - ka_i:] = next_a[:, :, :ka_i]
            ftv, fta = kv, ka_i
        # ── end source (additive): freeze the LAST SEGMENT's TAIL ─────────────
        # Deliberately an INDEPENDENT if, not another elif: an end source does not
        # replace any branch above, it ADDS a frozen band at the far end of the
        # last segment, on top of whatever that branch built.
        #
        # THE CONDITION IS THE TIMELINE INDEX, NOT THE GENERATION ORDER: the band
        # belongs at the end of the VIDEO. In ``reverse`` mode that segment is the
        # one generated FIRST, which is the whole point of the schedule.
        #
        # WHAT "THE LAST SEGMENT" IS depends on the mode:
        #
        #   * ``internal_segment`` (API-unreachable) — it is the appended BAND
        #     SEGMENT, never a user clip, and always i >= 1 (n_seg == n + 1 >= 2),
        #     so the carry branch above has necessarily built ``init_v``: kv
        #     latents of carry-over followed by exactly this band.
        #   * ``in_window`` (1 clip) — it is the user's ONLY clip, i == 0, so any
        #     of the three i==0 branches may have run. The V2V and retake ones
        #     leave a real ``init_v``; the plain t2v one leaves it None, and the
        #     band still has to be written somewhere. Hence the zeros arm below,
        #     which the internal-segment design had made unreachable.
        #   * ``reverse`` (2+ clips) — it is the user's LAST clip, whose head is
        #     free and whose tail is this band, so it too arrives with ``init_v``
        #     None and takes the same zeros arm. A start source cannot reach it
        #     (chain_math 422s that combination) and the reverse carry cannot have
        #     claimed the same tail (``seg_tail_source[-1] is None``, asserted in
        #     chain_math and again here).
        if end_source is not None and i == n_seg - 1:
            # The band and a reverse carry would write the SAME latents; the
            # layout guarantees they never both apply, and this is where that
            # guarantee is spent.
            assert layout.seg_tail_source[i] is None, (i, layout.seg_tail_source)
            if init_v is None:
                # Only a free-head branch can land here (i == 0 with no source and
                # no retake, or the last clip of a ``reverse`` schedule).
                assert layout.end_source_mode in ("in_window", "reverse"), (
                    layout.end_source_mode
                )
                v_half_shape = VideoLatentShape.from_pixel_shape(
                    seg_shape,
                    latent_channels=components.video_latent_channels,
                    scale_factors=components.video_scale_factors,
                ).to_torch_shape()
                init_v = torch.zeros(tuple(v_half_shape), dtype=DTYPE, device=device)
                # ``init_a`` is NOT allocated here — see the audio block below,
                # which allocates it under its own condition. The two are not the
                # same question: a segment can arrive with a real ``init_v`` and
                # no ``init_a`` at all.
            # THE INDEX IS THIS SEGMENT'S OWN LATENT LENGTH — never blindly
            # ``layout.f_total``. f_total is the length of the WHOLE assembled
            # timeline; it coincides with a segment's length only when there is
            # exactly one segment (retake's degenerate shape, and ``in_window``
            # mode). Copying that idiom into a multi-segment mode would freeze the
            # band at the wrong position (or out of range).
            seg_L = init_v.shape[2]
            if layout.end_source_mode == "in_window":
                assert seg_L == layout.f_total, (seg_L, layout.f_total)
            elif layout.end_source_mode == "reverse":
                assert seg_L == layout.seg_latent[-1], (seg_L, layout.seg_latent)
            else:
                assert seg_L == layout.end_segment_latent, (
                    seg_L, layout.end_segment_latent
                )
            init_v[:, :, seg_L - n_end_v:] = end_v_half.to(init_v.dtype)
            ftv = n_end_v
            # The two ends want DIFFERENT strengths whenever this segment has a
            # head at all (an inter-segment carry at 1 - overlap_strength, or a
            # V2V context head), while the tail defaults to a HARD freeze so the
            # chain really lands on the given material. That split is exactly
            # what tail_mask_value is for. A plain t2v in-window segment has no
            # frozen head (fkv == 0), so only the tail value is doing anything.
            # ``end_source.strength`` softens the VIDEO tail alone (stage 2
            # always hard-freezes regardless): mask value ``1.0 - strength``,
            # clamped the same way ``stage1_mask_value`` clamps
            # ``overlap_strength``. The AUDIO tail below is unaffected by it.
            seg_tail_mask_value = 1.0 - max(0.0, min(1.0, float(end_source.strength)))
            # ── the material's own AUDIO band, on the same segment's tail ─────
            # THE CONDITION IS ``end_a``, NEVER ``end_source``: with
            # END_SOURCE_FREEZE_AUDIO off, or a still image, or a video with no
            # audio track, ``end_a`` is None and this block does not exist — which
            # is what makes the rollback exact rather than approximate.
            if end_a is not None:
                # ALLOCATED UNDER ITS OWN ``if``, not inside the ``init_v is
                # None`` arm above: "``init_v`` real, ``init_a`` None" is a REAL
                # combination — V2V + end source on ONE clip (the interpolation
                # use the API explicitly allows) with a source video that has no
                # audio track leaves exactly that, and an indexed write into None
                # would crash the job.
                if init_a is None:
                    a_shape = AudioLatentShape.from_video_pixel_shape(
                        seg_shape).to_torch_shape()
                    init_a = torch.zeros(tuple(a_shape), dtype=DTYPE, device=device)
                # This segment's OWN audio length, for the same reason seg_L is
                # the segment's own video length.
                seg_aL = init_a.shape[2]
                assert seg_aL == layout.seg_audio[i], (seg_aL, layout.seg_audio)
                init_a[:, :, seg_aL - n_end_a_eff:] = end_a.to(init_a.dtype)
                fta = n_end_a_eff
                # ALWAYS 0.0 — a HARD freeze, whatever ``strength`` says. The
                # split between this and the video tail's ``1 - strength`` is the
                # whole reason freeze_mask_values grew a fourth argument.
                seg_audio_tail_mask_value = 0.0
        # ── clip-wise IC-LoRA reference (§1-15, additive) ─────────────────────
        # THIS segment's window of the long reference, decoded + VAE-encoded right
        # here (one window per iteration, never pre-batched) and appended to
        # whatever conditioning the branch above produced — which for i >= 1 is the
        # empty list that used to make every clip but the first unconditioned.
        # ``ref_windows is None`` (no reference) leaves every branch above exactly
        # as it was; clips=1 yields the single (0, clip_frames[0]) window, i.e. the
        # historical clip-0-only injection. ``None`` from the generator == the
        # reference ran out: that segment generates WITHOUT one (owner decision:
        # never an error). The encoded latent stays on the GPU —
        # VideoConditionByReferenceLatent.apply_to indexes it against the latent
        # state's device, so a CPU copy would break, not save.
        if ref_windows is not None:
            ref_px = next(ref_windows, None)
            if ref_px is not None:
                conds = conds + pipe._reference_conditioning_from_pixels(
                    ref_px,
                    cond_kwargs=ref_cond_kwargs,
                    scale=ref_scale,
                    strength=ref_strength,
                )
                del ref_px
        # audio-to-video (additive): override the audio init/freeze with the
        # uploaded audio latent's window for this segment and HARD-freeze that
        # window (audio mask 0.0) over the whole segment. The VIDEO mask is left
        # at the ordinary ``stage1_mask_value`` (= 1 - overlap_strength), which is
        # what makes LONG A2V (2..24 clips) correct: from i>=1 the video head
        # freeze is fkv == kv > 0, so reusing 0.0 as the single mask_value would
        # weld every segment seam shut and silently discard overlap_strength.
        # At i == 0 (fkv == 0) the video mask addresses nothing, so single-clip
        # A2V is bit-identical to the pre-long-A2V behaviour.
        seg_mask_value = stage1_mask_value
        seg_audio_mask_value: float | None = None
        if retake is not None:
            # HARD freeze at stage 1 (see the retake branch above); the same
            # value covers head and tail, so ``seg_tail_mask_value`` stays None
            # (which _denoise_av_with_carry resolves back to mask_value). The END
            # SOURCE is the case that broke that symmetry — its head is an
            # ordinary carry while its tail must be hard-frozen — and that is why
            # the stage-1 call below now passes ``tail_mask_value`` explicitly
            # instead of leaving it implicit as it did before.
            seg_mask_value = RETAKE_STAGE1_MASK_VALUE
        if audio_source is not None:
            ws, wl = a_seg_windows[i]
            a_shape = AudioLatentShape.from_video_pixel_shape(seg_shape).to_torch_shape()
            init_a = torch.zeros(tuple(a_shape), dtype=DTYPE, device=device)
            init_a[:, :, :wl] = a2v_a[:, :, ws:ws + wl].to(DTYPE)
            fka = wl
            seg_audio_mask_value = 0.0
        vctx, actx = seg_ctx[i]
        vstate, astate = _denoise_av_with_carry(
            output_shape=seg_shape, components=components, transformer=transformer,
            video_context=vctx, audio_context=actx, video_conditionings=conds,
            noiser=noiser, stepper=stepper, sigmas=stage1_sigmas, noise_scale=1.0,
            initial_video_latent=init_v, initial_audio_latent=init_a,
            freeze_kv=fkv, freeze_ka=fka, mask_value=seg_mask_value, device=device,
            freeze_tail_v=ftv, freeze_tail_a=fta,
            tail_mask_value=seg_tail_mask_value,
            audio_mask_value=seg_audio_mask_value,
            audio_tail_mask_value=seg_audio_tail_mask_value,
        )
        seg_v[i] = vstate.latent.detach().clone()
        seg_a[i] = astate.latent.detach().clone()
        stage1_order.append(i)
        stage1_freezes.append(
            {"seg": i, "fkv": int(fkv), "ftv": int(ftv),
             "fka": int(fka), "fta": int(fta)}
        )
        if progress:
            # The NUMERATOR IS THE SCHEDULE POSITION, not the timeline index: a
            # progress bar must go forwards, and under ``reverse`` the timeline
            # index counts down.
            progress("stage1", order_idx, n_seg)

    # ── Assemble ONE continuous stage-1 AV latent. ────────────────────────────
    # Assembly is TIMELINE order, always — it knows nothing about the schedule.
    # The slots were pre-allocated, so this is where a schedule that skipped one
    # would show up (and it would show up as a shape error much later otherwise).
    assert all(x is not None for x in seg_v), layout.seg_generation_order
    assert all(x is not None for x in seg_a), layout.seg_generation_order
    assembled_v = seg_v[0]
    for i in range(1, n_seg):
        assembled_v = _crossfade_concat(assembled_v, seg_v[i], kv)
    assembled_a = seg_a[0]
    for i in range(1, n_seg):
        assembled_a = _crossfade_concat(assembled_a, seg_a[i], ka_list[i - 1])

    exp_v = VideoLatentShape.from_pixel_shape(
        VideoPixelShape(1, total_px, height // 2, width // 2, frame_rate),
        latent_channels=components.video_latent_channels,
        scale_factors=components.video_scale_factors,
    ).to_torch_shape()
    exp_a = AudioLatentShape.from_video_pixel_shape(
        VideoPixelShape(1, total_px, height, width, frame_rate)).to_torch_shape()
    assert tuple(assembled_v.shape) == tuple(exp_v), (tuple(assembled_v.shape), tuple(exp_v))
    assert tuple(assembled_a.shape) == tuple(exp_a), (tuple(assembled_a.shape), tuple(exp_a))

    # ── end source: the stage-1 half of the END-TO-END freeze proof ───────────
    # Taken off the ASSEMBLED timeline, not off ``seg_v[-1]``, so what is proven is
    # the band as it actually enters the upsample — the crossfade at the band
    # segment's own seam is upstream of this slice, so a mistake there would show.
    # CLONE IS MANDATORY: a plain slice is a VIEW, and a view keeps the whole
    # assembled stage-1 storage (the full half-res timeline) alive until after the
    # decode. The clone is n_end_v latent frames — kilobytes against gigabytes.
    s1_assembled_tail = None
    s1_assembled_tail_a = None
    if end_source is not None:
        s1_assembled_tail = assembled_v[:, :, assembled_v.shape[2] - n_end_v:].detach().clone()
    if end_a is not None:
        s1_assembled_tail_a = (
            assembled_a[:, :, assembled_a.shape[2] - n_end_a_eff:].detach().clone()
        )

    # ── ONE upsample over the whole timeline. ─────────────────────────────────
    if chunked_upsample:
        # Opt-in memory-bounded path: upsample in halo-padded temporal chunks and
        # keep the result on CPU (VRAM stays flat instead of scaling with total
        # length -> long 768p chains no longer OOM). The stage-2 slice below
        # transfers each tile back to the GPU on demand.
        upsampler = ledger.spatial_upsampler()
        upscaled_v = _chunked_upsample_cpu(
            assembled_v, video_encoder, upsampler, upsample_video, device, progress
        )
        del upsampler
        cleanup_memory()
    else:
        upscaled_v = upsample_video(assembled_v[:1], video_encoder, ledger.spatial_upsampler())
        torch.cuda.synchronize()
        cleanup_memory()

    # ── STAGE 2: always-tiled refine (video+audio jointly). ───────────────────
    # ONE context (the base/clip-0 prompt) for the ENTIRE stage-2 refine — this
    # is what the validated S2 spike did (single prompt everywhere). Per-segment
    # prompt variation lives in STAGE 1 (where the carry+freeze+crossfade absorbs
    # it smoothly — all segment seams stay continuous). Switching the AUDIO
    # context mid-tile-overlap in stage 2 injects a speech-context click at the
    # frozen tile seam (observed: Chain B J=456 audio ratio 13.67); a uniform
    # context removes that seam entirely.
    stage2_vctx, stage2_actx = seg_ctx[0]
    stage2_sigmas = torch.Tensor(STAGE_2_DISTILLED_SIGMA_VALUES).to(device)
    refined_v: list[torch.Tensor] = []
    refined_a: list[torch.Tensor] = []
    # Tiles the end-source band covers ENTIRELY (video fully frozen, audio still
    # refined). Observation only — reported in the metadata so a long band's cost
    # is visible; empty on every path without an end source.
    end_fully_frozen_tiles: list[int] = []
    # The same observation on the AUDIO grid. A fully-frozen audio tile is NOT
    # implied by a fully-frozen video one (the two grids advance differently), so
    # it is recorded separately rather than inferred.
    end_fully_frozen_tiles_a: list[int] = []
    for i in range(n_tiles):
        # F2: per-step shim phase for this tile's denoise (observation only).
        progress_shim.set_phase("stage2_denoise", outer_index=i, outer_total=n_tiles)
        vs, vlen = v_tiles[i]
        as_, alen = a_tiles[i]
        tile_px = (vlen - 1) * VIDEO_TIME_FACTOR + 1
        tile_shape = VideoPixelShape(1, tile_px, height, width, frame_rate)
        init_v = upscaled_v[:, :, vs:vs + vlen].contiguous().clone()
        if upscaled_v.device.type == "cpu":
            init_v = init_v.to(device)
        init_a = assembled_a[:, :, as_:as_ + alen].contiguous().clone()
        ftv = fta = 0
        if i == 0 and retake is not None:
            # ── retake stage 2: re-write AND re-freeze both ends ─────────────
            # Re-freezing here is MANDATORY, not belt-and-braces: stage 2
            # re-noises with noise_scale = sigmas[0] (~0.909), so an unfrozen
            # band would have the stage-1 freeze destroyed outright (spike
            # d1_both_side_freeze.py:663-669). Variant B: the bands come from
            # the FULL-res re-encode of the original, never from the upsampled
            # stage-1 approximation — the same choice the V2V head makes below.
            fkv, fka = n_head_v, n_head_a
            ftv, fta = n_tail_v, n_tail_a
            mv = 0.0
            init_v[:, :, :n_head_v] = rt_v_full[:, :, :n_head_v].to(DTYPE)
            init_v[:, :, layout.f_total - n_tail_v:] = (
                rt_v_full[:, :, layout.f_total - n_tail_v:].to(DTYPE)
            )
            if n_head_a > 0 or n_tail_a > 0:
                init_a[:, :, :n_head_a] = rt_a[:, :, :n_head_a].to(DTYPE)
                init_a[:, :, layout.a_total - n_tail_a:] = (
                    rt_a[:, :, layout.a_total - n_tail_a:].to(DTYPE)
                )
        elif i == 0 and source is not None:
            # video-to-video variant B: hard-freeze (mask 0.0) the source head at
            # tile-0's leading region using the FULL-res VAE re-encode, mirroring
            # how i>=1 tile joins freeze their leading kt_v. This is the one
            # genuinely new stage-2 orchestration piece (spike: eliminates the
            # variant-A color/tone drift; variant A failed G0).
            fkv, fka = n_ctx_v, freeze_ka
            mv = 0.0
            init_v[:, :, :n_ctx_v] = src_head_v_full.to(DTYPE)
            if freeze_ka > 0:
                init_a[:, :, :freeze_ka] = src_head_a.to(DTYPE)
        elif i == 0:
            fkv = fka = 0
            mv = 0.0
        else:
            fkv, fka = kt_v, kt_a
            mv = 0.0  # hard freeze on the leading overlap
            init_v[:, :, :kt_v] = refined_v[i - 1][:, :, refined_v[i - 1].shape[2] - kt_v:]
            init_a[:, :, :kt_a] = refined_a[i - 1][:, :, refined_a[i - 1].shape[2] - kt_a:]
        # ── end source (additive): re-write AND re-freeze the band in EVERY tile
        #    it reaches ──────────────────────────────────────────────────────────
        # An INDEPENDENT if for the same reason as its stage-1 twin: the band is
        # added on top of whichever head this tile already has (tile 0's V2V
        # context, tile i>=1's carry, or nothing at all).
        # Re-freezing here is MANDATORY, not belt-and-braces: stage 2 re-noises
        # with noise_scale = sigmas[0] (~0.909), so the stage-1 freeze alone would
        # be destroyed outright — the same argument the retake branch above makes.
        # Variant B: the band comes from the FULL-res re-encode of the material,
        # never from the upsampled stage-1 approximation (the same choice the V2V
        # head makes).
        #
        # THE BAND MAY SPAN SEVERAL TILES, and the plan for that is
        # ``layout.end_tile_bands`` — computed ONCE in chain_math and NEVER
        # re-derived here, so the geometry the app validated and the geometry the
        # engine writes cannot drift. ``t`` is how many band latents THIS tile
        # ends on and ``off`` the offset one past the last band latent it holds
        # (meaningless, and possibly negative, when t == 0 — hence the guard).
        # Because the band sits at the very END of the timeline, every tile that
        # reaches it reaches it on that TILE'S OWN TAIL, which is the trailing
        # hard-freeze shape retake and the tile seams were both validated under.
        #
        # WHY A SEAM *INSIDE* THE BAND IS SAFE. Tile i >= 1 hard-freezes its
        # leading kt_v from tile i-1's output, and the reassembly below crossfades
        # exactly that region. When the seam falls inside the band, both sides of
        # the fade hold the SAME source latents: tile i-1's tail is end_v_full
        # (it was frozen to it and mask 0.0 returns it bit-exact), and tile i's
        # lead is a copy of that tail. _crossfade_concat computes
        # ``a*(1-x) + b*x`` in fp32 and rounds back to bf16; with a == b that is
        # ``a*((1-x) + x)`` == a in fp32 exactly for the linspace weights (no
        # cancellation, both products share a's exponent), and rounding a bf16
        # value that survived fp32 arithmetic unchanged returns the same bf16
        # pattern. So a fade over identical values is the identity, bit for bit.
        # A tile the band SWALLOWS WHOLE (t == vlen) is therefore admissible too:
        # its video is fully frozen and only its audio is actually refined. That
        # is wasteful, not wrong, so it is merely made observable via
        # ``end_fully_frozen_tiles`` rather than special-cased.
        if end_source is not None:
            t, off = layout.end_tile_bands[i]
            if t > 0:
                # THE INDEX IS THIS TILE'S OWN LENGTH (vlen) — never
                # ``layout.f_total``; see the stage-1 note for why that idiom is
                # safe only in retake's degenerate single-tile shape.
                init_v[:, :, vlen - t:] = end_v_full[:, :, off - t: off].to(init_v.dtype)
                ftv = t
                if t == vlen:
                    end_fully_frozen_tiles.append(i)
                # Every stage-2 branch already hard-freezes, so head and tail agree
                # and no tail_mask_value split is needed here (unlike stage 1).
                assert mv == 0.0, mv
        # ── end source AUDIO band (additive): the same re-write and re-freeze ──
        # A SEPARATE ``if``, deliberately NOT nested inside the video block's
        # ``t > 0``: the two grids advance at different rates, so a tile can hold
        # no video band latents and some audio ones. ``end_bands_a`` is the audio
        # plan — the layout's own, or the narrower one a short encode forced —
        # and it is read here exactly as its video twin is read above.
        #
        # ORDER MATTERS ONLY IN ONE DIRECTION: tile i >= 1 has already written its
        # leading ``kt_a`` from tile i-1's output, and this write comes after. If
        # the two regions meet, they meet on identical values (tile i-1's tail was
        # frozen to this same band), so the crossfade over them is the identity —
        # the same argument the video block above spells out.
        # Audio latents are 4-D ``(b, c, T, f)``: the time axis is dim 2 and the
        # frequency axis must be carried through untouched.
        if end_a is not None:
            ta, off_a = end_bands_a[i]
            if ta > 0:
                init_a[:, :, alen - ta:] = end_a[:, :, off_a - ta: off_a, :].to(init_a.dtype)
                fta = ta
                if ta == alen:
                    end_fully_frozen_tiles_a.append(i)
                assert mv == 0.0, mv
        # audio-to-video (additive): refine this tile off the uploaded audio's
        # tile window (layout.a_tiles == (as_, alen)), HARD-frozen for the whole
        # tile. Stage-2 already runs mask 0.0 everywhere, so only the audio
        # init/freeze changes; the video refine (fkv/mv above) is untouched.
        if audio_source is not None:
            init_a = a2v_a[:, :, as_:as_ + alen].contiguous().clone().to(DTYPE)
            fka = alen
            mv = 0.0
        # clip-0 conditioning routed to the tile that owns each keyframe (full res).
        conds = _build_video_conditionings(
            _tile_images(clips[0].images, vs, vlen),
            height=height, width=width, video_encoder=video_encoder, device=device,
        )
        noiser2 = GaussianNoiser(generator=torch.Generator(device=device).manual_seed(base_seed + 100 + i))
        vstate2, astate2 = _denoise_av_with_carry(
            output_shape=tile_shape, components=components, transformer=transformer,
            video_context=stage2_vctx, audio_context=stage2_actx, video_conditionings=conds,
            noiser=noiser2, stepper=stepper, sigmas=stage2_sigmas,
            noise_scale=float(stage2_sigmas[0]),
            initial_video_latent=init_v, initial_audio_latent=init_a,
            freeze_kv=fkv, freeze_ka=fka, mask_value=mv, device=device,
            freeze_tail_v=ftv, freeze_tail_a=fta,
        )
        refined_v.append(vstate2.latent.detach().clone())
        refined_a.append(astate2.latent.detach().clone())
        if progress:
            progress("tile", i, n_tiles)

    # ── retake FREEZE PROOF (latent domain; computed while the sources live) ──
    # 8 numbers = {stage 1, stage 2} x {video, audio} x {head, tail}, each the
    # max-abs difference between what came OUT of the denoise and what was
    # frozen IN. This is the ONE check that looks at the latents themselves
    # rather than at pixels, and it is what the spike's main arm passed on all
    # counts (VERIFICATION_LOG §55.3). Stage 2 is always a hard freeze (mask
    # 0.0), so its four MUST be exactly 0; the stage-1 four are required to be 0
    # only when the stage-1 mask value was 0.0 (at 0.5 the band is a deliberate
    # 50% blend and a non-zero difference is correct).
    #
    # OBSERVATION ONLY — deliberately never raises. A wrong number here means
    # degraded output, not a corrupt job, and turning a metadata probe into a
    # new crash path would be a worse trade. ``pass`` carries the verdict.
    # The audio four are None (not 0.0) when the window had no audio: there is
    # nothing to compare, and inventing a 0.0 would fake a passing proof.
    retake_meta: dict | None = None
    if retake is not None:
        def _mad(a, b) -> float:
            return float((a.float() - b.float()).abs().max().item())

        s1_v, s1_a = seg_v[0], seg_a[0]
        f2_v, f2_a = refined_v[0], refined_a[0]
        f_tot, a_tot = layout.f_total, layout.a_total
        has_a = retake_had_audio and rt_a is not None and (n_head_a > 0 or n_tail_a > 0)
        checks: dict[str, float | None] = {
            "s1_video_head": _mad(s1_v[:, :, :n_head_v], rt_v_half[:, :, :n_head_v]),
            "s1_video_tail": _mad(s1_v[:, :, f_tot - n_tail_v:],
                                  rt_v_half[:, :, f_tot - n_tail_v:]),
            "s2_video_head": _mad(f2_v[:, :, :n_head_v], rt_v_full[:, :, :n_head_v]),
            "s2_video_tail": _mad(f2_v[:, :, f_tot - n_tail_v:],
                                  rt_v_full[:, :, f_tot - n_tail_v:]),
            "s1_audio_head": _mad(s1_a[:, :, :n_head_a], rt_a[:, :, :n_head_a]) if has_a else None,
            "s1_audio_tail": _mad(s1_a[:, :, a_tot - n_tail_a:],
                                  rt_a[:, :, a_tot - n_tail_a:]) if has_a else None,
            "s2_audio_head": _mad(f2_a[:, :, :n_head_a], rt_a[:, :, :n_head_a]) if has_a else None,
            "s2_audio_tail": _mad(f2_a[:, :, a_tot - n_tail_a:],
                                  rt_a[:, :, a_tot - n_tail_a:]) if has_a else None,
        }
        expected_zero = [k for k in checks if k.startswith("s2_")]
        if RETAKE_STAGE1_MASK_VALUE == 0.0:
            expected_zero += [k for k in checks if k.startswith("s1_")]
        checks["pass"] = all(
            checks[k] == 0.0 for k in expected_zero if checks[k] is not None
        )
        retake_meta = {"freeze_proof": checks}

    torch.cuda.synchronize()
    del transformer, video_encoder
    cleanup_memory()

    # ── Reassemble refined tiles (crossfade over hard-frozen overlaps). ───────
    final_v = refined_v[0]
    for i in range(1, n_tiles):
        final_v = _crossfade_concat(final_v, refined_v[i], kt_v)
    final_a = refined_a[0]
    for i in range(1, n_tiles):
        final_a = _crossfade_concat(final_a, refined_a[i], kt_a)
    assert final_v.shape[2] == layout.f_total, (final_v.shape[2], layout.f_total)
    assert final_a.shape[2] == layout.a_total, (final_a.shape[2], layout.a_total)

    # ── end source FREEZE PROOF (latent domain; the mirror of the retake block) ─
    # 4 numbers = {stage 1, stage 2} x {video tail, audio tail}, each the max-abs
    # difference between what came OUT and what was frozen IN. Stage 2 is ALWAYS
    # a HARD freeze (mask 0.0), so its numbers MUST be exactly 0.0. The VIDEO
    # stage-1 number is a hard freeze only when ``end_source.strength >= 1.0``
    # (mask 0.0); below that the band is a deliberate soft blend and a non-zero
    # difference is correct — the same shape the retake stage-1 pair has.
    #
    # THE AUDIO PAIR IS UNCONDITIONALLY EXPECTED TO BE 0.0, in stage 1 as well as
    # stage 2, because ``strength`` never reaches the audio tail (mask 0.0 in
    # both stages — see ``seg_audio_tail_mask_value``). ``s1_expected_zero``
    # below is therefore a VIDEO-only statement and must not be read as covering
    # the audio. Both audio numbers are None (not 0.0) when nothing was frozen —
    # no audio track, or the rollback constant — for the reason the retake proof
    # gives: an invented 0.0 would fake a passing proof. When a short encode made
    # the band narrower, they are measured over the ``n_end_a_frozen`` latents
    # that were actually written, which is where 0.0 is the correct expectation.
    #
    # THE MEASUREMENT IS END-TO-END, AND THAT IS THE WHOLE POINT OF ITS POSITION
    # HERE (after the tiles are reassembled, before the decode). The band now
    # spans an arbitrary number of stage-2 tiles, so a per-tile check would prove
    # each tile in isolation and prove nothing about the seams BETWEEN them. These
    # two numbers instead compare the material against the ONE assembled stage-1
    # timeline that entered the upsample and the ONE assembled stage-2 timeline
    # that is about to be decoded — every crossfade included.
    #
    # WHY EXACTLY 0.0 IS ATTAINABLE (and what it depends on) AT A MASK VALUE OF
    # 0.0: every write is a bf16 copy of the encoder's own output, the denoise
    # mask 0.0 returns those latents untouched, and a crossfade over two
    # identical values is the identity even through its fp32 intermediate (see
    # the stage-2 write site's note). The proof therefore rests on the bf16/fp32
    # rounding argument, not on a tolerance: if the pipeline ever moved to a
    # dtype where ``a*(1-x) + a*x != a``, these would go small-but-nonzero and
    # the assertion to relax would be "== 0.0".
    #
    # ``s1_expected_zero`` records the judgement basis in the metadata itself
    # (rather than only in ``pass``), because — unlike the retake stage-1 verdict,
    # which turns on a module constant — this one turns on a per-request value:
    # reading only ``pass`` from a log would risk misreading "the freeze was
    # proven" when the correct reading is "a soft blend behaved as configured".
    #
    # OBSERVATION ONLY — never raises, for the same reason the retake proof does
    # not: a wrong number means degraded output, not a corrupt job.
    end_source_meta: dict | None = None
    if end_source is not None:
        def _mad_end(a, b) -> float:
            return float((a.float() - b.float()).abs().max().item())

        # Each slice is taken off ITS OWN tensor's length (the same rule the two
        # write sites follow); here both happen to be the full timeline. The s1
        # number is always computed, even when a soft blend makes it nonzero by
        # design — it is the observed drift, not merely a pass/fail bit.
        s1_expected_zero = float(end_source.strength) >= 1.0
        end_checks: dict[str, float | bool | None] = {
            "s1_video_tail": _mad_end(s1_assembled_tail, end_v_half),
            "s2_video_tail": _mad_end(
                final_v[:, :, final_v.shape[2] - n_end_v:], end_v_full
            ),
            "s1_audio_tail": (
                None if s1_assembled_tail_a is None
                else _mad_end(s1_assembled_tail_a, end_a)
            ),
            "s2_audio_tail": (
                None if end_a is None
                else _mad_end(final_a[:, :, final_a.shape[2] - n_end_a_eff:], end_a)
            ),
        }
        end_checks["pass"] = (
            end_checks["s2_video_tail"] == 0.0
            and (not s1_expected_zero or end_checks["s1_video_tail"] == 0.0)
            and all(
                end_checks[k] == 0.0
                for k in ("s1_audio_tail", "s2_audio_tail")
                if end_checks[k] is not None
            )
        )
        end_source_meta = {
            "freeze_proof": end_checks,
            "context_frames": int(end_source.context_frames),
            "source_path": str(end_source.path),
            "strength": float(end_source.strength),
            "s1_expected_zero": s1_expected_zero,
            # WHAT STAGE 1 ACTUALLY DID, as opposed to what the geometry said it
            # should: the order the segments were generated in and the four
            # freeze widths each ran with. Every other end-source number in this
            # metadata (``mode``, ``generation_order``, ``end_tile_bands``, ...)
            # is a copy of a ``chain_math`` value, so those agreeing proves only
            # that the layout is self-consistent. These two come from the loop
            # itself, which is why the reverse schedule's machine gate reads them
            # and not the layout's own ``generation_order``.
            "stage1_order": list(stage1_order),
            "stage1_freezes": list(stage1_freezes),
            # Tiles whose video was entirely inside the band (cost observation).
            "end_fully_frozen_tiles": list(end_fully_frozen_tiles),
            # ── what the AUDIO side actually did ─────────────────────────────
            # Runtime observations, which is why they live here and not in the
            # layout's ``to_dict`` (that one publishes the GEOMETRY — how wide
            # the band is — and cannot know what the material turned out to
            # hold). ``end_source_provenance`` next door is the third kind: what
            # the APP prepared. The reading order for "did the audio freeze
            # happen" is this block, never the provenance.
            "audio_frozen": end_a is not None,
            # None when it did freeze; otherwise "no_audio" (no track, an
            # undecodable one, or an encode that yielded nothing) or "disabled"
            # (END_SOURCE_FREEZE_AUDIO). A SHORT encode is NOT a fallback: it
            # reads as ``audio_frozen: true`` with ``n_end_a_frozen`` below the
            # layout's ``n_end_a``.
            "audio_fallback_reason": None if end_a is not None else end_audio_status,
            "n_end_a_frozen": int(n_end_a_eff),
            # THE BAND THE ENGINE ACTUALLY WROTE, per tile. Equal to the layout's
            # ``end_tile_bands_a`` on a full freeze and narrower on a short one,
            # so a machine gate can check the freeze proof against the plan that
            # produced it rather than against a geometry that was not used.
            "end_tile_bands_a_frozen": [list(b) for b in end_bands_a],
            "end_fully_frozen_tiles_a": list(end_fully_frozen_tiles_a),
        }

    # ── ONE VAE decode -> ONE mp4. ────────────────────────────────────────────
    # F2: clear the shim phase — any further (unexpected) wheel loop would be
    # labelled with the generic "denoise", never a stale stage2 tag.
    progress_shim.end_op()
    if progress:
        progress("decode", 0, 1)
    gen = torch.Generator(device=device).manual_seed(base_seed)
    decoded_video = vae_decode_video(final_v, ledger.video_decoder(), tiling_cfg, gen)
    # A2V muxes the ORIGINAL waveform and a retake with regenerate_audio=False
    # re-muxes the window's own waveform, so in BOTH cases the vocoder is
    # skipped entirely — there is no point rendering audio that is discarded.
    _skip_vocoder = audio_source is not None or (
        retake is not None and not retake.regenerate_audio
    )
    decoded_audio = (
        None if _skip_vocoder
        else vae_decode_audio(final_a, ledger.audio_decoder(), ledger.vocoder())
    )
    v2v_meta: dict | None = None
    a2v_meta: dict | None = None
    if audio_source is not None:
        # ── audio-to-video: mux the ORIGINAL uploaded waveform (no vocoder),
        #    trimmed to the video duration. The video side streams through the
        #    same lazy chunk generator as the source-less path (no whole-timeline
        #    materialization — WDDM discipline: never assemble a GB-scale frame
        #    tensor); only the muxed audio differs (original waveform, not the
        #    vocoder render).
        from ltx_core.types import Audio

        n_mux = int(round(total_px / float(frame_rate) * a2v_sr))
        mux_wf = a2v_orig_wf[:, :n_mux].contiguous()          # (channels, samples)
        mux_audio = Audio(waveform=mux_wf.to(torch.float32), sampling_rate=a2v_sr)
        chunks = video_chunks_number(total_px, tiling_cfg)
        encode_video_output(
            video=decoded_video, audio=mux_audio, fps=int(frame_rate),
            output_path=str(output_path), video_chunks_number_value=chunks,
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
        torch.cuda.synchronize()
        del decoded_video
        torch.cuda.empty_cache()
    elif retake is not None:
        # ── retake: deliver the WHOLE WINDOW, untrimmed. ──────────────────────
        # The glue bands are NOT cut off. They are the overlap material the
        # timeline needs to lay this clip back over the original — the app puts
        # the seam at the window's outer edge, not at the regenerated region's
        # boundary, so the VAE round-trip's quality step never lands on the edit
        # point. That is also why NO audio-handle sidecar is emitted (unlike
        # V2V): the duplicate material a client would need for a true overlapped
        # crossfade is already inside this mp4. And no 30ms head fade either —
        # there is no butt-join here to guard against a click at.
        from ltx_core.types import Audio

        out_audio = decoded_audio
        if not retake.regenerate_audio and rt_orig_wf is not None:
            # Re-mux the window's OWN waveform verbatim, trimmed to the timeline.
            n_mux = int(round(total_px / float(frame_rate) * rt_sr))
            mux_wf = rt_orig_wf[:, :n_mux].contiguous()
            out_audio = Audio(waveform=mux_wf.to(torch.float32), sampling_rate=rt_sr)
        chunks = video_chunks_number(total_px, tiling_cfg)
        encode_video_output(
            video=decoded_video, audio=out_audio, fps=int(frame_rate),
            output_path=str(output_path), video_chunks_number_value=chunks,
        )
        retake_meta = dict(retake_meta or {})
        retake_meta.update({
            "regenerate_audio": bool(retake.regenerate_audio),
            "source_had_audio": bool(retake_had_audio),
            # Distinct from source_had_audio: the window can carry an audio
            # stream and still end up with no frozen band (regenerate_audio=False
            # with a short encode — see _encode_retake_window).
            "audio_frozen": bool(n_head_a > 0 or n_tail_a > 0),
            "muxed_original_waveform": bool(
                not retake.regenerate_audio and rt_orig_wf is not None
            ),
            "decoded_frames_px": int(total_px),
        })
        torch.cuda.synchronize()
        del decoded_video, decoded_audio
        torch.cuda.empty_cache()
    elif source is None:
        # ── source-less path: BYTE-IDENTICAL to today (gated). ────────────────
        chunks = video_chunks_number(total_px, tiling_cfg)
        encode_video_output(
            video=decoded_video, audio=decoded_audio, fps=int(frame_rate),
            output_path=str(output_path), video_chunks_number_value=chunks,
        )
        torch.cuda.synchronize()
        del decoded_video, decoded_audio
        torch.cuda.empty_cache()
    else:
        # ── video-to-video: trim the frozen context head so the delivered mp4
        #    is the NEW part only. vae_decode_video yields a LAZY generator of
        #    temporally-tiled (f,H,W,3) chunks -> materialize before slicing.
        from ltx_core.types import Audio

        if torch.is_tensor(decoded_video):
            full_video = decoded_video
        else:
            full_video = torch.cat(list(decoded_video), dim=0)
        f_total_px = int(full_video.shape[0])
        assert f_total_px == total_px, (f_total_px, total_px)
        trim_px = layout.trim_px
        new_video = full_video[trim_px:].contiguous()

        sr = decoded_audio.sampling_rate
        wf = decoded_audio.waveform
        if wf.dim() == 3:
            wf = wf.squeeze(0)                      # (channels, samples)
        n_trim_a = int(round(trim_px / float(frame_rate) * sr))

        # ── audio HANDLE sidecar (opt-in true-crossfade join material) ───────
        # Emit the FULL untrimmed timeline audio (context + new region, the very
        # waveform sliced just below) as a sidecar wav next to the delivered mp4,
        # with NO fade applied. A client join can then do a true overlapped
        # equal-power crossfade over the pre-junction context region (which BOTH
        # the source recording and this vocoder render depict) instead of a
        # no-overlap fade-pair that leaves an energy valley. The delivered mp4 is
        # untouched (still the trimmed new-part-only clip with its 30ms head
        # fade), so this is purely additive — byte-identical deliverable.
        audio_handle_filename: str | None = None
        handle_context_seconds = float(trim_px) / float(frame_rate)
        try:
            import os as _os
            import numpy as _np
            from scipy.io import wavfile as _wavfile

            _op = str(output_path)
            _root, _ = _os.path.splitext(_op)
            handle_path = _root + "_audio_handle.wav"
            handle_np = wf.detach().to(torch.float32).cpu().numpy()  # (channels, samples)
            handle_np = _np.ascontiguousarray(handle_np.T)           # (samples, channels)
            _wavfile.write(handle_path, int(sr), handle_np)
            audio_handle_filename = _os.path.basename(handle_path)
        except Exception as _exc:  # sidecar is best-effort; never fail the job on it
            logger.warning("chain: audio handle sidecar not written: %s", _exc)

        new_wf = wf[:, n_trim_a:].contiguous()
        # short linear fade-in (~30ms) on the continuation audio head: click
        # guard for the vocoder-vs-AAC noise-floor notch at the client-side join
        # (spike finding; video needs no fade).
        fade_n = min(int(round(0.030 * sr)), new_wf.shape[1])
        if fade_n > 1:
            ramp = torch.linspace(0.0, 1.0, fade_n, device=new_wf.device, dtype=new_wf.dtype)
            new_wf[:, :fade_n] = new_wf[:, :fade_n] * ramp
        new_audio = Audio(waveform=new_wf, sampling_rate=sr)

        new_chunks = video_chunks_number(int(new_video.shape[0]), tiling_cfg)
        encode_video_output(
            video=new_video, audio=new_audio, fps=int(frame_rate),
            output_path=str(output_path), video_chunks_number_value=new_chunks,
        )
        v2v_meta = {
            "context_frames": int(source.context_frames),
            "n_ctx_v": int(layout.n_ctx_v),
            "n_ctx_a": int(layout.n_ctx_a),
            "freeze_ka": int(freeze_ka),
            "trimmed_px": int(trim_px),
            "trimmed_audio_samples": int(n_trim_a),
            "audio_fade_in_samples": int(fade_n if fade_n > 1 else 0),
            "source_had_audio": bool(source_had_audio),
            # Distinct from source_had_audio: the source FILE can have an audio
            # stream (source_had_audio=True) while still ending up with a 0-frame
            # frozen audio head (freeze_ka=0) if the encoded source audio ran out
            # before n_ctx_a (see the audio-underrun warning above). This key is
            # unambiguous: "did the frozen head actually carry audio continuity".
            "audio_head_frozen": bool(freeze_ka > 0),
            "new_frames_px": int(new_video.shape[0]),
            "decoded_frames_px": int(f_total_px),
            "v2v_context_junction_px": layout.v2v_context_junction_px,
            # Opt-in true-crossfade join material (sidecar wav emitted above):
            # the full untrimmed timeline audio, junction at handle_context_seconds.
            "audio_handle_filename": audio_handle_filename,
            "handle_context_seconds": round(handle_context_seconds, 6),
        }
        torch.cuda.synchronize()
        del decoded_video, decoded_audio, full_video, new_video
        torch.cuda.empty_cache()

    wall = time.time() - t0
    peak = round(torch.cuda.max_memory_allocated(device) / 1e6, 1)
    meta = layout.to_dict()
    meta.update({
        "n_clips": n,
        "seed": base_seed,
        "seeds": seeds,
        "width": width,
        "height": height,
        "num_steps": num_steps,
        "overlap_frames": kv,
        "overlap_strength": float(overlap_strength),
        "wall_s": round(wall, 2),
        "vram_peak_mb": peak,
        "vram_within_16gb": peak < 16000,
        "output_mp4": str(output_path),
    })
    if v2v_meta is not None:
        # merge the runtime v2v fields into the geometry v2v sub-dict from
        # ChainLayout.to_dict() (single unified ``chain.v2v`` for the done event).
        meta.setdefault("v2v", {}).update(v2v_meta)
    if retake_meta is not None:
        # Merge the runtime retake fields into the geometry ``retake`` sub-dict
        # that ChainLayout.to_dict() already put there (one unified
        # ``chain.retake`` for the done event) — the same shape as v2v above.
        meta.setdefault("retake", {}).update(retake_meta)
    if end_source_meta is not None:
        # Merge the runtime end-source fields into the geometry ``end_source``
        # sub-dict ChainLayout.to_dict() already produced (one unified
        # ``chain.end_source`` for the done event) — same shape as v2v/retake.
        meta.setdefault("end_source", {}).update(end_source_meta)
    if a2v_meta is not None:
        # additive audio-to-video sub-dict (mirrors the v2v block; source-less +
        # v2v paths never set it, so those metas are unchanged).
        meta["a2v"] = a2v_meta
    return meta
