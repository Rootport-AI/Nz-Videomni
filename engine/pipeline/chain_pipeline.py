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
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from chain_math import VIDEO_TIME_FACTOR, ChainLayout, compute_chain_layout
from engine.api_types import ImageConditioningInput
from engine.pipeline.common import (
    default_tiling_config,
    encode_video_output,
    video_chunks_number,
)

DTYPE = torch.bfloat16

# progress(stage, index, total) — stage in {"stage1","tile","decode"}.
ProgressFn = Callable[[str, int, int], None]


@dataclass
class ChainClipSpec:
    """One clip in the chain (prompt already resolved to its effective value)."""

    prompt: str
    num_frames: int
    images: list[ImageConditioningInput] = field(default_factory=list)


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
):
    """Reimpl of helpers.denoise_audio_video with an overlap-freeze mask.

    Builds the video + audio initial states, overrides the denoise_mask on the
    first freeze_k{v,a} latent frames to ``mask_value`` (= 1 - overlap_strength),
    applies video conditionings, noises (mask-respecting), runs the euler loop
    (re-pins mask<1 tokens each step), and unpatchifies. Verbatim mechanics from
    the s1/s2 spikes (which faithfully reimplement the wheel helpers) with an
    added optional ``video_conditionings`` (empty for T2V segments/tiles).
    """
    from ltx_core.tools import AudioLatentTools, VideoLatentTools
    from ltx_core.types import AudioLatentShape, VideoLatentShape
    from ltx_pipelines.utils.helpers import simple_denoising_func, state_with_conditionings
    from ltx_pipelines.utils.samplers import euler_denoising_loop

    # ── VIDEO ──
    vshape = VideoLatentShape.from_pixel_shape(
        shape=output_shape,
        latent_channels=components.video_latent_channels,
        scale_factors=components.video_scale_factors,
    )
    vtools = VideoLatentTools(components.video_patchifier, vshape, output_shape.fps)
    vstate = vtools.create_initial_state(device, DTYPE, initial_video_latent)
    if freeze_kv > 0:
        hw = vshape.height * vshape.width
        m = vstate.denoise_mask.clone()
        assert m.shape[1] >= vshape.frames * hw
        m[:, : freeze_kv * hw, ...] = float(mask_value)
        vstate = dataclasses.replace(vstate, denoise_mask=m)
    vstate = state_with_conditionings(vstate, video_conditionings or [], vtools)
    vstate = noiser(vstate, noise_scale)

    # ── AUDIO ── (patchified audio token i == audio latent frame i; mask (b,T,1))
    ashape = AudioLatentShape.from_video_pixel_shape(output_shape)
    atools = AudioLatentTools(components.audio_patchifier, ashape)
    astate = atools.create_initial_state(device, DTYPE, initial_audio_latent)
    if freeze_ka > 0:
        m = astate.denoise_mask.clone()
        assert m.shape[1] >= ashape.frames
        m[:, :freeze_ka, ...] = float(mask_value)
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


def _seg_global_spans(seg_latent: list[int], kv: int) -> list[tuple[int, int]]:
    """Per-segment [start, end) in global stage-1 video latent frames."""
    spans: list[tuple[int, int]] = []
    for i, L in enumerate(seg_latent):
        s = 0 if i == 0 else spans[i - 1][0] + seg_latent[i - 1] - kv
        spans.append((s, s + L))
    return spans


def _dominant_segment(spans: list[tuple[int, int]], vs: int, ve: int) -> int:
    """Index of the segment whose global span overlaps [vs, ve) the most."""
    best_i, best_ov = 0, -1
    for i, (s, e) in enumerate(spans):
        ov = max(0, min(e, ve) - max(s, vs))
        if ov > best_ov:
            best_ov, best_i = ov, i
    return best_i


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
) -> dict:
    """Run a masked AV-latent chain to ONE mp4. Returns metadata incl. junctions.

    All heavy weights (transformer/video_encoder) are built ONCE and reused for
    every stage-1 segment and stage-2 tile; the text encoder is loaded once,
    used to encode all distinct clip prompts, then freed (mirrors the spikes +
    DistilledPipeline ordering).
    """
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
    layout: ChainLayout = compute_chain_layout(clip_frames, frame_rate, kv=kv)
    ka_list = layout.ka_list
    v_tiles = layout.v_tiles
    a_tiles = layout.a_tiles
    kt_v, kt_a = layout.kt_v, layout.kt_a
    n_tiles = layout.n_tiles
    total_px = layout.total_px

    base_seed = int(seed)
    seeds = [base_seed + i for i in range(n)]

    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()

    # ── Text encode ONCE for all DISTINCT prompts, then free the encoder. ─────
    text_encoder = ledger.text_encoder()
    distinct = list(dict.fromkeys(c.prompt for c in clips))  # preserves order, dedups
    ctx_by_prompt: dict[str, tuple] = {}
    for p in distinct:
        vctx, actx = encode_text(text_encoder, prompts=[p])[0]
        ctx_by_prompt[p] = (vctx, actx)
    torch.cuda.synchronize()
    del text_encoder
    cleanup_memory()

    seg_ctx = [ctx_by_prompt[c.prompt] for c in clips]

    # ── Build video_encoder + transformer ONCE (reuse for stage1 + stage2). ───
    video_encoder = ledger.video_encoder()
    transformer = ledger.transformer()
    # Drop the Windows-stranded reserved pool from the block-swap load-then-evict
    # (Phase 5B fix; the chain path bypasses the worker's denoise_audio_video
    # wrapper, so release explicitly here before the first heavy denoise).
    gc.collect()
    torch.cuda.empty_cache()

    stepper = EulerDiffusionStep()
    stage1_sigmas = torch.Tensor(DISTILLED_SIGMA_VALUES).to(device)

    # ── STAGE 1: per-segment (half-res) with carry+freeze. ────────────────────
    seg_v: list[torch.Tensor] = []
    seg_a: list[torch.Tensor] = []
    for i in range(n):
        seg_shape = VideoPixelShape(1, clip_frames[i], height // 2, width // 2, frame_rate)
        noiser = GaussianNoiser(generator=torch.Generator(device=device).manual_seed(seeds[i]))
        if i == 0:
            init_v = init_a = None
            fkv = fka = 0
            # clip-0 conditioning at HALF resolution (stage-1).
            conds = _build_video_conditionings(
                clips[0].images, height=height // 2, width=width // 2,
                video_encoder=video_encoder, device=device,
            )
        else:
            ka_i = ka_list[i - 1]
            prev_v, prev_a = seg_v[i - 1], seg_a[i - 1]
            init_v = torch.zeros_like(prev_v)
            init_v[:, :, :kv] = prev_v[:, :, prev_v.shape[2] - kv:]
            init_a = torch.zeros_like(prev_a)
            init_a[:, :, :ka_i] = prev_a[:, :, prev_a.shape[2] - ka_i:]
            fkv, fka = kv, ka_i
            conds = []
        vctx, actx = seg_ctx[i]
        vstate, astate = _denoise_av_with_carry(
            output_shape=seg_shape, components=components, transformer=transformer,
            video_context=vctx, audio_context=actx, video_conditionings=conds,
            noiser=noiser, stepper=stepper, sigmas=stage1_sigmas, noise_scale=1.0,
            initial_video_latent=init_v, initial_audio_latent=init_a,
            freeze_kv=fkv, freeze_ka=fka, mask_value=stage1_mask_value, device=device,
        )
        seg_v.append(vstate.latent.detach().clone())
        seg_a.append(astate.latent.detach().clone())
        if progress:
            progress("stage1", i, n)

    # ── Assemble ONE continuous stage-1 AV latent. ────────────────────────────
    assembled_v = seg_v[0]
    for i in range(1, n):
        assembled_v = _crossfade_concat(assembled_v, seg_v[i], kv)
    assembled_a = seg_a[0]
    for i in range(1, n):
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

    # ── ONE upsample over the whole timeline. ─────────────────────────────────
    upscaled_v = upsample_video(assembled_v[:1], video_encoder, ledger.spatial_upsampler())
    torch.cuda.synchronize()
    cleanup_memory()

    # ── STAGE 2: always-tiled refine (video+audio jointly). ───────────────────
    stage2_sigmas = torch.Tensor(STAGE_2_DISTILLED_SIGMA_VALUES).to(device)
    spans = _seg_global_spans(layout.seg_latent, kv)
    refined_v: list[torch.Tensor] = []
    refined_a: list[torch.Tensor] = []
    for i in range(n_tiles):
        vs, vlen = v_tiles[i]
        as_, alen = a_tiles[i]
        tile_px = (vlen - 1) * VIDEO_TIME_FACTOR + 1
        tile_shape = VideoPixelShape(1, tile_px, height, width, frame_rate)
        init_v = upscaled_v[:, :, vs:vs + vlen].contiguous().clone()
        init_a = assembled_a[:, :, as_:as_ + alen].contiguous().clone()
        if i == 0:
            fkv = fka = 0
            mv = 0.0
        else:
            fkv, fka = kt_v, kt_a
            mv = 0.0  # hard freeze on the leading overlap
            init_v[:, :, :kt_v] = refined_v[i - 1][:, :, refined_v[i - 1].shape[2] - kt_v:]
            init_a[:, :, :kt_a] = refined_a[i - 1][:, :, refined_a[i - 1].shape[2] - kt_a:]
        # clip-0 conditioning routed to the tile that owns each keyframe (full res).
        conds = _build_video_conditionings(
            _tile_images(clips[0].images, vs, vlen),
            height=height, width=width, video_encoder=video_encoder, device=device,
        )
        dom = _dominant_segment(spans, vs, vs + vlen)
        vctx, actx = seg_ctx[dom]
        noiser2 = GaussianNoiser(generator=torch.Generator(device=device).manual_seed(base_seed + 100 + i))
        vstate2, astate2 = _denoise_av_with_carry(
            output_shape=tile_shape, components=components, transformer=transformer,
            video_context=vctx, audio_context=actx, video_conditionings=conds,
            noiser=noiser2, stepper=stepper, sigmas=stage2_sigmas,
            noise_scale=float(stage2_sigmas[0]),
            initial_video_latent=init_v, initial_audio_latent=init_a,
            freeze_kv=fkv, freeze_ka=fka, mask_value=mv, device=device,
        )
        refined_v.append(vstate2.latent.detach().clone())
        refined_a.append(astate2.latent.detach().clone())
        if progress:
            progress("tile", i, n_tiles)

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

    # ── ONE VAE decode -> ONE mp4. ────────────────────────────────────────────
    if progress:
        progress("decode", 0, 1)
    gen = torch.Generator(device=device).manual_seed(base_seed)
    decoded_video = vae_decode_video(final_v, ledger.video_decoder(), tiling_cfg, gen)
    decoded_audio = vae_decode_audio(final_a, ledger.audio_decoder(), ledger.vocoder())
    chunks = video_chunks_number(total_px, tiling_cfg)
    encode_video_output(
        video=decoded_video, audio=decoded_audio, fps=int(frame_rate),
        output_path=str(output_path), video_chunks_number_value=chunks,
    )
    torch.cuda.synchronize()
    del decoded_video, decoded_audio
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
    return meta
