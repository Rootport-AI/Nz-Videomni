"""Outpainting: two-stage generation with a pixel-space blend between the stages.

Reproduces the official Lightricks ComfyUI workflow
``LTX-2.3_ICLoRA_Outpaint_Two_Stage_Distilled.json`` (a copy lives in
``uploads/_outpaint_verify/``) on top of THIS repo's low-VRAM pipeline. Design
context and the decision record are in
``../../Nz-LTX23-frontend-AviUtl2/Docs/OUTPAINTING_DESIGN_NOTES.md``.

Why this is a separate module rather than a flag on the single-generate path:
``LTXFastVideoPipeline._run_inference`` delegates two levels down into the
wheel's ``DistilledPipeline.__call__``, which runs stage 1 and stage 2 back to
back with a LATENT-space upsampler in between. Outpainting has to interrupt
exactly there — decode to pixels, blend the generated frame with the green
canvas, upscale in PIXEL space, re-encode — so it needs its own two-stage
driver. ``chain_pipeline.run_chain`` is the precedent for writing one: it also
reaches into ``pipe.pipeline`` for the ledger/components and arms every
acceleration knob through the same ``pipe._set_*_job`` pattern.

Flow (one worker invocation):

    text encode (+ NAG/VSF negative)
      -> freeze the SOURCE video's audio latent (official: "Frozen audio helps
         guiding outpainting to be consistent with the sounds in the video")
      -> STAGE 1 at half resolution, with the green canvas attached as the
         IC-LoRA reference conditioning
      -> decode stage 1 to pixels (half res)
      -> BLEND 1: Laplacian pyramid, dilation 5, against the green canvas
      -> 2x pixel upscale
      -> tiled VAE re-encode
      -> STAGE 2 at full resolution (no reference conditioning: the official
         graph strips the guide latents with ``LTXVCropGuides`` before stage 2)
      -> decode stage 2 to pixels (full res)
      -> BLEND 2: Laplacian pyramid, dilation 2, against the green canvas
      -> mux with the source's original waveform -> one mp4

Deliberate differences from the official workflow, all recorded rather than
hidden (see the module-level constants and the inline notes):

* sampler — official ``euler_ancestral_cfg_pp`` / ``euler_cfg_pp`` vs this
  repo's plain euler (the wheel offers no cfg++ stepper and CFG is 1.0 anyway);
* no distilled LoRA at 0.5 — official stacks one on a NON-distilled base, ours
  IS the distilled base;
* the 2x pixel upscale is bicubic, not lanczos (torch has no lanczos kernel);
* the IC-LoRA reference is TILE-encoded (``_reference_conditioning_for_stage``
  does this for every downscale-factor-1 adapter to survive a 16GB card),
  whereas the official ``LTXAddVideoICLoRAGuideAdvanced`` has
  ``use_tiled_encode=False``;
* the blend mask is generated analytically per resolution instead of being
  area-downscaled from the full-res one, and is carried as ONE frame instead of
  one per frame (see ``engine.outpaint``);
* stage-2 audio is re-frozen from the SOURCE audio, where the official graph
  freezes stage 1's own audio output. Stage 1 froze that audio hard
  (``mask_value=0.0``), so its output is the source audio — and this matches how
  ``chain_pipeline``'s A2V path already re-injects the uploaded audio per tile.
"""

from __future__ import annotations

import gc
import logging
import os
import time

import torch

from engine import progress_shim
from engine.outpaint.canvas import OutpaintGeometry, build_blend_mask
from engine.outpaint.pyramid_blend import blend_video_u8
from engine.pipeline.chain_pipeline import DTYPE, _denoise_av_with_carry
from engine.pipeline.common import (
    default_tiling_config,
    encode_video_output,
    video_chunks_number,
)
from engine.pipeline.fast_video_pipeline import _set_conv3d_memory_format
from engine.transformer.nag_service import encode_negative
from engine.transformer.vsf_service import VsfParams

logger = logging.getLogger(__name__)

# Stage-2 sigma schedule. The wheel's own STAGE_2_DISTILLED_SIGMA_VALUES starts
# at 0.909375; the official outpaint workflow's ManualSigmas node (5211) starts
# one step LOWER, at 0.725, and that difference is load-bearing here rather than
# cosmetic: stage 2's initial latent is the RE-ENCODE of the blended pixels, so
# the noise level it starts from decides how much of the blend survives. Starting
# where the wheel does would partly re-generate the very seam blend 1 just fixed.
# 0.421875 is the exact wheel value the workflow displays rounded as 0.4219.
OUTPAINT_STAGE2_SIGMAS = [0.725, 0.421875, 0.0]

# Frames per Laplacian-pyramid blend chunk. Blending pads each frame out to the
# next power of two per side (960x544 -> 1024x1024, 1920x1088 -> 2048x2048), so
# VRAM scales with this directly — roughly 2.2-2.6GB at 8 frames on a 2048^2
# canvas, on top of a transformer that is still resident. Overridable for the
# GPU gate without an API field, because it changes nothing about the result.
_BLEND_CHUNK_ENV = "LTX_OUTPAINT_BLEND_CHUNK"


def _blend_chunk_size() -> int | None:
    raw = os.environ.get(_BLEND_CHUNK_ENV, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        logger.warning("%s=%r is not an integer; using the default", _BLEND_CHUNK_ENV, raw)
        return None
    return value if value > 0 else None


def _load_canvas_pixels_u8(
    *, video_path: str, height: int, width: int, frame_cap: int, device: torch.device
) -> torch.Tensor:
    """Decode ``video_path`` to ``(F, H, W, 3)`` uint8 on the CPU.

    A uint8 twin of ``common.load_video_conditioning_cpu``, running the SAME
    per-frame ops on the same device (decode -> ``resize_and_center_crop`` on
    float32) but storing the result as uint8 instead of normalising to [-1, 1] in
    the model dtype.

    The storage format is the whole point. This tensor is one of the two blend
    operands, and the full-resolution case is 1920x1088x241 — 6.0GB as float32
    against 1.5GB as uint8. Since ``blend_video_u8`` converts per chunk anyway,
    keeping the whole-timeline copies in uint8 costs nothing and is the
    difference between ~4.5GB and ~18GB of resident host memory for the three
    full-res buffers (canvas, generated, output).

    The frame count is the caller's frame-shortfall detector: the app layer
    already rejects a source shorter than ``num_frames`` and
    ``video_io.pad_green_mp4`` clone-pads the tail as a backstop, so anything
    short here means one of those two failed and the caller must say so loudly
    rather than let stage 2's ``create_initial_state`` assert fire instead.
    """
    from ltx_pipelines.utils.media_io import decode_video_from_file, resize_and_center_crop

    frames: list[torch.Tensor] = []
    for f in decode_video_from_file(path=video_path, frame_cap=frame_cap, device=device):
        # (1, C, 1, H, W) float32 in [0, 255] -> (1, H, W, C) uint8
        frame = resize_and_center_crop(f.to(torch.float32), height, width)
        frame = frame.round().clamp(0, 255).to(torch.uint8)
        frames.append(frame[0].permute(1, 2, 3, 0)[0].cpu())
        del f, frame
    if not frames:
        raise ValueError(f"outpaint canvas decoded to 0 frames: {video_path}")
    return torch.stack(frames, dim=0)


def _decoded_to_u8(decoded) -> torch.Tensor:
    """Materialise ``video_vae.decode_video``'s output as ``(F, H, W, 3)`` uint8.

    The wheel yields a LAZY iterator of temporally-tiled ``(f, H, W, C)`` uint8
    chunks (untiled configs yield exactly one), so this is a concat, not a
    conversion — the dtype and layout are already what the blend wants.
    """
    if torch.is_tensor(decoded):
        return decoded.cpu()
    return torch.cat([chunk.cpu() for chunk in decoded], dim=0)


@torch.inference_mode()
def run_outpaint(
    pipe,
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
    ic_loras: list | None = None,
    ic_reference: tuple[str, float] | None = None,
    ic_attention_strength: float = 1.0,
    blend_dilation_stage1: int = 5,
    blend_dilation_stage2: int = 2,
    freeze_source_audio: bool = True,
    stage2_sigmas: list[float] | None = None,
    nag=None,
    progress=None,
) -> dict:
    """Extend ``canvas_path``'s footage into its green pad band -> one mp4.

    ``canvas_path`` is the lossless, video-only green canvas built by
    ``services.video_io.pad_green_mp4``; ``ic_reference`` points at that same
    file (the app substitutes it for the uploaded reference so the IC-LoRA
    plumbing needs no changes at all). ``source_path`` is the ORIGINAL upload and
    is read only for its audio, because the canvas is written without an audio
    stream on purpose.

    ``geometry`` describes the canvas and its four pads; ``num_frames`` /
    ``frame_rate`` are the generation timeline, which the app has already checked
    the source is long enough for.

    Returns the job metadata dict (geometry, blend settings, timings, VRAM peak).
    """
    from ltx_core.components.diffusion_steps import EulerDiffusionStep
    from ltx_core.components.noisers import GaussianNoiser
    from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
    from ltx_core.model.video_vae import decode_video as vae_decode_video
    from ltx_core.text_encoders.gemma import encode_text
    from ltx_core.types import AudioLatentShape, VideoPixelShape
    from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES
    from ltx_pipelines.utils.helpers import cleanup_memory

    geometry.validate()
    width = geometry.canvas_width
    height = geometry.canvas_height

    dp = pipe.pipeline  # DistilledPipeline
    device = dp.device
    ledger = dp.model_ledger
    components = dp.pipeline_components

    tiling_cfg = default_tiling_config(
        spatial_tile_size=pipe._vae_spatial_tile_size,
        temporal_tile_size=pipe._vae_temporal_tile_size,
    )
    chunk_size = _blend_chunk_size()
    full_shape = VideoPixelShape(1, num_frames, height, width, frame_rate)
    a_total = int(AudioLatentShape.from_video_pixel_shape(full_shape).to_torch_shape()[2])

    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()

    # ── Text encode ONCE, then free the encoder. ─────────────────────────────
    # NAG state must be armed BEFORE the encode: its negative prompt has to be
    # encoded against the SAME live text_encoder (a second Gemma load is what
    # this ordering avoids), and NagService.install() rejects "requested but not
    # encoded". Identical discipline to run_chain.
    pipe._set_nag_job(nag)
    if progress:
        progress("encode", 0, 1)
    text_encoder = ledger.text_encoder()
    video_ctx, audio_ctx = encode_text(text_encoder, prompts=[prompt])[0]
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

    # ── Freeze the source video's audio. ─────────────────────────────────────
    # The official note (node 5392): "Frozen audio helps guiding outpainting to
    # be consistent with the sounds in the video." Same shape as run_chain's A2V
    # path — encode once with the small audio encoder before the big models are
    # built, keep the ORIGINAL waveform on the CPU for the mux, skip the vocoder.
    #
    # Underrun is NOT fatal here. run_chain's A2V path errors out because there
    # the audio IS the subject; here it is guidance for a video that is already
    # fully specified, so a source whose audio ends early freezes what it has and
    # generates the rest (the same min() run_chain's V2V head does).
    frozen_audio = None
    source_waveform = None
    audio_sr = 0
    audio_frozen_frames = 0
    if freeze_source_audio and source_path:
        from ltx_core.model.audio_vae import encode_audio as vae_encode_audio
        from ltx_core.types import Audio
        from ltx_pipelines.utils.media_io import decode_audio_from_file

        src_audio = decode_audio_from_file(str(source_path), device)
        if src_audio is None:
            logger.warning(
                "outpaint: %s has no decodable audio stream; the model will "
                "generate audio for the widened frame instead of following it",
                source_path,
            )
        else:
            wf = src_audio.waveform
            if wf.dim() == 2:
                wf = wf.unsqueeze(0)
            # The audio VAE's conv_in expects stereo (weight [128, 2, 3, 3]), and
            # the mux writer is stereo-only, so a mono source is duplicated.
            if wf.shape[1] == 1:
                wf = wf.repeat(1, 2, 1)
            audio_sr = int(src_audio.sampling_rate)
            source_waveform = wf.squeeze(0).detach().to(torch.float32).cpu().contiguous()

            audio_encoder = ledger.audio_encoder()
            enc = vae_encode_audio(
                Audio(waveform=wf.to(DTYPE), sampling_rate=audio_sr), audio_encoder, None
            )
            audio_frozen_frames = min(a_total, int(enc.shape[2]))
            frozen_audio = enc[:, :, :audio_frozen_frames, :].detach().clone()
            if audio_frozen_frames < a_total:
                logger.warning(
                    "outpaint: source audio covers %d of %d audio latent frames; "
                    "the tail will be generated",
                    audio_frozen_frames, a_total,
                )
            del audio_encoder, enc, src_audio, wf
            cleanup_memory()

    def _audio_init() -> torch.Tensor | None:
        """A fresh a_total-length audio latent with the frozen head copied in."""
        if frozen_audio is None:
            return None
        shape = AudioLatentShape.from_video_pixel_shape(full_shape).to_torch_shape()
        init = torch.zeros(tuple(shape), dtype=DTYPE, device=device)
        init[:, :, :audio_frozen_frames] = frozen_audio.to(DTYPE)
        return init

    # ── IC-LoRA state MUST be set before the transformer is built. ───────────
    # The forward-time weight patch reads pipe._ic_loras through a provider wired
    # at transformer-build time, and _set_ic_job is also what resolves the
    # reference's downscale factor that _reference_conditioning_for_stage below
    # depends on. Passing the list explicitly (empty included) is the stale-detach
    # that stops a previous job's adapters bleeding in.
    pipe._set_ic_job(list(ic_loras or []), ic_reference, ic_attention_strength)

    video_encoder = ledger.video_encoder()
    transformer = ledger.transformer()
    gc.collect()
    torch.cuda.empty_cache()

    stepper = EulerDiffusionStep()
    stage1_sigmas = torch.Tensor(DISTILLED_SIGMA_VALUES).to(device)
    stage2_values = list(stage2_sigmas) if stage2_sigmas else list(OUTPAINT_STAGE2_SIGMAS)
    stage2_sigma_tensor = torch.Tensor(stage2_values).to(device)

    # ── STAGE 1: half resolution, green canvas as the IC-LoRA reference. ─────
    progress_shim.set_phase("stage1_denoise", outer_index=0, outer_total=1)
    half_shape = VideoPixelShape(1, num_frames, height // 2, width // 2, frame_rate)
    conds = []
    if ic_reference is not None:
        # The same builder the single-generate and chain paths use, called with
        # the half-res cond_kwargs so its own stage-1 discriminator
        # (cond_height == full_height // 2) accepts it. It returns [] for
        # anything else, which is exactly why stage 2 below can call nothing at
        # all and still be certain no guide latent leaks across — the official
        # graph achieves the same with an explicit LTXVCropGuides node.
        conds = pipe._reference_conditioning_for_stage(
            full_height=height,
            num_frames=num_frames,
            cond_kwargs={
                "height": height // 2,
                "width": width // 2,
                "video_encoder": video_encoder,
                "dtype": DTYPE,
                "device": device,
                "tiling_config": tiling_cfg,
            },
        )
    v1, _a1 = _denoise_av_with_carry(
        output_shape=half_shape,
        components=components,
        transformer=transformer,
        video_context=video_ctx,
        audio_context=audio_ctx,
        video_conditionings=conds,
        noiser=GaussianNoiser(generator=torch.Generator(device=device).manual_seed(int(seed))),
        stepper=stepper,
        sigmas=stage1_sigmas,
        noise_scale=1.0,
        initial_video_latent=None,
        initial_audio_latent=_audio_init(),
        freeze_kv=0,
        freeze_ka=audio_frozen_frames,
        mask_value=0.0,
        device=device,
    )
    stage1_latent = v1.latent.detach().clone()
    del v1, _a1, conds
    if progress:
        progress("stage1", 0, 1)

    # ── Decode stage 1 -> blend -> 2x upscale -> re-encode. ──────────────────
    # This is the whole reason outpainting cannot ride the wheel's own two-stage
    # call: the wheel upsamples in LATENT space between the stages, and the
    # blend has to happen on pixels.
    progress_shim.end_op()
    if progress:
        progress("decode", 0, 1)
    gen = torch.Generator(device=device).manual_seed(int(seed))
    stage1_pixels = _decoded_to_u8(
        vae_decode_video(stage1_latent, ledger.video_decoder(), tiling_cfg, gen)
    )
    del stage1_latent
    cleanup_memory()

    canvas_half = _load_canvas_pixels_u8(
        video_path=str(canvas_path),
        height=height // 2,
        width=width // 2,
        frame_cap=num_frames,
        device=device,
    )
    if canvas_half.shape[0] < num_frames or stage1_pixels.shape[0] < num_frames:
        raise ValueError(
            f"outpaint frame shortfall: canvas has {canvas_half.shape[0]} frames and "
            f"stage 1 decoded {stage1_pixels.shape[0]}, but the job needs {num_frames}. "
            "The canvas must be built with pad_green_mp4's exact-frame-count guarantee."
        )
    canvas_half = canvas_half[:num_frames]
    stage1_pixels = stage1_pixels[:num_frames]

    mask_half = build_blend_mask(geometry, height=height // 2, width=width // 2)
    blended_half = blend_video_u8(
        stage1_pixels,
        canvas_half,
        mask_half,
        mask_low_res_dilation=blend_dilation_stage1,
        chunk_size=chunk_size,
        device=device,
    )
    blend1_peak_mb = round(torch.cuda.max_memory_allocated(device) / 1e6, 1)
    del stage1_pixels, canvas_half, mask_half
    cleanup_memory()

    # 2x pixel upscale (official: lanczos; torch offers no lanczos kernel, so
    # bicubic — the closest windowed-sinc-like resampler available. Recorded as an
    # intentional difference and A/B'd in the GPU gate).
    upscaled = torch.empty(
        (num_frames, height, width, 3), dtype=torch.uint8, device="cpu"
    )
    step = chunk_size or 8
    for start in range(0, num_frames, step):
        end = min(start + step, num_frames)
        block = blended_half[start:end].to(device).permute(0, 3, 1, 2).to(torch.float32)
        block = torch.nn.functional.interpolate(
            block, size=(height, width), mode="bicubic", align_corners=False
        )
        block = block.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
        upscaled[start:end] = block.cpu()
        del block
    del blended_half
    cleanup_memory()

    # Tiled VAE re-encode. channels_last_3d on the Conv3d weights is the known
    # fix for the im2col intermediate that OOMs a 16GB card on a full-resolution
    # encode (fast_video_pipeline._reference_conditioning_for_stage carries the
    # measurements); restoring contiguous afterwards is a CORRECTNESS
    # requirement, not hygiene, because this same encoder is reused within the
    # job and its latents shift under the other layout.
    _accel = torch.device(device).type == "cuda" and callable(
        getattr(video_encoder, "modules", None)
    )
    if _accel:
        _set_conv3d_memory_format(video_encoder, torch.channels_last_3d)
    try:
        cleanup_memory()
        encode_input = (
            upscaled.permute(3, 0, 1, 2).unsqueeze(0).to(torch.float32) / 127.5 - 1.0
        ).to(DTYPE)
        stage2_init = video_encoder.tiled_encode(encode_input, tiling_cfg)
        del encode_input
    finally:
        if _accel:
            try:
                _set_conv3d_memory_format(video_encoder, torch.contiguous_format)
                cleanup_memory()
            except Exception:  # must never mask an in-flight encode failure
                logger.exception(
                    "outpaint: failed to restore the video encoder's contiguous layout"
                )
    del upscaled
    cleanup_memory()

    # ── STAGE 2: full resolution, no reference conditioning. ─────────────────
    progress_shim.set_phase("stage2_denoise", outer_index=0, outer_total=1)
    v2, a2 = _denoise_av_with_carry(
        output_shape=full_shape,
        components=components,
        transformer=transformer,
        video_context=video_ctx,
        audio_context=audio_ctx,
        video_conditionings=[],
        noiser=GaussianNoiser(
            generator=torch.Generator(device=device).manual_seed(int(seed) + 100)
        ),
        stepper=stepper,
        sigmas=stage2_sigma_tensor,
        noise_scale=float(stage2_sigma_tensor[0]),
        initial_video_latent=stage2_init.to(DTYPE),
        initial_audio_latent=_audio_init(),
        freeze_kv=0,
        freeze_ka=audio_frozen_frames,
        mask_value=0.0,
        device=device,
    )
    stage2_latent = v2.latent.detach().clone()
    stage2_audio_latent = a2.latent.detach().clone()
    del v2, a2, stage2_init
    if progress:
        progress("tile", 0, 1)

    torch.cuda.synchronize()
    del transformer, video_encoder
    cleanup_memory()

    # ── Decode stage 2 -> final blend -> mux. ────────────────────────────────
    progress_shim.end_op()
    if progress:
        progress("decode", 0, 1)
    gen2 = torch.Generator(device=device).manual_seed(int(seed))
    stage2_pixels = _decoded_to_u8(
        vae_decode_video(stage2_latent, ledger.video_decoder(), tiling_cfg, gen2)
    )[:num_frames]
    del stage2_latent
    cleanup_memory()

    decoded_audio = None
    if frozen_audio is None:
        decoded_audio = vae_decode_audio(
            stage2_audio_latent, ledger.audio_decoder(), ledger.vocoder()
        )
    del stage2_audio_latent
    cleanup_memory()

    canvas_full = _load_canvas_pixels_u8(
        video_path=str(canvas_path),
        height=height,
        width=width,
        frame_cap=num_frames,
        device=device,
    )[:num_frames]
    mask_full = build_blend_mask(geometry, height=height, width=width)
    final_pixels = blend_video_u8(
        stage2_pixels,
        canvas_full,
        mask_full,
        mask_low_res_dilation=blend_dilation_stage2,
        chunk_size=chunk_size,
        device=device,
    )
    blend2_peak_mb = round(torch.cuda.max_memory_allocated(device) / 1e6, 1)
    del stage2_pixels, canvas_full, mask_full
    cleanup_memory()

    if frozen_audio is not None:
        # Mux the ORIGINAL waveform, trimmed to the video duration — the vocoder
        # is skipped entirely, exactly as run_chain's A2V path does, so the
        # delivered audio track is bit-for-bit the source's.
        from ltx_core.types import Audio

        n_mux = int(round(num_frames / float(frame_rate) * audio_sr))
        mux_wf = source_waveform[:, :n_mux].contiguous()
        mux_audio = Audio(waveform=mux_wf.to(torch.float32), sampling_rate=audio_sr)
        muxed_samples = int(mux_wf.shape[-1])
    else:
        mux_audio = decoded_audio
        muxed_samples = 0

    encode_video_output(
        video=final_pixels,
        audio=mux_audio,
        fps=int(frame_rate),
        output_path=str(output_path),
        video_chunks_number_value=video_chunks_number(num_frames, tiling_cfg),
    )
    torch.cuda.synchronize()
    del final_pixels, mux_audio, decoded_audio
    torch.cuda.empty_cache()

    wall = time.time() - t0
    peak = round(torch.cuda.max_memory_allocated(device) / 1e6, 1)
    meta = {
        "outpaint": {
            **geometry.as_dict(),
            "blend_dilation_stage1": int(blend_dilation_stage1),
            "blend_dilation_stage2": int(blend_dilation_stage2),
            "blend_chunk_size": chunk_size,
            "blend1_peak_vram_mb": blend1_peak_mb,
            "blend2_peak_vram_mb": blend2_peak_mb,
            "stage2_sigmas": stage2_values,
            "freeze_source_audio": bool(freeze_source_audio),
            "audio_frozen_latent_frames": int(audio_frozen_frames),
            "audio_latent_frames_required": int(a_total),
            "muxed_original_waveform": frozen_audio is not None,
            "muxed_audio_samples": muxed_samples,
            "audio_sampling_rate": int(audio_sr),
            "canvas_path": str(canvas_path),
        },
        "seed": int(seed),
        "width": width,
        "height": height,
        "num_frames": int(num_frames),
        "num_steps": int(num_steps),
        "wall_s": round(wall, 2),
        "vram_peak_mb": peak,
        "vram_within_16gb": peak < 16000,
        "output_mp4": str(output_path),
    }
    return meta
