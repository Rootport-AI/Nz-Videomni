"""Inpainting: repaint the masked region of a video, two stages, pixel blends.

The sibling of :mod:`engine.pipeline.outpaint_pipeline` — same two-stage
skeleton, same In-Outpainting IC-LoRA, same Laplacian pyramid blend — with the
geometry inverted: outpainting invents a band AROUND the footage, inpainting
repaints a region INSIDE it. Design canon: ``Docs/INPAINTING_DESIGN.md`` §7.

**Why a separate module rather than a flag on ``run_outpaint``.** The two
differ in six places (the blend mask, the de-green, the restore, the final crop,
the metadata block and the mask decode), which as hook parameters on one
function would be six new arguments threaded through a 400-line body. The
decisive reason though is regression risk, not tidiness: outpainting ships on
BOTH engines (LTX 2.3 here, LTX 2.5 through ``engine25/outpaint25.py``, which
shares this repo's ``engine.outpaint`` blend), and its output has to stay
bit-identical. Keeping ``run_outpaint`` to three pure extractions
(``_freeze_source_audio`` / ``_audio_init`` / ``_mux_audio``, all imported
below) is what makes that provable rather than hoped for.

Flow (one worker invocation):

    text encode (+ NAG/VSF negative)
      -> freeze the SOURCE window's audio latent
      -> STAGE 1 at half resolution, with the GREEN CANVAS attached as the
         IC-LoRA reference conditioning (the masked region is painted #66FF00,
         which is what the In-Outpainting adapter reads as "invent this")
      -> decode stage 1 to pixels (half res)
      -> DE-GREEN, twice: the pad bands AND the masked region are replaced by
         this stage's own generated pixels
      -> BLEND 1: Laplacian pyramid against that canvas, PER-FRAME mask
      -> 2x pixel upscale -> tiled VAE re-encode
      -> STAGE 2 at full resolution (no reference conditioning)
      -> decode, DE-GREEN again, BLEND 2
      -> RESTORE: every pixel the stage-2 dilated mask never reached goes back
         to the source, byte for byte
      -> CROP off the right/bottom pad bands (uint8, lossless)
      -> mux the source window's own waveform -> one mp4

Deliberate differences from outpainting, beyond the geometry:

* **the mask is decoded twice**, once per stage, instead of being kept at both
  resolutions. A full-resolution 1920x1088x481 mask is 1.0GB as uint8 and the
  half-resolution one another 0.25GB; decoding again costs a few seconds of
  ffmpeg and keeps only one of them resident at a time;
* **restore runs once, after the FINAL blend only.** A restore between the
  stages would change stage 2's input (it re-encodes the blended pixels as its
  initial latent), so the promise "outside the mask is the original" is made
  where it can be kept and nowhere else;
* **the de-green is two calls, not one.** Outpainting's pad bands are the whole
  generated region, so one call covers it. Here the pad bands and the mask are
  two disjoint regions with different shapes — the bands come from the geometry,
  the mask from the video.
"""

from __future__ import annotations

import gc
import logging
import time

import torch

from engine import progress_shim
from engine.inpaint.canvas import (
    InpaintGeometry,
    fill_mask_with_generated_,
    fill_pad_bands_with_generated_,
    half_res_mask,
    place_mask_on_canvas,
)
from engine.inpaint.canvas import restore_and_measure_ as _restore_and_measure_
from engine.outpaint.pyramid_blend import blend_video_u8
from engine.pipeline.chain_pipeline import DTYPE, _denoise_av_with_carry
from engine.pipeline.common import (
    decode_mask_video,
    default_tiling_config,
    encode_video_output,
    video_chunks_number,
)
from engine.pipeline.fast_video_pipeline import _set_conv3d_memory_format
from engine.pipeline.outpaint_pipeline import (
    OUTPAINT_STAGE2_SIGMAS,
    _audio_init,
    _blend_chunk_size,
    _decoded_to_u8,
    _freeze_source_audio,
    _load_canvas_pixels_u8,
    _mux_audio,
)
from engine.transformer.nag_service import encode_negative
from engine.transformer.vsf_service import VsfParams

logger = logging.getLogger(__name__)

__all__ = ["run_inpaint"]


@torch.inference_mode()
def run_inpaint(
    pipe,
    *,
    prompt: str,
    canvas_path: str,
    source_path: str | None,
    mask_path: str,
    geometry: InpaintGeometry,
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
    """Repaint ``mask_path``'s white region inside ``canvas_path`` -> one mp4.

    ``canvas_path`` is the lossless, video-only canvas built by
    ``services.video_io.fill_mask_green_mp4``: the source window with the mask's
    white region painted #66FF00 and the right/bottom bands padded out to the
    128-multiple canvas. ``ic_reference`` points at that same file (the app
    substitutes it for the uploaded reference, so the IC-LoRA plumbing needs no
    changes at all). ``source_path`` is the CUT WINDOW — the same footage the
    canvas was built from, still carrying its audio, which the canvas
    deliberately does not.

    ``mask_path`` is the uploaded mask video at the SOURCE resolution; it is
    decoded here rather than baked into the canvas because the blend needs the
    region as numbers, not as green pixels.

    ``geometry`` describes the canvas and the source rectangle inside it;
    ``num_frames`` / ``frame_rate`` are the generation timeline, which the app
    has already checked the source and the mask both cover.

    Returns the job metadata dict (geometry, blend settings, timings, VRAM peak,
    audio record and ``mask_proof``).
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
    half_w, half_h, half_sw, half_sh = geometry.half_dims()

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
    # NAG state must be armed BEFORE the encode (NagService.install() rejects
    # "requested but not encoded" and a second Gemma load is what this ordering
    # avoids). Identical discipline to run_outpaint and run_chain.
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

    # ── Freeze the source window's audio (shared with run_outpaint). ─────────
    audio = _freeze_source_audio(
        ledger=ledger,
        device=device,
        source_path=source_path,
        enabled=freeze_source_audio,
        a_total=a_total,
        label="inpaint",
    )

    # ── IC-LoRA state MUST be set before the transformer is built. ───────────
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
    half_shape = VideoPixelShape(1, num_frames, half_h, half_w, frame_rate)
    conds = []
    if ic_reference is not None:
        conds = pipe._reference_conditioning_for_stage(
            full_height=height,
            num_frames=num_frames,
            cond_kwargs={
                "height": half_h,
                "width": half_w,
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
        initial_audio_latent=_audio_init(audio, full_shape=full_shape, device=device),
        freeze_kv=0,
        freeze_ka=audio.frozen_frames,
        mask_value=0.0,
        device=device,
    )
    stage1_latent = v1.latent.detach().clone()
    del v1, _a1, conds
    if progress:
        progress("stage1", 0, 1)

    # ── Decode stage 1 -> de-green -> blend -> 2x upscale -> re-encode. ──────
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
        height=half_h,
        width=half_w,
        frame_cap=num_frames,
        device=device,
    )
    if canvas_half.shape[0] < num_frames or stage1_pixels.shape[0] < num_frames:
        raise ValueError(
            f"inpaint frame shortfall: canvas has {canvas_half.shape[0]} frames and "
            f"stage 1 decoded {stage1_pixels.shape[0]}, but the job needs {num_frames}. "
            "The canvas must be built with fill_mask_green_mp4's measured-frame-count "
            "guarantee."
        )
    canvas_half = canvas_half[:num_frames]
    stage1_pixels = stage1_pixels[:num_frames]

    # The mask at half resolution: decode at the source size, paste into the
    # canvas (pads = 0), THEN area-downscale. Downscaling before the paste would
    # blur the mask across the pad boundary.
    mask_half = half_res_mask(
        place_mask_on_canvas(
            decode_mask_video(
                str(mask_path),
                num_frames=num_frames,
                height=geometry.source_height,
                width=geometry.source_width,
                device=device,
            ),
            geometry,
        ),
        half_h,
        half_w,
    )
    blend_chunk = chunk_size or 8
    fill_pad_bands_with_generated_(
        canvas_half,
        generated=stage1_pixels,
        source_height=half_sh,
        source_width=half_sw,
    )
    fill_mask_with_generated_(
        canvas_half, generated=stage1_pixels, mask=mask_half, chunk_size=blend_chunk
    )
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

    # 2x pixel upscale (bicubic; torch has no lanczos kernel — the same recorded
    # difference from the official graph run_outpaint carries).
    upscaled = torch.empty((num_frames, height, width, 3), dtype=torch.uint8, device="cpu")
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
    # encode; restoring contiguous afterwards is a CORRECTNESS requirement, not
    # hygiene, because this same encoder is reused within the job.
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
                    "inpaint: failed to restore the video encoder's contiguous layout"
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
        initial_audio_latent=_audio_init(audio, full_shape=full_shape, device=device),
        freeze_kv=0,
        freeze_ka=audio.frozen_frames,
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

    # ── Decode stage 2 -> final blend -> restore -> crop -> mux. ─────────────
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
    if audio.latent is None:
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
    # Second decode of the mask, at full resolution this time (see the module
    # docstring: holding both resolutions at once buys nothing).
    mask_full = place_mask_on_canvas(
        decode_mask_video(
            str(mask_path),
            num_frames=num_frames,
            height=geometry.source_height,
            width=geometry.source_width,
            device=device,
        ),
        geometry,
    )
    fill_pad_bands_with_generated_(
        canvas_full,
        generated=stage2_pixels,
        source_height=geometry.source_height,
        source_width=geometry.source_width,
    )
    fill_mask_with_generated_(
        canvas_full, generated=stage2_pixels, mask=mask_full, chunk_size=blend_chunk
    )
    final_pixels = blend_video_u8(
        stage2_pixels,
        canvas_full,
        mask_full,
        mask_low_res_dilation=blend_dilation_stage2,
        chunk_size=chunk_size,
        device=device,
    )
    blend2_peak_mb = round(torch.cuda.max_memory_allocated(device) / 1e6, 1)
    del stage2_pixels, canvas_full
    cleanup_memory()

    # ── RESTORE. ─────────────────────────────────────────────────────────────
    # A FRESH decode of the canvas, not the de-greened one above: that copy had
    # its mask region and its pad bands overwritten with generated pixels, so it
    # is no longer the original anywhere it matters. Outside the mask the canvas
    # IS the source window (that is what fill_mask_green_mp4 writes), which is
    # why the source does not have to be decoded separately here. The pad bands
    # come back green and are cropped off two steps later.
    canvas_restore = _load_canvas_pixels_u8(
        video_path=str(canvas_path),
        height=height,
        width=width,
        frame_cap=num_frames,
        device=device,
    )[:num_frames]
    mask_proof = _restore_and_measure_(
        blended=final_pixels,
        source=canvas_restore,
        mask=mask_full,
        dilation=blend_dilation_stage2,
        chunk_size=blend_chunk,
        device=device,
    )
    del canvas_restore, mask_full
    cleanup_memory()

    # ── CROP back to the source resolution. ──────────────────────────────────
    # The source is anchored at (0, 0), so this is a slice of the right and
    # bottom bands and nothing else — lossless, at uint8, before the encode.
    final_pixels = final_pixels[
        :, : geometry.source_height, : geometry.source_width, :
    ].contiguous()
    logger.info(
        "inpaint crop: %dx%d canvas -> %dx%d delivered (pads r/b=%d/%d)",
        width, height, geometry.source_width, geometry.source_height,
        geometry.pad_right, geometry.pad_bottom,
    )

    mux_audio, muxed_samples = _mux_audio(
        audio, decoded_audio=decoded_audio, num_frames=num_frames, frame_rate=frame_rate
    )

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
        "inpaint": {
            **geometry.as_dict(),
            "blend_dilation_stage1": int(blend_dilation_stage1),
            "blend_dilation_stage2": int(blend_dilation_stage2),
            "blend_chunk_size": chunk_size,
            "blend1_peak_vram_mb": blend1_peak_mb,
            "blend2_peak_vram_mb": blend2_peak_mb,
            "stage2_sigmas": stage2_values,
            "freeze_source_audio": bool(freeze_source_audio),
            "audio_frozen_latent_frames": int(audio.frozen_frames),
            "audio_latent_frames_required": int(a_total),
            "muxed_original_waveform": audio.latent is not None,
            "muxed_audio_samples": muxed_samples,
            "audio_sampling_rate": int(audio.sampling_rate),
            "canvas_path": str(canvas_path),
            "mask_path": str(mask_path),
            # The evidence that a mask was really decoded and really used. The
            # mock backend cannot fabricate these — it never opens a video — so
            # a metadata.json carrying them is proof the real engine ran the
            # real mask (Docs/INPAINTING_DESIGN.md §7.6).
            "mask_proof": mask_proof,
        },
        "seed": int(seed),
        # The DELIVERED size, not the canvas: this is what the mp4 comes out at.
        "width": geometry.source_width,
        "height": geometry.source_height,
        "num_frames": int(num_frames),
        "num_steps": int(num_steps),
        "wall_s": round(wall, 2),
        "vram_peak_mb": peak,
        "vram_within_16gb": peak < 16000,
        "output_mp4": str(output_path),
    }
    return meta
