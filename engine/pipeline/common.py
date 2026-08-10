"""Shared helpers and primitives for LTX video pipeline wrappers."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, cast

import torch

from engine.api_types import ImageConditioningInput
from engine.pipeline.utils import AudioOrNone, TilingConfigType, device_supports_fp8, sync_device

if TYPE_CHECKING:
    from ltx_core.components.guiders import MultiModalGuiderParams
    from ltx_core.types import LatentState


def default_tiling_config(
    spatial_tile_size: int = 0,
    temporal_tile_size: int = 0,
) -> TilingConfigType:
    """Return a TilingConfig, optionally overriding the spatial/temporal tile sizes.

    Passing 0 (the default) uses the library defaults:
      spatial  512 px tiles with 64 px overlap
      temporal  64 frame tiles with 24 frame overlap
    """
    from ltx_core.model.video_vae import TilingConfig, SpatialTilingConfig, TemporalTilingConfig

    if spatial_tile_size <= 0 and temporal_tile_size <= 0:
        return TilingConfig.default()

    default = TilingConfig.default()
    spatial = (
        SpatialTilingConfig(
            tile_size_in_pixels=max(64, (spatial_tile_size // 32) * 32),
            tile_overlap_in_pixels=64,
        )
        if spatial_tile_size > 0
        else default.spatial_config
    )
    temporal = (
        TemporalTilingConfig(
            tile_size_in_frames=max(16, (temporal_tile_size // 8) * 8),
            tile_overlap_in_frames=24,
        )
        if temporal_tile_size > 0
        else default.temporal_config
    )
    return TilingConfig(spatial_config=spatial, temporal_config=temporal)


def default_guiders() -> tuple[MultiModalGuiderParams, MultiModalGuiderParams]:
    from ltx_core.components.guiders import MultiModalGuiderParams

    return MultiModalGuiderParams(cfg_scale=3.0), MultiModalGuiderParams(cfg_scale=3.0)


def video_chunks_number(num_frames: int, tiling_config: TilingConfigType | None) -> int:
    from ltx_core.model.video_vae import get_video_chunks_number

    return int(get_video_chunks_number(num_frames, tiling_config))


def encode_video_output(
    video: torch.Tensor | Iterator[torch.Tensor],
    audio: AudioOrNone,
    fps: int,
    output_path: str,
    video_chunks_number_value: int,
) -> None:
    from ltx_pipelines.utils.media_io import encode_video

    encode_video(
        video=video,
        fps=fps,
        audio=audio,
        output_path=output_path,
        video_chunks_number=video_chunks_number_value,
    )


def iter_video_conditioning_cpu(
    *,
    video_path: str,
    height: int,
    width: int,
    frame_cap: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Iterator[torch.Tensor]:
    """Frame-by-frame generator behind :func:`load_video_conditioning_cpu`.

    Yields ONE (1,C,1,H,W) CPU tensor per decoded frame, in decode order, having
    run the exact same per-frame ops on the exact same device as the whole-tensor
    loader below — ``decode_video_from_file`` -> ``resize_and_center_crop`` on
    float32 -> ``normalize_latent`` to ``dtype`` -> ``.to("cpu")``. The loader is
    literally ``torch.cat(list(this), dim=2)``, so the two are byte-identical by
    construction; nothing about the numerics, the order, the dtype or the device
    changed when this generator was split out (§1-15 B4).

    Split out for the clip-wise chain reference (§1-15): a 24-clip chain needs
    24 DIFFERENT overlapping pixel windows of one long reference video, and
    materialising the whole video just to slice it would cost the full
    ``total_px`` (up to 11544 frames) on CPU at once. ``chain_pipeline``'s window
    generator consumes this stream lazily instead, holding at most
    ``max(clip_frames)`` frames.

    The GPU-side per-frame temporaries are released BEFORE the yield, so a slow
    consumer never pins a decoded frame on the device.
    """
    from ltx_pipelines.utils.media_io import (
        decode_video_from_file,
        normalize_latent,
        resize_and_center_crop,
    )

    for f in decode_video_from_file(path=video_path, frame_cap=frame_cap, device=device):
        # Same ops, same device (GPU), same dtype flow as load_video_conditioning.
        frame = resize_and_center_crop(f.to(torch.float32), height, width)
        frame = normalize_latent(frame, device, dtype)
        frame_cpu = frame.to("cpu")
        del f, frame
        yield frame_cpu


def load_video_conditioning_cpu(
    *,
    video_path: str,
    height: int,
    width: int,
    frame_cap: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Numerics-preserving, low-VRAM twin of ``media_io.load_video_conditioning``.

    The installed ``load_video_conditioning`` decodes every frame onto ``device``
    (GPU), preprocesses it there, and ``torch.cat``s the growing (1,C,F,H,W)
    tensor **on the GPU** — allocating a fresh full-size buffer per frame while
    the previous one is still live, so the total allocated volume grows with the
    SQUARE of the frame count (F(F+1)/2 frame-sized buffers). At 640x384x257
    that is a measured 46.8GB of reserved VRAM for a tensor whose final size is
    ~380MB; on Windows it does not even fail, it silently spills into shared
    memory and drags the whole job down.

    This twin runs the EXACT same per-frame ops on the EXACT same device (GPU) in
    the same dtype flow — ``decode_video_from_file`` -> ``resize_and_center_crop``
    on float32 -> ``normalize_latent`` to ``dtype`` — but moves each processed
    frame to CPU immediately and assembles the (1,C,F,H,W) tensor on CPU with a
    single ``torch.cat``. The GPU holds at most one frame at a time. Because the
    resize/normalize math still runs on the GPU, the per-frame values are
    bit-identical to the installed loader; the CPU copy is a pure device transfer
    (no arithmetic), and ``torch.cat`` is a deterministic copy, so the assembled
    tensor is byte-identical to the installed loader's output apart from residing
    on CPU. Measured on the same 640x384x257 case: reserved 46,794MB -> 20MB,
    9.45s -> 0.90s, output ``torch.equal`` with the installed loader.

    Callers that hand the result to ``VideoEncoder.tiled_encode`` can pass the
    CPU tensor straight through (tiled_encode streams tiles back to the GPU one
    at a time); callers that call the encoder directly must ``.to(device)`` first.

    Behaviour difference on a 0-frame video: the installed loader returns ``None``
    (its accumulator never gets a first frame), whereas this twin raises from
    ``torch.cat`` on an empty list. Both are failure paths for callers that
    require pixels; no production caller feeds a 0-frame source.

    The per-frame work lives in :func:`iter_video_conditioning_cpu`; this is the
    single ``torch.cat`` over it, so the two cannot drift apart.
    """
    return torch.cat(
        list(
            iter_video_conditioning_cpu(
                video_path=video_path,
                height=height,
                width=width,
                frame_cap=frame_cap,
                dtype=dtype,
                device=device,
            )
        ),
        dim=2,
    )


class DistilledNativePipeline:
    """Fast native pipeline implementation moved from ltx2_server.py."""

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str | None,
        device: torch.device | None = None,
        fp8transformer: bool = False,
    ) -> None:
        from ltx_pipelines.utils import ModelLedger
        from ltx_pipelines.utils.helpers import get_device
        from ltx_pipelines.utils.types import PipelineComponents

        if device is None:
            device = get_device()

        self.device = device
        self.dtype = torch.bfloat16

        from ltx_core.quantization import QuantizationPolicy

        self.model_ledger = ModelLedger(
            dtype=self.dtype,
            device=device,
            checkpoint_path=checkpoint_path,
            gemma_root_path=gemma_root,
            loras=None,
            quantization=QuantizationPolicy.fp8_cast() if fp8transformer and device_supports_fp8(device) else None,
        )
        self.pipeline_components = PipelineComponents(dtype=self.dtype, device=device)

    @torch.inference_mode()
    def __call__(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        tiling_config: TilingConfigType | None = None,
    ) -> tuple[torch.Tensor | Iterator[torch.Tensor], AudioOrNone]:
        from ltx_core.components.diffusion_steps import EulerDiffusionStep
        from ltx_core.components.noisers import GaussianNoiser
        from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
        from ltx_core.model.video_vae import decode_video as vae_decode_video
        from ltx_core.text_encoders.gemma import encode_text
        from ltx_core.types import VideoPixelShape
        from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES
        from ltx_pipelines.utils.args import ImageConditioningInput as _LtxImageInput
        from ltx_pipelines.utils.helpers import (
            cleanup_memory,
            denoise_audio_video,
            image_conditionings_by_replacing_latent,
            simple_denoising_func,
        )
        from ltx_pipelines.utils.samplers import euler_denoising_loop

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()
        dtype = torch.bfloat16

        text_encoder = self.model_ledger.text_encoder()
        context_p = encode_text(text_encoder, prompts=[prompt])[0]
        video_context, audio_context = context_p

        sync_device(self.device)
        del text_encoder
        cleanup_memory()

        video_encoder = self.model_ledger.video_encoder()
        transformer = self.model_ledger.transformer()
        sigmas = torch.Tensor(DISTILLED_SIGMA_VALUES).to(self.device)

        def denoising_loop(
            sigmas: torch.Tensor,
            video_state: LatentState,
            audio_state: LatentState,
            stepper: EulerDiffusionStep,
        ) -> tuple[LatentState, LatentState]:
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=simple_denoising_func(
                    video_context=video_context,
                    audio_context=audio_context,
                    transformer=transformer,
                ),
            )

        output_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=frame_rate)
        # NOTE (latent-index bug): image_conditionings_by_replacing_latent builds
        # VideoConditionByLatentIndex(latent_idx=img.frame_idx) for EVERY image,
        # treating our PIXEL frame_idx as a LATENT index.  For frame_idx > 0 this
        # overflows the latent token buffer and crashes (same root cause fixed in
        # LTXFastVideoPipeline._run_inference via the keyframe hybrid).  This
        # DistilledNativePipeline is NOT on the production path (worker.py uses
        # only LTXFastVideoPipeline), so it is left as-is; if this one-stage path
        # is ever put into production it MUST adopt the same hybrid routing
        # (idx == 0 → replace, idx > 0 → image_conditionings_by_adding_guiding_latent).
        conditionings = image_conditionings_by_replacing_latent(
            images=[_LtxImageInput(img.path, img.frame_idx, img.strength) for img in images],
            height=output_shape.height,
            width=output_shape.width,
            video_encoder=video_encoder,
            dtype=dtype,
            device=self.device,
        )

        video_state, audio_state = denoise_audio_video(
            output_shape=output_shape,
            conditionings=conditionings,
            noiser=noiser,
            sigmas=sigmas,
            stepper=stepper,
            denoising_loop_fn=cast(Any, denoising_loop),
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
        )

        sync_device(self.device)
        del transformer
        del video_encoder
        cleanup_memory()

        decoded_video = vae_decode_video(video_state.latent, self.model_ledger.video_decoder(), tiling_config)
        decoded_audio = vae_decode_audio(
            audio_state.latent,
            self.model_ledger.audio_decoder(),
            self.model_ledger.vocoder(),
        )
        return decoded_video, decoded_audio
