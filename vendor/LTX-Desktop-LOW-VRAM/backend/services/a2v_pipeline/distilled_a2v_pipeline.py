"""Distilled A2V (Audio-to-Video) pipeline.

Combines the distilled denoising approach (fixed sigmas, simple_denoising_func,
single ModelLedger) with A2V-specific behaviour (audio encoding, video-only
denoising with frozen audio, returning original audio).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import torch

from services.services_utils import AudioOrNone, TilingConfigType, sync_device

if TYPE_CHECKING:
    from ltx_core.loader.primitives import LoraPathStrengthAndSDOps
    from ltx_core.types import LatentState


class DistilledA2VPipeline:
    """Two-stage distilled audio-to-video pipeline.

    Stage 1 generates video at half resolution with frozen audio conditioning,
    then Stage 2 upsamples by 2x and refines with additional distilled steps.
    Uses a single ModelLedger (no LoRA swap between stages).
    """

    def __init__(
        self,
        distilled_checkpoint_path: str,
        gemma_root: str,
        spatial_upsampler_path: str,
        loras: LoraPathStrengthAndSDOps | None = None,
        device: torch.device | None = None,
        quantization: Any | None = None,
    ) -> None:
        from ltx_pipelines.utils import ModelLedger
        from ltx_pipelines.utils.helpers import get_device
        from ltx_pipelines.utils.types import PipelineComponents

        if device is None:
            device = get_device()

        self.device = device
        self.dtype = torch.bfloat16

        self.model_ledger = ModelLedger(
            dtype=self.dtype,
            device=device,
            checkpoint_path=distilled_checkpoint_path,
            gemma_root_path=gemma_root,
            spatial_upsampler_path=spatial_upsampler_path,
            loras=loras,
            quantization=quantization,
        )

        self.pipeline_components = PipelineComponents(
            dtype=self.dtype,
            device=device,
        )

    @torch.inference_mode()
    def __call__(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[tuple[str, int, float]],
        audio_path: str,
        audio_start_time: float = 0.0,
        audio_max_duration: float | None = None,
        tiling_config: TilingConfigType | None = None,
    ) -> tuple[Iterator[torch.Tensor], AudioOrNone]:
        from ltx_core.components.diffusion_steps import EulerDiffusionStep
        from ltx_core.components.noisers import GaussianNoiser
        from ltx_core.components.protocols import DiffusionStepProtocol
        from ltx_core.model.audio_vae import encode_audio as vae_encode_audio
        from ltx_core.model.upsampler import upsample_video
        from ltx_core.model.video_vae import decode_video as vae_decode_video
        from ltx_core.text_encoders.gemma import encode_text
        from ltx_core.types import Audio, AudioLatentShape, VideoPixelShape
        from ltx_pipelines.utils.args import ImageConditioningInput as LtxImageInput
        from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES
        from ltx_pipelines.utils.helpers import (
            assert_resolution,
            cleanup_memory,
            denoise_video_only,
            image_conditionings_by_replacing_latent,
            simple_denoising_func,
        )
        from ltx_pipelines.utils.media_io import decode_audio_from_file
        from ltx_pipelines.utils.samplers import euler_denoising_loop

        assert_resolution(height=height, width=width, is_two_stage=True)

        ltx_images = [LtxImageInput(path, frame_idx, strength) for path, frame_idx, strength in images]
        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()
        dtype = torch.bfloat16

        # Text encode (positive only).
        text_encoder = self.model_ledger.text_encoder()
        context_p = encode_text(text_encoder, prompts=[prompt])[0]
        video_context, audio_context = context_p

        sync_device(self.device)
        del text_encoder
        cleanup_memory()

        # Audio encode.
        decoded_audio = decode_audio_from_file(audio_path, self.device, audio_start_time, audio_max_duration)
        assert decoded_audio is not None, "Audio file contains no audio stream"
        encoded_audio_latent = vae_encode_audio(
            decoded_audio, self.model_ledger.audio_encoder()
        )
        audio_shape = AudioLatentShape.from_duration(batch=1, duration=num_frames / frame_rate, channels=8, mel_bins=16)
        target_frames = audio_shape.frames
        if encoded_audio_latent.shape[2] < target_frames:
            pad_size = target_frames - encoded_audio_latent.shape[2]
            encoded_audio_latent = torch.nn.functional.pad(encoded_audio_latent, (0, 0, 0, pad_size))
        else:
            encoded_audio_latent = encoded_audio_latent[:, :, :target_frames]

        cleanup_memory()

        # Shared denoising closure (simple, no guidance).
        video_encoder = self.model_ledger.video_encoder()
        transformer = self.model_ledger.transformer()

        def denoising_loop(
            sigmas: torch.Tensor,
            video_state: LatentState,
            audio_state: LatentState,
            stepper: DiffusionStepProtocol,
        ) -> tuple[LatentState, LatentState]:
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=simple_denoising_func(
                    video_context=video_context,
                    audio_context=audio_context,
                    transformer=transformer,  # noqa: F821
                ),
            )

        # Stage 1: Half-resolution video generation with frozen audio.
        stage_1_sigmas = torch.Tensor(DISTILLED_SIGMA_VALUES).to(self.device)
        stage_1_output_shape = VideoPixelShape(
            batch=1,
            frames=num_frames,
            width=width // 2,
            height=height // 2,
            fps=frame_rate,
        )
        stage_1_conditionings = image_conditionings_by_replacing_latent(
            images=ltx_images,
            height=stage_1_output_shape.height,
            width=stage_1_output_shape.width,
            video_encoder=video_encoder,
            dtype=dtype,
            device=self.device,
        )

        video_state = denoise_video_only(
            output_shape=stage_1_output_shape,
            conditionings=stage_1_conditionings,
            noiser=noiser,
            sigmas=stage_1_sigmas,
            stepper=stepper,
            denoising_loop_fn=denoising_loop,  # type: ignore[arg-type]
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
            initial_audio_latent=encoded_audio_latent,
        )

        # Upsample video 2x.
        upscaled_video_latent = upsample_video(
            latent=video_state.latent[:1],
            video_encoder=video_encoder,
            upsampler=self.model_ledger.spatial_upsampler(),
        )

        sync_device(self.device)
        cleanup_memory()

        # Stage 2: Full-resolution refinement with frozen audio.
        stage_2_sigmas = torch.Tensor(STAGE_2_DISTILLED_SIGMA_VALUES).to(self.device)
        stage_2_output_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=frame_rate)
        stage_2_conditionings = image_conditionings_by_replacing_latent(
            images=ltx_images,
            height=stage_2_output_shape.height,
            width=stage_2_output_shape.width,
            video_encoder=video_encoder,
            dtype=dtype,
            device=self.device,
        )

        video_state = denoise_video_only(
            output_shape=stage_2_output_shape,
            conditionings=stage_2_conditionings,
            noiser=noiser,
            sigmas=stage_2_sigmas,
            stepper=stepper,
            denoising_loop_fn=denoising_loop,  # type: ignore[arg-type]
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
            noise_scale=stage_2_sigmas[0].item(),
            initial_video_latent=upscaled_video_latent,
            initial_audio_latent=encoded_audio_latent,
        )

        sync_device(self.device)
        del transformer
        del video_encoder
        cleanup_memory()

        # Decode video; return original audio (not VAE-decoded) for fidelity.
        decoded_video = vae_decode_video(video_state.latent, self.model_ledger.video_decoder(), tiling_config, generator)

        # Trim waveform to target video duration so the muxed output doesn't
        # extend beyond the generated video frames.
        max_samples = round(num_frames / frame_rate * decoded_audio.sampling_rate)
        trimmed_waveform = decoded_audio.waveform.squeeze(0)[..., :max_samples]
        original_audio = Audio(waveform=trimmed_waveform, sampling_rate=decoded_audio.sampling_rate)

        return decoded_video, original_audio
