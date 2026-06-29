"""Video generation orchestration handler."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING

from PIL import Image

from api_types import ConditioningImageRequest, GenerateVideoRequest, GenerateVideoResponse, ImageConditioningInput, VideoCameraMotion
from _routes._errors import HTTPError
from handlers.base import StateHandlerBase
from handlers.generation_handler import GenerationHandler
from handlers.pipelines_handler import PipelinesHandler
from handlers.text_handler import TextHandler
from runtime_config.model_download_specs import resolve_model_path
from server_utils.media_validation import (
    normalize_optional_path,
    validate_audio_file,
    validate_image_file,
)
from services.interfaces import LTXAPIClient
from state.app_state_types import AppState
from state.app_settings import should_video_generate_with_ltx_api

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig

logger = logging.getLogger(__name__)

FORCED_API_MODEL_MAP: dict[str, str] = {
    "fast": "ltx-2-3-fast",
    "pro": "ltx-2-3-pro",
}
FORCED_API_RESOLUTION_MAP: dict[str, dict[str, str]] = {
    "1080p": {"16:9": "1920x1080", "9:16": "1080x1920"},
    "1440p": {"16:9": "2560x1440", "9:16": "1440x2560"},
    "2160p": {"16:9": "3840x2160", "9:16": "2160x3840"},
}
A2V_FORCED_API_RESOLUTION = "1920x1080"
FORCED_API_ALLOWED_ASPECT_RATIOS = {"16:9", "9:16"}
FORCED_API_ALLOWED_FPS = {24, 25, 48, 50}


def _get_allowed_durations(model_id: str, resolution_label: str, fps: int) -> set[int]:
    if model_id == "ltx-2-3-fast" and resolution_label == "1080p" and fps in {24, 25}:
        return {6, 8, 10, 12, 14, 16, 18, 20}
    return {6, 8, 10}


class VideoGenerationHandler(StateHandlerBase):
    def __init__(
        self,
        state: AppState,
        lock: RLock,
        generation_handler: GenerationHandler,
        pipelines_handler: PipelinesHandler,
        text_handler: TextHandler,
        ltx_api_client: LTXAPIClient,
        config: RuntimeConfig,
    ) -> None:
        super().__init__(state, lock, config)
        self._generation = generation_handler
        self._pipelines = pipelines_handler
        self._text = text_handler
        self._ltx_api_client = ltx_api_client
        self._completed_generation_count: int = 0

    def generate(self, req: GenerateVideoRequest) -> GenerateVideoResponse:
        if should_video_generate_with_ltx_api(
            force_api_generations=self.config.force_api_generations,
            settings=self.state.app_settings,
        ):
            return self._generate_forced_api(req)

        if self._generation.is_generation_running():
            raise HTTPError(409, "Generation already in progress")

        resolution = req.resolution

        duration = int(float(req.duration))
        fps = int(float(req.fps))

        audio_path = normalize_optional_path(req.audioPath)
        if audio_path:
            return self._generate_a2v(req, duration, fps, audio_path=audio_path)

        if req.model == "dev":
            return self._generate_dev(req, duration, fps)

        logger.info("Resolution %s - using fast pipeline", resolution)

        RESOLUTION_MAP_16_9: dict[str, tuple[int, int]] = {
            "540p": (960, 544),
            "720p": (1280, 704),
            "1080p": (1920, 1088),
        }

        def get_16_9_size(res: str) -> tuple[int, int]:
            return RESOLUTION_MAP_16_9.get(res, (960, 544))

        def get_9_16_size(res: str) -> tuple[int, int]:
            w, h = get_16_9_size(res)
            return h, w

        match req.aspectRatio:
            case "9:16":
                width, height = get_9_16_size(resolution)
            case "16:9":
                width, height = get_16_9_size(resolution)

        num_frames = self._compute_num_frames(duration, fps)

        # Multi-frame conditioning: if conditioningImages provided, use those;
        # otherwise fall back to single imagePath (backward compat).
        pre_built_images: list[ImageConditioningInput] | None = None
        image = None
        if req.conditioningImages:
            pre_built_images = self._prepare_conditioning_images(req.conditioningImages, width, height, num_frames)
            logger.info("Multi-frame conditioning: %d images", len(pre_built_images))
        else:
            image_path = normalize_optional_path(req.imagePath)
            if image_path:
                image = self._prepare_image(image_path, width, height)
                logger.info("Image: %s -> %sx%s", image_path, width, height)

        generation_id = self._make_generation_id()
        seed = self._resolve_seed(req.seed)

        try:
            self._pipelines.load_gpu_pipeline("fast", should_warm=False)
            self._generation.start_generation(generation_id)

            t_gen_start = time.perf_counter()
            output_path = self.generate_video(
                prompt=req.enhancedPrompt or req.prompt,
                image=image,
                height=height,
                width=width,
                num_frames=num_frames,
                fps=fps,
                seed=seed,
                camera_motion=req.cameraMotion,
                negative_prompt=req.negativePrompt,
                pre_built_images=pre_built_images,
                num_steps_override=req.numSteps,
                stg_scale_override=req.stgScale,
                stg_block_index_override=req.stgBlockIndex,
            )
            render_time = time.perf_counter() - t_gen_start

            self._write_sidecar(
                video_path=output_path,
                prompt=req.prompt,
                enhanced_prompt=req.enhancedPrompt,
                negative_prompt=req.negativePrompt,
                resolution=resolution,
                width=width,
                height=height,
                num_frames=num_frames,
                fps=fps,
                duration=duration,
                aspect_ratio=req.aspectRatio,
                camera_motion=req.cameraMotion,
                model=req.model,
                seed=seed,
                render_time_seconds=render_time,
            )

            self._generation.complete_generation(output_path)
            return GenerateVideoResponse(status="complete", video_path=output_path, seed_used=seed)

        except Exception as e:
            self._generation.fail_generation(str(e))
            if "cancelled" in str(e).lower():
                logger.info("Generation cancelled by user")
                return GenerateVideoResponse(status="cancelled")

            # After a CUDA OOM the GPU memory state is corrupted — any subsequent
            # CUDA operation risks an access violation (Windows exit code 0xC0000005).
            # Unload the pipeline immediately so VRAM is freed and the next generation
            # starts clean rather than crashing the process.
            if "out of memory" in str(e).lower():
                logger.warning("CUDA OOM detected — unloading pipeline to recover VRAM")
                try:
                    self._pipelines.unload_gpu_pipeline()
                except Exception:
                    logger.warning("Pipeline unload after OOM failed", exc_info=True)

            raise HTTPError(500, str(e)) from e

    @staticmethod
    def _raw_frame_to_latent_idx(raw_idx: int, num_frames: int) -> int:
        """Convert a raw video frame index to a latent frame index (8:1 temporal compression).

        frame_idx=0 → latent 0 (first)
        frame_idx=-1 → latent (latent_frames - 1) (last)
        frame_idx=N → latent N // 8, clamped to valid range
        """
        temporal_compression = 8
        latent_frames = (num_frames - 1) // temporal_compression + 1
        if raw_idx < 0:
            return latent_frames - 1
        return min(raw_idx // temporal_compression, latent_frames - 1)

    def _prepare_conditioning_images(
        self,
        conditioning: list[ConditioningImageRequest],
        width: int,
        height: int,
        num_frames: int,
    ) -> list[ImageConditioningInput]:
        """Prepare multi-frame conditioning images: resize each and save to temp files."""
        result: list[ImageConditioningInput] = []
        for ci in conditioning:
            ci_path = normalize_optional_path(ci.path)
            if not ci_path or not Path(ci_path).exists():
                logger.warning("Conditioning image not found, skipping: %s", ci.path)
                continue
            latent_idx = self._raw_frame_to_latent_idx(ci.frameIdx, num_frames)
            pil_img = self._prepare_image(ci_path, width, height)
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
            pil_img.save(tmp)
            result.append(ImageConditioningInput(path=tmp, frame_idx=latent_idx, strength=ci.strength))
            logger.info("Conditioning frame: %s raw_idx=%d latent_idx=%d strength=%.2f",
                        ci_path, ci.frameIdx, latent_idx, ci.strength)
        return result

    def generate_video(
        self,
        prompt: str,
        image: Image.Image | None,
        height: int,
        width: int,
        num_frames: int,
        fps: float,
        seed: int,
        camera_motion: VideoCameraMotion,
        negative_prompt: str,
        pre_built_images: list[ImageConditioningInput] | None = None,
        num_steps_override: int | None = None,
        stg_scale_override: float | None = None,
        stg_block_index_override: int | None = None,
    ) -> str:
        t_total_start = time.perf_counter()
        gen_mode = "i2v" if image is not None else "t2v"
        logger.info("[%s] Generation started (model=fast, %dx%d, %d frames, %d fps)", gen_mode, width, height, num_frames, int(fps))

        if self._generation.is_generation_cancelled():
            raise RuntimeError("Generation was cancelled")

        if not resolve_model_path(self.models_dir, self.config.model_download_specs,"checkpoint").exists():
            raise RuntimeError("Models not downloaded. Please download the AI models first using the Model Status menu.")

        settings = self.state.app_settings
        total_steps = num_steps_override if num_steps_override is not None else settings.distilled_num_steps
        eff_stg_scale = stg_scale_override if stg_scale_override is not None else settings.stg_scale
        eff_stg_block_index = stg_block_index_override if stg_block_index_override is not None else settings.stg_block_index
        sigma_schedule = settings.distilled_sigma_schedule
        denoising_loop = settings.denoising_loop
        ge_gamma = settings.ge_gamma
        res2s_bongmath = settings.res2s_bongmath
        res2s_bongmath_max_iter = settings.res2s_bongmath_max_iter

        # OOM prevention: force pipeline eviction every N completions so
        # VRAM fragmentation doesn't accumulate across generations.
        reload_every = settings.reload_pipeline_every_n_gens
        if reload_every > 0 and self._completed_generation_count > 0 and self._completed_generation_count % reload_every == 0:
            logger.info(
                "VRAM defrag: reloading pipeline after %d generations (reload_every=%d)",
                self._completed_generation_count, reload_every,
            )
            self._pipelines.unload_gpu_pipeline()

        self._generation.update_progress("loading_model", 5, 0, total_steps)
        t_load_start = time.perf_counter()
        pipeline_state = self._pipelines.load_gpu_pipeline("fast", should_warm=False)
        t_load_end = time.perf_counter()
        logger.info("[%s] Pipeline load: %.2fs", gen_mode, t_load_end - t_load_start)

        self._generation.update_progress("encoding_text", 10, 0, total_steps)

        enhanced_prompt = prompt + self.config.camera_motion_prompts.get(camera_motion, "")

        images: list[ImageConditioningInput] = []
        temp_image_path: str | None = None
        if pre_built_images is not None:
            # Multi-frame path: temp files already created by _prepare_conditioning_images
            images = pre_built_images
        elif image is not None:
            temp_image_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
            image.save(temp_image_path)
            images = [ImageConditioningInput(path=temp_image_path, frame_idx=0, strength=1.0)]

        output_path = self._make_output_path()

        try:
            use_api_encoding = not self._text.should_use_local_encoding()
            if image is not None:
                enhance = use_api_encoding and settings.prompt_enhancer_enabled_i2v
            else:
                enhance = use_api_encoding and settings.prompt_enhancer_enabled_t2v

            encoding_method = "api" if use_api_encoding else "local"
            t_text_start = time.perf_counter()
            self._text.prepare_text_encoding(enhanced_prompt, enhance_prompt=enhance)
            t_text_end = time.perf_counter()
            logger.info("[%s] Text encoding (%s): %.2fs", gen_mode, encoding_method, t_text_end - t_text_start)

            self._generation.update_progress("inference", 15, 0, total_steps)

            height = round(height / 64) * 64
            width = round(width / 64) * 64

            logger.info(
                "[%s] Inference params: steps=%d stg_scale=%.2f stg_block=%d sigma_schedule=%s denoising_loop=%s ge_gamma=%.2f res2s_bongmath=%s(%d)",
                gen_mode, total_steps, eff_stg_scale, eff_stg_block_index, sigma_schedule,
                denoising_loop, ge_gamma, res2s_bongmath, res2s_bongmath_max_iter,
            )
            t_inference_start = time.perf_counter()
            pipeline_state.pipeline.generate(
                prompt=enhanced_prompt,
                seed=seed,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=fps,
                images=images,
                output_path=str(output_path),
                num_steps=total_steps,
                stg_scale=eff_stg_scale,
                stg_block_index=eff_stg_block_index,
                sigma_schedule=sigma_schedule,
                denoising_loop=denoising_loop,
                ge_gamma=ge_gamma,
                res2s_bongmath=res2s_bongmath,
                res2s_bongmath_max_iter=res2s_bongmath_max_iter,
            )
            t_inference_end = time.perf_counter()
            logger.info("[%s] Inference: %.2fs", gen_mode, t_inference_end - t_inference_start)

            if self._generation.is_generation_cancelled():
                if output_path.exists():
                    output_path.unlink()
                raise RuntimeError("Generation was cancelled")

            t_total_end = time.perf_counter()
            logger.info("[%s] Total generation: %.2fs (load=%.2fs, text=%.2fs, inference=%.2fs)",
                        gen_mode, t_total_end - t_total_start,
                        t_load_end - t_load_start, t_text_end - t_text_start, t_inference_end - t_inference_start)

            self._completed_generation_count += 1
            self._generation.update_progress("complete", 100, total_steps, total_steps)
            return str(output_path)
        finally:
            self._text.clear_api_embeddings()
            if temp_image_path and os.path.exists(temp_image_path):
                os.unlink(temp_image_path)
            if pre_built_images:
                for img_input in pre_built_images:
                    if os.path.exists(img_input.path):
                        try:
                            os.unlink(img_input.path)
                        except OSError:
                            pass

    def _generate_dev(
        self, req: GenerateVideoRequest, duration: int, fps: int
    ) -> GenerateVideoResponse:
        """Generate using the dev (TI2VidTwoStages) pipeline for higher quality output."""
        resolution = req.resolution

        RESOLUTION_MAP_16_9: dict[str, tuple[int, int]] = {
            "540p": (960, 544),
            "720p": (1280, 704),
            "1080p": (1920, 1088),
        }

        def get_16_9_size(res: str) -> tuple[int, int]:
            return RESOLUTION_MAP_16_9.get(res, (960, 544))

        match req.aspectRatio:
            case "9:16":
                w, h = get_16_9_size(resolution)
                width, height = h, w
            case _:
                width, height = get_16_9_size(resolution)

        num_frames = self._compute_num_frames(duration, fps)

        pre_built_images: list[ImageConditioningInput] | None = None
        image = None
        if req.conditioningImages:
            pre_built_images = self._prepare_conditioning_images(req.conditioningImages, width, height, num_frames)
        else:
            image_path = normalize_optional_path(req.imagePath)
            if image_path:
                image = self._prepare_image(image_path, width, height)

        generation_id = self._make_generation_id()
        seed = self._resolve_seed(req.seed)

        settings = self.state.app_settings
        num_steps = req.numSteps if req.numSteps is not None else 30
        eff_stg_scale = req.stgScale if req.stgScale is not None else settings.stg_scale
        eff_stg_block_index = req.stgBlockIndex if req.stgBlockIndex is not None else settings.stg_block_index
        cfg_scale = req.cfgScale if req.cfgScale is not None else 3.0
        audio_cfg_scale = req.audioCfgScale if req.audioCfgScale is not None else 7.0
        rescale_scale = req.rescaleScale if req.rescaleScale is not None else 0.7
        modality_scale = req.modalityScale if req.modalityScale is not None else 3.0

        try:
            self._generation.start_generation(generation_id)
            t_gen_start = time.perf_counter()

            enhanced_prompt = (req.enhancedPrompt or req.prompt) + self.config.camera_motion_prompts.get(req.cameraMotion, "")
            negative_prompt = req.negativePrompt

            images: list[ImageConditioningInput] = []
            temp_image_path: str | None = None
            if pre_built_images is not None:
                images = pre_built_images
            elif image is not None:
                temp_image_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
                image.save(temp_image_path)
                images = [ImageConditioningInput(path=temp_image_path, frame_idx=0, strength=1.0)]

            output_path = self._make_output_path()

            self._generation.update_progress("loading_model", 5, 0, num_steps)
            pipeline_state = self._pipelines.load_gpu_pipeline("dev", should_warm=False)

            use_api_encoding = not self._text.should_use_local_encoding()
            enhance = use_api_encoding and (
                settings.prompt_enhancer_enabled_i2v if image is not None
                else settings.prompt_enhancer_enabled_t2v
            )
            self._generation.update_progress("encoding_text", 10, 0, num_steps)
            self._text.prepare_text_encoding(enhanced_prompt, enhance_prompt=enhance)
            self._generation.update_progress("inference", 15, 0, num_steps)

            height = round(height / 64) * 64
            width = round(width / 64) * 64

            logger.info(
                "[dev] Inference: steps=%d cfg=%.2f audio_cfg=%.2f stg=%.2f stg_block=%d rescale=%.2f modality=%.2f",
                num_steps, cfg_scale, audio_cfg_scale, eff_stg_scale, eff_stg_block_index, rescale_scale, modality_scale,
            )
            t_inference_start = time.perf_counter()
            pipeline_state.pipeline.generate(
                prompt=enhanced_prompt,
                negative_prompt=negative_prompt,
                seed=seed,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=fps,
                images=images,
                output_path=str(output_path),
                num_steps=num_steps,
                cfg_scale=cfg_scale,
                audio_cfg_scale=audio_cfg_scale,
                stg_scale=eff_stg_scale,
                stg_block_index=eff_stg_block_index,
                rescale_scale=rescale_scale,
                modality_scale=modality_scale,
            )
            t_inference_end = time.perf_counter()
            render_time = time.perf_counter() - t_gen_start
            logger.info("[dev] Inference: %.2fs total: %.2fs", t_inference_end - t_inference_start, render_time)

            if self._generation.is_generation_cancelled():
                if output_path.exists():
                    output_path.unlink()
                raise RuntimeError("Generation was cancelled")

            self._write_sidecar(
                video_path=str(output_path),
                prompt=req.prompt,
                enhanced_prompt=req.enhancedPrompt,
                negative_prompt=negative_prompt,
                resolution=resolution,
                width=width,
                height=height,
                num_frames=num_frames,
                fps=fps,
                duration=duration,
                aspect_ratio=req.aspectRatio,
                camera_motion=req.cameraMotion,
                model=req.model,
                seed=seed,
                render_time_seconds=render_time,
            )

            self._completed_generation_count += 1
            self._generation.update_progress("complete", 100, num_steps, num_steps)
            self._generation.complete_generation(str(output_path))
            return GenerateVideoResponse(status="complete", video_path=str(output_path), seed_used=seed)

        except Exception as e:
            self._generation.fail_generation(str(e))
            if "cancelled" in str(e).lower():
                logger.info("Generation cancelled by user")
                return GenerateVideoResponse(status="cancelled")
            if "out of memory" in str(e).lower():
                logger.warning("CUDA OOM detected — unloading pipeline to recover VRAM")
                try:
                    self._pipelines.unload_gpu_pipeline()
                except Exception:
                    logger.warning("Pipeline unload after OOM failed", exc_info=True)
            raise HTTPError(500, str(e)) from e
        finally:
            self._text.clear_api_embeddings()
            if temp_image_path and os.path.exists(temp_image_path):
                os.unlink(temp_image_path)
            if pre_built_images:
                for img_input in pre_built_images:
                    if os.path.exists(img_input.path):
                        try:
                            os.unlink(img_input.path)
                        except OSError:
                            pass

    def _generate_a2v(
        self, req: GenerateVideoRequest, duration: int, fps: int, *, audio_path: str
    ) -> GenerateVideoResponse:
        if req.model != "pro":
            logger.warning("A2V local requested with model=%s; A2V always uses pro pipeline", req.model)
        validated_audio_path = validate_audio_file(audio_path)
        audio_path_str = str(validated_audio_path)

        RESOLUTION_MAP: dict[str, tuple[int, int]] = {
            "540p": (960, 576),
            "720p": (1280, 704),
            "1080p": (1920, 1088),
        }
        width, height = RESOLUTION_MAP.get(req.resolution, (960, 576))

        num_frames = self._compute_num_frames(duration, fps)

        image = None
        temp_image_path: str | None = None
        image_path = normalize_optional_path(req.imagePath)
        if image_path:
            image = self._prepare_image(image_path, width, height)

        seed = self._resolve_seed()

        generation_id = self._make_generation_id()

        try:
            a2v_state = self._pipelines.load_a2v_pipeline()
            self._generation.start_generation(generation_id)

            enhanced_prompt = req.prompt + self.config.camera_motion_prompts.get(req.cameraMotion, "")
            neg = req.negativePrompt if req.negativePrompt else self.config.default_negative_prompt

            images: list[ImageConditioningInput] = []
            if image is not None:
                temp_image_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
                image.save(temp_image_path)
                images = [ImageConditioningInput(path=temp_image_path, frame_idx=0, strength=1.0)]

            output_path = self._make_output_path()

            total_steps = 11  # distilled: 8 steps (stage 1) + 3 steps (stage 2)

            a2v_settings = self.state.app_settings
            a2v_use_api = not self._text.should_use_local_encoding()
            if image is not None:
                a2v_enhance = a2v_use_api and a2v_settings.prompt_enhancer_enabled_i2v
            else:
                a2v_enhance = a2v_use_api and a2v_settings.prompt_enhancer_enabled_t2v

            self._generation.update_progress("loading_model", 5, 0, total_steps)
            self._generation.update_progress("encoding_text", 10, 0, total_steps)
            self._text.prepare_text_encoding(enhanced_prompt, enhance_prompt=a2v_enhance)
            self._generation.update_progress("inference", 15, 0, total_steps)

            a2v_state.pipeline.generate(
                prompt=enhanced_prompt,
                negative_prompt=neg,
                seed=seed,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=fps,
                num_inference_steps=total_steps,
                images=images,
                audio_path=audio_path_str,
                audio_start_time=0.0,
                audio_max_duration=None,
                output_path=str(output_path),
            )

            if self._generation.is_generation_cancelled():
                if output_path.exists():
                    output_path.unlink()
                raise RuntimeError("Generation was cancelled")

            self._generation.update_progress("complete", 100, total_steps, total_steps)
            self._generation.complete_generation(str(output_path))
            return GenerateVideoResponse(status="complete", video_path=str(output_path))

        except Exception as e:
            self._generation.fail_generation(str(e))
            if "cancelled" in str(e).lower():
                logger.info("Generation cancelled by user")
                return GenerateVideoResponse(status="cancelled")
            raise HTTPError(500, str(e)) from e
        finally:
            self._text.clear_api_embeddings()
            if temp_image_path and os.path.exists(temp_image_path):
                os.unlink(temp_image_path)

    def _prepare_image(self, image_path: str, width: int, height: int) -> Image.Image:
        validated_path = validate_image_file(image_path)
        try:
            img = Image.open(validated_path).convert("RGB")
        except Exception:
            raise HTTPError(400, f"Invalid image file: {image_path}") from None
        img_w, img_h = img.size
        target_ratio = width / height
        img_ratio = img_w / img_h
        if img_ratio > target_ratio:
            new_h = height
            new_w = int(img_w * (height / img_h))
        else:
            new_w = width
            new_h = int(img_h * (width / img_w))
        resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        left = (new_w - width) // 2
        top = (new_h - height) // 2
        return resized.crop((left, top, left + width, top + height))

    @staticmethod
    def _make_generation_id() -> str:
        return uuid.uuid4().hex[:8]

    @staticmethod
    def _compute_num_frames(duration: int, fps: int) -> int:
        n = ((duration * fps) // 8) * 8 + 1
        return max(n, 9)

    def _resolve_seed(self, request_seed: int | None = None) -> int:
        if request_seed is not None:
            logger.info("Using request seed: %s", request_seed)
            return request_seed
        settings = self.state.app_settings
        if settings.seed_locked:
            logger.info("Using locked seed: %s", settings.locked_seed)
            return settings.locked_seed
        return int(time.time()) % 2147483647

    def _make_output_path(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.config.outputs_dir / f"ltx2_video_{timestamp}_{self._make_generation_id()}.mp4"

    def _write_sidecar(
        self,
        video_path: str,
        prompt: str,
        negative_prompt: str,
        resolution: str,
        width: int,
        height: int,
        num_frames: int,
        fps: int,
        duration: int,
        aspect_ratio: str,
        camera_motion: str,
        model: str,
        seed: int,
        enhanced_prompt: str | None = None,
        render_time_seconds: float | None = None,
    ) -> None:
        settings = self.state.app_settings
        # Collect active LoRAs (name + strength) for the sidecar
        active_loras: list[dict] = []
        if settings.civitai_loras:
            try:
                import json as _json
                for entry in _json.loads(settings.civitai_loras):
                    if entry.get("enabled", True):
                        active_loras.append({
                            "name": Path(entry["path"]).name,
                            "strength": entry.get("strength", 1.0),
                        })
            except Exception:
                pass
        sidecar = {
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "enhanced_prompt": enhanced_prompt,
            "negative_prompt": negative_prompt,
            "model": model,
            "resolution": resolution,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "duration_seconds": duration,
            "fps": fps,
            "aspect_ratio": aspect_ratio,
            "camera_motion": camera_motion,
            "seed": seed,
            "render_time_seconds": round(render_time_seconds, 1) if render_time_seconds is not None else None,
            "block_swap_blocks_on_gpu": settings.block_swap_blocks_on_gpu,
            "use_fp8_transformer": settings.use_fp8_transformer,
            "attention_tile_size": settings.attention_tile_size,
            "use_multi_gpu": settings.use_multi_gpu,
            "loras": active_loras if active_loras else None,
        }
        sidecar_path = Path(video_path).with_suffix(".json")
        try:
            sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to write sidecar metadata: %s", exc)

    def _generate_forced_api(self, req: GenerateVideoRequest) -> GenerateVideoResponse:
        if self._generation.is_generation_running():
            raise HTTPError(409, "Generation already in progress")

        generation_id = self._make_generation_id()
        self._generation.start_api_generation(generation_id)

        audio_path = normalize_optional_path(req.audioPath)
        image_path = normalize_optional_path(req.imagePath)
        has_input_audio = bool(audio_path)
        has_input_image = bool(image_path)

        try:
            self._generation.update_progress("validating_request", 5, None, None)

            api_key = self.state.app_settings.ltx_api_key.strip()
            logger.info("Forced API generation route selected (key_present=%s)", bool(api_key))
            if not api_key:
                raise HTTPError(400, "PRO_API_KEY_REQUIRED")

            requested_model = req.model.strip().lower()
            api_model_id = FORCED_API_MODEL_MAP.get(requested_model)
            if api_model_id is None:
                raise HTTPError(400, "INVALID_FORCED_API_MODEL")

            resolution_label = req.resolution
            resolution_by_aspect = FORCED_API_RESOLUTION_MAP.get(resolution_label)
            if resolution_by_aspect is None:
                raise HTTPError(400, "INVALID_FORCED_API_RESOLUTION")

            aspect_ratio = req.aspectRatio.strip()
            if aspect_ratio not in FORCED_API_ALLOWED_ASPECT_RATIOS:
                raise HTTPError(400, "INVALID_FORCED_API_ASPECT_RATIO")

            api_resolution = resolution_by_aspect[aspect_ratio]

            prompt = req.prompt

            if self._generation.is_generation_cancelled():
                raise RuntimeError("Generation was cancelled")

            if has_input_audio:
                if requested_model != "pro":
                    logger.warning("A2V requested with model=%s; overriding to 'pro'", requested_model)
                api_model_id = FORCED_API_MODEL_MAP["pro"]
                if api_resolution != A2V_FORCED_API_RESOLUTION:
                    logger.warning("A2V requested with resolution=%s; overriding to '%s'", api_resolution, A2V_FORCED_API_RESOLUTION)
                api_resolution = A2V_FORCED_API_RESOLUTION
                validated_audio_path = validate_audio_file(audio_path)
                validated_image_path: Path | None = None
                if image_path is not None:
                    validated_image_path = validate_image_file(image_path)

                self._generation.update_progress("uploading_audio", 20, None, None)
                audio_uri = self._ltx_api_client.upload_file(
                    api_key=api_key,
                    file_path=str(validated_audio_path),
                )
                image_uri: str | None = None
                if validated_image_path is not None:
                    self._generation.update_progress("uploading_image", 35, None, None)
                    image_uri = self._ltx_api_client.upload_file(
                        api_key=api_key,
                        file_path=str(validated_image_path),
                    )
                self._generation.update_progress("inference", 55, None, None)
                video_bytes = self._ltx_api_client.generate_audio_to_video(
                    api_key=api_key,
                    prompt=prompt,
                    audio_uri=audio_uri,
                    image_uri=image_uri,
                    model=api_model_id,
                    resolution=api_resolution,
                )
                self._generation.update_progress("downloading_output", 85, None, None)
            elif has_input_image:
                validated_image_path = validate_image_file(image_path)

                duration = self._parse_forced_numeric_field(req.duration, "INVALID_FORCED_API_DURATION")
                fps = self._parse_forced_numeric_field(req.fps, "INVALID_FORCED_API_FPS")
                if fps not in FORCED_API_ALLOWED_FPS:
                    raise HTTPError(400, "INVALID_FORCED_API_FPS")
                if duration not in _get_allowed_durations(api_model_id, resolution_label, fps):
                    raise HTTPError(400, "INVALID_FORCED_API_DURATION")

                generate_audio = self._parse_audio_flag(req.audio)
                self._generation.update_progress("uploading_image", 20, None, None)
                image_uri = self._ltx_api_client.upload_file(
                    api_key=api_key,
                    file_path=str(validated_image_path),
                )
                self._generation.update_progress("inference", 55, None, None)
                video_bytes = self._ltx_api_client.generate_image_to_video(
                    api_key=api_key,
                    prompt=prompt,
                    image_uri=image_uri,
                    model=api_model_id,
                    resolution=api_resolution,
                    duration=float(duration),
                    fps=float(fps),
                    generate_audio=generate_audio,
                    camera_motion=req.cameraMotion,
                )
                self._generation.update_progress("downloading_output", 85, None, None)
            else:
                duration = self._parse_forced_numeric_field(req.duration, "INVALID_FORCED_API_DURATION")
                fps = self._parse_forced_numeric_field(req.fps, "INVALID_FORCED_API_FPS")
                if fps not in FORCED_API_ALLOWED_FPS:
                    raise HTTPError(400, "INVALID_FORCED_API_FPS")
                if duration not in _get_allowed_durations(api_model_id, resolution_label, fps):
                    raise HTTPError(400, "INVALID_FORCED_API_DURATION")

                generate_audio = self._parse_audio_flag(req.audio)
                self._generation.update_progress("inference", 55, None, None)
                video_bytes = self._ltx_api_client.generate_text_to_video(
                    api_key=api_key,
                    prompt=prompt,
                    model=api_model_id,
                    resolution=api_resolution,
                    duration=float(duration),
                    fps=float(fps),
                    generate_audio=generate_audio,
                    camera_motion=req.cameraMotion,
                )
                self._generation.update_progress("downloading_output", 85, None, None)

            if self._generation.is_generation_cancelled():
                raise RuntimeError("Generation was cancelled")

            output_path = self._write_forced_api_video(video_bytes)
            if self._generation.is_generation_cancelled():
                output_path.unlink(missing_ok=True)
                raise RuntimeError("Generation was cancelled")

            self._generation.update_progress("complete", 100, None, None)
            self._generation.complete_generation(str(output_path))
            return GenerateVideoResponse(status="complete", video_path=str(output_path))
        except HTTPError as e:
            self._generation.fail_generation(e.detail)
            raise
        except Exception as e:
            self._generation.fail_generation(str(e))
            if "cancelled" in str(e).lower():
                logger.info("Generation cancelled by user")
                return GenerateVideoResponse(status="cancelled")
            raise HTTPError(500, str(e)) from e

    def _write_forced_api_video(self, video_bytes: bytes) -> Path:
        output_path = self._make_output_path()
        output_path.write_bytes(video_bytes)
        return output_path

    @staticmethod
    def _parse_forced_numeric_field(raw_value: str, error_detail: str) -> int:
        try:
            return int(float(raw_value))
        except (TypeError, ValueError):
            raise HTTPError(400, error_detail) from None

    @staticmethod
    def _parse_audio_flag(audio_value: str | bool) -> bool:
        if isinstance(audio_value, bool):
            return audio_value
        normalized = audio_value.strip().lower()
        return normalized in {"1", "true", "yes", "on"}
