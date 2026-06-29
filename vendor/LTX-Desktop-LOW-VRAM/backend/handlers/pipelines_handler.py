"""Pipeline lifecycle and warmup handler."""

from __future__ import annotations

import json
import logging
import torch
from threading import RLock
from typing import TYPE_CHECKING

from handlers.base import StateHandlerBase
from handlers.text_handler import TextHandler
from runtime_config.model_download_specs import resolve_model_path
from services.fp8_export_service import fp8_transformer_path
from services.interfaces import (
    A2VPipeline,
    DepthProcessorPipeline,
    DevVideoPipeline,
    FastVideoPipeline,
    ImageGenerationPipeline,
    GpuCleaner,
    IcLoraPipeline,
    PoseProcessorPipeline,
    RetakePipeline,
    VideoPipelineModelType,
)
from services.services_utils import device_supports_fp8, get_device_type
from state.app_state_types import (
    A2VPipelineState,
    AppState,
    CpuSlot,
    GenerationRunning,
    GpuSlot,
    ICLoraState,
    RetakePipelineState,
    VideoPipelineState,
    VideoPipelineWarmth,
)

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig

logger = logging.getLogger(__name__)


class PipelinesHandler(StateHandlerBase):
    def __init__(
        self,
        state: AppState,
        lock: RLock,
        text_handler: TextHandler,
        gpu_cleaner: GpuCleaner,
        fast_video_pipeline_class: type[FastVideoPipeline],
        image_generation_pipeline_class: type[ImageGenerationPipeline],
        ic_lora_pipeline_class: type[IcLoraPipeline],
        depth_processor_pipeline_class: type[DepthProcessorPipeline],
        pose_processor_pipeline_class: type[PoseProcessorPipeline],
        a2v_pipeline_class: type[A2VPipeline],
        retake_pipeline_class: type[RetakePipeline],
        config: RuntimeConfig,
    ) -> None:
        super().__init__(state, lock, config)
        self._text_handler = text_handler
        self._gpu_cleaner = gpu_cleaner
        self._fast_video_pipeline_class = fast_video_pipeline_class
        self._image_generation_pipeline_class = image_generation_pipeline_class
        self._ic_lora_pipeline_class = ic_lora_pipeline_class
        self._depth_processor_pipeline_class = depth_processor_pipeline_class
        self._pose_processor_pipeline_class = pose_processor_pipeline_class
        self._a2v_pipeline_class = a2v_pipeline_class
        self._retake_pipeline_class = retake_pipeline_class
        self._runtime_device = get_device_type(self.config.device)

    def _ensure_no_running_generation(self) -> None:
        match self.state.gpu_slot:
            case GpuSlot(generation=GenerationRunning()):
                raise RuntimeError("Generation already running; cannot swap pipelines")
            case _:
                return

    def _pipeline_config_key(self, model_type: VideoPipelineModelType) -> tuple[object, ...]:
        s = self.state.app_settings
        return (
            model_type,
            s.block_swap_blocks_on_gpu,
            s.attention_tile_size,
            s.use_fp8_transformer,
            s.gguf_transformer_path,
            s.gguf_per_layer_quant,
            s.vae_spatial_tile_size,
            s.vae_temporal_tile_size,
            s.use_multi_gpu,
            s.transformer_device,
            s.civitai_loras,
        )

    def _pipeline_matches_model_type(self, model_type: VideoPipelineModelType) -> bool:
        match self.state.gpu_slot:
            case GpuSlot(active_pipeline=VideoPipelineState(pipeline=pipeline, config_key=key)):
                return pipeline.pipeline_kind == model_type and key == self._pipeline_config_key(model_type)
            case _:
                return False

    def _assert_invariants(self) -> None:
        gpu_is_zit = False
        match self.state.gpu_slot:
            case GpuSlot(active_pipeline=VideoPipelineState() | ICLoraState() | A2VPipelineState() | RetakePipelineState()):
                gpu_is_zit = False
            case GpuSlot():
                gpu_is_zit = True
            case _:
                gpu_is_zit = False

        if gpu_is_zit and self.state.cpu_slot is not None:
            raise RuntimeError("Invariant violation: ZIT cannot be in both GPU and CPU slots")

    def _install_text_patches_if_needed(self) -> None:
        te = self.state.text_encoder
        if te is None:
            return
        te.service.install_patches(lambda: self.state)

    def _compile_if_enabled(self, state: VideoPipelineState) -> VideoPipelineState:
        if not self.state.app_settings.use_torch_compile:
            return state
        if state.is_compiled:
            return state
        if self._runtime_device == "mps":
            logger.info("Skipping torch.compile() for %s - not supported on MPS", state.pipeline.pipeline_kind)
            return state

        try:
            state.pipeline.compile_transformer()
            state.is_compiled = True
        except Exception as exc:
            logger.warning("Failed to compile transformer: %s", exc, exc_info=True)
        return state

    def _resolve_transformer_device(self, settings) -> torch.device:
        """Resolve effective transformer device from settings."""
        transformer_device = self.config.device
        if settings.use_multi_gpu:
            if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
                transformer_device = torch.device("cuda:0")
                logger.info("Multi-GPU: video/transformer on cuda:0, text encoder on cuda:1")
            else:
                logger.warning("use_multi_gpu=True but <2 GPUs detected — single GPU mode")
        if settings.transformer_device:
            try:
                transformer_device = torch.device(settings.transformer_device)
            except Exception:
                pass
        return transformer_device

    def _parse_lora_entries(self, settings) -> "list":
        from services.lora_service import LoraEntry
        lora_entries: list[LoraEntry] = []
        if settings.civitai_loras:
            try:
                raw = json.loads(settings.civitai_loras)
                lora_entries = [
                    LoraEntry(**item)
                    for item in raw
                    if isinstance(item, dict) and item.get("path")
                ]
            except Exception:
                logger.warning("Failed to parse civitai_loras setting — no LoRAs applied")
        return lora_entries

    def _create_video_pipeline(self, model_type: VideoPipelineModelType) -> VideoPipelineState:
        gemma_root = self._text_handler.resolve_gemma_root()

        upsampler_path = str(resolve_model_path(self.models_dir, self.config.model_download_specs, "upsampler"))

        settings = self.state.app_settings
        transformer_device = self._resolve_transformer_device(settings)
        lora_entries = self._parse_lora_entries(settings)

        if model_type == "dev":
            # Use the dev checkpoint (full model, not distilled) for the two-stage pipeline.
            # Falls back to the distilled checkpoint if dev_checkpoint is not present so
            # existing setups without the dev file don't hard-fail (quality will be lower).
            dev_ckpt = resolve_model_path(self.models_dir, self.config.model_download_specs, "dev_checkpoint")
            if dev_ckpt.exists():
                checkpoint_path = str(dev_ckpt)
                logger.info("Dev pipeline: using dev checkpoint %s", dev_ckpt.name)
            else:
                checkpoint_path = str(resolve_model_path(self.models_dir, self.config.model_download_specs, "checkpoint"))
                logger.warning(
                    "Dev pipeline: dev checkpoint (%s) not found — falling back to distilled checkpoint."
                    " Download %s for correct two-stage quality.",
                    dev_ckpt.name, dev_ckpt.name,
                )
            pipeline = self._create_dev_pipeline(
                checkpoint_path=checkpoint_path,
                upsampler_path=upsampler_path,
                gemma_root=gemma_root,
                settings=settings,
                transformer_device=transformer_device,
                lora_entries=lora_entries,
            )
        else:
            checkpoint_path = str(resolve_model_path(self.models_dir, self.config.model_download_specs, "checkpoint"))
            pipeline = self._create_fast_pipeline(
                checkpoint_path=checkpoint_path,
                upsampler_path=upsampler_path,
                gemma_root=gemma_root,
                settings=settings,
                transformer_device=transformer_device,
                lora_entries=lora_entries,
            )

        state = VideoPipelineState(
            pipeline=pipeline,
            warmth=VideoPipelineWarmth.COLD,
            is_compiled=False,
            config_key=self._pipeline_config_key(model_type),
        )
        return self._compile_if_enabled(state)

    def _create_fast_pipeline(
        self,
        checkpoint_path: str,
        upsampler_path: str,
        gemma_root: str | None,
        settings,
        transformer_device: torch.device,
        lora_entries: list,
    ):
        # Use pre-quantized FP8 file if it exists (avoids on-the-fly downcast on every load).
        fp8_pre_quantized = ""
        _fp8_path = fp8_transformer_path(self.models_dir)
        if _fp8_path.exists():
            fp8_pre_quantized = str(_fp8_path)
            logger.info("Pre-quantized FP8 transformer found: %s", fp8_pre_quantized)

        return self._fast_video_pipeline_class.create(
            checkpoint_path,
            gemma_root,
            upsampler_path,
            self.config.device,
            transformer_device=transformer_device,
            block_swap_blocks_on_gpu=settings.block_swap_blocks_on_gpu,
            attention_tile_size=settings.attention_tile_size,
            use_fp8_transformer=settings.use_fp8_transformer,
            gguf_transformer_path=settings.gguf_transformer_path,
            gguf_per_layer_quant=getattr(settings, 'gguf_per_layer_quant', True),
            vae_spatial_tile_size=settings.vae_spatial_tile_size,
            vae_temporal_tile_size=settings.vae_temporal_tile_size,
            pre_quantized_transformer_path=fp8_pre_quantized,
            loras=lora_entries or None,
        )

    def _create_dev_pipeline(
        self,
        checkpoint_path: str,
        upsampler_path: str,
        gemma_root: str | None,
        settings,
        transformer_device: torch.device,
        lora_entries: list,
    ):
        from services.dev_video_pipeline.ltx_dev_video_pipeline import LTXDevVideoPipeline

        distilled_lora_path = resolve_model_path(
            self.models_dir, self.config.model_download_specs, "distilled_lora"
        )
        if not distilled_lora_path.exists():
            raise RuntimeError(
                f"Dev pipeline requires the distilled LoRA weight "
                f"({distilled_lora_path.name}). "
                f"Please download it via the Model Status menu."
            )

        gguf_path = settings.gguf_transformer_path or ""
        gguf_per_layer = getattr(settings, 'gguf_per_layer_quant', True)
        if gguf_path:
            logger.info(
                "Dev pipeline: GGUF transformer from %s (per_layer_quant=%s)",
                gguf_path, gguf_per_layer,
            )

        logger.info("Creating dev (two-stage) pipeline from checkpoint: %s", checkpoint_path)
        return LTXDevVideoPipeline.create(
            checkpoint_path=checkpoint_path,
            distilled_lora_path=str(distilled_lora_path),
            gemma_root=gemma_root,
            upsampler_path=upsampler_path,
            device=self.config.device,
            transformer_device=transformer_device,
            block_swap_blocks_on_gpu=settings.block_swap_blocks_on_gpu,
            attention_tile_size=settings.attention_tile_size,
            use_fp8_transformer=settings.use_fp8_transformer,
            vae_spatial_tile_size=settings.vae_spatial_tile_size,
            vae_temporal_tile_size=settings.vae_temporal_tile_size,
            loras=lora_entries or None,
            gguf_transformer_path=gguf_path,
            gguf_per_layer_quant=gguf_per_layer,
        )

    def unload_gpu_pipeline(self) -> None:
        with self._lock:
            self._ensure_no_running_generation()
            self.state.gpu_slot = None
            self._assert_invariants()
        self._gpu_cleaner.cleanup()

    def park_zit_on_cpu(self) -> None:
        zit: ImageGenerationPipeline | None = None

        with self._lock:
            if self.state.gpu_slot is None:
                return

            active = self.state.gpu_slot.active_pipeline
            if isinstance(active, (VideoPipelineState, ICLoraState, A2VPipelineState, RetakePipelineState)):
                return

            generation = self.state.gpu_slot.generation
            if isinstance(generation, GenerationRunning):
                raise RuntimeError("Cannot park ZIT while generation is running")

            zit = active
            self.state.gpu_slot = None

        assert zit is not None
        zit.to("cpu")
        self._gpu_cleaner.cleanup()

        with self._lock:
            self.state.cpu_slot = CpuSlot(active_pipeline=zit)
            self._assert_invariants()

    def load_zit_to_gpu(self) -> ImageGenerationPipeline:
        with self._lock:
            if self.state.gpu_slot is not None:
                active = self.state.gpu_slot.active_pipeline
                if not isinstance(active, (VideoPipelineState, ICLoraState, A2VPipelineState, RetakePipelineState)):
                    return active
                self._ensure_no_running_generation()

        zit_service: ImageGenerationPipeline | None = None

        with self._lock:
            match self.state.cpu_slot:
                case CpuSlot(active_pipeline=stored):
                    zit_service = stored
                    self.state.cpu_slot = None
                case _:
                    zit_service = None

        if zit_service is None:
            zit_path = resolve_model_path(self.models_dir, self.config.model_download_specs,"zit")
            if not (zit_path.exists() and any(zit_path.iterdir())):
                raise RuntimeError("Z-Image-Turbo model not downloaded. Please download the AI models first using the Model Status menu.")
            zit_service = self._image_generation_pipeline_class.create(str(zit_path), self._runtime_device)
        else:
            zit_service.to(self._runtime_device)

        self._gpu_cleaner.cleanup()

        with self._lock:
            self.state.gpu_slot = GpuSlot(active_pipeline=zit_service, generation=None)
            self._assert_invariants()

        return zit_service

    def preload_zit_to_cpu(self) -> ImageGenerationPipeline:
        with self._lock:
            match self.state.cpu_slot:
                case CpuSlot(active_pipeline=existing):
                    return existing
                case _:
                    pass

        zit_path = resolve_model_path(self.models_dir, self.config.model_download_specs,"zit")
        if not (zit_path.exists() and any(zit_path.iterdir())):
            raise RuntimeError("Z-Image-Turbo model not downloaded. Please download the AI models first using the Model Status menu.")

        zit_service = self._image_generation_pipeline_class.create(str(zit_path), None)
        with self._lock:
            if self.state.cpu_slot is None:
                self.state.cpu_slot = CpuSlot(active_pipeline=zit_service)
                self._assert_invariants()
                return zit_service
            return self.state.cpu_slot.active_pipeline

    def _evict_gpu_pipeline_for_swap(self) -> None:
        should_park_zit = False
        should_cleanup = False

        with self._lock:
            self._ensure_no_running_generation()
            if self.state.gpu_slot is None:
                return

            active = self.state.gpu_slot.active_pipeline
            if isinstance(active, (VideoPipelineState, ICLoraState, A2VPipelineState, RetakePipelineState)):
                self.state.gpu_slot = None
                self._assert_invariants()
                should_cleanup = True
            else:
                should_park_zit = True

        if should_park_zit:
            self.park_zit_on_cpu()
        elif should_cleanup:
            # Drop the local reference before cleanup so gc.collect() inside
            # the cleaner can actually free the old pipeline's CPU tensors
            # (block-swap buffers, VAE weights, etc.) rather than finding them
            # still reachable via this stack frame.
            del active
            self._gpu_cleaner.cleanup()

    def load_gpu_pipeline(self, model_type: VideoPipelineModelType, should_warm: bool = False) -> VideoPipelineState:
        self._install_text_patches_if_needed()

        state: VideoPipelineState | None = None
        with self._lock:
            if self._pipeline_matches_model_type(model_type):
                match self.state.gpu_slot:
                    case GpuSlot(active_pipeline=VideoPipelineState() as existing_state):
                        state = existing_state
                    case _:
                        pass

        if state is None:
            self._evict_gpu_pipeline_for_swap()
            state = self._create_video_pipeline(model_type)
            with self._lock:
                self.state.gpu_slot = GpuSlot(active_pipeline=state, generation=None)
                self._assert_invariants()

        if should_warm and state.warmth == VideoPipelineWarmth.COLD:
            with self._lock:
                state.warmth = VideoPipelineWarmth.WARMING

            self.warmup_pipeline(model_type)
            with self._lock:
                if state.warmth == VideoPipelineWarmth.WARMING:
                    state.warmth = VideoPipelineWarmth.WARM

        return state

    def load_ic_lora(
        self,
        lora_path: str,
        depth_model_path: str,
    ) -> ICLoraState:
        self._install_text_patches_if_needed()

        with self._lock:
            match self.state.gpu_slot:
                case GpuSlot(
                    active_pipeline=ICLoraState(
                        lora_path=current_lora_path,
                        depth_model_path=current_depth_model_path,
                    ) as state
                ) if (
                    current_lora_path == lora_path
                    and current_depth_model_path == depth_model_path
                ):
                    return state
                case _:
                    pass

        self._evict_gpu_pipeline_for_swap()

        pipeline = self._ic_lora_pipeline_class.create(
            str(resolve_model_path(self.models_dir, self.config.model_download_specs,"checkpoint")),
            self._text_handler.resolve_gemma_root(),
            str(resolve_model_path(self.models_dir, self.config.model_download_specs,"upsampler")),
            lora_path,
            self.config.device,
        )
        depth_pipeline = self._depth_processor_pipeline_class.create(depth_model_path, self.config.device)
        state = ICLoraState(
            pipeline=pipeline,
            lora_path=lora_path,
            depth_pipeline=depth_pipeline,
            depth_model_path=depth_model_path,
        )

        with self._lock:
            self.state.gpu_slot = GpuSlot(active_pipeline=state, generation=None)
            self._assert_invariants()
        return state

    def load_a2v_pipeline(self) -> A2VPipelineState:
        self._install_text_patches_if_needed()

        with self._lock:
            match self.state.gpu_slot:
                case GpuSlot(active_pipeline=A2VPipelineState() as state):
                    return state
                case _:
                    pass

        self._evict_gpu_pipeline_for_swap()

        pipeline = self._a2v_pipeline_class.create(
            str(resolve_model_path(self.models_dir, self.config.model_download_specs,"checkpoint")),
            self._text_handler.resolve_gemma_root(),
            str(resolve_model_path(self.models_dir, self.config.model_download_specs,"upsampler")),
            self.config.device,
        )
        state = A2VPipelineState(pipeline=pipeline)

        with self._lock:
            self.state.gpu_slot = GpuSlot(active_pipeline=state, generation=None)
            self._assert_invariants()
        return state

    def load_retake_pipeline(self, *, distilled: bool = True) -> RetakePipelineState:
        self._install_text_patches_if_needed()

        quantized = device_supports_fp8(self.config.device)

        with self._lock:
            match self.state.gpu_slot:
                case GpuSlot(
                    active_pipeline=RetakePipelineState(distilled=current_distilled, quantized=current_quantized) as state
                ) if current_distilled == distilled and current_quantized == quantized:
                    return state
                case _:
                    pass

        self._evict_gpu_pipeline_for_swap()

        from ltx_core.quantization import QuantizationPolicy

        quantization = QuantizationPolicy.fp8_cast() if quantized else None
        pipeline = self._retake_pipeline_class.create(
            checkpoint_path=str(resolve_model_path(self.models_dir, self.config.model_download_specs,"checkpoint")),
            gemma_root=self._text_handler.resolve_gemma_root(),
            device=self.config.device,
            loras=[],
            quantization=quantization,
        )
        state = RetakePipelineState(pipeline=pipeline, distilled=distilled, quantized=quantized)

        with self._lock:
            self.state.gpu_slot = GpuSlot(active_pipeline=state, generation=None)
            self._assert_invariants()
        return state

    def warmup_pipeline(self, model_type: VideoPipelineModelType) -> None:
        state = self.load_gpu_pipeline(model_type, should_warm=False)
        warmup_path = self.config.outputs_dir / f"_warmup_{model_type}.mp4"
        state.pipeline.warmup(output_path=str(warmup_path))