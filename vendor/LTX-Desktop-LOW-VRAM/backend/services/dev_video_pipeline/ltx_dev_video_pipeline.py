"""LTX dev (two-stage) video pipeline wrapper."""

from __future__ import annotations

from collections.abc import Iterator
import logging
import os
from pathlib import Path
from typing import Final, cast

import torch

from api_types import ImageConditioningInput
from services.lora_service import LoraEntry
from services.ltx_pipeline_common import default_tiling_config, encode_video_output, video_chunks_number
from services.services_utils import AudioOrNone, TilingConfigType, device_supports_fp8

_log = logging.getLogger(__name__)


class LTXDevVideoPipeline:
    """Wraps TI2VidTwoStagesPipeline (distilled + refinement) for high-quality generation."""

    pipeline_kind: Final = "dev"

    @staticmethod
    def create(
        checkpoint_path: str,
        distilled_lora_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        device: torch.device,
        transformer_device: torch.device | None = None,
        block_swap_blocks_on_gpu: int = 0,
        attention_tile_size: int = 0,
        use_fp8_transformer: bool = False,
        vae_spatial_tile_size: int = 0,
        vae_temporal_tile_size: int = 0,
        loras: list[LoraEntry] | None = None,
        gguf_transformer_path: str = "",
        gguf_per_layer_quant: bool = True,
    ) -> "LTXDevVideoPipeline":
        return LTXDevVideoPipeline(
            checkpoint_path=checkpoint_path,
            distilled_lora_path=distilled_lora_path,
            gemma_root=gemma_root,
            upsampler_path=upsampler_path,
            device=device,
            transformer_device=transformer_device,
            block_swap_blocks_on_gpu=block_swap_blocks_on_gpu,
            attention_tile_size=attention_tile_size,
            use_fp8_transformer=use_fp8_transformer,
            vae_spatial_tile_size=vae_spatial_tile_size,
            vae_temporal_tile_size=vae_temporal_tile_size,
            loras=loras,
            gguf_transformer_path=gguf_transformer_path,
            gguf_per_layer_quant=gguf_per_layer_quant,
        )

    def __init__(
        self,
        checkpoint_path: str,
        distilled_lora_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        device: torch.device,
        transformer_device: torch.device | None = None,
        block_swap_blocks_on_gpu: int = 0,
        attention_tile_size: int = 0,
        use_fp8_transformer: bool = False,
        vae_spatial_tile_size: int = 0,
        vae_temporal_tile_size: int = 0,
        loras: list[LoraEntry] | None = None,
        gguf_transformer_path: str = "",
        gguf_per_layer_quant: bool = True,
    ) -> None:
        from ltx_core.quantization import QuantizationPolicy
        from ltx_pipelines import TI2VidTwoStagesPipeline

        self._transformer_device = transformer_device or device
        self._block_swap_blocks_on_gpu = block_swap_blocks_on_gpu
        self._attention_tile_size = attention_tile_size
        self._vae_spatial_tile_size = vae_spatial_tile_size
        self._vae_temporal_tile_size = vae_temporal_tile_size

        # If the checkpoint file is already pre-quantized FP8 (filename contains _fp8)
        # do NOT apply fp8_cast on top — the weights are already in FP8 format.
        # For BF16 dev checkpoints, fp8_cast downcasts at load time to save VRAM.
        checkpoint_already_fp8 = "_fp8" in Path(checkpoint_path).stem
        if checkpoint_already_fp8:
            quantization = None
            _log.info("Dev pipeline: pre-quantized FP8 checkpoint detected — skipping fp8_cast")
        else:
            use_fp8 = use_fp8_transformer or device_supports_fp8(device)
            quantization = QuantizationPolicy.fp8_cast() if use_fp8 else None

        self.pipeline = TI2VidTwoStagesPipeline(
            checkpoint_path=checkpoint_path,
            distilled_lora=distilled_lora_path,
            spatial_upsampler_path=upsampler_path,
            gemma_root=cast(str, gemma_root),
            loras=[],
            device=device,
            quantization=quantization,
        )

        # GGUF must be installed first — it replaces transformer_builder on both
        # stage ledgers before any other service wraps the transformer() callable.
        if gguf_transformer_path:
            self._install_gguf(gguf_transformer_path, per_layer_quant=gguf_per_layer_quant)

        if block_swap_blocks_on_gpu > 0:
            self._install_block_swap(block_swap_blocks_on_gpu)

        if attention_tile_size > 0:
            self._install_attention_tiling(attention_tile_size)

        if loras:
            self._install_loras(loras)

    def _install_gguf(self, gguf_path: str, per_layer_quant: bool = True) -> None:
        """Install GGUF transformer on both stage ledgers.

        TI2VidTwoStagesPipeline uses stage_1_model_ledger for the main denoising pass
        and stage_2_model_ledger (derived from stage 1 via with_loras) for the
        refinement pass.  Both share the same underlying 22B checkpoint so both need
        the same loader.

        per_layer_quant=True (default): weights stay compressed in VRAM, dequantised
        one layer at a time during the forward pass — significantly lower VRAM.
        per_layer_quant=False: dequantise everything at load time → full BF16 in VRAM.
        """
        try:
            if per_layer_quant:
                from services.gguf_quant_service import GGUFQuantLoaderService
                svc_cls = GGUFQuantLoaderService
                mode = "per-layer quant (VRAM-efficient)"
            else:
                from services.gguf_loader_service import GGUFLoaderService
                svc_cls = GGUFLoaderService
                mode = "load-time dequant (full BF16 VRAM)"

            service1 = svc_cls(gguf_path=gguf_path)
            service1.install(self.pipeline.stage_1_model_ledger)
            service2 = svc_cls(gguf_path=gguf_path)
            service2.install(self.pipeline.stage_2_model_ledger)
            self._gguf_service = (service1, service2)
            _log.info("GGUF %s installed on dev pipeline (both stages): %s", mode, gguf_path)
        except Exception as exc:
            _log.warning("GGUF install failed on dev pipeline (%s)", exc)

    def _install_block_swap(self, blocks_on_gpu: int) -> None:
        try:
            ledger = self.pipeline.stage_1_model_ledger
            from services.block_swap_service import BlockSwapService
            service = BlockSwapService(
                blocks_on_gpu=blocks_on_gpu,
                device=self._transformer_device,
            )
            original_transformer = ledger.transformer

            def patched_transformer() -> torch.nn.Module:
                t = original_transformer()
                service.install(t)
                return t

            ledger.transformer = patched_transformer
            self._block_swap_service = service
            _log.info("BlockSwap configured: %d blocks on GPU", blocks_on_gpu)
        except Exception as exc:
            _log.warning("BlockSwap install failed (%s)", exc)

    def _install_attention_tiling(self, tile_size: int) -> None:
        try:
            from services.attention_tile_service import AttentionTileService
            service = AttentionTileService(tile_size=tile_size)
            service.install()
            self._attention_tile_service = service
            _log.info("AttentionTiling installed: tile_size=%d", tile_size)
        except Exception as exc:
            _log.warning("AttentionTiling install failed (%s)", exc)

    def _install_loras(self, entries: list[LoraEntry]) -> None:
        try:
            ledger = self.pipeline.stage_1_model_ledger
            from services.lora_service import LoraService
            service = LoraService(device=self._transformer_device)
            loaded = service.load_loras(entries)
            if not loaded:
                _log.warning("No LoRAs were successfully loaded")
                return
            _original_transformer_fn = ledger.transformer

            def _transformer_with_loras() -> torch.nn.Module:
                t = _original_transformer_fn()
                service.apply_hooks_to_transformer(t, loaded)
                return t

            ledger.transformer = _transformer_with_loras
            _log.info("LoRAs applied: %d loaded", len(loaded))
        except Exception as exc:
            _log.warning("LoRA install failed (%s)", exc)

    def _run_inference(
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        tiling_config: TilingConfigType,
        num_steps: int = 30,
        cfg_scale: float = 3.0,
        audio_cfg_scale: float = 7.0,
        stg_scale: float = 1.0,
        stg_block_index: int = 28,
        rescale_scale: float = 0.7,
        modality_scale: float = 3.0,
    ) -> tuple[torch.Tensor | Iterator[torch.Tensor], AudioOrNone]:
        from ltx_pipelines.utils.args import ImageConditioningInput as _LtxImageInput
        from ltx_core.components.guiders import MultiModalGuiderParams

        torch.cuda.empty_cache()

        video_guider_params = MultiModalGuiderParams(
            cfg_scale=cfg_scale,
            stg_scale=stg_scale,
            stg_blocks=[stg_block_index] if stg_scale > 0.0 else [],
            rescale_scale=rescale_scale,
            modality_scale=modality_scale,
            skip_step=0,
        )
        audio_guider_params = MultiModalGuiderParams(
            cfg_scale=audio_cfg_scale,
            stg_scale=0.0,
            stg_blocks=[],
            rescale_scale=0.0,
            modality_scale=1.0,
            skip_step=0,
        )

        return self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=num_steps,
            video_guider_params=video_guider_params,
            audio_guider_params=audio_guider_params,
            images=[_LtxImageInput(img.path, img.frame_idx, img.strength) for img in images],
            tiling_config=tiling_config,
        )

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        output_path: str,
        num_steps: int = 30,
        cfg_scale: float = 3.0,
        audio_cfg_scale: float = 7.0,
        stg_scale: float = 1.0,
        stg_block_index: int = 28,
        rescale_scale: float = 0.7,
        modality_scale: float = 3.0,
    ) -> None:
        tiling_config = default_tiling_config(
            spatial_tile_size=self._vae_spatial_tile_size,
            temporal_tile_size=self._vae_temporal_tile_size,
        )
        video, audio = self._run_inference(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            images=images,
            tiling_config=tiling_config,
            num_steps=num_steps,
            cfg_scale=cfg_scale,
            audio_cfg_scale=audio_cfg_scale,
            stg_scale=stg_scale,
            stg_block_index=stg_block_index,
            rescale_scale=rescale_scale,
            modality_scale=modality_scale,
        )
        chunks = video_chunks_number(num_frames, tiling_config)
        encode_video_output(video=video, audio=audio, fps=int(frame_rate), output_path=output_path, video_chunks_number_value=chunks)
        torch.cuda.synchronize()
        del video, audio
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def warmup(self, output_path: str) -> None:
        warmup_frames = 9
        tiling_config = default_tiling_config()
        try:
            video, audio = self._run_inference(
                prompt="test warmup",
                negative_prompt="",
                seed=42,
                height=256,
                width=384,
                num_frames=warmup_frames,
                frame_rate=8,
                images=[],
                tiling_config=tiling_config,
                num_steps=5,
                cfg_scale=3.0,
                audio_cfg_scale=7.0,
                stg_scale=0.0,
                stg_block_index=28,
                rescale_scale=0.0,
                modality_scale=1.0,
            )
            chunks = video_chunks_number(warmup_frames, tiling_config)
            encode_video_output(video=video, audio=audio, fps=8, output_path=output_path, video_chunks_number_value=chunks)
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def compile_transformer(self) -> None:
        ledger = self.pipeline.stage_1_model_ledger
        transformer = ledger.transformer()
        compiled = cast(
            torch.nn.Module,
            torch.compile(transformer, mode="reduce-overhead", fullgraph=False),
        )
        setattr(ledger, "transformer", lambda: compiled)
