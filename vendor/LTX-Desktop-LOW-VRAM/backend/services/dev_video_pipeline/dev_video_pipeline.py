"""Dev (two-stage) video pipeline protocol definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal, Protocol

from api_types import ImageConditioningInput

if TYPE_CHECKING:
    import torch
    from services.lora_service import LoraEntry


class DevVideoPipeline(Protocol):
    pipeline_kind: ClassVar[Literal["dev"]]

    @staticmethod
    def create(
        checkpoint_path: str,
        distilled_lora_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        device: "torch.device",
        transformer_device: "torch.device | None" = None,
        block_swap_blocks_on_gpu: int = 0,
        attention_tile_size: int = 0,
        use_fp8_transformer: bool = False,
        vae_spatial_tile_size: int = 0,
        vae_temporal_tile_size: int = 0,
        loras: "list[LoraEntry] | None" = None,
        gguf_transformer_path: str = "",
        gguf_per_layer_quant: bool = True,
    ) -> "DevVideoPipeline":
        ...

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
        ...

    def warmup(self, output_path: str) -> None:
        ...

    def compile_transformer(self) -> None:
        ...
