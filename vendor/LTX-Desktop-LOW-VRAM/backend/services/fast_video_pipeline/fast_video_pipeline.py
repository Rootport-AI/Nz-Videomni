"""Fast video pipeline protocol definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal, Protocol

from api_types import ImageConditioningInput

if TYPE_CHECKING:
    import torch
    from services.lora_service import LoraEntry


class FastVideoPipeline(Protocol):
    pipeline_kind: ClassVar[Literal["fast"]]

    @staticmethod
    def create(
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        device: torch.device,
        transformer_device: torch.device | None = None,
        block_swap_blocks_on_gpu: int = 0,
        attention_tile_size: int = 0,
        use_fp8_transformer: bool = False,
        gguf_transformer_path: str = "",
        loras: "list[LoraEntry] | None" = None,
    ) -> "FastVideoPipeline":
        ...

    def generate(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        output_path: str,
        num_steps: int = 8,
        stg_scale: float = 0.0,
        stg_block_index: int = 19,
    ) -> None:
        ...

    def warmup(self, output_path: str) -> None:
        ...

    def compile_transformer(self) -> None:
        ...
