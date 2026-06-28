"""Low-VRAM baseline settings (spec 0.2 / ch.13).

Phase 1 does NOT port fragile custom patches. This service simply collects the
config's low-VRAM flags into a plain object that ``ltx_runner`` reads when
building the official pipeline, and provides a safe memory-cleanup helper.

The advanced optimizations (attention tiling, block swap, custom VAE tiling)
are reported but stay disabled in Phase 1.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

from config import AppConfig


# Keys that form the frozen GET /status ``vram_optimization`` contract (spec
# 7.4). New internal fields (block_swap_blocks_on_gpu, vae_*_tile_size) must NOT
# leak into this block, so status_block() filters to exactly these.
_STATUS_KEYS = (
    "low_vram_mode",
    "low_vram_profile",
    "fp8_transformer",
    "cpu_offload_text_encoder",
    "vae_tiling",
    "attention_tiling",
    "block_swap",
)


@dataclass
class LowVramSettings:
    low_vram_mode: bool
    low_vram_profile: str
    fp8_transformer: bool
    cpu_offload_text_encoder: bool
    vae_tiling: bool
    attention_tiling: bool
    block_swap: bool
    # Internal knobs for the real GGUF engine (NOT part of status/metadata
    # contracts). Read directly by services.ltx_runner._RealBackend.
    block_swap_blocks_on_gpu: int | None = None
    vae_spatial_tile_size: int = 0
    vae_temporal_tile_size: int = 0

    def as_dict(self) -> dict:
        return asdict(self)

    def status_block(self, *, low_vram_disabled_required: bool) -> dict:
        """The ``vram_optimization`` block for GET /status (spec 7.4)."""
        data = {k: getattr(self, k) for k in _STATUS_KEYS}
        data["low_vram_disabled_required"] = low_vram_disabled_required
        return data

    def metadata_block(self, *, peak_vram_mb: int | None) -> dict:
        """The ``vram_optimization`` block for metadata.json (spec 10.2)."""
        return {
            "low_vram_mode": self.low_vram_mode,
            "low_vram_profile": self.low_vram_profile,
            "fp8_transformer": self.fp8_transformer,
            "cpu_offload_text_encoder": self.cpu_offload_text_encoder,
            "vae_tiling": self.vae_tiling,
            "peak_vram_mb": peak_vram_mb,
        }


def build_low_vram_settings(config: AppConfig) -> LowVramSettings:
    v = config.vram
    return LowVramSettings(
        low_vram_mode=v.low_vram_mode,
        low_vram_profile=v.low_vram_profile,
        fp8_transformer=v.fp8_transformer,
        cpu_offload_text_encoder=v.cpu_offload_text_encoder,
        vae_tiling=v.vae_tiling,
        attention_tiling=v.attention_tiling,
        block_swap=v.block_swap,
        block_swap_blocks_on_gpu=v.block_swap_blocks_on_gpu,
        vae_spatial_tile_size=v.vae_spatial_tile_size,
        vae_temporal_tile_size=v.vae_temporal_tile_size,
    )


def safe_memory_cleanup() -> None:
    """Best-effort GPU memory release. No-op when torch/CUDA is absent."""
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
