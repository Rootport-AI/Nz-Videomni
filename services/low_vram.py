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


@dataclass
class LowVramSettings:
    low_vram_mode: bool
    low_vram_profile: str
    fp8_transformer: bool
    cpu_offload_text_encoder: bool
    vae_tiling: bool
    attention_tiling: bool
    block_swap: bool

    def as_dict(self) -> dict:
        return asdict(self)

    def status_block(self, *, low_vram_disabled_required: bool) -> dict:
        """The ``vram_optimization`` block for GET /status (spec 7.4)."""
        data = self.as_dict()
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
