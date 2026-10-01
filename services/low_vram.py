"""Low-VRAM baseline settings (spec §9).

This service collects the config's low-VRAM flags into a plain object that the
engine adapters under ``services/engines/`` and ``PipelineManager`` read, and
provides a safe memory-cleanup helper.

The ``attention_tiling`` and ``block_swap`` flags are reported in the status
block and are not passed to the workers; block swap itself is driven by
``block_swap_blocks_on_gpu`` and the VAE tile sizes by the ``vae_*_tile_size``
fields.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

from config import AppConfig


# Keys that form the frozen GET /status ``vram_optimization`` contract (spec
# §6.5). Internal fields (block_swap_blocks_on_gpu, vae_*_tile_size,
# te_offload_text_encoder, dit_cpu_load) must NOT leak into this block, so
# status_block() filters to exactly these.
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
    # Internal knobs for the real engine workers (NOT part of status/metadata
    # contracts). Read by ``_RealBackend`` in services/engines/ltx/adapter.py;
    # ``block_swap_blocks_on_gpu`` is also read by ``_RealBackend25`` in
    # services/engines/ltx25/adapter.py.
    block_swap_blocks_on_gpu: int | None = None
    vae_spatial_tile_size: int = 0
    vae_temporal_tile_size: int = 0
    # Internal knob (like the above): sequential per-layer CPU offload of the
    # GGUF Gemma during text-encode. Read by ``_RealBackend`` in
    # services/engines/ltx/adapter.py (emitted as LTX_TE_OFFLOAD). NOT part of
    # status/metadata contracts.
    te_offload_text_encoder: bool = False
    # Internal knob (like the above): build the DiT (transformer) on CPU and move
    # only non-block submodules to GPU, removing the load-time GPU spike measured
    # in Docs/VERIFICATION_LOG.md §12. Read by ``_RealBackend`` in
    # services/engines/ltx/adapter.py (emitted as LTX_DIT_CPU_LOAD). NOT part of
    # status/metadata contracts.
    dit_cpu_load: bool = False

    def as_dict(self) -> dict:
        return asdict(self)

    def status_block(self, *, low_vram_disabled_required: bool) -> dict:
        """The ``vram_optimization`` block for GET /status (spec §6.5)."""
        data = {k: getattr(self, k) for k in _STATUS_KEYS}
        data["low_vram_disabled_required"] = low_vram_disabled_required
        return data

    def metadata_block(self, *, peak_vram_mb: int | None) -> dict:
        """The ``vram_optimization`` block for metadata.json (spec §6.6)."""
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
        te_offload_text_encoder=v.te_offload_text_encoder,
        dit_cpu_load=v.dit_cpu_load,
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
