"""Configuration loading for LTX-AviUtl2-Bridge.

Reads ``config.yaml`` into typed Pydantic models. This is the single source of
truth for server, model, VRAM, upload, limits and output settings (spec ch.11).

CLI overrides (``--listen``, ``--port``, ``--api-key``, ``--allow-all-cors``)
are applied in ``main.py`` on top of the loaded ``ServerConfig``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 18620
    allow_all_cors: bool = False
    api_key: str | None = None
    log_dir: str = "./logs"


class ModelConfig(BaseModel):
    checkpoint_dir: str = "./models"
    checkpoint_name: str = "ltx-2.3-22b-distilled"
    text_encoder: str = "google/gemma-2-2b-it"
    pipeline_type: str = "distilled"
    auto_load_on_generate: bool = True
    reload_interval: int = 0


class VramConfig(BaseModel):
    low_vram_mode: bool = True
    low_vram_profile: str = "16gb_safe"
    fp8_transformer: bool = True
    cpu_offload_text_encoder: bool = True
    vae_tiling: bool = True
    attention_tiling: bool = False
    attention_tile_size: int | None = None
    block_swap: bool = False
    block_swap_blocks_on_gpu: int | None = None
    allow_disable_low_vram: bool = True


class CropOutputPreset(BaseModel):
    width: int
    height: int


class GenerationPreset(BaseModel):
    width: int
    height: int
    crop_output: CropOutputPreset | None = None
    num_frames: int


class GenerationDefaults(BaseModel):
    width: int = 512
    height: int = 288
    crop_output: CropOutputPreset | None = None
    num_frames: int = 49
    frame_rate: float = 24.0
    num_inference_steps: int = 8
    guidance_scale: float = 1.0
    seed: int = -1
    pipeline: str = "distilled"
    conditioning_images: list[Any] = Field(default_factory=list)


class UploadConfig(BaseModel):
    dir: str = "./uploads"
    max_image_size_mb: int = 20
    allowed_image_extensions: list[str] = Field(
        default_factory=lambda: [".png", ".jpg", ".jpeg", ".webp"]
    )
    normalize_to_png: bool = True


class LimitsConfig(BaseModel):
    max_width: int = 1920
    max_height: int = 1088
    max_num_frames: int = 257
    max_conditioning_images_phase1: int = 1
    phase1_max_concurrent_jobs: int = 1
    low_vram_disabled_required: bool = False


class OutputConfig(BaseModel):
    dir: str = "./outputs"
    format: str = "mp4"
    save_metadata_json: bool = True
    keep_raw_frames: bool = False


class AppConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    vram: VramConfig = Field(default_factory=VramConfig)
    generation_presets: dict[str, GenerationPreset] = Field(default_factory=dict)
    generation_defaults: GenerationDefaults = Field(default_factory=GenerationDefaults)
    upload: UploadConfig = Field(default_factory=UploadConfig)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    # ----- convenience path helpers (always absolute, project-rooted) -----

    def _abs(self, value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()

    @property
    def output_dir(self) -> Path:
        return self._abs(self.output.dir)

    @property
    def upload_dir(self) -> Path:
        return self._abs(self.upload.dir)

    @property
    def log_dir(self) -> Path:
        return self._abs(self.server.log_dir)

    @property
    def checkpoint_dir(self) -> Path:
        return self._abs(self.model.checkpoint_dir)


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load ``config.yaml`` into an :class:`AppConfig`.

    Missing file falls back to model defaults so the server can still boot.
    """
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        return AppConfig()
    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return AppConfig.model_validate(raw)
