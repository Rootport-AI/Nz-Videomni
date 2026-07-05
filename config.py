"""Configuration loading for LTX-AviUtl2-Bridge.

Reads ``config.yaml`` into typed Pydantic models. This is the single source of
truth for server, model, VRAM, upload, limits and output settings (spec ch.11).

CLI overrides (``--listen``, ``--port``, ``--api-key``, ``--allow-all-cors``)
are applied in ``main.py`` on top of the loaded ``ServerConfig``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


class IcLoraEntry(BaseModel):
    """Dict-value form of an ``ic_loras`` registry entry (Phase C).

    Adds a ``preprocess`` kind alongside the safetensors ``path`` so one
    adapter file (e.g. the Union-Control LoRA) can be exposed under several
    logical names that each imply a different raw-video -> control-signal
    conversion in the engine worker. Plain string registry values (Phase B)
    remain valid and are equivalent to ``preprocess="none"``.
    """

    path: str
    preprocess: Literal["none", "canny", "dwpose"] = "none"


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 18620
    allow_all_cors: bool = False
    api_key: str | None = None
    log_dir: str = "./logs"


class ModelConfig(BaseModel):
    checkpoint_dir: str = "./models"
    checkpoint_name: str = "ltx-2.3-22b-distilled"
    text_encoder: str = "google/gemma-3-12b-it-qat-q4_0-unquantized"
    pipeline_type: str = "distilled"
    auto_load_on_generate: bool = True
    reload_interval: int = 0

    # Step 7 (real LTX) runtime paths. Populated by scripts/install_ltx.ps1 and
    # consumed only by services/ltx_runner.py. None until the model is installed.
    ltx_repo_dir: str = "./vendor/LTX-2"  # reference only (upstream LTX-2 clone).
    # reference-only. The 43GB monolith it named was physically deleted in
    # Stage 3; the GGUF + component-file path never opens it (proven by a rename
    # test: load still passed). Kept because it is still passed in the worker
    # payload for the DistilledPipeline signature (path stored, not read).
    checkpoint_path: str | None = None
    spatial_upsampler_path: str | None = None
    # tokenizer-only dir (~40MB). The GGUF Gemma path needs only the tokenizer/
    # processor files: DistilledPipeline is built with gemma_root=None so the wheel's
    # weight glob (model*.safetensors) is bypassed and the engine
    # rebuilds a shard-less text-encoder builder, loading tokenizer/processor
    # module_ops from this dir (globs only tokenizer.model + preprocessor_config.json).
    # The directory must still exist (gated below) -- it is the tokenizer source, not
    # a loaded weight checkpoint.
    gemma_root: str | None = None
    backend: str = "auto"  # "auto" | "mock" | "real"

    # Phase 5 (real GGUF engine) runtime paths. Consumed only by the
    # subprocess-worker real backend in services/ltx_runner.py. Defaults are the
    # spike-proven 16GB recipe (Q4_K_M transformer + Q4_K_M GGUF Gemma on GPU).
    gguf_transformer_path: str = "./models/ltx-2.3-gguf/LTX-2.3-distilled-1.1/LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf"
    gguf_gemma_path: str = "./models/gemma-3-12b-it-gguf/gemma-3-12b-it-Q4_K_M.gguf"
    # First-party engine package (project root ./engine). ltx_runner launches
    # `python -m engine.worker` with this on PYTHONPATH.
    engine_dir: str = "./engine"
    # Interpreter that runs the first-party engine worker (torch + cu128 + ltx_core
    # / ltx_pipelines + gguf). The dedicated ./.venv-engine, relocated out of the
    # (now-deleted) fork tree in Stage 2b. Separate from the app's torch-free
    # ./.venv. Dependency snapshot: engine/venv-engine.freeze.txt.
    engine_python: str = "./.venv-engine/Scripts/python.exe"
    gguf_per_layer_quant: bool = True

    # Component-file re-sourcing (Phase 1): standalone small files replacing the
    # 46GB monolith for VAE/audio (and, later, text projection). Resolved via
    # AppConfig._abs (relative -> project-rooted absolute). Consumed by the real
    # worker / reuse harness only when vram.use_component_files is True.
    component_video_vae_path: str = "./models/ltx-2.3-components/vae/LTX23_video_vae_bf16.safetensors"
    component_audio_vae_path: str = "./models/ltx-2.3-components/vae/LTX23_audio_vae_bf16.safetensors"
    component_text_projection_path: str = "./models/ltx-2.3-components/text_encoders/ltx-2.3_text_projection_bf16.safetensors"

    # IC-LoRA adapter registry (Phase B, extended Phase C). Maps a server-side
    # adapter NAME (what the API accepts in GenerateRequest.loras[].name — never
    # a filesystem path) to either a bare safetensors path (string, legacy Phase B
    # form, implies preprocess="none") or an IcLoraEntry (Phase C: path +
    # preprocess kind, for control adapters like Union-Control that need a raw
    # reference video converted to a control signal before use). Absent/empty
    # section -> any loras request is rejected (fail loud, no silent skip).
    ic_loras: dict[str, str | IcLoraEntry] = Field(default_factory=dict)

    # Model-management registries (additive, Docs/MODEL_MANAGEMENT_DESIGN.md).
    # Category-scoped NAME -> path maps mirroring ic_loras: a server-side model
    # NAME (what GET /models lists and POST /pipeline/load accepts in its
    # optional ``models`` block — never a filesystem path) to a project-relative
    # (or absolute) weight file. Absent/empty sections are the norm:
    # services/model_registry.py always injects a "default" entry per category
    # from the fixed default-path fields above (so the default combination stays
    # byte-identical), and directory scanning discovers additional files in the
    # existing layout without any config edit.
    transformers: dict[str, str] = Field(default_factory=dict)
    text_encoders: dict[str, str] = Field(default_factory=dict)
    video_vaes: dict[str, str] = Field(default_factory=dict)
    audio_models: dict[str, str] = Field(default_factory=dict)


class VramConfig(BaseModel):
    low_vram_mode: bool = True
    low_vram_profile: str = "16gb_safe"
    # fp8_transformer / cpu_offload_text_encoder: FROZEN API CONTRACT. Both are
    # emitted by services/low_vram.py (_STATUS_KEYS + metadata_block) into
    # GET /status and metadata.json, so they must NOT be removed. Neither is
    # propagated to the worker payload: fp8 is chosen at runtime by
    # device_supports_fp8 auto-detection, and CPU text-encode offload is driven
    # by the separate te_offload_text_encoder field (LTX_TE_OFFLOAD env) below.
    fp8_transformer: bool = True
    cpu_offload_text_encoder: bool = True
    # Sequential per-layer CPU offload of the GGUF Gemma during text-encode
    # (caps the ~15GB encode peak to ~a few GB; compute stays on GPU). Wired to
    # the real worker via LTX_TE_OFFLOAD. Default ON.
    te_offload_text_encoder: bool = True
    # Build the DiT (transformer) on CPU and move only non-block submodules to
    # GPU, eliminating the ~16.9GB load-time GPU spike. Wired to the real worker
    # via LTX_DIT_CPU_LOAD. Default ON.
    dit_cpu_load: bool = True
    vae_tiling: bool = True
    attention_tiling: bool = False
    attention_tile_size: int | None = None
    block_swap: bool = False
    block_swap_blocks_on_gpu: int | None = None
    # VAE tiling sizes for the real GGUF engine (0 -> engine default, proven to
    # fit 16GB at small resolutions). Consumed by the subprocess worker.
    vae_spatial_tile_size: int = 0
    vae_temporal_tile_size: int = 0
    allow_disable_low_vram: bool = True
    # Phase 1 gate: re-source VIDEO VAE + AUDIO VAE/vocoder from standalone
    # component files (model.component_*_path) instead of the 46GB monolith.
    # Off by default; flip to True to exercise the component-file path.
    use_component_files: bool = False


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
    # Reference-video upload (Phase B, POST /upload/video). Stored as-is (no
    # re-encode) under uploads/videos/{video_id}/; the engine's ffmpeg-based
    # video IO reads these containers.
    max_video_size_mb: int = 200
    allowed_video_extensions: list[str] = Field(
        default_factory=lambda: [".mp4", ".mov", ".webm", ".mkv"]
    )
    # Audio-to-video upload (A2V, POST /upload/audio). Stored as-is (no
    # re-encode) under uploads/audios/{audio_id}/; the engine's PyAV-based audio
    # decode (decode_audio_from_file) reads these containers. Codec validity is
    # verified at preflight (ffprobe), not on upload — only extension + size gate.
    max_audio_size_mb: int = 50
    allowed_audio_extensions: list[str] = Field(
        default_factory=lambda: [".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"]
    )


class LimitsConfig(BaseModel):
    max_width: int = 1920
    max_height: int = 1088
    max_num_frames: int = 481
    max_conditioning_images: int = 5
    # frame_idx grid advertised via /config so clients can build the UI grid.
    # Keyframes snap to the latent-frame-START grid: frame_idx 0 is the start-frame
    # (latent-replace path); every OTHER keyframe sits on offset + multiple*n, i.e.
    # the 8n+1 pixels (1, 9, 17, ...). Server snap: (f-1)//8*8+1 clamped in-range.
    conditioning_frame_idx_multiple: int = 8
    conditioning_keyframe_grid_offset: int = 1
    phase1_max_concurrent_jobs: int = 1
    low_vram_disabled_required: bool = False
    # 解像度別 spill-free フレーム数（16GB 実測, §8.4）。API は 481f まで受けるが、
    # これを超えると shared へ溢れ ~2-4x 低速化（OOM せず）→ クライアント UI で警告する。
    # キーは "WxH" 生成サイズ文字列（client が引きやすい形式）。
    spill_free_frames: dict[str, int] = Field(default_factory=dict)
    # V2V continuation (POST /generate/chain source_video.context_frames). Bounds
    # advertised via /config so a UI can build the control. context_frames is 8n+1;
    # the 145 max is a conservative v1 cap (keeps the frozen head inside one
    # stage-2 tile — see api.models.SourceVideoSpec). Defaulted so an old
    # config.yaml (without these keys) still parses (spill_free_frames precedent).
    v2v_context_frames_default: int = 73
    v2v_context_frames_min: int = 25
    v2v_context_frames_max: int = 145


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
