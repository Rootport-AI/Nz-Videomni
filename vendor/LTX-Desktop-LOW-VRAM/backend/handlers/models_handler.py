"""Model availability and model status handlers."""

from __future__ import annotations

from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING

from api_types import CheckpointVariant, ModelFileStatus, ModelInfo, ModelsStatusResponse, TextEncoderStatus
from handlers.base import StateHandlerBase, with_state_lock
from runtime_config.model_download_specs import MODEL_FILE_ORDER, resolve_model_path, resolve_required_model_types
from state.app_state_types import AppState, AvailableFiles, ModelFileType

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig


class ModelsHandler(StateHandlerBase):
    def __init__(
        self,
        state: AppState,
        lock: RLock,
        config: RuntimeConfig,
    ) -> None:
        super().__init__(state, lock, config)

    @staticmethod
    def _path_size(path: Path, is_folder: bool) -> int:
        if not is_folder:
            return path.stat().st_size
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())

    def _scan_available_files(self) -> AvailableFiles:
        files: AvailableFiles = {}
        for model_type in MODEL_FILE_ORDER:
            spec = self.config.spec_for(model_type)
            path = resolve_model_path(self.models_dir, self.config.model_download_specs, model_type)
            if spec.is_folder:
                ready = path.exists() and any(path.iterdir()) if path.exists() else False
                files[model_type] = path if ready else None
            else:
                files[model_type] = path if path.exists() else None
        return files

    @with_state_lock
    def refresh_available_files(self) -> AvailableFiles:
        self.state.available_files = self._scan_available_files()
        return self.state.available_files.copy()

    def get_text_encoder_status(self) -> TextEncoderStatus:
        files = self.refresh_available_files()
        text_encoder_path = files["text_encoder"]
        exists = text_encoder_path is not None
        text_spec = self.config.spec_for("text_encoder")
        size_bytes = self._path_size(text_encoder_path, is_folder=True) if exists else 0
        expected = text_spec.expected_size_bytes

        return TextEncoderStatus(
            downloaded=exists,
            size_bytes=size_bytes if exists else expected,
            size_gb=round((size_bytes if exists else expected) / (1024**3), 1),
            expected_size_gb=round(expected / (1024**3), 1),
        )

    def get_checkpoint_variants(self) -> list[CheckpointVariant]:
        """Scan models dir and return all usable checkpoint variants."""
        d = self.models_dir
        variants: list[CheckpointVariant] = []

        distilled = d / "ltx-2.3-22b-distilled.safetensors"
        fp8_preq  = d / "fp8_transformer.safetensors"
        distilled_lora = d / "ltx-2.3-22b-distilled-lora-384.safetensors"

        def _gb(path: Path) -> float:
            try:
                return round(path.stat().st_size / (1024 ** 3), 1)
            except OSError:
                return 0.0

        # ── Fast BF16 ──────────────────────────────────────────────────
        variants.append(CheckpointVariant(
            id="fast-bf16",
            label="Fast — BF16",
            description="Full-precision distilled model. Best accuracy, highest VRAM (~24 GB).",
            available=distilled.exists(),
            pipeline_type="fast",
            gguf_path="",
            use_fp8=False,
            size_gb=_gb(distilled) if distilled.exists() else 43.0,
        ))

        # ── Fast FP8 (pre-quantized) ───────────────────────────────────
        if fp8_preq.exists():
            variants.append(CheckpointVariant(
                id="fast-fp8",
                label="Fast — FP8 (pre-quantized)",
                description="FP8 pre-quantized transformer. Faster load, ~10–12 GB VRAM.",
                available=distilled.exists(),  # still needs the full checkpoint for non-transformer parts
                pipeline_type="fast",
                gguf_path="",
                use_fp8=True,
                size_gb=_gb(fp8_preq),
            ))

        # ── Fast GGUF variants (one per .gguf file in models dir) ──────
        gguf_files = sorted(d.glob("*.gguf"))
        for gguf_file in gguf_files:
            stem = gguf_file.stem
            # Parse quantisation tag from filename (e.g. "...-Q8_0" → "Q8_0")
            quant = stem.rsplit("-", 1)[-1].upper() if "-" in stem else stem.upper()
            variants.append(CheckpointVariant(
                id=f"fast-gguf-{quant.lower()}",
                label=f"Fast — GGUF {quant}",
                description=(
                    f"GGUF {quant} transformer. Smaller file; VRAM same as BF16 "
                    f"(dequantized to BF16 at load). File: {gguf_file.name}"
                ),
                available=True,
                pipeline_type="fast",
                gguf_path=str(gguf_file),
                use_fp8=False,
                size_gb=_gb(gguf_file),
            ))

        # ── Dev (two-stage, requires distilled_lora) ───────────────────
        dev_available = distilled.exists() and distilled_lora.exists()
        variants.append(CheckpointVariant(
            id="dev",
            label="Dev — High Quality (CFG + LoRA)",
            description=(
                "Two-stage pipeline: real CFG, negative prompt, STG. "
                "Requires distilled LoRA (7 GB). Slower but higher quality."
            ),
            available=dev_available,
            pipeline_type="dev",
            gguf_path="",
            use_fp8=False,
            size_gb=round(_gb(distilled) + _gb(distilled_lora), 1) if dev_available else None,
        ))

        # ── Dev GGUF variants (dev pipeline + GGUF transformer) ────────
        for gguf_file in gguf_files:
            stem = gguf_file.stem
            quant = stem.rsplit("-", 1)[-1].upper() if "-" in stem else stem.upper()
            variants.append(CheckpointVariant(
                id=f"dev-gguf-{quant.lower()}",
                label=f"Dev — GGUF {quant} (CFG + LoRA)",
                description=(
                    f"Two-stage dev pipeline with GGUF {quant} transformer. "
                    f"Real CFG + STG, smaller GGUF file. Requires distilled LoRA."
                ),
                available=dev_available,
                pipeline_type="dev",
                gguf_path=str(gguf_file),
                use_fp8=False,
                size_gb=round(_gb(gguf_file) + _gb(distilled_lora), 1) if dev_available else None,
            ))

        return variants

    def get_models_list(self) -> list[ModelInfo]:
        pro_steps = self.state.app_settings.pro_model.steps
        pro_upscaler = self.state.app_settings.pro_model.use_upscaler
        return [
            ModelInfo(id="fast", name="Fast (Distilled)", description="8 steps + 2x upscaler"),
            ModelInfo(
                id="pro",
                name="Pro (Full)",
                description=f"{pro_steps} steps" + (" + 2x upscaler" if pro_upscaler else " (native resolution)"),
            ),
        ]

    @with_state_lock
    def get_required_model_types(self, skip_text_encoder: bool = False) -> list[ModelFileType]:
        settings = self.state.app_settings
        required = resolve_required_model_types(
            self._config.required_model_types,
            has_api_key=bool(settings.ltx_api_key),
            use_local_text_encoder=settings.use_local_text_encoder,
        )
        return [
            model_type
            for model_type in MODEL_FILE_ORDER
            if model_type in required and not (skip_text_encoder and model_type == "text_encoder")
        ]

    def get_models_status(self, has_api_key: bool | None = None) -> ModelsStatusResponse:
        files = self.refresh_available_files()
        settings = self.state.app_settings.model_copy(deep=True)

        if has_api_key is None:
            has_api_key = bool(settings.ltx_api_key)

        models: list[ModelFileStatus] = []
        total_size = 0
        downloaded_size = 0
        required_types = resolve_required_model_types(
            self.config.required_model_types,
            has_api_key,
            settings.use_local_text_encoder,
        )

        for model_type in MODEL_FILE_ORDER:
            spec = self.config.spec_for(model_type)
            path = files[model_type]
            exists = path is not None
            actual_size = self._path_size(path, is_folder=spec.is_folder) if exists else 0
            required = model_type in required_types
            if required:
                total_size += spec.expected_size_bytes
                if exists:
                    downloaded_size += actual_size

            description = spec.description
            optional_reason: str | None = None
            if model_type == "text_encoder":
                description += " (optional with API key)" if has_api_key else ""
                optional_reason = "Uses LTX API for text encoding" if has_api_key else None

            models.append(
                ModelFileStatus(
                    id=model_type,
                    name=spec.name,
                    description=description,
                    downloaded=exists,
                    size=actual_size if exists else spec.expected_size_bytes,
                    expected_size=spec.expected_size_bytes,
                    required=required,
                    is_folder=spec.is_folder,
                    optional_reason=optional_reason if model_type == "text_encoder" else None,
                )
            )

        all_downloaded = all(model.downloaded for model in models if model.required)

        return ModelsStatusResponse(
            models=models,
            all_downloaded=all_downloaded,
            total_size=total_size,
            downloaded_size=downloaded_size,
            total_size_gb=round(total_size / (1024**3), 1),
            downloaded_size_gb=round(downloaded_size / (1024**3), 1),
            models_path=str(self.models_dir),
            has_api_key=has_api_key,
            text_encoder_status=self.get_text_encoder_status(),
            use_local_text_encoder=settings.use_local_text_encoder,
        )
