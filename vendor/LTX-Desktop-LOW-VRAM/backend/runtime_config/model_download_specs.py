"""Canonical model download specs and required-model policy."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from state.app_state_types import ModelFileType


@dataclass(frozen=True, slots=True)
class ModelFileDownloadSpec:
    relative_path: Path
    expected_size_bytes: int
    is_folder: bool
    repo_id: str
    description: str

    @property
    def name(self) -> str:
        return self.relative_path.name


MODEL_FILE_ORDER: tuple[ModelFileType, ...] = (
    "checkpoint",
    "dev_checkpoint",
    "upsampler",
    "distilled_lora",
    "ic_lora",
    "depth_processor",
    "person_detector",
    "pose_processor",
    "text_encoder",
    "zit",
)


DEFAULT_MODEL_DOWNLOAD_SPECS: dict[ModelFileType, ModelFileDownloadSpec] = {
    "checkpoint": ModelFileDownloadSpec(
        relative_path=Path("ltx-2.3-22b-distilled.safetensors"),
        expected_size_bytes=43_000_000_000,
        is_folder=False,
        repo_id="Lightricks/LTX-2.3",
        description="Main transformer model (distilled, fast pipeline)",
    ),
    # Full dev/pro checkpoint — used by the two-stage pipeline (TI2VidTwoStagesPipeline).
    # Stage 1 runs this model with CFG guidance at half resolution; stage 2 applies the
    # distilled LoRA to refine at full resolution.  This is the highest-quality mode.
    # FP8 variant (~23 GB) is the practical choice for consumer GPUs.
    "dev_checkpoint": ModelFileDownloadSpec(
        relative_path=Path("ltx-2.3-22b-dev_fp8.safetensors"),
        expected_size_bytes=23_000_000_000,
        is_folder=False,
        repo_id="Lightricks/LTX-2.3",
        description="Dev/Pro transformer (two-stage pipeline, FP8)",
    ),
    "upsampler": ModelFileDownloadSpec(
        relative_path=Path("ltx-2.3-spatial-upscaler-x2-1.1.safetensors"),
        expected_size_bytes=1_900_000_000,
        is_folder=False,
        repo_id="Lightricks/LTX-2.3",
        description="2x Upscaler",
    ),
    "distilled_lora": ModelFileDownloadSpec(
        relative_path=Path("ltx-2.3-22b-distilled-lora-384.safetensors"),
        expected_size_bytes=7_500_000_000,
        is_folder=False,
        repo_id="Lightricks/LTX-2.3",
        description="LoRA for Pro model",
    ),
    "ic_lora": ModelFileDownloadSpec(
        relative_path=Path("ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"),
        expected_size_bytes=654_465_352,
        is_folder=False,
        repo_id="Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control",
        description="Union IC-LoRA control model",
    ),
    "depth_processor": ModelFileDownloadSpec(
        relative_path=Path("dpt-hybrid-midas"),
        expected_size_bytes=500_000_000,
        is_folder=True,
        repo_id="Intel/dpt-hybrid-midas",
        description="DPT-Hybrid MiDaS depth processor",
    ),
    "person_detector": ModelFileDownloadSpec(
        relative_path=Path("yolox_l.torchscript.pt"),
        expected_size_bytes=217_697_649,
        is_folder=False,
        repo_id="hr16/yolox-onnx",
        description="YOLOX person detector for pose preprocessing",
    ),
    "pose_processor": ModelFileDownloadSpec(
        relative_path=Path("dw-ll_ucoco_384_bs5.torchscript.pt"),
        expected_size_bytes=135_059_124,
        is_folder=False,
        repo_id="hr16/DWPose-TorchScript-BatchSize5",
        description="DW Pose TorchScript processor",
    ),
    "text_encoder": ModelFileDownloadSpec(
        relative_path=Path("gemma-3-12b-it-qat-q4_0-unquantized"),
        expected_size_bytes=25_000_000_000,
        is_folder=True,
        repo_id="Lightricks/gemma-3-12b-it-qat-q4_0-unquantized",
        description="Gemma text encoder (bfloat16)",
    ),
    "zit": ModelFileDownloadSpec(
        relative_path=Path("Z-Image-Turbo"),
        expected_size_bytes=31_000_000_000,
        is_folder=True,
        repo_id="Tongyi-MAI/Z-Image-Turbo",
        description="Z-Image-Turbo model for text-to-image generation",
    ),
}


DEFAULT_REQUIRED_MODEL_TYPES: frozenset[ModelFileType] = frozenset(
    {"checkpoint", "upsampler", "zit"}
)


def _normalized_relative_path(
    specs: Mapping[ModelFileType, ModelFileDownloadSpec],
    model_type: ModelFileType,
) -> Path:
    """Validate and normalize relative_path from specs — pure function."""
    relative_path = specs[model_type].relative_path
    if relative_path.is_absolute():
        raise ValueError(f"Model path for {model_type} must be relative: {relative_path}")

    normalized_parts = [part for part in relative_path.parts if part not in ("", ".")]
    if not normalized_parts:
        raise ValueError(f"Model path for {model_type} cannot be empty: {relative_path}")
    if ".." in normalized_parts:
        raise ValueError(f"Model path for {model_type} cannot traverse parents: {relative_path}")

    return Path(*normalized_parts)


def resolve_model_path(
    models_dir: Path,
    specs: Mapping[ModelFileType, ModelFileDownloadSpec],
    model_type: ModelFileType,
) -> Path:
    return models_dir / _normalized_relative_path(specs, model_type)


def resolve_downloading_dir(models_dir: Path) -> Path:
    return models_dir / ".downloading"


def resolve_downloading_target_path(
    models_dir: Path,
    specs: Mapping[ModelFileType, ModelFileDownloadSpec],
    model_type: ModelFileType,
) -> Path:
    return resolve_downloading_dir(models_dir) / _normalized_relative_path(specs, model_type)


def resolve_downloading_path(
    models_dir: Path,
    specs: Mapping[ModelFileType, ModelFileDownloadSpec],
    model_type: ModelFileType,
) -> Path:
    """Return the staging path under downloading_dir for a model type."""
    spec = specs[model_type]
    relative_path = _normalized_relative_path(specs, model_type)
    downloading_dir = resolve_downloading_dir(models_dir)
    if not spec.is_folder:
        parent = relative_path.parent
        if parent == Path("."):
            return downloading_dir
        return downloading_dir / parent
    return downloading_dir / relative_path


def resolve_required_model_types(
    base_required: frozenset[ModelFileType],
    has_api_key: bool,
    use_local_text_encoder: bool = False,
) -> frozenset[ModelFileType]:
    if not base_required:
        return base_required
    if has_api_key and not use_local_text_encoder:
        return base_required
    return cast(frozenset[ModelFileType], base_required | {"text_encoder"})
