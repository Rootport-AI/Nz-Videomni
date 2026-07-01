"""Pydantic request/response models and TypedDicts for ltx2_server."""

from __future__ import annotations

from typing import Literal, NamedTuple, TypeAlias, TypedDict
from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

NonEmptyPrompt = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
ModelFileType = Literal[
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
]


class ImageConditioningInput(NamedTuple):
    """Image conditioning triplet used by all video pipelines."""

    path: str
    frame_idx: int
    strength: float


# ============================================================
# TypedDicts for module-level state globals
# ============================================================


class GenerationState(TypedDict):
    id: str | None
    cancelled: bool
    result: str | list[str] | None
    error: str | None
    status: str  # "idle" | "running" | "complete" | "cancelled" | "error"
    phase: str
    progress: int
    current_step: int
    total_steps: int


JsonObject: TypeAlias = dict[str, object]
VideoCameraMotion = Literal[
    "none",
    "dolly_in",
    "dolly_out",
    "dolly_left",
    "dolly_right",
    "jib_up",
    "jib_down",
    "static",
    "focus_shift",
]


# ============================================================
# Response Models
# ============================================================


class CheckpointVariant(BaseModel):
    """A selectable model variant — combination of pipeline type and checkpoint file."""
    id: str
    label: str
    description: str
    available: bool
    pipeline_type: str   # "fast" | "dev"
    gguf_path: str = ""  # non-empty → use GGUF transformer
    use_fp8: bool = False  # True → use pre-quantized FP8 transformer
    size_gb: float | None = None


class ModelStatusItem(BaseModel):
    id: str
    name: str
    loaded: bool
    downloaded: bool


class GpuTelemetry(BaseModel):
    name: str
    vram: int
    vramUsed: int


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    active_model: str | None
    gpu_info: GpuTelemetry
    sage_attention: bool
    models_status: list[ModelStatusItem]


class GpuInfoResponse(BaseModel):
    cuda_available: bool
    mps_available: bool = False
    gpu_available: bool = False
    gpu_name: str | None
    vram_gb: int | None
    gpu_info: GpuTelemetry


class RuntimePolicyResponse(BaseModel):
    force_api_generations: bool


class GenerationProgressResponse(BaseModel):
    status: str
    phase: str
    progress: int
    currentStep: int | None
    totalSteps: int | None
    enhancedPrompt: str | None = None


class ModelInfo(BaseModel):
    id: str
    name: str
    description: str


class ModelFileStatus(BaseModel):
    id: ModelFileType
    name: str
    description: str
    downloaded: bool
    size: int
    expected_size: int
    required: bool = True
    is_folder: bool = False
    optional_reason: str | None = None


class TextEncoderStatus(BaseModel):
    downloaded: bool
    size_bytes: int
    size_gb: float
    expected_size_gb: float


class ModelsStatusResponse(BaseModel):
    models: list[ModelFileStatus]
    all_downloaded: bool
    total_size: int
    downloaded_size: int
    total_size_gb: float
    downloaded_size_gb: float
    models_path: str
    has_api_key: bool
    text_encoder_status: TextEncoderStatus
    use_local_text_encoder: bool


class DownloadProgressResponse(BaseModel):
    status: str
    current_downloading_file: ModelFileType | None
    current_file_progress: float
    total_progress: float
    total_downloaded_bytes: int
    expected_total_bytes: int
    completed_files: set[ModelFileType]
    all_files: set[ModelFileType]
    error: str | None
    speed_bytes_per_sec: float


class SuggestGapPromptResponse(BaseModel):
    status: str = "success"
    suggested_prompt: str


class GenerateVideoResponse(BaseModel):
    status: str
    video_path: str | None = None
    seed_used: int | None = None


class GenerateImageResponse(BaseModel):
    status: str
    image_paths: list[str] | None = None


class CancelResponse(BaseModel):
    status: str
    id: str | None = None


class RetakeResponse(BaseModel):
    status: str
    video_path: str | None = None
    result: JsonObject | None = None


class IcLoraExtractResponse(BaseModel):
    conditioning: str
    original: str
    conditioning_type: Literal["canny", "depth"]
    frame_time: float


class IcLoraGenerateResponse(BaseModel):
    status: str
    video_path: str | None = None


class ModelDownloadStartResponse(BaseModel):
    status: str
    message: str | None = None
    sessionId: str | None = None


class TextEncoderDownloadResponse(BaseModel):
    status: str
    message: str | None = None
    sessionId: str | None = None


class StatusResponse(BaseModel):
    status: str


class ErrorResponse(BaseModel):
    error: str
    message: str | None = None


# ============================================================
# Request Models
# ============================================================


class ConditioningImageRequest(BaseModel):
    """A single conditioning image for multi-frame generation."""
    path: str
    frameIdx: int = 0       # 0 = first frame, -1 = last frame, or specific frame number
    strength: float = 1.0   # 0.0–1.0


class GenerateVideoRequest(BaseModel):
    prompt: NonEmptyPrompt
    resolution: str = "512p"
    model: str = "fast"
    cameraMotion: VideoCameraMotion = "none"
    negativePrompt: str = ""
    duration: str = "2"
    fps: str = "24"
    audio: str = "false"
    imagePath: str | None = None
    audioPath: str | None = None
    aspectRatio: Literal["16:9", "9:16"] = "16:9"
    seed: int | None = None
    enhancedPrompt: str | None = None
    conditioningImages: list[ConditioningImageRequest] | None = None  # multi-frame conditioning
    numSteps: int | None = None        # per-request override for distilledNumSteps
    stgScale: float | None = None     # per-request override for stgScale
    stgBlockIndex: int | None = None  # per-request override for stgBlockIndex
    # Dev (two-stage) pipeline overrides
    cfgScale: float | None = None        # video CFG scale (dev pipeline, default 3.0)
    audioCfgScale: float | None = None   # audio CFG scale (dev pipeline, default 7.0)
    rescaleScale: float | None = None    # CFG rescale factor (dev pipeline, default 0.7)
    modalityScale: float | None = None   # modality coupling scale (dev pipeline, default 3.0)


class GenerateImageRequest(BaseModel):
    prompt: NonEmptyPrompt
    width: int = 1024
    height: int = 1024
    numSteps: int = 4
    numImages: int = 1


def _default_model_types() -> set[ModelFileType]:
    return set()


class ModelDownloadRequest(BaseModel):
    modelTypes: set[ModelFileType] = Field(default_factory=_default_model_types)


class RequiredModelsResponse(BaseModel):
    modelTypes: list[ModelFileType]


class SuggestGapPromptRequest(BaseModel):
    beforePrompt: str = ""
    afterPrompt: str = ""
    beforeFrame: str | None = None
    afterFrame: str | None = None
    gapDuration: float = 5
    mode: str = "t2v"
    inputImage: str | None = None


class RetakeRequest(BaseModel):
    video_path: str
    start_time: float
    duration: float
    prompt: str = ""
    mode: str = "replace_audio_and_video"


class IcLoraExtractRequest(BaseModel):
    video_path: str
    conditioning_type: Literal["canny", "depth"] = "canny"
    frame_time: float = 0


class IcLoraImageInput(BaseModel):
    path: str
    frame: int = 0
    strength: float = 1.0


def _default_ic_lora_images() -> list[IcLoraImageInput]:
    return []


class IcLoraGenerateRequest(BaseModel):
    video_path: str
    conditioning_type: Literal["canny", "depth"]
    prompt: NonEmptyPrompt
    conditioning_strength: float = 1.0
    num_inference_steps: int = 30
    cfg_guidance_scale: float = 1.0
    negative_prompt: str = ""
    images: list[IcLoraImageInput] = Field(default_factory=_default_ic_lora_images)


# ============================================================
# MagiHuman request / response types
# ============================================================


class MagiGenerateRequest(BaseModel):
    prompt: NonEmptyPrompt
    image_path: str       # Windows filesystem path to conditioning image
    seconds: int = 5
    width: int = 448
    height: int = 256
    gpus: int = 2
    seed: int | None = None
    sr: bool = False      # Super-resolution 2× upscale via 540p_sr model


class MagiProgressResponse(BaseModel):
    status: str           # idle | running | complete | error | cancelled
    output_path: str | None = None
    error: str | None = None
    log_tail: str = ""
    sr_model_ready: bool = False   # True once 540p_sr model is downloaded


class MagiGenerateResponse(BaseModel):
    status: str


# ============================================================
# MMAudio request / response types
# ============================================================


class MMAudioGenerateRequest(BaseModel):
    video_path: str           # Windows filesystem path to input video
    prompt: str = ""          # Optional audio description; defaults to ambient inference
    duration: float = 8.0    # Audio duration in seconds (match video length for best results)
    seed: int | None = None
    negative_prompt: str = ""
    cfg_strength: float = 4.5
    num_steps: int = 25


class MMAudioProgressResponse(BaseModel):
    status: str               # idle | running | complete | error | cancelled
    output_path: str | None = None
    error: str | None = None
    log_tail: str = ""


# ============================================================
# PrismAudio request / response types
# ============================================================


class PrismAudioGenerateRequest(BaseModel):
    video_path: str           # Windows filesystem path to input video
    prompt: str = ""          # Optional Foley/SFX description
    seed: int | None = None


class PrismAudioProgressResponse(BaseModel):
    status: str               # idle | running | complete | error | cancelled
    output_path: str | None = None
    error: str | None = None
    log_tail: str = ""


# ============================================================
# Qwen3-TTS request / response types
# ============================================================


class QwenTTSGenerateRequest(BaseModel):
    text: str                               # Text to synthesise
    language: str = "English"               # e.g. "English", "Chinese", "Japanese"
    mode: str = "custom_voice"              # "custom_voice" | "voice_clone"
    speaker: str = "Ryan"                   # CustomVoice preset speaker name
    instruct: str = ""                      # Optional style instruction (CustomVoice)
    ref_audio_path: str | None = None       # Filesystem path to reference WAV (voice_clone)
    ref_text: str = ""                      # Transcript of reference audio (voice_clone)
    model_size: str = "1.7b"               # "1.7b" | "0.6b"


class QwenTTSProgressResponse(BaseModel):
    status: str                             # idle | running | complete | error | cancelled
    output_path: str | None = None
    error: str | None = None
    log_tail: str = ""
