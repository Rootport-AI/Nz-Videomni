"""Custom API errors and error codes (spec ch.17).

All domain errors raise :class:`APIError`, which is translated to the spec's
error envelope by an exception handler registered in ``main.py``::

    {"error": {"code": ..., "message": ..., "job_id": ..., "detail": ...}}
"""

from __future__ import annotations


class APIError(Exception):
    """A domain error carrying an HTTP status and a stable error ``code``."""

    def __init__(
        self,
        code: str,
        message: str,
        status_code: int,
        *,
        job_id: str | None = None,
        detail: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.job_id = job_id
        self.detail = detail

    def to_envelope(self) -> dict:
        error: dict = {"code": self.code, "message": self.message}
        if self.job_id is not None:
            error["job_id"] = self.job_id
        if self.detail is not None:
            error["detail"] = self.detail
        return {"error": error}


# --- factory helpers for the spec's named error codes (spec 17.3) ---


def job_busy(detail: str | None = None) -> APIError:
    return APIError(
        "JOB_BUSY",
        "A job is already running (Phase 1 allows one concurrent job)",
        409,
        detail=detail,
    )


def upload_invalid_type(detail: str | None = None) -> APIError:
    return APIError("UPLOAD_INVALID_TYPE", "Unsupported image format", 400, detail=detail)


def upload_too_large(detail: str | None = None) -> APIError:
    return APIError("UPLOAD_TOO_LARGE", "Image file size exceeds the limit", 400, detail=detail)


def image_not_found(image_id: str) -> APIError:
    return APIError("IMAGE_NOT_FOUND", f"image_id not found: {image_id}", 404)


def reference_video_not_found(video_id: str) -> APIError:
    return APIError("REFERENCE_VIDEO_NOT_FOUND", f"reference_video_id not found: {video_id}", 404)


def source_video_not_found(video_id: str) -> APIError:
    """V2V continuation: the ``source_video.video_id`` does not resolve to a
    stored upload. Mirrors :func:`reference_video_not_found` (404) with its own
    stable code so a client can distinguish the continuation source from an
    IC-LoRA reference video."""
    return APIError("SOURCE_VIDEO_NOT_FOUND", f"source_video.video_id not found: {video_id}", 404)


def source_video_too_short(detail: str | None = None) -> APIError:
    """V2V continuation: the stored source video has fewer frames (after any fps
    resample) than the requested ``context_frames``, so there is no full context
    tail to freeze. Rejected up front (422) before any GPU work."""
    return APIError(
        "SOURCE_VIDEO_TOO_SHORT",
        "source video has fewer frames than the requested context_frames",
        422,
        detail=detail,
    )


def source_audio_not_found(audio_id: str) -> APIError:
    """A2V: the ``source_audio.audio_id`` does not resolve to a stored upload.
    Mirrors :func:`source_video_not_found` (404) with its own stable code so a
    client can distinguish the audio-to-video source from a continuation video."""
    return APIError("SOURCE_AUDIO_NOT_FOUND", f"source_audio.audio_id not found: {audio_id}", 404)


def source_audio_too_short(detail: str | None = None) -> APIError:
    """A2V: the uploaded source audio VAE-encodes to fewer audio-latent frames
    than the chain timeline requires (video length is authoritative; audio is
    truncated, never padded). Rejected up front (422) before any GPU work —
    mirrors :func:`source_video_too_short`."""
    return APIError(
        "SOURCE_AUDIO_TOO_SHORT",
        "source audio is shorter than the requested chain timeline",
        422,
        detail=detail,
    )


def lora_not_found(name: str, detail: str | None = None) -> APIError:
    return APIError("LORA_NOT_FOUND", f"unknown IC-LoRA adapter name: {name}", 404, detail=detail)


def model_not_found(category: str, name: str, detail: str | None = None) -> APIError:
    """Model management: an unknown model NAME (or unknown category) was
    requested. Mirrors :func:`lora_not_found` (404) — names come from the
    category registry (GET /models); arbitrary filesystem paths are never
    accepted."""
    return APIError(
        "MODEL_NOT_FOUND",
        f"unknown model name '{name}' in category '{category}'",
        404,
        detail=detail,
    )


def model_file_missing(category: str, name: str, detail: str | None = None) -> APIError:
    """Model management: the NAME is registered but its weight file is gone
    from disk (deleted/moved after registration or scan). 422 — the request is
    well-formed; the server-side artifact is what is unusable."""
    return APIError(
        "MODEL_FILE_MISSING",
        f"registered model file for '{category}/{name}' is missing on disk",
        422,
        detail=detail,
    )


def model_incompatible(category: str, name: str, detail: str | None = None) -> APIError:
    """Model management: the selected file failed the cheap compatibility
    precheck (wrong extension / not a GGUF / broken safetensors header) that
    runs BEFORE the worker is restarted — guarding against a native loader
    crash deep in the engine. 422."""
    return APIError(
        "MODEL_INCOMPATIBLE",
        f"selected model '{category}/{name}' failed the compatibility precheck",
        422,
        detail=detail,
    )


def lora_preprocess_conflict(kinds: list[str]) -> APIError:
    """Phase C: the requested loras imply more than one control preprocess kind
    (e.g. one canny-control + one pose-control adapter) for a single reference
    video. Only one control signal can be derived from the one uploaded video."""
    return APIError(
        "LORA_PREPROCESS_CONFLICT",
        "loras request more than one control preprocess kind for a single reference video",
        400,
        detail=f"conflicting preprocess kinds: {sorted(kinds)}",
    )


def reference_resolution_invalid(width: int, height: int) -> APIError:
    """Phase C: registered IC-LoRA adapters use reference_downscale_factor=2, so
    the reference video is consumed at half the output resolution on VAE's
    64-grid. If width/height are not divisible by 128, that half-resolution
    reference lands off the 64-grid and the worker's VAE encode fails with an
    unfriendly einops error deep in the job -- reject it up front instead."""
    return APIError(
        "REFERENCE_RESOLUTION_INVALID",
        "reference-video jobs require width/height divisible by 128 "
        "(reference is used at half resolution on the 64-grid)",
        422,
        detail=f"width={width}, height={height}",
    )


def job_not_found(job_id: str) -> APIError:
    return APIError("JOB_NOT_FOUND", f"job not found: {job_id}", 404, job_id=job_id)


def video_not_ready(job_id: str) -> APIError:
    return APIError("VIDEO_NOT_READY", "Video is not ready yet", 409, job_id=job_id)


def pipeline_load_failed(detail: str | None = None) -> APIError:
    return APIError("PIPELINE_LOAD_FAILED", "Failed to load the pipeline", 503, detail=detail)


def gpu_oom(job_id: str | None = None, detail: str | None = None) -> APIError:
    return APIError(
        "GPU_OOM",
        "CUDA out of memory during generation",
        503,
        job_id=job_id,
        detail=detail or "Try smaller resolution or fewer frames.",
    )


def generation_failed(job_id: str | None = None, detail: str | None = None) -> APIError:
    return APIError("GENERATION_FAILED", "Generation failed", 503, job_id=job_id, detail=detail)
