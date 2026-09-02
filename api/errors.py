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


def retake_video_not_found(video_id: str) -> APIError:
    """Retake: the ``retake.video_id`` does not resolve to a stored upload.
    Mirrors :func:`source_video_not_found` (404) with its own stable code so a
    client can tell a missing retake source from a missing continuation video."""
    return APIError("RETAKE_VIDEO_NOT_FOUND", f"retake.video_id not found: {video_id}", 404)


def retake_window_out_of_range(detail: str | None = None) -> APIError:
    """Retake: the requested window does not fit the uploaded video.

    Either ``window_start_sec + clips[0].num_frames`` runs past the end of the
    upload (at the request frame rate), or ``regenerate_audio=False`` was asked
    for on an upload with no audio stream to keep. Rejected up front (422) before
    any GPU work — mirrors :func:`source_video_too_short`. The window's own
    geometry (8n+1, [73, 169], glue grids) is a schema/``chain_math`` 422 and
    never reaches here."""
    return APIError(
        "RETAKE_WINDOW_OUT_OF_RANGE",
        "the requested retake window does not fit the uploaded video",
        422,
        detail=detail,
    )


def end_source_not_found(source_id: str) -> APIError:
    """End source: the ``end_source.video_id`` / ``image_id`` does not resolve to
    a stored upload. ONE code for both stores — from a client's point of view the
    end source is one slot that happens to accept either kind, and the id it sent
    is echoed in the message, so splitting this into two codes would only add a
    branch nobody can act on differently."""
    return APIError("END_SOURCE_NOT_FOUND", f"end_source id not found: {source_id}", 404)


def end_source_too_short(detail: str | None = None) -> APIError:
    """End source: the stored video has fewer frames (after any fps resample)
    than the requested ``context_frames`` PLUS the one primer frame the causal
    VAE spends on its lone keyframe latent, so there is no full tail band to
    freeze. Rejected up front (422) before any GPU work — mirrors
    :func:`source_video_too_short`. Never raised for an image end source: a still
    is looped to whatever length is asked for."""
    return APIError(
        "END_SOURCE_TOO_SHORT",
        "end source video has fewer frames than context_frames + 1",
        422,
        detail=detail,
    )


def job_not_joinable(job_id: str, detail: str | None = None) -> APIError:
    """V2V join: the target job is not a V2V continuation job (its metadata has
    no ``v2v`` block — a plain chain / A2V / single generate), so there is no
    source video + continuation pair to join. Rejected up front (422)."""
    return APIError(
        "JOB_NOT_JOINABLE",
        "job is not a V2V continuation job (nothing to join)",
        422,
        job_id=job_id,
        detail=detail,
    )


def join_failed(job_id: str | None = None, detail: str | None = None) -> APIError:
    """V2V join: the server-side ffmpeg join (normalize / crossfade / concat)
    failed. Mirrors :func:`generation_failed` (503) with its own stable code."""
    return APIError("JOIN_FAILED", "V2V join failed", 503, job_id=job_id, detail=detail)


def joined_not_ready(job_id: str) -> APIError:
    """V2V join: GET /jobs/{id}/joined was called before a successful
    POST /jobs/{id}/join produced ``joined.mp4``."""
    return APIError(
        "JOINED_NOT_READY",
        "joined video has not been created yet (POST /jobs/{job_id}/join first)",
        404,
        job_id=job_id,
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


def lora_requires_reference(names: list[str]) -> APIError:
    """S1: a CONTROL-type IC-LoRA (union-control / pixel-spatial-upscaler — it
    derives its conditioning from a reference video) was requested without a
    ``reference_video_id``. Style/character LoRAs need no reference, so the old
    all-or-nothing ``loras <=> reference_video_id`` rule was relaxed to this
    kind-aware endpoint check. 422 — the request is well-formed; the required
    companion input (a reference video) is what is missing."""
    return APIError(
        "LORA_REQUIRES_REFERENCE",
        "a control-type IC-LoRA requires a reference_video_id",
        422,
        detail=f"control loras needing a reference video: {sorted(names)}",
    )


def reference_requires_control_lora(names: list[str]) -> APIError:
    """The reverse of :func:`lora_requires_reference`: a ``reference_video_id``
    was supplied but not one requested adapter is CONTROL-type. A reference
    video is only ever consumed through a control adapter (the reference
    downscale factor is read from that adapter's metadata), so a
    reference + style-only request has nothing to feed the video to. 422 — the
    request is well-formed; the required companion input (a control adapter) is
    what is missing."""
    return APIError(
        "REFERENCE_REQUIRES_CONTROL_LORA",
        "a reference_video_id requires at least one control-type IC-LoRA",
        422,
        detail=f"requested loras, none of them control-type: {sorted(names)}",
    )


def lora_format_unsupported(name: str, detail: str) -> APIError:
    """§3-108: the adapter file is a real safetensors, but its weight layout is
    one the engine loader cannot read — DoRA, LoHa, LoKr, kohya keys joined by
    underscores instead of dots (they resolve to no ``named_modules()`` name),
    or no LoRA weight keys at all. Only ``.lora_A``/``.lora_B`` (A/B) and
    ``.lora_down``/``.lora_up`` (+ ``.alpha``, kohya) are supported. Refusing
    here is what turns the old SILENT no-op (0 pairs -> a video identical to the
    LoRA-free one) into a loud rejection. 422 — the request is well-formed; the
    server-side artifact is what is unusable."""
    return APIError(
        "LORA_FORMAT_UNSUPPORTED",
        f"LoRA '{name}' uses an unsupported weight layout",
        422,
        detail=detail,
    )


def lora_depth_chain_unsupported(names: list[str]) -> APIError:
    """Chain LoRA (owner decision 2026-08-11): a depth-preprocess CONTROL IC-LoRA
    (Video-Depth-Anything) was requested on a chain with clips > 1. Multi-clip
    reference-video conditioning is otherwise supported (chain_math.
    video_segment_windows slices one long reference into per-clip windows), but
    the depth preprocessor is a whole-clip, all-frames-in-memory design (32-frame
    windows + full-clip min-max normalization) that cannot be chunked to a
    chain-length reference without OOM or breaking its normalization — so it is
    v1-scoped to single-clip chains and single-shot /generate only. Mirrors
    :func:`lora_requires_reference` (422) — the request is well-formed but the
    adapter kind is unsupported on this route."""
    return APIError(
        "LORA_DEPTH_CHAIN_UNSUPPORTED",
        "depth-type IC-LoRA is not supported on a multi-clip chain in this "
        "version (the depth preprocessor cannot process a chain-length "
        "reference); use pose/canny/deblur, or a single clip.",
        422,
        detail=f"depth loras rejected on multi-clip chain: {sorted(names)}",
    )


def lora_thumbnail_not_found(name: str) -> APIError:
    """S1: GET /loras/{name}/thumbnail for an adapter that has no sibling
    ``<stem>.png`` (or an unknown adapter name). Mirrors :func:`lora_not_found`
    (404)."""
    return APIError(
        "LORA_THUMBNAIL_NOT_FOUND",
        f"no thumbnail for IC-LoRA adapter: {name}",
        404,
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


def outpaint_preprocess_conflict(kinds: list[str]) -> APIError:
    """Docs/PENDING_TASKS_CLOSED.md §3-70 (filed as §1-13 at the time): outpainting
    hands the engine a green-padded CANVAS as the reference
    video, so a control adapter that would first run it through a preprocessor
    (canny / dwpose / depth) is incoherent — the edge map or depth map of a
    sentinel-green border is meaningless, and the In-Outpainting IC-LoRA expects
    the raw pixels. Only ``preprocess: none`` control adapters can outpaint."""
    return APIError(
        "OUTPAINT_PREPROCESS_CONFLICT",
        "outpaint requires a control lora with no reference preprocessing "
        "(the green canvas must reach the model as raw pixels)",
        422,
        detail=f"requested preprocess kinds: {sorted(kinds)}",
    )


def outpaint_source_mismatch(
    expected: tuple[int, int], actual: tuple[int, int] | None
) -> APIError:
    """Docs/PENDING_TASKS_CLOSED.md §3-70 (filed as §1-13 at the time):
    ``width``/``height`` are the final canvas and the four pads are cut
    out of it, so the keep rectangle is fully determined by the request. If the
    reference video's own resolution differs, the source would be silently
    rescaled and centre-cropped into the canvas (``resize_and_center_crop``),
    quietly breaking outpainting's one invariant — that the original pixels come
    through untouched. Reject instead of rescaling."""
    got = "unreadable (ffprobe unavailable or failed)" if actual is None else f"{actual[0]}x{actual[1]}"
    return APIError(
        "OUTPAINT_SOURCE_MISMATCH",
        "the reference video's resolution must equal the outpaint keep region "
        "(width/height minus the pads)",
        422,
        detail=f"keep region {expected[0]}x{expected[1]}, reference video {got}",
    )


def outpaint_source_too_short(available: int, required: int) -> APIError:
    """Docs/PENDING_TASKS_CLOSED.md §3-70 (filed as §1-13 at the time): the two
    blends pair frame *i* of the generation with frame *i* of
    the green canvas, and stage 2 asserts its initial latent matches the target
    shape, so a source shorter than ``num_frames`` cannot be honoured — the tail
    would be a frozen clone of the last frame while the request claims real
    footage. Reject up front instead of failing deep inside the denoiser."""
    return APIError(
        "OUTPAINT_SOURCE_TOO_SHORT",
        "num_frames exceeds the reference video's own frame count",
        422,
        detail=f"reference video has {available} frames, request needs {required}",
    )


def reference_resolution_invalid(width: int, height: int) -> APIError:
    """Phase C: the reference video is consumed on the VAE's 64-grid. The
    downscale factor is 2 (union-control family) or 1 (deblur); the divisible-by-128
    requirement is unchanged either way. If width/height are not divisible by 128,
    the reference lands off the 64-grid and the worker's VAE encode fails with an
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


def pipeline_loading(detail: str | None = None) -> APIError:
    """A load/reload arrived while one is already in flight (§3-97 P6).

    409, the same "you are asking at the wrong moment" family as
    :func:`job_busy` — nothing is wrong with the request, it just has to wait.
    A model load takes minutes, so a double-click on the frontend's Load button
    is the ordinary way to reach this, not an exotic race.

    THE MESSAGE IS BILINGUAL ON PURPOSE. Auto-load-on-generate calls the same
    load path from inside a job, where this error is re-wrapped as
    ``generation_failed(detail=str(exc))`` — and ``str(APIError)`` is the
    MESSAGE, not the detail. The Japanese half is therefore the only part that
    survives into what the operator reads on a failed job.
    """
    return APIError(
        "PIPELINE_LOADING",
        "The pipeline is already loading (モデルの読み込み中です)",
        409,
        detail=detail,
    )


def feature_unsupported(feature: str, detail: str | None = None) -> APIError:
    """The request asks for something THIS base model's engine cannot do (§3-98).

    422, not 400: the request is perfectly well-formed and would have been
    accepted by another base model — what makes it unrunnable is the engine
    currently selected. That is also why the code is stable and the feature is
    NAMED: the frontend disables the controls it knows about up front
    (``unsupported_features`` on GET /models), and this is the server-side
    backstop for everything that still slips through — a stale page, a script,
    the MCP server, a base-model switch between page load and submit.

    THE MESSAGE IS BILINGUAL for the same reason as :func:`pipeline_loading`:
    when a job path re-wraps this as ``generation_failed(detail=str(exc))``,
    only the MESSAGE survives into what the operator reads.
    """
    return APIError(
        "FEATURE_UNSUPPORTED",
        f"'{feature}' is not supported by the selected base model "
        f"(選択中のベースモデルでは使えない機能です)",
        422,
        detail=detail,
    )


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
