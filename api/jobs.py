"""Job endpoints: list / get / video / cancel-delete (spec §6.1 / §7), the
ADDITIVE V2V join pair (POST /jobs/{id}/join, GET /jobs/{id}/joined) and the
ADDITIVE Alpha Gen matte download (GET /jobs/{id}/matte) and the ADDITIVE
Alpha Gen exit (POST /jobs/{id}/alpha/rgba | chroma and their downloads)."""

from __future__ import annotations

import shutil

from fastapi import APIRouter, Body, Depends
from fastapi.responses import FileResponse

from api.context import AppContext
from api.deps import get_context, require_auth
from api.errors import job_not_found, video_not_ready
from api.models import (
    AlphaChromaColor,
    AlphaChromaRequest,
    AlphaOutputResponse,
    AlphaRgbaRequest,
    JobResponse,
    JobStatus,
    JoinRequest,
    JoinResponse,
)

router = APIRouter()


@router.get("/jobs", response_model=list[JobResponse])
def list_jobs(context: AppContext = Depends(get_context)) -> list[JobResponse]:
    return [
        r.to_response(context.config.output_dir) for r in context.job_store.list()
    ]


@router.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str, context: AppContext = Depends(get_context)) -> JobResponse:
    record = context.job_store.get(job_id)
    if record is None:
        raise job_not_found(job_id)
    return record.to_response(context.config.output_dir)


@router.get("/jobs/{job_id}/video")
def get_job_video(job_id: str, context: AppContext = Depends(get_context)) -> FileResponse:
    record = context.job_store.get(job_id)
    if record is None:
        raise job_not_found(job_id)
    if record.status != JobStatus.completed:
        raise video_not_ready(job_id)
    video_path = context.config.output_dir / job_id / "output.mp4"
    if not video_path.exists():
        raise job_not_found(job_id)
    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=f"{job_id}.mp4",
    )


@router.get("/jobs/{job_id}/matte")
def get_job_matte(job_id: str, context: AppContext = Depends(get_context)) -> FileResponse:
    """Download the lossless grey matte (``matte.mkv``, FFV1 gray, full range)
    a POST /generate/alpha job writes next to its ``output.mp4``. Mirrors
    GET /jobs/{id}/video: unknown job or no matte on disk -> 404, not yet
    completed -> 409 VIDEO_NOT_READY."""
    record = context.job_store.get(job_id)
    if record is None:
        raise job_not_found(job_id)
    if record.status != JobStatus.completed:
        raise video_not_ready(job_id)
    matte_path = context.config.output_dir / job_id / "matte.mkv"
    if not matte_path.exists():
        raise job_not_found(job_id)
    return FileResponse(
        matte_path,
        media_type="video/x-matroska",
        filename=f"{job_id}_matte.mkv",
    )


@router.post(
    "/jobs/{job_id}/join",
    response_model=JoinResponse,
    dependencies=[Depends(require_auth)],
)
def join_job(
    job_id: str,
    request: JoinRequest | None = Body(default=None),
    context: AppContext = Depends(get_context),
) -> JoinResponse:
    """Server-side V2V join (ADDITIVE endpoint): source video + continuation ->
    ``joined.mp4``. Synchronous 200 — ffmpeg only, no GPU, independent of the
    single-job guard (FastAPI runs sync endpoints on the thread pool). The body
    is optional; ``{}`` (or none) joins with the ``JoinRequest`` defaults."""
    return context.join_manager.join(job_id, request or JoinRequest())


@router.get("/jobs/{job_id}/joined")
def get_job_joined(job_id: str, context: AppContext = Depends(get_context)) -> FileResponse:
    """Download a previously-joined ``joined.mp4`` (404 JOINED_NOT_READY before
    a successful POST /jobs/{id}/join). Mirrors GET /jobs/{id}/video."""
    path = context.join_manager.joined_path(job_id)
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=f"{job_id}_joined.mp4",
    )


@router.post(
    "/jobs/{job_id}/alpha/rgba",
    response_model=AlphaOutputResponse,
    dependencies=[Depends(require_auth)],
)
def make_alpha_rgba(
    job_id: str,
    request: AlphaRgbaRequest | None = Body(default=None),
    context: AppContext = Depends(get_context),
) -> AlphaOutputResponse:
    """Alpha Gen exit (ADDITIVE endpoint): window + levelled matte ->
    ``rgba.mov`` (ProRes 4444) and ``alpha_preview.mp4`` (checkerboard).
    Synchronous 200 -- ffmpeg only, no GPU, independent of the single-job
    guard. Reused (``reused: true``) when both exist with the same thresholds.
    The body is optional; ``{}`` (or none) uses the default thresholds."""
    return context.alpha_output_manager.rgba(job_id, request or AlphaRgbaRequest())


@router.post(
    "/jobs/{job_id}/alpha/chroma",
    response_model=AlphaOutputResponse,
    dependencies=[Depends(require_auth)],
)
def make_alpha_chroma(
    job_id: str,
    request: AlphaChromaRequest,
    context: AppContext = Depends(get_context),
) -> AlphaOutputResponse:
    """Alpha Gen exit (ADDITIVE endpoint): window + levelled matte over a solid
    colour -> ``chroma_<color>.mp4``. Synchronous 200 like the rgba route; one
    file per colour, reused when it exists with the same thresholds. ``color``
    is required."""
    return context.alpha_output_manager.chroma(job_id, request)


@router.get("/jobs/{job_id}/alpha/rgba")
def get_alpha_rgba(job_id: str, context: AppContext = Depends(get_context)) -> FileResponse:
    """Download ``rgba.mov`` (404 ALPHA_OUTPUT_NOT_READY before a successful
    POST /jobs/{id}/alpha/rgba). Mirrors GET /jobs/{id}/joined."""
    path = context.alpha_output_manager.rgba_path(job_id)
    return FileResponse(
        path,
        media_type="video/quicktime",
        filename=f"{job_id}_rgba.mov",
    )


@router.get("/jobs/{job_id}/alpha/preview")
def get_alpha_preview(job_id: str, context: AppContext = Depends(get_context)) -> FileResponse:
    """Download ``alpha_preview.mp4``, the checkerboard preview made with
    ``rgba.mov`` (404 ALPHA_OUTPUT_NOT_READY before it exists)."""
    path = context.alpha_output_manager.preview_path(job_id)
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=f"{job_id}_alpha_preview.mp4",
    )


@router.get("/jobs/{job_id}/alpha/chroma/{color}")
def get_alpha_chroma(
    job_id: str,
    color: AlphaChromaColor,
    context: AppContext = Depends(get_context),
) -> FileResponse:
    """Download ``chroma_<color>.mp4`` (404 ALPHA_OUTPUT_NOT_READY before a
    successful POST /jobs/{id}/alpha/chroma with that colour; a name outside
    ``ALPHA_CHROMA_COLORS`` is 422)."""
    path = context.alpha_output_manager.chroma_path(job_id, color)
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=f"{job_id}_chroma_{color}.mp4",
    )


@router.delete("/jobs/{job_id}", dependencies=[Depends(require_auth)])
def delete_job(job_id: str, context: AppContext = Depends(get_context)) -> dict:
    record = context.job_store.get(job_id)
    if record is None:
        raise job_not_found(job_id)

    if record.is_active:
        # Not started yet: cancel in place so the single-job guard frees up
        # immediately (a queued job stuck behind the worker would otherwise
        # block the next /generate until the server restarts). The transition
        # is a compare-and-set under the store lock, mutually exclusive with
        # the worker's queued -> running promotion (JobStore.start_job) — a
        # cancelled job can never be resurrected into a running one.
        if context.job_store.cancel_if_queued(record):
            return {"job_id": job_id, "cancelled": True, "status": record.status.value}
        # Running (or won the race to running): inference cannot be safely
        # interrupted mid-flight, so this is best-effort — the worker checks
        # cancel_requested where it can.
        record.cancel_requested = True
        return {"job_id": job_id, "cancel_requested": True, "status": record.status.value}

    # Terminal job: drop record and remove its output directory.
    context.job_store.remove(job_id)
    out_dir = context.config.output_dir / job_id
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    return {"job_id": job_id, "deleted": True}
