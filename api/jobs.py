"""Job endpoints: list / get / video / cancel-delete (spec 5.2 / 8.2) and the
ADDITIVE V2V join pair (POST /jobs/{id}/join, GET /jobs/{id}/joined)."""

from __future__ import annotations

import shutil

from fastapi import APIRouter, Body, Depends
from fastapi.responses import FileResponse

from api.context import AppContext
from api.deps import get_context, require_auth
from api.errors import job_not_found, video_not_ready
from api.models import JobResponse, JobStatus, JoinRequest, JoinResponse

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
    is optional; ``{}`` (or none) gives the default smoothed join."""
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
        # interrupted mid-flight (Phase 1), so this stays best-effort — the
        # worker checks cancel_requested where it can.
        record.cancel_requested = True
        return {"job_id": job_id, "cancel_requested": True, "status": record.status.value}

    # Terminal job: drop record and remove its output directory.
    context.job_store.remove(job_id)
    out_dir = context.config.output_dir / job_id
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    return {"job_id": job_id, "deleted": True}
