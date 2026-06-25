"""Job endpoints: list / get / video / cancel-delete (spec 5.2 / 8.2)."""

from __future__ import annotations

import shutil

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse

from api.context import AppContext
from api.deps import get_context, require_auth
from api.errors import job_not_found, video_not_ready
from api.models import JobResponse, JobStatus

router = APIRouter()


@router.get("/jobs", response_model=list[JobResponse])
def list_jobs(context: AppContext = Depends(get_context)) -> list[JobResponse]:
    return [r.to_response() for r in context.job_store.list()]


@router.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str, context: AppContext = Depends(get_context)) -> JobResponse:
    record = context.job_store.get(job_id)
    if record is None:
        raise job_not_found(job_id)
    return record.to_response()


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


@router.delete("/jobs/{job_id}", dependencies=[Depends(require_auth)])
def delete_job(job_id: str, context: AppContext = Depends(get_context)) -> dict:
    record = context.job_store.get(job_id)
    if record is None:
        raise job_not_found(job_id)

    if record.is_active:
        # Best-effort cancel: inference cannot be safely interrupted in Phase 1.
        record.cancel_requested = True
        return {"job_id": job_id, "cancel_requested": True, "status": record.status.value}

    # Terminal job: drop record and remove its output directory.
    context.job_store.remove(job_id)
    out_dir = context.config.output_dir / job_id
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    return {"job_id": job_id, "deleted": True}
