"""Routes for /api/queue — generation queue management."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api_types import GenerateVideoRequest
from app_handler import AppHandler
from state import get_state_service

router = APIRouter(prefix="/api", tags=["queue"])


@router.post("/queue/add")
def route_queue_add(
    req: GenerateVideoRequest,
    handler: AppHandler = Depends(get_state_service),
) -> dict:
    """POST /api/queue/add — enqueue a video generation job."""
    # Snapshot the current LoRA config so that changes made after this call
    # don't affect the job when it eventually runs.
    civitai_loras_now = handler.state.app_settings.civitai_loras
    job = handler.queue.add_job(req, civitai_loras=civitai_loras_now)
    return {"id": job.id, "status": job.status, "created_at": job.created_at}


@router.get("/queue")
def route_queue_list(
    handler: AppHandler = Depends(get_state_service),
) -> dict:
    """GET /api/queue — list all queued jobs."""
    jobs = handler.queue.get_jobs()
    return {
        "jobs": [
            {
                "id": j.id,
                "status": j.status,
                "result_path": j.result_path,
                "error": j.error,
                "created_at": j.created_at,
                "prompt": j.request.prompt,
                "resolution": j.request.resolution,
                "duration": j.request.duration,
                "fps": j.request.fps,
                "model": j.request.model,
                "aspect_ratio": j.request.aspectRatio,
                "civitai_loras_snapshot": j.civitai_loras_snapshot,
            }
            for j in jobs
        ]
    }


@router.delete("/queue/{job_id}")
def route_queue_remove(
    job_id: str,
    handler: AppHandler = Depends(get_state_service),
) -> dict:
    """DELETE /api/queue/{job_id} — cancel a pending job."""
    success = handler.queue.remove_job(job_id)
    return {"success": success}
