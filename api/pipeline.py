"""POST /pipeline/load and /pipeline/unload (spec 5.2)."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.context import AppContext
from api.deps import get_context, require_auth
from api.errors import job_busy

router = APIRouter()


@router.post("/pipeline/load", dependencies=[Depends(require_auth)])
def load_pipeline(context: AppContext = Depends(get_context)) -> dict:
    pm = context.pipeline_manager
    pm.load()  # raises PIPELINE_LOAD_FAILED (503) on failure
    return {"pipeline_loaded": pm.loaded, "pipeline_type": pm.pipeline_type, "state": pm.state}


@router.post("/pipeline/unload", dependencies=[Depends(require_auth)])
def unload_pipeline(context: AppContext = Depends(get_context)) -> dict:
    if context.job_store.has_active():
        raise job_busy(detail="cannot unload while a job is running")
    pm = context.pipeline_manager
    pm.unload()
    return {"pipeline_loaded": pm.loaded, "state": pm.state}
