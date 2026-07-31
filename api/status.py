"""GET /status and GET /config (spec 7.4 / 5.2)."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.context import AppContext
from api.deps import get_context
from services import gpu_info

router = APIRouter()

VERSION = "0.4.0"


@router.get("/status")
def get_status(context: AppContext = Depends(get_context)) -> dict:
    pm = context.pipeline_manager
    return {
        "server": "running",
        "version": VERSION,
        "host": context.runtime.host,
        "port": context.runtime.port,
        "pipeline_loaded": pm.loaded,
        "pipeline_type": pm.pipeline_type if pm.loaded else None,
        "gpu": gpu_info.get_gpu_info(),
        "vram_optimization": pm.vram_status_block(),
        # Acceleration capability (ADDITIVE, top level — the FROZEN
        # vram_optimization block above is deliberately left untouched):
        # which attention backends this build understands, and whether the
        # non-default one can actually run here. See
        # PipelineManager.acceleration_status_block for the truth table.
        "acceleration": pm.acceleration_status_block(),
        "queue": {
            "mode": "single_job_in_memory",
            **context.job_store.counts(),
        },
    }


@router.get("/config")
def get_config(context: AppContext = Depends(get_context)) -> dict:
    """Expose the effective configuration (spec 5.2)."""
    return context.config.model_dump()
