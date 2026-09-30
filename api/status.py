"""GET /status and GET /config (spec §6.1)."""

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
        # ADDITIVE. ``pipeline_loaded`` is a bool and therefore cannot express
        # the state a base-model switch spends minutes in: "loading".
        # ``state`` is the full lifecycle value (PipelineManager's STATE_*
        # constants) a client needs to show a progress state and to keep its
        # Load button disabled meanwhile. ``base_model`` is the descriptor id
        # in effect — retained across an unload, like GET /models' ``active``,
        # because it is a SELECTION, not a load state.
        "state": pm.state,
        "base_model": pm.active_base_model,
        "gpu": gpu_info.get_gpu_info(),
        "vram_optimization": pm.vram_status_block(),
        # Acceleration capability (ADDITIVE, top level — the FROZEN
        # vram_optimization block above is deliberately left untouched):
        # which attention backends this build understands, and whether the
        # non-default one can actually run here. See
        # PipelineManager.acceleration_status_block for the truth table.
        "acceleration": pm.acceleration_status_block(),
        # Object tracking, ADDITIVE and deliberately its own block:
        # {"available": true} or {"available": false, "reason": ...}, the
        # reason being one of services.tracking_manager's REASON_* constants.
        # It is NOT folded into ``queue`` or ``state`` because tracking takes
        # no queue slot and has no bearing on the pipeline's lifecycle — the
        # frontend reads it to grey out the Toolbox tracking controls and to
        # refuse the tracking right-click.
        "tracking": context.tracking_manager.status_block(),
        "queue": {
            "mode": "single_job_in_memory",
            **context.job_store.counts(),
        },
    }


@router.get("/config")
def get_config(context: AppContext = Depends(get_context)) -> dict:
    """Expose the effective configuration (spec §6.1)."""
    return context.config.model_dump()
