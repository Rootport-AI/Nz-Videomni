"""POST /pipeline/load and /pipeline/unload (spec 5.2).

Model management (additive): /pipeline/load accepts an OPTIONAL body with a
``models`` block mapping a category (GET /models) to a registered model NAME.
No body / no block keeps the legacy behavior byte-identical (the current
active selection is loaded — all-default on boot). A selection differing from
the live one forces a worker rebuild (unload -> load); the same selection is a
no-op. Swap failures do NOT fall back (design ruling §9-1).
"""

from __future__ import annotations

from fastapi import APIRouter, Body, Depends
from pydantic import BaseModel

from api.context import AppContext
from api.deps import get_context, require_auth
from api.errors import job_busy, model_not_found
from services.model_registry import CATEGORIES, DEFAULT_NAME, precheck_model_file

router = APIRouter()


class LoadPipelineRequest(BaseModel):
    """Optional body for POST /pipeline/load (additive).

    ``models``: category -> registered NAME (never a filesystem path). Absent
    categories keep the current active selection, so a partial block only
    swaps what it names.
    """

    models: dict[str, str] | None = None


@router.post("/pipeline/load", dependencies=[Depends(require_auth)])
def load_pipeline(
    body: LoadPipelineRequest | None = Body(default=None),
    context: AppContext = Depends(get_context),
) -> dict:
    pm = context.pipeline_manager
    requested = (body.models if body is not None else None) or {}

    if not requested:
        # Legacy path: bodyless (or empty) load. Response shape unchanged.
        pm.load()  # raises PIPELINE_LOAD_FAILED (503) on failure
        return {"pipeline_loaded": pm.loaded, "pipeline_type": pm.pipeline_type, "state": pm.state}

    for category, name in requested.items():
        if category not in CATEGORIES:
            raise model_not_found(
                category, name, detail=f"unknown category; known: {list(CATEGORIES)}"
            )

    if context.job_store.has_active():
        raise job_busy(detail="cannot swap models while a job is running")

    registry = context.model_registry
    registry.rescan()  # a just-downloaded file must be selectable without a restart

    # Effective NAME per category: requested wins, else the current active.
    effective = {
        c: requested.get(c, pm.active_models.get(c, DEFAULT_NAME)) for c in CATEGORIES
    }
    # Resolve only non-default names: "default" means "use the config default
    # field", i.e. NO payload override — that keeps the all-default load
    # byte-identical to the legacy one (golden-snapshot guarantee).
    selection: dict[str, str] = {}
    for category, name in effective.items():
        if name == DEFAULT_NAME:
            continue
        path = registry.resolve(category, name)  # MODEL_NOT_FOUND / MODEL_FILE_MISSING
        # The category descriptor of the ACTIVE base model states which
        # extensions this category accepts. The returned GGUF KV metadata is
        # what the engine-generation ruling (check_kv) will judge on once the
        # base-model axis reaches this endpoint (§3-97 P6); until then the
        # precheck's own structural verdict is all this call needs.
        precheck_model_file(  # MODEL_INCOMPATIBLE (422)
            category,
            name,
            path,
            descriptor=registry.descriptor().categories.get(category),
        )
        selection[category] = str(path)

    if pm.loaded and effective == pm.active_models:
        # Already live: do NOT restart the worker for an identical selection.
        return {
            "pipeline_loaded": pm.loaded,
            "pipeline_type": pm.pipeline_type,
            "state": pm.state,
            "models": dict(pm.active_models),
        }

    if pm.loaded:
        pm.reload(selection, effective)  # swap: unload -> forced rebuild
    else:
        pm.load(selection=selection, active_names=effective)
    return {
        "pipeline_loaded": pm.loaded,
        "pipeline_type": pm.pipeline_type,
        "state": pm.state,
        "models": dict(pm.active_models),
    }


@router.post("/pipeline/unload", dependencies=[Depends(require_auth)])
def unload_pipeline(context: AppContext = Depends(get_context)) -> dict:
    if context.job_store.has_active():
        raise job_busy(detail="cannot unload while a job is running")
    pm = context.pipeline_manager
    pm.unload()
    return {"pipeline_loaded": pm.loaded, "state": pm.state}
