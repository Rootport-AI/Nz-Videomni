"""GET /models — category-scoped model enumeration (model management S1).

Additive endpoint (frozen API is extended, never changed): lists, per fixed
category (transformer / text_encoder / video_vae / audio), the selectable model
names the server knows about — the injected "default", explicit config
registrations, and files discovered by scanning the existing models/ layout.
Rescans on every call so a newly downloaded file appears without a restart.

``active`` is the NAME used by the last successful pipeline load per category
("default" until an explicit selection succeeds). It is retained while the
worker is unloaded — whether anything is loaded right now is GET /status's
``pipeline_loaded``, not this endpoint's business (design ruling §9-6).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.context import AppContext
from api.deps import get_context
from services.model_registry import CATEGORIES, DEFAULT_NAME

router = APIRouter()


@router.get("/models")
def list_models(context: AppContext = Depends(get_context)) -> dict:
    registry = context.model_registry
    registry.rescan()
    active = context.pipeline_manager.active_models
    return {
        "categories": {
            category: {
                "default": DEFAULT_NAME,
                "active": active.get(category, DEFAULT_NAME),
                "entries": [e.as_dict() for e in registry.entries(category)],
            }
            for category in CATEGORIES
        }
    }
