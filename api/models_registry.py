"""GET /models — base-model / category-scoped model enumeration.

Additive endpoint (frozen API is extended, never changed): lists, per category
(transformer / text_encoder / video_vae / audio), the selectable model names
the server knows about — the injected "default", explicit config registrations,
and files discovered by scanning the layout each base-model descriptor
declares. Rescans on every call so a newly downloaded file appears without a
restart.

``active`` is the NAME used by the last successful pipeline load per category
("default" until an explicit selection succeeds). It is retained while the
worker is unloaded — whether anything is loaded right now is GET /status's
``pipeline_loaded``, not this endpoint's business (design ruling §9-6).

MULTI-ENGINE (P3a, Docs/MULTI_ENGINE_DESIGN.md §5.1): the response gained a
base-model layer WITHOUT reshaping anything that existed. The top-level
``categories`` block is preserved verbatim — same keys, same order, same values
— and describes the ACTIVE base model, exactly as it always described the only
one. Added alongside it:

``active_base_model``
    Id of the base model the ``categories`` block describes — the one the
    pipeline is on (P6: ``POST /pipeline/load``'s ``base_model`` axis moves it,
    and the runtime state restores it at startup).
``base_models[]``
    Every declared base model with its own three-layer listing, plus install
    state: ``installed`` (every category's default file is on disk),
    ``present`` (at least one is — i.e. a partial install worth showing),
    ``missing_categories`` (which ones are not), ``category_order`` (the
    descriptor's declaration order as an ARRAY — see below) and
    ``unsupported_features`` (§3-98 P5: feature names that base model's engine
    cannot run — ``[]`` for LTX 2.3, so nothing about the 2.3 response changed).

CATEGORY ORDER IS CARRIED BY AN ARRAY, NOT BY OBJECT KEY ORDER. Both
``categories`` blocks are emitted in declaration order and Python dicts keep
it, but a JSON OBJECT's key order is not something a client can rely on after
transport: the WebUI reaches this response through the AviUtl2 plugin's
WebView2 message channel, and the order the descriptor declares came out
alphabetised on the other side (2026-08-20 owner sighting: the Settings
dropdowns rendered audio / text_encoder / transformer / video_vae). A JSON
ARRAY has no such ambiguity — every transport preserves element order — so
``category_order`` is the authoritative display order and the object keys are
left as the convenience they always were.

Clients that only know the old shape (gradio_ui/adapters.py) keep working
unchanged, by construction.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.context import AppContext
from api.deps import get_context
from services import engines
from services.model_registry import CATEGORIES, DEFAULT_NAME, ModelRegistry

router = APIRouter()


def _category_block(
    registry: ModelRegistry, category: str, base_model: str, active: dict[str, str]
) -> dict:
    return {
        "default": DEFAULT_NAME,
        "active": active.get(category, DEFAULT_NAME),
        "entries": [
            e.as_dict() for e in registry.entries(category, base_model=base_model)
        ],
    }


@router.get("/models")
def list_models(context: AppContext = Depends(get_context)) -> dict:
    registry = context.model_registry
    registry.rescan()
    # The PIPELINE owns which base model is active (P6); the registry keeps a
    # copy so base-less calls resolve consistently. Read from the pipeline here
    # so the listing can never lag a switch by one request.
    active_base = context.pipeline_manager.active_base_model
    active = context.pipeline_manager.active_models

    base_models = []
    for base_id in registry.base_model_ids:
        is_active = base_id == active_base
        descriptor = registry.descriptor(base_id)
        presence = registry.default_file_presence(base_model=base_id)
        # Only the active base model has a live selection; every other one is
        # listed at its defaults until it is actually loaded.
        base_active = active if is_active else {}
        base_models.append(
            {
                "id": base_id,
                "display_name": descriptor.display_name,
                "engine_family": descriptor.engine_family,
                "active": is_active,
                "installed": all(presence.values()),
                "present": any(presence.values()),
                "missing_categories": [c for c, ok in presence.items() if not ok],
                # Feature names this base model's ENGINE cannot run (§3-98 P5).
                # Purely ADDITIVE and per-base-model: LTX 2.3 declares none, so
                # its entry is `[]` and a client that has never heard of this
                # key is unaffected. The frontend greys out the controls it
                # recognises; POST /generate refuses the rest with the matching
                # FEATURE_UNSUPPORTED, so this list is a courtesy, never the
                # enforcement (a page can be stale, a script never asked).
                "unsupported_features": list(
                    engines.unsupported_features(descriptor.engine_family)
                ),
                # The display order, as an array (see the module docstring on
                # why object key order is not trusted to survive transport).
                # Built from the descriptor's declaration order, which is the
                # single source of truth for it: scripts/manifests/<base>.json's
                # `categories` key order, nothing else.
                "category_order": list(descriptor.categories),
                "categories": {
                    category: _category_block(registry, category, base_id, base_active)
                    for category in descriptor.categories
                },
            }
        )

    return {
        # Legacy two-layer block, byte-for-byte as before: the fixed CATEGORIES
        # order (not the descriptor's) so the shape cannot drift from the three
        # modules that import that literal.
        "categories": {
            category: _category_block(registry, category, active_base, active)
            for category in CATEGORIES
        },
        "active_base_model": active_base,
        "base_models": base_models,
    }
