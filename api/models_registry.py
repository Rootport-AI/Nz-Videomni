"""GET /models — base-model / category-scoped model enumeration.

Additive endpoint (frozen API is extended, never changed): lists, per category
(``services.model_registry.CATEGORIES``), the selectable model names the
server knows about — the injected "default", explicit config registrations,
and files discovered by scanning the layout each base-model descriptor
declares. Rescans on every call so a newly downloaded file appears without a
restart.

``active`` is the NAME used by the last successful pipeline load per category
("default" until an explicit selection succeeds). It is retained while the
worker is unloaded — whether anything is loaded right now is GET /status's
``pipeline_loaded``, not this endpoint's business (design ruling:
Docs/MODEL_MANAGEMENT_DESIGN.md §9, item 6).

MULTI-ENGINE (Docs/MULTI_ENGINE_DESIGN.md §5.1): the response carries a
base-model layer next to the top-level ``categories`` block, which keeps its
two-layer shape and describes the ACTIVE base model. Alongside it:

``active_base_model``
    Id of the base model the ``categories`` block describes — the one the
    pipeline is on (``POST /pipeline/load``'s ``base_model`` axis moves it,
    and the runtime state restores it at startup).
``base_models[]``
    Every declared base model with its own three-layer listing, plus install
    state: ``installed`` (every category's default file is on disk),
    ``present`` (at least one is — i.e. a partial install worth showing),
    ``missing_categories`` (which ones are not), ``category_order`` (the
    descriptor's declaration order as an ARRAY — see below) and
    ``unsupported_features`` (feature names that base model's engine cannot
    run, straight from that engine adapter's own ``UNSUPPORTED_FEATURES``;
    see Docs/MULTI_ENGINE_DESIGN.md §5.6), and ``transformer_weight_class``
    (``"4bit"``/``"8bit"``/``"q6k"``/``null``: the weight class of the
    selected transformer file, the key a client matches against
    ``limits.comfort_budgets`` rows' ``requires.weight_class``). Only the
    ACTIVE base model gets a value; every other one, and any file that cannot
    be classified, is ``null``.

CATEGORY ORDER IS CARRIED BY AN ARRAY, NOT BY OBJECT KEY ORDER. Both
``categories`` blocks are emitted in declaration order and Python dicts keep
it, but a JSON OBJECT's key order is not something a client can rely on after
transport: the WebUI reaches this response through the AviUtl2 plugin's
WebView2 message channel, and the order the descriptor declares has been seen
to come out alphabetised on the other side (the Settings dropdowns rendered
audio / text_encoder / transformer / video_vae). A JSON ARRAY has no such
ambiguity — every transport preserves element order — so ``category_order``
is the authoritative display order and the object keys are a convenience.

Clients that read only the top-level ``categories`` block keep working, by
construction.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.context import AppContext
from api.deps import get_context
from api.errors import APIError
from services import engines
from services.model_registry import CATEGORIES, DEFAULT_NAME, ModelRegistry
from services.weight_class import classify_transformer

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


def _transformer_weight_class(
    registry: ModelRegistry, base_id: str, selection: dict[str, str]
) -> str | None:
    """Weight class of the transformer ``selection`` names for ``base_id``.

    ``None`` when the name does not resolve (unknown name, missing file, no
    transformer category) or the file cannot be classified.
    """
    name = selection.get("transformer", DEFAULT_NAME)
    try:
        path = registry.resolve("transformer", name, base_model=base_id)
    except APIError:
        return None
    return classify_transformer(path)


@router.get("/models")
def list_models(context: AppContext = Depends(get_context)) -> dict:
    registry = context.model_registry
    registry.rescan()
    # The PIPELINE owns which base model is active; the registry keeps a
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
                # Feature names this base model's ENGINE cannot run.
                # Purely ADDITIVE and per-base-model: the content is owned
                # by each adapter's own UNSUPPORTED_FEATURES, not by this
                # endpoint. A client that has never heard of this key is
                # unaffected either way. The frontend greys out the controls it
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
                # Weight class of the selected transformer (see the module
                # docstring). Only the active base model has a selection the
                # server actually holds; the others stay null rather than
                # guessing from their default file.
                "transformer_weight_class": (
                    _transformer_weight_class(registry, base_id, base_active)
                    if is_active
                    else None
                ),
                "categories": {
                    category: _category_block(registry, category, base_id, base_active)
                    for category in descriptor.categories
                },
            }
        )

    return {
        # Two-layer block for the active base model: the fixed CATEGORIES
        # order (not the descriptor's) so the shape cannot drift from the
        # other modules that import that literal.
        "categories": {
            category: _category_block(registry, category, active_base, active)
            for category in CATEGORIES
        },
        "active_base_model": active_base,
        "base_models": base_models,
    }
