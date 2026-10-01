"""POST /pipeline/load and /pipeline/unload (spec §6.1 / §6.9).

Model management (additive): /pipeline/load accepts an OPTIONAL body with a
``models`` block mapping a category (GET /models) to a registered model NAME.
No body / no block loads the current active selection (at startup, the one
remembered in the runtime state file, or the defaults when there is none). A
selection differing from the live one forces a worker rebuild (unload ->
load); the same selection is a no-op. Swap failures do NOT fall back (design
ruling: Docs/MODEL_MANAGEMENT_DESIGN.md §9, item 1).

MULTI-ENGINE (Docs/MULTI_ENGINE_DESIGN.md §6.2): the body's ``base_model``
field is the OTHER axis. ``models`` picks a file WITHIN a base model;
``base_model`` picks the base model itself (LTX 2.3 / LTX 2.5 / ...), and the
two can arrive together or alone.

WHY A BASE-MODEL CHANGE PRECHECKS EVERY CATEGORY, INCLUDING THE DEFAULTS. On
an unchanged base, a category on ``"default"`` is deliberately left out of the
selection: "no override" is what keeps an all-default load's worker payload
equal to the snapshot ``tests/test_model_swap_load.py`` fixes
(``test_load_payload_byte_identical_without_selection``). But when the base
model CHANGES, every default is a different file, and skipping it would hand
the engine the new base model's weights without ever having looked inside
them — which is exactly where the KV ruling (``services.engines.check_kv``:
does this file's engine FAMILY match the base model being selected, and can
that family run this generation?) has to fire. So a base change resolves,
prechecks and rules on every category the target base model's descriptor
declares, and passes them all as explicit paths.

The price (recorded in Docs/MODEL_MANAGEMENT_DESIGN.md §9): after a
base-model change the selection carries explicit paths even for
``"default"`` names, and a later body-less load reuses it, so its
payload has the same VALUES as an all-default one but not the same BYTES.
The NAMES stay ``"default"`` throughout, so nothing a client displays — nor
metadata.json — changes.
"""

from __future__ import annotations

from fastapi import APIRouter, Body, Depends
from pydantic import BaseModel

from api.context import AppContext
from api.deps import get_context, require_auth
from api.errors import job_busy, model_not_found
from services import engines
from services.model_registry import DEFAULT_NAME, precheck_model_file

router = APIRouter()


class LoadPipelineRequest(BaseModel):
    """Optional body for POST /pipeline/load (additive).

    ``models``: category -> registered NAME (never a filesystem path). Absent
    categories keep the current active selection while the base model stays
    the same, so a partial block only swaps what it names; after a base-model
    change they start from that base model's defaults.

    ``base_model``: descriptor id from ``GET /models``' ``base_models[]``.
    Absent keeps the current base model, so a client that never sends it sees
    no change at all.
    """

    models: dict[str, str] | None = None
    base_model: str | None = None


@router.post("/pipeline/load", dependencies=[Depends(require_auth)])
def load_pipeline(
    body: LoadPipelineRequest | None = Body(default=None),
    context: AppContext = Depends(get_context),
) -> dict:
    pm = context.pipeline_manager
    requested = (body.models if body is not None else None) or {}
    requested_base = body.base_model if body is not None else None

    if not requested and requested_base is None:
        # Bodyless (or empty) load. The response carries no ``models`` key and
        # no ``base_model`` key, so a client that sends neither field sees only
        # the keys below (test_a_bodyless_load_is_unaffected_by_the_new_axis
        # fixes the key set).
        pm.load()  # raises PIPELINE_LOAD_FAILED (503) / PIPELINE_LOADING (409)
        return {"pipeline_loaded": pm.loaded, "pipeline_type": pm.pipeline_type, "state": pm.state}

    registry = context.model_registry
    # Unknown base model -> 404 MODEL_NOT_FOUND, naming the known ids (the
    # registry's own _base_id raises exactly that). Done FIRST: which
    # categories are legal is a question about this base model, not about the
    # one that happens to be active.
    effective_base = requested_base if requested_base is not None else pm.active_base_model
    descriptor = registry.descriptor(effective_base)
    base_changed = effective_base != pm.active_base_model
    categories = tuple(descriptor.categories)

    for category, name in requested.items():
        if category not in categories:
            raise model_not_found(
                category,
                name,
                detail=(
                    f"unknown category for base model '{effective_base}'; "
                    f"known: {list(categories)}"
                ),
            )

    if context.job_store.has_active():
        raise job_busy(detail="cannot swap models while a job is running")

    registry.rescan()  # a just-downloaded file must be selectable without a restart

    # Effective NAME per category: requested wins, else the current active —
    # but only while the BASE MODEL is unchanged. A different base model has a
    # different set of registered names, so carrying the old ones over would
    # produce a 404 for a selection the user never made; its own defaults are
    # the right starting point.
    carried = pm.active_models if not base_changed else {}
    effective = {c: requested.get(c, carried.get(c, DEFAULT_NAME)) for c in categories}

    selection: dict[str, str] = {}
    for category, name in effective.items():
        if name == DEFAULT_NAME and not base_changed:
            # "default" on the SAME base model means "no payload override",
            # which keeps an all-default load's worker payload equal to the
            # snapshot tests/test_model_swap_load.py fixes.
            continue
        path = registry.resolve(category, name, base_model=effective_base)
        # The category descriptor of the TARGET base model states which
        # extensions this category accepts, and the KV dict read on the way
        # back (from a GGUF header, or in the same keys from a quantized
        # safetensors transformer's header) is what the engine-generation
        # ruling judges on.
        kv = precheck_model_file(  # MODEL_INCOMPATIBLE (422)
            category,
            name,
            path,
            descriptor=descriptor.categories.get(category),
        )
        # The KV ruling (Docs/MULTI_ENGINE_DESIGN.md §2.1 / §2.3). ``engines.check_kv``
        # first rules on WHICH ENGINE the file belongs to — the KV is the judge,
        # the chosen base model is what it is judged against — and refuses a
        # mismatch with a message naming the base model to pick instead. Only
        # then does the family's own adapter rule on architecture + generation.
        # The dispatcher imports adapters lazily, so this endpoint still pulls
        # in exactly one engine: the one being loaded.
        engines.check_kv(descriptor, category, name, kv)
        selection[category] = str(path)

    if pm.loaded and not base_changed and effective == pm.active_models:
        # Already live: do NOT restart the worker for an identical selection.
        return _response(pm)

    if pm.loaded:
        pm.reload(selection, effective, base_model=effective_base)  # forced rebuild
    else:
        pm.load(selection=selection, active_names=effective, base_model=effective_base)
    return _response(pm)


def _response(pm) -> dict:
    return {
        "pipeline_loaded": pm.loaded,
        "pipeline_type": pm.pipeline_type,
        "state": pm.state,
        "base_model": pm.active_base_model,
        "models": dict(pm.active_models),
    }


@router.post("/pipeline/unload", dependencies=[Depends(require_auth)])
def unload_pipeline(context: AppContext = Depends(get_context)) -> dict:
    if context.job_store.has_active():
        raise job_busy(detail="cannot unload while a job is running")
    pm = context.pipeline_manager
    pm.unload()
    return {"pipeline_loaded": pm.loaded, "state": pm.state}
