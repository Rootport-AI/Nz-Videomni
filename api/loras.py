"""GET/POST /loras — style/character + control IC-LoRA enumeration (S1).

Additive endpoints (the frozen API is extended, never changed): list the
selectable IC-LoRA adapter names the server knows about — ``config.model.ic_loras``
registrations plus ``*.safetensors`` files discovered by scanning
``config.model.lora_dir`` — with each adapter's ``kind`` (``style`` | ``control``,
so a GUI can group a "Style LoRA" tab), whether it has a sibling ``<stem>.png``
thumbnail, and its source. Rescans on every call so a newly dropped-in file
appears without a restart (mirrors GET /models).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse

from api.context import AppContext
from api.deps import get_context
from api.errors import lora_thumbnail_not_found

router = APIRouter()


@router.get("/loras")
def list_loras(context: AppContext = Depends(get_context)) -> dict:
    registry = context.lora_registry
    registry.rescan()
    return {"loras": [e.as_dict() for e in registry.entries()]}


@router.post("/loras/reload")
def reload_loras(context: AppContext = Depends(get_context)) -> dict:
    """Explicit rescan of the registry (config + lora_dir), returning counts by
    kind so a GUI can confirm what a folder drop-in picked up."""
    registry = context.lora_registry
    registry.rescan()
    entries = registry.entries()
    styles = sum(1 for e in entries if e.kind == "style")
    controls = sum(1 for e in entries if e.kind == "control")
    return {"total": len(entries), "styles": styles, "controls": controls}


@router.get("/loras/{name}/thumbnail")
def get_lora_thumbnail(
    name: str, context: AppContext = Depends(get_context)
) -> FileResponse:
    """Serve the sibling ``<stem>.png`` thumbnail for an adapter, else 404.

    Rescans first so a thumbnail added next to an already-known weight is picked
    up. The registry resolves the (validated, non-path) name to its weight file;
    the thumbnail is that file with a ``.png`` suffix (mirrors GET /jobs/{id}/video).
    """
    registry = context.lora_registry
    registry.rescan()
    entry = context.lora_registry.info(name)  # 404 LORA_NOT_FOUND if unknown
    thumb = entry.path.with_suffix(".png")
    if not entry.has_thumbnail or not thumb.exists():
        raise lora_thumbnail_not_found(name)
    return FileResponse(
        thumb,
        media_type="image/png",
        filename=f"{name}.png",
    )
