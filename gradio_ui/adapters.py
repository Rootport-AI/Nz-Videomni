"""IC-LoRA reference-video control (S4). The Generate tab exposes an adapter
Dropdown whose choices are rebuilt on page load from /config model.ic_loras;
the static list below is the offline fallback (server /config unavailable).
``ADAPTER_NONE`` is the sentinel value meaning "no adapter" (payload omits
loras + reference_video_id entirely). The five known adapter keys get a
friendly label; any unknown registered key is shown as-is.
"""

from __future__ import annotations

import numpy as np

from .i18n import L, _DEFAULT_LANG

ADAPTER_NONE = "__none__"

ADAPTER_FRIENDLY: dict[str, str] = {
    "pixel-spatial-upscaler-x2": "Upscale ×2 (pixel-spatial-upscaler-x2)",
    "canny-control": "Canny edge control (canny-control)",
    "pose-control": "Pose control (pose-control)",
    "depth-control": "Depth control (depth-control)",
    "deblur": "Deblur (deblur)",
}

# Fallbacks used when /config is unavailable (mirrors config.yaml upload.*).
_FALLBACK_VIDEO_EXTS = [".mp4", ".mov", ".webm", ".mkv"]
_FALLBACK_MAX_VIDEO_MB = 200


def build_adapter_choices(config: dict | None, lang: str = _DEFAULT_LANG) -> list[tuple[str, str]]:
    """Build the adapter Dropdown ``choices`` (list of (label, value)).

    "None" is always first (value :data:`ADAPTER_NONE`). The remaining entries
    come from the fetched /config ``model.ic_loras`` keys (value == key); each
    key gets a friendly label when known, else is shown verbatim. Falls back to
    the static known adapters (:data:`ADAPTER_FRIENDLY`) when the server config
    has no ic_loras.
    """
    none_choice = (L("adapter_none", lang), ADAPTER_NONE)
    ic_loras = ((config or {}).get("model") or {}).get("ic_loras") or {}
    if not ic_loras:
        return [none_choice] + [(label, key) for key, label in ADAPTER_FRIENDLY.items()]
    return [none_choice] + [(ADAPTER_FRIENDLY.get(key, key), key) for key in ic_loras]


# --------------------------------------------------------------------------- #
# Model-management dropdowns (Settings tab "Models" section, S3). Same shape
# as the IC-LoRA adapter dropdown above: choices are (label, value) pairs whose
# VALUE is always the server-side registered NAME from GET /models — never a
# filesystem path.
# --------------------------------------------------------------------------- #

#: Fixed category order — must match services.model_registry.CATEGORIES.
MODEL_CATEGORIES = ("transformer", "text_encoder", "video_vae", "audio")

#: The injected per-category default NAME (server: model_registry.DEFAULT_NAME).
MODEL_DEFAULT = "default"


def _path_basename(path: str) -> str:
    """Last path component of ``path``, tolerant of both ``/`` and ``\\``
    separators regardless of the OS this happens to run on (the string comes
    verbatim from the server's GET /models response)."""
    return path.replace("\\", "/").rsplit("/", 1)[-1]


def _category_block(
    models_json: dict | None, category: str, base_model: str | None = None
) -> dict:
    """The ``{default, active, entries}`` block for one category.

    ``base_model=None`` reads the legacy top-level ``categories`` block, which
    always describes the ACTIVE base model (unchanged behaviour for every
    pre-multi-engine caller). A non-empty ``base_model`` instead reads that
    base model's own listing out of ``base_models[]`` — the shape the Settings
    tab needs while the user is browsing a base model that is not loaded yet.
    An unknown id (or a server too old to send ``base_models``) yields ``{}``,
    which the callers below degrade to the "default" fallback.
    """
    root = models_json or {}
    if base_model:
        for entry in root.get("base_models") or []:
            if isinstance(entry, dict) and entry.get("id") == base_model:
                return ((entry.get("categories") or {}).get(category)) or {}
        return {}
    return ((root.get("categories") or {}).get(category)) or {}


def build_base_model_choices(models_json: dict | None) -> list[tuple[str, str]]:
    """Dropdown ``choices`` for the base-model selector: ``(display_name, id)``
    over ``base_models[]``, in server order. The VALUE is the base-model id (the
    string ``POST /pipeline/load`` takes as ``base_model``); the label is the
    descriptor's human display name, which is language-independent — so a
    language switch never has to rebuild these choices.

    A response without ``base_models`` (a server predating the multi-engine
    layer) yields an empty list; the caller leaves its dropdown untouched."""
    choices: list[tuple[str, str]] = []
    for entry in (models_json or {}).get("base_models") or []:
        if not isinstance(entry, dict):
            continue
        base_id = entry.get("id")
        if not base_id:
            continue
        choices.append((entry.get("display_name") or base_id, base_id))
    return choices


def active_base_model(models_json: dict | None) -> str:
    """The id of the base model the pipeline is on (``""`` when the response
    does not carry one)."""
    return (models_json or {}).get("active_base_model") or ""


def build_model_choices(
    models_json: dict | None, category: str, lang: str = _DEFAULT_LANG,
    base_model: str | None = None,
) -> list[tuple[str, str]]:
    """Dropdown ``choices`` for one category from a GET /models response.

    The server already orders entries default-first. An entry whose file is
    absent on disk (``exists: false`` — registered but not downloaded) is kept
    selectable but labeled so the user knows why a load would fail. An empty /
    missing response falls back to the lone "default" choice (the server-side
    default entry always exists).

    The injected default entry (``name == "default"``) gets a descriptive
    label of the form ``"default — <filename>"`` built from its ``path``, so
    the user can tell which file the config-side default actually points at
    instead of seeing a bare, uninformative "default". This is a DISPLAY-ONLY
    change: the choice's VALUE stays ``"default"`` (the server-side resolution
    logic and callers key off that name, never the label). When ``path`` is
    empty or missing the label falls back to plain "default", same as before.

    ``base_model`` (optional) reads the listing of THAT base model instead of
    the active one's legacy block — see :func:`_category_block`.
    """
    block = _category_block(models_json, category, base_model)
    choices: list[tuple[str, str]] = []
    for entry in block.get("entries") or []:
        name = entry.get("name")
        if not name:
            continue
        label = name
        if name == MODEL_DEFAULT:
            path = entry.get("path") or ""
            filename = _path_basename(path) if path else ""
            if filename:
                label = f"{MODEL_DEFAULT} — {filename}"
        if not entry.get("exists", True):
            label = f"{label} ({L('model_missing', lang)})"
        choices.append((label, name))
    return choices or [(MODEL_DEFAULT, MODEL_DEFAULT)]


def model_active_value(models_json: dict | None, category: str,
                       base_model: str | None = None) -> str:
    """The currently active NAME for a category (``"default"`` fallback).

    With ``base_model`` set to a base model that is NOT the active one the
    server sends an empty ``active`` (no live selection exists for it), so this
    naturally falls back to ``"default"`` — the right pre-selection for a base
    model the user is only browsing."""
    block = _category_block(models_json, category, base_model)
    return block.get("active") or MODEL_DEFAULT


# --------------------------------------------------------------------------- #
# Style/character LoRA gallery (Style LoRA tab, S2). GET /loras enumerates every
# adapter with its ``kind``; the gallery shows ONLY the ``style`` ones (control
# adapters — canny/pose/upscaler — stay in the Generate tab's reference-video
# field). Each entry's thumbnail is served by the API (GET /loras/{name}/
# thumbnail); entries without one get a neutral placeholder so the tile still
# renders with its name caption.
# --------------------------------------------------------------------------- #

#: Neutral placeholder tile for a style LoRA that has no sibling .png thumbnail
#: (a light-gray square; gradio postprocesses the ndarray into an <img>).
_STYLE_PLACEHOLDER = np.full((144, 144, 3), 210, dtype=np.uint8)


def _style_lora_entries(loras: list[dict] | None) -> list[dict]:
    """The ``kind == "style"`` entries from a GET /loras list, in server order."""
    return [
        e for e in (loras or [])
        if isinstance(e, dict) and e.get("kind") == "style" and e.get("name")
    ]


def build_style_gallery(loras: list[dict] | None, base_url: str | None) -> list[tuple]:
    """gr.Gallery ``value`` for the Style LoRA tab: ``[(image, caption), ...]``
    over the style LoRAs only. ``image`` is the thumbnail URL when the entry has
    one (fetched by the browser from ``base_url``), else the neutral
    placeholder; ``caption`` is the LoRA name."""
    base = (base_url or "").rstrip("/")
    items: list[tuple] = []
    for entry in _style_lora_entries(loras):
        name = entry["name"]
        if entry.get("has_thumbnail"):
            image: object = f"{base}/api/v1/loras/{name}/thumbnail"
        else:
            image = _STYLE_PLACEHOLDER
        items.append((image, name))
    return items


def style_lora_names(loras: list[dict] | None) -> list[str]:
    """Style-LoRA names in the SAME order as :func:`build_style_gallery`, so a
    gallery select index resolves 1:1 to a name."""
    return [e["name"] for e in _style_lora_entries(loras)]
