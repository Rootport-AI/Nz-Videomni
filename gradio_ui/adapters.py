"""IC-LoRA reference-video control (S4). The Generate tab exposes an adapter
Dropdown whose choices are rebuilt on page load from /config model.ic_loras;
the static list below is the offline fallback (server /config unavailable).
``ADAPTER_NONE`` is the sentinel value meaning "no adapter" (payload omits
loras + reference_video_id entirely). The three known adapter keys get a
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
}

# Fallbacks used when /config is unavailable (mirrors config.yaml upload.*).
_FALLBACK_VIDEO_EXTS = [".mp4", ".mov", ".webm", ".mkv"]
_FALLBACK_MAX_VIDEO_MB = 200


def build_adapter_choices(config: dict | None, lang: str = _DEFAULT_LANG) -> list[tuple[str, str]]:
    """Build the adapter Dropdown ``choices`` (list of (label, value)).

    "None" is always first (value :data:`ADAPTER_NONE`). The remaining entries
    come from the fetched /config ``model.ic_loras`` keys (value == key); each
    key gets a friendly label when known, else is shown verbatim. Falls back to
    the three static known adapters when the server config has no ic_loras.
    """
    none_choice = (L("adapter_none", lang), ADAPTER_NONE)
    ic_loras = ((config or {}).get("model") or {}).get("ic_loras") or {}
    if not ic_loras:
        return [none_choice] + [
            (ADAPTER_FRIENDLY[k], k)
            for k in ("pixel-spatial-upscaler-x2", "canny-control", "pose-control")
        ]
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


def build_model_choices(
    models_json: dict | None, category: str, lang: str = _DEFAULT_LANG
) -> list[tuple[str, str]]:
    """Dropdown ``choices`` for one category from a GET /models response.

    The server already orders entries default-first. An entry whose file is
    absent on disk (``exists: false`` — registered but not downloaded) is kept
    selectable but labeled so the user knows why a load would fail. An empty /
    missing response falls back to the lone "default" choice (the server-side
    default entry always exists).
    """
    block = ((models_json or {}).get("categories") or {}).get(category) or {}
    choices: list[tuple[str, str]] = []
    for entry in block.get("entries") or []:
        name = entry.get("name")
        if not name:
            continue
        label = name
        if not entry.get("exists", True):
            label = f"{name} ({L('model_missing', lang)})"
        choices.append((label, name))
    return choices or [(MODEL_DEFAULT, MODEL_DEFAULT)]


def model_active_value(models_json: dict | None, category: str) -> str:
    """The currently active NAME for a category (``"default"`` fallback)."""
    block = ((models_json or {}).get("categories") or {}).get(category) or {}
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
