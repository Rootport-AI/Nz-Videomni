"""IC-LoRA reference-video control (S4). The Generate tab exposes an adapter
Dropdown whose choices are rebuilt on page load from /config model.ic_loras;
the static list below is the offline fallback (server /config unavailable).
``ADAPTER_NONE`` is the sentinel value meaning "no adapter" (payload omits
loras + reference_video_id entirely). The three known adapter keys get a
friendly label; any unknown registered key is shown as-is.
"""

from __future__ import annotations

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
