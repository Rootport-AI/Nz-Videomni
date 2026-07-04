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
