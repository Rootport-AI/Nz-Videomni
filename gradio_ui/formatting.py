"""Pure formatters: the top status line and the REST error envelope. Both are
unit-testable without a live server (format_status against a fake /status
body, format_api_error against a fake error envelope).
"""

from __future__ import annotations

import json

from .i18n import LABELS, L, _DEFAULT_LANG


# --------------------------------------------------------------------------- #
# Pure formatter for the top status line (unit-testable with a fake /status).
# Field names verified against api/status.py + services/{gpu_info,low_vram,
# job_store}.py: gpu.{name,vram_free_mb,vram_total_mb}, vram_optimization.
# {low_vram_mode,low_vram_profile}, queue.{running,pending,completed}.
# --------------------------------------------------------------------------- #
def format_status(s: dict, lang: str = _DEFAULT_LANG) -> str:
    gpu = s.get("gpu") or {}
    v = s.get("vram_optimization") or {}
    q = s.get("queue") or {}

    loaded = s.get("pipeline_loaded")
    ptype = s.get("pipeline_type")
    pipe = L("st_loaded", lang) if loaded else L("st_not_loaded", lang)
    if loaded and ptype:
        pipe = f"{pipe} ({ptype})"

    low = L("st_on", lang) if v.get("low_vram_mode") else L("st_off", lang)

    parts = [
        f"server={s.get('server')} v{s.get('version')}",
        f"{L('st_pipeline', lang)}: {pipe}",
        (f"GPU: {gpu.get('name')} "
         f"(VRAM {L('st_free', lang)} {gpu.get('vram_free_mb')} / {gpu.get('vram_total_mb')} MB)"),
        f"{L('st_lowvram', lang)}: {low} (profile={v.get('low_vram_profile')})",
    ]
    if q:
        parts.append(
            f"{L('st_queue', lang)}: "
            f"{L('st_running', lang)} {q.get('running', 0)} / "
            f"{L('st_waiting', lang)} {q.get('pending', 0)} / "
            f"{L('st_done', lang)} {q.get('completed', 0)}"
        )
    return " | ".join(parts)


def format_api_error(body: object, lang: str = _DEFAULT_LANG) -> str:
    """Render a REST error envelope into a localized, actionable message.

    ``body`` is the parsed JSON dict (``{"error": {"code", "message", "detail"}}``)
    or, when the response was not JSON, the raw text. Maps ``error.code`` to a
    one-line hint (all 15 real codes). For ``VALIDATION_ERROR`` the ``detail`` is
    a list of ``{loc, msg, type}`` rendered as ``loc: msg`` lines; for other
    codes ``detail`` is a string appended when present. An unknown code or an
    unparseable body falls back to the raw text.
    """
    if not isinstance(body, dict):
        return str(body)
    error = body.get("error")
    if not isinstance(error, dict):
        return json.dumps(body, ensure_ascii=False)
    code = error.get("code")
    hint_key = f"apierr_{code}" if code else None
    if not hint_key or hint_key not in LABELS["en"]:
        # Unknown / unmapped code -> raw text (still useful for debugging).
        return json.dumps(body, ensure_ascii=False)

    lines = [L(hint_key, lang)]
    detail = error.get("detail")
    if code == "VALIDATION_ERROR" and isinstance(detail, list):
        for item in detail:
            loc = ".".join(str(x) for x in (item.get("loc") or []))
            msg = item.get("msg", "")
            lines.append(f"{loc}: {msg}" if loc else str(msg))
    elif isinstance(detail, str) and detail:
        lines.append(detail)
    return "\n".join(lines)
