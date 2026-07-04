"""Gradio verification UI (spec ch.12) — thin client over the frozen REST API.

The UI is a *thin client* over the frozen REST API (/api/v1/*). It never calls
LTX directly; it exercises the very same endpoints future frontends (AviUtl2,
DaVinci Resolve) will use:

    [optional] POST /api/v1/upload/image  -> image_id
    POST /api/v1/generate                 -> job_id
    poll GET /api/v1/jobs/{job_id}         -> progress
    GET /api/v1/jobs/{job_id}/video        -> mp4

Package layout (split from the original single-file gradio_ui.py, one
cohesive concern per module):
  * ``i18n``       — ``LABELS`` / ``L()``, the i18n-ready label table (English
                      default, Japanese ported). Every user-visible string is
                      looked up via ``L(key)``.
  * ``api_client`` — ``ApiClient``, the only place /api/v1/* paths + auth
                      headers live. Holds an injectable ``httpx.Client`` so
                      tests can feed it an ``httpx.MockTransport``.
  * ``presets``    — ``PRESETS`` fallback table + preset/spill-warning logic
                      (``build_preset_choices``, ``pick_default_preset``,
                      ``compute_spill_warning``, ``apply_preset``).
  * ``adapters``   — IC-LoRA control-adapter dropdown choices
                      (``ADAPTER_NONE``, ``ADAPTER_FRIENDLY``,
                      ``build_adapter_choices``).
  * ``formatting`` — pure formatters: ``format_status()`` for the top status
                      line, ``format_api_error()`` for the REST error
                      envelope.
  * ``validation`` — ``check_chain_total()``, the clip-chain total-timeline
                      precheck that mirrors the server's chain_math validator.
  * ``handlers``   — ``make_generate_handler()`` / ``make_chain_handler()``,
                      the yield-based generate/chain flows factored out so
                      they are unit-testable with a mock transport.
  * ``ui``         — ``build_ui()``, assembling the top common bar + gr.Tabs
                      (Generate / Clip Chain / Jobs / Settings).

Dark theme + language switching are wired at the mount site (main.py passes a
``js=`` dark-default) and, for the Theme dropdown, a pure-frontend js handler.
No HTTP is performed at build time (build_ui runs before uvicorn listens); the
initial /status + /config fetch happens in ``demo.load``.

This ``__init__`` re-exports the public API so existing imports
(``from gradio_ui import build_ui``, ``from gradio_ui import ApiClient``,
etc.) keep working unchanged -- this package split is a pure refactor with no
behavior change.
"""

from __future__ import annotations

from .adapters import ADAPTER_FRIENDLY, ADAPTER_NONE, build_adapter_choices
from .api_client import ApiClient
from .formatting import format_api_error, format_status
from .handlers import make_chain_handler, make_generate_handler
from .i18n import LABELS, L
from .presets import (
    PRESETS,
    apply_preset,
    build_preset_choices,
    compute_spill_warning,
    pick_default_preset,
)
from .ui import build_ui
from .validation import MAX_CHAIN_TOTAL_PIXEL_FRAMES, check_chain_total

__all__ = [
    "ADAPTER_FRIENDLY",
    "ADAPTER_NONE",
    "ApiClient",
    "LABELS",
    "L",
    "MAX_CHAIN_TOTAL_PIXEL_FRAMES",
    "PRESETS",
    "apply_preset",
    "build_adapter_choices",
    "build_preset_choices",
    "build_ui",
    "check_chain_total",
    "compute_spill_warning",
    "format_api_error",
    "format_status",
    "make_chain_handler",
    "make_generate_handler",
    "pick_default_preset",
]
