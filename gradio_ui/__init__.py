"""Gradio verification UI (spec §12) — thin client over the frozen REST API.

The UI is a *thin client* over the frozen REST API (/api/v1/*). It never calls
LTX directly; it exercises the very same endpoints the AviUtl2 WebView2
frontend uses (and a future DaVinci Resolve frontend would use):

    [optional] POST /api/v1/upload/image  -> image_id
    POST /api/v1/generate                 -> job_id
    poll GET /api/v1/jobs/{job_id}         -> progress
    GET /api/v1/jobs/{job_id}/video        -> mp4

Package layout (one cohesive concern per module):
  * ``i18n``       — ``LABELS`` / ``L()``, the i18n-ready label table (English
                      default, Japanese ported). User-visible strings are
                      looked up via ``L(key)``.
  * ``api_client`` — ``ApiClient``, where the /api/v1/* calls + auth headers
                      live. Holds an injectable ``httpx.Client`` so
                      tests can feed it an ``httpx.MockTransport``.
  * ``presets``    — ``PRESETS`` fallback table + preset/spill-warning logic
                      (``build_preset_choices``, ``pick_default_preset``,
                      ``compute_spill_warning``, ``apply_preset``).
  * ``adapters``   — IC-LoRA control-adapter dropdown choices
                      (``ADAPTER_NONE``, ``ADAPTER_FRIENDLY``,
                      ``build_adapter_choices``).
  * ``feature_scope`` — server feature name -> Gradio control ids: the only
                      module allowed to know a server feature name. Hides (and
                      resets to the server default) every control the active
                      base model cannot use. Mirrors the WebView2 frontend's
                      ``webui/src/shell/featureScope.ts``.
  * ``formatting`` — pure formatters: ``format_status()`` for the top status
                      line, ``format_api_error()`` for the REST error
                      envelope.
  * ``validation`` — ``check_chain_total()``, the clip-chain total-timeline
                      precheck that mirrors the server's chain_math validator.
  * ``comfort``    — the Clip Chain tab's Stage-2 window dropdown: the window
                      choices, the chain comfort budget for the current
                      engine + acceleration settings, and the option labels
                      (ports of the WebUI's rules).
  * ``manifest``   — Batch A2V CSV manifest, the pure-Python data layer: scan
                      a wav folder into rows, read/write/merge the CSV that
                      lives next to the audio files, resolve output paths. No
                      gradio / threading / HTTP imports.
  * ``batch``      — ``BatchRunner``, the Batch A2V execution body: a daemon
                      thread inside the server process feeds one row at a time
                      into the REST API, so an unattended overnight run
                      survives the browser's SSE stream dropping.
  * ``handlers``   — ``make_generate_handler()`` / ``make_chain_handler()``,
                      the yield-based generate/chain flows factored out so
                      they are unit-testable with a mock transport.
  * ``styles``     — ``CUSTOM_CSS``, the stylesheet for the Blocks UI; ``ui``
                      alone imports it and injects it as an in-tree
                      ``gr.HTML`` ``<style>`` block (``mount_gradio_app``
                      overwrites ``blocks.css``, so ``gr.Blocks(css=...)``
                      would be dropped).
  * ``ui``         — ``build_ui()``, assembling the top common bar, the shared
                      prompt area above the tabs, and the ``gr.Tabs``.

The dark default is wired at the mount site (main.py passes a ``js=``
dark-default); the Theme dropdown uses a pure-frontend js handler, and the
Language dropdown is a server-side handler in ``ui`` (``switch_language``).
No HTTP is performed at build time (build_ui runs before uvicorn listens); the
initial /status, /config, /models and /loras fetches happen in ``demo.load``
handlers.

This ``__init__`` re-exports the public API so callers can import from the
package root (``from gradio_ui import build_ui``,
``from gradio_ui import ApiClient``, etc.).
"""

from __future__ import annotations

from .adapters import (
    ADAPTER_FRIENDLY,
    ADAPTER_NONE,
    build_adapter_choices,
    build_style_gallery,
    style_lora_names,
)
from .api_client import ApiClient
from .formatting import (
    build_jobs_rows,
    format_api_error,
    format_job_error,
    format_status,
    jobs_table_headers,
)
from .handlers import (
    delete_finished_jobs,
    make_chain_handler,
    make_generate_handler,
    parse_prompt_loras,
)
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
    "build_jobs_rows",
    "build_preset_choices",
    "build_style_gallery",
    "build_ui",
    "check_chain_total",
    "compute_spill_warning",
    "delete_finished_jobs",
    "format_api_error",
    "format_job_error",
    "format_status",
    "jobs_table_headers",
    "make_chain_handler",
    "make_generate_handler",
    "parse_prompt_loras",
    "pick_default_preset",
    "style_lora_names",
]
