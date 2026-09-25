"""Stage-2 window dropdown for the Clip Chain tab (§3-165): the window names,
the chain comfort budget that applies to the current engine + acceleration
settings, and the option labels built from it.

Every rule here is a port of the WebUI, so the two GUIs show the same label for
the same inputs (tests/test_gradio_stage2_window.py pins the numbers):

* the effective acceleration fields -> ``shell/accelerationSettings.ts``
  (``effectiveAcceleration`` + ``effectiveAccelerationFields``);
* the budget -> ``shell/comfortTable.ts`` ``resolveComfortRow`` followed by the
  ``?? resolveChainComfortBudget(limits.chain_comfort_token_budget)`` fallback
  in ``modes/chained/useChainForm.ts``;
* the recommended 16:9 size and the label -> ``shell/tokenBudget.ts``
  (``chainComfortAxisMax`` / ``chainComfortSize16x9`` /
  ``stage2WindowOptionLabel``).

The window table itself is NOT copied: the names and ``v_tile`` values are read
from ``chain_math.STAGE2_WINDOW_PRESETS`` (the single source of truth), minus
``full_length``, which is the fixed A2V wire value and never a choice.
"""

from __future__ import annotations

import math

import chain_math

from .i18n import L, _DEFAULT_LANG

#: The selectable windows, in table order (standard, high_resolution, w25..w61).
STAGE2_WINDOW_CHOICES: tuple[str, ...] = tuple(
    name for name in chain_math.STAGE2_WINDOW_PRESETS
    if name != chain_math.STAGE2_WINDOW_FULL_LENGTH
)

#: The dropdown's default value (the payload omits the key at this value).
STAGE2_WINDOW_DEFAULT = chain_math.STAGE2_WINDOW_DEFAULT

#: The slider grid the Chain tab's width/height step on. This tab has no
#: reference-video input, so it is always the WebUI's plain 64 grid.
GUIDE_GRID_DEFAULT = 64

_VAE_SPATIAL_FACTOR = 32


def _usable_budget(published, fallback: int) -> int:
    """A served budget, or ``fallback`` when it is not a positive finite number
    (missing / 0 / negative / NaN / not a number) — the WebUI's guard."""
    if isinstance(published, bool) or not isinstance(published, (int, float)):
        return fallback
    if not math.isfinite(published) or published <= 0:
        return fallback
    return published


def effective_acceleration_fields(attention_backend, block_swap_prefetch,
                                  keep_resident, fused_gguf_dequant_kernel,
                                  vae_mode, keep_resident_embeddings,
                                  sage_available=None,
                                  prefetch_available=None) -> dict:
    """The six acceleration settings as they would EFFECTIVELY run, in the
    server's request vocabulary — what a ``comfort_budgets`` row's ``requires``
    is matched against.

    ``sage_available`` / ``prefetch_available`` are ``GET /status``
    ``acceleration.sage_available`` / ``block_swap_prefetch_available``;
    ``None`` (unknown) counts as available, only an explicit ``False``
    demotes. keep-resident only counts while prefetch is effectively on (the
    backend folds the other combination back off)."""
    prefetch_on = bool(block_swap_prefetch) and prefetch_available is not False
    return {
        "attention_backend": ("sage" if attention_backend == "sage"
                              and sage_available is not False else "sdpa"),
        "block_swap_prefetch": bool(block_swap_prefetch),
        "keep_resident": bool(keep_resident) and prefetch_on,
        "fused_gguf_dequant_kernel": bool(fused_gguf_dequant_kernel),
        "vae_mode": vae_mode or "default",
        "keep_resident_embeddings": bool(keep_resident_embeddings),
    }


def _matches_requires(requires: dict, fields: dict) -> bool:
    """Every ``requires`` key present in ``fields`` AND equal (an unknown key
    never matches — the safe side)."""
    for key, want in (requires or {}).items():
        if key not in fields or fields[key] != want:
            return False
    return True


def resolve_chain_budget(limits: dict | None, engine_family: str | None,
                         accel_fields: dict) -> int:
    """The chain stage-2 comfort budget (tokens per window) for this engine and
    these EFFECTIVE acceleration fields (:func:`effective_acceleration_fields`).

    ``limits.comfort_budgets[engine_family].rows`` is scanned top-down and the
    first row whose ``requires`` all match gives ``chain_budget`` (40,000 if
    that number is unusable). No table, an unknown engine (``""`` — before the
    first ``GET /models``), an engine without a profile, or no matching row
    all fall back to ``limits.chain_comfort_token_budget`` (40,000 if that is
    unusable too)."""
    limits = limits or {}
    scalar = _usable_budget(limits.get("chain_comfort_token_budget"),
                            chain_math.CHAIN_COMFORT_TOKEN_BUDGET)
    table = limits.get("comfort_budgets")
    if not table or not engine_family:
        return scalar
    profile = table.get(engine_family)
    if not profile:
        return scalar
    for row in profile.get("rows") or []:
        if not row:
            continue
        if _matches_requires(row.get("requires") or {}, accel_fields):
            return _usable_budget(row.get("chain_budget"),
                                  chain_math.CHAIN_COMFORT_TOKEN_BUDGET)
    return scalar


def _v_tile(window: str) -> int:
    return chain_math.STAGE2_WINDOW_PRESETS[window][0]


def _floor_to_grid(value: float, grid: int) -> int:
    return int(math.floor(value / grid) * grid)


def chain_comfort_axis_max(other: int, window: str, budget: int,
                           grid: int = GUIDE_GRID_DEFAULT) -> int:
    """Largest value for ONE axis that keeps a ``window`` stage-2 tile inside
    ``budget`` while the other axis is ``other`` px, floored onto ``grid``
    (0 when ``other`` is below one 32px cell)."""
    other_cells = int(other) // _VAE_SPATIAL_FACTOR
    if other_cells < 1:
        return 0
    cells = int(budget // (_v_tile(window) * other_cells))
    return _floor_to_grid(_VAE_SPATIAL_FACTOR * cells, grid)


def chain_comfort_size_16x9(window: str, budget: int,
                            grid: int = GUIDE_GRID_DEFAULT) -> tuple[int, int]:
    """The recommended roughly-16:9 ``(width, height)`` for ``window`` at
    ``budget``: width from ``sqrt(floor(budget / v_tile) * 1024 * 16 / 9)``
    floored onto ``grid``, height = :func:`chain_comfort_axis_max` of it."""
    cells = int(budget // _v_tile(window))
    width = _floor_to_grid(math.sqrt(cells * 1024 * 16 / 9), grid)
    return width, chain_comfort_axis_max(width, window, budget, grid)


def stage2_window_option_label(window: str, template: str, engine_label: str,
                               budget: int, grid: int = GUIDE_GRID_DEFAULT) -> str:
    """``template`` with ``{frames}``/``{engine}``/``{width}``/``{height}``
    filled. Just ``"{frames}f"`` when no size fits the budget; the
    ``"{engine} "`` placeholder (with its space) is dropped while the engine
    name is unknown."""
    frames = str(_v_tile(window))
    width, height = chain_comfort_size_16x9(window, budget, grid)
    if width <= 0 or height <= 0:
        return f"{frames}f"
    if not engine_label:
        template = template.replace("{engine} ", "", 1)
    return (template.replace("{frames}", frames, 1)
            .replace("{engine}", engine_label, 1)
            .replace("{width}", str(width), 1)
            .replace("{height}", str(height), 1))


def build_stage2_window_choices(lang: str = _DEFAULT_LANG, engine_label: str = "",
                                budget: int | None = None) -> list[tuple[str, str]]:
    """Dropdown ``choices`` ``[(label, window_name), ...]`` over
    :data:`STAGE2_WINDOW_CHOICES`. ``budget=None`` -> the 40,000 default."""
    if budget is None:
        budget = chain_math.CHAIN_COMFORT_TOKEN_BUDGET
    template = L("stage2_window_option", lang)
    return [(stage2_window_option_label(name, template, engine_label, budget), name)
            for name in STAGE2_WINDOW_CHOICES]


def engine_info_from_models(models_json: dict | None) -> tuple[str, str]:
    """``(engine_family, display_name)`` of the ACTIVE base model from a
    ``GET /models`` response (``("", "")`` when it cannot be told)."""
    root = models_json or {}
    active = root.get("active_base_model") or ""
    if not active:
        return "", ""
    for entry in root.get("base_models") or []:
        if isinstance(entry, dict) and entry.get("id") == active:
            return (entry.get("engine_family") or "",
                    entry.get("display_name") or "")
    return "", ""


def status_availability(status_json: dict | None) -> tuple[bool | None, bool | None]:
    """``(sage_available, block_swap_prefetch_available)`` from a ``GET
    /status`` body; ``None`` for a missing / non-bool flag (= unknown)."""
    accel = (status_json or {}).get("acceleration") or {}
    sage = accel.get("sage_available")
    prefetch = accel.get("block_swap_prefetch_available")
    return (sage if isinstance(sage, bool) else None,
            prefetch if isinstance(prefetch, bool) else None)


def stage2_window_choices_for(lang, config, engine_state, attention_backend,
                              block_swap_prefetch, keep_resident,
                              fused_gguf_dequant_kernel, vae_mode,
                              keep_resident_embeddings) -> list[tuple[str, str]]:
    """The whole chain from the Gradio inputs to the dropdown ``choices``:
    ``engine_state`` is the tab's ``{"engine_family", "engine_label",
    "sage_available", "prefetch_available"}`` State (filled from ``GET
    /models`` + ``GET /status``), ``config`` the ``/config`` dict and the six
    acceleration values are the Settings tab's controls."""
    state = engine_state or {}
    fields = effective_acceleration_fields(
        attention_backend, block_swap_prefetch, keep_resident,
        fused_gguf_dequant_kernel, vae_mode, keep_resident_embeddings,
        sage_available=state.get("sage_available"),
        prefetch_available=state.get("prefetch_available"),
    )
    budget = resolve_chain_budget((config or {}).get("limits"),
                                  state.get("engine_family") or "", fields)
    return build_stage2_window_choices(lang, state.get("engine_label") or "", budget)
