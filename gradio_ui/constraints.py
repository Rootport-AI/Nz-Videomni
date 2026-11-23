"""Per-base-model request constraints for the Generate (Single) tab.

``GET /models`` serves, on every ``base_models[]`` entry, a ``constraints``
block built from the base model's own request rules (the server's adapter
table is the one source; the operation panel, this UI, MCP and the mock all
read it). Its shape::

    {"num_frames": {"step": 8, "offset": 1, "min": 9, "max": 481},
     "dimension_multiple": 64,
     "frame_rate": {"min": 1, "max": 60},
     "sampling": null}

``sampling`` is ``null`` for a base model whose step count is fixed (LTX:
distilled, 8 steps) and an object of three ``{min, max, step, default}``
sliders for one that takes them (MiniMax H3: ``num_inference_steps`` /
``shift_video`` / ``shift_audio``).

This module reads that block for the LOADED base model -- the same judgment
:func:`gradio_ui.adapters.active_unsupported_features` makes, because a
request is checked against what the pipeline is on, not against a pending
dropdown selection -- and turns it into the Single tab's labels, precheck and
sampling sliders. It never branches on an engine id: a server that sends no
block (an older backend, or before the first ``GET /models``) reads as
:data:`LTX_CONSTRAINTS`, so LTX keeps its fixed look (8 steps, CFG 1.0, 8n+1, ÷64).

The Clip Chain tab, the batch and A2V paths (POST /generate/chain) are not
touched: they are LTX-only. They send LTX's frame grid (8n+1) and 8 steps,
which MiniMax H3's chain refuses with a 422 (it takes 17n+5 clips; a chain
that happens to be on both grids, e.g. 73 / 209 / 345 / 481 frames, runs on
H3 at 8 steps). The operation panel's Chained screen is H3's chain client.
"""

from __future__ import annotations

import copy

import gradio as gr

from .i18n import L, _DEFAULT_LANG

#: The constraints of the LTX base models -- the fallback for a server that
#: serves none. Mirrors the server's LTX request rules (8n+1, multiple of 64,
#: 1-60 fps, no sampling sliders).
LTX_CONSTRAINTS: dict = {
    "num_frames": {"step": 8, "offset": 1, "min": 9, "max": 481},
    "dimension_multiple": 64,
    "frame_rate": {"min": 1, "max": 60},
    "sampling": None,
}

#: The three sampling sliders, in display (and payload) order.
SAMPLING_FIELDS: tuple[str, ...] = ("num_inference_steps", "shift_video", "shift_audio")

#: The fixed values the Single tab sends when the base model has no sampling
#: sliders (LTX distilled: the frozen contract).
FIXED_STEPS = 8
FIXED_CFG = 1.0

#: The pi-H3 radio's "not in use" value (the server field's default).
PI_H3_OFF = "off"
#: The Fast VAE radio's Sol-H3-VAE value (it feeds ``sol_vae``, not ``pi_h3``).
FAST_VAE_SOL = "sol"
#: The pi-H3 choices of the Fast VAE radio (the server's ``pi_h3`` values).
PI_H3_CHOICES = ("4steps", "2steps")


def split_fast_vae(value) -> tuple[str, bool]:
    """The Fast VAE radio's one value -> ``(pi_h3, sol_vae)``, the two request
    fields: a pi-H3 choice is ``(choice, False)``, Sol is ``("off", True)``,
    anything else ``("off", False)``. The one place the radio is split; what
    follows reads the pi-H3 value as before."""
    if value in PI_H3_CHOICES:
        return value, False
    if value == FAST_VAE_SOL:
        return PI_H3_OFF, True
    return PI_H3_OFF, False


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _usable_slider(spec) -> bool:
    return isinstance(spec, dict) and all(
        _number(spec.get(key)) for key in ("min", "max", "step", "default"))


def normalize_constraints(raw) -> dict:
    """A complete constraints dict: every top-level key the server sent in a
    usable shape, the LTX value for anything missing or malformed. ``sampling``
    is kept only when all three sliders are complete (``None`` otherwise, i.e.
    "no sliders")."""
    out = copy.deepcopy(LTX_CONSTRAINTS)
    if not isinstance(raw, dict):
        return out
    nf = raw.get("num_frames")
    if isinstance(nf, dict) and all(_number(nf.get(k))
                                    for k in ("step", "offset", "min", "max")) \
            and nf["step"] > 0:
        out["num_frames"] = {k: int(nf[k]) for k in ("step", "offset", "min", "max")}
    dm = raw.get("dimension_multiple")
    if _number(dm) and dm > 0:
        out["dimension_multiple"] = int(dm)
    fr = raw.get("frame_rate")
    if isinstance(fr, dict) and _number(fr.get("min")) and _number(fr.get("max")):
        out["frame_rate"] = {"min": fr["min"], "max": fr["max"]}
    sampling = raw.get("sampling")
    if isinstance(sampling, dict) and all(_usable_slider(sampling.get(f))
                                          for f in SAMPLING_FIELDS):
        out["sampling"] = {f: dict(sampling[f]) for f in SAMPLING_FIELDS}
    # pi-H3 (MiniMax H3): ``{choice: {num_inference_steps, shift_video,
    # shift_audio}}``, the sampling values each choice runs with. Kept only
    # in that shape (each entry with all three numbers); a missing or broken
    # block adds no key, so a server without pi-H3 (LTX) reads as before.
    pi_h3 = raw.get("pi_h3")
    if isinstance(pi_h3, dict):
        presets = {
            choice: {f: values[f] for f in SAMPLING_FIELDS}
            for choice, values in pi_h3.items()
            if isinstance(choice, str) and choice != PI_H3_OFF
            and isinstance(values, dict) and all(_number(values.get(f)) for f in SAMPLING_FIELDS)
        }
        if presets:
            out["pi_h3"] = presets
    return out


def active_scope(models_json: dict | None) -> dict:
    """``{"engine_family": str, "constraints": dict}`` of the LOADED base model
    (the ``base_models[]`` entry flagged ``active``). No active entry -- or no
    ``base_models`` at all -- yields an unknown family (``""``) and the LTX
    constraints."""
    for entry in (models_json or {}).get("base_models") or []:
        if isinstance(entry, dict) and entry.get("active"):
            family = entry.get("engine_family")
            return {"engine_family": family if isinstance(family, str) else "",
                    "constraints": normalize_constraints(entry.get("constraints"))}
    return default_scope()


def default_scope() -> dict:
    """The scope before the first ``GET /models``: unknown family, LTX rules."""
    return {"engine_family": "", "constraints": copy.deepcopy(LTX_CONSTRAINTS)}


def scope_constraints(scope) -> dict:
    """The constraints held in a scope State value (LTX when it holds none)."""
    if isinstance(scope, dict):
        return normalize_constraints(scope.get("constraints"))
    return copy.deepcopy(LTX_CONSTRAINTS)


def frame_rule(constraints: dict) -> str:
    """The frame grid as the label shows it: ``"8n+1"`` / ``"17n+5"``."""
    nf = normalize_constraints(constraints)["num_frames"]
    return f"{nf['step']}n+{nf['offset']}"


def is_valid_num_frames(n: int, constraints: dict) -> bool:
    nf = normalize_constraints(constraints)["num_frames"]
    return nf["min"] <= n <= nf["max"] and (n - nf["offset"]) % nf["step"] == 0


def field_labels(constraints: dict, lang: str = _DEFAULT_LANG) -> dict[str, str]:
    """The Single tab's width / height / frames labels for these constraints.
    For LTX they are exactly the static ``lbl_width`` / ``lbl_height`` /
    ``lbl_frames`` strings."""
    c = normalize_constraints(constraints)
    m = c["dimension_multiple"]
    return {
        "width": L("lbl_width_rule", lang).format(m=m),
        "height": L("lbl_height_rule", lang).format(m=m),
        "frames": L("lbl_frames_rule", lang).format(rule=frame_rule(c)),
    }


def single_shape_error(width, height, num_frames, constraints: dict,
                       lang: str = _DEFAULT_LANG) -> str | None:
    """The Single tab's shape precheck, built from the constraints: width /
    height a multiple of ``dimension_multiple`` and ``num_frames`` on the grid
    within ``[min, max]``. A localized message on the first violation, else
    ``None``. The server re-checks all of it (422); this only saves the round
    trip.

    The frame rate is deliberately NOT checked here: its range is the
    server's (an out-of-range fps is sent unchanged so the 422 names it --
    see handlers._snap_frame_rate), and a base model with a fixed fps
    (``frame_rate.min == max``) is no different."""
    c = normalize_constraints(constraints)
    m = c["dimension_multiple"]
    try:
        w_i, h_i = int(width), int(height)
    except (TypeError, ValueError):
        return L("msg_bad_dimension_rule", lang).format(m=m)
    if w_i % m != 0 or h_i % m != 0:
        return L("msg_bad_dimension_rule", lang).format(m=m)
    nf = c["num_frames"]
    frames_msg = L("msg_bad_frames_rule", lang).format(
        rule=frame_rule(c), min=nf["min"], max=nf["max"])
    try:
        nf_i = int(num_frames)
    except (TypeError, ValueError):
        return frames_msg
    if not is_valid_num_frames(nf_i, c):
        return frames_msg
    return None


def sampling_payload(constraints: dict, steps, shift_video, shift_audio) -> dict:
    """The sampling keys the Single tab sends. With sliders: all three, ALWAYS
    (one rule), each value falling back to its slider default when unreadable.
    Without: NONE -- the server writes the loaded family's own default back
    (LTX: 8). Not sending a fixed 8 also keeps
    a stale scope (another client switched the server to a sampling family
    and this page was not refreshed) from silently running that family at 8
    steps. ``guidance_scale`` is sent by the caller."""
    sampling = normalize_constraints(constraints)["sampling"]
    if sampling is None:
        return {}
    values = {"num_inference_steps": steps, "shift_video": shift_video,
              "shift_audio": shift_audio}
    out: dict = {}
    for field in SAMPLING_FIELDS:
        raw = values[field]
        try:
            v = float(raw)
        except (TypeError, ValueError):
            v = float(sampling[field]["default"])
        out[field] = int(round(v)) if field == "num_inference_steps" else v
    return out


def pi_h3_locked_values(constraints: dict, choice) -> dict | None:
    """The sampling values a pi-H3 ``choice`` locks the Single tab to, or
    ``None`` when it locks nothing.

    Locked only when the choice is not "off", the constraints carry
    ``sampling`` and the served ``pi_h3`` block has an entry for the choice;
    the values are that entry (``{num_inference_steps, shift_video,
    shift_audio}``), never numbers of this UI's own. The ONE judgment behind
    both the locked sliders (:func:`pi_h3_sampling_updates`) and the request
    (handlers.generate sends ``pi_h3`` instead of the three values), so the
    screen and the wire cannot disagree."""
    if not isinstance(choice, str) or choice == PI_H3_OFF:
        return None
    c = normalize_constraints(constraints)
    if c["sampling"] is None:
        return None
    values = (c.get("pi_h3") or {}).get(choice)
    return dict(values) if values else None


def pi_h3_sampling_updates(constraints: dict, choice) -> tuple:
    """``gr.update``s for the steps, shift (video) and shift (audio) sliders
    and the "fixed because pi-H3" caption, for these constraints and this
    pi-H3 ``choice`` -- the part both wirings share (the scope change through
    :func:`single_scope_updates`, and the pi-H3 radio's own change).

    Locked (:func:`pi_h3_locked_values`): the three sliders show the choice's
    values and cannot be moved, and the caption is shown. Otherwise exactly
    what the constraints give: with sampling sliders, live sliders at their
    served range and default (turning pi-H3 off writes the defaults back);
    without (LTX), steps fixed at 8 and no shift sliders. The caption is
    hidden."""
    c = normalize_constraints(constraints)
    sampling = c["sampling"]
    if sampling is None:
        return (
            gr.update(minimum=1, maximum=50, step=1, value=FIXED_STEPS,
                      interactive=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )
    locked = pi_h3_locked_values(c, choice)

    def _slider(field):
        spec = sampling[field]
        value = spec["default"] if locked is None else locked[field]
        return gr.update(minimum=spec["min"], maximum=spec["max"],
                         step=spec["step"], value=value,
                         interactive=locked is None, visible=True)

    return (
        _slider("num_inference_steps"),
        _slider("shift_video"),
        _slider("shift_audio"),
        gr.update(visible=locked is not None),
    )


def single_scope_updates(constraints: dict, lang: str = _DEFAULT_LANG,
                         pi_h3=PI_H3_OFF) -> tuple:
    """``gr.update``s for the Single tab when the loaded base model's
    constraints change, in this order: steps slider, CFG slider, shift
    (video) slider, shift (audio) slider, the "fixed at 8 / 1.0" caption,
    width, height, frames.

    With sampling sliders: steps / shift become live sliders at their served
    range and default (a base-model switch resets them to the defaults), the
    decorative CFG slider and its "fixed" caption are hidden. Without: exactly
    the LTX look -- steps 1-50 fixed at 8, CFG fixed at 1.0, no shift sliders.
    The width / height / frames labels and spinner steps follow the grid.
    With a pi-H3 choice that locks (``pi_h3``, the Settings radio's value;
    :func:`pi_h3_locked_values`), the three sliders keep the choice's values
    and stay locked instead (:func:`pi_h3_sampling_updates`), so a reload onto
    another H3 weight does not make them movable again."""
    c = normalize_constraints(constraints)
    labels = field_labels(c, lang)
    steps_u, shift_video_u, shift_audio_u, _caption = pi_h3_sampling_updates(c, pi_h3)
    no_sampling = c["sampling"] is None
    slider_updates = (
        steps_u,
        gr.update(visible=no_sampling),
        shift_video_u,
        shift_audio_u,
        gr.update(visible=no_sampling),
    )
    m = c["dimension_multiple"]
    return slider_updates + (
        gr.update(label=labels["width"], step=m),
        gr.update(label=labels["height"], step=m),
        gr.update(label=labels["frames"], step=c["num_frames"]["step"]),
    )


def comfort_table_missing(config: dict | None, engine_family: str | None) -> bool:
    """True when the served ``limits.comfort_budgets`` has no table for the
    loaded engine family, or when that family's profile carries a
    ``single_rows_budget`` that is not ``None`` (a rows-line family, MiniMax
    H3) -- the cases the Single tab drops its spill-free warning (the
    ``spill_free_frames`` table is LTX's). LTX's ``spill_free_frames``
    warning is not shown for a rows-line family; Gradio builds no H3 marker.
    The judgment is on the value, not on the key: the server serves
    ``single_rows_budget: None`` on the LTX profiles too. An unknown family,
    or a server that serves no ``comfort_budgets`` at all, is NOT "missing":
    the LTX behaviour stays."""
    if not engine_family:
        return False
    table = ((config or {}).get("limits") or {}).get("comfort_budgets")
    if not isinstance(table, dict):
        return False
    profile = table.get(engine_family)
    if not profile:
        return True
    return profile.get("single_rows_budget") is not None
