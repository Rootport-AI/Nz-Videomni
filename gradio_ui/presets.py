"""Presets (S1 fallback only — S2 replaces this with /config generation_presets).
width/height are multiples of 64 (two-stage distilled).
"""

from __future__ import annotations

import gradio as gr

from config import MAX_CONDITIONING_IMAGES

from .i18n import L, _DEFAULT_LANG
from .validation import MAX_CHAIN_TOTAL_PIXEL_FRAMES

PRESETS: dict[str, dict] = {
    "smoke_test": {"width": 384, "height": 256, "num_frames": 17, "crop_w": 0, "crop_h": 0},
    "minimal": {"width": 512, "height": 320, "num_frames": 49, "crop_w": 0, "crop_h": 0},
    "small": {"width": 960, "height": 576, "num_frames": 121, "crop_w": 960, "crop_h": 540},
}


# --------------------------------------------------------------------------- #
# Preset handling (S2). The hardcoded PRESETS dict above is now only a fallback
# for when the server /config fetch (demo.load) failed or returned no presets;
# the normal path reads config["generation_presets"] (GET /config, which
# model_dumps config.py's GenerationPreset -- width/height/num_frames/
# crop_output, the last being {"width", "height"} or None).
# --------------------------------------------------------------------------- #
def build_preset_choices(config: dict | None) -> list[tuple[str, str]]:
    """Build Dropdown ``choices`` from the fetched /config generation_presets.

    Falls back to the hardcoded ``PRESETS`` keys (label == value, matching S1
    behaviour) when the server config is empty/unavailable.
    """
    presets = (config or {}).get("generation_presets") or {}
    if not presets:
        return [(name, name) for name in PRESETS]

    choices: list[tuple[str, str]] = []
    for key, p in presets.items():
        w, h, nf = p.get("width"), p.get("height"), p.get("num_frames")
        crop = p.get("crop_output")
        if crop:
            label = f"{key} ({w}×{h} → {crop.get('width')}×{crop.get('height')}, {nf}f)"
        else:
            label = f"{key} ({w}×{h}, {nf}f)"
        choices.append((label, key))
    return choices


def pick_default_preset(config: dict | None) -> str:
    """Pick the Dropdown's initial value: ``standard_720p`` if present, else the
    first server preset, else the S1 fallback default."""
    presets = (config or {}).get("generation_presets") or {}
    if "standard_720p" in presets:
        return "standard_720p"
    if presets:
        return next(iter(presets))
    return "minimal"


def compute_spill_warning(width, height, num_frames, config: dict | None,
                           lang: str = _DEFAULT_LANG):
    """Look up limits.spill_free_frames[f"{width}x{height}"] from the fetched
    /config. Returns a gr.update for a Markdown warning: visible+worded when
    num_frames exceeds the comfortable (spill-free) threshold for that
    resolution, hidden when the resolution is unknown or within budget."""
    try:
        w, h, nf = int(width), int(height), int(num_frames)
    except (TypeError, ValueError):
        return gr.update(value="", visible=False)

    spill = ((config or {}).get("limits") or {}).get("spill_free_frames") or {}
    key = f"{w}x{h}"
    threshold = spill.get(key)
    if threshold is not None and nf > threshold:
        text = L("warn_spill_limit", lang).format(res=key, limit=threshold)
        return gr.update(value=text, visible=True)
    return gr.update(value="", visible=False)


def apply_preset(name: str, config: dict | None, lang: str = _DEFAULT_LANG):
    """Resolve a preset name to the Generate-tab field values.

    Reads the server preset (config["generation_presets"][name]) when present;
    falls back to the hardcoded PRESETS dict only when the server config is
    unavailable or does not contain ``name``. Returns a tuple matching the
    ``preset.change`` outputs: (width, height, num_frames, crop_enabled,
    crop_w, crop_h, crop_row_update, spill_warning_update).
    """
    presets = (config or {}).get("generation_presets") or {}
    if name in presets:
        p = presets[name]
        width_v, height_v, frames_v = p["width"], p["height"], p["num_frames"]
        crop = p.get("crop_output")
        crop_w_v = crop["width"] if crop else 0
        crop_h_v = crop["height"] if crop else 0
    else:
        p = PRESETS.get(name) or PRESETS["minimal"]
        width_v, height_v, frames_v = p["width"], p["height"], p["num_frames"]
        crop_w_v, crop_h_v = p.get("crop_w", 0), p.get("crop_h", 0)

    crop_enabled_v = bool(crop_w_v) and bool(crop_h_v)
    crop_row_update = gr.update(visible=crop_enabled_v)
    spill_update = compute_spill_warning(width_v, height_v, frames_v, config, lang)
    return (width_v, height_v, frames_v, crop_enabled_v, crop_w_v, crop_h_v,
            crop_row_update, spill_update)


def format_duration_label(num_frames, fps) -> str:
    """Format a pixel-frame count + frame rate as a rounded seconds label.

    Pure formatter: ``duration = num_frames / fps``, rendered with 2 decimal
    places and an ``s`` suffix (e.g. ``format_duration_label(273, 24)`` ->
    ``"11.38s"``, ``format_duration_label(121, 24)`` -> ``"5.04s"``).

    Defensive: returns ``""`` (empty string, so a caller can drop it straight
    into a ``gr.update(value=...)``/label with no visible garbage) when either
    argument is ``None``, non-numeric, or when ``fps`` is <= 0 (a zero/negative
    frame rate makes the division meaningless).
    """
    try:
        frames_f = float(num_frames)
        fps_f = float(fps)
    except (TypeError, ValueError):
        return ""
    if fps_f <= 0:
        return ""
    return f"{frames_f / fps_f:.2f}s"


# --------------------------------------------------------------------------- #
# Collapsible clip / keyframe slot bounds (owner decision 2026-07-13). The Clip
# Chain tab shows a growable list of clip slots (min CHAIN_MIN_OPEN, max
# CHAIN_MAX_CLIPS); the Generate tab's keyframe grid is likewise collapsible
# (min KF_MIN_OPEN, max KF_MAX_SLOTS). The ± state transition + the estimated-
# duration readout are pure module-level functions (not UI closures) so they are
# unit-testable without a Gradio runtime.
# --------------------------------------------------------------------------- #
CHAIN_MIN_OPEN = 2
CHAIN_MAX_CLIPS = 24
KF_MIN_OPEN = 1
KF_MAX_SLOTS = MAX_CONDITIONING_IMAGES


def clamp_open_count(count, delta, lo, hi) -> int:
    """Pure ± state transition: return ``count + delta`` clamped to ``[lo, hi]``.

    ``count`` / ``delta`` are coerced to int; a ``None``/garbage ``count`` falls
    back to ``lo`` and a garbage ``delta`` to 0 (a no-op step)."""
    try:
        c = int(count)
    except (TypeError, ValueError):
        c = lo
    try:
        d = int(delta)
    except (TypeError, ValueError):
        d = 0
    return max(lo, min(hi, c + d))


def collapsible_slot_transition(old_count, new_count, n_slots,
                                enable_new: bool = False) -> list[tuple[bool, object]]:
    """Per-extra-slot ``(visible, use_value)`` for slots 2..``n_slots`` after a
    ± step from ``old_count`` open slots to ``new_count``.

    ``visible`` is ``slot_i <= new_count``. ``use_value`` is the slot's Use
    checkbox directive: ``False`` when the slot ends up hidden (a hidden slot
    must never be submitted — the handler collects every ENABLED slot regardless
    of visibility), ``True`` when ``enable_new`` and the slot was NEWLY revealed
    by this step (``old_count < slot_i <= new_count`` — owner request: "+" on
    the Clip Chain enables the new slot), and ``None`` for "leave as is" (a slot
    that stays visible keeps whatever the user set). Slot 1 is always visible
    and is NOT part of the returned list. Returns ``n_slots - 1`` tuples in slot
    2..``n_slots`` order. Counts are assumed pre-clamped (``clamp_open_count``).
    """
    old_c, new_c = int(old_count), int(new_count)
    out: list[tuple[bool, object]] = []
    for slot_i in range(2, int(n_slots) + 1):
        visible = slot_i <= new_c
        if not visible:
            use: object = False
        elif enable_new and old_c < slot_i:
            use = True
        else:
            use = None
        out.append((visible, use))
    return out


def slot_step_state(count, delta, min_open, max_slots,
                    enable_new: bool = False):
    """Full semantic state of a ± step over a collapsible slot list (pure,
    unit-testable; the UI handlers only wrap this into ``gr.update`` calls).
    Shared by the Generate tab's keyframe grid (min 1 / max KF_MAX_SLOTS) and the Clip
    Chain tab's clip list (min 2 / max 24, ``enable_new=True``).

    Returns ``(new_count, slot_states, minus_interactive, plus_interactive,
    counter_text)``:
      * ``new_count``         -- ``count + delta`` clamped to
                                  [``min_open``, ``max_slots``].
      * ``slot_states``       -- :func:`collapsible_slot_transition` tuples for
                                  slots 2..``max_slots`` (``enable_new`` checks a
                                  newly-revealed slot's Use box on — Clip Chain
                                  only, per owner request).
      * ``minus_interactive`` -- False at the floor (nothing left to close, the
                                  − button greys out instead of hiding; both
                                  tabs START at their floor, so − is greyed out
                                  on page load).
      * ``plus_interactive``  -- False once every slot is open (the ＋ button
                                  greys out at the ceiling instead of hiding).
      * ``counter_text``      -- the "n/max" readout (digits only, language-
                                  independent, so it is not i18n-registered).
    """
    old_count = clamp_open_count(count, 0, min_open, max_slots)
    new_count = clamp_open_count(count, delta, min_open, max_slots)
    states = collapsible_slot_transition(old_count, new_count, max_slots,
                                         enable_new)
    return (new_count, states, new_count > min_open, new_count < max_slots,
            f"{new_count}/{max_slots}")


def compute_chain_duration_label(enabled_flags, frames_list, fps,
                                 overlap_frames, lang: str = _DEFAULT_LANG,
                                 stage2_window=None) -> str:
    """Estimated total chain duration for the currently-enabled clips, shown
    under the "Clip list" heading. Uses the SAME geometry as the server
    (``chain_math.compute_chain_layout`` + ``MAX_CHAIN_TOTAL_PIXEL_FRAMES``, the
    identical pair the authoritative precheck/API validator use) -- no formula is
    re-implemented here. Only slots whose Use flag is truthy feed the layout, in
    slot order. Degenerate geometry (e.g. the overlap not fitting inside a tiny
    clip) is swallowed and rendered as the neutral ``chain_est_none`` placeholder,
    since this is a live readout, not a submit-time gate. An over-cap total gets
    the ``chain_est_over`` warning wording. Returns a Markdown string.

    ``stage2_window`` (ADDITIVE, appended AFTER ``lang`` because ``ui.py`` wires
    this function's arguments POSITIONALLY through Gradio ``inputs=[...]``):
    the stage-2 window preset NAME; ``None`` -> the default, i.e. the exact
    geometry this readout used before the knob existed. See
    ``gradio_ui/validation.check_chain_total`` for why this is plumbed through
    even though this tab has no window selector yet."""
    import chain_math

    flags = list(enabled_flags or [])
    frames = list(frames_list or [])
    clip_frames: list[int] = []
    for flag, nf in zip(flags, frames):
        if not flag:
            continue
        try:
            clip_frames.append(int(nf))
        except (TypeError, ValueError):
            return L("chain_est_none", lang)
    if not clip_frames:
        return L("chain_est_none", lang)

    try:
        fps_v = float(fps) if fps else 24.0
        if fps_v <= 0:
            return L("chain_est_none", lang)
        kv = (int(overlap_frames) if overlap_frames is not None
              else chain_math.DEFAULT_OVERLAP_FRAMES)
        v_tile, v_adv = chain_math.resolve_stage2_window(stage2_window)
        layout = chain_math.compute_chain_layout(clip_frames, fps_v, kv=kv,
                                                 v_tile=v_tile, v_adv=v_adv)
    except (ValueError, TypeError, ZeroDivisionError):
        return L("chain_est_none", lang)

    total_px = layout.total_px
    secs = f"{total_px / fps_v:.2f}"
    if total_px > MAX_CHAIN_TOTAL_PIXEL_FRAMES:
        return L("chain_est_over", lang).format(
            sec=secs, frames=total_px, maxf=MAX_CHAIN_TOTAL_PIXEL_FRAMES)
    return L("chain_est", lang).format(sec=secs, frames=total_px)


def _chain_preset_clip_recommendation(width, height, fallback_num_frames,
                                       config: dict | None) -> int:
    """Recommended per-clip frame count for a chain preset: the resolution's
    ``limits.spill_free_frames["{width}x{height}"]`` comfortable cap when the
    server publishes one for this exact (width, height), else the preset's own
    ``num_frames``."""
    spill = ((config or {}).get("limits") or {}).get("spill_free_frames") or {}
    threshold = spill.get(f"{int(width)}x{int(height)}")
    return int(threshold) if threshold is not None else int(fallback_num_frames)


def _chain_preset_total_warning(recommended_frames, n_enabled_clips, fps,
                                 overlap_frames, lang: str = _DEFAULT_LANG,
                                 stage2_window=None):
    """Chain-total-timeline warning for ``n_enabled_clips`` clips all set to
    ``recommended_frames``. Uses the SAME geometry as the server
    (``chain_math.compute_chain_layout`` + ``MAX_CHAIN_TOTAL_PIXEL_FRAMES``,
    the identical pair ``gradio_ui.validation.check_chain_total`` uses) so this
    preview never contradicts the authoritative precheck/API validator.
    Degenerate geometry (e.g. the overlap not fitting inside a tiny clip) is
    treated as "nothing to warn about yet" rather than raised, since this is
    only a preset-preview convenience, not a submit-time gate. Returns a
    ``gr.update`` for a Markdown component (hidden with ``value=""`` when
    there is fewer than 1 clip, the geometry is degenerate, or the total is
    within budget)."""
    if n_enabled_clips is None or n_enabled_clips < 1:
        return gr.update(value="", visible=False)
    import chain_math

    try:
        fps_v = float(fps) if fps else 24.0
        kv = int(overlap_frames) if overlap_frames is not None else chain_math.DEFAULT_OVERLAP_FRAMES
        clip_frames = [int(recommended_frames)] * int(n_enabled_clips)
        # ``stage2_window`` (ADDITIVE, after ``lang`` — see
        # ``compute_chain_duration_label``): ``None`` -> the default preset.
        v_tile, v_adv = chain_math.resolve_stage2_window(stage2_window)
        layout = chain_math.compute_chain_layout(clip_frames, fps_v, kv=kv,
                                                 v_tile=v_tile, v_adv=v_adv)
    except (ValueError, TypeError):
        return gr.update(value="", visible=False)

    if layout.total_px > MAX_CHAIN_TOTAL_PIXEL_FRAMES:
        text = L("warn_chain_preset_total", lang).format(
            total=layout.total_px, max=MAX_CHAIN_TOTAL_PIXEL_FRAMES,
        )
        return gr.update(value=text, visible=True)
    return gr.update(value="", visible=False)


def apply_chain_preset(name: str, config: dict | None,
                        enabled_flags: list | None = None,
                        fps=24.0, overlap_frames=None,
                        lang: str = _DEFAULT_LANG):
    """Resolve a preset name to the Clip Chain tab's field values.

    Mirrors :func:`apply_preset`'s server-config-first / ``PRESETS``-fallback
    lookup, but targets the Clip Chain tab's components (``chain_width`` /
    ``chain_height`` / ``chain_crop_*`` + the 24 fixed clip-length ``Number``
    fields from ``chain_clip_slots`` in ``gradio_ui/ui.py``, which has NO
    single ``num_frames`` field of its own — each of the 24 slots gets the
    preset's recommended per-clip length instead) plus a chain-total-timeline
    warning, since applying a preset to every enabled slot can overshoot the
    server's total-pixel-frame cap.

    Argument order (CONTRACT for the future ``preset.change`` wiring in
    ``ui.py`` — callers MUST pass positionally/keyword in this shape):
      1. ``name``           -- selected chain-preset key (Dropdown value).
      2. ``config``         -- fetched ``/config`` dict, or ``None``. Reads
                                ``config["generation_presets"][name]`` first;
                                falls back to the hardcoded ``PRESETS`` dict
                                only when the server config is unavailable or
                                does not contain ``name`` (same precedence as
                                ``apply_preset``).
      3. ``enabled_flags``  -- optional list of the 24 clip-slot Checkbox
                                current values, in slot 1..24 order (i.e.
                                ``[s[0].value for s in chain_clip_slots]``).
                                Only its COUNT of truthy entries feeds the
                                total-timeline warning (which slots are on
                                does not matter, since every slot receives
                                the same recommended length). ``None``
                                (the default) assumes all 24 slots enabled --
                                the conservative worst case. A list shorter
                                than 24 is padded with ``True``; longer lists
                                are truncated to the first 24 entries.
      4. ``fps``             -- ``chain_fps`` Number's current value (used
                                only for the warning's audio-latent-aware
                                total-pixel-frame math). Falls back to 24.0
                                when falsy/non-numeric.
      5. ``overlap_frames``  -- ``chain_overlap`` Slider's current value (the
                                join overlap, K_v). ``None`` (the default)
                                falls back to ``chain_math.DEFAULT_OVERLAP_FRAMES``
                                (3), the same default the Slider itself uses.
      6. ``lang``            -- current UI language for ``L()`` lookups.

    Returns a 31-tuple of ``gr.update()``/plain values, in this exact order
    (CONTRACT for the ``chain_preset.change`` ``outputs=[...]`` wiring):
      ``(width, height, crop_enabled, crop_w, crop_h, crop_row_update,
      clip1_frames, clip2_frames, ..., clip24_frames, chain_total_warning_update)``
      (6 leading values + 24 per-clip frame updates + 1 warning update)

    Notes on the per-element semantics:
      * ``width`` / ``height`` -- plain ints, straight into ``chain_width`` /
        ``chain_height``.
      * ``crop_enabled`` -- bool for ``chain_crop_enabled`` (``chk_crop``),
        True iff the preset carries a non-empty ``crop_output``.
      * ``crop_w`` / ``crop_h`` -- plain ints for ``chain_crop_w`` /
        ``chain_crop_h`` (0/0 when the preset has no crop).
      * ``crop_row_update`` -- ``gr.update(visible=crop_enabled)`` for
        ``chain_crop_row`` (mirrors ``apply_preset``'s ``crop_row_update``).
      * ``clip1_frames`` .. ``clip24_frames`` -- ``gr.update(value=...)``, ALL
        24 set to the SAME recommended per-clip frame count: the preset
        resolution's ``limits.spill_free_frames["{width}x{height}"]``
        comfortable cap when the server publishes one for this exact
        resolution, else the preset's own ``num_frames`` (e.g. ``minimal``'s
        512x320 has no ``spill_free_frames`` entry -> falls back to its
        ``num_frames`` 49; ``FHD_1080p``'s 1920x1088 -> 161 from
        ``spill_free_frames``). Callers may still hand-edit individual slots
        afterwards; this only sets the initial suggestion.
      * ``chain_total_warning_update`` -- ``gr.update`` for a Markdown
        warning (hidden with ``value=""`` when within budget), worded with
        the ``warn_chain_preset_total`` i18n key ({total}/{max}
        placeholders), computed via :func:`_chain_preset_total_warning` --
        the SAME ``chain_math.compute_chain_layout`` + cap
        (``MAX_CHAIN_TOTAL_PIXEL_FRAMES`` == 11544, imported from
        ``gradio_ui.validation``) the server/`` check_chain_total`` precheck
        use, run over ``n_enabled_clips`` copies of the recommended length.
    """
    presets = (config or {}).get("generation_presets") or {}
    if name in presets:
        p = presets[name]
        width_v, height_v, frames_v = p["width"], p["height"], p["num_frames"]
        crop = p.get("crop_output")
        crop_w_v = crop["width"] if crop else 0
        crop_h_v = crop["height"] if crop else 0
    else:
        p = PRESETS.get(name) or PRESETS["minimal"]
        width_v, height_v, frames_v = p["width"], p["height"], p["num_frames"]
        crop_w_v, crop_h_v = p.get("crop_w", 0), p.get("crop_h", 0)

    crop_enabled_v = bool(crop_w_v) and bool(crop_h_v)
    crop_row_update = gr.update(visible=crop_enabled_v)

    recommended = _chain_preset_clip_recommendation(width_v, height_v, frames_v, config)
    clip_updates = tuple(gr.update(value=recommended) for _ in range(CHAIN_MAX_CLIPS))

    if enabled_flags is None:
        flags = [True] * CHAIN_MAX_CLIPS
    else:
        flags = list(enabled_flags)[:CHAIN_MAX_CLIPS]
        if len(flags) < CHAIN_MAX_CLIPS:
            flags = flags + [True] * (CHAIN_MAX_CLIPS - len(flags))
    n_enabled = sum(1 for f in flags if f)

    warning_update = _chain_preset_total_warning(
        recommended, n_enabled, fps, overlap_frames, lang,
    )

    return (width_v, height_v, crop_enabled_v, crop_w_v, crop_h_v, crop_row_update,
            *clip_updates, warning_update)
