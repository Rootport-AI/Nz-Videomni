"""Presets (S1 fallback only — S2 replaces this with /config generation_presets).
width/height are multiples of 64 (two-stage distilled).
"""

from __future__ import annotations

import gradio as gr

from .i18n import L, _DEFAULT_LANG

PRESETS: dict[str, dict] = {
    "smoke_test": {"width": 384, "height": 256, "num_frames": 17, "crop_w": 0, "crop_h": 0},
    "phase1_default": {"width": 512, "height": 320, "num_frames": 49, "crop_w": 0, "crop_h": 0},
    "phase1_target": {"width": 960, "height": 576, "num_frames": 121, "crop_w": 960, "crop_h": 540},
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
    return "phase1_default"


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
        p = PRESETS.get(name) or PRESETS["phase1_default"]
        width_v, height_v, frames_v = p["width"], p["height"], p["num_frames"]
        crop_w_v, crop_h_v = p.get("crop_w", 0), p.get("crop_h", 0)

    crop_enabled_v = bool(crop_w_v) and bool(crop_h_v)
    crop_row_update = gr.update(visible=crop_enabled_v)
    spill_update = compute_spill_warning(width_v, height_v, frames_v, config, lang)
    return (width_v, height_v, frames_v, crop_enabled_v, crop_w_v, crop_h_v,
            crop_row_update, spill_update)
