"""Unit tests for gradio_ui/presets.py (WP-PRESETS):

  * the smoke_test/minimal/small preset rename (was phase1_default/
    phase1_target) -- pick_default_preset / apply_preset fallback behaviour,
  * format_duration_label -- the pure frames+fps -> "N.NNs" formatter,
  * apply_chain_preset -- the Clip Chain tab's preset-apply contract (width/
    height/crop/8 clip-length fields/total-timeline warning).

These never touch the server (no HTTP, no live backend) -- pure-function
tests only, matching tests/test_gradio_handlers.py's S2 preset test style.
"""

from __future__ import annotations

from gradio_ui.presets import (
    PRESETS,
    apply_chain_preset,
    apply_preset,
    format_duration_label,
    pick_default_preset,
)


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _fake_config_with_chain_presets() -> dict:
    return {
        "generation_presets": {
            "minimal": {"width": 512, "height": 320, "crop_output": None, "num_frames": 49},
            "FHD_1080p": {
                "width": 1920, "height": 1088,
                "crop_output": {"width": 1920, "height": 1080},
                "num_frames": 153,
            },
            # No spill_free_frames entry for this resolution on purpose --
            # exercises the "fall back to the preset's own num_frames" path
            # with a per-clip length large enough to blow the chain-total cap
            # once several slots are enabled.
            "huge_test": {
                "width": 999, "height": 999, "crop_output": None, "num_frames": 1000,
            },
        },
        "limits": {
            "spill_free_frames": {"1280x768": 257, "1920x1088": 153, "2560x1472": 81},
        },
    }


# --------------------------------------------------------------------------- #
# renamed presets: smoke_test / minimal / small (was phase1_default / phase1_target)
# --------------------------------------------------------------------------- #
def test_presets_dict_uses_renamed_keys():
    assert set(PRESETS.keys()) == {"smoke_test", "minimal", "small"}
    assert "phase1_default" not in PRESETS
    assert "phase1_target" not in PRESETS


def test_pick_default_preset_fallback_is_minimal():
    assert pick_default_preset({}) == "minimal"
    assert pick_default_preset(None) == "minimal"


def test_apply_preset_fallback_uses_small_by_name():
    (width, height, frames, crop_enabled, crop_w, crop_h,
     crop_row_update, _spill_update) = apply_preset("small", {})
    assert (width, height, frames) == (960, 576, 121)
    assert crop_enabled is True
    assert (crop_w, crop_h) == (960, 540)
    assert crop_row_update["visible"] is True


def test_apply_preset_fallback_unknown_name_uses_minimal():
    (width, height, frames, *_rest) = apply_preset("does_not_exist", {})
    assert (width, height, frames) == (512, 320, 49)


# --------------------------------------------------------------------------- #
# format_duration_label
# --------------------------------------------------------------------------- #
def test_format_duration_label_typical_values():
    assert format_duration_label(257, 24) == "10.71s"
    assert format_duration_label(121, 24) == "5.04s"


def test_format_duration_label_zero_fps_returns_empty():
    assert format_duration_label(257, 0) == ""
    assert format_duration_label(257, -1) == ""


def test_format_duration_label_none_returns_empty():
    assert format_duration_label(None, 24) == ""
    assert format_duration_label(257, None) == ""
    assert format_duration_label(None, None) == ""


def test_format_duration_label_non_numeric_returns_empty():
    assert format_duration_label("abc", 24) == ""
    assert format_duration_label(257, "abc") == ""


# --------------------------------------------------------------------------- #
# apply_chain_preset
# --------------------------------------------------------------------------- #
def test_apply_chain_preset_fhd_1080p_fills_resolution_and_clip_lengths():
    cfg = _fake_config_with_chain_presets()
    result = apply_chain_preset("FHD_1080p", cfg)
    assert len(result) == 15
    (width, height, crop_enabled, crop_w, crop_h, crop_row_update,
     *clip_updates_and_warning) = result
    clip_updates = clip_updates_and_warning[:8]
    warning_update = clip_updates_and_warning[8]

    assert (width, height) == (1920, 1088)
    assert crop_enabled is True
    assert (crop_w, crop_h) == (1920, 1080)
    assert crop_row_update["visible"] is True
    # spill_free_frames["1920x1088"] == 153 -> every one of the 8 slots gets it.
    assert len(clip_updates) == 8
    for upd in clip_updates:
        assert upd["value"] == 153
    # Only 2 clips assumed enabled -> well within the 3848-pixel-frame cap.
    assert warning_update["visible"] is False


def test_apply_chain_preset_minimal_falls_back_to_preset_num_frames():
    # 512x320 has no limits.spill_free_frames entry -> falls back to the
    # preset's own num_frames (49), not a spill-derived value.
    cfg = _fake_config_with_chain_presets()
    result = apply_chain_preset("minimal", cfg)
    clip_updates = result[6:14]
    for upd in clip_updates:
        assert upd["value"] == 49
    assert result[2] is False  # crop_enabled: minimal has crop_output=None


def test_apply_chain_preset_warns_when_total_exceeds_cap():
    cfg = _fake_config_with_chain_presets()
    # All 8 slots enabled, each recommended at 1000 frames (huge_test's
    # num_frames fallback, no spill entry for 999x999) -> total_px (computed
    # via chain_math.compute_chain_layout, same as the server) is 7825,
    # comfortably over MAX_CHAIN_TOTAL_PIXEL_FRAMES (3848).
    result = apply_chain_preset("huge_test", cfg, enabled_flags=[True] * 8)
    warning_update = result[-1]
    assert warning_update["visible"] is True
    assert "7825" in warning_update["value"]
    assert "3848" in warning_update["value"]


def test_apply_chain_preset_no_warning_when_few_clips_enabled():
    cfg = _fake_config_with_chain_presets()
    # Same oversized per-clip length, but only 1 clip enabled -> no join
    # overlap subtracted anywhere near the cap.
    result = apply_chain_preset("huge_test", cfg, enabled_flags=[True] + [False] * 7)
    warning_update = result[-1]
    assert warning_update["visible"] is False


def test_apply_chain_preset_default_enabled_flags_assumes_all_eight():
    # enabled_flags omitted -> conservative worst case (all 8 slots), so the
    # oversized huge_test preset still warns without the caller passing flags.
    cfg = _fake_config_with_chain_presets()
    result = apply_chain_preset("huge_test", cfg)
    assert result[-1]["visible"] is True


def test_apply_chain_preset_fallback_when_config_empty():
    # No server config at all -> PRESETS fallback dict ("small": 960x576,
    # num_frames=121, crop 960x540).
    result = apply_chain_preset("small", {})
    (width, height, crop_enabled, crop_w, crop_h, crop_row_update,
     *clip_updates_and_warning) = result
    assert (width, height) == (960, 576)
    assert crop_enabled is True
    assert (crop_w, crop_h) == (960, 540)
    assert crop_row_update["visible"] is True
    for upd in clip_updates_and_warning[:8]:
        assert upd["value"] == 121


def test_apply_chain_preset_unknown_name_falls_back_to_minimal():
    result = apply_chain_preset("does_not_exist", {})
    assert (result[0], result[1]) == (512, 320)
    for upd in result[6:14]:
        assert upd["value"] == 49
