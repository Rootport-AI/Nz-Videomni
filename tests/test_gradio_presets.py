"""Unit tests for gradio_ui/presets.py (WP-PRESETS):

  * the smoke_test/minimal/small preset rename (was phase1_default/
    phase1_target) -- pick_default_preset / apply_preset fallback behaviour,
  * format_duration_label -- the pure frames+fps -> "N.NNs" formatter,
  * apply_chain_preset -- the Clip Chain tab's preset-apply contract (width/
    height/crop/24 clip-length fields/total-timeline warning),
  * compute_chain_duration_label + the pure ± slot-count state transition
    (clamp_open_count / collapsible_slot_transition / slot_step_state)
    for the 24-slot expansion + collapsible keyframe grid.

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
    assert len(result) == 31
    (width, height, crop_enabled, crop_w, crop_h, crop_row_update,
     *clip_updates_and_warning) = result
    clip_updates = clip_updates_and_warning[:24]
    warning_update = clip_updates_and_warning[24]

    assert (width, height) == (1920, 1088)
    assert crop_enabled is True
    assert (crop_w, crop_h) == (1920, 1080)
    assert crop_row_update["visible"] is True
    # spill_free_frames["1920x1088"] == 153 -> every one of the 24 slots gets it.
    assert len(clip_updates) == 24
    for upd in clip_updates:
        assert upd["value"] == 153
    # Default (None) flags assume all 24 slots enabled: 24×153 -> total_px 3281,
    # well within the 11544-pixel-frame cap.
    assert warning_update["visible"] is False


def test_apply_chain_preset_minimal_falls_back_to_preset_num_frames():
    # 512x320 has no limits.spill_free_frames entry -> falls back to the
    # preset's own num_frames (49), not a spill-derived value.
    cfg = _fake_config_with_chain_presets()
    result = apply_chain_preset("minimal", cfg)
    clip_updates = result[6:30]
    for upd in clip_updates:
        assert upd["value"] == 49
    assert result[2] is False  # crop_enabled: minimal has crop_output=None


def test_apply_chain_preset_warns_when_total_exceeds_cap():
    cfg = _fake_config_with_chain_presets()
    # All 24 slots enabled, each recommended at 1000 frames (huge_test's
    # num_frames fallback, no spill entry for 999x999) -> total_px (computed
    # via chain_math.compute_chain_layout, same as the server) is 23441,
    # comfortably over MAX_CHAIN_TOTAL_PIXEL_FRAMES (11544).
    result = apply_chain_preset("huge_test", cfg, enabled_flags=[True] * 24)
    warning_update = result[-1]
    assert warning_update["visible"] is True
    assert "23441" in warning_update["value"]
    assert "11544" in warning_update["value"]


def test_apply_chain_preset_no_warning_when_few_clips_enabled():
    cfg = _fake_config_with_chain_presets()
    # Same oversized per-clip length, but only 1 clip enabled -> no join
    # overlap subtracted anywhere near the cap.
    result = apply_chain_preset("huge_test", cfg, enabled_flags=[True] + [False] * 23)
    warning_update = result[-1]
    assert warning_update["visible"] is False


def test_apply_chain_preset_default_enabled_flags_assumes_all_slots():
    # enabled_flags omitted -> conservative worst case (all 24 slots), so the
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
    for upd in clip_updates_and_warning[:24]:
        assert upd["value"] == 121


def test_apply_chain_preset_unknown_name_falls_back_to_minimal():
    result = apply_chain_preset("does_not_exist", {})
    assert (result[0], result[1]) == (512, 320)
    for upd in result[6:30]:
        assert upd["value"] == 49


# --------------------------------------------------------------------------- #
# compute_chain_duration_label (live chain-duration readout, 24-slot expansion)
# --------------------------------------------------------------------------- #
from gradio_ui.presets import (  # noqa: E402
    CHAIN_MAX_CLIPS,
    CHAIN_MIN_OPEN,
    KF_MAX_SLOTS,
    KF_MIN_OPEN,
    clamp_open_count,
    collapsible_slot_transition,
    compute_chain_duration_label,
    slot_step_state,
)
from gradio_ui.validation import MAX_CHAIN_TOTAL_PIXEL_FRAMES  # noqa: E402


def test_compute_chain_duration_label_basic_two_clips():
    # [121, 121] @ fps 24, overlap 3 -> chain_math total_px 225 (= 9.38s). Same
    # geometry the server uses (compute_chain_layout), not a re-implemented sum.
    txt = compute_chain_duration_label([True, True], [121, 121], 24.0, 3)
    assert "225" in txt
    assert "9.3" in txt  # 225 / 24 == 9.375 -> "9.38s"


def test_compute_chain_duration_label_excludes_disabled_slots():
    # Only the two enabled slots feed the layout; the disabled 999 is ignored, so
    # the result matches the two-clip case exactly.
    txt = compute_chain_duration_label(
        [True, False, True], [121, 999, 121], 24.0, 3)
    assert "225" in txt


def test_compute_chain_duration_label_over_cap_warns():
    # 26 clips of 481f blow past MAX_CHAIN_TOTAL_PIXEL_FRAMES; the over-cap
    # wording carries the cap number.
    txt = compute_chain_duration_label([True] * 26, [481] * 26, 24.0, 1)
    assert str(MAX_CHAIN_TOTAL_PIXEL_FRAMES) in txt  # "11544"


def test_compute_chain_duration_label_degenerate_geometry_is_neutral():
    # overlap >= the tiny clip's stage-1 frames -> chain_math raises; the readout
    # degrades to the neutral placeholder rather than surfacing an exception.
    assert compute_chain_duration_label([True, True], [9, 9], 24.0, 5) == "—"


def test_compute_chain_duration_label_all_disabled_is_neutral():
    assert compute_chain_duration_label([False, False], [121, 121], 24.0, 3) == "—"


# --------------------------------------------------------------------------- #
# ± slot-count pure state transition (clamp + per-slot visibility/Use)
# --------------------------------------------------------------------------- #
def test_clamp_open_count_clamps_to_bounds():
    # Clip chain: floor 2, ceiling 24.
    assert clamp_open_count(2, -1, CHAIN_MIN_OPEN, CHAIN_MAX_CLIPS) == 2  # Clip 2 never closes
    assert clamp_open_count(24, +1, CHAIN_MIN_OPEN, CHAIN_MAX_CLIPS) == 24  # ceiling
    assert clamp_open_count(3, +1, CHAIN_MIN_OPEN, CHAIN_MAX_CLIPS) == 4
    assert clamp_open_count(5, -1, CHAIN_MIN_OPEN, CHAIN_MAX_CLIPS) == 4
    # Keyframe: floor 1, ceiling 5.
    assert clamp_open_count(1, -1, KF_MIN_OPEN, KF_MAX_SLOTS) == 1  # KF 1 never closes
    assert clamp_open_count(5, +1, KF_MIN_OPEN, KF_MAX_SLOTS) == 5
    # Garbage count falls back to the floor before the delta is applied
    # (None -> floor 2, then a no-op step stays at the floor).
    assert clamp_open_count(None, 0, CHAIN_MIN_OPEN, CHAIN_MAX_CLIPS) == CHAIN_MIN_OPEN
    assert clamp_open_count("x", 0, KF_MIN_OPEN, KF_MAX_SLOTS) == KF_MIN_OPEN


def test_collapsible_slot_transition_visibility_and_use_off():
    # No-move "transition" at count 2 over 24 slots: slot 2 visible (Use
    # untouched -> None), slots 3..24 hidden (Use forced off -> False).
    # Returns 23 (slot 2..24) tuples.
    states = collapsible_slot_transition(2, 2, CHAIN_MAX_CLIPS)
    assert len(states) == CHAIN_MAX_CLIPS - 1
    assert states[0] == (True, None)    # slot 2 visible, Use left as is
    assert all(vis is False and use is False for vis, use in states[1:])  # 3..24 hidden+off

    # Shrink 4 -> 3: slots 2-3 stay visible (None), 4..24 hidden+off.
    states3 = collapsible_slot_transition(4, 3, CHAIN_MAX_CLIPS)
    assert states3[0] == (True, None) and states3[1] == (True, None)
    assert all(use is False for _vis, use in states3[2:])


def test_collapsible_slot_transition_plus_enables_new_slots_only():
    # Grow 2 -> 4 with enable_new (Clip Chain "＋"): the NEWLY revealed slots 3-4
    # get Use forced ON; the already-visible slot 2 is left alone (a
    # hand-unchecked box must not be re-checked); 5..24 stay hidden+off.
    states = collapsible_slot_transition(2, 4, CHAIN_MAX_CLIPS, enable_new=True)
    assert states[0] == (True, None)    # slot 2: untouched
    assert states[1] == (True, True)    # slot 3: newly revealed -> Use on
    assert states[2] == (True, True)    # slot 4: newly revealed -> Use on
    assert all(vis is False and use is False for vis, use in states[3:])

    # Shrink with enable_new never turns anything on: 4 -> 2 forces 3..24 off.
    states_dn = collapsible_slot_transition(4, 2, CHAIN_MAX_CLIPS, enable_new=True)
    assert states_dn[0] == (True, None)
    assert all(use is False for _vis, use in states_dn[1:])


def test_collapsible_slot_transition_keyframe_grid_no_auto_enable():
    # KF grid (enable_new omitted): slot 1 always visible (not in the list);
    # growing 1 -> 2 reveals slot 2 WITHOUT checking its Use box (owner asked
    # for auto-enable on the Clip Chain only).
    states = collapsible_slot_transition(1, 2, KF_MAX_SLOTS)
    assert len(states) == KF_MAX_SLOTS - 1
    assert states[0] == (True, None)    # revealed but Use untouched
    assert all(vis is False and use is False for vis, use in states[1:])


# --------------------------------------------------------------------------- #
# slot_step_state (± semantic state, shared by the keyframe grid — min 1/max 5 —
# and the Clip Chain clip list — min 2/max 24: slot visibility/Use directives +
# −/＋ grey-out flags + the "n/max" counter text)
# --------------------------------------------------------------------------- #
def _kf_step(count, delta):
    return slot_step_state(count, delta, KF_MIN_OPEN, KF_MAX_SLOTS)


def _chain_step(count, delta):
    return slot_step_state(count, delta, CHAIN_MIN_OPEN, CHAIN_MAX_CLIPS,
                           enable_new=True)


def test_slot_step_state_kf_plus_from_default():
    # Startup floor (1 open): ＋ reveals slot 2 WITHOUT checking its Use box,
    # and − comes back to life (no longer at the floor).
    new_count, states, minus_on, plus_on, counter = _kf_step(1, +1)
    assert new_count == 2
    assert states[0][0] is True          # slot 2 now visible
    assert states[0][1] is None          # ...but its Use box untouched
    assert minus_on is True              # above the floor again
    assert plus_on is True               # still below the 5-slot ceiling
    assert counter == "2/5"


def test_slot_step_state_kf_minus_greys_out_at_floor():
    # 2 -> 1 lands ON the floor: − greys out (nothing left to close), ＋ stays
    # available. This is also the page-load state (both tabs start at the
    # floor, so − is created interactive=False).
    new_count, states, minus_on, plus_on, counter = _kf_step(2, -1)
    assert new_count == 1
    assert states[0] == (False, False)   # slot 2 hidden + Use forced off
    assert minus_on is False
    assert plus_on is True
    assert counter == "1/5"
    # A further − is a clamped no-op that stays greyed out.
    new_count2, _s, minus_on2, plus_on2, counter2 = _kf_step(1, -1)
    assert (new_count2, minus_on2, plus_on2, counter2) == (1, False, True, "1/5")


def test_slot_step_state_kf_plus_greys_out_at_ceiling():
    # Reaching the ceiling disables ＋ (grey-out, not hidden); − stays usable.
    new_count, states, minus_on, plus_on, counter = _kf_step(4, +1)
    assert new_count == 5
    assert all(vis for vis, _use in states)  # all 5 rows open
    assert (minus_on, plus_on) == (True, False)
    assert counter == "5/5"
    # ...and a further ＋ is a clamped no-op that stays greyed out.
    new_count2, _s, minus_on2, plus_on2, counter2 = _kf_step(5, +1)
    assert (new_count2, minus_on2, plus_on2, counter2) == (5, True, False, "5/5")


def test_slot_step_state_kf_mid_range_both_enabled():
    new_count, _states, minus_on, plus_on, counter = _kf_step(2, +1)
    assert new_count == 3
    assert (minus_on, plus_on) == (True, True)
    assert counter == "3/5"


def test_slot_step_state_chain_minus_greys_out_at_floor():
    # Clip Chain floor is 2: 3 -> 2 greys out −; Clip 2 never closes.
    new_count, states, minus_on, plus_on, counter = _chain_step(3, -1)
    assert new_count == 2
    assert states[0] == (True, None)     # slot 2 stays visible, Use untouched
    assert states[1] == (False, False)   # slot 3 hidden + Use forced off
    assert (minus_on, plus_on) == (False, True)
    assert counter == "2/24"
    # A further − is a clamped no-op (startup state: − greyed out).
    new_count2, _s, minus_on2, _p, counter2 = _chain_step(2, -1)
    assert (new_count2, minus_on2, counter2) == (2, False, "2/24")


def test_slot_step_state_chain_plus_greys_out_at_ceiling_and_enables_new():
    # 23 -> 24 hits the ceiling: ＋ greys out, − stays usable, and the newly
    # revealed slot 24 gets Use forced ON (enable_new, Clip Chain behaviour).
    new_count, states, minus_on, plus_on, counter = _chain_step(23, +1)
    assert new_count == 24
    assert states[-1] == (True, True)    # slot 24: newly revealed -> Use on
    assert (minus_on, plus_on) == (True, False)
    assert counter == "24/24"


def test_slot_step_state_chain_mid_range_both_enabled():
    new_count, states, minus_on, plus_on, counter = _chain_step(2, +1)
    assert new_count == 3
    assert states[1] == (True, True)     # slot 3: newly revealed -> Use on
    assert (minus_on, plus_on) == (True, True)
    assert counter == "3/24"
