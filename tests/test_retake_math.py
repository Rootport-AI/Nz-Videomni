"""Retake (temporal inpainting) geometry — the pure ``chain_math`` layer.

Pins the numbers the app validator, the engine and the mock ALL resolve from
``compute_chain_layout(retake_glue_px=...)``. If any expectation here moves, the
three of them have stopped agreeing and the freeze bands land on the wrong
latents (a silent quality failure, not a crash) — see VERIFICATION_LOG §55.
"""

from __future__ import annotations

import pytest

import chain_math as cm

# The owner-approved default: a 169-frame window with 25/24 glue at 24fps
# (VERIFICATION_LOG §55.5 — 9/8 is too weak, 49/48 weakens the audio seam).
WINDOW, HEAD, TAIL, FPS = 169, 25, 24, 24.0


def _layout(window=WINDOW, head=HEAD, tail=TAIL, fps=FPS, **kw):
    return cm.compute_chain_layout([window], fps, retake_glue_px=(head, tail), **kw)


# ── the canonical expected values ────────────────────────────────────────────
def test_default_window_pins_every_derived_number():
    layout = _layout()
    assert layout.f_total == 22
    assert layout.total_px == 169
    assert layout.a_total == 176            # a_win
    assert layout.n_tiles == 1
    d = layout.to_dict()["retake"]
    assert d == {
        "window_px": 169,
        "head_px": 25,
        "tail_px": 24,
        "n_head_v": 4,
        "n_tail_v": 3,
        "n_head_a": 26,
        "n_tail_a": 24,
        "free_middle_px": [25, 145],
    }


def test_primary_data_only_is_stored_on_the_layout():
    # The dataclass carries the two INPUTS and nothing derived: a stored copy of
    # n_head_a (etc.) could drift away from the function the engine calls.
    layout = _layout()
    assert layout.retake_window_px == 169
    assert layout.retake_glue_px == (25, 24)
    for absent in ("n_head_v", "n_tail_v", "n_head_a", "n_tail_a", "free_middle_px"):
        assert not hasattr(layout, absent)


def test_no_retake_key_when_glue_is_none():
    # A normal chain's to_dict() key set must be byte-unchanged.
    assert "retake" not in cm.compute_chain_layout([73, 73], 24.0).to_dict()
    assert cm.compute_chain_layout([73, 73], 24.0).retake_glue_px is None


# ── the single-stage-2-tile invariant (the reason the window is capped) ──────
def test_every_legal_window_on_the_8n1_grid_is_one_stage2_tile():
    lo, hi = cm.RETAKE_WINDOW_MIN_PX, cm.retake_max_window_px(cm.STAGE2_V_TILE)
    assert (lo, hi) == (73, 169)
    windows = list(range(lo, hi + 1, 8))
    assert windows[0] == 73 and windows[-1] == 169
    for w in windows:
        assert cm.compute_chain_layout([w], FPS).n_tiles == 1, w
        assert _layout(window=w, head=9, tail=8).n_tiles == 1, w


def test_177_frames_would_split_into_two_tiles():
    # 177 is the next 8n+1 step past the cap; this is WHY the cap is 169.
    assert cm.compute_chain_layout([177], FPS).n_tiles == 2
    with pytest.raises(ValueError, match=r"\[73, 169\]"):
        _layout(window=177)


def test_max_window_tracks_v_tile_and_is_the_sibling_of_stage2_max_context():
    assert cm.retake_max_window_px(22) == 169
    assert cm.retake_max_window_px(19) == 145      # "high_resolution" preset
    # V2V's ceiling is one latent frame lower (tile 0 must keep something to
    # generate); retake deliberately fills the whole tile.
    assert cm.stage2_max_context_px(22) == 161


# ── the causal audio grid + the H-A1' scan ───────────────────────────────────
@pytest.mark.parametrize("i", [0, 1, 2, 25, 151, 152, 175])
def test_audio_latent_support_is_the_wheels_causal_closed_form(i):
    assert cm.audio_latent_support_sec(i) == (max(4 * i - 3, 0) / 100.0, (4 * i + 1) / 100.0)


def test_audio_latent_zero_is_clamped_at_time_zero():
    assert cm.audio_latent_support_sec(0) == (0.0, 0.01)


def test_scan_never_overshoots_into_the_regenerated_middle():
    # The whole point of the scan: no frozen latent may overlap the free middle.
    n_head, n_tail = cm.retake_audio_glue_latents(
        window_px=WINDOW, head_px=HEAD, tail_px=TAIL, fps=FPS, a_win=176
    )
    assert (n_head, n_tail) == (26, 24)
    head_band_end = HEAD / FPS
    tail_band_start = (WINDOW - TAIL) / FPS
    assert cm.audio_latent_support_sec(n_head - 1)[1] <= head_band_end + 1e-9
    assert cm.audio_latent_support_sec(n_head)[1] > head_band_end + 1e-9
    first_tail = 176 - n_tail
    assert cm.audio_latent_support_sec(first_tail)[0] >= tail_band_start - 1e-9
    assert cm.audio_latent_support_sec(first_tail - 1)[0] < tail_band_start - 1e-9


def test_scan_holds_over_the_whole_legal_grid():
    for window in range(73, 170, 8):
        for head in range(9, window, 8):
            for tail in range(8, window - head, 8):
                a_win = cm.a_frames_for_px(window, FPS)
                n_head, n_tail = cm.retake_audio_glue_latents(
                    window_px=window, head_px=head, tail_px=tail, fps=FPS, a_win=a_win
                )
                assert n_head + n_tail <= a_win
                if n_head:
                    assert cm.audio_latent_support_sec(n_head - 1)[1] <= head / FPS + 1e-9
                if n_tail:
                    first = a_win - n_tail
                    assert (
                        cm.audio_latent_support_sec(first)[0]
                        >= (window - tail) / FPS - 1e-9
                    )


def _glue_latents_before_the_split(*, window_px, head_px, tail_px, fps, a_win):
    """The pre-refactor body of ``retake_audio_glue_latents``, VERBATIM.

    The end source needed the tail scan on its own, so the two loops moved into
    :func:`chain_math.audio_head_latents` / :func:`chain_math.audio_tail_latents`
    and the retake entry point became a wrapper. This copy is what the wrapper is
    measured against below: retake's non-regression is then a machine fact rather
    than a reading of the diff. It must NOT be updated to follow the module — if
    the two ever disagree, the module moved and retake moved with it.
    """
    eps = 1e-9
    head_s = head_px / float(fps)
    tail_start_s = (window_px - tail_px) / float(fps)
    n_head = 0
    for i in range(a_win):
        if cm.audio_latent_support_sec(i)[1] <= head_s + eps:
            n_head = i + 1
        else:
            break
    first_tail = a_win
    for i in range(a_win - 1, -1, -1):
        if cm.audio_latent_support_sec(i)[0] >= tail_start_s - eps:
            first_tail = i
        else:
            break
    n_head = max(0, min(n_head, a_win))
    first_tail = max(0, min(first_tail, a_win))
    return n_head, a_win - first_tail


def test_the_split_into_head_and_tail_scans_returns_the_same_numbers():
    """Exhaustive over every fps the app can generate at and every legal
    window/head/tail on the grid: the wrapper's pair equals the old body's pair,
    and each half equals the wrapper's corresponding entry."""
    checked = 0
    for fps in (23.976, 24.0, 25.0, 29.97, 30.0, 48.0, 50.0, 59.94, 60.0):
        for window in range(73, 258, 8):
            a_win = cm.a_frames_for_px(window, fps)
            for head in range(9, window, 16):
                for tail in range(8, window - head, 16):
                    kw = dict(
                        window_px=window, head_px=head, tail_px=tail,
                        fps=fps, a_win=a_win,
                    )
                    got = cm.retake_audio_glue_latents(**kw)
                    assert got == _glue_latents_before_the_split(**kw), kw
                    assert got[0] == cm.audio_head_latents(
                        head_px=head, fps=fps, a_win=a_win
                    )
                    assert got[1] == cm.audio_tail_latents(
                        window_px=window, tail_px=tail, fps=fps, a_win=a_win
                    )
                    checked += 1
    assert checked > 1000


# ── head/tail latent asymmetry (causal VAE) ──────────────────────────────────
def test_head_and_tail_use_different_latent_grids():
    assert cm.v_latent_frames(25) == 4        # head: 8n+1 px -> (px-1)//8+1
    assert cm.v_tail_latents(24) == 3         # tail: multiple of 8 -> px//8
    assert cm.v_latent_frames(9) == 2 and cm.v_tail_latents(8) == 1
    assert cm.v_latent_frames(1) == 1 and cm.v_tail_latents(0) == 0
    assert cm.v_latent_frames(169) == 22 and cm.v_tail_latents(168) == 21


# ── absolute tail token addressing ───────────────────────────────────────────
def test_tail_token_range_is_absolute_and_ignores_appended_tokens():
    lo, hi = cm.retake_tail_token_range(latent_frames=22, hw=240, n_tail=3)
    assert (lo, hi) == (19 * 240, 22 * 240) == (4560, 5280)
    # The negative form a caller might reach for grabs a DIFFERENT range as soon
    # as conditioning tokens have been appended — that is the bug this guards.
    total_with_conditioning = 22 * 240 + 512
    assert total_with_conditioning - 3 * 240 != lo


def test_tail_token_range_covers_everything_and_rejects_bad_counts():
    assert cm.retake_tail_token_range(22, 240, 22) == (0, 5280)
    for bad in (0, -1, 23):
        with pytest.raises(ValueError):
            cm.retake_tail_token_range(22, 240, bad)


# ── rejections ───────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(window=65), r"\[73, 169\]"),                 # below the floor
        (dict(window=177), r"\[73, 169\]"),                # above the ceiling
        (dict(window=100), "8n\\+1"),                      # off the window grid
        (dict(head=24), "head_px must be 8n\\+1"),         # head off-grid
        (dict(head=1), "head_px must be >= 9"),            # degenerate keyframe head
        (dict(tail=25), "multiple of 8"),                  # tail off-grid
        (dict(tail=0), "tail_px must be >= 8"),            # empty tail
        (dict(head=81, tail=88), "free middle"),           # pixel middle exhausted
    ],
)
def test_geometric_rejections(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _layout(**kwargs)


def test_retake_and_v2v_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        cm.compute_chain_layout([169], FPS, retake_glue_px=(25, 24), source_context_px=25)


def test_retake_requires_exactly_one_clip():
    with pytest.raises(ValueError, match="exactly 1 clip"):
        cm.compute_chain_layout([169, 169], FPS, retake_glue_px=(25, 24))


def test_free_audio_middle_is_checked_independently_of_the_pixel_middle():
    # A pixel middle can survive while the audio grid's does not; both are
    # checked, so this window/glue combination must be caught by ONE of them.
    layout = _layout(window=73, head=25, tail=24)
    d = layout.to_dict()["retake"]
    assert d["n_head_a"] + d["n_tail_a"] < layout.a_total


def test_non_default_stage2_window_narrows_the_ceiling():
    v_tile, v_adv = cm.resolve_stage2_window("high_resolution")
    assert cm.compute_chain_layout(
        [145], FPS, retake_glue_px=(25, 24), v_tile=v_tile, v_adv=v_adv
    ).n_tiles == 1
    with pytest.raises(ValueError, match=r"\[73, 145\]"):
        cm.compute_chain_layout(
            [169], FPS, retake_glue_px=(25, 24), v_tile=v_tile, v_adv=v_adv
        )
