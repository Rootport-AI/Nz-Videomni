"""chain_math.alpha_gen_geometry — AlphaGen working size and padded canvas.

Examples are the ones fixed in the AlphaGen phase-1 plan (section 4 (2));
the no-shrink cases match gate 0's B1/B2 (VERIFICATION_LOG §151).
"""

from __future__ import annotations

import pytest

from chain_math import alpha_gen_geometry, v_latent_frames


def _tokens(cw: int, ch: int, frames: int) -> int:
    return (cw // 32) * (ch // 32) * v_latent_frames(frames)


@pytest.mark.parametrize(
    ("src", "frames", "budget", "expected"),
    [
        # 4bit (B=23,460): 1920x1088x145f shrinks to 1468x832 on a 1472x832 canvas
        ((1920, 1088), 145, 23_460, (1468, 832, 1472, 832)),
        # Q6_K (B=21,672): 1920x1088x81f shrinks to 1808x1024 on a 1856x1024 canvas
        ((1920, 1088), 81, 21_672, (1808, 1024, 1856, 1024)),
    ],
)
def test_plan_examples(src, frames, budget, expected):
    assert alpha_gen_geometry(src[0], src[1], frames, budget) == expected
    cw, ch = expected[2], expected[3]
    assert _tokens(cw, ch, frames) <= budget


def test_plan_example_token_counts():
    assert _tokens(1472, 832, 145) == 22_724
    assert _tokens(1856, 1024, 81) == 20_416


@pytest.mark.parametrize(
    ("src", "frames"),
    [((1280, 768), 121), ((1920, 1088), 81)],
)
def test_no_shrink_within_budget(src, frames):
    w, h = src
    assert alpha_gen_geometry(w, h, frames, 23_460) == (w, h, w, h)


def test_budget_none_never_shrinks():
    assert alpha_gen_geometry(3840, 2160, 145, None) == (3840, 2160, 3840, 2176)


def test_portrait():
    ww, wh, cw, ch = alpha_gen_geometry(1088, 1920, 145, 23_460)
    assert (ww, wh, cw, ch) == (832, 1468, 832, 1472)
    assert _tokens(cw, ch, 145) <= 23_460


def test_monotonic_in_budget():
    prev = (0, 0)
    for budget in range(4_000, 40_001, 500):
        ww, wh, _, _ = alpha_gen_geometry(1920, 1088, 145, budget)
        assert ww >= prev[0] and wh >= prev[1], budget
        prev = (ww, wh)


@pytest.mark.parametrize("budget", [6_000, 12_345, 21_672, 19_380, 23_460, 30_000])
@pytest.mark.parametrize(
    ("src", "frames"),
    [((1920, 1088), 145), ((1088, 1920), 121), ((1280, 720), 145), ((2560, 1440), 97), ((1366, 768), 145)],
)
def test_shape_invariants(src, frames, budget):
    ww, wh, cw, ch = alpha_gen_geometry(src[0], src[1], frames, budget)
    assert ww % 2 == 0 and wh % 2 == 0
    assert cw % 64 == 0 and ch % 64 == 0
    assert 0 <= cw - ww < 64 and 0 <= ch - wh < 64
    assert ww <= src[0] and wh <= src[1]
    assert _tokens(cw, ch, frames) <= budget


def test_no_fit_raises():
    with pytest.raises(ValueError):
        alpha_gen_geometry(1920, 1088, 145, 1)
