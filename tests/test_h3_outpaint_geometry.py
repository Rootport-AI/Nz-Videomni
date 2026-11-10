"""engine_h3.outpaint_geometry: the Outpainting geometry and paste-back (台帳 §1-125, CPU gate G-C).

Runs in ``.venv-engine-h3`` through the engine-venv runner::

    .venv-engine-h3/Scripts/python.exe scripts/run_h3_pytest.py tests/test_h3_outpaint_geometry.py

The expected values are computed here independently (token by token, pixel by
pixel) from the definitions in Docs/MINIMAX_H3_DESIGN.md §3.1 and the plan's §2.2.
"""

from __future__ import annotations

import math
import sys

import numpy as np
import pytest

from engine_h3 import outpaint_geometry as G


def _geom(sw, sh, pl, pr, pt, pb):
    return G.compute_outpaint_geometry(sw, sh, sw + pl + pr, sh + pt + pb, pl, pr, pt, pb)


def _fixed_by_hand(g):
    """Token (i, j) fixed iff [32j, 32j+32) x [32i, 32i+32) lies inside the kept rectangle."""
    want = np.zeros((g.canvas_h // 32, g.canvas_w // 32), dtype=bool)
    for i in range(want.shape[0]):
        for j in range(want.shape[1]):
            want[i, j] = (32 * j >= g.pad_left and 32 * j + 32 <= g.pad_left + g.source_w
                          and 32 * i >= g.pad_top and 32 * i + 32 <= g.pad_top + g.source_h)
    return want


def _frames(f, h, w, seed=0):
    return np.random.default_rng(seed).integers(0, 256, size=(f, h, w, 3), dtype=np.uint8)


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #

def test_torch_free():
    # The module must not pull torch in (the plan's §6.1: numpy only).
    import importlib
    import subprocess

    code = "import sys; import engine_h3.outpaint_geometry; print('torch' in sys.modules)"
    root = str(__import__("pathlib").Path(__file__).resolve().parents[1])
    out = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
    assert importlib.import_module("engine_h3.constants").SPATIAL_FACTOR == G.TOKEN_PX == 32
    assert G.LATENT_PX == 16


def test_720p_to_1920x1088_by_hand():
    g = _geom(1280, 720, 320, 320, 184, 184)
    assert (g.canvas_w, g.canvas_h) == (1920, 1088)
    assert (g.x0, g.y0, g.x1, g.y1) == (320, 160, 1600, 928)
    assert (g.aligned_w, g.aligned_h) == (1280, 768)
    assert (g.grid_w, g.grid_h) == (60, 34)
    # Columns 10..49 (40), rows ceil(184/32)=6 .. floor(904/32)-1=27 (22).
    cols = np.flatnonzero(g.fixed.any(0))
    rows = np.flatnonzero(g.fixed.any(1))
    assert cols.tolist() == list(range(10, 50)) and rows.tolist() == list(range(6, 28))
    assert g.fixed_count == 40 * 22 and g.generated_count == 60 * 34 - 880
    assert np.array_equal(g.fixed, _fixed_by_hand(g))
    assert g.latent_offset == (10, 20)
    assert g.source_offset_in_aligned == (24, 0)


def test_1080p_bottom_edge():
    g = _geom(1920, 1080, 0, 0, 0, 8)
    assert (g.canvas_w, g.canvas_h) == (1920, 1088)
    assert (g.x0, g.y0, g.x1, g.y1) == (0, 0, 1920, 1088)
    # The last token row [1056, 1088) holds 8 px of pad: generated.
    assert g.fixed[:-1].all() and not g.fixed[-1].any()
    assert g.fixed_count == 60 * 33
    assert np.array_equal(g.fixed, _fixed_by_hand(g))


@pytest.mark.parametrize("pad", [0, 1, 31, 32, 33])
def test_pad_boundaries_left(pad):
    # Left pad varies; the right pad makes the canvas a multiple of 32.
    sw, sh = 320, 256
    pr = (-(sw + pad)) % 32 + (32 if pad == 0 else 0)
    g = _geom(sw, sh, pad, pr, 0, 0)
    assert g.canvas_w % 32 == 0 and g.x1 <= g.canvas_w and g.y1 <= g.canvas_h
    assert g.x0 == (pad // 32) * 32 and g.x1 == math.ceil((pad + sw) / 32) * 32
    assert g.x0 <= pad and g.x1 >= pad + sw and g.x0 % 32 == 0 and g.x1 % 32 == 0
    assert g.aligned_w % 32 == 0 and g.aligned_h % 32 == 0
    assert np.array_equal(g.fixed, _fixed_by_hand(g))
    first = math.ceil(pad / 32)
    assert np.flatnonzero(g.fixed.any(0))[0] == first
    assert g.fixed_count >= 1


@pytest.mark.parametrize("pads", [(0, 32, 0, 0), (12, 20, 0, 0), (0, 0, 7, 25), (1, 31, 33, 31)])
def test_pad_zero_sides_and_odd_pads(pads):
    pl, pr, pt, pb = pads
    sw, sh = 256, 256
    g = _geom(sw, sh, pl, pr, pt, pb)
    assert g.x1 <= g.canvas_w and g.y1 <= g.canvas_h
    assert np.array_equal(g.fixed, _fixed_by_hand(g))
    if pl == 0:
        assert g.fixed[:, 0].any()
    if pt == 0:
        assert g.fixed[0, :].any()


def test_256_source_has_a_fixed_token():
    g = _geom(256, 256, 31, 1, 31, 1)
    assert g.fixed_count == 7 * 7


@pytest.mark.parametrize("args", [
    (256, 256, 256, 256, 0, 0, 0, 0),          # no pad
    (256, 256, 288, 256, -32, 64, 0, 0),       # negative pad
    (250, 256, 260, 256, 5, 5, 0, 0),          # canvas not a multiple of 32
    (256, 256, 320, 256, 32, 0, 0, 0),         # canvas != source + pads
    (16, 16, 64, 64, 1, 47, 1, 47),            # no token fully inside
    (256, 256, 288, 256, True, 31, 0, 0),      # bool is not an int
])
def test_refuses_bad_geometry(args):
    with pytest.raises(ValueError):
        G.compute_outpaint_geometry(*args)


def test_fixed_is_read_only():
    g = _geom(256, 256, 32, 32, 0, 0)
    with pytest.raises(ValueError):
        g.fixed[0, 0] = False


# --------------------------------------------------------------------------- #
# aligned frames (reflect gaps) and token values
# --------------------------------------------------------------------------- #

def test_aligned_frames_mirror_the_edges():
    g = _geom(256, 256, 12, 20, 5, 27)
    src = _frames(2, 256, 256)
    out = G.aligned_frames(src, g)
    assert out.shape == (2, g.aligned_h, g.aligned_w, 3) and out.dtype == np.uint8
    top, left = g.source_offset_in_aligned
    assert (top, left) == (5, 12)
    bottom = g.y1 - (g.pad_top + g.source_h)
    right = g.x1 - (g.pad_left + g.source_w)
    assert (bottom, right) == (27, 20)
    # The source sits unchanged at the offset.
    assert np.array_equal(out[:, top:top + 256, left:left + 256], src)
    # Each gap pixel is the source's reflection (the edge pixel not repeated).
    for k in range(1, left + 1):
        assert np.array_equal(out[:, top:top + 256, left - k], src[:, :, k])
    for k in range(1, right + 1):
        assert np.array_equal(out[:, top:top + 256, left + 255 + k], src[:, :, 255 - k])
    for k in range(1, top + 1):
        assert np.array_equal(out[:, top - k, left:left + 256], src[:, k])
    for k in range(1, bottom + 1):
        assert np.array_equal(out[:, top + 255 + k, left:left + 256], src[:, 255 - k])
    # Corners: reflected along both axes.
    assert np.array_equal(out[:, 0, 0], src[:, top, left])


def test_aligned_frames_no_gap_is_a_copy_and_shape_is_checked():
    g = _geom(256, 256, 32, 32, 0, 0)
    src = _frames(1, 256, 256)
    out = G.aligned_frames(src, g)
    assert np.array_equal(out, src) and out is not src
    with pytest.raises(ValueError):
        G.aligned_frames(_frames(1, 256, 250), g)
    with pytest.raises(ValueError):
        G.aligned_frames(src.astype(np.float32), g)


def test_token_values():
    g = _geom(1280, 720, 320, 320, 184, 184)
    v = G.token_values(g, 3)
    assert v.shape == (3, 34, 60) and v.dtype == np.float32
    assert np.array_equal(v[0] == 0, g.fixed) and np.array_equal(v[1], v[0]) and np.array_equal(v[2], v[0])
    assert set(np.unique(v).tolist()) == {0.0, 1.0}
    with pytest.raises(ValueError):
        G.token_values(g, 0)


# --------------------------------------------------------------------------- #
# feather weights and composite
# --------------------------------------------------------------------------- #

def test_feather_zero_is_all_ones():
    g = _geom(256, 256, 12, 20, 0, 0)
    w = G.feather_weights(g, 0)
    assert w.shape == (256, 256) and w.dtype == np.float32 and bool((w == 1).all())


def test_feather_by_hand_and_pad_zero_side_is_not_counted():
    # Left / right padded, top / bottom not: w depends on the column only.
    g = _geom(256, 256, 12, 20, 0, 0)
    f = 8
    w = G.feather_weights(g, f)
    for x in range(256):
        d = min(x, 255 - x)
        assert np.allclose(w[:, x], min(1.0, d / f))
    assert (w[0] == 1).sum() == 256 - 2 * f  # top row (pad 0 side) is not darkened by the top edge
    assert w[0, 0] == 0 and w[0, 255] == 0 and w[128, 128] == 1


def test_feather_single_side_monotone():
    g = _geom(256, 256, 0, 0, 0, 32)  # only the bottom is padded
    w = G.feather_weights(g, 40)
    col = w[:, 100]
    assert col[-1] == 0 and col[0] == 1
    assert bool((np.diff(col) <= 0).all())  # non-increasing toward the bottom seam
    assert bool((w == w[:, :1]).all())      # left / right sides (pad 0) not counted
    assert np.isclose(col[255 - 20], 0.5)


def test_feather_monotone_toward_each_padded_side():
    g = _geom(256, 256, 32, 32, 32, 32)
    w = G.feather_weights(g, 50)
    mid = 128
    assert bool((np.diff(w[mid, :mid]) >= 0).all()) and bool((np.diff(w[mid, mid:]) <= 0).all())
    assert bool((np.diff(w[:mid, mid]) >= 0).all()) and bool((np.diff(w[mid:, mid]) <= 0).all())
    assert float(w.min()) == 0.0 and float(w.max()) == 1.0


def test_composite_hard_paste_is_bit_identical_and_outside_is_untouched():
    g = _geom(256, 256, 12, 20, 5, 27)
    src = _frames(5, 256, 256, seed=1)
    dec = _frames(5, g.canvas_h, g.canvas_w, seed=2)
    before = dec.copy()
    out = G.composite(dec, src, G.feather_weights(g, 0), g)
    assert out is dec
    pt, pl = g.pad_top, g.pad_left
    assert np.array_equal(out[:, pt:pt + 256, pl:pl + 256], src)
    outside = np.ones(out.shape[1:3], dtype=bool)
    outside[pt:pt + 256, pl:pl + 256] = False
    assert np.array_equal(out[:, outside], before[:, outside])


def test_composite_feather_band_full_and_zero_regions():
    g = _geom(256, 256, 32, 32, 32, 32)
    src = np.full((5, 256, 256, 3), 200, dtype=np.uint8)
    src[0, 0, 0] = 7  # a corner pixel that is weight 0: must not show up
    dec = np.full((5, g.canvas_h, g.canvas_w, 3), 40, dtype=np.uint8)
    before = dec.copy()
    w = G.feather_weights(g, 16)
    out = G.composite(dec, src, w, g)
    region = out[:, 32:288, 32:288]
    full, zero = w == 1, w == 0
    band = ~full & ~zero
    assert np.array_equal(region[:, full], src[:, full])
    assert np.array_equal(region[:, zero], before[:, 32:288, 32:288][:, zero])
    vals = region[:, band]
    assert bool((vals >= 40).all()) and bool((vals <= 200).all())
    # Rounded half up: w * 200 + (1 - w) * 40.
    want = np.floor(w[band] * 200 + (1 - w[band]) * 40 + 0.5).astype(np.uint8)
    assert np.array_equal(vals[0, :, 0], want)
    # Monotone along a row: from the seam inward the value rises toward the source.
    row = region[2, 128, :128, 0].astype(int)
    assert bool((np.diff(row) >= 0).all())
    outside = np.ones(out.shape[1:3], dtype=bool)
    outside[32:288, 32:288] = False
    assert np.array_equal(out[:, outside], before[:, outside])


def test_composite_refuses_mismatches():
    g = _geom(256, 256, 32, 32, 0, 0)
    src = _frames(2, 256, 256)
    dec = _frames(3, g.canvas_h, g.canvas_w)
    w = G.feather_weights(g, 0)
    with pytest.raises(ValueError):
        G.composite(dec, src, w, g)  # frame counts differ
    with pytest.raises(ValueError):
        G.composite(dec[:2], src, w[:10], g)  # weights shape
    with pytest.raises(ValueError):
        G.composite(dec[:2], src, w * 2, g)  # weights out of [0, 1]
