"""engine_h3.inpaint_geometry: the Inpainting geometry, token mask and paste-back (台帳 §1-124).

Runs in ``.venv-engine-h3`` through the engine-venv runner::

    .venv-engine-h3/Scripts/python.exe scripts/run_h3_pytest.py tests/test_h3_inpaint_geometry.py

The expected values are computed here independently (brute force on small arrays).
"""

from __future__ import annotations

import numpy as np
import pytest

from engine_h3 import inpaint_geometry as I


def _frames(n, h, w, seed):
    return np.random.default_rng(seed).integers(0, 256, size=(n, h, w, 3), dtype=np.uint8)


def _brute_weights(block_mask, geom, f):
    """Min Euclidean distance from every source pixel to every masked canvas pixel, then min(1, d/F)."""
    pix = np.repeat(np.repeat(block_mask, 32, axis=0), 32, axis=1)
    my, mx = np.nonzero(pix)
    ys, xs = np.mgrid[0:geom.source_h, 0:geom.source_w]
    d = np.sqrt((ys[..., None] - my) ** 2 + (xs[..., None] - mx) ** 2).min(axis=-1)
    if f == 0:
        return np.where(d == 0, 0.0, 1.0)
    return np.minimum(1.0, d / f)


def test_round_up_32():
    assert I.round_up_32(768) == 768
    assert I.round_up_32(1080) == 1088
    assert I.round_up_32(1) == 32
    with pytest.raises(ValueError):
        I.round_up_32(0)


def test_geometry():
    with pytest.raises(ValueError):
        I.compute_inpaint_geometry(1280, 768, 1280, 800)
    g = I.compute_inpaint_geometry(1280, 768, 1280, 768)
    assert (g.pad_left, g.pad_right, g.pad_top, g.pad_bottom) == (0, 0, 0, 0)
    assert g.fixed.all()
    assert (g.aligned_w, g.aligned_h) == (1280, 768)
    g = I.compute_inpaint_geometry(1920, 1080, 1920, 1088)
    assert (g.pad_left, g.pad_right, g.pad_top, g.pad_bottom) == (0, 0, 0, 8)
    assert g.fixed.shape == (34, 60)
    assert g.fixed[:33].all() and not g.fixed[33].any()
    assert (g.aligned_w, g.aligned_h) == (1920, 1088)


def test_chunk_ranges():
    assert I.chunk_ranges(73) == [(0, 17), (17, 34), (34, 51), (51, 68), (68, 73)]
    assert I.chunk_ranges(22) == [(0, 17), (17, 22)]


def test_block_mask_and_chunks_and_tokens():
    g = I.compute_inpaint_geometry(1280, 768, 1280, 768)
    blocks = np.zeros((73, g.grid_h, g.grid_w), dtype=bool)
    frame = np.zeros((768, 1280), dtype=bool)
    frame[100, 70] = True  # one pixel -> block (3, 2)
    blocks[20] = I.block_mask_from_pixels(frame, g)
    assert blocks[20].sum() == 1 and blocks[20][3, 2]
    chunks = I.chunk_block_masks(blocks, 73)
    assert chunks.shape == (5, 24, 40)
    assert chunks[1, 3, 2] and chunks.sum() == 1
    tv = I.token_values_from_chunks(chunks, 22)
    assert tv.shape == (22, 24, 40) and tv.dtype == np.float32
    assert set(np.unique(tv)) <= {0.0, 1.0}
    assert tv[5:10, 3, 2].all() and tv.sum() == 5
    with pytest.raises(ValueError):
        I.token_values_from_chunks(chunks, 23)


def test_block_mask_beyond_source_is_false():
    g = I.compute_inpaint_geometry(70, 40, 96, 64)
    frame = np.zeros((40, 70), dtype=bool)
    frame[39, 69] = True
    b = I.block_mask_from_pixels(frame, g)
    assert b.shape == (2, 3) and b.sum() == 1 and b[1, 2]


@pytest.mark.parametrize("f", [0, 1, 5, 20, 45])
def test_feather_matches_brute_force(f):
    g = I.compute_inpaint_geometry(150, 100, 160, 128)
    m = np.zeros((g.grid_h, g.grid_w), dtype=bool)
    m[1, 2] = True
    got = I.feather_weights_from_mask(m, g, f)
    want = _brute_weights(m, g, f)
    assert got.shape == (100, 150) and got.dtype == np.float32
    assert np.allclose(got, want, atol=1e-6)
    assert (got[32:64, 64:96] == 0).all()


def test_feather_two_blocks_and_far_pixels():
    g = I.compute_inpaint_geometry(150, 100, 160, 128)
    m = np.zeros((g.grid_h, g.grid_w), dtype=bool)
    m[0, 0] = m[3, 4] = True
    got = I.feather_weights_from_mask(m, g, 12)
    assert np.allclose(got, _brute_weights(m, g, 12), atol=1e-6)
    assert got[31, 32] == pytest.approx(1 / 12)  # touches block (0, 0): d = 1
    assert got[60, 60] == 1.0  # far from both
    with pytest.raises(ValueError):
        I.feather_weights_from_mask(m, g, -1)


def test_compose_hard_paste_bit_identical():
    g = I.compute_inpaint_geometry(70, 40, 96, 64)
    n = 22
    src = _frames(n, 40, 70, 1)
    dec = _frames(n, 64, 96, 2)
    before = dec.copy()
    chunks = np.zeros((2, g.grid_h, g.grid_w), dtype=bool)
    chunks[0, 0, 0] = True
    chunks[1, 1, 2] = True
    out = I.compose_inpaint(dec, src, chunks, g, 0)
    assert out is dec
    for k, (a, b) in enumerate(I.chunk_ranges(n)):
        inside = np.repeat(np.repeat(chunks[k], 32, 0), 32, 1)
        src_in = inside[:40, :70]
        region = out[a:b, :40, :70]
        assert np.array_equal(region[:, src_in], before[a:b, :40, :70][:, src_in])
        assert np.array_equal(region[:, ~src_in], src[a:b][:, ~src_in])
    # beyond the source (the canvas pad) stays decoded
    assert np.array_equal(out[:, 40:], before[:, 40:])
    assert np.array_equal(out[:, :, 70:], before[:, :, 70:])


def test_compose_feather_mixes_only_the_band():
    g = I.compute_inpaint_geometry(96, 64, 96, 64)
    n = 18
    src = np.full((n, 64, 96, 3), 200, dtype=np.uint8)
    dec = np.full((n, 64, 96, 3), 10, dtype=np.uint8)
    chunks = np.zeros((2, g.grid_h, g.grid_w), dtype=bool)
    chunks[:, 0, 1] = True
    f = 8
    I.compose_inpaint(dec, src, chunks, g, f)
    w = I.feather_weights_from_mask(chunks[0], g, f)
    band = (w > 0) & (w < 1)
    assert band.any()
    assert (dec[:, w == 0] == 10).all()
    assert (dec[:, w == 1] == 200).all()
    mixed = dec[:, band]
    assert ((mixed > 10) & (mixed < 200)).any()
    assert ((mixed >= 10) & (mixed <= 200)).all()


def test_compose_refuses_mismatches():
    g = I.compute_inpaint_geometry(96, 64, 96, 64)
    src = _frames(22, 64, 96, 3)
    dec = _frames(21, 64, 96, 4)
    chunks = np.zeros((2, g.grid_h, g.grid_w), dtype=bool)
    with pytest.raises(ValueError):
        I.compose_inpaint(dec, src, chunks, g, 0)  # frame counts differ
    with pytest.raises(ValueError):
        I.compose_inpaint(_frames(22, 64, 96, 4), src, chunks[:1], g, 0)  # chunk count
