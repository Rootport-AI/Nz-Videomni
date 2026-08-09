"""Outpaint canvas geometry — pure-arithmetic unit tests (no torch).

``engine.outpaint.canvas`` keeps its ``import torch`` inside ``build_blend_mask``
precisely so this file runs in the app ``.venv``. Import the submodule directly
(``engine.outpaint.canvas``) rather than the package attribute, and note that
``engine/outpaint/__init__.py`` re-exports the torch-backed names lazily so that
importing the submodule does not drag torch in through the package ``__init__``.

  .venv\\Scripts\\python.exe -m pytest tests/test_outpaint_canvas.py -q
"""

from __future__ import annotations

import pytest

from engine.outpaint.canvas import (
    CANVAS_MULTIPLE,
    GREEN_RGB,
    MIN_INNER_SIDE,
    OutpaintGeometry,
)


def _valid(**overrides) -> OutpaintGeometry:
    """A geometry that passes ``validate()``; override one field to break it."""
    base = {
        "canvas_width": 1280,
        "canvas_height": 768,
        "pad_left": 128,
        "pad_right": 256,
        "pad_top": 64,
        "pad_bottom": 192,
    }
    base.update(overrides)
    return OutpaintGeometry(**base)


# ── constants ───────────────────────────────────────────────────────────────
def test_green_sentinel_is_the_official_66ff00():
    assert GREEN_RGB == (102, 255, 0)


def test_canvas_multiple_and_min_inner_side():
    assert CANVAS_MULTIPLE == 128
    assert MIN_INNER_SIDE == 256


# ── inner_* geometry ────────────────────────────────────────────────────────
def test_inner_rect_with_asymmetric_pads():
    """Asymmetric on BOTH axes, and left != top, so a swapped x/y is caught."""
    geom = _valid()
    assert geom.inner_x == 128  # pad_left, not pad_top
    assert geom.inner_y == 64  # pad_top, not pad_left
    assert geom.inner_width == 1280 - 128 - 256  # 896
    assert geom.inner_height == 768 - 64 - 192  # 512


def test_inner_rect_does_not_mix_up_the_axes():
    """Transposing the pads must transpose the inner rect, nothing else."""
    geom = OutpaintGeometry(
        canvas_width=768,
        canvas_height=1280,
        pad_left=64,
        pad_right=192,
        pad_top=128,
        pad_bottom=256,
    )
    assert (geom.inner_x, geom.inner_y) == (64, 128)
    assert (geom.inner_width, geom.inner_height) == (512, 896)


def test_zero_pad_on_one_side_only():
    geom = _valid(pad_left=0, pad_right=384)
    assert geom.inner_x == 0
    assert geom.inner_width == 1280 - 384


def test_geometry_is_frozen():
    geom = _valid()
    with pytest.raises(Exception):
        geom.pad_left = 0  # type: ignore[misc]


# ── validate(): one rule at a time ──────────────────────────────────────────
def test_validate_accepts_a_sane_geometry():
    _valid().validate()  # must not raise


@pytest.mark.parametrize(
    "field", ["pad_left", "pad_right", "pad_top", "pad_bottom"]
)
def test_validate_rejects_negative_pad(field):
    with pytest.raises(ValueError, match=f"{field} must be >= 0"):
        _valid(**{field: -1}).validate()


def test_validate_rejects_non_multiple_canvas_width():
    with pytest.raises(ValueError, match="canvas_width must be a multiple of 128"):
        _valid(canvas_width=1280 + 1).validate()


def test_validate_rejects_non_multiple_canvas_height():
    with pytest.raises(ValueError, match="canvas_height must be a multiple of 128"):
        _valid(canvas_height=768 + 64).validate()


def test_validate_rejects_zero_total_pad():
    """A 128-multiple canvas with no padding is a well-formed rectangle but has
    nothing to outpaint, so it must be rejected on its own dedicated rule."""
    geom = OutpaintGeometry(
        canvas_width=1280,
        canvas_height=768,
        pad_left=0,
        pad_right=0,
        pad_top=0,
        pad_bottom=0,
    )
    with pytest.raises(ValueError, match="at least one pad band"):
        geom.validate()


def test_validate_rejects_narrow_inner_width():
    # 1280 - 640 - 512 = 128 < 256; the height stays comfortably valid.
    geom = _valid(pad_left=640, pad_right=512)
    assert geom.inner_width == 128
    with pytest.raises(ValueError, match="inner_width must be >= 256"):
        geom.validate()


def test_validate_rejects_short_inner_height():
    # 768 - 384 - 256 = 128 < 256; the width stays comfortably valid.
    geom = _valid(pad_top=384, pad_bottom=256)
    assert geom.inner_height == 128
    with pytest.raises(ValueError, match="inner_height must be >= 256"):
        geom.validate()


def test_validate_accepts_exactly_the_minimum_inner_side():
    geom = _valid(pad_left=512, pad_right=512, pad_top=256, pad_bottom=256)
    assert geom.inner_width == MIN_INNER_SIDE
    assert geom.inner_height == MIN_INNER_SIDE
    geom.validate()


# ── as_dict() ───────────────────────────────────────────────────────────────
def test_as_dict_keys_and_values():
    geom = _valid()
    d = geom.as_dict()
    assert set(d) == {
        "canvas_width",
        "canvas_height",
        "pad_left",
        "pad_right",
        "pad_top",
        "pad_bottom",
        "inner_x",
        "inner_y",
        "inner_width",
        "inner_height",
    }
    assert d["canvas_width"] == 1280
    assert d["pad_left"] == 128
    assert d["inner_x"] == 128
    assert d["inner_y"] == 64
    assert d["inner_width"] == 896
    assert d["inner_height"] == 512
    assert all(isinstance(v, int) for v in d.values())
