"""Frame and box arithmetic (``tracking/geometry.py``).

Runs on the APP venv even though the code it covers is used by the worker: the
module is torch-free precisely so these rules are checked on every ordinary
``pytest`` run, not only on a machine that has installed the tracker.
"""

from __future__ import annotations

import numpy as np
import pytest

from tracking.geometry import (
    clamp_box_to_frame,
    rgba_bytes_to_rgb,
    xywh_to_xyxy,
    xyxy_to_xywh,
)


# --------------------------------------------------------------------------- #
# RGBA -> RGB
# --------------------------------------------------------------------------- #


def _rgba(width: int, height: int) -> bytes:
    """A deterministic RGBA buffer whose every byte is its own index mod 256."""
    return bytes((i % 256) for i in range(width * height * 4))


def test_drops_alpha_and_keeps_row_major_order():
    raw = bytes(
        [
            1, 2, 3, 255, 4, 5, 6, 255,      # row 0
            7, 8, 9, 255, 10, 11, 12, 255,   # row 1
        ]
    )
    rgb = rgba_bytes_to_rgb(raw, 2, 2)
    assert rgb.shape == (2, 2, 3)
    assert rgb.dtype == np.uint8
    assert rgb.tolist() == [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]


def test_result_is_contiguous_and_writable():
    """Not cosmetic: OpenCV's crop step needs contiguous memory, and the
    ``np.frombuffer`` view over an immutable ``bytes`` body is read-only."""
    rgb = rgba_bytes_to_rgb(_rgba(8, 4), 8, 4)
    assert rgb.flags["C_CONTIGUOUS"]
    assert rgb.flags["WRITEABLE"]


def test_does_not_alias_the_input_buffer():
    """The copy is what lets the request body be released while tracking runs."""
    raw = bytearray(_rgba(4, 4))
    rgb = rgba_bytes_to_rgb(bytes(raw), 4, 4)
    before = rgb[0, 0].tolist()
    raw[0] = (raw[0] + 7) % 256
    assert rgb[0, 0].tolist() == before


@pytest.mark.parametrize("delta", [-1, 1, -192])
def test_wrong_length_is_refused(delta: int):
    """One byte short, one byte long, and a whole row missing."""
    raw = _rgba(8, 6)
    mangled = raw[:delta] if delta < 0 else raw + b"\x00" * delta
    with pytest.raises(ValueError) as exc:
        rgba_bytes_to_rgb(mangled, 8, 6)
    assert "expected 192" in str(exc.value)


def test_zero_size_frame_is_refused():
    with pytest.raises(ValueError):
        rgba_bytes_to_rgb(b"", 0, 0)


# --------------------------------------------------------------------------- #
# xywh <-> xyxy
# --------------------------------------------------------------------------- #


def test_xywh_xyxy_round_trip():
    box = (12.5, 30.25, 100.0, 40.5)
    assert xyxy_to_xywh(xywh_to_xyxy(box)) == box


def test_xyxy_is_corner_plus_corner():
    assert xywh_to_xyxy((10, 20, 5, 7)) == (10.0, 20.0, 15.0, 27.0)


# --------------------------------------------------------------------------- #
# clamping
# --------------------------------------------------------------------------- #


def test_a_box_already_inside_is_returned_unchanged():
    box = (10.5, 20.25, 30.0, 40.0)
    assert clamp_box_to_frame(box, 1920, 1080) == box


def test_a_box_touching_the_far_corner_is_unchanged():
    assert clamp_box_to_frame((1820.0, 980.0, 100.0, 100.0), 1920, 1080) == (
        1820.0,
        980.0,
        100.0,
        100.0,
    )


def test_overhang_is_clipped_not_slid():
    """The visible half comes back. THE BOX IS NOT MOVED to keep its size --
    a size-preserving slide would report the object somewhere it is not."""
    assert clamp_box_to_frame((-20.0, -10.0, 50.0, 40.0), 100, 100) == (0.0, 0.0, 30.0, 30.0)
    assert clamp_box_to_frame((80.0, 70.0, 50.0, 60.0), 100, 100) == (80.0, 70.0, 20.0, 30.0)


def test_a_box_entirely_outside_collapses_to_one_pixel_inside():
    """Degenerate, but total: every consumer downstream needs a positive size,
    and a helper both the seeding and the per-frame path call must not raise."""
    assert clamp_box_to_frame((-500.0, 10.0, 100.0, 20.0), 100, 100) == (0.0, 10.0, 1.0, 20.0)
    assert clamp_box_to_frame((500.0, 10.0, 100.0, 20.0), 100, 100) == (99.0, 10.0, 1.0, 20.0)


def test_a_box_larger_than_the_frame_becomes_the_frame():
    assert clamp_box_to_frame((-10.0, -10.0, 400.0, 400.0), 64, 48) == (0.0, 0.0, 64.0, 48.0)


def test_zero_size_box_becomes_one_pixel():
    assert clamp_box_to_frame((10.0, 10.0, 0.0, 0.0), 100, 100) == (10.0, 10.0, 1.0, 1.0)


def test_frame_size_must_be_positive():
    with pytest.raises(ValueError):
        clamp_box_to_frame((0, 0, 1, 1), 0, 10)
