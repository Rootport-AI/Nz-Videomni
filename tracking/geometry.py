"""Frame and box arithmetic for object tracking.

PURE apart from numpy: no torch, no cv2, no project imports. It is deliberately
importable from the APP venv as well as ``.venv-utils``, so every rule about
what a frame buffer is and where a box may sit is unit-tested on the ordinary
``pytest`` run rather than only on the interpreter the tracker happens to live
on.

Two conventions, fixed everywhere in this feature:

* **xywh** ``(x, y, w, h)`` -- top-left corner plus size. This is what UETrack
  takes and returns, what the HTTP contract carries, and therefore what the
  native plugin converts its partial-filter values to and from.
* **xyxy** ``(x0, y0, x1, y1)`` -- the two corners, ``x1``/``y1`` EXCLUSIVE.
  Used only as an intermediate here, because clipping is trivial on corners and
  fiddly on a corner-plus-size.

Coordinates are floats: the tracker's output is sub-pixel, and rounding it here
would throw away the only information the plugin's smoothing step has to work
with.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = [
    "BYTES_PER_PIXEL",
    "rgba_bytes_to_rgb",
    "clamp_box_to_frame",
    "xywh_to_xyxy",
    "xyxy_to_xywh",
]

#: The wire format is raw RGBA -- red, green, blue, alpha, one byte each. Fixed
#: by what AviUtl2's scene renderer hands the plugin; see the API contract.
BYTES_PER_PIXEL = 4


def rgba_bytes_to_rgb(buf: bytes, width: int, height: int) -> np.ndarray:
    """Raw RGBA bytes -> an ``(height, width, 3)`` uint8 RGB array.

    EXACTLY ONE COPY, and it is not optional. ``np.frombuffer`` over the request
    body is free (a view), and dropping alpha with ``[..., :3]`` is free too --
    but it leaves a strided, read-only array, and the preprocessing step that
    consumes this calls into OpenCV, which needs contiguous memory. Asking for
    contiguity here makes that copy once, visibly, instead of letting cv2 make
    it invisibly (or refuse the array) further down.

    The alpha channel is dropped rather than composited: the tracker was trained
    on RGB, and the frames arrive already composited by AviUtl2's renderer, so
    alpha carries nothing the model can use.

    A length that does not match ``width * height * 4`` raises ``ValueError``.
    The API layer checks the same thing first, with its own error code -- this
    check is the backstop for every other caller (the worker, tests, a future
    tool), not a duplicate of that one.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"frame size must be positive, got {width}x{height}")
    expected = width * height * BYTES_PER_PIXEL
    if len(buf) != expected:
        raise ValueError(
            f"RGBA buffer is {len(buf)} bytes, expected {expected} "
            f"({width}x{height}x{BYTES_PER_PIXEL})"
        )
    flat = np.frombuffer(buf, dtype=np.uint8)
    return np.ascontiguousarray(flat.reshape(height, width, BYTES_PER_PIXEL)[:, :, :3])


def xywh_to_xyxy(box: Sequence[float]) -> tuple[float, float, float, float]:
    """``(x, y, w, h)`` -> ``(x0, y0, x1, y1)``."""
    x, y, w, h = (float(v) for v in box)
    return x, y, x + w, y + h


def xyxy_to_xywh(box: Sequence[float]) -> tuple[float, float, float, float]:
    """``(x0, y0, x1, y1)`` -> ``(x, y, w, h)``."""
    x0, y0, x1, y1 = (float(v) for v in box)
    return x0, y0, x1 - x0, y1 - y0


def clamp_box_to_frame(
    box: Sequence[float], width: int, height: int
) -> tuple[float, float, float, float]:
    """Clip ``box`` into ``width`` x ``height``. Never raises on the box itself.

    THE RULE IS CLIPPING, NOT SLIDING: an edge that falls outside the frame is
    pulled onto the frame border and the opposite edge stays put, so a box that
    is half off-screen comes back as its visible half rather than as a
    same-sized box moved somewhere the user never put it.

    The one exception is degeneracy. Clipping can leave zero (or negative) width
    -- a box entirely off the left edge, say -- and every consumer downstream,
    UETrack's own ``initialize`` included, requires a positive size. Rather than
    raise from a helper that both the seeding path and the per-frame path call,
    such a box collapses to the nearest 1-pixel box inside the frame. A caller
    that cares about the difference compares its input to the result.

    ``width``/``height`` must themselves be positive; that is a programming
    error, not a geometry case, and does raise.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"frame size must be positive, got {width}x{height}")
    fw = float(width)
    fh = float(height)
    x0, y0, x1, y1 = xywh_to_xyxy(box)

    x0 = min(max(x0, 0.0), fw)
    x1 = min(max(x1, 0.0), fw)
    if x1 - x0 < 1.0:
        x0 = min(x0, max(fw - 1.0, 0.0))
        x1 = min(x0 + 1.0, fw)

    y0 = min(max(y0, 0.0), fh)
    y1 = min(max(y1, 0.0), fh)
    if y1 - y0 < 1.0:
        y0 = min(y0, max(fh - 1.0, 0.0))
        y1 = min(y0 + 1.0, fh)

    return xyxy_to_xywh((x0, y0, x1, y1))
