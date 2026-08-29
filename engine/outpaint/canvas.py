"""Outpainting canvas geometry — the padded canvas and its blend mask.

This module is deliberately **importable without torch**: everything except
``build_blend_mask`` is plain integer arithmetic, and ``build_blend_mask``
imports torch inside the function body. That keeps the geometry unit tests
runnable in the app ``.venv`` (which has no torch) while the engine ``.venv``
still gets the tensor helper.

Vocabulary used throughout:

* **canvas**  — the full padded frame that is actually generated.
* **inner**   — the rectangle inside the canvas occupied by the user's source
  video (the region we want to *keep*).
* **pad**     — the four bands around the inner rectangle that the model has to
  invent (the outpainted region).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    import torch

logger = logging.getLogger(__name__)


# Official LTXV outpainting sentinel colour (#66FF00). The reference workflow
# paints the pad bands with this exact RGB triple before encoding, so the model
# recognises "this area is to be invented" from the pixel values alone.
GREEN_RGB: tuple[int, int, int] = (102, 255, 0)

# The VAE / patchifier stride chain requires both canvas sides to be multiples
# of 128 (64px VAE stride x 2 for the two-stage chain's stage-2 upscale).
CANVAS_MULTIPLE = 128

# Minimum side length of the kept (inner) rectangle. See ``OutpaintGeometry``.
MIN_INNER_SIDE = 256


@dataclass(frozen=True)
class OutpaintGeometry:
    """Canvas size plus the four pad bands around the kept source rectangle.

    ``MIN_INNER_SIDE`` (256) is not arbitrary. The Laplacian blend dilates the
    mask at a fixed low resolution: the mask is first resized so its **long side
    is 64px**, then max-pooled with radius ``r``, then resized back. The
    effective transition width at full resolution is therefore

        r * (canvas_long_side / 64)

    which for a 1920-wide canvas and the default ``r = 5`` is roughly 150px of
    full-resolution feathering that eats *inwards*, into the kept rectangle.
    An inner rectangle of, say, 64px would be consumed outright — the "kept"
    source would be entirely replaced by generated pixels. 256px leaves a solid
    untouched core even at the largest canvas and the largest dilation radius.
    """

    canvas_width: int
    canvas_height: int
    pad_left: int
    pad_right: int
    pad_top: int
    pad_bottom: int

    @property
    def inner_x(self) -> int:
        """Left edge of the kept rectangle, in canvas pixels."""
        return self.pad_left

    @property
    def inner_y(self) -> int:
        """Top edge of the kept rectangle, in canvas pixels."""
        return self.pad_top

    @property
    def inner_width(self) -> int:
        return self.canvas_width - self.pad_left - self.pad_right

    @property
    def inner_height(self) -> int:
        return self.canvas_height - self.pad_top - self.pad_bottom

    def validate(self) -> None:
        """Raise ``ValueError`` on the first violated invariant.

        Each rule is an independent ``if`` with its own message rather than one
        compound condition, so the caller (and the API error surfaced to the
        UI) always names exactly which rule failed.
        """
        if self.pad_left < 0:
            raise ValueError(f"pad_left must be >= 0, got {self.pad_left}")
        if self.pad_right < 0:
            raise ValueError(f"pad_right must be >= 0, got {self.pad_right}")
        if self.pad_top < 0:
            raise ValueError(f"pad_top must be >= 0, got {self.pad_top}")
        if self.pad_bottom < 0:
            raise ValueError(f"pad_bottom must be >= 0, got {self.pad_bottom}")

        if self.canvas_width % CANVAS_MULTIPLE != 0:
            raise ValueError(
                f"canvas_width must be a multiple of {CANVAS_MULTIPLE}, "
                f"got {self.canvas_width}"
            )
        if self.canvas_height % CANVAS_MULTIPLE != 0:
            raise ValueError(
                f"canvas_height must be a multiple of {CANVAS_MULTIPLE}, "
                f"got {self.canvas_height}"
            )

        total_pad = self.pad_left + self.pad_right + self.pad_top + self.pad_bottom
        if total_pad <= 0:
            raise ValueError(
                "at least one pad band must be greater than 0; "
                "an outpaint job with no padding has nothing to generate"
            )

        if self.inner_width < MIN_INNER_SIDE:
            raise ValueError(
                f"inner_width must be >= {MIN_INNER_SIDE}, got {self.inner_width} "
                "(the blend's low-resolution mask dilation would consume it)"
            )
        if self.inner_height < MIN_INNER_SIDE:
            raise ValueError(
                f"inner_height must be >= {MIN_INNER_SIDE}, got {self.inner_height} "
                "(the blend's low-resolution mask dilation would consume it)"
            )

    def as_dict(self) -> dict[str, int]:
        """Flat dict for ``metadata.json`` — the six stored values plus inner_*."""
        return {
            "canvas_width": self.canvas_width,
            "canvas_height": self.canvas_height,
            "pad_left": self.pad_left,
            "pad_right": self.pad_right,
            "pad_top": self.pad_top,
            "pad_bottom": self.pad_bottom,
            "inner_x": self.inner_x,
            "inner_y": self.inner_y,
            "inner_width": self.inner_width,
            "inner_height": self.inner_height,
        }


def build_blend_mask(
    geom: OutpaintGeometry,
    *,
    height: int,
    width: int,
    dtype: Any = None,
    device: Any = None,
) -> "torch.Tensor":
    """Build the ``(1, 1, height, width)`` blend mask for ``geom``.

    ``1.0`` marks the pad bands (pixels the model generates and that the blend
    should take from the *generated* video); ``0.0`` marks the kept rectangle.

    Only **one** frame is produced. The outpaint mask is a static rectangular
    frame, identical for every frame of the clip, so materialising it per-frame
    would cost 2.0GB of float32 at 1920x1088x241 for no information gain. The
    blender broadcasts this single plane across the batch.

    ``height`` / ``width`` may be lower than the canvas: the pipeline needs the
    same mask at half resolution for the stage-1 pass. The inner rectangle is
    therefore obtained by **proportionally projecting** the canvas-space
    rectangle onto the requested resolution rather than by resampling a
    full-resolution mask. The official workflow downsamples a full-resolution
    mask with ``area`` interpolation; the half-pixel differences that produces
    are erased downstream anyway, because the blender resizes the mask to a
    64px long side before dilating it and then back up again. Generating the
    rectangle analytically is exact and costs nothing.

    ``dtype`` defaults to ``torch.float32``; it is not spelled as a default
    argument value because evaluating ``torch.float32`` at import time would
    make this module unimportable without torch (see the module docstring).
    """
    import torch

    if dtype is None:
        dtype = torch.float32
    if height <= 0 or width <= 0:
        raise ValueError(f"mask size must be positive, got height={height} width={width}")

    x0 = round(geom.inner_x * width / geom.canvas_width)
    x1 = round((geom.inner_x + geom.inner_width) * width / geom.canvas_width)
    y0 = round(geom.inner_y * height / geom.canvas_height)
    y1 = round((geom.inner_y + geom.inner_height) * height / geom.canvas_height)

    mask = torch.ones((1, 1, height, width), dtype=dtype, device=device)
    mask[..., y0:y1, x0:x1] = 0.0
    return mask


def fill_pad_with_generated_(
    canvas: "torch.Tensor",
    *,
    generated: "torch.Tensor",
    mask: "torch.Tensor",
) -> "torch.Tensor":
    """Replace the canvas' pad bands with the same frame's generated pixels, in place.

    ``canvas`` and ``generated`` are both ``(F, H, W, 3)`` uint8, and ``mask`` is
    the ``(1, 1, H, W)`` plane :func:`build_blend_mask` produced for that same
    resolution. The kept rectangle of the canvas is left exactly as it was; only
    the four pad bands around it are overwritten. The canvas tensor handed in is
    the one returned — nothing is copied.

    Why this exists, stated correctly. The Laplacian blend computes, at every
    pyramid level, ``L_generated * m + L_canvas * (1 - m)``. At the coarse levels
    the mask is no longer a rectangle but a smooth blob: measured at level 6 it
    only reaches 0.76 deep inside the pad band, never 1.0. So the canvas' DC
    component **always** mixes into the generated area, and no change to this
    module can stop it. What this function changes is not *whether* the canvas
    mixes in, but *what* mixes in: with the pad bands carrying generated pixels
    instead of the #66FF00 sentinel, the thing that bleeds inwards is the picture
    the model just drew rather than a flat green.

    The blend still does its real job. The seam work happens in the dilated band
    just **inside** the kept rectangle, and there the two operands are still
    different pictures (source footage on one side, generated frame on the other)
    — that band is untouched here. What does become an identity is the deep pad,
    where both operands are now the same pixels and mixing them is a no-op. That
    is harmless: there was never anything to reconcile out there.

    This is not the earlier, rejected idea of taking the green out of the
    *conditioning*. The model is still shown the green canvas and still reads
    "invent this area" from it; only the operand handed to the pixel-space blend,
    minutes later in the pipeline, has its pad bands swapped.

    The pad is written as four slice assignments rather than a boolean index.
    ``canvas[:, pad_mask, :] = ...`` builds a fresh tensor of every selected
    pixel first — 847 MB of resident memory measured on a 1920x1152 canvas at 241
    frames — whereas assigning the top, bottom, left and right bands one slice at
    a time copies straight into the canvas' own storage and adds nothing. The
    four bands are still derived **from the mask**, so a caller that changes how
    the mask is built cannot leave this function writing the wrong pixels.

    Deriving the rectangle from the mask relies on the mask being an exact
    rectangle of 0.0 inside and 1.0 outside, which is what ``build_blend_mask``
    constructs (one ``mask[..., y0:y1, x0:x1] = 0.0`` on a tensor of ones). It is
    read once here, not per frame, because the mask carries a single plane that
    the blender broadcasts across the batch.

    ``generated`` and ``mask`` are keyword-only on purpose. The neighbouring
    :func:`engine.outpaint.pyramid_blend.blend_video_u8` takes its operands in the
    opposite order (``generated`` first, canvas second), and a call site that
    swapped them here would silently overwrite the generated video with green
    instead. Keyword arguments make that mistake impossible to write.

    ``torch`` is imported inside the body, like :func:`build_blend_mask`, so this
    module stays importable in the app ``.venv`` where torch is absent (see the
    module docstring).
    """
    import torch

    if tuple(mask.shape[-2:]) != tuple(canvas.shape[1:3]):
        raise ValueError(
            "the blend mask and the canvas must have the same spatial resolution, "
            f"mask is {tuple(mask.shape[-2:])} and canvas is {tuple(canvas.shape[1:3])}"
        )

    height = int(canvas.shape[1])
    width = int(canvas.shape[2])

    # The kept rectangle's four edges, read once off the mask plane. This depends
    # on the mask being a rectangle (build_blend_mask guarantees it); a feathered
    # or non-rectangular mask would make these bounds meaningless.
    keep = mask[0, 0] <= 0.5
    rows = keep.any(dim=1).nonzero()
    cols = keep.any(dim=0).nonzero()
    if rows.numel() == 0 or cols.numel() == 0:
        # No kept rectangle at all: every pixel is pad. The bands below then
        # degenerate to "bottom = the whole frame", which is exactly right.
        y0 = y1 = x0 = x1 = 0
    else:
        y0, y1 = int(rows[0]), int(rows[-1]) + 1
        x0, x1 = int(cols[0]), int(cols[-1]) + 1

    bands = (
        (slice(0, y0), slice(0, width)),       # top band, full width
        (slice(y1, height), slice(0, width)),  # bottom band, full width
        (slice(y0, y1), slice(0, x0)),         # left band, between the two above
        (slice(y0, y1), slice(x1, width)),     # right band, between the two above
    )

    def _frame0_pad_mean(video: "torch.Tensor") -> list[float]:
        """Mean RGB of frame 0's pad bands. Frame 0 only: the whole-timeline mean
        would materialise hundreds of MB of float for one log line."""
        total = torch.zeros(3, dtype=torch.float64)
        count = 0
        for band_rows, band_cols in bands:
            band = video[0, band_rows, band_cols, :]
            if band.numel() == 0:
                continue
            total += band.reshape(-1, 3).to(torch.float64).sum(dim=0)
            count += int(band.shape[0]) * int(band.shape[1])
        if count == 0:
            return [0.0, 0.0, 0.0]
        return [round(float(v), 1) for v in total / count]

    before = _frame0_pad_mean(canvas)
    for band_rows, band_cols in bands:
        canvas[:, band_rows, band_cols, :] = generated[:, band_rows, band_cols, :]
    after = _frame0_pad_mean(canvas)

    logger.info(
        "outpaint de-green: canvas %s, pad %d px/frame, frame-0 pad mean RGB %s -> %s",
        tuple(int(v) for v in canvas.shape),
        height * width - (y1 - y0) * (x1 - x0),
        before,
        after,
    )
    return canvas
