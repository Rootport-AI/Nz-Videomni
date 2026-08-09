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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    import torch


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
