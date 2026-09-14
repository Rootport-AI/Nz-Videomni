"""Inpainting canvas geometry — an arbitrary mask inside a 128-multiple canvas.

The sibling of :mod:`engine.outpaint.canvas`, and deliberately shaped like it:
this module is **importable without torch** (every tensor helper imports torch
inside its own body), so the geometry unit tests run in the app ``.venv`` as
well as in both engine venvs.

Vocabulary used throughout:

* **canvas** — the full frame that is actually generated. Both sides are
  multiples of ``CANVAS_MULTIPLE`` (128) because the VAE / patchifier stride
  chain demands it for the two-stage path.
* **source** — the user's video, always placed at the canvas' TOP-LEFT corner
  (0, 0). Its own resolution is what the delivered mp4 comes out at.
* **pad** — the right and bottom bands between the source and the canvas edge.
  There are only two of them (never a left or a top band) precisely because the
  source is anchored at the origin: a 1920x1080 source on a 1920x1088 canvas
  has an 8px bottom band and nothing else. The bands are sentinel green in the
  canvas file and are cut off again, losslessly, at the very end of the job.
* **mask** — the white region the model is asked to repaint. Unlike outpainting's
  static rectangle it is a DIFFERENT PICTURE PER FRAME (the plugin bakes the
  AviUtl2 partial filter's moving frame into a video), which is why every helper
  here takes ``(F, 1, H, W)`` rather than the single plane outpainting uses.

The two conventions a caller has to know:

* a mask carried as **uint8** is ``0`` / ``255`` (255 = repaint), which is what
  :func:`engine.pipeline.common.decode_mask_video` produces and what the blend
  accepts directly;
* a mask carried as **float** is ``0.0`` / ``1.0``.

Every predicate below is written as ``mask > 0.5``, which is true for 255 and
for 1.0 and false for 0 and 0.0 — so both conventions travel through the same
code without a dtype switch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from engine.outpaint.canvas import CANVAS_MULTIPLE, GREEN_RGB

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    import torch

logger = logging.getLogger(__name__)

__all__ = [
    "CANVAS_MULTIPLE",
    "GREEN_RGB",
    "INPAINT_MIN_SOURCE_SIDE",
    "InpaintGeometry",
    "fill_mask_with_generated_",
    "fill_pad_bands_with_generated_",
    "half_res_mask",
    "place_mask_on_canvas",
    "restore_outside_mask_",
    "round_up_128",
]


# Minimum side length of the SOURCE video. Numerically the same 256 as
# ``engine.outpaint.canvas.MIN_INNER_SIDE`` and ``api.models.OUTPAINT_MIN_KEEP_SIDE``,
# but a separate name with a separate reason, because the two rules protect
# different things and could legitimately drift apart:
#
#   outpainting's floor protects the KEPT rectangle from a mask dilation that
#   eats inwards from the canvas edge, whereas this floor protects the CANVAS
#   ITSELF. A source below 256px on a side would round up to a 128 or 256px
#   canvas, and the two-stage path halves the canvas for stage 1 before handing
#   it to a VAE with a 32px patch stride — at 128px the stage-1 frame is 64px
#   and there is no room left for the model to see any context around the mask.
#
# 256 is also the point below which the stage-2 blend's own dilation band
# (``r * canvas_long_side / 64`` plus a measured constant ~18px, see
# Docs/VERIFICATION_LOG.md) would cover the whole frame at the default r=2.
INPAINT_MIN_SOURCE_SIDE = 256


def round_up_128(value: int) -> int:
    """Smallest multiple of ``CANVAS_MULTIPLE`` (128) that is >= ``value``.

    This is the ONE place the canvas size is derived from a source size. The API
    layer uses it to check the request's ``width``/``height``, the app layer
    uses it to build the geometry, and the engine re-derives the same number on
    the other side of the worker pipe — three readers, one rule.

    >>> round_up_128(1280), round_up_128(1920), round_up_128(1080)
    (1280, 1920, 1152)
    """
    if value <= 0:
        raise ValueError(f"round_up_128 needs a positive size, got {value}")
    return -(-int(value) // CANVAS_MULTIPLE) * CANVAS_MULTIPLE


@dataclass(frozen=True)
class InpaintGeometry:
    """Canvas size plus the source rectangle anchored at its top-left corner.

    Only four numbers are stored. The two pad bands are DERIVED rather than
    carried, which is the whole point: the source resolution is the single
    source of truth (the file on disk can be ffprobed, a pad field cannot be
    checked against anything), so a request that disagrees with the file is a
    422 instead of a silently wrong canvas.
    """

    canvas_width: int
    canvas_height: int
    source_width: int
    source_height: int

    @property
    def pad_right(self) -> int:
        """Width of the green band to the right of the source."""
        return self.canvas_width - self.source_width

    @property
    def pad_bottom(self) -> int:
        """Height of the green band below the source."""
        return self.canvas_height - self.source_height

    def validate(self) -> None:
        """Raise ``ValueError`` on the first violated invariant.

        Each rule is an independent ``if`` with its own message rather than one
        compound condition, so the caller (and the API error surfaced to the
        UI) always names exactly which rule failed — the same discipline
        ``OutpaintGeometry.validate`` uses.
        """
        if self.source_width <= 0 or self.source_height <= 0:
            raise ValueError(
                "source size must be positive, got "
                f"{self.source_width}x{self.source_height}"
            )
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
        if self.source_width < INPAINT_MIN_SOURCE_SIDE:
            raise ValueError(
                f"source_width must be >= {INPAINT_MIN_SOURCE_SIDE}, "
                f"got {self.source_width}"
            )
        if self.source_height < INPAINT_MIN_SOURCE_SIDE:
            raise ValueError(
                f"source_height must be >= {INPAINT_MIN_SOURCE_SIDE}, "
                f"got {self.source_height}"
            )
        if not 0 <= self.pad_right < CANVAS_MULTIPLE:
            raise ValueError(
                f"pad_right must be in 0..{CANVAS_MULTIPLE - 1} (the canvas is the "
                f"source rounded UP to the next multiple of {CANVAS_MULTIPLE}), got "
                f"{self.pad_right}"
            )
        if not 0 <= self.pad_bottom < CANVAS_MULTIPLE:
            raise ValueError(
                f"pad_bottom must be in 0..{CANVAS_MULTIPLE - 1} (the canvas is the "
                f"source rounded UP to the next multiple of {CANVAS_MULTIPLE}), got "
                f"{self.pad_bottom}"
            )
        # Stage 1 runs at exactly half the canvas resolution and the source
        # rectangle has to land on whole pixels there too, or the half-res
        # de-green would be off by one row/column against the half-res mask.
        # The canvas sides are multiples of 128 and therefore always even; the
        # source sides are what has to be checked.
        if self.source_width % 2 != 0:
            raise ValueError(
                f"source_width must be even (stage 1 runs at half resolution), "
                f"got {self.source_width}"
            )
        if self.source_height % 2 != 0:
            raise ValueError(
                f"source_height must be even (stage 1 runs at half resolution), "
                f"got {self.source_height}"
            )

    def half_dims(self) -> tuple[int, int, int, int]:
        """``(canvas_w, canvas_h, source_w, source_h)`` at HALF resolution.

        The single place the stage-1 geometry is computed, so the stage-1
        de-green and the stage-1 mask can never disagree about where the source
        rectangle ends. Half and only half: stage 1 is the only reduced pass the
        pipeline has, and a general "any divisor" helper would be four callers'
        worth of unused generality plus a divisibility rule to maintain.

        :meth:`validate` already guarantees all four sides are even, so this
        cannot produce a fractional size for a geometry that passed it.
        """
        return (
            self.canvas_width // 2,
            self.canvas_height // 2,
            self.source_width // 2,
            self.source_height // 2,
        )

    def as_dict(self) -> dict[str, int]:
        """Flat dict for ``metadata.json`` — the four stored values plus the pads."""
        return {
            "canvas_width": self.canvas_width,
            "canvas_height": self.canvas_height,
            "source_width": self.source_width,
            "source_height": self.source_height,
            "pad_right": self.pad_right,
            "pad_bottom": self.pad_bottom,
        }


def place_mask_on_canvas(
    mask: "torch.Tensor", geometry: InpaintGeometry
) -> "torch.Tensor":
    """Paste a source-resolution ``(F, 1, sh, sw)`` mask into the canvas.

    Returns ``(F, 1, canvas_height, canvas_width)`` with the SAME dtype: the pad
    bands are filled with zeros, i.e. "do not repaint". That is the correct
    value and not merely a convenient one — the bands carry sentinel green that
    the de-green step replaces with generated pixels before the blend, and they
    are cropped off at the end, so asking the blend to treat them as mask would
    only widen the dilated band that eats into the real picture.

    A mask already at canvas resolution is returned untouched, which is what
    makes the function safe to call twice on the same job (stage 1 and stage 2
    decode the mask separately to avoid holding both resolutions at once).
    """
    import torch

    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError(f"mask must be (F, 1, H, W), got {tuple(mask.shape)}")

    height = int(mask.shape[2])
    width = int(mask.shape[3])
    if (width, height) == (geometry.canvas_width, geometry.canvas_height):
        return mask
    if (width, height) != (geometry.source_width, geometry.source_height):
        raise ValueError(
            "mask must be at the source resolution "
            f"{geometry.source_width}x{geometry.source_height} (or already at the "
            f"canvas resolution {geometry.canvas_width}x{geometry.canvas_height}), "
            f"got {width}x{height}"
        )

    out = torch.zeros(
        (mask.shape[0], 1, geometry.canvas_height, geometry.canvas_width),
        dtype=mask.dtype,
        device=mask.device,
    )
    out[:, :, :height, :width] = mask
    return out


def half_res_mask(
    mask: "torch.Tensor", height: int, width: int, *, chunk_size: int = 8
) -> "torch.Tensor":
    """Area-downscale ``mask`` to ``(height, width)`` and re-binarise at >= 0.5.

    Returns ``(F, 1, height, width)`` **uint8** with values 0 / 255, the same
    convention :func:`engine.pipeline.common.decode_mask_video` produces.

    ``area`` rather than ``nearest`` is load-bearing. Nearest-neighbour sampling
    of a half-resolution grid keeps one pixel out of four, so a one-pixel-wide
    white line survives or vanishes depending on which parity it happens to sit
    on — for a mask traced around a moving object that reads as the mask
    flickering between frames. Area interpolation gives the fraction of the
    source pixel that was white, and thresholding that at 0.5 is a majority
    vote: a line one pixel wide covers exactly half of the 2x1 footprint in the
    axis it is thin in and therefore survives.

    CHUNKED, like :func:`fill_mask_with_generated_` and for the same reason. The
    interpolation needs float32, and converting the whole clip at once would
    materialise the FULL-RESOLUTION mask as float — 4.0GB at 1920x1088x481,
    against the 1.0GB the uint8 input already costs — for a result that is only
    a quarter of that size. Eight frames at a time costs ~65MB and writes
    straight into the pre-allocated uint8 output. Every op involved is
    independent per frame, so the result is identical to the whole-clip form.
    """
    import torch
    import torch.nn.functional as F

    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError(f"mask must be (F, 1, H, W), got {tuple(mask.shape)}")
    if height <= 0 or width <= 0:
        raise ValueError(f"mask size must be positive, got height={height} width={width}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")

    frames = int(mask.shape[0])
    same_size = (int(mask.shape[3]), int(mask.shape[2])) == (width, height)
    out = torch.empty(
        (frames, 1, height, width), dtype=torch.uint8, device=mask.device
    )
    for start in range(0, frames, chunk_size):
        end = min(start + chunk_size, frames)
        work = mask[start:end].to(torch.float32)
        if mask.dtype == torch.uint8:
            work = work.div_(255.0)
        if not same_size:
            work = F.interpolate(work, size=(height, width), mode="area")
        out[start:end] = (work >= 0.5).to(torch.uint8).mul_(255)
        del work
    return out


def fill_pad_bands_with_generated_(
    canvas: "torch.Tensor",
    *,
    generated: "torch.Tensor",
    source_height: int,
    source_width: int,
) -> "torch.Tensor":
    """Replace the canvas' right and bottom pad bands with the generated pixels.

    ``canvas`` and ``generated`` are both ``(F, H, W, 3)`` uint8 at the SAME
    resolution, and ``source_height`` / ``source_width`` describe where the
    source rectangle ends AT THAT RESOLUTION (use
    :meth:`InpaintGeometry.half_dims` rather than dividing by hand). The
    canvas tensor handed in is the one returned — nothing is copied.

    Same reasoning as ``engine.outpaint.canvas.fill_pad_with_generated_``, which
    this is the two-band twin of: the coarse levels of the Laplacian pyramid mix
    the canvas' DC component into the generated area whatever the mask says, so
    what bleeds inwards should be the picture the model just drew rather than a
    flat #66FF00. Two slice assignments rather than a boolean index, for the
    same reason as well — a boolean index materialises every selected pixel in a
    fresh tensor first.

    The bands are derived from the SOURCE SIZE here rather than from the mask
    (outpainting reads them off its rectangular mask plane), because an inpaint
    mask is an arbitrary per-frame shape and carries no information at all about
    where the pad starts.
    """
    height = int(canvas.shape[1])
    width = int(canvas.shape[2])
    if canvas.shape != generated.shape:
        raise ValueError(
            "canvas and generated must have the same shape, "
            f"{tuple(canvas.shape)} != {tuple(generated.shape)}"
        )
    if not 0 < source_width <= width or not 0 < source_height <= height:
        raise ValueError(
            f"source rectangle {source_width}x{source_height} does not fit the "
            f"canvas {width}x{height}"
        )

    bands = (
        (slice(0, source_height), slice(source_width, width)),  # right band
        (slice(source_height, height), slice(0, width)),        # bottom band, full width
    )
    filled = 0
    for band_rows, band_cols in bands:
        canvas[:, band_rows, band_cols, :] = generated[:, band_rows, band_cols, :]
        filled += (band_rows.stop - band_rows.start) * (band_cols.stop - band_cols.start)

    logger.info(
        "inpaint de-green (pad bands): canvas %s, source %dx%d, %d px/frame replaced",
        tuple(int(v) for v in canvas.shape),
        source_width,
        source_height,
        filled,
    )
    return canvas


def fill_mask_with_generated_(
    canvas: "torch.Tensor",
    *,
    generated: "torch.Tensor",
    mask: "torch.Tensor",
    chunk_size: int = 8,
) -> "torch.Tensor":
    """Replace the canvas' MASKED pixels with the generated ones, in place.

    ``canvas`` / ``generated`` are ``(F, H, W, 3)`` uint8; ``mask`` is
    ``(F, 1, H, W)`` (uint8 0/255 or float 0/1) at the same resolution, one
    plane per frame. The canvas tensor handed in is the one returned.

    This is the arbitrary-shape counterpart of
    :func:`fill_pad_bands_with_generated_` and exists for exactly the same
    reason: the green the model was asked to paint over must not be what the
    coarse pyramid levels bleed outwards into the picture that surrounds it.
    Note the asymmetry with outpainting — there the green sits OUTSIDE the kept
    rectangle, here it sits INSIDE the frame with real footage all around it, so
    leaving it in place would tint the whole neighbourhood rather than a border.

    Chunked ``torch.where`` rather than one whole-timeline call: the boolean
    mask has to be broadcast to ``(n, H, W, 3)`` and ``torch.where`` allocates a
    fresh output, so a 1920x1088x481 canvas would need a second 3.0GB buffer at
    once. Eight frames at a time costs ~50MB and writes straight back into the
    canvas' own storage.
    """
    import torch

    if canvas.shape != generated.shape:
        raise ValueError(
            "canvas and generated must have the same shape, "
            f"{tuple(canvas.shape)} != {tuple(generated.shape)}"
        )
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError(f"mask must be (F, 1, H, W), got {tuple(mask.shape)}")
    if tuple(mask.shape[-2:]) != tuple(canvas.shape[1:3]):
        raise ValueError(
            "the mask and the canvas must have the same spatial resolution, "
            f"mask is {tuple(mask.shape[-2:])} and canvas is {tuple(canvas.shape[1:3])}"
        )
    if mask.shape[0] not in (1, canvas.shape[0]):
        raise ValueError(
            f"mask must carry 1 or {canvas.shape[0]} frames, got {mask.shape[0]}"
        )
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")

    frames = int(canvas.shape[0])
    one_plane = mask.shape[0] == 1
    white = 0
    for start in range(0, frames, chunk_size):
        end = min(start + chunk_size, frames)
        plane = mask[0:1] if one_plane else mask[start:end]
        # (n, 1, H, W) -> (n, H, W, 1): broadcasts across the three channels.
        selector = (plane > 0.5).permute(0, 2, 3, 1)
        white += int(selector.sum()) * (1 if not one_plane else (end - start))
        canvas[start:end] = torch.where(
            selector, generated[start:end], canvas[start:end]
        )

    logger.info(
        "inpaint de-green (mask): canvas %s, %d masked px over the clip",
        tuple(int(v) for v in canvas.shape),
        white,
    )
    return canvas


def restore_outside_mask_(
    blended: "torch.Tensor",
    *,
    source: "torch.Tensor",
    dilated_mask: "torch.Tensor",
    chunk_size: int = 8,
) -> "torch.Tensor":
    """Put the original pixels back wherever the DILATED mask is exactly zero.

    ``blended`` / ``source`` are ``(F, H, W, 3)`` uint8 at the same resolution
    and ``dilated_mask`` is ``(F, 1, H, W)`` or ``(1, 1, H, W)`` — the output of
    ``pyramid_blend.apply_low_res_mask_dilation`` at the blend's own dilation
    radius. ``blended`` is modified in place and returned.

    A **bool** mask is accepted as well as a float one, and means the same thing:
    ``True`` where the dilation reached. The predicate below is ``< 1e-6``, which
    on a bool tensor reads ``False`` as zero and ``True`` as one, so the pipeline
    can do its comparison on the GPU and send only the verdict back (see
    ``inpaint_pipeline._restore_and_measure_``) without a second convention.

    **Why "exactly zero" and not "below a half".** The dilated mask is not
    binary: the dilation resizes to a 64px long side, max-pools, and resizes
    back, so around the mask there is a ramp from 1.0 down to 0.0 that is
    roughly ``r * long_side / 64 + 18`` pixels wide (measured — see
    Docs/VERIFICATION_LOG.md). That ramp is exactly the band the Laplacian blend
    used to hide the seam. Restoring any of it would put a hard edge back where
    the blend had just removed one, so the rule is the weakest one that still
    delivers the promise: only pixels the blend provably never touched —
    ``dilated < 1e-6`` — go back to the original.

    The guarantee this buys is pixel-exact: outside the dilated support the
    delivered frame IS the source frame, byte for byte, up to the final mp4
    encode.
    """
    import torch

    if blended.shape != source.shape:
        raise ValueError(
            "blended and source must have the same shape, "
            f"{tuple(blended.shape)} != {tuple(source.shape)}"
        )
    if dilated_mask.ndim != 4 or dilated_mask.shape[1] != 1:
        raise ValueError(
            f"dilated_mask must be (F, 1, H, W), got {tuple(dilated_mask.shape)}"
        )
    if tuple(dilated_mask.shape[-2:]) != tuple(blended.shape[1:3]):
        raise ValueError(
            "the dilated mask and the video must have the same spatial resolution, "
            f"mask is {tuple(dilated_mask.shape[-2:])} and video is "
            f"{tuple(blended.shape[1:3])}"
        )
    if dilated_mask.shape[0] not in (1, blended.shape[0]):
        raise ValueError(
            f"dilated_mask must carry 1 or {blended.shape[0]} frames, "
            f"got {dilated_mask.shape[0]}"
        )
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")

    frames = int(blended.shape[0])
    one_plane = dilated_mask.shape[0] == 1
    for start in range(0, frames, chunk_size):
        end = min(start + chunk_size, frames)
        plane = dilated_mask[0:1] if one_plane else dilated_mask[start:end]
        outside = (plane < 1e-6).permute(0, 2, 3, 1)
        blended[start:end] = torch.where(
            outside, source[start:end], blended[start:end]
        )
    return blended
