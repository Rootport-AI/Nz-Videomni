"""Laplacian pyramid blending — a pure-torch port of ``LTXVLaplacianPyramidBlend``.

The official LTXV ComfyUI node (``ltxv_nodes/pyramid_blending.py``, mirrored
under ``uploads/_outpaint_verify/``) implements outpaint seam removal with
kornia's pyramid helpers. This project does not ship kornia, so the four kornia
primitives the node touches -- ``pyrdown`` / ``pyrup`` / ``build_pyramid`` /
``build_laplacian_pyramid`` -- are reimplemented here directly on top of
``torch.nn.functional``. They are numerically equal to kornia 0.8.3 (verified
against a golden fixture generated with the real kornia; see
``tests/test_outpaint_pyramid_blend.py``).

Details that are easy to get wrong and are therefore reproduced verbatim:

* The 5x5 binomial kernel is **already normalised** (``/256``) and is applied as
  a *correlation* (no flip), depthwise, with ``reflect`` padding of 2 on each
  side.
* ``pyrup`` does **not** apply the OpenCV ``x4`` gain -- it upsamples
  bilinearly and then blurs with the same normalised kernel. Adding the gain
  would look "correct" in isolation but would not match the node.
* ``pyrdown``'s target size is written asymmetrically upstream
  (``int(float(h) / factor)`` for the height, ``int(float(w) // factor)`` for
  the width). Kept as-is: for ``factor = 2.0`` the two agree, and diverging
  from the original buys nothing.
* ``build_laplacian_pyramid`` here assumes its input is already a power of two
  on both sides (the caller guarantees it), so kornia's internal re-padding --
  which is guarded by an ``or`` and would never fire for our inputs anyway --
  is not ported.

Deliberate deviations from the official node are documented on
``_prepare_mask_pyramid`` and ``blend_video_u8``.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

# Frames processed per device round-trip. Matches the official node.
_CHUNK_SIZE = 8

# The mask is dilated at this (long-side) resolution, not at full resolution.
_MASK_LOW_RES_LONG_SIDE = 64

_KERNEL_2D = (
    torch.tensor(
        [
            [1.0, 4.0, 6.0, 4.0, 1.0],
            [4.0, 16.0, 24.0, 16.0, 4.0],
            [6.0, 24.0, 36.0, 24.0, 6.0],
            [4.0, 16.0, 24.0, 16.0, 4.0],
            [1.0, 4.0, 6.0, 4.0, 1.0],
        ]
    )
    / 256.0
)


# ---------------------------------------------------------------------------
# power-of-two helpers (kornia parity)
# ---------------------------------------------------------------------------
def is_powerof_two(x: int) -> bool:
    """True when ``x`` is a non-zero power of two."""
    return bool(x) and (not (x & (x - 1)))


def find_next_powerof_two(x: int) -> int:
    """Smallest power of two >= ``x`` (``x`` itself when already a power of two)."""
    return 1 << (x - 1).bit_length()


# ---------------------------------------------------------------------------
# pyramid primitives
# ---------------------------------------------------------------------------
def _filter2d(x: Tensor, border_type: str) -> Tensor:
    """Depthwise correlation with the normalised 5x5 binomial kernel.

    Equivalent to ``kornia.filters.filter2d(x, kernel, border_type)`` with the
    library defaults (``normalized=False``, ``padding="same"``,
    ``behaviour="corr"``): pad by 2 on every side with ``border_type``, then a
    stride-1, zero-padding, grouped convolution with one kernel copy per
    channel.
    """
    channels = x.shape[1]
    weight = _KERNEL_2D.to(device=x.device, dtype=x.dtype)
    weight = weight.reshape(1, 1, 5, 5).expand(channels, 1, 5, 5)
    padded = F.pad(x, [2, 2, 2, 2], mode=border_type)
    return F.conv2d(padded, weight, groups=channels, padding=0, stride=1)


def pyrdown(
    x: Tensor,
    border_type: str = "reflect",
    align_corners: bool = False,
    factor: float = 2.0,
) -> Tensor:
    """Blur then bilinearly downsample by ``factor`` (kornia ``pyrdown``)."""
    _, _, height, width = x.shape
    x_blur = _filter2d(x, border_type)
    return F.interpolate(
        x_blur,
        size=(int(float(height) / factor), int(float(width) // factor)),
        mode="bilinear",
        align_corners=align_corners,
    )


def pyrup(x: Tensor, border_type: str = "reflect", align_corners: bool = False) -> Tensor:
    """Bilinearly upsample x2 then blur (kornia ``pyrup``).

    No gain factor is applied -- see the module docstring.
    """
    _, _, height, width = x.shape
    x_up = F.interpolate(
        x, size=(height * 2, width * 2), mode="bilinear", align_corners=align_corners
    )
    return _filter2d(x_up, border_type)


def build_pyramid(
    x: Tensor,
    max_level: int,
    border_type: str = "reflect",
    align_corners: bool = False,
) -> list[Tensor]:
    """Gaussian pyramid: ``[x, pyrdown(x), pyrdown(pyrdown(x)), ...]``.

    ``max_level`` counts levels including the original, so the result has
    ``max_level`` entries.
    """
    pyramid = [x]
    for _ in range(max_level - 1):
        pyramid.append(pyrdown(pyramid[-1], border_type, align_corners))
    return pyramid


def build_laplacian_pyramid(
    x: Tensor,
    max_level: int,
    border_type: str = "reflect",
    align_corners: bool = False,
) -> list[Tensor]:
    """Laplacian pyramid; the last entry is the Gaussian residual itself.

    ``x`` must already have power-of-two height and width (the caller pads).
    """
    gaussian = build_pyramid(x, max_level, border_type, align_corners)
    laplacian: list[Tensor] = []
    for i in range(max_level - 1):
        laplacian.append(gaussian[i] - pyrup(gaussian[i + 1], border_type, align_corners))
    laplacian.append(gaussian[-1])
    return laplacian


# ---------------------------------------------------------------------------
# mask preprocessing
# ---------------------------------------------------------------------------
def _resize_preserving_aspect_ratio(x: Tensor, long_side: int, mode: str) -> Tensor:
    h, w = x.shape[-2:]
    current_long_side = max(h, w)
    if current_long_side == long_side:
        return x

    scale = long_side / current_long_side
    resized_h = max(1, int(round(h * scale)))
    resized_w = max(1, int(round(w * scale)))

    if mode == "nearest":
        return F.interpolate(x, size=(resized_h, resized_w), mode=mode)
    return F.interpolate(x, size=(resized_h, resized_w), mode=mode, align_corners=False)


def apply_low_res_mask_dilation(
    mask: Tensor,
    spatial_radius: int,
    long_side: int = _MASK_LOW_RES_LONG_SIDE,
) -> Tensor:
    """Grow the white region of ``mask`` by ``spatial_radius`` at low resolution.

    The mask is resized to a ``long_side``-px long side, max-pooled with a
    ``2r+1`` window, then resized back. Working at low resolution is what makes
    the transition band scale with the canvas: the effective full-resolution
    growth is ``r * (long_side_px / long_side)``.
    """
    if spatial_radius <= 0:
        return mask

    original_size = mask.shape[-2:]
    low_res = _resize_preserving_aspect_ratio(mask.float(), long_side, mode="bilinear")
    low_res = F.max_pool2d(
        low_res,
        kernel_size=spatial_radius * 2 + 1,
        stride=1,
        padding=spatial_radius,
    )
    return F.interpolate(low_res, size=original_size, mode="bilinear", align_corners=False)


# ---------------------------------------------------------------------------
# internal plumbing shared by both public entry points
# ---------------------------------------------------------------------------
def _power_of_two_padding(height: int, width: int) -> tuple[int, int]:
    """``(pad_right, pad_down)`` needed to reach power-of-two sides.

    Note the ``and``: padding is applied unless **both** sides are already
    powers of two. (kornia's own internal guard uses ``or``; the official node
    deliberately uses ``and``, so a 64x80 input -- one side already a power of
    two -- is still padded, to 64x128.)
    """
    if is_powerof_two(height) and is_powerof_two(width):
        return 0, 0
    return find_next_powerof_two(width) - width, find_next_powerof_two(height) - height


def _pad_for_laplacian(x: Tensor) -> tuple[Tensor, tuple[int, int]]:
    """Reflect-pad right/bottom so both sides are powers of two."""
    pad_right, pad_down = _power_of_two_padding(x.shape[2], x.shape[3])
    if pad_right or pad_down:
        x = F.pad(x, (0, pad_right, 0, pad_down), mode="reflect")
    return x, (pad_right, pad_down)


def _prepare_mask_pyramid(
    mask: Tensor,
    *,
    height: int,
    width: int,
    padding: tuple[int, int],
    max_level: int,
    mask_low_res_dilation: int,
    device: torch.device | None,
    dtype: torch.dtype,
) -> list[Tensor]:
    """Dilate, pad and pyramid-decompose the single-frame mask, once.

    **Deliberate deviation from the official node.** Upstream carries a mask of
    ``B`` frames and dilates / pads / decomposes all ``B`` of them. Our
    outpainting mask is a *static rectangular frame*, identical for every frame,
    so we do the whole thing on a single ``(1, 1, H, W)`` plane and let the
    per-chunk arithmetic broadcast it across the batch. Mathematically this is
    exactly equivalent (every op involved -- ``interpolate``, ``max_pool2d``,
    ``pad``, grouped ``conv2d`` -- is independent per sample), and it avoids a
    2.0GB float32 allocation at 1920x1088x241: ``F.pad`` and ``F.interpolate``
    cannot preserve the stride-0 of an ``expand``ed view and would materialise
    the full batch.
    """
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError(f"mask must have shape (1, 1, H, W), got {tuple(mask.shape)}")
    if mask.shape[0] != 1:
        raise ValueError(
            "mask must carry exactly one frame with shape (1, 1, H, W) -- the "
            f"outpaint mask is static across the clip, got {tuple(mask.shape)}"
        )
    if tuple(mask.shape[-2:]) != (height, width):
        raise ValueError(
            "image_a, image_b, and mask must have the same spatial resolution "
            f"for blending, got mask {tuple(mask.shape[-2:])} vs image ({height}, {width})"
        )

    if device is not None:
        mask = mask.to(device)
    mask = mask.to(dtype)
    mask = apply_low_res_mask_dilation(mask, mask_low_res_dilation).to(dtype)

    pad_right, pad_down = padding
    if pad_right or pad_down:
        mask = F.pad(mask, (0, pad_right, 0, pad_down), mode="reflect")

    return build_pyramid(mask, max_level)


def _blend_chunk(
    chunk_a: Tensor,
    chunk_b: Tensor,
    mask_pyramid: list[Tensor],
    max_level: int,
    orig_h: int,
    orig_w: int,
) -> Tensor:
    """Blend one already-on-device chunk and crop it back to the original size.

    ``chunk_a`` / ``chunk_b`` are ``(n, C, H, W)`` float tensors that have NOT
    yet been power-of-two padded; ``mask_pyramid`` holds ``(1, 1, h, w)`` levels
    that broadcast over the batch and channel axes.
    """
    padded_a, _ = _pad_for_laplacian(chunk_a)
    padded_b, _ = _pad_for_laplacian(chunk_b)

    pyr_a = build_laplacian_pyramid(padded_a, max_level=max_level)
    pyr_b = build_laplacian_pyramid(padded_b, max_level=max_level)

    pm = mask_pyramid[-1]
    out = pyr_a[-1] * pm + pyr_b[-1] * (1 - pm)
    for i in range(len(pyr_a) - 2, -1, -1):
        pm = mask_pyramid[i]
        residual = pyr_a[i] * pm + pyr_b[i] * (1 - pm)
        out = pyrup(out) + residual

    return out[..., :orig_h, :orig_w].clamp(0, 1)


def _plan(
    height: int,
    width: int,
    mask: Tensor,
    *,
    max_level: int,
    mask_low_res_dilation: int,
    device: torch.device | None,
    dtype: torch.dtype,
) -> tuple[int, list[Tensor]]:
    """Resolve the effective pyramid depth and build the mask pyramid."""
    padding = _power_of_two_padding(height, width)
    padded_min = min(height + padding[1], width + padding[0])
    effective_level = min(max_level, int(math.log2(padded_min)))

    mask_pyramid = _prepare_mask_pyramid(
        mask,
        height=height,
        width=width,
        padding=padding,
        max_level=effective_level,
        mask_low_res_dilation=mask_low_res_dilation,
        device=device,
        dtype=dtype,
    )
    return effective_level, mask_pyramid


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def laplacian_pyramid_blend(
    image_a: Tensor,
    image_b: Tensor,
    mask: Tensor,
    *,
    max_level: int = 7,
    mask_low_res_dilation: int = 5,
    chunk_size: int | None = None,
    device: torch.device | None = None,
    output_device: torch.device | None = None,
) -> Tensor:
    """Seamlessly blend ``image_a`` into ``image_b`` under ``mask``.

    Args:
        image_a: ``(B, 3, H, W)`` float32 in ``[0, 1]`` -- taken where mask == 1.
        image_b: ``(B, 3, H, W)`` float32 in ``[0, 1]`` -- taken where mask == 0.
        mask: ``(1, 1, H, W)`` float32 in ``[0, 1]``. Exactly one frame; the
            outpaint mask is static, so a per-frame mask is rejected rather
            than silently broadcast from the wrong axis.
        max_level: requested pyramid depth; clamped to ``log2`` of the padded
            short side.
        mask_low_res_dilation: dilation radius applied at the 64px working
            resolution (0 disables).
        chunk_size: frames per pass (default ``_CHUNK_SIZE``).
        device: compute device; ``None`` computes where the inputs already live.
        output_device: where results are collected; ``None`` keeps them on the
            compute device.

    Returns:
        ``(B, 3, H, W)`` float32 in ``[0, 1]``.
    """
    if image_a.shape != image_b.shape:
        raise ValueError(
            f"input images must have the same size, {image_a.shape} != {image_b.shape}"
        )
    if image_a.ndim != 4:
        raise ValueError(f"input images must be (B, C, H, W), got {tuple(image_a.shape)}")

    orig_h, orig_w = image_a.shape[-2], image_a.shape[-1]
    effective_level, mask_pyramid = _plan(
        orig_h,
        orig_w,
        mask,
        max_level=max_level,
        mask_low_res_dilation=mask_low_res_dilation,
        device=device,
        dtype=image_a.dtype,
    )

    step = _CHUNK_SIZE if chunk_size is None else chunk_size
    if step <= 0:
        raise ValueError(f"chunk_size must be > 0, got {step}")

    results: list[Tensor] = []
    for start in range(0, image_a.shape[0], step):
        end = min(start + step, image_a.shape[0])
        chunk_a = image_a[start:end]
        chunk_b = image_b[start:end]
        if device is not None:
            chunk_a = chunk_a.to(device)
            chunk_b = chunk_b.to(device)

        blended = _blend_chunk(chunk_a, chunk_b, mask_pyramid, effective_level, orig_h, orig_w)
        results.append(blended.to(output_device) if output_device is not None else blended)

    return torch.cat(results, dim=0)


def blend_video_u8(
    generated: Tensor,
    original: Tensor,
    mask: Tensor,
    *,
    max_level: int = 7,
    mask_low_res_dilation: int = 5,
    chunk_size: int | None = None,
    device: torch.device | None = None,
) -> Tensor:
    """uint8 NHWC front end for :func:`laplacian_pyramid_blend`.

    Args:
        generated: ``(F, H, W, 3)`` uint8 -- the generated video (mask == 1).
        original: ``(F, H, W, 3)`` uint8 -- the green-composited canvas
            (mask == 0).
        mask: ``(1, 1, H, W)`` float32.

    Returns:
        ``(F, H, W, 3)`` uint8 on the CPU.

    Why this exists rather than "just call ``laplacian_pyramid_blend``": a full
    resolution 1920x1088x241 clip held as float32 ``(B, 3, H, W)`` is 6.0GB per
    copy, and the blend needs three of them (a, b, out) -- 18GB, which the
    machines this ships to do not have. Keeping the two inputs and the output as
    uint8 NHWC costs ~4.5GB resident, and each chunk is converted to float32,
    blended, quantised back and written into the pre-allocated output. Following
    ``chain_pipeline.py``'s V2V path -- which hands ``encode_video_output`` a
    fully materialised tensor -- this returns a finished tensor rather than a
    generator, so the caller's encode path is unchanged.
    """
    if generated.shape != original.shape:
        raise ValueError(
            f"input videos must have the same size, {generated.shape} != {original.shape}"
        )
    if generated.ndim != 4 or generated.shape[-1] != 3:
        raise ValueError(f"input videos must be (F, H, W, 3), got {tuple(generated.shape)}")
    if generated.dtype != torch.uint8 or original.dtype != torch.uint8:
        # Not just a contract check: the per-chunk conversion below scales in
        # place, which is only safe because `.to(torch.float32)` on a uint8
        # tensor always copies. A float32 input would be scaled under the
        # caller's feet.
        raise ValueError(
            "input videos must be uint8, got "
            f"generated={generated.dtype} original={original.dtype}"
        )

    frames, orig_h, orig_w, _ = generated.shape
    effective_level, mask_pyramid = _plan(
        orig_h,
        orig_w,
        mask,
        max_level=max_level,
        mask_low_res_dilation=mask_low_res_dilation,
        device=device,
        dtype=torch.float32,
    )

    step = _CHUNK_SIZE if chunk_size is None else chunk_size
    if step <= 0:
        raise ValueError(f"chunk_size must be > 0, got {step}")

    out = torch.empty((frames, orig_h, orig_w, 3), dtype=torch.uint8, device="cpu")
    for start in range(0, frames, step):
        end = min(start + step, frames)
        chunk_a = generated[start:end].permute(0, 3, 1, 2)
        chunk_b = original[start:end].permute(0, 3, 1, 2)
        if device is not None:
            chunk_a = chunk_a.to(device)
            chunk_b = chunk_b.to(device)
        chunk_a = chunk_a.to(torch.float32).div_(255.0)
        chunk_b = chunk_b.to(torch.float32).div_(255.0)

        blended = _blend_chunk(chunk_a, chunk_b, mask_pyramid, effective_level, orig_h, orig_w)
        quantised = blended.mul(255.0).round().clamp(0, 255).to(torch.uint8)
        out[start:end] = quantised.permute(0, 2, 3, 1).cpu()

    return out
