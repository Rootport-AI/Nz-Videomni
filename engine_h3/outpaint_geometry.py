"""MiniMax H3 Outpainting geometry and paste-back -- numpy only, torch-free (台帳 §1-125).

Docs/MINIMAX_H3_DESIGN.md §3.1 is the home of the design; the plan's §2.2
spells out the formulas implemented here. Pure functions, so the CPU tests
(tests/test_h3_outpaint_geometry.py) can check every boundary case.

Symbols: the source video is ``W_s x H_s``; the canvas ``W_c x H_c`` (multiples
of 32) is the source with ``pad_left / pad_right / pad_top / pad_bottom``
pixels added. The kept rectangle is ``[pl, pl + W_s) x [pt, pt + H_s)`` on the
canvas.

* 32-aligned bounding box: ``x0 = floor(pl/32)*32``, ``y0 = floor(pt/32)*32``,
  ``x1 = ceil((pl+W_s)/32)*32``, ``y1 = ceil((pt+H_s)/32)*32``. The source is
  encoded alone at this size (a multiple of 32, so of the encoder's 16) and its
  latent is placed at ``(y0/16, x0/16)`` of the canvas latent.
* The gaps between the bounding box and the source (0-31 px per side) are the
  source's edge mirrored (``numpy.pad(mode="reflect")``, the same kind of
  padding the encoder itself uses at an image edge): no new colour comes in.
* Token (i, j) (pixels ``[32j, 32j+32) x [32i, 32i+32)``) is FIXED (value 0)
  only when it lies entirely inside the kept rectangle; any token touching a
  gap or the pad is generated (value 1). 保持矩形に完全に含まれるトークンだけを固定する。
* Feather (paste-back): the source pixel weight ``w = min(1, d / F)``, where
  ``d`` is the distance to the nearest side whose pad is > 0 (the outermost
  pixel row / column of the kept rectangle on that side has ``d = 0``, so it
  is fully generated when F > 0); sides with pad 0 touch the canvas edge and
  have no seam, so they are not counted. ``F = 0`` gives ``w = 1`` everywhere
  (a hard paste).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from engine_h3 import constants as _C

#: One DiT token is 32 px square (VAE 16 x patch 2); constants.SPATIAL_FACTOR.
TOKEN_PX = _C.SPATIAL_FACTOR
#: One video latent cell is 16 px square; constants.VAE_SPATIAL.
LATENT_PX = _C.VAE_SPATIAL
#: Frames blended per chunk in ``composite`` (keeps the float32 peak to a few frames).
_COMPOSITE_CHUNK = 4


@dataclass(frozen=True)
class OutpaintGeometry:
    """The canvas, the pads, the 32-aligned bounding box and the fixed-token grid of one Outpainting job.

    ``x0, y0, x1, y1``: the bounding box in canvas pixels (multiples of 32).
    ``fixed``: read-only bool ``[grid_h, grid_w]``, True = a fixed token (it lies
    entirely inside the kept rectangle). It is derived from the other fields,
    so it is left out of comparison and hashing.
    """

    source_w: int
    source_h: int
    canvas_w: int
    canvas_h: int
    pad_left: int
    pad_right: int
    pad_top: int
    pad_bottom: int
    x0: int
    y0: int
    x1: int
    y1: int
    fixed: np.ndarray = field(compare=False, repr=False)

    @property
    def aligned_w(self) -> int:
        return self.x1 - self.x0

    @property
    def aligned_h(self) -> int:
        return self.y1 - self.y0

    @property
    def grid_w(self) -> int:
        return self.canvas_w // TOKEN_PX

    @property
    def grid_h(self) -> int:
        return self.canvas_h // TOKEN_PX

    @property
    def fixed_count(self) -> int:
        return int(self.fixed.sum())

    @property
    def generated_count(self) -> int:
        return int(self.fixed.size) - self.fixed_count

    @property
    def latent_offset(self) -> tuple[int, int]:
        """``(y0 / 16, x0 / 16)``: where the bounding box's latent goes in the canvas latent."""
        return self.y0 // LATENT_PX, self.x0 // LATENT_PX

    @property
    def source_offset_in_aligned(self) -> tuple[int, int]:
        """``(pad_top - y0, pad_left - x0)``: the source's top-left inside the bounding box (0-31 each)."""
        return self.pad_top - self.y0, self.pad_left - self.x0


def _int(value, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"Outpainting: {what} は整数で指定してください（{value!r}）")
    return int(value)


def compute_outpaint_geometry(
    source_w, source_h, canvas_w, canvas_h, pad_left, pad_right, pad_top, pad_bottom,
) -> OutpaintGeometry:
    """The geometry of placing a ``source_w x source_h`` video on the canvas with the four pads.

    Raises ValueError (理由は日本語) when a pad is negative or all are 0, the
    canvas is not a multiple of 32, the canvas is not the source plus the pads,
    or no token lies entirely inside the kept rectangle.
    """
    sw, sh = _int(source_w, "source_w"), _int(source_h, "source_h")
    cw, ch = _int(canvas_w, "canvas_w"), _int(canvas_h, "canvas_h")
    pl, pr = _int(pad_left, "pad_left"), _int(pad_right, "pad_right")
    pt, pb = _int(pad_top, "pad_top"), _int(pad_bottom, "pad_bottom")
    if sw <= 0 or sh <= 0:
        raise ValueError(f"Outpainting: 元の映像の寸法が正ではありません（{sw}x{sh}）")
    if min(pl, pr, pt, pb) < 0:
        raise ValueError(f"Outpainting: pad は 0 以上にしてください（左 {pl}・右 {pr}・上 {pt}・下 {pb}）")
    if pl + pr + pt + pb <= 0:
        raise ValueError("Outpainting: pad の合計が 0 です（広げる辺がありません）")
    if cw <= 0 or ch <= 0 or cw % TOKEN_PX or ch % TOKEN_PX:
        raise ValueError(f"Outpainting: キャンバスの寸法は {TOKEN_PX} の倍数にしてください（{cw}x{ch}）")
    if cw != sw + pl + pr or ch != sh + pt + pb:
        raise ValueError(
            f"Outpainting: キャンバス {cw}x{ch} が元の映像 {sw}x{sh} と pad の和"
            f"（{sw + pl + pr}x{sh + pt + pb}）に一致しません")
    x0 = (pl // TOKEN_PX) * TOKEN_PX
    y0 = (pt // TOKEN_PX) * TOKEN_PX
    x1 = -(-(pl + sw) // TOKEN_PX) * TOKEN_PX
    y1 = -(-(pt + sh) // TOKEN_PX) * TOKEN_PX
    gw, gh = cw // TOKEN_PX, ch // TOKEN_PX
    # Token j covers [32j, 32j+32): fixed when pl <= 32j and 32j+32 <= pl+W_s.
    starts_x = np.arange(gw) * TOKEN_PX
    starts_y = np.arange(gh) * TOKEN_PX
    cols = (starts_x >= pl) & (starts_x + TOKEN_PX <= pl + sw)
    rows = (starts_y >= pt) & (starts_y + TOKEN_PX <= pt + sh)
    fixed = rows[:, None] & cols[None, :]
    fixed.setflags(write=False)
    if not bool(fixed.any()):
        raise ValueError(
            f"Outpainting: 元の映像 {sw}x{sh} に完全に含まれる {TOKEN_PX} 画素のトークンが 1 つもありません")
    return OutpaintGeometry(
        source_w=sw, source_h=sh, canvas_w=cw, canvas_h=ch,
        pad_left=pl, pad_right=pr, pad_top=pt, pad_bottom=pb,
        x0=x0, y0=y0, x1=x1, y1=y1, fixed=fixed,
    )


def _check_frames(frames: np.ndarray, h: int, w: int, what: str) -> None:
    if not isinstance(frames, np.ndarray) or frames.dtype != np.uint8 or frames.ndim != 4 \
            or frames.shape[1:] != (h, w, 3):
        shape = list(getattr(frames, "shape", ()))
        dtype = getattr(frames, "dtype", type(frames).__name__)
        raise ValueError(f"Outpainting: {what} は uint8 [F, {h}, {w}, 3] が必要です（{shape}, {dtype}）")


def aligned_frames(frames: np.ndarray, geom: OutpaintGeometry) -> np.ndarray:
    """uint8 ``[F, H_s, W_s, 3]`` -> uint8 ``[F, aligned_h, aligned_w, 3]``: the source with the gaps mirrored.

    The gap on each side (0-31 px) is the source's edge reflected
    (``numpy.pad(mode="reflect")``, the edge pixel itself not repeated). A gap
    is always narrower than the source, so the reflection never runs out.
    """
    _check_frames(frames, geom.source_h, geom.source_w, "元の映像")
    top, left = geom.source_offset_in_aligned
    bottom = geom.y1 - (geom.pad_top + geom.source_h)
    right = geom.x1 - (geom.pad_left + geom.source_w)
    if top == bottom == left == right == 0:
        return frames.copy()
    return np.pad(frames, ((0, 0), (top, bottom), (left, right), (0, 0)), mode="reflect")


def token_values(geom: OutpaintGeometry, t_lat: int) -> np.ndarray:
    """float32 ``[t_lat, grid_h, grid_w]``: 0.0 for the fixed tokens, 1.0 for the generated ones, every frame alike.

    The per-token values ``engine_h3.denoise_mask.video_mask`` takes.
    """
    t = _int(t_lat, "t_lat")
    if t <= 0:
        raise ValueError(f"Outpainting: 潜在フレーム数が正ではありません（{t}）")
    one = np.where(geom.fixed, np.float32(0.0), np.float32(1.0)).astype(np.float32)
    return np.ascontiguousarray(np.broadcast_to(one, (t,) + one.shape))


def feather_weights(geom: OutpaintGeometry, feather_px: int) -> np.ndarray:
    """float32 ``[H_s, W_s]``: the source pixel's weight ``w = min(1, d / F)`` in the paste-back.

    ``d``: the distance in pixels to the nearest side whose pad is > 0, counted
    so that the kept rectangle's outermost pixel on that side has ``d = 0``
    (w = 0 there: fully generated when F > 0). Sides with pad 0 are not
    counted (they touch the canvas edge, so there is no seam). ``F = 0`` -> all 1.
    """
    f = _int(feather_px, "feather_px")
    if f < 0:
        raise ValueError(f"Outpainting: なじませの幅は 0 以上にしてください（{f}）")
    h, w = geom.source_h, geom.source_w
    if f == 0:
        return np.ones((h, w), dtype=np.float32)
    far = np.float32(np.inf)
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    dx = np.full(w, far, dtype=np.float32)
    dy = np.full(h, far, dtype=np.float32)
    if geom.pad_left > 0:
        dx = np.minimum(dx, xs)
    if geom.pad_right > 0:
        dx = np.minimum(dx, (w - 1) - xs)
    if geom.pad_top > 0:
        dy = np.minimum(dy, ys)
    if geom.pad_bottom > 0:
        dy = np.minimum(dy, (h - 1) - ys)
    d = np.minimum(dy[:, None], dx[None, :])
    return np.minimum(np.float32(1.0), d / np.float32(f)).astype(np.float32)


def composite(
    decoded: np.ndarray, source: np.ndarray, weights: np.ndarray, geom: OutpaintGeometry,
) -> np.ndarray:
    """Paste the source back into the decoded canvas frames IN PLACE and return ``decoded`` itself.

    ``decoded``: uint8 ``[F, H_c, W_c, 3]`` (written in place); ``source``:
    uint8 ``[F, H_s, W_s, 3]``; ``weights``: float32 ``[H_s, W_s]``
    (``feather_weights``). Inside the kept rectangle each pixel becomes
    ``w source + (1 - w) decoded`` (float32, rounded half up to uint8). The
    w == 1 pixels are the source copied as is (no arithmetic, so bit-identical),
    the w == 0 pixels stay decoded, and nothing outside the kept rectangle is
    touched. 数フレームずつ処理して float32 の山を抑える。
    """
    _check_frames(decoded, geom.canvas_h, geom.canvas_w, "復号した映像")
    _check_frames(source, geom.source_h, geom.source_w, "元の映像")
    if decoded.shape[0] != source.shape[0]:
        raise ValueError(f"Outpainting: フレーム数が一致しません（復号 {decoded.shape[0]}・元 {source.shape[0]}）")
    weights = np.asarray(weights)
    if weights.shape != (geom.source_h, geom.source_w):
        raise ValueError(
            f"Outpainting: 重みは [{geom.source_h}, {geom.source_w}] が必要です（{list(weights.shape)}）")
    w = weights.astype(np.float32, copy=False)
    if not bool(np.isfinite(w).all()) or bool((w < 0).any()) or bool((w > 1).any()):
        raise ValueError("Outpainting: 重みは有限で 0〜1 にしてください")
    pt, pl = geom.pad_top, geom.pad_left
    region = decoded[:, pt:pt + geom.source_h, pl:pl + geom.source_w]
    full = w == 1
    if bool(full.all()):
        region[...] = source
        return decoded
    band = ~full & (w > 0)
    w3 = w[:, :, None]
    for s in range(0, source.shape[0], _COMPOSITE_CHUNK):
        e = min(s + _COMPOSITE_CHUNK, source.shape[0])
        dec = region[s:e]
        src = source[s:e]
        if bool(band.any()):
            mixed = w3 * src.astype(np.float32) + (np.float32(1.0) - w3) * dec.astype(np.float32)
            mixed = np.clip(np.floor(mixed + np.float32(0.5)), 0, 255).astype(np.uint8)
            dec[:, band] = mixed[:, band]
        dec[:, full] = src[:, full]
    return decoded
