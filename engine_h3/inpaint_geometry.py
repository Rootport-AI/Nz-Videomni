"""MiniMax H3 Inpainting geometry, token mask and paste-back -- numpy only, torch-free (台帳 §1-124).

The Inpainting job reuses the Outpainting geometry (engine_h3/outpaint_geometry.py):
the source ``W_s x H_s`` is placed at the top-left of a canvas rounded up to the
next multiple of 32 (``pad_left = pad_top = 0``; the right / bottom gaps, 0-31 px,
are the source's edge mirrored by ``aligned_frames``), and the output is cut back
to the source size afterwards.

* Time: the frames are grouped in chunks of 17 (the last one is the remainder);
  each chunk's pixel mask is the union (any) over its frames. Each chunk spans 5
  latent frames, and the last 3 latent frames of the repeated sequence are
  dropped, so ``t_lat = 5 * n_chunks - 3``.
* Space: a 32 x 32 token is generated when at least one of its pixels is white
  in the mask (生成側に倒す). The canvas beyond the source counts as not masked.
* Paste-back: the source pixel's weight is ``w = min(1, d / F)``, ``d`` being the
  exact Euclidean distance to the masked blocks (a pixel touching a masked block
  has ``d = 1``, a pixel inside has ``d = 0``). ``F = 0`` is a hard paste
  (0 inside the mask, 1 outside). ``composite`` (Outpainting) does the blending.
"""

from __future__ import annotations

import numpy as np

from engine_h3.outpaint_geometry import (
    TOKEN_PX,
    OutpaintGeometry,
    composite,
    compute_outpaint_geometry,
)

#: Pixel frames per time chunk (one chunk = 5 latent frames).
CHUNK_FRAMES = 17
#: Latent frames per chunk before the trailing ones are dropped.
CHUNK_LATENTS = 5
#: Latent frames dropped at the end of the repeated sequence.
DROPPED_LATENTS = 3


def _int(value, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"Inpainting: {what} は整数で指定してください（{value!r}）")
    return int(value)


def round_up_32(n: int) -> int:
    """The smallest multiple of 32 that is >= ``n`` (``n`` > 0)."""
    v = _int(n, "寸法")
    if v <= 0:
        raise ValueError(f"Inpainting: 寸法が正ではありません（{v}）")
    return -(-v // TOKEN_PX) * TOKEN_PX


def compute_inpaint_geometry(source_w, source_h, canvas_w, canvas_h) -> OutpaintGeometry:
    """The geometry of an Inpainting job: the source at the top-left of a canvas rounded up to 32.

    Raises ValueError (理由は日本語) unless ``canvas == round_up_32(source)`` on both axes.
    """
    sw, sh = _int(source_w, "source_w"), _int(source_h, "source_h")
    cw, ch = _int(canvas_w, "canvas_w"), _int(canvas_h, "canvas_h")
    if sw <= 0 or sh <= 0:
        raise ValueError(f"Inpainting: 元の映像の寸法が正ではありません（{sw}x{sh}）")
    want_w, want_h = round_up_32(sw), round_up_32(sh)
    if cw != want_w or ch != want_h:
        raise ValueError(
            f"Inpainting: キャンバス {cw}x{ch} が元の映像 {sw}x{sh} を {TOKEN_PX} の倍数へ"
            f"切り上げた寸法（{want_w}x{want_h}）に一致しません")
    return compute_outpaint_geometry(
        sw, sh, cw, ch, 0, cw - sw, 0, ch - sh, require_pad=False)


def chunk_ranges(num_frames: int) -> list[tuple[int, int]]:
    """``[(a, b), ...]``: the frames ``[a, b)`` of each chunk of 17 (the last one is the remainder)."""
    f = _int(num_frames, "フレーム数")
    if f <= 0:
        raise ValueError(f"Inpainting: フレーム数が正ではありません（{f}）")
    return [(a, min(a + CHUNK_FRAMES, f)) for a in range(0, f, CHUNK_FRAMES)]


def block_mask_from_pixels(frame_bool: np.ndarray, geom: OutpaintGeometry) -> np.ndarray:
    """bool ``[H_s, W_s]`` -> bool ``[grid_h, grid_w]``: True where a 32 x 32 block has any True pixel.

    The source is placed on the canvas at ``(pad_top, pad_left)``; the rest of the
    canvas is False.
    """
    m = np.asarray(frame_bool)
    if m.dtype != np.bool_ or m.shape != (geom.source_h, geom.source_w):
        raise ValueError(
            f"Inpainting: マスクは bool [{geom.source_h}, {geom.source_w}] が必要です"
            f"（{list(m.shape)}, {m.dtype}）")
    canvas = np.zeros((geom.canvas_h, geom.canvas_w), dtype=bool)
    pt, pl = geom.pad_top, geom.pad_left
    canvas[pt:pt + geom.source_h, pl:pl + geom.source_w] = m
    return canvas.reshape(geom.grid_h, TOKEN_PX, geom.grid_w, TOKEN_PX).any(axis=(1, 3))


def chunk_block_masks(blocks: np.ndarray, num_frames: int) -> np.ndarray:
    """bool ``[F, gh, gw]`` -> bool ``[n_chunks, gh, gw]``: the union (any) over each chunk's frames."""
    b = np.asarray(blocks)
    f = _int(num_frames, "フレーム数")
    if b.dtype != np.bool_ or b.ndim != 3 or b.shape[0] != f:
        raise ValueError(
            f"Inpainting: ブロックのマスクは bool [{f}, gh, gw] が必要です（{list(b.shape)}, {b.dtype}）")
    return np.stack([b[a:e].any(axis=0) for a, e in chunk_ranges(f)])


def token_values_from_chunks(chunk_blocks: np.ndarray, t_lat: int) -> np.ndarray:
    """bool ``[n_chunks, gh, gw]`` -> float32 ``[t_lat, gh, gw]`` (1.0 = generated, 0.0 = kept).

    Each chunk is repeated 5 times and the last 3 latent frames are dropped, so
    ``t_lat`` must be ``5 * n_chunks - 3``.
    """
    c = np.asarray(chunk_blocks)
    t = _int(t_lat, "t_lat")
    if c.dtype != np.bool_ or c.ndim != 3 or c.shape[0] <= 0:
        raise ValueError(
            f"Inpainting: 塊ごとのマスクは bool [n_chunks, gh, gw] が必要です（{list(c.shape)}, {c.dtype}）")
    want = CHUNK_LATENTS * c.shape[0] - DROPPED_LATENTS
    if t != want:
        raise ValueError(
            f"Inpainting: 潜在フレーム数 {t} が塊の数 {c.shape[0]} から決まる数（{want}）に一致しません")
    rep = np.repeat(c, CHUNK_LATENTS, axis=0)[:t]
    return np.ascontiguousarray(np.where(rep, np.float32(1.0), np.float32(0.0)).astype(np.float32))


def _axis_distance(block_line: np.ndarray, n: int, offset: int) -> np.ndarray:
    """float64 ``[n]``: per pixel ``p`` (source coordinates), the 1-D distance to the masked blocks of one line.

    Block ``j`` covers canvas pixels ``[32j, 32j + 31]``; the source pixel ``p`` is the
    canvas pixel ``p + offset``. Distance ``max(32j - q, 0, q - (32j + 31))``, minimised over
    the masked ``j`` (inf when none is masked).
    """
    js = np.flatnonzero(block_line)
    q = np.arange(n, dtype=np.float64)[:, None] + offset
    if js.size == 0:
        return np.full(n, np.inf)
    lo = (js * TOKEN_PX).astype(np.float64)[None, :]
    d = np.maximum(np.maximum(lo - q, 0.0), q - (lo + (TOKEN_PX - 1)))
    return d.min(axis=1)


def feather_weights_from_mask(
    chunk_block_mask: np.ndarray, geom: OutpaintGeometry, feather_px: int,
) -> np.ndarray:
    """float32 ``[H_s, W_s]``: the source pixel's weight ``w = min(1, d / F)`` in the paste-back.

    ``d``: the exact Euclidean distance (in pixels) to the masked 32 x 32 blocks
    (1 for a pixel touching a masked block, 0 inside). ``F = 0`` -> 0 inside the
    mask, 1 outside (a hard paste). Computed as ``d^2 = min_i (vy_i^2 + hx_i^2)``
    over the block rows ``i``; rows farther than ``F`` from a pixel are skipped
    for it (``d >= F`` there already gives ``w = 1``).
    """
    m = np.asarray(chunk_block_mask)
    f = _int(feather_px, "feather_px")
    if f < 0:
        raise ValueError(f"Inpainting: なじませの幅は 0 以上にしてください（{f}）")
    if m.dtype != np.bool_ or m.shape != (geom.grid_h, geom.grid_w):
        raise ValueError(
            f"Inpainting: 塊のマスクは bool [{geom.grid_h}, {geom.grid_w}] が必要です"
            f"（{list(m.shape)}, {m.dtype}）")
    h, w = geom.source_h, geom.source_w
    pt, pl = geom.pad_top, geom.pad_left
    if f == 0:
        pix = np.repeat(np.repeat(m, TOKEN_PX, axis=0), TOKEN_PX, axis=1)[pt:pt + h, pl:pl + w]
        return np.where(pix, np.float32(0.0), np.float32(1.0)).astype(np.float32)
    cap = float(f) * float(f)
    d2 = np.full((h, w), cap, dtype=np.float64)
    for i in np.flatnonzero(m.any(axis=1)):
        top = int(i) * TOKEN_PX - pt          # block row i in source coordinates: [top, top + 31]
        y_lo = max(0, top - f + 1)
        y_hi = min(h, top + TOKEN_PX + f - 1)
        if y_lo >= y_hi:
            continue
        ys = np.arange(y_lo, y_hi, dtype=np.float64)
        vy = np.maximum(np.maximum(top - ys, 0.0), ys - (top + TOKEN_PX - 1))
        hx = _axis_distance(m[i], w, pl)
        band = vy[:, None] ** 2 + hx[None, :] ** 2
        np.minimum(d2[y_lo:y_hi], band, out=d2[y_lo:y_hi])
    return np.minimum(1.0, np.sqrt(d2) / float(f)).astype(np.float32)


def compose_inpaint(
    decoded: np.ndarray, source: np.ndarray, chunk_blocks: np.ndarray,
    geom: OutpaintGeometry, feather_px: int,
) -> np.ndarray:
    """Paste the source back into the decoded canvas frames IN PLACE, chunk by chunk; return ``decoded``.

    ``decoded``: uint8 ``[F, H_c, W_c, 3]``; ``source``: uint8 ``[F, H_s, W_s, 3]``;
    ``chunk_blocks``: bool ``[n_chunks, gh, gw]`` (``chunk_block_masks``). Each chunk's
    weights come from ``feather_weights_from_mask`` and are applied by ``composite``
    on the chunk's slice (a view, so the writes land in ``decoded``).
    """
    dec = decoded
    if not isinstance(dec, np.ndarray) or dec.ndim != 4 or not isinstance(source, np.ndarray) \
            or source.ndim != 4:
        raise ValueError("Inpainting: 復号した映像と元の映像は [F, H, W, 3] の配列が必要です")
    if dec.shape[0] != source.shape[0]:
        raise ValueError(f"Inpainting: フレーム数が一致しません（復号 {dec.shape[0]}・元 {source.shape[0]}）")
    ranges = chunk_ranges(dec.shape[0])
    c = np.asarray(chunk_blocks)
    if c.ndim != 3 or c.shape[0] != len(ranges):
        raise ValueError(
            f"Inpainting: 塊の数が一致しません（マスク {c.shape[0] if c.ndim else 0}・"
            f"フレーム {dec.shape[0]} 枚から {len(ranges)}）")
    for k, (a, b) in enumerate(ranges):
        w_c = feather_weights_from_mask(c[k], geom, feather_px)
        composite(dec[a:b], source[a:b], w_c, geom)
    return decoded
