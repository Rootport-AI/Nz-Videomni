"""Picture measurements of the Sol experiment (CPU, torch only; no new package).

Every function takes CPU uint8 frames ``[F, H, W, 3]`` (the decoders' output) unless it says
otherwise. Luma is BT.601 (0.299 R + 0.587 G + 0.114 B) on 0..255.

* :func:`psnr_u8` -- PSNR over the whole clip (one MSE) and per frame (min / mean), max |diff|.
* :func:`ssim_luma` -- SSIM of the luma (Gaussian window 11, sigma 1.5, the usual K1 0.01 / K2 0.03),
  averaged over every ``stride``-th frame.
* :func:`grid_edges` -- the strength of vertical and horizontal edges ON a pixel grid of period p
  (the boundary between column x - 1 and x for x = p, 2p, ...) divided by the strength at every
  OTHER boundary: mean |Y[x] - Y[x - 1]| over frames and rows. 1.0 = no grid; a visible 128-pixel
  line raises the p = 128 ratio. Periods 16 (one token), 128 and 256 are reported; for 128 and 256 also
  with a denominator that leaves out the multiples of 16 (``ratio_excluding_16``).
* :func:`flicker` -- the mean |Y[f + 1] - Y[f]| of frame pairs that straddle a native block boundary
  (the last frame of block b and the first of block b + 1) against the pairs inside a block.
* :func:`grid_crops` / :func:`save_sheet` -- crops centred on 128-pixel grid crossings, upscaled
  (nearest) side by side for the eyes.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F

_LUMA = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32)


def luma(frames: torch.Tensor) -> torch.Tensor:
    """uint8 [F, H, W, 3] -> float32 [F, H, W] on 0..255."""
    return (frames.to(torch.float32) * _LUMA).sum(-1)


def psnr_u8(a: torch.Tensor, b: torch.Tensor, chunk: int = 8) -> dict:
    if a.shape != b.shape:
        return {"error": f"shapes differ {list(a.shape)} vs {list(b.shape)}"}
    se_total, n_total, per, mx = 0.0, 0, [], 0
    for f0 in range(0, a.shape[0], chunk):
        d = a[f0:f0 + chunk].to(torch.int16) - b[f0:f0 + chunk].to(torch.int16)
        mx = max(mx, int(d.abs().max()))
        se = (d.to(torch.float64) ** 2).flatten(1).sum(1)
        n = d[0].numel()
        for v in se.tolist():
            mse = v / n
            per.append(100.0 if mse == 0 else 10.0 * math.log10(255.0 ** 2 / mse))
        se_total += float(se.sum())
        n_total += n * d.shape[0]
    mse = se_total / n_total
    return {"psnr_db": round(100.0 if mse == 0 else 10.0 * math.log10(255.0 ** 2 / mse), 3),
            "per_frame_min_db": round(min(per), 3), "per_frame_mean_db": round(sum(per) / len(per), 3),
            "max_abs": mx}


def psnr_float(a: torch.Tensor, b: torch.Tensor, peak: float = 1.0) -> dict:
    """Float tensors of the same shape (the cross-check's [0, 1] pixels)."""
    d = (a.double() - b.double())
    mse = float((d ** 2).mean())
    return {"psnr_db": round(200.0 if mse == 0 else 10.0 * math.log10(peak ** 2 / mse), 3),
            "max_abs": float(d.abs().max()), "mean_abs": float(d.abs().mean())}


def _gauss(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    x = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def ssim_luma(a: torch.Tensor, b: torch.Tensor, stride: int = 1) -> dict:
    if a.shape != b.shape:
        return {"error": f"shapes differ {list(a.shape)} vs {list(b.shape)}"}
    g = _gauss()
    kx, ky = g.view(1, 1, 1, -1), g.view(1, 1, -1, 1)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2

    def blur(x):
        return F.conv2d(F.conv2d(x, kx), ky)

    vals = []
    for f in range(0, a.shape[0], stride):
        x = luma(a[f:f + 1])[:, None]
        y = luma(b[f:f + 1])[:, None]
        mx, my = blur(x), blur(y)
        sxx, syy, sxy = blur(x * x) - mx * mx, blur(y * y) - my * my, blur(x * y) - mx * my
        s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))
        vals.append(float(s.mean()))
    return {"ssim": round(sum(vals) / len(vals), 6), "ssim_min_frame": round(min(vals), 6),
            "frames_used": len(vals), "stride": stride}


def grid_edges(frames: torch.Tensor, periods=(16, 128, 256), stride: int = 1, chunk: int = 8) -> dict:
    Fn, H, W, _ = frames.shape
    gx = torch.zeros(W - 1, dtype=torch.float64)    # boundary between x and x + 1 -> index x
    gy = torch.zeros(H - 1, dtype=torch.float64)
    used = 0
    idx = list(range(0, Fn, stride))
    for c0 in range(0, len(idx), chunk):
        y = luma(frames[idx[c0:c0 + chunk]])
        gx += (y[:, :, 1:] - y[:, :, :-1]).abs().double().sum((0, 1))
        gy += (y[:, 1:, :] - y[:, :-1, :]).abs().double().sum((0, 2))
        used += y.shape[0]
    gx /= used * H
    gy /= used * W
    out = {"frames_used": used, "stride": stride}
    for p in periods:
        row = {}
        for axis, prof in (("x", gx), ("y", gy)):
            pos = torch.arange(1, prof.numel() + 1)            # boundary index i sits at pixel position i + 1
            on = (pos % p) == 0
            if not bool(on.any()):
                row[axis] = None
                continue
            e_on, e_off = float(prof[on].mean()), float(prof[~on].mean())
            row[axis] = {"on": round(e_on, 4), "off": round(e_off, 4), "ratio": round(e_on / e_off, 4) if e_off else None,
                         "lines": int(on.sum())}
            if p != 16:
                # the denominator without the 16-pixel (one token) boundaries, which may carry their own edge
                off16 = ~on & ((pos % 16) != 0)
                e16 = float(prof[off16].mean()) if bool(off16.any()) else None
                row[axis].update(off_excluding_16=None if e16 is None else round(e16, 4),
                                 ratio_excluding_16=round(e_on / e16, 4) if e16 else None)
        out[f"p{p}"] = row
    return out


def flicker(frames: torch.Tensor, block_frames: list[int], chunk: int = 16) -> dict:
    Fn = frames.shape[0]
    diffs = []
    prev = None
    for f0 in range(0, Fn, chunk):
        y = luma(frames[f0:f0 + chunk])
        if prev is not None:
            diffs.append(float((y[0] - prev).abs().mean()))
        diffs += [float(v) for v in (y[1:] - y[:-1]).abs().mean((1, 2))]
        prev = y[-1]
    ends = []
    acc = 0
    for n in block_frames[:-1]:
        acc += n
        ends.append(acc - 1)                                    # pair (acc - 1, acc) straddles the boundary
    bset = set(ends)
    on = [diffs[i] for i in ends if i < len(diffs)]
    off = [d for i, d in enumerate(diffs) if i not in bset]
    m_on = sum(on) / len(on) if on else None
    m_off = sum(off) / len(off) if off else None
    return {"pairs": len(diffs), "boundary_pairs": len(on), "boundary_mean": None if m_on is None else round(m_on, 4),
            "inside_mean": None if m_off is None else round(m_off, 4),
            "ratio": round(m_on / m_off, 4) if m_on is not None and m_off else None,
            "boundary_values": [round(v, 3) for v in on], "boundary_pair_index": ends}


def grid_centres(width: int, height: int, period: int = 128) -> list[tuple[int, int]]:
    """Three 128-pixel grid crossings: the one nearest the centre and those nearest the quarter points."""
    def near(v, lim):
        k = max(1, min(round(v / period), (lim - 1) // period))
        return k * period
    pts = [(near(width / 2, width), near(height / 2, height)), (near(width / 4, width), near(height / 4, height)),
           (near(3 * width / 4, width), near(3 * height / 4, height))]
    out = []
    for p in pts:
        if p not in out:
            out.append(p)
    return out


def grid_crops(frames: torch.Tensor, centres, frame_idx, half: int = 48) -> list:
    H, W = frames.shape[1:3]
    tiles = []
    for f in frame_idx:
        for cx, cy in centres:
            x0, y0 = max(0, cx - half), max(0, cy - half)
            x1, y1 = min(W, cx + half), min(H, cy + half)
            tiles.append(frames[f, y0:y1, x0:x1].contiguous().numpy())
    return tiles


def save_sheet(path: Path, rows: list, col_labels: list, upscale: int = 4) -> str:
    """``rows``: [(label, [uint8 [h, w, 3], ...])], every tile the same size, upscaled (nearest)."""
    from PIL import Image, ImageDraw

    h, w = rows[0][1][0].shape[:2]
    tw, th = w * upscale, h * upscale
    sheet = Image.new("RGB", (len(col_labels) * tw, len(rows) * th), (0, 0, 0))
    draw = ImageDraw.Draw(sheet)
    for r, (label, tiles) in enumerate(rows):
        for c, tile in enumerate(tiles):
            im = Image.fromarray(tile).resize((tw, th), Image.NEAREST)
            sheet.paste(im, (c * tw, r * th))
            text = f"{label} | {col_labels[c]}"
            draw.rectangle([c * tw, r * th, c * tw + 7 * len(text) + 8, r * th + 14], fill=(0, 0, 0))
            draw.text((c * tw + 4, r * th + 2), text, fill=(255, 255, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return str(path)
