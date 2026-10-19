"""The I2V key frame: an image file put onto the generation canvas (MINIMAX_H3_DESIGN §3).

One rule for the user, shared with LTX: the image is scaled to COVER the
canvas with its aspect ratio kept, then the centre is cut out (never a
stretch, never letterboxing). The geometry -- ``scale = max(H/h, W/w)``, the
sizes rounded up with ``ceil``, the offsets ``(new - target) // 2`` -- is the
LTX pipelines' ``resize_and_center_crop``. The interpolation is PIL's LANCZOS,
which all three MiniMax H3 references use (DiffSynth, diffusers, ComfyUI); LTX's
one-frame H.264 re-compression is an LTX-specific step and is not done here.

The cropped canvas image is what both consumers see, for different key
frames: the video VAE encoder takes every key frame (as ``[1, 3, 1, H, W]``
float32 in [0, 1]; the ImageNet normalisation happens inside the encoder),
and the vision tower of the text encoder takes only the key frames at frame 0
and F-1 (as the PIL image itself, through the Qwen3-VL image processor); see
MINIMAX_H3_DESIGN §3.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps


def load_canvas_image(path: str, width: int, height: int) -> tuple[Image.Image, tuple[int, int]]:
    """Open ``path``; return ``(RGB image of exactly width x height, (source width, source height))``.

    Cover + centre crop (module docstring). The source size is only reported.

    Raises ``ValueError`` (Japanese, it reaches the job's ``error``) when the
    file is missing or is not an image PIL can read.
    """
    width, height = int(width), int(height)
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"キーフレームの画像が見つかりません: {path}")
    try:
        with Image.open(p) as opened:
            # The upload store applies the same two steps (EXIF orientation, RGB);
            # repeating them makes a hand-picked selftest image behave the same.
            img = ImageOps.exif_transpose(opened).convert("RGB")
    except (OSError, ValueError) as exc:
        raise ValueError(f"キーフレームの画像を読めません（{path}）: {exc}") from exc
    src_w, src_h = img.size
    if src_w < 1 or src_h < 1:
        raise ValueError(f"キーフレームの画像の大きさが不正です（{src_w}×{src_h}）: {path}")
    scale = max(height / src_h, width / src_w)
    new_h = math.ceil(src_h * scale)
    new_w = math.ceil(src_w * scale)
    if (new_w, new_h) != (src_w, src_h):
        img = img.resize((new_w, new_h), Image.LANCZOS)
    top = (new_h - height) // 2
    left = (new_w - width) // 2
    return img.crop((left, top, left + width, top + height)), (src_w, src_h)


def canvas_to_tensor(img: Image.Image) -> torch.Tensor:
    """RGB canvas image -> ``[1, 3, 1, H, W]`` float32 in [0, 1] (uint8 / 255, as the references do)."""
    arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
    x = torch.from_numpy(arr.copy()).to(torch.float32) / 255.0      # [H, W, 3]
    return x.permute(2, 0, 1)[None, :, None]                         # [1, 3, 1, H, W]
