"""Ref2VA reference media: the image / video readers and the Qwen-VL inputs (台帳 §1-107, plan §3.1).

Everything here runs on the CPU and is tested there (tests/test_h3_reference_media.py).
The behaviour follows ComfyUI's MiniMax H3 reference node as a specification
(sizes, truncation, the 2-frames-per-second sampling and its time labels); no
ComfyUI code is copied (GPL-3.0). The patch flattening of :func:`pair_patches`
is ported from Hugging Face transformers (Apache-2.0),
``models/qwen2_vl/image_processing_qwen2_vl.py`` ``Qwen2VLImageProcessor._preprocess``
and ``video_processing_qwen2_vl.py`` (the temporal axis), transformers 5.14.1.
Copyright the Hugging Face team, licensed under the Apache License, Version 2.0
(http://www.apache.org/licenses/LICENSE-2.0). Changes: one pair of frames at a
time, no resize (the caller already sized the frames to multiples of 32).

Sizes
-----
* reference image (``match``): ``scale = min(1, sqrt(W*H / (w*h)))`` with the
  aspect ratio kept, each side rounded to the NEAREST multiple of 32 (Python
  ``round``; at least 32). The scale never enlarges; the rounding can move a
  side by at most 16 px either way, as in ComfyUI.
* reference video: the canvas with a 768 px short edge, its area capped at
  768 x 1344, each side rounded to the nearest multiple of 32; when the source
  has fewer pixels than that canvas, the source size rounded to multiples of 32
  is used instead (no enlargement beyond that rounding).

Reference audio
---------------
No function here: the pipeline reads it with
``engine_h3.source_audio.load_source_waveform(path, keep_samples=1,
read_samples=int(min(max_sec, REF_CLIP_MAX_SEC) * 32000))`` (``keep_samples=1``
because that reader refuses files shorter than ``keep_samples``).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from engine_h3.image_cond import canvas_to_tensor

#: One reference clip is cut to at most this many seconds (the official 15 s limit;
#: the audio reference's cap -- the video reference uses REF_VIDEO_MAX_FRAMES).
REF_CLIP_MAX_SEC = 15.0
#: The video reference's frame cap: the official 15 s clip on the 17k + 5 grid
#: (17 x 21 + 5). 15 s x 24 = 360 would round down to 345 (14.4 s).
REF_VIDEO_MAX_FRAMES = 362
#: The reference-video canvas: 768 px short edge, area at most 768 x 1344.
REF_VIDEO_SHORT_EDGE = 768
REF_VIDEO_MAX_AREA = 768 * 1344
#: The text encoder sees every 12th frame of a reference video (2 per second at 24 fps).
REF_TE_FRAME_STRIDE = 12
#: Reference videos are resampled to this frame rate.
REF_FPS = 24
#: Every side is a multiple of this (VAE 16 x Qwen-VL merge 2).
CANVAS_MULTIPLE = 32
#: The VAE takes 17k + 5 frames; fewer than 5 cannot be encoded.
MIN_VIDEO_FRAMES = 5
#: The Qwen3-VL image processor's normalisation, as the product's tokenizer folder
#: (``models/MiniMaxH3/TextEncoder/qwen3vl_tokenizer/preprocessor_config.json``)
#: sets it: ``image_mean`` / ``image_std`` 0.5; ``rescale_factor`` is not in the
#: file, so the class default 1/255 applies. The CPU test compares with the
#: processor built from that folder.
QWEN_IMAGE_MEAN = (0.5, 0.5, 0.5)
QWEN_IMAGE_STD = (0.5, 0.5, 0.5)
QWEN_RESCALE_FACTOR = 1 / 255
#: Tolerance (seconds) when comparing a frame time with a 1/24 s sampling time.
_TIME_EPS = 1e-4


def _round32(x: float) -> int:
    return max(CANVAS_MULTIPLE, round(x / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)


def match_size(w: int, h: int, W: int, H: int) -> tuple[int, int]:
    """Reference-image size for a ``W x H`` generation (``match``; module docstring)."""
    w, h, W, H = int(w), int(h), int(W), int(H)
    if min(w, h, W, H) < 1:
        raise ValueError(f"大きさが不正です（参照 {w}×{h}・生成 {W}×{H}）")
    scale = min(1.0, math.sqrt((W * H) / (w * h)))
    return _round32(w * scale), _round32(h * scale)


def video_canvas(w: int, h: int) -> tuple[int, int]:
    """Reference-video size for a ``w x h`` source (module docstring)."""
    w, h = int(w), int(h)
    if min(w, h) < 1:
        raise ValueError(f"参照動画の大きさが不正です（{w}×{h}）")
    ratio = w / h
    if ratio >= 1.0:
        nom_w, nom_h = REF_VIDEO_SHORT_EDGE * ratio, float(REF_VIDEO_SHORT_EDGE)
    else:
        nom_w, nom_h = float(REF_VIDEO_SHORT_EDGE), REF_VIDEO_SHORT_EDGE / ratio
    if nom_w * nom_h > REF_VIDEO_MAX_AREA:
        s = math.sqrt(REF_VIDEO_MAX_AREA / (nom_w * nom_h))
        nom_w, nom_h = nom_w * s, nom_h * s
    cw, ch = _round32(nom_w), _round32(nom_h)
    if w * h < cw * ch:
        return _round32(w), _round32(h)
    return cw, ch


def video_frame_count(n: int) -> int:
    """The largest 17k + 5 that is <= ``n`` (0 when ``n`` < 5)."""
    n = int(n)
    if n < MIN_VIDEO_FRAMES:
        return 0
    return n - (n - MIN_VIDEO_FRAMES) % 17


def frames_for_seconds(sec: float) -> int:
    """The smallest 17k + 5 that is >= ``round(sec * 24)`` (5 s -> 124, 10 s -> 243, 15 s -> 362)."""
    n = round(float(sec) * REF_FPS)
    if n <= MIN_VIDEO_FRAMES:
        return MIN_VIDEO_FRAMES
    return MIN_VIDEO_FRAMES + -(-(n - MIN_VIDEO_FRAMES) // 17) * 17


def load_reference_image(path: str, W: int, H: int) -> tuple[torch.Tensor, dict]:
    """Open a reference image for a ``W x H`` generation.

    Returns ``([1, 3, 1, h, w] float32 in [0, 1], {"source_size": [w0, h0], "size": [w, h]})``
    with ``(w, h) = match_size(w0, h0, W, H)``. Opened as
    ``engine_h3.image_cond.load_canvas_image`` opens a key frame (EXIF
    orientation applied, RGB), resized with LANCZOS (no crop). The tensor is
    ``uint8 / 255``, so ``round(x * 255)`` gives the image back exactly (the text
    encoder takes it that way). Raises ``ValueError`` with a Japanese reason.
    """
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"参照画像が見つかりません: {path}")
    try:
        with Image.open(p) as opened:
            img = ImageOps.exif_transpose(opened).convert("RGB")
    except (OSError, ValueError) as exc:
        raise ValueError(f"参照画像を読めません（{path}）: {exc}") from exc
    src_w, src_h = img.size
    if src_w < 1 or src_h < 1:
        raise ValueError(f"参照画像の大きさが不正です（{src_w}×{src_h}）: {path}")
    w, h = match_size(src_w, src_h, W, H)
    if (w, h) != (src_w, src_h):
        img = img.resize((w, h), Image.LANCZOS)
    return canvas_to_tensor(img), {"source_size": [src_w, src_h], "size": [w, h]}


def _frame_seconds(frame, index: int, rate: float) -> float:
    t = frame.time
    return float(t) if t is not None else index / rate


def read_reference_video(path: str, *, num_frames: int, max_sec: float | None = None) -> tuple[np.ndarray, dict]:
    """Read a reference video as ``uint8 [F, h, w, 3]`` at 24 fps on the :func:`video_canvas`.

    Timing: times are measured from the FIRST decoded frame's time
    (``frame.time``), so a file whose first PTS is not 0 keeps its first frame.
    Sample ``k`` (time ``k/24`` s) takes the last frame whose time is at or
    before ``k/24`` (the frame on screen at that moment). Samples run until the
    end of the last frame (its time plus its duration, or 1 / average rate).

    Length: ``F`` is ``min(samples read, num_frames, frames_for_seconds(max_sec)
    when given, REF_VIDEO_MAX_FRAMES)`` rounded down to 17k + 5. The cap is 362, not
    15 s * 24 = 360, because 360 would round down to 345 (14.4 s). ``max_sec`` is
    the user's budget, so it is rounded UP to the 17k + 5 grid (at most 16 frames
    more), the same idea as 15 s -> 362. Fewer than 5 is refused with a Japanese
    ``ValueError``.

    Not handled: rotation metadata (display matrix) is ignored; the frames are
    used as stored. Resize: PIL LANCZOS from the decoded RGB frame. Only the
    first video stream is read; the audio of the file is not used.

    Info: ``source_size`` [w, h], ``size`` [w, h], ``source_fps`` (average rate
    or None), ``source_frames`` (24 fps samples read, before the 17k + 5 cut),
    ``used_frames`` (F).
    """
    import av

    p = Path(path)
    if not p.is_file():
        raise ValueError(f"参照動画が見つかりません: {path}")
    cap = min(int(num_frames), int(REF_VIDEO_MAX_FRAMES))
    if max_sec is not None:
        cap = min(cap, frames_for_seconds(max_sec))
    samples: list[np.ndarray] = []
    try:
        with av.open(str(p)) as container:
            if not container.streams.video:
                raise ValueError(f"参照動画に映像がありません: {path}")
            stream = container.streams.video[0]
            source_fps = float(stream.average_rate) if stream.average_rate else None
            rate = source_fps or float(REF_FPS)
            size: tuple[int, int] | None = None
            source_size: tuple[int, int] | None = None
            t0: float | None = None
            prev = None             # the last decoded frame (av.VideoFrame)
            prev_rel = 0.0
            prev_arr: np.ndarray | None = None

            def convert(frame) -> np.ndarray:
                img = frame.to_image()   # RGB PIL image
                if img.size != size:
                    img = img.resize(size, Image.LANCZOS)
                return np.asarray(img, dtype=np.uint8)

            for index, frame in enumerate(container.decode(stream)):
                if size is None:
                    source_size = (int(frame.width), int(frame.height))
                    size = video_canvas(*source_size)
                rel = _frame_seconds(frame, index, rate)
                if t0 is None:
                    t0 = rel
                rel -= t0
                # Every sample before this frame's time shows the previous frame.
                while prev is not None and len(samples) < cap and rel > len(samples) / REF_FPS + _TIME_EPS:
                    if prev_arr is None:
                        prev_arr = convert(prev)
                    samples.append(prev_arr)
                if len(samples) >= cap:
                    break
                prev, prev_rel, prev_arr = frame, rel, None
            if prev is not None and len(samples) < cap:
                if prev.duration and prev.time_base:
                    dur = float(prev.duration * prev.time_base)
                else:
                    dur = 1.0 / rate
                end = prev_rel + dur
                while len(samples) < cap and len(samples) / REF_FPS < end - _TIME_EPS:
                    if prev_arr is None:
                        prev_arr = convert(prev)
                    samples.append(prev_arr)
    except (OSError, av.error.FFmpegError) as exc:   # FFmpeg errors can also be ValueErrors: caught first
        raise ValueError(f"参照動画を読めません（{path}）: {exc}") from exc
    used = video_frame_count(len(samples))
    if used < MIN_VIDEO_FRAMES:
        raise ValueError(
            f"参照動画が短すぎます（24fps で {len(samples)} フレーム。5 フレーム（約 0.2 秒）以上が必要です）: {path}"
        )
    frames = np.stack(samples[:used])
    info = {
        "source_size": list(source_size),
        "size": list(size),
        "source_fps": source_fps,
        "source_frames": len(samples),
        "used_frames": used,
    }
    return frames, info


def qwen_pairs(frames: np.ndarray) -> list[tuple[np.ndarray, str]]:
    """The text encoder's view of a reference video: ``[(pair uint8 [2, h, w, 3], label)]``.

    Every 12th frame from the first (indices 0, 12, 24, ...); the i-th picked
    frame stands for ``i / 2`` seconds. An odd count repeats the last picked
    frame (with its time). Pairs in order; the label is the pair's mean time
    as ``"<%.1f seconds>"`` (e.g. 362 frames -> 31 picked -> 32 -> 16 pairs
    labelled 0.2, 1.2, ..., 14.2, 15.0).
    """
    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.shape[0] < 1:
        raise ValueError(f"qwen_pairs: [F, h, w, 3] を期待しましたが {frames.shape} でした")
    picked = list(range(0, frames.shape[0], REF_TE_FRAME_STRIDE))
    times = [i / 2.0 for i in range(len(picked))]
    if len(picked) % 2:
        picked.append(picked[-1])
        times.append(times[-1])
    out = []
    for j in range(0, len(picked), 2):
        pair = np.stack([frames[picked[j]], frames[picked[j + 1]]])
        ts = (times[j] + times[j + 1]) / 2.0
        out.append((pair, "<%.1f seconds>" % ts))
    return out


def pair_patches(pair: np.ndarray, *, patch: int = 16, merge: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """Two frames -> one Qwen-VL temporal patch: ``(patches f32 [gh*gw, 3*2*patch*patch], grid long [[1, gh, gw]])``.

    The same normalisation and flattening as the Hugging Face processor (module
    docstring): ``(x - mean*255) / (std*255)`` per channel, then the patch
    order ``[gh/m, gw/m, m, m, C, T, p, p]``. Two equal frames give exactly
    what ``Qwen2VLImageProcessor`` gives for that frame as an image. The sides
    must already be multiples of ``patch * merge`` (checked, never resized).
    """
    pair = np.asarray(pair)
    if pair.ndim != 4 or pair.shape[0] != 2 or pair.shape[-1] != 3 or pair.dtype != np.uint8:
        raise ValueError(f"pair_patches: uint8 [2, h, w, 3] を期待しましたが {pair.dtype} {pair.shape} でした")
    _, h, w, c = pair.shape
    unit = patch * merge
    if h % unit or w % unit:
        raise ValueError(f"pair_patches: 高さと幅は {unit} の倍数が必要です（{h}×{w}）")
    x = torch.from_numpy(np.ascontiguousarray(pair)).permute(0, 3, 1, 2).to(torch.float32)   # [T, C, H, W]
    mean = torch.tensor(QWEN_IMAGE_MEAN) * (1.0 / QWEN_RESCALE_FACTOR)
    std = torch.tensor(QWEN_IMAGE_STD) * (1.0 / QWEN_RESCALE_FACTOR)
    x = (x - mean.view(1, c, 1, 1)) / std.view(1, c, 1, 1)
    gh, gw = h // patch, w // patch
    x = x.reshape(2, c, gh // merge, merge, patch, gw // merge, merge, patch)
    x = x.permute(2, 5, 3, 6, 1, 0, 4, 7)          # [gh/m, gw/m, m, m, C, T, p, p]
    patches = x.reshape(gh * gw, c * 2 * patch * patch)
    grid = torch.tensor([[1, gh, gw]], dtype=torch.long)
    return patches, grid
