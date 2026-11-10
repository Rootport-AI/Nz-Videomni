"""The chain's input video tail, read on the CPU (video-to-video, 台帳 §1-121).

The app has already cut the input video's last ``context_frames`` frames into
their own mp4 at 24 fps with ffmpeg (``video_io.cut_tail_mp4``, the LTX
chain's path), audio included when the input has any. Here the first video
stream is decoded with PyAV, every frame is put onto the generation canvas
with the key frames' rule (:func:`engine_h3.image_cond.cover_crop`: cover,
then centre crop) and the count is checked; :func:`has_audio_stream` tells the
caller whether to read the audio (``engine_h3.source_audio`` refuses a file
without one). Both run before any model is loaded.

Outpainting (台帳 §1-125) reads its cut source video with
:func:`read_source_frames`: the same read at the file's own size (no canvas
step), so the decoded pixels can be pasted back after the decode.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from engine_h3.image_cond import cover_crop


def _decode_frames(path: str, frames: int, label: str, convert) -> tuple[list[np.ndarray], list[int] | None, int]:
    """The shared PyAV read of :func:`read_source_tail` / :func:`read_source_frames`.

    Decodes the first video stream of ``path`` in order and passes each frame
    (PyAV ``VideoFrame``) to ``convert`` -> uint8 ``[H, W, 3]``. Stops one frame
    past ``frames`` (the caller's count check refuses it). Returns
    ``(arrays, source_size [w, h] of the first frame, frames_read)``. ``label``
    names the file in the Japanese errors (``ValueError``) for a missing file,
    no video, or a decode failure; the count is checked by the caller.
    """
    import av

    p = Path(path)
    if not p.is_file():
        raise ValueError(f"{label}が見つかりません: {path}")
    out: list[np.ndarray] = []
    source_size: list[int] | None = None
    read = 0
    try:
        with av.open(str(p)) as container:
            if not container.streams.video:
                raise ValueError(f"{label}に映像がありません: {path}")
            stream = container.streams.video[0]
            for frame in container.decode(stream):
                read += 1
                if read > frames:
                    break   # already too many: the count check refuses it
                if source_size is None:
                    source_size = [int(frame.width), int(frame.height)]
                out.append(convert(frame))
    except (OSError, av.error.FFmpegError) as exc:   # FFmpeg errors can also be ValueErrors: caught first
        raise ValueError(f"{label}を読めません（{path}）: {exc}") from exc
    return out, source_size, read


def _check_count(read: int, frames: int, label: str, path: str) -> None:
    """Refuse (Japanese ``ValueError``) unless exactly ``frames`` frames were read."""
    if read != frames:
        more = "以上" if read > frames else ""
        raise ValueError(
            f"{label}のフレーム数が合いません（{read} フレーム{more}。"
            f"ちょうど {frames} フレームが必要です）: {path}"
        )


def read_source_tail(path: str, width: int, height: int, frames: int) -> tuple[np.ndarray, dict]:
    """Decode ``path`` -> ``(uint8 [frames, height, width, 3], info)``.

    Every decoded frame of the first video stream is used, in order (the file
    is already exactly ``frames`` frames at 24 fps). Raises ``ValueError``
    (Japanese, it reaches the job's ``error``) when the file is missing, has no
    video, cannot be decoded or does not hold exactly ``frames`` frames.
    ``info``: ``source_size`` [w, h] (the first frame's) and ``frames_read``.
    """
    width, height, frames = int(width), int(height), int(frames)
    label = "入力動画の切り出し"
    out, source_size, read = _decode_frames(
        path, frames, label,
        lambda frame: np.asarray(cover_crop(frame.to_image(), width, height), dtype=np.uint8),
    )
    _check_count(read, frames, label, path)
    return np.stack(out), {"source_size": source_size, "frames_read": read}


def read_source_frames(path: str, frames: int) -> np.ndarray:
    """Decode ``path`` at its own size -> uint8 ``[frames, H_s, W_s, 3]`` (Outpainting, 台帳 §1-125).

    The same PyAV read as :func:`read_source_tail` without the canvas step: no
    resize and no crop, so the pixels are the cut mp4's decoded pixels (the
    paste-back after the decode relies on this; 設計正本 ``MINIMAX_H3_DESIGN.md``
    §3.4・計画 §2.3). The app has already cut the source video to exactly
    ``frames`` frames at 24 fps (``video_io.cut_window_mp4``). Raises
    ``ValueError`` (Japanese) when the file is missing, has no video, cannot be
    decoded, does not hold exactly ``frames`` frames, or a frame's size differs
    from the first frame's.
    元の寸法のまま読む（拡縮も切り抜きもしない）。寸法の照合（キャンバス − pad）は呼び出し側。
    """
    frames = int(frames)
    label = "元動画の切り出し"
    out, source_size, read = _decode_frames(
        path, frames, label, lambda frame: frame.to_ndarray(format="rgb24"),
    )
    _check_count(read, frames, label, path)
    for i, arr in enumerate(out):
        if [int(arr.shape[1]), int(arr.shape[0])] != source_size:
            raise ValueError(
                f"{label}のフレームの寸法がそろっていません（{i + 1} フレーム目が "
                f"{arr.shape[1]}×{arr.shape[0]}、1 フレーム目は {source_size[0]}×{source_size[1]}）: {path}"
            )
    return np.stack(out).astype(np.uint8, copy=False)


def has_audio_stream(path: str) -> bool:
    """Whether ``path`` has an audio stream (PyAV). Raises ``ValueError`` (Japanese)
    when the file cannot be opened."""
    import av

    try:
        with av.open(str(path)) as container:
            return bool(container.streams.audio)
    except (OSError, av.error.FFmpegError) as exc:
        raise ValueError(f"入力動画の切り出しを開けませんでした（{path}）: {exc}") from exc
