"""The chain's input video tail, read on the CPU (video-to-video, 台帳 §1-121).

The app has already cut the input video's last ``context_frames`` frames into
their own mp4 at 24 fps with ffmpeg (``video_io.cut_tail_mp4``, the LTX
chain's path), audio included when the input has any. Here the first video
stream is decoded with PyAV, every frame is put onto the generation canvas
with the key frames' rule (:func:`engine_h3.image_cond.cover_crop`: cover,
then centre crop) and the count is checked; :func:`has_audio_stream` tells the
caller whether to read the audio (``engine_h3.source_audio`` refuses a file
without one). Both run before any model is loaded.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from engine_h3.image_cond import cover_crop


def read_source_tail(path: str, width: int, height: int, frames: int) -> tuple[np.ndarray, dict]:
    """Decode ``path`` -> ``(uint8 [frames, height, width, 3], info)``.

    Every decoded frame of the first video stream is used, in order (the file
    is already exactly ``frames`` frames at 24 fps). Raises ``ValueError``
    (Japanese, it reaches the job's ``error``) when the file is missing, has no
    video, cannot be decoded or does not hold exactly ``frames`` frames.
    ``info``: ``source_size`` [w, h] (the first frame's) and ``frames_read``.
    """
    import av

    width, height, frames = int(width), int(height), int(frames)
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"入力動画（末尾の切り出し）が見つかりません: {path}")
    out: list[np.ndarray] = []
    source_size: list[int] | None = None
    read = 0
    try:
        with av.open(str(p)) as container:
            if not container.streams.video:
                raise ValueError(f"入力動画（末尾の切り出し）に映像がありません: {path}")
            stream = container.streams.video[0]
            for frame in container.decode(stream):
                read += 1
                if read > frames:
                    break   # already too many: the count below refuses it
                if source_size is None:
                    source_size = [int(frame.width), int(frame.height)]
                img = frame.to_image()   # RGB PIL image
                out.append(np.asarray(cover_crop(img, width, height), dtype=np.uint8))
    except (OSError, av.error.FFmpegError) as exc:   # FFmpeg errors can also be ValueErrors: caught first
        raise ValueError(f"入力動画（末尾の切り出し）を読めません（{path}）: {exc}") from exc
    if read != frames:
        more = "以上" if read > frames else ""
        raise ValueError(
            f"入力動画（末尾の切り出し）のフレーム数が合いません（{read} フレーム{more}。"
            f"ちょうど {frames} フレームが必要です）: {path}"
        )
    return np.stack(out), {"source_size": source_size, "frames_read": read}


def has_audio_stream(path: str) -> bool:
    """Whether ``path`` has an audio stream (PyAV). Raises ``ValueError`` (Japanese)
    when the file cannot be opened."""
    import av

    try:
        with av.open(str(path)) as container:
            return bool(container.streams.audio)
    except (OSError, av.error.FFmpegError) as exc:
        raise ValueError(f"入力動画（末尾の切り出し）を開けませんでした（{path}）: {exc}") from exc
