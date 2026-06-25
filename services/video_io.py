"""Video encoding (ffmpeg) and metadata persistence (spec ch.10).

The LTX runner hands us decoded frames (PIL images); this module is the single
place that shells out to ffmpeg to produce ``output.mp4`` with an optional
centered crop, and writes ``metadata.json``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


class FFmpegError(RuntimeError):
    pass


def ffmpeg_path() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise FFmpegError("ffmpeg not found on PATH. Install ffmpeg and add it to PATH.")
    return exe


def encode_frames_to_mp4(
    frames: Iterable[Image.Image],
    output_path: Path,
    frame_rate: float,
    crop: tuple[int, int] | None = None,
    *,
    keep_raw: bool = False,
    raw_dir: Path | None = None,
) -> Path:
    """Encode PIL frames into an H.264 MP4.

    ``crop`` is a ``(width, height)`` target applied as a centered crop via the
    ffmpeg ``crop`` filter (spec 10.3). Returns ``output_path``.
    """
    exe = ffmpeg_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames = list(frames)
    if not frames:
        raise FFmpegError("no frames to encode")

    tmp_dir = Path(raw_dir) if (keep_raw and raw_dir) else Path(tempfile.mkdtemp(prefix="ltx_frames_"))
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        for i, frame in enumerate(frames):
            img = frame.convert("RGB") if frame.mode != "RGB" else frame
            img.save(tmp_dir / f"{i:06d}.png")

        vf_filters = []
        if crop is not None:
            cw, ch = crop
            vf_filters.append(f"crop={cw}:{ch}:(in_w-{cw})/2:(in_h-{ch})/2")

        cmd = [
            exe,
            "-y",
            "-framerate",
            str(frame_rate),
            "-i",
            str(tmp_dir / "%06d.png"),
        ]
        if vf_filters:
            cmd += ["-vf", ",".join(vf_filters)]
        cmd += [
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(frame_rate),
            str(output_path),
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise FFmpegError(f"ffmpeg failed (code {proc.returncode}): {proc.stderr[-2000:]}")
        return output_path
    finally:
        if not keep_raw:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def probe_resolution(path: Path) -> tuple[int, int] | None:
    """Return (width, height) of the first video stream, via ffprobe if present."""
    exe = shutil.which("ffprobe")
    if not exe:
        return None
    cmd = [
        exe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "json",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return None
    try:
        data = json.loads(proc.stdout)
        stream = data["streams"][0]
        return int(stream["width"]), int(stream["height"])
    except Exception:
        return None


def save_metadata(path: Path, metadata: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2)
    return path
