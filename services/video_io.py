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


def crop_mp4(input_path: Path, output_path: Path, width: int, height: int) -> Path:
    """Center-crop an existing MP4 to ``width`` x ``height`` via ffmpeg.

    Used by the real LTX backend: the pipeline encodes at the (multiple-of-64)
    generation size, then this re-encodes a centered crop to the requested final
    display size. Reuses the same crop filter as :func:`encode_frames_to_mp4`.
    Returns ``output_path``.
    """
    exe = ffmpeg_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    vf = f"crop={width}:{height}:(in_w-{width})/2:(in_h-{height})/2"
    cmd = [
        exe,
        "-y",
        "-i",
        str(input_path),
        "-vf",
        vf,
        "-map",
        "0:v",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        str(output_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise FFmpegError(f"ffmpeg crop failed (code {proc.returncode}): {proc.stderr[-2000:]}")
    return output_path


def has_audio_stream(path: Path) -> bool:
    """True if ``path`` has at least one audio stream (via ffprobe)."""
    exe = shutil.which("ffprobe")
    if not exe:
        return False
    cmd = [
        exe,
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index",
        "-of",
        "csv=p=0",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode == 0 and proc.stdout.strip() != ""


def probe_duration(path: Path) -> float | None:
    """Return the container duration in seconds (via ffprobe), or None."""
    exe = shutil.which("ffprobe")
    if not exe:
        return None
    cmd = [
        exe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return None
    try:
        return float(json.loads(proc.stdout)["format"]["duration"])
    except Exception:
        return None


def concat_mp4s(
    clip_paths: list[Path],
    output_path: Path,
    frame_rate: float,
    *,
    trim_leading_pixels_per_nonfirst_clip: int = 0,
    crop: tuple[int, int] | None = None,
) -> Path:
    """Concatenate clip mp4s into one continuous timeline (Phase 3 clip-concat).

    Each clip after the first has its leading ``trim_leading_pixels_per_nonfirst_clip``
    PIXEL frames dropped — those frames are the frozen overlap (the previous
    clip's tail re-used at this clip's head), so dropping them yields a seamless
    continuation. The trim count is derived from the K latent overlap frames
    under the causal VAE temporal mapping (latent frame 0 -> 1 pixel, each
    subsequent latent frame -> 8 pixels): ``1 + (K-1)*8``.

    Audio is carried through when every clip has an audio stream. ``crop`` applies
    a centered crop to the final output (applied once, here). Robust to a single
    clip (trim/concat is a no-op copy+optional crop). Re-encodes (fine for the
    single-user local server). Returns ``output_path``.
    """
    exe = ffmpeg_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not clip_paths:
        raise FFmpegError("no clips to concat")

    with_audio = all(has_audio_stream(p) for p in clip_paths)

    # Build a filter graph: for each input, optionally trim the leading N frames
    # of every non-first clip, then concat the (video[, audio]) segments.
    inputs: list[str] = []
    for p in clip_paths:
        inputs += ["-i", str(p)]

    parts: list[str] = []
    concat_labels: list[str] = []
    n = len(clip_paths)
    for i in range(n):
        vlabel = f"v{i}"
        trim_frames = trim_leading_pixels_per_nonfirst_clip if i > 0 else 0
        if trim_frames > 0:
            # Drop the first ``trim_frames`` frames, then reset PTS so concat sees
            # a contiguous, zero-based timeline for this segment.
            parts.append(
                f"[{i}:v]select='gte(n,{trim_frames})',setpts=PTS-STARTPTS[{vlabel}]"
            )
        else:
            parts.append(f"[{i}:v]setpts=PTS-STARTPTS[{vlabel}]")
        concat_labels.append(f"[{vlabel}]")
        if with_audio:
            alabel = f"a{i}"
            if trim_frames > 0:
                # Trim the matching leading audio by time = frames / fps.
                start_t = trim_frames / float(frame_rate)
                parts.append(
                    f"[{i}:a]atrim=start={start_t:.6f},asetpts=PTS-STARTPTS[{alabel}]"
                )
            else:
                parts.append(f"[{i}:a]asetpts=PTS-STARTPTS[{alabel}]")
            concat_labels.append(f"[{alabel}]")

    concat_v = "".join(concat_labels)
    if with_audio:
        parts.append(f"{concat_v}concat=n={n}:v=1:a=1[cv][ca]")
        vout, aout = "[cv]", "[ca]"
    else:
        parts.append(f"{concat_v}concat=n={n}:v=1:a=0[cv]")
        vout, aout = "[cv]", None

    # Optional final centered crop, chained off the concatenated video.
    if crop is not None:
        cw, ch = crop
        parts.append(f"{vout}crop={cw}:{ch}:(in_w-{cw})/2:(in_h-{ch})/2[cvc]")
        vout = "[cvc]"

    filter_complex = ";".join(parts)

    cmd = [exe, "-y", *inputs, "-filter_complex", filter_complex, "-map", vout]
    if aout is not None:
        cmd += ["-map", aout, "-c:a", "aac"]
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
        raise FFmpegError(f"ffmpeg concat failed (code {proc.returncode}): {proc.stderr[-2000:]}")
    return output_path


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


def frame_count(mp4: Path) -> int:
    """Return the exact number of decoded video frames in ``mp4`` (via ffprobe).

    Counts frames rather than trusting the container's ``nb_frames`` tag (which
    can be absent/approximate for some encoders), so this is safe to use as the
    source of truth for frame-accurate indexing (Phase 3 boundary verification).
    """
    exe = shutil.which("ffprobe")
    if not exe:
        raise FFmpegError("ffprobe not found on PATH. Install ffmpeg and add it to PATH.")
    cmd = [
        exe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames",
        "-of",
        "csv=p=0",
        str(mp4),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise FFmpegError(f"ffprobe frame_count failed (code {proc.returncode}): {proc.stderr[-2000:]}")
    try:
        return int(proc.stdout.strip())
    except ValueError:
        raise FFmpegError(f"ffprobe frame_count: unparsable output {proc.stdout!r}")


def extract_frame_at(mp4: Path, frame_index: int, out_png: Path) -> Path:
    """Extract the exact frame at 0-based ``frame_index`` from ``mp4`` to a PNG.

    Uses the ffmpeg ``select`` filter (frame-number based, not timestamp-based)
    so this is exact-frame accurate even for variable-framerate or short clips
    where seeking by time could land on the wrong frame. Returns ``out_png``.
    """
    exe = ffmpeg_path()
    if frame_index < 0:
        raise FFmpegError(f"frame_index must be >= 0, got {frame_index}")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        exe,
        "-y",
        "-i",
        str(mp4),
        "-vf",
        f"select='eq(n\\,{frame_index})'",
        "-vsync",
        "0",
        "-frames:v",
        "1",
        str(out_png),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise FFmpegError(f"ffmpeg extract_frame_at failed (code {proc.returncode}): {proc.stderr[-2000:]}")
    if not out_png.exists():
        raise FFmpegError(f"ffmpeg extract_frame_at produced no output for frame {frame_index} (out of range?)")
    return out_png


def extract_last_frame(mp4: Path, out_png: Path) -> Path:
    """Extract the last decoded frame of ``mp4`` to a PNG. Returns ``out_png``."""
    n = frame_count(mp4)
    if n <= 0:
        raise FFmpegError(f"extract_last_frame: {mp4} reports {n} frames")
    return extract_frame_at(mp4, n - 1, out_png)


def save_metadata(path: Path, metadata: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2)
    return path
