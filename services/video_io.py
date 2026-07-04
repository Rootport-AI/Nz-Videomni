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


def probe_fps(path: Path) -> float | None:
    """Return the average frame rate (fps) of the first video stream, via ffprobe.

    Uses ``avg_frame_rate`` (a ``num/den`` rational) which reflects the real
    decoded cadence better than ``r_frame_rate`` for VFR sources. Returns None
    when ffprobe is missing or the value is unavailable/degenerate.
    """
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
        "stream=avg_frame_rate,r_frame_rate",
        "-of",
        "json",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return None
    try:
        stream = json.loads(proc.stdout)["streams"][0]
    except Exception:
        return None
    for key in ("avg_frame_rate", "r_frame_rate"):
        val = stream.get(key)
        if not val or val == "0/0":
            continue
        try:
            if "/" in val:
                num, den = val.split("/", 1)
                den_f = float(den)
                if den_f == 0:
                    continue
                return float(num) / den_f
            return float(val)
        except Exception:
            continue
    return None


def cut_tail_mp4(
    src: Path,
    out: Path,
    context_frames: int,
    fps: float,
) -> dict:
    """Write to ``out`` an mp4 holding EXACTLY the last ``context_frames`` frames
    of ``src`` at ``fps`` (the fps-correct source tail the V2V engine consumes).

    When ``src``'s native fps differs from ``fps`` the whole source is first
    resampled (ffmpeg ``fps`` filter) so the tail cadence matches the request;
    the frame-exact tail is then selected with the ``select`` filter (the same
    frame-number-based selection used by :func:`extract_frame_at`, so it is exact
    even for VFR/short sources where time seeking could miss). Audio, when
    present, is carried through and trimmed to the same tail window (the engine
    freezes/fades the audio head itself). Re-encodes (the VAE re-encodes anyway).

    Returns ``{source_fps, resampled, total_frames}`` where ``total_frames`` is
    the frame count of the (possibly resampled) source. Raises
    :class:`FFmpegError` if the (resampled) source has fewer than
    ``context_frames`` frames (a defensive backstop — the app preflights this).
    """
    exe = ffmpeg_path()
    out.parent.mkdir(parents=True, exist_ok=True)

    source_fps = probe_fps(src)
    resampled = source_fps is not None and abs(source_fps - float(fps)) > 1e-3

    work = src
    tmp_resampled: Path | None = None
    if resampled:
        tmp_resampled = out.parent / (out.stem + "_resampled.mp4")
        rs_cmd = [
            exe, "-y", "-i", str(src),
            "-vf", f"fps={fps}",
            "-map", "0:v", "-map", "0:a?",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(fps),
            "-c:a", "aac",
            str(tmp_resampled),
        ]
        proc = subprocess.run(rs_cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise FFmpegError(f"ffmpeg fps-resample failed (code {proc.returncode}): {proc.stderr[-2000:]}")
        work = tmp_resampled

    try:
        total = frame_count(work)
        if total < context_frames:
            raise FFmpegError(
                f"source has {total} frames (after resample={resampled}) < "
                f"context_frames={context_frames}"
            )
        start = total - context_frames
        with_audio = has_audio_stream(work)

        parts = [f"[0:v]select='gte(n\\,{start})',setpts=PTS-STARTPTS[v]"]
        vout = "[v]"
        aout = None
        if with_audio:
            start_t = start / float(fps)
            parts.append(f"[0:a]atrim=start={start_t},asetpts=PTS-STARTPTS[a]")
            aout = "[a]"
        filter_complex = ";".join(parts)

        cmd = [exe, "-y", "-i", str(work), "-filter_complex", filter_complex, "-map", vout]
        if aout is not None:
            cmd += ["-map", aout, "-c:a", "aac"]
        cmd += [
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(fps),
            str(out),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise FFmpegError(f"ffmpeg tail-cut failed (code {proc.returncode}): {proc.stderr[-2000:]}")
    finally:
        if tmp_resampled is not None:
            tmp_resampled.unlink(missing_ok=True)

    return {"source_fps": source_fps, "resampled": resampled, "total_frames": total}


def concat_mp4s(
    clip_paths: list[Path],
    output_path: Path,
    frame_rate: float,
    *,
    crop: tuple[int, int] | None = None,
) -> Path:
    """Concatenate clip mp4s into one continuous timeline.

    Audio is carried through when every clip has an audio stream. ``crop`` applies
    a centered crop to the final output (applied once, here). Robust to a single
    clip (concat is a no-op copy+optional crop). Re-encodes (fine for the
    single-user local server). Returns ``output_path``.
    """
    exe = ffmpeg_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not clip_paths:
        raise FFmpegError("no clips to concat")

    with_audio = all(has_audio_stream(p) for p in clip_paths)

    # Build a filter graph: reset each segment's PTS, then concat the
    # (video[, audio]) segments.
    inputs: list[str] = []
    for p in clip_paths:
        inputs += ["-i", str(p)]

    parts: list[str] = []
    concat_labels: list[str] = []
    n = len(clip_paths)
    for i in range(n):
        vlabel = f"v{i}"
        parts.append(f"[{i}:v]setpts=PTS-STARTPTS[{vlabel}]")
        concat_labels.append(f"[{vlabel}]")
        if with_audio:
            alabel = f"a{i}"
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
