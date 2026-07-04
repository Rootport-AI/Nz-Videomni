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


def probe_audio_stream(path: Path) -> tuple[int | None, int | None]:
    """Return ``(sample_rate, channels)`` of the first audio stream via ffprobe.

    Either value may be None when ffprobe is missing, the file has no audio
    stream, or the field is unavailable. Used by the A2V mock backend to report a
    faithful ``chain.a2v`` sample rate / channel count without a real audio decode.
    """
    exe = shutil.which("ffprobe")
    if not exe:
        return None, None
    cmd = [
        exe,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate,channels",
        "-of",
        "json",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return None, None
    try:
        streams = json.loads(proc.stdout).get("streams", [])
        if not streams:
            return None, None
        stream = streams[0]
        sr = int(stream["sample_rate"]) if stream.get("sample_rate") else None
        ch = int(stream["channels"]) if stream.get("channels") is not None else None
        return sr, ch
    except Exception:
        return None, None


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


def _loudnorm_measure(exe: str, path: Path) -> dict[str, float]:
    """Run ffmpeg's ``loudnorm`` filter in pass-1 (measure-only) mode and parse
    the JSON stats block it writes to stderr. Used by :func:`join_v2v` to read
    the actual integrated loudness (LUFS) of an existing audio track without
    modifying it (spec ch.10 V2V join: "measure before you touch anything").

    The ``I``/``TP``/``LRA`` target values passed here only affect the
    *analysis* thresholds, not the reported ``input_i`` (the file's own
    measured loudness) -- neutral defaults are used since this call never
    writes output.
    """
    cmd = [
        exe,
        "-hide_banner",
        "-nostats",
        "-i",
        str(path),
        "-vn",
        "-af",
        "loudnorm=I=-23:TP=-2:LRA=7:print_format=json",
        "-f",
        "null",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise FFmpegError(f"ffmpeg loudnorm measure failed (code {proc.returncode}): {proc.stderr[-2000:]}")
    text = proc.stderr
    start, end = text.rfind("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise FFmpegError(f"ffmpeg loudnorm measure: could not parse JSON stats from stderr: {text[-1000:]}")
    try:
        return json.loads(text[start : end + 1])
    except Exception as exc:
        raise FFmpegError(f"ffmpeg loudnorm measure: bad JSON stats: {exc}") from exc


def join_v2v(
    source: Path,
    continuation: Path,
    out: Path,
    *,
    audio_fade_ms: int = 400,
    loudness_match: bool = True,
    handle_audio: Path | None = None,
    handle_crossfade_ms: int = 300,
) -> dict:
    """Join a real ``source`` clip to a generated ``continuation`` clip (V2V).

    Unlike :func:`concat_mp4s` (used for same-generation clip chains, where
    segments share one continuous vocoder-audio stream), the two inputs here
    come from independent audio sources -- a real source recording and a
    freshly generated (vocoder) continuation with a different noise floor.
    Research (R0) confirmed a true overlapped ``acrossfade`` is not possible:
    neither file has audio *past* the visual join point, and shrinking either
    track by the fade duration would desync audio from video. So this
    implements the no-handle standard instead:

    - Video: a hard cut, concatenated the same way as :func:`concat_mp4s`
      (re-encoded; ``source`` and ``continuation`` must share resolution and
      fps -- raises :class:`FFmpegError` otherwise).
    - Audio: fade the source's last ``audio_fade_ms`` out and the
      continuation's first ``audio_fade_ms`` in (both ``curve=qsin``, the
      "equal-power" crossfade curve), so the discontinuity is masked by
      silence at the seam rather than heard as a hard cut. Total duration is
      preserved (no audio/video trimming -- only fades, which do not change
      length).
    - Loudness: when ``loudness_match`` is True and both clips have audio, the
      source's integrated loudness is measured (two-pass ``loudnorm``,
      pass-1 JSON) and the continuation's audio is normalized toward it
      *before* fading, so the fade blends between two clips at matched
      levels rather than masking a volume jump. The source's audio is never
      altered except for its fade-out.

    If either input lacks an audio stream, no audio treatment is applied at
    all (mirrors :func:`concat_mp4s`'s fallback): the output is video-only,
    matching the "both must have audio" gate that function also uses.

    ``audio_fade_ms`` is clamped to the shorter of the two clips' audio
    durations if it would otherwise exceed one side's available audio.

    Writes the joined video to ``out``. Returns a small dict for
    metadata/logging: ``{source_lufs, continuation_lufs_before,
    fade_ms_applied, loudness_matched}`` (``source_lufs`` /
    ``continuation_lufs_before`` are ``None`` when loudness was not measured).

    HANDLE TRUE-CROSSFADE mode (opt-in, ``handle_audio`` given). The engine can
    now emit a sidecar wav (``<stem>_audio_handle.wav``) holding the FULL
    untrimmed timeline audio — the pre-junction *context* region (which BOTH the
    real source recording and this vocoder render depict) plus the continuation
    region, in one sample-continuous stream. Given that handle, this does a real
    overlapped equal-power crossfade instead of the no-overlap fade pair (which
    leaves an energy valley at the seam):

    - Video: the SAME hard concat as the default path (unchanged).
    - Audio: ``acrossfade=d=<handle_crossfade_ms>:c1=qsin:c2=qsin`` between the
      real source audio (stream A) and the handle stream (stream B), where B is
      the sidecar trimmed to start at ``handle_context_seconds − crossfade`` so
      the crossfade window sits ENTIRELY BEFORE the junction — both A and B
      depict the same music there. From the junction onward the audio is the
      handle's continuation region: sample-continuous, NO fade at the junction.
      ``handle_context_seconds`` (the junction offset inside the handle) is
      derived as ``handle_duration − continuation_duration`` (the handle = the
      context region + the continuation region, so this is exact given the
      sample geometry). B is loudness-matched to the source (same two-pass
      ``loudnorm``) before the fade.
    - Duration: ``acrossfade`` shrinks the summed stream by ``d``; because A ends
      at the junction and B's first ``d`` seconds are the pre-junction overlap,
      the output audio length works out to ``source_audio + continuation_audio``
      — matching the hard-concatenated video. Asserted within a small tolerance.

    Handle mode requires both inputs to carry audio (``with_audio``); otherwise it
    falls back to the default video-only behavior. When ``handle_audio`` is None
    the behavior is byte-identical to before (the default fade-pair join).
    """
    exe = ffmpeg_path()
    out.parent.mkdir(parents=True, exist_ok=True)

    src_res = probe_resolution(source)
    cont_res = probe_resolution(continuation)
    if src_res is None or cont_res is None or src_res != cont_res:
        raise FFmpegError(
            f"join_v2v: resolution mismatch/unavailable: source={src_res} continuation={cont_res}"
        )

    src_fps = probe_fps(source)
    cont_fps = probe_fps(continuation)
    if src_fps is None or cont_fps is None or abs(src_fps - cont_fps) > 1e-3:
        raise FFmpegError(
            f"join_v2v: fps mismatch/unavailable: source={src_fps} continuation={cont_fps}"
        )

    with_audio = has_audio_stream(source) and has_audio_stream(continuation)
    use_handle = handle_audio is not None and with_audio

    info: dict[str, Any] = {
        "source_lufs": None,
        "continuation_lufs_before": None,
        "fade_ms_applied": 0,
        "loudness_matched": False,
        "join_mode": "handle_crossfade" if use_handle else ("fade_pair" if with_audio else "video_only"),
        "handle_crossfade_ms_applied": 0,
        "handle_context_seconds": None,
    }

    parts: list[str] = [
        "[0:v]setpts=PTS-STARTPTS[v0]",
        "[1:v]setpts=PTS-STARTPTS[v1]",
    ]

    aout: str | None = None
    if use_handle:
        # ── HANDLE true-crossfade (opt-in) ───────────────────────────────────
        src_dur = probe_duration(source) or 0.0
        cont_dur = probe_duration(continuation) or 0.0
        handle_dur = probe_duration(handle_audio) or 0.0
        # junction offset inside the handle = context region duration.
        hcs = handle_dur - cont_dur
        if hcs <= 0.0:
            raise FFmpegError(
                f"join_v2v handle: derived handle_context_seconds={hcs:.4f} <= 0 "
                f"(handle_dur={handle_dur:.4f}, continuation_dur={cont_dur:.4f})"
            )
        info["handle_context_seconds"] = round(hcs, 6)

        # crossfade must fit entirely in the pre-junction context region (<= hcs)
        # and have enough source tail to fade out (<= src_dur).
        requested_cf = max(0, handle_crossfade_ms) / 1000.0
        cf = max(0.0, min(requested_cf, hcs, src_dur))
        info["handle_crossfade_ms_applied"] = round(cf * 1000.0)
        trim_start = max(0.0, hcs - cf)

        # loudness-match the handle (stream B) to the source before the fade.
        handle_label = "[2:a]"
        loud_prefix = ""
        if loudness_match:
            src_stats = _loudnorm_measure(exe, source)
            h_stats = _loudnorm_measure(exe, handle_audio)
            source_lufs = float(src_stats["input_i"])
            info["source_lufs"] = source_lufs
            info["continuation_lufs_before"] = float(h_stats["input_i"])
            info["loudness_matched"] = True
            loud_prefix = (
                f"loudnorm=I={source_lufs}:TP=-2:LRA=7:"
                f"measured_I={h_stats['input_i']}:measured_TP={h_stats['input_tp']}:"
                f"measured_LRA={h_stats['input_lra']}:measured_thresh={h_stats['input_thresh']}:"
                f"offset={h_stats['target_offset']}:linear=true,"
            )

        afmt = "aformat=sample_rates=48000:channel_layouts=stereo"
        # A = real source audio (its tail is the pre-junction crossfade window).
        parts.append(f"[0:a]{afmt},asetpts=PTS-STARTPTS[a0]")
        # B = handle trimmed to start `cf` before the junction, loudness-matched.
        parts.append(
            f"{handle_label}{loud_prefix}atrim=start={trim_start:.6f},"
            f"{afmt},asetpts=PTS-STARTPTS[a2]"
        )
        parts.append(f"[a0][a2]acrossfade=d={cf:.6f}:c1=qsin:c2=qsin[ca]")
        parts.append("[v0][v1]concat=n=2:v=1:a=0[cv]")
        vout, aout = "[cv]", "[ca]"
    elif with_audio:
        src_dur = probe_duration(source) or 0.0
        cont_dur = probe_duration(continuation) or 0.0
        requested_fade_sec = max(0, audio_fade_ms) / 1000.0
        max_fade_sec = max(0.0, min(src_dur, cont_dur))
        applied_fade_sec = min(requested_fade_sec, max_fade_sec)
        info["fade_ms_applied"] = round(applied_fade_sec * 1000.0)

        cont_audio_label = "[1:a]"
        if loudness_match:
            src_stats = _loudnorm_measure(exe, source)
            cont_stats = _loudnorm_measure(exe, continuation)
            source_lufs = float(src_stats["input_i"])
            info["source_lufs"] = source_lufs
            info["continuation_lufs_before"] = float(cont_stats["input_i"])
            info["loudness_matched"] = True

            loudnorm_filter = (
                f"loudnorm=I={source_lufs}:TP=-2:LRA=7:"
                f"measured_I={cont_stats['input_i']}:measured_TP={cont_stats['input_tp']}:"
                f"measured_LRA={cont_stats['input_lra']}:measured_thresh={cont_stats['input_thresh']}:"
                f"offset={cont_stats['target_offset']}:linear=true"
            )
            parts.append(f"[1:a]{loudnorm_filter}[a1n]")
            cont_audio_label = "[a1n]"

        if applied_fade_sec > 0:
            fade_out_start = max(0.0, src_dur - applied_fade_sec)
            parts.append(
                f"[0:a]afade=t=out:st={fade_out_start}:d={applied_fade_sec}:curve=qsin,"
                f"asetpts=PTS-STARTPTS[a0]"
            )
            parts.append(
                f"{cont_audio_label}afade=t=in:st=0:d={applied_fade_sec}:curve=qsin,"
                f"asetpts=PTS-STARTPTS[a1]"
            )
        else:
            parts.append("[0:a]asetpts=PTS-STARTPTS[a0]")
            parts.append(f"{cont_audio_label}asetpts=PTS-STARTPTS[a1]")

        parts.append("[v0][a0][v1][a1]concat=n=2:v=1:a=1[cv][ca]")
        vout, aout = "[cv]", "[ca]"
    else:
        parts.append("[v0][v1]concat=n=2:v=1:a=0[cv]")
        vout = "[cv]"

    filter_complex = ";".join(parts)

    cmd = [
        exe,
        "-y",
        "-i",
        str(source),
        "-i",
        str(continuation),
    ]
    if use_handle:
        cmd += ["-i", str(handle_audio)]
    cmd += [
        "-filter_complex",
        filter_complex,
        "-map",
        vout,
    ]
    if aout is not None:
        cmd += ["-map", aout, "-c:a", "aac"]
    cmd += [
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(src_fps),
        str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise FFmpegError(f"ffmpeg join_v2v failed (code {proc.returncode}): {proc.stderr[-2000:]}")

    if use_handle:
        # A/V duration sanity: acrossfade output = source_audio + continuation_audio
        # which should match the hard-concatenated video (source_video + cont_video).
        out_v_frames = frame_count(out)
        expected_frames = frame_count(source) + frame_count(continuation)
        if out_v_frames != expected_frames:
            raise FFmpegError(
                f"join_v2v handle: joined video frame count {out_v_frames} != "
                f"source+continuation {expected_frames}"
            )
        out_dur = probe_duration(out) or 0.0
        expected_dur = (probe_duration(source) or 0.0) + (probe_duration(continuation) or 0.0)
        if abs(out_dur - expected_dur) > 0.15:
            raise FFmpegError(
                f"join_v2v handle: output duration {out_dur:.3f}s deviates from "
                f"expected {expected_dur:.3f}s by > 0.15s (A/V desync risk)"
            )
        info["output_duration_seconds"] = round(out_dur, 4)

    return info


def save_metadata(path: Path, metadata: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2)
    return path
