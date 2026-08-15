"""Video encoding (ffmpeg) and metadata persistence (spec ch.10).

The LTX runner hands us decoded frames (PIL images); this module is the single
place that shells out to ffmpeg to produce ``output.mp4`` with an optional
centered crop, and writes ``metadata.json``.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

logger = logging.getLogger("ltx.video_io")


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


# The outpainting sentinel colour, RGB(102, 255, 0) = #66FF00. The official
# ComfyUI node paints it under the mask before diffusion
# (``LTXVInpaintPreprocess``, uploads/_outpaint_verify/vanish_nodes.py:92) and
# the In-Outpainting IC-LoRA was trained to replace exactly this colour. Black
# would collide with genuinely dark scene content; this green does not occur in
# natural footage.
OUTPAINT_GREEN_HEX = "0x66FF00"


def pad_green_mp4(
    input_path: Path,
    output_path: Path,
    *,
    canvas_width: int,
    canvas_height: int,
    pad_left: int,
    pad_top: int,
    frame_rate: float,
    num_frames: int,
) -> Path:
    """Place ``input_path`` inside a green canvas and write a LOSSLESS, video-only
    MP4 with EXACTLY ``num_frames`` frames at ``frame_rate`` fps.

    This is the outpainting pre-processing step (`Docs/OUTPAINTING_DESIGN_NOTES.md`
    §3-2 step 1): the source video is centred/aligned inside a larger canvas and
    the surrounding pad band is filled with the sentinel green the In-Outpainting
    IC-LoRA was trained on. The result is what `engine/pipeline/outpaint_pipeline`
    feeds BOTH as the IC-LoRA reference AND as the ``image_b`` side of the two
    Laplacian-pyramid blends, so three properties are load-bearing:

    * **Lossless RGB.** ``libx264rgb -crf 0 -pix_fmt rgb24`` keeps the green at
      exactly (102, 255, 0). A yuv420p round trip shifts it by several units and
      chroma-subsamples the pad/keep boundary, which is precisely where the model
      has to decide what is sentinel and what is content. ``ffv1`` is the
      fallback for ffmpeg builds without libx264rgb (also lossless RGB).
    * **Exact frame count.** The blends pair frame *i* of the generation with
      frame *i* of this canvas, and stage 2 asserts its initial latent matches the
      target shape (``ltx_core/tools.py:106`` ``create_initial_state``), so a
      canvas that is even one frame short crashes the job deep inside the
      denoiser. ``tpad=stop=-1:stop_mode=clone`` extends the last frame
      indefinitely and ``-frames:v`` cuts at exactly ``num_frames`` — the source
      is normally long enough (the app validates that) and this is the structural
      backstop against VFR/rounding losing a frame in the ``fps`` filter.
    * **Video only.** ``-an``: the audio never travels through this file. The
      engine reads the ORIGINAL upload for the frozen-audio guidance, which
      avoids both a lossy re-encode and the "codec X has no tag in the MP4
      container" failure that ``-c:a copy`` hits on PCM/Vorbis/FLAC audio inside
      a user-supplied container.

    The source is NOT resized: the caller has already verified that its
    resolution equals the keep rectangle (canvas minus pads).
    """
    exe = ffmpeg_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # fps first: the pad/tpad geometry is per frame, so normalising the cadence
    # before extending keeps `-frames:v` counting output frames, not source ones.
    #
    # ``format=rgb24`` before ``pad`` is NOT cosmetic. ffmpeg specifies the pad
    # colour in RGB but applies it in the filter chain's current pixel format —
    # for a normal yuv420p input that means the sentinel is converted RGB -> YUV
    # here and YUV -> RGB again at the rgb24 encoder, and the round trip lands on
    # (101, 253, 0) instead of (102, 255, 0) (measured). The IC-LoRA was trained
    # on the exact colour, so the chain is forced into RGB before the pad is
    # painted and stays there through the lossless encode.
    vf = (
        f"fps={frame_rate},"
        f"format=rgb24,"
        f"pad={canvas_width}:{canvas_height}:{pad_left}:{pad_top}:color={OUTPAINT_GREEN_HEX},"
        f"tpad=stop=-1:stop_mode=clone"
    )

    def _run(vcodec: list[str]) -> subprocess.CompletedProcess:
        cmd = [
            exe, "-y",
            "-i", str(input_path),
            "-vf", vf,
            "-map", "0:v",
            "-an",
            *vcodec,
            "-frames:v", str(int(num_frames)),
            str(output_path),
        ]
        return subprocess.run(cmd, capture_output=True, text=True)

    proc = _run(["-c:v", "libx264rgb", "-pix_fmt", "rgb24", "-crf", "0"])
    if proc.returncode != 0:
        # Second lossless-RGB option for builds compiled without libx264rgb.
        fallback = _run(["-c:v", "ffv1", "-pix_fmt", "rgb24"])
        if fallback.returncode != 0:
            raise FFmpegError(
                f"ffmpeg green-pad failed (libx264rgb code {proc.returncode}, "
                f"ffv1 code {fallback.returncode}): {fallback.stderr[-2000:]}"
            )
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


def cut_range_mp4(
    src: Path,
    out: Path,
    start_sec: float,
    duration_sec: float,
    *,
    start_frame: int | None = None,
    num_frames: int | None = None,
) -> dict:
    """Write to ``out`` an mp4 holding EXACTLY the ``[start_sec, start_sec +
    duration_sec)`` window of ``src``, at ``src``'s own measured frame rate.

    Unlike :func:`cut_tail_mp4` this NEVER resamples: the caller wants the very
    same material the user sees on the timeline ribbon, so the source cadence is
    preserved verbatim (``-r src_fps``) and only the frame window changes.

    UNIT NOTE (deliberate hedge, now exercised): the public signature takes
    SECONDS, but this function internally converts to frames immediately
    (``start_frame = round(start_sec * src_fps)``) and does all of its work in
    frame space via the ``select`` filter. ``start_frame``/``num_frames`` are the
    additive short-circuit this note always promised: when given, each one
    replaces the corresponding seconds->frames conversion outright (``start_sec``/
    ``duration_sec`` are ignored for the side that has a frame value), and none of
    the selection logic below has to move. §1-15's upload-time frame-count trim
    (``services/video_upload_store.save``'s ``max_frames``) is the first caller —
    it wants "first N frames", which is a frame count, not a duration computed
    from a frame rate that may not even matter to it.

    The window is inclusive on both ends in frame space
    (``select='between(n,start,end)'``) and ``end`` is clamped to the last
    available frame, so asking for more than the source holds simply yields the
    remainder rather than failing. Audio, when present, is trimmed to the same
    window (``atrim``) and re-encoded to AAC.

    Returns ``{source_fps, total_frames, start_frame, end_frame,
    written_frames}``. Raises :class:`FFmpegError` when the frame rate cannot be
    probed (the caller must not silently guess a cadence), when the requested
    duration is non-positive, or when the window starts past the end of the
    source.
    """
    exe = ffmpeg_path()
    out.parent.mkdir(parents=True, exist_ok=True)

    source_fps = probe_fps(src)
    if source_fps is None or source_fps <= 0:
        raise FFmpegError(f"cut_range_mp4: could not probe a usable frame rate for {src}")

    start = max(0, int(start_frame)) if start_frame is not None else max(
        0, round(start_sec * source_fps)
    )
    n = int(num_frames) if num_frames is not None else round(duration_sec * source_fps)
    if n <= 0:
        if num_frames is not None:
            raise FFmpegError(f"cut_range_mp4: num_frames={num_frames} must be positive")
        raise FFmpegError(
            f"cut_range_mp4: duration_sec={duration_sec} maps to {n} frames at {source_fps} fps"
        )

    total = frame_count(src)
    if start >= total:
        raise FFmpegError(
            f"cut_range_mp4: start_sec={start_sec} maps to frame {start} but source has {total} frames"
        )
    end = min(total - 1, start + n - 1)

    with_audio = has_audio_stream(src)

    parts = [f"[0:v]select='between(n\\,{start}\\,{end})',setpts=PTS-STARTPTS[v]"]
    vout = "[v]"
    aout = None
    if with_audio:
        start_t = start / float(source_fps)
        end_t = (end + 1) / float(source_fps)
        parts.append(f"[0:a]atrim=start={start_t}:end={end_t},asetpts=PTS-STARTPTS[a]")
        aout = "[a]"
    filter_complex = ";".join(parts)

    cmd = [exe, "-y", "-i", str(src), "-filter_complex", filter_complex, "-map", vout]
    if aout is not None:
        cmd += ["-map", aout, "-c:a", "aac"]
    cmd += [
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(source_fps),
        str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise FFmpegError(f"ffmpeg range-cut failed (code {proc.returncode}): {proc.stderr[-2000:]}")

    return {
        "source_fps": source_fps,
        "total_frames": total,
        "start_frame": start,
        "end_frame": end,
        "written_frames": end - start + 1,
    }


def cut_window_mp4(
    src: Path,
    out: Path,
    window_start_sec: float,
    num_frames: int,
    fps: float,
) -> dict:
    """Write to ``out`` EXACTLY ``num_frames`` frames of ``src`` starting at
    ``window_start_sec``, at ``fps`` — the retake window the engine consumes.

    A third cutter next to :func:`cut_tail_mp4` and :func:`cut_range_mp4` because
    its CONTRACT is different from both, not just its arguments:

      * it RESAMPLES when the source cadence differs (``cut_range_mp4`` never
        does — it deliberately preserves the material's own cadence), reusing
        ``cut_tail_mp4``'s two-pass shape;
      * it is FRAME-EXACT OR IT FAILS. ``cut_range_mp4`` clamps a too-long
        request to the remainder and reports a PREDICTED ``written_frames``;
        here the frame count is load-bearing geometry (the whole window must be
        8n+1 and must land on one stage-2 tile), so the written file is
        re-probed with :func:`frame_count` and a mismatch raises;
      * it encodes for a VAE round trip rather than for delivery — see the
        codec note below.

    CODEC CHOICE (deliberate, not defaults): ``-crf 12`` instead of the usual 23
    and CFR output. The window's two glue bands are re-encoded, VAE-encoded and
    then frozen bit-exact, so whatever this file loses is a permanent ceiling on
    the frozen ends' quality — the one place in the pipeline where the
    intermediate's fidelity shows up in the deliverable. Audio is written as
    PCM when the container/build accepts it (``pcm_s16le``), falling back to AAC
    192k, for the same reason: a lossy intermediate would be baked into the
    frozen audio latents.

    Returns ``{source_fps, resampled, total_frames, start_frame, written_frames,
    has_audio}`` where ``written_frames`` is MEASURED, not predicted. Raises
    :class:`FFmpegError` on an unprobeable frame rate, a window that runs past
    the end of the (resampled) source, or a frame-count mismatch.
    """
    exe = ffmpeg_path()
    out.parent.mkdir(parents=True, exist_ok=True)

    if num_frames <= 0:
        raise FFmpegError(f"cut_window_mp4: num_frames must be >= 1 (got {num_frames})")
    source_fps = probe_fps(src)
    if source_fps is None or source_fps <= 0:
        raise FFmpegError(f"cut_window_mp4: could not probe a usable frame rate for {src}")
    resampled = abs(source_fps - float(fps)) > 1e-3

    work = src
    tmp_resampled: Path | None = None
    if resampled:
        tmp_resampled = out.parent / (out.stem + "_resampled.mp4")
        rs_cmd = [
            exe, "-y", "-i", str(src),
            "-vf", f"fps={fps}",
            "-map", "0:v", "-map", "0:a?",
            "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p",
            "-fps_mode", "cfr", "-r", str(fps),
            "-c:a", "aac", "-b:a", "192k",
            str(tmp_resampled),
        ]
        proc = subprocess.run(rs_cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise FFmpegError(
                f"ffmpeg fps-resample failed (code {proc.returncode}): {proc.stderr[-2000:]}"
            )
        work = tmp_resampled

    try:
        total = frame_count(work)
        # The window is defined on the timeline the REQUEST fps describes, which
        # is the cadence ``work`` now has.
        start = max(0, round(window_start_sec * float(fps)))
        end = start + num_frames - 1
        if end > total - 1:
            raise FFmpegError(
                f"cut_window_mp4: window [{start}, {end}] runs past the source "
                f"(after resample={resampled} it has {total} frames)"
            )
        with_audio = has_audio_stream(work)

        parts = [f"[0:v]select='between(n\\,{start}\\,{end})',setpts=PTS-STARTPTS[v]"]
        aout = None
        if with_audio:
            start_t = start / float(fps)
            end_t = (end + 1) / float(fps)
            parts.append(f"[0:a]atrim=start={start_t}:end={end_t},asetpts=PTS-STARTPTS[a]")
            aout = "[a]"
        filter_complex = ";".join(parts)

        def _cut(audio_args: list[str]) -> subprocess.CompletedProcess:
            cmd = [exe, "-y", "-i", str(work), "-filter_complex", filter_complex,
                   "-map", "[v]"]
            if aout is not None:
                cmd += ["-map", aout] + audio_args
            cmd += [
                "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p",
                "-fps_mode", "cfr", "-r", str(fps),
                str(out),
            ]
            return subprocess.run(cmd, capture_output=True, text=True)

        # PCM first; fall back to AAC 192k when this ffmpeg/container pair
        # refuses it (decided by the actual build, never guessed).
        proc = _cut(["-c:a", "pcm_s16le"])
        if proc.returncode != 0 and aout is not None:
            proc = _cut(["-c:a", "aac", "-b:a", "192k"])
        if proc.returncode != 0:
            raise FFmpegError(
                f"ffmpeg window-cut failed (code {proc.returncode}): {proc.stderr[-2000:]}"
            )
    finally:
        if tmp_resampled is not None:
            tmp_resampled.unlink(missing_ok=True)

    # MEASURED, not predicted: the engine's whole geometry rests on this count.
    written = frame_count(out)
    if written != num_frames:
        raise FFmpegError(
            f"cut_window_mp4 wrote {written} frames but {num_frames} were "
            f"requested (start_frame={start}, fps={fps}, resampled={resampled})"
        )
    return {
        "source_fps": source_fps,
        "resampled": resampled,
        "total_frames": total,
        "start_frame": start,
        "written_frames": written,
        "has_audio": has_audio_stream(out),
    }


def still_image_mp4(
    src: Path,
    out: Path,
    num_frames: int,
    fps: float,
) -> dict:
    """Write to ``out`` EXACTLY ``num_frames`` identical frames of the still
    ``src``, at ``fps`` — a silent video the end-source path can feed to the
    engine through the SAME code path an uploaded video takes.

    That is the whole reason this exists: an end source may be a picture, but the
    engine only ever freezes a decoded video tail, so the picture is turned into
    a video HERE (app side, with ffmpeg) rather than teaching the engine a second
    kind of input. There is deliberately no cutter counterpart — a video end
    source is cut by :func:`cut_window_mp4` with ``window_start_sec=0.0``, whose
    contract (fps resample, crf 12, CFR, MEASURED frame count) is already exactly
    what this path needs.

    THE CALLER MUST PASS ``context_frames + 1``, never ``context_frames``. The
    causal video VAE spends the first frame on its lone keyframe latent, which is
    not part of the tail band; a file one frame short would leave the band's
    latents empty.

    Frames are written as ``-crf 12`` yuv420p CFR for the same reason
    :func:`cut_window_mp4` does: the material is VAE-encoded and then frozen, so
    whatever this intermediate loses is a permanent ceiling on the frozen tail's
    quality. No audio track is produced (v1 freezes video only). Odd-sized
    stills are scaled to the nearest even width/height — libx264 + yuv420p cannot
    encode an odd dimension — and an alpha channel is simply dropped by the
    yuv420p conversion.

    Returns ``{written_frames, width, height}`` where ``written_frames`` is
    MEASURED, not predicted. Raises :class:`FFmpegError` on a failed encode or a
    frame-count mismatch.
    """
    exe = ffmpeg_path()
    out.parent.mkdir(parents=True, exist_ok=True)

    if num_frames <= 0:
        raise FFmpegError(f"still_image_mp4: num_frames must be >= 1 (got {num_frames})")

    cmd = [
        exe, "-y",
        "-loop", "1", "-framerate", str(fps), "-i", str(src),
        "-frames:v", str(int(num_frames)),
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-an",
        "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p",
        "-fps_mode", "cfr", "-r", str(fps),
        str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise FFmpegError(
            f"ffmpeg still-image encode failed (code {proc.returncode}): "
            f"{proc.stderr[-2000:]}"
        )

    # MEASURED, not predicted: the engine's tail geometry rests on this count.
    written = frame_count(out)
    if written != num_frames:
        raise FFmpegError(
            f"still_image_mp4 wrote {written} frames but {num_frames} were "
            f"requested (fps={fps})"
        )
    size = probe_resolution(out)
    return {
        "written_frames": written,
        "width": size[0] if size else None,
        "height": size[1] if size else None,
    }


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


def _loudnorm_stats_usable(stats: dict, *, as_target: bool) -> bool:
    """Return True only when ``stats`` (a pass-1 ``loudnorm`` JSON block) can be
    safely expanded into a pass-2 ``loudnorm`` filter argument.

    Digitally-silent audio (a present-but-zero stream, e.g. a source recording
    with a muted track) makes ffmpeg report ``input_i`` (and friends) as
    ``-inf``. Feeding that straight back into ``loudnorm=I=-inf`` /
    ``measured_I=-inf`` makes ffmpeg reject the filter graph
    (``Value -inf for parameter 'I' out of range [-70 - -5]``), which surfaced as
    a 503 on the V2V join. This guard detects non-finite / out-of-range measured
    values so the caller can skip loudness matching instead of crashing.

    ``as_target`` selects the range that ``input_i`` must satisfy: True when the
    value will be passed as the pass-2 *target* ``I=`` (ffmpeg accepts
    ``[-70, -5]``); False when it is a *measured_* input (ffmpeg accepts
    ``[-99, 0]``). The remaining fields only need to be finite floats.
    """
    keys = ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")
    vals: dict[str, float] = {}
    for key in keys:
        raw = stats.get(key)
        try:
            val = float(raw)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(val):
            return False
        vals[key] = val
    lo, hi = (-70.0, -5.0) if as_target else (-99.0, 0.0)
    if not (lo <= vals["input_i"] <= hi):
        return False
    return True


def normalize_clip(src: Path, out: Path, width: int, height: int, fps: float) -> Path:
    """Re-encode ``src`` to exactly ``width`` x ``height`` @ ``fps`` for joining.

    The V2V join (:func:`join_v2v` / :func:`concat_mp4s`) requires both inputs to
    share resolution and fps, but the user's uploaded source video generally does
    not match the generated continuation (the engine only fps-aligns the context
    TAIL it consumes — the full source is untouched). This is the app-side
    normalization pass (R3 smoke, 2026-07-05): scale-to-cover + centered crop
    (mirroring the engine's resize+center-crop conditioning semantics) + fps
    resample. ``setsar=1`` is REQUIRED — scaling can leave a fractional sample
    aspect ratio and ffmpeg's ``concat`` filter then rejects the pair even though
    the pixel dimensions match (verified in the R3 smoke). Audio is carried
    through untouched (``-c:a aac`` re-mux). Returns ``out``.
    """
    exe = ffmpeg_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    vf = (
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height}:(in_w-{width})/2:(in_h-{height})/2,"
        f"setsar=1,fps={fps}"
    )
    cmd = [
        exe, "-y", "-i", str(src),
        "-vf", vf, "-map", "0:v", "-map", "0:a?",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(fps),
        "-c:a", "aac",
        str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise FFmpegError(
            f"ffmpeg normalize_clip failed (code {proc.returncode}): {proc.stderr[-2000:]}"
        )
    return out


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
            if _loudnorm_stats_usable(src_stats, as_target=True) and _loudnorm_stats_usable(
                h_stats, as_target=False
            ):
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
            else:
                info["loudness_skip_reason"] = (
                    "loudnorm stats unusable (likely digitally-silent audio); "
                    "skipped loudness matching to avoid an out-of-range filter argument"
                )
                logger.warning(
                    "join_v2v handle: skipping loudness match "
                    "(source input_i=%r, handle input_i=%r)",
                    src_stats.get("input_i"),
                    h_stats.get("input_i"),
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
            if _loudnorm_stats_usable(src_stats, as_target=True) and _loudnorm_stats_usable(
                cont_stats, as_target=False
            ):
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
            else:
                info["loudness_skip_reason"] = (
                    "loudnorm stats unusable (likely digitally-silent audio); "
                    "skipped loudness matching to avoid an out-of-range filter argument"
                )
                logger.warning(
                    "join_v2v: skipping loudness match "
                    "(source input_i=%r, continuation input_i=%r)",
                    src_stats.get("input_i"),
                    cont_stats.get("input_i"),
                )

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
