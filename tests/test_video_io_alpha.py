"""AlphaGen pre/post-processing in ``services.video_io`` (AlphaGen phase 1).

* ``finalize_alpha_matte`` -- crop the pad off the engine canvas, resize to the
  source size, and write ``matte.mkv`` (FFV1 gray, FULL range) plus the H.264
  preview ``output.mp4`` in one ffmpeg run. The input here is written the way the
  official ``encode_video`` writes it: limited-range yuv420p H.264 (luma 16-235),
  so the 0/255 round trip proves the tv->pc stretch.
* ``pad_green_mp4`` -- the new ``scale``/``color`` keywords, and that leaving
  them out keeps the outpainting command unchanged.

App venv (no torch needed); skipped without ffmpeg on PATH. Same manners as
``tests/test_inpaint_green_fill.py``.
"""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from services import video_io

has_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="the whole point of this module is what ffmpeg actually writes",
)

np = pytest.importorskip("numpy")

FPS = 24
FRAMES = 9
CANVAS_W, CANVAS_H = 192, 128  # engine canvas (64 grid)
WORK_W, WORK_H = 160, 96  # working size = top-left of the canvas
SRC_W, SRC_H = 320, 192  # delivered (source) size
PAD_GRAY = 128  # pad band value: must NOT survive the crop


def _ffmpeg() -> str:
    return shutil.which("ffmpeg")


def _canvas_frame():
    """White left half / black right half of the working area; mid-gray pad."""
    img = np.full((CANVAS_H, CANVAS_W, 3), PAD_GRAY, dtype=np.uint8)
    img[:WORK_H, : WORK_W // 2] = 255
    img[:WORK_H, WORK_W // 2 : WORK_W] = 0
    return img


def _write_engine_canvas(path) -> None:
    """Limited-range yuv420p H.264, crf 0 -- the official encode_video's format."""
    raw = np.stack([_canvas_frame()] * FRAMES).tobytes()
    subprocess.run(
        [
            _ffmpeg(), "-y", "-v", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{CANVAS_W}x{CANVAS_H}", "-r", str(FPS),
            "-i", "-",
            "-vf", "scale=out_color_matrix=bt709:out_range=tv",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "0",
            "-color_range", "tv", "-colorspace", "bt709",
            str(path),
        ],
        input=raw, check=True, capture_output=True,
    )


def _read_gray(path, width, height):
    proc = subprocess.run(
        [_ffmpeg(), "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        check=True, capture_output=True,
    )
    return np.frombuffer(proc.stdout, dtype=np.uint8).reshape(-1, height, width)


def _read_rgb(path, width, height):
    proc = subprocess.run(
        [_ffmpeg(), "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        check=True, capture_output=True,
    )
    return np.frombuffer(proc.stdout, dtype=np.uint8).reshape(-1, height, width, 3)


def _stream(path) -> dict:
    proc = subprocess.run(
        [
            shutil.which("ffprobe"), "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,pix_fmt,width,height",
            "-of", "json", str(path),
        ],
        check=True, capture_output=True, text=True,
    )
    return json.loads(proc.stdout)["streams"][0]


# --- finalize_alpha_matte ------------------------------------------------------ #


@pytest.fixture
def finalized(tmp_path):
    engine = tmp_path / "_alpha_engine.mp4"
    _write_engine_canvas(engine)
    preview = tmp_path / "output.mp4"
    matte = tmp_path / "matte.mkv"
    info = video_io.finalize_alpha_matte(
        engine, crop=(WORK_W, WORK_H), target=(SRC_W, SRC_H),
        preview_mp4=preview, matte_mkv=matte,
    )
    return info, preview, matte


@has_ffmpeg
def test_finalize_returns_the_summary(finalized):
    info, preview, matte = finalized
    assert info == {
        "matte_codec": "ffv1/gray",
        "crop": [WORK_W, WORK_H],
        "target": [SRC_W, SRC_H],
    }
    # The frame counts are measured on the written files, not reported.
    assert video_io.frame_count(matte) == FRAMES
    assert video_io.frame_count(preview) == FRAMES


@has_ffmpeg
def test_matte_is_ffv1_gray_at_the_source_size(finalized):
    _info, _preview, matte = finalized
    s = _stream(matte)
    assert (s["codec_name"], s["pix_fmt"]) == ("ffv1", "gray")
    assert (s["width"], s["height"]) == (SRC_W, SRC_H)
    assert video_io.frame_count(matte) == FRAMES


@has_ffmpeg
def test_matte_returns_to_full_range_zero_and_255(finalized):
    """Limited-range 235/16 come back as 255/0 away from the band edge."""
    _info, _preview, matte = finalized
    frames = _read_gray(matte, SRC_W, SRC_H)
    assert frames.shape[0] == FRAMES
    margin = 8
    white = frames[:, :, : SRC_W // 2 - margin]
    black = frames[:, :, SRC_W // 2 + margin :]
    assert white.min() == 255 and white.max() == 255
    assert black.min() == 0 and black.max() == 0


@has_ffmpeg
def test_pad_band_is_cut_off(finalized):
    """The mid-gray pad (right of and below the working area) is gone: the
    right edge and the bottom rows are pure band values, not gray."""
    _info, _preview, matte = finalized
    frames = _read_gray(matte, SRC_W, SRC_H)
    assert (frames[:, :, -4:] == 0).all()  # right edge = black band
    assert (frames[:, -4:, :40] == 255).all()  # bottom rows, white band
    assert (frames[:, -4:, -40:] == 0).all()  # bottom rows, black band
    mid = (frames > 64) & (frames < 192)
    # only the soft white/black boundary may be gray -- never a whole edge
    assert not mid[:, :, -8:].any() and not mid[:, -8:, :].any()


@has_ffmpeg
def test_preview_is_h264_yuv420p_at_the_source_size(finalized):
    _info, preview, _matte = finalized
    s = _stream(preview)
    assert (s["codec_name"], s["pix_fmt"]) == ("h264", "yuv420p")
    assert (s["width"], s["height"]) == (SRC_W, SRC_H)
    assert video_io.frame_count(preview) == FRAMES


@has_ffmpeg
def test_finalize_failure_raises_ffmpeg_error(tmp_path):
    with pytest.raises(video_io.FFmpegError):
        video_io.finalize_alpha_matte(
            tmp_path / "missing.mp4", crop=(WORK_W, WORK_H), target=(SRC_W, SRC_H),
            preview_mp4=tmp_path / "output.mp4", matte_mkv=tmp_path / "matte.mkv",
        )


# --- pad_green_mp4: scale / color ----------------------------------------------- #


def _write_source(path, width=SRC_W, height=SRC_H, color="white") -> None:
    subprocess.run(
        [
            _ffmpeg(), "-y", "-v", "error",
            "-f", "lavfi", "-i", f"color=c={color}:size={width}x{height}:rate={FPS}:duration=1",
            "-frames:v", str(FRAMES),
            "-c:v", "libx264rgb", "-pix_fmt", "rgb24", "-crf", "0",
            str(path),
        ],
        check=True, capture_output=True,
    )


@has_ffmpeg
def test_pad_scale_and_black_color(tmp_path):
    """AlphaGen reference: resize to the working size, pad black on the right
    and bottom to the canvas."""
    src = tmp_path / "window.mp4"
    _write_source(src)
    out = tmp_path / "alpha_reference.mp4"
    video_io.pad_green_mp4(
        src, out,
        canvas_width=CANVAS_W, canvas_height=CANVAS_H, pad_left=0, pad_top=0,
        frame_rate=FPS, num_frames=FRAMES,
        scale=(WORK_W, WORK_H), color="black",
    )
    frames = _read_rgb(out, CANVAS_W, CANVAS_H)
    assert frames.shape[0] == FRAMES
    assert (frames[:, :WORK_H, :WORK_W] == 255).all()
    assert (frames[:, :, WORK_W:] == 0).all()
    assert (frames[:, WORK_H:, :] == 0).all()


def test_pad_defaults_keep_the_outpainting_command(tmp_path, monkeypatch):
    """Leaving ``scale``/``color`` out builds exactly the pre-AlphaGen command."""
    seen: list[list[str]] = []

    class _Done:
        returncode = 0
        stderr = ""

    def fake_run(cmd, **_kw):
        seen.append(list(cmd))
        return _Done()

    monkeypatch.setattr(video_io, "ffmpeg_path", lambda: "ffmpeg")
    monkeypatch.setattr(video_io.subprocess, "run", fake_run)
    video_io.pad_green_mp4(
        tmp_path / "in.mp4", tmp_path / "out.mp4",
        canvas_width=384, canvas_height=256, pad_left=32, pad_top=16,
        frame_rate=24, num_frames=9,
    )
    assert seen == [[
        "ffmpeg", "-y",
        "-i", str(tmp_path / "in.mp4"),
        "-vf", "fps=24,format=rgb24,pad=384:256:32:16:color=0x66FF00,tpad=stop=-1:stop_mode=clone",
        "-map", "0:v",
        "-an",
        "-c:v", "libx264rgb", "-pix_fmt", "rgb24", "-crf", "0",
        "-frames:v", "9",
        str(tmp_path / "out.mp4"),
    ]]
