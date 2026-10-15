"""AlphaGen pre/post-processing in ``services.video_io`` (AlphaGen phase 1).

* ``finalize_alpha_matte`` -- crop the pad off the engine canvas, resize to the
  source size, and write ``matte.mkv`` (FFV1 gray, FULL range) plus the H.264
  preview ``output.mp4`` in one ffmpeg run. The input here is written the way the
  official ``encode_video`` writes it: limited-range yuv420p H.264 (luma 16-235),
  so the 0/255 round trip proves the tv->pc stretch.
* ``pad_green_mp4`` -- the new ``scale``/``color`` keywords, and that leaving
  them out keeps the outpainting command unchanged.
* ``compose_alpha_rgba`` / ``compose_alpha_chroma`` (第 3 弾 出口) -- the
  commands they build, and what ffmpeg writes: ProRes 4444 with alpha, the
  matte levels on the alpha (L < R linear, L = R black and white, 0 / 255
  unchanged), the chroma fill, and the audio codecs.

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


class _Done:
    returncode = 0
    stderr = ""
    stdout = ""


def _spy_finalize(tmp_path, monkeypatch, **kw) -> list[str]:
    seen: list[list[str]] = []

    def fake_run(cmd, **_kw):
        seen.append(list(cmd))
        return _Done()

    monkeypatch.setattr(video_io, "ffmpeg_path", lambda: "ffmpeg")
    monkeypatch.setattr(video_io.subprocess, "run", fake_run)
    video_io.finalize_alpha_matte(
        tmp_path / "e.mp4", crop=(160, 96), target=(320, 192),
        preview_mp4=tmp_path / "output.mp4", matte_mkv=tmp_path / "matte.mkv", **kw,
    )
    assert len(seen) == 1
    return seen[0]


_FINALIZE_GRAPH = (
    "[0:v]crop=160:96:0:0,split=2[a][b];"
    "[a]scale=320:192:flags=bicubic:in_range=tv:out_range=pc,format=gray[m];"
    "[b]scale=320:192:flags=bicubic,format=yuv420p[p]"
)


def test_finalize_without_output_fps_keeps_the_phase1_command(tmp_path, monkeypatch):
    """Leaving ``output_fps`` out builds exactly the 第 1 弾 command."""
    cmd = _spy_finalize(tmp_path, monkeypatch)
    assert cmd == [
        "ffmpeg", "-y",
        "-i", str(tmp_path / "e.mp4"),
        "-filter_complex", _FINALIZE_GRAPH,
        "-map", "[m]", "-c:v", "ffv1", "-level", "3", "-pix_fmt", "gray", "-an",
        str(tmp_path / "matte.mkv"),
        "-map", "[p]", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", "-an",
        str(tmp_path / "output.mp4"),
    ]


def test_finalize_with_output_fps_retimes_and_writes_both_at_that_rate(tmp_path, monkeypatch):
    """§1-88: the frames are re-timed first (frame N at N / rate) and both
    outputs are written at the source's exact rational rate."""
    cmd = _spy_finalize(tmp_path, monkeypatch, output_fps="30000/1001")
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert graph == "[0:v]setpts=N/(30000/1001)/TB," + _FINALIZE_GRAPH[len("[0:v]"):]
    matte_at = cmd.index(str(tmp_path / "matte.mkv"))
    preview_at = cmd.index(str(tmp_path / "output.mp4"))
    assert cmd[matte_at - 2 : matte_at] == ["-r", "30000/1001"]
    assert cmd[preview_at - 2 : preview_at] == ["-r", "30000/1001"]
    assert cmd.count("-r") == 2


@has_ffmpeg
def test_finalize_with_output_fps_writes_ntsc_rate_and_keeps_frames(tmp_path):
    """The engine canvas is at an integer rate (24 here); with
    ``output_fps="30000/1001"`` the preview reports exactly 30000/1001, the
    matte is within 1e-3 of it, and no frame is dropped or repeated."""
    engine = tmp_path / "_alpha_engine.mp4"
    _write_engine_canvas(engine)
    preview = tmp_path / "output.mp4"
    matte = tmp_path / "matte.mkv"
    video_io.finalize_alpha_matte(
        engine, crop=(WORK_W, WORK_H), target=(SRC_W, SRC_H),
        preview_mp4=preview, matte_mkv=matte, output_fps="30000/1001",
    )
    assert video_io.probe_fps_rational(preview) == ("30000/1001", 30000 / 1001)
    assert abs(video_io.probe_fps(matte) - 30000 / 1001) < 1e-3
    assert video_io.frame_count(preview) == FRAMES
    assert video_io.frame_count(matte) == FRAMES


# --- probe_fps_rational ----------------------------------------------------------- #


@has_ffmpeg
def test_probe_fps_rational_returns_the_raw_string_and_value(tmp_path):
    src = tmp_path / "ntsc.mp4"
    subprocess.run(
        [
            _ffmpeg(), "-y", "-v", "error",
            "-f", "lavfi", "-i", f"color=c=white:size={SRC_W}x{SRC_H}:rate=30000/1001:duration=1",
            "-frames:v", str(FRAMES),
            "-c:v", "libx264rgb", "-pix_fmt", "rgb24", "-crf", "0",
            str(src),
        ],
        check=True, capture_output=True,
    )
    assert video_io.probe_fps_rational(src) == ("30000/1001", 30000 / 1001)
    # probe_fps is the value half, unchanged.
    assert video_io.probe_fps(src) == 30000 / 1001


def test_probe_fps_rational_falls_back_to_r_frame_rate(tmp_path, monkeypatch):
    class _Probe(_Done):
        stdout = json.dumps({"streams": [{"avg_frame_rate": "0/0", "r_frame_rate": "25/1"}]})

    monkeypatch.setattr(video_io.shutil, "which", lambda _n: "ffprobe")
    monkeypatch.setattr(video_io.subprocess, "run", lambda *_a, **_k: _Probe())
    assert video_io.probe_fps_rational(tmp_path / "x.mp4") == ("25/1", 25.0)


def test_probe_fps_rational_is_none_without_ffprobe(tmp_path, monkeypatch):
    monkeypatch.setattr(video_io.shutil, "which", lambda _n: None)
    assert video_io.probe_fps_rational(tmp_path / "x.mp4") is None
    assert video_io.probe_fps(tmp_path / "x.mp4") is None


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


# --- compose_alpha_rgba / compose_alpha_chroma (第 3 弾 出口) ------------------- #

# The matte has four vertical bands: 250 / 128 / 100 / 0 (left to right).
BAND_VALUES = (250, 128, 100, 0)
COMP_W, COMP_H = 64, 48
WINDOW_RGB = (200, 50, 50)


def _write_window(path) -> None:
    """A solid-colour window with a sine tone, like ``_alpha_window.mp4``."""
    subprocess.run(
        [
            _ffmpeg(), "-y", "-v", "error",
            "-f", "lavfi", "-i", f"color=c=0xC83232:size={COMP_W}x{COMP_H}:rate={FPS}:duration=1",
            "-f", "lavfi", "-i", f"sine=f=440:duration={FRAMES / FPS:.4f}",
            "-frames:v", str(FRAMES),
            "-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv444p",
            "-c:a", "pcm_s16le",
            str(path),
        ],
        check=True, capture_output=True,
    )


def _write_band_matte(path, width=COMP_W, height=COMP_H) -> None:
    """FFV1 gray, full range: four equal vertical bands of ``BAND_VALUES``."""
    band = width // 4
    boxes = ",".join(
        f"drawbox=x={i * band}:y=0:w={band}:h={height}:color=0x{v:02X}{v:02X}{v:02X}:t=fill"
        for i, v in enumerate(BAND_VALUES) if v
    )
    subprocess.run(
        [
            _ffmpeg(), "-y", "-v", "error",
            "-f", "lavfi", "-i", f"color=c=black:size={width}x{height}:rate={FPS}:duration=1,{boxes},format=gray",
            "-frames:v", str(FRAMES),
            "-c:v", "ffv1", "-pix_fmt", "gray",
            str(path),
        ],
        check=True, capture_output=True,
    )


def _read_alpha(path, width, height):
    proc = subprocess.run(
        [_ffmpeg(), "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgba", "-"],
        check=True, capture_output=True,
    )
    return np.frombuffer(proc.stdout, dtype=np.uint8).reshape(-1, height, width, 4)[..., 3]


def _band_centres(frames, width=COMP_W):
    """The value at the centre of each band, per frame -> (frames, 4)."""
    band = width // 4
    xs = [i * band + band // 2 for i in range(4)]
    return frames[:, frames.shape[1] // 2, xs].astype(int)


def _probe_streams(path) -> list[dict]:
    proc = subprocess.run(
        [
            shutil.which("ffprobe"), "-v", "error",
            "-show_entries", "stream=codec_type,codec_name,profile,pix_fmt,width,height,r_frame_rate",
            "-of", "json", str(path),
        ],
        check=True, capture_output=True, text=True,
    )
    return json.loads(proc.stdout)["streams"]


@pytest.fixture
def window_and_matte(tmp_path):
    window = tmp_path / "_alpha_window.mp4"
    matte = tmp_path / "matte.mkv"
    _write_window(window)
    _write_band_matte(matte)
    return window, matte


def _rgba(tmp_path, window, matte, transparent, opaque, fps=f"{FPS}/1"):
    mov = tmp_path / f"rgba_{transparent}_{opaque}.mov"
    preview = tmp_path / f"preview_{transparent}_{opaque}.mp4"
    video_io.compose_alpha_rgba(
        window, matte, transparent=transparent, opaque=opaque, fps=fps,
        rgba_mov=mov, preview_mp4=preview,
    )
    return mov, preview


def test_alpha_preview_size_is_640_wide_and_even():
    assert video_io.alpha_preview_size((1280, 768)) == (640, 384)
    assert video_io.alpha_preview_size((320, 256)) == (640, 512)
    assert video_io.alpha_preview_size((1920, 1080)) == (640, 360)
    # 640 * 270 / 1000 = 172.8 -> 172 (even)
    assert video_io.alpha_preview_size((1000, 270)) == (640, 172)


@pytest.mark.parametrize(
    "transparent, opaque, fragment",
    [
        (7, 248, ",lutyuv=y='clip((val-7)*255/(248-7),0,255)',format=gray"),
        (128, 128, ",lutyuv=y='if(gte(val,128),255,0)',format=gray"),
        (0, 255, ""),
    ],
)
def test_alpha_levels_fragment(transparent, opaque, fragment):
    assert video_io._alpha_levels(transparent, opaque) == fragment


@pytest.mark.parametrize("transparent, opaque", [(200, 100), (-1, 10), (0, 256)])
def test_alpha_levels_rejects_bad_thresholds(transparent, opaque):
    with pytest.raises(ValueError):
        video_io._alpha_levels(transparent, opaque)


def _spy(monkeypatch) -> list[list[str]]:
    seen: list[list[str]] = []

    def fake_run(cmd, **_kw):
        seen.append(list(cmd))
        return _Done()

    monkeypatch.setattr(video_io, "ffmpeg_path", lambda: "ffmpeg")
    monkeypatch.setattr(video_io.subprocess, "run", fake_run)
    return seen


def test_compose_rgba_command(tmp_path, monkeypatch):
    seen = _spy(monkeypatch)
    monkeypatch.setattr(video_io, "probe_resolution", lambda _p: (1280, 768))
    video_io.compose_alpha_rgba(
        tmp_path / "w.mp4", tmp_path / "m.mkv", transparent=7, opaque=248,
        fps="30000/1001", rgba_mov=tmp_path / "r.mov", preview_mp4=tmp_path / "p.mp4",
    )
    assert seen == [[
        "ffmpeg", "-y",
        "-i", str(tmp_path / "w.mp4"),
        "-i", str(tmp_path / "m.mkv"),
        "-f", "lavfi", "-i",
        "nullsrc=s=640x384:r=30000/1001,"
        "geq=lum='if(eq(mod(floor(X/16)+floor(Y/16),2),0),200,140)':cb=128:cr=128",
        "-filter_complex",
        "[0:v]setpts=N/(30000/1001)/TB[w];"
        "[1:v]setpts=N/(30000/1001)/TB,format=gray,"
        "lutyuv=y='clip((val-7)*255/(248-7),0,255)',format=gray[a];"
        "[w][a]alphamerge,split[f1][f2];"
        "[f1]format=yuva444p10le[rgba];"
        "[f2]scale=640:384[fs];"
        "[2:v][fs]overlay=shortest=1,format=yuv420p[prev]",
        "-map", "[rgba]", "-map", "0:a?",
        "-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le",
        "-alpha_bits", "16", "-c:a", "pcm_s16le", "-r", "30000/1001",
        str(tmp_path / "r.mov"),
        "-map", "[prev]", "-map", "0:a?",
        "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-r", "30000/1001",
        str(tmp_path / "p.mp4"),
    ]]


def test_compose_chroma_command(tmp_path, monkeypatch):
    seen = _spy(monkeypatch)
    monkeypatch.setattr(video_io, "probe_resolution", lambda _p: (1280, 768))
    video_io.compose_alpha_chroma(
        tmp_path / "w.mp4", tmp_path / "m.mkv", color_hex="00FF00",
        transparent=0, opaque=255, fps="24/1", out_mp4=tmp_path / "c.mp4",
    )
    assert seen == [[
        "ffmpeg", "-y",
        "-i", str(tmp_path / "w.mp4"),
        "-i", str(tmp_path / "m.mkv"),
        "-f", "lavfi", "-i", "color=c=0x00FF00:s=1280x768:r=24/1",
        "-filter_complex",
        "[0:v]setpts=N/(24/1)/TB[w];"
        "[1:v]setpts=N/(24/1)/TB,format=gray[a];"
        "[w][a]alphamerge[fg];"
        "[2:v][fg]overlay=shortest=1,format=yuv420p[out]",
        "-map", "[out]", "-map", "0:a?",
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-r", "24/1",
        str(tmp_path / "c.mp4"),
    ]]


@has_ffmpeg
def test_rgba_is_prores_4444_with_alpha_and_pcm(tmp_path, window_and_matte):
    window, matte = window_and_matte
    mov, preview = _rgba(tmp_path, window, matte, 7, 248)

    video, audio = _probe_streams(mov)
    assert (video["codec_name"], video["profile"]) == ("prores", "4444")
    # Written as yuva444p10le; ffprobe reports the 12-bit name.
    assert video["pix_fmt"].startswith("yuva")
    assert (video["width"], video["height"]) == (COMP_W, COMP_H)
    assert video["r_frame_rate"] == f"{FPS}/1"
    assert audio["codec_name"] == "pcm_s16le"
    assert video_io.frame_count(mov) == FRAMES

    pv, pa = _probe_streams(preview)
    assert (pv["codec_name"], pv["pix_fmt"]) == ("h264", "yuv420p")
    assert (pv["width"], pv["height"]) == video_io.alpha_preview_size((COMP_W, COMP_H))
    assert pv["r_frame_rate"] == f"{FPS}/1"
    assert pa["codec_name"] == "aac"
    assert video_io.frame_count(preview) == FRAMES


@has_ffmpeg
@pytest.mark.parametrize(
    "transparent, opaque, expected",
    [
        # default levels: 250 -> 255, 128 -> 128, 100 -> 98, 0 -> 0
        (7, 248, (255, 128, 98, 0)),
        # equal thresholds: black and white
        (128, 128, (255, 255, 0, 0)),
        # 0 / 255: the matte unchanged
        (0, 255, (250, 128, 100, 0)),
    ],
)
def test_rgba_alpha_follows_the_levels(tmp_path, window_and_matte, transparent, opaque, expected):
    window, matte = window_and_matte
    mov, _preview = _rgba(tmp_path, window, matte, transparent, opaque)
    alpha = _band_centres(_read_alpha(mov, COMP_W, COMP_H))
    assert alpha.shape[0] == FRAMES
    assert np.abs(alpha - np.array(expected)).max() <= 1, alpha[0]


@has_ffmpeg
def test_chroma_fills_the_transparent_part(tmp_path, window_and_matte):
    window, matte = window_and_matte
    out = tmp_path / "chroma_green.mp4"
    video_io.compose_alpha_chroma(
        window, matte, color_hex="00FF00", transparent=128, opaque=128,
        fps=f"{FPS}/1", out_mp4=out,
    )
    video, audio = _probe_streams(out)
    assert (video["codec_name"], video["pix_fmt"]) == ("h264", "yuv420p")
    assert (video["width"], video["height"]) == (COMP_W, COMP_H)
    assert audio["codec_name"] == "aac"
    assert video_io.frame_count(out) == FRAMES

    frames = _read_rgb(out, COMP_W, COMP_H)
    band = COMP_W // 4
    row = COMP_H // 2
    # L = R = 128: the 250 and 128 bands are opaque, the 100 and 0 bands are fill.
    for i, opaque_band in enumerate((True, True, False, False)):
        pixel = frames[:, row, i * band + band // 2].astype(int)
        target = WINDOW_RGB if opaque_band else (0, 255, 0)
        assert np.abs(pixel - np.array(target)).max() <= 6, (i, pixel[0])


@has_ffmpeg
def test_compose_failure_raises_ffmpeg_error(tmp_path):
    with pytest.raises(video_io.FFmpegError):
        video_io.compose_alpha_rgba(
            tmp_path / "missing.mp4", tmp_path / "missing.mkv", transparent=7, opaque=248,
            fps="24/1", rgba_mov=tmp_path / "r.mov", preview_mp4=tmp_path / "p.mp4",
        )
    with pytest.raises(video_io.FFmpegError):
        video_io.compose_alpha_chroma(
            tmp_path / "missing.mp4", tmp_path / "missing.mkv", color_hex="00FF00",
            transparent=7, opaque=248, fps="24/1", out_mp4=tmp_path / "c.mp4",
        )


@has_ffmpeg
def test_compose_keeps_window_and_matte_aligned_frame_by_frame(tmp_path):
    """The setpts / -r alignment: a window cut at the float value of
    30000/1001 (like cut_window_mp4) and a matte whose value changes every
    frame (N*4 mod 256, written like finalize_alpha_matte) give an alpha that
    matches frame for frame, with no frame dropped or repeated."""
    rate, frames = "30000/1001", 61
    window = tmp_path / "_alpha_window.mp4"
    matte = tmp_path / "matte.mkv"
    subprocess.run(
        [
            _ffmpeg(), "-y", "-v", "error",
            "-f", "lavfi", "-i", f"color=c=0xC83232:s={COMP_W}x{COMP_H}:r={rate}",
            "-f", "lavfi", "-i", "sine=f=440:duration=10",
            "-frames:v", str(frames), "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p",
            "-fps_mode", "cfr", "-r", str(30000 / 1001), "-c:a", "aac", "-shortest",
            str(window),
        ],
        check=True, capture_output=True,
    )
    subprocess.run(
        [
            _ffmpeg(), "-y", "-v", "error",
            "-f", "lavfi",
            "-i", f"color=c=black:s={COMP_W}x{COMP_H}:r=30,format=gray,geq=lum='mod(N*4\\,256)'",
            "-frames:v", str(frames),
            "-vf", f"setpts=N/({rate})/TB", "-c:v", "ffv1", "-pix_fmt", "gray", "-r", rate,
            str(matte),
        ],
        check=True, capture_output=True,
    )
    assert video_io.frame_count(window) == frames
    assert video_io.frame_count(matte) == frames

    mov = tmp_path / "rgba.mov"
    preview = tmp_path / "alpha_preview.mp4"
    video_io.compose_alpha_rgba(
        window, matte, transparent=0, opaque=255, fps=rate,
        rgba_mov=mov, preview_mp4=preview,
    )
    alpha = _read_alpha(mov, COMP_W, COMP_H)[:, COMP_H // 2, COMP_W // 2].astype(int)
    expected = np.array([(i * 4) % 256 for i in range(frames)])
    assert len(alpha) == frames
    assert video_io.frame_count(preview) == frames
    assert np.abs(alpha - expected).max() <= 1, list(zip(alpha, expected))

    out = tmp_path / "chroma_green.mp4"
    video_io.compose_alpha_chroma(
        window, matte, color_hex="00FF00", transparent=0, opaque=255, fps=rate, out_mp4=out,
    )
    assert video_io.frame_count(out) == frames
