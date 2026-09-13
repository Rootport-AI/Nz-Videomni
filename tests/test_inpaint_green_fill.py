"""``services.video_io.fill_mask_green_mp4`` — the inpaint canvas (台帳 §3-55).

Real ffmpeg, real mp4 files, raw RGB readback. Nothing here is a synthetic
tensor, because every property this function has to hold is a property of what
ffmpeg actually wrote:

* the sentinel is EXACTLY (102, 255, 0) — the In-Outpainting IC-LoRA was trained
  on that triple and the engine's de-green replaces pixels by value, so a
  two-unit drift leaves a green haze in the delivered video;
* everything OUTSIDE the mask is the source, byte for byte;
* the mask is BINARISED at 128 before the merge, so a soft H.264 edge does not
  become a ring of half-green pixels;
* the frame count is MEASURED, because this ffmpeg's ``maskedmerge`` has no
  ``shortest`` option (see the function's own docstring) and a short mask would
  otherwise be padded by repetition and pass unnoticed.

App venv (no torch needed); skipped without ffmpeg on PATH.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from services import video_io

has_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="the whole point of this module is what ffmpeg actually writes",
)

np = pytest.importorskip("numpy")

GREEN = (102, 255, 0)
SRC_W, SRC_H, FRAMES, FPS = 320, 192, 9, 24
CANVAS_W, CANVAS_H = 384, 256  # SRC rounded up to the 128 grid
RECT = (64, 48, 192, 144)  # x0, y0, x1, y1, half-open


def _ffmpeg() -> str:
    return shutil.which("ffmpeg")


def _write_source(path, *, width=SRC_W, height=SRC_H, frames=FRAMES) -> None:
    """A moving test pattern, lossless so "outside the mask is unchanged" can be
    an EXACT claim rather than an approximate one."""
    subprocess.run(
        [
            _ffmpeg(), "-y", "-v", "error",
            "-f", "lavfi",
            "-i", f"testsrc=size={width}x{height}:rate={FPS}:duration=2",
            "-frames:v", str(frames),
            "-c:v", "libx264rgb", "-pix_fmt", "rgb24", "-crf", "0",
            str(path),
        ],
        check=True, capture_output=True,
    )


def _write_mask(path, *, frames=FRAMES, width=SRC_W, height=SRC_H, lossy=True,
                rect=RECT, moving=False) -> None:
    """A white rectangle on black, encoded the way the plugin's writer would.

    Written as a PNG sequence rather than through ``drawbox`` so the moving case
    is exact: this ffmpeg's ``drawbox`` has no frame-number variable, and driving
    the position off the timestamp instead makes the fixture depend on float
    rounding. One image per frame says precisely what each frame contains.
    """
    from PIL import Image, ImageDraw

    x0, y0, x1, y1 = rect
    stage = path.parent / f"{path.stem}_png"
    stage.mkdir(exist_ok=True)
    for i in range(frames):
        img = Image.new("RGB", (width, height), (0, 0, 0))
        shift = i if moving else 0
        ImageDraw.Draw(img).rectangle(
            [x0 + shift, y0, x1 - 1 + shift, y1 - 1], fill=(255, 255, 255)
        )
        img.save(stage / f"{i:06d}.png")

    codec = (
        ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
        if lossy
        else ["-c:v", "libx264rgb", "-pix_fmt", "rgb24", "-crf", "0"]
    )
    subprocess.run(
        [
            _ffmpeg(), "-y", "-v", "error",
            "-framerate", str(FPS),
            "-i", str(stage / "%06d.png"),
            "-frames:v", str(frames), *codec, str(path),
        ],
        check=True, capture_output=True,
    )


def _write_flat(path, *, level: int, frames=FRAMES) -> None:
    """A flat grey field — the binarisation probe."""
    hexc = f"0x{level:02X}{level:02X}{level:02X}"
    subprocess.run(
        [
            _ffmpeg(), "-y", "-v", "error",
            "-f", "lavfi",
            "-i", f"color=c={hexc}:s={SRC_W}x{SRC_H}:rate={FPS}:duration=2",
            "-frames:v", str(frames),
            "-c:v", "libx264rgb", "-pix_fmt", "rgb24", "-crf", "0",
            str(path),
        ],
        check=True, capture_output=True,
    )


def _rgb(path, width, height):
    """Every frame of ``path`` as ``(F, H, W, 3)`` uint8 — no YUV round trip."""
    raw = subprocess.run(
        [_ffmpeg(), "-v", "error", "-i", str(path), "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-"],
        check=True, capture_output=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, height, width, 3)


def _fill(tmp_path, source, mask, **over):
    out = tmp_path / "canvas.mp4"
    kwargs = dict(
        canvas_width=CANVAS_W,
        canvas_height=CANVAS_H,
        frame_rate=float(FPS),
        num_frames=FRAMES,
    )
    kwargs.update(over)
    return out, video_io.fill_mask_green_mp4(source, mask, out, **kwargs)


# ── 1. the sentinel ─────────────────────────────────────────────────────────
@has_ffmpeg
def test_the_masked_region_is_exactly_the_sentinel_green(tmp_path):
    src, mask = tmp_path / "src.mp4", tmp_path / "mask.mp4"
    _write_source(src)
    _write_mask(mask, lossy=False)
    out, info = _fill(tmp_path, src, mask)

    frames = _rgb(out, CANVAS_W, CANVAS_H)
    x0, y0, x1, y1 = RECT
    rect = frames[:, y0:y1, x0:x1, :]
    assert (rect == np.array(GREEN, dtype=np.uint8)).all(), (
        "every masked pixel must be (102, 255, 0) exactly; a colour-input green "
        "plate would land on (101, 253, 0)"
    )
    assert info["codec"] in ("libx264rgb", "ffv1")


@has_ffmpeg
def test_the_pad_bands_are_the_same_sentinel(tmp_path):
    src, mask = tmp_path / "src.mp4", tmp_path / "mask.mp4"
    _write_source(src)
    _write_mask(mask, lossy=False)
    out, _ = _fill(tmp_path, src, mask)

    frames = _rgb(out, CANVAS_W, CANVAS_H)
    right = frames[:, :SRC_H, SRC_W:, :]
    bottom = frames[:, SRC_H:, :, :]
    assert (right == np.array(GREEN, dtype=np.uint8)).all()
    assert (bottom == np.array(GREEN, dtype=np.uint8)).all()


@has_ffmpeg
def test_everything_outside_the_mask_is_the_source_byte_for_byte(tmp_path):
    src, mask = tmp_path / "src.mp4", tmp_path / "mask.mp4"
    _write_source(src)
    _write_mask(mask, lossy=False)
    out, _ = _fill(tmp_path, src, mask)

    canvas = _rgb(out, CANVAS_W, CANVAS_H)[:, :SRC_H, :SRC_W, :].astype(int)
    source = _rgb(src, SRC_W, SRC_H)[:FRAMES].astype(int)
    x0, y0, x1, y1 = RECT
    keep = np.ones((SRC_H, SRC_W), dtype=bool)
    keep[y0:y1, x0:x1] = False
    assert np.abs(canvas[:, keep] - source[:, keep]).max() == 0


@has_ffmpeg
def test_a_moving_mask_moves_the_green(tmp_path):
    """The mask really is a different picture per frame, and the pairing is by
    frame NUMBER — ``setpts=N/(fr*TB)`` on both inputs."""
    src, mask = tmp_path / "src.mp4", tmp_path / "mask.mp4"
    # A FLAT source here rather than the test pattern: ``testsrc``'s colour ramps
    # contain the sentinel triple by coincidence, and this test locates the green
    # by searching for it.
    _write_flat(src, level=40)
    _write_mask(mask, lossy=False, moving=True)
    out, _ = _fill(tmp_path, src, mask)

    frames = _rgb(out, CANVAS_W, CANVAS_H)
    x0, y0, x1, y1 = RECT
    green = np.array(GREEN, dtype=np.uint8)
    for i in range(FRAMES):
        row = frames[i, (y0 + y1) // 2, :, :]
        cols = np.flatnonzero((row == green).all(axis=-1))
        # The pad band on the right is green too, so only look left of it.
        cols = cols[cols < SRC_W]
        assert cols.min() == x0 + i, f"frame {i}: green starts at {cols.min()}"


# ── 2. binarisation ─────────────────────────────────────────────────────────
@has_ffmpeg
@pytest.mark.parametrize("level,masked", [(0, False), (100, False), (127, False),
                                          (128, True), (200, True), (255, True)])
def test_the_mask_is_binarised_at_128(tmp_path, level, masked):
    """A FLAT mask at one grey level: below 128 nothing is painted, at or above
    128 everything is. The same rule ``decode_mask_video`` applies on the other
    side of the worker pipe."""
    src, mask = tmp_path / "src.mp4", tmp_path / f"flat{level}.mp4"
    _write_source(src)
    _write_flat(mask, level=level)
    out, _ = _fill(tmp_path, src, mask)

    inner = _rgb(out, CANVAS_W, CANVAS_H)[:, :SRC_H, :SRC_W, :]
    if masked:
        assert (inner == np.array(GREEN, dtype=np.uint8)).all()
    else:
        # Stated as "identical to the source" rather than "no green anywhere":
        # ``testsrc``'s colour ramps happen to contain the sentinel triple, so
        # counting green pixels would be measuring the fixture, not the filter.
        source = _rgb(src, SRC_W, SRC_H)[:FRAMES].astype(int)
        assert np.abs(inner.astype(int) - source).max() == 0


@has_ffmpeg
def test_a_soft_mask_edge_never_produces_a_half_green_pixel(tmp_path):
    """The failure this guards against: ``maskedmerge`` is a LINEAR blend, so an
    un-binarised H.264 edge would produce a ring of pixels that are neither the
    source nor the sentinel — and the engine's de-green, which matches the exact
    triple, would leave every one of them in the delivered video."""
    src, mask = tmp_path / "src.mp4", tmp_path / "mask.mp4"
    _write_source(src)
    _write_mask(mask, lossy=True)  # a real yuv420p mask, soft edges and all
    out, _ = _fill(tmp_path, src, mask)

    canvas = _rgb(out, CANVAS_W, CANVAS_H)[:, :SRC_H, :SRC_W, :]
    source = _rgb(src, SRC_W, SRC_H)[:FRAMES]
    green = (canvas == np.array(GREEN, dtype=np.uint8)).all(axis=-1)
    same = (canvas == source).all(axis=-1)
    # Every pixel is EITHER the sentinel OR untouched source. Nothing in between.
    assert bool((green | same).all()), (
        f"{int((~(green | same)).sum())} pixels are neither sentinel nor source"
    )


# ── 3. the frame count, measured ────────────────────────────────────────────
@has_ffmpeg
def test_the_written_frame_count_is_measured_and_returned(tmp_path):
    src, mask = tmp_path / "src.mp4", tmp_path / "mask.mp4"
    _write_source(src)
    _write_mask(mask, lossy=False)
    out, info = _fill(tmp_path, src, mask)
    assert info["written_frames"] == FRAMES == video_io.frame_count(out)
    assert video_io.probe_resolution(out) == (CANVAS_W, CANVAS_H)


@has_ffmpeg
@pytest.mark.parametrize("short", ["mask", "source"])
def test_a_short_input_is_refused_before_the_merge_runs(tmp_path, short):
    """THE CASE THE POST-WRITE COUNT CANNOT CATCH. This ffmpeg's ``maskedmerge``
    declares only ``planes`` — ``shortest``/``repeatlast`` are rejected outright
    (pinned below) — so framesync repeats the last frame of whichever stream ran
    out and the file still reaches ``num_frames``: a short MASK leaves the tail
    masked by a stale frame, a short SOURCE leaves it frozen on a stale picture.
    Measuring what was written would report success in BOTH cases (verified: 4
    against 9 writes 9 either way), so both lengths are checked up front."""
    src, mask = tmp_path / "src.mp4", tmp_path / "mask.mp4"
    _write_source(src, frames=4 if short == "source" else FRAMES)
    _write_mask(mask, frames=4 if short == "mask" else FRAMES, lossy=False)
    out = tmp_path / "canvas.mp4"
    with pytest.raises(video_io.FFmpegError) as ei:
        _fill(tmp_path, src, mask)
    assert f"the {short} has 4 frames but 9 are needed" in str(ei.value)
    assert not out.exists(), "nothing may be written once an input is refused"


@has_ffmpeg
def test_framesync_really_does_pad_a_short_input(tmp_path):
    """The measured fact the check above exists for, pinned directly: with the
    guard bypassed, a 4-frame mask against a 9-frame source produces NINE
    frames, not four. If a future ffmpeg stops doing this, the guard becomes
    belt-and-braces rather than the only defence — worth knowing."""
    src, mask = tmp_path / "src.mp4", tmp_path / "mask.mp4"
    _write_source(src)
    _write_mask(mask, frames=4, lossy=False)
    out = tmp_path / "raw.mp4"
    graph = (
        f"[0:v]setpts=N/({FPS}*TB),format=rgb24,split[s][t];"
        f"[t]lutrgb=r=102:g=255:b=0[g];"
        f"[1:v]setpts=N/({FPS}*TB),format=gray,"
        f"lut=y='if(gte(val,128),255,0)',format=rgb24[m];"
        f"[s][g][m]maskedmerge[o]"
    )
    subprocess.run(
        [_ffmpeg(), "-y", "-v", "error", "-i", str(src), "-i", str(mask),
         "-filter_complex", graph, "-map", "[o]", "-an",
         "-fps_mode", "passthrough", "-frames:v", str(FRAMES),
         "-c:v", "libx264rgb", "-pix_fmt", "rgb24", "-crf", "0", str(out)],
        check=True, capture_output=True,
    )
    assert video_io.frame_count(out) == FRAMES


@has_ffmpeg
def test_this_ffmpeg_really_does_reject_the_framesync_options(tmp_path):
    """The measured fact the design rests on, pinned so a future ffmpeg upgrade
    that ADDS the options shows up as a failing test rather than as a silent
    change of what the code could have relied on."""
    proc = subprocess.run(
        [
            _ffmpeg(), "-hide_banner", "-v", "error",
            "-f", "lavfi", "-i", "color=c=red:s=64x64:d=0.2:r=24",
            "-f", "lavfi", "-i", "color=c=blue:s=64x64:d=0.2:r=24",
            "-f", "lavfi", "-i", "color=c=white:s=64x64:d=0.2:r=24",
            "-filter_complex", "[0:v][1:v][2:v]maskedmerge=shortest=1:repeatlast=0[o]",
            "-map", "[o]", "-frames:v", "2", "-f", "null", "-",
        ],
        capture_output=True, text=True,
    )
    assert proc.returncode != 0, (
        "maskedmerge now accepts shortest/repeatlast on this ffmpeg. The design "
        "note in services/video_io.fill_mask_green_mp4 and the endpoint's "
        "INPAINT_MASK_FRAME_MISMATCH check were written because it does not; "
        "revisit both before relying on the new behaviour."
    )
    assert "Option not found" in proc.stderr or "shortest" in proc.stderr


@has_ffmpeg
def test_num_frames_must_be_positive(tmp_path):
    src, mask = tmp_path / "src.mp4", tmp_path / "mask.mp4"
    _write_source(src)
    _write_mask(mask, lossy=False)
    with pytest.raises(video_io.FFmpegError, match="num_frames must be >= 1"):
        _fill(tmp_path, src, mask, num_frames=0)


# ── 4. an off-grid source, the case R8 names ────────────────────────────────
@has_ffmpeg
def test_a_1920x1080_shaped_source_gets_a_bottom_band_only(tmp_path):
    """The shape almost every real project has: the width is already on the grid
    and only the height needs a band. Scaled down here so the test is quick —
    the arithmetic is the same one ``round_up_128`` does."""
    src, mask = tmp_path / "src.mp4", tmp_path / "mask.mp4"
    _write_source(src, width=384, height=280)
    _write_mask(mask, width=384, height=280, lossy=False, rect=(32, 32, 160, 160))
    out, info = _fill(
        tmp_path, src, mask, canvas_width=384, canvas_height=384
    )
    assert info["written_frames"] == FRAMES
    frames = _rgb(out, 384, 384)
    assert (frames[:, 280:, :, :] == np.array(GREEN, dtype=np.uint8)).all()
    # ...and no right band at all, because 384 is already a multiple of 128.
    src_frames = _rgb(src, 384, 280)[:FRAMES].astype(int)
    keep = np.ones((280, 384), dtype=bool)
    keep[32:160, 32:160] = False
    assert np.abs(frames[:, :280, :, :].astype(int)[:, keep] - src_frames[:, keep]).max() == 0
