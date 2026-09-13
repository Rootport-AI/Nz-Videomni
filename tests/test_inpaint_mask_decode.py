"""``engine.pipeline.common.decode_mask_video`` — the mask's way in (台帳 §3-55).

``.venv-engine`` ONLY — the LTX 2.3 engine venv, not the 2.5 one. Inpainting is
an LTX 2.3 feature (``services/engines/ltx25/adapter.py``'s ``REJECT_TABLE``
says so), and ``engine/pipeline/common.py`` sits on 2.3's wheel: the 2.5 venv
ships a ``decode_video_from_file`` with a different signature, so running this
file there would report a wheel difference, not a bug. The geometry tests
(``test_inpaint_geometry.py``) DO belong in both, because the blend they feed is
shared.

Unlike most engine-side tests this one also needs ffmpeg: the whole point of the
function is what survives a real video round trip, so the fixtures here are
actual mp4 files written by ffmpeg rather than synthetic tensors. A mask that has
never been through H.264 cannot demonstrate why the binarisation exists.

  .venv-engine\\Scripts\\python.exe -m pytest tests\\test_inpaint_mask_decode.py ^
      -p no:warnings -p no:cacheprovider --noconftest --rootdir .
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

torch = pytest.importorskip("torch", reason="engine venv only")
pytest.importorskip("ltx_core", reason="decode_mask_video goes through the wheel's media_io")

from engine.pipeline.common import decode_mask_video  # noqa: E402

has_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="the fixtures are real mp4 files"
)

# Both sides clear ``INPAINT_MIN_SOURCE_SIDE`` (256) so the geometry in the last
# test is a legal one, and the height is deliberately NOT a multiple of 128
# (288 -> a 384 canvas, a 96px pad band) so that test exercises a real band.
W, H, N = 384, 288, 5
RECT = (96, 72, 288, 216)  # x0, y0, x1, y1 (half-open), all even


def _write_rect_mask(path, *, width=W, height=H, frames=N, lossless=False) -> None:
    """A white rectangle on black, encoded the way the plugin's writer would."""
    x0, y0, x1, y1 = RECT
    codec = (
        ["-c:v", "libx264rgb", "-pix_fmt", "rgb24", "-crf", "0"]
        if lossless
        else ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    )
    subprocess.run(
        [
            shutil.which("ffmpeg"), "-y", "-v", "error",
            "-f", "lavfi",
            "-i", f"color=c=black:s={width}x{height}:rate=24:duration=2",
            "-vf",
            f"drawbox=x={x0}:y={y0}:w={x1 - x0}:h={y1 - y0}:color=white@1.0:t=fill",
            "-frames:v", str(frames), *codec, str(path),
        ],
        check=True, capture_output=True,
    )


def _write_grey_ramp(path, *, level: int, width=W, height=H, frames=2) -> None:
    """A FLAT grey field at a known 8-bit level — the threshold probe."""
    hexc = f"0x{level:02X}{level:02X}{level:02X}"
    subprocess.run(
        [
            shutil.which("ffmpeg"), "-y", "-v", "error",
            "-f", "lavfi",
            "-i", f"color=c={hexc}:s={width}x{height}:rate=24:duration=1",
            "-frames:v", str(frames),
            "-c:v", "libx264rgb", "-pix_fmt", "rgb24", "-crf", "0",
            str(path),
        ],
        check=True, capture_output=True,
    )


@has_ffmpeg
def test_a_rectangle_mask_decodes_to_exactly_that_rectangle(tmp_path):
    path = tmp_path / "mask.mp4"
    _write_rect_mask(path, lossless=True)
    out = decode_mask_video(
        str(path), num_frames=N, height=H, width=W, device=torch.device("cpu")
    )
    assert tuple(out.shape) == (N, 1, H, W)
    assert out.dtype == torch.uint8
    assert set(out.unique().tolist()) == {0, 255}, "binary, never a grey in between"

    x0, y0, x1, y1 = RECT
    assert int(out[:, :, y0:y1, x0:x1].min()) == 255
    assert int(out[:, :, :y0, :].max()) == 0
    assert int(out[:, :, y1:, :].max()) == 0


@has_ffmpeg
def test_a_lossy_mask_still_decodes_to_two_values(tmp_path):
    """The reason the threshold exists. H.264's 4:2:0 chroma and its deblocking
    filter leave a grey ring around the rectangle; without binarisation that ring
    would reach the blend as fractional mask values and the de-green as
    not-quite-green pixels."""
    path = tmp_path / "lossy.mp4"
    _write_rect_mask(path, lossless=False)
    out = decode_mask_video(
        str(path), num_frames=N, height=H, width=W, device=torch.device("cpu")
    )
    assert set(out.unique().tolist()) <= {0, 255}
    # The interior and the far exterior are unambiguous even after the encode.
    x0, y0, x1, y1 = RECT
    assert int(out[:, :, y0 + 4 : y1 - 4, x0 + 4 : x1 - 4].min()) == 255
    assert int(out[:, :, :4, :4].max()) == 0


@has_ffmpeg
@pytest.mark.parametrize(
    "level,expected",
    [
        (0, 0),
        (127, 0),      # just below the threshold -> black
        (128, 255),    # AT the threshold -> white (>= 128, the same rule the
        (129, 255),    # ffmpeg filtergraph's lut uses: if(gte(val,128),255,0))
        (255, 255),
    ],
)
def test_the_threshold_is_gte_128(tmp_path, level, expected):
    path = tmp_path / f"grey{level}.mp4"
    _write_grey_ramp(path, level=level)
    out = decode_mask_video(
        str(path), num_frames=2, height=H, width=W, device=torch.device("cpu")
    )
    assert int(out.min()) == int(out.max()) == expected


@has_ffmpeg
def test_the_threshold_is_a_parameter_not_a_constant(tmp_path):
    """Not exposed through the API, but the GPU gate has to be able to move it
    without editing the module."""
    path = tmp_path / "grey100.mp4"
    _write_grey_ramp(path, level=100)
    dev = torch.device("cpu")
    assert int(decode_mask_video(str(path), num_frames=2, height=H, width=W, device=dev).max()) == 0
    assert (
        int(
            decode_mask_video(
                str(path), num_frames=2, height=H, width=W, device=dev, threshold=64
            ).min()
        )
        == 255
    )


@has_ffmpeg
def test_a_mask_of_the_wrong_resolution_is_rejected_not_resized(tmp_path):
    """The contract (Docs/INPAINTING_DESIGN.md §6.2): the mask is never resized,
    because rescaling moves the boundary by an amount the user cannot see."""
    path = tmp_path / "small.mp4"
    _write_rect_mask(path, width=256, height=192, lossless=True)
    with pytest.raises(ValueError, match="never resized"):
        decode_mask_video(
            str(path), num_frames=N, height=H, width=W, device=torch.device("cpu")
        )


@has_ffmpeg
def test_a_short_mask_is_rejected_rather_than_padded(tmp_path):
    """A short mask would leave the tail of the window unrepainted. The endpoint
    checks the frame count before a job exists; this is the backstop on the other
    side of the worker pipe."""
    path = tmp_path / "short.mp4"
    _write_rect_mask(path, frames=3, lossless=True)
    with pytest.raises(ValueError, match="decoded 3 frames"):
        decode_mask_video(
            str(path), num_frames=N, height=H, width=W, device=torch.device("cpu")
        )


@has_ffmpeg
def test_a_long_mask_is_capped_at_num_frames(tmp_path):
    """The other direction is NOT an error here: ``frame_cap`` stops the decode
    at ``num_frames``, so extra material is simply never read. The API refuses
    the mismatch before it gets this far — this only pins that the engine does
    not read, and then hold, a longer mask than the job needs."""
    path = tmp_path / "long.mp4"
    _write_rect_mask(path, frames=N + 6, lossless=True)
    out = decode_mask_video(
        str(path), num_frames=N, height=H, width=W, device=torch.device("cpu")
    )
    assert out.shape[0] == N


def test_num_frames_must_be_positive(tmp_path):
    with pytest.raises(ValueError, match="num_frames must be >= 1"):
        decode_mask_video(
            str(tmp_path / "nothing.mp4"),
            num_frames=0,
            height=H,
            width=W,
            device=torch.device("cpu"),
        )


@has_ffmpeg
def test_the_decoded_mask_feeds_the_blend_and_the_geometry_unchanged(tmp_path):
    """The three pieces meet here: decode -> place on the canvas -> half
    resolution. Each is tested on its own elsewhere; this is the one place the
    dtype/value convention (uint8 0/255) is shown to survive all three."""
    from engine.inpaint.canvas import InpaintGeometry, half_res_mask, place_mask_on_canvas

    path = tmp_path / "mask.mp4"
    _write_rect_mask(path, lossless=True)
    mask = decode_mask_video(
        str(path), num_frames=N, height=H, width=W, device=torch.device("cpu")
    )
    geom = InpaintGeometry(
        canvas_width=384, canvas_height=384, source_width=W, source_height=H
    )
    geom.validate()
    on_canvas = place_mask_on_canvas(mask, geom)
    assert tuple(on_canvas.shape) == (N, 1, 384, 384)
    assert int(on_canvas[:, :, H:, :].max()) == 0, "the pad band is never repainted"

    half = half_res_mask(on_canvas, 192, 192)
    assert half.dtype == torch.uint8
    assert set(half.unique().tolist()) <= {0, 255}
    x0, y0, x1, y1 = RECT
    assert int(half[:, :, y0 // 2 + 1 : y1 // 2 - 1, x0 // 2 + 1 : x1 // 2 - 1].min()) == 255
