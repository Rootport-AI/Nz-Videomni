"""Laplacian pyramid blending — kornia-parity tests for the pure-torch port.

``engine.outpaint.pyramid_blend`` reimplements the four kornia primitives the
official ``LTXVLaplacianPyramidBlend`` node uses. The load-bearing test here is
the **golden comparison** against ``tests/fixtures/outpaint_pyramid_golden.npz``,
which was produced by running the upstream node's own ``_pyramid_blend`` on top
of a real kornia 0.8.3 install.

That fixture is not optional belt-and-braces. Self-consistency tests cannot
police this port: ``build_laplacian_pyramid`` and the reconstruction loop call
the *same* ``pyrup``, so a wrong gain factor (OpenCV's x4) or an unnormalised
kernel telescopes out of the sum and the identity checks below still pass
perfectly. Only comparing against externally generated numbers pins the
primitives down.

Run this file in the ENGINE venv -- the app ``.venv`` has no torch, so
``importorskip("torch")`` skips the whole module there:

  .venv-engine\\Scripts\\python.exe -m pytest tests\\test_outpaint_pyramid_blend.py ^
      -p no:warnings -p no:cacheprovider --noconftest --rootdir .
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from engine.outpaint.pyramid_blend import (  # noqa: E402
    _CHUNK_SIZE,
    _pad_for_laplacian,
    apply_low_res_mask_dilation,
    blend_video_u8,
    build_laplacian_pyramid,
    build_pyramid,
    find_next_powerof_two,
    is_powerof_two,
    laplacian_pyramid_blend,
    pyrdown,
    pyrup,
)

from engine.outpaint.canvas import (  # noqa: E402
    GREEN_RGB,
    OutpaintGeometry,
    build_blend_mask,
    fill_pad_with_generated_,
)

_GOLDEN = pathlib.Path(__file__).with_name("fixtures") / "outpaint_pyramid_golden.npz"


def _rect_mask(h: int, w: int, y0: int, y1: int, x0: int, x1: int):
    """Outpaint convention: 1.0 in the generated band, 0.0 in the kept rect."""
    m = torch.ones(1, 1, h, w)
    m[..., y0:y1, x0:x1] = 0.0
    return m


# ── 1. golden comparison against real kornia ────────────────────────────────
@pytest.mark.skipif(not _GOLDEN.exists(), reason=f"missing golden fixture {_GOLDEN}")
def test_matches_kornia_golden():
    """Bit-for-bit parity target with the upstream node (see module docstring)."""
    data = np.load(_GOLDEN)
    n_cases = sum(1 for k in data.files if k.endswith("_out"))
    assert n_cases == 3

    for i in range(n_cases):
        image_a = torch.from_numpy(data[f"case{i}_a"])
        image_b = torch.from_numpy(data[f"case{i}_b"])
        mask = torch.from_numpy(data[f"case{i}_mask"])
        dilation = int(data[f"case{i}_dilation"])
        expected = torch.from_numpy(data[f"case{i}_out"])

        assert mask.shape[0] == 1, "fixture stores the single-frame mask"

        got = laplacian_pyramid_blend(
            image_a, image_b, mask, max_level=7, mask_low_res_dilation=dilation
        )
        assert got.shape == expected.shape
        assert torch.allclose(got, expected, atol=1e-5), (
            f"case{i} max |diff| = {(got - expected).abs().max().item()}"
        )


# ── 2. degenerate masks reproduce their source exactly ──────────────────────
def test_all_white_mask_returns_image_a():
    torch.manual_seed(0)
    image_a = torch.rand(3, 3, 32, 32)
    image_b = torch.rand(3, 3, 32, 32)
    mask = torch.ones(1, 1, 32, 32)
    out = laplacian_pyramid_blend(image_a, image_b, mask)
    assert torch.allclose(out, image_a, atol=1e-4)


def test_all_black_mask_returns_image_b():
    torch.manual_seed(0)
    image_a = torch.rand(3, 3, 32, 32)
    image_b = torch.rand(3, 3, 32, 32)
    mask = torch.zeros(1, 1, 32, 32)
    out = laplacian_pyramid_blend(image_a, image_b, mask)
    assert torch.allclose(out, image_b, atol=1e-4)


# ── 3. non-power-of-two sizes ───────────────────────────────────────────────
@pytest.mark.parametrize("height,width", [(96, 80), (64, 80), (40, 24)])
def test_shape_and_identity_survive_non_power_of_two(height, width):
    torch.manual_seed(1)
    image_a = torch.rand(2, 3, height, width)
    image_b = torch.rand(2, 3, height, width)

    out = laplacian_pyramid_blend(image_a, image_b, torch.ones(1, 1, height, width))
    assert out.shape == image_a.shape
    assert torch.allclose(out, image_a, atol=1e-4)

    out = laplacian_pyramid_blend(image_a, image_b, torch.zeros(1, 1, height, width))
    assert out.shape == image_b.shape
    assert torch.allclose(out, image_b, atol=1e-4)


def test_pad_for_laplacian_uses_and_not_or():
    """64x80 has ONE power-of-two side; upstream still pads it (``and``).

    kornia's own internal guard is an ``or`` and would leave this untouched --
    pinning 64x128 here keeps the port on the node's side of that difference.
    """
    padded, padding = _pad_for_laplacian(torch.zeros(1, 3, 64, 80))
    assert padded.shape[-2:] == (64, 128)
    assert padding == (48, 0)  # (pad_right, pad_down)

    padded, padding = _pad_for_laplacian(torch.zeros(1, 3, 64, 128))
    assert padded.shape[-2:] == (64, 128)
    assert padding == (0, 0)


# ── 4. structured mask touching a frame edge ────────────────────────────────
def test_mask_rect_touching_the_right_edge():
    """A mask region flush against the right border, on a padded (48 -> 64) canvas.

    The right-hand 16 columns of the working canvas are pure reflect padding, so
    a wrong border mode (zeros / replicate) would bleed into the flush rectangle
    and pull it away from ``image_a``.
    """
    torch.manual_seed(1234)
    h = w = 48
    mask = torch.zeros(1, 1, h, w)
    mask[..., 8:24, 8:24] = 1.0  # interior rectangle
    mask[..., 30:40, 36:48] = 1.0  # flush against the right edge

    image_a = torch.rand(2, 3, h, w)
    image_b = torch.rand(2, 3, h, w)
    out = laplacian_pyramid_blend(image_a, image_b, mask, mask_low_res_dilation=0)

    def mean_err(target, ys, xs):
        return (out[:, :, ys, xs] - target[:, :, ys, xs]).abs().mean().item()

    interior = (slice(11, 21), slice(11, 21))
    flush = (slice(33, 38), slice(40, 48))
    outside = (slice(0, 6), slice(0, 48))

    assert mean_err(image_a, *interior) < 0.05
    assert mean_err(image_b, *interior) > 0.2
    assert mean_err(image_a, *flush) < 0.05
    assert mean_err(image_b, *flush) > 0.2
    assert mean_err(image_b, *outside) < 0.05
    assert mean_err(image_a, *outside) > 0.2

    assert out.min() >= 0.0 and out.max() <= 1.0


# ── 5. chunking does not change the result ──────────────────────────────────
def test_chunk_size_does_not_change_the_result():
    """CPU float32 is deterministic per chunk, so demand near-exact agreement.

    Not asserted on CUDA: a different batch size can select a different conv
    kernel there, so only CPU is expected to be bit-for-bit stable.
    """
    torch.manual_seed(5)
    frames = 20
    image_a = torch.rand(frames, 3, 64, 64)
    image_b = torch.rand(frames, 3, 64, 64)
    mask = _rect_mask(64, 64, 16, 48, 16, 48)

    chunked = laplacian_pyramid_blend(image_a, image_b, mask, chunk_size=_CHUNK_SIZE)
    single = laplacian_pyramid_blend(image_a, image_b, mask, chunk_size=frames)
    assert torch.allclose(chunked, single, atol=1e-6)


# ── 6. dilation grows the generated side inward ─────────────────────────────
def test_dilation_pulls_the_boundary_into_the_kept_region():
    """Larger dilation must move the seam inward, toward ``image_a``.

    This is the mechanism ``OutpaintGeometry.MIN_INNER_SIDE`` guards against:
    the radius is applied at a 64px long side, so at 128px it is worth 2x as
    many real pixels, and a small kept rectangle would be eaten whole.
    """
    torch.manual_seed(7)
    h = w = 128
    mask = _rect_mask(h, w, 32, 96, 32, 96)
    image_a = torch.rand(2, 3, h, w)
    image_b = torch.rand(2, 3, h, w)

    band = (slice(36, 92), slice(36, 52))  # just inside the kept rectangle
    core = (slice(56, 72), slice(56, 72))  # deep inside it

    band_errors = []
    core_errors = []
    for dilation in (0, 2, 5, 8):
        out = laplacian_pyramid_blend(
            image_a, image_b, mask, mask_low_res_dilation=dilation
        )
        band_errors.append(
            (out[:, :, band[0], band[1]] - image_a[:, :, band[0], band[1]])
            .abs()
            .mean()
            .item()
        )
        core_errors.append(
            (out[:, :, core[0], core[1]] - image_a[:, :, core[0], core[1]])
            .abs()
            .mean()
            .item()
        )

    assert band_errors == sorted(band_errors, reverse=True), band_errors
    assert band_errors[-1] < band_errors[0] * 0.5, band_errors
    # the deep interior is untouched by any of these radii
    assert max(core_errors) - min(core_errors) < 1e-3, core_errors


def test_dilation_zero_is_a_no_op():
    mask = _rect_mask(64, 64, 16, 48, 16, 48)
    assert apply_low_res_mask_dilation(mask, 0) is mask


def test_dilation_only_grows_the_mask():
    mask = _rect_mask(128, 128, 32, 96, 32, 96)
    grown = apply_low_res_mask_dilation(mask, 5)
    assert grown.shape == mask.shape
    assert (grown >= mask - 1e-6).all()
    assert grown.sum() > mask.sum()


# ── 7. uint8 NHWC front end agrees with the float NCHW one ──────────────────
def test_blend_video_u8_matches_the_float_path():
    torch.manual_seed(11)
    frames, h, w = 12, 64, 48
    generated = (torch.rand(frames, h, w, 3) * 255).to(torch.uint8)
    original = (torch.rand(frames, h, w, 3) * 255).to(torch.uint8)
    mask = _rect_mask(h, w, 12, 52, 8, 40)

    got = blend_video_u8(generated, original, mask)
    assert got.dtype == torch.uint8
    assert got.shape == generated.shape
    assert got.device.type == "cpu"

    reference = laplacian_pyramid_blend(
        generated.permute(0, 3, 1, 2).float() / 255.0,
        original.permute(0, 3, 1, 2).float() / 255.0,
        mask,
    ).permute(0, 2, 3, 1)

    assert torch.allclose(got.float() / 255.0, reference, atol=2 / 255)


def test_blend_video_u8_rejects_float_inputs():
    """A float input would be rescaled in place by the chunk conversion."""
    mask = _rect_mask(32, 32, 8, 24, 8, 24)
    with pytest.raises(ValueError, match="must be uint8"):
        blend_video_u8(
            torch.zeros(4, 32, 32, 3),
            torch.zeros(4, 32, 32, 3, dtype=torch.uint8),
            mask,
        )


def test_blend_video_u8_does_not_mutate_its_inputs():
    torch.manual_seed(13)
    generated = (torch.rand(4, 32, 32, 3) * 255).to(torch.uint8)
    original = (torch.rand(4, 32, 32, 3) * 255).to(torch.uint8)
    mask = _rect_mask(32, 32, 8, 24, 8, 24)
    before_a = generated.clone()
    before_b = original.clone()
    before_m = mask.clone()
    blend_video_u8(generated, original, mask)
    assert torch.equal(generated, before_a)
    assert torch.equal(original, before_b)
    # The mask too. Both engines now hand the SAME mask object to
    # ``fill_pad_with_generated_`` and then to this function, and the helper
    # derives the pad rectangle from it -- so the two calls must not be
    # order-dependent. (It holds today: the blend dilates its own copy.)
    assert torch.equal(mask, before_m)


def test_blend_video_u8_rejects_mismatched_shapes():
    mask = _rect_mask(32, 32, 8, 24, 8, 24)
    with pytest.raises(ValueError, match="same size"):
        blend_video_u8(
            torch.zeros(4, 32, 32, 3, dtype=torch.uint8),
            torch.zeros(5, 32, 32, 3, dtype=torch.uint8),
            mask,
        )


# ── 8. power-of-two helpers ─────────────────────────────────────────────────
@pytest.mark.parametrize(
    "value,expected",
    [(0, False), (1, True), (2, True), (3, False), (4, True), (63, False), (64, True), (80, False)],
)
def test_is_powerof_two(value, expected):
    assert is_powerof_two(value) is expected


@pytest.mark.parametrize(
    "value,expected", [(1, 1), (2, 2), (3, 4), (5, 8), (24, 32), (40, 64), (64, 64), (96, 128)]
)
def test_find_next_powerof_two(value, expected):
    assert find_next_powerof_two(value) == expected


# ── 9. pyramid structure ────────────────────────────────────────────────────
def test_laplacian_tail_is_the_gaussian_residual():
    torch.manual_seed(3)
    x = torch.rand(2, 3, 64, 64)
    gaussian = build_pyramid(x, 5)
    laplacian = build_laplacian_pyramid(x, 5)

    assert len(gaussian) == len(laplacian) == 5
    assert torch.equal(laplacian[-1], gaussian[-1])
    assert laplacian[0].shape == x.shape
    assert gaussian[-1].shape[-2:] == (4, 4)


def test_pyrdown_and_pyrup_shapes():
    x = torch.rand(1, 3, 64, 32)
    assert pyrdown(x).shape[-2:] == (32, 16)
    assert pyrup(x).shape[-2:] == (128, 64)


def test_pyramid_reconstructs_the_input():
    """Collapsing the Laplacian pyramid returns the original image.

    Necessary but NOT sufficient (see the module docstring): the same ``pyrup``
    appears on both sides, so this passes even with a wrong gain factor.
    """
    torch.manual_seed(4)
    x = torch.rand(1, 3, 64, 64)
    pyr = build_laplacian_pyramid(x, 5)
    out = pyr[-1]
    for i in range(len(pyr) - 2, -1, -1):
        out = pyrup(out) + pyr[i]
    assert torch.allclose(out, x, atol=1e-5)


# ── 10. input validation ────────────────────────────────────────────────────
def test_multi_frame_mask_is_rejected():
    image_a = torch.rand(2, 3, 32, 32)
    image_b = torch.rand(2, 3, 32, 32)
    with pytest.raises(ValueError, match="exactly one frame"):
        laplacian_pyramid_blend(image_a, image_b, torch.ones(2, 1, 32, 32))


def test_mismatched_image_shapes_are_rejected():
    with pytest.raises(ValueError, match="same size"):
        laplacian_pyramid_blend(
            torch.rand(2, 3, 32, 32), torch.rand(2, 3, 32, 16), torch.ones(1, 1, 32, 32)
        )


def test_mismatched_mask_resolution_is_rejected():
    with pytest.raises(ValueError, match="same spatial resolution"):
        laplacian_pyramid_blend(
            torch.rand(2, 3, 32, 32), torch.rand(2, 3, 32, 32), torch.ones(1, 1, 16, 16)
        )


def test_non_4d_mask_is_rejected():
    with pytest.raises(ValueError, match=r"shape \(1, 1, H, W\)"):
        laplacian_pyramid_blend(
            torch.rand(2, 3, 32, 32), torch.rand(2, 3, 32, 32), torch.ones(1, 32, 32)
        )


# ── canvas.build_blend_mask (torch half of the geometry module) ─────────────
def test_build_blend_mask_full_and_half_resolution():
    from engine.outpaint.canvas import OutpaintGeometry, build_blend_mask

    geom = OutpaintGeometry(
        canvas_width=1280,
        canvas_height=768,
        pad_left=128,
        pad_right=256,
        pad_top=64,
        pad_bottom=192,
    )
    geom.validate()

    full = build_blend_mask(geom, height=768, width=1280)
    assert full.shape == (1, 1, 768, 1280)
    assert full.dtype == torch.float32
    # kept rectangle is 0, the pad bands are 1
    assert full[0, 0, 64:256, 128:1024].max().item() == 0.0
    assert full[0, 0, :64, :].min().item() == 1.0
    assert full[0, 0, 576:, :].min().item() == 1.0
    assert full[0, 0, :, :128].min().item() == 1.0
    assert full[0, 0, :, 1024:].min().item() == 1.0

    half = build_blend_mask(geom, height=384, width=640)
    assert half.shape == (1, 1, 384, 640)
    assert half[0, 0, 32:288, 64:512].max().item() == 0.0
    assert half[0, 0, :32, :].min().item() == 1.0
    assert half[0, 0, :, 512:].min().item() == 1.0


# ── canvas.fill_pad_with_generated_ (the de-greened blend operand) ──────────
#
# The smallest canvas the geometry allows: both sides a multiple of 128 and a
# kept rectangle of at least 256px. All four pads are ODD and no two are equal,
# so an off-by-one in the rectangle derivation shows up as a wrong band rather
# than cancelling out.
_DEGREEN_GEOM = OutpaintGeometry(
    canvas_width=384,
    canvas_height=384,
    pad_left=63,
    pad_right=65,
    pad_top=61,
    pad_bottom=67,
)


def _green_canvas(frames: int, h: int, w: int):
    """A canvas painted the #66FF00 sentinel edge to edge, as ``pad_green_mp4``
    paints the pad bands before the source rectangle is composited over it."""
    canvas = torch.empty((frames, h, w, 3), dtype=torch.uint8)
    for channel, value in enumerate(GREEN_RGB):
        canvas[..., channel] = value
    return canvas


def _picture(frames: int, h: int, w: int, seed: int):
    torch.manual_seed(seed)
    return (torch.rand(frames, h, w, 3) * 255).to(torch.uint8)


def _pad_green_excess(video, mask) -> float:
    """``G - (R + B) / 2`` over the pad band -- the headline number of the spike
    report. Positive means the region leans green."""
    pad = mask[0, 0] > 0.5
    mean = video[:, pad, :].to(torch.float64).mean(dim=(0, 1))
    return float(mean[1] - (mean[0] + mean[2]) / 2.0)


def test_fill_pad_replaces_every_pad_pixel_and_leaves_the_kept_rect_alone():
    """T1. Odd pads on every side; the pad band must end up bit-identical to the
    generated video and the kept rectangle bit-identical to what it was."""
    _DEGREEN_GEOM.validate()
    mask = build_blend_mask(_DEGREEN_GEOM, height=384, width=384)
    canvas = _green_canvas(3, 384, 384)
    generated = _picture(3, 384, 384, seed=101)
    before = canvas.clone()

    fill_pad_with_generated_(canvas, generated=generated, mask=mask)

    pad = mask[0, 0] > 0.5
    keep = ~pad
    assert torch.equal(canvas[:, pad, :], generated[:, pad, :])
    assert torch.equal(canvas[:, keep, :], before[:, keep, :])
    # and the kept rectangle really is the geometry's, not a shrunken one
    assert int(keep.sum()) == _DEGREEN_GEOM.inner_width * _DEGREEN_GEOM.inner_height


def test_fill_pad_is_in_place_and_allocates_no_second_canvas():
    """T2. The engines pass a local that is blended and then ``del``'d, so the
    write must land in the caller's own storage: same object, same data_ptr."""
    mask = build_blend_mask(_DEGREEN_GEOM, height=384, width=384)
    canvas = _green_canvas(2, 384, 384)
    generated = _picture(2, 384, 384, seed=102)
    pointer = canvas.data_ptr()

    returned = fill_pad_with_generated_(canvas, generated=generated, mask=mask)

    assert returned is canvas
    assert returned.data_ptr() == pointer


def test_fill_pad_follows_the_half_resolution_mask():
    """T3. Stage 1 blends at half resolution against a mask whose rectangle is
    PROPORTIONALLY PROJECTED, not resampled. The bands the helper writes must be
    that projection's, to the pixel."""
    mask_half = build_blend_mask(_DEGREEN_GEOM, height=192, width=192)
    canvas = _green_canvas(2, 192, 192)
    generated = _picture(2, 192, 192, seed=103)
    before = canvas.clone()

    fill_pad_with_generated_(canvas, generated=generated, mask=mask_half)

    # round(63 * 192/384) = 32, round((63+256) * 192/384) = 160, and the same
    # pair vertically: round(61/2) = 30, round((61+256)/2) = 158.
    y0, y1, x0, x1 = 30, 158, 32, 160
    assert mask_half[0, 0, y0:y1, x0:x1].max().item() == 0.0
    assert int((mask_half[0, 0] == 0.0).sum()) == (y1 - y0) * (x1 - x0)
    assert torch.equal(canvas[:, y0:y1, x0:x1, :], before[:, y0:y1, x0:x1, :])
    pad = mask_half[0, 0] > 0.5
    assert torch.equal(canvas[:, pad, :], generated[:, pad, :])


def test_fill_pad_does_not_touch_the_generated_video():
    """T4. The generated tensor is the blend's other operand and is read again
    immediately afterwards; writing into it would corrupt the blend itself."""
    mask = build_blend_mask(_DEGREEN_GEOM, height=384, width=384)
    canvas = _green_canvas(2, 384, 384)
    generated = _picture(2, 384, 384, seed=104)
    before = generated.clone()

    fill_pad_with_generated_(canvas, generated=generated, mask=mask)

    assert torch.equal(generated, before)


def test_fill_pad_rejects_a_mask_of_the_wrong_resolution():
    """T5. A half-resolution mask against a full-resolution canvas is the exact
    mistake the two call sites per engine invite, and it must be named."""
    canvas = _green_canvas(1, 384, 384)
    generated = _picture(1, 384, 384, seed=105)
    wrong = build_blend_mask(_DEGREEN_GEOM, height=192, width=192)
    with pytest.raises(ValueError, match="same spatial resolution"):
        fill_pad_with_generated_(canvas, generated=generated, mask=wrong)


def test_the_blend_of_a_de_greened_canvas_returns_the_generated_video_exactly():
    """T6a -- the permanent green regression gate.

    This fixture is ARTIFICIAL on purpose: the canvas is the generated video
    everywhere except the pad, so once the helper has run the two blend operands
    are bit-identical over the whole frame and the blend has nothing left to do.
    That makes the expected output exact rather than approximate -- and it is the
    strongest possible statement of the property under test, because the moment
    the helper stops replacing the pad the equality breaks by a wide margin (the
    control arm below). A production-shaped version of the same claim, where the
    two operands still differ inside the kept rectangle, is the next test.
    """
    mask = build_blend_mask(_DEGREEN_GEOM, height=384, width=384)
    generated = _picture(2, 384, 384, seed=106)

    canvas = generated.clone()
    pad = mask[0, 0] > 0.5
    for channel, value in enumerate(GREEN_RGB):
        canvas[:, pad, channel] = value

    de_greened = canvas.clone()
    fill_pad_with_generated_(de_greened, generated=generated, mask=mask)
    blended = blend_video_u8(generated, de_greened, mask, mask_low_res_dilation=5)
    assert torch.equal(blended, generated)

    # Control arm: the same fixture WITHOUT the helper. If this ever passes too,
    # the test above has stopped proving anything.
    plain = blend_video_u8(generated, canvas, mask, mask_low_res_dilation=5)
    assert not torch.equal(plain, generated)


def test_the_de_greened_blend_drops_the_pad_bands_green_by_an_order_of_magnitude():
    """T6b -- the production-shaped version of T6a.

    Green canvas, and a kept rectangle holding a DIFFERENT picture from the
    generated one, which is the real situation: source footage on one side of the
    seam, the model's frame on the other. The blend still mixes the canvas in --
    the coarse pyramid levels guarantee it -- so the claim is not "zero" but "the
    sentinel is no longer what mixes in". Measured as green excess over the pad
    band, which is the spike report's headline number; a per-pixel bound is not
    used, because individual pixels move by up to 132 codes for reasons that have
    nothing to do with green.
    """
    mask = build_blend_mask(_DEGREEN_GEOM, height=384, width=384)
    generated = _picture(2, 384, 384, seed=107)
    keep = ~(mask[0, 0] > 0.5)

    canvas = _green_canvas(2, 384, 384)
    footage = _picture(2, 384, 384, seed=207)
    canvas[:, keep, :] = footage[:, keep, :]

    de_greened = canvas.clone()
    fill_pad_with_generated_(de_greened, generated=generated, mask=mask)

    before = _pad_green_excess(
        blend_video_u8(generated, canvas, mask, mask_low_res_dilation=5), mask
    )
    after = _pad_green_excess(
        blend_video_u8(generated, de_greened, mask, mask_low_res_dilation=5), mask
    )

    assert before > 1.0, "the control arm must actually be green, or nothing is proven"
    assert abs(after) < abs(before) / 10.0
