"""Inpainting canvas geometry and the per-frame mask helpers (台帳 §3-55).

``engine/inpaint/canvas.py`` is importable WITHOUT torch, so the pure-integer
half of this file runs in all three venvs — the app ``.venv`` included. The
tensor half skips itself where torch is absent, which is what lets one file be
the single home for "the geometry is right".

Run it in the app venv with the rest of ``tests``; in either engine venv with:

  .venv-engine\\Scripts\\python.exe -m pytest tests\\test_inpaint_geometry.py ^
      -p no:warnings -p no:cacheprovider --noconftest --rootdir .
"""

from __future__ import annotations

import pytest

from engine.inpaint.canvas import (
    CANVAS_MULTIPLE,
    GREEN_RGB,
    INPAINT_MIN_SOURCE_SIDE,
    InpaintGeometry,
    round_up_128,
)

try:  # torch lives only in the engine venvs; the integer half below never needs it
    import torch

    from engine.inpaint.canvas import (
        fill_mask_with_generated_,
        fill_pad_bands_with_generated_,
        half_res_mask,
        place_mask_on_canvas,
        restore_and_measure_,
        restore_outside_mask_,
    )
except ImportError:  # pragma: no cover - the app venv has no torch
    torch = None

#: Applied to every test that builds a tensor. The integer tests carry no marker
#: and therefore run in ALL THREE venvs, which is the point of keeping one file:
#: ``round_up_128`` and ``InpaintGeometry`` are the geometry contract, and the
#: app side and the engine side have to agree about them.
needs_torch = pytest.mark.skipif(
    torch is None, reason="the tensor helpers need an engine venv"
)


# ── 1. round_up_128 ─────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "value,expected",
    [
        (128, 128),      # already on the grid: unchanged, not bumped
        (1280, 1280),    # 10 x 128, the common project width
        (768, 768),      # 6 x 128
        (1920, 1920),    # 15 x 128
        (1080, 1152),    # the one everybody hits: 1080 -> 1152, an 72px band
        (1, 128),
        (129, 256),
    ],
)
def test_round_up_128(value, expected):
    assert round_up_128(value) == expected


def test_round_up_128_refuses_a_nonpositive_size():
    with pytest.raises(ValueError, match="positive"):
        round_up_128(0)


def test_the_app_side_twin_agrees_everywhere_it_matters():
    """``api.models.round_up_128`` is a DUPLICATE (the app venv and the engine
    venv never import each other), so the two are held against each other here
    over every size the feature can see. Skipped in an engine venv, which has no
    app package."""
    api_models = pytest.importorskip("api.models", reason="app venv only")
    for value in list(range(256, 4097, 1)):
        assert api_models.round_up_128(value) == round_up_128(value), value


# ── 2. InpaintGeometry ──────────────────────────────────────────────────────
def _geom(sw=1280, sh=768, cw=None, ch=None) -> InpaintGeometry:
    return InpaintGeometry(
        canvas_width=cw if cw is not None else round_up_128(sw),
        canvas_height=ch if ch is not None else round_up_128(sh),
        source_width=sw,
        source_height=sh,
    )


def test_pads_are_derived_not_stored():
    g = _geom(1920, 1080)
    assert (g.canvas_width, g.canvas_height) == (1920, 1152)
    assert g.pad_right == 0
    assert g.pad_bottom == 72


def test_a_source_already_on_the_grid_has_no_pads_at_all():
    """The common case, and the one outpainting's geometry REFUSES: there,
    zero pad on every side means "nothing to generate". Here it is normal —
    the region to repaint is inside the frame, not around it."""
    g = _geom(1280, 768)
    g.validate()
    assert g.pad_right == 0 and g.pad_bottom == 0


def test_as_dict_carries_the_four_stored_values_and_the_two_derived_ones():
    assert _geom(1920, 1080).as_dict() == {
        "canvas_width": 1920,
        "canvas_height": 1152,
        "source_width": 1920,
        "source_height": 1080,
        "pad_right": 0,
        "pad_bottom": 72,
    }


@pytest.mark.parametrize("size", [1280, 1920, 640, 512, 384, 256])
def test_validate_accepts_every_canvas_derived_from_round_up_128(size):
    _geom(size, size).validate()


def test_validate_rejects_a_canvas_off_the_128_grid():
    with pytest.raises(ValueError, match="canvas_width must be a multiple"):
        InpaintGeometry(
            canvas_width=1290, canvas_height=768, source_width=1280, source_height=768
        ).validate()
    with pytest.raises(ValueError, match="canvas_height must be a multiple"):
        InpaintGeometry(
            canvas_width=1280, canvas_height=770, source_width=1280, source_height=768
        ).validate()


def test_validate_rejects_a_source_below_the_floor():
    small = INPAINT_MIN_SOURCE_SIDE - 2
    with pytest.raises(ValueError, match="source_width must be >="):
        InpaintGeometry(
            canvas_width=256, canvas_height=256, source_width=small, source_height=256
        ).validate()
    with pytest.raises(ValueError, match="source_height must be >="):
        InpaintGeometry(
            canvas_width=256, canvas_height=256, source_width=256, source_height=small
        ).validate()


def test_validate_rejects_a_pad_that_is_not_the_round_up():
    """A pad of 128 or more means the canvas was NOT the source rounded up —
    somebody computed it from something else, and the crop at the end would take
    the wrong rectangle."""
    with pytest.raises(ValueError, match="pad_right must be in"):
        InpaintGeometry(
            canvas_width=1536, canvas_height=768, source_width=1280, source_height=768
        ).validate()
    with pytest.raises(ValueError, match="pad_bottom must be in"):
        InpaintGeometry(
            canvas_width=1280, canvas_height=1024, source_width=1280, source_height=768
        ).validate()


def test_validate_rejects_a_negative_pad():
    """A source LARGER than its canvas is the same mistake read the other way."""
    with pytest.raises(ValueError, match="pad_right must be in"):
        InpaintGeometry(
            canvas_width=1280, canvas_height=768, source_width=1408, source_height=768
        ).validate()


def test_validate_rejects_an_odd_source_side():
    """Stage 1 runs at exactly half resolution; an odd side has no half."""
    with pytest.raises(ValueError, match="source_width must be even"):
        InpaintGeometry(
            canvas_width=1280, canvas_height=768, source_width=1279, source_height=768
        ).validate()
    with pytest.raises(ValueError, match="source_height must be even"):
        InpaintGeometry(
            canvas_width=1280, canvas_height=768, source_width=1280, source_height=767
        ).validate()


def test_half_dims_halves_all_four():
    assert _geom(1280, 768).half_dims() == (640, 384, 640, 384)
    assert _geom(1920, 1080).half_dims() == (960, 576, 960, 540)


def test_half_dims_is_exact_for_every_geometry_validate_accepts():
    """``validate`` already guarantees all four sides are even, so halving can
    never lose a pixel. Swept rather than asserted once, because that guarantee
    is the ONLY thing standing between stage 1 and an off-by-one de-green."""
    for sw in range(256, 2049, 2):
        for sh in (256, 768, 1080, 1088):
            g = _geom(sw, sh)
            g.validate()
            cw, ch, ssw, ssh = g.half_dims()
            assert (cw * 2, ch * 2, ssw * 2, ssh * 2) == (
                g.canvas_width, g.canvas_height, g.source_width, g.source_height
            )


def test_the_sentinel_and_the_grid_come_from_the_outpaint_module():
    """One green, one grid. Re-declaring either would let the two features drift
    while every test still passed."""
    from engine.outpaint.canvas import CANVAS_MULTIPLE as OUT_MULT
    from engine.outpaint.canvas import GREEN_RGB as OUT_GREEN

    assert GREEN_RGB is OUT_GREEN and GREEN_RGB == (102, 255, 0)
    assert CANVAS_MULTIPLE is OUT_MULT and CANVAS_MULTIPLE == 128


# ── 3. place_mask_on_canvas ─────────────────────────────────────────────────
def _mask(frames, h, w, fill=255):
    return torch.full((frames, 1, h, w), fill, dtype=torch.uint8)


@needs_torch
def test_place_mask_pads_with_zeros_and_keeps_the_source_rectangle():
    g = _geom(1920, 1080)
    m = _mask(3, 1080, 1920)
    out = place_mask_on_canvas(m, g)
    assert tuple(out.shape) == (3, 1, 1152, 1920)
    assert out.dtype == torch.uint8
    assert torch.equal(out[:, :, :1080, :1920], m)
    assert int(out[:, :, 1080:, :].max()) == 0


@needs_torch
def test_place_mask_is_idempotent_on_a_canvas_sized_mask():
    g = _geom(1920, 1080)
    already = _mask(2, 1152, 1920)
    assert place_mask_on_canvas(already, g) is already


@needs_torch
def test_place_mask_rejects_a_mask_that_is_neither_size():
    g = _geom(1920, 1080)
    with pytest.raises(ValueError, match="source resolution"):
        place_mask_on_canvas(_mask(2, 720, 1280), g)


# ── 4. half_res_mask ────────────────────────────────────────────────────────
@needs_torch
def test_half_res_mask_keeps_a_one_pixel_line():
    """The reason area interpolation is used rather than nearest: a 1px line
    covers half of the 2x1 footprint in the axis it is thin in, so a >= 0.5
    majority keeps it. Nearest would keep or drop it depending on parity."""
    m = torch.zeros(1, 1, 64, 64, dtype=torch.uint8)
    m[..., :, 31] = 255  # one vertical line, odd column
    out = half_res_mask(m, 32, 32)
    assert out.dtype == torch.uint8
    assert int(out.max()) == 255
    assert int((out > 0).sum()) == 32, "the whole line survives, one column of it"

    m2 = torch.zeros(1, 1, 64, 64, dtype=torch.uint8)
    m2[..., 30, :] = 255  # one horizontal line, even row
    assert int((half_res_mask(m2, 32, 32) > 0).sum()) == 32


@needs_torch
def test_half_res_mask_is_a_majority_vote_not_a_sample():
    """A 2x2 block with ONE white pixel is 25% white -> black; with two it is
    50% -> white. That is the rule, spelled out on the smallest case."""
    quarter = torch.zeros(1, 1, 2, 2, dtype=torch.uint8)
    quarter[0, 0, 0, 0] = 255
    assert int(half_res_mask(quarter, 1, 1).max()) == 0

    half = torch.zeros(1, 1, 2, 2, dtype=torch.uint8)
    half[0, 0, 0, :] = 255
    assert int(half_res_mask(half, 1, 1).max()) == 255


@needs_torch
def test_half_res_mask_accepts_float_and_uint8_alike():
    m = torch.zeros(2, 1, 32, 32, dtype=torch.uint8)
    m[..., 8:24, 8:24] = 255
    as_float = (m.to(torch.float32) / 255.0)
    assert torch.equal(half_res_mask(m, 16, 16), half_res_mask(as_float, 16, 16))


@needs_torch
def test_half_res_mask_of_a_solid_mask_is_solid():
    assert int(half_res_mask(_mask(2, 32, 32), 16, 16).min()) == 255


# ── 5. the two de-green helpers ─────────────────────────────────────────────
def _video(frames, h, w, value):
    return torch.full((frames, h, w, 3), value, dtype=torch.uint8)


@needs_torch
def test_fill_pad_bands_touches_only_the_right_and_bottom_bands():
    canvas = _video(2, 128, 128, 10)
    generated = _video(2, 128, 128, 200)
    fill_pad_bands_with_generated_(canvas, generated=generated, source_height=100, source_width=96)
    assert int(canvas[:, :100, :96, :].max()) == 10, "the source rectangle is untouched"
    assert int(canvas[:, :100, 96:, :].min()) == 200, "right band"
    assert int(canvas[:, 100:, :, :].min()) == 200, "bottom band, full width"


@needs_torch
def test_fill_pad_bands_is_a_no_op_when_there_are_no_bands():
    canvas = _video(2, 128, 128, 10)
    before = canvas.clone()
    fill_pad_bands_with_generated_(
        canvas, generated=_video(2, 128, 128, 200), source_height=128, source_width=128
    )
    assert torch.equal(canvas, before)


@needs_torch
def test_fill_mask_replaces_exactly_the_white_pixels():
    canvas = _video(3, 32, 32, 10)
    generated = _video(3, 32, 32, 200)
    mask = torch.zeros(3, 1, 32, 32, dtype=torch.uint8)
    for i in range(3):
        mask[i, 0, 4 + i : 12 + i, 4:12] = 255
    fill_mask_with_generated_(canvas, generated=generated, mask=mask, chunk_size=2)
    for i in range(3):
        assert int(canvas[i, 4 + i : 12 + i, 4:12, :].min()) == 200
        assert int(canvas[i, 20:, 20:, :].max()) == 10


@needs_torch
def test_fill_mask_accepts_a_float_mask_and_a_single_plane():
    canvas = _video(3, 32, 32, 10)
    plane = torch.zeros(1, 1, 32, 32, dtype=torch.float32)
    plane[..., 8:16, 8:16] = 1.0
    fill_mask_with_generated_(canvas, generated=_video(3, 32, 32, 200), mask=plane)
    assert int(canvas[:, 8:16, 8:16, :].min()) == 200
    assert int(canvas[:, 0:8, 0:8, :].max()) == 10


@needs_torch
def test_fill_mask_rejects_a_mask_frame_count_that_is_neither_one_nor_f():
    canvas = _video(3, 32, 32, 10)
    with pytest.raises(ValueError, match="1 or 3 frames"):
        fill_mask_with_generated_(
            canvas, generated=_video(3, 32, 32, 200), mask=_mask(2, 32, 32)
        )


@needs_torch
def test_the_chunk_size_does_not_change_the_de_green():
    args = dict(generated=_video(5, 32, 32, 200), mask=_mask(5, 32, 32, 0))
    args["mask"][:, :, 8:24, 8:24] = 255
    a = _video(5, 32, 32, 10)
    b = _video(5, 32, 32, 10)
    fill_mask_with_generated_(a, chunk_size=1, **args)
    fill_mask_with_generated_(b, chunk_size=5, **args)
    assert torch.equal(a, b)


# ── 6. restore_outside_mask_ ────────────────────────────────────────────────
@needs_torch
def test_restore_puts_back_exactly_the_pixels_the_dilation_never_reached():
    blended = _video(2, 32, 32, 200)
    source = _video(2, 32, 32, 10)
    dilated = torch.zeros(2, 1, 32, 32, dtype=torch.float32)
    dilated[..., 8:24, 8:24] = 1.0
    dilated[..., 7, 7] = 1e-3  # inside the ramp: NOT restored

    restore_outside_mask_(blended, source=source, dilated_mask=dilated)
    assert int(blended[:, 8:24, 8:24, :].min()) == 200, "the mask itself stays generated"
    assert int(blended[:, 7, 7, :].min()) == 200, "the ramp stays generated too"
    assert int(blended[:, 0, 0, :].max()) == 10, "hard zero -> original"


@needs_torch
def test_restore_of_an_all_zero_mask_gives_the_source_back_exactly():
    blended = _video(3, 32, 32, 200)
    source = _video(3, 32, 32, 10)
    restore_outside_mask_(
        blended, source=source, dilated_mask=torch.zeros(3, 1, 32, 32)
    )
    assert torch.equal(blended, source)


@needs_torch
def test_restore_of_an_all_one_mask_changes_nothing():
    blended = _video(3, 32, 32, 200)
    before = blended.clone()
    restore_outside_mask_(
        blended, source=_video(3, 32, 32, 10), dilated_mask=torch.ones(3, 1, 32, 32)
    )
    assert torch.equal(blended, before)


@needs_torch
def test_restore_is_chunk_size_independent():
    def run(step):
        blended = _video(5, 32, 32, 200)
        dil = torch.zeros(5, 1, 32, 32)
        for i in range(5):
            dil[i, 0, 4 + i : 20 + i, 4:20] = 1.0
        restore_outside_mask_(
            blended, source=_video(5, 32, 32, 10), dilated_mask=dil, chunk_size=step
        )
        return blended

    assert torch.equal(run(1), run(5))


# ── 7. restore_and_measure_ — one definition, both engines ──────────────────
def test_the_2_3_pipeline_calls_THIS_modules_restore_and_measure_():
    """The restore-and-measure pass lives here, next to the mask helpers it is
    made of, because the LTX 2.5 inpaint driver cannot import
    ``engine.pipeline.inpaint_pipeline`` (that module is bound to the 2.3
    wheel). 2.3 now imports it back under its old private name, so this
    identity is what says the move was a move and not a fork — two copies of a
    ramp rule would let the two engines restore different pixels."""
    pipeline = pytest.importorskip(
        "engine.pipeline.inpaint_pipeline",
        reason="the 2.3 pipeline needs an engine venv",
    )
    from engine.inpaint import canvas

    assert pipeline._restore_and_measure_ is canvas.restore_and_measure_


@needs_torch
def test_restore_and_measure_puts_the_outside_back_and_reports_honest_ratios():
    """``dilation=0`` makes the dilation the identity, so the two ratios have to
    agree and both are a quarter of the frame — which is also what makes the
    ratios checkable without pinning the dilation's resampling."""
    blended = _video(4, 32, 32, 200)
    source = _video(4, 32, 32, 10)
    mask = _mask(4, 32, 32, 0)
    mask[:, :, 8:24, 8:24] = 255

    proof = restore_and_measure_(
        blended=blended,
        source=source,
        mask=mask,
        dilation=0,
        chunk_size=2,
        device=None,
    )

    assert proof == {"decoded_frames": 4, "white_ratio": 0.25, "dilated_ratio": 0.25}
    assert int(blended[:, 8:24, 8:24, :].min()) == 200, "inside the mask stays generated"
    assert int(blended[:, 0:8, 0:8, :].max()) == 10, "outside is the source, byte for byte"
