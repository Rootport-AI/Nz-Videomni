"""The LTX 2.5 inpainting driver: its mask, its borrowings and its edges.

台帳 §3-150 added ``engine25/inpaint25.py`` -- the masked two-stage driver that
lets a 2.5 base model run the Edit tab's Inpainting sub-tab. What the real-GPU
gate measures is the picture; what is checkable HERE, without a GPU and without
building a single model, is everything else that could go wrong:

1. **The mask decode is a CONTRACT, not a convenience.** ``(F, 1, H, W)`` uint8
   0/255, red channel, threshold at exactly 128, NEVER resized, and a frame
   count that matches the job. Every one of those is a promise the blend and
   the restore rest on, and the wrong answer to any of them is a silently
   mis-framed video rather than a failed job.
2. **This driver's 2.3 borrowings really are the 2.3 objects**, not same-named
   twins. The geometry, the two de-greens, the half-res mask and the restore
   are SHARED with the 2.3 engine and with the API validator; two definitions
   of any of them would let the canvas the app validated and the canvas this
   engine repaints disagree with nothing failing.
3. **The restore is byte-exact where it promises to be.** Outside the dilated
   support the delivered frame IS the source frame -- that is the whole
   guarantee of the feature, and ``mask_proof`` is what says so in
   ``metadata.json``.
4. **The de-green really removes the green.** Two calls, two disjoint regions;
   a build that made only one of them would leave #66FF00 for the coarse
   pyramid levels to bleed into the picture.
5. **The crop is right/bottom only.** The source is anchored at (0, 0), so a
   crop that took anything off the left or the top would shift every delivered
   frame by a few pixels -- invisible in a single still, obvious in a cut.
6. **``InpaintResult`` is a superset of ``GenerationResult``** -- derived from
   the dataclass rather than restated -- with ``inpaint`` as its one additive
   ``as_dict`` key, because the worker's ``done`` builder reads it by name.
7. **The module imports without building anything.**

CPU-only and model-free. Every function under test is arithmetic over tensors
and dicts; the one that touches a decoder takes it through the module global
this file monkeypatches. The single exception is the ffmpeg round trip at the
end, which skips when ffmpeg is absent -- a mask that has never been through
H.264 cannot demonstrate why the binarisation exists.

Run with ``.venv-engine-ltx25`` and ``--noconftest`` (the app conftest builds a
FastAPI app that venv does not have)::

    .venv-engine-ltx25/Scripts/python.exe \\
        outputs/start-end-bridge-2026-09-07/implA_engine_runner/run_ltx25_pytest.py \\
        tests/test_ltx25_inpaint.py
"""

from __future__ import annotations

import inspect
import shutil
import subprocess
from dataclasses import fields
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("ltx_core")

import torch  # noqa: E402

from engine.inpaint import canvas as inpaint_canvas_2_3  # noqa: E402
from engine.outpaint import pyramid_blend as blend_2_3  # noqa: E402
from engine.outpaint.canvas import GREEN_RGB  # noqa: E402
from engine25 import inpaint25, outpaint25  # noqa: E402
from engine25.inpaint25 import (  # noqa: E402
    InpaintGeometry,
    InpaintResult,
    _decode_mask_u8,
)
from engine25.pipeline25 import GenerationResult  # noqa: E402

CPU = torch.device("cpu")

has_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="the round-trip fixture is a real mp4"
)

# Both sides clear ``INPAINT_MIN_SOURCE_SIDE`` (256) so the geometry below is a
# legal one, and the height is deliberately NOT a multiple of 128 (288 -> a 384
# canvas, a 96px bottom band) so the pad-band tests exercise a real band. The
# same three numbers 2.3's ``tests/test_inpaint_mask_decode.py`` uses, which is
# what makes the round trip at the end of this file a comparison rather than a
# coincidence.
W, H, N = 384, 288, 5
RECT = (96, 72, 288, 216)  # x0, y0, x1, y1 (half-open), all even


def _geometry() -> InpaintGeometry:
    """A 384x384 canvas holding a 384x288 source: a 96px bottom band, no right one."""
    geom = InpaintGeometry(
        canvas_width=384, canvas_height=384, source_width=W, source_height=H
    )
    geom.validate()
    return geom


def _fake_mask_decoder(
    frames: int, *, height: int = H, width: int = W, level: int = 255
):
    """A ``decode_video_by_frame`` twin yielding ``frames`` (1, H, W, C) uint8 frames.

    The RECT region is filled with ``level`` and everything else is 0, in ALL
    THREE channels -- a genuinely grey mask, which is what the plugin writes.
    The red-channel rule is proved separately by :func:`_fake_colour_decoder`.
    """

    def decode(*, path: str, device, frame_cap: int | None = None):  # noqa: ANN001, ARG001
        x0, y0, x1, y1 = RECT
        for index in range(frames):
            if frame_cap is not None and index >= frame_cap:
                return
            frame = torch.zeros((1, height, width, 3), dtype=torch.uint8)
            if (y1 <= height) and (x1 <= width):
                frame[0, y0:y1, x0:x1, :] = level
            yield frame

    return decode


def _fake_flat_decoder(level: int, *, height: int = 8, width: int = 8):
    """A decoder yielding a FLAT grey field at a known 8-bit level.

    The threshold probe: no edge anywhere, so what is measured is the decision
    rule and nothing about how an edge survives an encode."""

    def decode(*, path: str, device, frame_cap: int | None = None):  # noqa: ANN001, ARG001
        for _ in range(frame_cap or 1):
            yield torch.full((1, height, width, 3), level, dtype=torch.uint8)

    return decode


def _fake_colour_decoder(red: int, green: int, blue: int):
    """A decoder whose three channels DISAGREE -- the red-channel probe."""

    def decode(*, path: str, device, frame_cap: int | None = None):  # noqa: ANN001, ARG001
        for _ in range(frame_cap or 1):
            frame = torch.zeros((1, 8, 8, 3), dtype=torch.uint8)
            frame[0, :, :, 0] = red
            frame[0, :, :, 1] = green
            frame[0, :, :, 2] = blue
            yield frame

    return decode


# ---------------------------------------------------------------------------
# 1. The mask decode contract (Docs/INPAINTING_DESIGN.md §6.2)
# ---------------------------------------------------------------------------


def test_the_mask_decodes_to_F1HW_uint8_zero_or_255_on_the_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shape and the dtype ARE the contract: ``place_mask_on_canvas``,
    ``half_res_mask``, ``fill_mask_with_generated_`` and ``blend_video_u8`` all
    take ``(F, 1, H, W)`` and all read ``mask > 0.5``, which is true for 255 and
    false for 0 and for nothing in between. The CPU is where the uint8 video
    buffers live, so a mask left on the compute device would be copied per
    chunk for the rest of the job."""
    monkeypatch.setattr(inpaint25, "decode_video_by_frame", _fake_mask_decoder(N))

    mask = _decode_mask_u8(
        mask_path="mask.mp4", num_frames=N, height=H, width=W, device=CPU
    )
    assert tuple(mask.shape) == (N, 1, H, W)
    assert mask.dtype is torch.uint8
    assert mask.device.type == "cpu"
    assert set(mask.unique().tolist()) == {0, 255}, "binary, never a grey in between"

    x0, y0, x1, y1 = RECT
    assert int(mask[:, :, y0:y1, x0:x1].min()) == 255
    assert int(mask[:, :, :y0, :].max()) == 0
    assert int(mask[:, :, y1:, :].max()) == 0


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
def test_the_threshold_is_gte_128(
    monkeypatch: pytest.MonkeyPatch, level: int, expected: int
) -> None:
    """ONE binarisation rule, spelt three times -- in the green-fill
    filtergraph's ``lut``, in 2.3's ``decode_mask_video`` and here -- because
    ffmpeg and the two wheels cannot share a constant. A threshold that drifted
    on one of the three would paint the canvas green over one region and repaint
    a slightly different one."""
    monkeypatch.setattr(inpaint25, "decode_video_by_frame", _fake_flat_decoder(level))
    mask = _decode_mask_u8(
        mask_path="grey.mp4", num_frames=2, height=8, width=8, device=CPU
    )
    assert int(mask.min()) == int(mask.max()) == expected


def test_the_threshold_is_a_parameter_not_a_constant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Not exposed through the API, but the GPU gate has to be able to move it
    without editing the module -- which is how 128 was chosen in the first
    place. 2.3's decode carries the same parameter for the same reason."""
    monkeypatch.setattr(inpaint25, "decode_video_by_frame", _fake_flat_decoder(100))
    at_default = _decode_mask_u8(
        mask_path="grey.mp4", num_frames=2, height=8, width=8, device=CPU
    )
    assert int(at_default.max()) == 0
    at_64 = _decode_mask_u8(
        mask_path="grey.mp4", num_frames=2, height=8, width=8, device=CPU, threshold=64
    )
    assert int(at_64.min()) == 255


def test_only_the_red_channel_decides(monkeypatch: pytest.MonkeyPatch) -> None:
    """For a genuinely grey mask all three channels are equal, so this only
    matters for one that somehow is not -- and then RED is the channel
    ``video_io.fill_mask_green_mp4``'s ``lut`` reads too. Two readers, one
    channel: a decode that averaged the three, or took the green one, would
    disagree with the canvas it is blended against."""
    monkeypatch.setattr(
        inpaint25, "decode_video_by_frame", _fake_colour_decoder(200, 0, 0)
    )
    assert int(
        _decode_mask_u8(
            mask_path="m.mp4", num_frames=1, height=8, width=8, device=CPU
        ).min()
    ) == 255

    monkeypatch.setattr(
        inpaint25, "decode_video_by_frame", _fake_colour_decoder(0, 255, 255)
    )
    assert int(
        _decode_mask_u8(
            mask_path="m.mp4", num_frames=1, height=8, width=8, device=CPU
        ).max()
    ) == 0


def test_a_mask_of_the_wrong_resolution_is_rejected_not_resized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE HEADLINE OF §6.2. Rescaling a mask moves the boundary by a sub-pixel
    amount the user cannot see and cannot correct, so the contract is that the
    mask arrives at the source's own resolution or the request is a 422. A
    mismatch HERE means the API guard was bypassed -- which is why the message
    names the design section rather than only the two sizes."""
    monkeypatch.setattr(
        inpaint25, "decode_video_by_frame", _fake_mask_decoder(N, height=192, width=256)
    )
    with pytest.raises(ValueError, match="never resized") as ei:
        _decode_mask_u8(
            mask_path="small.mp4", num_frames=N, height=H, width=W, device=CPU
        )
    assert "INPAINTING_DESIGN.md §6.2" in str(ei.value)
    assert "256x192" in str(ei.value) and f"{W}x{H}" in str(ei.value)


def test_a_short_mask_is_a_named_shortfall_rather_than_padding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A short mask would silently leave the tail of the window unrepainted.
    The endpoint checks the frame count before a job exists; this is the
    backstop on the other side of the worker pipe, and it speaks in
    ``_require_frames``' words under THIS driver's label so an operator grepping
    a log can tell an inpaint shortfall from an outpaint one."""
    monkeypatch.setattr(inpaint25, "decode_video_by_frame", _fake_mask_decoder(3))
    with pytest.raises(ValueError, match="^inpaint frame shortfall: the mask video"):
        _decode_mask_u8(
            mask_path="short.mp4", num_frames=N, height=H, width=W, device=CPU
        )


def test_an_empty_mask_decode_is_named_separately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Zero frames gets its OWN message: "decoded to 0 frames" says the file is
    unreadable or empty, which is a different fault from a file that is merely
    short, and the two are diagnosed differently. ``_canvas_u8`` draws the same
    distinction for the canvas."""
    monkeypatch.setattr(inpaint25, "decode_video_by_frame", _fake_mask_decoder(0))
    with pytest.raises(ValueError, match="inpaint mask decoded to 0 frames"):
        _decode_mask_u8(
            mask_path="empty.mp4", num_frames=N, height=H, width=W, device=CPU
        )


def test_num_frames_must_be_positive() -> None:
    """Checked BEFORE the decoder is touched, so a zero-length job names the
    number rather than an empty iterator."""
    with pytest.raises(ValueError, match="num_frames must be >= 1"):
        _decode_mask_u8(
            mask_path="nothing.mp4", num_frames=0, height=H, width=W, device=CPU
        )


def test_a_frame_of_the_wrong_rank_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The decoder's own contract, restated where it would bite: 1.2.0 yields
    ``(1, H, W, C)`` where 2.3's wheel yielded ``(1, C, 1, H, W)``. A wheel that
    changed shape under us must fail here, naming the shape, rather than have
    its width read as a channel count."""

    def decode(*, path: str, device, frame_cap: int | None = None):  # noqa: ANN001, ARG001
        yield torch.zeros((1, 3, 1, H, W), dtype=torch.uint8)

    monkeypatch.setattr(inpaint25, "decode_video_by_frame", decode)
    with pytest.raises(ValueError, match=r"expected one \(1, H, W, C\) frame"):
        _decode_mask_u8(
            mask_path="odd.mp4", num_frames=1, height=H, width=W, device=CPU
        )


def test_a_long_mask_is_capped_at_num_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    """The other direction is NOT an error: ``frame_cap`` stops the decode at
    ``num_frames``, so extra material is simply never read -- and never held.
    The API refuses the mismatch before it gets this far."""
    monkeypatch.setattr(inpaint25, "decode_video_by_frame", _fake_mask_decoder(N + 6))
    mask = _decode_mask_u8(
        mask_path="long.mp4", num_frames=N, height=H, width=W, device=CPU
    )
    assert int(mask.shape[0]) == N


# ---------------------------------------------------------------------------
# 2. The borrowings are the 2.3 objects
# ---------------------------------------------------------------------------


def test_the_inpaint_canvas_helpers_are_the_SAME_objects_2_3_uses() -> None:
    """Identity, not equality. ``InpaintGeometry`` is what the API validator
    constructs and what ``services.video_io.fill_mask_green_mp4`` is sized from;
    ``restore_and_measure_`` is what produces ``mask_proof``, the number that
    proves a real mask reached a real engine. A second definition of either
    would let the two engines answer the same question differently with nothing
    failing."""
    assert inpaint25.InpaintGeometry is inpaint_canvas_2_3.InpaintGeometry
    assert inpaint25.place_mask_on_canvas is inpaint_canvas_2_3.place_mask_on_canvas
    assert inpaint25.half_res_mask is inpaint_canvas_2_3.half_res_mask
    assert (
        inpaint25.fill_pad_bands_with_generated_
        is inpaint_canvas_2_3.fill_pad_bands_with_generated_
    )
    assert (
        inpaint25.fill_mask_with_generated_
        is inpaint_canvas_2_3.fill_mask_with_generated_
    )
    assert inpaint25.restore_and_measure_ is inpaint_canvas_2_3.restore_and_measure_
    # ...and the blend itself, shared by all FOUR drivers (2.3 and 2.5,
    # in- and outpainting). A second implementation would be a second set of
    # seams to gate.
    assert inpaint25.blend_video_u8 is blend_2_3.blend_video_u8


def test_the_2_3_engine_imports_the_same_restore_function() -> None:
    """The move that made this driver possible: ``_restore_and_measure_`` used
    to live inside ``engine/pipeline/inpaint_pipeline.py``, which cannot be
    imported under the 2.5 wheel. It is in ``engine.inpaint.canvas`` now and
    BOTH engines import it from there -- so the guarantee 2.3's gate G6
    measured is the one this driver makes, by identity rather than by
    inspection.

    The 2.3 module itself is only importable in ``.venv-engine`` (it binds 2.3's
    wheel at import time), so the identity is asserted from the shared side and
    ``tests/test_inpaint_geometry.py`` closes the loop in the other venv."""
    assert inpaint25.restore_and_measure_.__module__ == "engine.inpaint.canvas"
    assert "restore_and_measure_" in inpaint_canvas_2_3.__all__


def test_the_driver_borrows_outpaint25s_helpers_rather_than_copying_them() -> None:
    """The six blocks lifted out of ``run_outpaint`` for exactly this driver,
    plus the pixel and audio helpers around them. Each is either a MEASURED
    path (the stage-2 re-encode's ``channels_last_3d`` re-layout) or a
    user-visible CONTRACT (the ``ltx25`` sub-dict lands in metadata.json
    verbatim), so a copy would be a second answer to one question."""
    shared = (
        "_freeze_source_audio",
        "_audio_init",
        "_encode_reference_conditionings",
        "_reencode_stage2",
        "_mux_audio",
        "_ltx25_block",
        "_canvas_u8",
        "_decoded_to_u8",
        "_upscale_u8",
        "_iter_encode_chunks",
        "_encode_chunk_count",
        "_audio_freeze",
        "_freeze_proof",
        "_mux_plan",
        "_phase_peak_mb",
        "_resolve_stage2_sigmas",
        "_blend_chunk_size",
        "_require_frames",
    )
    for name in shared:
        assert getattr(inpaint25, name) is getattr(outpaint25, name), name
        assert getattr(inpaint25, name).__module__ == outpaint25.__name__, name


def test_the_two_drivers_share_one_stage2_schedule_and_one_pixel_chunk() -> None:
    """Stage 2 starts from a re-encode of BLENDED pixels in BOTH features, so
    the rung it starts at is one fact rather than two -- and the driver reaches
    it through the shared resolver rather than through a literal of its own.
    The pixel chunk is the same knob under the same environment variable."""
    assert inpaint25._resolve_stage2_sigmas(None) == outpaint25.OUTPAINT_STAGE2_SIGMAS
    assert inpaint25._PIXEL_CHUNK_FRAMES is outpaint25._PIXEL_CHUNK_FRAMES == 8
    # The stage-2 audio policy is recorded by the SHARED ``ltx25`` builder, so
    # an inpaint job cannot report a policy the outpaint driver does not run.
    assert "STAGE2_AUDIO_INIT_POLICY" in inspect.getsource(outpaint25._ltx25_block)


def test_the_module_imports_without_building_anything() -> None:
    """The import smoke test. Reaching ``run_inpaint`` must not need a model, a
    GPU or a checkpoint -- the whole file above this line is proof by execution,
    and this states it."""
    import importlib

    module = importlib.import_module("engine25.inpaint25")
    assert callable(module.run_inpaint)
    assert issubclass(module.InpaintError, RuntimeError)


# ---------------------------------------------------------------------------
# 3. The restore, and the proof it returns
# ---------------------------------------------------------------------------


def _clip(frames: int, height: int, width: int, fill: int) -> torch.Tensor:
    return torch.full((frames, height, width, 3), fill, dtype=torch.uint8)


def test_the_restore_puts_the_original_back_byte_for_byte_outside_the_dilation() -> None:
    """THE GUARANTEE OF THE WHOLE FEATURE, and the reason ``mask_proof`` is in
    metadata.json: outside the dilated support the delivered frame IS the source
    frame, byte for byte, up to the mp4 encode.

    Byte-EXACT rather than approximate is attainable because the restore is a
    ``torch.where`` over uint8 -- nothing is interpolated -- and the predicate
    is "the dilation provably never reached here" (``< 1e-6``) rather than a
    threshold somewhere in the ramp."""
    from engine.outpaint.pyramid_blend import apply_low_res_mask_dilation

    frames, side = 3, 128
    source = _clip(frames, side, side, 40)
    blended = _clip(frames, side, side, 200)
    mask = torch.zeros((frames, 1, side, side), dtype=torch.uint8)
    mask[:, :, 48:80, 48:80] = 255

    proof = inpaint25.restore_and_measure_(
        blended=blended, source=source, mask=mask, dilation=2, chunk_size=8, device=CPU
    )

    dilated = apply_low_res_mask_dilation(mask.to(torch.float32).div(255.0), 2)
    outside = (dilated < 1e-6).permute(0, 2, 3, 1).expand(-1, -1, -1, 3)
    assert torch.equal(blended[outside], source[outside])
    # ...and INSIDE the support the blend survives untouched: a restore that
    # ate into the ramp would put back the hard edge the blend just removed.
    assert int(blended[~outside].min()) == 200

    # The ratios are the honest by-product of a pass that had to look at every
    # pixel anyway, and they are what a gate reads.
    assert proof["decoded_frames"] == frames
    assert proof["white_ratio"] == pytest.approx(32 * 32 / (side * side), rel=1e-3)
    assert proof["dilated_ratio"] > proof["white_ratio"], (
        "the dilation grows the region; a ratio that did not would mean the "
        "restore measured the raw mask instead of the dilated one"
    )


def test_the_restore_is_chunk_invariant() -> None:
    """The chunk size is reachable from the environment
    (``LTX_OUTPAINT_BLEND_CHUNK``), so an operator narrowing it for a 16 GB card
    must not be changing the deliverable. Every op in the restore is independent
    per frame, and this is what keeps that true."""
    frames, side = 5, 128
    mask = torch.zeros((frames, 1, side, side), dtype=torch.uint8)
    mask[:, :, 32:96, 32:96] = 255

    results = []
    for chunk in (1, 2, 8, 99):
        blended = _clip(frames, side, side, 200)
        proof = inpaint25.restore_and_measure_(
            blended=blended,
            source=_clip(frames, side, side, 40),
            mask=mask,
            dilation=2,
            chunk_size=chunk,
            device=CPU,
        )
        results.append((blended, proof))

    first, first_proof = results[0]
    for other, proof in results[1:]:
        assert torch.equal(first, other)
        assert proof == first_proof


# ---------------------------------------------------------------------------
# 4. The de-green: two calls, two disjoint regions, no green left
# ---------------------------------------------------------------------------


def test_the_two_de_greens_leave_no_sentinel_pixel_anywhere() -> None:
    """The pad bands and the mask are DISJOINT regions with different shapes --
    the bands come from the geometry, the mask from the video -- which is why
    this takes two calls rather than one. What is at stake is not tidiness: the
    coarse levels of the Laplacian pyramid mix the canvas' DC component into the
    generated area whatever the mask says, so a #66FF00 left anywhere tints its
    whole neighbourhood in the delivered frame."""
    geom = _geometry()
    frames = 2
    canvas = torch.zeros((frames, geom.canvas_height, geom.canvas_width, 3), dtype=torch.uint8)
    canvas[..., 0], canvas[..., 1], canvas[..., 2] = 10, 20, 30  # the real picture
    # The sentinel, in BOTH places a real canvas carries it: the bottom band...
    canvas[:, geom.source_height :, :, :] = torch.tensor(GREEN_RGB, dtype=torch.uint8)
    # ...and the masked region inside the picture.
    x0, y0, x1, y1 = RECT
    canvas[:, y0:y1, x0:x1, :] = torch.tensor(GREEN_RGB, dtype=torch.uint8)

    generated = _clip(frames, geom.canvas_height, geom.canvas_width, 77)
    mask = torch.zeros((frames, 1, geom.canvas_height, geom.canvas_width), dtype=torch.uint8)
    mask[:, :, y0:y1, x0:x1] = 255

    assert _greens(canvas) > 0, "the fixture has to start green or it proves nothing"
    inpaint25.fill_pad_bands_with_generated_(
        canvas,
        generated=generated,
        source_height=geom.source_height,
        source_width=geom.source_width,
    )
    assert _greens(canvas) > 0, "one call cannot cover two disjoint regions"
    inpaint25.fill_mask_with_generated_(
        canvas, generated=generated, mask=mask, chunk_size=8
    )
    assert _greens(canvas) == 0

    # ...and the untouched picture is still the picture: the de-green replaces
    # the two regions and nothing else.
    assert int(canvas[:, :y0, :, 0].max()) == 10
    assert int(canvas[:, y0:y1, x0:x1, :].min()) == 77
    assert int(canvas[:, geom.source_height :, :, :].min()) == 77


def _greens(clip: torch.Tensor) -> int:
    """How many pixels of ``clip`` are exactly the sentinel #66FF00."""
    green = torch.tensor(GREEN_RGB, dtype=torch.uint8)
    return int((clip == green).all(dim=-1).sum())


def test_the_half_res_mask_is_pasted_then_halved() -> None:
    """PASTE, THEN HALVE -- the order the driver uses, and the one that is easy
    to get backwards. Halving first would area-average the mask across the pad
    boundary and bleed the region into a band that is cropped off at the end,
    which is a seam in the delivered frame rather than a visible bug in a
    fixture."""
    geom = _geometry()
    x0, y0, x1, y1 = RECT
    source_mask = torch.zeros((N, 1, H, W), dtype=torch.uint8)
    source_mask[:, :, y0:y1, x0:x1] = 255

    on_canvas = inpaint25.place_mask_on_canvas(source_mask, geom)
    assert tuple(on_canvas.shape) == (N, 1, geom.canvas_height, geom.canvas_width)
    assert int(on_canvas[:, :, H:, :].max()) == 0, "the pad band is never repainted"

    half = inpaint25.half_res_mask(on_canvas, geom.canvas_height // 2, geom.canvas_width // 2)
    assert half.dtype is torch.uint8
    assert set(half.unique().tolist()) <= {0, 255}
    assert int(half[:, :, y0 // 2 + 1 : y1 // 2 - 1, x0 // 2 + 1 : x1 // 2 - 1].min()) == 255
    # The half-res pad band is still zero, which is what the other order would
    # have blurred away.
    assert int(half[:, :, H // 2 :, :].max()) == 0


# ---------------------------------------------------------------------------
# 5. The crop is right/bottom only
# ---------------------------------------------------------------------------


def test_the_crop_takes_the_top_left_rectangle_and_nothing_else() -> None:
    """The source is anchored at (0, 0), so the delivered frame is a slice of
    the right and bottom bands and nothing else -- lossless, at uint8, before
    the encode. A crop that centred instead would shift every frame by a few
    pixels: invisible in one still, obvious the moment the clip is cut back into
    the timeline next to its own material."""
    geom = _geometry()
    canvas = torch.arange(
        2 * geom.canvas_height * geom.canvas_width * 3, dtype=torch.int32
    ).remainder(251).to(torch.uint8).reshape(2, geom.canvas_height, geom.canvas_width, 3)

    cropped = canvas[:, : geom.source_height, : geom.source_width, :].contiguous()

    assert tuple(cropped.shape) == (2, geom.source_height, geom.source_width, 3)
    assert (geom.source_height, geom.source_width) == (H, W)
    assert torch.equal(cropped, canvas[:, :H, :W, :])
    # The first delivered pixel is the canvas' first pixel: the origin did not
    # move, which is the half a centring bug would break.
    assert torch.equal(cropped[:, 0, 0, :], canvas[:, 0, 0, :])


def test_the_driver_crops_with_the_geometrys_own_source_dimensions() -> None:
    """The behaviour test above is arithmetic anyone could write; this is what
    ties it to the shipped code. The slice has to read the GEOMETRY, because
    that is the object the API validated -- a crop that used ``msg['width']`` or
    a re-derived number would be a second source of truth for the delivered
    size."""
    source = inspect.getsource(inpaint25.run_inpaint)
    assert ": int(geometry.source_height), : int(geometry.source_width)" in source
    # ...and the restore happens BEFORE it: cropping first would leave the pad
    # bands' green in the file, and restoring after would have nothing to
    # restore the bands from.
    assert source.index("restore_and_measure_(") < source.index("inpaint crop:")


# ---------------------------------------------------------------------------
# 6. InpaintResult
# ---------------------------------------------------------------------------


def _result(**overrides) -> InpaintResult:
    base = {
        "output_path": "out.mp4",
        "seed": 123,
        # The DELIVERED size, not the canvas: 1280x768 source inside a
        # 1280x768 canvas needs no pads, so the interesting case is a source
        # whose height is not a multiple of 128.
        "width": 1280,
        "height": 720,
        "num_frames": 121,
        "frame_rate": 24.0,
        "encode_fps": 24,
        "num_images": 0,
        "size_bytes": 4096,
        "seconds": 81.345,
        "metadata": {"inpaint": {"canvas_width": 1280, "canvas_height": 768}},
        "phases": {"21_stage1_denoise": {"seconds": 1.0}},
        "peak_allocated_gib": 9.0,
        "peak_reserved_gib": 10.0,
        "rss_peak_gib": 12.5,
        "video_chunks": 16,
        "tiling": "TileSizeConfig(...)",
    }
    base.update(overrides)
    return InpaintResult(**base)


def test_inpaint_result_answers_to_every_generation_result_field() -> None:
    """DERIVED from the dataclass, not restated: a field added to
    ``GenerationResult`` and forgotten would otherwise be found only by the
    worker, at run time, on a real job. The ``ClassVar`` that renames the
    additive key is NOT a field, which is what keeps this derivation seeing
    exactly the generation contract plus ``metadata``."""
    expected = {f.name for f in fields(GenerationResult)}
    got = {f.name for f in fields(InpaintResult)}
    assert expected <= got, sorted(expected - got)
    assert got - expected == {"metadata"}
    assert got == {f.name for f in fields(outpaint25.OutpaintResult)}
    assert "_JOB_KEY" not in got


def test_as_dict_is_generation_results_dict_in_its_order_plus_inpaint() -> None:
    """The ``GENERATE_REPORT`` line is read by eye as often as by machine, so an
    inpaint report whose leading keys sit exactly where a plain generation's do
    is one a reader can compare across job kinds without re-learning it -- and
    the LAST key names which kind it was."""
    plain = GenerationResult(
        output_path="out.mp4", seed=123, width=1280, height=720, num_frames=121,
        frame_rate=24.0, encode_fps=24, num_images=0, size_bytes=4096, seconds=81.345,
        phases={"21_stage1_denoise": {"seconds": 1.0}},
        peak_allocated_gib=9.0, peak_reserved_gib=10.0, rss_peak_gib=12.5,
        video_chunks=16, tiling="TileSizeConfig(...)",
    ).as_dict()
    got = _result().as_dict()

    assert list(got)[: len(plain)] == list(plain)
    for key, value in plain.items():
        assert got[key] == value, key
    assert set(got) - set(plain) == {"inpaint"}
    assert list(got)[-1] == "inpaint"
    assert InpaintResult._JOB_KEY == "inpaint"
    # ...and the sibling still says "outpaint": one ``ClassVar``, two answers.
    assert outpaint25.OutpaintResult._JOB_KEY == "outpaint"
    # The value under that key is the job metadata ITSELF, by identity -- the
    # worker reads ``result.metadata["inpaint"]`` off the same object, so a
    # report that carried a copy would be a second thing to keep honest.
    result = _result()
    assert result.as_dict()["inpaint"] is result.metadata


def test_the_result_is_a_subclass_rather_than_a_twin() -> None:
    """A twin dataclass would be a second copy of the generation contract, and
    the worker reads that contract by name. One ``ClassVar`` is the whole
    difference between the two job kinds' results."""
    assert issubclass(InpaintResult, outpaint25.OutpaintResult)
    assert InpaintResult.as_dict is outpaint25.OutpaintResult.as_dict


# ---------------------------------------------------------------------------
# 7. The real round trip: an H.264 mask through THIS wheel's decoder
# ---------------------------------------------------------------------------


def _write_rect_mask(path: Path, *, width=W, height=H, frames=N, lossless=False) -> None:
    """A white rectangle on black, encoded the way the plugin's writer would.

    The same fixture 2.3's ``tests/test_inpaint_mask_decode.py`` writes, so the
    two decoders are compared on one file rather than on two descriptions of
    one."""
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


@has_ffmpeg
def test_a_real_mask_round_trips_to_the_plane_the_2_3_contract_describes(tmp_path) -> None:
    """THE ONE TEST THAT COULD NOT BE WRITTEN WITH A DOUBLE. Two wheels decode
    this mask -- 2.3's ``decode_video_from_file`` and 1.2.0's
    ``decode_video_by_frame`` -- and the whole argument for not sharing 2.3's
    function is that the two produce the same ``(1, H, W, C)`` uint8 RGB frame.
    If that were wrong by one code, the boundary of every repainted region would
    move by a pixel on one engine and not the other.

    The expected plane is written out from the RECT rather than read from 2.3
    (whose module does not import under this wheel), which is the same fixture
    2.3's own test asserts against."""
    path = tmp_path / "mask.mp4"
    _write_rect_mask(path, lossless=True)

    mask = _decode_mask_u8(
        mask_path=str(path), num_frames=N, height=H, width=W, device=CPU
    )
    expected = torch.zeros((N, 1, H, W), dtype=torch.uint8)
    x0, y0, x1, y1 = RECT
    expected[:, :, y0:y1, x0:x1] = 255
    assert torch.equal(mask, expected)


@has_ffmpeg
def test_a_lossy_mask_still_decodes_to_two_values(tmp_path) -> None:
    """The reason the threshold exists. H.264's 4:2:0 chroma and its deblocking
    filter leave a grey ring around the rectangle; without binarisation that
    ring would reach the blend as fractional mask values and the de-green as
    not-quite-green pixels."""
    path = tmp_path / "lossy.mp4"
    _write_rect_mask(path, lossless=False)
    mask = _decode_mask_u8(
        mask_path=str(path), num_frames=N, height=H, width=W, device=CPU
    )
    assert set(mask.unique().tolist()) <= {0, 255}
    x0, y0, x1, y1 = RECT
    assert int(mask[:, :, y0 + 4 : y1 - 4, x0 + 4 : x1 - 4].min()) == 255
    assert int(mask[:, :, :4, :4].max()) == 0


@has_ffmpeg
def test_a_real_mask_survives_the_paste_and_the_halve(tmp_path) -> None:
    """The three pieces meet here: decode -> place on the canvas -> half
    resolution, on a real file. Each is tested on its own above; this is the one
    place the dtype/value convention (uint8 0/255) is shown to survive all three
    on this wheel."""
    path = tmp_path / "mask.mp4"
    _write_rect_mask(path, lossless=True)
    geom = _geometry()
    mask = _decode_mask_u8(
        mask_path=str(path), num_frames=N, height=H, width=W, device=CPU
    )
    on_canvas = inpaint25.place_mask_on_canvas(mask, geom)
    assert tuple(on_canvas.shape) == (N, 1, 384, 384)
    assert int(on_canvas[:, :, H:, :].max()) == 0

    half = inpaint25.half_res_mask(on_canvas, 192, 192, chunk_size=8)
    assert half.dtype is torch.uint8
    assert set(half.unique().tolist()) <= {0, 255}
    x0, y0, x1, y1 = RECT
    assert int(half[:, :, y0 // 2 + 1 : y1 // 2 - 1, x0 // 2 + 1 : x1 // 2 - 1].min()) == 255
