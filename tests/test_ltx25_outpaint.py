"""The LTX 2.5 outpainting driver: its constants, its borrowings and its edges.

§3-102's Outpainting increment, commit C1, added
``engine25/outpaint25.py`` -- a two-stage driver with ZERO call sites. The
adapter still refuses ``outpaint`` and the worker has no branch for it, so
nothing about a shipped job changed; the evidence for that is gate G1(a)'s
unchanged selftest digests. What is checkable HERE, without a GPU and without
building a single model, is everything else that could go wrong later:

1. **The stage-2 sigma schedule is DERIVED, and is not a tensor view.** It is
   ``STAGE_2_DISTILLED_SIGMAS[1:]``, and slicing a ``torch.Tensor`` yields a
   VIEW -- a module global holding one would let anything that mutated the
   wheel's constant in place change every outpaint job. ``.tolist()`` is what
   stops that, and this file is what stops the ``.tolist()`` from being
   dropped as noise.
2. **The borrowings really are the 2.3 objects**, not same-named twins. The
   geometry class and the blend function are SHARED with the 2.3 engine, the
   API validator and the mock; two definitions of either would let the canvas
   the app validated and the canvas this engine blends disagree.
3. **The audio freeze is one mechanism over three cases** (full, partial,
   none), and the band item is FIRST in the list it is handed in. The band
   arithmetic is elementwise over the whole token axis, so anything that
   appends tokens must not run before it.
4. **The uint8 <-> float boundaries (C-1)**. 2.5's pixel domain is not 2.3's:
   the decode yields float ``[0, 1]`` where 2.3's yielded uint8, and the mp4
   encoder WANTS float ``[0, 1]`` where 2.3's wanted a uint8 tensor. Getting
   either backwards produces a video 255x too dark or an encoder error, and
   both are cheap to pin here.
5. **The upscale is chunk-invariant.** Its chunk size is reachable from the
   environment, so an operator narrowing it for a 16 GB card must not be
   changing the deliverable.
6. **The three audio outcomes are told apart in the metadata**, including the
   two that share an outcome (freeze disabled vs. no source audio).
7. **A frame shortfall is a NAMED error**, not stage 2's ``create_initial_state``
   assertion several minutes later.
8. **``OutpaintResult`` is a superset of ``GenerationResult``** -- derived from
   the dataclass rather than restated -- because the worker's ``done`` builder
   reads it by name and the app stores ``peak_vram_mb`` out of it.
9. **The engine runs in ONE inference mode (§3-124).**
   ``torch.inference_mode`` appears nowhere under ``engine25/``. This driver
   was decorated with it once, borrowed from 2.3 where every path is, and the
   buffers a job allocated under it could no longer be written by the NEXT
   ordinary generation on the same resident worker. The scan keeps the next
   borrowing from bringing that back.

CPU-only and model-free. Every function under test is arithmetic over tensors,
floats and dicts; the one that touches a decoder takes it through the module
global the test monkeypatches.

Run with ``.venv-engine-ltx25`` and ``--noconftest`` (the app conftest builds a
FastAPI app that venv does not have)::

    PYTHONPATH=. .venv-engine-ltx25/Scripts/python.exe -m pytest \\
        tests/test_ltx25_outpaint.py --noconftest -q
"""

from __future__ import annotations

import inspect
import math
from dataclasses import fields
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("ltx_core")

import torch  # noqa: E402

from engine.outpaint import canvas as canvas_2_3  # noqa: E402
from engine.outpaint import pyramid_blend as blend_2_3  # noqa: E402
from engine25 import chain25, outpaint25  # noqa: E402
from engine25.ltxcore_compat import STAGE_2_DISTILLED_SIGMAS  # noqa: E402
from engine25.outpaint25 import (  # noqa: E402
    OUTPAINT_STAGE2_SIGMAS,
    OutpaintResult,
    _audio_freeze,
    _canvas_u8,
    _encode_chunk_count,
    _encode_input_from_u8,
    _freeze_proof,
    _iter_encode_chunks,
    _mux_plan,
    _resolve_stage2_sigmas,
    _upscale_u8,
)
from engine25.pipeline25 import GenerationResult  # noqa: E402

CPU = torch.device("cpu")


def _audio_latent(frames: int = 6) -> torch.Tensor:
    """A ``(1, 8, frames, 16)`` audio latent with DISTINCT values everywhere.

    ``arange`` rather than ``zeros``: a mask bug that freezes the wrong frames
    is invisible against an all-zero latent.
    """
    return torch.arange(8 * frames * 16, dtype=torch.float32).reshape(1, 8, frames, 16)


# ---------------------------------------------------------------------------
# 1. The stage-2 sigma schedule
# ---------------------------------------------------------------------------


def test_stage2_sigmas_are_the_wheel_ladder_minus_its_first_rung() -> None:
    """DERIVED, not transcribed. 2.3 writes the three numbers as a literal; here
    they are a slice of the constant the wheel ships, so an upstream re-tune
    moves this with it instead of leaving a stale literal behind."""
    assert OUTPAINT_STAGE2_SIGMAS == STAGE_2_DISTILLED_SIGMAS[1:].tolist()
    assert len(OUTPAINT_STAGE2_SIGMAS) == 3
    # The first rung is the workflow's own ManualSigmas value (node 5211).
    # ``isclose`` rather than ``==`` because the value arrives through float32.
    assert math.isclose(OUTPAINT_STAGE2_SIGMAS[0], 0.725, abs_tol=1e-6)
    assert math.isclose(OUTPAINT_STAGE2_SIGMAS[1], 0.421875, abs_tol=1e-6)
    assert OUTPAINT_STAGE2_SIGMAS[-1] == 0.0
    # ...and it really is one rung LOWER than the stock stage-2 schedule, which
    # is the whole point: stage 2 starts from a re-encode of the BLENDED
    # pixels, so starting where the stock schedule does would partly regenerate
    # the seam blend 1 just fixed.
    assert float(STAGE_2_DISTILLED_SIGMAS[0]) > OUTPAINT_STAGE2_SIGMAS[0]


def test_stage2_sigmas_is_a_plain_list_and_not_a_view_of_the_wheel_tensor() -> None:
    """``[1:]`` on a tensor is a VIEW; a module global holding one would let an
    in-place mutation of the wheel's constant change every outpaint job. The
    mutation below is what makes that a measurement rather than a claim."""
    assert isinstance(OUTPAINT_STAGE2_SIGMAS, list)
    assert not torch.is_tensor(OUTPAINT_STAGE2_SIGMAS)
    assert all(type(v) is float for v in OUTPAINT_STAGE2_SIGMAS)

    before = list(OUTPAINT_STAGE2_SIGMAS)
    original = STAGE_2_DISTILLED_SIGMAS.clone()
    try:
        STAGE_2_DISTILLED_SIGMAS.mul_(0.0)
        assert OUTPAINT_STAGE2_SIGMAS == before
    finally:
        STAGE_2_DISTILLED_SIGMAS.copy_(original)
    assert STAGE_2_DISTILLED_SIGMAS[1:].tolist() == before


def test_the_sigma_override_replaces_the_schedule_and_the_default_is_a_copy() -> None:
    """Gate O4's arm, and the ONE place the override and the default meet.

    The default must be a COPY: a caller that mutated the returned list would
    otherwise be editing the module constant for the rest of the process."""
    assert _resolve_stage2_sigmas(None) == OUTPAINT_STAGE2_SIGMAS
    assert _resolve_stage2_sigmas(None) is not OUTPAINT_STAGE2_SIGMAS
    assert _resolve_stage2_sigmas([]) == OUTPAINT_STAGE2_SIGMAS

    override = [0.909375, 0.725, 0.421875, 0.0]
    assert _resolve_stage2_sigmas(override) == override
    # Ints and numpy-ish scalars arrive as plain floats.
    assert all(type(v) is float for v in _resolve_stage2_sigmas([1, 0]))

    mutated = _resolve_stage2_sigmas(None)
    mutated[0] = -1.0
    assert OUTPAINT_STAGE2_SIGMAS[0] != -1.0


def test_the_stage2_noise_scale_follows_the_override(  # C-4
) -> None:
    """``noise_scale`` is the schedule's OWN first sigma, read once and handed to
    both modalities. Two separate reads is how that link gets broken, and a
    partial audio freeze (gate O7) is the only arm that would expose the break
    -- which is why the link is pinned here as well."""
    for arm in (None, [0.909375, 0.725, 0.421875, 0.0], [0.5, 0.0]):
        values = _resolve_stage2_sigmas(arm)
        tensor = torch.tensor(values, dtype=torch.float32, device=CPU)
        # ``approx`` and not ``==``: the list is float64 and the tensor float32,
        # which is exactly what the engine does too -- the scale is read off the
        # TENSOR, so what is being pinned is "the same rung", not "the same bits".
        assert float(tensor[0].item()) == pytest.approx(values[0], rel=1e-6)
        assert float(tensor[0].item()) == pytest.approx(
            max(values), rel=1e-6
        ), "the noise scale must be the schedule's FIRST (highest) rung"


# ---------------------------------------------------------------------------
# 2. The borrowings are the 2.3 objects
# ---------------------------------------------------------------------------


def test_the_geometry_and_the_blend_are_the_SAME_objects_the_2_3_engine_uses() -> None:
    """Identity, not equality. ``OutpaintGeometry`` is what the API validator
    constructs and what ``services.video_io.pad_green_mp4`` is sized from; a
    second class here would let the canvas the app validated and the canvas this
    engine blends disagree with nothing failing."""
    assert outpaint25.OutpaintGeometry is canvas_2_3.OutpaintGeometry
    assert outpaint25.build_blend_mask is canvas_2_3.build_blend_mask
    assert outpaint25.blend_video_u8 is blend_2_3.blend_video_u8
    # The de-green step likewise: two copies would let the pad band this engine
    # replaces and the pad band the mask defines drift apart.
    assert outpaint25.fill_pad_with_generated_ is canvas_2_3.fill_pad_with_generated_


def test_the_chain_borrowings_are_the_same_objects_the_chain_uses() -> None:
    """The band item, the dtype, the sampler kwargs and the two seed/eta
    constants come from ``chain25`` rather than being restated. ``AudioBandMask``
    in particular is pinned against upstream's video twin by
    ``ltxcore_compat.verify``; a copy would not be pinned."""
    assert outpaint25.AudioBandMask is chain25.AudioBandMask
    assert outpaint25.DTYPE is chain25.DTYPE
    assert outpaint25._stage1_sampler_kwargs is chain25._stage1_sampler_kwargs
    assert outpaint25._vram_summary is chain25._vram_summary
    assert outpaint25.STAGE2_SEED_OFFSET == chain25.STAGE2_SEED_OFFSET == 100
    assert outpaint25.STAGE1_SAMPLER == chain25.STAGE1_SAMPLER == "ancestral"


def test_the_module_imports_without_building_anything() -> None:
    """The import smoke test. Reaching ``run_outpaint`` must not need a model, a
    GPU or a checkpoint -- the whole file above this line is proof by execution,
    and this states it."""
    import importlib

    module = importlib.import_module("engine25.outpaint25")
    assert callable(module.run_outpaint)
    assert issubclass(module.OutpaintError, RuntimeError)


# ---------------------------------------------------------------------------
# 3. The audio freeze: one mechanism, three cases
# ---------------------------------------------------------------------------


def test_audio_freeze_full_partial_and_none() -> None:
    """The three cases the outpaint contract has to answer for.

    FULL is the ordinary job. PARTIAL is a source whose audio ends early --
    NOT an error here (2.3 ruled the same): the audio is guidance for a video
    that is already fully specified, so what there is gets frozen and the rest
    is generated. NONE is a silent source, and it must produce an EMPTY list so
    the call site stays branch-free."""
    latent = _audio_latent(frames=6)

    # -- full --------------------------------------------------------------
    items = _audio_freeze(latent, 6)
    assert len(items) == 1
    assert torch.equal(items[0].mask, torch.ones((1, 6), dtype=torch.float32))

    # -- partial -----------------------------------------------------------
    items = _audio_freeze(latent, 2)
    assert len(items) == 1
    expected = torch.zeros((1, 6), dtype=torch.float32)
    expected[:, :2] = 1.0
    assert torch.equal(items[0].mask, expected)
    # THE LATENT IS THE FULL-LENGTH ONE, never the frozen head alone: the item
    # patchifies latent and mask against the state's whole token axis, so a
    # short tensor would not broadcast. The MASK is what selects the head.
    assert items[0].latent.shape[2] == 6

    # -- none --------------------------------------------------------------
    assert _audio_freeze(latent, 0) == []
    assert _audio_freeze(latent, -3) == []
    assert _audio_freeze(None, 4) == []


def test_audio_freeze_is_a_hard_freeze_and_clamps_an_over_long_request() -> None:
    """``strength=1.0`` is the complement of 2.3's ``mask_value=0.0`` -- the two
    engines freeze the same way, exactly. And a request wider than the latent is
    clamped rather than raising: the caller already took ``min(a_total, avail)``,
    so this is the second line of defence, not the first."""
    latent = _audio_latent(frames=4)
    (item,) = _audio_freeze(latent, 4)
    assert item.strength == 1.0

    (clamped,) = _audio_freeze(latent, 99)
    assert torch.equal(clamped.mask, torch.ones((1, 4), dtype=torch.float32))


def test_the_band_item_is_first_in_the_conditioning_list_it_is_handed_in() -> None:
    """The band arithmetic is elementwise over the whole token axis, so an item
    that APPENDS tokens must not run before it. Outpainting's audio list is the
    band and nothing else today; this is what turns a future addition into a
    decision rather than an accident."""
    items = _audio_freeze(_audio_latent(), 3)
    assert isinstance(items[0], chain25.AudioBandMask)
    assert len(items) == 1


def test_the_freeze_proof_is_None_when_nothing_was_frozen() -> None:
    """``None``, NEVER 0.0. There is nothing to compare, and an invented zero
    would fake a passing proof of exactly the thing being proven -- so ``pass``
    is ``None`` too ("no verdict"), not ``True``."""
    empty = _freeze_proof(frozen_audio=None, s1_audio_head=None, s2_audio_head=None)
    assert empty == {"s1_audio_head": None, "s2_audio_head": None, "pass": None}

    head = _audio_latent(frames=2)
    exact = _freeze_proof(frozen_audio=head, s1_audio_head=head.clone(), s2_audio_head=head.clone())
    assert exact["s1_audio_head"] == 0.0
    assert exact["s2_audio_head"] == 0.0
    assert exact["pass"] is True

    drifted = head.clone()
    drifted[0, 0, 0, 0] += 4.0
    failed = _freeze_proof(frozen_audio=head, s1_audio_head=drifted, s2_audio_head=head.clone())
    assert failed["s1_audio_head"] == 4.0
    assert failed["pass"] is False


# ---------------------------------------------------------------------------
# 4. The uint8 <-> float boundaries (C-1)
# ---------------------------------------------------------------------------


def test_the_encoder_feed_is_a_generator_of_float_0_1_chunks() -> None:
    """Boundary (3). ``encode_video`` wants ``(F, H, W, C)`` float ``[0, 1]``;
    2.3's encoder wanted a materialised uint8 tensor, so a straight port would
    have produced a video 255x too dark or an encoder error.

    A GENERATOR, not a converted tensor: at 1920x1152x241 one float32 timeline
    is 6.4 GB on top of the uint8 original, which is still live because the
    generator reads from it."""
    pixels = torch.randint(0, 256, (10, 4, 6, 3), dtype=torch.uint8)
    chunks = list(_iter_encode_chunks(pixels, 4))

    assert len(chunks) == 3                     # 4 + 4 + 2
    assert [int(c.shape[0]) for c in chunks] == [4, 4, 2]
    for chunk in chunks:
        assert chunk.dtype is torch.float32
        assert float(chunk.min()) >= 0.0
        assert float(chunk.max()) <= 1.0
        assert chunk.shape[1:] == (4, 6, 3)     # (F, H, W, C) is preserved

    # The values are exactly ``u8 / 255`` -- and the caller's uint8 timeline is
    # untouched, which is what makes the in-place divide safe.
    rebuilt = torch.cat(chunks, dim=0)
    assert torch.equal(rebuilt, pixels.to(torch.float32) / 255.0)
    assert pixels.dtype is torch.uint8
    assert int(pixels.max()) <= 255


@pytest.mark.parametrize("frames,step,expected", [(10, 4, 3), (8, 8, 1), (1, 8, 1), (25, 8, 4), (0, 8, 1)])
def test_the_declared_chunk_count_matches_what_the_generator_yields(
    frames: int, step: int, expected: int
) -> None:
    """The encoder drives its progress bar off ``video_chunks_number``, so a
    count that disagreed with the feed would be a bar that never finishes (or an
    iterator the encoder stops reading). One function owns the count precisely
    so a caller cannot disagree with it."""
    assert _encode_chunk_count(frames, step) == expected
    if frames:
        pixels = torch.zeros((frames, 2, 2, 3), dtype=torch.uint8)
        assert len(list(_iter_encode_chunks(pixels, step))) == expected


def test_the_vae_encode_input_is_minus_one_to_one_in_the_wheels_layout() -> None:
    """Boundary (2). ``x / 127.5 - 1`` into ``(1, C, F, H, W)`` -- 2.3's
    expression, character for character, because it is the same VAE input
    convention and a second spelling is a second chance to get a sign wrong."""
    pixels = torch.zeros((3, 4, 6, 3), dtype=torch.uint8)
    pixels[..., 0] = 255
    encoded = _encode_input_from_u8(pixels)

    assert tuple(encoded.shape) == (1, 3, 3, 4, 6)          # (B, C, F, H, W)
    assert encoded.dtype is chain25.DTYPE
    assert encoded.device.type == "cpu"                      # tiled_encode moves tiles itself
    assert float(encoded[0, 0].max()) == pytest.approx(1.0, abs=1e-2)
    assert float(encoded[0, 1].min()) == pytest.approx(-1.0, abs=1e-2)


def test_the_upscale_is_byte_identical_at_two_chunk_sizes() -> None:
    """The chunk size is reachable from the environment
    (``LTX_OUTPAINT_BLEND_CHUNK``), so an operator narrowing it for a 16 GB card
    must not be changing the deliverable. The resample is SPATIAL only, so
    frames do not see each other -- and this is what keeps that true."""
    generator = torch.Generator().manual_seed(20260826)
    pixels = torch.randint(
        0, 256, (9, 8, 12, 3), dtype=torch.uint8, generator=generator
    )
    at_two = _upscale_u8(pixels, height=16, width=24, device=CPU, step=2)
    at_eight = _upscale_u8(pixels, height=16, width=24, device=CPU, step=8)

    assert at_two.dtype is torch.uint8
    assert tuple(at_two.shape) == (9, 16, 24, 3)
    assert torch.equal(at_two, at_eight)
    # ...and a step wider than the timeline is the same single-pass answer.
    assert torch.equal(at_two, _upscale_u8(pixels, height=16, width=24, device=CPU, step=99))


# ---------------------------------------------------------------------------
# 5. The three audio outcomes
# ---------------------------------------------------------------------------


def test_the_mux_plan_tells_the_three_audio_branches_apart() -> None:
    """Three branches, two outcomes -- and the branch is worth recording even
    where the outcome is shared, because a gate reading "the vocoder ran" needs
    to know whether that was the request or a source with no audio track."""
    waveform = torch.zeros((2, 480), dtype=torch.float32)

    frozen = _mux_plan(
        freeze_source_audio=True, source_waveform=waveform, audio_frozen_frames=7
    )
    assert frozen == {
        "audio_branch": "frozen_source_waveform",
        "muxed_original_waveform": True,
        "vocoder_skipped": True,
    }

    disabled = _mux_plan(
        freeze_source_audio=False, source_waveform=None, audio_frozen_frames=0
    )
    assert disabled["audio_branch"] == "generated_freeze_disabled"
    assert disabled["muxed_original_waveform"] is False
    assert disabled["vocoder_skipped"] is False

    silent = _mux_plan(
        freeze_source_audio=True, source_waveform=None, audio_frozen_frames=0
    )
    assert silent["audio_branch"] == "generated_no_source_audio"
    assert silent["vocoder_skipped"] is False

    # A waveform that encoded to ZERO latent frames is the silent case too: the
    # freeze is what the verbatim mux rests on, so "has a track" alone must not
    # skip the vocoder.
    assert (
        _mux_plan(freeze_source_audio=True, source_waveform=waveform, audio_frozen_frames=0)[
            "audio_branch"
        ]
        == "generated_no_source_audio"
    )


# ---------------------------------------------------------------------------
# 6. A frame shortfall is a NAMED error
# ---------------------------------------------------------------------------


def _fake_decoder(frames: int, height: int = 6, width: int = 8):
    """A ``decode_video_by_frame`` twin yielding ``frames`` (1, H, W, C) uint8 frames."""

    def decode(*, path: str, device, frame_cap: int | None = None):  # noqa: ANN001, ARG001
        for index in range(frames):
            if frame_cap is not None and index >= frame_cap:
                return
            yield torch.full((1, height, width, 3), index % 256, dtype=torch.uint8)

    return decode


def test_a_short_canvas_raises_a_named_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """The app already rejects a short source and ``pad_green_mp4`` clone-pads
    the tail as a backstop, so a shortfall here means one of those two failed.
    The alternative to raising is worse than a failed job: stage 2's
    ``create_initial_state`` would assert several minutes later, naming a latent
    shape rather than the file that was short."""
    monkeypatch.setattr(outpaint25, "decode_video_by_frame", _fake_decoder(5))

    with pytest.raises(ValueError, match="outpaint frame shortfall"):
        _canvas_u8(video_path="canvas.mp4", height=4, width=4, frame_cap=9, device=CPU)


def test_an_empty_canvas_decode_is_named_too(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero frames gets its OWN message: "decoded to 0 frames" says the file is
    unreadable or empty, which is a different fault from a file that is merely
    short, and the two are diagnosed differently."""
    monkeypatch.setattr(outpaint25, "decode_video_by_frame", _fake_decoder(0))

    with pytest.raises(ValueError, match="decoded to 0 frames"):
        _canvas_u8(video_path="canvas.mp4", height=4, width=4, frame_cap=3, device=CPU)


def test_a_long_enough_canvas_decodes_to_uint8_FHWC(monkeypatch: pytest.MonkeyPatch) -> None:
    """Boundary (4). ``decode_video_by_frame`` yields ``(1, H, W, C)`` uint8 in
    1.2.0 (2.3's decoder gave ``(1, C, 1, H, W)`` float ``[0, 255]``), and what
    the blend wants is ``(F, H, W, 3)`` uint8 on the CPU."""
    monkeypatch.setattr(outpaint25, "decode_video_by_frame", _fake_decoder(4, height=6, width=8))

    out = _canvas_u8(video_path="canvas.mp4", height=6, width=8, frame_cap=4, device=CPU)
    assert out.dtype is torch.uint8
    assert tuple(out.shape) == (4, 6, 8, 3)
    assert out.device.type == "cpu"
    # Frame ``i`` was filled with ``i``; the resize is the identity at matching
    # dimensions, so the values must survive it exactly.
    assert [int(out[i].min()) for i in range(4)] == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# 7. OutpaintResult is a superset of GenerationResult
# ---------------------------------------------------------------------------


def _result(**overrides) -> OutpaintResult:
    base = {
        "output_path": "out.mp4",
        "seed": 123,
        "width": 1280,
        "height": 768,
        "num_frames": 73,
        "frame_rate": 24.0,
        "encode_fps": 24,
        "num_images": 0,
        "size_bytes": 4096,
        "seconds": 81.345,
        "metadata": {"outpaint": {"canvas_width": 1280}},
        "phases": {"21_stage1_denoise": {"seconds": 1.0}},
        "peak_allocated_gib": 7.0,
        "peak_reserved_gib": 8.0,
        "rss_peak_gib": 12.5,
        "video_chunks": 10,
        "tiling": "TileSizeConfig(...)",
    }
    base.update(overrides)
    return OutpaintResult(**base)


def test_outpaint_result_answers_to_every_generation_result_field() -> None:
    """DERIVED from the dataclass, not restated: a field added to
    ``GenerationResult`` and forgotten here would otherwise be found only by the
    worker, at run time, on a real job."""
    expected = {f.name for f in fields(GenerationResult)}
    got = {f.name for f in fields(OutpaintResult)}
    assert expected <= got, sorted(expected - got)
    # ...and the one addition is the job metadata, named for the feature.
    assert got - expected == {"metadata"}


def test_outpaint_result_carries_the_nine_things_the_done_event_reads() -> None:
    """``engine25/worker.py``'s generate ``done`` reads these eight attributes by
    name and logs ``as_dict()`` as the ``GENERATE_REPORT`` line -- nine reads in
    all. The app then stores ``peak_vram_mb`` out of the first of them into
    ``metadata.json``'s ``vram_optimization`` block, which is what the job-VRAM
    gate reads; an outpaint result answering to fewer names would leave that
    number empty and the gate blind."""
    result = _result()
    assert result.peak_allocated_gib == 7.0
    assert result.peak_reserved_gib == 8.0
    assert result.rss_peak_gib == 12.5
    assert result.seconds == 81.345
    assert result.num_frames == 73
    assert result.encode_fps == 24
    assert result.size_bytes == 4096
    assert result.phases == {"21_stage1_denoise": {"seconds": 1.0}}
    assert callable(result.as_dict)


def test_as_dict_is_generation_results_dict_in_its_order_plus_outpaint() -> None:
    """The ``GENERATE_REPORT`` line is read by eye as often as by machine, so an
    outpaint report whose leading keys sit exactly where a plain generation's do
    is one a reader can compare across job kinds without re-learning it."""
    plain = GenerationResult(
        output_path="out.mp4", seed=123, width=1280, height=768, num_frames=73,
        frame_rate=24.0, encode_fps=24, num_images=0, size_bytes=4096, seconds=81.345,
        phases={"21_stage1_denoise": {"seconds": 1.0}},
        peak_allocated_gib=7.0, peak_reserved_gib=8.0, rss_peak_gib=12.5,
        video_chunks=10, tiling="TileSizeConfig(...)",
    ).as_dict()
    got = _result().as_dict()

    assert list(got)[: len(plain)] == list(plain)
    for key, value in plain.items():
        assert got[key] == value, key
    assert set(got) - set(plain) == {"outpaint"}
    # ``seconds`` is ROUNDED by both, identically -- the report is a report, not
    # the timing source. Compared against ``round`` rather than a literal,
    # because 81.345 is not representable and rounds DOWN here (81.34); a
    # literal would be pinning the float repr rather than the rounding.
    assert got["seconds"] == round(81.345, 2) == plain["seconds"]


# ---------------------------------------------------------------------------
# 9. One inference mode across engine25 (§3-124)
# ---------------------------------------------------------------------------


def test_no_engine25_module_asks_for_torchs_inference_mode() -> None:
    """§3-124: ``run_outpaint`` carried an ``@torch.inference_mode()``
    decorator (copied from 2.3's outpaint pipeline, where EVERY path is that
    mode), and the pinned staging buffers a picture-widening job allocated under
    it were stamped "inference tensors" for good -- so the next ordinary
    generation on the same resident worker died writing into them
    (``RuntimeError: Inplace update to inference tensor outside InferenceMode``).
    The fix was subtraction: engine25 runs under ``torch.no_grad()`` everywhere
    now, as ``generate`` and ``run_chain`` always had. VERIFICATION_LOG §81
    carries the mechanism.

    WHY A MACHINE KEEPS THIS RULE. The bug was one line, it was correct-looking,
    and it was CHEAP TO REPEAT: the next person porting a 2.3 path will find
    that decorator on it. Nothing else here would notice -- the mismatch needs a
    GPU, a resident worker and two jobs in the right order to show itself, which
    is a gate an hour long rather than a test. Text is what the ban is on, since
    a decorator, a ``with`` block and a bare call all spell the same name; that
    is also why the comment left at the old site says "inference-mode" in words
    rather than in code.
    """
    package_dir = Path(inspect.getsourcefile(outpaint25)).parent
    modules = sorted(package_dir.rglob("*.py"))

    # The GLOB is the weak link, not the scan: a renamed or moved package would
    # hand this test an empty list and pass. The package's own contents are the
    # tripwire for that.
    names = {path.name for path in modules}
    assert {"outpaint25.py", "chain25.py", "pipeline25.py", "worker.py"} <= names, sorted(names)
    assert len(modules) >= 10, sorted(names)

    offenders = sorted(
        path.name for path in modules
        if "inference_mode" in path.read_text(encoding="utf-8")
    )
    assert offenders == [], (
        "engine25 is no_grad from end to end; these ask for the other mode: "
        f"{offenders}"
    )
