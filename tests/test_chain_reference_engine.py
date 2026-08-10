"""Clip-wise IC-LoRA reference — ENGINE-side unit tests (§1-15 B4/B5/B6).

The API-layer half of this feature lives in ``tests/test_chain_reference.py`` and
the pure window arithmetic in ``tests/test_chain_math_reference.py``. This file
covers the part in between — how ``chain_pipeline`` READS the long reference
video and hands each stage-1 segment its own window:

  1. ``_iter_reference_windows``: the pure "frame stream -> per-window pixel
     tensor" generator. Driven by a FAKE frame iterator whose n-th frame carries
     the value ``n``, so a test can assert the actual OFFSETS that come out
     (a correct ``video_segment_windows`` paired with an off-by-one reader is
     exactly the bug the window arithmetic tests cannot see).
  2. Short references: partial windows, then ``None`` — never an error.
  3. ``run_chain`` wiring: the reference conditioning is built ONCE PER CLIP
     (n calls for n clips, not the old clip-0-only single call), with each call
     receiving that clip's window, and the calls stop when the reference runs out.

VENV: torch-only, no fastapi -> this file runs in the ENGINE venv, which is the
one that actually has torch:

  .venv-engine\\Scripts\\python.exe -m pytest tests\\test_chain_reference_engine.py ^
      -p no:warnings -p no:cacheprovider --noconftest --rootdir .

(The app ``.venv`` has no torch, so ``importorskip`` skips the whole module
there; ``--noconftest`` is required because tests/conftest.py imports fastapi,
which the engine venv does not have. Same recipe as
tests/test_ic_lora_engine_conditioning.py.)
"""

from __future__ import annotations

import sys
import types

import pytest

torch = pytest.importorskip("torch")

from chain_math import compute_chain_layout, video_segment_windows  # noqa: E402


# ── fake frame stream ────────────────────────────────────────────────────────


def _fake_frames(count: int, *, frame_cap: int | None = None):
    """Yield ``count`` (1,1,1,1,1) CPU frames whose single pixel is the frame's
    GLOBAL index — the same shape contract as
    ``engine.pipeline.common.iter_video_conditioning_cpu`` (which yields
    (1,C,1,H,W) per pixel frame), shrunk to one pixel so the value IS the index.
    ``frame_cap`` mirrors the decoder's cap."""
    n = count if frame_cap is None else min(count, frame_cap)
    for i in range(n):
        yield torch.full((1, 1, 1, 1, 1), float(i))


def _values(window: torch.Tensor) -> list[float]:
    """The frame indices packed into a window tensor, in order."""
    assert window.shape[2] == window.numel(), window.shape
    return [float(v) for v in window.reshape(-1)]


# ── 1. window read-out ───────────────────────────────────────────────────────


def _windows(frames, windows):
    from engine.pipeline.chain_pipeline import _iter_reference_windows

    return list(_iter_reference_windows(frames, windows))


def test_window_offsets_and_lengths_are_read_exactly():
    """Each window comes out starting at its ``start_px`` and exactly
    ``len_px`` frames long — the read-out, not just the arithmetic."""
    wins = [(0, 25), (17, 25), (34, 9)]
    out = _windows(_fake_frames(64), wins)

    assert len(out) == len(wins)
    for tensor, (start, length) in zip(out, wins):
        assert tensor is not None
        assert tensor.shape[2] == length
        vals = _values(tensor)
        assert vals[0] == start
        assert vals == [float(start + k) for k in range(length)]


def test_overlapping_frames_are_carried_not_reread():
    """Adjacent windows share their overlap: the tail of window i and the head of
    window i+1 are the SAME frames (this is what makes the seam consistent), and
    the stream is walked exactly once — the shared frames are buffered, never
    re-pulled."""
    wins = [(0, 25), (17, 25)]
    pulled: list[float] = []

    def counting():
        for f in _fake_frames(64):
            pulled.append(float(f.reshape(-1)[0]))
            yield f

    out = _windows(counting(), wins)
    a, b = _values(out[0]), _values(out[1])
    overlap = 25 - 17  # 8 frames shared
    assert a[-overlap:] == b[:overlap]
    # one pass: 42 distinct frames for a 0..41 span, no repeats
    assert pulled == sorted(set(pulled))
    assert len(pulled) == 42


def test_windows_from_video_segment_windows_line_up():
    """The production pairing: real ``video_segment_windows`` output read through
    the real reader. Window i must start at 8*s_i and the overlap must be the
    8*kv-7 carry band the geometry promises."""
    kv = 3
    layout = compute_chain_layout([25, 25, 25], 24.0, kv=kv)
    wins = video_segment_windows(layout)
    out = _windows(_fake_frames(layout.total_px), wins)

    for tensor, (start, length) in zip(out, wins):
        assert _values(tensor)[0] == start
        assert tensor.shape[2] == length
    # neighbours overlap by exactly 8*kv - 7 identical frames
    ov = 8 * kv - 7
    for i in range(len(wins) - 1):
        assert _values(out[i])[-ov:] == _values(out[i + 1])[:ov]
    # and the last window ends exactly at total_px - 1
    assert _values(out[-1])[-1] == layout.total_px - 1


def test_single_window_is_the_whole_reference():
    """clips=1 degenerates to the historical single (0, num_frames) window."""
    out = _windows(_fake_frames(25), [(0, 25)])
    assert len(out) == 1
    assert _values(out[0]) == [float(i) for i in range(25)]


def test_unequal_clip_lengths_still_line_up():
    """A long clip followed by a short one: the leftover buffered overlap is
    larger than the next window, and must be trimmed to that window's length."""
    wins = [(0, 49), (41, 25), (58, 9)]
    out = _windows(_fake_frames(80), wins)
    for tensor, (start, length) in zip(out, wins):
        assert tensor.shape[2] == length
        assert _values(tensor)[0] == start


# ── 2. short reference ───────────────────────────────────────────────────────


def test_short_reference_yields_partial_then_none():
    """Two full windows' worth of reference for a 4-window chain: window 2 comes
    out partial (only the frames that exist) and every window after it is None.
    Not an error — the owner rule is "generate the rest without a reference"."""
    wins = [(0, 25), (17, 25), (34, 25), (51, 25)]
    out = _windows(_fake_frames(42), wins)

    assert out[0] is not None and out[0].shape[2] == 25
    assert out[1] is not None and out[1].shape[2] == 25
    assert out[2] is not None
    assert out[2].shape[2] == 8            # frames 34..41, all that is left
    assert _values(out[2]) == [float(34 + k) for k in range(8)]
    assert out[3] is None


def test_reference_running_out_mid_first_window():
    """A reference shorter than even clip 0: window 0 is partial, every later
    window is None (their start is past the end of the stream)."""
    wins = [(0, 25), (17, 25), (34, 25)]
    out = _windows(_fake_frames(10), wins)
    assert out[0] is not None
    assert _values(out[0]) == [float(i) for i in range(10)]
    assert out[1] is None
    assert out[2] is None


def test_zero_frame_reference_yields_all_none():
    """A reference that decodes to nothing must not raise (and must never yield a
    0-frame tensor, which is what the VAE encode would choke on) — every window
    is simply None."""
    out = _windows(iter(()), [(0, 25), (17, 25)])
    assert out == [None, None]


def test_partial_length_is_not_rounded_down_to_8n_plus_1():
    """No 8n+1 rounding on a partial window: the wheel's VAE crops the tail of a
    short pixel run itself, so trimming here would only throw away reference
    frames. 8 leftover frames stay 8, not 1."""
    out = _windows(_fake_frames(42), [(0, 25), (17, 25), (34, 25)])
    assert out[2].shape[2] == 8


# ── 3. run_chain wiring: one reference conditioning per clip ─────────────────


class _Stop(Exception):
    """Halts run_chain once the stage-1 loop has visited every segment."""


class _LenientShape:
    def __init__(self, *a, **k):
        pass


V_LATENT = (1, 2, 4, 4, 4)   # (B, C, latent frames, H, W) for a 25-frame clip
A_LATENT = (1, 2, 64, 4)     # deliberately generous: only its K_a tail is sliced


class _FakeVideoLatentShape(_LenientShape):
    @classmethod
    def from_pixel_shape(cls, *a, **k):
        return cls()

    def to_torch_shape(self):
        return V_LATENT


class _FakeAudioLatentShape(_LenientShape):
    @classmethod
    def from_video_pixel_shape(cls, *a, **k):
        return cls()

    def to_torch_shape(self):
        return A_LATENT


_LTX_STUBS = {
    "ltx_core": {},
    "ltx_core.components": {},
    "ltx_core.model": {},
    "ltx_core.text_encoders": {},
    "ltx_pipelines": {},
    "ltx_pipelines.utils": {},
    "ltx_core.components.diffusion_steps": {"EulerDiffusionStep": object},
    "ltx_core.components.noisers": {"GaussianNoiser": _LenientShape},
    "ltx_core.model.audio_vae": {
        "decode_audio": lambda *a, **k: None,
        "encode_audio": lambda *a, **k: None,
    },
    "ltx_core.model.upsampler": {"upsample_video": lambda *a, **k: None},
    "ltx_core.model.video_vae": {"decode_video": lambda *a, **k: None},
    "ltx_core.text_encoders.gemma": {
        "encode_text": lambda te, prompts: [(object(), object())]
    },
    "ltx_core.types": {
        "AudioLatentShape": _FakeAudioLatentShape,
        "VideoLatentShape": _FakeVideoLatentShape,
        "VideoPixelShape": _LenientShape,
        "Audio": object,
    },
    "ltx_pipelines.utils.args": {"ImageConditioningInput": object},
    "ltx_pipelines.utils.constants": {
        "DISTILLED_SIGMA_VALUES": [1.0],
        "STAGE_2_DISTILLED_SIGMA_VALUES": [1.0],
    },
    "ltx_pipelines.utils.helpers": {
        "cleanup_memory": lambda *a, **k: None,
        "image_conditionings_by_adding_guiding_latent": lambda *a, **k: [],
        "image_conditionings_by_replacing_latent": lambda *a, **k: [],
    },
}

REF_SENTINEL = "ref-cond"


def _drive_stage1(monkeypatch, *, num_clips, ref_frames, clip_frames=25, kv=2):
    """Run ``chain_pipeline.run_chain``'s stage-1 loop with everything heavy
    faked out, and report what the reference plumbing did.

    The result carries:
      ``ref_calls``     one ``(first_frame_index, frame_count)`` per
                        ``_reference_conditioning_from_pixels`` call,
      ``seg_conds``     the conditioning list each segment's denoise received,
      ``decode_kwargs`` one dict per ``iter_video_conditioning_cpu`` call,
      ``events``        an ordered log of the heavy-build / job-state calls.
    """
    for name, attrs in _LTX_STUBS.items():
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        monkeypatch.setitem(sys.modules, name, mod)

    import engine.pipeline.chain_pipeline as cp

    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda *a, **k: None)
    monkeypatch.setattr(cp, "default_tiling_config", lambda **k: object())

    decode_kwargs: list[dict] = []

    def fake_iter(**kw):
        decode_kwargs.append(kw)
        return _fake_frames(ref_frames, frame_cap=int(kw["frame_cap"]))

    monkeypatch.setattr(cp, "iter_video_conditioning_cpu", fake_iter)

    ref_calls: list[tuple[float, int]] = []
    seg_conds: list[list] = []
    events: list = []
    cond_kwargs_seen: list[dict] = []
    strengths: list[float] = []

    def fake_denoise(**kw):
        seg_conds.append(list(kw["video_conditionings"]))
        if len(seg_conds) >= num_clips:
            raise _Stop()
        return (
            types.SimpleNamespace(latent=torch.zeros(V_LATENT)),
            types.SimpleNamespace(latent=torch.zeros(A_LATENT)),
        )

    monkeypatch.setattr(cp, "_denoise_av_with_carry", fake_denoise)

    class FakeLedger:
        def text_encoder(self):
            events.append("text_encoder")
            return object()

        def video_encoder(self):
            events.append("video_encoder")
            return object()

        def transformer(self):
            events.append("transformer")
            return object()

    class FakeDP:
        device = "cpu"
        model_ledger = FakeLedger()
        pipeline_components = types.SimpleNamespace(
            video_latent_channels=2, video_scale_factors=(8, 32, 32),
        )

    class FakeNag:
        requested = False

    class FakePipe:
        pipeline = FakeDP()
        _vae_spatial_tile_size = 0
        _vae_temporal_tile_size = 0
        _nag = FakeNag()

        def _set_nag_job(self, nag):
            events.append(("set_nag_job", nag))

        def _set_ic_job(self, loras, ref, attn):
            events.append(("set_ic_job", list(loras), ref, attn))

        def _reference_pixel_dims(self, cond_height, cond_width):
            events.append(("reference_pixel_dims", cond_height, cond_width))
            return 2, cond_height // 2, cond_width // 2

        def _reference_conditioning_from_pixels(self, video, *, cond_kwargs, scale, strength):
            ref_calls.append((float(video.reshape(-1)[0]), int(video.shape[2])))
            cond_kwargs_seen.append(cond_kwargs)
            strengths.append(strength)
            return [REF_SENTINEL]

    clips = [
        cp.ChainClipSpec(prompt=f"clip{i}", num_frames=clip_frames, images=[])
        for i in range(num_clips)
    ]
    with pytest.raises(_Stop):
        cp.run_chain(
            FakePipe(), clips=clips, width=384, height=256, frame_rate=24.0,
            num_steps=8, seed=1, overlap_frames=kv, overlap_strength=0.5,
            output_path="x.mp4", ic_loras=[],
            ic_reference=("/p/ref.mp4", 0.75), ic_attention_strength=1.0,
        )
    return types.SimpleNamespace(
        ref_calls=ref_calls, seg_conds=seg_conds, decode_kwargs=decode_kwargs,
        events=events, cond_kwargs=cond_kwargs_seen, strengths=strengths,
    )


def test_run_chain_injects_reference_once_per_clip(monkeypatch):
    """n clips -> n reference conditionings (the pre-§1-15 behaviour built
    exactly ONE, at clip 0), each carrying that clip's own window."""
    n = 4
    res = _drive_stage1(monkeypatch, num_clips=n, ref_frames=4000)

    layout = compute_chain_layout([25] * n, 24.0, kv=2)
    expected = video_segment_windows(layout)

    assert len(res.ref_calls) == n
    assert res.ref_calls == [(float(s), l) for s, l in expected]
    # every segment — including i >= 1, whose conds used to be hard-coded [] —
    # receives the reference conditioning
    assert len(res.seg_conds) == n
    for conds in res.seg_conds:
        assert REF_SENTINEL in conds


def test_run_chain_single_clip_matches_legacy_single_window(monkeypatch):
    """clips=1 still builds exactly one reference conditioning over the whole
    clip — the historical clip-0-only injection, unchanged."""
    res = _drive_stage1(monkeypatch, num_clips=1, ref_frames=4000)
    assert res.ref_calls == [(0.0, 25)]
    assert res.seg_conds[0] == [REF_SENTINEL]


def test_run_chain_encodes_at_half_res_with_the_requested_strength(monkeypatch):
    """Every per-clip encode uses the SAME half-resolution cond_kwargs the single
    generate() path uses at stage 1, and the reference strength from
    ``ic_reference`` — not a default."""
    res = _drive_stage1(monkeypatch, num_clips=3, ref_frames=4000)
    assert res.strengths == [0.75, 0.75, 0.75]
    for ck in res.cond_kwargs:
        assert ck["height"] == 128 and ck["width"] == 192   # 256//2, 384//2
        assert ck["dtype"] is torch.bfloat16
        assert ck["device"] == "cpu"
        assert "tiling_config" in ck                        # deblur's tiled encode
    # ...and the dims came through the shared helper, at stage-1 resolution.
    assert ("reference_pixel_dims", 128, 192) in res.events


def test_run_chain_forwards_reference_to_set_ic_job_before_the_build(monkeypatch):
    """_set_ic_job (which resolves the downscale factor + attention wrapper) still
    runs before the video_encoder/transformer are built — the reference plumbing
    did not disturb that ordering."""
    res = _drive_stage1(monkeypatch, num_clips=2, ref_frames=4000)
    set_ic = [e for e in res.events if isinstance(e, tuple) and e[0] == "set_ic_job"]
    assert set_ic == [("set_ic_job", [], ("/p/ref.mp4", 0.75), 1.0)]
    assert res.events.index(set_ic[0]) < res.events.index("video_encoder")
    assert res.events.index(set_ic[0]) < res.events.index("transformer")
    # NAG's job state is set even earlier (it must be encoded with the positives)
    assert res.events.index(("set_nag_job", None)) < res.events.index("text_encoder")


def test_run_chain_stops_injecting_when_reference_runs_out(monkeypatch):
    """A reference covering only the first couple of windows: the segments it
    reaches get one, the rest denoise with an EMPTY conditioning list rather
    than erroring."""
    n = 4
    layout = compute_chain_layout([25] * n, 24.0, kv=2)
    wins = video_segment_windows(layout)
    covered = wins[1][0] + wins[1][1]          # exactly two full windows

    res = _drive_stage1(monkeypatch, num_clips=n, ref_frames=covered)
    rc = res.ref_calls
    assert len(rc) == 3                        # 2 full + 1 partial
    assert rc[0] == (float(wins[0][0]), wins[0][1])
    assert rc[1] == (float(wins[1][0]), wins[1][1])
    assert rc[2][0] == float(wins[2][0])
    assert 0 < rc[2][1] < wins[2][1]           # partial, non-empty
    assert res.seg_conds[3] == []              # last clip: no reference, no error


def test_run_chain_caps_the_decode_at_the_last_window(monkeypatch):
    """The decode is asked for exactly what the last window can reach — decoding
    the rest of an over-long reference would be pure waste."""
    n = 3
    res = _drive_stage1(monkeypatch, num_clips=n, ref_frames=4000)
    layout = compute_chain_layout([25] * n, 24.0, kv=2)
    wins = video_segment_windows(layout)
    dk = res.decode_kwargs
    assert len(dk) == 1                        # ONE decode for the whole chain
    assert dk[0]["frame_cap"] == wins[-1][0] + wins[-1][1] == layout.total_px
    # ...at the reference resolution the shared helper resolved (scale 2 of the
    # HALF-res stage-1 dims 128x192)
    assert (dk[0]["height"], dk[0]["width"]) == (64, 96)


def test_run_chain_without_reference_opens_no_decode(monkeypatch):
    """ic_reference=None -> no decode, no injection, every segment's conditioning
    exactly what it was before §1-15 (byte-identity of the no-reference chain)."""
    import engine.pipeline.chain_pipeline as cp  # noqa: F401  (stubs installed below)

    ref_calls, seg_conds, decode_kwargs = _drive_stage1_no_reference(monkeypatch)
    assert ref_calls == []
    assert decode_kwargs == []
    assert all(c == [] for c in seg_conds)


def _drive_stage1_no_reference(monkeypatch, *, num_clips=2):
    """``_drive_stage1``'s ic_reference=None twin (the control case)."""
    for name, attrs in _LTX_STUBS.items():
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        monkeypatch.setitem(sys.modules, name, mod)

    import engine.pipeline.chain_pipeline as cp

    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda *a, **k: None)
    monkeypatch.setattr(cp, "default_tiling_config", lambda **k: object())

    decode_kwargs: list[dict] = []
    monkeypatch.setattr(
        cp, "iter_video_conditioning_cpu",
        lambda **kw: (decode_kwargs.append(kw), iter(()))[1],
    )

    seg_conds: list[list] = []

    def fake_denoise(**kw):
        seg_conds.append(list(kw["video_conditionings"]))
        if len(seg_conds) >= num_clips:
            raise _Stop()
        return (
            types.SimpleNamespace(latent=torch.zeros(V_LATENT)),
            types.SimpleNamespace(latent=torch.zeros(A_LATENT)),
        )

    monkeypatch.setattr(cp, "_denoise_av_with_carry", fake_denoise)

    ref_calls: list = []

    class FakeLedger:
        def text_encoder(self):
            return object()

        def video_encoder(self):
            return object()

        def transformer(self):
            return object()

    class FakeDP:
        device = "cpu"
        model_ledger = FakeLedger()
        pipeline_components = types.SimpleNamespace(
            video_latent_channels=2, video_scale_factors=(8, 32, 32),
        )

    class FakeNag:
        requested = False

    class FakePipe:
        pipeline = FakeDP()
        _vae_spatial_tile_size = 0
        _vae_temporal_tile_size = 0
        _nag = FakeNag()

        def _set_nag_job(self, nag):
            pass

        def _set_ic_job(self, loras, ref, attn):
            pass

        def _reference_conditioning_from_pixels(self, *a, **k):
            ref_calls.append(a)
            return ["unexpected"]

    clips = [
        cp.ChainClipSpec(prompt=f"clip{i}", num_frames=25, images=[])
        for i in range(num_clips)
    ]
    with pytest.raises(_Stop):
        cp.run_chain(
            FakePipe(), clips=clips, width=384, height=256, frame_rate=24.0,
            num_steps=8, seed=1, overlap_frames=2, overlap_strength=0.5,
            output_path="x.mp4", ic_loras=[], ic_reference=None,
        )
    return ref_calls, seg_conds, decode_kwargs
