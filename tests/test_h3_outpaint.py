"""engine_h3.outpaint_h3 and the single ``generate()``'s Outpainting path (台帳 §1-125, 計画 §2.3・§6.1).

Runs in ``.venv-engine-h3`` through the engine-venv runner::

    .venv-engine-h3/Scripts/python.exe scripts/run_h3_pytest.py tests/test_h3_outpaint.py

CPU only, no weights: the fixed tokens and the audio values of ``outpaint_fix``,
the self-check, the source read's audio rule, and ``generate()`` end to end on
the tiny DiT (real ``_denoise`` / ``_decode`` / ``_encode_source_tail``; the
text encoder, the VAEs and ``write_mp4`` are stand-ins).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
pytest.importorskip("av")
pytest.importorskip("comfy_kitchen")

from engine_h3 import media_out as MO  # noqa: E402
from engine_h3 import outpaint_h3 as OH  # noqa: E402
from engine_h3 import pipeline_h3 as P  # noqa: E402
from engine_h3.outpaint_geometry import compute_outpaint_geometry  # noqa: E402
from engine_h3.source_tail import read_source_frames  # noqa: E402

#: The end-to-end job: a 64 x 64 source, 22 frames (T' = 7, A = 37), on a 128 x 96
#: canvas with pads left 32 / right 32 / top 0 / bottom 32 (grid 3 x 4; the box
#: [32, 96) x [0, 64) is token-aligned, so its 2 x 2 tokens are all fixed).
F, T_LAT, A_LAT = 22, 7, 37
SRC_W, SRC_H, CAN_W, CAN_H = 64, 64, 128, 96
PADS = {"pad_left": 32, "pad_right": 32, "pad_top": 0, "pad_bottom": 32}
DECODED_GREY = 77


def _geom():
    return compute_outpaint_geometry(SRC_W, SRC_H, CAN_W, CAN_H, 32, 32, 0, 32)


def _write_source(path, audio=False, frames=F):
    """A synthetic mp4 written with the real ``write_mp4`` (captured before any stub)."""
    g = torch.Generator().manual_seed(5)
    video = torch.randint(0, 256, (frames, SRC_H, SRC_W, 3), generator=g, dtype=torch.uint8)
    wav = None
    if audio:
        n = round(frames * 32000 / 24)
        t = torch.arange(n, dtype=torch.float32) / 32000
        wav = torch.stack([torch.sin(2 * torch.pi * 440 * t), torch.sin(2 * torch.pi * 220 * t)]) * 0.2
    MO.write_mp4(str(path), video, fps=24, audio=wav)
    return path


# --------------------------------------------------------------------------- #
# outpaint_fix / check_outpaint_fixed
# --------------------------------------------------------------------------- #

def _bbox_latent(seed=3):
    geom = _geom()
    return torch.randn((1, 24, T_LAT, geom.aligned_h // 16, geom.aligned_w // 16),
                       generator=torch.Generator().manual_seed(seed))


def test_outpaint_fix_places_the_box_and_marks_the_tokens():
    geom = _geom()
    bbox = _bbox_latent()
    (clean_v, values), audio_fix = OH.outpaint_fix(bbox, None, geom, 0.5, T_LAT, A_LAT)
    assert audio_fix is None
    assert clean_v.shape == (1, 24, T_LAT, CAN_H // 16, CAN_W // 16) and clean_v.dtype == torch.float32
    oy, ox = geom.latent_offset
    assert (oy, ox) == (0, 2)
    assert torch.equal(clean_v[:, :, :, oy:oy + 4, ox:ox + 4], bbox)
    outside = clean_v.clone()
    outside[:, :, :, oy:oy + 4, ox:ox + 4] = 0
    assert not bool(outside.any())                         # zero outside the box
    assert values.shape == (T_LAT, 3, 4) and values.dtype == torch.float32
    want = torch.ones(3, 4)
    want[0:2, 1:3] = 0.0                                   # the 2 x 2 tokens inside the kept rectangle
    assert all(torch.equal(values[t], want) for t in range(T_LAT))


@pytest.mark.parametrize("s, kept", [(0.5, 0.5), (0.25, 0.75), (1.0, 0.0)])
def test_outpaint_fix_audio_values(s, kept):
    clean_a = torch.randn(2, 32, 30, generator=torch.Generator().manual_seed(4))
    _video_fix, (full_a, per_a) = OH.outpaint_fix(_bbox_latent(), clean_a, _geom(), s, T_LAT, A_LAT)
    assert full_a.shape == (2, 32, A_LAT)
    assert torch.equal(full_a[:, :, :30], clean_a) and not bool(full_a[:, :, 30:].any())
    assert per_a.tolist() == [kept] * 30 + [1.0] * 7


def test_outpaint_fix_refuses_a_wrong_box_latent():
    with pytest.raises(RuntimeError, match="source latent"):
        OH.outpaint_fix(torch.zeros(1, 24, T_LAT, 4, 3), None, _geom(), 0.0, T_LAT, A_LAT)


def _fixed_result(s=1.0):
    clean_a = torch.randn(2, 32, 36, generator=torch.Generator().manual_seed(6))
    video_fix, audio_fix = OH.outpaint_fix(_bbox_latent(), clean_a, _geom(), s, T_LAT, A_LAT)
    g = torch.Generator().manual_seed(9)
    video = torch.randn(video_fix[0].shape, generator=g)
    cells = OH.fixed_cells(video_fix[1])
    video[:, :, cells] = video_fix[0][:, :, cells]
    audio = torch.randn(2, 32, A_LAT, generator=g)
    audio[:, :, :36] = audio_fix[0][:, :, :36]
    return video, video_fix, audio, audio_fix


def test_fixed_cells_are_the_tokens_doubled():
    cells = OH.fixed_cells(_fixed_result()[1][1])
    assert cells.shape == (T_LAT, CAN_H // 16, CAN_W // 16) and cells.dtype == torch.bool
    assert int(cells.sum()) == T_LAT * 4 * 4 and bool(cells[:, 0:4, 2:6].all())


def test_check_outpaint_fixed_passes_and_refuses():
    video, video_fix, audio, audio_fix = _fixed_result()
    OH.check_outpaint_fixed(video, video_fix, audio, audio_fix, 1.0, 36)
    OH.check_outpaint_fixed(video, video_fix, audio, None, 0.0, 0)
    bad = video.clone()
    bad[0, 5, 3, 1, 3] += 1e-3                              # one fixed latent cell moved
    with pytest.raises(RuntimeError, match="自己検査"):
        OH.check_outpaint_fixed(bad, video_fix, audio, audio_fix, 1.0, 36)
    moved = video.clone()
    moved[0, 5, 3, 5, 0] += 1.0                              # a generated cell: not checked
    OH.check_outpaint_fixed(moved, video_fix, audio, audio_fix, 1.0, 36)
    bad_a = audio.clone()
    bad_a[0, 0, 2] += 1.0
    with pytest.raises(RuntimeError, match="自己検査"):
        OH.check_outpaint_fixed(video, video_fix, bad_a, audio_fix, 1.0, 36)
    # Below strength 1.0 the read audio rows are not fixed: not checked.
    OH.check_outpaint_fixed(video, video_fix, bad_a, audio_fix, 0.5, 36)


# --------------------------------------------------------------------------- #
# read_outpaint_source (the audio rule; the video read is source_tail's)
# --------------------------------------------------------------------------- #

def test_read_outpaint_source_without_audio(tmp_path):
    path = _write_source(tmp_path / "v.mp4", audio=False)
    frames, wave, info = OH.read_outpaint_source(str(path), F, 24, 0.0)
    assert frames.shape == (F, SRC_H, SRC_W, 3) and wave is None
    assert info == {"source_size": [SRC_W, SRC_H], "frames_read": F, "source_had_audio": False,
                    "required_samples": 29333, "read_samples": 0}
    with pytest.raises(RuntimeError, match="音声トラックがありません"):
        OH.read_outpaint_source(str(path), F, 24, 0.5)


def test_read_outpaint_source_strength_zero_reads_no_audio(tmp_path, monkeypatch):
    path = _write_source(tmp_path / "a.mp4", audio=True)
    import engine_h3.source_audio as SA

    monkeypatch.setattr(SA, "load_source_waveform", lambda *a, **k: pytest.fail("read at strength 0"))
    _frames, wave, info = OH.read_outpaint_source(str(path), F, 24, 0.0)
    assert wave is None and info["source_had_audio"] is True


@pytest.mark.parametrize("s, keep", [(0.5, 1), (1.0, 29333)])
def test_read_outpaint_source_audio_rule(tmp_path, monkeypatch, s, keep):
    path = _write_source(tmp_path / "a.mp4", audio=True)
    from test_h3_engine import _a2v_fakes

    wave_t, calls = _a2v_fakes(monkeypatch, samples=29333)
    _frames, wave, info = OH.read_outpaint_source(str(path), F, 24, s)
    assert calls["load"] == [(str(path), keep, 29333)]
    assert wave is wave_t and info["read_samples"] == 29333


def test_read_outpaint_source_unreadable_audio_below_one_is_free(tmp_path, monkeypatch):
    path = _write_source(tmp_path / "a.mp4", audio=True)
    import engine_h3.source_audio as SA

    def fail(*a, **k):
        raise ValueError("no samples")

    monkeypatch.setattr(SA, "load_source_waveform", fail)
    logs = []
    _frames, wave, _info = OH.read_outpaint_source(str(path), F, 24, 0.5, log=logs.append)
    assert wave is None and any("generating audio freely" in m for m in logs)
    with pytest.raises(ValueError, match="no samples"):
        OH.read_outpaint_source(str(path), F, 24, 1.0)


# --------------------------------------------------------------------------- #
# generate() end to end on the tiny DiT
# --------------------------------------------------------------------------- #

def _outpaint_job(monkeypatch, tmp_path, *, s=0.0, audio=False, feather=0):
    from test_h3_engine import TEXT, _fake_int8, _tiny_dit_tensors, _write_sft

    from engine_h3 import audio_vae as AV
    import h3_format
    from engine_h3 import video_vae as VV

    src = _write_source(tmp_path / "src.mp4", audio=audio)
    source_u8 = read_source_frames(str(src), F)
    _fake_int8(monkeypatch)
    dit_path = tmp_path / "tiny.safetensors"
    _write_sft(dit_path, _tiny_dit_tensors(layers=3, refiner=2, grid=1025, rank=8))
    _header, cfg = h3_format.inspect_h3_dit(str(dit_path))
    progress, logs = [], []
    pipe = P.H3Pipeline(P.H3Paths(str(dit_path), "te", "vv", "av", "tok"), P.H3LoadOptions(blocks_on_gpu=1),
                        device="cpu", progress=lambda *a, **k: progress.append(a), log=logs.append)
    pipe._dit_cfg = cfg
    rec = {"encode_kw": [], "enc_frames": [], "latents": [], "written": []}

    def encode(prompt, phases, **kw):
        rec["encode_kw"].append(kw)
        return (torch.randn(5, TEXT, generator=torch.Generator().manual_seed(3)).to(torch.bfloat16),
                torch.ones(5, dtype=torch.long))

    class FakeVideoEncoder:
        def encode_clips(self, frames, *, device, tiled=True):
            rec["enc_frames"].append(frames.copy())
            out = torch.randn((1, 24, T_LAT, frames.shape[1] // 16, frames.shape[2] // 16),
                              generator=torch.Generator().manual_seed(17))
            rec["enc_out"] = out.clone()
            return out

    class FakeDecoder(torch.nn.Module):
        def decode(self, z, tiled=True):
            rec["latents"].append(z.clone())
            return torch.full((F, CAN_H, CAN_W, 3), DECODED_GREY, dtype=torch.uint8)

    class FakeAudio(torch.nn.Module):
        def decode(self, a):
            return torch.zeros((2, 800 * a.shape[-1]))

    def write_mp4(path, frames, *, fps, audio=None, sample_rate=32000):
        rec["written"].append((frames.clone(), None if audio is None else audio.clone()))
        return 1

    monkeypatch.setattr(pipe, "_encode", encode)
    monkeypatch.setattr(pipe, "_build_video_encoder", lambda: FakeVideoEncoder())
    monkeypatch.setattr(VV.H3VideoDecoder, "from_file", classmethod(lambda cls, *a, **k: FakeDecoder()))
    monkeypatch.setattr(AV.H3AudioVAE, "from_file", classmethod(lambda cls, *a, **k: FakeAudio()))
    monkeypatch.setattr(MO, "write_mp4", write_mp4)
    req = P.H3GenerateRequest(
        prompt="p", seed=7, width=CAN_W, height=CAN_H, num_frames=F, output_path=str(tmp_path / "o.mp4"),
        num_inference_steps=2,
        outpaint=P.H3OutpaintInput(path=str(src), audio_strength=s, feather_px=feather, **PADS),
    )
    return pipe, req, source_u8, rec, progress, logs


def test_generate_outpaint_end_to_end_strength_zero(monkeypatch, tmp_path):
    pipe, req, source_u8, rec, progress, logs = _outpaint_job(monkeypatch, tmp_path, s=0.0)
    monkeypatch.setattr(pipe, "_encode_audio", lambda *a, **k: pytest.fail("audio encoder built at strength 0"))
    done = pipe.generate(req)
    # The encode stage counts the source's encode as the key frames' step.
    assert rec["encode_kw"][0]["has_cond"] is True and ("encode", 4, 4) in progress
    # The source alone at the 32-aligned box (here the box is the source itself).
    assert len(rec["enc_frames"]) == 1 and np.array_equal(rec["enc_frames"][0], source_u8)
    # The self-check passed; the decoder got the box's latents on the fixed cells bit for bit.
    z = rec["latents"][0]
    assert z.shape == (1, 24, T_LAT, CAN_H // 16, CAN_W // 16)
    assert torch.equal(z[:, :, :, 0:4, 2:6], rec["enc_out"])
    # The paste-back (F = 0): the kept rectangle is the source bit for bit, the rest is the decode.
    frames, audio = rec["written"][0]
    assert frames.shape == (F, CAN_H, CAN_W, 3) and frames.dtype == torch.uint8
    assert np.array_equal(frames[:, 0:64, 32:96].numpy(), source_u8)
    outside = frames.clone()
    outside[:, 0:64, 32:96] = DECODED_GREY
    assert bool((outside == DECODED_GREY).all())
    assert audio is not None and audio.shape[0] == 2          # the decoded (free) audio
    h3 = done["h3"]
    assert h3["outpaint"] == {
        "canvas": [CAN_W, CAN_H], "pads": {"left": 32, "right": 32, "top": 0, "bottom": 32},
        "source": [SRC_W, SRC_H], "aligned": [32, 0, 96, 64], "fixed_tokens": 4, "generated_tokens": 8,
        "feather_px": 0, "audio_strength": 0.0, "source_had_audio": False, "audio_latents_encoded": 0,
        "audio_output": "generated", "inject_clean": 1.0,
    }
    assert h3["phases"]["denoise"]["mask"]["video"]["inject_clean"] == 1.0
    assert h3["phases"]["denoise"]["mask"]["video"]["rows"] == T_LAT * 12
    assert "audio" not in h3["phases"]["denoise"]["mask"]
    assert h3["phases"]["source_encode"]["aligned"] == [32, 0, 96, 64]
    assert "a2v" not in h3 and done["width"] == CAN_W and done["height"] == CAN_H
    assert any(m.startswith("outpaint: canvas 128x96") and "fixed=4 generated=8" in m for m in logs)
    assert any(m.startswith("outpaint: fixed rows from the source video") for m in logs)


def test_generate_outpaint_strength_one_muxes_the_source_audio(monkeypatch, tmp_path):
    from test_h3_engine import _a2v_fakes

    pipe, req, source_u8, rec, _progress, _logs = _outpaint_job(monkeypatch, tmp_path, s=1.0, audio=True,
                                                                feather=8)
    wave, calls = _a2v_fakes(monkeypatch, samples=29333)
    done = pipe.generate(req)
    assert calls["build"] == 1 and calls["encode"][0].shape == (2, 29333)
    frames, audio = rec["written"][0]
    assert torch.equal(audio, wave)                          # the original waveform, not a decode
    # A feather band: the inner rows / columns are the source, the seam side is mixed.
    assert np.array_equal(frames[:, 0:56, 40:88].numpy(), source_u8[:, 0:56, 8:56])
    rec_op = done["h3"]["outpaint"]
    assert rec_op["audio_output"] == "original" and rec_op["audio_latents_encoded"] == 29333 // 800
    assert rec_op["source_had_audio"] is True and rec_op["feather_px"] == 8
    assert done["h3"]["phases"]["denoise"]["mask"]["audio"]["max"] == 1.0
    assert done["h3"]["phases"]["decode"]["audio_vae_source"] == "skipped"


def test_generate_outpaint_fixed_cells_that_move_are_refused(monkeypatch, tmp_path):
    pipe, req, _source_u8, rec, _progress, _logs = _outpaint_job(monkeypatch, tmp_path)
    real = pipe._denoise

    def broken(*a, **k):
        video, audio, used = real(*a, **k)
        video = video.clone()
        video[0, 0, 0, 0, 2] += 1.0
        return video, audio, used

    monkeypatch.setattr(pipe, "_denoise", broken)
    with pytest.raises(RuntimeError, match="自己検査"):
        pipe.generate(req)
    assert rec["latents"] == []                              # stopped before the decode


def test_generate_without_outpaint_passes_no_video_fix(monkeypatch, tmp_path):
    pipe, req, _source_u8, _rec, _progress, _logs = _outpaint_job(monkeypatch, tmp_path)
    seen = []

    class Stop(Exception):
        pass

    def denoise(*a, **k):
        seen.append(k)
        raise Stop

    monkeypatch.setattr(pipe, "_denoise", denoise)
    from dataclasses import replace

    with pytest.raises(Stop):
        pipe.generate(replace(req, outpaint=None))
    assert "video_fix" not in seen[0] and "video_inject_clean" not in seen[0]
