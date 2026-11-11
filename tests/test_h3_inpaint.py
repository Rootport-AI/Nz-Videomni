"""engine_h3.inpaint_h3 and the single ``generate()``'s Inpainting path (台帳 §1-124, 計画 §4.6).

Runs in ``.venv-engine-h3`` through the engine-venv runner::

    .venv-engine-h3/Scripts/python.exe scripts/run_h3_pytest.py tests/test_h3_inpaint.py

CPU only, no weights: the mask read, the token values, ``_decode``'s
``crop_to``, ``validate_request``'s Inpainting block, and ``generate()`` end
to end on the tiny DiT (real ``_denoise`` / ``_decode`` /
``_encode_source_tail``; the text encoder, the VAEs and ``write_mp4`` are
stand-ins, as in ``test_h3_outpaint.py``).
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
pytest.importorskip("av")
pytest.importorskip("comfy_kitchen")

from engine_h3 import inpaint_h3 as IH  # noqa: E402
from engine_h3 import media_out as MO  # noqa: E402
from engine_h3 import pipeline_h3 as P  # noqa: E402
from engine_h3.inpaint_geometry import compute_inpaint_geometry  # noqa: E402
from engine_h3.outpaint_geometry import token_values  # noqa: E402
from engine_h3.source_tail import read_source_frames  # noqa: E402

#: 22 frames = chunks [0, 17) and [17, 22) = T' 7 latent frames.
F, T_LAT = 22, 7
_WRITE_MP4 = MO.write_mp4   # captured before any stub


def _write_video(path, frames_u8):
    _WRITE_MP4(str(path), torch.from_numpy(np.ascontiguousarray(frames_u8)), fps=24, audio=None)
    return path


# --------------------------------------------------------------------------- #
# read_inpaint_mask
# --------------------------------------------------------------------------- #

def _half_white_mask(tmp_path, frames=F, size=64):
    m = np.zeros((frames, size, size, 3), dtype=np.uint8)
    m[:, :, size // 2:] = 255
    return _write_video(tmp_path / "mask.mp4", m)


def test_read_inpaint_mask_blocks_and_proof(tmp_path):
    path = _half_white_mask(tmp_path)
    geom = compute_inpaint_geometry(64, 64, 64, 64)
    blocks, proof = IH.read_inpaint_mask(str(path), F, (64, 64), geom)
    assert blocks.dtype == np.bool_ and blocks.shape == (F, 2, 2)
    assert bool((blocks == np.array([[False, True], [False, True]])).all())
    assert proof["decoded_frames"] == F
    assert proof["white_ratio"] == pytest.approx(0.5, abs=0.02)


def test_read_inpaint_mask_refuses_a_wrong_count_or_size(tmp_path):
    path = _half_white_mask(tmp_path)
    geom = compute_inpaint_geometry(64, 64, 64, 64)
    with pytest.raises(RuntimeError, match="フレーム数が合いません"):
        IH.read_inpaint_mask(str(path), 39, (64, 64), geom)
    with pytest.raises(RuntimeError, match="フレーム数が合いません"):
        IH.read_inpaint_mask(str(path), 5, (64, 64), geom)
    with pytest.raises(RuntimeError, match="寸法が元動画と合いません"):
        IH.read_inpaint_mask(str(path), F, (96, 64), compute_inpaint_geometry(96, 64, 96, 64))


# --------------------------------------------------------------------------- #
# apply_inpaint_tokens
# --------------------------------------------------------------------------- #

def test_apply_inpaint_tokens_is_the_max_of_geometry_and_mask():
    # A 64 x 48 source on a 64 x 64 canvas: the bottom token row straddles the gap (value 1).
    geom = compute_inpaint_geometry(64, 48, 64, 64)
    values = torch.from_numpy(token_values(geom, T_LAT))
    assert bool((values[:, 1] == 1).all()) and bool((values[:, 0] == 0).all())
    mask = np.zeros((T_LAT, 2, 2), dtype=np.float32)
    mask[0:5, 0, 0] = 1.0
    clean = torch.zeros((1, 24, T_LAT, 4, 4))
    out_clean, out = IH.apply_inpaint_tokens((clean, values), mask)
    assert out_clean is clean and out.dtype == values.dtype and out.shape == values.shape
    assert bool((out[:, 1] == 1).all())
    assert bool((out[0:5, 0, 0] == 1).all()) and bool((out[5:, 0, 0] == 0).all())
    assert bool((out[:, 0, 1] == 0).all())
    with pytest.raises(RuntimeError, match="mask token values"):
        IH.apply_inpaint_tokens((clean, values), mask[:6])


# --------------------------------------------------------------------------- #
# _decode(crop_to=)
# --------------------------------------------------------------------------- #

def _decode_pipe(monkeypatch, frames_hw=(800, 1280), num_frames=5):
    from engine_h3 import audio_vae as AV
    from engine_h3 import video_vae as VV

    h, w = frames_hw
    rec = {"written": [], "composite_shape": []}

    class FakeDecoder(torch.nn.Module):
        def decode(self, z, tiled=True):
            out = torch.zeros((num_frames, h, w, 3), dtype=torch.uint8)
            out[:, :, :, 0] = torch.arange(h, dtype=torch.int64).remainder(256).to(torch.uint8)[None, :, None]
            return out

    class FakeAudio(torch.nn.Module):
        def decode(self, a):
            return torch.zeros((2, 800 * a.shape[-1]))

    def write_mp4(path, frames, *, fps, audio=None, sample_rate=32000):
        rec["written"].append(frames.clone())
        return 1

    monkeypatch.setattr(VV.H3VideoDecoder, "from_file", classmethod(lambda cls, *a, **k: FakeDecoder()))
    monkeypatch.setattr(AV.H3AudioVAE, "from_file", classmethod(lambda cls, *a, **k: FakeAudio()))
    monkeypatch.setattr(MO, "write_mp4", write_mp4)
    pipe = P.H3Pipeline(P.H3Paths("dit", "te", "vv", "av", "tok"), P.H3LoadOptions(blocks_on_gpu=1),
                        device="cpu", progress=lambda *a, **k: None, log=lambda m: None)
    req = SimpleNamespace(num_frames=num_frames, frame_rate=24, output_path="o.mp4")
    return pipe, req, rec


def test_decode_crop_to_cuts_after_the_composite(monkeypatch):
    pipe, req, rec = _decode_pipe(monkeypatch)

    def composite(frames_t):
        rec["composite_shape"].append(tuple(frames_t.shape))
        return frames_t

    entry: dict = {}
    pipe._decode(req, torch.zeros(1), torch.zeros((1, 64, 4)), entry, composite=composite, crop_to=(768, 1280))
    assert rec["composite_shape"] == [(5, 800, 1280, 3)]          # the paste-back sees the canvas
    frames = rec["written"][0]
    assert tuple(frames.shape) == (5, 768, 1280, 3) and frames.is_contiguous()
    assert int(frames[0, 767, 0, 0]) == 767 % 256                  # top-left kept
    assert entry["cropped_to"] == [1280, 768] and entry["frames"] == 5


def test_decode_without_crop_to_is_unchanged(monkeypatch):
    pipe, req, rec = _decode_pipe(monkeypatch)
    entry: dict = {}
    pipe._decode(req, torch.zeros(1), torch.zeros((1, 64, 4)), entry)
    assert tuple(rec["written"][0].shape) == (5, 800, 1280, 3) and "cropped_to" not in entry


def test_decode_crop_to_larger_than_the_frames_is_refused(monkeypatch):
    pipe, req, rec = _decode_pipe(monkeypatch)
    with pytest.raises(RuntimeError, match="crop_to"):
        pipe._decode(req, torch.zeros(1), torch.zeros((1, 64, 4)), {}, crop_to=(832, 1280))
    assert rec["written"] == []


# --------------------------------------------------------------------------- #
# validate_request
# --------------------------------------------------------------------------- #

def _req(tmp_path, **ip_kw):
    src, mask = tmp_path / "src.mp4", tmp_path / "mask.mp4"
    src.write_bytes(b"x")
    mask.write_bytes(b"x")
    kw = {"path": str(src), "mask_path": str(mask), "audio_strength": 0.5, "feather_px": 12, **ip_kw}
    return P.H3GenerateRequest(prompt="p", seed=1, width=1280, height=736, num_frames=124, output_path="o.mp4",
                               inpaint=P.H3InpaintInput(**kw))


def test_validate_request_accepts_an_inpaint_job(tmp_path):
    P.validate_request(_req(tmp_path))
    P.validate_request(_req(tmp_path, audio_strength=0, feather_px=0))
    P.validate_request(_req(tmp_path, audio_strength=1.0))


@pytest.mark.parametrize("ip_kw, match", [
    ({"path": ""}, "元動画のパスが空"),
    ({"mask_path": ""}, "マスク動画のパスが空"),
    ({"path": "none.mp4"}, "元動画が見つかりません"),
    ({"mask_path": "none.mp4"}, "マスク動画が見つかりません"),
    ({"audio_strength": 2.0}, "音声の強度は 0〜1"),
    ({"audio_strength": -0.1}, "音声の強度は 0〜1"),
    ({"audio_strength": float("nan")}, "音声の強度は 0〜1"),
    ({"audio_strength": "x"}, "音声の強度は数値"),
    ({"feather_px": -1}, "なじませ幅は 0 以上の整数"),
    ({"feather_px": 2.0}, "なじませ幅は 0 以上の整数"),
    ({"feather_px": True}, "なじませ幅は 0 以上の整数"),
])
def test_validate_request_refuses_bad_inpaint_values(tmp_path, ip_kw, match):
    if ip_kw.get("path") == "none.mp4":
        ip_kw = {"path": str(tmp_path / "none.mp4")}
    if ip_kw.get("mask_path") == "none.mp4":
        ip_kw = {"mask_path": str(tmp_path / "none.mp4")}
    with pytest.raises(ValueError, match=match):
        P.validate_request(_req(tmp_path, **ip_kw))


def test_validate_request_refuses_inpaint_with_other_conditioning(tmp_path):
    req = _req(tmp_path)
    other = tmp_path / "a.wav"
    other.write_bytes(b"x")
    with pytest.raises(ValueError, match="Outpainting と併用できません"):
        P.validate_request(replace(req, outpaint=P.H3OutpaintInput(path=str(other), pad_left=32)))
    with pytest.raises(ValueError, match="キーフレーム"):
        P.validate_request(replace(req, images=(P.H3ImageInput(str(other), 0),)))
    with pytest.raises(ValueError, match="A2V"):
        P.validate_request(replace(req, source_audio=P.H3AudioInput(str(other), 0.5)))
    with pytest.raises(ValueError, match="Ref2VA"):
        P.validate_request(replace(req, references=(P.H3ReferenceInput("audio", str(other), 1),)))


# --------------------------------------------------------------------------- #
# generate() end to end on the tiny DiT
# --------------------------------------------------------------------------- #

#: A 64 x 48 source on a 64 x 64 canvas (the bottom 16 px mirrored, cut back after the decode).
SRC_W, SRC_H, CAN_W, CAN_H = 64, 48, 64, 64
DECODED_GREY = 77


def _inpaint_job(monkeypatch, tmp_path, *, feather=0, src_wh=(SRC_W, SRC_H), mask="block"):
    """``mask``: "block" (the top-left block in the first chunk), "empty" (no white) or "full" (all white)."""
    from test_h3_engine import TEXT, _fake_int8, _tiny_dit_tensors, _write_sft

    from engine_h3 import audio_vae as AV
    import h3_format
    from engine_h3 import video_vae as VV

    g = torch.Generator().manual_seed(5)
    sw, sh = src_wh
    cw, ch = -(-sw // 32) * 32, -(-sh // 32) * 32
    src_u8 = torch.randint(0, 256, (F, sh, sw, 3), generator=g, dtype=torch.uint8).numpy()
    src = _write_video(tmp_path / "src.mp4", src_u8)
    source_u8 = read_source_frames(str(src), F)
    # White on the top-left block in the first chunk's frames only.
    mask_u8 = np.zeros((F, sh, sw, 3), dtype=np.uint8)
    if mask == "block":
        mask_u8[:17, 0:32, 0:32] = 255
    elif mask == "full":
        mask_u8[...] = 255
    mask = _write_video(tmp_path / "mask.mp4", mask_u8)
    _fake_int8(monkeypatch)
    dit_path = tmp_path / "tiny.safetensors"
    _write_sft(dit_path, _tiny_dit_tensors(layers=3, refiner=2, grid=1025, rank=8))
    _header, cfg = h3_format.inspect_h3_dit(str(dit_path))
    logs = []
    pipe = P.H3Pipeline(P.H3Paths(str(dit_path), "te", "vv", "av", "tok"), P.H3LoadOptions(blocks_on_gpu=1),
                        device="cpu", progress=lambda *a, **k: None, log=logs.append)
    pipe._dit_cfg = cfg
    rec = {"enc_frames": [], "latents": [], "written": [], "denoise_kw": []}

    def encode(prompt, phases, **kw):
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
            return torch.full((F, ch, cw, 3), DECODED_GREY, dtype=torch.uint8)

    class FakeAudio(torch.nn.Module):
        def decode(self, a):
            return torch.zeros((2, 800 * a.shape[-1]))

    def write_mp4(path, frames, *, fps, audio=None, sample_rate=32000):
        rec["written"].append(frames.clone())
        return 1

    real_denoise = pipe._denoise

    def denoise(*a, **k):
        rec["denoise_kw"].append(k)
        return real_denoise(*a, **k)

    monkeypatch.setattr(pipe, "_encode", encode)
    monkeypatch.setattr(pipe, "_denoise", denoise)
    monkeypatch.setattr(pipe, "_build_video_encoder", lambda: FakeVideoEncoder())
    monkeypatch.setattr(VV.H3VideoDecoder, "from_file", classmethod(lambda cls, *a, **k: FakeDecoder()))
    monkeypatch.setattr(AV.H3AudioVAE, "from_file", classmethod(lambda cls, *a, **k: FakeAudio()))
    monkeypatch.setattr(MO, "write_mp4", write_mp4)
    req = P.H3GenerateRequest(
        prompt="p", seed=7, width=cw, height=ch, num_frames=F, output_path=str(tmp_path / "o.mp4"),
        num_inference_steps=2,
        inpaint=P.H3InpaintInput(path=str(src), mask_path=str(mask), audio_strength=0.0, feather_px=feather),
    )
    return pipe, req, source_u8, rec, logs


def test_generate_inpaint_end_to_end(monkeypatch, tmp_path):
    pipe, req, source_u8, rec, logs = _inpaint_job(monkeypatch, tmp_path)
    done = pipe.generate(req)
    # The source alone at the canvas, its bottom 16 rows mirrored (aligned_frames).
    assert rec["enc_frames"][0].shape == (F, CAN_H, CAN_W, 3)
    assert np.array_equal(rec["enc_frames"][0][:, :SRC_H], source_u8)
    # The token values: the bottom row (the gap) always, the top-left token in the first chunk's latents.
    values = rec["denoise_kw"][0]["video_fix"][1]
    assert values.shape == (T_LAT, 2, 2)
    assert bool((values[:, 1] == 1).all()) and bool((values[:, 0, 1] == 0).all())
    assert values[:, 0, 0].tolist() == [1, 1, 1, 1, 1, 0, 0]
    # The fixed cells came back as the source's latents (the self-check passed).
    z = rec["latents"][0]
    assert torch.equal(z[:, :, 5:, 0:2, 0:2], rec["enc_out"][:, :, 5:, 0:2, 0:2])
    assert torch.equal(z[:, :, :, 0:2, 2:4], rec["enc_out"][:, :, :, 0:2, 2:4])
    # Cut back to the source size; F = 0 is a hard paste: the mask's block is the decode, the rest the source.
    frames = rec["written"][0]
    assert tuple(frames.shape) == (F, SRC_H, SRC_W, 3)
    out = frames.numpy()
    assert bool((out[:17, 0:32, 0:32] == DECODED_GREY).all())
    assert np.array_equal(out[:17, :, 32:], source_u8[:17, :, 32:])
    assert np.array_equal(out[:17, 32:], source_u8[:17, 32:])
    assert np.array_equal(out[17:], source_u8[17:])
    h3 = done["h3"]
    assert "outpaint" not in h3
    ip = h3["inpaint"]
    proof = ip.pop("mask_proof")
    assert ip == {
        "canvas": [CAN_W, CAN_H], "pads": {"left": 0, "right": 0, "top": 0, "bottom": 16},
        "source": [SRC_W, SRC_H], "mask_rule": IH.MASK_RULE, "chunk_frames": 17,
        "chunks": [[0, 17], [17, 22]], "latent_frames": T_LAT,
        "fixed_tokens_total": 9, "generated_tokens_total": 19,
        "generated_per_latent_frame": [3, 3, 3, 3, 3, 2, 2],
        "audio_strength": 0.0, "source_had_audio": False, "audio_latents_encoded": 0,
        "audio_output": "generated", "feather_px": 0, "inject_clean": 1.0,
        "self_check": {"fixed_latents_equal": True}, "cropped_to": [SRC_W, SRC_H],
    }
    assert proof["decoded_frames"] == F and proof["token_ratio"] == pytest.approx(5 / 28)
    assert proof["white_ratio"] == pytest.approx(17 * 32 * 32 / (F * SRC_H * SRC_W), abs=0.01)
    assert h3["phases"]["decode"]["cropped_to"] == [SRC_W, SRC_H]
    assert any(m.startswith("inpaint: canvas 64x64 source 64x48") for m in logs)
    assert not any(m.startswith("outpaint:") for m in logs)


def test_generate_inpaint_feather_blends_outward(monkeypatch, tmp_path):
    pipe, req, source_u8, rec, _logs = _inpaint_job(monkeypatch, tmp_path, feather=8)
    done = pipe.generate(req)
    out = rec["written"][0].numpy()
    assert bool((out[:17, 0:32, 0:32] == DECODED_GREY).all())          # inside the mask: the decode
    assert np.array_equal(out[:17, :, 40:], source_u8[:17, :, 40:])     # 8 px and more away: the source
    assert np.array_equal(out[17:], source_u8[17:])                     # the second chunk has no mask
    assert done["h3"]["inpaint"]["feather_px"] == 8


def test_generate_inpaint_refuses_a_canvas_not_rounded_up(monkeypatch, tmp_path):
    pipe, req, _source_u8, rec, _logs = _inpaint_job(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="切り上げた寸法"):
        pipe.generate(replace(req, width=96))
    assert rec["denoise_kw"] == []


def test_generate_inpaint_empty_mask_on_an_aligned_source_is_the_source(monkeypatch, tmp_path):
    # 64 x 64 on 64 x 64 (no pad) with no white pixel: every token fixed, the output is the source.
    pipe, req, source_u8, rec, _logs = _inpaint_job(monkeypatch, tmp_path, src_wh=(64, 64), mask="empty")
    done = pipe.generate(req)
    assert torch.equal(rec["latents"][0], rec["enc_out"])                # every cell fixed (self-check passed)
    frames = rec["written"][0].numpy()
    assert frames.shape == (F, 64, 64, 3) and np.array_equal(frames, source_u8)
    ip = done["h3"]["inpaint"]
    assert ip["pads"] == {"left": 0, "right": 0, "top": 0, "bottom": 0}
    assert ip["cropped_to"] == [64, 64] and ip["self_check"] == {"fixed_latents_equal": True}
    assert ip["generated_tokens_total"] == 0 and ip["fixed_tokens_total"] == T_LAT * 2 * 2
    assert ip["mask_proof"]["white_ratio"] == 0.0 and ip["mask_proof"]["token_ratio"] == 0.0


def test_generate_inpaint_full_mask_generates_every_token(monkeypatch, tmp_path):
    # All white: no fixed cell (the self-check has nothing to compare and passes), all decoded.
    pipe, req, _source_u8, rec, _logs = _inpaint_job(monkeypatch, tmp_path, src_wh=(64, 64), mask="full")
    done = pipe.generate(req)
    frames = rec["written"][0].numpy()
    assert frames.shape == (F, 64, 64, 3) and bool((frames == DECODED_GREY).all())
    ip = done["h3"]["inpaint"]
    assert ip["generated_tokens_total"] == ip["latent_frames"] * 2 * 2 == T_LAT * 4
    assert ip["fixed_tokens_total"] == 0 and ip["self_check"] == {"fixed_latents_equal": True}
    assert ip["mask_proof"]["token_ratio"] == 1.0
    assert ip["mask_proof"]["white_ratio"] == pytest.approx(1.0, abs=0.01)
