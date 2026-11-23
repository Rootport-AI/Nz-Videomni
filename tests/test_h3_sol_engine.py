"""Sol-H3-VAE in the MiniMax H3 engine (``engine_h3.sol_vae``, ``pipeline_h3``'s
``sol_start`` / ``_decode`` / ``sol_record``, the chain, the worker's ``sol_vae`` key).

Runs in ``.venv-engine-h3`` through the engine-venv runner, without a GPU::

    CUDA_VISIBLE_DEVICES=-1 .venv-engine-h3/Scripts/python.exe scripts/run_h3_pytest.py tests/test_h3_sol_engine.py

What is pinned:
  * the decoder's structure (windows, shifts, global tokens, the block-by-block order, the
    frame trimming) against an all-at-once reference built from the pi-H3 tests' all-rows
    layer (``test_h3_pih3_engine._old_layer`` over ``dwd.build_geometry``), small random
    float64 weights on the CPU; a wrong temporal shift is caught; the fp16 code path stays
    near the reference; a synthetic INT8 file is read and decoded through comfy-kitchen's
    int8_linear on the CPU;
  * the file check: what does not fit is stated, nothing raised;
  * ``sol_start`` (once at the start: no file -> "file_missing", a broken file refuses the job
    before the text encoder) and ``validate_request``'s pi-H3 + Sol refusal;
  * ``_decode``: the size judged on the latents (inside / exactly / above
    ``MAX_LATENT_TOKENS``), the stock decoder untouched when Sol decodes, the records;
  * ``generate`` and the chain on the tiny DiT: "on" / "on->off" and ``h3.sol_vae``; a job
    without ``sol_vae`` reaches none of the Sol code and its records are unchanged; the chain's
    Sol decoder gets the joined latents;
  * the worker's ``sol_vae`` key (single and chain), an empty path allowed.
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("comfy_kitchen")

import torch.nn.functional as F  # noqa: E402

from engine_h3 import dwd  # noqa: E402
from engine_h3 import sol_vae as SV  # noqa: E402
from engine_h3 import video_vae as VV  # noqa: E402
from test_h3_pih3_engine import _old_layer, _pi_job_pipe  # noqa: E402

#: The released decoder's rules with tiny widths (dim 128, 2 heads of 64, 4 layers).
TINY = SV.SolDims(dim=128, heads=2, head_dim=64, num_layers=4)


def _random_model(dtype=torch.float64, seed=20261123, dims=TINY):
    g = torch.Generator().manual_seed(seed)
    model = SV.SolModel(dims, None, dtype)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith(("scale1", "scale2")):
                p.copy_(0.5 * torch.randn(p.shape, generator=g, dtype=torch.float64))
            elif "norm" in name and name.endswith("weight"):
                p.copy_(1 + 0.1 * torch.randn(p.shape, generator=g, dtype=torch.float64))
            elif p.ndim > 1:
                p.copy_(torch.randn(p.shape, generator=g, dtype=torch.float64) / max(1, p.shape[-1]) ** 0.5)
            else:
                p.copy_(0.05 * torch.randn(p.shape, generator=g, dtype=torch.float64))
    g2 = torch.Generator().manual_seed(seed + 1)
    mean = torch.randn(24, generator=g2).tolist()
    std = (torch.rand(24, generator=g2) + 0.5).tolist()
    return model, mean, std


def _reference_decode(model, mean, std, z):
    """All at once (float64): every token row and every block's global rows in one stream,
    each layer through the pi-H3 tests' all-rows layer on ``dwd.build_geometry``."""
    dims, dec = model.dims, model.decoder
    T, h, w = (int(v) for v in z.shape[2:])
    N, D = T * h * w, dims.dim
    geo = dwd.build_geometry(T, h, w, dims.rope_dim, SV.NUM_GLOBAL)
    m = torch.tensor(mean, dtype=torch.float64).view(-1, 1, 1, 1)
    s = torch.tensor(std, dtype=torch.float64).view(-1, 1, 1, 1)
    raw = (z[0].double() * s + m).permute(1, 2, 3, 0).reshape(N, -1)
    pq = model.post_quant_conv
    X = torch.empty(geo.R, D, dtype=torch.float64)
    X[:N] = F.linear(F.linear(raw, pq.weight.reshape(pq.weight.shape[0], -1), pq.bias),
                     dec.x_embedder.weight, dec.x_embedder.bias)
    glob = torch.cat([dec.register_tokens[0], torch.zeros(1, D, dtype=torch.float64)])
    X[N:] = glob.repeat(geo.nb, 1)
    rope = {k: (c, sn) for k, (c, sn) in geo.rope.items()}
    for li, blk in enumerate(dec.blocks):
        _old_layer(blk, X, geo, li % 2 == 1, dims, 8192, 16, rope)
    y = F.layer_norm(X[:N], (D,), dec.norm_out.weight, dec.norm_out.bias, SV.EPS)
    p = F.linear(y, dec.proj_out.weight, dec.proj_out.bias)
    px = p.view(T, h, w, 3, 4, 16, 16).permute(0, 4, 3, 1, 5, 2, 6).reshape(T, 4, 3, h * 16, w * 16)
    frames = []
    for t in range(T):
        f = px[t][SV.TRIM:] if t % SV.BLOCK == 0 else px[t]
        frames.append(f)
    out = torch.cat(frames)
    pm = torch.tensor(VV.IMAGENET_MEAN, dtype=torch.float64).view(1, 3, 1, 1)
    ps = torch.tensor(VV.IMAGENET_STD, dtype=torch.float64).view(1, 3, 1, 1)
    return (out * ps + pm).clamp(0, 1).permute(0, 2, 3, 1)


def _psnr(a, b) -> float:
    mse = float(((a.double() - b.double()) ** 2).mean())
    return 200.0 if mse == 0 else 10 * math.log10(1.0 / mse)


# --------------------------------------------------------------------------- #
# the decoder against the all-at-once reference
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("T, h, w", [(14, 23, 37), (12, 20, 32), (7, 17, 33), (5, 16, 16)])
@pytest.mark.parametrize("pieces", [False, True], ids=["default", "small_pieces"])
def test_block_by_block_decoder_matches_the_all_at_once_reference(monkeypatch, T, h, w, pieces):
    """Shifted windows over two blocks, partial windows at the edges, a shorter last block, a
    single block; with ``pieces`` the row pieces are smaller than a window and few windows go
    per attention call."""
    if pieces:
        monkeypatch.setattr(SV, "ROW_CHUNK", 97)
        monkeypatch.setattr(SV, "WIN_BATCH", 3)
    model, mean, std = _random_model()
    z = torch.randn(1, 24, T, h, w, generator=torch.Generator().manual_seed(T * 100 + h), dtype=torch.float64)
    dec = SV.SolDecoder.from_model(model, mean=mean, std=std, device="cpu")
    got = dec.decode(z, output="float")
    want = _reference_decode(model, mean, std, z)
    assert got.dtype == torch.float64 and got.shape == want.shape == (dwd.frames_for_latents(T), h * 16, w * 16, 3)
    assert float((got - want).abs().max()) <= 1e-10
    assert dec.last_stats["max_sections_alive"] <= 2
    assert dec.last_stats["latent_tokens"] == T * h * w and dec.last_stats["nonfinite_pixels"] == 0
    u8 = dec.decode(z)
    assert u8.dtype == torch.uint8 and u8.shape == got.shape and u8.device.type == "cpu"


def test_a_wrong_temporal_shift_is_caught(monkeypatch):
    """The comparison can fail: the decoder with the temporal shift 1 differs from the reference."""
    model, mean, std = _random_model()
    z = torch.randn(1, 24, 14, 23, 37, generator=torch.Generator().manual_seed(9), dtype=torch.float64)
    want = _reference_decode(model, mean, std, z)
    monkeypatch.setattr(SV, "T_SHIFT", 1)
    got = SV.SolDecoder.from_model(model, mean=mean, std=std, device="cpu").decode(z, output="float")
    assert float((got - want).abs().max()) > 1e-3


def test_the_fp16_path_stays_near_the_reference():
    """The production steps (every tensor fp16, the statistics and the rope tables rounded to
    fp16) on the CPU, the torch rope standing in for the CUDA kernel."""
    model, mean, std = _random_model()
    z = torch.randn(1, 24, 12, 20, 32, generator=torch.Generator().manual_seed(4), dtype=torch.float64)
    want = _reference_decode(model, mean, std, z)
    m16 = SV.SolModel(TINY, None, torch.float16)
    m16.load_state_dict({k: v.half() for k, v in model.state_dict().items()})
    dec = SV.SolDecoder.from_model(m16, mean=mean, std=std, device="cpu")
    assert dec.rope_impl == "torch" and dec.dtype == torch.float16
    got = dec.decode(z.float(), output="float")
    assert got.dtype == torch.float32
    assert _psnr(got, want) >= 40.0


def test_the_cuda_path_uses_the_kitchen_rope():
    model, mean, std = _random_model(torch.float16)
    assert SV.SolDecoder.from_model(model, mean=mean, std=std, device="cuda").rope_impl == "kitchen"


def test_decode_refuses_another_shape():
    model, mean, std = _random_model()
    with pytest.raises(ValueError, match="Sol-H3-VAE"):
        SV.SolDecoder.from_model(model, mean=mean, std=std, device="cpu").decode(torch.zeros(1, 16, 7, 4, 4))


# --------------------------------------------------------------------------- #
# the file
# --------------------------------------------------------------------------- #

_DT = {torch.float32: "F32", torch.float16: "F16", torch.int8: "I8", torch.uint8: "U8"}


def _write(path: Path, tensors: dict, metadata: dict | None = None) -> None:
    doc, blobs, off = {}, [], 0
    if metadata is not None:
        doc["__metadata__"] = metadata
    for name, ten in tensors.items():
        data = ten.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        doc[name] = {"dtype": _DT[ten.dtype], "shape": list(ten.shape), "data_offsets": [off, off + len(data)]}
        blobs.append(data)
        off += len(data)
    head = json.dumps(doc).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(head)))
        f.write(head)
        for b in blobs:
            f.write(b)


def _mark():
    return torch.frombuffer(bytearray(json.dumps({"format": "int8_tensorwise", "convrot": False}).encode()),
                            dtype=torch.uint8)


def _int8_file(path: Path, *, mutate=None, metadata=True):
    """A synthetic INT8 Sol file at the tiny widths: the 4 Linears of each layer int8 (a
    marker, an F32 ``weight_scale`` and an F16 bias), everything else F16, an encoder tensor
    the decoder ignores, and the 24 + 24 statistics. Returns ``(path, the float64 model with the
    dequantized weights, mean, std)``."""
    model, mean, std = _random_model()
    sd = model.state_dict()
    tensors: dict = {"encoder.conv_in.weight": torch.zeros(4, dtype=torch.float16)}
    ref = {}
    int8 = {f"decoder.blocks.{i}.{leaf}" for i in range(TINY.num_layers) for leaf in SV._INT8_LEAVES}
    for k, v in sd.items():
        layer = k.rsplit(".", 1)[0]
        if layer in int8 and k.endswith(".weight"):
            scale = (v.abs().amax(dim=1, keepdim=True) / 127).float()
            q = torch.round(v / scale.double()).clamp(-127, 127).to(torch.int8)
            tensors[k] = q
            tensors[f"{layer}.weight_scale"] = scale
            tensors[f"{layer}.comfy_quant"] = _mark()
            ref[k] = q.double() * scale.double()
        else:
            tensors[k] = v.half()
            ref[k] = v.half().double()
    if mutate is not None:
        mutate(tensors)
    meta = {VV.METADATA_KEY: json.dumps({"latents_mean": mean, "latents_std": std})} if metadata else {}
    _write(path, tensors, meta)
    model.load_state_dict(ref)
    return path, model, mean, std


def test_a_synthetic_int8_file_is_read_and_decodes_near_its_float_reference(tmp_path):
    path, ref_model, mean, std = _int8_file(tmp_path / "sol.safetensors")
    assert SV.file_problems(path, dims=TINY) == []
    dec = SV.SolDecoder.from_file(path, device="cpu", dims=TINY)
    from engine_h3.quant_linear import H3Int8Linear

    assert sum(isinstance(m, H3Int8Linear) for m in dec.model.modules()) == 4 * TINY.num_layers
    assert dec.dtype == torch.float16
    assert dec.load_report["int8_layers"] == 4 * TINY.num_layers and dec.load_report["file"] == "sol.safetensors"
    z = torch.randn(1, 24, 7, 17, 20, generator=torch.Generator().manual_seed(2))
    got = dec.decode(z, output="float")
    want = _reference_decode(ref_model, [float(torch.tensor(v).half()) for v in mean],
                             [float(torch.tensor(v).half()) for v in std], z.double())
    assert _psnr(got, want) >= 30.0
    assert dec.decode(z).dtype == torch.uint8


@pytest.mark.parametrize("mutate, fragment", [
    (lambda t: [t.pop(k) for k in list(t) if k.startswith("decoder.blocks.3.")], "復号器の層が 3 層で、層 0〜3 と合いません"),
    (lambda t: t.pop("decoder.blocks.1.ff.w2.comfy_quant"), "int8 の量子化の印が 15 層にあり"),
    (lambda t: t.pop("decoder.blocks.0.attn.to_out.weight_scale"), "decoder.blocks.0.attn.to_out.weight_scale がありません"),
    (lambda t: t.update({"decoder.proj_out.weight": torch.zeros(8, 8, dtype=torch.float16)}),
     "decoder.proj_out.weight が F16 [8, 8] です（F16 [3072, 128] のはず）"),
    (lambda t: t.pop("decoder.norm_out.bias"), "復号器のテンソルが 1 個足りません（decoder.norm_out.bias）"),
])
def test_each_misfit_of_the_file_is_stated(tmp_path, mutate, fragment):
    path, *_ = _int8_file(tmp_path / "bad.safetensors", mutate=mutate)
    problems = SV.file_problems(path, dims=TINY)
    assert any(fragment in p for p in problems), problems
    with pytest.raises(RuntimeError, match="Sol-H3-VAE のファイル（bad.safetensors）"):
        SV.SolDecoder.from_file(path, device="cpu", dims=TINY)


def test_a_file_without_statistics_or_a_header_is_stated(tmp_path):
    path, *_ = _int8_file(tmp_path / "nometa.safetensors", metadata=False)
    assert any("minimax_h3_video_vae" in p for p in SV.file_problems(path, dims=TINY))
    junk = tmp_path / "junk.safetensors"
    junk.write_bytes(b"not a safetensors file")
    (problem,) = SV.file_problems(junk, dims=TINY)
    assert problem.startswith("ヘッダーを読めません")


# --------------------------------------------------------------------------- #
# the decisions
# --------------------------------------------------------------------------- #

def test_the_fallback_reasons():
    assert SV.file_fallback("") == SV.file_fallback("   ") == "file_missing"
    assert SV.file_fallback("Z:/no/such/file.safetensors") == "file_missing"
    assert SV.file_fallback(__file__) is None
    limit = SV.MAX_LATENT_TOKENS
    assert limit == 1_650_000
    # The three chains the experiment measured are inside the limit.
    for frames, w, h in ((940, 1280, 768), (685, 1920, 1088), (362, 2560, 1472)):
        from engine_h3.constants import latent_frames

        shape = (1, 24, latent_frames(frames), h // 16, w // 16)
        assert SV.size_fallback(shape) is None, (frames, w, h)
    assert SV.latent_tokens((1, 24, 10, 20, 30)) == 6000


@pytest.mark.parametrize("tokens, reason", [(5999, "too_large"), (6000, None), (6001, None)],
                         ids=["above", "exactly", "inside"])
def test_the_size_limit_is_above_not_at(monkeypatch, tokens, reason):
    monkeypatch.setattr(SV, "MAX_LATENT_TOKENS", tokens)
    assert SV.size_fallback((1, 24, 10, 20, 30)) == reason


def test_kitchen_problem_reads_the_cuda_capabilities(monkeypatch):
    import comfy_kitchen

    caps = {"int8_linear", "rms_rope_split_half_"}
    monkeypatch.setattr(comfy_kitchen, "list_backends", lambda: {
        "cuda": {"available": True, "disabled": False, "capabilities": sorted(caps)}})
    assert SV.kitchen_problem() is None
    caps.discard("rms_rope_split_half_")
    assert SV.kitchen_problem() == "comfy-kitchen の CUDA バックエンドに rms_rope_split_half_ がありません"
    monkeypatch.setattr(comfy_kitchen, "list_backends", lambda: {"eager": {"capabilities": ["rms_rope_split_half_"]}})
    assert SV.kitchen_problem() == "comfy-kitchen の CUDA バックエンドに rms_rope_split_half_ がありません"


def _req(**kw):
    from engine_h3 import pipeline_h3 as P

    base = dict(prompt="p", seed=1, width=640, height=384, num_frames=124, output_path="o.mp4")
    return P.H3GenerateRequest(**{**base, **kw})


def test_sol_start(monkeypatch, tmp_path):
    from engine_h3 import pipeline_h3 as P

    assert P.sol_start(_req(), "cpu") is None
    for path in ("", str(tmp_path / "missing.safetensors")):
        assert P.sol_start(_req(sol_vae=P.H3SolVaeInput(path)), "cpu") == {"path": path, "fallback": "file_missing"}
    bad = tmp_path / "bad.safetensors"
    _int8_file(bad)            # the tiny widths: the released layout finds fault with it
    with pytest.raises(ValueError, match=r"^Sol-H3-VAE のファイル（bad.safetensors）の照合で、合わない所が \d+ か所ありました: "):
        P.sol_start(_req(sol_vae=P.H3SolVaeInput(str(bad))), "cpu")
    monkeypatch.setattr(SV, "file_problems", lambda path: [])
    assert P.sol_start(_req(sol_vae=P.H3SolVaeInput(str(bad))), "cpu") == {"path": str(bad), "fallback": None}
    monkeypatch.setattr(SV, "kitchen_problem", lambda: "comfy-kitchen の CUDA バックエンドに rms_rope_split_half_ がありません")
    with pytest.raises(ValueError, match="rms_rope_split_half_ がありません"):
        P.sol_start(_req(sol_vae=P.H3SolVaeInput(str(bad))), "cuda")
    assert P.sol_start(_req(sol_vae=P.H3SolVaeInput(str(bad))), "cpu")["fallback"] is None  # CPU: not asked


def test_validate_request_refuses_pi_h3_with_sol():
    from engine_h3 import pipeline_h3 as P

    with pytest.raises(ValueError, match="pi-H3 と Sol-H3-VAE は同時に指定できません"):
        P.validate_request(_req(num_inference_steps=4, pi_h3=P.H3PiH3Input(4, "pi.safetensors"),
                                sol_vae=P.H3SolVaeInput("sol.safetensors")))
    P.validate_request(_req(sol_vae=P.H3SolVaeInput("")))  # no raise: the file is sol_start's


# --------------------------------------------------------------------------- #
# _decode
# --------------------------------------------------------------------------- #

class _FakeSol:
    """``SolDecoder``'s shape: CPU uint8 frames for the latents it is given."""

    built: list = []

    def __init__(self, path):
        self.load_report = {"file": Path(path).name}
        self.last_stats = {}
        self.seen = []
        _FakeSol.built.append(self)

    def decode(self, latents):
        self.seen.append(tuple(latents.shape))
        t, h, w = (int(v) for v in latents.shape[2:])
        self.last_stats = {"latent_tokens": t * h * w, "nonfinite_pixels": 0}
        return torch.full((dwd.frames_for_latents(t), 16 * h, 16 * w, 3), 7, dtype=torch.uint8)


def _decode_pipe(monkeypatch):
    from engine_h3 import audio_vae as AV
    from engine_h3 import media_out as MO
    from engine_h3 import pipeline_h3 as P

    rec = {"stock": 0, "frames": []}

    class Stock(torch.nn.Module):
        def decode(self, z, tiled=True):
            rec["stock"] += 1
            t, h, w = (int(v) for v in z.shape[2:])
            return torch.zeros((dwd.frames_for_latents(t), 16 * h, 16 * w, 3), dtype=torch.uint8)

    class Audio(torch.nn.Module):
        def decode(self, a):
            return torch.zeros((2, 800 * a.shape[-1]))

    monkeypatch.setattr(VV.H3VideoDecoder, "from_file", classmethod(lambda cls, *a, **k: Stock()))
    monkeypatch.setattr(AV.H3AudioVAE, "from_file", classmethod(lambda cls, *a, **k: Audio()))
    monkeypatch.setattr(MO, "write_mp4", lambda path, frames, **k: rec["frames"].append(frames.clone()) or 1)
    _FakeSol.built = []
    monkeypatch.setattr(SV.SolDecoder, "from_file", classmethod(lambda cls, path, *, device: _FakeSol(path)))
    pipe = P.H3Pipeline(P.H3Paths("t", "te", "vv", "av", "tok"), P.H3LoadOptions(), device="cpu",
                        progress=lambda *a, **k: None, log=lambda m: None)
    return pipe, rec


def _lat(t=7, h=4, w=4):
    return torch.zeros(1, 24, t, h, w), torch.zeros(1, 8, 9, 16)


@pytest.mark.parametrize("limit, used", [(112 + 1, True), (112, True), (111, False)],
                         ids=["inside", "exactly", "above"])
def test_decode_judges_the_size_on_the_latents(monkeypatch, limit, used):
    from engine_h3 import pipeline_h3 as P

    monkeypatch.setattr(SV, "MAX_LATENT_TOKENS", limit)
    pipe, rec = _decode_pipe(monkeypatch)
    video, audio = _lat()             # 7 x 4 x 4 = 112 latent tokens
    entry: dict = {}
    pipe._decode(_req(num_frames=22), video, audio, entry, sol={"path": "m/sol.safetensors", "fallback": None})
    if used:
        assert entry["vae_source"] == "sol" and rec["stock"] == 0 and len(_FakeSol.built) == 1
        assert _FakeSol.built[0].seen == [(1, 24, 7, 4, 4)]
        assert entry["sol"]["decoder"] == {"file": "sol.safetensors"} and entry["sol"]["latent_tokens"] == 112
        assert entry["sol"]["nonfinite_pixels"] == 0 and "sol_fallback" not in entry
        assert int(rec["frames"][-1][0, 0, 0, 0]) == 7
        assert P.sol_record(entry)[0] == "on"
    else:
        assert entry["vae_source"] == "file" and rec["stock"] == 1 and _FakeSol.built == []
        assert entry["sol_fallback"] == {"reason": "too_large", "latent_tokens": 112, "max_latent_tokens": 111}
        assert "sol" not in entry
        assert P.sol_record(entry) == ("on->off", {"used": "on->off", "reason": "too_large",
                                                   "latent_tokens": 112, "max_latent_tokens": 111})
    assert entry["frames"] == 22


def test_decode_with_the_start_fallback_never_builds_sol(monkeypatch):
    from engine_h3 import pipeline_h3 as P

    pipe, rec = _decode_pipe(monkeypatch)
    entry: dict = {}
    pipe._decode(_req(num_frames=22), *_lat(), entry, sol={"path": "", "fallback": "file_missing"})
    assert entry["vae_source"] == "file" and rec["stock"] == 1 and _FakeSol.built == []
    assert entry["sol_fallback"]["reason"] == "file_missing"
    assert P.sol_record(entry)[0] == "on->off"


def test_decode_without_sol_is_as_before(monkeypatch):
    pipe, rec = _decode_pipe(monkeypatch)
    monkeypatch.setattr(SV, "size_fallback", lambda shape: pytest.fail("a job without sol_vae judged the size"))
    entry: dict = {}
    pipe._decode(_req(num_frames=22), *_lat(), entry)
    assert entry["vae_source"] == "file" and rec["stock"] == 1 and _FakeSol.built == []
    assert not {k for k in entry if k.startswith("sol")}


def test_sol_leaves_a_kept_video_decoder_where_it_is(monkeypatch):
    pipe, rec = _decode_pipe(monkeypatch)
    kept = object()
    pipe._resident = {"decoder": kept}
    entry: dict = {}
    pipe._decode(_req(num_frames=22), *_lat(), entry, keep=True, sol={"path": "s", "fallback": None})
    assert entry["vae_source"] == "sol" and pipe._resident["decoder"] is kept and rec["stock"] == 0


# --------------------------------------------------------------------------- #
# generate and the chain on the tiny DiT
# --------------------------------------------------------------------------- #

def _sol_job_pipe(monkeypatch, tmp_path):
    pipe, rec, _job = _pi_job_pipe(monkeypatch, tmp_path)
    _FakeSol.built = []
    monkeypatch.setattr(SV.SolDecoder, "from_file", classmethod(lambda cls, path, *, device: _FakeSol(path)))

    def job(sol_path=None):
        from engine_h3 import pipeline_h3 as P

        req = P.H3GenerateRequest(prompt="p", seed=7, width=64, height=64, num_frames=22, output_path="o.mp4",
                                  num_inference_steps=3, shift_video=12.0, shift_audio=3.0,
                                  sol_vae=None if sol_path is None else P.H3SolVaeInput(sol_path))
        return pipe.generate(req)

    return pipe, rec, job


def test_generate_records_on_and_on_to_off(monkeypatch, tmp_path):
    pipe, rec, job = _sol_job_pipe(monkeypatch, tmp_path)
    monkeypatch.setattr(SV, "file_problems", lambda path: [])
    sol = tmp_path / "sol.safetensors"
    sol.write_bytes(b"x")
    done = job(str(sol))
    assert done["sol_vae_used"] == "on" and list(done)[-2:] == ["sol_vae_used", "h3"]
    assert done["h3"]["sol_vae"]["used"] == "on" and done["h3"]["sol_vae"]["latent_tokens"] == 7 * 4 * 4
    assert done["h3"]["phases"]["decode"]["vae_source"] == "sol"
    assert rec["decoders"] == 0 and _FakeSol.built[0].seen == [(1, 24, 7, 4, 4)]
    off = job(str(tmp_path / "missing.safetensors"))
    assert off["sol_vae_used"] == "on->off"
    assert off["h3"]["sol_vae"] == {"used": "on->off", "reason": "file_missing", "latent_tokens": 112,
                                    "max_latent_tokens": SV.MAX_LATENT_TOKENS}
    assert off["h3"]["phases"]["decode"]["vae_source"] == "file" and rec["decoders"] == 1


def test_a_broken_file_fails_the_job_before_the_text_encoder(monkeypatch, tmp_path):
    pipe, rec, job = _sol_job_pipe(monkeypatch, tmp_path)
    encodes = []
    monkeypatch.setattr(pipe, "_encode", lambda *a, **k: encodes.append(1))
    bad = tmp_path / "bad.safetensors"
    bad.write_bytes(b"not a safetensors file")
    with pytest.raises(ValueError, match="Sol-H3-VAE のファイル（bad.safetensors）の照合で、合わない所が 1 か所"):
        job(str(bad))
    assert encodes == [] and _FakeSol.built == []


def test_a_job_without_sol_reaches_none_of_the_sol_code(monkeypatch, tmp_path):
    pipe, rec, job = _sol_job_pipe(monkeypatch, tmp_path)

    def boom(*a, **k):
        raise AssertionError("a job without sol_vae reached the Sol code")

    for name in ("file_fallback", "file_problems", "size_fallback", "latent_tokens", "kitchen_problem"):
        monkeypatch.setattr(SV, name, boom)
    monkeypatch.setattr(SV.SolDecoder, "from_file", classmethod(boom))
    done = job(None)
    assert "sol_vae_used" not in done and "sol_vae" not in done["h3"]
    decode = done["h3"]["phases"]["decode"]
    assert decode["vae_source"] == "file" and not {k for k in decode if k.startswith("sol")}


def test_the_chain_decodes_the_joined_latents_with_sol(monkeypatch, tmp_path):
    """22 + 39 frames: the Sol decoder gets the joined 12 latents (7 + 12 - 7), judged there;
    an off chain afterwards passes no ``sol`` key."""
    from engine_h3 import chain_h3 as CH
    from engine_h3 import pipeline_h3 as P

    pipe, rec, _job = _sol_job_pipe(monkeypatch, tmp_path)
    monkeypatch.setattr(SV, "file_problems", lambda path: [])
    sol = tmp_path / "sol.safetensors"
    sol.write_bytes(b"x")
    decodes = []
    real = pipe._decode

    def spy(*a, **k):
        decodes.append(k)
        return real(*a, **k)

    monkeypatch.setattr(pipe, "_decode", spy)

    class TimelineDecoder(torch.nn.Module):     # the stock decoder, for the timeline's frames
        def decode(self, z, tiled=True):
            return torch.zeros((dwd.frames_for_latents(int(z.shape[2])), 64, 64, 3), dtype=torch.uint8)

    monkeypatch.setattr(VV.H3VideoDecoder, "from_file", classmethod(lambda cls, *a, **k: TimelineDecoder()))

    def spec(path):
        base = P.H3GenerateRequest(prompt="p", seed=0, width=64, height=64, num_frames=22, output_path="o.mp4",
                                   num_inference_steps=3, shift_video=12.0, shift_audio=3.0,
                                   sol_vae=None if path is None else P.H3SolVaeInput(path))
        return CH.H3ChainSpec(base=base, clips=(CH.H3ChainClipSpec("a", 22), CH.H3ChainClipSpec("b", 39)), seed=11)

    done = CH.run_chain_h3(pipe, spec(str(sol)), lambda *a, **k: None, lambda m: None)
    assert decodes[-1]["sol"] == {"path": str(sol), "fallback": None}
    assert _FakeSol.built[-1].seen == [(1, 24, 12, 4, 4)]
    assert done["sol_vae_used"] == "on" and done["h3"]["sol_vae"]["latent_tokens"] == 12 * 16
    assert tuple(rec["frames"][-1].shape) == (39, 64, 64, 3)
    monkeypatch.setattr(SV, "MAX_LATENT_TOKENS", 12 * 16 - 1)
    done = CH.run_chain_h3(pipe, spec(str(sol)), lambda *a, **k: None, lambda m: None)
    assert done["sol_vae_used"] == "on->off" and done["h3"]["sol_vae"]["reason"] == "too_large"
    assert len(_FakeSol.built) == 1
    off = CH.run_chain_h3(pipe, spec(None), lambda *a, **k: None, lambda m: None)
    assert "sol" not in decodes[-1] and "sol_vae_used" not in off and "sol_vae" not in off["h3"]


# --------------------------------------------------------------------------- #
# the worker
# --------------------------------------------------------------------------- #

def test_worker_sol_vae_key(monkeypatch):
    from engine_h3 import worker
    from engine_h3.pipeline_h3 import H3SolVaeInput

    msg = {"op": "generate", "prompt": "p", "seed": 1, "width": 640, "height": 384, "num_frames": 124,
           "output_path": "o.mp4"}
    assert worker._generate_request(msg).sol_vae is None
    assert worker._generate_request({**msg, "sol_vae": None}).sol_vae is None
    assert worker._generate_request({**msg, "sol_vae": {"path": "C:/m/sol.safetensors"}}).sol_vae == \
        H3SolVaeInput("C:/m/sol.safetensors")
    assert worker._generate_request({**msg, "sol_vae": {"path": ""}}).sol_vae == H3SolVaeInput("")  # allowed
    assert worker._sol_vae({"sol_vae": {"path": "p", "extra": 1}})[1] == ["extra"]
    for bad in ({}, {"path": None}, {"path": 3}, ["p"], "p"):
        with pytest.raises(ValueError, match="sol_vae"):
            worker._sol_vae({"sol_vae": bad})
    assert "sol_vae" in worker._GENERATE_FIELDS and "sol_vae" in worker._CHAIN_FIELDS

    seen, events, logs = [], [], []

    class FakePipe:
        def generate(self, req):
            seen.append(req)
            return {"seed_used": 1, **({"sol_vae_used": "on"} if req.sol_vae is not None else {})}

    monkeypatch.setattr(worker, "_PIPE", FakePipe())
    monkeypatch.setattr(worker, "_emit", lambda event, **f: events.append({"event": event, **f}))
    monkeypatch.setattr(worker, "_log", logs.append)
    worker._do_generate({**msg, "sol_vae": {"path": "C:/m/sol.safetensors", "x": 1}})
    worker._do_generate({**msg, "sol_vae": {"path": ""}})
    worker._do_generate(dict(msg))
    assert seen[0].sol_vae == H3SolVaeInput("C:/m/sol.safetensors") and seen[2].sol_vae is None
    assert events[0]["sol_vae_used"] == "on" and "sol_vae_used" not in events[2]
    assert "sol_vae=sol.safetensors" in logs[0] and "ignored keys: sol_vae.x" in logs[0]
    assert "sol_vae=(no path)" in logs[1]
    assert "sol_vae" not in logs[2] and "ignored keys" not in logs[2]


def test_worker_chain_reads_sol_vae(monkeypatch):
    from engine_h3 import chain_h3 as CH
    from engine_h3 import worker
    from engine_h3.pipeline_h3 import H3SolVaeInput

    seen, events, logs = [], [], []

    def run(pipe, spec, progress, log):
        seen.append(spec)
        return {"seed_used": 1, **({"sol_vae_used": "on->off"} if spec.base.sol_vae is not None else {})}

    monkeypatch.setattr(CH, "run_chain_h3", run)
    monkeypatch.setattr(worker, "_PIPE", object())
    monkeypatch.setattr(worker, "_emit", lambda event, **f: events.append({"event": event, **f}))
    monkeypatch.setattr(worker, "_log", logs.append)
    msg = {"op": "generate_chain", "seed": 1, "width": 640, "height": 384, "output_path": "o.mp4",
           "clips": [{"prompt": "a", "num_frames": 124}, {"prompt": "b", "num_frames": 124}]}
    worker._do_generate_chain({**msg, "sol_vae": {"path": "C:/m/sol.safetensors", "x": 1}})
    worker._do_generate_chain(dict(msg))
    assert seen[0].base.sol_vae == H3SolVaeInput("C:/m/sol.safetensors") and seen[1].base.sol_vae is None
    assert events[0]["sol_vae_used"] == "on->off" and "sol_vae_used" not in events[1]
    assert "sol_vae=sol.safetensors" in logs[0] and "ignored keys: sol_vae.x" in logs[0]
    assert "sol_vae" not in logs[1]
