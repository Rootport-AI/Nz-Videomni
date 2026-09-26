"""§3-168: ``engine.sft_quant.dequant`` — dequantize / normalize_aux / hadamard (CPU only).

  * ``hadamard``: symmetric, orthogonal, involutory (H·H = I), every element
    +/- 1/16, the ComfyUI H4 seed, cached per device and not an inference tensor;
  * ``dequantize``: fp8 / fp8_scaled bit-identical to the §3-167 expression,
    int8 and int8_convrot round-trip a quantized weight, the rotation matters;
  * ``normalize_aux``: the dtype and shape of ``sft_quant_format.aux_specs``.

Run with ``.venv-engine`` and ``--noconftest``; the app venv skips the module.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from engine.sft_quant import dequant  # noqa: E402
from engine.sft_quant.dequant import dequantize, hadamard, normalize_aux  # noqa: E402

BF16 = torch.bfloat16
F32 = torch.float32


def _u8(t: torch.Tensor) -> torch.Tensor:
    return t.detach().reshape(-1).view(torch.uint8).clone()


def _quantize_int8(w: torch.Tensor, per_row: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric int8 quantization, as ComfyUI's int8_tensorwise stores it."""
    amax = w.abs().amax(dim=1, keepdim=True) if per_row else w.abs().amax().reshape(1, 1)
    scale = (amax / 127.0).to(F32)
    q = torch.clamp(torch.round(w / scale), -127, 127).to(torch.int8)
    return q, scale


# ── hadamard ────────────────────────────────────────────────────────────────


def test_hadamard_properties():
    h = hadamard("cpu")
    assert h.shape == (256, 256) and h.dtype == F32
    assert torch.equal(h, h.T)
    assert torch.equal(h.abs(), torch.full((256, 256), 1.0 / 16))
    eye = torch.eye(256)
    assert torch.allclose(h @ h, eye, atol=1e-6)
    assert torch.allclose(h @ h.T, eye, atol=1e-6)


def test_hadamard_is_the_comfyui_h4_kronecker_power():
    h4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=F32
    )
    ref = h4
    for _ in range(3):
        ref = torch.kron(ref, h4)
    assert torch.equal(hadamard("cpu"), ref / 16)


def test_hadamard_cached_per_device_and_not_an_inference_tensor(monkeypatch):
    monkeypatch.setattr(dequant, "_HADAMARD", {})
    with torch.inference_mode():
        h = hadamard("cpu")
    assert not h.is_inference()
    assert hadamard(torch.device("cpu")) is h
    assert list(dequant._HADAMARD) == [torch.device("cpu")]


# ── dequantize ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("fp8", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_fp8_schemes_bit_identical_to_the_3_167_expression(fp8):
    torch.manual_seed(1)
    w = (torch.randn(32, 64) * 4).to(fp8)
    s = torch.tensor(0.0625, dtype=F32)
    assert torch.equal(_u8(dequantize("fp8", w, {}, BF16)), _u8(w.to(BF16)))
    assert torch.equal(_u8(dequantize(None, w, {}, BF16)), _u8(w.to(BF16)))
    got = dequantize("fp8_scaled", w, {"weight_scale": s}, BF16)
    assert torch.equal(_u8(got), _u8((w.to(F32) * s).to(BF16)))


@pytest.mark.parametrize("per_row", [False, True], ids=["scalar", "per_row"])
def test_int8_round_trip(per_row):
    torch.manual_seed(2)
    w = torch.randn(48, 64)
    q, scale = _quantize_int8(w, per_row)
    aux = {"weight_scale": scale.expand(48, 1).contiguous()}
    got = dequantize("int8", q, aux, F32)
    assert got.dtype == F32
    assert torch.equal(got, q.to(F32) * aux["weight_scale"])
    # within half a quantization step of the original
    assert (got - w).abs().max() <= aux["weight_scale"].max() / 2 + 1e-6
    assert torch.equal(dequantize("int8", q, aux, BF16), got.to(BF16))


@pytest.mark.parametrize("per_row", [False, True], ids=["scalar", "per_row"])
def test_int8_convrot_round_trip(per_row):
    """A weight rotated by W @ H before quantization comes back to W."""
    torch.manual_seed(3)
    o, i = 24, 512
    w = torch.randn(o, i)
    h = hadamard("cpu")
    rotated = (w.view(o, i // 256, 256) @ h).view(o, i)
    q, scale = _quantize_int8(rotated, per_row)
    aux = {"weight_scale": scale.expand(o, 1).contiguous()}
    got = dequantize("int8_convrot", q, aux, F32)
    ref = ((q.to(F32) * aux["weight_scale"]).view(o, i // 256, 256) @ h).view(o, i)
    assert torch.equal(got, ref)
    # the rotation is orthogonal, so the error per row stays that of int8
    step = aux["weight_scale"].max()
    assert (got - w).abs().max() < step * 16
    assert torch.allclose(got, w, atol=float(step) * 4, rtol=0)
    # without the rotation the weight is NOT recovered
    plain = dequantize("int8", q, aux, F32)
    assert not torch.allclose(plain, w, atol=float(step) * 4, rtol=0)
    assert torch.equal(dequantize("int8_convrot", q, aux, BF16), ref.to(BF16))


def test_dequantize_does_not_write_its_inputs():
    torch.manual_seed(4)
    q = torch.randint(-127, 128, (8, 256), dtype=torch.int8)
    s = torch.rand(8, 1)
    q0, s0 = q.clone(), s.clone()
    for scheme in ("int8", "int8_convrot"):
        dequantize(scheme, q, {"weight_scale": s}, BF16)
    assert torch.equal(q, q0) and torch.equal(s, s0)


def test_unknown_scheme_raises():
    with pytest.raises(RuntimeError, match="unknown scheme"):
        dequantize("w4a8", torch.zeros(2, 2, dtype=torch.int8), {}, BF16)


# ── normalize_aux ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("raw", [torch.tensor(0.5), torch.tensor([0.5])], ids=["0dim", "(1,)"])
def test_normalize_fp8_scale_is_0_dim_f32(raw):
    got = normalize_aux("fp8_scaled", "weight_scale", raw, 4, 8)
    assert got.shape == () and got.dtype == F32 and float(got) == 0.5


@pytest.mark.parametrize(
    "raw",
    [torch.tensor(0.25), torch.tensor([0.25]), torch.full((6, 1), 0.25)],
    ids=["0dim", "(1,)", "(o,1)"],
)
@pytest.mark.parametrize("scheme", ["int8", "int8_convrot"])
def test_normalize_int8_scale_is_o_by_1_f32(scheme, raw):
    got = normalize_aux(scheme, "weight_scale", raw, 6, 256)
    assert got.shape == (6, 1) and got.dtype == F32 and got.is_contiguous()
    assert torch.equal(got, torch.full((6, 1), 0.25))


def test_normalize_int8_per_row_keeps_values():
    raw = torch.arange(6, dtype=F32).reshape(6, 1)
    assert torch.equal(normalize_aux("int8", "weight_scale", raw, 6, 16), raw)


def test_normalize_rejects_unknown_leaf_and_wrong_shape():
    with pytest.raises(RuntimeError, match="not an auxiliary tensor"):
        normalize_aux("fp8", "weight_scale", torch.tensor(1.0), 4, 4)
    with pytest.raises(RuntimeError):
        normalize_aux("int8", "weight_scale", torch.ones(3, 1), 6, 16)
    with pytest.raises(RuntimeError, match="expected"):
        normalize_aux("int8", "weight_scale", torch.ones(6, 1, dtype=torch.int8), 6, 16)
