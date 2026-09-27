"""§3-168: ``engine.sft_quant.dequant`` — dequantize / normalize_aux / hadamard (CPU only).

  * ``hadamard``: symmetric, orthogonal, involutory (H·H = I), every element
    +/- 1/16, the ComfyUI H4 seed, cached per device and not an inference tensor;
  * ``dequantize``: fp8 / fp8_scaled bit-identical to the §3-167 expression,
    int8 and int8_convrot round-trip a quantized weight, the rotation matters;
  * ``dequantize`` w4a8 (C-3): against an independent element-by-element
    reference; the low nibble is the even column; the int8-grid rounding and
    the codebook values matter; U8 and fp8 s_rel give the same bytes;
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
        dequantize("nvfp4", torch.zeros(2, 2, dtype=torch.int8), {}, BF16)


# ── w4a8 (§3-168 C-3) ───────────────────────────────────────────────────────
#
# The reference below is written independently of dequant.py: a plain Python
# loop over every element (nibble, codebook, s_rel decoded by hand from its
# E4M3 bits, float32 products, round half to even, clamp, s_channel), then the
# ConvRot rotation as a float64 sum with a Hadamard matrix built here.

_H4_ROWS = ((1, 1, 1, -1), (1, 1, -1, 1), (1, -1, 1, 1), (-1, 1, 1, 1))


def _e4m3_value(byte: int) -> float:
    """float8_e4m3fn decoded from its bits (1 sign, 4 exponent bias 7, 3 mantissa)."""
    sign = -1.0 if byte & 0x80 else 1.0
    exp = (byte >> 3) & 0x0F
    man = byte & 0x07
    if exp == 0x0F and man == 0x07:
        return float("nan")
    if exp == 0:
        return sign * (man / 8.0) * 2.0 ** -6
    return sign * (1.0 + man / 8.0) * 2.0 ** (exp - 7)


def _hadamard_f64(n: int = 256) -> list[list[float]]:
    m = [[float(v) for v in row] for row in _H4_ROWS]
    while len(m) < n:
        m = [[a * b for a in ra for b in rb] for ra in m for rb in _H4_ROWS]
    return [[v / 16.0 for v in row] for row in m]


def _naive_w4a8(packed_i8, s_rel_u8, s_channel, codebook, *, low_first=True, rounding=True):
    import numpy as np

    f32 = np.float32
    o, half = packed_i8.shape
    i = half * 2
    packed = packed_i8.view(torch.uint8).tolist()
    s_rel_bytes = s_rel_u8.tolist()
    cb = [f32(v) for v in codebook.tolist()]
    sc = [f32(v) for v in s_channel.tolist()]
    grid = [[0.0] * i for _ in range(o)]
    for r in range(o):
        for c in range(i):
            byte = packed[r][c // 2]
            lo, hi = byte & 0x0F, byte >> 4
            code = (lo if c % 2 == 0 else hi) if low_first else (hi if c % 2 == 0 else lo)
            v = cb[code] * f32(_e4m3_value(s_rel_bytes[r][c // 16]))
            if rounding:
                v = f32(np.rint(v))
                v = f32(min(max(v, -127.0), 127.0))
            grid[r][c] = float(v * sc[r])
    h = _hadamard_f64()
    out = [[0.0] * i for _ in range(o)]
    for r in range(o):
        for g0 in range(0, i, 256):
            row = grid[r][g0:g0 + 256]
            for c in range(256):
                out[r][g0 + c] = sum(row[k] * h[k][c] for k in range(256))
    return torch.tensor(out, dtype=torch.float64)


def _w4a8_case(o=4, i=512, seed=11):
    """Random packed codes, a NON-uniform sorted codebook, fp8 s_rel wide
    enough that some products leave the int8 grid (the clamp) and most are
    off-integer (the round)."""
    g = torch.Generator().manual_seed(seed)
    packed = torch.randint(-128, 128, (o, i // 2), dtype=torch.int8, generator=g)
    codebook = torch.sort(torch.randn(16, generator=g) * 0.6).values.to(F32)
    s_rel = (torch.rand(o, i // 16, generator=g) * 180 + 20).to(torch.float8_e4m3fn)
    s_channel = (torch.rand(o, generator=g) * 0.02 + 0.001).to(F32)
    aux = {"weight_s_rel": s_rel, "weight_s_channel": s_channel, "weight_codebook": codebook}
    return packed, aux


def _ref_args(packed, aux):
    return (packed, aux["weight_s_rel"].view(torch.uint8), aux["weight_s_channel"],
            aux["weight_codebook"])


def test_w4a8_matches_the_naive_loop_reference():
    packed, aux = _w4a8_case()
    got = dequantize("w4a8", packed, aux, F32)
    assert got.shape == (4, 512) and got.dtype == F32
    ref = _naive_w4a8(*_ref_args(packed, aux))
    tol = 1e-5 * float(ref.abs().max())
    assert float((got.double() - ref).abs().max()) <= tol
    # the codebook values actually reach the clamp
    assert float(aux["weight_codebook"].abs().max()) * 200 > 127
    assert torch.equal(_u8(dequantize("w4a8", packed, aux, BF16)), _u8(got.to(BF16)))


def test_w4a8_even_column_is_the_low_nibble():
    packed, aux = _w4a8_case(o=2, i=256, seed=12)
    got = dequantize("w4a8", packed, aux, F32).double()
    tol = 1e-5 * float(got.abs().max())
    assert float((got - _naive_w4a8(*_ref_args(packed, aux))).abs().max()) <= tol
    swapped = _naive_w4a8(*_ref_args(packed, aux), low_first=False)
    assert float((got - swapped).abs().max()) > 100 * tol


def test_w4a8_rounds_onto_the_int8_grid():
    packed, aux = _w4a8_case(o=2, i=256, seed=13)
    got = dequantize("w4a8", packed, aux, F32)
    # undo the rotation (H is involutory) and s_channel: integers in [-127, 127]
    grid = (got.view(2, 1, 256) @ hadamard("cpu")).view(2, 256) / aux["weight_s_channel"][:, None]
    assert float((grid - grid.round()).abs().max()) < 1e-3
    assert 126.5 <= float(grid.abs().max()) <= 127 + 1e-3
    # without the rounding the weight is measurably different
    unrounded = _naive_w4a8(*_ref_args(packed, aux), rounding=False)
    tol = 1e-5 * float(got.abs().max())
    assert float((got.double() - unrounded).abs().max()) > 100 * tol


def test_w4a8_codebook_values_matter():
    packed, aux = _w4a8_case(o=2, i=256, seed=14)
    got = dequantize("w4a8", packed, aux, F32)
    uniform = dict(aux, weight_codebook=torch.linspace(-1, 1, 16, dtype=F32))
    assert not torch.allclose(dequantize("w4a8", packed, uniform, F32), got)


def test_w4a8_s_rel_as_u8_or_fp8_is_the_same():
    o, i = 4, 256
    packed, aux = _w4a8_case(o=o, i=i, seed=15)
    as_fp8 = normalize_aux("w4a8", "weight_s_rel", aux["weight_s_rel"], o, i)
    raw_u8 = aux["weight_s_rel"].view(torch.uint8).clone()
    as_u8 = normalize_aux("w4a8", "weight_s_rel", raw_u8, o, i)
    assert as_fp8.dtype == as_u8.dtype == torch.float8_e4m3fn
    assert torch.equal(as_fp8.view(torch.uint8), as_u8.view(torch.uint8))
    a = dequantize("w4a8", packed, dict(aux, weight_s_rel=as_fp8), BF16)
    b = dequantize("w4a8", packed, dict(aux, weight_s_rel=as_u8), BF16)
    assert torch.equal(_u8(a), _u8(b))


def test_w4a8_does_not_write_its_inputs():
    packed, aux = _w4a8_case(o=2, i=256, seed=16)
    before = {k: _u8(v) for k, v in aux.items()}
    p0 = packed.clone()
    dequantize("w4a8", packed, aux, BF16)
    assert torch.equal(packed, p0)
    for k, v in aux.items():
        assert torch.equal(_u8(v), before[k])


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


def test_normalize_w4a8_aux_dtypes_and_shapes():
    o, i = 6, 512
    s_rel = normalize_aux("w4a8", "weight_s_rel", torch.zeros(o, i // 16, dtype=torch.uint8), o, i)
    assert s_rel.shape == (o, i // 16) and s_rel.dtype == torch.float8_e4m3fn
    s_ch = normalize_aux("w4a8", "weight_s_channel", torch.ones(o), o, i)
    assert s_ch.shape == (o,) and s_ch.dtype == F32
    cb = normalize_aux("w4a8", "weight_codebook", torch.arange(16, dtype=F32), o, i)
    assert cb.shape == (16,) and cb.dtype == F32
    assert torch.equal(cb, torch.arange(16, dtype=F32))


def test_normalize_w4a8_rejects_other_dtypes():
    o, i = 6, 512
    # only weight_s_rel reads U8 as fp8
    with pytest.raises(RuntimeError, match="expected"):
        normalize_aux("w4a8", "weight_s_channel", torch.ones(o, dtype=torch.uint8), o, i)
    with pytest.raises(RuntimeError, match="expected"):
        normalize_aux("w4a8", "weight_codebook", torch.ones(16, dtype=torch.float16), o, i)
    with pytest.raises(RuntimeError, match="expected"):
        normalize_aux("w4a8", "weight_s_rel", torch.ones(o, i // 16, dtype=torch.int8), o, i)
    with pytest.raises(RuntimeError, match="expected"):
        normalize_aux("w4a8", "weight_s_rel", torch.ones(o, i // 16, dtype=F32), o, i)


def test_normalize_rejects_unknown_leaf_and_wrong_shape():
    with pytest.raises(RuntimeError, match="not an auxiliary tensor"):
        normalize_aux("fp8", "weight_scale", torch.tensor(1.0), 4, 4)
    with pytest.raises(RuntimeError):
        normalize_aux("int8", "weight_scale", torch.ones(3, 1), 6, 16)
    with pytest.raises(RuntimeError, match="expected"):
        normalize_aux("int8", "weight_scale", torch.ones(6, 1, dtype=torch.int8), 6, 16)
    # the reshape to aux_shape refuses a wrong element count (w4a8 s_channel / s_rel)
    with pytest.raises(RuntimeError):
        normalize_aux("w4a8", "weight_s_channel", torch.ones(5), 6, 512)
    with pytest.raises(RuntimeError):
        normalize_aux("w4a8", "weight_s_rel", torch.zeros(6, 16, dtype=torch.uint8), 6, 512)
