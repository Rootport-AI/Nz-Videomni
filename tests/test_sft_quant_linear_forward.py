"""§3-167 / §3-168: the ``sft_quant_linear`` forward — test_ic_lora_forward.py's twin (CPU only).

Pins the forward that ``engine.sft_quant.quant_service`` installs on every Linear of
a quantized (fp8 / int8) transformer:

  * a bf16 layer without LoRA is byte-identical to a plain ``nn.Linear``,
  * fp8 layers (scaled and plain cast), with and without (IC-)LoRA, match the
    reference ``(w.float() * s).to(bf16) + sum(delta)`` bit for bit, where each
    delta is ``matmul(B.float() * strength, A.float()).to(x.dtype)`` — cast to
    the ACTIVATION dtype, never to the fp8 weight dtype,
  * the forward never writes the stored weight or scale (keep_resident cache),
  * the meta skeleton takes an fp8 Parameter and the 0-dim f32 persistent
    ``weight_scale`` buffer through ``load_state_dict(strict=False, assign=True)``;
  * §3-168: int8 / int8_convrot layers — the skeleton takes an int8 Parameter
    (``requires_grad=False``) and an ``(o, 1)`` f32 scale, the forward matches
    ``F.linear`` over the dequantized weight plus the unchanged LoRA delta, and
    nothing stored is written;
  * §3-168 C-3: w4a8 — the skeleton takes the packed int8 Parameter ``(o, i//2)``
    and the three auxiliary buffers (fp8 ``(o, i/16)``, f32 ``(o,)``, f32 ``(16,)``),
    the forward matches an independent reference plus the unchanged LoRA delta.

Run with ``.venv-engine`` and ``--noconftest``; the app venv skips the module.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from engine.sft_quant.quant_service import _patch_model_for_quant  # noqa: E402
from engine.gguf.ic_lora_common import IC_LORA_SPECS_ATTR  # noqa: E402

OUT_F, IN_F, RANK = 16, 12, 4
BF16 = torch.bfloat16
FP8 = torch.float8_e4m3fn


def _u8(t: torch.Tensor) -> torch.Tensor:
    return t.detach().reshape(-1).view(torch.uint8).clone()


def _linear(weight: torch.Tensor, scale: float | None = None) -> nn.Linear:
    """A Linear patched exactly as the module op patches one, weight as given."""
    root = nn.Module()
    root.lin = nn.Linear(IN_F, OUT_F, bias=True)
    root.lin.bias = nn.Parameter(torch.randn(OUT_F).to(BF16), requires_grad=False)
    if scale is not None:
        layers = {"lin": "fp8_scaled"}
    elif weight.dtype in (FP8, torch.float8_e5m2):
        layers = {"lin": "fp8"}
    else:
        layers = {}
    _patch_model_for_quant(root, layers)
    root.lin.weight = nn.Parameter(weight, requires_grad=False)
    if scale is not None:
        root.lin._buffers["weight_scale"] = torch.tensor(scale, dtype=torch.float32)
    return root.lin


def _set_specs(m: nn.Linear, factors) -> None:
    specs = []
    for i, (a, b, s) in enumerate(factors):
        m.register_buffer(f"_ic_lora_A_{i}", a, persistent=False)
        m.register_buffer(f"_ic_lora_B_{i}", b, persistent=False)
        specs.append((f"_ic_lora_A_{i}", f"_ic_lora_B_{i}", s))
    setattr(m, IC_LORA_SPECS_ATTR, specs)


def _factors(n: int):
    return [
        (torch.randn(RANK, IN_F).to(BF16), torch.randn(OUT_F, RANK).to(BF16), s)
        for s in (0.75, -0.3, 0.5)[:n]
    ]


def _reference(w: torch.Tensor, scale: float | None, factors, x_dtype=BF16) -> torch.Tensor:
    wd = w.float() if scale is None else w.float() * torch.tensor(scale, dtype=torch.float32)
    wd = wd.to(x_dtype)
    acc = None
    for a, b, s in factors:
        d = torch.matmul(b.float() * s, a.float()).to(x_dtype)
        acc = d if acc is None else acc + d
    return wd if acc is None else wd + acc


def test_bf16_layer_without_lora_is_byte_identical_to_plain_linear():
    torch.manual_seed(1)
    W = torch.randn(OUT_F, IN_F).to(BF16)
    x = torch.randn(5, IN_F).to(BF16)
    m = _linear(W.clone())
    plain = nn.Linear(IN_F, OUT_F, bias=True).to(BF16)
    plain.weight = nn.Parameter(W.clone(), requires_grad=False)
    plain.bias = nn.Parameter(m.bias.detach().clone(), requires_grad=False)
    assert torch.equal(m(x), plain(x))
    assert torch.equal(m(x), F.linear(x, W, m.bias))


@pytest.mark.parametrize("scale", [0.0625, None], ids=["scaled", "plain_cast"])
@pytest.mark.parametrize("n_lora", [0, 1, 2])
def test_fp8_layer_matches_reference(scale, n_lora):
    torch.manual_seed(2 + n_lora)
    W8 = (torch.randn(OUT_F, IN_F) * 4).to(FP8)
    x = torch.randn(5, IN_F).to(BF16)
    m = _linear(W8.clone(), scale)
    factors = _factors(n_lora)
    if factors:
        _set_specs(m, factors)
    ref = _reference(W8, scale, factors)
    assert torch.equal(m(x), F.linear(x, ref, m.bias))


def test_bf16_layer_with_lora_matches_reference():
    torch.manual_seed(5)
    W = torch.randn(OUT_F, IN_F).to(BF16)
    x = torch.randn(5, IN_F).to(BF16)
    m = _linear(W.clone())
    factors = _factors(2)
    _set_specs(m, factors)
    assert torch.equal(m(x), F.linear(x, _reference(W, None, factors), m.bias))


def test_lora_delta_is_not_rounded_to_fp8():
    """The delta must be cast to x.dtype: an fp8-rounded delta would differ."""
    torch.manual_seed(6)
    # A small base weight so the delta dominates: fp8 keeps 3 mantissa bits,
    # bf16 keeps 7, and the difference must reach the output.
    W8 = (torch.randn(OUT_F, IN_F) * 0.01).to(FP8)
    x = torch.randn(5, IN_F).to(BF16)
    m = _linear(W8.clone(), 0.5)
    factors = [(torch.randn(RANK, IN_F).to(BF16), torch.randn(OUT_F, RANK).to(BF16), 1.0)]
    _set_specs(m, factors)
    a, b, s = factors[0]
    fp8_rounded = (W8.float() * 0.5).to(BF16) + torch.matmul(b.float() * s, a.float()).to(FP8).to(BF16)
    out = m(x)
    assert torch.equal(out, F.linear(x, _reference(W8, 0.5, factors), m.bias))
    assert not torch.equal(out, F.linear(x, fp8_rounded, m.bias))


@pytest.mark.parametrize("scale", [0.25, None], ids=["scaled", "plain_cast"])
def test_forward_does_not_mutate_weight_or_scale(scale):
    torch.manual_seed(7)
    W8 = (torch.randn(OUT_F, IN_F) * 4).to(FP8)
    x = torch.randn(5, IN_F).to(BF16)
    m = _linear(W8.clone(), scale)
    _set_specs(m, _factors(2))
    w_before = _u8(m.weight)
    s_before = None if scale is None else _u8(m.weight_scale)
    _ = m(x)
    _ = m(x)
    assert m.weight.dtype == FP8
    assert torch.equal(_u8(m.weight), w_before)
    if scale is not None:
        assert m.weight_scale.dtype == torch.float32
        assert torch.equal(_u8(m.weight_scale), s_before)


def test_fp8_bias_is_cast_to_activation_dtype():
    torch.manual_seed(8)
    W8 = (torch.randn(OUT_F, IN_F) * 4).to(FP8)
    x = torch.randn(5, IN_F).to(BF16)
    m = _linear(W8.clone(), 0.5)
    m.bias = nn.Parameter(torch.randn(OUT_F).to(FP8), requires_grad=False)
    assert torch.equal(m(x), F.linear(x, _reference(W8, 0.5, []), m.bias.to(BF16)))



def test_e5m2_weight_without_scale_matches_reference():
    torch.manual_seed(10)
    W5 = (torch.randn(OUT_F, IN_F) * 4).to(torch.float8_e5m2)
    x = torch.randn(5, IN_F).to(BF16)
    m = _linear(W5.clone())
    assert torch.equal(m(x), F.linear(x, W5.float().to(BF16), m.bias))
    factors = _factors(1)
    _set_specs(m, factors)
    assert torch.equal(m(x), F.linear(x, _reference(W5, None, factors), m.bias))
    assert m.weight.dtype == torch.float8_e5m2

def test_meta_skeleton_takes_fp8_parameter_and_scale_buffer():
    torch.manual_seed(9)
    with torch.device("meta"):
        root = nn.Module()
        root.blk = nn.Module()
        root.blk.lin = nn.Linear(IN_F, OUT_F, bias=True)
        root.blk.plain = nn.Linear(IN_F, OUT_F, bias=False)
    _patch_model_for_quant(root, {"blk.lin": "fp8_scaled", "blk.plain": "fp8"})
    # The scale buffer is persistent: it is part of the skeleton's state_dict keys.
    assert "blk.lin.weight_scale" in root.state_dict()
    assert "weight_scale" not in root.blk.plain._buffers

    W8 = (torch.randn(OUT_F, IN_F) * 4).to(FP8)
    P8 = (torch.randn(OUT_F, IN_F) * 4).to(FP8)
    bias = torch.randn(OUT_F).to(BF16)
    sd = {
        "blk.lin.weight": W8,
        "blk.lin.bias": bias,
        "blk.lin.weight_scale": torch.tensor(0.125, dtype=torch.float32),
        "blk.plain.weight": P8,
    }
    result = root.load_state_dict(sd, strict=False, assign=True)
    assert not result.unexpected_keys and not result.missing_keys

    assert isinstance(root.blk.lin.weight, nn.Parameter)
    assert root.blk.lin.weight.dtype == FP8 and root.blk.lin.weight.device.type == "cpu"
    # §3-168 ruling 8: fp8 layers take the same requires_grad=False rule as int8
    assert not root.blk.lin.weight.requires_grad and not root.blk.plain.weight.requires_grad
    assert root.blk.lin.weight_scale.dim() == 0
    assert root.blk.lin.weight_scale.item() == 0.125
    assert not [n for n, t in root.state_dict().items() if t.device.type == "meta"]

    x = torch.randn(5, IN_F).to(BF16)
    assert torch.equal(root.blk.lin(x), F.linear(x, _reference(W8, 0.125, []), bias))
    assert torch.equal(root.blk.plain(x), F.linear(x, _reference(P8, None, [])))


def test_unknown_scaled_layer_raises():
    root = nn.Module()
    root.lin = nn.Linear(IN_F, OUT_F)
    with pytest.raises(AttributeError):
        _patch_model_for_quant(root, {"missing.lin": "fp8"})


# ──────────────────────────────────────────────────────────────────────────────
# §3-168: int8 / int8_convrot
# ──────────────────────────────────────────────────────────────────────────────

I8 = torch.int8
IN_ROT = 512  # ConvRot needs in_features % 256 == 0; two groups


def _h256() -> torch.Tensor:
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float32)
    m = h4
    for _ in range(3):
        m = torch.kron(m, h4)
    return m / 16


def _int8_reference(q: torch.Tensor, scale: torch.Tensor, rotate: bool) -> torch.Tensor:
    """Written out independently of engine.sft_quant.dequant (fp32, then bf16)."""
    w = q.float() * scale
    if rotate:
        o, i = w.shape
        w = (w.view(o, i // 256, 256) @ _h256()).view(o, i)
    return w.to(BF16)


def _int8_root(scheme: str, in_f: int):
    """A meta skeleton with one quantized Linear, loaded like the builder does."""
    with torch.device("meta"):
        root = nn.Module()
        root.blk = nn.Module()
        root.blk.lin = nn.Linear(in_f, OUT_F, bias=True)
    _patch_model_for_quant(root, {"blk.lin": scheme})
    q = torch.randint(-127, 128, (OUT_F, in_f), dtype=I8)
    scale = (torch.rand(OUT_F, 1) * 0.01 + 1e-3).to(torch.float32)
    bias = torch.randn(OUT_F).to(BF16)
    sd = {"blk.lin.weight": q, "blk.lin.bias": bias, "blk.lin.weight_scale": scale}
    result = root.load_state_dict(sd, strict=False, assign=True)
    assert not result.unexpected_keys and not result.missing_keys
    return root, q, scale, bias


def test_int8_parameter_cannot_require_grad():
    """Why the module op re-makes the weight: the default Parameter fails on int8."""
    with pytest.raises(RuntimeError):
        nn.Parameter(torch.zeros(2, 2, dtype=I8))
    with torch.device("meta"):
        lin = nn.Linear(4, 2)
    with pytest.raises(RuntimeError):
        lin.load_state_dict({"weight": torch.zeros(2, 4, dtype=I8), "bias": torch.zeros(2)}, assign=True)


@pytest.mark.parametrize("scheme,in_f", [("int8", IN_F), ("int8_convrot", IN_ROT)])
def test_meta_skeleton_takes_int8_parameter_and_row_scale(scheme, in_f):
    torch.manual_seed(20)
    with torch.device("meta"):
        root = nn.Module()
        root.blk = nn.Module()
        root.blk.lin = nn.Linear(in_f, OUT_F, bias=True)
    _patch_model_for_quant(root, {"blk.lin": scheme})
    skel = root.state_dict()
    assert skel["blk.lin.weight"].shape == (OUT_F, in_f)
    assert skel["blk.lin.weight_scale"].shape == (OUT_F, 1)
    assert skel["blk.lin.weight_scale"].dtype == torch.float32
    assert not root.blk.lin.weight.requires_grad
    assert root.blk.lin._sft_scheme == scheme

    root, q, scale, bias = _int8_root(scheme, in_f)
    w = root.blk.lin.weight
    assert isinstance(w, nn.Parameter) and w.dtype == I8 and not w.requires_grad
    assert root.blk.lin.weight_scale.dtype == torch.float32
    assert root.blk.lin.weight_scale.shape == (OUT_F, 1)
    assert not [n for n, t in root.state_dict().items() if t.device.type == "meta"]


@pytest.mark.parametrize("scheme,in_f", [("int8", IN_F), ("int8_convrot", IN_ROT)])
@pytest.mark.parametrize("n_lora", [0, 1, 2])
def test_int8_layer_matches_reference(scheme, in_f, n_lora):
    torch.manual_seed(21 + n_lora)
    root, q, scale, bias = _int8_root(scheme, in_f)
    m = root.blk.lin
    x = torch.randn(5, in_f).to(BF16)
    factors = [
        (torch.randn(RANK, in_f).to(BF16), torch.randn(OUT_F, RANK).to(BF16), s)
        for s in (0.75, -0.3)[:n_lora]
    ]
    if factors:
        _set_specs(m, factors)
    wd = _int8_reference(q, scale, scheme == "int8_convrot")
    acc = None
    for a, b, s in factors:  # the LoRA delta, exactly as for fp8
        d = torch.matmul(b.float() * s, a.float()).to(BF16)
        acc = d if acc is None else acc + d
    ref = wd if acc is None else wd + acc
    assert torch.equal(m(x), F.linear(x, ref, bias))


def test_int8_convrot_differs_from_int8():
    torch.manual_seed(25)
    root, q, scale, bias = _int8_root("int8_convrot", IN_ROT)
    x = torch.randn(5, IN_ROT).to(BF16)
    rotated = root.blk.lin(x)
    root.blk.lin._sft_scheme = "int8"
    assert not torch.equal(rotated, root.blk.lin(x))


@pytest.mark.parametrize("scheme,in_f", [("int8", IN_F), ("int8_convrot", IN_ROT)])
def test_int8_forward_does_not_mutate_weight_or_scale(scheme, in_f):
    torch.manual_seed(26)
    root, q, scale, bias = _int8_root(scheme, in_f)
    m = root.blk.lin
    _set_specs(m, [(torch.randn(RANK, in_f).to(BF16), torch.randn(OUT_F, RANK).to(BF16), 0.5)])
    w_before, s_before = _u8(m.weight), _u8(m.weight_scale)
    x = torch.randn(5, in_f).to(BF16)
    _ = m(x)
    _ = m(x)
    assert m.weight.dtype == I8 and m.weight_scale.dtype == torch.float32
    assert torch.equal(_u8(m.weight), w_before)
    assert torch.equal(_u8(m.weight_scale), s_before)


# ──────────────────────────────────────────────────────────────────────────────
# §3-168 C-3: w4a8
# ──────────────────────────────────────────────────────────────────────────────


def _w4a8_reference(packed: torch.Tensor, aux: dict) -> torch.Tensor:
    """Written out independently of engine.sft_quant.dequant (fp32, then bf16):
    low nibble = even column, codebook, x s_rel per 16 columns, int8 grid,
    x s_channel, ConvRot."""
    o, half = packed.shape
    i = half * 2
    u = packed.view(torch.uint8).to(torch.int64)
    codes = torch.empty(o, i, dtype=torch.int64)
    codes[:, 0::2] = u & 0x0F
    codes[:, 1::2] = u >> 4
    v = aux["weight_codebook"][codes]
    s_rel = aux["weight_s_rel"].float().repeat_interleave(16, dim=1)
    v = torch.clamp(torch.round(v * s_rel), -127, 127)
    v = v * aux["weight_s_channel"][:, None]
    return (v.view(o, i // 256, 256) @ _h256()).view(o, i).to(BF16)


def _w4a8_root(in_f: int = IN_ROT, s_rel_raw_u8: bool = False):
    """A meta skeleton with one w4a8 Linear, loaded like the builder does."""
    from engine.sft_quant.dequant import normalize_aux

    with torch.device("meta"):
        root = nn.Module()
        root.blk = nn.Module()
        root.blk.lin = nn.Linear(in_f, OUT_F, bias=True)
    _patch_model_for_quant(root, {"blk.lin": "w4a8"})
    packed = torch.randint(-128, 128, (OUT_F, in_f // 2), dtype=I8)
    s_rel = (torch.rand(OUT_F, in_f // 16) * 150 + 20).to(FP8)
    aux = {
        "weight_s_rel": s_rel,
        "weight_s_channel": (torch.rand(OUT_F) * 0.01 + 1e-3).to(torch.float32),
        "weight_codebook": torch.sort(torch.randn(16) * 0.7).values.to(torch.float32),
    }
    raw = dict(aux, weight_s_rel=s_rel.view(torch.uint8).clone()) if s_rel_raw_u8 else aux
    bias = torch.randn(OUT_F).to(BF16)
    sd = {"blk.lin.weight": packed, "blk.lin.bias": bias}
    for leaf, t in raw.items():
        sd[f"blk.lin.{leaf}"] = normalize_aux("w4a8", leaf, t, OUT_F, in_f)
    result = root.load_state_dict(sd, strict=False, assign=True)
    assert not result.unexpected_keys and not result.missing_keys
    return root, packed, aux, bias


def test_meta_skeleton_takes_w4a8_packed_parameter_and_three_aux():
    with torch.device("meta"):
        root = nn.Module()
        root.blk = nn.Module()
        root.blk.lin = nn.Linear(IN_ROT, OUT_F, bias=True)
    _patch_model_for_quant(root, {"blk.lin": "w4a8"})
    skel = root.state_dict()
    assert skel["blk.lin.weight"].shape == (OUT_F, IN_ROT // 2)
    assert (skel["blk.lin.weight_s_rel"].shape, skel["blk.lin.weight_s_rel"].dtype) == (
        (OUT_F, IN_ROT // 16), FP8)
    assert (skel["blk.lin.weight_s_channel"].shape, skel["blk.lin.weight_s_channel"].dtype) == (
        (OUT_F,), torch.float32)
    assert (skel["blk.lin.weight_codebook"].shape, skel["blk.lin.weight_codebook"].dtype) == (
        (16,), torch.float32)
    assert not root.blk.lin.weight.requires_grad and root.blk.lin._sft_scheme == "w4a8"

    torch.manual_seed(40)
    root, packed, aux, bias = _w4a8_root(s_rel_raw_u8=True)  # U8 on disk -> fp8 buffer
    w = root.blk.lin.weight
    assert isinstance(w, nn.Parameter) and w.dtype == I8 and not w.requires_grad
    assert w.shape == (OUT_F, IN_ROT // 2)
    assert root.blk.lin.weight_s_rel.dtype == FP8
    assert torch.equal(_u8(root.blk.lin.weight_s_rel), _u8(aux["weight_s_rel"]))
    assert not [n for n, t in root.state_dict().items() if t.device.type == "meta"]


@pytest.mark.parametrize("n_lora", [0, 1, 2])
def test_w4a8_layer_matches_reference(n_lora):
    torch.manual_seed(41 + n_lora)
    root, packed, aux, bias = _w4a8_root()
    m = root.blk.lin
    x = torch.randn(5, IN_ROT).to(BF16)
    factors = [
        (torch.randn(RANK, IN_ROT).to(BF16), torch.randn(OUT_F, RANK).to(BF16), s)
        for s in (0.75, -0.3)[:n_lora]
    ]
    if factors:
        _set_specs(m, factors)
    wd = _w4a8_reference(packed, aux)
    acc = None
    for a, b, s in factors:  # the LoRA delta, exactly as for fp8 / int8
        d = torch.matmul(b.float() * s, a.float()).to(BF16)
        acc = d if acc is None else acc + d
    ref = wd if acc is None else wd + acc
    assert torch.equal(m(x), F.linear(x, ref, bias))


def test_w4a8_forward_does_not_mutate_weight_or_aux():
    torch.manual_seed(45)
    root, packed, aux, bias = _w4a8_root()
    m = root.blk.lin
    _set_specs(m, [(torch.randn(RANK, IN_ROT).to(BF16), torch.randn(OUT_F, RANK).to(BF16), 0.5)])
    names = ("weight", "weight_s_rel", "weight_s_channel", "weight_codebook")
    before = {n: _u8(getattr(m, n)) for n in names}
    x = torch.randn(5, IN_ROT).to(BF16)
    _ = m(x)
    _ = m(x)
    for n in names:
        assert torch.equal(_u8(getattr(m, n)), before[n]), n
