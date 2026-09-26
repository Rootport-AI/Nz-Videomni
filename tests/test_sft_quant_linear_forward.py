"""§3-167: the ``sft_quant_linear`` forward — test_ic_lora_forward.py's twin (CPU only).

Pins the forward that ``engine.sft_quant.quant_service`` installs on every Linear of
an fp8 transformer:

  * a bf16 layer without LoRA is byte-identical to a plain ``nn.Linear``,
  * fp8 layers (scaled and plain cast), with and without (IC-)LoRA, match the
    reference ``(w.float() * s).to(bf16) + sum(delta)`` bit for bit, where each
    delta is ``matmul(B.float() * strength, A.float()).to(x.dtype)`` — cast to
    the ACTIVATION dtype, never to the fp8 weight dtype,
  * the forward never writes the stored weight or scale (keep_resident cache),
  * the meta skeleton takes an fp8 Parameter and the 0-dim f32 persistent
    ``weight_scale`` buffer through ``load_state_dict(strict=False, assign=True)``.

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
    _patch_model_for_quant(root, frozenset({"lin"}) if scale is not None else frozenset())
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
    _patch_model_for_quant(root, frozenset({"blk.lin"}))
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
        _patch_model_for_quant(root, frozenset({"missing.lin"}))
