"""IC-LoRA Phase B — forward-time weight-patch unit tests (CPU-only).

These exercise the per-layer-quant forward path (``ggml_linear_forward``) and the
attach/detach machinery (``engine.gguf.ic_lora_common``) with synthetic tensors —
no GPU, no real model. The delta formula is asserted byte-equal to the reference
``matmul(B.float()*strength, A.float())`` (the same formula the bf16 fuse path
uses, gate G2), and the no-LoRA path is asserted byte-identical to today (gate G1).

The full ``attach_ic_loras`` resolution test loads a tiny safetensors through the
wheel's ``SafetensorsStateDictLoader`` + rename map, so it is skipped when the LTX
wheel is absent (app venv). Everything else runs torch-only.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from engine.gguf.ic_lora_common import (  # noqa: E402
    IC_LORA_SPECS_ATTR,
    detach_ic_loras,
)
from engine.gguf.quant_service import (  # noqa: E402
    GGMLQuantizedTensor,
    _GGML_BF16,
    _patch_linear_for_ggml_dequant,
    dequantize_ggml_tensor,
)

OUT_F, IN_F, RANK = 16, 12, 4


def _quant_linear(weight: torch.Tensor) -> nn.Linear:
    """A ggml-patched Linear whose weight is a BF16-typed GGMLQuantizedTensor.

    ``weight`` must already be bf16. The forward is exercised with a float32
    input so the BF16 dequant undergoes a real bf16->f32 conversion and returns a
    FRESH tensor — exactly like the production quantized kernels (Q4_K/Q8_0 do
    arithmetic and never alias the stored bytes). Using an F32-typed tensor with
    a matching out-dtype would instead return a byte-reinterpret VIEW aliasing the
    buffer (a case the loader never produces), so we deliberately avoid it here.
    """
    m = nn.Linear(IN_F, OUT_F, bias=True)
    bias = m.bias.detach().clone()
    _patch_linear_for_ggml_dequant(m)
    raw = weight.to(torch.bfloat16).contiguous().view(torch.uint8).reshape(-1)
    m.weight = GGMLQuantizedTensor(raw, _GGML_BF16, (OUT_F, IN_F))
    m.bias = nn.Parameter(bias)
    return m


def _set_specs(m: nn.Linear, factors: list[tuple[torch.Tensor, torch.Tensor, float]]) -> None:
    specs = []
    for i, (a, b, s) in enumerate(factors):
        setattr(m, f"_ic_lora_A_{i}", a)
        setattr(m, f"_ic_lora_B_{i}", b)
        specs.append((f"_ic_lora_A_{i}", f"_ic_lora_B_{i}", s))
    setattr(m, IC_LORA_SPECS_ATTR, specs)


def _base(m: nn.Linear, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return dequantize_ggml_tensor(
        m.weight.as_subclass(torch.Tensor).view(torch.uint8), _GGML_BF16, (OUT_F, IN_F), dtype
    )


def test_no_lora_forward_byte_identical():
    torch.manual_seed(1)
    W = torch.randn(OUT_F, IN_F)
    x = torch.randn(5, IN_F)
    m = _quant_linear(W)
    dq = _base(m)
    assert torch.equal(dq, W.to(torch.bfloat16).float())  # bf16-exact
    assert torch.equal(m(x), F.linear(x, dq, m.bias))


def test_single_lora_delta_formula():
    torch.manual_seed(2)
    W = torch.randn(OUT_F, IN_F)
    x = torch.randn(5, IN_F)
    A = torch.randn(RANK, IN_F)
    B = torch.randn(OUT_F, RANK)
    s = 0.75
    m = _quant_linear(W)
    dq = _base(m)
    _set_specs(m, [(A, B, s)])
    delta = torch.matmul(B.float() * s, A.float())
    assert torch.equal(m(x), F.linear(x, dq + delta.to(dq.dtype), m.bias))


def test_multiple_loras_accumulate():
    torch.manual_seed(3)
    W = torch.randn(OUT_F, IN_F)
    x = torch.randn(5, IN_F)
    A1, B1, s1 = torch.randn(RANK, IN_F), torch.randn(OUT_F, RANK), 0.5
    A2, B2, s2 = torch.randn(RANK, IN_F), torch.randn(OUT_F, RANK), -0.3
    m = _quant_linear(W)
    dq = _base(m)
    _set_specs(m, [(A1, B1, s1), (A2, B2, s2)])
    # Sequential accumulation (matches both the forward's per-spec in-place add
    # order and Phase A's per-LoRA add_ order — float add is not associative).
    ref = dq.clone()
    ref += torch.matmul(B1.float() * s1, A1.float()).to(ref.dtype)
    ref += torch.matmul(B2.float() * s2, A2.float()).to(ref.dtype)
    assert torch.equal(m(x), F.linear(x, ref, m.bias))


def test_quant_bytes_not_mutated_by_forward():
    torch.manual_seed(4)
    W = torch.randn(OUT_F, IN_F)
    x = torch.randn(5, IN_F)
    A, B = torch.randn(RANK, IN_F), torch.randn(OUT_F, RANK)
    m = _quant_linear(W)
    before = _base(m).clone()
    _set_specs(m, [(A, B, 1.0)])
    _ = m(x)
    assert torch.equal(_base(m), before)  # compressed bytes untouched → cache-safe


def test_float_weight_out_of_place():
    torch.manual_seed(5)
    W = torch.randn(OUT_F, IN_F)
    x = torch.randn(5, IN_F)
    A, B, s = torch.randn(RANK, IN_F), torch.randn(OUT_F, RANK), 0.6
    m = nn.Linear(IN_F, OUT_F, bias=True)
    _patch_linear_for_ggml_dequant(m)
    m.weight = nn.Parameter(W.clone(), requires_grad=False)  # plain float weight
    m.bias = nn.Parameter(torch.randn(OUT_F))
    _set_specs(m, [(A, B, s)])
    before = m.weight.detach().clone()
    delta = torch.matmul(B.float() * s, A.float())
    assert torch.equal(m(x), F.linear(x, W + delta.to(W.dtype), m.bias))
    assert torch.equal(m.weight.detach(), before)  # stored buffer NOT mutated


def test_detach_clears_state_and_restores_byte_identity():
    torch.manual_seed(6)
    W = torch.randn(OUT_F, IN_F)
    x = torch.randn(5, IN_F)
    A, B = torch.randn(RANK, IN_F), torch.randn(OUT_F, RANK)
    m = _quant_linear(W)
    root = nn.Module()
    root.lin = m  # tiny tree; _find_target_model falls back to root itself
    # register as buffers (as attach does) so detach's _buffers.pop path runs
    m.register_buffer("_ic_lora_A_0", A, persistent=False)
    m.register_buffer("_ic_lora_B_0", B, persistent=False)
    setattr(m, IC_LORA_SPECS_ATTR, [("_ic_lora_A_0", "_ic_lora_B_0", 1.0)])
    cleared = detach_ic_loras(root)
    assert cleared == 1
    assert getattr(m, IC_LORA_SPECS_ATTR, None) is None
    assert "_ic_lora_A_0" not in m._buffers
    assert torch.equal(m(x), F.linear(x, _base(m), m.bias))  # no-LoRA byte identity


def test_attach_resolves_prefix_to_module():
    """Full attach via wheel loader + COMFY rename — needs the LTX wheel."""
    pytest.importorskip("ltx_core")
    import tempfile
    from pathlib import Path

    from safetensors.torch import save_file

    from engine.gguf.ic_lora_common import attach_ic_loras

    torch.manual_seed(7)
    W = torch.randn(OUT_F, IN_F)
    x = torch.randn(5, IN_F)
    A, B, s = torch.randn(RANK, IN_F), torch.randn(OUT_F, RANK), 0.5

    inner = nn.Module()
    inner.lin = nn.Linear(IN_F, OUT_F, bias=False)
    _patch_linear_for_ggml_dequant(inner.lin)
    inner.lin.weight = nn.Parameter(W.clone(), requires_grad=False)
    root = nn.Module()
    root.block = inner  # named_modules() name of the Linear == "block.lin"

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "lora.safetensors"
        save_file(
            {
                "diffusion_model.block.lin.lora_A.weight": A.to(torch.bfloat16),
                "diffusion_model.block.lin.lora_B.weight": B.to(torch.bfloat16),
            },
            str(path),
        )
        n = attach_ic_loras(root, [(str(path), s)])
        assert n == 1
        assert "_ic_lora_A_0" in inner.lin._buffers  # rides .to(device)
        Abf = A.to(torch.bfloat16).float()
        Bbf = B.to(torch.bfloat16).float()
        ref = F.linear(x, W + torch.matmul(Bbf * s, Abf).to(W.dtype))
        assert torch.equal(inner.lin(x), ref)
        assert detach_ic_loras(root) == 1
