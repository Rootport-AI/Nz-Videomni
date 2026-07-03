"""CPU-only synthetic checks for the IC-LoRA Phase B forward-time patch.

Run with the ENGINE venv:
    ./.venv-engine/Scripts/python.exe logs_diag/ic_lora_phaseb_synthetic_check.py

Checks (all CPU, no GPU, no real model):
  1. No-LoRA forward is byte-identical to F.linear(x, dequant(weight), bias).
  2. LoRA forward == F.linear(x, dequant(weight) + delta, bias) exactly (fp32),
     where delta = matmul(B.float()*s, A.float()).
  3. Multiple LoRAs accumulate sequentially.
  4. Float-weight (non-quantized) module uses out-of-place add, does NOT mutate
     the stored buffer, and equals the reference.
  5. attach_ic_loras resolves prefix->module via a real (tiny) safetensors +
     COMFY rename, registers buffers that .to() moves, and detach clears state.
  6. 0-match adapter WARNs and attaches nothing.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.gguf import quant_service as qs
from engine.gguf.quant_service import (
    GGMLQuantizedTensor,
    _GGML_F32,
    _patch_linear_for_ggml_dequant,
    dequantize_ggml_tensor,
)
from engine.gguf.ic_lora_common import (
    IC_LORA_SPECS_ATTR,
    attach_ic_loras,
    detach_ic_loras,
)

torch.manual_seed(0)
OK = "OK"
FAIL = "FAIL"
results = []


def _make_quant_linear(out_f: int, in_f: int, weight: torch.Tensor) -> nn.Linear:
    """Return a ggml-patched Linear whose weight is an F32 GGMLQuantizedTensor."""
    m = nn.Linear(in_f, out_f, bias=True)
    with torch.no_grad():
        bias = m.bias.clone()
    _patch_linear_for_ggml_dequant(m)
    raw = weight.contiguous().view(torch.uint8).reshape(-1)
    m.weight = GGMLQuantizedTensor(raw, _GGML_F32, (out_f, in_f))
    m.bias = nn.Parameter(bias)
    return m


def check(name: str, cond: bool) -> None:
    results.append((name, cond))
    print(f"[{OK if cond else FAIL}] {name}")


# ── 1 + 2 + 3: quantized-weight forward ──────────────────────────────────────
out_f, in_f, rank = 16, 12, 4
W = torch.randn(out_f, in_f, dtype=torch.float32)
x = torch.randn(3, in_f, dtype=torch.float32)

m = _make_quant_linear(out_f, in_f, W)
dq = dequantize_ggml_tensor(
    m.weight.as_subclass(torch.Tensor).view(torch.uint8), _GGML_F32, (out_f, in_f), x.dtype
)
check("dequant(F32) reproduces the source weight exactly", torch.equal(dq, W))

# 1: no-LoRA path byte-identical
out_no_lora = m(x)
ref_no_lora = F.linear(x, dq, m.bias)
check("no-LoRA forward == F.linear(x, dequant, bias) [byte-identical]",
      torch.equal(out_no_lora, ref_no_lora))

# 2: single LoRA
A = torch.randn(rank, in_f, dtype=torch.float32)
B = torch.randn(out_f, rank, dtype=torch.float32)
s = 0.7
m._ic_lora_A_0 = A  # noqa: SLF001 - simulate attach
m._ic_lora_B_0 = B
setattr(m, IC_LORA_SPECS_ATTR, [("_ic_lora_A_0", "_ic_lora_B_0", s)])
delta = torch.matmul(B.float() * s, A.float())
ref_lora = F.linear(x, dq + delta.to(dq.dtype), m.bias)
out_lora = m(x)
check("single-LoRA forward == F.linear(x, dequant+delta, bias)",
      torch.equal(out_lora, ref_lora))

# quant bytes untouched by forward (cache-safety)
check("GGMLQuantizedTensor bytes unchanged after LoRA forward", torch.equal(
    dequantize_ggml_tensor(m.weight.as_subclass(torch.Tensor).view(torch.uint8),
                           _GGML_F32, (out_f, in_f), x.dtype), W))

# 3: two LoRAs accumulate
A2 = torch.randn(rank, in_f, dtype=torch.float32)
B2 = torch.randn(out_f, rank, dtype=torch.float32)
s2 = -0.4
m._ic_lora_A_1 = A2
m._ic_lora_B_1 = B2
getattr(m, IC_LORA_SPECS_ATTR).append(("_ic_lora_A_1", "_ic_lora_B_1", s2))
d2 = torch.matmul(B2.float() * s2, A2.float())
ref2 = F.linear(x, dq + delta.to(dq.dtype) + d2.to(dq.dtype), m.bias)
check("two-LoRA forward accumulates sequentially",
      torch.equal(m(x), ref2))

# ── 4: float-weight (non-quantized) module, out-of-place ──────────────────────
mf = nn.Linear(in_f, out_f, bias=True)
Wf = torch.randn(out_f, in_f, dtype=torch.float32)
_patch_linear_for_ggml_dequant(mf)
mf.weight = torch.nn.Parameter(Wf.clone(), requires_grad=False)  # plain float weight
mf.bias = nn.Parameter(torch.randn(out_f))
mf._ic_lora_A_0 = A
mf._ic_lora_B_0 = B
setattr(mf, IC_LORA_SPECS_ATTR, [("_ic_lora_A_0", "_ic_lora_B_0", s)])
before = mf.weight.detach().clone()
ref_f = F.linear(x, Wf + delta.to(Wf.dtype), mf.bias)
out_f_res = mf(x)
check("float-weight LoRA forward == F.linear(x, W+delta, bias)",
      torch.equal(out_f_res, ref_f))
check("float-weight buffer NOT mutated by forward (out-of-place)",
      torch.equal(mf.weight.detach(), before))

# ── 5 + 6: attach/detach via a real tiny safetensors + COMFY rename ───────────
from safetensors.torch import save_file  # noqa: E402


class Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(in_f, out_f, bias=False)


class Root(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = Inner()


root = Root()
# ggml-patch the target so its forward is the dequant path (weight left float,
# exercising the float branch through the real attach machinery).
_patch_linear_for_ggml_dequant(root.block.lin)
root.block.lin.weight = torch.nn.Parameter(W.clone(), requires_grad=False)

with tempfile.TemporaryDirectory() as td:
    good = Path(td) / "good.safetensors"
    save_file(
        {
            "diffusion_model.block.lin.lora_A.weight": A.to(torch.bfloat16),
            "diffusion_model.block.lin.lora_B.weight": B.to(torch.bfloat16),
        },
        str(good),
    )
    n = attach_ic_loras(root, [(str(good), s)])
    check("attach resolves prefix 'block.lin' -> Linear (1 match)", n == 1)
    specs = getattr(root.block.lin, IC_LORA_SPECS_ATTR, None)
    check("attach registered specs on the target Linear", bool(specs))
    check("attached A/B live in _buffers (so .to(device) moves them)",
          "_ic_lora_A_0" in root.block.lin._buffers
          and "_ic_lora_B_0" in root.block.lin._buffers)

    # .to() moves the buffers (simulate block-swap device move; CPU->CPU no-op
    # but must preserve them and dtype).
    root.block.to("cpu")
    check(".to() preserves attached buffers",
          "_ic_lora_A_0" in root.block.lin._buffers)

    # forward through the attached module equals reference (bf16 A/B cast to f32)
    Abf = A.to(torch.bfloat16).to(torch.float32)
    Bbf = B.to(torch.bfloat16).to(torch.float32)
    ref_attached = F.linear(x, W + torch.matmul(Bbf * s, Abf).to(W.dtype))
    check("attached-module forward == reference (bf16 factors)",
          torch.equal(root.block.lin(x), ref_attached))

    cleared = detach_ic_loras(root)
    check("detach clears 1 Linear", cleared == 1)
    check("detach removed specs", getattr(root.block.lin, IC_LORA_SPECS_ATTR, None) is None)
    check("detach removed buffers", "_ic_lora_A_0" not in root.block.lin._buffers)
    # post-detach forward == no-LoRA reference (byte-identical toggle-off)
    check("post-detach forward == no-LoRA reference",
          torch.equal(root.block.lin(x), F.linear(x, W)))

    # 6: 0-match adapter (wrong prefix) attaches nothing, WARNs
    bad = Path(td) / "bad.safetensors"
    save_file(
        {
            "diffusion_model.nonexistent.lora_A.weight": A.to(torch.bfloat16),
            "diffusion_model.nonexistent.lora_B.weight": B.to(torch.bfloat16),
        },
        str(bad),
    )
    n_bad = attach_ic_loras(root, [(str(bad), s)])
    check("0-match adapter attaches nothing (returns 0)", n_bad == 0)


passed = sum(1 for _, c in results if c)
print(f"\n{passed}/{len(results)} checks passed")
raise SystemExit(0 if passed == len(results) else 1)
