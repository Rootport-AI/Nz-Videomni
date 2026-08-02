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
    classify_lora_axis,
    detach_ic_loras,
    normalize_ic_loras,
    strength_for_prefix,
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


# --------------------------------------------------------------------- #
# Per-axis strength (audio_strength)                                     #
# --------------------------------------------------------------------- #

# (module_prefix, expected axis) — the exact-match specials must beat the
# `audio_` prefix rule, so ordering itself is under test here.
_AXIS_CASES = [
    ("transformer_blocks.0.audio_attn1.to_q", "audio"),
    ("transformer_blocks.0.audio_to_video_attn.to_q", "video"),
    ("transformer_blocks.0.video_to_audio_attn.to_k", "audio"),
    ("transformer_blocks.0.attn1.to_q", "video"),
    ("av_ca_v2a_gate_adaln_single.linear", "audio"),
    ("av_ca_a2v_gate_adaln_single.linear", "video"),
    ("audio_proj_out", "audio"),
    ("patchify_proj", "video"),
]


def test_classify_lora_axis():
    for prefix, axis in _AXIS_CASES:
        assert classify_lora_axis(prefix) == axis, prefix


def test_strength_for_prefix_none_follows_video():
    for prefix, _axis in _AXIS_CASES:  # G-BC: None → strength, audio keys included
        assert strength_for_prefix(prefix, 0.8, None) == 0.8
    assert strength_for_prefix("transformer_blocks.0.audio_attn1.to_q", 0.8, 0.0) == 0.0
    assert strength_for_prefix("transformer_blocks.0.attn1.to_q", 0.8, 0.0) == 0.8


def test_normalize_ic_loras_accepts_2_and_3_tuples():
    assert normalize_ic_loras([("a.safetensors", 1)]) == [("a.safetensors", 1.0, None)]
    assert normalize_ic_loras([("a.safetensors", 1, 0)]) == [("a.safetensors", 1.0, 0.0)]


def _av_tree(w_video: torch.Tensor, w_audio: torch.Tensor):
    """Tree whose named_modules() are transformer_blocks.0.{attn1,audio_attn1}.to_q."""
    def _lin(w: torch.Tensor) -> nn.Linear:
        lin = nn.Linear(IN_F, OUT_F, bias=False)
        _patch_linear_for_ggml_dequant(lin)
        lin.weight = nn.Parameter(w.clone(), requires_grad=False)
        return lin

    block = nn.Module()
    block.attn1 = nn.Module()
    block.attn1.to_q = _lin(w_video)
    block.audio_attn1 = nn.Module()
    block.audio_attn1.to_q = _lin(w_audio)
    root = nn.Module()
    root.transformer_blocks = nn.ModuleList([block])
    return root, block.attn1.to_q, block.audio_attn1.to_q


def _save_av_lora(path, A: torch.Tensor, B: torch.Tensor, *, video: bool = True) -> None:
    from safetensors.torch import save_file

    sd = {}
    if video:
        base = "diffusion_model.transformer_blocks.0.attn1.to_q"
        sd[base + ".lora_A.weight"] = A.to(torch.bfloat16)
        sd[base + ".lora_B.weight"] = B.to(torch.bfloat16)
    base = "diffusion_model.transformer_blocks.0.audio_attn1.to_q"
    sd[base + ".lora_A.weight"] = A.to(torch.bfloat16)
    sd[base + ".lora_B.weight"] = B.to(torch.bfloat16)
    save_file(sd, str(path))


def test_attach_audio_zero_skips_audio_axis_linear():
    """audio_strength=0 → audio Linear stays byte-identical to the no-LoRA path."""
    pytest.importorskip("ltx_core")
    import tempfile
    from pathlib import Path

    from engine.gguf.ic_lora_common import attach_ic_loras

    torch.manual_seed(8)
    Wv, Wa = torch.randn(OUT_F, IN_F), torch.randn(OUT_F, IN_F)
    x = torch.randn(5, IN_F)
    A, B = torch.randn(RANK, IN_F), torch.randn(OUT_F, RANK)
    root, vid, aud = _av_tree(Wv, Wa)

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "lora.safetensors"
        _save_av_lora(path, A, B)
        assert attach_ic_loras(root, [(str(path), 1.0, 0.0)]) == 1
        assert "_ic_lora_A_0" in vid._buffers
        assert "_ic_lora_A_0" not in aud._buffers
        assert getattr(aud, IC_LORA_SPECS_ATTR, None) is None
        assert torch.equal(aud(x), F.linear(x, Wa))  # no-LoRA byte identity
        delta = torch.matmul(B.to(torch.bfloat16).float(), A.to(torch.bfloat16).float())
        assert torch.equal(vid(x), F.linear(x, Wv + delta.to(Wv.dtype)))


def test_attach_legacy_two_tuple_hits_both_axes():
    """No audio_strength → classification is not consulted, both Linears attach."""
    pytest.importorskip("ltx_core")
    import tempfile
    from pathlib import Path

    from engine.gguf.ic_lora_common import attach_ic_loras

    torch.manual_seed(9)
    Wv, Wa = torch.randn(OUT_F, IN_F), torch.randn(OUT_F, IN_F)
    A, B = torch.randn(RANK, IN_F), torch.randn(OUT_F, RANK)
    root, vid, aud = _av_tree(Wv, Wa)

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "lora.safetensors"
        _save_av_lora(path, A, B)
        assert attach_ic_loras(root, [(str(path), 1.0)]) == 2
        assert "_ic_lora_A_0" in vid._buffers
        assert "_ic_lora_A_0" in aud._buffers


def test_full_mute_does_not_warn_key_format_mismatch(caplog):
    """All keys muted is not a key-format mismatch — the loud WARN must stay quiet."""
    pytest.importorskip("ltx_core")
    import logging
    import tempfile
    from pathlib import Path

    from engine.gguf.ic_lora_common import attach_ic_loras

    torch.manual_seed(10)
    Wv, Wa = torch.randn(OUT_F, IN_F), torch.randn(OUT_F, IN_F)
    A, B = torch.randn(RANK, IN_F), torch.randn(OUT_F, RANK)
    root, _vid, aud = _av_tree(Wv, Wa)

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "audio_only.safetensors"
        _save_av_lora(path, A, B, video=False)  # audio-axis keys only
        with caplog.at_level(logging.WARNING, logger="engine.gguf.ic_lora_common"):
            assert attach_ic_loras(root, [(str(path), 1.0, 0.0)]) == 0
        assert "KEY-FORMAT" not in caplog.text
        assert getattr(aud, IC_LORA_SPECS_ATTR, None) is None
