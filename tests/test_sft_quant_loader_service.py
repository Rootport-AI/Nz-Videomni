"""§3-167: ``SftQuantLoaderService.install`` against a mock ledger (CPU only).

Pins the three install steps without any weights or GPU:

  * the policy is replaced WHOLESALE — ``sd_ops`` is None (fp8_cast's
    TRANSFORMER_LINEAR_DOWNCAST_MAP would push biases and the bf16 blocks down
    to fp8) and fp8_cast's ``UPCAST_DURING_INFERENCE`` is gone, leaving only
    ``sft_quant_linear``;
  * the builder's loader is an ``SftQuantStateDictLoader`` sharing the ONE Layout
    ``inspect`` returned;
  * errors are never swallowed (inspect's refusal, ``dit_cpu_load=0``, a stray
    fp8 tensor outside a Linear, and the pipeline's ``_install_safetensors``);
  * the pipeline accepts exactly one transformer source.

B-2 additions (shared with engine25): the loader's 0-dim scale and
text_embedding_projection skip, ``sft_transformer_sd_ops`` and
``load_connector_bf16``.

§3-168 additions: int8 / int8_convrot through the loader (quantized weight as
stored, auxiliary tensors F32 in the one ``(o, 1)`` shape, a surplus auxiliary
key raises, F16 -> bf16), loader -> skeleton -> forward end to end, the int8
connector (row / scalar scale, ConvRot, F16), and int8 outside a Linear.
C-3: w4a8 through the loader (packed weight ``(o, i//2)``, fp8 ``s_rel`` ``(o, i/16)``
whether stored as fp8 or U8, f32 ``s_channel`` ``(o,)`` and codebook ``(16,)``),
loader -> skeleton -> forward, and the w4a8 connector (REDGraft).

Run with ``.venv-engine`` and ``--noconftest``; the app venv skips the module.
"""

from __future__ import annotations

import types
from dataclasses import dataclass

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("ltx_core")

# Pre-warm ltx_core.loader first, as engine/worker.py does: importing
# ltx_core.quantization cold hits the wheel's circular loader import.
import ltx_core.loader  # noqa: E402,F401
import torch.nn as nn  # noqa: E402
from ltx_core.quantization import QuantizationPolicy  # noqa: E402

import sft_quant_format  # noqa: E402
from engine.sft_quant.quant_service import (  # noqa: E402
    SftQuantLoaderService,
    SftQuantStateDictLoader,
    sft_transformer_sd_ops,
    load_connector_bf16,
)

_LAYOUT = types.SimpleNamespace(
    config={"transformer": {"num_layers": 48}},
    model_version="2.3.0",
    prefix="model.diffusion_model.",
    layers={"transformer_blocks.2.attn1.to_q": "fp8_scaled"},
    connector_keys=("model.diffusion_model.video_embeddings_connector.w",),
    n_blocks=48,
)


@dataclass(frozen=True)
class _Builder:
    model_loader: object = None
    model_sd_ops: object = None


class _Ledger:
    """Just what install() touches: transformer_builder, quantization, transformer()."""

    def __init__(self, built: nn.Module | None = None) -> None:
        self.transformer_builder = _Builder()
        self.quantization = QuantizationPolicy.fp8_cast()  # what CUDA pipelines start with
        self._built = built if built is not None else nn.Linear(2, 2)
        self.calls = 0

    def transformer(self):
        self.calls += 1
        return self._built


@pytest.fixture
def fake_inspect(monkeypatch):
    seen = []

    def _inspect(path, **_kw):
        seen.append(path)
        return _LAYOUT

    monkeypatch.setattr(sft_quant_format, "inspect", _inspect)
    return seen


def test_install_replaces_loader_and_policy(fake_inspect):
    ledger = _Ledger()
    SftQuantLoaderService("X.safetensors", dit_cpu_load=True).install(ledger)

    assert fake_inspect == ["X.safetensors"]  # inspected exactly once
    loader = ledger.transformer_builder.model_loader
    assert isinstance(loader, SftQuantStateDictLoader)
    assert loader.path == "X.safetensors" and loader.layout is _LAYOUT
    assert loader.metadata("") == _LAYOUT.config
    # prefixed file: the very SDOps the wheel's ModelLedger puts there (2.3 unchanged)
    from ltx_core.model.transformer import LTXV_MODEL_COMFY_RENAMING_MAP

    assert ledger.transformer_builder.model_sd_ops is LTXV_MODEL_COMFY_RENAMING_MAP

    policy = ledger.quantization
    assert policy.sd_ops is None
    assert [op.name for op in policy.module_ops] == ["sft_quant_linear"]  # fp8_cast is gone


def test_module_op_registers_scale_from_the_shared_layout(fake_inspect):
    from ltx_core.model.transformer.model import LTXModel

    ledger = _Ledger()
    SftQuantLoaderService("X.safetensors", dit_cpu_load=True).install(ledger)
    (op,) = ledger.quantization.module_ops
    assert not op.matcher(nn.Linear(2, 2))
    assert op.matcher(LTXModel.__new__(LTXModel))

    # The mutator itself on a stand-in tree carrying the scaled layer's name.
    root = nn.Module()
    root.transformer_blocks = nn.ModuleList([nn.Module() for _ in range(3)])
    root.transformer_blocks[2].attn1 = nn.Module()
    root.transformer_blocks[2].attn1.to_q = nn.Linear(2, 2)
    op.mutator(root)
    assert "transformer_blocks.2.attn1.to_q.weight_scale" in root.state_dict()


def test_dit_cpu_load_off_raises_before_anything(monkeypatch):
    def _never(*_a, **_k):
        raise AssertionError("inspect must not run when dit_cpu_load is off")

    monkeypatch.setattr(sft_quant_format, "inspect", _never)
    ledger = _Ledger()
    before = (ledger.transformer_builder, ledger.quantization)
    with pytest.raises(RuntimeError, match="dit_cpu_load"):
        SftQuantLoaderService("X.safetensors", dit_cpu_load=False).install(ledger)
    assert (ledger.transformer_builder, ledger.quantization) == before


def test_inspect_refusal_propagates(monkeypatch):
    def _refuse(*_a, **_k):
        raise sft_quant_format.QuantFormatError("流儀 F8_E5M2 は扱えません")

    monkeypatch.setattr(sft_quant_format, "inspect", _refuse)
    ledger = _Ledger()
    with pytest.raises(sft_quant_format.QuantFormatError, match="F8_E5M2"):
        SftQuantLoaderService("X.safetensors", dit_cpu_load=True).install(ledger)
    assert ledger.quantization.sd_ops is not None  # still fp8_cast: untouched
    assert ledger.transformer_builder.model_loader is None


def test_pipeline_install_safetensors_does_not_swallow(monkeypatch):
    from engine.pipeline.fast_video_pipeline import LTXFastVideoPipeline

    def _refuse(*_a, **_k):
        raise sft_quant_format.QuantFormatError("per-row weight_scale")

    monkeypatch.setattr(sft_quant_format, "inspect", _refuse)
    fake_self = types.SimpleNamespace(
        _dit_cpu_load=True,
        _ic_loras=[],
        pipeline=types.SimpleNamespace(model_ledger=_Ledger()),
    )
    with pytest.raises(sft_quant_format.QuantFormatError, match="per-row"):
        LTXFastVideoPipeline._install_safetensors(fake_self, "X.safetensors")

    fake_self._dit_cpu_load = False
    with pytest.raises(RuntimeError, match="dit_cpu_load"):
        LTXFastVideoPipeline._install_safetensors(fake_self, "X.safetensors")


def test_wrapped_transformer_detaches_and_passes_clean_model(fake_inspect, monkeypatch):
    built = nn.Module()
    built.lin = nn.Linear(2, 2)
    built.lin.weight = nn.Parameter(torch.zeros(2, 2).to(torch.float8_e4m3fn), requires_grad=False)
    ledger = _Ledger(built)
    detached = []
    import engine.gguf.ic_lora_common as ic

    monkeypatch.setattr(ic, "detach_ic_loras", lambda t: detached.append(t) or 0)
    SftQuantLoaderService("X.safetensors", dit_cpu_load=True, ic_loras_provider=lambda: []).install(ledger)
    assert ledger.transformer() is built
    assert ledger.calls == 1 and detached == [built]


def test_wrapped_transformer_rejects_fp8_outside_linear(fake_inspect):
    built = nn.Module()
    built.norm = nn.LayerNorm(2)
    built.norm.weight = nn.Parameter(torch.ones(2).to(torch.float8_e4m3fn), requires_grad=False)
    ledger = _Ledger(built)
    SftQuantLoaderService("X.safetensors", dit_cpu_load=True).install(ledger)
    with pytest.raises(RuntimeError, match="outside Linear"):
        ledger.transformer()


def _pipeline_kwargs(**over):
    kw = dict(
        checkpoint_path="",
        gemma_root=None,
        upsampler_path="u.safetensors",
        device=torch.device("cpu"),
        gguf_gemma_path="g.gguf",
        component_video_vae_path="v.safetensors",
        component_audio_vae_path="a.safetensors",
        component_text_projection_path="p.safetensors",
    )
    kw.update(over)
    return kw


def test_pipeline_requires_exactly_one_transformer_source():
    from engine.pipeline.fast_video_pipeline import LTXFastVideoPipeline

    with pytest.raises(RuntimeError, match="gguf_transformer_path or safetensors_transformer_path"):
        LTXFastVideoPipeline(**_pipeline_kwargs())
    with pytest.raises(RuntimeError, match="exactly one transformer source"):
        LTXFastVideoPipeline(
            **_pipeline_kwargs(
                gguf_transformer_path="t.gguf", safetensors_transformer_path="t.safetensors"
            )
        )



def test_gemma_connectors_via_fp8_safetensors(tmp_path):
    """The Gemma side reads the connectors out of the fp8 file (prefixed keys,
    F32 -> bf16, BF16 as is) with the same seek + readinto reader."""
    from safetensors.torch import save_file

    from engine.gemma.gguf_quant_service import GemmaGGUFQuantStateDictLoader

    p = "model.diffusion_model."
    v = torch.randn(3, 2).to(torch.bfloat16)
    a = torch.randn(4)
    path = tmp_path / "t.safetensors"
    save_file(
        {
            p + "video_embeddings_connector.w": v,
            p + "audio_embeddings_connector.b": a,
            p + "transformer_blocks.0.attn1.to_q.weight": torch.zeros(2, 2).to(torch.float8_e4m3fn),
        },
        str(path),
    )
    loader = GemmaGGUFQuantStateDictLoader.__new__(GemmaGGUFQuantStateDictLoader)
    loader._connector_gguf_path = str(path)
    loader._connector_sd_ops = None
    out = loader._load_gguf_connectors(torch.device("cpu"))
    assert set(out) == {p + "video_embeddings_connector.w", p + "audio_embeddings_connector.b"}
    assert all(t.dtype == torch.bfloat16 for t in out.values())
    assert torch.equal(out[p + "video_embeddings_connector.w"], v)
    assert torch.equal(out[p + "audio_embeddings_connector.b"], a.to(torch.bfloat16))

    loader._connector_gguf_path = str(tmp_path / "t.bin")
    with pytest.raises(RuntimeError, match="expected a .gguf or .safetensors"):
        loader._load_gguf_connectors(torch.device("cpu"))


def test_gemma_connectors_from_bare_fp8_safetensors(tmp_path):
    """A bare-named file with a scaled fp8 connector: keys come back in the
    original ``model.diffusion_model.`` form, upcast times the scale."""
    from safetensors.torch import save_file

    from engine.gemma.gguf_quant_service import GemmaGGUFQuantStateDictLoader

    w = torch.randn(3, 2).to(torch.float8_e4m3fn)
    path = tmp_path / "bare.safetensors"
    save_file(
        {
            "video_embeddings_connector.w.weight": w,
            "video_embeddings_connector.w.weight_scale": torch.tensor([0.5]),
            "transformer_blocks.0.attn1.to_q.weight": torch.zeros(2, 2).to(torch.float8_e4m3fn),
        },
        str(path),
    )
    loader = GemmaGGUFQuantStateDictLoader.__new__(GemmaGGUFQuantStateDictLoader)
    loader._connector_gguf_path = str(path)
    loader._connector_sd_ops = None
    out = loader._load_gguf_connectors(torch.device("cpu"))
    key = "model.diffusion_model.video_embeddings_connector.w.weight"
    assert set(out) == {key}
    assert torch.equal(out[key], (w.to(torch.float32) * 0.5).to(torch.bfloat16))


# ──────────────────────────────────────────────────────────────────────────────
# B-2: pieces shared with engine25
# ──────────────────────────────────────────────────────────────────────────────


def _bare_layout(**over):
    kw = dict(
        config={"transformer": {}},
        model_version="2.5.0",
        prefix="",
        layers={"transformer_blocks.0.attn1.to_q": "fp8_scaled"},
        connector_keys=("video_embeddings_connector.w",),
        n_blocks=1,
    )
    kw.update(over)
    return types.SimpleNamespace(**kw)


def test_loader_scale_shape_1_becomes_0_dim_and_text_projection_is_skipped(tmp_path):
    from safetensors.torch import save_file

    lin = "transformer_blocks.0.attn1.to_q"
    w = torch.randn(2, 2).to(torch.float8_e4m3fn)
    path = tmp_path / "bare.safetensors"
    save_file(
        {
            f"{lin}.weight": w,
            f"{lin}.weight_scale": torch.tensor([0.25]),
            f"{lin}.input_scale": torch.tensor([1.0]),
            "transformer_blocks.0.scale_shift_table": torch.randn(2),
            "text_embedding_projection.video_aggregate_embed.weight": torch.randn(2, 2).to(torch.bfloat16),
            "video_embeddings_connector.w": torch.randn(2).to(torch.bfloat16),
        },
        str(path),
    )
    loader = SftQuantStateDictLoader(str(path), _bare_layout())
    sd = loader.load("", sd_ops=sft_transformer_sd_ops("")).sd
    assert set(sd) == {f"{lin}.weight", f"{lin}.weight_scale", "transformer_blocks.0.scale_shift_table"}
    assert sd[f"{lin}.weight_scale"].shape == () and sd[f"{lin}.weight_scale"].dtype == torch.float32
    assert float(sd[f"{lin}.weight_scale"]) == 0.25
    assert sd[f"{lin}.weight"].dtype == torch.float8_e4m3fn
    assert sd["transformer_blocks.0.scale_shift_table"].dtype == torch.bfloat16


def test_loader_prefixed_file_keeps_0_dim_scale(tmp_path):
    from safetensors.torch import save_file

    p = "model.diffusion_model."
    lin = "transformer_blocks.0.attn1.to_q"
    path = tmp_path / "p.safetensors"
    save_file(
        {
            f"{p}{lin}.weight": torch.randn(2, 2).to(torch.float8_e4m3fn),
            f"{p}{lin}.weight_scale": torch.tensor(0.5),
            "text_embedding_projection.aggregate_embed.weight": torch.randn(2, 2).to(torch.bfloat16),
        },
        str(path),
    )
    loader = SftQuantStateDictLoader(str(path), _bare_layout(prefix=p, connector_keys=()))
    sd = loader.load("", sd_ops=sft_transformer_sd_ops(p)).sd
    assert set(sd) == {f"{lin}.weight", f"{lin}.weight_scale"}
    assert sd[f"{lin}.weight_scale"].shape == ()


def test_sft_transformer_sd_ops_two_branches():
    from ltx_core.model.transformer import LTXV_MODEL_COMFY_RENAMING_MAP

    assert sft_transformer_sd_ops("model.diffusion_model.") is LTXV_MODEL_COMFY_RENAMING_MAP
    bare = sft_transformer_sd_ops("")
    # identity: every key matches and nothing is renamed (a matcher-less SDOps
    # would return None for everything)
    for key in ("transformer_blocks.0.attn1.to_q.weight", "patchify_proj.bias", "x"):
        assert bare.apply_to_key(key) == key
    with pytest.raises(ValueError, match="prefix"):
        sft_transformer_sd_ops("diffusion_model.")


@pytest.mark.parametrize("prefix", ["model.diffusion_model.", ""], ids=["prefixed", "bare"])
def test_load_connector_bf16_four_dtypes_and_excluded_keys(tmp_path, prefix):
    from safetensors.torch import save_file

    c = prefix + "video_embeddings_connector."
    a = prefix + "audio_embeddings_connector."
    bf16 = torch.randn(2, 3).to(torch.bfloat16)
    f32 = torch.randn(3)
    fp8 = torch.randn(2, 3).to(torch.float8_e4m3fn)
    fp8s = torch.randn(2, 3).to(torch.float8_e5m2)
    marker = torch.tensor(list(b'{"format":"float8_e5m2"}'), dtype=torch.uint8)
    path = tmp_path / "c.safetensors"
    save_file(
        {
            prefix + "transformer_blocks.0.attn1.to_q.weight": torch.zeros(2, 2).to(torch.float8_e4m3fn),
            c + "bf.weight": bf16,
            c + "f.bias": f32,
            a + "plain.weight": fp8,
            a + "scaled.weight": fp8s,
            a + "scaled.weight_scale": torch.tensor([2.0]),
            a + "scaled.input_scale": torch.tensor([1.0]),
            a + "scaled.comfy_quant": marker,
        },
        str(path),
    )
    out = load_connector_bf16(str(path))
    assert set(out) == {
        "video_embeddings_connector.bf.weight",
        "video_embeddings_connector.f.bias",
        "audio_embeddings_connector.plain.weight",
        "audio_embeddings_connector.scaled.weight",
    }
    assert all(t.dtype == torch.bfloat16 for t in out.values())
    assert torch.equal(out["video_embeddings_connector.bf.weight"], bf16)
    assert torch.equal(out["video_embeddings_connector.f.bias"], f32.to(torch.bfloat16))
    assert torch.equal(out["audio_embeddings_connector.plain.weight"], fp8.to(torch.bfloat16))
    assert torch.equal(
        out["audio_embeddings_connector.scaled.weight"],
        (fp8s.to(torch.float32) * 2.0).to(torch.bfloat16),
    )


def test_load_connector_bf16_fp8_bias_is_not_scaled(tmp_path):
    """A connector layer with a scaled fp8 .weight and an fp8 .bias: the scale
    belongs to the weight only, so the bias comes back as a plain cast."""
    from safetensors.torch import save_file

    lay = "video_embeddings_connector.proj"
    w = torch.full((2, 3), 1.5).to(torch.float8_e4m3fn)
    b = torch.full((2,), 1.5).to(torch.float8_e4m3fn)
    path = tmp_path / "bias.safetensors"
    save_file(
        {
            "transformer_blocks.0.attn1.to_q.weight": torch.zeros(2, 2).to(torch.float8_e4m3fn),
            f"{lay}.weight": w,
            f"{lay}.weight_scale": torch.tensor([2.0]),
            f"{lay}.bias": b,
        },
        str(path),
    )
    out = load_connector_bf16(str(path))
    assert set(out) == {f"{lay}.weight", f"{lay}.bias"}
    assert out[f"{lay}.bias"].dtype == torch.bfloat16
    assert torch.equal(out[f"{lay}.bias"], b.to(torch.bfloat16))
    assert torch.equal(out[f"{lay}.weight"], (w.to(torch.float32) * 2.0).to(torch.bfloat16))



# ──────────────────────────────────────────────────────────────────────────────
# §3-168: int8 / int8_convrot
# ──────────────────────────────────────────────────────────────────────────────

_LIN = "transformer_blocks.0.attn1.to_q"


def _marker(convrot: bool) -> torch.Tensor:
    import json

    conf = {"format": "int8_tensorwise", "convrot": convrot, "convrot_groupsize": 256}
    return torch.tensor(list(json.dumps(conf).encode()), dtype=torch.uint8)


def _int8_weight(o: int, i: int) -> torch.Tensor:
    return torch.randint(-127, 128, (o, i), dtype=torch.int8)


@pytest.mark.parametrize(
    "raw_scale",
    [torch.tensor(0.5), torch.tensor([0.5]), torch.full((4, 1), 0.5)],
    ids=["0dim", "(1,)", "(o,1)"],
)
def test_loader_int8_keeps_weight_and_normalizes_scale(tmp_path, raw_scale):
    from safetensors.torch import save_file

    q = _int8_weight(4, 8)
    path = tmp_path / "i8.safetensors"
    save_file(
        {
            f"{_LIN}.weight": q,
            f"{_LIN}.weight_scale": raw_scale,
            f"{_LIN}.comfy_quant": _marker(False),
            f"{_LIN}.input_scale": torch.tensor([1.0]),
            "transformer_blocks.0.norm.weight": torch.randn(8).to(torch.float16),
            "transformer_blocks.0.scale_shift_table": torch.randn(2),
        },
        str(path),
    )
    layout = _bare_layout(layers={_LIN: "int8"}, connector_keys=())
    sd = SftQuantStateDictLoader(str(path), layout).load("", sd_ops=sft_transformer_sd_ops("")).sd
    assert set(sd) == {
        f"{_LIN}.weight", f"{_LIN}.weight_scale",
        "transformer_blocks.0.norm.weight", "transformer_blocks.0.scale_shift_table",
    }
    assert sd[f"{_LIN}.weight"].dtype == torch.int8 and torch.equal(sd[f"{_LIN}.weight"], q)
    s = sd[f"{_LIN}.weight_scale"]
    assert s.dtype == torch.float32 and s.shape == (4, 1) and torch.equal(s, torch.full((4, 1), 0.5))
    # the one rule for non-quantized floats: F16 and F32 -> bf16
    assert sd["transformer_blocks.0.norm.weight"].dtype == torch.bfloat16
    assert sd["transformer_blocks.0.scale_shift_table"].dtype == torch.bfloat16


@pytest.mark.parametrize("scheme", [None, "fp8"], ids=["bf16_layer", "plain_fp8_layer"])
def test_loader_surplus_auxiliary_key_raises(tmp_path, scheme):
    from safetensors.torch import save_file

    w = torch.randn(4, 8).to(torch.bfloat16 if scheme is None else torch.float8_e4m3fn)
    path = tmp_path / "extra.safetensors"
    save_file({f"{_LIN}.weight": w, f"{_LIN}.weight_scale": torch.tensor(0.5)}, str(path))
    layout = _bare_layout(layers={} if scheme is None else {_LIN: scheme}, connector_keys=())
    with pytest.raises(RuntimeError, match="no quantized"):
        SftQuantStateDictLoader(str(path), layout).load("", sd_ops=sft_transformer_sd_ops(""))


@pytest.mark.parametrize("scheme,in_f", [("int8", 16), ("int8_convrot", 256)])
def test_loader_to_skeleton_to_forward(tmp_path, scheme, in_f):
    """File -> loader -> patched meta skeleton -> load_state_dict(assign) -> forward."""
    from safetensors.torch import save_file

    from engine.sft_quant.dequant import dequantize
    from engine.sft_quant.quant_service import _patch_model_for_quant

    torch.manual_seed(31)
    o = 8
    q = _int8_weight(o, in_f)
    scale = torch.rand(o, 1) * 0.01 + 1e-3
    bias = torch.randn(o).to(torch.bfloat16)
    path = tmp_path / "e2e.safetensors"
    save_file(
        {
            f"{_LIN}.weight": q,
            f"{_LIN}.weight_scale": scale,
            f"{_LIN}.bias": bias,
            f"{_LIN}.comfy_quant": _marker(scheme == "int8_convrot"),
        },
        str(path),
    )
    layout = _bare_layout(layers={_LIN: scheme}, connector_keys=())
    sd = SftQuantStateDictLoader(str(path), layout).load("", sd_ops=sft_transformer_sd_ops("")).sd

    with torch.device("meta"):
        root = nn.Module()
        root.transformer_blocks = nn.ModuleList([nn.Module()])
        root.transformer_blocks[0].attn1 = nn.Module()
        root.transformer_blocks[0].attn1.to_q = nn.Linear(in_f, o)
    _patch_model_for_quant(root, layout.layers)
    res = root.load_state_dict(sd, strict=False, assign=True)
    assert not res.missing_keys and not res.unexpected_keys
    m = root.transformer_blocks[0].attn1.to_q
    assert m.weight.dtype == torch.int8 and not m.weight.requires_grad

    x = torch.randn(3, in_f).to(torch.bfloat16)
    ref = dequantize(scheme, q, {"weight_scale": scale}, torch.bfloat16)
    assert torch.equal(m(x), torch.nn.functional.linear(x, ref, bias))


def test_wrapped_transformer_rejects_int8_outside_linear_but_not_uint8(fake_inspect):
    built = nn.Module()
    built.norm = nn.LayerNorm(2)
    built.norm.register_buffer("ggml", torch.zeros(4, dtype=torch.uint8))  # GGUF-like: allowed
    ledger = _Ledger(built)
    SftQuantLoaderService("X.safetensors", dit_cpu_load=True).install(ledger)
    assert ledger.transformer() is built
    built.norm.register_buffer("q", torch.zeros(2, dtype=torch.int8))
    with pytest.raises(RuntimeError, match="outside Linear"):
        ledger.transformer()


@pytest.mark.parametrize("prefix", ["model.diffusion_model.", ""], ids=["prefixed", "bare"])
def test_load_connector_bf16_int8_convrot_and_f16(tmp_path, prefix):
    from safetensors.torch import save_file

    from engine.sft_quant.dequant import dequantize

    torch.manual_seed(32)
    c = prefix + "video_embeddings_connector."
    row_q, row_s = _int8_weight(4, 16), torch.rand(4, 1) * 0.01
    sca_q, sca_s = _int8_weight(4, 16), torch.tensor([0.02])
    rot_q, rot_s = _int8_weight(4, 256), torch.rand(4, 1) * 0.01
    f16 = torch.randn(3).to(torch.float16)
    path = tmp_path / "ic.safetensors"
    save_file(
        {
            prefix + "transformer_blocks.0.attn1.to_q.weight": torch.zeros(2, 2).to(torch.float8_e4m3fn),
            c + "row.weight": row_q,
            c + "row.weight_scale": row_s,
            c + "row.comfy_quant": _marker(False),
            c + "sca.weight": sca_q,
            c + "sca.weight_scale": sca_s,
            c + "sca.comfy_quant": _marker(False),
            c + "rot.weight": rot_q,
            c + "rot.weight_scale": rot_s,
            c + "rot.comfy_quant": _marker(True),
            c + "rot.bias": torch.randn(4).to(torch.bfloat16),
            c + "norm.bias": f16,
        },
        str(path),
    )
    out = load_connector_bf16(str(path))
    v = "video_embeddings_connector."
    assert set(out) == {v + "row.weight", v + "sca.weight", v + "rot.weight", v + "rot.bias", v + "norm.bias"}
    assert all(t.dtype == torch.bfloat16 for t in out.values())
    bf = torch.bfloat16
    assert torch.equal(out[v + "row.weight"], (row_q.float() * row_s).to(bf))
    assert torch.equal(out[v + "sca.weight"], (sca_q.float() * 0.02).to(bf))
    assert torch.equal(
        out[v + "rot.weight"],
        dequantize("int8_convrot", rot_q, {"weight_scale": rot_s}, bf),
    )
    assert not torch.equal(out[v + "rot.weight"], (rot_q.float() * rot_s).to(bf))
    assert torch.equal(out[v + "norm.bias"], f16.to(bf))


def test_load_connector_bf16_int8_without_marker_is_refused(tmp_path):
    """The connector goes through the same judge (layer_schemes) as inspect."""
    from safetensors.torch import save_file

    path = tmp_path / "nomark.safetensors"
    save_file(
        {
            "transformer_blocks.0.attn1.to_q.weight": torch.zeros(2, 2).to(torch.float8_e4m3fn),
            "video_embeddings_connector.w.weight": _int8_weight(4, 16),
            "video_embeddings_connector.w.weight_scale": torch.rand(4, 1),
        },
        str(path),
    )
    with pytest.raises(sft_quant_format.QuantFormatError):
        load_connector_bf16(str(path))


# ──────────────────────────────────────────────────────────────────────────────
# §3-168 C-3: w4a8
# ──────────────────────────────────────────────────────────────────────────────


def _w4a8_marker() -> torch.Tensor:
    import json

    conf = {"format": "asym_w4a8_int8", "group_size": 16, "convrot": True, "convrot_groupsize": 256}
    return torch.tensor(list(json.dumps(conf).encode()), dtype=torch.uint8)


def _w4a8_tensors(o: int, i: int, s_rel_u8: bool) -> dict[str, torch.Tensor]:
    """Raw file tensors of one w4a8 layer (leaf -> tensor)."""
    s_rel = (torch.rand(o, i // 16) * 150 + 20).to(torch.float8_e4m3fn)
    return {
        "weight": torch.randint(-128, 128, (o, i // 2), dtype=torch.int8),
        "weight_s_rel": s_rel.view(torch.uint8).clone() if s_rel_u8 else s_rel,
        "weight_s_channel": (torch.rand(o) * 0.01 + 1e-3).to(torch.float32),
        "weight_codebook": torch.sort(torch.randn(16) * 0.7).values.to(torch.float32),
        "comfy_quant": _w4a8_marker(),
    }


def _w4a8_aux(raw: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The normalized auxiliary tensors (U8 s_rel read as fp8)."""
    s_rel = raw["weight_s_rel"]
    if s_rel.dtype == torch.uint8:
        s_rel = s_rel.view(torch.float8_e4m3fn)
    return {
        "weight_s_rel": s_rel,
        "weight_s_channel": raw["weight_s_channel"],
        "weight_codebook": raw["weight_codebook"],
    }


@pytest.mark.parametrize("s_rel_u8", [False, True], ids=["s_rel_fp8", "s_rel_u8"])
def test_loader_w4a8_keeps_packed_weight_and_normalizes_aux(tmp_path, s_rel_u8):
    from safetensors.torch import save_file

    torch.manual_seed(50)
    o, i = 4, 256
    raw = _w4a8_tensors(o, i, s_rel_u8)
    path = tmp_path / "w4.safetensors"
    save_file({f"{_LIN}.{leaf}": t for leaf, t in raw.items()}, str(path))
    layout = _bare_layout(layers={_LIN: "w4a8"}, connector_keys=())
    sd = SftQuantStateDictLoader(str(path), layout).load("", sd_ops=sft_transformer_sd_ops("")).sd
    assert set(sd) == {f"{_LIN}.{leaf}" for leaf in raw if leaf != "comfy_quant"}
    w = sd[f"{_LIN}.weight"]
    assert w.dtype == torch.int8 and w.shape == (o, i // 2) and torch.equal(w, raw["weight"])
    s_rel = sd[f"{_LIN}.weight_s_rel"]
    assert s_rel.dtype == torch.float8_e4m3fn and s_rel.shape == (o, i // 16)
    assert torch.equal(s_rel.view(torch.uint8), raw["weight_s_rel"].view(torch.uint8))
    s_ch = sd[f"{_LIN}.weight_s_channel"]
    assert s_ch.dtype == torch.float32 and s_ch.shape == (o,)
    cb = sd[f"{_LIN}.weight_codebook"]
    assert cb.dtype == torch.float32 and cb.shape == (16,)
    assert torch.equal(cb, raw["weight_codebook"])


def test_loader_w4a8_to_skeleton_to_forward(tmp_path):
    """File -> loader -> patched meta skeleton (weight slot (o, i//2)) ->
    load_state_dict(assign) -> forward."""
    from safetensors.torch import save_file

    from engine.sft_quant.dequant import dequantize
    from engine.sft_quant.quant_service import _patch_model_for_quant

    torch.manual_seed(51)
    o, i = 8, 512
    raw = _w4a8_tensors(o, i, s_rel_u8=True)
    bias = torch.randn(o).to(torch.bfloat16)
    tensors = {f"{_LIN}.{leaf}": t for leaf, t in raw.items()}
    tensors[f"{_LIN}.bias"] = bias
    path = tmp_path / "w4e2e.safetensors"
    save_file(tensors, str(path))
    layout = _bare_layout(layers={_LIN: "w4a8"}, connector_keys=())
    sd = SftQuantStateDictLoader(str(path), layout).load("", sd_ops=sft_transformer_sd_ops("")).sd

    with torch.device("meta"):
        root = nn.Module()
        root.transformer_blocks = nn.ModuleList([nn.Module()])
        root.transformer_blocks[0].attn1 = nn.Module()
        root.transformer_blocks[0].attn1.to_q = nn.Linear(i, o)
    _patch_model_for_quant(root, layout.layers)
    assert root.transformer_blocks[0].attn1.to_q.weight.shape == (o, i // 2)
    res = root.load_state_dict(sd, strict=False, assign=True)
    assert not res.missing_keys and not res.unexpected_keys
    m = root.transformer_blocks[0].attn1.to_q
    assert m.weight.dtype == torch.int8 and not m.weight.requires_grad
    assert m.weight_s_rel.dtype == torch.float8_e4m3fn

    x = torch.randn(3, i).to(torch.bfloat16)
    ref = dequantize("w4a8", raw["weight"], _w4a8_aux(raw), torch.bfloat16)
    assert torch.equal(m(x), torch.nn.functional.linear(x, ref, bias))


@pytest.mark.parametrize("prefix", ["model.diffusion_model.", ""], ids=["prefixed", "bare"])
def test_load_connector_bf16_w4a8(tmp_path, prefix):
    """REDGraft's connector layers are w4a8: the two passes (auxiliary first,
    then one weight at a time) dequantize them with i = stored width x 2."""
    from safetensors.torch import save_file

    from engine.sft_quant.dequant import dequantize

    torch.manual_seed(52)
    c = prefix + "video_embeddings_connector."
    fp8_raw = _w4a8_tensors(4, 256, s_rel_u8=False)
    u8_raw = _w4a8_tensors(4, 512, s_rel_u8=True)
    tensors = {
        prefix + "transformer_blocks.0.attn1.to_q.weight": torch.zeros(2, 2).to(torch.float8_e4m3fn),
        c + "bias_only.bias": torch.randn(4).to(torch.bfloat16),
    }
    tensors.update({f"{c}a.{leaf}": t for leaf, t in fp8_raw.items()})
    tensors.update({f"{c}b.{leaf}": t for leaf, t in u8_raw.items()})
    tensors[c + "b.bias"] = torch.randn(4).to(torch.float16)
    path = tmp_path / "w4c.safetensors"
    save_file(tensors, str(path))

    out = load_connector_bf16(str(path))
    v = "video_embeddings_connector."
    assert set(out) == {v + "bias_only.bias", v + "a.weight", v + "b.weight", v + "b.bias"}
    assert all(t.dtype == torch.bfloat16 for t in out.values())
    bf = torch.bfloat16
    assert out[v + "a.weight"].shape == (4, 256) and out[v + "b.weight"].shape == (4, 512)
    assert torch.equal(
        out[v + "a.weight"], dequantize("w4a8", fp8_raw["weight"], _w4a8_aux(fp8_raw), bf)
    )
    assert torch.equal(
        out[v + "b.weight"], dequantize("w4a8", u8_raw["weight"], _w4a8_aux(u8_raw), bf)
    )
    assert torch.equal(out[v + "b.bias"], tensors[c + "b.bias"].to(bf))
