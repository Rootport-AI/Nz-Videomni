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
    flavor="scaled",
    config={"transformer": {"num_layers": 48}},
    model_version="2.3.0",
    prefix="model.diffusion_model.",
    scaled_layers=frozenset({"transformer_blocks.2.attn1.to_q"}),
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
        flavor="scaled",
        config={"transformer": {}},
        model_version="2.5.0",
        prefix="",
        scaled_layers=frozenset({"transformer_blocks.0.attn1.to_q"}),
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
