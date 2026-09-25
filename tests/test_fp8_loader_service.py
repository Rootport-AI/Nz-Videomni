"""§3-167: ``Fp8LoaderService.install`` against a mock ledger (CPU only).

Pins the three install steps without any weights or GPU:

  * the policy is replaced WHOLESALE — ``sd_ops`` is None (fp8_cast's
    TRANSFORMER_LINEAR_DOWNCAST_MAP would push biases and the bf16 blocks down
    to fp8) and fp8_cast's ``UPCAST_DURING_INFERENCE`` is gone, leaving only
    ``fp8_linear``;
  * the builder's loader is an ``Fp8StateDictLoader`` sharing the ONE Layout
    ``inspect`` returned;
  * errors are never swallowed (inspect's refusal, ``dit_cpu_load=0``, a stray
    fp8 tensor outside a Linear, and the pipeline's ``_install_fp8``);
  * the pipeline accepts exactly one transformer source.

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

import sft_fp8_format  # noqa: E402
from engine.fp8.quant_service import Fp8LoaderService, Fp8StateDictLoader  # noqa: E402

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

    monkeypatch.setattr(sft_fp8_format, "inspect", _inspect)
    return seen


def test_install_replaces_loader_and_policy(fake_inspect):
    ledger = _Ledger()
    Fp8LoaderService("X.safetensors", dit_cpu_load=True).install(ledger)

    assert fake_inspect == ["X.safetensors"]  # inspected exactly once
    loader = ledger.transformer_builder.model_loader
    assert isinstance(loader, Fp8StateDictLoader)
    assert loader.path == "X.safetensors" and loader.layout is _LAYOUT
    assert loader.metadata("") == _LAYOUT.config

    policy = ledger.quantization
    assert policy.sd_ops is None
    assert [op.name for op in policy.module_ops] == ["fp8_linear"]  # fp8_cast is gone


def test_module_op_registers_scale_from_the_shared_layout(fake_inspect):
    from ltx_core.model.transformer.model import LTXModel

    ledger = _Ledger()
    Fp8LoaderService("X.safetensors", dit_cpu_load=True).install(ledger)
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

    monkeypatch.setattr(sft_fp8_format, "inspect", _never)
    ledger = _Ledger()
    before = (ledger.transformer_builder, ledger.quantization)
    with pytest.raises(RuntimeError, match="dit_cpu_load"):
        Fp8LoaderService("X.safetensors", dit_cpu_load=False).install(ledger)
    assert (ledger.transformer_builder, ledger.quantization) == before


def test_inspect_refusal_propagates(monkeypatch):
    def _refuse(*_a, **_k):
        raise sft_fp8_format.Fp8FormatError("流儀 F8_E5M2 は扱えません")

    monkeypatch.setattr(sft_fp8_format, "inspect", _refuse)
    ledger = _Ledger()
    with pytest.raises(sft_fp8_format.Fp8FormatError, match="F8_E5M2"):
        Fp8LoaderService("X.safetensors", dit_cpu_load=True).install(ledger)
    assert ledger.quantization.sd_ops is not None  # still fp8_cast: untouched
    assert ledger.transformer_builder.model_loader is None


def test_pipeline_install_fp8_does_not_swallow(monkeypatch):
    from engine.pipeline.fast_video_pipeline import LTXFastVideoPipeline

    def _refuse(*_a, **_k):
        raise sft_fp8_format.Fp8FormatError("per-row weight_scale")

    monkeypatch.setattr(sft_fp8_format, "inspect", _refuse)
    fake_self = types.SimpleNamespace(
        _dit_cpu_load=True,
        _ic_loras=[],
        pipeline=types.SimpleNamespace(model_ledger=_Ledger()),
    )
    with pytest.raises(sft_fp8_format.Fp8FormatError, match="per-row"):
        LTXFastVideoPipeline._install_fp8(fake_self, "X.safetensors")

    fake_self._dit_cpu_load = False
    with pytest.raises(RuntimeError, match="dit_cpu_load"):
        LTXFastVideoPipeline._install_fp8(fake_self, "X.safetensors")


def test_wrapped_transformer_detaches_and_passes_clean_model(fake_inspect, monkeypatch):
    built = nn.Module()
    built.lin = nn.Linear(2, 2)
    built.lin.weight = nn.Parameter(torch.zeros(2, 2).to(torch.float8_e4m3fn), requires_grad=False)
    ledger = _Ledger(built)
    detached = []
    import engine.gguf.ic_lora_common as ic

    monkeypatch.setattr(ic, "detach_ic_loras", lambda t: detached.append(t) or 0)
    Fp8LoaderService("X.safetensors", dit_cpu_load=True, ic_loras_provider=lambda: []).install(ledger)
    assert ledger.transformer() is built
    assert ledger.calls == 1 and detached == [built]


def test_wrapped_transformer_rejects_fp8_outside_linear(fake_inspect):
    built = nn.Module()
    built.norm = nn.LayerNorm(2)
    built.norm.weight = nn.Parameter(torch.ones(2).to(torch.float8_e4m3fn), requires_grad=False)
    ledger = _Ledger(built)
    Fp8LoaderService("X.safetensors", dit_cpu_load=True).install(ledger)
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



def test_gemma_connectors_from_fp8_safetensors(tmp_path):
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
    assert torch.equal(out[p + "video_embeddings_connector.w"], v)
    assert torch.equal(out[p + "audio_embeddings_connector.b"], a.to(torch.bfloat16))

    loader._connector_gguf_path = str(tmp_path / "t.bin")
    with pytest.raises(RuntimeError, match="expected a .gguf or .safetensors"):
        loader._load_gguf_connectors(torch.device("cpu"))
