"""LTX 2.3 pipeline: a failed GGUF install fails the load (§1-51).

Neither GGUF install has anything to fall back to — the pipeline is built with
``gemma_root=None`` and, in production, an empty ``checkpoint_path`` — so the
exception must leave ``_install_gemma_gguf`` / ``_install_gguf`` (and with it
the worker's ``load``). Unbound methods are called on a fake ``self``.
"""

from __future__ import annotations

import types

import pytest

pytest.importorskip("torch")
pytest.importorskip("ltx_pipelines")

import engine.gemma.gguf_quant_service as gemma_gqs  # noqa: E402
from engine.pipeline.fast_video_pipeline import LTXFastVideoPipeline  # noqa: E402


def test_gemma_gguf_install_failure_is_raised(monkeypatch):
    def _boom(self, _ledger):
        raise RuntimeError("corrupt gemma gguf")

    monkeypatch.setattr(gemma_gqs.GemmaGGUFQuantLoaderService, "install", _boom)
    fake_self = types.SimpleNamespace(pipeline=types.SimpleNamespace(model_ledger=types.SimpleNamespace()))
    with pytest.raises(RuntimeError, match="corrupt gemma gguf"):
        LTXFastVideoPipeline._install_gemma_gguf(fake_self, "missing.gguf", gemma_tokenizer_root=None)
    assert not hasattr(fake_self, "_gemma_gguf_service")


def test_transformer_gguf_install_failure_is_raised_without_lora(tmp_path):
    fake_self = types.SimpleNamespace(
        _ic_loras=[],
        pipeline=types.SimpleNamespace(model_ledger=types.SimpleNamespace()),
    )
    with pytest.raises(FileNotFoundError, match="missing.gguf"):
        LTXFastVideoPipeline._install_gguf(
            fake_self, str(tmp_path / "missing.gguf"), per_layer_quant=True, ic_loras=[],
        )
    assert not hasattr(fake_self, "_gguf_service")
