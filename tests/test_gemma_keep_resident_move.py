"""Keep-resident (StateDictRegistry) CPU-build -> compute-device move helpers.

Covers ``engine.gemma.gguf_quant_service``'s replacement for the old
"skip tensors that are already on CPU" proxy, which collapsed under a
StateDictRegistry (everything is CPU-built there, so the whole Gemma model was
skipped and ran on the CPU). The new rule is DiT-style: move every CPU leaf
EXCEPT the sub-trees of the decoder layers the offload service streams.

Almost everything here is device-free: the selection rule is a pure function
(``_leaf_tensors_to_move``) precisely so it can be asserted without a GPU. Only
one test performs a real move, and it is skipped when CUDA is absent.

Run with ``.venv-engine`` and ``--noconftest`` (the app conftest builds a FastAPI
app the engine venv does not have).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from engine.gemma.gguf_quant_service import (  # noqa: E402
    _assert_no_stray_cpu_leaves,
    _leaf_tensors_to_move,
    _move_leaf_tensors_to_device,
)

N_LAYERS = 6
DIM = 4


class _Leafy(nn.Module):
    """A stand-in decoder layer / norm / connector: one param + one buffer.

    The buffer models the GGUF-quantized Linear weights, which are registered as
    BUFFERS (not parameters) — the case a ``parameters()``-only walk would miss.
    """

    def __init__(self, device: str = "cpu") -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(DIM, device=device))
        self.register_buffer("quant", torch.zeros(DIM, dtype=torch.uint8, device=device))


class _FakeGemma(nn.Module):
    """Minimal shape-alike of the built text encoder.

    ``layers`` = the streamed decoder stack (invariant L), ``norm`` = the
    compute-device anchor (N), ``connector`` = the LTX-side bf16 tensors (C),
    ``scale`` = a param attached directly to the root module (the DiT precedent's
    ``scale_shift_table`` case), ``embed_tokens`` = the held-back embedding, still
    on meta at move time (E).
    """

    def __init__(self, non_layer_device: str = "cpu", embed_device: str = "meta") -> None:
        super().__init__()
        self.layers = nn.ModuleList(_Leafy() for _ in range(N_LAYERS))
        self.norm = _Leafy(non_layer_device)
        self.connector = _Leafy(non_layer_device)
        self.scale = nn.Parameter(torch.zeros(2, device=non_layer_device))
        self.embed_tokens = nn.Embedding(8, DIM)
        self.embed_tokens.weight = nn.Parameter(
            torch.empty(8, DIM, device=embed_device), requires_grad=False
        )


def _selected_names(model: nn.Module, excluded=()) -> list[str]:
    id_to_name = {id(m): n for n, m in model.named_modules()}
    return sorted(
        f"{id_to_name[id(mod)]}.{name}".lstrip(".")
        for mod, _kind, name in _leaf_tensors_to_move(model, excluded)
    )


# ── selection rule ───────────────────────────────────────────────────────────


def test_excluded_layers_are_not_selected() -> None:
    model = _FakeGemma()
    names = _selected_names(model, tuple(model.layers))
    # N / C / root-attached leaves move; the streamed layers and the meta
    # embedding do not.
    assert names == [
        "connector.quant",
        "connector.weight",
        "norm.quant",
        "norm.weight",
        "scale",
    ]
    assert not any(n.startswith("layers.") for n in names)
    assert not any("embed_tokens" in n for n in names)


def test_empty_exclusion_selects_every_cpu_leaf() -> None:
    """Non-regression proxy for the old non-offload branch (moved everything)."""
    model = _FakeGemma()
    names = _selected_names(model)
    assert len(names) == 5 + 2 * N_LAYERS
    for i in range(N_LAYERS):
        assert f"layers.{i}.weight" in names
        assert f"layers.{i}.quant" in names
    # meta stays out regardless of the exclusion set.
    assert not any("embed_tokens" in n for n in names)


# ── post-condition check ─────────────────────────────────────────────────────


def test_assert_reports_stray_cpu_leaves_by_name() -> None:
    model = _FakeGemma()  # nothing moved yet -> N/C/root are still on CPU
    with pytest.raises(RuntimeError) as exc:
        _assert_no_stray_cpu_leaves(model, tuple(model.layers))
    msg = str(exc.value)
    for expected in ("norm.weight", "norm.quant", "connector.weight", "scale"):
        assert expected in msg
    assert "layers.0" not in msg


def test_assert_passes_when_only_layers_and_embed_are_on_cpu() -> None:
    # Post-move shape: everything outside the streamed layers is off CPU, and the
    # CPU token embedding (Lever 3) is the one allowed exception.
    model = _FakeGemma(non_layer_device="meta", embed_device="cpu")
    _assert_no_stray_cpu_leaves(model, tuple(model.layers))


# ── real move (CUDA only) ────────────────────────────────────────────────────


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_move_leaf_tensors_to_device_cuda() -> None:
    model = _FakeGemma()
    device = torch.device("cuda:0")
    excluded = tuple(model.layers)
    kept_param = model.norm.weight  # object identity must survive (weight tying)
    cached_buf = model.connector.quant  # the "cached" tensor must not be mutated

    moved = _move_leaf_tensors_to_device(model, device, excluded)

    assert moved == 5
    for layer in model.layers:
        assert layer.weight.device.type == "cpu"
        assert layer.quant.device.type == "cpu"
    assert model.norm.weight.device.type == "cuda"
    assert model.norm.quant.device.type == "cuda"
    assert model.connector.weight.device.type == "cuda"
    assert model.scale.device.type == "cuda"
    # param: same Parameter object, re-pointed .data (tie-preserving).
    assert model.norm.weight is kept_param
    # buffer: slot re-bound out-of-place; the original object is untouched.
    assert model.connector.quant is not cached_buf
    assert cached_buf.device.type == "cpu"
    # meta is never dragged along.
    assert model.embed_tokens.weight.device.type == "meta"

    _assert_no_stray_cpu_leaves(model, excluded)
