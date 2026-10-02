"""GGUF dequantisation against the gguf-py reference (§1-61).

Every type in ``SUPPORTED_GGML_TYPES`` must decode random blocks exactly as
``gguf.quants.dequantize`` does (CPU, pure torch — the fused Triton kernels are
only reached for CUDA tensors). Any other type is refused with ValueError, both
by ``dequantize_ggml_tensor`` and by ``GGUFQuantStateDictLoader.load``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
gguf = pytest.importorskip("gguf")
pytest.importorskip("ltx_core")

from gguf import GGMLQuantizationType as T  # noqa: E402
from gguf.constants import GGML_QUANT_SIZES  # noqa: E402

import engine.gguf.quant_service as qs  # noqa: E402

# Byte offsets of the f16 scale fields inside one block: overwritten with sane
# values so both sides decode the same finite numbers.
_F16_FIELDS = {"Q8_0": [0], "Q4_K": [0, 2], "Q5_K": [0, 2], "Q6_K": [208]}


def test_supported_types_match_gguf_numbering():
    assert qs.SUPPORTED_GGML_TYPES == {
        int(T.F32): "F32",
        int(T.F16): "F16",
        int(T.BF16): "BF16",
        int(T.Q8_0): "Q8_0",
        int(T.Q4_K): "Q4_K",
        int(T.Q5_K): "Q5_K",
        int(T.Q6_K): "Q6_K",
    }
    for t, name in qs.SUPPORTED_GGML_TYPES.items():
        assert T(t).name == name


@pytest.mark.parametrize("name", sorted(_F16_FIELDS))
@pytest.mark.parametrize("rows,cols", [(1, 256), (3, 256), (2, 512)])
def test_quantised_types_match_reference(name, rows, cols):
    qt = T[name]
    blk, tsize = GGML_QUANT_SIZES[qt]
    rng = np.random.default_rng(int(qt) * 1000 + rows * 10 + cols)
    nblk = rows * cols // blk
    raw = rng.integers(0, 256, size=(nblk, tsize), dtype=np.uint8)
    for o in _F16_FIELDS[name]:
        vals = rng.uniform(0.01, 1.0, size=nblk).astype(np.float16)
        raw[:, o:o + 2] = vals.view(np.uint8).reshape(nblk, 2)
    ref = gguf.quants.dequantize(raw.reshape(rows, -1), qt).astype(np.float32)
    got = qs.dequantize_ggml_tensor(
        torch.from_numpy(raw.reshape(-1).copy()), int(qt), (rows, cols), torch.float32,
    ).numpy()
    assert got.shape == ref.shape
    assert np.array_equal(got, ref)


@pytest.mark.parametrize("name", ["F32", "F16", "BF16"])
def test_float_types_pass_through(name):
    values = torch.linspace(-2.0, 2.0, 12).reshape(3, 4)
    dtype = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16}[name]
    raw = values.to(dtype).contiguous().view(torch.uint8).reshape(-1)
    got = qs.dequantize_ggml_tensor(raw, int(T[name]), (3, 4), torch.float32)
    assert torch.equal(got, values.to(dtype).to(torch.float32))


@pytest.mark.parametrize("ggml_type", [2, 3, 6, 7, 10, 11, 20, 22, 23])
def test_unsupported_types_are_refused(ggml_type):
    raw = torch.zeros(4096, dtype=torch.uint8)
    with pytest.raises(ValueError, match=f"unsupported GGML tensor type {ggml_type};"):
        qs.dequantize_ggml_tensor(raw, ggml_type, (1, 256), torch.float32)


def test_state_dict_loader_refuses_unsupported_tensor(monkeypatch):
    tensors = [
        SimpleNamespace(
            name="blocks.0.ok.weight", tensor_type=T.Q4_K,
            shape=np.array([256, 1]), data=np.zeros((1, 144), dtype=np.uint8),
        ),
        SimpleNamespace(
            name="blocks.0.bad.weight", tensor_type=T.Q3_K,
            shape=np.array([256, 1]), data=np.zeros((1, 110), dtype=np.uint8),
        ),
    ]
    monkeypatch.setattr(gguf, "GGUFReader", lambda _path, mode="r": SimpleNamespace(tensors=tensors))
    loader = qs.GGUFQuantStateDictLoader("C:/models/user.gguf")
    with pytest.raises(ValueError, match=r"user\.gguf: tensor 'blocks\.0\.bad\.weight': unsupported GGML tensor type 11;"):
        loader.load("unused")
