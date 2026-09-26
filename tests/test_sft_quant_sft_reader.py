"""§3-167: ``engine.sft_quant.sft_reader`` — seek + readinto, never mmap (CPU only).

A tiny safetensors file is written by hand (so the data section can be laid
out in an order different from the header's key order) and read back; every
tensor must come back byte-identical, in file order, without ``mmap.mmap`` or
``safetensors.safe_open`` being usable at all.

Run with ``.venv-engine`` and ``--noconftest`` (torch lives only there; the app
venv skips the whole module, like test_ic_lora_forward.py).
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import sft_quant_format  # noqa: E402
from engine.sft_quant import sft_reader  # noqa: E402
from engine.sft_quant.sft_reader import read_tensors  # noqa: E402

_DTYPE_NAMES = {
    torch.bfloat16: "BF16",
    torch.float32: "F32",
    torch.float16: "F16",
    torch.float8_e4m3fn: "F8_E4M3",
    torch.float8_e5m2: "F8_E5M2",
    torch.uint8: "U8",
    torch.int32: "I32",
}


def _raw(t: torch.Tensor) -> bytes:
    return t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def write_sft(
    path: Path,
    tensors: dict[str, torch.Tensor],
    *,
    data_order: list[str] | None = None,
    metadata: dict[str, str] | None = None,
) -> Path:
    """Write a safetensors file whose data section follows ``data_order``."""
    order = data_order or list(tensors)
    header: dict = {}
    blobs = []
    offset = 0
    for name in order:
        raw = _raw(tensors[name])
        header[name] = {
            "dtype": _DTYPE_NAMES[tensors[name].dtype],
            "shape": list(tensors[name].shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        blobs.append(raw)
        offset += len(raw)
    # Header key order = the dict's insertion order, deliberately NOT data order.
    doc = {k: header[k] for k in tensors}
    if metadata:
        doc["__metadata__"] = metadata
    blob = json.dumps(doc).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"".join(blobs))
    return path


def _sample() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {
        "a.weight": (torch.randn(6, 4) * 3).to(torch.float8_e4m3fn),
        "a.weight_scale": torch.tensor(0.125, dtype=torch.float32),
        "a.bias": torch.randn(6).to(torch.bfloat16),
        "e5.weight": (torch.randn(3, 4) * 4).to(torch.float8_e5m2),
        "norm.weight": torch.randn(5, dtype=torch.float32),
        "half": torch.randn(3, 2).to(torch.float16),
        "a.comfy_quant": torch.tensor(list(b'{"format": "float8_e4m3fn"}'), dtype=torch.uint8),
    }


def _same_bytes(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and _raw(a) == _raw(b)


def test_round_trip_is_byte_identical(tmp_path):
    src = _sample()
    path = write_sft(tmp_path / "m.safetensors", src)
    got = dict(read_tensors(str(path), list(src)))
    assert set(got) == set(src)
    for key, t in src.items():
        assert _same_bytes(got[key], t), key
    assert got["a.weight_scale"].dim() == 0  # 0-dim survives the 1-D detour
    assert got["a.weight_scale"].item() == 0.125


def test_tensors_own_their_bytes(tmp_path):
    src = _sample()
    path = write_sft(tmp_path / "m.safetensors", src)
    got = dict(read_tensors(str(path), ["norm.weight"]))
    path.write_bytes(b"")  # the file changing underneath must not matter
    assert _same_bytes(got["norm.weight"], src["norm.weight"])


def test_yields_in_data_offset_order(tmp_path):
    src = _sample()
    data_order = ["half", "norm.weight", "e5.weight", "a.bias", "a.comfy_quant", "a.weight", "a.weight_scale"]
    path = write_sft(tmp_path / "m.safetensors", src, data_order=data_order)
    asked = list(reversed(list(src)))  # neither header order nor data order
    assert [k for k, _ in read_tensors(str(path), asked)] == data_order


def test_no_mmap_and_no_safe_open(tmp_path, monkeypatch):
    import mmap

    import numpy
    import safetensors
    import safetensors.torch

    def _forbidden(*_a, **_k):
        raise AssertionError("mmap / safe_open / load_file / memmap must not be used")

    monkeypatch.setattr(mmap, "mmap", _forbidden)
    monkeypatch.setattr(numpy, "memmap", _forbidden)
    monkeypatch.setattr(safetensors, "safe_open", _forbidden)
    monkeypatch.setattr(safetensors.torch, "safe_open", _forbidden)
    monkeypatch.setattr(safetensors.torch, "load_file", _forbidden)
    src = _sample()
    path = write_sft(tmp_path / "m.safetensors", src)
    got = dict(read_tensors(str(path), list(src)))
    for key, t in src.items():
        assert _same_bytes(got[key], t), key


def test_short_read_raises(tmp_path):
    src = _sample()
    path = write_sft(tmp_path / "m.safetensors", src, data_order=list(src))
    header = sft_quant_format.read_header(path)
    # Truncate inside the LAST tensor after the header was validated.
    path.write_bytes(path.read_bytes()[:-3])
    with pytest.raises(OSError, match="short read"):
        list(read_tensors(str(path), list(src), header))


def test_unsupported_dtype_raises(tmp_path):
    path = write_sft(tmp_path / "m.safetensors", {"i": torch.arange(4, dtype=torch.int32)})
    with pytest.raises(ValueError, match="I32"):
        list(read_tensors(str(path), ["i"]))


def test_unknown_key_raises(tmp_path):
    path = write_sft(tmp_path / "m.safetensors", _sample())
    with pytest.raises(KeyError):
        list(read_tensors(str(path), ["nope"]))


class _CountingFile:
    def __init__(self, f, counter: dict) -> None:
        self._f = f
        self._counter = counter

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._f.close()
        return False

    def seek(self, *a):
        return self._f.seek(*a)

    def readinto(self, b):
        self._counter["n"] += 1
        return self._f.readinto(b)


def test_loader_skips_keys_that_sd_ops_drops(tmp_path, monkeypatch):
    """SftQuantStateDictLoader: keys mapped to None by sd_ops are never read
    (counted at readinto), nor are comfy_quant / input_scale / connectors."""
    pytest.importorskip("ltx_core")
    import types

    from ltx_core.model.transformer.model_configurator import LTXV_MODEL_COMFY_RENAMING_MAP

    from engine.sft_quant.quant_service import SftQuantStateDictLoader

    p = "model.diffusion_model."
    src = {
        p + "blk.lin.weight": torch.randn(4, 4).to(torch.float8_e4m3fn),
        p + "blk.lin.weight_scale": torch.tensor(0.5, dtype=torch.float32),
        p + "blk.lin.input_scale": torch.tensor(1.0, dtype=torch.float32),
        p + "blk.lin.comfy_quant": torch.tensor(list(b'{"format": "float8_e4m3fn"}'), dtype=torch.uint8),
        p + "blk.lin.bias": torch.randn(4).to(torch.bfloat16),
        p + "norm.weight": torch.randn(4, dtype=torch.float32),
        p + "video_embeddings_connector.w": torch.randn(3).to(torch.bfloat16),
        "vae.decoder.w": torch.randn(8).to(torch.bfloat16),
        "vocoder.w": torch.randn(8).to(torch.bfloat16),
    }
    path = write_sft(tmp_path / "m.safetensors", src)
    layout = types.SimpleNamespace(
        config={"transformer": {}},
        layers={"blk.lin": "fp8_scaled"},
        prefix=p,
        connector_keys=(p + "video_embeddings_connector.w",),
    )
    counter = {"n": 0}
    real_open = open
    monkeypatch.setattr(
        sft_reader, "open", lambda *a, **k: _CountingFile(real_open(*a, **k), counter), raising=False
    )

    loader = SftQuantStateDictLoader(str(path), layout)
    assert loader.metadata("") == {"transformer": {}}
    sd = loader.load([""], sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP, device=torch.device("cpu"))

    assert set(sd.sd) == {"blk.lin.weight", "blk.lin.weight_scale", "blk.lin.bias", "norm.weight"}
    assert counter["n"] == 4  # one readinto per loaded tensor, nothing else touched
    assert sd.sd["blk.lin.weight"].dtype == torch.float8_e4m3fn
    assert sd.sd["blk.lin.weight_scale"].dtype == torch.float32  # scale stays f32
    assert sd.sd["norm.weight"].dtype == torch.bfloat16  # other F32 -> bf16
    assert torch.equal(sd.sd["norm.weight"], src[p + "norm.weight"].to(torch.bfloat16))
    assert _same_bytes(sd.sd["blk.lin.weight"], src[p + "blk.lin.weight"])
    assert sd.device == torch.device("cpu")
    assert sd.size == sum(t.nbytes for t in sd.sd.values())
    assert sd.dtype == {torch.float8_e4m3fn, torch.float32, torch.bfloat16}
