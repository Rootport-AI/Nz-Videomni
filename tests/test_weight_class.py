"""§1-31: weight class of a transformer file (services/weight_class.py) and the
GGUF tensor-info reader it uses (services/gguf_kv.read_gguf_tensor_types).

Rules' source of truth: Docs/COMFORT_LIMIT_TABLE.md §1. Synthetic headers only
(no real multi-GB weights): GGUFs from conftest.write_gguf_with_tensors,
safetensors from the builders of tests/test_sft_quant_format.py.
"""

from __future__ import annotations

import os
import struct

import pytest

from conftest import write_gguf_with_tensors
from services import weight_class
from services.gguf_kv import GgufParseError, read_gguf_tensor_types
from services.weight_class import (
    WEIGHT_CLASSES,
    classify_gguf_tensors,
    classify_sft_layers,
    classify_transformer,
)
from test_sft_quant_format import _int8_model, _model, _w4a8_model, _write

Q4_K, Q5_K, Q6_K, Q8_0, F16, F32 = 12, 13, 14, 8, 1, 0


def _blocks(counts: dict[int, int]) -> dict[str, tuple[int, int]]:
    """``{ggml type: n}`` -> that many 2-D transformer-block ``.weight`` tensors."""
    tensors = {}
    i = 0
    for ggml_type, n in counts.items():
        for _ in range(n):
            tensors[f"transformer_blocks.{i}.attn1.to_q.weight"] = (2, ggml_type)
            i += 1
    return tensors


@pytest.fixture(autouse=True)
def _fresh_cache():
    weight_class._classify_cached.cache_clear()
    yield
    weight_class._classify_cached.cache_clear()


def test_weight_classes_are_the_three_fixed_names():
    assert WEIGHT_CLASSES == ("4bit", "8bit", "q6k")


# --- GGUF (pure function) ----------------------------------------------------- #


def test_gguf_q4_k_m_mix_is_4bit():
    """The official Q4_K_M: Q4_K 1242 / Q6_K 322 / Q5_K 68 -> majority Q4_K."""
    assert classify_gguf_tensors(_blocks({Q4_K: 1242, Q6_K: 322, Q5_K: 68})) == "4bit"


def test_gguf_q6_k_is_q6k():
    assert classify_gguf_tensors(_blocks({Q6_K: 1632})) == "q6k"


@pytest.mark.parametrize("ggml_type", [Q8_0, F16, 30])
def test_gguf_other_majority_is_none(ggml_type):
    assert classify_gguf_tensors(_blocks({ggml_type: 10, Q4_K: 2})) is None


def test_gguf_without_block_weights_is_none():
    assert classify_gguf_tensors({}) is None


def test_gguf_only_2d_block_weights_are_counted():
    """1-D tensors, ``.bias`` and tensors outside ``transformer_blocks.<n>.`` do
    not vote, however many there are."""
    tensors = _blocks({Q6_K: 3})
    for i in range(10):
        tensors[f"transformer_blocks.{i}.norm.weight"] = (1, Q4_K)
        tensors[f"transformer_blocks.{i}.attn1.to_q.bias"] = (2, Q4_K)
        tensors[f"patchify_proj{i}.weight"] = (2, Q4_K)
    assert classify_gguf_tensors(tensors) == "q6k"


# --- safetensors (pure function) ---------------------------------------------- #


@pytest.mark.parametrize("scheme", ["fp8", "fp8_scaled", "int8", "int8_convrot"])
def test_sft_8bit_schemes(scheme):
    assert classify_sft_layers({"a": scheme, "b": scheme}) == "8bit"


def test_sft_w4a8_is_4bit():
    assert classify_sft_layers({"a": "w4a8"}) == "4bit"


def test_sft_majority_wins():
    """REDGraft mix (int8_convrot 831 + w4a8 513) is 8bit; a w4a8-major file
    with some int8 layers is 4bit."""
    redgraft = {f"a{i}": "int8_convrot" for i in range(831)}
    redgraft.update({f"b{i}": "w4a8" for i in range(513)})
    assert classify_sft_layers(redgraft) == "8bit"
    w4a8_major = {f"a{i}": "w4a8" for i in range(10)}
    w4a8_major.update({f"b{i}": "int8" for i in range(3)})
    assert classify_sft_layers(w4a8_major) == "4bit"


def test_sft_tie_goes_to_8bit():
    assert classify_sft_layers({"a": "w4a8", "b": "fp8"}) == "8bit"


def test_sft_empty_is_none():
    assert classify_sft_layers({}) is None


# --- read_gguf_tensor_types ---------------------------------------------------- #


def test_read_gguf_tensor_types_skips_kv_and_reads_tensors(tmp_path):
    path = write_gguf_with_tensors(
        tmp_path / "t.gguf",
        {"transformer_blocks.0.attn1.to_q.weight": (2, Q4_K), "x.bias": (1, F32)},
        **{"general.architecture": "ltxv", "model_version": "2.3.0"},
    )
    assert read_gguf_tensor_types(path) == {
        "transformer_blocks.0.attn1.to_q.weight": (2, Q4_K),
        "x.bias": (1, F32),
    }


def test_read_gguf_tensor_types_refuses_a_non_gguf(tmp_path):
    path = tmp_path / "fake.gguf"
    path.write_bytes(b"NOPE" + b"\0" * 32)
    with pytest.raises(GgufParseError, match="bad magic"):
        read_gguf_tensor_types(path)


def test_read_gguf_tensor_types_refuses_a_truncated_file(tmp_path):
    path = write_gguf_with_tensors(tmp_path / "t.gguf", _blocks({Q4_K: 3}))
    path.write_bytes(path.read_bytes()[:-5])
    with pytest.raises(GgufParseError, match="unexpected end of file"):
        read_gguf_tensor_types(path)


def test_read_gguf_tensor_types_refuses_too_many_dimensions(tmp_path):
    path = write_gguf_with_tensors(tmp_path / "t.gguf", {"w": (5, Q4_K)})
    with pytest.raises(GgufParseError, match="5 dimensions"):
        read_gguf_tensor_types(path)


def test_read_gguf_tensor_types_refuses_an_oversized_tensor_count(tmp_path):
    path = tmp_path / "t.gguf"
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, (1 << 16) + 1, 0))
    with pytest.raises(GgufParseError, match="tensor_count"):
        read_gguf_tensor_types(path)


# --- classify_transformer (files) --------------------------------------------- #


def test_classify_gguf_file(tmp_path):
    path = write_gguf_with_tensors(
        tmp_path / "q4.gguf", _blocks({Q4_K: 5, Q6_K: 2}), general_architecture="ltxv"
    )
    assert classify_transformer(path) == "4bit"


def test_classify_fp8_safetensors_is_8bit(tmp_path):
    path = _write(tmp_path / "fp8.safetensors", *_model("scaled"))
    assert classify_transformer(path) == "8bit"


def test_classify_int8_safetensors_is_8bit(tmp_path):
    path = _write(tmp_path / "int8.safetensors", *_int8_model())
    assert classify_transformer(path) == "8bit"


def test_classify_w4a8_safetensors_is_4bit(tmp_path):
    path = _write(tmp_path / "w4a8.safetensors", *_w4a8_model())
    assert classify_transformer(path) == "4bit"


def test_classify_bf16_only_safetensors_is_none(tmp_path):
    """No quantized layer: ``inspect`` refuses -> None, not an exception."""
    path = _write(tmp_path / "bf16.safetensors", *_model("plain", fp8_blocks=()))
    assert classify_transformer(path) is None


def test_classify_missing_and_broken_files_are_none(tmp_path):
    assert classify_transformer(tmp_path / "absent.gguf") is None
    broken = tmp_path / "broken.gguf"
    broken.write_bytes(b"GGUF\x03")
    assert classify_transformer(broken) is None
    other = tmp_path / "weights.bin"
    other.write_bytes(b"\0" * 16)
    assert classify_transformer(other) is None


def test_cache_is_invalidated_by_a_new_mtime(tmp_path):
    """The cache key is (path, size, mtime): rewriting the file under the same
    name (same size here) re-classifies it once the mtime moves. The mtime is
    moved explicitly -- two quick writes on Windows can share one."""
    path = write_gguf_with_tensors(tmp_path / "w.gguf", _blocks({Q4_K: 4}))
    assert classify_transformer(path) == "4bit"
    before = path.stat().st_mtime_ns
    write_gguf_with_tensors(path, _blocks({Q6_K: 4}))
    os.utime(path, ns=(before, before))
    assert classify_transformer(path) == "4bit"  # same key -> cached answer
    os.utime(path, ns=(before + 10_000_000_000, before + 10_000_000_000))
    assert classify_transformer(path) == "q6k"
