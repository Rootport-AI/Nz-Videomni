"""Unit tests for services/gguf_kv.py -- a synthetic-bytes test suite (no real
GGUF file needed): every case builds a minimal valid-or-broken GGUF header by
hand and feeds it to read_gguf_kv. See MULTI_ENGINE_DESIGN.md §2.5 for why
this module exists (no ``gguf`` package in the app venv) and what it is
scoped to (KV header only, types 0-12, wanted-only capture).
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from services.gguf_kv import GgufParseError, read_gguf_kv

MAGIC = b"GGUF"

_TYPE_UINT32 = 4
_TYPE_STRING = 8
_TYPE_ARRAY = 9


def _u32(v: int) -> bytes:
    return struct.pack("<I", v)


def _u64(v: int) -> bytes:
    return struct.pack("<Q", v)


def _gguf_string(s: str) -> bytes:
    b = s.encode("utf-8")
    return _u64(len(b)) + b


def _kv_string(key: str, value: str) -> bytes:
    return _gguf_string(key) + _u32(_TYPE_STRING) + _gguf_string(value)


def _kv_uint32(key: str, value: int) -> bytes:
    return _gguf_string(key) + _u32(_TYPE_UINT32) + _u32(value)


def _kv_string_array(key: str, values: list[str]) -> bytes:
    body = _u32(_TYPE_STRING) + _u64(len(values))
    for v in values:
        body += _gguf_string(v)
    return _gguf_string(key) + _u32(_TYPE_ARRAY) + body


def _header(*, version: int, kv_count: int, tensor_count: int = 0) -> bytes:
    return MAGIC + _u32(version) + _u64(tensor_count) + _u64(kv_count)


def _write(tmp_path: Path, name: str, data: bytes) -> Path:
    p = tmp_path / name
    p.write_bytes(data)
    return p


# --------------------------------------------------------------------- happy


@pytest.mark.parametrize("version", [2, 3])
def test_reads_wanted_string_keys(tmp_path, version):
    kvs = [
        _kv_string("general.architecture", "ltxv"),
        _kv_uint32("some.unrelated.count", 42),
        _kv_string("model_version", "2.3.0"),
        _kv_string("license", "MIT" * 100),  # exercises string skip too
    ]
    data = _header(version=version, kv_count=len(kvs)) + b"".join(kvs)
    p = _write(tmp_path, "ok.gguf", data)

    result = read_gguf_kv(p, {"general.architecture", "model_version"})

    assert result == {"general.architecture": "ltxv", "model_version": "2.3.0"}
    assert "some.unrelated.count" not in result
    assert "license" not in result


def test_wanted_key_absent_is_not_an_error(tmp_path):
    kvs = [_kv_string("general.architecture", "gemma3")]
    data = _header(version=3, kv_count=len(kvs)) + b"".join(kvs)
    p = _write(tmp_path, "no_version.gguf", data)

    result = read_gguf_kv(p, {"general.architecture", "model_version"})

    assert result == {"general.architecture": "gemma3"}
    assert "model_version" not in result


def test_numeric_wanted_value_is_stringified(tmp_path):
    kvs = [_kv_uint32("some.count", 7)]
    data = _header(version=3, kv_count=len(kvs)) + b"".join(kvs)
    p = _write(tmp_path, "numeric.gguf", data)

    result = read_gguf_kv(p, {"some.count"})

    assert result == {"some.count": "7"}


def test_string_array_spanning_multiple_elements_is_skipped_correctly(tmp_path):
    # An unwanted ARRAY-of-STRING (e.g. a tokenizer vocab list) with elements
    # of different lengths, followed by a wanted plain STRING. If the skip
    # logic ever assumed a fixed element width instead of reading each
    # element's own length prefix, this would desync the cursor and either
    # raise or return garbage for model_version.
    kvs = [
        _kv_string_array("tokenizer.tokens", ["a", "bb", "ccc", "dddd", ""]),
        _kv_string("model_version", "2.5.0"),
    ]
    data = _header(version=2, kv_count=len(kvs)) + b"".join(kvs)
    p = _write(tmp_path, "array_span.gguf", data)

    result = read_gguf_kv(p, {"model_version"})

    assert result == {"model_version": "2.5.0"}


def test_wanted_string_array_is_captured_and_stringified(tmp_path):
    kvs = [_kv_string_array("some.tags", ["x", "yy"])]
    data = _header(version=3, kv_count=len(kvs)) + b"".join(kvs)
    p = _write(tmp_path, "wanted_array.gguf", data)

    result = read_gguf_kv(p, {"some.tags"})

    assert result == {"some.tags": "['x', 'yy']"}


def test_early_return_skips_a_corrupted_kv_after_all_wanted_found(tmp_path):
    # architecture + version come first; the third entry claims an unknown
    # value type. If the parser did not stop as soon as both wanted keys were
    # captured, this would raise instead of returning cleanly.
    good = (
        _kv_string("general.architecture", "ltxv")
        + _kv_string("model_version", "2.3.0")
    )
    corrupted_key = _gguf_string("broken.entry") + _u32(255)  # unknown type
    data = _header(version=3, kv_count=3) + good + corrupted_key
    p = _write(tmp_path, "early_return.gguf", data)

    result = read_gguf_kv(p, {"general.architecture", "model_version"})

    assert result == {"general.architecture": "ltxv", "model_version": "2.3.0"}


# -------------------------------------------------------------------- errors


def test_bad_magic_raises(tmp_path):
    data = b"GGUX" + _u32(3) + _u64(0) + _u64(0)
    p = _write(tmp_path, "bad_magic.gguf", data)

    with pytest.raises(GgufParseError):
        read_gguf_kv(p, {"general.architecture"})


@pytest.mark.parametrize("version", [1, 4])
def test_unsupported_version_raises(tmp_path, version):
    data = _header(version=version, kv_count=0)
    p = _write(tmp_path, "bad_version.gguf", data)

    with pytest.raises(GgufParseError):
        read_gguf_kv(p, {"general.architecture"})


def test_truncated_file_raises(tmp_path):
    # Header claims one KV entry but the file ends right after the header.
    data = _header(version=3, kv_count=1)
    p = _write(tmp_path, "truncated.gguf", data)

    with pytest.raises(GgufParseError):
        read_gguf_kv(p, {"general.architecture"})


def test_truncated_mid_string_raises(tmp_path):
    # A key length prefix promises more bytes than the file actually has.
    data = (
        _header(version=3, kv_count=1)
        + _u64(100)  # key claims 100 bytes
        + b"short"
    )
    p = _write(tmp_path, "truncated_mid.gguf", data)

    with pytest.raises(GgufParseError):
        read_gguf_kv(p, {"general.architecture"})


def test_unknown_value_type_raises(tmp_path):
    data = (
        _header(version=3, kv_count=1)
        + _gguf_string("weird.key")
        + _u32(255)  # not a valid GGUF value type
    )
    p = _write(tmp_path, "unknown_type.gguf", data)

    with pytest.raises(GgufParseError):
        read_gguf_kv(p, {"weird.key"})


def test_unsupported_nested_array_type_raises(tmp_path):
    # ARRAY-of-ARRAY: element type 9 is itself ARRAY, which this minimal
    # parser deliberately does not support.
    data = (
        _header(version=3, kv_count=1)
        + _gguf_string("nested")
        + _u32(_TYPE_ARRAY)
        + _u32(_TYPE_ARRAY)  # element type = ARRAY
        + _u64(0)  # count
    )
    p = _write(tmp_path, "nested_array.gguf", data)

    with pytest.raises(GgufParseError):
        read_gguf_kv(p, {"nested"})


def test_huge_kv_count_raises(tmp_path):
    # kv_count alone exceeds the sanity ceiling -- must raise before ever
    # trying to read a KV entry (there are none in this file).
    data = _header(version=3, kv_count=5000)
    p = _write(tmp_path, "huge_kv_count.gguf", data)

    with pytest.raises(GgufParseError):
        read_gguf_kv(p, {"general.architecture"})


def test_huge_array_length_raises(tmp_path):
    data = (
        _header(version=3, kv_count=1)
        + _gguf_string("huge.array")
        + _u32(_TYPE_ARRAY)
        + _u32(_TYPE_UINT32)  # element type
        + _u64(1 << 30)  # count, way over the limit
    )
    p = _write(tmp_path, "huge_array.gguf", data)

    with pytest.raises(GgufParseError):
        read_gguf_kv(p, {"huge.array"})


def test_oversized_string_length_raises(tmp_path):
    data = (
        _header(version=3, kv_count=1)
        + _gguf_string("bloated")
        + _u32(_TYPE_STRING)
        + _u64(16 * 1024 * 1024)  # claims a 16MB string, over the 8MB limit
    )
    p = _write(tmp_path, "oversized_string.gguf", data)

    with pytest.raises(GgufParseError):
        read_gguf_kv(p, {"bloated"})
