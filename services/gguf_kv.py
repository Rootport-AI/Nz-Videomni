"""Minimal, dependency-free GGUF key-value header parser.

MULTI_ENGINE_DESIGN.md §2.2/§2.5: engine selection at load time reads two
string KV fields out of a transformer GGUF's header -- ``general.architecture``
and ``model_version`` -- to tell LTX 2.3 from LTX 2.5 (and to catch a
completely wrong file) BEFORE a worker process is ever started. The app venv
(``.venv``) deliberately does not have the ``gguf`` PyPI package installed (it
stays torch-free, §5.3), so this module reads just enough of the GGUF binary
format by hand -- there is no dependency on any GGUF library.

Deliberately narrow scope: this is NOT a general GGUF reader. It reads ONLY
the header's KV section (magic / version / tensor_count / kv_count, then each
KV entry in file order) and stops as soon as every requested key has been
found -- tensor metadata and tensor data (the multi-gigabyte bulk of the file)
are never touched. GGUF value types 0-12 (see ``_SCALAR_FORMATS`` /
``_TYPE_STRING`` / ``_TYPE_ARRAY`` below) are understood; anything else, or
any structural inconsistency (bad magic, unsupported version, truncation, an
oversized count) raises :class:`GgufParseError`.

A real GGUF orders its KV entries arbitrarily (license/config text, tokenizer
arrays, quantization metadata, ...), so callers must not assume the wanted
keys come first -- this parser scans in file order and returns the moment
every wanted key has been captured, so a damaged or huge KV that happens to
sit AFTER all the wanted keys never turns a perfectly good read into an
error.
"""

from __future__ import annotations

import struct
from pathlib import Path


class GgufParseError(ValueError):
    """The GGUF header is malformed, truncated, or uses a KV structure this
    deliberately minimal parser does not understand (see module docstring)."""


_MAGIC = b"GGUF"
_SUPPORTED_VERSIONS = (2, 3)

# GGUF value-type tags -> (struct format, byte width) for the FIXED-WIDTH
# scalar types. STRING (8) and ARRAY (9) are variable-width and handled by
# _read_string/_skip_string and _read_array separately below. This dict's key
# set IS "the types 0-12 this parser understands" minus STRING/ARRAY.
_SCALAR_FORMATS: dict[int, str] = {
    0: "<B",   # UINT8
    1: "<b",   # INT8
    2: "<H",   # UINT16
    3: "<h",   # INT16
    4: "<I",   # UINT32
    5: "<i",   # INT32
    6: "<f",   # FLOAT32
    7: "<?",   # BOOL
    10: "<Q",  # UINT64
    11: "<q",  # INT64
    12: "<d",  # FLOAT64
}
_TYPE_STRING = 8
_TYPE_ARRAY = 9

# Sanity ceilings (§2.5 "上限ガード"). These bound how much a single call can
# be made to read/allocate even when handed a hostile or corrupted file --
# real GGUFs from Nz-GGUF-Converter-LTX23 stay far under all four.
_MAX_KV_COUNT = 4096
_MAX_KEY_LEN = 1024
_MAX_STRING_LEN = 8 * 1024 * 1024
_MAX_ARRAY_LEN = 1 << 24


def _read_exact(f, n: int) -> bytes:
    data = f.read(n)
    if len(data) != n:
        raise GgufParseError(
            f"unexpected end of file (wanted {n} bytes, got {len(data)})"
        )
    return data


def _read_u32(f) -> int:
    return struct.unpack("<I", _read_exact(f, 4))[0]


def _read_u64(f) -> int:
    return struct.unpack("<Q", _read_exact(f, 8))[0]


def _read_string(f, max_len: int) -> str:
    length = _read_u64(f)
    if length > max_len:
        raise GgufParseError(
            f"string length {length} exceeds the {max_len}-byte limit"
        )
    data = _read_exact(f, length)
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise GgufParseError(f"string value is not valid UTF-8: {exc}") from exc


def _skip_string(f) -> None:
    length = _read_u64(f)
    if length > _MAX_STRING_LEN:
        raise GgufParseError(
            f"string length {length} exceeds the {_MAX_STRING_LEN}-byte limit"
        )
    f.seek(length, 1)


def _read_array(f, capture: bool):
    elem_type = _read_u32(f)
    count = _read_u64(f)
    if count > _MAX_ARRAY_LEN:
        raise GgufParseError(
            f"array length {count} exceeds the {_MAX_ARRAY_LEN} limit"
        )

    if elem_type in _SCALAR_FORMATS:
        # Fixed-width elements: skip (or read) the whole run in one seek/read
        # rather than one element at a time.
        fmt = _SCALAR_FORMATS[elem_type]
        width = struct.calcsize(fmt)
        if not capture:
            f.seek(width * count, 1)
            return None
        data = _read_exact(f, width * count)
        return list(struct.unpack(f"<{count}{fmt[1]}", data))

    if elem_type == _TYPE_STRING:
        # Variable-width elements: each carries its own length prefix, so
        # there is no bulk seek -- walk them one at a time.
        if capture:
            return [_read_string(f, _MAX_STRING_LEN) for _ in range(count)]
        for _ in range(count):
            _skip_string(f)
        return None

    raise GgufParseError(
        f"unsupported GGUF array element type {elem_type} "
        "(only scalar or string array elements are supported)"
    )


def _read_value(f, value_type: int, capture: bool):
    """Read (``capture=True``) or skip (``capture=False``) one KV value of
    the given type. Returns the decoded Python value when capturing, ``None``
    when skipping.
    """
    if value_type in _SCALAR_FORMATS:
        if not capture:
            f.seek(struct.calcsize(_SCALAR_FORMATS[value_type]), 1)
            return None
        return struct.unpack(
            _SCALAR_FORMATS[value_type],
            _read_exact(f, struct.calcsize(_SCALAR_FORMATS[value_type])),
        )[0]
    if value_type == _TYPE_STRING:
        if not capture:
            _skip_string(f)
            return None
        return _read_string(f, _MAX_STRING_LEN)
    if value_type == _TYPE_ARRAY:
        return _read_array(f, capture)
    raise GgufParseError(f"unknown GGUF value type {value_type}")


def read_gguf_kv(path: Path, wanted: set[str]) -> dict[str, str]:
    """Read the requested KV metadata keys from a GGUF file's header.

    Scans the KV section in file order and returns as soon as every key in
    ``wanted`` has been captured. A ``wanted`` key the file does not have is
    simply absent from the returned dict -- that is NOT an error (a foreign
    or hand-edited GGUF may legitimately lack it; the caller decides what to
    do about a missing key). Values are returned as ``str``; a captured
    non-string value (e.g. an integer KV) is converted with ``str()``.

    Raises :class:`GgufParseError` on any structural problem: bad magic, an
    unsupported version (only 2 and 3 are understood), truncation, a value
    type this parser does not understand, or a size that exceeds this
    module's sanity limits (kv_count / key length / string length / array
    length).
    """
    wanted = set(wanted)
    found: dict[str, str] = {}
    with open(path, "rb") as f:
        magic = _read_exact(f, 4)
        if magic != _MAGIC:
            raise GgufParseError(f"bad magic {magic!r} (expected {_MAGIC!r})")
        version = _read_u32(f)
        if version not in _SUPPORTED_VERSIONS:
            raise GgufParseError(
                f"unsupported GGUF version {version} (expected 2 or 3)"
            )
        _read_u64(f)  # tensor_count -- unused, tensor info is never read.
        kv_count = _read_u64(f)
        if kv_count > _MAX_KV_COUNT:
            raise GgufParseError(
                f"kv_count {kv_count} exceeds the {_MAX_KV_COUNT} limit"
            )

        for _ in range(kv_count):
            if wanted and set(found) >= wanted:
                # Early return: everything asked for is already in hand, so
                # a damaged or oversized KV further down the file can never
                # turn an otherwise-good read into an error.
                return found
            key = _read_string(f, _MAX_KEY_LEN)
            value_type = _read_u32(f)
            capture = key in wanted
            value = _read_value(f, value_type, capture)
            if capture:
                found[key] = value if isinstance(value, str) else str(value)

    return found
