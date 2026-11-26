"""The copy of Spark-H3's files (``engine_h3/third_party/spark_h3/``) is the author's, byte for byte.

Reads files only (no engine import), so it runs in the app venv::

    .venv/Scripts/python.exe -m pytest tests/test_spark_h3_vendored.py

The SHA-256 table of the folder's README.md is the record; this test holds the files to it,
and holds the table to the folder (no file missing from it, none extra).
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "engine_h3" / "third_party" / "spark_h3"
#: Added by this project (not the author's): not in the table.
OWN = {"__init__.py", "README.md"}

_ROW = re.compile(r"^\| `([^`]+)` \| ([\d,]+) \| `([0-9a-f]{64})` \|$")


def _table() -> dict[str, tuple[int, str]]:
    rows = {}
    for line in (DIR / "README.md").read_text(encoding="utf-8").splitlines():
        m = _ROW.match(line)
        if m:
            rows[m.group(1)] = (int(m.group(2).replace(",", "")), m.group(3))
    return rows


def _files() -> set[str]:
    return {
        p.relative_to(DIR).as_posix()
        for p in DIR.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.relative_to(DIR).as_posix() not in OWN
    }


def test_the_table_lists_every_copied_file():
    table = _table()
    assert len(table) == 23
    assert set(table) == _files()
    assert {"LICENSE", "comfyui_backend.py", "comfyui_reblock_plan.py", "h3_sparse_attention/__init__.py"} <= set(table)


def test_every_copied_file_has_the_recorded_sha256():
    for name, (size, sha) in _table().items():
        data = (DIR / name).read_bytes()
        assert (len(data), hashlib.sha256(data).hexdigest()) == (size, sha), name


def test_nothing_that_imports_comfyui_was_copied():
    for name in _files():
        if name.endswith(".py"):
            text = (DIR / name).read_text(encoding="utf-8")
            assert not re.search(r"^\s*(from|import)\s+comfy(\.|_extras|\s|$)", text, re.M), name


def test_git_keeps_the_bytes():
    lines = (ROOT / ".gitattributes").read_text(encoding="utf-8").splitlines()
    assert "engine_h3/third_party/spark_h3/** -text" in lines
