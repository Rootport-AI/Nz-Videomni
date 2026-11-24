"""engine_h3.ref_latent_cache: the reference latents kept between jobs (台帳 §1-297).

torch-free; runs in the app venv and in ``.venv-engine-h3``::

    .venv/Scripts/python.exe -m pytest tests/test_h3_ref_latent_cache.py
    .venv-engine-h3/Scripts/python.exe scripts/run_h3_pytest.py tests/test_h3_ref_latent_cache.py

What is pinned: ``begin`` drops what the job's list does not name (all with an
empty list); a job that ends well keeps only what it used; a failed job drops
only what it put (what an earlier job left stays); ``end`` without ``begin``
changes nothing; the identity changes with the size, the modification time
and the path. The engine's use of it is in tests/test_h3_engine.py and
tests/test_h3_chain.py.
"""

from __future__ import annotations

import os

from engine_h3 import ref_latent_cache as RC


def _identity(tmp_path, name, data=b"x", kind="video"):
    p = tmp_path / name
    p.write_bytes(data)
    return RC.media_identity(kind, p)


def _filled(tmp_path):
    """A cache that a finished job left with a (2 extents) and b."""
    a, b = _identity(tmp_path, "a.mp4"), _identity(tmp_path, "b.mp4")
    c = RC.RefLatentCache()
    c.begin({a, b})
    c.put((a, 39), "a39")
    c.put((a, 22), "a22")
    c.put((b, 39), "b39")
    c.end(True)
    return c, a, b


def test_identity_is_kind_path_size_and_mtime_ns(tmp_path):
    p = tmp_path / "v.mp4"
    p.write_bytes(b"abc")
    st = os.stat(p)
    assert RC.media_identity("video", p) == ("video", str(p), 3, st.st_mtime_ns)
    assert RC.media_identity("video", str(p)) == RC.media_identity("video", p)


def test_identity_changes_with_size_mtime_and_path(tmp_path):
    p = tmp_path / "v.mp4"
    p.write_bytes(b"abc")
    first = RC.media_identity("video", p)
    p.write_bytes(b"abcd")                                    # size
    os.utime(p, ns=(first[3], first[3]))
    assert RC.media_identity("video", p)[2] == 4 and RC.media_identity("video", p) != first
    p.write_bytes(b"abc")
    os.utime(p, ns=(first[3] + 1_000_000, first[3] + 1_000_000))   # modification time only
    assert RC.media_identity("video", p)[2] == 3 and RC.media_identity("video", p) != first
    os.utime(p, ns=(first[3], first[3]))
    assert RC.media_identity("video", p) == first
    q = tmp_path / "w.mp4"                                    # the same bytes and time elsewhere
    q.write_bytes(b"abc")
    os.utime(q, ns=(first[3], first[3]))
    assert RC.media_identity("video", q) != first
    assert RC.media_identity("audio", p) != first             # the kind is part of it


def test_get_and_put_return_the_value_itself(tmp_path):
    a = _identity(tmp_path, "a.mp4")
    c = RC.RefLatentCache()
    c.begin({a})
    value = (object(), object())
    assert c.get((a, 39)) is None
    c.put((a, 39), value)
    assert c.get((a, 39)) is value and len(c) == 1


def test_begin_drops_what_the_list_does_not_name(tmp_path):
    c, a, b = _filled(tmp_path)
    assert len(c) == 3
    c.begin({a})
    assert len(c) == 2 and c.get((a, 39)) == "a39" and c.get((a, 22)) == "a22" and c.get((b, 39)) is None


def test_begin_with_an_empty_list_drops_everything(tmp_path):
    c, _a, _b = _filled(tmp_path)
    c.begin(set())
    assert len(c) == 0
    c.end(True)
    assert len(c) == 0


def test_end_ok_keeps_only_what_the_job_used(tmp_path):
    c, a, b = _filled(tmp_path)
    c.begin({a, b})
    assert c.get((a, 39)) == "a39"                            # used (a hit)
    c.put((b, 22), "b22")                                     # used (put)
    c.end(True)
    assert len(c) == 2
    c.begin({a, b})
    assert c.get((a, 39)) == "a39" and c.get((b, 22)) == "b22"
    assert c.get((a, 22)) is None and c.get((b, 39)) is None   # not used by the job before: gone


def test_end_failed_drops_only_what_the_job_put(tmp_path):
    c, a, b = _filled(tmp_path)
    c.begin({a, b})
    assert c.get((a, 39)) == "a39"
    c.put((a, 56), "a56")
    c.end(False)
    assert len(c) == 3                                        # what the earlier job left stays, used or not
    c.begin({a, b})
    assert c.get((a, 56)) is None
    assert c.get((a, 39)) == "a39" and c.get((a, 22)) == "a22" and c.get((b, 39)) == "b39"


def test_end_without_begin_changes_nothing(tmp_path):
    c, a, b = _filled(tmp_path)
    c.end(False)                                              # a job refused before its begin
    assert len(c) == 3
    c.begin({a, b})
    assert c.get((a, 39)) == "a39" and c.get((a, 22)) == "a22" and c.get((b, 39)) == "b39"
    fresh = RC.RefLatentCache()
    fresh.end(False)
    fresh.end(True)
    assert len(fresh) == 0


def test_module_is_torch_free():
    from pathlib import Path

    src = Path(RC.__file__).read_text(encoding="utf-8")
    assert "import torch" not in src
