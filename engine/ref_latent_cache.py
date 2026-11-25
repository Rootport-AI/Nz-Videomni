"""A generic cache of encoded values kept in main memory between jobs (台帳 §1-297).

torch-free: the values are held as they are given (never copied, never looked
into), and the keys are not looked into either. Each owner holds its own
instance, apart from what ``keep_resident`` keeps: ``release_resident()``
never touches it. The owners:

* MiniMax H3: the references' encoded latents (``H3Pipeline._ref_latents``)
  and the text encoder's conditioning (``H3Pipeline._te_conditioning``).
* LTX 2.3: the text encoder's conditioning
  (``LTXFastVideoPipeline._te_conditioning``; keyed per string).
* LTX 2.5: the text encoder's conditioning (``Ltx25Pipeline._te_conditioning``;
  keyed by the whole sequence of strings handed to the encoding).

Identity of a file (:func:`media_identity`): ``(kind, str(path), st_size,
st_mtime_ns)``. A key is ``(identity, extent)``: the identity plus what was
actually encoded (a picture's resized ``(w, h)``, a video's used frame count,
an audio's sample count), so one file read to two extents is held twice.

One job, four calls (:class:`RefLatentCache`):

* ``begin(identities)``: every key whose identity is not in ``identities`` is
  dropped (a job that keeps nothing passes an empty set: all go); the job's
  notes ("used" / "put") are cleared.
* ``get(key)``: the held value or None; a hit is noted as used.
* ``put(key, value)``: held, noted as used and as put by this job.
* ``end(ok)``: a job that ended well keeps only what it used; a failed job
  drops only what it put (what an earlier job left stays). The notes are cleared.
"""

from __future__ import annotations

import os


def media_identity(kind: str, path) -> tuple:
    """``(kind, str(path), st_size, st_mtime_ns)`` of one reference file (``OSError`` when it cannot be stat'ed)."""
    st = os.stat(path)
    return (str(kind), str(path), int(st.st_size), int(st.st_mtime_ns))


class RefLatentCache:
    """Key -> value, plus the current job's notes of the keys it used and put."""

    def __init__(self) -> None:
        self._values: dict = {}
        self._used: set = set()
        self._put: set = set()

    def begin(self, identities) -> None:
        keep = set(identities)
        for key in [k for k in self._values if k[0] not in keep]:
            del self._values[key]
        self._used.clear()
        self._put.clear()

    def get(self, key):
        value = self._values.get(key)
        if value is not None:
            self._used.add(key)
        return value

    def put(self, key, value) -> None:
        self._values[key] = value
        self._used.add(key)
        self._put.add(key)

    def end(self, ok: bool) -> None:
        if ok:
            for key in [k for k in self._values if k not in self._used]:
                del self._values[key]
        else:
            for key in self._put:
                self._values.pop(key, None)
        self._used.clear()
        self._put.clear()

    def __len__(self) -> int:
        return len(self._values)
