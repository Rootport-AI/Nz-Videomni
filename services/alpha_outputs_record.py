"""The record of an Alpha Gen job's exit outputs (``alpha_outputs.json``).

The Alpha Gen exit (第 3 弾, Docs/ALPHAGEN_DESIGN.md §10.1) makes its outputs
on demand and reuses them while the thresholds (and, for the chroma-key video,
the colour) are unchanged. To answer "made with which thresholds?" it keeps a
small JSON file next to the outputs::

    {"rgba": {"transparent_threshold": 7, "opaque_threshold": 248},
     "chroma": {"green": {"transparent_threshold": 7, "opaque_threshold": 248}}}

One transparent video (``rgba.mov`` + its preview ``alpha_preview.mp4``) and
one chroma-key video per colour (``chroma_<color>.mp4``).

The one place that reads and writes the file. Both ``services/job_store.py``
(``JobResponse.alpha``) and ``services/alpha_output_manager.py`` use it; it
imports neither, so there is no import cycle.

An output counts as present only when BOTH its record entry and its file(s)
exist (:func:`existing`): a record left behind by a deleted file must not
promise a file that answers 404. A missing or unreadable record is read as
empty, and an entry that does not look like a pair of thresholds is ignored.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

RECORD_FILENAME = "alpha_outputs.json"
RGBA_FILENAME = "rgba.mov"
PREVIEW_FILENAME = "alpha_preview.mp4"


def chroma_filename(color: str) -> str:
    return f"chroma_{color}.mp4"


@dataclass(frozen=True)
class AlphaThresholds:
    transparent: int
    opaque: int

    def to_json(self) -> dict:
        return {"transparent_threshold": self.transparent, "opaque_threshold": self.opaque}


@dataclass
class AlphaOutputsRecord:
    rgba: AlphaThresholds | None = None
    chroma: dict[str, AlphaThresholds] = field(default_factory=dict)


def _thresholds(entry: object) -> AlphaThresholds | None:
    """An entry of the file as thresholds, or None when it is not a valid pair."""
    if not isinstance(entry, dict):
        return None
    transparent = entry.get("transparent_threshold")
    opaque = entry.get("opaque_threshold")
    if type(transparent) is not int or type(opaque) is not int:
        return None
    if not 0 <= transparent <= opaque <= 255:
        return None
    return AlphaThresholds(transparent, opaque)


def read(job_dir: Path) -> AlphaOutputsRecord:
    """The record as written, without looking at the output files."""
    try:
        data = json.loads((job_dir / RECORD_FILENAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return AlphaOutputsRecord()
    if not isinstance(data, dict):
        return AlphaOutputsRecord()

    record = AlphaOutputsRecord(rgba=_thresholds(data.get("rgba")))
    chroma = data.get("chroma")
    if isinstance(chroma, dict):
        for color, entry in chroma.items():
            thresholds = _thresholds(entry)
            if isinstance(color, str) and thresholds is not None:
                record.chroma[color] = thresholds
    return record


def write(job_dir: Path, record: AlphaOutputsRecord) -> None:
    """Write the record (temporary name, then ``os.replace``). Raises OSError.

    Known and left as is: on Windows a ``GET /jobs`` reading the file at the
    very moment of the replace can make the replace fail (a rare 503)."""
    data = {
        "rgba": record.rgba.to_json() if record.rgba is not None else None,
        "chroma": {color: t.to_json() for color, t in record.chroma.items()},
    }
    path = job_dir / RECORD_FILENAME
    tmp = job_dir / "_alpha_outputs.tmp.json"
    try:
        # Text mode, like metadata.json (CRLF on Windows).
        with tmp.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps(data, ensure_ascii=False, indent=2))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def existing(job_dir: Path) -> AlphaOutputsRecord:
    """The outputs that are really there: the record's entries whose files exist.

    The transparent video needs both ``rgba.mov`` and ``alpha_preview.mp4``.
    """
    record = read(job_dir)
    rgba = record.rgba
    if rgba is not None:
        if not (job_dir / RGBA_FILENAME).exists() or not (job_dir / PREVIEW_FILENAME).exists():
            rgba = None
    chroma = {
        color: thresholds
        for color, thresholds in record.chroma.items()
        if (job_dir / chroma_filename(color)).exists()
    }
    return AlphaOutputsRecord(rgba=rgba, chroma=chroma)
