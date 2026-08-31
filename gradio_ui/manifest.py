"""Batch A2V CSV manifest — pure-Python data layer (no Gradio/thread/HTTP deps).

Owns the on-disk contract for the "Batch A2V" feature (bedside/overnight
keyframe-audio-to-video batches, see ``Docs/BATCH_A2V_WORKORDER.md``): scanning
a wav folder into rows, reading/writing the CSV manifest that lives next to the
audio files, merging a rescan into a previously-edited manifest without losing
user edits or generation results, and resolving/deconflicting output paths.

Deliberately dependency-light: no ``gradio``, no ``threading``, no HTTP client.
This keeps the module trivially unit-testable and safe to import from any
layer (batch worker thread, Gradio handlers, future WebView2-facing tooling)
without dragging in a GUI framework. In particular this is why the wav-length
probe below is a small local reimplementation of
``gradio_ui.handlers._wav_duration_seconds`` rather than an import of it: that
function lives in a module that imports ``gradio`` at module scope, and
importing it here would make every consumer of :mod:`gradio_ui.manifest`
(including a future headless batch worker) pay for a ``gradio`` import it may
not otherwise need. The frame-count suggestion (``frames_for``), by contrast,
IS injected as a plain callable by the caller — that keeps the "which frame
count does the app actually pick" policy in exactly one place
(``gradio_ui.handlers.suggest_frames_for_audio``) while this module stays
policy-free.
"""

from __future__ import annotations

import csv
import os
import time
import wave
from dataclasses import dataclass
from math import floor
from pathlib import Path
from typing import Callable

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# Manifest file lives directly under the audio folder (WORKORDER §2.9/§4.5).
MANIFEST_NAME = "batch_a2v_manifest.csv"
# Fallback write target when the primary CSV is locked (e.g. open in Excel)
# even after retrying ``os.replace`` — see write_manifest_atomic().
AUTOSAVE_NAME = "batch_a2v_manifest.autosave.csv"

# Same set as config.yaml's upload.allowed_audio_extensions (kept as a literal
# constant here rather than read from config.yaml, per the WP spec — the batch
# scan step never talks to the API/config layer).
ALLOWED_AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg")

# Row status values (WORKORDER §2.3).
STAT_WAITING = "Waiting"
STAT_GENERATING = "Generating"
STAT_DONE = "Done"
STAT_FAILED = "Failed"
STAT_SKIP = "Skip"

# image column sentinel meaning "use the Generate tab's common i2v keyframe".
IMAGE_SHARED = "Shared"

# Server hard cap (num_frames <= 481, api/models.py). Mirrors
# gradio_ui.handlers.suggest_frames_for_audio's raw-frame-count formula.
MAX_FRAMES = 481

# CSV header — English lowercase, 10 columns, fixed order (shared contract
# with the WebView2/React frontend per WORKORDER §2.9).
CSV_FIELDS = [
    "queue", "wav", "duration", "image", "prompt",
    "stat", "output", "frames", "skip_reason", "error",
]

# write_manifest_atomic() retry budget for a locked target file (e.g. open in
# Excel) before falling back to the autosave path.
_WRITE_RETRY_ATTEMPTS = 5
_WRITE_RETRY_BASE_DELAY_S = 0.05


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class BatchRow:
    """One manifest row. ``wav``/``image``/``output`` are filenames only (no
    directory component) — the wav folder / output folder are resolved
    separately (see :func:`resolve_output_dir`)."""

    queue: int
    wav: str
    duration_s: float = 0.0
    image: str = IMAGE_SHARED
    prompt: str = ""
    stat: str = STAT_WAITING
    output: str = ""
    frames: int = 0
    skip_reason: str = ""
    error: str = ""


@dataclass
class WriteResult:
    """Result of :func:`write_manifest_atomic`.

    ``ok=True``  -> the primary manifest CSV was written/replaced normally.
    ``ok=False`` -> the primary path stayed locked through every retry; when
    ``autosave_path`` is not ``None`` the SAME rows were written there instead
    (the caller should surface ``locked`` to the user, e.g. via ``gr.Warning``,
    so they know to close whatever has the file open and rescan)."""

    ok: bool
    locked: bool = False
    path: Path | None = None
    autosave_path: Path | None = None


# --------------------------------------------------------------------------- #
# Small coercion helpers (safe defaulting for read_manifest — a hand-edited or
# truncated CSV must never raise).
# --------------------------------------------------------------------------- #
def _to_int(value, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _to_float(value, default: float = 0.0) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def _to_str(value, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def _row_from_dict(d: dict, queue_fallback: int) -> BatchRow:
    """Build a :class:`BatchRow` from a ``csv.DictReader`` row dict. Missing
    columns (``d.get(...)`` -> ``None``) and type-mismatched values (e.g. a
    non-numeric ``duration``) fall back to safe defaults instead of raising —
    a broken/hand-edited CSV degrades gracefully rather than dying on load."""
    return BatchRow(
        queue=_to_int(d.get("queue"), queue_fallback),
        wav=_to_str(d.get("wav")),
        duration_s=_to_float(d.get("duration")),
        image=_to_str(d.get("image")) or IMAGE_SHARED,
        prompt=_to_str(d.get("prompt")),
        stat=_to_str(d.get("stat")) or STAT_WAITING,
        output=_to_str(d.get("output")),
        frames=_to_int(d.get("frames")),
        skip_reason=_to_str(d.get("skip_reason")),
        error=_to_str(d.get("error")),
    )


# --------------------------------------------------------------------------- #
# CSV I/O
# --------------------------------------------------------------------------- #
def read_manifest(wav_dir) -> list[BatchRow] | None:
    """Read ``{wav_dir}/batch_a2v_manifest.csv``.

    Returns ``None`` when the file does not exist (caller treats this as "no
    prior manifest — first scan"). Returns a (possibly empty) list otherwise;
    a malformed row never aborts the whole read — see :func:`_row_from_dict`.
    A totally unreadable file (I/O error, garbage encoding) degrades to an
    empty list rather than raising, for the same "never die on a bad CSV"
    reason."""
    path = Path(wav_dir) / MANIFEST_NAME
    if not path.exists():
        return None

    rows: list[BatchRow] = []
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for i, raw in enumerate(reader, start=1):
                try:
                    rows.append(_row_from_dict(raw, i))
                except Exception:
                    # Should not happen (_row_from_dict itself never raises),
                    # but keep the "one bad row must not lose the rest" promise
                    # airtight even if a future edit adds a raising path.
                    rows.append(BatchRow(queue=i, wav=""))
    except Exception:
        return rows
    return rows


def _write_csv(fh, rows: list[BatchRow]) -> None:
    writer = csv.writer(fh)
    writer.writerow(CSV_FIELDS)
    for r in rows:
        writer.writerow([
            r.queue, r.wav, r.duration_s, r.image, r.prompt,
            r.stat, r.output, r.frames, r.skip_reason, r.error,
        ])


def write_manifest_atomic(wav_dir, rows: list[BatchRow]) -> WriteResult:
    """Write ``rows`` to ``{wav_dir}/batch_a2v_manifest.csv`` atomically.

    Writes the full CSV to a same-directory temp file, then ``os.replace()``s
    it into place (atomic on both POSIX and Windows). If the target is locked
    by another process (e.g. open in Excel) ``os.replace`` raises
    ``PermissionError`` on Windows; this is retried with a short exponential
    backoff (5 attempts). If every retry fails, the same rows are written to
    ``batch_a2v_manifest.autosave.csv`` instead (best effort) and the result
    carries ``locked=True`` so the caller can warn the user (e.g.
    ``gr.Warning``) instead of silently losing the edit.
    """
    wav_dir = Path(wav_dir)
    path = wav_dir / MANIFEST_NAME
    tmp_path = wav_dir / (MANIFEST_NAME + ".tmp")

    with open(tmp_path, "w", encoding="utf-8-sig", newline="") as f:
        _write_csv(f, rows)

    delay = _WRITE_RETRY_BASE_DELAY_S
    for attempt in range(_WRITE_RETRY_ATTEMPTS):
        try:
            os.replace(tmp_path, path)
            return WriteResult(ok=True, path=path)
        except PermissionError:
            if attempt == _WRITE_RETRY_ATTEMPTS - 1:
                break
            time.sleep(delay)
            delay *= 2

    # Locked through every retry -> best-effort autosave fallback.
    autosave_path: Path | None = wav_dir / AUTOSAVE_NAME
    try:
        with open(autosave_path, "w", encoding="utf-8-sig", newline="") as f:
            _write_csv(f, rows)
    except Exception:
        autosave_path = None
    try:
        if tmp_path.exists():
            tmp_path.unlink()
    except Exception:
        pass
    return WriteResult(ok=False, locked=True, path=path, autosave_path=autosave_path)


# --------------------------------------------------------------------------- #
# Frame-count arithmetic (pure) — the ONE place the raw 8n+1 frame-count formula
# and the 481-frame Skip test live, so scan_wav_folder (initial scan) and the
# batch runner's start-time re-judgment (gradio_ui.batch) share exactly the same
# math instead of re-deriving it. Deliberately policy-free (no clamp/shrink):
# that "which frame count does the app actually pick" policy stays in
# gradio_ui.handlers.suggest_frames_for_audio, injected as ``frames_for``.
# --------------------------------------------------------------------------- #
def raw_frame_count(dur: float, fps) -> int:
    """The raw (unclamped) 8n+1 frame count ``dur`` seconds covers at ``fps`` —
    the exact expression the 481-frame Skip test keys off
    (``((floor(dur*fps) - 1) // 8) * 8 + 1``)."""
    return ((floor(dur * float(fps)) - 1) // 8) * 8 + 1


def over_frame_limit(dur: float, fps) -> bool:
    """True when ``dur`` at ``fps`` needs more than :data:`MAX_FRAMES` raw
    frames (i.e. the row must be Skipped with ``skip_reason="over-481f"``)."""
    return raw_frame_count(dur, fps) > MAX_FRAMES


# --------------------------------------------------------------------------- #
# Wav-length probe (local reimplementation — see module docstring for why this
# is not an import of gradio_ui.handlers._wav_duration_seconds).
# --------------------------------------------------------------------------- #
def _wav_duration_seconds(path) -> float | None:
    """Duration in seconds of a ``.wav`` file via the stdlib :mod:`wave`
    module, or ``None`` when it cannot be read (bad header, zero frame rate,
    not actually a wav). Caller is expected to have already filtered to a
    ``.wav`` extension."""
    try:
        with wave.open(str(path), "rb") as w:
            frames = w.getnframes()
            rate = w.getframerate()
    except Exception:
        return None
    if not rate or rate <= 0:
        return None
    return frames / float(rate)


# --------------------------------------------------------------------------- #
# Scan
# --------------------------------------------------------------------------- #
# MCPサーバー側 mcp_server/batch_planning.py に写経あり（scan_wav_folder の走査
# 規約 / raw_frame_count / over_frame_limit）。変更時は両方＋パリティテストを更新
def scan_wav_folder(
    wav_dir, fps, frames_for: Callable[[float, float], int]
) -> list[BatchRow]:
    """Scan ``wav_dir`` for audio files and build fresh :class:`BatchRow`\\ s.

    * Only ``ALLOWED_AUDIO_EXTENSIONS`` files are considered; the manifest CSV
      itself and its ``.tmp``/autosave siblings are always excluded.
    * Sorted by **mtime ascending** (VOICEROID writes files in script order —
      WORKORDER §2.7).
    * Non-``.wav`` files, and ``.wav`` files whose length cannot be read, get
      ``stat=Skip, skip_reason="wav-only-alpha"`` (kept in the list — visible
      but excluded — rather than dropped silently).
    * ``.wav`` files whose raw frame count (``((floor(dur*fps)-1)//8)*8+1``)
      exceeds 481 get ``stat=Skip, skip_reason="over-481f"``.
    * Everything else gets its ``frames`` from the injected ``frames_for(dur,
      fps)`` callable (the caller wires in
      ``gradio_ui.handlers.suggest_frames_for_audio``, which additionally
      shrinks/clamps against the latent-frame budget — that policy is
      deliberately NOT duplicated here).
    """
    wav_dir = Path(wav_dir)
    reserved = {MANIFEST_NAME, AUTOSAVE_NAME}

    candidates: list[tuple[float, Path]] = []
    for p in wav_dir.iterdir():
        if not p.is_file():
            continue
        if p.name in reserved or p.name.endswith(".tmp"):
            continue
        if p.suffix.lower() not in ALLOWED_AUDIO_EXTENSIONS:
            continue
        try:
            mtime = p.stat().st_mtime
        except OSError:
            continue
        candidates.append((mtime, p))
    candidates.sort(key=lambda t: t[0])

    rows: list[BatchRow] = []
    for i, (_mtime, p) in enumerate(candidates, start=1):
        is_wav = p.suffix.lower() == ".wav"
        dur = _wav_duration_seconds(p) if is_wav else None
        if dur is None:
            rows.append(BatchRow(
                queue=i, wav=p.name, duration_s=0.0,
                stat=STAT_SKIP, skip_reason="wav-only-alpha",
            ))
            continue

        if over_frame_limit(dur, fps):
            rows.append(BatchRow(
                queue=i, wav=p.name, duration_s=dur,
                stat=STAT_SKIP, skip_reason="over-481f",
            ))
            continue

        frames = int(frames_for(dur, fps))
        rows.append(BatchRow(queue=i, wav=p.name, duration_s=dur, frames=frames))

    return rows


def compute_spill_warnings(
    rows: list[BatchRow], width, height, spill_free_frames: dict
) -> list[str]:
    """Return one warning string per non-Skip row whose ``frames`` exceeds the
    comfortable (spill-free) threshold for ``{width}x{height}`` in
    ``spill_free_frames`` (config.yaml's ``limits.spill_free_frames`` shape,
    e.g. ``{"1280x768": 273}``). Unknown resolution -> no warnings. This never
    changes ``stat`` — spill is a slowdown, not an exclusion (WORKORDER §2.6 /
    §4.3: only the 481-frame cap Skips)."""
    key = f"{int(width)}x{int(height)}"
    threshold = (spill_free_frames or {}).get(key)
    if threshold is None:
        return []
    warnings: list[str] = []
    for r in rows:
        if r.stat == STAT_SKIP:
            continue
        if r.frames > threshold:
            warnings.append(f"{r.wav}: {r.frames}f > {threshold}f ({key})")
    return warnings


# --------------------------------------------------------------------------- #
# Merge (rescan via "Set audios" over an existing manifest)
# --------------------------------------------------------------------------- #
def merge_rows(
    existing: list[BatchRow] | None, scanned: list[BatchRow]
) -> tuple[list[BatchRow], list[str]]:
    """Merge a fresh :func:`scan_wav_folder` result into a previously-saved
    manifest, per WORKORDER's 4 merge rules:

    1. A wav present in both keeps the existing row's ``prompt``/``image``/
       ``stat``/``output``/``error`` (user edits + generation results
       survive); ``duration``/``frames``/``skip_reason`` are refreshed from
       the rescan. If the rescan newly flags the row Skip, ``stat`` is
       overwritten to Skip UNLESS the existing row was already ``Done`` — a
       completed row is never demoted, it just earns a warning instead.
    2. A wav present only in the rescan is a brand-new row, inserted as-is
       (its own ``scan_wav_folder`` stat — normally Waiting, or Skip when the
       scan itself flagged it).
    3. A wav present only in the existing manifest (the file is gone) is
       dropped — UNLESS its ``stat`` is ``Done``, in which case it is kept at
       the tail with a warning (never silently lose a completed generation).
    4. The result is ordered by the rescan's order (mtime), any Done leftovers
       appended at the end, then ``queue`` is renumbered from 1.

    Returns ``(merged_rows, warnings)``.
    """
    existing = existing or []
    existing_by_wav = {r.wav: r for r in existing}
    scanned_wavs = {r.wav for r in scanned}

    warnings: list[str] = []
    merged: list[BatchRow] = []

    for srow in scanned:
        erow = existing_by_wav.get(srow.wav)
        if erow is None:
            # Rule 2: brand new — keep the scan's own row (incl. its Skip
            # status, if any) untouched.
            merged.append(BatchRow(
                queue=0, wav=srow.wav, duration_s=srow.duration_s,
                image=srow.image, prompt=srow.prompt, stat=srow.stat,
                output=srow.output, frames=srow.frames,
                skip_reason=srow.skip_reason, error=srow.error,
            ))
            continue

        # Rule 1: known wav — existing wins for the user-owned columns.
        stat = erow.stat
        if srow.stat == STAT_SKIP:
            if erow.stat == STAT_DONE:
                warnings.append(
                    f"{srow.wav}: 再スキャンでSkip判定 (理由={srow.skip_reason}) "
                    "になりましたが、既にDoneのため一覧に保持します。"
                )
                stat = STAT_DONE
            else:
                stat = STAT_SKIP
        merged.append(BatchRow(
            queue=0, wav=srow.wav, duration_s=srow.duration_s,
            image=erow.image, prompt=erow.prompt, stat=stat,
            output=erow.output, frames=srow.frames,
            skip_reason=srow.skip_reason, error=erow.error,
        ))

    # Rule 3: existing-only rows (the wav file vanished).
    tail: list[BatchRow] = []
    for erow in existing:
        if erow.wav in scanned_wavs:
            continue
        if erow.stat == STAT_DONE:
            warnings.append(
                f"{erow.wav}: 音声ファイルが見つかりませんが、Done実績があるため一覧に残します。"
            )
            tail.append(erow)
        # else: silently dropped (no generation result to protect).

    result = merged + tail
    # Rule 4: renumber queue from 1 in the final order.
    for i, row in enumerate(result, start=1):
        row.queue = i
    return result, warnings


# --------------------------------------------------------------------------- #
# Output path resolution
# --------------------------------------------------------------------------- #
def resolve_output_dir(wav_dir, mode: str, custom_dir=None) -> Path:
    """Resolve the batch's output directory. Path resolution only — the
    caller creates the directory if needed.

    * ``mode="auto"``   -> the sibling folder ``"{wav_dir名}_a2v_out"`` next to
      ``wav_dir`` (NOT created automatically here).
    * ``mode="custom"`` -> ``custom_dir`` verbatim.
    """
    wav_dir = Path(wav_dir)
    if mode == "custom":
        return Path(custom_dir)
    return wav_dir.parent / f"{wav_dir.name}_a2v_out"


def unique_output_name(out_dir, wav_name: str) -> str:
    """Return ``{wav basename}.mp4``, or ``..._2.mp4``, ``..._3.mp4``, … the
    first name that does not already exist under ``out_dir`` — so a batch run
    never overwrites a prior output for a same-named wav."""
    out_dir = Path(out_dir)
    stem = Path(wav_name).stem
    candidate = f"{stem}.mp4"
    if not (out_dir / candidate).exists():
        return candidate
    n = 2
    while True:
        candidate = f"{stem}_{n}.mp4"
        if not (out_dir / candidate).exists():
            return candidate
        n += 1
