"""Batch A2V execution engine — the in-process worker thread (``BatchRunner``).

This is the *execution body* of the "Batch A2V" overnight feature: it feeds a
wav folder's audio files, one at a time, into ``POST /generate/chain`` (each a
single-clip A2V chain carrying ``source_audio``) and drives every job to
completion, copying the finished mp4 into the batch's output folder. It is
built to run unattended for 8-17 hours (bedside / overnight) against the frozen
REST API's single-job queue.

Why a thread and not a Gradio event handler
--------------------------------------------
A Gradio event handler that loops over the rows dies the moment the browser's
SSE stream drops: Gradio abandons the generator (queueing.py 898-973 on Gradio
6.19) and the whole batch stops. So the loop lives here, in a **daemon thread
inside the server process**, decoupled from any browser connection. The UI
layer (WP3) only ever reads state back via :meth:`BatchRunner.snapshot_rows` /
:meth:`BatchRunner.summary` on a ``gr.Timer`` tick and never runs the loop
itself.

Source of truth
---------------
The canonical per-row state is the CSV manifest next to the audio files
(:mod:`gradio_ui.manifest`). Every state transition is flushed to that CSV via
:func:`gradio_ui.manifest.write_manifest_atomic`, so a crash / power loss / app
restart can resume exactly where it left off (Waiting/Failed/Generating rows
are re-run; Done/Skip are left alone).

Design invariants
-----------------
* **Never let one bad row kill the batch.** Every row is processed inside a
  try/except; any exception marks that row Failed and moves on.
* **Never leave the runner stuck "running".** The worker's outer body is a
  try/finally that always returns the runner to ``idle`` — even on an
  unexpected crash — so the overnight run can always be restarted.
* **No duplicate uploads.** Uploaded image IDs are cached by absolute local
  path for the whole run; the reference video (if any) is uploaded once.
* This module has NO ``gradio`` import — it is pure threading + HTTP (through
  the injected :class:`gradio_ui.api_client.ApiClient`) + the manifest data
  layer, so it stays trivially unit-testable with a mock transport and safe to
  run off the UI thread.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .handlers import build_a2v_chain_payload, suggest_frames_for_audio
from .manifest import (
    IMAGE_SHARED,
    STAT_DONE,
    STAT_FAILED,
    STAT_GENERATING,
    STAT_SKIP,
    STAT_WAITING,
    BatchRow,
    over_frame_limit,
    unique_output_name,
    write_manifest_atomic,
)

logger = logging.getLogger(__name__)

# Runner lifecycle states (distinct from the per-ROW STAT_* values above).
STATE_IDLE = "idle"
STATE_RUNNING = "running"
STATE_STOPPING = "stopping"

# Rows in one of these states are "unfinished" — the runner will (re)process
# them. STAT_GENERATING is included so a crash-remnant (a row that was mid-flight
# when the app died) is retried on resume rather than stranded.
_UNFINISHED = (STAT_WAITING, STAT_FAILED, STAT_GENERATING)

# generate_chain 409 (JOB_BUSY) retry policy: the server is single-job, so a
# 409 means a previous job has not yet been reaped. Back off a few seconds and
# retry a small number of times before giving up on the row.
_JOB_BUSY_MAX_ATTEMPTS = 3
_JOB_BUSY_BACKOFF_S = 3.0


# --------------------------------------------------------------------------- #
# BatchSnapshot — the WP3 -> WP2 interface contract.
# --------------------------------------------------------------------------- #
@dataclass
class BatchSnapshot:
    """Frozen copy of the Generate-tab settings at batch-start time.

    WP3 (the UI wiring) builds ONE of these when the user presses "Run batch",
    capturing every Generate-tab knob so the overnight run is immune to later
    UI edits, then hands it to :meth:`BatchRunner.start`. Everything here is a
    plain value (no Gradio component, no live state) so it can cross the thread
    boundary safely.

    Path fields
        wav_dir      Absolute path to the folder of source audio files. Each
                     row's ``wav`` filename is resolved against this.
        img_dir      Absolute path to the folder the per-row image choices were
                     drawn from (the UI's "Image folder path"). A row's
                     individually-named image (a bare filename) is resolved
                     against this when set; empty -> fall back to ``wav_dir``
                     (so an unset image folder keeps the old behaviour).
        out_dir      Absolute path to the (already resolved, see
                     :func:`gradio_ui.manifest.resolve_output_dir`) output
                     folder. Created on demand if missing.

    Prompt fields
        prompt_common  The Generate-tab prompt, used as the shared/common base.
        negative       Negative prompt (verbatim for every row).
        prompt_mode    "add"     -> per-row prompt is appended to the common one
                       "replace" -> per-row prompt replaces the common one
                       (empty per-row prompt always falls back to the common one).

    Generation geometry (identical to the frozen Generate-tab A2V send)
        width, height   ints (÷64, validated upstream).
        crop_output     Pre-computed ``{"width","height"}`` dict, or ``None``.
                        The runner passes this THROUGH unchanged — the "is crop
                        enabled" decision is made by WP3 when it snapshots.
        frame_rate      float fps.
        seed            int seed (applied verbatim to every row).

    LoRA / adapter
        loras           The FINAL merged lora list — WP3 must have already run
                        :func:`gradio_ui.handlers._combine_generate_loras`
                        (adapter-first, prompt order, name last-wins). The
                        runner does NOT merge; it forwards this list as-is.
        use_adapter     Whether a reference-video CONTROL adapter is in play.
        ref_video_path  Local path to the reference video, or ``None``. Uploaded
                        once (first row) and the id reused for the rest.
        control_adherence   float (conditioning_attention_strength; sent <1.0).
        reference_strength   float (reference_video_strength; sent <1.0).

    Shared keyframe images (Generate-tab common i2v slots)
        shared_images   ``list[(image_path, frame_idx, strength)]``. Used for
                        every row whose ``image`` column is the ``"Shared"``
                        sentinel. May be empty (pure A2V, no keyframe) — but see
                        the start-time guard in :meth:`BatchRunner.start`.

    Polling
        poll_interval   seconds between GET /jobs/{id} polls.
        poll_timeout_s  per-row polling ceiling in seconds (a row that has not
                        reached a terminal state by then is marked Failed).

    NAG (non-CFG Negative)
        nag_enabled, nag_scale, nag_tau, nag_alpha — forwarded to
        build_a2v_chain_payload for every row; defaults reproduce the pre-NAG
        payload (NAG off) byte-for-byte.
        neg_method, vsf_scale — the method selector + VSF's own
        param, forwarded the same way (only reach the payload when
        nag_enabled is True, per build_a2v_chain_payload's discipline).
    """

    wav_dir: str
    out_dir: str
    img_dir: str = ""
    prompt_common: str = ""
    negative: str = ""
    prompt_mode: str = "add"
    width: int = 512
    height: int = 320
    crop_output: Optional[dict] = None
    frame_rate: float = 24.0
    seed: int = -1
    loras: List[dict] = field(default_factory=list)
    shared_images: List[Tuple[str, int, float]] = field(default_factory=list)
    use_adapter: bool = False
    ref_video_path: Optional[str] = None
    control_adherence: float = 1.0
    reference_strength: float = 1.0
    poll_interval: float = 2.0
    poll_timeout_s: float = 7200.0
    nag_enabled: bool = False
    nag_scale: float = 11.0
    nag_tau: float = 2.5
    nag_alpha: float = 0.25
    neg_method: str = "nag"
    vsf_scale: float = 1.5


# --------------------------------------------------------------------------- #
# Prompt composition (pure function).
# --------------------------------------------------------------------------- #
def compose_prompt(common: str, row_prompt: str, mode: str) -> str:
    """Combine the batch's common prompt with a per-row prompt.

    * An empty (whitespace-only) ``row_prompt`` -> the ``common`` prompt
      verbatim, regardless of ``mode``.
    * ``mode == "replace"`` -> the ``row_prompt`` verbatim.
    * ``mode == "add"`` (default) -> ``f"{common} {row_prompt}"`` with both ends
      stripped (so an empty ``common`` yields just the row prompt).
    """
    common = common or ""
    row_prompt = row_prompt or ""
    if not row_prompt.strip():
        return common
    if mode == "replace":
        return row_prompt
    return f"{common} {row_prompt}".strip()


# --------------------------------------------------------------------------- #
# Error summarising (kept short so the CSV ``error`` column stays readable).
# --------------------------------------------------------------------------- #
def _summarize_exc(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:500]


def _summarize_job_error(job: dict) -> str:
    err = job.get("error")
    return (str(err)[:500]) if err else "failed"


def _resp_error(resp) -> str:
    try:
        body = resp.json()
    except Exception:
        body = resp.text
    return str(body)[:500]


# --------------------------------------------------------------------------- #
# BatchRunner — the in-process singleton worker.
# --------------------------------------------------------------------------- #
class BatchRunner:
    """Runs an A2V batch on a daemon thread, one row at a time, in queue order.

    Thread-safety: a single :class:`threading.Lock` guards the mutable state
    (``_state``, the ``_rows`` list contents, ``_current_job_id``, the id
    caches). Only the worker thread mutates row fields, so reads inside the
    worker are lock-free; every *externally visible* read
    (:meth:`snapshot_rows`, :meth:`summary`, :attr:`state`) takes the lock and
    returns copies. ``stop_event`` is a :class:`threading.Event` the UI sets to
    ask for a graceful stop between rows.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = STATE_IDLE
        self._rows: List[BatchRow] = []
        self._snapshot: Optional[BatchSnapshot] = None
        self._api = None
        self._thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self._current_job_id: Optional[str] = None
        # Per-run caches (reset on every start): absolute-image-path -> image_id,
        # and the single reference-video id.
        self._image_id_cache: dict = {}
        self._ref_video_id: Optional[str] = None

    # --- public state accessors ------------------------------------------- #
    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    def snapshot_rows(self) -> List[BatchRow]:
        """A copy of the current rows (for the UI's Timer re-render). Copies so
        the caller can never mutate the runner's canonical rows."""
        with self._lock:
            return [dataclasses.replace(r) for r in self._rows]

    def summary(self) -> dict:
        """Per-status counts + the runner state, for the UI status line."""
        with self._lock:
            counts = {STAT_DONE: 0, STAT_FAILED: 0, STAT_SKIP: 0,
                      STAT_WAITING: 0, STAT_GENERATING: 0}
            for r in self._rows:
                if r.stat in counts:
                    counts[r.stat] += 1
            return {
                "state": self._state,
                "done": counts[STAT_DONE],
                "failed": counts[STAT_FAILED],
                "skip": counts[STAT_SKIP],
                "waiting": counts[STAT_WAITING],
                "generating": counts[STAT_GENERATING],
            }

    # --- lifecycle -------------------------------------------------------- #
    def start(self, snapshot: BatchSnapshot, rows: List[BatchRow], api, *,
              sync: bool = False) -> Tuple[bool, str]:
        """Begin a batch run.

        Returns ``(started, reason)``. ``started=False`` when the runner is
        already busy (double-start guard) or a start-time validation fails
        (see below); ``reason`` is a human-readable string in either case.

        Start-time preflight (see :func:`_plan_rejudgement` / :func:`_validate`),
        run in this order so a rejected start leaves ``rows`` *and* the CSV
        completely untouched (only a start that actually proceeds mutates them):
          1. re-judge every unfinished row's frame count / 481-frame Skip at the
             snapshot's ``frame_rate`` (PROJECTED only — nothing mutated yet);
          2. validate against that projection: ``wav_dir`` must exist, at least
             one unfinished row must survive the re-judgment, and the image /
             prompt foolproof checks must pass;
          3. only once every check passes, APPLY the re-judgment (over-481 rows
             -> Skip, others -> refreshed frames), persist it to the CSV, and
             begin the run.

        ``sync=False`` (default) spawns a daemon thread and returns
        immediately; ``sync=True`` runs the whole batch inline before returning
        (used by the unit tests).
        """
        with self._lock:
            if self._state != STATE_IDLE:
                return False, "batch already running"
            # 1) project the frame/Skip re-judgment (pure — no mutation yet).
            plan = _plan_rejudgement(snapshot, rows)
            # 2) validate against that projection; on failure nothing is touched.
            ok, reason = _validate(snapshot, rows, plan)
            if not ok:
                return False, reason
            # 3) all checks passed -> it is now safe to mutate: apply the
            # re-judgment to the real rows before the run consumes them.
            applied = _apply_plan(rows, plan)
            # Take ownership under the lock (CAS on the idle state).
            self._state = STATE_RUNNING
            self._snapshot = snapshot
            self._rows = rows
            self._api = api
            self.stop_event.clear()
            self._current_job_id = None
            self._image_id_cache = {}
            self._ref_video_id = None

        # Persist the applied re-judgment (Skip re-marks / refreshed frames) so
        # the on-disk manifest matches the rows the run is about to process.
        if applied:
            self._write_csv()

        if sync:
            self._run()
            return True, "completed"

        self._thread = threading.Thread(
            target=self._run, name="batch-a2v-runner", daemon=True)
        self._thread.start()
        return True, "started"

    def request_stop(self) -> None:
        """Ask for a graceful stop: set ``stop_event`` (so no further rows are
        submitted) and best-effort cancel the in-flight job. A queued job
        cancels immediately (-> ``cancelled``, the row returns to Waiting); a
        running job may only be flagged and can still run to completion. Any
        error from the cancel is swallowed."""
        self.stop_event.set()
        with self._lock:
            if self._state == STATE_RUNNING:
                self._state = STATE_STOPPING
            job_id = self._current_job_id
            api = self._api
        if job_id and api is not None:
            try:
                api.delete_job(job_id)
            except Exception:
                logger.debug("batch: delete_job on stop failed", exc_info=True)

    # --- worker body ------------------------------------------------------ #
    def _run(self) -> None:
        try:
            self._process_all()
        except Exception:
            # Should be unreachable (every row is guarded), but keep the promise
            # airtight: a stray crash must never strand the runner as running.
            logger.exception("batch: worker crashed")
        finally:
            with self._lock:
                self._state = STATE_IDLE
                self._current_job_id = None
            # Final flush so the on-disk manifest matches the end state.
            self._write_csv()
            logger.info("batch: run finished (%s)", self.summary())

    def _process_all(self) -> None:
        # Iterate a stable snapshot of the row references in queue order. Each
        # row is processed at most once per run: a row we mark Failed is NOT
        # re-picked here (only a fresh start() — i.e. a resume — reprocesses
        # Failed rows).
        rows = sorted(list(self._rows), key=lambda r: r.queue)
        for row in rows:
            if self.stop_event.is_set():
                logger.info("batch: stop requested, halting before row %s", row.queue)
                break
            if row.stat not in _UNFINISHED:
                continue  # Done / Skip -> leave untouched, zero API calls
            self._process_row(row)

    def _process_row(self, row: BatchRow) -> None:
        api = self._api
        snap = self._snapshot
        logger.info("batch: start row %s (%s)", row.queue, row.wav)
        try:
            # 3) mark Generating + flush (locked CSV is non-fatal; keep going).
            self._set_and_write(row, stat=STAT_GENERATING, error="")

            # 4) upload audio (every row) + images (cached) + ref video (once).
            wav_path = os.path.join(snap.wav_dir, row.wav)
            audio_id = api.upload_audio(wav_path)
            conditioning = self._build_conditioning(row)
            ref_id = self._ensure_ref_video()

            # 5/6) build the byte-identical A2V chain payload + submit.
            prompt = compose_prompt(snap.prompt_common, row.prompt, snap.prompt_mode)
            payload = build_a2v_chain_payload(
                audio_id=audio_id,
                num_frames=row.frames,
                prompt=prompt,
                negative_prompt=snap.negative,
                width=snap.width,
                height=snap.height,
                crop_output=snap.crop_output,
                frame_rate=snap.frame_rate,
                seed=snap.seed,
                conditioning_images=conditioning,
                loras=snap.loras,
                use_adapter=snap.use_adapter,
                reference_video_id=ref_id,
                control_adherence=snap.control_adherence,
                reference_strength=snap.reference_strength,
                nag_enabled=snap.nag_enabled,
                nag_scale=snap.nag_scale,
                nag_tau=snap.nag_tau,
                nag_alpha=snap.nag_alpha,
                neg_method=snap.neg_method,
                vsf_scale=snap.vsf_scale,
            )
            job_id = self._submit_with_retry(payload)
            if job_id is None:
                self._set_and_write(row, stat=STAT_FAILED,
                                    error="server busy (409) after retries")
                logger.warning("batch: row %s failed — server busy", row.queue)
                return

            with self._lock:
                self._current_job_id = job_id

            # 7) poll to terminal (or timeout) and record the outcome.
            self._poll_row(row, job_id)
        except Exception as exc:
            # 8) any error on any row -> mark it Failed and move to the next.
            logger.exception("batch: row %s errored", row.queue)
            self._set_and_write(row, stat=STAT_FAILED, error=_summarize_exc(exc))
        finally:
            with self._lock:
                self._current_job_id = None

    # --- per-row helpers -------------------------------------------------- #
    def _build_conditioning(self, row: BatchRow) -> list:
        """Assemble the clip's ``conditioning_images`` (same shape as the frozen
        Generate-tab A2V path): a ``"Shared"`` row uses every snapshot shared
        image at its own frame/strength; an individually-named image attaches as
        a single frame-0, strength-1.0 keyframe."""
        snap = self._snapshot
        conditioning: list = []
        if row.image == IMAGE_SHARED:
            for image_path, frame_idx, strength in snap.shared_images:
                image_id = self._upload_image_cached(image_path)
                conditioning.append({
                    "image_id": image_id,
                    "frame_idx": int(frame_idx),
                    "strength": float(strength),
                })
        elif row.image:
            image_path = self._resolve_image_path(row.image)
            image_id = self._upload_image_cached(image_path)
            conditioning.append({
                "image_id": image_id, "frame_idx": 0, "strength": 1.0,
            })
        return conditioning

    def _resolve_image_path(self, name: str) -> str:
        """Resolve a per-row image reference to a local path. Absolute paths are
        used verbatim; a bare filename is joined against the snapshot's
        ``img_dir`` (the folder the UI's per-row image choices came from) when
        set, falling back to ``wav_dir`` for backwards compatibility (image
        folder == audio folder)."""
        if os.path.isabs(name):
            return name
        snap = self._snapshot
        base = snap.img_dir or snap.wav_dir
        return os.path.join(base, name)

    def _upload_image_cached(self, image_path: str) -> str:
        """Upload an image once per run, keyed by its absolute local path, so a
        shared keyframe reused across many rows is uploaded exactly once."""
        key = os.path.abspath(image_path)
        cached = self._image_id_cache.get(key)
        if cached is not None:
            return cached
        image_id = self._api.upload_image(image_path)
        self._image_id_cache[key] = image_id
        return image_id

    def _ensure_ref_video(self) -> Optional[str]:
        snap = self._snapshot
        if not snap.use_adapter or not snap.ref_video_path:
            return None
        if self._ref_video_id is None:
            self._ref_video_id = self._api.upload_video(snap.ref_video_path)
        return self._ref_video_id

    def _submit_with_retry(self, payload: dict) -> Optional[str]:
        """POST /generate/chain, retrying a 409 (JOB_BUSY) up to
        ``_JOB_BUSY_MAX_ATTEMPTS`` times with a fixed backoff. Returns the
        ``job_id`` on success, ``None`` if 409 persisted through every retry.
        Any other >=400 raises (caught by the row's handler -> Failed)."""
        for attempt in range(_JOB_BUSY_MAX_ATTEMPTS):
            resp = self._api.generate_chain(payload)
            if resp.status_code == 409:
                if attempt < _JOB_BUSY_MAX_ATTEMPTS - 1:
                    time.sleep(_JOB_BUSY_BACKOFF_S)
                continue
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"generate/chain {resp.status_code}: {_resp_error(resp)}")
            return resp.json()["job_id"]
        return None

    def _poll_row(self, row: BatchRow, job_id: str) -> None:
        """Poll GET /jobs/{id} until terminal or timeout, then record the row's
        outcome. ``cancelled`` (typically from a stop request) rewinds the row
        to Waiting so a later resume re-runs it; ``completed`` fetches + copies
        the mp4; ``failed`` / timeout mark it Failed."""
        api = self._api
        interval = self._snapshot.poll_interval
        timeout_s = self._snapshot.poll_timeout_s
        try:
            interval = float(interval)
            if interval < 0:
                interval = 0.0
            step = interval if interval > 0 else 1.0
            iterations = max(1, int(float(timeout_s) / step))
        except (TypeError, ValueError, ZeroDivisionError):
            interval, iterations = 1.0, 7200

        for _ in range(iterations):
            time.sleep(interval)
            try:
                job = api.get_job(job_id)
            except Exception:
                # Transient read error — keep polling; the job is still there.
                logger.debug("batch: get_job(%s) failed, retrying", job_id,
                             exc_info=True)
                continue
            status = job.get("status")
            if status == "completed":
                self._finish_completed(row, job_id)
                return
            if status == "failed":
                self._set_and_write(row, stat=STAT_FAILED,
                                    error=_summarize_job_error(job))
                logger.warning("batch: row %s job failed", row.queue)
                return
            if status == "cancelled":
                # Stop-driven cancel: rewind to Waiting for a clean resume.
                self._set_and_write(row, stat=STAT_WAITING, error="")
                logger.info("batch: row %s cancelled -> Waiting", row.queue)
                return
            # queued / running -> keep polling.
        self._set_and_write(row, stat=STAT_FAILED, error="poll timeout")
        logger.warning("batch: row %s timed out", row.queue)

    def _finish_completed(self, row: BatchRow, job_id: str) -> None:
        snap = self._snapshot
        tmp_path = self._api.fetch_video(job_id)
        out_dir = snap.out_dir
        os.makedirs(out_dir, exist_ok=True)
        out_name = unique_output_name(out_dir, row.wav)
        try:
            shutil.copy(tmp_path, os.path.join(out_dir, out_name))
        finally:
            # fetch_video() drops the mp4 into %TEMP%; delete it after the copy
            # (in a finally so a failed copy still cleans up) so an overnight
            # 200-clip run does not leave 200 temp mp4s behind. A remove failure
            # is non-fatal.
            try:
                os.remove(tmp_path)
            except OSError:
                logger.debug("batch: temp video cleanup failed: %s", tmp_path,
                             exc_info=True)
        self._set_and_write(row, stat=STAT_DONE, output=out_name, error="")
        logger.info("batch: row %s done -> %s", row.queue, out_name)

    # --- CSV flush -------------------------------------------------------- #
    def _set_and_write(self, row: BatchRow, **fields) -> None:
        """Apply field updates to a row under the lock, then flush the manifest."""
        with self._lock:
            for k, v in fields.items():
                setattr(row, k, v)
        self._write_csv()

    def _write_csv(self) -> None:
        with self._lock:
            snap = self._snapshot
            if snap is None:
                return
            rows_copy = [dataclasses.replace(r) for r in self._rows]
            wav_dir = snap.wav_dir
        try:
            res = write_manifest_atomic(wav_dir, rows_copy)
            if res.locked:
                logger.warning("batch: manifest locked, wrote autosave %s",
                               res.autosave_path)
        except Exception:
            logger.exception("batch: manifest write failed")


# --------------------------------------------------------------------------- #
# Start-time re-judgment (frames / 481-frame Skip), recomputed at the snapshot's
# frame rate. Pure + non-mutating: it returns a PLAN keyed by ``id(row)`` that
# the caller applies only after every validation check has passed, so a rejected
# start never half-mutates the rows / CSV. The raw-frame math itself lives in
# gradio_ui.manifest (over_frame_limit) — shared with the initial scan so the
# two never drift; the "which frame count does the app pick" policy stays in
# gradio_ui.handlers.suggest_frames_for_audio.
# --------------------------------------------------------------------------- #
def _plan_rejudgement(snapshot: BatchSnapshot, rows: List[BatchRow]) -> dict:
    """Project each unfinished row's re-judgment WITHOUT mutating anything.

    Returns ``{id(row): ("skip", "over-481f")}`` for a row whose duration now
    overruns the 481-frame cap at ``snapshot.frame_rate``, or
    ``{id(row): ("frames", n)}`` with its refreshed ``suggest_frames_for_audio``
    count otherwise. Rows with a non-positive ``duration_s`` (non-wav / already
    Skip remnants) are left out entirely — never touched."""
    fps = snapshot.frame_rate
    plan: dict = {}
    for r in rows:
        if r.stat not in _UNFINISHED or r.duration_s <= 0:
            continue
        if over_frame_limit(r.duration_s, fps):
            plan[id(r)] = ("skip", "over-481f")
        else:
            plan[id(r)] = ("frames", int(suggest_frames_for_audio(r.duration_s, fps)))
    return plan


def _apply_plan(rows: List[BatchRow], plan: dict) -> bool:
    """Apply a :func:`_plan_rejudgement` plan to ``rows`` in place. Returns True
    if any row's ``stat``/``skip_reason``/``frames`` actually changed (so the
    caller only re-flushes the CSV when there is something to persist)."""
    changed = False
    for r in rows:
        entry = plan.get(id(r))
        if entry is None:
            continue
        kind, val = entry
        if kind == "skip":
            if r.stat != STAT_SKIP or r.skip_reason != val:
                r.stat = STAT_SKIP
                r.skip_reason = val
                changed = True
        else:  # "frames"
            if r.frames != val:
                r.frames = val
                changed = True
    return changed


# --------------------------------------------------------------------------- #
# Start-time validation (module-level so it is testable in isolation).
#
# Structured top-down as a blanket check with a conditional branch at each leaf
# (per the owner's explicit preference) rather than a per-row enumeration: the
# processing targets are the unfinished rows that SURVIVE the projected Skip
# re-judgment (``plan``); the image + prompt foolproof gates then run against
# exactly that set.
# --------------------------------------------------------------------------- #
def _validate(snapshot: BatchSnapshot, rows: List[BatchRow],
              plan: Optional[dict] = None) -> Tuple[bool, str]:
    plan = plan or {}
    if not snapshot.wav_dir or not os.path.isdir(snapshot.wav_dir):
        return False, f"wav folder not found: {snapshot.wav_dir}"

    # Targets = unfinished rows that the projected re-judgment does NOT Skip.
    skip_ids = {rid for rid, (kind, _v) in plan.items() if kind == "skip"}
    targets = [r for r in rows
               if r.stat in _UNFINISHED and id(r) not in skip_ids]
    if not targets:
        return False, "no rows to process"

    # --- Image foolproof (blanket -> branch) ---
    # 1. the FIRST common keyframe (slot 1, frame_idx == 0) is set        -> OK
    # 2. it is not set -> every target must carry its OWN image; else fail.
    #    (a shared_images entry from slot 2+ alone, i.e. frame_idx > 0,
    #    does NOT satisfy the "Shared" rows — only the frame-0 slot does.)
    has_first_keyframe = any(int(frame_idx) == 0
                              for _path, frame_idx, _strength in snapshot.shared_images)
    if not has_first_keyframe:
        shared_targets = [r for r in targets if r.image == IMAGE_SHARED]
        if not all(r.image != IMAGE_SHARED for r in targets):
            return False, (
                "shared keyframe image(s) required: "
                f"{len(shared_targets)} row(s) reference '{IMAGE_SHARED}' but "
                "no shared image is set"
            )

    # --- Prompt foolproof (blanket -> branch) ---
    # 1. common prompt is non-empty                                  -> OK
    # 2. empty + mode="add"     -> nothing to send                   -> fail
    # 3. empty + mode="replace" -> every target row prompt must be
    #    non-empty; else fail (reason carries the empty-row count).
    if not (snapshot.prompt_common or "").strip():
        mode = snapshot.prompt_mode or "add"
        if mode == "add":
            return False, "prompt-empty-add"
        empty = [r for r in targets if not (r.prompt or "").strip()]
        if empty:
            return False, f"prompt-rows-empty:{len(empty)}"

    return True, "ok"


# --------------------------------------------------------------------------- #
# Process-wide singleton.
# --------------------------------------------------------------------------- #
_RUNNER: Optional[BatchRunner] = None
_RUNNER_LOCK = threading.Lock()


def get_runner() -> BatchRunner:
    """Return the process-wide :class:`BatchRunner` singleton (lazily created).

    The UI (WP3) and its ``gr.Timer`` share this ONE instance so the batch
    survives browser reconnects. Unit tests construct :class:`BatchRunner`
    directly for isolation rather than going through here."""
    global _RUNNER
    with _RUNNER_LOCK:
        if _RUNNER is None:
            _RUNNER = BatchRunner()
        return _RUNNER
