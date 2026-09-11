"""Supervises the object-tracking worker and owns the single tracking session.

Two things live here and nothing else does:

1. **The subprocess.** ``.venv-utils/Scripts/python.exe -u -m tracking.worker``,
   started lazily on the first session and kept resident afterwards (the model
   load is the expensive part, and it must not be repeated per session).
2. **The session.** Exactly one at a time, with the idle expiry, the byte-length
   check and the serialisation lock that the API contract promises.

WHY THIS IS NOT A SUBCLASS OF ``services/engines/ltx/adapter.py``'s _RealBackend
--------------------------------------------------------------------------------
The launch/teardown DISCIPLINE below is copied from it deliberately -- stderr to
a log file rather than a pipe (:1841-1845), a watchdog that kills the child so a
blocked read returns (:1617-1624), and the ``threading.Lock`` that serialises
every send/receive pair (:1537, used at :2108 and :2377) -- but the CLASS is not
reusable here:

* ``adapter.py:51`` imports ``GenerateRequest``; that module is welded to the
  generation vocabulary this feature has nothing to do with.
* ``SELECTION_FIELDS`` / ``REQUIRED_ASSETS`` / ``_build_load_payload`` (:1664)
  are all LTX base-model descriptor language. A tracker has one weight file and
  no categories.
* Tracking needs BINARY pipes for the raw frames, so ``_send`` (:1602) and
  ``_read_event`` (:1607) -- the only two methods worth inheriting -- would both
  be overridden. The reusable surface is empty.

THE LOCK IS LOAD-BEARING, NOT DEFENSIVE
---------------------------------------
FastAPI runs the frame endpoint through ``run_in_threadpool``, so two POSTs to
the same session genuinely execute on two threads. Without the lock, two
send/receive pairs interleave on one pipe, the length-prefixed framing loses its
alignment, and there is no way back short of killing the worker. The lock is
held across the WHOLE round trip for that reason -- taking it only around the
write would not help.

NO SHUTDOWN HOOK AT APP EXIT
----------------------------
By design, matching the LTX worker: closing our stdin is the signal, and the
worker exits on EOF (``engine/worker.py:1330-1331``). Adding a lifespan handler
would be a second, racier path to the same place.
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from api.errors import (
    track_busy,
    track_failed,
    track_frame_invalid,
    track_session_not_found,
    track_unavailable,
)
from config import AppConfig
from tracking.geometry import BYTES_PER_PIXEL, clamp_box_to_frame
from tracking.protocol import (
    EVENT_BOX,
    EVENT_ERROR,
    EVENT_READY,
    OP_INIT,
    OP_LOAD,
    OP_TRACK,
    ProtocolEOF,
    encode_header,
    read_message,
)

logger = logging.getLogger("ltx.tracking")

# --------------------------------------------------------------------------- #
# Module constants -- deliberately NOT config keys (design document §6.4).
# A knob is a promise to support every value of it; neither of these has a
# second value anyone would want.
# --------------------------------------------------------------------------- #

#: How long an untouched session survives. Evaluated LAZILY, when the next
#: session is opened -- there is no timer thread, because the only thing an
#: expired session blocks is the next ``open``.
IDLE_TIMEOUT_S = 60.0

#: Largest single frame accepted, in bytes. 4K RGBA is ~33 MB, so this leaves
#: room for anything AviUtl2 can render while still refusing a nonsense
#: width x height before it becomes an allocation.
MAX_FRAME_BYTES = 64 * 1024 * 1024

#: Worker stderr sink under ``config.log_dir``. Named for the venv, not for the
#: model: the next utility AI to live on ``.venv-utils`` logs here too.
_LOG_NAME = "utils_worker.log"

_WORKER_MODULE = "tracking.worker"

#: Model load. Generous: it is a ~107 MB safetensors read on a cold file cache.
_LOAD_TIMEOUT_S = 180.0
#: One frame. 1080p measures ~25 ms on this class of CPU; this is two orders of
#: magnitude of headroom, and exists only so a wedged worker cannot hang the
#: request thread forever.
_FRAME_TIMEOUT_S = 120.0

#: The two ``/status.tracking.reason`` values. Exactly two, by contract.
REASON_NOT_INSTALLED = "not installed"
REASON_WORKER_FAILED = "worker failed"

# --- mock backend behaviour (see _MockBackend) ------------------------------
_MOCK_STEP_PX = 1.0
_MOCK_LOST_FIRST = 10
_MOCK_LOST_LAST = 19
_MOCK_LOST_SCORE = 0.1
_MOCK_GOOD_SCORE = 0.9


class TrackingWorkerError(RuntimeError):
    """The worker died, timed out, or answered with an ``error`` event."""


class TrackingUnavailable(RuntimeError):
    """The worker could not be started. Carries the ``/status`` reason."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(detail or reason)
        self.reason = reason
        self.detail = detail or reason


@dataclass
class _Session:
    session_id: str
    width: int
    height: int
    frame_bytes: int
    seed_box: tuple[float, float, float, float]
    search_factor: float | None
    #: False until the first frame arrives; that frame is the ``init``.
    initialized: bool = False
    #: Frames accepted so far, the init frame included.
    frames: int = 0
    #: ``time.monotonic`` of the last accepted request on this session.
    touched_at: float = field(default_factory=time.monotonic)

    def public(self) -> dict:
        return {
            "session_id": self.session_id,
            "width": self.width,
            "height": self.height,
            "frame_bytes": self.frame_bytes,
        }


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #


class _MockBackend:
    """In-process fake tracker. No subprocess, no weights, no torch.

    Used by the whole test suite and by the agent-side end-to-end walkthrough,
    so the API layer can be exercised on a machine that has never run
    ``install-UETrack.bat``.

    The motion is a straight diagonal drift of one pixel per frame from the seed
    box, and frames 10..19 of a session come back with a low score. Both are
    arbitrary but FIXED: the plugin's lost-detection and smoothing are pure
    functions over exactly this kind of sequence, so a deterministic ramp with a
    known bad stretch is the cheapest thing that exercises them end to end.
    The index counted here is the session's own frame ordinal, NOT the ``frame=``
    query value -- the server does not interpret that one (design document §4.1).
    """

    name = "mock"

    def __init__(self) -> None:
        self._width = 0
        self._height = 0
        self._seed = (0.0, 0.0, 0.0, 0.0)
        self._index = 0

    def availability(self) -> tuple[bool, str | None]:
        return True, None

    def ensure_loaded(self) -> None:
        return None

    def init(
        self,
        *,
        width: int,
        height: int,
        box: tuple[float, float, float, float],
        search_factor: float | None,
        data: bytes,
    ) -> tuple[tuple[float, float, float, float], float]:
        self._width = width
        self._height = height
        self._seed = clamp_box_to_frame(box, width, height)
        self._index = 0
        return self._seed, 1.0

    def track(self, data: bytes) -> tuple[tuple[float, float, float, float], float]:
        self._index += 1
        x, y, w, h = self._seed
        moved = clamp_box_to_frame(
            (x + _MOCK_STEP_PX * self._index, y + _MOCK_STEP_PX * self._index, w, h),
            self._width,
            self._height,
        )
        lost = _MOCK_LOST_FIRST <= self._index <= _MOCK_LOST_LAST
        return moved, (_MOCK_LOST_SCORE if lost else _MOCK_GOOD_SCORE)


class _WorkerBackend:
    """The real thing: one resident ``tracking.worker`` subprocess.

    Every method here assumes the caller holds :class:`TrackingManager`'s lock,
    except :meth:`availability`, which only reads two paths and a flag and is
    called from ``/status`` without it.
    """

    name = "uetrack"

    def __init__(self, config: AppConfig, log_dir: Path) -> None:
        self.config = config
        self._log_dir = log_dir
        self._proc: subprocess.Popen | None = None
        self._log_fh = None
        self._log_path: Path | None = None
        #: Latched by a failed launch so ``/status`` can distinguish "you never
        #: installed this" from "it is installed and it will not start". Cleared
        #: by the next successful load.
        self._failed = False

    # ------------------------------------------------------------- paths

    def _python_path(self) -> Path:
        """``config._abs`` is private-by-name; using it here is accepted rather
        than met with a new public method on AppConfig for one caller."""
        return self.config._abs(self.config.tracking.utils_python)

    def _checkpoint_path(self) -> Path:
        t = self.config.tracking
        return self.config.models_dir / t.model_dir / t.checkpoint

    # -------------------------------------------------------- availability

    def availability(self) -> tuple[bool, str | None]:
        """Can a session be opened right now? Never starts anything.

        Order matters: "not installed" is checked FIRST so that a machine
        without ``.venv-utils`` is told to run the installer, not told that a
        worker it does not have failed to start.
        """
        if not self._python_path().is_file():
            return False, REASON_NOT_INSTALLED
        if not self._checkpoint_path().is_file():
            return False, REASON_NOT_INSTALLED
        if self._failed:
            return False, REASON_WORKER_FAILED
        return True, None

    @property
    def loaded(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    # ------------------------------------------------------------- launch

    def ensure_loaded(self) -> None:
        """Start the worker if it is not already running. Idempotent.

        A PREVIOUS failure does not bar a retry: ``_failed`` only colours what
        ``/status`` says, and the operator who just fixed their install should
        not have to restart the server to find out.

        The project root comes from ``config._abs``, private by name; as in
        :meth:`_python_path` that is accepted here rather than widened into a
        public method for one caller.
        """
        if self.loaded:
            return
        # A process that died between sessions leaves a handle to close before
        # the next launch opens another one.
        if self._proc is not None:
            self._teardown()
        if not self._python_path().is_file() or not self._checkpoint_path().is_file():
            raise TrackingUnavailable(
                REASON_NOT_INSTALLED,
                f"python={self._python_path()} checkpoint={self._checkpoint_path()}",
            )

        python = self._python_path()
        checkpoint = self._checkpoint_path()
        project_root = self.config._abs(".")
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self._log_dir / _LOG_NAME

        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        # `python -m tracking.worker` must find the FIRST-PARTY package, and
        # only it: cwd already puts the project root on sys.path, and pinning
        # PYTHONPATH to the same place keeps an inherited one from a developer's
        # shell out of the child.
        env.pop("PYTHONPATH", None)
        env["PYTHONPATH"] = str(project_root)

        logger.info("Starting tracking worker. python=%s checkpoint=%s", python, checkpoint)
        log_fh = open(self._log_path, "a", encoding="utf-8")
        try:
            self._proc = subprocess.Popen(
                [str(python), "-u", "-m", _WORKER_MODULE],
                cwd=str(project_root),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                # stderr to a FILE, never a pipe: nobody drains a stderr pipe
                # while this side is blocked reading stdout, and torch is
                # perfectly capable of filling 64 KB of warnings on import.
                stderr=log_fh,
                # Binary and unbuffered -- frames are raw RGBA. text=True would
                # corrupt them on the first byte that is not valid UTF-8, and a
                # buffer layer would hide a partially written frame.
                text=False,
                bufsize=0,
                env=env,
            )
        except Exception as exc:  # noqa: BLE001
            log_fh.close()
            self._proc = None
            self._failed = True
            raise TrackingUnavailable(
                REASON_WORKER_FAILED, f"failed to launch tracking worker: {exc!r}"
            ) from exc
        self._log_fh = log_fh

        try:
            # No "device": the worker is CPU-only and says so itself. A field
            # with one possible value is a way for the two sides to disagree.
            event = self._exchange(
                {"op": OP_LOAD, "checkpoint_path": str(checkpoint)},
                timeout=_LOAD_TIMEOUT_S,
            )
        except TrackingWorkerError as exc:
            # Kill first: _teardown only drops OUR handles, and a worker that
            # timed out or answered badly is still running otherwise.
            self._kill()
            self._teardown()
            self._failed = True
            raise TrackingUnavailable(REASON_WORKER_FAILED, str(exc)) from exc
        if event.get("event") != EVENT_READY:
            self._kill()
            self._teardown()
            self._failed = True
            raise TrackingUnavailable(
                REASON_WORKER_FAILED, f"unexpected event during load: {event!r}"
            )
        self._failed = False
        logger.info(
            "Tracking worker ready. model=%s device=%s",
            event.get("model"),
            event.get("device"),
        )

    # ---------------------------------------------------------------- I/O

    def _stderr_tail(self, n: int = 2000) -> str:
        """Best-effort tail of ``utils_worker.log`` (for error messages)."""
        if self._log_path is None:
            return ""
        try:
            return self._log_path.read_text(encoding="utf-8", errors="replace")[-n:]
        except Exception:  # noqa: BLE001
            return ""

    def _send(self, msg: dict, payload: bytes | None = None) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise TrackingWorkerError("tracking worker is not running")
        # The header line and the frame are written SEPARATELY. Joining them
        # would copy the whole 8.3 MB payload onto a ~100 byte line, once per
        # frame, to produce bytes the pipe is about to take in pieces anyway.
        # The reader does not care where our writes fall: it reads a line and
        # then the declared number of bytes.
        parts = [encode_header(msg, payload)]
        if payload:
            parts.append(payload)
        try:
            for part in parts:
                # Raw (bufsize=0) streams may report a short write. A pipe write
                # on a blocking handle normally consumes everything, but
                # "normally" is not a contract, and losing eight bytes of an
                # eight-megabyte frame would desynchronise the pipe permanently.
                view = memoryview(part)
                while view:
                    written = proc.stdin.write(view)
                    if not written:
                        raise TrackingWorkerError(
                            "tracking worker stdin accepted no bytes"
                        )
                    view = view[written:]
            proc.stdin.flush()
        except OSError as exc:
            # A dead worker shows up here as a broken pipe. It has to become a
            # TrackingWorkerError or it escapes the manager as a 500 instead of
            # the contract's TRACK_FAILED.
            raise TrackingWorkerError(
                f"tracking worker pipe write failed: {exc!r} {self._stderr_tail()}"
            ) from exc

    def _read_event(self, timeout: float | None = None) -> dict:
        """Block until the worker answers, killing it if ``timeout`` elapses.

        The watchdog has to KILL rather than signal: the read below is blocking
        on a pipe, and the only portable way to make it return is to close the
        far end. Copied from ``adapter.py:1607-1651``.
        """
        proc = self._proc
        if proc is None or proc.stdout is None:
            raise TrackingWorkerError("tracking worker is not running")
        timer: threading.Timer | None = None
        timed_out = {"v": False}
        if timeout is not None:

            def _kill_on_timeout() -> None:
                timed_out["v"] = True
                self._kill()

            timer = threading.Timer(timeout, _kill_on_timeout)
            timer.daemon = True
            timer.start()
        try:
            msg, _payload = read_message(
                proc.stdout,
                on_noise=lambda raw: logger.debug("ignored worker stdout noise: %r", raw[:200]),
            )
            return msg
        except ProtocolEOF as exc:
            if timed_out["v"]:
                raise TrackingWorkerError(
                    f"tracking worker timed out after {timeout:.0f}s: {self._stderr_tail()}"
                ) from exc
            raise TrackingWorkerError(
                f"tracking worker died: {self._stderr_tail()}"
            ) from exc
        except Exception as exc:  # noqa: BLE001
            if timed_out["v"]:
                raise TrackingWorkerError(
                    f"tracking worker timed out after {timeout:.0f}s: {self._stderr_tail()}"
                ) from exc
            raise TrackingWorkerError(
                f"tracking worker protocol failure: {exc!r} {self._stderr_tail()}"
            ) from exc
        finally:
            if timer is not None:
                timer.cancel()

    def _exchange(
        self, msg: dict, payload: bytes | None = None, timeout: float | None = None
    ) -> dict:
        """One send + one receive. The pairing the manager's lock protects."""
        self._send(msg, payload)
        return self._read_event(timeout=timeout)

    def _box_exchange(
        self, msg: dict, payload: bytes
    ) -> tuple[tuple[float, float, float, float], float]:
        event = self._exchange(msg, payload, timeout=_FRAME_TIMEOUT_S)
        kind = event.get("event")
        if kind == EVENT_BOX:
            return (
                (
                    float(event["x"]),
                    float(event["y"]),
                    float(event["w"]),
                    float(event["h"]),
                ),
                float(event["score"]),
            )
        if kind == EVENT_ERROR:
            # The worker is still alive and still loaded -- it caught this and
            # kept serving -- so the session dies but the process does not.
            raise TrackingWorkerError(f"tracking worker error: {event.get('detail', '')}")
        raise TrackingWorkerError(f"unexpected event from tracking worker: {event!r}")

    # ------------------------------------------------------------ tracking

    def init(
        self,
        *,
        width: int,
        height: int,
        box: tuple[float, float, float, float],
        search_factor: float | None,
        data: bytes,
    ) -> tuple[tuple[float, float, float, float], float]:
        msg = {
            "op": OP_INIT,
            "width": width,
            "height": height,
            "box": list(box),
            "search_factor": search_factor,
        }
        return self._box_exchange(msg, data)

    def track(self, data: bytes) -> tuple[tuple[float, float, float, float], float]:
        return self._box_exchange({"op": OP_TRACK}, data)

    # ------------------------------------------------------------ teardown

    def _kill(self) -> None:
        proc = self._proc
        if proc is None:
            return
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass

    def _teardown(self) -> None:
        self._proc = None
        if self._log_fh is not None:
            try:
                self._log_fh.close()
            except Exception:  # noqa: BLE001
                pass
            self._log_fh = None


# --------------------------------------------------------------------------- #
# Manager
# --------------------------------------------------------------------------- #


class TrackingManager:
    """The one object the API layer talks to. Built at startup, starts nothing.

    Construction is free and side-effect-free on purpose: ``AppContext`` builds
    one unconditionally, including in the two ``main.build_app`` calls the test
    suite makes, and a server that has never been asked to track must never have
    paid for a subprocess.
    """

    def __init__(self, config: AppConfig, log_dir: Path | None = None) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._session: _Session | None = None
        if config.tracking.backend == "mock":
            self._backend: _MockBackend | _WorkerBackend = _MockBackend()
        else:
            self._backend = _WorkerBackend(config, log_dir or config.log_dir)

    # ------------------------------------------------------------ status

    @property
    def backend_name(self) -> str:
        return self._backend.name

    def availability(self) -> tuple[bool, str | None]:
        """``(available, reason)``. ``reason`` is None when available."""
        try:
            return self._backend.availability()
        except Exception:  # noqa: BLE001 - /status must never fail over this
            logger.warning("tracking availability probe failed", exc_info=True)
            return False, REASON_NOT_INSTALLED

    def status_block(self) -> dict:
        """The ``tracking`` block of ``GET /status``.

        ``reason`` appears only when unavailable -- the contract is two keys at
        most, so a client can write ``tracking.reason`` without checking for a
        ``null`` that would mean the same as an absent key.
        """
        available, reason = self.availability()
        if available:
            return {"available": True}
        return {"available": False, "reason": reason}

    # ----------------------------------------------------------- sessions

    def open(
        self,
        *,
        width: int,
        height: int,
        box: tuple[float, float, float, float],
        search_factor: float | None = None,
    ) -> dict:
        """Open THE session. Returns the ``POST /utils/track/sessions`` body."""
        frame_bytes = int(width) * int(height) * BYTES_PER_PIXEL
        if frame_bytes > MAX_FRAME_BYTES:
            raise track_frame_invalid(
                detail=(
                    f"frame would be {frame_bytes} bytes "
                    f"({width}x{height}), limit is {MAX_FRAME_BYTES}"
                )
            )
        with self._lock:
            self._evict_if_idle_locked()
            if self._session is not None:
                raise track_busy(
                    detail=f"session {self._session.session_id} is still open"
                )
            # No separate availability pre-check: ``ensure_loaded`` is the one
            # place that decides, and asking twice would let the two answers
            # disagree (and would turn one failed launch into a permanent 503,
            # since a retry is exactly what should happen here).
            try:
                self._backend.ensure_loaded()
            except TrackingUnavailable as exc:
                logger.warning("tracking worker unavailable: %s", exc.detail)
                raise track_unavailable(detail=exc.detail) from exc

            session = _Session(
                session_id=uuid.uuid4().hex,
                width=int(width),
                height=int(height),
                frame_bytes=frame_bytes,
                seed_box=clamp_box_to_frame(box, int(width), int(height)),
                search_factor=search_factor,
            )
            self._session = session
            logger.info(
                "tracking session %s opened: %dx%d frame_bytes=%d box=%s",
                session.session_id,
                session.width,
                session.height,
                session.frame_bytes,
                [round(v, 2) for v in session.seed_box],
            )
            return session.public()

    def frame(
        self, session_id: str, frame_no: int, data: bytes
    ) -> tuple[dict, float]:
        """One frame in, one box out. Blocking -- call it from a worker thread.

        The FIRST frame of a session is its ``init``; every later one is a
        ``track``. The endpoint is the same either way, which is what lets the
        plugin's render loop send frame 1 down the same path as frame 240.
        """
        with self._lock:
            session = self._session
            if session is None or session.session_id != session_id:
                raise track_session_not_found(session_id)
            if len(data) != session.frame_bytes:
                raise track_frame_invalid(
                    detail=(
                        f"frame body is {len(data)} bytes, session {session_id} "
                        f"expects {session.frame_bytes}"
                    )
                )
            try:
                if session.initialized:
                    box, score = self._backend.track(data)
                else:
                    box, score = self._backend.init(
                        width=session.width,
                        height=session.height,
                        box=session.seed_box,
                        search_factor=session.search_factor,
                        data=data,
                    )
                    session.initialized = True
            except TrackingWorkerError as exc:
                # The session cannot continue -- the tracker's internal state is
                # gone or suspect -- so drop it here rather than make the plugin
                # guess. The process is left to the backend: an ``error`` event
                # means it is alive and can serve the next session.
                logger.error("tracking session %s failed: %s", session_id, exc)
                self._session = None
                raise track_failed(detail=str(exc)) from exc
            session.frames += 1
            session.touched_at = time.monotonic()
            # ``frame_no`` is AviUtl2's own numbering and is NOT interpreted
            # anywhere (the API echoes it back verbatim). It is logged so a
            # utils_worker.log can be lined up against the plugin's progress
            # events, and for nothing else.
            logger.debug(
                "tracking %s frame=%s -> box=%s score=%.3f",
                session_id,
                frame_no,
                [round(v, 2) for v in box],
                score,
            )
            return (
                {"x": box[0], "y": box[1], "w": box[2], "h": box[3]},
                float(score),
            )

    def close(self, session_id: str) -> dict:
        """Close THE session. Unknown id -> 404, including a second close."""
        with self._lock:
            session = self._session
            if session is None or session.session_id != session_id:
                raise track_session_not_found(session_id)
            logger.info(
                "tracking session %s closed after %d frame(s)",
                session.session_id,
                session.frames,
            )
            self._session = None
            return {"closed": True}

    # ------------------------------------------------------------ lifetime

    def _evict_if_idle_locked(self) -> None:
        """Drop a session nobody has touched for :data:`IDLE_TIMEOUT_S`.

        Lazy by design: the ONLY thing a stale session can block is the next
        ``open``, so this is the only moment the answer matters, and a timer
        thread would be a second lifetime to reason about. The constant is read
        here, not cached, so a test can shorten it.
        """
        session = self._session
        if session is None:
            return
        idle = time.monotonic() - session.touched_at
        # ">=" so that a timeout of zero means "always evict". The boundary is
        # otherwise academic (the shipped value is 60 seconds against a clock
        # whose resolution on Windows is ~16 ms), but it is what makes the rule
        # statable without reference to the clock.
        if idle < IDLE_TIMEOUT_S:
            return
        logger.warning(
            "evicting tracking session %s: idle for %.1fs (limit %.1fs)",
            session.session_id,
            idle,
            IDLE_TIMEOUT_S,
        )
        self._session = None
