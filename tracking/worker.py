"""Resident object-tracking worker. Runs on ``.venv-utils``, CPU only.

Launched by ``services/tracking_manager.py`` as::

    .venv-utils/Scripts/python.exe -u -m tracking.worker

and talked to over stdin/stdout with the framing in :mod:`tracking.protocol`.
One process serves every tracking session for the life of the server: the model
is loaded once, and a new session is just another ``init``.

THE SHAPE OF THE LOOP IS ``engine/worker.py``'s (:1279-1335), on purpose
-----------------------------------------------------------------------
Same four rules, so an operator who has debugged one worker has debugged both:

* before ``load``, every op except ``load``/``shutdown`` is ignored (logged);
* a failure DURING ``load`` emits ``error`` and exits 1 -- the parent must see
  the worker die rather than wait for a ``ready`` that is never coming;
* a failure AFTER ``load`` emits ``error`` and KEEPS SERVING -- one bad frame
  must not cost the next session a model load;
* EOF on stdin is a normal shutdown, which is why the parent needs no
  termination handshake at exit (it closes the pipe and we leave).

WHAT IS DIFFERENT, AND WHY
--------------------------
* **Binary pipes.** Frames are raw RGBA, so stdin/stdout are used through
  ``.buffer`` and never through ``print``. Anything written to ``sys.stdout``
  by accident would corrupt a frame boundary; that is why every diagnostic in
  this file goes to stderr (which the parent redirects to
  ``logs/utils_worker.log``).
* **No ``reset`` op.** Re-seeding is what ``init`` already does, and a second
  way to do it would be a second thing to keep correct.
* **torch is imported lazily**, inside the ``load`` handler. An import error
  (a half-built ``.venv-utils``, a torch DLL that will not load) then reaches
  the parent as an ``error`` event with a traceback, instead of killing this
  process before the protocol has said anything at all.
"""

from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime

from tracking.geometry import clamp_box_to_frame, rgba_bytes_to_rgb
from tracking.protocol import (
    EVENT_BOX,
    EVENT_ERROR,
    EVENT_READY,
    OP_INIT,
    OP_LOAD,
    OP_SHUTDOWN,
    OP_TRACK,
    ProtocolEOF,
    encode_message,
    read_message,
)

#: The one tracker this process owns. Never more than one: the API allows a
#: single session at a time (see the design document's session lifetime rules).
_TRACKER = None
#: Frame size of the CURRENT session, taken from ``init``. ``track`` carries no
#: size of its own -- the parent has already checked every frame's byte length
#: against it, so re-sending it per frame would only create a way for the two to
#: disagree.
_WIDTH = 0
_HEIGHT = 0


def _log(msg: str) -> None:
    """Diagnostics -> STDERR only (stdout is the binary protocol channel).

    Same timestamp format as ``engine/worker.py``'s ``_log`` so ``server.log``,
    ``ltx_worker.log`` and ``utils_worker.log`` line up by wall clock.
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
    print(f"[trk_worker] {ts} {msg}", file=sys.stderr, flush=True)


def _emit(event: str, **fields: object) -> None:
    """Write one framed event to STDOUT and flush.

    The same short-write loop as the parent's ``_send``
    (``services/tracking_manager.py``): a stream is allowed to take fewer bytes
    than it was given, and half an event on the pipe desynchronises it
    permanently. Events are small, so this loop almost always runs once.
    """
    out = sys.stdout.buffer
    view = memoryview(encode_message({"event": event, **fields}))
    while view:
        written = out.write(view)
        if not written:
            raise RuntimeError("stdout accepted no bytes")
        view = view[written:]
    out.flush()


def _detail(exc: BaseException) -> str:
    """repr + head/tail of the traceback, like ``engine/worker.py:_detail``.

    Both ends are kept because the OUTER frames say which of OUR call sites
    raised; a tail-only truncation leaves only some numpy or torch internal.
    """
    tb = "".join(traceback.format_exc())
    if len(tb) > 2800:
        tb = f"{tb[:1200]}\n...[traceback truncated]...\n{tb[-1600:]}"
    return f"{exc!r}\n{tb}"


def _do_load(msg: dict) -> None:
    """Handle ``load``: build the tracker and answer ``ready``."""
    global _TRACKER
    checkpoint_path = msg.get("checkpoint_path")
    if not checkpoint_path:
        raise ValueError("load message has no checkpoint_path")
    # Lazy on purpose -- see the module docstring.
    from tracking.uetrack_runtime import load_tracker

    # CPU, always. The load message carries no device: this worker exists to
    # stay off the GPU, so a field for it would only be a way to ask for
    # something that is refused.
    device = "cpu"
    _log(f"loading tracker: {checkpoint_path} device={device}")
    tracker = load_tracker(checkpoint_path, device=device)
    _TRACKER = tracker
    # Logged, not emitted: the parent has no use for the key lists, but whoever
    # is reading utils_worker.log after a checkpoint swap very much does. A
    # non-empty MISSING list means the vendored geometry drifted from the
    # weights (see load_tracker's docstring); the unexpected list is expected to
    # be non-empty only when the OFFICIAL .tar is read directly.
    _log(
        f"tracker ready: missing_keys={len(tracker.missing_keys)} "
        f"unexpected_keys={len(tracker.unexpected_keys)}"
    )
    if tracker.missing_keys:
        _log(f"MISSING KEYS: {json.dumps(tracker.missing_keys[:20])}")
    _emit(EVENT_READY, model="uetrack_base", device=device)


def _do_init(msg: dict, payload: bytes | None) -> None:
    """Handle ``init``: seed the tracker from the first frame of a session."""
    global _WIDTH, _HEIGHT
    if payload is None:
        raise ValueError("init message carries no frame payload")
    width = int(msg["width"])
    height = int(msg["height"])
    frame = rgba_bytes_to_rgb(payload, width, height)
    seed = clamp_box_to_frame(msg["box"], width, height)
    kwargs = {}
    # Absent -> uetrack_runtime.BASE_SEARCH_FACTOR decides. The default lives in
    # exactly one place and this is not it.
    if msg.get("search_factor") is not None:
        kwargs["search_factor"] = float(msg["search_factor"])
    _TRACKER.initialize(frame, seed, **kwargs)
    _WIDTH, _HEIGHT = width, height
    _log(f"init {width}x{height} box={[round(v, 2) for v in seed]} {kwargs}")
    # Score 1.0 by definition: the seed box is the user's, not a prediction, so
    # there is nothing to be confident or unconfident about. Keeping the reply
    # the same SHAPE as track's is what lets the plugin send frame 1 down the
    # same path as every other frame.
    _emit(EVENT_BOX, x=seed[0], y=seed[1], w=seed[2], h=seed[3], score=1.0)


def _do_track(payload: bytes | None) -> None:
    """Handle ``track``: advance one frame."""
    if payload is None:
        raise ValueError("track message carries no frame payload")
    if _TRACKER is None or _WIDTH <= 0:
        raise RuntimeError("track before init: no session is open on this worker")
    frame = rgba_bytes_to_rgb(payload, _WIDTH, _HEIGHT)
    box, score = _TRACKER.track(frame)
    box = clamp_box_to_frame(box, _WIDTH, _HEIGHT)
    _emit(EVENT_BOX, x=box[0], y=box[1], w=box[2], h=box[3], score=float(score))


def main() -> None:
    stdin = sys.stdin.buffer
    loaded = False
    while True:
        try:
            # No ``on_noise`` here, unlike the parent's side of the pipe: the
            # only writer of our stdin is the manager, which writes framed
            # messages and nothing else. There is no banner to tolerate, so
            # anything unexpected is a fault, not noise worth logging.
            msg, payload = read_message(stdin)
        except ProtocolEOF:
            break
        except Exception as exc:  # noqa: BLE001
            # A framing fault is unrecoverable: the stream position is unknown,
            # so there is no safe place to resume reading from.
            _log("PROTOCOL_FAILED")
            _emit(EVENT_ERROR, detail=_detail(exc))
            sys.exit(1)

        op = msg.get("op")

        if not loaded:
            if op == OP_SHUTDOWN:
                break
            if op != OP_LOAD:
                _log(f"ignoring op={op!r} before load")
                continue
            try:
                _do_load(msg)
                loaded = True
            except BaseException as exc:  # noqa: BLE001
                _log("LOAD_FAILED")
                _emit(EVENT_ERROR, detail=_detail(exc))
                sys.exit(1)
            continue

        # Serving loop (post-load). Every failure here keeps the process alive.
        if op == OP_INIT:
            try:
                _do_init(msg, payload)
            except BaseException as exc:  # noqa: BLE001
                _log("INIT_FAILED")
                _emit(EVENT_ERROR, detail=_detail(exc))
            continue
        if op == OP_TRACK:
            try:
                _do_track(payload)
            except BaseException as exc:  # noqa: BLE001
                _log("TRACK_FAILED")
                _emit(EVENT_ERROR, detail=_detail(exc))
            continue
        if op == OP_SHUTDOWN:
            break
        _log(f"ignoring unknown op={op!r}")

    _log("shutting down")
    sys.exit(0)


if __name__ == "__main__":
    main()
