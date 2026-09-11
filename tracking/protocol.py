"""Framing for the object-tracking worker pipe (app venv <-> .venv-utils).

PURE: no torch, no numpy, no project imports. It is imported by BOTH sides --
``services/tracking_manager.py`` on the app interpreter and ``tracking/worker.py``
on ``.venv-utils`` -- so the wire format has exactly one definition and cannot
drift between them.

THE FORMAT
----------
One message is the marker :data:`PREFIX`, one line of UTF-8 JSON, a newline,
and -- only when that JSON carries a ``"bytes": N`` key -- exactly N further
bytes of payload::

    @@TRK@@{"op":"track","bytes":8294400}
    <8294400 raw bytes, no trailing newline>

Both pipes are opened in binary with no buffering on the parent side, so
nothing here may assume text mode or newline translation.

WHY A BINARY PIPE AND NOT base64/JSON ARRAYS
--------------------------------------------
A 1080p RGBA frame is 8.3 MB. base64 would add ~10 ms of encode+decode per
frame against a ~25 ms inference (measured 2026-09-11) -- a 40% tax on the
whole loop to buy nothing. If a binary pipe ever proves unworkable on some
Windows configuration, base64 can be reintroduced INSIDE THIS MODULE alone: no
caller looks at the bytes on the wire.

WHY THE PAYLOAD LENGTH LIVES IN THE JSON
----------------------------------------
There is then a single length-bearing field instead of a JSON header plus a
separate binary length prefix, and a human reading a hexdump of the pipe sees
the frame size in the same line as the op.

SHORT READS ARE THE NORMAL CASE, NOT AN EDGE CASE
-------------------------------------------------
An 8 MB payload never arrives in one ``read()``: an OS pipe buffer is 64 KB.
:func:`read_message` therefore loops until it has the declared number of bytes
and treats a premature EOF as an error, never as a short message. This is the
one thing in this file that MUST NOT be "simplified".
"""

from __future__ import annotations

import json
from typing import Any, BinaryIO

__all__ = [
    "PREFIX",
    "BYTES_KEY",
    "OP_LOAD",
    "OP_INIT",
    "OP_TRACK",
    "OP_SHUTDOWN",
    "EVENT_READY",
    "EVENT_BOX",
    "EVENT_ERROR",
    "MAX_LINE_BYTES",
    "ProtocolError",
    "ProtocolEOF",
    "encode_header",
    "encode_message",
    "read_message",
]

#: Line marker. Distinct from the LTX worker's ``@@LTX@@`` so a log or a dump of
#: one worker's pipe can never be mistaken for the other's.
PREFIX = b"@@TRK@@"

#: The JSON key whose presence means "a raw payload follows this line".
BYTES_KEY = "bytes"

# --- ops (parent -> worker) ------------------------------------------------
OP_LOAD = "load"
OP_INIT = "init"
OP_TRACK = "track"
OP_SHUTDOWN = "shutdown"

# --- events (worker -> parent) ---------------------------------------------
EVENT_READY = "ready"
EVENT_BOX = "box"
EVENT_ERROR = "error"

#: Guard against a desynchronised pipe (or a foreign process on the other end)
#: making us buffer without bound while hunting for a newline. No legitimate
#: message comes close: the largest is an ``error`` event carrying a truncated
#: traceback, a few KB.
MAX_LINE_BYTES = 1 << 20


class ProtocolError(Exception):
    """The bytes on the pipe are not a well-formed message."""


class ProtocolEOF(ProtocolError):
    """The pipe closed cleanly on a message boundary.

    Not a fault on either side: it is how the worker learns the parent is gone
    (the parent closes stdin) and how the parent learns the worker exited. Both
    ends treat it as "stop", which is why it is its own type rather than a
    detail of :class:`ProtocolError`.
    """


def encode_header(obj: dict[str, Any], payload: bytes | None = None) -> bytes:
    """The marker + JSON line of one message, WITHOUT the payload after it.

    ONE RULE ABOUT PAYLOADS: a payload is either ``None`` or a non-empty byte
    string. An empty one is the same statement as no payload at all, so it gets
    no ``bytes`` key -- there is no third case for either end to handle.

    The caller must NOT put ``bytes`` in ``obj``: the length is derived from
    ``payload`` here so there is no second place for it to be wrong. Passing it
    anyway is a programming error and is refused rather than silently
    overwritten -- a mismatched length desynchronises the pipe permanently.

    Separate from :func:`encode_message` so a sender can write the line and the
    payload as two writes: concatenating an 8.3 MB frame onto a ~100 byte header
    copies the whole frame for nothing (``services/tracking_manager.py::_send``).
    """
    if BYTES_KEY in obj:
        raise ProtocolError(
            f"{BYTES_KEY!r} is derived from the payload; do not set it in the message"
        )
    body = dict(obj)
    if payload:
        body[BYTES_KEY] = len(payload)
    return PREFIX + json.dumps(body, separators=(",", ":")).encode("utf-8") + b"\n"


def encode_message(obj: dict[str, Any], payload: bytes | None = None) -> bytes:
    """One whole message as a single byte string: header line, then payload.

    Convenient where the payload is small or absent (the worker's events, the
    tests). The frame-sending path uses :func:`encode_header` and two writes
    instead, to avoid copying the frame.
    """
    line = encode_header(obj, payload)
    if not payload:
        return line
    return line + payload


def _read_line(stream: BinaryIO) -> bytes:
    """One newline-terminated line, read WITHOUT over-reading past it.

    Deliberately one byte at a time. ``stream`` may be a raw ``FileIO`` (the
    parent opens the pipes with ``bufsize=0``), and a raw stream has no place to
    put bytes it read past the newline -- they would be lost, taking the first
    bytes of the payload with them. A header line is ~100 bytes, i.e. ~100
    reads against a ~25 ms inference, so the cost is noise.
    """
    chunks = bytearray()
    while True:
        ch = stream.read(1)
        if not ch:
            if not chunks:
                raise ProtocolEOF("pipe closed on a message boundary")
            raise ProtocolError(
                f"pipe closed mid-line after {len(chunks)} bytes: {bytes(chunks[:120])!r}"
            )
        if ch == b"\n":
            return bytes(chunks)
        chunks += ch
        if len(chunks) > MAX_LINE_BYTES:
            raise ProtocolError(
                f"no newline within {MAX_LINE_BYTES} bytes; the pipe is desynchronised"
            )


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    """Exactly ``size`` bytes, looping over short reads (see module docstring)."""
    if size == 0:
        return b""
    buf = bytearray()
    while len(buf) < size:
        chunk = stream.read(size - len(buf))
        if not chunk:
            raise ProtocolError(f"pipe closed after {len(buf)} of {size} payload bytes")
        buf += chunk
    return bytes(buf)


def read_message(stream: BinaryIO, on_noise=None) -> tuple[dict[str, Any], bytes | None]:
    """Read one message. Returns ``(json_object, payload_or_None)``.

    Lines that do not start with :data:`PREFIX` are skipped, with the raw line
    handed to ``on_noise`` when one is given. This is the same tolerance the LTX
    adapter has (``services/engines/ltx/adapter.py:1607``) and for the same
    reason: a library on the other side may print to stdout, and a stray banner
    must not look like a dead worker. A skipped line can never carry a payload,
    so skipping cannot desynchronise the stream.

    Raises :class:`ProtocolEOF` at a clean end of stream and
    :class:`ProtocolError` for anything malformed.
    """
    while True:
        line = _read_line(stream)
        if line.startswith(PREFIX):
            break
        if on_noise is not None and line.strip():
            on_noise(line)

    try:
        obj = json.loads(line[len(PREFIX) :].decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - re-raised as a protocol fault
        raise ProtocolError(f"malformed JSON header: {line[:200]!r}") from exc
    if not isinstance(obj, dict):
        raise ProtocolError(f"message header is not a JSON object: {line[:200]!r}")

    if BYTES_KEY not in obj:
        return obj, None
    size = obj[BYTES_KEY]
    # Zero is refused with the negatives: a payload is None or non-empty (see
    # encode_header), so a declared length of 0 is a sender this reader has no
    # rule for -- and returning b"" would put back the third case on this side.
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise ProtocolError(f"invalid {BYTES_KEY!r}: {size!r}")
    return obj, _read_exact(stream, size)
