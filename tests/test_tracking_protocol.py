"""Wire format of the tracking worker pipe (``tracking/protocol.py``).

This module is the ONE place the frame framing is defined, and both ends import
it, so these tests are what stands between the feature and a class of bug that
has no good failure mode: a desynchronised binary pipe does not raise, it hands
the tracker half a frame and the other half of the next one.

The awkward cases are therefore the point. In particular
:func:`test_payload_arrives_in_fragments` and
:func:`test_header_line_arrives_in_fragments` drive the reader through a stream
that never returns everything asked for -- which is not a pathological case but
the NORMAL one for an 8 MB frame over a 64 KB OS pipe buffer.
"""

from __future__ import annotations

import io
import json

import pytest

from tracking import protocol


class _DribbleStream(io.RawIOBase):
    """A binary stream that returns at most ``chunk`` bytes per ``read``.

    Stands in for an OS pipe: ``read(n)`` on one is free to return less than
    ``n``, and the only defence is looping until the declared length is there.
    """

    def __init__(self, data: bytes, chunk: int = 1) -> None:
        self._data = data
        self._chunk = chunk
        self._pos = 0

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = len(self._data) - self._pos
        take = min(size, self._chunk, len(self._data) - self._pos)
        out = self._data[self._pos : self._pos + take]
        self._pos += take
        return out


def test_round_trip_without_payload():
    blob = protocol.encode_message({"op": protocol.OP_SHUTDOWN})
    obj, payload = protocol.read_message(io.BytesIO(blob))
    assert obj == {"op": "shutdown"}
    # None, not b"": "this message has no payload" and "this message has an
    # empty payload" are different statements about the sender.
    assert payload is None


def test_round_trip_with_payload():
    blob = protocol.encode_message({"op": protocol.OP_TRACK}, b"\x00\x01\x02\x03")
    obj, payload = protocol.read_message(io.BytesIO(blob))
    assert obj == {"op": "track", "bytes": 4}
    assert payload == b"\x00\x01\x02\x03"


def test_an_empty_payload_is_the_same_as_no_payload():
    """The one rule: a payload is ``None`` or a non-empty byte string.

    ``b""`` therefore produces no ``bytes`` key at all, so neither end has a
    third case to tell apart from the other two.
    """
    blob = protocol.encode_message({"op": "x"}, b"")
    assert blob == protocol.encode_message({"op": "x"})
    obj, payload = protocol.read_message(io.BytesIO(blob))
    assert protocol.BYTES_KEY not in obj
    assert payload is None


def test_encode_refuses_a_caller_supplied_length():
    """The length is derived from the payload; a second source may disagree."""
    with pytest.raises(protocol.ProtocolError):
        protocol.encode_message({"op": "track", "bytes": 99}, b"ab")


def test_payload_may_contain_the_prefix_and_newlines():
    """Frames are arbitrary bytes -- the framing must not be self-terminating.

    A raw RGBA frame is perfectly capable of containing ``@@TRK@@`` and any
    number of ``\\n`` bytes. Because the length is declared up front the reader
    never looks inside the payload, which is exactly why the format has a length
    at all instead of a sentinel.
    """
    nasty = protocol.PREFIX + b'{"event":"box"}\n' + b"\n\n" + b"\x00\xff"
    blob = protocol.encode_message({"op": "track"}, nasty)
    obj, payload = protocol.read_message(io.BytesIO(blob))
    assert payload == nasty
    assert obj["bytes"] == len(nasty)


def test_eight_megabyte_payload_round_trips():
    """1080p RGBA, the size this feature actually moves per frame."""
    frame = bytes(1920 * 1080 * 4 % 251 for _ in range(8)) * (1024 * 1024)
    assert len(frame) == 8 * 1024 * 1024
    blob = protocol.encode_message({"op": protocol.OP_TRACK}, frame)
    obj, payload = protocol.read_message(io.BytesIO(blob))
    assert obj["bytes"] == len(frame)
    assert payload == frame


def test_payload_arrives_in_fragments():
    blob = protocol.encode_message({"op": "track"}, bytes(range(256)) * 40)
    obj, payload = protocol.read_message(_DribbleStream(blob, chunk=7))
    assert obj["bytes"] == 256 * 40
    assert payload == bytes(range(256)) * 40


def test_header_line_arrives_one_byte_at_a_time():
    blob = protocol.encode_message({"op": "init", "width": 1920, "height": 1080})
    obj, payload = protocol.read_message(_DribbleStream(blob, chunk=1))
    assert obj == {"op": "init", "width": 1920, "height": 1080}
    assert payload is None


def test_two_messages_back_to_back_on_one_stream():
    """The reader must stop at its own boundary, not swallow what follows."""
    stream = io.BytesIO(
        protocol.encode_message({"op": "track"}, b"AAAA")
        + protocol.encode_message({"op": protocol.OP_SHUTDOWN})
    )
    first, payload = protocol.read_message(stream)
    assert (first["op"], payload) == ("track", b"AAAA")
    second, payload2 = protocol.read_message(stream)
    assert (second["op"], payload2) == ("shutdown", None)


def test_clean_eof_raises_protocol_eof():
    """How both ends learn the other is gone. Not a fault; a stop signal."""
    with pytest.raises(protocol.ProtocolEOF):
        protocol.read_message(io.BytesIO(b""))


def test_eof_midway_through_a_line_is_an_error_not_an_eof():
    with pytest.raises(protocol.ProtocolError) as exc:
        protocol.read_message(io.BytesIO(protocol.PREFIX + b'{"op":"tr'))
    assert not isinstance(exc.value, protocol.ProtocolEOF)


def test_eof_midway_through_a_payload_is_an_error():
    blob = protocol.encode_message({"op": "track"}, b"12345678")
    with pytest.raises(protocol.ProtocolError) as exc:
        protocol.read_message(io.BytesIO(blob[:-3]))
    assert "5 of 8" in str(exc.value)


def test_non_protocol_lines_are_skipped_and_reported():
    """A library banner on stdout must not look like a dead worker."""
    seen: list[bytes] = []
    stream = io.BytesIO(b"warning: something\n" + protocol.encode_message({"op": "x"}))
    obj, _ = protocol.read_message(stream, on_noise=seen.append)
    assert obj == {"op": "x"}
    assert seen == [b"warning: something"]


def test_malformed_json_header_is_a_protocol_error():
    with pytest.raises(protocol.ProtocolError):
        protocol.read_message(io.BytesIO(protocol.PREFIX + b"{not json}\n"))


def test_non_object_header_is_a_protocol_error():
    with pytest.raises(protocol.ProtocolError):
        protocol.read_message(io.BytesIO(protocol.PREFIX + b"[1,2,3]\n"))


@pytest.mark.parametrize("size", [-1, 0])
def test_a_length_that_is_not_positive_is_refused(size: int):
    """Zero belongs with the negatives: no sender of this format emits it."""
    line = protocol.PREFIX + json.dumps({"op": "track", "bytes": size}).encode() + b"\n"
    with pytest.raises(protocol.ProtocolError):
        protocol.read_message(io.BytesIO(line))


def test_a_line_without_a_newline_is_bounded():
    """A desynchronised pipe must not be able to make us buffer without limit."""
    with pytest.raises(protocol.ProtocolError) as exc:
        protocol.read_message(io.BytesIO(protocol.PREFIX + b"x" * (protocol.MAX_LINE_BYTES + 8)))
    assert "desynchronised" in str(exc.value)


def test_header_is_one_line_of_utf8_json_after_the_marker():
    """Pins the shape itself: the plugin-side log and any hexdump depend on it."""
    blob = protocol.encode_message({"op": "track"}, b"xy")
    line, rest = blob.split(b"\n", 1)
    assert line.startswith(protocol.PREFIX)
    assert json.loads(line[len(protocol.PREFIX) :].decode("utf-8")) == {
        "op": "track",
        "bytes": 2,
    }
    assert rest == b"xy"
