"""The three ``/utils/track/...`` endpoints and the session rules behind them.

Everything here runs on the MOCK tracking backend (``tests/conftest.py`` pins
``tracking.backend: "mock"`` for both apps it builds), so the suite exercises the
whole HTTP contract -- session lifetime, the byte-length check, the five error
codes, the serialisation lock -- on a machine that has never run
``install-UETrack.bat``. The real worker is covered by
``tests/test_tracking_runtime_smoke.py`` on ``.venv-utils`` and by the manual
end-to-end walkthrough.

The mock's motion is fixed and documented in ``services/tracking_manager.py``:
one pixel down-right per frame from the seed box, with a low score on frames
10..19 of the session.
"""

from __future__ import annotations

import argparse
import threading
import time

import anyio
import httpx
import pytest
import yaml
from conftest import build_model_layout
from fastapi.testclient import TestClient

import main
from services import tracking_manager

WIDTH, HEIGHT = 64, 48
FRAME_BYTES = WIDTH * HEIGHT * 4
FRAME = b"\x00" * FRAME_BYTES
SEED = {"x": 10.0, "y": 8.0, "w": 12.0, "h": 9.0}

SESSIONS = "/api/v1/utils/track/sessions"


def _open_body(**over) -> dict:
    body = {"width": WIDTH, "height": HEIGHT, "box": dict(SEED), "search_factor": 4.0}
    body.update(over)
    return body


def _open(client, **over):
    return client.post(SESSIONS, json=_open_body(**over))


def _code(response) -> str:
    return response.json()["error"]["code"]


# --------------------------------------------------------------------------- #
# the happy path
# --------------------------------------------------------------------------- #


def test_open_send_close(client):
    """One session, twenty-five frames, one close -- the whole plugin loop."""
    opened = _open(client)
    assert opened.status_code == 201
    body = opened.json()
    assert set(body) == {"session_id", "width", "height", "frame_bytes"}
    assert (body["width"], body["height"]) == (WIDTH, HEIGHT)
    # frame_bytes is the ONLY thing later frames are checked against, so it has
    # to be exactly width * height * 4 -- the plugin sizes its buffer from it.
    assert body["frame_bytes"] == FRAME_BYTES
    sid = body["session_id"]

    scores = []
    boxes = []
    for index in range(25):
        reply = client.post(f"{SESSIONS}/{sid}/frame?frame={1000 + index}", content=FRAME)
        assert reply.status_code == 200
        payload = reply.json()
        # The AviUtl2 frame number is echoed, never interpreted.
        assert payload["frame"] == 1000 + index
        assert set(payload["box"]) == {"x", "y", "w", "h"}
        boxes.append(payload["box"])
        scores.append(payload["score"])

    # Frame 0 IS the initialisation: the seed box back, with no opinion to have
    # about it.
    assert boxes[0] == SEED
    assert scores[0] == 1.0
    # ...and every later frame is a prediction that drifts by one pixel.
    assert boxes[5] == {"x": 15.0, "y": 13.0, "w": 12.0, "h": 9.0}
    # The mock's deliberate bad stretch, so the plugin's lost-detection has
    # something to find.
    assert [i for i, s in enumerate(scores) if s < 0.5] == list(range(10, 20))

    closed = client.delete(f"{SESSIONS}/{sid}")
    assert closed.status_code == 200
    assert closed.json() == {"closed": True}


def test_an_omitted_search_factor_reaches_the_worker_as_none(client, monkeypatch):
    """``search_factor`` is optional, and absent must stay absent.

    The default belongs to the runtime (``uetrack_runtime.BASE_SEARCH_FACTOR``),
    so nothing between the request body and the worker may fill one in: a
    default substituted here would be a second copy to keep in step with it.
    """
    manager = client.app_context.tracking_manager
    seen: dict = {}
    original = manager._backend.init

    def _record(**kwargs):
        seen.update(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(manager._backend, "init", _record)

    body = _open_body()
    del body["search_factor"]
    opened = client.post(SESSIONS, json=body)
    assert opened.status_code == 201
    sid = opened.json()["session_id"]
    assert client.post(f"{SESSIONS}/{sid}/frame?frame=0", content=FRAME).status_code == 200
    assert seen["search_factor"] is None


def test_a_closed_session_frees_the_slot(client):
    first = _open(client).json()["session_id"]
    client.delete(f"{SESSIONS}/{first}")
    second = _open(client)
    assert second.status_code == 201
    assert second.json()["session_id"] != first


# --------------------------------------------------------------------------- #
# the five error codes
# --------------------------------------------------------------------------- #


def test_second_session_is_busy(client):
    _open(client)
    refused = _open(client)
    assert refused.status_code == 409
    assert _code(refused) == "TRACK_BUSY"


def test_unknown_session_on_frame_is_404(client):
    reply = client.post(f"{SESSIONS}/does-not-exist/frame?frame=0", content=FRAME)
    assert reply.status_code == 404
    assert _code(reply) == "TRACK_SESSION_NOT_FOUND"


def test_unknown_session_on_delete_is_404(client):
    """Including the second DELETE of a session that was already closed: the id
    genuinely does not exist, and a soft answer would hide a plugin that lost
    track of its own session."""
    sid = _open(client).json()["session_id"]
    assert client.delete(f"{SESSIONS}/{sid}").status_code == 200
    again = client.delete(f"{SESSIONS}/{sid}")
    assert again.status_code == 404
    assert _code(again) == "TRACK_SESSION_NOT_FOUND"


@pytest.mark.parametrize(
    "size",
    [0, FRAME_BYTES - 1, FRAME_BYTES + 1],
    ids=["empty", "one-byte-short", "one-byte-long"],
)
def test_wrong_frame_length_is_400(client, size: int):
    """Raw RGBA carries no header, so the byte count is the only check there is
    -- and it has to be exact, not a lower bound."""
    sid = _open(client).json()["session_id"]
    reply = client.post(f"{SESSIONS}/{sid}/frame?frame=0", content=b"\x00" * size)
    assert reply.status_code == 400
    assert _code(reply) == "TRACK_FRAME_INVALID"


def test_a_rejected_frame_does_not_end_the_session(client):
    """A bad body is the caller's mistake, not a reason to lose the tracker."""
    sid = _open(client).json()["session_id"]
    client.post(f"{SESSIONS}/{sid}/frame?frame=0", content=FRAME)
    assert client.post(f"{SESSIONS}/{sid}/frame?frame=1", content=b"x").status_code == 400
    good = client.post(f"{SESSIONS}/{sid}/frame?frame=2", content=FRAME)
    assert good.status_code == 200
    assert good.json()["box"]["x"] == SEED["x"] + 1


def test_a_frame_larger_than_the_ceiling_is_refused_at_open(client):
    """The 64 MB ceiling is checked before anything is allocated: an 8K RGBA
    frame is 132 MB, and the failure should be a 400, not an OOM."""
    refused = _open(client, width=7680, height=4320)
    assert refused.status_code == 400
    assert _code(refused) == "TRACK_FRAME_INVALID"


def test_a_content_length_over_the_ceiling_is_refused_before_the_body(client):
    """The same ceiling on the frame endpoint, read from the HEADER.

    The oversized length is declared rather than sent: the point is that the
    refusal costs nothing, so a body big enough to hurt is exactly the body this
    test must not transmit.
    """
    sid = _open(client).json()["session_id"]
    reply = client.post(
        f"{SESSIONS}/{sid}/frame?frame=0",
        content=b"\x00" * 8,
        headers={"Content-Length": str(tracking_manager.MAX_FRAME_BYTES + 1)},
    )
    assert reply.status_code == 400
    assert _code(reply) == "TRACK_FRAME_INVALID"
    # The HEADER is what refused it. Without this line the test would pass just
    # as well on the manager's exact-length check, which reads the body first --
    # the very thing this check exists to avoid.
    assert "Content-Length" in reply.json()["error"]["detail"]


def test_a_worker_failure_returns_503_and_drops_the_session(client, monkeypatch):
    """The contract's fifth code. The session dies so the plugin can open a new
    one; the process is the backend's business."""
    manager = client.app_context.tracking_manager
    sid = _open(client).json()["session_id"]
    client.post(f"{SESSIONS}/{sid}/frame?frame=0", content=FRAME)

    def _boom(_data):
        raise tracking_manager.TrackingWorkerError("worker died mid-track")

    monkeypatch.setattr(manager._backend, "track", _boom)
    reply = client.post(f"{SESSIONS}/{sid}/frame?frame=1", content=FRAME)
    assert reply.status_code == 503
    assert _code(reply) == "TRACK_FAILED"
    assert "worker died mid-track" in reply.json()["error"]["detail"]
    # The slot is free again -- no operator intervention needed.
    assert _open(client).status_code == 201


@pytest.mark.parametrize(
    "body",
    [
        {"width": 0, "height": 48, "box": SEED},
        {"width": 64, "height": 48, "box": {"x": 0, "y": 0, "w": 0, "h": 10}},
        {"width": 64, "height": 48},
        # Above the server-side ceiling on the search window (the panel's own
        # range stops at 6.0; this is the outer edge).
        {"width": 64, "height": 48, "box": SEED, "search_factor": 8.5},
    ],
)
def test_a_malformed_open_body_is_422(client, body: dict):
    """Shape errors stay in FastAPI's validation layer: the five tracking codes
    describe STATES, not typos."""
    assert client.post(SESSIONS, json=body).status_code == 422


# --------------------------------------------------------------------------- #
# session lifetime
# --------------------------------------------------------------------------- #


def test_an_idle_session_is_evicted_by_the_next_open(client, monkeypatch):
    """The expiry is evaluated lazily, on the next open -- there is no timer
    thread, and this is the only moment the answer can matter."""
    monkeypatch.setattr(tracking_manager, "IDLE_TIMEOUT_S", 0.0)
    stale = _open(client).json()["session_id"]
    fresh = _open(client)
    assert fresh.status_code == 201
    assert fresh.json()["session_id"] != stale
    # The evicted id is gone, not merely superseded.
    gone = client.post(f"{SESSIONS}/{stale}/frame?frame=0", content=FRAME)
    assert gone.status_code == 404


def test_a_frame_in_between_is_what_saves_a_session_from_eviction(client, monkeypatch):
    """The clock is reset by every accepted frame, so a slow render never loses
    a session that is still being fed.

    Stated as a contrast over the SAME wait: with a frame in the middle the
    session survives, without one it is evicted. Asserting only the first half
    would pass just as well if the expiry never fired at all.
    """
    monkeypatch.setattr(tracking_manager, "IDLE_TIMEOUT_S", 0.05)
    wait = 0.08

    # Fed: the frame lands mid-wait and moves the deadline with it.
    fed = _open(client).json()["session_id"]
    time.sleep(wait)
    assert client.post(f"{SESSIONS}/{fed}/frame?frame=0", content=FRAME).status_code == 200
    assert _code(_open(client)) == "TRACK_BUSY"
    assert client.delete(f"{SESSIONS}/{fed}").status_code == 200

    # Starved: the same wait, no frame, and the next open takes the slot.
    stale = _open(client).json()["session_id"]
    time.sleep(wait)
    fresh = _open(client)
    assert fresh.status_code == 201
    assert fresh.json()["session_id"] != stale


# --------------------------------------------------------------------------- #
# the serialisation lock
# --------------------------------------------------------------------------- #


#: How long the instrumented backend stays "busy" per frame. Long enough that
#: two unserialised calls cannot miss each other by luck: the failure being
#: tested is two threads inside the backend AT THE SAME TIME, and a call that
#: returns in microseconds can interleave without ever overlapping.
_HOLD_S = 0.005


def _record_spans(monkeypatch, backend, spans: list[tuple[float, float]]) -> None:
    """Make ``backend.track`` record the interval it spent serving each frame.

    The assertion is then about the intervals themselves (see
    :func:`_assert_never_overlapped`) rather than about the mock's counter
    coming out unique -- a counter can survive a race by accident, and its
    uniqueness says nothing about the pipe the real worker writes to.
    """
    original = backend.track
    append_lock = threading.Lock()

    def _timed(data):
        entered = time.monotonic()
        time.sleep(_HOLD_S)
        result = original(data)
        left = time.monotonic()
        with append_lock:
            spans.append((entered, left))
        return result

    monkeypatch.setattr(backend, "track", _timed)


def _assert_never_overlapped(spans: list[tuple[float, float]]) -> None:
    """No two recorded intervals share a moment."""
    ordered = sorted(spans)
    for (_, earlier_left), (later_entered, _) in zip(ordered, ordered[1:]):
        assert earlier_left <= later_entered, (
            f"two tracking calls were inside the backend at once: {ordered}"
        )


def test_concurrent_frames_on_one_session_are_serialised(tmp_path, monkeypatch):
    """Two POSTs in flight at once must never be inside the backend together.

    This is the failure the lock exists for. FastAPI runs the frame endpoint in
    a threadpool, so two send/receive pairs genuinely execute on two threads,
    and on the real worker that means two writes interleaved on one pipe with no
    way back short of killing it. The backend is instrumented to record when it
    was entered and when it left, and those intervals must be disjoint.

    Driven through ``httpx.ASGITransport`` + ``anyio`` rather than
    ``TestClient`` because the concurrency has to be real (the same convention
    as ``tests/test_mcp_tools_system.py``).
    """
    from conftest import _build_app

    app = _build_app(tmp_path)
    spans: list[tuple[float, float]] = []
    _record_spans(monkeypatch, app.state.context.tracking_manager._backend, spans)
    posts = 16

    async def _run() -> list[dict]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
            opened = await ac.post(SESSIONS, json=_open_body())
            sid = opened.json()["session_id"]
            # Frame 0 is the init; keep it out of the race so every request
            # below is the same operation.
            await ac.post(f"{SESSIONS}/{sid}/frame?frame=0", content=FRAME)

            results: list[dict] = []

            async def _one(index: int) -> None:
                reply = await ac.post(
                    f"{SESSIONS}/{sid}/frame?frame={index}", content=FRAME
                )
                assert reply.status_code == 200, reply.text
                results.append(reply.json())

            async with anyio.create_task_group() as tg:
                for index in range(1, posts + 1):
                    tg.start_soon(_one, index)
            await ac.delete(f"{SESSIONS}/{sid}")
            return results

    results = anyio.run(_run)
    assert len(results) == posts
    assert len(spans) == posts
    _assert_never_overlapped(spans)


def test_the_manager_lock_survives_two_threads_on_one_session(client, monkeypatch):
    """The same guarantee one layer down, without HTTP in the picture."""
    manager = client.app_context.tracking_manager
    spans: list[tuple[float, float]] = []
    _record_spans(monkeypatch, manager._backend, spans)

    sid = _open(client).json()["session_id"]
    manager.frame(sid, 0, FRAME)  # init -- goes to `init`, not `track`

    pushes = 10

    def _push() -> None:
        for _ in range(pushes):
            manager.frame(sid, 0, FRAME)

    threads = [threading.Thread(target=_push) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(spans) == 2 * pushes
    _assert_never_overlapped(spans)


# --------------------------------------------------------------------------- #
# /status and availability
# --------------------------------------------------------------------------- #


def test_status_reports_tracking_available(client):
    status = client.get("/api/v1/status").json()
    # Two keys at most, and no "reason" when it is available.
    assert status["tracking"] == {"available": True}
    # The blocks the contract says must not move.
    assert status["queue"]["mode"] == "single_job_in_memory"
    assert "state" in status


def test_all_three_endpoints_require_the_api_key(tmp_path):
    """Tracking is behind the same bearer token as everything else.

    Worth its own test because the three routes declare ``require_auth``
    individually -- there is no router-wide dependency to fall back on, so an
    endpoint added later without the line would simply be open.
    """
    app = _tracking_app(tmp_path, api_key="s3cret", backend="mock")
    with TestClient(app) as c:
        assert c.post(SESSIONS, json=_open_body()).status_code == 401
        assert c.post(f"{SESSIONS}/x/frame?frame=0", content=FRAME).status_code == 401
        assert c.delete(f"{SESSIONS}/x").status_code == 401
        # ...and with the token, the same calls reach the tracking logic.
        auth = {"Authorization": "Bearer s3cret"}
        opened = c.post(SESSIONS, json=_open_body(), headers=auth)
        assert opened.status_code == 201
        sid = opened.json()["session_id"]
        assert c.post(f"{SESSIONS}/{sid}/frame?frame=0", content=FRAME, headers=auth).status_code == 200
        assert c.delete(f"{SESSIONS}/{sid}", headers=auth).status_code == 200


def _tracking_app(tmp_path, *, api_key: str | None = None, backend: str = "uetrack"):
    """An app whose ``tracking`` section this test chose.

    ``conftest``'s fixtures always pin the mock; this builds the other cases --
    the REAL backend pointed at a ``.venv-utils`` that is not there (the shape of
    a machine that never ran ``install-UETrack.bat``), and an app with a bearer
    token configured.
    """
    cfg = {
        "server": {"log_dir": (tmp_path / "logs").as_posix()},
        "model": {"backend": "mock", **build_model_layout(tmp_path)},
        "output": {"dir": (tmp_path / "outputs").as_posix()},
        "upload": {"dir": (tmp_path / "uploads").as_posix()},
        "state_file": (tmp_path / "state.json").as_posix(),
        "tracking": {
            "backend": backend,
            "utils_python": (tmp_path / "no-such-venv" / "python.exe").as_posix(),
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    args = argparse.Namespace(
        listen=False, port=None, api_key=api_key, allow_all_cors=False,
        config=path.as_posix(), te_offload=None, dit_cpu_load=None,
    )
    return main.build_app(args)


def _uninstalled_app(tmp_path):
    return _tracking_app(tmp_path, backend="uetrack")


def test_status_says_not_installed_when_the_venv_is_missing(tmp_path):
    with TestClient(_uninstalled_app(tmp_path)) as c:
        assert c.get("/api/v1/status").json()["tracking"] == {
            "available": False,
            "reason": "not installed",
        }


def test_opening_a_session_without_the_module_is_503(tmp_path):
    with TestClient(_uninstalled_app(tmp_path)) as c:
        refused = c.post(SESSIONS, json=_open_body())
        assert refused.status_code == 503
        assert _code(refused) == "TRACK_UNAVAILABLE"


def test_building_the_manager_starts_no_process(tmp_path, monkeypatch):
    """The whole point of lazy launch: every app this suite builds constructs a
    TrackingManager, and none of them may cost a subprocess."""
    import subprocess

    def _forbidden(*_a, **_k):
        raise AssertionError("TrackingManager must not spawn anything at construction")

    monkeypatch.setattr(subprocess, "Popen", _forbidden)
    app = _uninstalled_app(tmp_path)
    assert app.state.context.tracking_manager.backend_name == "uetrack"


def test_a_failed_launch_is_reported_as_worker_failed(tmp_path):
    """The second (and last) ``reason`` value, produced by a real failed launch.

    "not installed" is checked first on purpose: a machine with no
    ``.venv-utils`` must be told to run the installer, not told that a worker it
    does not have refused to start. So this case needs both files to EXIST --
    and the interpreter here exists and is not one, which is what Windows
    refuses to start. Driven through ``open()`` rather than by setting the
    ``_failed`` flag by hand: the flag being SET is the thing under test, and a
    test that sets it only checks that ``/status`` can read it back.
    """
    from api.errors import APIError
    from config import AppConfig
    from services.tracking_manager import TrackingManager

    fake_python = tmp_path / "venv" / "python.exe"
    fake_python.parent.mkdir(parents=True)
    fake_python.write_bytes(b"not really an interpreter")
    weights = tmp_path / "models" / "UETrack" / "uetrack_base.safetensors"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"not really a checkpoint")

    config = AppConfig.model_validate(
        {
            "model": {"models_dir": (tmp_path / "models").as_posix()},
            "tracking": {"backend": "uetrack", "utils_python": fake_python.as_posix()},
        }
    )
    manager = TrackingManager(config, log_dir=tmp_path / "logs")
    # Both files are there, so nothing yet says this install is broken.
    assert manager.availability() == (True, None)

    with pytest.raises(APIError) as refused:
        manager.open(width=WIDTH, height=HEIGHT, box=(10.0, 8.0, 12.0, 9.0))
    assert refused.value.code == "TRACK_UNAVAILABLE"
    assert refused.value.status_code == 503

    # ...and the launch failure is remembered, with the other reason.
    assert manager.status_block() == {"available": False, "reason": "worker failed"}
