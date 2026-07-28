"""Contract tests for mcp_server/tools/jobs.py (W3, 7 tools).

Same two-double pattern as the other mcp_server tool test modules:
  * ``_client_for_app`` -- the real mock-backend app via ``httpx.ASGITransport``
    (the ``mcp_app`` fixture), for job_status/list_jobs/join_job's happy and
    real-error paths (a real completed job, a real non-V2V 422).
  * ``httpx.MockTransport`` with a hand-written handler, for wait_for_job's
    state-sequence/timeout/clamp behavior and cancel_job/delete_job/
    purge_terminal_jobs's state-guard behavior, where the test needs to
    assert exactly which HTTP calls happened (or didn't).
"""

from __future__ import annotations

import json
from pathlib import Path

import anyio
import httpx
import pytest
from mcp.server.fastmcp.exceptions import ToolError

from mcp_server.client import BackendClient, set_client
from mcp_server.settings import Settings
from mcp_server.tools import generate, jobs


def _client_for_app(app, output_dir: Path) -> BackendClient:
    transport = httpx.ASGITransport(app=app)
    settings = Settings(base_url="http://testserver", api_key=None, output_dir=output_dir)
    return BackendClient(settings, transport=transport)


def _client_for_handler(handler) -> BackendClient:
    settings = Settings(base_url="http://testserver", api_key=None, output_dir=Path("."))
    return BackendClient(settings, transport=httpx.MockTransport(handler))


@pytest.fixture(autouse=True)
def _reset_client():
    set_client(None)
    yield
    set_client(None)


def _run_completed_job(app, output_dir: Path) -> str:
    """Submit + let the mock backend complete a plain T2V job; return its id."""
    set_client(_client_for_app(app, output_dir))
    result = anyio.run(generate.submit_generate, "a bustling town square at dusk")
    return result["job_id"]


# --------------------------------------------------------- job_status/list_jobs


def test_job_status_and_list_jobs_summary_has_no_request_key(mcp_app):
    app, output_dir = mcp_app
    job_id = _run_completed_job(app, output_dir)

    full = anyio.run(jobs.job_status, job_id)
    assert full["job_id"] == job_id
    assert full["status"] == "completed"
    assert "request" in full  # job_status IS the full-text tool

    listing = anyio.run(jobs.list_jobs)
    assert listing["counts"]["completed"] >= 1
    summary = next(j for j in listing["jobs"] if j["job_id"] == job_id)
    assert "request" not in summary
    assert summary["status"] == "completed"
    assert summary["has_result"] is True
    expected_keys = {
        "job_id", "status", "progress", "stage", "clip", "clip_count",
        "is_v2v", "joined", "created_at", "error", "has_result",
    }
    assert set(summary.keys()) == expected_keys


# --------------------------------------------------------------- wait_for_job


def test_wait_for_job_state_sequence_queued_running_completed():
    responses = iter(["queued", "running", "completed"])
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(200, json={"job_id": "j1", "status": next(responses)})

    set_client(_client_for_handler(handler))

    result = anyio.run(jobs.wait_for_job, "j1", 5, 0.1)

    assert result["status"] == "completed"
    assert result["timed_out"] is False
    assert result["waited_sec"] >= 0
    assert len(calls) == 3


def test_wait_for_job_timeout_path_returns_timed_out_true_not_an_exception():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"job_id": "j1", "status": "running"})

    set_client(_client_for_handler(handler))

    result = anyio.run(jobs.wait_for_job, "j1", 1, 0.5)

    assert result["status"] == "running"
    assert result["timed_out"] is True
    assert "hint" in result
    assert result["waited_sec"] >= 1


def test_clamp_helpers_are_pure_and_cheap():
    assert jobs._clamp_wait_timeout_sec(0.1) == 1.0
    assert jobs._clamp_wait_timeout_sec(999) == 45.0
    assert jobs._clamp_wait_timeout_sec(10) == 10
    assert jobs._clamp_poll_interval_sec(0.01) == 0.5
    assert jobs._clamp_poll_interval_sec(999) == 10.0
    assert jobs._clamp_join_wait_sec(0) == 5.0
    assert jobs._clamp_join_wait_sec(999) == 45.0


# ------------------------------------------------------------ cancel/delete


def test_cancel_job_on_terminal_raises_and_issues_no_delete():
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        assert request.method == "GET", "must not DELETE an already-terminal job"
        return httpx.Response(200, json={"job_id": "j1", "status": "completed"})

    set_client(_client_for_handler(handler))

    with pytest.raises(ToolError) as exc_info:
        anyio.run(jobs.cancel_job, "j1")

    assert "JOB_ALREADY_TERMINAL" in str(exc_info.value)
    assert calls == ["GET"]


def test_delete_job_on_active_raises_and_issues_no_delete():
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        assert request.method == "GET", "must not DELETE a still-active job"
        return httpx.Response(200, json={"job_id": "j1", "status": "running"})

    set_client(_client_for_handler(handler))

    with pytest.raises(ToolError) as exc_info:
        anyio.run(jobs.delete_job, "j1")

    assert "JOB_STILL_ACTIVE" in str(exc_info.value)
    assert calls == ["GET"]


def test_delete_job_on_terminal_issues_delete():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"job_id": "j1", "status": "completed"})
        assert request.method == "DELETE"
        return httpx.Response(200, json={"job_id": "j1", "deleted": True})

    set_client(_client_for_handler(handler))

    result = anyio.run(jobs.delete_job, "j1")
    assert result == {"job_id": "j1", "deleted": True}


# -------------------------------------------------------- purge_terminal_jobs


def _purge_fixture_records() -> list[dict]:
    return [
        {"job_id": "queued-1", "status": "queued"},
        {"job_id": "running-1", "status": "running"},
        {"job_id": "completed-ok", "status": "completed"},
        {"job_id": "failed-boom", "status": "failed"},
        {"job_id": "cancelled-ok", "status": "cancelled"},
    ]


def test_purge_terminal_jobs_skips_active_deletes_terminal_and_continues_past_a_failure():
    delete_calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json=_purge_fixture_records())
        assert request.method == "DELETE"
        job_id = request.url.path.rsplit("/", 1)[-1]
        delete_calls.append(job_id)
        if job_id == "failed-boom":
            return httpx.Response(
                500,
                json={"error": {"code": "INTERNAL", "message": "boom"}},
            )
        return httpx.Response(200, json={"job_id": job_id, "deleted": True})

    set_client(_client_for_handler(handler))

    result = anyio.run(jobs.purge_terminal_jobs, False)

    # queued/running never got a DELETE.
    assert "queued-1" not in delete_calls
    assert "running-1" not in delete_calls
    # both terminal successes AND the failing one were attempted (loop continued).
    assert set(delete_calls) == {"completed-ok", "failed-boom", "cancelled-ok"}
    assert result["attempted"] == 3
    assert result["deleted"] == 2
    assert len(result["failed"]) == 1
    assert result["failed"][0]["job_id"] == "failed-boom"
    assert "INTERNAL" in result["failed"][0]["error"]
    assert result["dry_run"] is False


def test_purge_terminal_jobs_dry_run_issues_zero_deletes():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET", "dry_run must not DELETE anything"
        return httpx.Response(200, json=_purge_fixture_records())

    set_client(_client_for_handler(handler))

    result = anyio.run(jobs.purge_terminal_jobs, True)

    assert result == {"attempted": 3, "deleted": 0, "failed": [], "dry_run": True}


# --------------------------------------------------------------------- join_job


def test_join_job_on_non_v2v_job_surfaces_job_not_joinable(mcp_app):
    app, output_dir = mcp_app
    job_id = _run_completed_job(app, output_dir)

    with pytest.raises(ToolError) as exc_info:
        anyio.run(jobs.join_job, job_id)

    assert "JOB_NOT_JOINABLE" in str(exc_info.value)


def test_join_job_read_timeout_returns_finished_false_not_an_exception():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("simulated timeout", request=request)

    set_client(_client_for_handler(handler))

    result = anyio.run(jobs.join_job, "j1")

    assert result == {
        "finished": False,
        "hint": (
            "応答がタイムアウトしましたが、バックエンド側では処理が継続している"
            "可能性があります。job_status の joined を確認してください。"
        ),
    }


def test_join_job_audio_smoothing_false_appears_in_captured_body():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "job_id": "j1",
                "joined_path": "outputs/j1/joined.mp4",
                "join_mode": "hard_concat",
                "source_normalized": False,
            },
        )

    set_client(_client_for_handler(handler))

    result = anyio.run(jobs.join_job, "j1", False)

    assert captured["body"]["audio_smoothing"] is False
    assert result["finished"] is True
    assert result["join_mode"] == "hard_concat"
