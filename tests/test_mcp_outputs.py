"""Contract tests for mcp_server/paths.py and mcp_server/tools/outputs.py (W5).

Same two-double pattern as the other mcp_server tool test modules: the real
mock-backend app via ``httpx.ASGITransport`` (``mcp_app`` fixture) for
happy-path/real-file behavior, and ``httpx.MockTransport`` with a hand-written
handler for asserting exactly which HTTP calls happen (or don't -- notably
``save_job_video`` must never call the backend at all).
"""

from __future__ import annotations

from pathlib import Path

import anyio
import httpx
import pytest
from mcp.server.fastmcp.exceptions import ToolError

from mcp_server.client import BackendClient, set_client
from mcp_server.paths import job_joined_path, job_output_path, unique_dest
from mcp_server.settings import Settings
from mcp_server.tools import generate, outputs


def _client_for_app(app, output_dir: Path) -> BackendClient:
    transport = httpx.ASGITransport(app=app)
    settings = Settings(base_url="http://testserver", api_key=None, output_dir=output_dir)
    return BackendClient(settings, transport=transport)


def _client_for_handler(handler, output_dir: Path) -> BackendClient:
    settings = Settings(base_url="http://testserver", api_key=None, output_dir=output_dir)
    return BackendClient(settings, transport=httpx.MockTransport(handler))


@pytest.fixture(autouse=True)
def _reset_client():
    set_client(None)
    yield
    set_client(None)


def _run_completed_job(app, output_dir: Path) -> str:
    set_client(_client_for_app(app, output_dir))
    result = anyio.run(generate.submit_generate, "a bustling town square at dusk")
    return result["job_id"]


# ------------------------------------------------------------------- paths.py


def test_job_output_path_and_joined_path_are_pure_path_arithmetic(tmp_path):
    assert job_output_path(tmp_path, "abc") == tmp_path / "abc" / "output.mp4"
    assert job_joined_path(tmp_path, "abc") == tmp_path / "abc" / "joined.mp4"
    # str input works the same as Path input.
    assert job_output_path(str(tmp_path), "abc") == tmp_path / "abc" / "output.mp4"


def test_unique_dest_no_collision_returns_filename_unchanged(tmp_path):
    assert unique_dest(tmp_path, "video.mp4") == tmp_path / "video.mp4"


def test_unique_dest_collision_suffixes_incrementally(tmp_path):
    (tmp_path / "video.mp4").write_bytes(b"1")
    assert unique_dest(tmp_path, "video.mp4") == tmp_path / "video_2.mp4"

    (tmp_path / "video_2.mp4").write_bytes(b"2")
    assert unique_dest(tmp_path, "video.mp4") == tmp_path / "video_3.mp4"


# ------------------------------------------------------------ get_job_video_path


def test_get_job_video_path_matches_real_output_dir_and_exists_true(mcp_app):
    app, output_dir = mcp_app
    job_id = _run_completed_job(app, output_dir)

    result = anyio.run(outputs.get_job_video_path, job_id)

    assert result["path"] == str(output_dir / job_id / "output.mp4")
    assert result["exists"] is True
    assert result["size_bytes"] > 0
    assert result["status"] == "completed"
    assert "note" not in result


def test_get_job_video_path_derives_from_settings_not_response(mcp_app, tmp_path):
    """Regression: even when the client is (mis)configured with a DIFFERENT
    output_dir than where the mock backend actually wrote files, the returned
    path is derived purely from client.output_dir -- never trusted from the
    job response."""
    app, real_output_dir = mcp_app
    job_id = _run_completed_job(app, real_output_dir)

    other_dir = tmp_path / "elsewhere"
    other_dir.mkdir()
    set_client(_client_for_app(app, other_dir))

    result = anyio.run(outputs.get_job_video_path, job_id)

    assert result["path"] == str(other_dir / job_id / "output.mp4")
    assert result["exists"] is False  # the real file lives under real_output_dir
    assert result["status"] == "completed"


def test_get_job_video_path_incomplete_job_exists_false_no_exception(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"job_id": "j1", "status": "running"})

    set_client(_client_for_handler(handler, tmp_path))

    result = anyio.run(outputs.get_job_video_path, "j1")

    assert result["exists"] is False
    assert result["size_bytes"] is None
    assert result["status"] == "running"
    assert "note" in result


# --------------------------------------------------------- get_joined_video_path


def test_get_joined_video_path_includes_joined_flag(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"job_id": "j1", "status": "completed", "joined": True})

    set_client(_client_for_handler(handler, tmp_path))

    result = anyio.run(outputs.get_joined_video_path, "j1")

    assert result["joined"] is True
    assert result["path"] == str(tmp_path / "j1" / "joined.mp4")
    assert "note" not in result


def test_get_joined_video_path_not_completed_has_note_and_joined_false(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"job_id": "j1", "status": "running", "joined": False})

    set_client(_client_for_handler(handler, tmp_path))

    result = anyio.run(outputs.get_joined_video_path, "j1")

    assert result["joined"] is False
    assert result["exists"] is False
    assert "note" in result


# -------------------------------------------------------------- save_job_video


def test_save_job_video_never_calls_the_backend(mcp_app, tmp_path):
    """save_job_video is a pure local copy -- no HTTP round trip at all."""
    app, output_dir = mcp_app
    job_id = _run_completed_job(app, output_dir)

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"save_job_video must not call the backend, got {request.url.path}")

    set_client(_client_for_handler(handler, output_dir))

    dest_dir = tmp_path / "saved"
    result = anyio.run(outputs.save_job_video, job_id, str(dest_dir))

    assert Path(result["saved_path"]) == dest_dir / f"{job_id}.mp4"
    assert Path(result["saved_path"]).exists()
    assert result["clobbered"] is False
    assert result["size_bytes"] > 0


def test_save_job_video_default_filename_and_mkdir_nested(mcp_app, tmp_path):
    app, output_dir = mcp_app
    job_id = _run_completed_job(app, output_dir)

    dest_dir = tmp_path / "saved" / "nested"  # does not exist yet
    result = anyio.run(outputs.save_job_video, job_id, str(dest_dir))

    saved = Path(result["saved_path"])
    assert saved == dest_dir / f"{job_id}.mp4"
    assert saved.exists()
    assert result["source_path"] == str(output_dir / job_id / "output.mp4")


def test_save_job_video_custom_filename(mcp_app, tmp_path):
    app, output_dir = mcp_app
    job_id = _run_completed_job(app, output_dir)

    result = anyio.run(outputs.save_job_video, job_id, str(tmp_path), "custom.mp4")

    assert Path(result["saved_path"]) == tmp_path / "custom.mp4"


def test_save_job_video_no_clobber_twice_suffixes(mcp_app, tmp_path):
    app, output_dir = mcp_app
    job_id = _run_completed_job(app, output_dir)
    dest_dir = tmp_path / "saved"

    first = anyio.run(outputs.save_job_video, job_id, str(dest_dir))
    second = anyio.run(outputs.save_job_video, job_id, str(dest_dir))

    assert Path(first["saved_path"]).name == f"{job_id}.mp4"
    assert Path(second["saved_path"]).name == f"{job_id}_2.mp4"
    assert first["clobbered"] is False
    assert second["clobbered"] is False


def test_save_job_video_no_clobber_false_overwrites(mcp_app, tmp_path):
    app, output_dir = mcp_app
    job_id = _run_completed_job(app, output_dir)
    dest_dir = tmp_path / "saved"

    first = anyio.run(outputs.save_job_video, job_id, str(dest_dir))
    second = anyio.run(outputs.save_job_video, job_id, str(dest_dir), None, False)

    assert first["saved_path"] == second["saved_path"]
    assert second["clobbered"] is True


def test_save_job_video_which_joined_missing_raises(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("save_job_video must not call the backend")

    set_client(_client_for_handler(handler, tmp_path))  # output_dir has no job folders at all

    with pytest.raises(ToolError) as exc_info:
        anyio.run(outputs.save_job_video, "no-such-job", str(tmp_path / "dest"), None, True, "joined")

    assert "VIDEO_NOT_FOUND" in str(exc_info.value)


def test_save_job_video_which_joined_succeeds_after_join(mcp_app, tmp_path):
    app, output_dir = mcp_app
    job_id = _run_completed_job(app, output_dir)

    # Simulate a completed join by writing joined.mp4 next to output.mp4 --
    # save_job_video only ever reads the local filesystem for the copy.
    joined_src = output_dir / job_id / "joined.mp4"
    joined_src.write_bytes(b"joined-bytes")

    result = anyio.run(
        outputs.save_job_video, job_id, str(tmp_path / "dest"), None, True, "joined"
    )

    assert Path(result["saved_path"]).name == f"{job_id}_joined.mp4"
    assert result["source_path"] == str(joined_src)


def test_save_job_video_invalid_which_raises(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("save_job_video must not call the backend")

    set_client(_client_for_handler(handler, tmp_path))

    with pytest.raises(ToolError):
        anyio.run(outputs.save_job_video, "j1", str(tmp_path / "dest"), None, True, "bogus")
