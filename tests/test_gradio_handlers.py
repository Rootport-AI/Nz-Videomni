"""Unit tests for the Gradio UI handlers (S1).

These never touch the server: ApiClient is fed an ``httpx.MockTransport`` so we
can assert on the exact request paths/headers/JSON it emits, and drive the
yield-based generate flow without a live backend.
"""

from __future__ import annotations

import httpx
import pytest

from gradio_ui import (
    ApiClient,
    format_status,
    make_generate_handler,
)


def _make_client(handler, *, api_key: str | None = "secret") -> ApiClient:
    transport = httpx.MockTransport(handler)
    hc = httpx.Client(transport=transport)
    return ApiClient("http://test", api_key=api_key, client=hc)


# --------------------------------------------------------------------------- #
# ApiClient: paths, method, auth header.
# --------------------------------------------------------------------------- #
def test_get_status_path_and_auth():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"server": "running", "version": "0.4.0"})

    api = _make_client(handler)
    data = api.get_status()
    assert seen["url"] == "http://test/api/v1/status"
    assert seen["auth"] == "Bearer secret"
    assert data["server"] == "running"


def test_no_auth_header_when_no_key():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={})

    api = _make_client(handler, api_key=None)
    api.get_status()
    assert seen["auth"] is None


def test_get_config_path():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"generation_presets": {}})

    api = _make_client(handler)
    cfg = api.get_config()
    assert seen["url"] == "http://test/api/v1/config"
    assert "generation_presets" in cfg


def test_load_and_unload_paths_and_methods():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, str(request.url)))
        return httpx.Response(200, json={"pipeline_loaded": True, "state": "ready"})

    api = _make_client(handler)
    api.load_pipeline()
    api.unload_pipeline()
    assert seen == [
        ("POST", "http://test/api/v1/pipeline/load"),
        ("POST", "http://test/api/v1/pipeline/unload"),
    ]


def test_get_job_path():
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://test/api/v1/jobs/abc123"
        return httpx.Response(200, json={"status": "running", "progress": 0.5})

    api = _make_client(handler)
    job = api.get_job("abc123")
    assert job["status"] == "running"


def test_upload_image_path_and_returns_id(tmp_path):
    img = tmp_path / "in.png"
    img.write_bytes(b"\x89PNG\r\n")
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["is_multipart"] = request.headers.get("content-type", "").startswith("multipart/")
        return httpx.Response(200, json={"image_id": "img-42"})

    api = _make_client(handler)
    image_id = api.upload_image(str(img))
    assert seen["url"] == "http://test/api/v1/upload/image"
    assert seen["is_multipart"] is True
    assert image_id == "img-42"


def test_fetch_video_downloads_to_tempfile():
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://test/api/v1/jobs/j9/video"
        return httpx.Response(200, content=b"MP4DATA")

    api = _make_client(handler)
    path = api.fetch_video("j9")
    assert path.endswith(".mp4")
    with open(path, "rb") as fh:
        assert fh.read() == b"MP4DATA"


# --------------------------------------------------------------------------- #
# format_status: pure formatter against a fake /status body.
# --------------------------------------------------------------------------- #
def _fake_status() -> dict:
    return {
        "server": "running",
        "version": "0.4.0",
        "pipeline_loaded": True,
        "pipeline_type": "distilled",
        "gpu": {"name": "RTX 4080", "vram_free_mb": 9812, "vram_total_mb": 16376},
        "vram_optimization": {"low_vram_mode": True, "low_vram_profile": "16gb_safe"},
        "queue": {"mode": "single_job_in_memory", "pending": 1, "running": 0, "completed": 3, "failed": 0},
    }


def test_format_status_english():
    line = format_status(_fake_status())
    assert "server=running v0.4.0" in line
    assert "Pipeline: loaded (distilled)" in line
    assert "RTX 4080" in line
    assert "free 9812 / 16376 MB" in line
    assert "Low-VRAM: ON (profile=16gb_safe)" in line
    assert "Queue: running 0 / waiting 1 / done 3" in line


def test_format_status_not_loaded_and_no_queue():
    s = {
        "server": "running", "version": "0.4.0", "pipeline_loaded": False,
        "gpu": {"name": None, "vram_free_mb": 0, "vram_total_mb": 0},
        "vram_optimization": {"low_vram_mode": False, "low_vram_profile": "default"},
    }
    line = format_status(s)
    assert "Pipeline: not loaded" in line
    assert "Low-VRAM: OFF" in line
    assert "Queue" not in line  # omitted when absent


def test_format_status_japanese():
    line = format_status(_fake_status(), lang="ja")
    assert "パイプライン: 読込済 (distilled)" in line
    assert "省VRAM: ON" in line
    assert "キュー: 実行中 0 / 待機 1 / 完了 3" in line


# --------------------------------------------------------------------------- #
# Generate flow: correct payload assembly (captured via mock transport).
# --------------------------------------------------------------------------- #
def _run_until_job_started(gen):
    """Advance the generator just past the /generate POST (first yield after it),
    then close it so the poll loop (with its 1s sleeps) never runs."""
    first = next(gen)
    gen.close()
    return first


def test_generate_t2v_payload():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://test/api/v1/generate"
        import json
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"job_id": "job-1"})

    api = _make_client(handler)
    generate = make_generate_handler(api)
    gen = generate(
        "A calm river", "blurry", None, 0.8,
        512, 320, False, 0, 0, 49, 24.0, -1,
    )
    progress, job_id, video = _run_until_job_started(gen)

    assert captured["prompt"] == "A calm river"
    assert captured["negative_prompt"] == "blurry"
    assert captured["width"] == 512 and captured["height"] == 320
    assert captured["num_frames"] == 49
    assert captured["frame_rate"] == 24.0
    assert captured["num_inference_steps"] == 8
    assert captured["guidance_scale"] == 1.0
    assert captured["pipeline"] == "distilled"
    assert captured["crop_output"] is None  # checkbox off
    assert captured["conditioning_images"] == []  # no image => t2v
    assert job_id == "job-1"
    assert "t2v" in progress


def test_generate_crop_output_when_enabled():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"job_id": "job-2"})

    api = _make_client(handler)
    generate = make_generate_handler(api)
    gen = generate(
        "prompt", "", None, 0.8,
        1280, 768, True, 1280, 720, 257, 24.0, 7,
    )
    _run_until_job_started(gen)
    assert captured["crop_output"] == {"width": 1280, "height": 720}
    assert captured["seed"] == 7


def test_generate_empty_prompt_short_circuits():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={"job_id": "x"})

    api = _make_client(handler)
    generate = make_generate_handler(api)
    out = list(generate("   ", "", None, 0.8, 512, 320, False, 0, 0, 49, 24.0, -1))
    assert calls["n"] == 0  # no HTTP performed
    assert len(out) == 1
    progress, job_id, video = out[0]
    assert job_id == "" and video is None


def test_generate_409_reports_busy():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(409, json={"error": {"code": "JOB_BUSY"}})

    api = _make_client(handler)
    generate = make_generate_handler(api)
    out = list(generate("prompt", "", None, 0.8, 512, 320, False, 0, 0, 49, 24.0, -1))
    progress, job_id, video = out[-1]
    assert "409" in progress
    assert job_id == ""


def test_generate_i2v_uploads_then_generates(tmp_path):
    img = tmp_path / "in.png"
    img.write_bytes(b"\x89PNG\r\n")
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/upload/image"):
            return httpx.Response(200, json={"image_id": "img-7"})
        import json
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"job_id": "job-3"})

    api = _make_client(handler)
    generate = make_generate_handler(api)
    gen = generate("prompt", "", str(img), 0.65, 512, 320, False, 0, 0, 49, 24.0, -1)
    # first yield = upload-done; advance again to trigger the generate POST.
    next(gen)
    second = next(gen)
    gen.close()
    assert captured["conditioning_images"] == [
        {"image_id": "img-7", "frame_idx": 0, "strength": 0.65}
    ]
    assert "i2v" in second[0]


# --------------------------------------------------------------------------- #
# build_ui smoke: constructs without HTTP and exposes the label registry.
# --------------------------------------------------------------------------- #
def test_build_ui_constructs_and_registers_labels():
    from gradio_ui import build_ui

    demo = build_ui("http://127.0.0.1:8000", api_key=None)
    registry = getattr(demo, "label_registry")
    assert len(registry) > 0
    # every registry entry references a key that exists in the English table.
    from gradio_ui import LABELS
    for _component, key, _attr in registry:
        assert key in LABELS["en"], f"missing en label for {key}"
        assert key in LABELS["ja"], f"missing ja label for {key}"
