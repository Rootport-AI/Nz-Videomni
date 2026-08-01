"""Unit tests for the Gradio UI handlers (S1).

These never touch the server: ApiClient is fed an ``httpx.MockTransport`` so we
can assert on the exact request paths/headers/JSON it emits, and drive the
yield-based generate flow without a live backend.
"""

from __future__ import annotations

import httpx
import pytest

from gradio_ui import (
    ADAPTER_NONE,
    MAX_CHAIN_TOTAL_PIXEL_FRAMES,
    PRESETS,
    ApiClient,
    apply_preset,
    build_adapter_choices,
    build_jobs_rows,
    build_preset_choices,
    check_chain_total,
    compute_spill_warning,
    delete_finished_jobs,
    format_api_error,
    format_job_error,
    format_status,
    jobs_table_headers,
    make_chain_handler,
    make_generate_handler,
    pick_default_preset,
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


# --------------------------------------------------------------------------- #
# S3: the 5 fixed keyframe slots are flattened into 20 positional args
# (enabled, image_path, frame_idx, strength) x 5. This helper builds that flat
# list from as few slots as a test cares about; the rest default to
# disabled/empty (pure T2V when all 5 are left out).
# --------------------------------------------------------------------------- #
def _kf_args(*slots):
    filled = list(slots) + [(False, None, 0, 0.8)] * (5 - len(slots))
    args = []
    for enabled, image, frame_idx, strength in filled[:5]:
        args.extend([enabled, image, frame_idx, strength])
    return args


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
        "A calm river", "blurry", *_kf_args(),
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
    assert captured["conditioning_images"] == []  # no keyframes => t2v
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
        "prompt", "", *_kf_args(),
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
    out = list(generate("   ", "", *_kf_args(), 512, 320, False, 0, 0, 49, 24.0, -1))
    assert calls["n"] == 0  # no HTTP performed
    assert len(out) == 1
    progress, job_id, video = out[0]
    assert job_id == "" and video is None


def test_generate_409_reports_busy():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(409, json={"error": {"code": "JOB_BUSY"}})

    api = _make_client(handler)
    generate = make_generate_handler(api)
    out = list(generate("prompt", "", *_kf_args(), 512, 320, False, 0, 0, 49, 24.0, -1))
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
    gen = generate(
        "prompt", "", *_kf_args((True, str(img), 0, 0.65)),
        512, 320, False, 0, 0, 49, 24.0, -1,
    )
    # first yield = "uploading keyframe 1/1"; advance again to trigger the
    # generate POST.
    first = next(gen)
    second = next(gen)
    gen.close()
    assert "1/1" in first[0]
    assert captured["conditioning_images"] == [
        {"image_id": "img-7", "frame_idx": 0, "strength": 0.65}
    ]
    assert "i2v" in second[0]


# --------------------------------------------------------------------------- #
# S3: multi-keyframe conditioning (5 fixed slots).
# --------------------------------------------------------------------------- #
def test_generate_multi_keyframe_uploads_in_slot_order(tmp_path):
    img1 = tmp_path / "a.png"
    img2 = tmp_path / "b.png"
    img3 = tmp_path / "c.png"
    for p in (img1, img2, img3):
        p.write_bytes(b"\x89PNG\r\n")

    uploads = []
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/upload/image"):
            uploads.append(request.content)
            return httpx.Response(200, json={"image_id": f"img-{len(uploads)}"})
        import json
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"job_id": "job-multi"})

    api = _make_client(handler)
    generate = make_generate_handler(api)
    gen = generate(
        "prompt", "",
        *_kf_args(
            (True, str(img1), 0, 0.8),
            (True, str(img2), 32, 0.6),
            (True, str(img3), 64, 0.4),
        ),
        512, 320, False, 0, 0, 49, 24.0, -1,
    )
    progress_msgs = []
    for out in gen:
        progress_msgs.append(out[0])
        if out[1]:  # job started -> stop before the poll loop's 1s sleeps
            gen.close()
            break

    assert len(uploads) == 3
    assert any("1/3" in m for m in progress_msgs)
    assert any("2/3" in m for m in progress_msgs)
    assert any("3/3" in m for m in progress_msgs)
    assert captured["conditioning_images"] == [
        {"image_id": "img-1", "frame_idx": 0, "strength": 0.8},
        {"image_id": "img-2", "frame_idx": 32, "strength": 0.6},
        {"image_id": "img-3", "frame_idx": 64, "strength": 0.4},
    ]


def test_generate_disabled_slot_with_image_is_skipped(tmp_path):
    img = tmp_path / "in.png"
    img.write_bytes(b"\x89PNG\r\n")
    uploads = {"n": 0}
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/upload/image"):
            uploads["n"] += 1
            return httpx.Response(200, json={"image_id": "img-x"})
        import json
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"job_id": "job-skip"})

    api = _make_client(handler)
    generate = make_generate_handler(api)
    gen = generate(
        "prompt", "",
        *_kf_args((False, str(img), 0, 0.8)),  # slot 1 filled but disabled
        512, 320, False, 0, 0, 49, 24.0, -1,
    )
    _run_until_job_started(gen)
    assert uploads["n"] == 0
    assert captured["conditioning_images"] == []


def test_generate_enabled_slot_missing_image_errors():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={"job_id": "x"})

    api = _make_client(handler)
    generate = make_generate_handler(api)
    out = list(generate(
        "prompt", "",
        *_kf_args((True, None, 0, 0.8)),  # slot 1 enabled but no image
        512, 320, False, 0, 0, 49, 24.0, -1,
    ))
    assert calls["n"] == 0  # no HTTP performed at all
    assert len(out) == 1
    progress, job_id, video = out[0]
    assert "1" in progress  # localized message references slot 1
    assert job_id == "" and video is None


def test_generate_all_slots_disabled_is_pure_t2v():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert not request.url.path.endswith("/upload/image")
        import json
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"job_id": "job-t2v"})

    api = _make_client(handler)
    generate = make_generate_handler(api)
    gen = generate("prompt", "", *_kf_args(), 512, 320, False, 0, 0, 49, 24.0, -1)
    _run_until_job_started(gen)
    assert captured["conditioning_images"] == []


def test_generate_negative_frame_idx_errors_before_any_api_call(tmp_path):
    img = tmp_path / "in.png"
    img.write_bytes(b"\x89PNG\r\n")
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={"job_id": "x"})

    api = _make_client(handler)
    generate = make_generate_handler(api)
    out = list(generate(
        "prompt", "",
        *_kf_args((True, str(img), -1, 0.8)),
        512, 320, False, 0, 0, 49, 24.0, -1,
    ))
    assert calls["n"] == 0  # neither upload nor generate performed
    assert len(out) == 1
    progress, job_id, video = out[0]
    assert job_id == "" and video is None


# --------------------------------------------------------------------------- #
# S2: dynamic presets (/config generation_presets) + spill-free warning.
# --------------------------------------------------------------------------- #
def _fake_config_with_presets() -> dict:
    return {
        "generation_presets": {
            "smoke_test": {"width": 384, "height": 256, "crop_output": None, "num_frames": 17},
            "standard_720p": {
                "width": 1280, "height": 768,
                "crop_output": {"width": 1280, "height": 720},
                "num_frames": 257,
            },
        },
        "limits": {
            "spill_free_frames": {"1280x768": 257, "1920x1088": 153},
        },
    }


def test_build_preset_choices_from_server_config():
    choices = build_preset_choices(_fake_config_with_presets())
    values = [v for _label, v in choices]
    assert values == ["smoke_test", "standard_720p"]
    labels_by_value = {v: label for label, v in choices}
    # crop_output present -> "WxH -> cropW x cropH, Nf" in the label.
    assert "1280×768" in labels_by_value["standard_720p"]
    assert "1280×720" in labels_by_value["standard_720p"]
    assert "257f" in labels_by_value["standard_720p"]
    # crop_output absent -> no arrow.
    assert "→" not in labels_by_value["smoke_test"]


def test_build_preset_choices_falls_back_when_config_empty():
    choices = build_preset_choices({})
    assert choices == [(name, name) for name in PRESETS]


def test_pick_default_preset_prefers_standard_720p():
    assert pick_default_preset(_fake_config_with_presets()) == "standard_720p"


def test_pick_default_preset_first_key_when_no_standard_720p():
    cfg = {"generation_presets": {"smoke_test": {"width": 384, "height": 256,
                                                  "crop_output": None, "num_frames": 17}}}
    assert pick_default_preset(cfg) == "smoke_test"


def test_pick_default_preset_fallback_when_config_empty():
    assert pick_default_preset({}) == "minimal"


def test_apply_preset_from_server_config_with_crop():
    cfg = _fake_config_with_presets()
    (width, height, frames, crop_enabled, crop_w, crop_h,
     crop_row_update, spill_update) = apply_preset("standard_720p", cfg)
    assert (width, height, frames) == (1280, 768, 257)
    assert crop_enabled is True
    assert (crop_w, crop_h) == (1280, 720)
    assert crop_row_update["visible"] is True
    # 257 frames == the spill-free threshold exactly -> not exceeded, no warning.
    assert spill_update["visible"] is False


def test_apply_preset_from_server_config_without_crop():
    cfg = _fake_config_with_presets()
    (width, height, frames, crop_enabled, crop_w, crop_h,
     crop_row_update, _spill_update) = apply_preset("smoke_test", cfg)
    assert (width, height, frames) == (384, 256, 17)
    assert crop_enabled is False
    assert (crop_w, crop_h) == (0, 0)
    assert crop_row_update["visible"] is False


def test_apply_preset_fallback_when_config_empty():
    # "small" (was "phase1_target") -- 960x576, 121 frames, 960x540 crop.
    (width, height, frames, crop_enabled, crop_w, crop_h,
     crop_row_update, _spill_update) = apply_preset("small", {})
    assert (width, height, frames) == (960, 576, 121)
    assert crop_enabled is True
    assert (crop_w, crop_h) == (960, 540)
    assert crop_row_update["visible"] is True


def test_apply_preset_fallback_unknown_name_uses_minimal():
    (width, height, frames, *_rest) = apply_preset("does_not_exist", {})
    assert (width, height, frames) == (512, 320, 49)


def test_spill_warning_hidden_at_exact_threshold():
    cfg = _fake_config_with_presets()
    upd = compute_spill_warning(1280, 768, 257, cfg)
    assert upd["visible"] is False
    assert upd["value"] == ""


def test_spill_warning_shown_when_exceeded():
    cfg = _fake_config_with_presets()
    upd = compute_spill_warning(1280, 768, 321, cfg)
    assert upd["visible"] is True
    assert "1280x768" in upd["value"]
    assert "257" in upd["value"]


def test_spill_warning_hidden_for_unknown_resolution():
    cfg = _fake_config_with_presets()
    upd = compute_spill_warning(999, 999, 999, cfg)
    assert upd["visible"] is False


def test_spill_warning_hidden_when_config_empty():
    upd = compute_spill_warning(1280, 768, 321, {})
    assert upd["visible"] is False


def test_spill_warning_japanese():
    cfg = _fake_config_with_presets()
    upd = compute_spill_warning(1280, 768, 321, cfg, lang="ja")
    assert upd["visible"] is True
    assert "1280x768" in upd["value"]
    assert "快適上限" in upd["value"]


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


# --------------------------------------------------------------------------- #
# S4: IC-LoRA reference-video control. The generate handler gains trailing
# args (adapter, adapter_strength, control_adherence, reference_strength,
# ref_video_path, config); a helper appends them so the intent of each test is
# explicit. control_adherence / reference_strength (S3) are keyword-only here so
# the existing positional call sites (adapter, strength, ref_path, config) stay
# valid; the returned list places them in the real UI/handler order.
# --------------------------------------------------------------------------- #
def _adapter_args(adapter=ADAPTER_NONE, strength=1.0, ref_path=None, config=None,
                  control_adherence=1.0, reference_strength=1.0):
    return [adapter, strength, control_adherence, reference_strength, ref_path, config]


def test_upload_video_path_and_returns_id(tmp_path):
    vid = tmp_path / "ref.mp4"
    vid.write_bytes(b"MP4DATA")
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["is_multipart"] = request.headers.get("content-type", "").startswith("multipart/")
        return httpx.Response(200, json={"video_id": "vid-1"})

    api = _make_client(handler)
    video_id = api.upload_video(str(vid))
    assert seen["url"] == "http://test/api/v1/upload/video"
    assert seen["is_multipart"] is True
    assert video_id == "vid-1"


def test_generate_adapter_none_omits_lora_and_reference():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert not request.url.path.endswith("/upload/video")
        import json
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"job_id": "job-none"})

    api = _make_client(handler)
    generate = make_generate_handler(api)
    gen = generate(
        "prompt", "", *_kf_args(),
        512, 320, False, 0, 0, 49, 24.0, -1,
        *_adapter_args(),  # adapter = ADAPTER_NONE
    )
    _run_until_job_started(gen)
    assert "loras" not in captured
    assert "reference_video_id" not in captured


def test_generate_adapter_uploads_video_and_correct_payload(tmp_path):
    vid = tmp_path / "ref.mp4"
    vid.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    uploads = {"video": 0}
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/upload/video"):
            uploads["video"] += 1
            return httpx.Response(200, json={"video_id": "vid-9"})
        import json
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"job_id": "job-ref"})

    api = _make_client(handler)
    generate = make_generate_handler(api)
    gen = generate(
        "prompt", "", *_kf_args(),
        1280, 768, False, 0, 0, 257, 24.0, -1,
        *_adapter_args("canny-control", 1.5, str(vid), {}),
    )
    for out in gen:
        if out[1]:  # job started -> stop before the poll loop's sleeps
            gen.close()
            break

    assert uploads["video"] == 1
    assert captured["loras"] == [{"name": "canny-control", "strength": 1.5}]
    assert captured["reference_video_id"] == "vid-9"
    # keyframes and adapter can combine; here no keyframes -> empty list.
    assert captured["conditioning_images"] == []


def test_generate_adapter_missing_video_errors_zero_calls():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={"job_id": "x"})

    api = _make_client(handler)
    generate = make_generate_handler(api)
    out = list(generate(
        "prompt", "", *_kf_args(),
        1280, 768, False, 0, 0, 257, 24.0, -1,
        *_adapter_args("canny-control", 1.0, None, {}),
    ))
    assert calls["n"] == 0  # no upload, no generate
    assert len(out) == 1
    progress, job_id, video = out[0]
    assert job_id == "" and video is None


def test_generate_adapter_bad_extension_errors_zero_calls(tmp_path):
    bad = tmp_path / "ref.txt"
    bad.write_bytes(b"not a video")
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={"job_id": "x"})

    api = _make_client(handler)
    generate = make_generate_handler(api)
    out = list(generate(
        "prompt", "", *_kf_args(),
        1280, 768, False, 0, 0, 257, 24.0, -1,
        *_adapter_args("canny-control", 1.0, str(bad), {}),
    ))
    assert calls["n"] == 0
    assert len(out) == 1
    assert out[0][1] == "" and out[0][2] is None


def test_generate_adapter_resolution_not_128_errors_zero_calls(tmp_path):
    vid = tmp_path / "ref.mp4"
    vid.write_bytes(b"data")
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={"job_id": "x"})

    api = _make_client(handler)
    generate = make_generate_handler(api)
    # 1280 % 128 == 0 but 720 % 128 == 80 -> the ÷128 precheck fires.
    out = list(generate(
        "prompt", "", *_kf_args(),
        1280, 720, False, 0, 0, 257, 24.0, -1,
        *_adapter_args("canny-control", 1.0, str(vid), {}),
    ))
    assert calls["n"] == 0
    assert len(out) == 1
    assert out[0][1] == "" and out[0][2] is None


def test_generate_adapter_resolution_128_passes_precheck(tmp_path):
    vid = tmp_path / "ref.mp4"
    vid.write_bytes(b"data")
    seen = {"video": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/upload/video"):
            seen["video"] += 1
            return httpx.Response(200, json={"video_id": "v"})
        return httpx.Response(200, json={"job_id": "j"})

    api = _make_client(handler)
    generate = make_generate_handler(api)
    gen = generate(
        "prompt", "", *_kf_args(),
        1280, 768, False, 0, 0, 257, 24.0, -1,
        *_adapter_args("pose-control", 1.0, str(vid), {}),
    )
    for out in gen:
        if out[1]:
            gen.close()
            break
    assert seen["video"] == 1  # precheck passed -> reference video uploaded


def test_generate_adapter_too_large_errors_zero_calls(tmp_path):
    vid = tmp_path / "ref.mp4"
    vid.write_bytes(b"x" * 2048)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={"job_id": "x"})

    api = _make_client(handler)
    generate = make_generate_handler(api)
    # A tiny configured limit (0 MB) makes the 2KB file "too large".
    cfg = {"upload": {"max_video_size_mb": 0, "allowed_video_extensions": [".mp4"]}}
    out = list(generate(
        "prompt", "", *_kf_args(),
        1280, 768, False, 0, 0, 257, 24.0, -1,
        *_adapter_args("canny-control", 1.0, str(vid), cfg),
    ))
    assert calls["n"] == 0
    assert len(out) == 1
    assert out[0][1] == "" and out[0][2] is None


# --------------------------------------------------------------------------- #
# S3: control-adherence + reference-strength sliders. Each key is sent ONLY when
# the adapter flow is active AND the slider value is below 1.0, so the default
# (both 1.0) op stays byte-identical to the pre-S3 request.
# --------------------------------------------------------------------------- #
def test_generate_adapter_strengths_default_omits_keys(tmp_path):
    vid = tmp_path / "ref.mp4"
    vid.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/upload/video"):
            return httpx.Response(200, json={"video_id": "vid-9"})
        import json
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"job_id": "job-def"})

    api = _make_client(handler)
    generate = make_generate_handler(api)
    gen = generate(
        "prompt", "", *_kf_args(),
        1280, 768, False, 0, 0, 257, 24.0, -1,
        *_adapter_args("canny-control", 1.0, str(vid), {}),  # both sliders 1.0
    )
    for out in gen:
        if out[1]:  # job started -> stop before the poll loop's sleeps
            gen.close()
            break
    assert "conditioning_attention_strength" not in captured
    assert "reference_video_strength" not in captured


def test_generate_adapter_strengths_below_one_included(tmp_path):
    vid = tmp_path / "ref.mp4"
    vid.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/upload/video"):
            return httpx.Response(200, json={"video_id": "vid-9"})
        import json
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"job_id": "job-str"})

    api = _make_client(handler)
    generate = make_generate_handler(api)
    gen = generate(
        "prompt", "", *_kf_args(),
        1280, 768, False, 0, 0, 257, 24.0, -1,
        *_adapter_args("canny-control", 1.0, str(vid), {},
                       control_adherence=0.6, reference_strength=0.8),
    )
    for out in gen:
        if out[1]:  # job started -> stop before the poll loop's sleeps
            gen.close()
            break
    assert captured["conditioning_attention_strength"] == 0.6
    assert captured["reference_video_strength"] == 0.8


def test_generate_non_adapter_ignores_strength_sliders():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert not request.url.path.endswith("/upload/video")
        import json
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"job_id": "job-noadapt"})

    api = _make_client(handler)
    generate = make_generate_handler(api)
    # adapter = ADAPTER_NONE but sliders below 1.0 -> keys must stay absent
    # (no adapter flow, so no reference-conditioning strengths apply).
    gen = generate(
        "prompt", "", *_kf_args(),
        512, 320, False, 0, 0, 49, 24.0, -1,
        *_adapter_args(control_adherence=0.5, reference_strength=0.5),
    )
    _run_until_job_started(gen)
    assert "conditioning_attention_strength" not in captured
    assert "reference_video_strength" not in captured


# --------------------------------------------------------------------------- #
# Generate-tab: ÷64 / 8n+1 precheck (moved over from the chain handler's
# identical rule) — zero API calls on violation.
# --------------------------------------------------------------------------- #
def test_generate_bad_width_precheck_zero_calls():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={"job_id": "x"})

    api = _make_client(handler)
    generate = make_generate_handler(api)
    # 500 is not a multiple of 64.
    out = list(generate("prompt", "", *_kf_args(), 500, 320, False, 0, 0, 49, 24.0, -1))
    assert calls["n"] == 0
    assert len(out) == 1
    assert out[0][1] == "" and out[0][2] is None


def test_generate_bad_num_frames_precheck_zero_calls():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={"job_id": "x"})

    api = _make_client(handler)
    generate = make_generate_handler(api)
    # 50 is not 8n+1 (49 or 57 would be).
    out = list(generate("prompt", "", *_kf_args(), 512, 320, False, 0, 0, 50, 24.0, -1))
    assert calls["n"] == 0
    assert len(out) == 1
    assert out[0][1] == "" and out[0][2] is None


# --------------------------------------------------------------------------- #
# A2V (Generate tab, 案A): a src_audio upload routes make_generate_handler's
# ``generate`` to POST /generate/chain as a single-clip chain instead of
# POST /generate. Style/character <lora:...> tokens ARE allowed (wired into the
# chain payload's ``loras``); only the reference-video CONTROL adapter is not.
# --------------------------------------------------------------------------- #
def test_generate_a2v_conflict_with_adapter_without_reference_zero_calls(tmp_path):
    """A2V + a reference-video CONTROL adapter (canny/pose/upscaler) WITHOUT a
    reference video selected is rejected up front by the existing adapter
    precheck (:466-468 -- ``msg_ref_video_required``), zero API calls. This is
    the SAME precheck the non-A2V adapter flow already uses; A2V+adapter has no
    dedicated conflict message anymore now that reference-video IC-LoRA can be
    combined with A2V (chains carry ``reference_video_id`` for a single clip)."""
    aud = tmp_path / "a.wav"
    aud.write_bytes(b"RIFF....WAVEfmt ")
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={"job_id": "x"})

    api = _make_client(handler)
    generate = make_generate_handler(api)
    out = list(generate(
        "prompt", "", *_kf_args(),
        512, 320, False, 0, 0, 49, 24.0, -1,
        adapter="canny-control", src_audio=str(aud),
    ))
    assert calls["n"] == 0  # no upload_video, no upload_audio, no generate_chain
    assert len(out) == 1
    assert out[0][1] == "" and out[0][2] is None
    from gradio_ui import LABELS
    assert LABELS["en"]["msg_ref_video_required"] in out[0][0]


def test_generate_a2v_with_adapter_and_reference_wires_chain_payload(tmp_path):
    """A2V + a reference-video CONTROL adapter WITH a reference video selected
    is accepted: the reference video is uploaded, the chain payload carries
    both ``reference_video_id`` and ``loras`` (the control adapter), and
    /generate/chain (not /generate) is called."""
    aud = tmp_path / "a.wav"
    aud.write_bytes(b"RIFF....WAVEfmt ")
    vid = tmp_path / "ref.mp4"
    vid.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    uploads = {"video": 0, "audio": 0}
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/upload/video"):
            uploads["video"] += 1
            return httpx.Response(200, json={"video_id": "vid-a2v-ref"})
        if request.url.path.endswith("/upload/audio"):
            uploads["audio"] += 1
            return httpx.Response(200, json={"audio_id": "aud-a2v-ref"})
        assert str(request.url) == "http://test/api/v1/generate/chain"
        import json
        captured.update(json.loads(request.content))
        return httpx.Response(202, json={"job_id": "chain-a2v-ref"})

    api = _make_client(handler)
    generate = make_generate_handler(api)
    gen = generate(
        "prompt", "", *_kf_args(),
        512, 384, False, 0, 0, 49, 24.0, -1,
        *_adapter_args("canny-control", 1.0, str(vid), {}),
        src_audio=str(aud),
    )
    outs = []
    for out in gen:
        outs.append(out)
        if out[1]:  # job started -> stop before the poll loop's sleeps
            gen.close()
            break

    assert uploads["video"] == 1
    assert uploads["audio"] == 1
    assert captured["reference_video_id"] == "vid-a2v-ref"
    assert captured["loras"] == [{"name": "canny-control", "strength": 1.0}]
    assert captured["source_audio"] == {"audio_id": "aud-a2v-ref"}
    assert captured["clips"] == [{"num_frames": 49}]
    assert outs[-1][1] == "chain-a2v-ref"


def test_generate_a2v_with_style_lora_token_wires_chain_loras(tmp_path):
    """A2V + a prompt <lora:...> STYLE token: the token is resolved against
    GET /loras, stripped from the prompt, and sent in the chain payload's
    ``loras`` (uniform strength across the clip)."""
    aud = tmp_path / "a.wav"
    aud.write_bytes(b"RIFF....WAVEfmt ")  # unreadable wav -> length precheck skipped
    captured = {}
    uploads = {"audio": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/loras") and request.method == "GET":
            return httpx.Response(200, json={"loras": [
                {"name": "neon-city", "kind": "style", "has_thumbnail": False,
                 "exists": True, "source": "scan"},
            ]})
        if request.url.path.endswith("/upload/audio"):
            uploads["audio"] += 1
            return httpx.Response(200, json={"audio_id": "aud-lora"})
        assert str(request.url) == "http://test/api/v1/generate/chain"
        import json
        captured.update(json.loads(request.content))
        return httpx.Response(202, json={"job_id": "chain-a2v-lora"})

    api = _make_client(handler)
    generate = make_generate_handler(api)
    gen = generate(
        "a singer <lora:neon-city:0.7> performing", "", *_kf_args(),
        512, 320, False, 0, 0, 49, 24.0, -1,
        src_audio=str(aud),
    )
    outs = []
    for out in gen:
        outs.append(out)
        if out[1]:  # job started -> stop before the poll loop's sleeps
            gen.close()
            break
    assert uploads["audio"] == 1
    assert captured["loras"] == [{"name": "neon-city", "strength": 0.7}]
    assert captured["prompt"] == "a singer performing"  # token stripped
    assert captured["source_audio"] == {"audio_id": "aud-lora"}
    assert captured["clips"] == [{"num_frames": 49}]
    assert outs[-1][1] == "chain-a2v-lora"


def test_generate_a2v_unknown_style_token_aborts_zero_calls(tmp_path):
    """A2V + an UNKNOWN <lora:...> token aborts with zero upload/generate calls
    (same precheck discipline as the single /generate path)."""
    aud = tmp_path / "a.wav"
    aud.write_bytes(b"RIFF....WAVEfmt ")
    posts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/loras") and request.method == "GET":
            return httpx.Response(200, json={"loras": [
                {"name": "neon-city", "kind": "style", "has_thumbnail": False,
                 "exists": True, "source": "scan"},
            ]})
        if request.method == "POST":
            posts["n"] += 1
        return httpx.Response(200, json={"job_id": "x"})

    api = _make_client(handler)
    generate = make_generate_handler(api)
    out = list(generate(
        "hero <lora:does-not-exist:1.0> walking", "", *_kf_args(),
        512, 320, False, 0, 0, 49, 24.0, -1,
        src_audio=str(aud),
    ))
    assert posts["n"] == 0  # no upload_audio, no generate_chain
    assert out[-1][1] == "" and out[-1][2] is None


def test_generate_a2v_audio_only_uses_generate_chain(tmp_path):
    aud = tmp_path / "a.wav"
    aud.write_bytes(b"RIFF....WAVEfmt ")
    uploads = {"audio": 0}
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/upload/audio"):
            uploads["audio"] += 1
            return httpx.Response(200, json={"audio_id": "aud-1"})
        assert str(request.url) == "http://test/api/v1/generate/chain"
        import json
        captured.update(json.loads(request.content))
        return httpx.Response(202, json={"job_id": "chain-a2v"})

    api = _make_client(handler)
    generate = make_generate_handler(api)
    gen = generate(
        "a singer performing", "", *_kf_args(),
        512, 320, False, 0, 0, 49, 24.0, -1,
        src_audio=str(aud),
    )
    outs = []
    for out in gen:
        outs.append(out)
        if out[1]:  # job started -> stop before the poll loop's sleeps
            gen.close()
            break

    assert uploads["audio"] == 1
    assert captured["clips"] == [{"num_frames": 49}]
    assert captured["source_audio"] == {"audio_id": "aud-1"}
    assert captured["width"] == 512 and captured["height"] == 320
    assert captured["prompt"] == "a singer performing"
    assert captured["overlap_frames"] == 3
    assert captured["overlap_strength"] == 0.5
    assert "num_frames" not in captured  # lives on the clip, not top-level
    assert "loras" not in captured  # GenerateChainRequest carries no loras field
    assert outs[-1][1] == "chain-a2v"


def test_generate_a2v_upload_failure_reports_error_zero_generate_chain(tmp_path):
    aud = tmp_path / "a.wav"
    aud.write_bytes(b"RIFF....WAVEfmt ")
    calls = {"chain": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/upload/audio"):
            return httpx.Response(500, json={"error": "boom"})
        calls["chain"] += 1
        return httpx.Response(202, json={"job_id": "x"})

    api = _make_client(handler)
    generate = make_generate_handler(api)
    out = list(generate(
        "prompt", "", *_kf_args(),
        512, 320, False, 0, 0, 49, 24.0, -1,
        src_audio=str(aud),
    ))
    assert calls["chain"] == 0  # upload failed -> generate/chain never called
    assert out[-1][1] == "" and out[-1][2] is None


def test_generate_a2v_with_keyframe_builds_clip_conditioning(tmp_path):
    aud = tmp_path / "a.wav"
    aud.write_bytes(b"RIFF....WAVEfmt ")
    img = tmp_path / "kf.png"
    img.write_bytes(b"\x89PNG\r\n")
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/upload/audio"):
            return httpx.Response(200, json={"audio_id": "aud-2"})
        if request.url.path.endswith("/upload/image"):
            return httpx.Response(200, json={"image_id": "img-1"})
        import json
        captured.update(json.loads(request.content))
        return httpx.Response(202, json={"job_id": "chain-a2v-kf"})

    api = _make_client(handler)
    generate = make_generate_handler(api)
    gen = generate(
        "prompt", "",
        *_kf_args((True, str(img), 0, 0.7)),
        512, 320, False, 0, 0, 49, 24.0, -1,
        src_audio=str(aud),
    )
    for out in gen:
        if out[1]:
            gen.close()
            break

    assert captured["clips"] == [{
        "num_frames": 49,
        "conditioning_images": [{"image_id": "img-1", "frame_idx": 0, "strength": 0.7}],
    }]


def _real_wav(path, seconds, sr=16000):
    """Write a real (readable) mono 16-bit PCM wav of ``seconds`` at ``path``."""
    import struct
    import wave

    n = int(seconds * sr)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(struct.pack("<%dh" % n, *([0] * n)))
    return str(path)


def test_generate_a2v_short_wav_rejected_zero_calls(tmp_path):
    """A2V length precheck: a wav shorter than the timeline is rejected BEFORE
    any upload/API call (mirrors the server's 422 SOURCE_AUDIO_TOO_SHORT). 2s
    audio vs. 121 frames @ 24fps (~5.04s video, ~126 audio-latent frames
    required, only ~50 available)."""
    aud = _real_wav(tmp_path / "short.wav", 2.0)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={"job_id": "x"})

    api = _make_client(handler)
    generate = make_generate_handler(api)
    out = list(generate(
        "prompt", "", *_kf_args(),
        512, 320, False, 0, 0, 121, 24.0, -1,
        src_audio=aud,
    ))
    assert calls["n"] == 0  # no upload_audio, no generate_chain
    assert len(out) == 1
    assert out[0][1] == "" and out[0][2] is None
    # The message states both the required and the attached seconds.
    assert "5.04" in out[0][0] and "2.00" in out[0][0]


def test_generate_a2v_long_wav_passes_precheck_and_uploads(tmp_path):
    """A2V length precheck: a wav at least as long as the timeline sails through
    the precheck and proceeds to upload + /generate/chain. 6s audio vs. the same
    121-frame @ 24fps timeline (~150 available >= ~126 required)."""
    aud = _real_wav(tmp_path / "long.wav", 6.0)
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path.endswith("/upload/audio"):
            return httpx.Response(200, json={"audio_id": "aud-1"})
        return httpx.Response(202, json={"job_id": "chain-a2v"})

    api = _make_client(handler)
    generate = make_generate_handler(api)
    gen = generate(
        "prompt", "", *_kf_args(),
        512, 320, False, 0, 0, 121, 24.0, -1,
        src_audio=aud,
    )
    outs = []
    for out in gen:
        outs.append(out)
        if out[1]:
            gen.close()
            break
    assert any(p.endswith("/upload/audio") for p in calls)
    assert outs[-1][1] == "chain-a2v"


def test_generate_a2v_non_wav_defers_to_server(tmp_path):
    """A non-wav audio (stdlib cannot decode) skips the local length precheck and
    proceeds to upload — the server's ffprobe preflight is the authority. An
    otherwise-too-short setting must still reach the API (zero client rejection)."""
    aud = tmp_path / "clip.mp3"
    aud.write_bytes(b"ID3\x03\x00\x00\x00")  # not a real mp3; length unmeasurable
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path.endswith("/upload/audio"):
            return httpx.Response(200, json={"audio_id": "aud-1"})
        return httpx.Response(202, json={"job_id": "chain-a2v"})

    api = _make_client(handler)
    generate = make_generate_handler(api)
    gen = generate(
        "prompt", "", *_kf_args(),
        512, 320, False, 0, 0, 121, 24.0, -1,
        src_audio=str(aud),
    )
    outs = []
    for out in gen:
        outs.append(out)
        if out[1]:
            gen.close()
            break
    assert any(p.endswith("/upload/audio") for p in calls)
    assert outs[-1][1] == "chain-a2v"


def test_generate_no_audio_uses_plain_generate_endpoint_unchanged():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"job_id": "job-plain"})

    api = _make_client(handler)
    generate = make_generate_handler(api)
    gen = generate(
        "prompt", "", *_kf_args(),
        512, 320, False, 0, 0, 49, 24.0, -1,
        src_audio=None,
    )
    _run_until_job_started(gen)
    assert seen["url"] == "http://test/api/v1/generate"  # not /generate/chain


# --------------------------------------------------------------------------- #
# S4: format_api_error — the shared error-envelope formatter.
# --------------------------------------------------------------------------- #
def test_format_api_error_reference_resolution_hint():
    body = {"error": {"code": "REFERENCE_RESOLUTION_INVALID",
                      "message": "reference-video jobs require width/height divisible by 128",
                      "detail": "width=1280, height=720"}}
    msg = format_api_error(body)
    assert "128" in msg              # actionable hint
    assert "width=1280" in msg       # detail appended


def test_format_api_error_validation_list_renders_loc_msg():
    body = {"error": {"code": "VALIDATION_ERROR", "message": "Request validation failed",
                      "detail": [
                          {"loc": ["body", "width"], "msg": "width must be a multiple of 64",
                           "type": "value_error"},
                          {"loc": ["body", "num_frames"], "msg": "num_frames must be 8n+1",
                           "type": "value_error"},
                      ]}}
    msg = format_api_error(body)
    assert "body.width: width must be a multiple of 64" in msg
    assert "body.num_frames: num_frames must be 8n+1" in msg


def test_format_api_error_unknown_code_falls_back_to_raw():
    body = {"error": {"code": "SOMETHING_WEIRD", "message": "boom"}}
    msg = format_api_error(body)
    assert "SOMETHING_WEIRD" in msg  # raw text (no hint for unknown code)


def test_format_api_error_raw_text_when_not_a_dict():
    assert format_api_error("upstream 502 bad gateway") == "upstream 502 bad gateway"


def test_format_api_error_japanese_hint():
    body = {"error": {"code": "GPU_OOM", "message": "CUDA OOM"}}
    msg = format_api_error(body, lang="ja")
    assert "GPU" in msg  # Japanese hint mentions GPU memory


# --------------------------------------------------------------------------- #
# S4: build_adapter_choices — dynamic from /config, static fallback.
# --------------------------------------------------------------------------- #
def test_build_adapter_choices_from_config_with_unknown_key():
    cfg = {"model": {"ic_loras": {
        "pixel-spatial-upscaler-x2": "path",
        "canny-control": {"path": "x", "preprocess": "canny"},
        "pose-control": {"path": "y", "preprocess": "dwpose"},
        "my-custom-adapter": "z",
    }}}
    choices = build_adapter_choices(cfg)
    values = [v for _label, v in choices]
    assert values[0] == ADAPTER_NONE
    assert "pixel-spatial-upscaler-x2" in values
    assert "my-custom-adapter" in values
    labels = {v: label for label, v in choices}
    assert labels["canny-control"] == "Canny edge control (canny-control)"
    assert labels["my-custom-adapter"] == "my-custom-adapter"  # unknown shown as-is


def test_build_adapter_choices_fallback_when_no_ic_loras():
    values = [v for _label, v in build_adapter_choices({})]
    assert values == [ADAPTER_NONE, "pixel-spatial-upscaler-x2",
                      "canny-control", "pose-control"]


# --------------------------------------------------------------------------- #
# S5: Clip Chain handler. The 24 FIXED clip slots are flattened into positional
# args; only slot 1 carries a start image + strength. ``_chain_args`` builds the
# full arg tuple from as few clip specs as a test cares about (rest disabled).
# --------------------------------------------------------------------------- #
def _chain_args(prompt="Base prompt", negative="", width=1280, height=768,
                crop_enabled=False, crop_w=0, crop_h=0, fps=24.0, seed=-1,
                overlap=3, overlap_strength=0.5, clips=None, config=None):
    clips = list(clips or [])
    filled = clips + [None] * (24 - len(clips))
    args = [prompt, negative, width, height, crop_enabled, crop_w, crop_h, fps, seed,
            overlap, overlap_strength]
    for i, spec in enumerate(filled[:24]):
        spec = spec or {}
        enabled = spec.get("enabled", False)
        p = spec.get("prompt", "")
        frames = spec.get("frames", 121)
        if i == 0:  # slot 1 has image + strength
            args.extend([enabled, p, frames, spec.get("image"), spec.get("strength", 0.8)])
        else:
            args.extend([enabled, p, frames])
    args.append(config)
    return args


def _run_chain_until_started(gen):
    """Drive the chain generator until the /generate/chain POST produced a job
    id, then close it so the poll loop's 1s sleeps never run. Returns all yields."""
    outs = []
    for out in gen:
        outs.append(out)
        if out[1]:  # job started
            gen.close()
            break
    return outs


def test_chain_payload_two_clips():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://test/api/v1/generate/chain"
        import json
        captured.update(json.loads(request.content))
        return httpx.Response(202, json={"job_id": "chain-1"})

    api = _make_client(handler)
    chain = make_chain_handler(api)
    gen = chain(*_chain_args(
        prompt="Shared base", negative="blurry", width=1280, height=768, fps=24.0, seed=7,
        overlap=3, overlap_strength=0.5,
        clips=[
            {"enabled": True, "prompt": "Clip one override", "frames": 121},
            {"enabled": True, "prompt": "", "frames": 129},  # blank -> omitted
        ],
    ))
    outs = _run_chain_until_started(gen)

    assert captured["prompt"] == "Shared base"
    assert captured["negative_prompt"] == "blurry"
    assert captured["width"] == 1280 and captured["height"] == 768
    assert captured["frame_rate"] == 24.0
    assert captured["seed"] == 7
    assert captured["num_inference_steps"] == 8
    assert captured["guidance_scale"] == 1.0
    assert captured["pipeline"] == "distilled"
    assert captured["overlap_frames"] == 3
    assert captured["overlap_strength"] == 0.5
    assert captured["crop_output"] is None
    # order + per-clip prompt omission when blank; no conditioning (no image).
    assert captured["clips"] == [
        {"num_frames": 121, "prompt": "Clip one override"},
        {"num_frames": 129},
    ]
    assert outs[-1][1] == "chain-1"
    assert "chain-1" in outs[-1][0]


def test_chain_payload_three_clips_order_and_prompt_omission():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        captured.update(json.loads(request.content))
        return httpx.Response(202, json={"job_id": "chain-3"})

    api = _make_client(handler)
    chain = make_chain_handler(api)
    gen = chain(*_chain_args(clips=[
        {"enabled": True, "prompt": "", "frames": 121},        # blank
        {"enabled": True, "prompt": "second", "frames": 121},
        {"enabled": True, "prompt": "third", "frames": 121},
    ]))
    _run_chain_until_started(gen)
    assert captured["clips"] == [
        {"num_frames": 121},
        {"num_frames": 121, "prompt": "second"},
        {"num_frames": 121, "prompt": "third"},
    ]


def test_chain_payload_chunked_upsample_flag():
    # The chunked-upsample checkbox (bound after v2v_context) rides into the
    # /generate/chain payload as a bool. _chain_args stops at ``config``, so it is
    # passed as a keyword here (the real UI binds it as a trailing positional).
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        captured.update(json.loads(request.content))
        return httpx.Response(202, json={"job_id": "chain-cu"})

    api = _make_client(handler)
    chain = make_chain_handler(api)
    gen = chain(*_chain_args(clips=[
        {"enabled": True, "prompt": "", "frames": 121},
        {"enabled": True, "prompt": "", "frames": 121},
    ]), chunked_upsample=True)
    _run_chain_until_started(gen)
    assert captured["chunked_upsample"] is True


def test_chain_clip0_conditioning_present_only_when_image_set(tmp_path):
    img = tmp_path / "start.png"
    img.write_bytes(b"\x89PNG\r\n")
    uploads = {"n": 0}
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/upload/image"):
            uploads["n"] += 1
            return httpx.Response(200, json={"image_id": "img-c0"})
        import json
        captured.update(json.loads(request.content))
        return httpx.Response(202, json={"job_id": "chain-i2v"})

    api = _make_client(handler)
    chain = make_chain_handler(api)
    gen = chain(*_chain_args(clips=[
        {"enabled": True, "prompt": "", "frames": 121, "image": str(img), "strength": 0.65},
        {"enabled": True, "prompt": "", "frames": 121},
    ]))
    _run_chain_until_started(gen)
    assert uploads["n"] == 1
    assert captured["clips"][0] == {
        "num_frames": 121,
        "conditioning_images": [{"image_id": "img-c0", "frame_idx": 0, "strength": 0.65}],
    }
    # clip 1 never carries conditioning.
    assert "conditioning_images" not in captured["clips"][1]


def test_chain_clip0_no_conditioning_key_when_no_image():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert not request.url.path.endswith("/upload/image")
        import json
        captured.update(json.loads(request.content))
        return httpx.Response(202, json={"job_id": "chain-t2v"})

    api = _make_client(handler)
    chain = make_chain_handler(api)
    gen = chain(*_chain_args(clips=[
        {"enabled": True, "frames": 121},
        {"enabled": True, "frames": 121},
    ]))
    _run_chain_until_started(gen)
    assert "conditioning_images" not in captured["clips"][0]


def test_chain_fewer_than_two_clips_errors_zero_calls():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(202, json={"job_id": "x"})

    api = _make_client(handler)
    chain = make_chain_handler(api)
    out = list(chain(*_chain_args(clips=[{"enabled": True, "frames": 121}])))
    assert calls["n"] == 0  # no API call
    assert len(out) == 1
    progress, job_id, video = out[0]
    assert job_id == "" and video is None


def test_chain_bad_num_frames_errors_zero_calls():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(202, json={"job_id": "x"})

    api = _make_client(handler)
    chain = make_chain_handler(api)
    # 120 is not 8n+1 (119 or 121 would be) -> precheck fires.
    out = list(chain(*_chain_args(clips=[
        {"enabled": True, "frames": 121},
        {"enabled": True, "frames": 120},
    ])))
    assert calls["n"] == 0
    assert len(out) == 1
    assert out[0][1] == "" and out[0][2] is None


def test_chain_empty_prompt_fires_toast_warning_zero_calls(monkeypatch):
    # A chain-precheck rejection must ALSO surface as a gr.Warning toast --
    # the progress textbox alone was missed by a real user ("the button does
    # nothing"). gr.Warning is a plain notification call, so capture it.
    from gradio_ui import handlers as handlers_mod

    toasts: list[str] = []
    monkeypatch.setattr(
        handlers_mod.gr, "Warning",
        lambda message, *args, **kwargs: toasts.append(message),
    )

    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(202, json={"job_id": "x"})

    api = _make_client(handler)
    chain = make_chain_handler(api)
    out = list(chain(*_chain_args(prompt="", clips=[
        {"enabled": True, "frames": 121},
        {"enabled": True, "frames": 121},
    ])))

    assert calls["n"] == 0  # rejected before any API call
    assert len(out) == 1
    assert out[0][1] == "" and out[0][2] is None
    # exactly one toast, carrying the SAME localized message as the textbox.
    assert toasts == [out[0][0]]


def test_chain_overlap_too_large_for_shortest_clip_errors_zero_calls():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(202, json={"job_id": "x"})

    api = _make_client(handler)
    chain = make_chain_handler(api)
    # shortest clip 9 frames -> stage-1 latent frames = 2 -> max overlap = 1;
    # overlap 3 is too large.
    out = list(chain(*_chain_args(overlap=3, clips=[
        {"enabled": True, "frames": 9},
        {"enabled": True, "frames": 121},
    ])))
    assert calls["n"] == 0
    assert len(out) == 1
    assert out[0][1] == "" and out[0][2] is None


def test_chain_total_frames_cap_violation_via_helper():
    # The 11544 cap is unreachable through the 24-slot handler (24×481 minus the
    # join overlaps stays under it), so exercise the mirror-math helper directly
    # with more clips than the UI can request.
    err = check_chain_total([481] * 26, 24.0, 1)
    assert err is not None
    assert str(MAX_CHAIN_TOTAL_PIXEL_FRAMES) in err  # "11544"


def test_chain_total_frames_within_cap_returns_none():
    assert check_chain_total([121, 129], 24.0, 3) is None


def test_chain_geometry_degenerate_short_clips_via_helper():
    # Clips too short for a continuous audio cross-fade -> chain_math raises;
    # helper surfaces it as a localized geometry error (not None).
    err = check_chain_total([9, 9], 24.0, 1)
    assert err is not None


# --------------------------------------------------------------------------- #
# Chain LoRA (A2V+LoRA解禁): the SHARED (draft) prompt feeds the chain flow. Clip
# chaining now wires style/character <lora:...> tokens into the chain payload's
# ``loras`` (uniform strength across the chain); a token-free prompt is forwarded
# byte-identical, and per-clip prompts are left untouched (out of scope).
# --------------------------------------------------------------------------- #
def test_chain_shared_prompt_loras_wired_and_stripped():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/loras") and request.method == "GET":
            return httpx.Response(200, json={"loras": [
                {"name": "neon-city", "kind": "style", "has_thumbnail": False,
                 "exists": True, "source": "scan"},
            ]})
        assert str(request.url) == "http://test/api/v1/generate/chain"
        import json
        captured.update(json.loads(request.content))
        return httpx.Response(202, json={"job_id": "chain-lora"})

    api = _make_client(handler)
    chain = make_chain_handler(api)
    gen = chain(*_chain_args(prompt="a city <lora:neon-city:0.7> at dusk", clips=[
        {"enabled": True, "frames": 121},
        {"enabled": True, "frames": 121},
    ]))
    _run_chain_until_started(gen)
    # token removed + whitespace collapsed; wired into loras (uniform strength).
    assert captured["prompt"] == "a city at dusk"
    assert captured["loras"] == [{"name": "neon-city", "strength": 0.7}]


def test_chain_unknown_shared_prompt_lora_aborts_zero_calls():
    posts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/loras") and request.method == "GET":
            return httpx.Response(200, json={"loras": [
                {"name": "neon-city", "kind": "style", "has_thumbnail": False,
                 "exists": True, "source": "scan"},
            ]})
        if request.method == "POST":
            posts["n"] += 1
        return httpx.Response(202, json={"job_id": "x"})

    api = _make_client(handler)
    chain = make_chain_handler(api)
    out = list(chain(*_chain_args(prompt="a city <lora:missing:0.7> at dusk", clips=[
        {"enabled": True, "frames": 121},
        {"enabled": True, "frames": 121},
    ]))
    )
    assert posts["n"] == 0  # unknown token -> no upload, no generate/chain
    assert out[-1][1] == "" and out[-1][2] is None


def test_chain_token_free_shared_prompt_unchanged_no_warning(monkeypatch):
    from gradio_ui import handlers as handlers_mod

    toasts: list[str] = []
    monkeypatch.setattr(
        handlers_mod.gr, "Warning",
        lambda message, *args, **kwargs: toasts.append(message),
    )

    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        captured.update(json.loads(request.content))
        return httpx.Response(202, json={"job_id": "chain-plain"})

    api = _make_client(handler)
    chain = make_chain_handler(api)
    gen = chain(*_chain_args(prompt="a plain shared prompt", clips=[
        {"enabled": True, "frames": 121},
        {"enabled": True, "frames": 121},
    ]))
    _run_chain_until_started(gen)
    assert captured["prompt"] == "a plain shared prompt"  # byte-identical
    assert toasts == []  # no <lora:...> token -> no warning


def test_chain_per_clip_prompt_lora_is_preserved():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        captured.update(json.loads(request.content))
        return httpx.Response(202, json={"job_id": "chain-clip-lora"})

    api = _make_client(handler)
    chain = make_chain_handler(api)
    gen = chain(*_chain_args(prompt="base", clips=[
        {"enabled": True, "prompt": "hero <lora:foo:1.0>", "frames": 121},
        {"enabled": True, "frames": 121},
    ]))
    _run_chain_until_started(gen)
    # shared prompt has no token -> unchanged; per-clip token left as authored.
    assert captured["prompt"] == "base"
    assert captured["clips"][0]["prompt"] == "hero <lora:foo:1.0>"


def test_prompt_unification_i18n_keys_present_both_langs():
    from gradio_ui import LABELS

    for k in (
        "info_negative",
        "nag_accordion", "nag_note", "nag_enable",
        "nag_lbl_method", "nag_method_nag", "nag_method_vsf",
        "nag_lbl_scale", "nag_lbl_tau", "nag_lbl_alpha",
        "nag_msg_negative_required",
        "vsf_lbl_scale", "vsf_lbl_scale_info",
    ):
        assert LABELS["en"].get(k), f"missing EN: {k}"
        assert LABELS["ja"].get(k), f"missing JA: {k}"


def test_chain_409_reports_busy():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(409, json={"error": {"code": "JOB_BUSY"}})

    api = _make_client(handler)
    chain = make_chain_handler(api)
    out = list(chain(*_chain_args(clips=[
        {"enabled": True, "frames": 121},
        {"enabled": True, "frames": 121},
    ])))
    progress, job_id, video = out[-1]
    assert "409" in progress
    assert job_id == ""


def test_chain_validation_error_formatted():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"error": {
            "code": "VALIDATION_ERROR", "message": "bad",
            "detail": [{"loc": ["body", "clips", 0, "num_frames"],
                        "msg": "must be 8n+1", "type": "value_error"}],
        }})

    api = _make_client(handler)
    chain = make_chain_handler(api)
    out = list(chain(*_chain_args(clips=[
        {"enabled": True, "frames": 121},
        {"enabled": True, "frames": 121},
    ])))
    assert "must be 8n+1" in out[-1][0]
    assert out[-1][1] == ""


# --------------------------------------------------------------------------- #
# S6: Jobs tab — ApiClient list_jobs / delete_job paths.
# --------------------------------------------------------------------------- #
def test_list_jobs_path():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert str(request.url) == "http://test/api/v1/jobs"
        return httpx.Response(200, json=[{"job_id": "j1", "status": "completed"}])

    api = _make_client(handler)
    jobs = api.list_jobs()
    assert jobs[0]["job_id"] == "j1"


def test_delete_job_path_and_method():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"job_id": "j9", "deleted": True})

    api = _make_client(handler)
    resp = api.delete_job("j9")
    assert seen["method"] == "DELETE"
    assert seen["url"] == "http://test/api/v1/jobs/j9"
    assert resp["deleted"] is True


# --------------------------------------------------------------------------- #
# S6: build_jobs_rows — Dataframe rows from a fake /jobs list.
# --------------------------------------------------------------------------- #
def _fake_jobs() -> list:
    return [
        {"job_id": "a3f8", "status": "completed", "progress": 1.0,
         "created_at": "2026-07-04T14:22:07", "completed_at": "2026-07-04T14:25:41",
         "error": None},
        {"job_id": "b7d1", "status": "running", "progress": 0.62,
         "created_at": "2026-07-04T14:31:55", "completed_at": None, "error": None},
        {"job_id": "c0e4", "status": "failed", "progress": 0.0,
         "created_at": "2026-07-04T13:58:40", "completed_at": "2026-07-04T13:59:10",
         "error": "GPU_OOM: CUDA out of memory (detail)"},
    ]


def test_build_jobs_rows_columns_and_formatting():
    rows = build_jobs_rows(_fake_jobs())
    assert len(rows) == 3
    # completed -> 100%, no error summary.
    assert rows[0] == ["a3f8", "completed", "100%", "2026-07-04T14:22:07",
                       "2026-07-04T14:25:41", ""]
    # running -> 62%.
    assert rows[1][2] == "62%"
    # failed -> progress dash + error CODE summary.
    assert rows[2][1] == "failed"
    assert rows[2][2] == "—"
    assert rows[2][5] == "GPU_OOM"


def test_build_jobs_rows_handles_missing_and_empty():
    assert build_jobs_rows([]) == []
    assert build_jobs_rows(None) == []
    rows = build_jobs_rows([{"job_id": "x", "status": "queued", "progress": None}])
    assert rows[0][2] == "—"  # no progress -> dash


def test_jobs_table_headers_localized():
    en = jobs_table_headers("en")
    ja = jobs_table_headers("ja")
    assert en[0] == "Job ID" and ja[0] == "ジョブID"
    assert len(en) == 6 and len(ja) == 6


# --------------------------------------------------------------------------- #
# S6: format_job_error — parse "CODE: message (detail)" and reuse apierr_* hints.
# --------------------------------------------------------------------------- #
def test_format_job_error_known_code_english_hint():
    msg = format_job_error("GPU_OOM: CUDA out of memory (detail)")
    assert "Out of GPU memory" in msg          # localized apierr_GPU_OOM hint
    assert "CUDA out of memory (detail)" in msg  # raw server text kept


def test_format_job_error_known_code_japanese_hint():
    msg = format_job_error("GENERATION_FAILED: boom", lang="ja")
    assert "生成に失敗" in msg  # localized apierr_GENERATION_FAILED (ja)


def test_format_job_error_unknown_prefix_returns_raw():
    raw = "SOMETHING_WEIRD: unexpected failure"
    assert format_job_error(raw) == raw


def test_format_job_error_none_and_non_string():
    assert format_job_error(None) == ""
    assert format_job_error("") == ""


# --------------------------------------------------------------------------- #
# S6: delete_finished_jobs — client-side loop deletes ONLY terminal jobs.
# --------------------------------------------------------------------------- #
def test_delete_finished_jobs_only_terminal_states():
    deleted = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path.endswith("/jobs"):
            return httpx.Response(200, json=[
                {"job_id": "done1", "status": "completed"},
                {"job_id": "run1", "status": "running"},
                {"job_id": "fail1", "status": "failed"},
                {"job_id": "queue1", "status": "queued"},
                {"job_id": "cancel1", "status": "cancelled"},
            ])
        if request.method == "DELETE":
            deleted.append(request.url.path.rsplit("/", 1)[-1])
            return httpx.Response(200, json={"job_id": "x", "deleted": True})
        return httpx.Response(404, json={})

    api = _make_client(handler)
    msg = delete_finished_jobs(api)
    # running + queued are active -> never deleted.
    assert sorted(deleted) == ["cancel1", "done1", "fail1"]
    assert "3" in msg


def test_delete_finished_jobs_list_failure_localized():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"code": "X"}})

    api = _make_client(handler)
    msg = delete_finished_jobs(api, lang="ja")
    assert "失敗" in msg  # msg_purge_failed (ja)


# --------------------------------------------------------------------------- #
# S6: polling cadence — _poll_job_until_done honours a custom interval/timeout,
# and _resolve_poll plumbs the Settings gr.Number values (with safe defaults).
# --------------------------------------------------------------------------- #
def test_resolve_poll_defaults_and_custom():
    from gradio_ui.handlers import _resolve_poll

    assert _resolve_poll(None, None) == (1.0, 7200.0)
    assert _resolve_poll(0.5, 2) == (0.5, 120.0)
    # non-positive / invalid inputs fall back to the 1s / 120min defaults.
    assert _resolve_poll(0, 0) == (1.0, 7200.0)
    assert _resolve_poll("x", "y") == (1.0, 7200.0)


def test_poll_respects_custom_interval(monkeypatch):
    from gradio_ui import handlers

    slept: list[float] = []
    monkeypatch.setattr(handlers.time, "sleep", lambda s: slept.append(s))

    def handler(request: httpx.Request) -> httpx.Response:
        # Any request (poll or video fetch) returns a completed job body.
        return httpx.Response(200, json={"status": "completed", "progress": 1.0})

    api = _make_client(handler)
    outs = list(handlers._poll_job_until_done(api, "j1", "en",
                                              interval=0.25, timeout_s=10))
    assert slept and slept[0] == 0.25       # custom interval used
    assert outs[-1][1] == "j1"              # returned after completion


def test_poll_queued_emits_waiting_then_stuck(monkeypatch):
    # B-1: a queued job surfaces a "waiting" tick; once it stays queued past
    # QUEUED_WARN_SECONDS the wording flips to the "stuck — cancel from Jobs"
    # hint. The poll never aborts on its own — it keeps going until completion.
    from gradio_ui import handlers
    from gradio_ui.i18n import L

    monkeypatch.setattr(handlers.time, "sleep", lambda s: None)
    monkeypatch.setattr(handlers, "QUEUED_WARN_SECONDS", 3)

    # queued x5 (interval=1s -> elapsed 1..5) then completed.
    seq = iter(["queued", "queued", "queued", "queued", "queued", "completed"])

    def handler(request: httpx.Request) -> httpx.Response:
        try:
            status = next(seq)
        except StopIteration:
            status = "completed"  # trailing video-fetch request
        return httpx.Response(200, json={"status": status, "progress": 0.0})

    api = _make_client(handler)
    outs = list(handlers._poll_job_until_done(api, "jq", "en",
                                              interval=1.0, timeout_s=60))
    texts = [o[0] for o in outs]

    # Below threshold (elapsed 1, 2): plain waiting message with the second count.
    assert texts[0] == L("msg_queued", "en").format(secs=1)
    assert texts[1] == L("msg_queued", "en").format(secs=2)
    # At/after threshold (elapsed 3, 4, 5): stuck message.
    assert texts[2] == L("msg_queued_stuck", "en").format(secs=3)
    assert texts[4] == L("msg_queued_stuck", "en").format(secs=5)
    # Still reaches completion afterwards (never auto-aborted).
    assert outs[-1][1] == "jq"
    assert L("msg_completed", "en").format(job_id="jq") in texts[-1]


def test_queued_i18n_keys_present_both_langs():
    # B-1: both new queued keys exist (and are non-empty) in EN and JA.
    from gradio_ui import LABELS

    for k in ("msg_queued", "msg_queued_stuck"):
        assert LABELS["en"].get(k), f"missing EN: {k}"
        assert LABELS["ja"].get(k), f"missing JA: {k}"


def test_generate_runtime_lang_localizes_messages():
    # ui_lang is the trailing arg; an empty prompt short-circuits with the
    # Japanese message when ui_lang="ja".
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"job_id": "x"})

    api = _make_client(handler)
    generate = make_generate_handler(api)
    out = list(generate(
        "   ", "", *_kf_args(), 512, 320, False, 0, 0, 49, 24.0, -1,
        *_adapter_args(),  # adapter none + default strengths
        "ja",  # ui_lang
    ))
    assert "プロンプト" in out[0][0]  # msg_prompt_required (ja)


# --------------------------------------------------------------------------- #
# S6: language switch — build_ui exposes switch_language; it returns one update
# per registered component plus the two localized-header Dataframes.
# --------------------------------------------------------------------------- #
def test_language_switch_returns_update_per_registry_entry():
    from gradio_ui import build_ui

    demo = build_ui("http://127.0.0.1:8000", api_key="secret")
    registry = demo.label_registry
    updates = demo.switch_language("ja", {})
    # One update per registry entry + 2 extra header-only Dataframes.
    assert len(updates) == len(registry) + 2
    assert len(updates) == len(demo.lang_switch_outputs)
    # A Japanese label made it into the batch (spot-check).
    assert any(u.get("label") == "生成" or u.get("value") == "生成" for u in updates)
    # The last two updates carry localized Dataframe headers (jobs + spill).
    assert updates[-2]["headers"][0] == "ジョブID"
    assert updates[-1]["headers"] == ["解像度", "最大フレーム数"]


# --------------------------------------------------------------------------- #
# Settings /config bug fix: build_spill_rows (pure formatter for the Settings
# spill-free Dataframe) + fetch_config_safe / on_config_retry_tick (the
# testable retry logic that replaces the old "swallow into {} forever" bug).
# --------------------------------------------------------------------------- #
def test_build_spill_rows_normal():
    from gradio_ui.ui import build_spill_rows

    cfg = {"limits": {"spill_free_frames": {"1280x768": 257, "1920x1088": 153}}}
    rows = build_spill_rows(cfg)
    assert rows == [["1280x768", 257], ["1920x1088", 153]]


def test_build_spill_rows_missing_key():
    from gradio_ui.ui import build_spill_rows

    # "limits" present but no "spill_free_frames" key -> empty rows, no crash.
    assert build_spill_rows({"limits": {}}) == []
    # "limits" itself absent.
    assert build_spill_rows({"generation_presets": {}}) == []


def test_build_spill_rows_empty_config():
    from gradio_ui.ui import build_spill_rows

    assert build_spill_rows({}) == []
    assert build_spill_rows(None) == []


def test_fetch_config_safe_success():
    from gradio_ui.handlers import fetch_config_safe

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"limits": {"spill_free_frames": {"512x320": 49}}})

    api = _make_client(handler)
    cfg, err = fetch_config_safe(api)
    assert err is None
    assert cfg["limits"]["spill_free_frames"]["512x320"] == 49


def test_fetch_config_safe_failure_returns_warning_not_raise():
    from gradio_ui.handlers import fetch_config_safe

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    api = _make_client(handler)
    cfg, err = fetch_config_safe(api)
    assert cfg is None
    assert err is not None
    assert "Failed to load server settings" in err


def test_fetch_config_safe_failure_localized_japanese():
    from gradio_ui.handlers import fetch_config_safe

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={})

    api = _make_client(handler)
    _cfg, err = fetch_config_safe(api, lang="ja")
    assert "サーバ設定の取得に失敗" in err


def test_on_config_retry_tick_success_stops_timer():
    from gradio_ui.handlers import on_config_retry_tick

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"limits": {}})

    api = _make_client(handler)
    cfg, err, next_attempt, keep_retrying = on_config_retry_tick(api, attempt=1)
    assert err is None
    assert cfg == {"limits": {}}
    assert next_attempt == 1  # unchanged on success
    assert keep_retrying is False


def test_on_config_retry_tick_failure_below_budget_keeps_retrying():
    from gradio_ui.handlers import CONFIG_RETRY_MAX_ATTEMPTS, on_config_retry_tick

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={})

    api = _make_client(handler)
    cfg, err, next_attempt, keep_retrying = on_config_retry_tick(api, attempt=0)
    assert cfg is None
    assert err is not None
    assert next_attempt == 1
    assert next_attempt < CONFIG_RETRY_MAX_ATTEMPTS
    assert keep_retrying is True


def test_on_config_retry_tick_exhausted_stops_and_final_message():
    from gradio_ui.handlers import CONFIG_RETRY_MAX_ATTEMPTS, on_config_retry_tick

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={})

    api = _make_client(handler)
    cfg, err, next_attempt, keep_retrying = on_config_retry_tick(
        api, attempt=CONFIG_RETRY_MAX_ATTEMPTS - 1)
    assert cfg is None
    assert next_attempt == CONFIG_RETRY_MAX_ATTEMPTS
    assert keep_retrying is False
    assert str(CONFIG_RETRY_MAX_ATTEMPTS) in err  # exhaustion count in the message
    assert "Use Refresh" in err


def test_on_config_retry_tick_localized_japanese_exhausted():
    from gradio_ui.handlers import CONFIG_RETRY_MAX_ATTEMPTS, on_config_retry_tick

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={})

    api = _make_client(handler)
    _cfg, err, _next, keep_retrying = on_config_retry_tick(
        api, attempt=CONFIG_RETRY_MAX_ATTEMPTS - 1, lang="ja")
    assert keep_retrying is False
    assert "更新ボタン" in err


# --------------------------------------------------------------------------- #
# Settings /config bug fix, wired end-to-end through build_ui: on_page_load
# and on_config_retry are exposed on the returned demo (like switch_language),
# and demo.api is the SAME ApiClient the Blocks graph calls -- swapping in a
# mock transport exercises the exact closures wired to demo.load /
# config_retry_timer.tick, without a live server.
# --------------------------------------------------------------------------- #
def test_on_page_load_success_populates_and_disables_timer():
    from gradio_ui import build_ui

    demo = build_ui("http://127.0.0.1:8000", api_key=None)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/status"):
            return httpx.Response(200, json={"server": "running", "version": "0.4.0",
                                              "pipeline_loaded": False,
                                              "gpu": {"name": None, "vram_free_mb": 0,
                                                      "vram_total_mb": 0},
                                              "vram_optimization": {"low_vram_mode": False,
                                                                    "low_vram_profile": "d"}})
        return httpx.Response(200, json={
            "generation_presets": {}, "limits": {"spill_free_frames": {"512x320": 49}},
        })

    demo.api._client = httpx.Client(transport=httpx.MockTransport(handler))
    (status, cfg, preset_upd, adapter_upd, server_json_upd, spill_upd,
     timer_upd) = demo.on_page_load({}, "en")

    assert cfg["limits"]["spill_free_frames"]["512x320"] == 49
    assert spill_upd["value"] == [["512x320", 49]]
    assert server_json_upd["value"] == cfg
    assert timer_upd["active"] is False  # success -> timer stays off


def test_on_page_load_failure_warns_keeps_state_and_arms_timer():
    from gradio_ui import build_ui

    demo = build_ui("http://127.0.0.1:8000", api_key=None)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/status"):
            return httpx.Response(500, json={})
        return httpx.Response(500, json={"error": "server not ready"})

    demo.api._client = httpx.Client(transport=httpx.MockTransport(handler))
    stale_cfg = {"limits": {"spill_free_frames": {"stale": 1}}}
    with pytest.warns(UserWarning):
        (_status, cfg, preset_upd, adapter_upd, server_json_upd, spill_upd,
         timer_upd) = demo.on_page_load(stale_cfg, "en")

    # No-op-on-failure: existing good state is NOT clobbered with {} (a plain
    # gr.update() carries no "value"/"choices" key, so nothing changes).
    assert cfg is stale_cfg
    assert "choices" not in preset_upd
    assert "choices" not in adapter_upd
    assert "value" not in server_json_upd
    assert "value" not in spill_upd
    assert timer_upd["active"] is True  # failure -> retry timer armed


def test_on_config_retry_success_repopulates_and_stops_timer():
    from gradio_ui import build_ui

    demo = build_ui("http://127.0.0.1:8000", api_key=None)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "generation_presets": {}, "limits": {"spill_free_frames": {"1280x768": 257}},
        })

    demo.api._client = httpx.Client(transport=httpx.MockTransport(handler))
    (cfg, preset_upd, adapter_upd, server_json_upd, spill_upd,
     next_attempt, timer_upd) = demo.on_config_retry({}, 1, "en")

    assert cfg["limits"]["spill_free_frames"]["1280x768"] == 257
    assert spill_upd["value"] == [["1280x768", 257]]
    assert "choices" in preset_upd
    assert "choices" in adapter_upd
    assert next_attempt == 1
    assert timer_upd["active"] is False


def test_on_config_retry_dead_backend_no_op_until_exhausted():
    from gradio_ui import build_ui
    from gradio_ui.handlers import CONFIG_RETRY_MAX_ATTEMPTS

    demo = build_ui("http://127.0.0.1:8000", api_key=None)

    # Simulate a dead port: MockTransport that always raises a connection error.
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    demo.api._client = httpx.Client(transport=httpx.MockTransport(handler))
    stale_cfg = {"limits": {"spill_free_frames": {"stale": 1}}}

    # Not yet exhausted -> no-op on config/table, timer stays armed, no warning.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        (cfg, preset_upd, adapter_upd, server_json_upd, spill_upd,
         next_attempt, timer_upd) = demo.on_config_retry(stale_cfg, 0, "en")
    assert cfg is stale_cfg
    assert "choices" not in preset_upd and "choices" not in adapter_upd
    assert "value" not in server_json_upd and "value" not in spill_upd
    assert next_attempt == 1
    assert timer_upd["active"] is True

    # Exhausted -> stop retrying + final gr.Warning.
    with pytest.warns(UserWarning):
        (*_rest, last_attempt, timer_upd_final) = demo.on_config_retry(
            stale_cfg, CONFIG_RETRY_MAX_ATTEMPTS - 1, "en")
    assert last_attempt == CONFIG_RETRY_MAX_ATTEMPTS
    assert timer_upd_final["active"] is False


def test_language_switch_rebuilds_choice_components():
    from gradio_ui import build_ui

    demo = build_ui("http://127.0.0.1:8000", api_key=None)
    cfg = {"model": {"ic_loras": {"canny-control": {"path": "x"}}}}
    updates = demo.switch_language("ja", cfg)
    # Quality-mode radios get localized choices (value unchanged).
    qmode_updates = [u for u in updates if "choices" in u
                     and any(v == "distilled" for _lbl, v in u["choices"])]
    assert qmode_updates, "quality radios should get rebuilt localized choices"
    assert any("高速" in lbl for u in qmode_updates for lbl, _v in u["choices"])
    # Adapter dropdown choices are rebuilt from /config (localized "None").
    adapter_updates = [u for u in updates if "choices" in u
                       and any(v == ADAPTER_NONE for _lbl, v in u["choices"])]
    assert adapter_updates
    assert adapter_updates[0]["choices"][0][0] == "なし"  # adapter_none (ja)


# --------------------------------------------------------------------------- #
# F3 (G3 feedback): running-progress text — no "step None/None", optional
# localized stage label appended when the backend reports one.
# --------------------------------------------------------------------------- #
def test_running_progress_without_steps_hides_step_suffix():
    from gradio_ui.handlers import _format_running_progress

    text = _format_running_progress({"progress": 0.42}, "en")
    assert "42%" in text
    assert "None" not in text and "step" not in text

    text_ja = _format_running_progress({"progress": 0.42}, "ja")
    assert "42%" in text_ja and "None" not in text_ja


def test_running_progress_with_steps_keeps_classic_format():
    from gradio_ui.handlers import _format_running_progress

    text = _format_running_progress(
        {"progress": 0.375, "current_step": 3, "total_steps": 8}, "en"
    )
    assert "(step 3/8)" in text and "38%" in text


@pytest.mark.parametrize(
    "stage,en_label,ja_label",
    [
        ("encode", "Encoding", "エンコード中"),
        ("stage1_denoise", "Denoising (stage 1)", "デノイズ中 (stage 1)"),
        ("stage1", "Denoising (stage 1)", "デノイズ中 (stage 1)"),
        ("stage2_denoise", "Denoising (stage 2)", "デノイズ中 (stage 2)"),
        ("tile", "Upsampling", "アップサンプル中"),
        ("decode", "Decoding", "デコード中"),
    ],
)
def test_running_progress_stage_labels(stage, en_label, ja_label):
    from gradio_ui.handlers import _format_running_progress

    job = {"progress": 0.5, "current_step": 4, "total_steps": 8, "stage": stage}
    assert en_label in _format_running_progress(job, "en")
    assert ja_label in _format_running_progress(job, "ja")


def test_running_progress_unknown_or_absent_stage_shows_no_label():
    from gradio_ui.handlers import _format_running_progress

    assert "—" not in _format_running_progress({"progress": 0.5}, "en")
    assert "—" not in _format_running_progress(
        {"progress": 0.5, "stage": "mystery_stage"}, "en"
    )


def test_poll_loop_uses_graceful_progress_text(monkeypatch):
    """End-to-end through _poll_job_until_done: a running job with step=None
    yields the percent-only line; the poll machinery itself is untouched."""
    from gradio_ui import handlers

    monkeypatch.setattr(handlers.time, "sleep", lambda s: None)
    bodies = iter(
        [
            {"status": "running", "progress": 0.03,
             "current_step": None, "total_steps": None, "stage": "encode"},
            {"status": "running", "progress": 0.2,
             "current_step": 3, "total_steps": 8, "stage": "stage1_denoise"},
            {"status": "failed", "error": "X: boom"},
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=next(bodies))

    api = _make_client(handler)
    outs = [t for t, _jid, _vid in handlers._poll_job_until_done(api, "j1", "en")]
    assert "None" not in outs[0]
    assert "Encoding" in outs[0]
    assert "(step 3/8)" in outs[1] and "Denoising (stage 1)" in outs[1]


# --------------------------------------------------------------------------- #
# Chain clip progress: "clip n/N" appended while a chain job reports which clip
# it is on (additive JobResponse fields), and "all N clips processed" on the
# completion line. Non-chain jobs (no clip fields) are byte-identical.
# --------------------------------------------------------------------------- #
def test_running_progress_appends_clip_position():
    from gradio_ui.handlers import _format_running_progress

    job = {"progress": 0.2, "current_step": 3, "total_steps": 8,
           "stage": "stage1_denoise", "clip": 1, "clip_count": 2}
    text = _format_running_progress(job, "en")
    assert "(step 3/8)" in text and "Denoising (stage 1)" in text
    assert "clip 1/2" in text
    text_ja = _format_running_progress(job, "ja")
    assert "クリップ 1/2" in text_ja


def test_running_progress_without_clip_fields_shows_no_clip_text():
    from gradio_ui.handlers import _format_running_progress

    job = {"progress": 0.2, "current_step": 3, "total_steps": 8,
           "stage": "stage1_denoise"}
    assert "clip" not in _format_running_progress(job, "en")
    assert "クリップ" not in _format_running_progress(job, "ja")


def test_poll_loop_completion_notes_all_clips_processed(monkeypatch):
    from gradio_ui import handlers

    monkeypatch.setattr(handlers.time, "sleep", lambda s: None)
    bodies = iter(
        [
            {"status": "running", "progress": 0.3, "current_step": 4,
             "total_steps": 8, "stage": "stage1_denoise",
             "clip": 2, "clip_count": 2},
            {"status": "completed", "clip_count": 2},
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/video"):
            return httpx.Response(404)  # fetch_video fails -> video None path
        return httpx.Response(200, json=next(bodies))

    api = _make_client(handler)
    outs = [t for t, _jid, _vid in handlers._poll_job_until_done(api, "j1", "en")]
    assert "clip 2/2" in outs[0]
    assert "all 2 clips processed" in outs[-1]


def test_poll_loop_completion_without_clip_count_is_unchanged(monkeypatch):
    from gradio_ui import handlers

    monkeypatch.setattr(handlers.time, "sleep", lambda s: None)
    bodies = iter([{"status": "completed"}])

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/video"):
            return httpx.Response(404)
        return httpx.Response(200, json=next(bodies))

    api = _make_client(handler)
    outs = [t for t, _jid, _vid in handlers._poll_job_until_done(api, "j1", "en")]
    assert outs[-1] == "Completed: j1"  # single generate: no clip note


# --------------------------------------------------------------------------- #
# Feature 1: A2V audio-length -> Frames auto-adjust
# (suggest_frames_for_audio + a2v_audio_change_handler).
# --------------------------------------------------------------------------- #
def test_suggest_frames_for_audio_5s_24fps_matches_hand_derived_value():
    from gradio_ui.handlers import suggest_frames_for_audio

    # nf = ((floor(5*24) - 1) // 8) * 8 + 1 = ((120-1)//8)*8+1 = (14*8)+1 = 113.
    # Preflight check (kv=3): required = audio_latents_required([113], 24, kv=3)
    # = 118 latent frames; available = round(5*25) = 125 >= 118, no shrink.
    assert suggest_frames_for_audio(5.0, 24.0) == 113


def test_suggest_frames_for_audio_2s_24fps_matches_hand_derived_value():
    from gradio_ui.handlers import suggest_frames_for_audio

    # nf = ((floor(2*24) - 1) // 8) * 8 + 1 = ((48-1)//8)*8+1 = (5*8)+1 = 41.
    # required = audio_latents_required([41], 24, kv=3) = 43; available =
    # round(2*25) = 50 >= 43, no shrink.
    assert suggest_frames_for_audio(2.0, 24.0) == 41


def test_suggest_frames_for_audio_30s_clamped_to_481():
    from gradio_ui.handlers import suggest_frames_for_audio

    # Raw formula gives 713 (way above the server's num_frames ceiling);
    # clamped down to 481, which is itself on the 8n+1 grid (8*60+1).
    assert suggest_frames_for_audio(30.0, 24.0) == 481


def test_suggest_frames_for_audio_result_always_on_8n_plus_1_grid():
    from gradio_ui.handlers import suggest_frames_for_audio

    for dur in (0.1, 0.5, 1.0, 2.0, 3.7, 5.0, 9.9, 17.3, 30.0, 60.0):
        nf = suggest_frames_for_audio(dur, 24.0)
        assert 9 <= nf <= 481
        assert (nf - 1) % 8 == 0


def test_suggest_frames_for_audio_falls_back_to_24fps_on_falsy_or_bad_fps():
    from gradio_ui.handlers import suggest_frames_for_audio

    baseline = suggest_frames_for_audio(6.0, 24.0)
    assert suggest_frames_for_audio(6.0, None) == baseline
    assert suggest_frames_for_audio(6.0, 0) == baseline
    assert suggest_frames_for_audio(6.0, "not-a-number") == baseline


def test_suggest_frames_for_audio_very_short_clamps_to_9():
    from gradio_ui.handlers import suggest_frames_for_audio

    assert suggest_frames_for_audio(0.1, 24.0) == 9


def test_a2v_audio_change_handler_real_wav_sets_num_frames_and_notifies(
        tmp_path, capsys):
    from gradio_ui.handlers import a2v_audio_change_handler, suggest_frames_for_audio

    wav = _real_wav(tmp_path / "five.wav", 5.0)
    expected = suggest_frames_for_audio(5.0, 24.0)
    update = a2v_audio_change_handler(wav, 24.0, "en")
    assert update["value"] == expected == 113
    out = capsys.readouterr().out
    assert str(expected) in out  # gr.Info degrades to a console print outside a queue


def test_a2v_audio_change_handler_recomputes_on_fps_change(tmp_path):
    from gradio_ui.handlers import a2v_audio_change_handler

    wav = _real_wav(tmp_path / "two.wav", 2.0)
    at_24fps = a2v_audio_change_handler(wav, 24.0, "en")["value"]
    at_30fps = a2v_audio_change_handler(wav, 30.0, "en")["value"]
    assert at_24fps == 41
    assert at_30fps == 57
    assert at_24fps != at_30fps


def test_a2v_audio_change_handler_overwrites_current_frames_value(tmp_path):
    # The handler itself has no notion of "current Frames value" -- ui.py
    # wires it straight to num_frames' outputs, so its gr.update ALWAYS
    # carries a fresh "value" key on a readable wav, unconditionally (i.e. it
    # does not merely no-op just because *some* value was already there).
    from gradio_ui.handlers import a2v_audio_change_handler

    wav = _real_wav(tmp_path / "five.wav", 5.0)
    update = a2v_audio_change_handler(wav, 24.0, "en")
    assert "value" in update and update["value"] == 113


def test_a2v_audio_change_handler_non_wav_is_noop(tmp_path):
    from gradio_ui.handlers import a2v_audio_change_handler

    mp3 = tmp_path / "clip.mp3"
    mp3.write_bytes(b"ID3....")
    update = a2v_audio_change_handler(str(mp3), 24.0, "en")
    assert "value" not in update


def test_a2v_audio_change_handler_none_path_is_noop():
    from gradio_ui.handlers import a2v_audio_change_handler

    update = a2v_audio_change_handler(None, 24.0, "en")
    assert "value" not in update


def test_a2v_audio_change_handler_unreadable_wav_is_noop(tmp_path):
    from gradio_ui.handlers import a2v_audio_change_handler

    bad = tmp_path / "broken.wav"
    bad.write_bytes(b"not a real wav file")
    update = a2v_audio_change_handler(str(bad), 24.0, "en")
    assert "value" not in update


def test_a2v_msg_frames_adjusted_i18n_keys_present():
    from gradio_ui.i18n import LABELS

    for lang in ("en", "ja"):
        assert "a2v_msg_frames_adjusted" in LABELS[lang]
        # Must actually format with the {frames}/{dur} kwargs the handler
        # passes (a KeyError here means the template and the call site drifted).
        LABELS[lang]["a2v_msg_frames_adjusted"].format(frames=113, dur=5.0)


# --------------------------------------------------------------------------- #
# build_a2v_chain_payload (WP0): the pure POST /generate/chain body builder
# extracted from make_generate_handler's A2V branch. The batch runner reuses it,
# so these lock the exact keys + key ORDER + the optional-key gating (loras only
# when non-empty, conditioning_images only when present, reference_* only under
# an adapter and each strength only below 1.0).
# --------------------------------------------------------------------------- #
def test_build_a2v_chain_payload_minimal_no_loras_no_ref_no_cond():
    from gradio_ui.handlers import build_a2v_chain_payload

    payload = build_a2v_chain_payload(
        audio_id="aud-1",
        num_frames=113,
        prompt="a cat",
        negative_prompt="",
        width=512,
        height=512,
        crop_output=None,
        frame_rate=24.0,
        seed=7,
        conditioning_images=[],
        loras=[],
        use_adapter=False,
    )
    assert payload == {
        "prompt": "a cat",
        "negative_prompt": "",
        "width": 512,
        "height": 512,
        "crop_output": None,
        "frame_rate": 24.0,
        "num_inference_steps": 8,
        "guidance_scale": 1.0,
        "seed": 7,
        "pipeline": "distilled",
        "overlap_frames": 3,
        "overlap_strength": 0.5,
        "clips": [{"num_frames": 113}],
        "source_audio": {"audio_id": "aud-1"},
    }
    # Optional keys must be entirely absent on the byte-identical baseline path.
    assert "loras" not in payload
    assert "reference_video_id" not in payload
    assert "conditioning_images" not in payload["clips"][0]
    # Key order is part of the frozen contract (dicts preserve insertion order).
    assert list(payload.keys()) == [
        "prompt", "negative_prompt", "width", "height", "crop_output",
        "frame_rate", "num_inference_steps", "guidance_scale", "seed",
        "pipeline", "overlap_frames", "overlap_strength", "clips", "source_audio",
    ]


def test_build_a2v_chain_payload_with_conditioning_and_crop():
    from gradio_ui.handlers import build_a2v_chain_payload

    cond = [{"image_id": "img-9", "frame_idx": 0, "strength": 1.0}]
    payload = build_a2v_chain_payload(
        audio_id="aud-2",
        num_frames=57,
        prompt="p",
        negative_prompt="ugly",
        width=768,
        height=512,
        crop_output={"width": 640, "height": 480},
        frame_rate=30.0,
        seed=0,
        conditioning_images=cond,
    )
    assert payload["clips"] == [{"num_frames": 57, "conditioning_images": cond}]
    assert payload["crop_output"] == {"width": 640, "height": 480}
    assert payload["negative_prompt"] == "ugly"
    assert "loras" not in payload
    assert "reference_video_id" not in payload


def test_build_a2v_chain_payload_adds_loras_only_when_non_empty():
    from gradio_ui.handlers import build_a2v_chain_payload

    loras = [{"name": "style", "strength": 0.8}]
    payload = build_a2v_chain_payload(
        audio_id="aud-3",
        num_frames=113,
        prompt="p",
        negative_prompt="",
        width=512,
        height=512,
        crop_output=None,
        frame_rate=24.0,
        seed=1,
        loras=loras,
    )
    assert payload["loras"] == loras
    # loras is appended AFTER source_audio (last of the base dict).
    assert list(payload.keys())[-1] == "loras"
    assert "reference_video_id" not in payload


def test_build_a2v_chain_payload_adapter_defaults_send_ref_id_only():
    from gradio_ui.handlers import build_a2v_chain_payload

    # Adapter used but both strengths at the 1.0 default: only reference_video_id
    # is added; the two S3 strength keys stay absent (byte-identical default).
    payload = build_a2v_chain_payload(
        audio_id="aud-4",
        num_frames=113,
        prompt="p",
        negative_prompt="",
        width=512,
        height=512,
        crop_output=None,
        frame_rate=24.0,
        seed=1,
        loras=[{"name": "canny", "strength": 1.0}],
        use_adapter=True,
        reference_video_id="vid-1",
        control_adherence=1.0,
        reference_strength=1.0,
    )
    assert payload["reference_video_id"] == "vid-1"
    assert "conditioning_attention_strength" not in payload
    assert "reference_video_strength" not in payload
    # Order: base dict -> loras -> reference_video_id.
    assert list(payload.keys())[-2:] == ["loras", "reference_video_id"]


def test_build_a2v_chain_payload_adapter_below_one_adds_strength_keys():
    from gradio_ui.handlers import build_a2v_chain_payload

    payload = build_a2v_chain_payload(
        audio_id="aud-5",
        num_frames=113,
        prompt="p",
        negative_prompt="",
        width=512,
        height=512,
        crop_output=None,
        frame_rate=24.0,
        seed=1,
        use_adapter=True,
        reference_video_id="vid-2",
        control_adherence=0.5,
        reference_strength=0.25,
    )
    assert payload["reference_video_id"] == "vid-2"
    assert payload["conditioning_attention_strength"] == 0.5
    assert payload["reference_video_strength"] == 0.25
    assert list(payload.keys())[-3:] == [
        "reference_video_id",
        "conditioning_attention_strength",
        "reference_video_strength",
    ]


def test_build_a2v_chain_payload_coerces_numeric_types():
    from gradio_ui.handlers import build_a2v_chain_payload

    # Mirrors the handler's int()/float() coercion so a batch caller passing
    # numeric strings emits the same JSON as the Generate tab.
    payload = build_a2v_chain_payload(
        audio_id="aud-6",
        num_frames="113",
        prompt="p",
        negative_prompt=None,
        width="512",
        height="512",
        crop_output=None,
        frame_rate="24",
        seed="9",
    )
    assert payload["clips"][0]["num_frames"] == 113
    assert payload["width"] == 512 and payload["height"] == 512
    assert payload["frame_rate"] == 24.0 and isinstance(payload["frame_rate"], float)
    assert payload["seed"] == 9
    # negative_prompt=None coalesces to "" exactly like the handler branch.
    assert payload["negative_prompt"] == ""


# --------------------------------------------------------------------------- #
# NAG (Normalized Attention Guidance / non-CFG Negative) -- additive keys on
# build_a2v_chain_payload, the single/chain handlers' request bodies, and the
# precheck that rejects an enabled-but-empty negative prompt with zero API
# calls (same discipline as every other precheck above).
# --------------------------------------------------------------------------- #
def test_build_a2v_chain_payload_nag_enabled_appends_six_keys_in_order():
    from gradio_ui.handlers import build_a2v_chain_payload

    payload = build_a2v_chain_payload(
        audio_id="aud-7",
        num_frames=113,
        prompt="p",
        negative_prompt="blurry",
        width=512,
        height=512,
        crop_output=None,
        frame_rate=24.0,
        seed=1,
        nag_enabled=True,
        nag_scale=9.0,
        nag_tau=3.0,
        nag_alpha=0.4,
        neg_method="vsf",
        vsf_scale=2.0,
    )
    assert list(payload.keys())[-6:] == [
        "nag_enabled", "nag_scale", "nag_tau", "nag_alpha",
        "neg_method", "vsf_scale",
    ]
    assert payload["nag_enabled"] is True
    assert payload["nag_scale"] == 9.0
    assert payload["nag_tau"] == 3.0
    assert payload["nag_alpha"] == 0.4
    assert payload["neg_method"] == "vsf"
    assert payload["vsf_scale"] == 2.0


def test_build_a2v_chain_payload_nag_default_omits_all_six_keys():
    from gradio_ui.handlers import build_a2v_chain_payload

    payload = build_a2v_chain_payload(
        audio_id="aud-8",
        num_frames=113,
        prompt="p",
        negative_prompt="",
        width=512,
        height=512,
        crop_output=None,
        frame_rate=24.0,
        seed=1,
    )
    for key in ("nag_enabled", "nag_scale", "nag_tau", "nag_alpha",
                "neg_method", "vsf_scale"):
        assert key not in payload


def test_generate_handler_nag_enabled_adds_body_fields():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"job_id": "job-nag"})

    api = _make_client(handler)
    generate = make_generate_handler(api)
    gen = generate(
        "A calm river", "blurry", *_kf_args(),
        512, 320, False, 0, 0, 49, 24.0, -1,
        nag_enabled=True, nag_scale=9.0, nag_tau=3.0, nag_alpha=0.4,
    )
    _run_until_job_started(gen)
    assert captured["nag_enabled"] is True
    assert captured["nag_scale"] == 9.0
    assert captured["nag_tau"] == 3.0
    assert captured["nag_alpha"] == 0.4


def test_generate_handler_vsf_enabled_adds_body_fields():
    # neg_method/vsf_scale reach the request body ALONGSIDE the four
    # nag_* keys whenever nag_enabled is True, regardless of which method is
    # actually selected (the key-order contract stays simple).
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"job_id": "job-vsf"})

    api = _make_client(handler)
    generate = make_generate_handler(api)
    gen = generate(
        "A calm river", "blurry", *_kf_args(),
        512, 320, False, 0, 0, 49, 24.0, -1,
        nag_enabled=True, neg_method="vsf", vsf_scale=2.5,
    )
    _run_until_job_started(gen)
    assert captured["neg_method"] == "vsf"
    assert captured["vsf_scale"] == 2.5


def test_generate_handler_nag_default_omits_body_fields():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"job_id": "job-nonag"})

    api = _make_client(handler)
    generate = make_generate_handler(api)
    gen = generate(
        "A calm river", "", *_kf_args(),
        512, 320, False, 0, 0, 49, 24.0, -1,
    )
    _run_until_job_started(gen)
    for key in ("nag_enabled", "nag_scale", "nag_tau", "nag_alpha",
                "neg_method", "vsf_scale"):
        assert key not in captured


def test_generate_nag_enabled_empty_negative_precheck_zero_calls():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={"job_id": "x"})

    api = _make_client(handler)
    generate = make_generate_handler(api)
    out = list(generate(
        "prompt", "   ", *_kf_args(), 512, 320, False, 0, 0, 49, 24.0, -1,
        nag_enabled=True,
    ))
    assert calls["n"] == 0
    assert len(out) == 1
    assert out[0][1] == "" and out[0][2] is None


def test_chain_handler_nag_enabled_adds_body_fields_positional_order():
    # Exercises the POSITIONAL contract (chunked_upsample -> nag x4 ->
    # src_audio) that ui.py's click inputs / _chain_args rely on.
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        captured.update(json.loads(request.content))
        return httpx.Response(202, json={"job_id": "chain-nag"})

    api = _make_client(handler)
    chain = make_chain_handler(api)
    # _chain_args (this file's helper) stops at ``config``; nag_* are passed as
    # keywords directly to ``chain`` (they sit further along the signature,
    # after the S6/mode params this helper never fills in).
    gen = chain(*_chain_args(negative="blurry", clips=[
        {"enabled": True, "frames": 121},
        {"enabled": True, "frames": 121},
    ]), nag_enabled=True, nag_scale=8.0, nag_tau=4.0, nag_alpha=0.5)
    _run_chain_until_started(gen)
    assert captured["nag_enabled"] is True
    assert captured["nag_scale"] == 8.0
    assert captured["nag_tau"] == 4.0
    assert captured["nag_alpha"] == 0.5


def test_chain_handler_vsf_enabled_adds_body_fields():
    # Calls `chain` with vsf_* as KEYWORD arguments (see _chain_args's docstring
    # below), so this only checks the resulting request body's fields, not
    # positional order. The positional contract itself (chunked_upsample ->
    # nag x4 -> neg_method/vsf_scale -> src_audio) is verified against
    # ui.py's actual click-handler wiring by tests/test_gradio_v2v_a2v.py's
    # `_chain_args` helper.
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        captured.update(json.loads(request.content))
        return httpx.Response(202, json={"job_id": "chain-vsf"})

    api = _make_client(handler)
    chain = make_chain_handler(api)
    gen = chain(*_chain_args(negative="blurry", clips=[
        {"enabled": True, "frames": 121},
        {"enabled": True, "frames": 121},
    ]), nag_enabled=True, neg_method="vsf", vsf_scale=3.0)
    _run_chain_until_started(gen)
    assert captured["neg_method"] == "vsf"
    assert captured["vsf_scale"] == 3.0


def test_chain_handler_nag_default_omits_body_fields():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        captured.update(json.loads(request.content))
        return httpx.Response(202, json={"job_id": "chain-nonag"})

    api = _make_client(handler)
    chain = make_chain_handler(api)
    gen = chain(*_chain_args(clips=[
        {"enabled": True, "frames": 121},
        {"enabled": True, "frames": 121},
    ]))
    _run_chain_until_started(gen)
    for key in ("nag_enabled", "nag_scale", "nag_tau", "nag_alpha",
                "neg_method", "vsf_scale"):
        assert key not in captured


def test_chain_nag_enabled_empty_negative_precheck_zero_calls():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(202, json={"job_id": "x"})

    api = _make_client(handler)
    chain = make_chain_handler(api)
    out = list(chain(*_chain_args(negative="", clips=[
        {"enabled": True, "frames": 121},
        {"enabled": True, "frames": 121},
    ]), nag_enabled=True))
    assert calls["n"] == 0
    assert len(out) == 1
    assert out[0][1] == "" and out[0][2] is None


# --------------------------------------------------------------------------- #
# Acceleration (attention_backend): the Settings-tab selector reaches every
# request body, but ONLY when it differs from the "sdpa" default -- the sdpa
# payload must stay byte-identical to the pre-Acceleration contract the
# exact-match / key-order tests above lock in. Passed as a KEYWORD everywhere
# (it sits at the very end of each handler signature, after src_audio for the
# chain handler), so no positional helper needed changing.
# --------------------------------------------------------------------------- #
def test_accel_i18n_keys_present_in_both_languages():
    from gradio_ui.i18n import LABELS

    for key in ("accel_section_title", "accel_note", "accel_lbl_fused_gguf",
                "accel_lbl_attention", "accel_info_attention",
                "accel_lbl_vae", "accel_info_unimplemented"):
        for lang in ("en", "ja"):
            assert key in LABELS[lang], f"missing {lang} label for {key}"
            assert LABELS[lang][key].strip()
    # The sage caveat must actually say the output changes + name the speedup.
    assert "1.2-1.6x" in LABELS["en"]["accel_info_attention"]
    assert "1.2〜1.6倍" in LABELS["ja"]["accel_info_attention"]


def test_build_a2v_chain_payload_sage_appends_attention_backend_last():
    from gradio_ui.handlers import build_a2v_chain_payload

    payload = build_a2v_chain_payload(
        audio_id="aud-accel-1",
        num_frames=113,
        prompt="p",
        negative_prompt="",
        width=512,
        height=512,
        crop_output=None,
        frame_rate=24.0,
        seed=1,
        attention_backend="sage",
    )
    assert payload["attention_backend"] == "sage"
    assert list(payload.keys())[-1] == "attention_backend"


def test_build_a2v_chain_payload_sage_sits_after_the_nag_block():
    from gradio_ui.handlers import build_a2v_chain_payload

    payload = build_a2v_chain_payload(
        audio_id="aud-accel-2",
        num_frames=113,
        prompt="p",
        negative_prompt="blurry",
        width=512,
        height=512,
        crop_output=None,
        frame_rate=24.0,
        seed=1,
        nag_enabled=True,
        neg_method="vsf",
        vsf_scale=2.0,
        attention_backend="sage",
    )
    assert list(payload.keys())[-7:] == [
        "nag_enabled", "nag_scale", "nag_tau", "nag_alpha",
        "neg_method", "vsf_scale", "attention_backend",
    ]


def test_build_a2v_chain_payload_default_omits_attention_backend():
    from gradio_ui.handlers import build_a2v_chain_payload

    payload = build_a2v_chain_payload(
        audio_id="aud-accel-3",
        num_frames=113,
        prompt="p",
        negative_prompt="",
        width=512,
        height=512,
        crop_output=None,
        frame_rate=24.0,
        seed=1,
    )
    assert "attention_backend" not in payload
    # Explicitly passing the default must be indistinguishable from omitting it.
    assert payload == build_a2v_chain_payload(
        audio_id="aud-accel-3",
        num_frames=113,
        prompt="p",
        negative_prompt="",
        width=512,
        height=512,
        crop_output=None,
        frame_rate=24.0,
        seed=1,
        attention_backend="sdpa",
    )


def test_generate_handler_sage_adds_attention_backend():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"job_id": "job-sage"})

    api = _make_client(handler)
    generate = make_generate_handler(api)
    gen = generate(
        "A calm river", "", *_kf_args(),
        512, 320, False, 0, 0, 49, 24.0, -1,
        attention_backend="sage",
    )
    _run_until_job_started(gen)
    assert captured["attention_backend"] == "sage"
    # Appended last, after the (absent here) NAG block.
    assert list(captured.keys())[-1] == "attention_backend"


def test_generate_handler_default_omits_attention_backend():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"job_id": "job-sdpa"})

    api = _make_client(handler)
    generate = make_generate_handler(api)
    gen = generate(
        "A calm river", "", *_kf_args(),
        512, 320, False, 0, 0, 49, 24.0, -1,
    )
    _run_until_job_started(gen)
    assert "attention_backend" not in captured


def test_generate_handler_a2v_forwards_attention_backend(tmp_path):
    # The A2V branch builds its body through build_a2v_chain_payload -- the
    # selector must survive that hop too.
    aud = tmp_path / "voice.wav"
    aud.write_bytes(b"RIFF....WAVEfmt ")
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).endswith("/upload/audio"):
            return httpx.Response(200, json={"audio_id": "aud-accel"})
        import json
        captured.update(json.loads(request.content))
        return httpx.Response(202, json={"job_id": "chain-a2v-sage"})

    api = _make_client(handler)
    generate = make_generate_handler(api)
    gen = generate(
        "a singer", "", *_kf_args(),
        512, 320, False, 0, 0, 49, 24.0, -1,
        src_audio=str(aud), attention_backend="sage",
    )
    # The A2V branch yields an "uploading audio" line BEFORE the POST, so this
    # drives the generator to the job-started yield instead of using
    # _run_until_job_started (which stops at the first yield).
    for out in gen:
        if out[1]:
            gen.close()
            break
    assert captured["attention_backend"] == "sage"


def test_chain_handler_sage_adds_attention_backend():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        captured.update(json.loads(request.content))
        return httpx.Response(202, json={"job_id": "chain-sage"})

    api = _make_client(handler)
    chain = make_chain_handler(api)
    gen = chain(*_chain_args(clips=[
        {"enabled": True, "frames": 121},
        {"enabled": True, "frames": 121},
    ]), attention_backend="sage")
    _run_chain_until_started(gen)
    assert captured["attention_backend"] == "sage"
    assert list(captured.keys())[-1] == "attention_backend"


def test_chain_handler_default_omits_attention_backend():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        captured.update(json.loads(request.content))
        return httpx.Response(202, json={"job_id": "chain-sdpa"})

    api = _make_client(handler)
    chain = make_chain_handler(api)
    gen = chain(*_chain_args(clips=[
        {"enabled": True, "frames": 121},
        {"enabled": True, "frames": 121},
    ]))
    _run_chain_until_started(gen)
    assert "attention_backend" not in captured


# --------------------------------------------------------------------------- #
# Block-swap prefetch (block_swap_prefetch): the Settings-tab checkbox added
# alongside attention_backend. S4 (2026-08-01) flipped BOTH the API's own
# pydantic default (api/models.py) AND this module's mirror constant to True
# -- the real-device gate (bit-exact output + VRAM headroom, G1-G7) passed and
# the owner confirmed "gate green -> default on". The payload discipline
# flipped WITH it: a request now reaches the body ONLY when it differs from
# gradio_ui.handlers.BLOCK_SWAP_PREFETCH_DEFAULT (no longer "only when True"
# -- that pattern would have silently gone inert on this flip, since an
# explicit "off" would then never reach the wire and the server's new True
# default would turn it back on behind the caller's back). Appended right
# after attention_backend so the default payload stays byte-identical to the
# pre-prefetch contract either way.
# --------------------------------------------------------------------------- #
def test_block_swap_prefetch_i18n_keys_present_in_both_languages():
    from gradio_ui.i18n import LABELS

    for key in ("accel_lbl_prefetch", "accel_info_prefetch"):
        for lang in ("en", "ja"):
            assert key in LABELS[lang], f"missing {lang} label for {key}"
            assert LABELS[lang][key].strip()


def test_block_swap_prefetch_default_constant_is_true():
    from gradio_ui.handlers import BLOCK_SWAP_PREFETCH_DEFAULT

    assert BLOCK_SWAP_PREFETCH_DEFAULT is True


def test_build_a2v_chain_payload_prefetch_off_appends_key_last():
    from gradio_ui.handlers import build_a2v_chain_payload

    payload = build_a2v_chain_payload(
        audio_id="aud-prefetch-1",
        num_frames=113,
        prompt="p",
        negative_prompt="",
        width=512,
        height=512,
        crop_output=None,
        frame_rate=24.0,
        seed=1,
        block_swap_prefetch=False,
    )
    assert payload["block_swap_prefetch"] is False
    assert list(payload.keys())[-1] == "block_swap_prefetch"


def test_build_a2v_chain_payload_prefetch_sits_after_attention_backend():
    from gradio_ui.handlers import build_a2v_chain_payload

    payload = build_a2v_chain_payload(
        audio_id="aud-prefetch-2",
        num_frames=113,
        prompt="p",
        negative_prompt="blurry",
        width=512,
        height=512,
        crop_output=None,
        frame_rate=24.0,
        seed=1,
        nag_enabled=True,
        neg_method="vsf",
        vsf_scale=2.0,
        attention_backend="sage",
        block_swap_prefetch=False,
    )
    assert list(payload.keys())[-8:] == [
        "nag_enabled", "nag_scale", "nag_tau", "nag_alpha",
        "neg_method", "vsf_scale", "attention_backend", "block_swap_prefetch",
    ]


def test_build_a2v_chain_payload_default_omits_block_swap_prefetch():
    from gradio_ui.handlers import build_a2v_chain_payload

    payload = build_a2v_chain_payload(
        audio_id="aud-prefetch-3",
        num_frames=113,
        prompt="p",
        negative_prompt="",
        width=512,
        height=512,
        crop_output=None,
        frame_rate=24.0,
        seed=1,
    )
    assert "block_swap_prefetch" not in payload
    # Explicitly passing the default (True, post-S4) must be indistinguishable
    # from omitting it.
    assert payload == build_a2v_chain_payload(
        audio_id="aud-prefetch-3",
        num_frames=113,
        prompt="p",
        negative_prompt="",
        width=512,
        height=512,
        crop_output=None,
        frame_rate=24.0,
        seed=1,
        block_swap_prefetch=True,
    )


def test_build_a2v_chain_payload_explicit_off_sends_false():
    from gradio_ui.handlers import build_a2v_chain_payload

    payload = build_a2v_chain_payload(
        audio_id="aud-prefetch-4",
        num_frames=113,
        prompt="p",
        negative_prompt="",
        width=512,
        height=512,
        crop_output=None,
        frame_rate=24.0,
        seed=1,
        block_swap_prefetch=False,
    )
    assert payload["block_swap_prefetch"] is False


def test_generate_handler_prefetch_off_adds_block_swap_prefetch():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"job_id": "job-prefetch"})

    api = _make_client(handler)
    generate = make_generate_handler(api)
    gen = generate(
        "A calm river", "", *_kf_args(),
        512, 320, False, 0, 0, 49, 24.0, -1,
        block_swap_prefetch=False,
    )
    _run_until_job_started(gen)
    assert captured["block_swap_prefetch"] is False
    # Appended last, after the (absent here) attention_backend key.
    assert list(captured.keys())[-1] == "block_swap_prefetch"


def test_generate_handler_default_omits_block_swap_prefetch():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"job_id": "job-noprefetch"})

    api = _make_client(handler)
    generate = make_generate_handler(api)
    gen = generate(
        "A calm river", "", *_kf_args(),
        512, 320, False, 0, 0, 49, 24.0, -1,
    )
    _run_until_job_started(gen)
    assert "block_swap_prefetch" not in captured


def test_generate_handler_a2v_forwards_block_swap_prefetch(tmp_path):
    # The A2V branch builds its body through build_a2v_chain_payload -- the
    # checkbox must survive that hop too.
    aud = tmp_path / "voice.wav"
    aud.write_bytes(b"RIFF....WAVEfmt ")
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).endswith("/upload/audio"):
            return httpx.Response(200, json={"audio_id": "aud-prefetch"})
        import json
        captured.update(json.loads(request.content))
        return httpx.Response(202, json={"job_id": "chain-a2v-prefetch"})

    api = _make_client(handler)
    generate = make_generate_handler(api)
    gen = generate(
        "a singer", "", *_kf_args(),
        512, 320, False, 0, 0, 49, 24.0, -1,
        src_audio=str(aud), block_swap_prefetch=False,
    )
    for out in gen:
        if out[1]:
            gen.close()
            break
    assert captured["block_swap_prefetch"] is False


def test_chain_handler_prefetch_off_adds_block_swap_prefetch():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        captured.update(json.loads(request.content))
        return httpx.Response(202, json={"job_id": "chain-prefetch"})

    api = _make_client(handler)
    chain = make_chain_handler(api)
    gen = chain(*_chain_args(clips=[
        {"enabled": True, "frames": 121},
        {"enabled": True, "frames": 121},
    ]), block_swap_prefetch=False)
    _run_chain_until_started(gen)
    assert captured["block_swap_prefetch"] is False
    assert list(captured.keys())[-1] == "block_swap_prefetch"


def test_chain_handler_default_omits_block_swap_prefetch():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        captured.update(json.loads(request.content))
        return httpx.Response(202, json={"job_id": "chain-noprefetch"})

    api = _make_client(handler)
    chain = make_chain_handler(api)
    gen = chain(*_chain_args(clips=[
        {"enabled": True, "frames": 121},
        {"enabled": True, "frames": 121},
    ]))
    _run_chain_until_started(gen)
    assert "block_swap_prefetch" not in captured
