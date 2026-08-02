"""Contract tests for mcp_server/tools/uploads.py and tools/generate.py (W2).

Same two-double pattern as test_mcp_tools_system.py:
  * ``_client_for_app`` -- the real mock-backend FastAPI app via
    ``httpx.ASGITransport`` (the ``mcp_app`` fixture), for happy-path /
    server-round-trip behavior (upload succeeds, a T2V job actually
    completes, a malformed request really gets a 422 from the backend).
  * ``httpx.MockTransport`` with a hand-written handler, for asserting
    exactly which HTTP calls happen (or don't) and for capturing the exact
    JSON body submit_generate sends (the payload contract: no None/empty
    keys, no hidden fields).
"""

from __future__ import annotations

import functools
import json
import struct
import wave
from pathlib import Path

import anyio
import httpx
import pytest
from mcp.server.fastmcp.exceptions import ToolError

from mcp_server.client import BackendClient, set_client
from mcp_server.params import ChainClipArg, ConditioningImageArg, LoraArg
from mcp_server.server import build_server
from mcp_server.settings import Settings
from mcp_server.tools import generate, jobs, uploads


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


# --------------------------------------------------------------- uploads ---


def test_upload_image_happy_path_via_mcp_app(mcp_app, tmp_path, png_bytes):
    app, output_dir = mcp_app
    set_client(_client_for_app(app, output_dir))

    img_path = tmp_path / "photo.png"
    img_path.write_bytes(png_bytes)

    result = anyio.run(uploads.upload_image, str(img_path))

    assert result["image_id"]
    assert result["stored_path"].endswith("/input.png")
    assert result["width"] == 64
    assert result["height"] == 48


def test_upload_image_file_not_found():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"no HTTP call expected, got {request.method} {request.url.path}")

    set_client(_client_for_handler(handler))

    with pytest.raises(ToolError) as exc_info:
        anyio.run(uploads.upload_image, "S:/does/not/exist.png")

    assert "FILE_NOT_FOUND" in str(exc_info.value)


def test_upload_image_wrong_extension_precheck_makes_no_http_call(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"no HTTP call expected, got {request.method} {request.url.path}")

    set_client(_client_for_handler(handler))

    bad = tmp_path / "notes.txt"
    bad.write_text("hello")

    with pytest.raises(ToolError) as exc_info:
        anyio.run(uploads.upload_image, str(bad))

    assert "UPLOAD_INVALID_TYPE" in str(exc_info.value)


def test_upload_video_wrong_extension_precheck_makes_no_http_call(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"no HTTP call expected, got {request.method} {request.url.path}")

    set_client(_client_for_handler(handler))

    bad = tmp_path / "clip.txt"
    bad.write_text("hello")

    with pytest.raises(ToolError) as exc_info:
        anyio.run(uploads.upload_video, str(bad))

    assert "UPLOAD_INVALID_TYPE" in str(exc_info.value)


def test_upload_audio_wrong_extension_precheck_makes_no_http_call(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"no HTTP call expected, got {request.method} {request.url.path}")

    set_client(_client_for_handler(handler))

    bad = tmp_path / "song.txt"
    bad.write_text("hello")

    with pytest.raises(ToolError) as exc_info:
        anyio.run(uploads.upload_audio, str(bad))

    assert "UPLOAD_INVALID_TYPE" in str(exc_info.value)


# ------------------------------------------------------------ submit_generate


def test_submit_generate_minimal_t2v_completes(mcp_app):
    app, output_dir = mcp_app
    set_client(_client_for_app(app, output_dir))

    result = anyio.run(generate.submit_generate, "A red ball rolling on a white floor")

    assert result["job_id"]
    # POST /generate's response body is built at job-creation time (before the
    # background task runs), so it always reports "queued" here -- mirrors
    # GenerateResponse's contract (see api/generate.py). The mock backend still
    # runs the job to completion synchronously inside that same request/response
    # cycle (BackgroundTasks execute before the ASGI call returns), so a
    # follow-up job_status immediately sees "completed".
    assert result["status"] == "queued"
    assert result["next"] == "wait_for_job(job_id) を呼ぶ"

    status = anyio.run(jobs.job_status, result["job_id"])
    assert status["status"] == "completed"
    assert status["result"]["seed_used"] is not None


def test_submit_generate_payload_contract_defaults_only():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            202, json={"job_id": "j1", "status": "queued", "created_at": "2026-01-01T00:00:00Z"}
        )

    set_client(_client_for_handler(handler))

    anyio.run(generate.submit_generate, "a prompt")

    body = captured["body"]
    assert set(body.keys()) == {"prompt", "width", "height", "num_frames", "frame_rate", "seed"}
    assert "negative_prompt" not in body
    assert "nag_enabled" not in body
    assert "crop_output" not in body
    assert "conditioning_images" not in body
    assert "loras" not in body
    assert "reference_video_id" not in body


def test_submit_generate_with_nag_and_conditioning_images_includes_them():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            202, json={"job_id": "j1", "status": "queued", "created_at": "2026-01-01T00:00:00Z"}
        )

    set_client(_client_for_handler(handler))

    anyio.run(
        generate.submit_generate,
        "a prompt",
        "blurry",  # negative_prompt
        True,  # nag_enabled
        11.0, 2.5, 0.25,
        512, 320,
        None, None,
        49, 24.0, -1,
        [ConditioningImageArg(image_id="img-1", frame_idx=0, strength=0.9)],
    )

    body = captured["body"]
    assert body["negative_prompt"] == "blurry"
    assert body["nag_enabled"] is True
    assert body["nag_scale"] == 11.0
    assert body["conditioning_images"] == [
        {"image_id": "img-1", "frame_idx": 0, "strength": 0.9}
    ]


def test_submit_generate_with_vsf_fields_includes_them():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            202, json={"job_id": "j1", "status": "queued", "created_at": "2026-01-01T00:00:00Z"}
        )

    set_client(_client_for_handler(handler))

    anyio.run(
        functools.partial(
            generate.submit_generate,
            "a prompt",
            negative_prompt="blurry",
            nag_enabled=True,
            neg_method="vsf",
            vsf_scale=3.0,
        )
    )

    body = captured["body"]
    assert body["neg_method"] == "vsf"
    assert body["vsf_scale"] == 3.0


def test_submit_generate_nag_enabled_defaults_vsf_fields_to_nag():
    # nag_enabled=True with neg_method/vsf_scale omitted -> the
    # neg_method="nag"/vsf_scale=1.5 defaults still ride along
    # in the payload (mirrors the always-sent-inside-the-if-block contract).
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            202, json={"job_id": "j1", "status": "queued", "created_at": "2026-01-01T00:00:00Z"}
        )

    set_client(_client_for_handler(handler))

    anyio.run(
        functools.partial(
            generate.submit_generate,
            "a prompt",
            negative_prompt="blurry",
            nag_enabled=True,
        )
    )

    body = captured["body"]
    assert body["neg_method"] == "nag"
    assert body["vsf_scale"] == 1.5


def test_submit_generate_loras_without_audio_strength_omits_key():
    # WP4: LoraArg.model_dump(exclude_none=True) -- omitting audio_strength on
    # the client side must not put an "audio_strength" key in the loras
    # payload entries (backward-compat: audio side follows strength).
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            202, json={"job_id": "j1", "status": "queued", "created_at": "2026-01-01T00:00:00Z"}
        )

    set_client(_client_for_handler(handler))

    anyio.run(
        functools.partial(
            generate.submit_generate,
            "a prompt",
            loras=[LoraArg(name="style-x", strength=0.8)],
        )
    )

    body = captured["body"]
    assert body["loras"] == [{"name": "style-x", "strength": 0.8}]
    assert "audio_strength" not in body["loras"][0]


def test_submit_generate_loras_with_audio_strength_zero_included():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            202, json={"job_id": "j1", "status": "queued", "created_at": "2026-01-01T00:00:00Z"}
        )

    set_client(_client_for_handler(handler))

    anyio.run(
        functools.partial(
            generate.submit_generate,
            "a prompt",
            loras=[LoraArg(name="style-x", strength=0.8, audio_strength=0.0)],
        )
    )

    body = captured["body"]
    assert body["loras"] == [{"name": "style-x", "strength": 0.8, "audio_strength": 0.0}]


def test_submit_generate_sage_attention_backend_included_in_body():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            202, json={"job_id": "j1", "status": "queued", "created_at": "2026-01-01T00:00:00Z"}
        )

    set_client(_client_for_handler(handler))

    anyio.run(
        functools.partial(
            generate.submit_generate,
            "a prompt",
            attention_backend="sage",
        )
    )

    body = captured["body"]
    assert body["attention_backend"] == "sage"


def test_submit_generate_payload_contract_defaults_only_key_set_unchanged_with_default_attention_backend():
    # Guards D2's byte-identical contract: leaving attention_backend at its
    # "sdpa" default must not add a key to the payload (trip-wire alongside
    # the existing exact-match assertion at line ~165).
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            202, json={"job_id": "j1", "status": "queued", "created_at": "2026-01-01T00:00:00Z"}
        )

    set_client(_client_for_handler(handler))

    anyio.run(
        functools.partial(
            generate.submit_generate,
            "a prompt",
            attention_backend="sdpa",
        )
    )

    body = captured["body"]
    assert set(body.keys()) == {"prompt", "width", "height", "num_frames", "frame_rate", "seed"}
    assert "attention_backend" not in body


def test_submit_generate_block_swap_prefetch_off_included_in_body():
    # S4 (2026-08-01): BLOCK_SWAP_PREFETCH_DEFAULT flipped to True (real-device
    # gate G1-G7 passed), so it is now the OFF (False) call that diverges from
    # the default and reaches the wire — mirrors gradio_ui/handlers.py's
    # discipline (backend §44 / WORKORDER §8.8).
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            202, json={"job_id": "j1", "status": "queued", "created_at": "2026-01-01T00:00:00Z"}
        )

    set_client(_client_for_handler(handler))

    anyio.run(
        functools.partial(
            generate.submit_generate,
            "a prompt",
            block_swap_prefetch=False,
        )
    )

    body = captured["body"]
    assert body["block_swap_prefetch"] is False


def test_submit_generate_payload_contract_defaults_only_key_set_unchanged_with_default_block_swap_prefetch():
    # Explicitly passing the default (True, post-S4) must be indistinguishable
    # from omitting it.
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            202, json={"job_id": "j1", "status": "queued", "created_at": "2026-01-01T00:00:00Z"}
        )

    set_client(_client_for_handler(handler))

    anyio.run(
        functools.partial(
            generate.submit_generate,
            "a prompt",
            block_swap_prefetch=True,
        )
    )

    body = captured["body"]
    assert set(body.keys()) == {"prompt", "width", "height", "num_frames", "frame_rate", "seed"}
    assert "block_swap_prefetch" not in body


def test_submit_generate_crop_single_sided_raises_before_any_http_call():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"no HTTP call expected, got {request.method} {request.url.path}")

    set_client(_client_for_handler(handler))

    with pytest.raises(ToolError) as exc_info:
        anyio.run(generate.submit_generate, "p", "", False, 11.0, 2.5, 0.25, 512, 320, 100, None)

    assert "CROP_SIZE_INCOMPLETE" in str(exc_info.value)


def test_submit_generate_nag_enabled_and_empty_negative_prompt_surfaces_validation_error(mcp_app):
    app, output_dir = mcp_app
    set_client(_client_for_app(app, output_dir))

    with pytest.raises(ToolError) as exc_info:
        anyio.run(generate.submit_generate, "p", "", True)  # nag_enabled=True, negative_prompt=""

    assert "VALIDATION_ERROR" in str(exc_info.value)


def test_submit_generate_width_not_multiple_of_64_surfaces_validation_error(mcp_app):
    app, output_dir = mcp_app
    set_client(_client_for_app(app, output_dir))

    with pytest.raises(ToolError) as exc_info:
        anyio.run(
            generate.submit_generate,
            "p", "", False, 11.0, 2.5, 0.25,
            500,  # width, not a multiple of 64
        )

    assert "VALIDATION_ERROR" in str(exc_info.value)


def test_submit_generate_input_schema_has_no_hidden_fields():
    async def _run():
        mcp = build_server()
        return await mcp.list_tools()

    tools = anyio.run(_run)
    tool = next(t for t in tools if t.name == "submit_generate")
    props = set(tool.inputSchema.get("properties", {}))
    for hidden in ("pipeline", "num_inference_steps", "guidance_scale", "crf"):
        assert hidden not in props, f"hidden field {hidden!r} leaked into submit_generate schema"


# --------------------------------------------------------------- submit_chain


def _make_wav(path: Path, *, seconds: float = 3.0, sr: int = 16000, channels: int = 1) -> Path:
    """Same tiny silent-PCM wav fixture as tests/test_a2v_chain.py."""
    n = int(round(seconds * sr))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(struct.pack("<%dh" % (n * channels), *([0] * (n * channels))))
    return path


def test_submit_chain_source_xor_violation_raises_before_any_http_call():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"no HTTP call expected, got {request.method} {request.url.path}")

    set_client(_client_for_handler(handler))

    with pytest.raises(ToolError) as exc_info:
        anyio.run(
            generate.submit_chain,
            "p",
            [ChainClipArg(num_frames=49)],
            "", False, 11.0, 2.5, 0.25,
            512, 320, None, None,
            24.0, -1, 3, 0.5,
            "video-1",  # source_video_id
            73,
            "audio-1",  # source_audio_id
        )

    assert "SOURCE_XOR_VIOLATION" in str(exc_info.value)


def test_submit_chain_crop_single_sided_raises_before_any_http_call():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"no HTTP call expected, got {request.method} {request.url.path}")

    set_client(_client_for_handler(handler))

    with pytest.raises(ToolError) as exc_info:
        anyio.run(
            generate.submit_chain,
            "p",
            [ChainClipArg(num_frames=49), ChainClipArg(num_frames=49)],
            "", False, 11.0, 2.5, 0.25,
            512, 320, 100, None,  # crop_width set, crop_height omitted
        )

    assert "CROP_SIZE_INCOMPLETE" in str(exc_info.value)


def test_submit_chain_payload_contract_defaults_only_and_chunked_upsample_always_sent():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            202,
            json={"job_id": "j1", "status": "queued", "created_at": "2026-01-01T00:00:00Z", "num_clips": 2},
        )

    set_client(_client_for_handler(handler))

    anyio.run(
        generate.submit_chain,
        "a prompt",
        [ChainClipArg(num_frames=25), ChainClipArg(num_frames=25)],
    )

    body = captured["body"]
    assert set(body.keys()) == {
        "prompt", "width", "height", "frame_rate", "seed",
        "overlap_frames", "overlap_strength", "clips", "chunked_upsample",
    }
    assert body["chunked_upsample"] is True  # panel default, always sent explicitly
    assert body["clips"] == [{"num_frames": 25}, {"num_frames": 25}]
    assert "negative_prompt" not in body
    assert "nag_enabled" not in body
    assert "crop_output" not in body
    assert "source_video" not in body
    assert "source_audio" not in body
    assert "loras" not in body
    assert "reference_video_id" not in body


def test_submit_chain_loras_without_audio_strength_omits_key():
    # WP4: same exclude_none=True contract as submit_generate.
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            202,
            json={"job_id": "j1", "status": "queued", "created_at": "2026-01-01T00:00:00Z", "num_clips": 2},
        )

    set_client(_client_for_handler(handler))

    anyio.run(
        functools.partial(
            generate.submit_chain,
            "a prompt",
            [ChainClipArg(num_frames=25), ChainClipArg(num_frames=25)],
            loras=[LoraArg(name="style-x", strength=0.8)],
        )
    )

    body = captured["body"]
    assert body["loras"] == [{"name": "style-x", "strength": 0.8}]
    assert "audio_strength" not in body["loras"][0]


def test_submit_chain_loras_with_audio_strength_zero_included():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            202,
            json={"job_id": "j1", "status": "queued", "created_at": "2026-01-01T00:00:00Z", "num_clips": 2},
        )

    set_client(_client_for_handler(handler))

    anyio.run(
        functools.partial(
            generate.submit_chain,
            "a prompt",
            [ChainClipArg(num_frames=25), ChainClipArg(num_frames=25)],
            loras=[LoraArg(name="style-x", strength=0.8, audio_strength=0.0)],
        )
    )

    body = captured["body"]
    assert body["loras"] == [{"name": "style-x", "strength": 0.8, "audio_strength": 0.0}]


def test_submit_chain_sage_attention_backend_included_in_body():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            202,
            json={"job_id": "j1", "status": "queued", "created_at": "2026-01-01T00:00:00Z", "num_clips": 2},
        )

    set_client(_client_for_handler(handler))

    anyio.run(
        functools.partial(
            generate.submit_chain,
            "a prompt",
            [ChainClipArg(num_frames=25), ChainClipArg(num_frames=25)],
            attention_backend="sage",
        )
    )

    body = captured["body"]
    assert body["attention_backend"] == "sage"


def test_submit_chain_payload_contract_key_set_unchanged_with_default_attention_backend():
    # Guards D2's byte-identical contract: leaving attention_backend at its
    # "sdpa" default must not add a key to the payload (trip-wire alongside
    # the existing exact-match assertion at line ~382).
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            202,
            json={"job_id": "j1", "status": "queued", "created_at": "2026-01-01T00:00:00Z", "num_clips": 2},
        )

    set_client(_client_for_handler(handler))

    anyio.run(
        functools.partial(
            generate.submit_chain,
            "a prompt",
            [ChainClipArg(num_frames=25), ChainClipArg(num_frames=25)],
            attention_backend="sdpa",
        )
    )

    body = captured["body"]
    assert set(body.keys()) == {
        "prompt", "width", "height", "frame_rate", "seed",
        "overlap_frames", "overlap_strength", "clips", "chunked_upsample",
    }
    assert "attention_backend" not in body


def test_submit_chain_block_swap_prefetch_off_included_in_body():
    # S4 (2026-08-01): BLOCK_SWAP_PREFETCH_DEFAULT flipped to True, so it is
    # now the OFF (False) call that diverges from the default and reaches the
    # wire — mirrors test_submit_chain_sage_attention_backend_included_in_body
    # (backend §44 / WORKORDER §8.8).
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            202,
            json={"job_id": "j1", "status": "queued", "created_at": "2026-01-01T00:00:00Z", "num_clips": 2},
        )

    set_client(_client_for_handler(handler))

    anyio.run(
        functools.partial(
            generate.submit_chain,
            "a prompt",
            [ChainClipArg(num_frames=25), ChainClipArg(num_frames=25)],
            block_swap_prefetch=False,
        )
    )

    body = captured["body"]
    assert body["block_swap_prefetch"] is False


def test_submit_chain_payload_contract_key_set_unchanged_with_default_block_swap_prefetch():
    # Explicitly passing the default (True, post-S4) must be indistinguishable
    # from omitting it.
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            202,
            json={"job_id": "j1", "status": "queued", "created_at": "2026-01-01T00:00:00Z", "num_clips": 2},
        )

    set_client(_client_for_handler(handler))

    anyio.run(
        functools.partial(
            generate.submit_chain,
            "a prompt",
            [ChainClipArg(num_frames=25), ChainClipArg(num_frames=25)],
            block_swap_prefetch=True,
        )
    )

    body = captured["body"]
    assert set(body.keys()) == {
        "prompt", "width", "height", "frame_rate", "seed",
        "overlap_frames", "overlap_strength", "clips", "chunked_upsample",
    }
    assert "block_swap_prefetch" not in body


def test_submit_chain_chunked_upsample_false_is_sent_explicitly():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            202,
            json={"job_id": "j1", "status": "queued", "created_at": "2026-01-01T00:00:00Z", "num_clips": 2},
        )

    set_client(_client_for_handler(handler))

    anyio.run(
        generate.submit_chain,
        "a prompt",
        [ChainClipArg(num_frames=25), ChainClipArg(num_frames=25)],
        "", False, 11.0, 2.5, 0.25,
        512, 320, None, None,
        24.0, -1, 3, 0.5,
        None, 73, None, None, None, None, None,
        False,  # chunked_upsample
    )

    assert captured["body"]["chunked_upsample"] is False


def test_submit_chain_nag_and_clip_prompt_and_conditioning_images_included():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            202,
            json={"job_id": "j1", "status": "queued", "created_at": "2026-01-01T00:00:00Z", "num_clips": 2},
        )

    set_client(_client_for_handler(handler))

    anyio.run(
        generate.submit_chain,
        "base prompt",
        [
            ChainClipArg(
                prompt="clip0 override",
                num_frames=25,
                conditioning_images=[ConditioningImageArg(image_id="img-1", frame_idx=0, strength=0.9)],
            ),
            ChainClipArg(num_frames=25),
        ],
        "blurry",  # negative_prompt
        True,  # nag_enabled
    )

    body = captured["body"]
    assert body["negative_prompt"] == "blurry"
    assert body["nag_enabled"] is True
    assert body["nag_scale"] == 11.0
    assert body["clips"][0]["prompt"] == "clip0 override"
    assert body["clips"][0]["conditioning_images"] == [
        {"image_id": "img-1", "frame_idx": 0, "strength": 0.9}
    ]
    assert "prompt" not in body["clips"][1]
    assert "conditioning_images" not in body["clips"][1]


def test_submit_chain_with_vsf_fields_includes_them():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            202,
            json={"job_id": "j1", "status": "queued", "created_at": "2026-01-01T00:00:00Z", "num_clips": 2},
        )

    set_client(_client_for_handler(handler))

    anyio.run(
        functools.partial(
            generate.submit_chain,
            "a prompt",
            [ChainClipArg(num_frames=25), ChainClipArg(num_frames=25)],
            negative_prompt="blurry",
            nag_enabled=True,
            neg_method="vsf",
            vsf_scale=3.0,
        )
    )

    body = captured["body"]
    assert body["neg_method"] == "vsf"
    assert body["vsf_scale"] == 3.0


def test_submit_chain_nag_enabled_defaults_vsf_fields_to_nag():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            202,
            json={"job_id": "j1", "status": "queued", "created_at": "2026-01-01T00:00:00Z", "num_clips": 2},
        )

    set_client(_client_for_handler(handler))

    anyio.run(
        functools.partial(
            generate.submit_chain,
            "a prompt",
            [ChainClipArg(num_frames=25), ChainClipArg(num_frames=25)],
            negative_prompt="blurry",
            nag_enabled=True,
        )
    )

    body = captured["body"]
    assert body["neg_method"] == "nag"
    assert body["vsf_scale"] == 1.5


def test_submit_chain_input_schema_has_no_hidden_fields():
    async def _run():
        mcp = build_server()
        return await mcp.list_tools()

    tools = anyio.run(_run)
    tool = next(t for t in tools if t.name == "submit_chain")
    props = set(tool.inputSchema.get("properties", {}))
    for hidden in ("pipeline", "num_inference_steps", "guidance_scale", "crf"):
        assert hidden not in props, f"hidden field {hidden!r} leaked into submit_chain schema"


def test_submit_chain_no_source_two_clip_chain_completes_via_mcp_app(mcp_app):
    app, output_dir = mcp_app
    set_client(_client_for_app(app, output_dir))

    result = anyio.run(
        generate.submit_chain,
        "a bustling town square at dusk",
        [ChainClipArg(num_frames=25), ChainClipArg(num_frames=25)],
    )

    assert result["job_id"]
    assert result["num_clips"] == 2
    assert result["next"] == "wait_for_job(job_id) を呼ぶ"

    status = anyio.run(jobs.job_status, result["job_id"])
    assert status["status"] == "completed"


def test_submit_chain_a2v_single_clip_happy_path_via_mcp_app(mcp_app, tmp_path):
    app, output_dir = mcp_app
    set_client(_client_for_app(app, output_dir))

    wav_path = _make_wav(tmp_path / "voice.wav", seconds=3.0, sr=16000, channels=1)
    upload = anyio.run(uploads.upload_audio, str(wav_path))
    audio_id = upload["audio_id"]

    result = anyio.run(
        generate.submit_chain,
        "a bustling town square at dusk, a woman speaking",
        [ChainClipArg(num_frames=49)],
        "", False, 11.0, 2.5, 0.25,
        384, 256, None, None,
        24.0, -1, 3, 0.5,
        None, 73,
        audio_id,  # source_audio_id
    )

    assert result["num_clips"] == 1
    job_id = result["job_id"]

    status = anyio.run(jobs.job_status, job_id)
    assert status["status"] == "completed", status
