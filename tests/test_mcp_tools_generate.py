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
    # Docs/PENDING_TASKS_CLOSED.md §3-122 (closed 2026-09-01): the outpaint nest
    # rides only on a non-zero pad, so a defaults-only call's body is
    # byte-identical to what it was before the six outpaint_* arguments existed.
    assert "outpaint" not in body


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


def test_submit_generate_keep_resident_on_included_in_body():
    # keep_resident's default is OFF (§48), so — unlike block_swap_prefetch
    # right above — it is the ON call that diverges from the default and
    # reaches the wire. Same rule, opposite direction.
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
            keep_resident=True,
        )
    )

    body = captured["body"]
    assert body["keep_resident"] is True

    # ...and the default (False) is indistinguishable from omitting it.
    anyio.run(
        functools.partial(
            generate.submit_generate,
            "a prompt",
            keep_resident=False,
        )
    )
    body = captured["body"]
    assert set(body.keys()) == {"prompt", "width", "height", "num_frames", "frame_rate", "seed"}


def test_submit_generate_fused_dequant_off_included_in_body():
    # fused_gguf_dequant_kernel (§1-11): default ON since 2026-08-04 (§51), so
    # the OFF call is the one that reaches the wire. Sent LAST of the
    # Acceleration keys, so the default body's key set stays frozen.
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
            fused_gguf_dequant_kernel=False,
        )
    )

    body = captured["body"]
    assert body["fused_gguf_dequant_kernel"] is False

    # ...and the default (True) is indistinguishable from omitting it.
    anyio.run(
        functools.partial(
            generate.submit_generate,
            "a prompt",
            fused_gguf_dequant_kernel=True,
        )
    )
    body = captured["body"]
    assert set(body.keys()) == {"prompt", "width", "height", "num_frames", "frame_rate", "seed"}


def test_submit_generate_vae_mode_prune_vaed_included_in_body():
    # vae_mode (PrunaVAED, Docs/PENDING_TASKS_CLOSED.md §3-66, filed as §3-50
    # at the time): exposed to MCP on 2026-08-05 once the
    # real-device gates G1-G7 passed (owner ruling 0-8; before that the tool
    # deliberately hid it because the field was a mock). Default is "default"
    # and it is PERMANENT (0-11) -- so the "prune_vaed" call is the one that
    # reaches the wire, appended after fused_gguf_dequant_kernel so the default
    # body's key set stays frozen.
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
            vae_mode="prune_vaed",
        )
    )

    body = captured["body"]
    assert body["vae_mode"] == "prune_vaed"

    # ...and the default ("default") is indistinguishable from omitting it.
    anyio.run(
        functools.partial(
            generate.submit_generate,
            "a prompt",
            vae_mode="default",
        )
    )
    body = captured["body"]
    assert set(body.keys()) == {"prompt", "width", "height", "num_frames", "frame_rate", "seed"}


def test_submit_generate_input_schema_exposes_vae_mode_enum():
    # G9 (§9 of PRUNAVAED_WORKORDER.md): the schema must carry vae_mode with
    # exactly the two API literals and a "default" default -- an agent reads
    # only the schema to learn the switch exists.
    async def _run():
        mcp = build_server()
        return await mcp.list_tools()

    tools = anyio.run(_run)
    tool = next(t for t in tools if t.name == "submit_generate")
    prop = tool.inputSchema["properties"]["vae_mode"]
    assert prop["enum"] == ["default", "prune_vaed"]
    assert prop["default"] == "default"


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


def test_submit_chain_end_source_xor_violation_raises_before_any_http_call():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"no HTTP call expected, got {request.method} {request.url.path}")

    set_client(_client_for_handler(handler))

    with pytest.raises(ToolError) as exc_info:
        anyio.run(
            functools.partial(
                generate.submit_chain,
                "p",
                [ChainClipArg(num_frames=169)],
                end_source_video_id="video-1",
                end_source_image_id="image-1",
            )
        )

    assert "END_SOURCE_XOR_VIOLATION" in str(exc_info.value)


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
    assert "end_source" not in body
    # Docs/PENDING_TASKS_CLOSED.md §3-115 (closed 2026-09-01): the retake nest
    # rides only on retake_video_id, so a defaults-only call's body is
    # byte-identical to what it was before the five retake_* arguments existed.
    assert "retake" not in body


def test_submit_chain_end_source_video_id_included_in_body():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            202,
            json={"job_id": "j1", "status": "queued", "created_at": "2026-01-01T00:00:00Z", "num_clips": 1},
        )

    set_client(_client_for_handler(handler))

    anyio.run(
        functools.partial(
            generate.submit_chain,
            "a prompt",
            [ChainClipArg(num_frames=169)],
            end_source_video_id="v-1",
            end_source_context_frames=24,
        )
    )

    body = captured["body"]
    assert body["end_source"] == {"video_id": "v-1", "context_frames": 24, "strength": 1.0}


def test_submit_chain_end_source_strength_included_in_body():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            202,
            json={"job_id": "j1", "status": "queued", "created_at": "2026-01-01T00:00:00Z", "num_clips": 1},
        )

    set_client(_client_for_handler(handler))

    anyio.run(
        functools.partial(
            generate.submit_chain,
            "a prompt",
            [ChainClipArg(num_frames=169)],
            end_source_video_id="v-1",
            end_source_context_frames=24,
            end_source_strength=0.4,
        )
    )

    body = captured["body"]
    assert body["end_source"] == {"video_id": "v-1", "context_frames": 24, "strength": 0.4}


def test_submit_chain_end_source_strength_included_for_image_id_too():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            202,
            json={"job_id": "j1", "status": "queued", "created_at": "2026-01-01T00:00:00Z", "num_clips": 1},
        )

    set_client(_client_for_handler(handler))

    anyio.run(
        functools.partial(
            generate.submit_chain,
            "a prompt",
            [ChainClipArg(num_frames=169)],
            end_source_image_id="i-1",
            end_source_context_frames=8,
            end_source_strength=0.0,
        )
    )

    body = captured["body"]
    assert body["end_source"] == {"image_id": "i-1", "context_frames": 8, "strength": 0.0}


def test_submit_chain_input_schema_exposes_end_source_strength():
    async def _run():
        mcp = build_server()
        return await mcp.list_tools()

    tools = anyio.run(_run)
    tool = next(t for t in tools if t.name == "submit_chain")
    prop = tool.inputSchema["properties"]["end_source_strength"]
    assert prop["default"] == 1.0


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


def test_submit_chain_keep_resident_on_included_in_body():
    # Default OFF (§48) -> the ON call is the one that reaches the wire, and
    # the default call must leave the frozen chain key set untouched.
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
            keep_resident=True,
        )
    )
    assert captured["body"]["keep_resident"] is True

    anyio.run(
        functools.partial(
            generate.submit_chain,
            "a prompt",
            [ChainClipArg(num_frames=25), ChainClipArg(num_frames=25)],
            keep_resident=False,
        )
    )
    assert set(captured["body"].keys()) == {
        "prompt", "width", "height", "frame_rate", "seed",
        "overlap_frames", "overlap_strength", "clips", "chunked_upsample",
    }


def test_submit_chain_fused_dequant_off_included_in_body():
    # §1-11, default ON since 2026-08-04 (§51) -> only the OFF call reaches the
    # wire; the default call must leave the frozen chain key set untouched.
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
            fused_gguf_dequant_kernel=False,
        )
    )
    assert captured["body"]["fused_gguf_dequant_kernel"] is False

    anyio.run(
        functools.partial(
            generate.submit_chain,
            "a prompt",
            [ChainClipArg(num_frames=25), ChainClipArg(num_frames=25)],
            fused_gguf_dequant_kernel=True,
        )
    )
    assert set(captured["body"].keys()) == {
        "prompt", "width", "height", "frame_rate", "seed",
        "overlap_frames", "overlap_strength", "clips", "chunked_upsample",
    }


def test_submit_chain_vae_mode_prune_vaed_included_in_body():
    # PrunaVAED (Docs/PENDING_TASKS_CLOSED.md §3-66, filed as §3-50 at the
    # time): same rule and same direction as submit_generate above. In a
    # chain one vae_mode applies to every clip and every stage
    # (api/models.py:456-460).
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
            vae_mode="prune_vaed",
        )
    )
    assert captured["body"]["vae_mode"] == "prune_vaed"

    anyio.run(
        functools.partial(
            generate.submit_chain,
            "a prompt",
            [ChainClipArg(num_frames=25), ChainClipArg(num_frames=25)],
            vae_mode="default",
        )
    )
    assert set(captured["body"].keys()) == {
        "prompt", "width", "height", "frame_rate", "seed",
        "overlap_frames", "overlap_strength", "clips", "chunked_upsample",
    }


def test_submit_chain_input_schema_exposes_vae_mode_enum():
    # G9: same schema contract as submit_generate.
    async def _run():
        mcp = build_server()
        return await mcp.list_tools()

    tools = anyio.run(_run)
    tool = next(t for t in tools if t.name == "submit_chain")
    prop = tool.inputSchema["properties"]["vae_mode"]
    assert prop["enum"] == ["default", "prune_vaed"]
    assert prop["default"] == "default"


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


# ---------------- retake (Docs/PENDING_TASKS_CLOSED.md §3-115, closed 2026-09-01)
#
# The five retake_* arguments were appended to submit_chain's signature (the
# same "append at the tail" discipline D13-D16 used for end source), so every
# existing positional-argument call above keeps working untouched. The nest they
# build carries ALL FIVE keys even at their defaults, matching RetakeSpec's own
# defaults (api/models.py:547-599) -- the same shape end_source sends.


def test_submit_chain_retake_body_exact():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            202,
            json={"job_id": "j1", "status": "queued", "created_at": "2026-01-01T00:00:00Z", "num_clips": 1},
        )

    set_client(_client_for_handler(handler))

    anyio.run(
        functools.partial(
            generate.submit_chain,
            "a prompt",
            [ChainClipArg(num_frames=121)],
            retake_video_id="v-1",
            retake_window_start_sec=2.5,
        )
    )

    body = captured["body"]
    # Defaults for head/tail/regenerate_audio are still spelled out on the wire.
    assert body["retake"] == {
        "video_id": "v-1",
        "window_start_sec": 2.5,
        "head_px": 25,
        "tail_px": 24,
        "regenerate_audio": True,
    }
    # The window length has exactly one source: clips[0].num_frames.
    assert body["clips"] == [{"num_frames": 121}]


def test_submit_chain_retake_non_default_glue_and_audio_in_body():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            202,
            json={"job_id": "j1", "status": "queued", "created_at": "2026-01-01T00:00:00Z", "num_clips": 1},
        )

    set_client(_client_for_handler(handler))

    anyio.run(
        functools.partial(
            generate.submit_chain,
            "a prompt",
            [ChainClipArg(num_frames=73)],
            retake_video_id="v-2",
            retake_window_start_sec=0.0,
            retake_head_px=33,  # 8n+1
            retake_tail_px=32,  # multiple of 8
            retake_regenerate_audio=False,
        )
    )

    assert captured["body"]["retake"] == {
        "video_id": "v-2",
        "window_start_sec": 0.0,
        "head_px": 33,
        "tail_px": 32,
        "regenerate_audio": False,
    }


def test_submit_chain_retake_without_window_start_raises_before_any_http_call():
    # Owner ruling (2026-09-01): window_start_sec has NO default, so a forgotten
    # one must not silently become 0.0 and retake the wrong part of the footage.
    # Same "a specification silently turns into something else" category as
    # CROP_SIZE_INCOMPLETE.
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"no HTTP call expected, got {request.method} {request.url.path}")

    set_client(_client_for_handler(handler))

    with pytest.raises(ToolError) as exc_info:
        anyio.run(
            functools.partial(
                generate.submit_chain,
                "a prompt",
                [ChainClipArg(num_frames=121)],
                retake_video_id="v-1",
            )
        )

    assert "RETAKE_WINDOW_START_REQUIRED" in str(exc_info.value)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"retake_window_start_sec": 1.0},
        {"retake_head_px": 33},
        {"retake_tail_px": 32},
        {"retake_regenerate_audio": False},
    ],
)
def test_submit_chain_retake_helper_args_without_video_id_raise_before_any_http_call(kwargs):
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"no HTTP call expected, got {request.method} {request.url.path}")

    set_client(_client_for_handler(handler))

    with pytest.raises(ToolError) as exc_info:
        anyio.run(
            functools.partial(
                generate.submit_chain,
                "a prompt",
                [ChainClipArg(num_frames=49), ChainClipArg(num_frames=49)],
                **kwargs,
            )
        )

    assert "RETAKE_ARGS_WITHOUT_VIDEO_ID" in str(exc_info.value)


def test_submit_chain_input_schema_exposes_retake_arguments():
    async def _run():
        mcp = build_server()
        return await mcp.list_tools()

    tools = anyio.run(_run)
    tool = next(t for t in tools if t.name == "submit_chain")
    props = tool.inputSchema["properties"]
    for name in (
        "retake_video_id",
        "retake_window_start_sec",
        "retake_head_px",
        "retake_tail_px",
        "retake_regenerate_audio",
    ):
        assert name in props, f"{name} missing from submit_chain schema"
    assert props["retake_head_px"]["default"] == 25
    assert props["retake_tail_px"]["default"] == 24
    assert props["retake_regenerate_audio"]["default"] is True
    # No usable default for the window start (owner ruling): the schema must not
    # advertise a 0.0 an agent would read as "fine to omit".
    assert props["retake_window_start_sec"].get("default") is None


# -------------- outpaint (Docs/PENDING_TASKS_CLOSED.md §3-122, closed 2026-09-01)


def test_submit_generate_outpaint_body_default_blend_omits_both_keys():
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
            width=1280,
            height=768,
            num_frames=49,
            reference_video_id="v-1",
            outpaint_pad_left=128,
            outpaint_pad_right=128,
        )
    )

    body = captured["body"]
    # blend_dilation_* omitted at the default (5) -- the same backward-compat
    # discipline the panel's useOutpaintForm.ts follows; freeze_source_audio
    # omitted at its own default (True).
    assert body["outpaint"] == {
        "pad_left": 128,
        "pad_right": 128,
        "pad_top": 0,
        "pad_bottom": 0,
    }


def test_submit_generate_outpaint_non_default_blend_follows_5_to_2():
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
            width=1280,
            height=768,
            reference_video_id="v-1",
            outpaint_pad_top=64,
            outpaint_blend_dilation_stage1=8,
            outpaint_freeze_source_audio=False,
        )
    )

    body = captured["body"]
    # round(8 * 2 / 5) == round(3.2) == 3 -- the port of outpaintGeometry.ts's
    # stage2FromStage1, which the panel is the source of truth for.
    assert body["outpaint"] == {
        "pad_left": 0,
        "pad_right": 0,
        "pad_top": 64,
        "pad_bottom": 0,
        "blend_dilation_stage1": 8,
        "blend_dilation_stage2": 3,
        "freeze_source_audio": False,
    }


@pytest.mark.parametrize(
    "stage1,expected_stage2",
    [(0, 0), (1, 1), (2, 1), (3, 1), (4, 2), (6, 2), (7, 3), (8, 3), (10, 4), (15, 6)],
)
def test_outpaint_stage2_from_stage1_matches_the_panel_formula(stage1, expected_stage2):
    # Python's round() is banker's rounding and JavaScript's Math.round() is
    # half-up, but r*2/5 never lands on a .5 for an integer r, so the port is
    # exact over the whole 0-15 range the API accepts.
    assert generate._outpaint_stage2_from_stage1(stage1) == expected_stage2


def test_submit_generate_all_pads_zero_sends_no_outpaint_key():
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
            outpaint_pad_left=0,
            outpaint_pad_right=0,
            outpaint_pad_top=0,
            outpaint_pad_bottom=0,
        )
    )

    body = captured["body"]
    assert "outpaint" not in body
    assert "loras" not in body  # ...and no LoRA is injected either
    assert set(body.keys()) == {"prompt", "width", "height", "num_frames", "frame_rate", "seed"}


def test_submit_generate_outpaint_auto_injects_the_in_outpainting_lora():
    # Owner ruling (2026-09-01): the panel pins this LoRA, the server requires
    # exactly one preprocess-free control adapter alongside `outpaint`, and an
    # agent that forgets it only ever sees a 422. So the tool adds it.
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
            width=1280,
            height=768,
            reference_video_id="v-1",
            outpaint_pad_bottom=128,
        )
    )

    assert captured["body"]["loras"] == [{"name": "in-outpainting", "strength": 1.0}]

    # An unrelated style LoRA is kept, with the control adapter appended.
    anyio.run(
        functools.partial(
            generate.submit_generate,
            "a prompt",
            width=1280,
            height=768,
            loras=[LoraArg(name="Pixar_Toon", strength=0.6)],
            reference_video_id="v-1",
            outpaint_pad_bottom=128,
        )
    )

    assert captured["body"]["loras"] == [
        {"name": "Pixar_Toon", "strength": 0.6},
        {"name": "in-outpainting", "strength": 1.0},
    ]


def test_submit_generate_outpaint_does_not_duplicate_an_existing_in_outpainting_lora():
    # Match on NAME ONLY: a caller who deliberately dialled the strength down to
    # 0.7 keeps that value rather than getting a second 1.0 entry beside it.
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
            width=1280,
            height=768,
            loras=[LoraArg(name="in-outpainting", strength=0.7)],
            reference_video_id="v-1",
            outpaint_pad_bottom=128,
        )
    )

    assert captured["body"]["loras"] == [{"name": "in-outpainting", "strength": 0.7}]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"outpaint_blend_dilation_stage1": 8},
        {"outpaint_freeze_source_audio": False},
    ],
)
def test_submit_generate_outpaint_helper_args_with_zero_pads_raise_before_any_http_call(kwargs):
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"no HTTP call expected, got {request.method} {request.url.path}")

    set_client(_client_for_handler(handler))

    with pytest.raises(ToolError) as exc_info:
        anyio.run(functools.partial(generate.submit_generate, "a prompt", **kwargs))

    assert "OUTPAINT_PADS_ALL_ZERO" in str(exc_info.value)


def test_submit_generate_input_schema_exposes_outpaint_arguments():
    async def _run():
        mcp = build_server()
        return await mcp.list_tools()

    tools = anyio.run(_run)
    tool = next(t for t in tools if t.name == "submit_generate")
    props = tool.inputSchema["properties"]
    for name in (
        "outpaint_pad_left",
        "outpaint_pad_right",
        "outpaint_pad_top",
        "outpaint_pad_bottom",
        "outpaint_blend_dilation_stage1",
        "outpaint_freeze_source_audio",
    ):
        assert name in props, f"{name} missing from submit_generate schema"
    assert props["outpaint_pad_left"]["default"] == 0
    assert props["outpaint_blend_dilation_stage1"]["default"] == 5
    assert props["outpaint_freeze_source_audio"]["default"] is True
    # blend_dilation_stage2 is DERIVED (5:2), never an argument -- exposing it
    # would give the ratio a second source of truth.
    assert "outpaint_blend_dilation_stage2" not in props


# ---- upload_video max_frames (Docs/PENDING_TASKS_CLOSED.md §3-122 companion)


def test_upload_video_max_frames_rides_on_the_query_and_returns_measurements(tmp_path):
    # api/uploads.py:57 declares max_frames as a Query parameter, and it doubles
    # as the "please measure this" opt-in (frame_count / fps come back only when
    # it is sent) -- which is how an agent works out a retake window's start
    # second without leaving MCP.
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["params"] = dict(request.url.params)
        return httpx.Response(
            200,
            json={
                "video_id": "v-1",
                "original_filename": "clip.mp4",
                "stored_path": "videos/v-1/input.mp4",
                "content_type": "video/mp4",
                "size_bytes": 5,
                "trimmed": False,
                "frame_count": 480,
                "fps": 24.0,
            },
        )

    set_client(_client_for_handler(handler))

    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"fake\x00")

    result = anyio.run(functools.partial(uploads.upload_video, str(clip), max_frames=100000))

    assert captured["path"].endswith("/upload/video")
    assert captured["params"] == {"max_frames": "100000"}
    assert result["frame_count"] == 480
    assert result["fps"] == 24.0


def test_upload_video_without_max_frames_sends_no_query_at_all(tmp_path):
    # Tripwire: an ordinary upload's request line must stay byte-identical to
    # what it was before max_frames existed.
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["params"] = dict(request.url.params)
        return httpx.Response(
            200,
            json={
                "video_id": "v-1",
                "original_filename": "clip.mp4",
                "stored_path": "videos/v-1/input.mp4",
                "content_type": "video/mp4",
                "size_bytes": 5,
                "trimmed": False,
                "frame_count": None,
                "fps": None,
            },
        )

    set_client(_client_for_handler(handler))

    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"fake\x00")

    result = anyio.run(uploads.upload_video, str(clip))

    assert captured["params"] == {}
    assert result["frame_count"] is None
    assert result["fps"] is None


def test_upload_video_input_schema_exposes_max_frames():
    async def _run():
        mcp = build_server()
        return await mcp.list_tools()

    tools = anyio.run(_run)
    tool = next(t for t in tools if t.name == "upload_video")
    assert "max_frames" in tool.inputSchema["properties"]
