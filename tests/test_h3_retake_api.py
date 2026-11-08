"""MiniMax H3 Retake (``POST /generate/chain`` with ``retake``, 台帳 §1-123) — API + mock (no GPU).

Pins the app half of H3's Retake in one place: the family's rules (the 22-frame
glue bands written back when omitted and refused when explicit otherwise, the
``regenerate_audio=false`` refusal, the ``audio_strength`` default written
back), the geometry's 422s (17n+5, at least 73, at most 481), the schema's
exclusivity with ``clips[0].references``, the silent-upload preflight for a
Strength above 0, the mock's whole-window mp4 and ``metadata["retake"]``, and
what ``GET /models`` publishes (H3's ``constraints.retake``; nothing on LTX).
Also the LTX side of the split: LTX refuses ``audio_strength`` and still
refuses an explicit ``head_px=22`` (``chain_math``'s grid, now that the schema
no longer checks it).
"""

from __future__ import annotations

import json
import shutil

import pytest

import h3_chain_math
from services import video_io
from test_engine_h3_adapter import PROMPT, _activate, _base
from test_h3_ref_api import _refs
from test_retake_api import BASE as LTX_BASE
from test_retake_api import _upload_window_source

_HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
needs_ffmpeg = pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg/ffprobe not on PATH")

#: An H3 chain body (24 fps, 32-multiple size); the window is clips[0].
H3_BODY = {"prompt": PROMPT, "width": 384, "height": 256, "frame_rate": 24.0, "seed": 123}

#: The one window length used throughout (17 * 7 + 5).
WINDOW = 124


def _retake(**over) -> dict:
    return {"video_id": "vid", "window_start_sec": 0.0, **over}


def _post_h3(client, num_frames=WINDOW, **retake):
    return client.post(
        "/api/v1/generate/chain",
        json={**H3_BODY, "clips": [{"num_frames": num_frames}], "retake": _retake(**retake)},
    )


def _metadata(client, job_id: str) -> dict:
    ctx = client.app_context
    return json.loads((ctx.config.output_dir / job_id / "metadata.json").read_text("utf-8"))


def _job_count(client) -> int:
    return len(client.get("/api/v1/jobs").json())


# --------------------------------------------------------------------------- #
# GET /models
# --------------------------------------------------------------------------- #


def test_models_publishes_the_retake_strength_on_h3_only(three_family_client):
    h3_entry = _base(three_family_client, "MiniMaxH3")
    assert h3_entry["constraints"]["retake"] == {
        "audio_strength": {"min": 0.0, "max": 1.0, "step": 0.05, "default": 0.5}
    }
    assert "retake" not in h3_entry["unsupported_features"]
    for ltx in ("LTX23", "LTX25"):
        entry = _base(three_family_client, ltx)
        assert "retake" not in entry["constraints"]
        assert "retake" not in entry["unsupported_features"]


# --------------------------------------------------------------------------- #
# H3's 422s (all before any upload is looked up: the video_id is a dummy)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "num_frames, retake, message",
    [
        (WINDOW, {"head_px": 25}, "retake.head_px must be 22 on MiniMax H3"),
        (WINDOW, {"tail_px": 30}, "retake.tail_px must be 22 on MiniMax H3"),
        # what an older MCP sent on every call: the LTX defaults, explicit
        (WINDOW, {"head_px": 25, "tail_px": 24}, "retake.head_px must be 22 on MiniMax H3"),
        (WINDOW, {"head_px": 22, "tail_px": 24}, "retake.tail_px must be 22 on MiniMax H3"),
        (WINDOW, {"regenerate_audio": False}, "use retake.audio_strength=1.0 to keep the original audio"),
        # the geometry: on the 17n+5 grid but under the window floor
        (56, {}, f"must be at least {h3_chain_math.H3_RETAKE_MIN_FRAMES} frames"),
        # off the grid
        (100, {}, "clips[0]: num_frames must be 17n+5"),
    ],
)
def test_h3_retake_422s(three_family_client, num_frames, retake, message):
    _activate(three_family_client, "MiniMaxH3")
    before = _job_count(three_family_client)
    r = _post_h3(three_family_client, num_frames, **retake)
    assert r.status_code == 422, r.text
    assert r.json()["error"]["code"] == "VALIDATION_ERROR"
    assert message in r.text
    assert _job_count(three_family_client) == before


def test_h3_retake_window_over_481_is_422(three_family_client):
    _activate(three_family_client, "MiniMaxH3")
    # The schema's ceiling refuses it first (H3_FRAMES_MAX is derived from it,
    # so REQUEST_RULES.violation's range message is not reached).
    r = _post_h3(three_family_client, 482)
    assert r.status_code == 422, r.text
    assert r.json()["error"]["code"] == "VALIDATION_ERROR"
    assert "Input should be less than or equal to 481" in r.text


def test_h3_retake_strength_out_of_range_is_422(three_family_client):
    _activate(three_family_client, "MiniMaxH3")
    r = _post_h3(three_family_client, audio_strength=1.5)
    assert r.status_code == 422, r.text
    assert "audio_strength" in r.text


def test_retake_and_clip0_references_are_mutually_exclusive(three_family_client):
    _activate(three_family_client, "MiniMaxH3")
    r = three_family_client.post(
        "/api/v1/generate/chain",
        json={
            **H3_BODY,
            "clips": [{"num_frames": WINDOW, "references": _refs("picture", 1)}],
            "retake": _retake(),
        },
    )
    assert r.status_code == 422, r.text
    assert r.json()["error"]["code"] == "VALIDATION_ERROR"
    assert "retake is mutually exclusive with clips[0].references" in r.text


# --------------------------------------------------------------------------- #
# LTX's side of the split
# --------------------------------------------------------------------------- #


def test_ltx_refuses_audio_strength(three_family_client):
    for base_model in ("LTX23", "LTX25"):
        _activate(three_family_client, base_model)
        r = three_family_client.post(
            "/api/v1/generate/chain",
            json={**LTX_BASE, "clips": [{"num_frames": 169}], "retake": _retake(audio_strength=0.5)},
        )
        assert r.status_code == 422, r.text
        assert r.json()["error"]["code"] == "VALIDATION_ERROR"
        assert "retake.audio_strength is only accepted by MiniMax H3" in r.text


def test_ltx_still_refuses_an_explicit_22_frame_head(three_family_client):
    # The schema no longer checks the glue grids; chain_math still does.
    for base_model in ("LTX23", "LTX25"):
        _activate(three_family_client, base_model)
        r = three_family_client.post(
            "/api/v1/generate/chain",
            json={**LTX_BASE, "clips": [{"num_frames": 169}], "retake": _retake(head_px=22)},
        )
        assert r.status_code == 422, r.text
        assert "retake head_px must be 8n+1 (got 22)" in r.text


# --------------------------------------------------------------------------- #
# the material preflight and the mock end to end
# --------------------------------------------------------------------------- #


@needs_ffmpeg
def test_h3_strength_above_zero_on_a_silent_upload_is_422(three_family_client, tmp_path):
    _activate(three_family_client, "MiniMaxH3")
    vid = _upload_window_source(three_family_client, tmp_path, audio=False)
    r = _post_h3(three_family_client, video_id=vid, audio_strength=0.5)
    assert r.status_code == 422, r.text
    assert "RETAKE_VIDEO_NO_AUDIO" in r.text
    assert "no audio stream" in r.text


@needs_ffmpeg
def test_h3_mock_retake_writes_the_defaults_back_and_the_metadata(three_family_client, tmp_path):
    _activate(three_family_client, "MiniMaxH3")
    vid = _upload_window_source(three_family_client, tmp_path)
    r = _post_h3(three_family_client, video_id=vid, window_start_sec=2.0)
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    job = three_family_client.get(f"/api/v1/jobs/{job_id}").json()
    assert job["status"] == "completed", job
    meta = _metadata(three_family_client, job_id)

    # the omitted fields, as validate_chain wrote them back
    request = meta["request"]["retake"]
    assert (request["head_px"], request["tail_px"]) == (22, 22)
    assert request["audio_strength"] == 0.5
    assert request["regenerate_audio"] is True

    rt = meta["retake"]
    # geometry: h3_chain_math's eight keys, the same names as LTX's
    layout = h3_chain_math.compute_h3_chain_layout([WINDOW], retake=True).to_dict()["retake"]
    assert {k: rt[k] for k in layout} == layout
    assert (rt["window_px"], rt["head_px"], rt["tail_px"]) == (WINDOW, 22, 22)
    assert (rt["n_head_v"], rt["n_tail_v"], rt["n_head_a"]) == (7, 7, 36)
    assert rt["free_middle_px"] == [22, WINDOW - 22]
    # runtime: the LTX mock's names, audio_strength instead of regenerate_audio
    assert rt["audio_strength"] == 0.5
    assert rt["source_had_audio"] is True
    assert rt["audio_frozen"] is True
    assert rt["muxed_original_waveform"] is False
    assert rt["decoded_frames_px"] == WINDOW
    assert "regenerate_audio" not in rt
    assert "freeze_proof" not in rt
    # provenance (pipeline_manager, family-independent)
    assert rt["retake_video_id"] == vid
    assert rt["window_start_frame"] == 48
    assert rt["written_frames"] == WINDOW

    # the deliverable is the whole window
    out_dir = three_family_client.app_context.config.output_dir / job_id
    assert video_io.frame_count(out_dir / "output.mp4") == WINDOW
    assert video_io.frame_count(out_dir / "_retake_window.mp4") == WINDOW


@needs_ffmpeg
def test_h3_mock_retake_full_strength_records_the_original_waveform(three_family_client, tmp_path):
    _activate(three_family_client, "MiniMaxH3")
    vid = _upload_window_source(three_family_client, tmp_path)
    r = _post_h3(three_family_client, video_id=vid, head_px=22, tail_px=22, audio_strength=1.0)
    assert r.status_code == 202, r.text
    meta = _metadata(three_family_client, r.json()["job_id"])
    assert meta["request"]["retake"]["audio_strength"] == 1.0
    assert meta["retake"]["audio_strength"] == 1.0
    assert meta["retake"]["muxed_original_waveform"] is True


@needs_ffmpeg
def test_h3_mock_retake_on_a_silent_upload_at_strength_zero(three_family_client, tmp_path):
    _activate(three_family_client, "MiniMaxH3")
    vid = _upload_window_source(three_family_client, tmp_path, audio=False)
    # Strength 0 を明示（省略すると既定 0.5 が書き戻されて 422。次のテスト）
    r = _post_h3(three_family_client, 73, video_id=vid, audio_strength=0.0)
    assert r.status_code == 202, r.text
    meta = _metadata(three_family_client, r.json()["job_id"])
    rt = meta["retake"]
    assert rt["upload_has_audio"] is False
    assert rt["source_had_audio"] is False
    assert rt["audio_frozen"] is False
    assert rt["muxed_original_waveform"] is False
    assert rt["decoded_frames_px"] == 73


@needs_ffmpeg
def test_h3_omitted_strength_on_a_silent_upload_is_422(three_family_client, tmp_path):
    """The written-back default 0.5 is above zero, so the window's original
    audio is needed — a silent upload is refused by the preflight."""
    _activate(three_family_client, "MiniMaxH3")
    vid = _upload_window_source(three_family_client, tmp_path, audio=False)
    r = _post_h3(three_family_client, 73, video_id=vid)
    assert r.status_code == 422, r.text
    assert "RETAKE_VIDEO_NO_AUDIO" in r.text
    assert "has no audio stream" in r.json()["error"]["detail"]
