"""MiniMax H3 Chained with an input video (V2V) and clip 0's key frames (i2v),
over HTTP on the mock (台帳 §1-121).

Pins the app half: the family's ``context_frames`` rule (omitted -> 22 written
back, any other explicit value -> 422 in the H3 wording), clip 0's 39-frame
floor with a source (``h3_chain_math``), the app's tail cut / preflight running
on 22 (``SOURCE_VIDEO_TOO_SHORT``), the mock's ``metadata.v2v`` block (the
engine's sixteen keys plus the app's three provenance keys), the audio-handle
sidecar, the delivered duration, ``is_v2v`` and JOIN; clip 0's key frames
(accepted without a strength, counted as ``images_used``; a strength is the
single path's ``keyframe_strength`` refusal); the schema's exclusivity of a
source and a clip-0 image; and that LTX 2.3's V2V path is unchanged and never
answers in H3's words.
"""

from __future__ import annotations

import pytest

from test_engine_h3_adapter import PROMPT, _activate, _metadata
from test_h3_ref_api import _make_mp4, _need_ffmpeg, _upload_image
from test_v2v_chain import _make_source_mp4, _upload_source

#: The engine's ``done.chain.v2v`` keys (the LTX chain's sixteen).
ENGINE_V2V_KEYS = {
    "source_context_px",
    "n_ctx_v",
    "n_ctx_a",
    "trim_px",
    "v2v_context_junction_px",
    "new_frames_px",
    "context_frames",
    "freeze_ka",
    "trimmed_px",
    "trimmed_audio_samples",
    "audio_fade_in_samples",
    "source_had_audio",
    "audio_head_frozen",
    "decoded_frames_px",
    "audio_handle_filename",
    "handle_context_seconds",
}
#: What the app adds to ``metadata.v2v`` (provenance of the upload).
APP_V2V_KEYS = {"source_video_id", "source_fps", "resampled"}

H3_CONTEXT_WORDING = "source_video.context_frames must be 22 on MiniMax H3"
CHAIN_BODY = {"prompt": PROMPT, "width": 640, "height": 384}


def _post_chain(client, clips, **extra):
    return client.post("/api/v1/generate/chain", json={**CHAIN_BODY, "clips": clips, **extra})


def _upload_av_source(client, tmp_path, seconds: float = 3.0) -> str:
    """A small real mp4 with a 440 Hz audio track (72 frames at 24 fps)."""
    _need_ffmpeg()
    return _upload_source(client, _make_mp4(tmp_path / "src_av.mp4", audio="sine", seconds=seconds))


def _completed(client, response) -> str:
    assert response.status_code == 202, response.text
    job_id = response.json()["job_id"]
    job = client.get(f"/api/v1/jobs/{job_id}").json()
    assert job["status"] == "completed", job
    return job_id


# --------------------------------------------------------------------------- #
# V2V
# --------------------------------------------------------------------------- #


def test_v2v_two_clips_metadata_handle_duration_and_join(three_family_client, tmp_path):
    client = three_family_client
    _activate(client, "MiniMaxH3")
    vid = _upload_av_source(client, tmp_path)
    r = _post_chain(
        client, [{"num_frames": 56}, {"num_frames": 39}], source_video={"video_id": vid}
    )
    job_id = _completed(client, r)
    total = 56 + 39 - 22  # the timeline before the trim (one seam)

    meta = _metadata(client, job_id)
    v2v = meta["v2v"]
    assert set(v2v) == ENGINE_V2V_KEYS | APP_V2V_KEYS
    assert v2v["source_video_id"] == vid and v2v["resampled"] is False
    assert abs(v2v["source_fps"] - 24.0) < 0.5
    assert (v2v["context_frames"], v2v["source_context_px"], v2v["trim_px"]) == (22, 22, 22)
    assert (v2v["n_ctx_v"], v2v["n_ctx_a"], v2v["v2v_context_junction_px"]) == (7, 37, 21)
    assert v2v["new_frames_px"] == total - 22
    assert v2v["decoded_frames_px"] == total
    assert v2v["source_had_audio"] is True and v2v["audio_head_frozen"] is True
    assert v2v["freeze_ka"] == 37
    assert v2v["handle_context_seconds"] == round(22 / 24, 6)
    # The engine's clip 0 head proof is the real latents' only (the mock fabricates none).
    assert meta["h3"]["clips"][0]["head_equals_prev_tail"] is None

    job_dir = client.app_context.config.output_dir / job_id
    assert v2v["audio_handle_filename"] == "output_audio_handle.wav"
    assert (job_dir / v2v["audio_handle_filename"]).stat().st_size > 0
    assert meta["output"]["duration_seconds"] == round((total - 22) / 24, 3)
    assert meta["request"]["source_video"]["context_frames"] == 22

    job = client.get(f"/api/v1/jobs/{job_id}").json()
    assert job["is_v2v"] is True

    r = client.post(f"/api/v1/jobs/{job_id}/join", json={"audio_smoothing": True})
    assert r.status_code == 200, r.text
    assert (job_dir / "joined.mp4").stat().st_size > 0


def test_v2v_one_clip_is_accepted(three_family_client, tmp_path):
    client = three_family_client
    _activate(client, "MiniMaxH3")
    vid = _upload_av_source(client, tmp_path)
    job_id = _completed(
        client,
        _post_chain(client, [{"num_frames": 39}], source_video={"video_id": vid, "context_frames": 22}),
    )
    meta = _metadata(client, job_id)
    assert meta["v2v"]["new_frames_px"] == 39 - 22


def test_v2v_clip0_needs_39_frames(three_family_client):
    _activate(three_family_client, "MiniMaxH3")
    r = _post_chain(
        three_family_client, [{"num_frames": 22}, {"num_frames": 39}],
        source_video={"video_id": "not-looked-up"},
    )
    assert r.status_code == 422, r.text
    assert r.json()["error"]["code"] == "VALIDATION_ERROR"
    assert "clips[0]: num_frames must be at least 39 (source context 22 + one grid step)" in r.text


def test_v2v_explicit_context_other_than_22_is_422(three_family_client):
    _activate(three_family_client, "MiniMaxH3")
    r = _post_chain(
        three_family_client, [{"num_frames": 124}],
        source_video={"video_id": "not-looked-up", "context_frames": 73},
    )
    assert r.status_code == 422, r.text
    assert r.json()["error"]["code"] == "VALIDATION_ERROR"
    assert H3_CONTEXT_WORDING in r.text
    assert "send 22 or omit it" in r.text


def test_v2v_omitted_context_is_stored_as_22(three_family_client, tmp_path):
    client = three_family_client
    _activate(client, "MiniMaxH3")
    vid = _upload_av_source(client, tmp_path)
    job_id = _completed(client, _post_chain(client, [{"num_frames": 39}], source_video={"video_id": vid}))
    assert _metadata(client, job_id)["request"]["source_video"]["context_frames"] == 22


def test_v2v_source_shorter_than_22_frames_is_422(three_family_client, tmp_path):
    client = three_family_client
    _activate(client, "MiniMaxH3")
    vid = _upload_source(client, _make_source_mp4(tmp_path / "short.mp4", n_frames=15, fps=24.0))
    r = _post_chain(client, [{"num_frames": 39}], source_video={"video_id": vid})
    assert r.status_code == 422, r.text
    assert r.json()["error"]["code"] == "SOURCE_VIDEO_TOO_SHORT"


def test_v2v_and_a_clip0_image_are_the_schemas_exclusivity(three_family_client):
    _activate(three_family_client, "MiniMaxH3")
    r = _post_chain(
        three_family_client,
        [{"num_frames": 39, "conditioning_images": [{"image_id": "x", "frame_idx": 0}]}],
        source_video={"video_id": "x"},
    )
    assert r.status_code == 422, r.text
    assert "mutually exclusive" in r.text


# --------------------------------------------------------------------------- #
# i2v (clip 0's key frames)
# --------------------------------------------------------------------------- #


def test_i2v_start_and_middle_key_frames_are_accepted(three_family_client, png_bytes):
    client = three_family_client
    _activate(client, "MiniMaxH3")
    first, middle = _upload_image(client, png_bytes), _upload_image(client, png_bytes)
    job_id = _completed(
        client,
        _post_chain(
            client,
            [
                {
                    "num_frames": 124,
                    "conditioning_images": [
                        {"image_id": first, "frame_idx": 0},
                        {"image_id": middle, "frame_idx": 60},
                    ],
                },
                {"num_frames": 39},
            ],
        ),
    )
    meta = _metadata(client, job_id)
    assert meta["h3"]["images_used"] == 2
    assert [ci["strength"] for ci in meta["request"]["clips"][0]["conditioning_images"]] == [None, None]
    assert "v2v" not in meta


def test_i2v_key_frame_strength_is_refused(three_family_client):
    _activate(three_family_client, "MiniMaxH3")
    r = _post_chain(
        three_family_client,
        [
            {"num_frames": 124, "conditioning_images": [{"image_id": "x", "frame_idx": 0, "strength": 1.0}]},
            {"num_frames": 39},
        ],
    )
    assert r.status_code == 422, r.text
    error = r.json()["error"]
    assert error["code"] == "FEATURE_UNSUPPORTED"
    assert "'keyframe_strength'" in error["message"]


# --------------------------------------------------------------------------- #
# LTX 2.3 is unchanged
# --------------------------------------------------------------------------- #


def test_ltx_v2v_is_unchanged_after_h3(three_family_client, tmp_path):
    client = three_family_client
    _activate(client, "MiniMaxH3")
    _activate(client, "LTX23")
    vid = _upload_source(client, _make_source_mp4(tmp_path / "src24.mp4", n_frames=50, fps=24.0))
    # An LTX-shaped V2V request runs on LTX's path.
    job_id = _completed(
        client,
        client.post(
            "/api/v1/generate/chain",
            json={
                "prompt": PROMPT, "width": 384, "height": 256, "overlap_frames": 2,
                "clips": [{"num_frames": 49}],
                "source_video": {"video_id": vid, "context_frames": 25},
            },
        ),
    )
    assert _metadata(client, job_id)["v2v"]["context_frames"] == 25
    # H3's request on LTX: refused by LTX's own rules, never in H3's words.
    for body in (
        {"clips": [{"num_frames": 56}, {"num_frames": 39}], "source_video": {"video_id": vid}},
        {"clips": [{"num_frames": 49}], "source_video": {"video_id": vid, "context_frames": 22}},
    ):
        r = client.post("/api/v1/generate/chain", json={"prompt": PROMPT, **body})
        assert r.status_code == 422, r.text
        assert "MiniMax H3" not in r.text
