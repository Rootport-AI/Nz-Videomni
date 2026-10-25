"""MiniMax H3 single-shot A2V (``source_audio``) over the mock backend.

Pins the whole route in one place: the upload, the 404/422 guards (unknown id,
audio shorter than the video), the engine scope (LTX refuses it with the
H3-only sentence), ``metadata["a2v"]`` (engine-side mock keys + app-side
provenance, disjoint), and the real worker payload (``source_audio`` sits right
before ``output_path``; a request without it is today's golden, byte for byte).
"""

from __future__ import annotations

import struct
import wave

import pytest

from test_engine_h3_adapter import (
    GOLDEN_GENERATE_KEYS_H3,
    PROMPT,
    _activate,
    _capturing_backend,
    _metadata,
    _validated,
)

H3_ONLY_DETAIL = "単発の A2V は MiniMax H3 専用です。LTX の A2V は Chained の音声で使えます。"
# A 22-frame clip at 24 fps is 0.917 s; 2.0 s covers it, 0.5 s does not.
BODY = {"prompt": PROMPT, "width": 640, "height": 384, "num_frames": 22}

#: What the mock reports (engine side) and what the app adds (app side).
ENGINE_KEYS = {
    "mask_value", "audio_latents", "required_samples", "encoded_samples",
    "channels_used", "resampler", "encoder_tensors", "audio_output",
    "muxed_audio_samples", "mock",
}
APP_KEYS = {"source_audio_id", "strength", "source"}


def _wav(path, seconds: float, sr: int = 44100):
    n = int(round(seconds * sr))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(struct.pack("<%dh" % n, *([0] * n)))
    return path


def _upload(client, tmp_path, seconds: float) -> str:
    wav = _wav(tmp_path / f"voice_{seconds}.wav", seconds)
    r = client.post("/api/v1/upload/audio", files={"file": ("voice.wav", wav.read_bytes(), "audio/wav")})
    assert r.status_code == 200, r.text
    return r.json()["audio_id"]


def _run(client, **extra):
    r = client.post("/api/v1/generate", json={**BODY, **extra})
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    assert client.get(f"/api/v1/jobs/{job_id}").json()["status"] == "completed"
    return _metadata(client, job_id)


def test_a2v_metadata_carries_engine_and_app_records(three_family_client, tmp_path):
    _activate(three_family_client, "MiniMaxH3")
    audio_id = _upload(three_family_client, tmp_path, 2.0)
    meta = _run(three_family_client, source_audio={"audio_id": audio_id, "strength": 0.5})
    a2v = meta["a2v"]
    assert set(a2v) == ENGINE_KEYS | APP_KEYS
    assert not ENGINE_KEYS & APP_KEYS
    assert a2v["source_audio_id"] == audio_id and a2v["strength"] == 0.5
    assert a2v["source"]["duration_sec"] == pytest.approx(2.0, abs=0.01)
    assert (a2v["source"]["sample_rate"], a2v["source"]["channels"]) == (44100, 1)
    assert a2v["mask_value"] == 0.5 and a2v["audio_output"] == "decoded"
    assert a2v["mock"] is True
    assert "h3" not in meta  # the mock has nothing else to report


def test_full_strength_keeps_the_original_audio_and_the_default_is_half(three_family_client, tmp_path):
    from api.models import SingleSourceAudioSpec
    from services import engines

    _activate(three_family_client, "MiniMaxH3")
    audio_id = _upload(three_family_client, tmp_path, 2.0)
    explicit = _run(three_family_client, source_audio={"audio_id": audio_id, "strength": 1.0})["a2v"]
    assert explicit["strength"] == 1.0 and explicit["audio_output"] == "original"
    default = _run(three_family_client, source_audio={"audio_id": audio_id})["a2v"]
    assert default["strength"] == 0.5 and default["audio_output"] == "decoded"
    # the two default literals (published constraint / request model) cannot drift
    published = engines.request_constraints("h3")["source_audio"]["strength"]["default"]
    assert published == SingleSourceAudioSpec.model_fields["strength"].default


def test_audio_shorter_than_the_video_is_422(three_family_client, tmp_path):
    _activate(three_family_client, "MiniMaxH3")
    audio_id = _upload(three_family_client, tmp_path, 0.5)
    before = len(three_family_client.get("/api/v1/jobs").json())
    r = three_family_client.post("/api/v1/generate", json={**BODY, "source_audio": {"audio_id": audio_id}})
    assert r.status_code == 422, r.text
    error = r.json()["error"]
    assert error["code"] == "SOURCE_AUDIO_TOO_SHORT"
    assert error["message"] == "source audio is shorter than the requested video length"
    assert "0.917" in error["detail"]  # the length the 22-frame clip needs
    assert len(three_family_client.get("/api/v1/jobs").json()) == before


def test_unknown_audio_id_is_404(three_family_client):
    _activate(three_family_client, "MiniMaxH3")
    r = three_family_client.post("/api/v1/generate", json={**BODY, "source_audio": {"audio_id": "nope"}})
    assert r.status_code == 404, r.text
    assert r.json()["error"]["code"] == "SOURCE_AUDIO_NOT_FOUND"


def test_a_keyframe_and_source_audio_together_are_allowed(three_family_client, tmp_path, png_bytes):
    _activate(three_family_client, "MiniMaxH3")
    audio_id = _upload(three_family_client, tmp_path, 2.0)
    image_id = three_family_client.post(
        "/api/v1/upload/image", files={"file": ("first.png", png_bytes, "image/png")}
    ).json()["image_id"]
    meta = _run(
        three_family_client,
        source_audio={"audio_id": audio_id},
        conditioning_images=[{"image_id": image_id, "frame_idx": 0}],
    )
    assert meta["generation_mode"] == "i2v" and "a2v" in meta


@pytest.mark.parametrize("base_model", ["LTX23", "LTX25"])
def test_ltx_refuses_source_audio_with_the_h3_only_sentence(three_family_client, tmp_path, base_model):
    _activate(three_family_client, base_model)
    audio_id = _upload(three_family_client, tmp_path, 2.0)
    r = three_family_client.post(
        "/api/v1/generate",
        json={"prompt": PROMPT, "num_frames": 25, "source_audio": {"audio_id": audio_id}},
    )
    assert r.status_code == 422, r.text
    error = r.json()["error"]
    assert error["code"] == "FEATURE_UNSUPPORTED"
    assert error["detail"] == H3_ONLY_DETAIL


def test_a_job_without_source_audio_has_no_a2v_record(three_family_client):
    _activate(three_family_client, "MiniMaxH3")
    meta = _run(three_family_client)
    assert "a2v" not in meta and "h3" not in meta
    assert meta["request"]["source_audio"] is None


def test_real_payload_places_source_audio_before_output_path(tmp_path):
    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"wav")
    captured: list[dict] = []
    be = _capturing_backend(captured)
    req = _validated(
        source_audio={"audio_id": "aud-1", "strength": 0.5}, keep_resident=True
    )
    be.generate(req, tmp_path / "job", seed=1, source_audio_path=audio)
    (payload,) = captured
    assert payload["source_audio"] == {"path": str(audio), "strength": 0.5}
    keys = list(payload)
    assert keys[-3:] == ["source_audio", "output_path", "keep_resident"]
    assert [k for k in keys if k not in ("source_audio", "keep_resident")] == GOLDEN_GENERATE_KEYS_H3

