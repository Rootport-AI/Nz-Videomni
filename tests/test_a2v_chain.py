"""Audio-to-video (source_audio on POST /generate/chain) — API + mock e2e (no GPU).

The mock runner mirrors the engine's ``chain.a2v`` contract: it renders ONE
synthetic mp4 of the full timeline and emits the same ``chain.a2v`` key set the
real worker does (geometry from :mod:`chain_math`), so the additive contract
(upload endpoint, schema, endpoint 404/422, mutual exclusion, metadata a2v block)
is pinned without weights. Frozen-API discipline: a request omitting source_audio
is byte-shape identical to before (regression test below).
"""

from __future__ import annotations

import argparse
import json
import struct
import wave

import pytest
import yaml
from fastapi.testclient import TestClient
from PIL import Image

import chain_math
import main
from services import video_io


BASE = {
    "prompt": "a bustling town square at dusk, a woman speaking, cinematic trailer",
    "width": 384,
    "height": 256,
    "frame_rate": 24.0,
    "num_inference_steps": 8,
    "guidance_scale": 1.0,
    "seed": 123,
    "pipeline": "distilled",
    "overlap_frames": 2,
    "overlap_strength": 0.5,
}


def _run_chain(client, clips, **overrides):
    payload = {**BASE, "clips": clips, **overrides}
    return client.post("/api/v1/generate/chain", json=payload)


def _make_wav(path, *, seconds=3.0, sr=16000, channels=1):
    """Write a small silent PCM wav fixture (a few KB) with an exact duration."""
    n = int(round(seconds * sr))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(struct.pack("<%dh" % (n * channels), *([0] * (n * channels))))
    return path


def _upload_audio(client, wav_path, name="voice.wav", content_type="audio/wav") -> str:
    data = wav_path.read_bytes()
    r = client.post("/api/v1/upload/audio", files={"file": (name, data, content_type)})
    assert r.status_code == 200, r.text
    return r.json()["audio_id"]


def _build_client(tmp_path, upload_overrides=None):
    """A mock-backend TestClient with optional upload-config overrides (used by the
    oversize test to set a tiny max_audio_size_mb without a giant fixture)."""
    cfg = {
        "server": {"log_dir": (tmp_path / "logs").as_posix()},
        "model": {"backend": "mock"},
        "output": {"dir": (tmp_path / "outputs").as_posix()},
        "upload": {"dir": (tmp_path / "uploads").as_posix(), **(upload_overrides or {})},
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    args = argparse.Namespace(
        listen=False, port=None, api_key=None, allow_all_cors=False,
        config=cfg_path.as_posix(), te_offload=None, dit_cpu_load=None,
    )
    app = main.build_app(args)
    c = TestClient(app)
    c.app_context = app.state.context  # type: ignore[attr-defined]
    return c


# ---------------------------------------------------------------- (a) regression


def test_chain_without_source_regression(client):
    """A chain omitting source_audio: source_audio defaults to None, no top-level
    or chain.a2v block — the pre-A2V metadata key set is unchanged."""
    r = _run_chain(client, [{"num_frames": 25}, {"num_frames": 25}])
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    job = client.get(f"/api/v1/jobs/{job_id}").json()
    assert job["status"] == "completed", job

    ctx = client.app_context
    meta = json.loads((ctx.config.output_dir / job_id / "metadata.json").read_text(encoding="utf-8"))
    assert meta["request"]["source_audio"] is None
    assert "a2v" not in meta
    assert "a2v" not in meta["chain"]


# ------------------------------------------------------------- (b) upload/audio


def test_upload_audio_roundtrip(client, tmp_path):
    wav = _make_wav(tmp_path / "voice.wav", seconds=1.0)
    r = client.post("/api/v1/upload/audio", files={"file": ("voice.wav", wav.read_bytes(), "audio/wav")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["audio_id"]
    assert body["stored_path"].endswith("/input.wav")
    assert body["content_type"] == "audio/wav"
    assert body["size_bytes"] == wav.stat().st_size

    ctx = client.app_context
    stored = ctx.audio_upload_store.path_for(body["audio_id"])
    assert stored.exists()
    assert stored.read_bytes() == wav.read_bytes()


def test_upload_audio_bad_extension_rejected(client):
    r = client.post("/api/v1/upload/audio", files={"file": ("bad.txt", b"nope", "text/plain")})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "UPLOAD_INVALID_TYPE"


def test_upload_audio_empty_rejected(client):
    r = client.post("/api/v1/upload/audio", files={"file": ("empty.wav", b"", "audio/wav")})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "UPLOAD_INVALID_TYPE"


def test_upload_audio_too_large_rejected(tmp_path):
    client = _build_client(tmp_path, upload_overrides={"max_audio_size_mb": 1})
    # ~2 MB of bytes with an allowed extension -> over the 1 MB cap.
    big = b"\x00" * (2 * 1024 * 1024)
    r = client.post("/api/v1/upload/audio", files={"file": ("big.wav", big, "audio/wav")})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "UPLOAD_TOO_LARGE"


# ------------------------------------------------------------------- (c) 422s


def test_a2v_conflicts_with_source_video_422(client):
    r = _run_chain(
        client, [{"num_frames": 49}],
        source_audio={"audio_id": "a"},
        source_video={"video_id": "v", "context_frames": 25},
    )
    assert r.status_code == 422
    assert "mutually exclusive" in r.text


def test_a2v_requires_single_clip_422(client):
    r = _run_chain(
        client, [{"num_frames": 49}, {"num_frames": 49}],
        source_audio={"audio_id": "a"},
    )
    assert r.status_code == 422
    assert "exactly 1 clip" in r.text


def test_a2v_source_too_short_422(client, tmp_path):
    # [49]@24fps needs a_total=51 audio-latent frames (~2.04s). A 1s upload
    # encodes to ~25 frames < 51 -> rejected up front.
    wav = _make_wav(tmp_path / "short.wav", seconds=1.0)
    aid = _upload_audio(client, wav)
    r = _run_chain(client, [{"num_frames": 49}], source_audio={"audio_id": aid})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "SOURCE_AUDIO_TOO_SHORT"


# -------------------------------------------------------------------- (d) 404


def test_a2v_unknown_audio_id_404(client):
    r = _run_chain(client, [{"num_frames": 49}], source_audio={"audio_id": "no-such-id"})
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "SOURCE_AUDIO_NOT_FOUND"


# ---------------------------------------------------------------- (e) mock e2e


def test_a2v_mock_e2e(client, tmp_path):
    """Upload a long-enough audio, run a single-clip A2V chain -> completed; the
    metadata a2v block carries the engine's key set with chain_math geometry."""
    wav = _make_wav(tmp_path / "voice.wav", seconds=3.0, sr=16000, channels=1)
    aid = _upload_audio(client, wav)

    r = _run_chain(client, [{"num_frames": 49}], source_audio={"audio_id": aid})
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["num_clips"] == 1
    job_id = body["job_id"]

    job = client.get(f"/api/v1/jobs/{job_id}").json()
    assert job["status"] == "completed", job

    ctx = client.app_context
    out = ctx.config.output_dir / job_id / "output.mp4"
    assert out.exists() and out.stat().st_size > 0

    layout = chain_math.compute_chain_layout([49], 24.0, kv=2)
    meta = json.loads((ctx.config.output_dir / job_id / "metadata.json").read_text(encoding="utf-8"))
    a2v = meta["a2v"]
    # geometry (from chain_math)
    assert a2v["a_total"] == layout.a_total
    assert a2v["encoded_audio_frames_available"] >= a2v["a_total"]
    # fixed engine-contract flags
    assert a2v["muxed_original_waveform"] is True
    assert a2v["vocoder_skipped"] is True
    # probed audio properties
    assert a2v["audio_sampling_rate"] == 16000
    assert a2v["audio_channels"] == 1
    assert a2v["muxed_audio_samples"] > 0
    # app-side provenance
    assert a2v["source_audio_id"] == aid
    assert a2v["source_audio_path"].endswith("input.wav")
    # the additive request field round-trips
    assert meta["request"]["source_audio"]["audio_id"] == aid


def test_a2v_allows_clip0_conditioning(client, tmp_path, png_bytes):
    """source_audio + clip-0 conditioning image is ALLOWED (no added restriction)."""
    up = client.post("/api/v1/upload/image", files={"file": ("k.png", png_bytes, "image/png")})
    assert up.status_code == 200, up.text
    image_id = up.json()["image_id"]

    wav = _make_wav(tmp_path / "voice.wav", seconds=3.0)
    aid = _upload_audio(client, wav)

    clips = [{
        "num_frames": 49,
        "conditioning_images": [{"image_id": image_id, "frame_idx": 0, "strength": 0.8}],
    }]
    r = _run_chain(client, clips, source_audio={"audio_id": aid})
    assert r.status_code == 202, r.text
    job = client.get(f"/api/v1/jobs/{r.json()['job_id']}").json()
    assert job["status"] == "completed", job


# ---------------------------------------------------------------- (f) geometry


def test_a2v_geometry_matches_audio_latents_required():
    """The preflight length gate and the engine both source a_total from the SAME
    pure function — pin that identity so app/engine never drift."""
    layout = chain_math.compute_chain_layout([49], 24.0, kv=2)
    assert chain_math.audio_latents_required([49], 24.0, kv=2) == layout.a_total


def test_a2v_preflight_boundary(client, tmp_path):
    """Audio that encodes to EXACTLY a_total frames passes; one frame short 422s.
    available = round(duration * 25); required = a_total = 51 for [49]@24fps."""
    required = chain_math.audio_latents_required([49], 24.0, kv=2)
    assert required == 51

    # duration = 51/25 = 2.04s -> available == 51 (exactly enough): passes.
    ok = _make_wav(tmp_path / "exact.wav", seconds=required / chain_math.AUDIO_LATENTS_PER_SEC)
    aid_ok = _upload_audio(client, ok, name="exact.wav")
    r_ok = _run_chain(client, [{"num_frames": 49}], source_audio={"audio_id": aid_ok})
    assert r_ok.status_code == 202, r_ok.text

    # duration = 50/25 = 2.0s -> available == 50 < 51: rejected.
    short = _make_wav(tmp_path / "under.wav", seconds=(required - 1) / chain_math.AUDIO_LATENTS_PER_SEC)
    aid_short = _upload_audio(client, short, name="under.wav")
    r_short = _run_chain(client, [{"num_frames": 49}], source_audio={"audio_id": aid_short})
    assert r_short.status_code == 422
    assert r_short.json()["error"]["code"] == "SOURCE_AUDIO_TOO_SHORT"
