"""Audio-to-video (source_audio on POST /generate/chain) — API + mock e2e (no GPU).

The mock runner mirrors the engine's ``chain.a2v`` contract: it renders ONE
synthetic mp4 of the full timeline and emits the same ``chain.a2v`` key set the
real worker does (geometry from :mod:`chain_math`), so the additive contract
(upload endpoint, schema, endpoint 404/422, mutual exclusion, metadata a2v block)
is pinned without weights. Frozen-API discipline: a request omitting source_audio
is byte-shape identical to before (regression test below).

Long A2V (PENDING_TASKS §1-16): one uploaded audio drives 1..24 clips. There is
no per-clip audio upload — ``chain_math.audio_segment_windows`` tiles the ONE
global audio latent across the stage-1 segments — so the multi-clip cases below
differ from the single-clip ones only in how much audio the length gate demands.
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
        # §3-97 P5: the runtime-state file, in tmp like every other
        # writable location -- never the repository's own state.json.
        "state_file": (tmp_path / "state.json").as_posix(),
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


def test_a2v_source_too_short_422(client, tmp_path):
    # [49]@24fps needs a_total=51 audio-latent frames (~2.04s). A 1s upload
    # encodes to ~25 frames < 51 -> rejected up front.
    wav = _make_wav(tmp_path / "short.wav", seconds=1.0)
    aid = _upload_audio(client, wav)
    r = _run_chain(client, [{"num_frames": 49}], source_audio={"audio_id": aid})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "SOURCE_AUDIO_TOO_SHORT"


def test_a2v_multi_clip_source_too_short_422(client, tmp_path):
    """Long A2V uses the SAME length gate: [241,241]@24fps kv=2 assembles to 473
    pixel frames -> a_total=493 (~19.7s), so a 1s upload is still rejected up
    front. The clip-count guard is gone; the length guard is not."""
    wav = _make_wav(tmp_path / "short2.wav", seconds=1.0)
    aid = _upload_audio(client, wav)
    r = _run_chain(
        client, [{"num_frames": 241}, {"num_frames": 241}],
        source_audio={"audio_id": aid},
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "SOURCE_AUDIO_TOO_SHORT"
    assert chain_math.audio_latents_required([241, 241], 24.0, kv=2) == 493


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


def test_a2v_multi_clip_accepted(client, tmp_path):
    """Long A2V: 2 clips + one source audio is ACCEPTED (the old "exactly 1 clip
    in v1" guard is gone). [121,121]@24fps kv=2 -> a_total=243 (~9.72s)."""
    wav = _make_wav(tmp_path / "voice2.wav", seconds=10.0)
    aid = _upload_audio(client, wav)

    r = _run_chain(
        client, [{"num_frames": 121}, {"num_frames": 121}],
        source_audio={"audio_id": aid},
    )
    assert r.status_code == 202, r.text
    assert r.json()["num_clips"] == 2


def test_a2v_multi_clip_mock_e2e(client, tmp_path):
    """3 clips driven by ONE uploaded audio run to completion, and the metadata
    a2v block reports the geometry of the WHOLE assembled timeline (not of one
    clip): [121,121,121]@24fps kv=2 -> 345 pixel frames -> a_total=359 (14.36s)."""
    layout = chain_math.compute_chain_layout([121, 121, 121], 24.0, kv=2)
    assert (layout.total_px, layout.a_total) == (345, 359)

    wav = _make_wav(tmp_path / "long.wav", seconds=14.4, sr=16000, channels=1)
    aid = _upload_audio(client, wav)

    clips = [{"num_frames": 121}, {"num_frames": 121}, {"num_frames": 121}]
    r = _run_chain(client, clips, source_audio={"audio_id": aid})
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["num_clips"] == 3
    job_id = body["job_id"]

    job = client.get(f"/api/v1/jobs/{job_id}").json()
    assert job["status"] == "completed", job

    ctx = client.app_context
    out = ctx.config.output_dir / job_id / "output.mp4"
    assert out.exists() and out.stat().st_size > 0

    meta = json.loads((ctx.config.output_dir / job_id / "metadata.json").read_text(encoding="utf-8"))
    a2v = meta["a2v"]
    assert a2v["a_total"] == layout.a_total
    assert a2v["encoded_audio_frames_available"] >= a2v["a_total"]
    assert a2v["muxed_original_waveform"] is True
    assert a2v["vocoder_skipped"] is True
    assert a2v["source_audio_id"] == aid
    # the request echo keeps all three clips AND the single source audio
    assert len(meta["request"]["clips"]) == 3
    assert meta["request"]["source_audio"]["audio_id"] == aid


def test_a2v_multi_clip_with_start_image_mock_e2e(client, tmp_path, png_bytes):
    """Start-frame image + source audio + several clips combine (none of the
    three features blocks the others). Pins that the A2V unblock did not create
    a new conflict with clip-0 conditioning."""
    up = client.post("/api/v1/upload/image", files={"file": ("k.png", png_bytes, "image/png")})
    assert up.status_code == 200, up.text
    image_id = up.json()["image_id"]

    wav = _make_wav(tmp_path / "voice3.wav", seconds=10.0)
    aid = _upload_audio(client, wav)

    clips = [
        {
            "num_frames": 121,
            "conditioning_images": [{"image_id": image_id, "frame_idx": 0, "strength": 0.8}],
        },
        {"num_frames": 121},
    ]
    r = _run_chain(client, clips, source_audio={"audio_id": aid})
    assert r.status_code == 202, r.text
    job = client.get(f"/api/v1/jobs/{r.json()['job_id']}").json()
    assert job["status"] == "completed", job


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


# ------------------------------------------------- (e2) full_length window §1-19


def test_a2v_full_length_window_mock_e2e(client, tmp_path):
    """The a2v window: ``stage2_window="full_length"`` (61/61 -> kt_v 0) turns a
    481-frame single-clip A2V chain into ONE stage-2 tile over the whole
    timeline — no tile seam at all, i.e. the same refine pass plain
    ``POST /generate`` does. 481f @24fps -> 61 latent frames -> a_total=501
    (20.04s), so the uploaded audio must be at least that long."""
    layout = chain_math.compute_chain_layout(
        [481], 24.0, kv=2, v_tile=61, v_adv=61
    )
    assert (layout.f_total, layout.total_px, layout.n_tiles) == (61, 481, 1)

    wav = _make_wav(tmp_path / "full.wav", seconds=21.0, sr=16000, channels=1)
    aid = _upload_audio(client, wav)

    r = _run_chain(
        client, [{"num_frames": 481}],
        source_audio={"audio_id": aid}, stage2_window="full_length",
    )
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    job = client.get(f"/api/v1/jobs/{job_id}").json()
    assert job["status"] == "completed", job

    ctx = client.app_context
    meta = json.loads(
        (ctx.config.output_dir / job_id / "metadata.json").read_text(encoding="utf-8")
    )
    chain_meta = meta["chain"]
    assert chain_meta["stage2_window"] == "full_length"
    assert chain_meta["v_tile"] == 61
    assert chain_meta["kt_v"] == 0
    assert chain_meta["n_tiles"] == 1
    assert chain_meta["video_tiles"] == [[0, 61]]
    # The whole point: a seamless timeline.
    assert chain_meta["tile_seam_junctions"] == []
    assert meta["a2v"]["a_total"] == layout.a_total


def test_a2v_default_window_is_still_the_tiled_one(client, tmp_path):
    """Non-regression baseline for the above: an A2V request that does NOT send
    ``stage2_window`` keeps the frozen 22/18 tiled window (the new preset is
    opt-in on the wire, even though the WebUI/Gradio now always send it)."""
    wav = _make_wav(tmp_path / "plain.wav", seconds=21.0, sr=16000, channels=1)
    aid = _upload_audio(client, wav)

    r = _run_chain(client, [{"num_frames": 481}], source_audio={"audio_id": aid})
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    assert client.get(f"/api/v1/jobs/{job_id}").json()["status"] == "completed"

    ctx = client.app_context
    meta = json.loads(
        (ctx.config.output_dir / job_id / "metadata.json").read_text(encoding="utf-8")
    )
    assert meta["chain"]["stage2_window"] == "standard"
    assert meta["chain"]["v_tile"] == 22
    assert meta["chain"]["kt_v"] == 4
    assert meta["chain"]["n_tiles"] > 1


# ---------------------------------------------------------------- (f) geometry


def test_a2v_geometry_matches_audio_latents_required():
    """The preflight length gate and the engine both source a_total from the SAME
    pure function — pin that identity so app/engine never drift."""
    layout = chain_math.compute_chain_layout([49], 24.0, kv=2)
    assert chain_math.audio_latents_required([49], 24.0, kv=2) == layout.a_total


def test_a2v_segment_windows_tile_the_global_timeline():
    """Long A2V's whole geometry, spelled out with numbers so the front end's TS
    mirror has a reference implementation to check itself against.

    Worked example — clips [121, 121, 121] @ 24 fps, K_v = 2:

        seg_latent  = [16, 16, 16]        (121 -> (121-1)//8 + 1)
        f_total     = 16*3 - 2*2 = 44  -> total_px = 43*8 + 1 = 345
        a_total     = round(345/24 * 25) = 359
        seg_audio   = [126, 126, 126]     (round(121/24 * 25))
        ka_list     = [10, 9]             (sum(seg_audio) - a_total = 19, split)
        windows     = [(0, 126), (116, 126), (233, 126)]

    Every window is a slice of the ONE uploaded audio latent; the invariants
    below are what make that slicing a tiling: it starts at 0, it ends exactly on
    a_total, and consecutive windows overlap by exactly the same per-join K_a the
    assembler crossfades with.
    """
    cases = [
        ([121, 121, 121], 24.0, 2, [(0, 126), (116, 126), (233, 126)]),
        ([49, 49], 24.0, 2, [(0, 51), (42, 51)]),
        ([49, 73, 121], 24.0, 2, [(0, 51), (41, 76), (108, 126)]),   # unequal clips
        ([121, 121], 30.0, 3, [(0, 101), (87, 101)]),                # non-24fps
        ([121], 24.0, 2, [(0, 126)]),                                # degenerate: 1 clip
    ]
    for clip_frames, fps, kv, expected in cases:
        layout = chain_math.compute_chain_layout(clip_frames, fps, kv=kv)
        windows = chain_math.audio_segment_windows(layout)
        assert len(windows) == len(clip_frames)
        # (a) the first window starts at the head of the global timeline
        assert windows[0][0] == 0
        # (b) the last window ends exactly on a_total (no gap, no overrun)
        assert windows[-1][0] + windows[-1][1] == layout.a_total
        # (c) consecutive windows overlap by exactly ka_list[i]
        for i in range(len(windows) - 1):
            overlap = (windows[i][0] + windows[i][1]) - windows[i + 1][0]
            assert overlap == layout.ka_list[i], (clip_frames, fps, kv, i, overlap)
        # every window is a valid slice of the a_total-long uploaded latent
        for start, length in windows:
            assert 0 <= start and start + length <= layout.a_total
        if clip_frames == [121, 121, 121] and fps == 24.0 and kv == 2:
            assert layout.seg_audio == [126, 126, 126]
            assert layout.ka_list == [10, 9]
        assert windows == expected


def test_a2v_multi_clip_freezes_the_audio_window_but_not_the_video_seam():
    """The long-A2V engine fix, at the granularity that is testable without GPU.

    A stage-1 A2V segment calls ``_denoise_av_with_carry`` with
    ``mask_value = 1 - overlap_strength`` and ``audio_mask_value = 0.0``. From
    clip 2 onward the video head freeze is non-zero, so the two must NOT share
    one value: the audio window is hard-frozen while the video seam keeps
    carrying over at the user's overlap_strength. Before the fix both were 0.0,
    which welded every seam shut and discarded overlap_strength silently."""
    stage1_mask_value = 1.0 - 0.5          # overlap_strength = 0.5
    v_head, v_tail, a_head, a_tail = chain_math.freeze_mask_values(
        stage1_mask_value, None, 0.0
    )
    assert (v_head, v_tail) == (0.5, 0.5)  # seam still blends
    assert (a_head, a_tail) == (0.0, 0.0)  # uploaded audio hard-frozen

    # No override -> one value everywhere: every pre-A2V call site unchanged.
    assert chain_math.freeze_mask_values(0.5) == (0.5, 0.5, 0.5, 0.5)
    # retake's two-sided split is unaffected by the new modality split.
    assert chain_math.freeze_mask_values(0.3, 0.7) == (0.3, 0.7, 0.3, 0.7)
    assert chain_math.freeze_mask_values(0.0, None, None) == (0.0, 0.0, 0.0, 0.0)


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
