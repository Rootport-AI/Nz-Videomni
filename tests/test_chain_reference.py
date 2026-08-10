"""Reference-video CONTROL IC-LoRA on a chain (Phase C chain support; 1..24
clips — owner decision 2026-08-11 lifted the old 2026-07-11 "exactly 1 clip"
ALPHA scope): a ``GenerateChainRequest`` may now carry ``reference_video_id`` /
``conditioning_attention_strength`` / ``reference_video_strength`` exactly like
a single ``/generate``, on ANY clip count. A long reference video is auto-sliced
per stage-1 segment server-side (``chain_math.video_segment_windows``, tested in
test_chain_math_reference.py); this API-layer test suite only checks that the
endpoint no longer rejects >1 clip. Mutually exclusive with ``source_video``.
The one remaining clip-count restriction is a depth-preprocess control adapter,
still rejected on >1 clip (422 ``LORA_DEPTH_CHAIN_UNSUPPORTED`` — the depth
preprocessor is a whole-clip design that cannot process a chain-length
reference).

Coverage, mirroring tests/test_chain_lora.py's structure:
  (a) API branch coverage — schema (422, pydantic-level) vs endpoint (404/422,
      APIError-level) rejections, and the clips=1+control+reference happy path,
      PLUS the multi-clip acceptance (canny/pose) and multi-clip depth rejection;
  (b) real-backend payload — ``_RealBackend.generate_chain``'s additive
      ``reference_video`` block (present only when a reference video was
      requested; ``attention_strength`` present only when set);
  (c) ``chain_pipeline.run_chain`` plumbing — MOVED to
      tests/test_chain_reference_engine.py, the venv that can actually run it
      (see the marker below where it used to live);
  (d) mock e2e — A2V + control adapter + reference video (clips=1) completes and
      the recorded metadata carries ``reference_video_id`` + ``loras``.

The mock backend (fixture ``chain_ref_client``) has no weights, so the forward-
time patch itself is not exercised here (see test_ic_lora_forward.py); the real
weight effect is an owner-witnessed GPU check per project rule.
"""

from __future__ import annotations

import argparse
import json
import struct
import threading
import types
import wave
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

import main
from api.models import GenerateChainRequest
from services.ltx_runner import _RealBackend

STYLE_LORA = "style-adapter"
CANNY_LORA = "canny-control"
DEPTH_LORA = "depth-control"

# A tiny but non-empty mp4-ish blob (the video store validates extension/size
# only; the mock backend never opens it).
FAKE_MP4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64


def _make_args(config_path: str) -> argparse.Namespace:
    return argparse.Namespace(
        listen=False, port=None, api_key=None, allow_all_cors=False,
        config=config_path, te_offload=None, dit_cpu_load=None,
    )


@pytest.fixture()
def chain_ref_client(tmp_path):
    """A mock-backend client registering one STYLE adapter and one CONTROL
    adapter (dict entry w/ preprocess canny) — same registry shape as
    tests/test_chain_lora.py's ``chain_lora_client``."""
    style_file = tmp_path / "style.safetensors"
    style_file.write_bytes(b"\x00" * 8)
    control_file = tmp_path / "union-control.safetensors"
    control_file.write_bytes(b"\x00" * 8)

    cfg = {
        "server": {"log_dir": (tmp_path / "logs").as_posix()},
        "model": {
            "backend": "mock",
            "ic_loras": {
                STYLE_LORA: style_file.as_posix(),
                CANNY_LORA: {"path": control_file.as_posix(), "preprocess": "canny"},
                # Same file as the canny entry (one Union-Control adapter, several
                # preprocess kinds) -- only the ``preprocess`` label differs, which
                # is all the depth-chain-unsupported check reads.
                DEPTH_LORA: {"path": control_file.as_posix(), "preprocess": "depth"},
            },
        },
        "output": {"dir": (tmp_path / "outputs").as_posix()},
        "upload": {"dir": (tmp_path / "uploads").as_posix()},
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    app = main.build_app(_make_args(cfg_path.as_posix()))
    with TestClient(app) as c:
        c.app_context = app.state.context  # type: ignore[attr-defined]
        yield c


# ÷128-clean base (384x256): the reference is consumed at half output resolution
# on the 64-grid, so all CONTROL adapters need width/height divisible by 128.
BASE = {
    "prompt": "a bustling town square at dusk, cinematic trailer",
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


def _upload_video(client) -> str:
    r = client.post("/api/v1/upload/video", files={"file": ("ref.mp4", FAKE_MP4, "video/mp4")})
    assert r.status_code == 200, r.text
    return r.json()["video_id"]


def _upload_audio(client, tmp_path, seconds: float = 3.0) -> str:
    wav = tmp_path / "voice.wav"
    n = int(round(seconds * 16000))
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(struct.pack("<%dh" % n, *([0] * n)))
    r = client.post(
        "/api/v1/upload/audio",
        files={"file": ("voice.wav", wav.read_bytes(), "audio/wav")},
    )
    assert r.status_code == 200, r.text
    return r.json()["audio_id"]


# ------------------------------------------------------------------ (a) branch
# coverage: clips=1+control+reference / missing reference / ÷128 / schema 422s /
# unknown reference id.
# ------------------------------------------------------------------ #


def test_clips1_control_with_reference_accepted(chain_ref_client):
    vid = _upload_video(chain_ref_client)
    r = _run_chain(
        chain_ref_client, [{"num_frames": 25}],
        loras=[{"name": CANNY_LORA, "strength": 1.0}],
        reference_video_id=vid,
    )
    assert r.status_code == 202, r.text
    job = chain_ref_client.get(f"/api/v1/jobs/{r.json()['job_id']}").json()
    assert job["status"] == "completed", job


def test_clips1_control_without_reference_422_requires_reference(chain_ref_client, tmp_path):
    """A CONTROL adapter on a 1-clip chain WITHOUT a reference video is rejected
    (LORA_REQUIRES_REFERENCE) — the alpha counterpart to
    test_chain_lora.test_chain_control_lora_rejected_422 (clips>=2, no reference
    possible at all). A bare 1-clip chain without source_video/source_audio/
    reference_video_id fails the clip-count FLOOR first (schema-level), so this
    uses source_audio (A2V) to reach exactly 1 clip legitimately -- the same
    "A2V + control adapter, no reference video" combo the Gradio handler
    prechecks (see test_gradio_handlers.py's
    test_generate_a2v_conflict_with_adapter_without_reference_zero_calls)."""
    aid = _upload_audio(chain_ref_client, tmp_path)
    r = _run_chain(
        chain_ref_client, [{"num_frames": 49}],
        source_audio={"audio_id": aid},
        loras=[{"name": CANNY_LORA, "strength": 1.0}],
    )
    assert r.status_code == 422, r.text
    assert r.json()["error"]["code"] == "LORA_REQUIRES_REFERENCE"


def test_reference_resolution_not_divisible_by_128_422(chain_ref_client):
    """384x320: width is ÷128-clean but 320 % 128 == 64 -> rejected up front
    (REFERENCE_RESOLUTION_INVALID) rather than crashing the worker's VAE encode.
    320 % 64 == 0 so the (unrelated) ÷64 precheck still passes."""
    vid = _upload_video(chain_ref_client)
    r = _run_chain(
        chain_ref_client, [{"num_frames": 25}],
        height=320,
        loras=[{"name": CANNY_LORA, "strength": 1.0}],
        reference_video_id=vid,
    )
    assert r.status_code == 422, r.text
    assert r.json()["error"]["code"] == "REFERENCE_RESOLUTION_INVALID"


def test_reference_with_style_only_loras_422(chain_ref_client):
    """The chain counterpart of test_ic_lora_api.test_reference_with_style_only_loras_422:
    a 1-clip chain carrying a reference video but only STYLE adapters has nothing
    to consume the reference (the downscale factor comes from a CONTROL adapter's
    metadata) -> 422 REFERENCE_REQUIRES_CONTROL_LORA."""
    vid = _upload_video(chain_ref_client)
    r = _run_chain(
        chain_ref_client, [{"num_frames": 25}],
        loras=[{"name": STYLE_LORA, "strength": 1.0}],
        reference_video_id=vid,
    )
    assert r.status_code == 422, r.text
    assert r.json()["error"]["code"] == "REFERENCE_REQUIRES_CONTROL_LORA"
    assert STYLE_LORA in r.json()["error"]["detail"]


def test_reference_with_style_plus_control_accepted(chain_ref_client):
    """STYLE + CONTROL together on a 1-clip chain still passes -- only
    style-ONLY is rejected."""
    vid = _upload_video(chain_ref_client)
    r = _run_chain(
        chain_ref_client, [{"num_frames": 25}],
        loras=[
            {"name": STYLE_LORA, "strength": 1.0},
            {"name": CANNY_LORA, "strength": 1.0},
        ],
        reference_video_id=vid,
    )
    assert r.status_code == 202, r.text
    job = chain_ref_client.get(f"/api/v1/jobs/{r.json()['job_id']}").json()
    assert job["status"] == "completed", job


def test_reference_with_empty_loras_422_schema(chain_ref_client):
    vid = _upload_video(chain_ref_client)
    r = _run_chain(
        chain_ref_client, [{"num_frames": 25}],
        reference_video_id=vid,
    )
    assert r.status_code == 422, r.text
    assert "requires at least one lora" in r.text


def test_reference_with_two_clips_accepted(chain_ref_client):
    """reference_video_id is now accepted on any clip count (owner decision
    2026-08-11 lifted the old "exactly 1 clip" ALPHA scope) -- a 2-clip chain
    carrying a non-depth control adapter + reference completes."""
    vid = _upload_video(chain_ref_client)
    r = _run_chain(
        chain_ref_client, [{"num_frames": 25}, {"num_frames": 25}],
        loras=[{"name": CANNY_LORA, "strength": 1.0}],
        reference_video_id=vid,
    )
    assert r.status_code == 202, r.text
    job = chain_ref_client.get(f"/api/v1/jobs/{r.json()['job_id']}").json()
    assert job["status"] == "completed", job


def test_reference_with_many_clips_accepted(chain_ref_client):
    """A longer chain (4 clips) with a control adapter + reference also
    completes -- not just the 2-clip boundary case above."""
    vid = _upload_video(chain_ref_client)
    r = _run_chain(
        chain_ref_client,
        [{"num_frames": 25} for _ in range(4)],
        loras=[{"name": CANNY_LORA, "strength": 1.0}],
        reference_video_id=vid,
    )
    assert r.status_code == 202, r.text
    job = chain_ref_client.get(f"/api/v1/jobs/{r.json()['job_id']}").json()
    assert job["status"] == "completed", job


def test_depth_reference_with_two_clips_422(chain_ref_client):
    """The one remaining clip-count restriction: a depth-preprocess control
    adapter is still rejected on a >1-clip chain (owner decision 2026-08-11 —
    the depth preprocessor cannot process a chain-length reference)."""
    vid = _upload_video(chain_ref_client)
    r = _run_chain(
        chain_ref_client, [{"num_frames": 25}, {"num_frames": 25}],
        loras=[{"name": DEPTH_LORA, "strength": 1.0}],
        reference_video_id=vid,
    )
    assert r.status_code == 422, r.text
    assert r.json()["error"]["code"] == "LORA_DEPTH_CHAIN_UNSUPPORTED"


def test_depth_reference_with_one_clip_accepted(chain_ref_client):
    """A depth-preprocess control adapter is still fine on a 1-clip chain (the
    pre-existing single-clip alpha behaviour is unchanged)."""
    vid = _upload_video(chain_ref_client)
    r = _run_chain(
        chain_ref_client, [{"num_frames": 25}],
        loras=[{"name": DEPTH_LORA, "strength": 1.0}],
        reference_video_id=vid,
    )
    assert r.status_code == 202, r.text
    job = chain_ref_client.get(f"/api/v1/jobs/{r.json()['job_id']}").json()
    assert job["status"] == "completed", job


def test_reference_with_source_video_422_schema(chain_ref_client):
    """reference_video_id and source_video are mutually exclusive (both drive
    clip-0's head)."""
    ref_vid = _upload_video(chain_ref_client)
    src_vid = _upload_video(chain_ref_client)
    r = _run_chain(
        chain_ref_client, [{"num_frames": 81}],
        loras=[{"name": CANNY_LORA, "strength": 1.0}],
        reference_video_id=ref_vid,
        source_video={"video_id": src_vid, "context_frames": 73},
    )
    assert r.status_code == 422, r.text
    assert "mutually exclusive" in r.text


def test_unknown_reference_video_404(chain_ref_client):
    r = _run_chain(
        chain_ref_client, [{"num_frames": 25}],
        loras=[{"name": CANNY_LORA, "strength": 1.0}],
        reference_video_id="does-not-exist",
    )
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "REFERENCE_VIDEO_NOT_FOUND"


def test_strength_fields_with_empty_loras_422_schema(chain_ref_client):
    r = _run_chain(
        chain_ref_client, [{"num_frames": 25}, {"num_frames": 25}],
        conditioning_attention_strength=0.5,
    )
    assert r.status_code == 422, r.text
    assert "conditioning_attention_strength requires at least one lora" in r.text


# --------------------------------------------------------- (b) payload (real backend)


def _capturing_backend(captured: list[dict]) -> _RealBackend:
    be = _RealBackend.__new__(_RealBackend)
    be._proc = types.SimpleNamespace(poll=lambda: None)  # type: ignore[attr-defined]
    be._lock = threading.Lock()

    def _send(msg: dict) -> None:
        captured.append(msg)
        out = Path(msg["output_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"\x00" * 16)

    be._send = _send  # type: ignore[attr-defined]
    be._read_chain_events = (  # type: ignore[attr-defined]
        lambda cb: {"event": "done", "seed_used": 7, "peak_vram_mb": 100, "chain": {}}
    )
    return be


def _chain_kwargs(**over):
    base = dict(
        prompt="p", width=384, height=256,
        clips=[{"num_frames": 25}, {"num_frames": 25}],
    )
    base.update(over)
    return base


def _chain_request(**over) -> GenerateChainRequest:
    return GenerateChainRequest(**_chain_kwargs(**over))


def test_chain_payload_carries_reference_when_set(tmp_path):
    captured: list[dict] = []
    be = _capturing_backend(captured)
    req = _chain_request(
        clips=[{"num_frames": 25}],
        loras=[{"name": CANNY_LORA, "strength": 0.9}],
        reference_video_id="ref-id",
        reference_video_strength=0.8,
        conditioning_attention_strength=0.6,
    )
    be.generate_chain(
        req, tmp_path / "out",
        lora_paths=[(tmp_path / "union-control.safetensors", 0.9, "canny")],
        reference_video_path=tmp_path / "ref.mp4",
    )
    ref = captured[0]["reference_video"]
    assert ref["path"] == str(tmp_path / "ref.mp4")
    assert ref["strength"] == 0.8
    assert ref["preprocess"] == "canny"
    assert ref["attention_strength"] == 0.6


def test_chain_payload_reference_omits_attention_strength_when_unset(tmp_path):
    """attention_strength is spliced in ONLY when the request carries
    conditioning_attention_strength; strength defaults to 1.0 when
    reference_video_strength is unset."""
    captured: list[dict] = []
    be = _capturing_backend(captured)
    req = _chain_request(
        clips=[{"num_frames": 25}],
        loras=[{"name": CANNY_LORA, "strength": 1.0}],
        reference_video_id="ref-id",
    )
    be.generate_chain(
        req, tmp_path / "out",
        lora_paths=[(tmp_path / "union-control.safetensors", 1.0, "canny")],
        reference_video_path=tmp_path / "ref.mp4",
    )
    ref = captured[0]["reference_video"]
    assert ref["strength"] == 1.0
    assert "attention_strength" not in ref


def test_chain_payload_omits_reference_when_none(tmp_path):
    """No reference_video_path -> the payload's ``reference_video`` key is
    entirely absent, byte-identical to before this feature (mirrors
    test_chain_lora.test_chain_payload_omits_loras_when_empty)."""
    captured: list[dict] = []
    be = _capturing_backend(captured)
    req = _chain_request()  # no loras, no reference
    be.generate_chain(req, tmp_path / "out")
    assert "reference_video" not in captured[0]


# ------------------------------------------------- (c) run_chain plumbing
#
# MOVED to tests/test_chain_reference_engine.py (§1-15 B6). Driving run_chain
# needs torch, which only the ENGINE venv has -- and the engine venv cannot even
# COLLECT this module (no fastapi), so the ltx-stub tests that used to live here
# were skipped in one venv and unreachable in the other, i.e. never actually
# executed (they had rotted: their FakePipe predates ``pipe._set_nag_job``). The
# replacements run for real and cover the new per-clip injection:
#   test_run_chain_injects_reference_once_per_clip           (n clips -> n calls)
#   test_run_chain_stops_injecting_when_reference_runs_out
#   test_run_chain_forwards_reference_to_set_ic_job_before_the_build


# --------------------------------------------------------------- (d) mock e2e


def test_a2v_control_reference_clips1_completes_and_records(chain_ref_client, tmp_path):
    """A2V + a CONTROL adapter + its reference video, clips=1: the whole point
    of the alpha scope. Completes end to end (mock) and the additive
    reference_video_id/loras fields round-trip into the metadata request dump
    (mirrors test_chain_lora.test_chain_style_lora_with_source_audio_completes)."""
    aid = _upload_audio(chain_ref_client, tmp_path)
    vid = _upload_video(chain_ref_client)

    r = _run_chain(
        chain_ref_client, [{"num_frames": 49}],
        source_audio={"audio_id": aid},
        loras=[{"name": CANNY_LORA, "strength": 1.0}],
        reference_video_id=vid,
    )
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    job = chain_ref_client.get(f"/api/v1/jobs/{job_id}").json()
    assert job["status"] == "completed", job

    ctx = chain_ref_client.app_context
    meta = json.loads(
        (ctx.config.output_dir / job_id / "metadata.json").read_text(encoding="utf-8")
    )
    assert meta["request"]["reference_video_id"] == vid
    assert meta["request"]["loras"][0]["name"] == CANNY_LORA
    assert meta["request"]["loras"][0]["strength"] == 1.0


# ------------------------------------------------- (e) §1-15 ic_lora metadata block


def test_metadata_ic_lora_block_absent_without_reference(chain_ref_client):
    """A referenceless chain's metadata carries no ic_lora key at all (byte-
    identical to before this feature -- mirrors _write_metadata's own guard)."""
    r = _run_chain(chain_ref_client, [{"num_frames": 25}, {"num_frames": 25}])
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    assert chain_ref_client.get(f"/api/v1/jobs/{job_id}").json()["status"] == "completed"
    ctx = chain_ref_client.app_context
    meta = json.loads(
        (ctx.config.output_dir / job_id / "metadata.json").read_text(encoding="utf-8")
    )
    assert "ic_lora" not in meta


def test_metadata_ic_lora_block_records_segment_windows(chain_ref_client):
    """A multi-clip chain with a reference records the per-clip
    reference_segment_windows + reference_frames_needed, pinned against a
    direct call to the same chain_math.video_segment_windows the block is
    built from."""
    import chain_math

    vid = _upload_video(chain_ref_client)
    clip_frames = [25, 25, 25]
    r = _run_chain(
        chain_ref_client,
        [{"num_frames": f} for f in clip_frames],
        loras=[{"name": CANNY_LORA, "strength": 1.0}],
        reference_video_id=vid,
        overlap_frames=2,
    )
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    assert chain_ref_client.get(f"/api/v1/jobs/{job_id}").json()["status"] == "completed"

    ctx = chain_ref_client.app_context
    meta = json.loads(
        (ctx.config.output_dir / job_id / "metadata.json").read_text(encoding="utf-8")
    )
    ic_lora = meta["ic_lora"]
    assert ic_lora["reference_video_id"] == vid
    assert ic_lora["loras"][0]["name"] == CANNY_LORA

    layout = chain_math.compute_chain_layout(clip_frames, BASE["frame_rate"], kv=2)
    expected_windows = [list(w) for w in chain_math.video_segment_windows(layout)]
    assert ic_lora["reference_segment_windows"] == expected_windows
    assert len(expected_windows) == 3
    assert ic_lora["reference_frames_needed"] == layout.total_px


def test_metadata_ic_lora_block_records_strength_overrides_when_set(chain_ref_client):
    vid = _upload_video(chain_ref_client)
    r = _run_chain(
        chain_ref_client, [{"num_frames": 25}],
        loras=[{"name": CANNY_LORA, "strength": 1.0}],
        reference_video_id=vid,
        conditioning_attention_strength=0.4,
        reference_video_strength=0.7,
    )
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    assert chain_ref_client.get(f"/api/v1/jobs/{job_id}").json()["status"] == "completed"
    ctx = chain_ref_client.app_context
    meta = json.loads(
        (ctx.config.output_dir / job_id / "metadata.json").read_text(encoding="utf-8")
    )
    ic_lora = meta["ic_lora"]
    assert ic_lora["conditioning_attention_strength"] == 0.4
    assert ic_lora["reference_video_strength"] == 0.7


def test_metadata_ic_lora_omits_strength_overrides_when_unset(chain_ref_client):
    """Mirrors _write_metadata's own precedent: an omitted override field keeps
    the block's key set byte-unchanged rather than writing a null."""
    vid = _upload_video(chain_ref_client)
    r = _run_chain(
        chain_ref_client, [{"num_frames": 25}],
        loras=[{"name": CANNY_LORA, "strength": 1.0}],
        reference_video_id=vid,
    )
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    ctx = chain_ref_client.app_context
    meta = json.loads(
        (ctx.config.output_dir / job_id / "metadata.json").read_text(encoding="utf-8")
    )
    ic_lora = meta["ic_lora"]
    assert "conditioning_attention_strength" not in ic_lora
    assert "reference_video_strength" not in ic_lora


def test_metadata_ic_lora_omits_reference_frames_available_for_an_unprobeable_upload(chain_ref_client):
    """FAKE_MP4 (every other test in this module) is not a real, ffprobe-
    decodable file -- the best-effort frame_count() probe fails, and the
    metadata write must degrade by omitting the key rather than failing the
    (already-completed) job."""
    vid = _upload_video(chain_ref_client)
    r = _run_chain(
        chain_ref_client, [{"num_frames": 25}],
        loras=[{"name": CANNY_LORA, "strength": 1.0}],
        reference_video_id=vid,
    )
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    assert chain_ref_client.get(f"/api/v1/jobs/{job_id}").json()["status"] == "completed"
    ctx = chain_ref_client.app_context
    meta = json.loads(
        (ctx.config.output_dir / job_id / "metadata.json").read_text(encoding="utf-8")
    )
    assert "reference_frames_available" not in meta["ic_lora"]


def test_metadata_ic_lora_records_reference_frames_available_for_a_real_upload(chain_ref_client, tmp_path):
    """With a real, ffprobe-decodable upload, reference_frames_available is
    present and matches the source's actual (short-of-the-timeline) frame
    count -- the number a reviewer cross-checks against
    reference_frames_needed to see whether every clip got its reference."""
    from PIL import Image

    from services import video_io as video_io_mod

    frames = [Image.new("RGB", (64, 64), (i * 40 % 256, 80, 160)) for i in range(9)]
    src = tmp_path / "ref_real.mp4"
    video_io_mod.encode_frames_to_mp4(frames, src, frame_rate=24.0)
    r = chain_ref_client.post(
        "/api/v1/upload/video",
        files={"file": ("ref_real.mp4", src.read_bytes(), "video/mp4")},
    )
    assert r.status_code == 200, r.text
    vid = r.json()["video_id"]

    rr = _run_chain(
        chain_ref_client, [{"num_frames": 25}],
        loras=[{"name": CANNY_LORA, "strength": 1.0}],
        reference_video_id=vid,
    )
    assert rr.status_code == 202, rr.text
    job_id = rr.json()["job_id"]
    assert chain_ref_client.get(f"/api/v1/jobs/{job_id}").json()["status"] == "completed"
    ctx = chain_ref_client.app_context
    meta = json.loads(
        (ctx.config.output_dir / job_id / "metadata.json").read_text(encoding="utf-8")
    )
    assert meta["ic_lora"]["reference_frames_available"] == 9
    assert meta["ic_lora"]["reference_frames_needed"] == 25  # single clip -> total_px == num_frames
