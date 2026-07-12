"""Reference-video CONTROL IC-LoRA on a chain (Phase C chain support, ALPHA
scope — owner decision 2026-07-11): a ``GenerateChainRequest`` may now carry
``reference_video_id`` / ``conditioning_attention_strength`` /
``reference_video_strength`` exactly like a single ``/generate``, but ONLY when
the chain is exactly 1 clip (a per-clip reference video is out of scope, demoted
to a later research item). Mutually exclusive with ``source_video``.

Coverage, mirroring tests/test_chain_lora.py's structure:
  (a) API branch coverage — schema (422, pydantic-level) vs endpoint (404/422,
      APIError-level) rejections, and the clips=1+control+reference happy path;
  (b) real-backend payload — ``_RealBackend.generate_chain``'s additive
      ``reference_video`` block (present only when a reference video was
      requested; ``attention_strength`` present only when set);
  (c) ``chain_pipeline.run_chain`` plumbing — ``_set_ic_job`` receives
      ``ic_reference``/``ic_attention_strength``, and (best-effort) clip-0's
      stage-1 conditioning is built via ``pipe._reference_conditioning_for_stage``
      when a reference is present;
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
import sys
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


def test_reference_with_empty_loras_422_schema(chain_ref_client):
    vid = _upload_video(chain_ref_client)
    r = _run_chain(
        chain_ref_client, [{"num_frames": 25}],
        reference_video_id=vid,
    )
    assert r.status_code == 422, r.text
    assert "requires at least one lora" in r.text


def test_reference_with_two_clips_422_schema(chain_ref_client):
    """reference_video_id is alpha-scoped to exactly 1 clip; a 2-clip chain
    carrying one is rejected at the schema level regardless of loras."""
    vid = _upload_video(chain_ref_client)
    r = _run_chain(
        chain_ref_client, [{"num_frames": 25}, {"num_frames": 25}],
        loras=[{"name": CANNY_LORA, "strength": 1.0}],
        reference_video_id=vid,
    )
    assert r.status_code == 422, r.text
    assert "exactly 1 clip" in r.text


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


_LTX_STUB_MODULES = {
    "ltx_core": {},
    "ltx_core.components": {},
    "ltx_core.model": {},
    "ltx_core.text_encoders": {},
    "ltx_pipelines": {},
    "ltx_pipelines.utils": {},
    "ltx_core.components.diffusion_steps": {"EulerDiffusionStep": object},
    "ltx_core.components.noisers": {"GaussianNoiser": object},
    "ltx_core.model.audio_vae": {
        "decode_audio": lambda *a, **k: None,
        "encode_audio": lambda *a, **k: None,
    },
    "ltx_core.model.upsampler": {"upsample_video": lambda *a, **k: None},
    "ltx_core.model.video_vae": {"decode_video": lambda *a, **k: None},
    "ltx_core.text_encoders.gemma": {
        "encode_text": lambda te, prompts: [(object(), object())]
    },
    "ltx_core.types": {
        "AudioLatentShape": object,
        "VideoLatentShape": object,
        "VideoPixelShape": object,
        "Audio": object,
    },
    "ltx_pipelines.utils.constants": {
        "DISTILLED_SIGMA_VALUES": [1.0],
        "STAGE_2_DISTILLED_SIGMA_VALUES": [1.0],
    },
    "ltx_pipelines.utils.helpers": {"cleanup_memory": lambda *a, **k: None},
}


class _StopBeforeBuild(Exception):
    """Raised by the fake video_encoder() to halt run_chain right after the
    IC-LoRA state is set but before any heavy build."""


def _run_chain_capturing_set_ic_job(monkeypatch, ic_loras, ic_reference=None,
                                     ic_attention_strength=1.0):
    """Drive engine.pipeline.chain_pipeline.run_chain far enough to observe the
    _set_ic_job call ordering, GPU-free, with ltx_core stubbed out. Returns the
    ordered call log. (Mirrors tests/test_chain_lora.py's helper of the same
    name, extended with ic_reference/ic_attention_strength passthrough.)"""
    torch = pytest.importorskip("torch")
    for name, attrs in _LTX_STUB_MODULES.items():
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        monkeypatch.setitem(sys.modules, name, mod)

    import engine.pipeline.chain_pipeline as cp

    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda *a, **k: None)
    monkeypatch.setattr(cp, "default_tiling_config", lambda **k: object())

    calls: list = []

    class FakeLedger:
        def text_encoder(self):
            calls.append("text_encoder")
            return object()

        def video_encoder(self):
            calls.append("video_encoder")
            raise _StopBeforeBuild()

        def transformer(self):
            calls.append("transformer")
            raise _StopBeforeBuild()

    class FakeDP:
        device = "cpu"
        model_ledger = FakeLedger()
        pipeline_components = object()

    class FakePipe:
        pipeline = FakeDP()
        _vae_spatial_tile_size = 0
        _vae_temporal_tile_size = 0

        def _set_ic_job(self, loras, ref, attn):
            calls.append(("set_ic_job", list(loras), ref, attn))

    clips = [
        cp.ChainClipSpec(prompt="a", num_frames=25, images=[]),
        cp.ChainClipSpec(prompt="b", num_frames=25, images=[]),
    ]
    with pytest.raises(_StopBeforeBuild):
        cp.run_chain(
            FakePipe(), clips=clips, width=384, height=256, frame_rate=24.0,
            num_steps=8, seed=1, overlap_frames=2, overlap_strength=0.5,
            output_path="x.mp4", ic_loras=ic_loras,
            ic_reference=ic_reference, ic_attention_strength=ic_attention_strength,
        )
    return calls


def test_run_chain_sets_ic_reference_and_attention_strength(monkeypatch):
    """run_chain forwards ic_reference/ic_attention_strength to _set_ic_job
    (before the transformer is built), alongside the style loras."""
    calls = _run_chain_capturing_set_ic_job(
        monkeypatch, ic_loras=[("/p/style.safetensors", 0.7)],
        ic_reference=("/p/ref.mp4", 1.0), ic_attention_strength=0.8,
    )
    set_calls = [c for c in calls if isinstance(c, tuple) and c[0] == "set_ic_job"]
    assert set_calls == [
        ("set_ic_job", [("/p/style.safetensors", 0.7)], ("/p/ref.mp4", 1.0), 0.8)
    ]
    assert calls.index(set_calls[0]) < calls.index("video_encoder")


class _LenientShape:
    """A permissive stand-in for the stubbed ltx_core VideoPixelShape/
    VideoLatentShape (bare ``object`` cannot be called with constructor args)."""

    def __init__(self, *a, **k):
        pass


class _LenientNoiser:
    """A permissive stand-in for the stubbed GaussianNoiser (bare ``object``
    cannot be called with keyword args)."""

    def __init__(self, *a, **k):
        pass


def _run_chain_capturing_stage1_reference(monkeypatch, *, num_clips=1):
    """Drive run_chain past the video_encoder/transformer build (unlike
    ``_run_chain_capturing_set_ic_job``, which stops there) into the stage-1
    per-segment loop, halting exactly at clip-0's ic_reference conditioning
    build via a FakePipe._reference_conditioning_for_stage that raises to stop.
    Returns the ordered call log."""
    torch = pytest.importorskip("torch")
    stub_modules = dict(_LTX_STUB_MODULES)
    stub_modules["ltx_core.types"] = {
        "AudioLatentShape": _LenientShape,
        "VideoLatentShape": _LenientShape,
        "VideoPixelShape": _LenientShape,
        "Audio": object,
    }
    stub_modules["ltx_core.components.noisers"] = {"GaussianNoiser": _LenientNoiser}
    # _build_video_conditionings imports these at module-import time even when
    # ``images`` ends up empty (the early-return happens AFTER the imports), so
    # they must resolve even though this test never actually builds a keyframe
    # conditioning.
    stub_modules["ltx_pipelines.utils.args"] = {"ImageConditioningInput": object}
    stub_modules["ltx_pipelines.utils.helpers"] = {
        "cleanup_memory": lambda *a, **k: None,
        "image_conditionings_by_adding_guiding_latent": lambda *a, **k: [],
        "image_conditionings_by_replacing_latent": lambda *a, **k: [],
    }
    for name, attrs in stub_modules.items():
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        monkeypatch.setitem(sys.modules, name, mod)

    import engine.pipeline.chain_pipeline as cp

    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda *a, **k: None)
    monkeypatch.setattr(cp, "default_tiling_config", lambda **k: object())

    calls: list = []

    class _StopAtReference(Exception):
        pass

    class FakeLedger:
        def text_encoder(self):
            calls.append("text_encoder")
            return object()

        def video_encoder(self):
            calls.append("video_encoder")
            return object()

        def transformer(self):
            calls.append("transformer")
            return object()

    class FakeDP:
        device = "cpu"
        model_ledger = FakeLedger()
        pipeline_components = object()

    class FakePipe:
        pipeline = FakeDP()
        _vae_spatial_tile_size = 0
        _vae_temporal_tile_size = 0

        def _set_ic_job(self, loras, ref, attn):
            calls.append(("set_ic_job", list(loras), ref, attn))

        def _reference_conditioning_for_stage(self, **kwargs):
            calls.append("reference_conditioning_for_stage")
            raise _StopAtReference()

    clips = [
        cp.ChainClipSpec(prompt=f"clip{i}", num_frames=25, images=[])
        for i in range(num_clips)
    ]
    with pytest.raises(_StopAtReference):
        cp.run_chain(
            FakePipe(), clips=clips, width=384, height=256, frame_rate=24.0,
            num_steps=8, seed=1, overlap_frames=2, overlap_strength=0.5,
            output_path="x.mp4", ic_loras=[],
            ic_reference=("/p/ref.mp4", 1.0), ic_attention_strength=1.0,
        )
    return calls


def test_run_chain_injects_reference_conditioning_once_at_clip0(monkeypatch):
    """When ic_reference is set, clip-0's stage-1 conditioning is built via
    pipe._reference_conditioning_for_stage exactly once (the single-generate
    path's "stage-1 only" semantics, reused verbatim for the chain's clip-0)."""
    calls = _run_chain_capturing_stage1_reference(monkeypatch, num_clips=1)
    assert calls.count("reference_conditioning_for_stage") == 1
    # ordering: video_encoder/transformer are built before the stage-1 loop
    # reaches the reference-conditioning call.
    assert calls.index("video_encoder") < calls.index("reference_conditioning_for_stage")
    assert calls.index("transformer") < calls.index("reference_conditioning_for_stage")


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
