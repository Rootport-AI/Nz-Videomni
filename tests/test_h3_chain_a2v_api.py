"""MiniMax H3 A2V per chain clip (``clips[i].source_audio``, 台帳 §1-120).

Pins the app half in one place: the schema (``ChainClipSourceAudioSpec``: the
single spec plus ``offset_frames``; exclusive with ``source_video`` and
``retake``, accepted with ``references``), the engine scope (H3 still refuses
the top-level ``source_audio`` as ``a2v``; both LTX families refuse the clip
field as ``source_audio`` with the chain's sentence), the 404 naming
``clips[i].source_audio``, the preflight (no stream / silent / too short, the
single A2V's codes, the span from ``clip_audio_start_sample(offset_frames)``,
each id measured once), the orchestration and the mock (``metadata.a2v.clips``,
engine keys and app keys disjoint), the real worker payload (the clip key
sits last; the top-level keys are the golden) and the published constraint.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

import h3_chain_math
from api.models import ChainClip, ChainClipSourceAudioSpec, GenerateChainRequest
from services import engines
from services.engines.h3 import adapter as h3
from test_engine_h3_adapter import (
    CHAIN_DONE_EVENT,
    GOLDEN_CHAIN_KEYS_H3,
    PROMPT,
    _activate,
    _capturing_backend,
    _metadata,
    _validated_chain,
)
from test_h3_a2v_api import _upload as _upload_audio

CHAIN_BODY = {"prompt": PROMPT, "width": 640, "height": 384}
#: Three 124-frame clips: S_1 = 102 (a multiple of 3), S_2 = 204.
FRAMES = [124, 124, 124]
STARTS = [0, 102, 204]

#: What the engine (here the mock) reports per clip, and what the app adds.
ENGINE_KEYS = {
    "mask_value", "audio_latents", "fixed_head_latents", "start_sample", "required_samples",
    "encoded_samples", "channels_used", "resampler", "encoder_tensors", "audio_output", "mock",
}
APP_KEYS = {"source_audio_id", "strength", "offset_frames", "source"}

CHAIN_DETAIL = "クリップごとの A2V は MiniMax H3 専用です。LTX の A2V は Chained の音声で使えます。"


def _clips(*audios, frames=FRAMES) -> list[dict]:
    """One clip per frame count; ``audios[i]`` (a dict or None) is clip i's source_audio."""
    out = []
    for i, f in enumerate(frames):
        clip: dict = {"num_frames": f}
        if i < len(audios) and audios[i] is not None:
            clip["source_audio"] = audios[i]
        out.append(clip)
    return out


def _post_chain(client, clips, **extra):
    return client.post("/api/v1/generate/chain", json={**CHAIN_BODY, "clips": clips, **extra})


def _job_count(client) -> int:
    return len(client.get("/api/v1/jobs").json())


def _spy_runner(client) -> list[dict]:
    pm = client.app_context.pipeline_manager
    orig = pm.runner.generate_chain
    calls: list[dict] = []

    def wrapped(*a, **k):
        calls.append(dict(k))
        return orig(*a, **k)

    pm.runner.generate_chain = wrapped  # type: ignore[method-assign]
    return calls


def _count_measures(client) -> list[str]:
    """Wrap the shared measurement so a test can count it (one entry per call)."""
    pm = client.app_context.pipeline_manager
    orig = pm._measure_h3_source_audio
    seen: list[str] = []

    def wrapped(audio_id, where=""):
        seen.append(audio_id)
        return orig(audio_id, where)

    pm._measure_h3_source_audio = wrapped  # type: ignore[method-assign]
    return seen


# --------------------------------------------------------------------------- #
# the schema
# --------------------------------------------------------------------------- #


def test_the_defaults():
    chain = GenerateChainRequest(prompt=PROMPT, clips=_clips())
    assert all(c.source_audio is None for c in chain.clips)
    spec = ChainClipSourceAudioSpec(audio_id="a")
    assert (spec.strength, spec.offset_frames) == (0.5, 0)
    assert "source_audio" in ChainClip.model_fields


@pytest.mark.parametrize(
    "audio",
    [
        {"audio_id": "a", "strength": 1.5},
        {"audio_id": "a", "strength": -0.1},
        {"audio_id": "a", "offset_frames": -1},
        {"audio_id": ""},
    ],
)
def test_out_of_range_values_are_refused(audio):
    with pytest.raises(ValidationError):
        GenerateChainRequest(prompt=PROMPT, clips=_clips(None, audio))


def test_with_an_input_video_or_a_retake_it_is_refused():
    with pytest.raises(ValidationError, match=r"clips\[1\]\.source_audio and source_video are mutually exclusive"):
        GenerateChainRequest(
            prompt=PROMPT, clips=_clips(None, {"audio_id": "a"}), source_video={"video_id": "v"},
        )
    with pytest.raises(ValidationError, match=r"clips\[0\]\.source_audio and retake are mutually exclusive"):
        GenerateChainRequest(
            prompt=PROMPT, clips=_clips({"audio_id": "a"}, frames=[124]),
            retake={"video_id": "v", "window_start_sec": 0.0},
        )


def test_with_references_on_the_same_clip_it_is_accepted():
    clips = _clips(None, {"audio_id": "a", "offset_frames": 102})
    clips[1]["references"] = [{"kind": "picture", "id": "img", "slot": 1}]
    chain = _validated_chain(clips=clips)
    assert chain.clips[1].source_audio.offset_frames == 102 and chain.clips[1].references


def test_to_clip_request_does_not_copy_it():
    chain = GenerateChainRequest(prompt=PROMPT, clips=_clips({"audio_id": "a"}))
    assert chain.to_clip_request(0).source_audio is None


def test_the_audio_sample_rate_constants_agree():
    from fractions import Fraction

    assert h3_chain_math.H3_AUDIO_SAMPLE_RATE == 32000
    assert h3_chain_math.H3_AUDIO_SAMPLES_PER_FRAME == Fraction(h3_chain_math.H3_AUDIO_SAMPLE_RATE, 24)
    assert h3_chain_math.H3_AUDIO_SAMPLES_PER_LATENT * 40 == h3_chain_math.H3_AUDIO_SAMPLE_RATE


def test_the_start_sample_is_on_an_audio_latent_boundary():
    assert h3_chain_math.clip_audio_start_sample(0) == 0
    assert h3_chain_math.clip_audio_start_sample(102) == 170 * 800
    # Not a multiple of 3: round(221 * 5/3) = round(368.33) = 368.
    assert h3_chain_math.clip_audio_start_sample(221) == 368 * 800
    assert h3_chain_math.clip_audio_start_sample(1) == 2 * 800
    for bad in (-1, True, 1.0):
        with pytest.raises(ValueError):
            h3_chain_math.clip_audio_start_sample(bad)


# --------------------------------------------------------------------------- #
# the engine scope, the constraint and the 404
# --------------------------------------------------------------------------- #


def test_h3_still_refuses_the_top_level_source_audio_as_a2v(three_family_client):
    _activate(three_family_client, "MiniMaxH3")
    before = _job_count(three_family_client)
    r = _post_chain(three_family_client, _clips(), source_audio={"audio_id": "nope"})
    assert r.status_code == 422, r.text
    error = r.json()["error"]
    assert error["code"] == "FEATURE_UNSUPPORTED" and "'a2v'" in error["message"]
    assert _job_count(three_family_client) == before
    assert ("source_audio", "a2v") == h3.CHAIN_REJECT_TABLE[0][:2]
    assert "a2v" in h3.UNSUPPORTED_FEATURES


@pytest.mark.parametrize("base_model", ["LTX23", "LTX25"])
def test_ltx_refuses_a_clips_source_audio_with_the_chain_sentence(three_family_client, base_model):
    from services.engines.ltx import adapter as ltx23

    _activate(three_family_client, base_model)
    before = _job_count(three_family_client)
    r = three_family_client.post(
        "/api/v1/generate/chain",
        json={
            "prompt": PROMPT, "width": 384, "height": 256, "overlap_frames": 2,
            "clips": [{"num_frames": 25}, {"num_frames": 25, "source_audio": {"audio_id": "nope"}}],
        },
    )
    assert r.status_code == 422, r.text
    error = r.json()["error"]
    assert error["code"] == "FEATURE_UNSUPPORTED"
    assert "'source_audio'" in error["message"]
    assert error["detail"] == CHAIN_DETAIL == ltx23.H3_ONLY_CHAIN_REFUSALS["source_audio"]
    assert _job_count(three_family_client) == before


def test_the_single_ltx_refusal_keeps_its_sentence():
    from services.engines.ltx import adapter as ltx23

    assert ltx23.H3_ONLY_REFUSALS["source_audio"].startswith("単発の A2V")
    assert {k: v for k, v in ltx23.H3_ONLY_CHAIN_REFUSALS.items() if k != "source_audio"} == {
        k: v for k, v in ltx23.H3_ONLY_REFUSALS.items() if k != "source_audio"
    }


def test_the_published_unsupported_features_are_unchanged():
    """The new LTX chain rows name ``source_audio``, which the single tables
    already publish: ``unsupported_features`` keeps its entries."""
    from services.engines.ltx import adapter as ltx23
    from services.engines.ltx25 import adapter as ltx25

    for adapter in (ltx23, ltx25):
        assert adapter.UNSUPPORTED_FEATURES == tuple(feat for _f, feat, _p in adapter.REJECT_TABLE)
        assert "source_audio" in adapter.UNSUPPORTED_FEATURES
        rows = [row for row in adapter.CHAIN_REJECT_TABLE if row[1] == "source_audio"]
        assert len(rows) == 1 and rows[0][0] == "clips"
        assert adapter.CHAIN_REJECT_TABLE[-1][1] == "source_audio"   # added last


def test_constraints_publish_the_clip_strength_for_h3_only():
    chain = engines.request_constraints("h3")["chain"]
    assert chain["overlap_frames"] == 22
    assert chain["source_audio"] == {"strength": {"min": 0.0, "max": 1.0, "step": 0.05, "default": 0.5}}
    assert chain["source_audio"]["strength"]["default"] == ChainClipSourceAudioSpec.model_fields["strength"].default
    assert "chain" not in engines.request_constraints("ltx")
    assert "chain" not in engines.request_constraints("ltx25")


def test_models_publishes_the_chain_key(three_family_client):
    listing = three_family_client.get("/api/v1/models").json()["base_models"]
    by_id = {b["id"]: b for b in listing}
    assert by_id["MiniMaxH3"]["constraints"]["chain"] == {
        "overlap_frames": 22,
        "source_audio": {"strength": {"min": 0.0, "max": 1.0, "step": 0.05, "default": 0.5}},
    }
    assert "a2v" in by_id["MiniMaxH3"]["unsupported_features"]
    for ltx in ("LTX23", "LTX25"):
        assert "chain" not in by_id[ltx]["constraints"]
        assert "source_audio" in by_id[ltx]["unsupported_features"]


def test_an_unknown_id_is_404_naming_the_clip(three_family_client, tmp_path):
    _activate(three_family_client, "MiniMaxH3")
    audio_id = _upload_audio(three_family_client, tmp_path, 15.0)
    before = _job_count(three_family_client)
    r = _post_chain(three_family_client, _clips({"audio_id": audio_id}, {"audio_id": "nope", "offset_frames": 102}))
    assert r.status_code == 404, r.text
    error = r.json()["error"]
    assert error["code"] == "SOURCE_AUDIO_NOT_FOUND"
    assert error["message"] == "clips[1].source_audio.audio_id not found: nope"
    assert _job_count(three_family_client) == before


def test_the_404s_come_before_any_measurement(three_family_client, tmp_path):
    _activate(three_family_client, "MiniMaxH3")
    short_id = _upload_audio(three_family_client, tmp_path, 0.5)
    seen = _count_measures(three_family_client)
    r = _post_chain(three_family_client, _clips({"audio_id": short_id}, None, {"audio_id": "nope"}))
    assert r.status_code == 404, r.text
    assert seen == []


def test_the_single_404_message_is_unchanged(three_family_client):
    _activate(three_family_client, "MiniMaxH3")
    r = three_family_client.post(
        "/api/v1/generate",
        json={"prompt": PROMPT, "width": 640, "height": 384, "num_frames": 22, "source_audio": {"audio_id": "nope"}},
    )
    assert r.status_code == 404, r.text
    assert r.json()["error"]["message"] == "source_audio.audio_id not found: nope"


# --------------------------------------------------------------------------- #
# the preflight (422, the single A2V's codes)
# --------------------------------------------------------------------------- #


def test_no_audio_stream_is_422_naming_the_clip(three_family_client, tmp_path, monkeypatch):
    from services import pipeline_manager as pm_module

    _activate(three_family_client, "MiniMaxH3")
    audio_id = _upload_audio(three_family_client, tmp_path, 15.0)
    monkeypatch.setattr(pm_module.video_io, "has_audio_stream", lambda path: False)
    before = _job_count(three_family_client)
    r = _post_chain(three_family_client, _clips(None, {"audio_id": audio_id, "offset_frames": 102}))
    assert r.status_code == 422, r.text
    error = r.json()["error"]
    assert error["code"] == "SOURCE_AUDIO_TOO_SHORT"
    assert error["detail"] == (
        f"clips[1].source_audio: アップロードされた音声（{audio_id}）に読み取れる音声の流れがありません。"
    )
    assert _job_count(three_family_client) == before


def test_a_digitally_silent_track_is_422_naming_the_clip(three_family_client, tmp_path):
    import shutil

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not on PATH")
    from test_h3_ref_api import _make_mp4, _upload_mp4_as_audio

    _activate(three_family_client, "MiniMaxH3")
    audio_id = _upload_mp4_as_audio(three_family_client, _make_mp4(tmp_path / "zero.mp4", audio="zero", seconds=6.0))
    r = _post_chain(three_family_client, _clips({"audio_id": audio_id}))
    assert r.status_code == 422, r.text
    error = r.json()["error"]
    assert error["code"] == "SOURCE_AUDIO_SILENT"
    assert error["detail"].startswith("clips[0].source_audio: ") and audio_id in error["detail"]


def test_an_offset_past_the_audio_is_422(three_family_client, tmp_path):
    """9 s covers clip 0 (5.17 s) but not clip 1 from frame 102:
    (136,000 + 165,333) / 32,000 = 9.417 s."""
    _activate(three_family_client, "MiniMaxH3")
    audio_id = _upload_audio(three_family_client, tmp_path, 9.0)
    before = _job_count(three_family_client)
    r = _post_chain(
        three_family_client,
        _clips({"audio_id": audio_id}, {"audio_id": audio_id, "offset_frames": 102}),
    )
    assert r.status_code == 422, r.text
    error = r.json()["error"]
    assert error["code"] == "SOURCE_AUDIO_TOO_SHORT"
    assert error["detail"] == (
        "clips[1].source_audio: 音声が 9.000 秒しかありません。開始 102 フレームから "
        "124 フレーム（24 fps）を使うには、9.417 秒以上の音声が必要です。"
    )
    assert _job_count(three_family_client) == before
    # The same audio with offset 0 on every clip is long enough.
    r = _post_chain(three_family_client, _clips({"audio_id": audio_id}, {"audio_id": audio_id}))
    assert r.status_code == 202, r.text


def _preflight_with(three_family_client, monkeypatch, duration: float | None, clips) -> None:
    pm = three_family_client.app_context.pipeline_manager
    monkeypatch.setattr(pm, "_measure_h3_source_audio", lambda audio_id, where="": {"duration_sec": duration})
    chain = GenerateChainRequest(prompt=PROMPT, clips=clips)
    pm.preflight_chain_clip_source_audio(chain.clips, 24)


def test_the_length_rule_has_1ms_of_slack(three_family_client, monkeypatch):
    from api.errors import APIError

    clips = _clips(None, {"audio_id": "a", "offset_frames": 102})
    needed = (136000 + round(124 * 32000 / 24)) / 32000
    _preflight_with(three_family_client, monkeypatch, needed - 0.0009, clips)   # inside the 1 ms slack: passes
    _preflight_with(three_family_client, monkeypatch, needed, clips)
    with pytest.raises(APIError) as ei:
        _preflight_with(three_family_client, monkeypatch, needed - 0.0011, clips)
    assert ei.value.code == "SOURCE_AUDIO_TOO_SHORT"
    # An unmeasurable duration is left to the engine's guard.
    _preflight_with(three_family_client, monkeypatch, None, clips)


def test_the_length_rule_uses_the_engines_start_sample(three_family_client, monkeypatch):
    """Offset 1 starts at sample 1,600 (latent 2), not at 1,333 (frame 1 in time)."""
    from api.errors import APIError

    clips = _clips({"audio_id": "a", "offset_frames": 1}, frames=[22, 39])
    end = 1600 + round(22 * 32000 / 24)
    _preflight_with(three_family_client, monkeypatch, end / 32000 - 0.0009, clips)
    with pytest.raises(APIError):
        _preflight_with(three_family_client, monkeypatch, (1333 + round(22 * 32000 / 24)) / 32000, clips)


def test_each_id_is_measured_once(three_family_client, tmp_path):
    _activate(three_family_client, "MiniMaxH3")
    a = _upload_audio(three_family_client, tmp_path, 15.0)
    b = _upload_audio(three_family_client, tmp_path, 14.0)
    seen = _count_measures(three_family_client)
    r = _post_chain(
        three_family_client,
        _clips({"audio_id": a}, {"audio_id": a, "offset_frames": 102}, {"audio_id": a, "offset_frames": 204}),
    )
    assert r.status_code == 202, r.text
    assert seen == [a]
    seen.clear()
    r = _post_chain(three_family_client, _clips({"audio_id": a}, {"audio_id": b}, {"audio_id": a}))
    assert r.status_code == 202, r.text
    assert seen == [a, b]


def test_the_single_preflight_sentences_are_unchanged(three_family_client, tmp_path, monkeypatch):
    from services import pipeline_manager as pm_module

    _activate(three_family_client, "MiniMaxH3")
    audio_id = _upload_audio(three_family_client, tmp_path, 2.0)
    monkeypatch.setattr(pm_module.video_io, "has_audio_stream", lambda path: False)
    r = three_family_client.post(
        "/api/v1/generate",
        json={"prompt": PROMPT, "width": 640, "height": 384, "num_frames": 22, "source_audio": {"audio_id": audio_id}},
    )
    assert r.status_code == 422, r.text
    assert r.json()["error"]["detail"] == f"アップロードされた音声（{audio_id}）に読み取れる音声の流れがありません。"


# --------------------------------------------------------------------------- #
# orchestration, the mock and metadata.json
# --------------------------------------------------------------------------- #


def test_mock_chain_records_each_clips_a2v(three_family_client, tmp_path):
    _activate(three_family_client, "MiniMaxH3")
    audio_id = _upload_audio(three_family_client, tmp_path, 15.0)
    calls = _spy_runner(three_family_client)
    clips = _clips(
        {"audio_id": audio_id, "strength": 0.5},
        None,
        {"audio_id": audio_id, "strength": 1.0, "offset_frames": STARTS[2]},
    )
    r = _post_chain(three_family_client, clips)
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    assert three_family_client.get(f"/api/v1/jobs/{job_id}").json()["status"] == "completed"

    path = three_family_client.app_context.audio_upload_store.path_for(audio_id)
    (call,) = calls
    assert call["clip_source_audio_paths"] == [path, None, path]
    assert "clip_reference_paths" not in call

    layout = h3_chain_math.compute_h3_chain_layout(FRAMES)
    meta = _metadata(three_family_client, job_id)
    first, middle, last = meta["a2v"]["clips"]
    assert middle is None
    for block in (first, last):
        assert set(block) == ENGINE_KEYS | APP_KEYS
    assert not ENGINE_KEYS & APP_KEYS
    assert first["mask_value"] == 0.5 and last["mask_value"] == 0.0
    assert first["start_sample"] == 0 and last["start_sample"] == h3_chain_math.clip_audio_start_sample(204)
    assert first["fixed_head_latents"] == 0 and last["fixed_head_latents"] == layout.audio_overlap_latents[1]
    assert first["audio_latents"] == layout.clip_audio_latents[0]
    assert first["required_samples"] == round(124 * 32000 / 24)
    assert first["audio_output"] == "decoded" == last["audio_output"]   # even at 1.0
    assert first["channels_used"] == "mono->stereo" and first["mock"] is True
    assert first["encoded_samples"] is None and first["encoder_tensors"] is None
    assert (first["source_audio_id"], first["strength"], first["offset_frames"]) == (audio_id, 0.5, 0)
    assert (last["strength"], last["offset_frames"]) == (1.0, 204)
    assert first["source"]["duration_sec"] == pytest.approx(15.0, abs=0.01)
    assert (first["source"]["sample_rate"], first["source"]["channels"]) == (44100, 1)
    # The engine's per-clip record reaches metadata.h3.clips too.
    engine_clips = meta["h3"]["clips"]
    assert "a2v" in engine_clips[0] and "a2v" not in engine_clips[1] and "a2v" in engine_clips[2]
    # The stored request keeps every clip's field.
    assert meta["request"]["clips"][2]["source_audio"] == {
        "audio_id": audio_id, "strength": 1.0, "offset_frames": 204,
    }


def test_a_chain_without_a2v_passes_no_extra_keyword(three_family_client):
    _activate(three_family_client, "MiniMaxH3")
    calls = _spy_runner(three_family_client)
    r = _post_chain(three_family_client, _clips())
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    assert three_family_client.get(f"/api/v1/jobs/{job_id}").json()["status"] == "completed"
    (call,) = calls
    assert "clip_source_audio_paths" not in call
    meta = _metadata(three_family_client, job_id)
    assert "a2v" not in meta
    assert all("a2v" not in c for c in meta["h3"]["clips"])
    assert [c["source_audio"] for c in meta["request"]["clips"]] == [None, None, None]


def test_h3_runner_forwards_clip_source_audio_paths():
    seen: dict = {}

    class _Backend:
        loaded = True

        def generate_chain(self, chain_request, **kwargs):
            seen.update(kwargs)
            return "outcome"

    runner = h3.H3Runner.__new__(h3.H3Runner)
    runner._backend = _Backend()
    paths = ["a.wav", None]
    assert runner.generate_chain("chain", output_dir="out", seed=3, clip_source_audio_paths=paths) == "outcome"
    assert seen["clip_source_audio_paths"] is paths


def test_the_mock_refuses_unresolved_clip_audio(tmp_path):
    chain = _validated_chain(clips=_clips(None, {"audio_id": "a"}))
    mock = h3._MockBackendH3.__new__(h3._MockBackendH3)
    mock._loaded = True
    with pytest.raises(RuntimeError, match=r"clips\[1\]\.source_audio requested but its audio was not resolved"):
        mock.generate_chain(chain, tmp_path / "job", seed=1, clip_source_audio_paths=[None, None, None])


# --------------------------------------------------------------------------- #
# the real worker payload
# --------------------------------------------------------------------------- #


def test_clip_source_audio_payloads_is_one_entry_per_clip():
    chain = _validated_chain(clips=_clips({"audio_id": "a", "strength": 0.25}, None, {"audio_id": "a", "offset_frames": 204}))
    assert h3._clip_source_audio_payloads(chain, ["x.wav", None, "x.wav"]) == [
        {"path": "x.wav", "strength": 0.25, "offset_frames": 0},
        None,
        {"path": "x.wav", "strength": 0.5, "offset_frames": 204},
    ]
    assert h3._clip_source_audio_payloads(_validated_chain(), None) == [None, None, None]


def test_real_chain_payload_carries_source_audio_last_on_its_clip(tmp_path):
    voice, pic = tmp_path / "v.wav", tmp_path / "p.png"
    clips = _clips(None, {"audio_id": "a", "strength": 0.5, "offset_frames": 102}, None)
    clips[1]["references"] = [{"kind": "picture", "id": "img", "slot": 1}]
    chain = _validated_chain(clips=clips)
    captured: list[dict] = []
    be = _capturing_backend(captured, CHAIN_DONE_EVENT)
    be.generate_chain(
        chain, tmp_path / "job", seed=42,
        clip_reference_paths=[[], [pic], []], clip_source_audio_paths=[None, voice, None],
    )
    (payload,) = captured
    assert list(payload) == GOLDEN_CHAIN_KEYS_H3
    assert "source_audio" not in payload
    first, second, third = payload["clips"]
    assert list(first) == ["prompt", "num_frames"] and list(third) == ["prompt", "num_frames"]
    assert list(second) == ["prompt", "num_frames", "references", "source_audio"]
    assert second["source_audio"] == {"path": str(voice), "strength": 0.5, "offset_frames": 102}


def test_real_chain_payload_without_clip_audio_is_the_golden(tmp_path):
    captured: list[dict] = []
    _capturing_backend(captured, CHAIN_DONE_EVENT).generate_chain(_validated_chain(), tmp_path / "job", seed=42)
    (payload,) = captured
    assert list(payload) == GOLDEN_CHAIN_KEYS_H3
    assert all(list(clip) == ["prompt", "num_frames"] for clip in payload["clips"])


@pytest.mark.parametrize(
    "paths, message",
    [
        (None, r"clips\[1\]\.source_audio requested but its audio was not resolved"),
        ([None, None, None], r"clips\[1\]\.source_audio requested but its audio was not resolved"),
        ([None, "a.wav"], r"3 clip\(s\) in the chain but 2 source audio path\(s\) resolved"),
    ],
)
def test_real_chain_payload_refuses_unresolved_clip_audio(tmp_path, paths, message):
    chain = _validated_chain(clips=_clips(None, {"audio_id": "a"}))
    captured: list[dict] = []
    with pytest.raises(RuntimeError, match=message):
        _capturing_backend(captured, CHAIN_DONE_EVENT).generate_chain(
            chain, tmp_path / "job", seed=1, clip_source_audio_paths=paths
        )
    assert captured == []
