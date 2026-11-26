"""MiniMax H3 Spark-H3 (``spark_h3``) on the app side: the three request
fields, the worker payloads, the ``done`` echo, the mock job and metadata.json,
and the LTX refusals (Docs/MINIMAX_H3_DESIGN.md §6.11). ``/status``'s
``spark_h3_available`` is pinned in tests/test_smoke.py beside sage's.

Pinned here, none of it needing a GPU or the Spark-H3 kit:

1. ``spark_h3`` off (omitted or ``false``, whatever the two numbers say)
   changes nothing on the wire: the payloads are the goldens,
   ``outcome.spark_h3_used`` is None and metadata.json has no
   ``spark_h3_used`` key;
2. on, the payload carries ``spark_h3 = {topk_ratio, full_steps}`` with the
   ratio as given (not rounded), and the ``done`` echo becomes
   ``outcome.spark_h3_used``;
3. the mock job records ``spark_h3_used: "on->off"`` (it runs no kernel);
4. LTX 2.3 / 2.5 refuse ``spark_h3`` with the H3-only sentence and publish it,
   and let the two numbers through alone (governed by ``spark_h3``).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from api.errors import APIError
from api.models import GenerateChainRequest, GenerateRequest
from services.engines.h3 import adapter as h3
from services.engines.ltx import adapter as ltx23
from services.engines.ltx25 import adapter as ltx25
from test_engine_h3_adapter import (
    CHAIN_DONE_EVENT,
    DONE_EVENT,
    GOLDEN_CHAIN_KEYS_H3,
    GOLDEN_CHAIN_KEYS_H3_RETAKE,
    GOLDEN_CHAIN_KEYS_H3_V2V,
    GOLDEN_GENERATE_KEYS_H3,
    PROMPT,
    _activate,
    _capturing_backend,
    _chain,
    _metadata,
    _request,
    _validated,
    _validated_chain,
)

H3_ONLY_DETAIL = "Spark-H3 は MiniMax H3 専用です。LTX では使えません。"
FIELDS = ("spark_h3", "spark_h3_topk_ratio", "spark_h3_full_steps")
BODY = {"prompt": PROMPT, "width": 640, "height": 384, "num_frames": 22}
CHAIN_BODY = {"prompt": PROMPT, "width": 640, "height": 384, "clips": [{"num_frames": 124}, {"num_frames": 124}]}
#: The two numbers moved off their defaults, with spark_h3 itself off.
NUMBERS_ONLY = {"spark_h3_topk_ratio": 0.1, "spark_h3_full_steps": 5}


# --------------------------------------------------------------------------- #
# the fields and the tables
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("schema", [GenerateRequest, GenerateChainRequest])
def test_the_fields_and_their_defaults(schema):
    fields = schema.model_fields
    assert fields["spark_h3"].default is False
    assert fields["spark_h3_topk_ratio"].default == 0.2
    assert fields["spark_h3_full_steps"].default == 2


@pytest.mark.parametrize(
    "field, value, ok",
    [
        ("spark_h3_topk_ratio", 0.01, True),
        ("spark_h3_topk_ratio", 0.5, True),
        ("spark_h3_topk_ratio", 0.009, False),
        ("spark_h3_topk_ratio", 0.51, False),
        ("spark_h3_full_steps", 0, True),
        ("spark_h3_full_steps", 10, True),
        ("spark_h3_full_steps", -1, False),
        ("spark_h3_full_steps", 11, False),
    ],
)
def test_the_ranges(field, value, ok):
    for build in (_request, _chain):
        if ok:
            assert getattr(build(spark_h3=True, **{field: value}), field) == value
        else:
            with pytest.raises(ValidationError):
                build(spark_h3=True, **{field: value})


def test_h3_honours_them_and_does_not_publish_spark():
    for field in FIELDS:
        assert field in h3.HONOURED_FIELDS and field in h3.CHAIN_HONOURED_FIELDS
    assert "spark_h3" not in h3.UNSUPPORTED_FEATURES
    h3.reject_unsupported(_request(spark_h3=True))  # no raise
    h3.reject_chain(_chain(spark_h3=True))  # no raise


def test_to_clip_request_keeps_them():
    clip = _validated_chain(spark_h3=True, spark_h3_topk_ratio=0.137, spark_h3_full_steps=7).to_clip_request(0)
    assert (clip.spark_h3, clip.spark_h3_topk_ratio, clip.spark_h3_full_steps) == (True, 0.137, 7)
    clip = _validated_chain().to_clip_request(0)
    assert (clip.spark_h3, clip.spark_h3_topk_ratio, clip.spark_h3_full_steps) == (False, 0.2, 2)


def test_on_changes_no_step_or_shift():
    req = _validated(spark_h3=True)
    assert (req.num_inference_steps, req.shift_video, req.shift_audio) == (20, 12.0, 3.0)


# --------------------------------------------------------------------------- #
# the worker payloads
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "overrides",
    [{}, {"spark_h3": False}, NUMBERS_ONLY],
    ids=["omitted", "off", "numbers-only"],
)
def test_off_payloads_are_todays_goldens(tmp_path, overrides):
    captured: list[dict] = []
    be = _capturing_backend(captured, DONE_EVENT)
    outcome = be.generate(_validated(**overrides), tmp_path / "job", seed=42)
    assert list(captured[-1]) == GOLDEN_GENERATE_KEYS_H3
    assert outcome.spark_h3_used is None
    be = _capturing_backend(captured, CHAIN_DONE_EVENT)
    outcome = be.generate_chain(_validated_chain(**overrides), tmp_path / "job", seed=42)
    assert list(captured[-1]) == GOLDEN_CHAIN_KEYS_H3
    assert outcome.spark_h3_used is None
    be.generate_chain(
        _validated_chain(source_video={"video_id": "vid-1"}, clips=[{"num_frames": 56}, {"num_frames": 39}], **overrides),
        tmp_path / "job", seed=42, source_tail_path=tmp_path / "t.mp4", source_context_frames=22,
    )
    assert list(captured[-1]) == GOLDEN_CHAIN_KEYS_H3_V2V
    be.generate_chain(
        _validated_chain(retake={"video_id": "vid-1", "window_start_sec": 0.0}, clips=[{"num_frames": 124}], **overrides),
        tmp_path / "job", seed=42, retake_window_path=tmp_path / "w.mp4",
    )
    assert list(captured[-1]) == GOLDEN_CHAIN_KEYS_H3_RETAKE


@pytest.mark.parametrize("used", ["on", "on->off"])
def test_spark_payload_carries_the_numbers_and_the_echo(tmp_path, used):
    captured: list[dict] = []
    be = _capturing_backend(captured, {**DONE_EVENT, "spark_h3_used": used})
    outcome = be.generate(
        _validated(spark_h3=True, spark_h3_topk_ratio=0.137, spark_h3_full_steps=3, keep_resident=True),
        tmp_path / "job", seed=42,
    )
    (payload,) = captured
    assert payload["spark_h3"] == {"topk_ratio": 0.137, "full_steps": 3}
    assert type(payload["spark_h3"]["topk_ratio"]) is float
    assert type(payload["spark_h3"]["full_steps"]) is int
    keys = list(payload)
    assert keys[-3:] == ["spark_h3", "output_path", "keep_resident"]
    assert [k for k in keys if k not in ("spark_h3", "keep_resident")] == GOLDEN_GENERATE_KEYS_H3
    # The attention backend still rides as today (Spark is layered on it).
    assert payload["attention_backend"] == "sdpa"
    assert outcome.spark_h3_used == used


def test_spark_chain_payloads(tmp_path):
    captured: list[dict] = []
    be = _capturing_backend(captured, {**CHAIN_DONE_EVENT, "spark_h3_used": "on"})
    outcome = be.generate_chain(_validated_chain(spark_h3=True, keep_resident=True), tmp_path / "job", seed=42)
    assert list(captured[-1]) == GOLDEN_CHAIN_KEYS_H3 + ["spark_h3", "keep_resident"]
    assert captured[-1]["spark_h3"] == {"topk_ratio": 0.2, "full_steps": 2}
    assert outcome.spark_h3_used == "on"
    be.generate_chain(
        _validated_chain(spark_h3=True, source_video={"video_id": "vid-1"}, clips=[{"num_frames": 56}, {"num_frames": 39}]),
        tmp_path / "job", seed=42, source_tail_path=tmp_path / "t.mp4", source_context_frames=22,
    )
    assert list(captured[-1]) == GOLDEN_CHAIN_KEYS_H3_V2V + ["spark_h3"]
    be.generate_chain(
        _validated_chain(spark_h3=True, retake={"video_id": "vid-1", "window_start_sec": 0.0},
                         clips=[{"num_frames": 124}]),
        tmp_path / "job", seed=42, retake_window_path=tmp_path / "w.mp4",
    )
    assert list(captured[-1]) == GOLDEN_CHAIN_KEYS_H3_RETAKE + ["spark_h3"]


def test_the_engine_spark_records_ride_through(tmp_path):
    record = {"used": "on->off", "reason": "short_sequence", "sequence_tokens": 9000}
    event = {**DONE_EVENT, "spark_h3_used": "on->off", "h3": {**DONE_EVENT.get("h3", {}), "spark_h3": record}}
    captured: list[dict] = []
    outcome = _capturing_backend(captured, event).generate(_validated(spark_h3=True), tmp_path / "job", seed=42)
    assert outcome.h3["spark_h3"] == record
    # A chain's per-clip record rides chain.clips[i].spark_h3 untouched.
    clips = [{**clip, "spark_h3": record} for clip in CHAIN_DONE_EVENT["chain"]["clips"]]
    event = {**CHAIN_DONE_EVENT, "spark_h3_used": "on->off", "chain": {**CHAIN_DONE_EVENT["chain"], "clips": clips}}
    outcome = _capturing_backend(captured, event).generate_chain(
        _validated_chain(spark_h3=True), tmp_path / "job", seed=42
    )
    assert [c["spark_h3"] for c in outcome.chain_metadata["clips"]] == [record] * len(clips)


# --------------------------------------------------------------------------- #
# HTTP over the mock backend
# --------------------------------------------------------------------------- #


def _completed(client, path: str, body: dict) -> str:
    r = client.post(path, json=body)
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    job = client.get(f"/api/v1/jobs/{job_id}").json()
    assert job["status"] == "completed", job
    return job_id


def test_mock_spark_job_records_on_off(three_family_client):
    _activate(three_family_client, "MiniMaxH3")
    body = {"spark_h3": True, "spark_h3_topk_ratio": 0.15, "spark_h3_full_steps": 1}
    for path, base in (("/api/v1/generate", BODY), ("/api/v1/generate/chain", CHAIN_BODY)):
        meta = _metadata(three_family_client, _completed(three_family_client, path, {**base, **body}))
        assert meta["spark_h3_used"] == "on->off"
        assert {k: meta["request"][k] for k in FIELDS} == {
            "spark_h3": True, "spark_h3_topk_ratio": 0.15, "spark_h3_full_steps": 1,
        }


@pytest.mark.parametrize("overrides", [{}, {"spark_h3": False}, NUMBERS_ONLY], ids=["omitted", "off", "numbers-only"])
def test_mock_off_records_have_no_spark_key(three_family_client, overrides):
    _activate(three_family_client, "MiniMaxH3")
    for path, base in (("/api/v1/generate", BODY), ("/api/v1/generate/chain", CHAIN_BODY)):
        meta = _metadata(three_family_client, _completed(three_family_client, path, {**base, **overrides}))
        assert "spark_h3_used" not in meta
        assert meta["request"]["spark_h3"] is False


def test_models_publishes_it_on_ltx_only(three_family_client):
    listing = three_family_client.get("/api/v1/models").json()
    by_id = {b["id"]: b for b in listing["base_models"]}
    assert "spark_h3" not in by_id["MiniMaxH3"]["unsupported_features"]
    assert "spark_h3" in by_id["LTX23"]["unsupported_features"]
    assert "spark_h3" in by_id["LTX25"]["unsupported_features"]


# --------------------------------------------------------------------------- #
# LTX refuses it
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("adapter", [ltx23, ltx25], ids=["ltx23", "ltx25"])
def test_ltx_refuses_with_the_h3_only_sentence(adapter):
    for reject, request, off in (
        (adapter.reject_unsupported, GenerateRequest(prompt=PROMPT, spark_h3=True),
         GenerateRequest(prompt=PROMPT, **NUMBERS_ONLY)),
        (adapter.reject_chain, _chain(spark_h3=True), _chain(**NUMBERS_ONLY)),
    ):
        with pytest.raises(APIError) as ei:
            reject(request)
        assert ei.value.code == "FEATURE_UNSUPPORTED" and ei.value.status_code == 422
        assert ei.value.detail == H3_ONLY_DETAIL
        reject(off)  # the two numbers alone: no raise
    assert "spark_h3" in adapter.UNSUPPORTED_FEATURES
    assert adapter.GOVERNED_FIELDS["spark_h3_topk_ratio"] == "spark_h3"
    assert adapter.GOVERNED_FIELDS["spark_h3_full_steps"] == "spark_h3"


@pytest.mark.parametrize("base_model", ["LTX23", "LTX25"])
def test_ltx_refuses_over_http(three_family_client, base_model):
    _activate(three_family_client, base_model)
    for path, body in (
        ("/api/v1/generate", {"prompt": PROMPT, "num_frames": 25}),
        ("/api/v1/generate/chain", {"prompt": PROMPT, "clips": [{"num_frames": 25}, {"num_frames": 25}]}),
    ):
        r = three_family_client.post(path, json={**body, "spark_h3": True})
        assert r.status_code == 422, r.text
        error = r.json()["error"]
        assert error["code"] == "FEATURE_UNSUPPORTED"
        assert error["detail"] == H3_ONLY_DETAIL
