"""``keep_text_conditioning`` on LTX 2.3 and LTX 2.5, app side (台帳 §1-299).

The field is an honoured field on both LTX adapters now (it was refused with 422
FEATURE_UNSUPPORTED while only MiniMax H3 knew it — see
tests/test_h3_keep_text_conditioning_api.py for H3's half). What the app side
owes the engines, and what is pinned here without a GPU or a worker:

* the field is classified as honoured, on the single and the chain path, and is
  no longer a published limitation;
* the worker payload carries ``keep_text_conditioning: true`` ONLY when the
  request says true (a false job's payload is today's, key for key);
* LTX 2.3's own record (the ``ltx23`` dict, only on a job that asked) travels
  worker -> adapter -> ``metadata.json`` unchanged: top level of the ``done``
  event for a single job, inside ``done.chain`` for a chain; a job without the
  key writes no ``ltx23``;
* Alpha Gen's request relays the field into the internal ``GenerateRequest``.

The engines' cache itself is not touched here.
"""

from __future__ import annotations

import json
import threading
import types
from pathlib import Path

import pytest

from api.models import AlphaGenRequest, GenerateChainRequest, GenerateRequest
from services import engines
from services.engines.ltx import adapter as ltx23
from services.engines.ltx25 import adapter as ltx25
from test_ltx25_adapter import (
    GOLDEN_ACCEL_KEYS_25,
    GOLDEN_CHAIN_KEYS_25,
    GOLDEN_GENERATE_KEYS_25,
    _capturing_backend as _capturing_backend25,
    _capturing_chain_backend as _capturing_chain_backend25,
    _chain_request as _chain_request25,
    _request as _request25,
)

FIELD = "keep_text_conditioning"
PROMPT = "a quiet harbour at first light"

#: What an LTX 2.3 worker reports for a job that asked (the shape is the
#: engine's; the app relays it verbatim, so any dict proves the road).
LTX23_RECORD = {
    "te_cache": {"positive": "hit", "negative": "miss"},
    "conditioning_sha256": {"positive": {"video": "aa", "audio": "bb"}},
}
LTX23_CHAIN_RECORD = {
    "te_cache": {"clips": ["hit", "miss"], "stage2": "hit", "negative": "miss"},
    "conditioning_sha256": {"clips": [{"video": "aa", "audio": "bb"}]},
}

ADAPTERS = [ltx23, ltx25]
ADAPTER_IDS = ["ltx23", "ltx25"]


def _request23(**overrides) -> GenerateRequest:
    return GenerateRequest(prompt=PROMPT, width=512, height=320, num_frames=25, **overrides)


def _chain_request23(**overrides) -> GenerateChainRequest:
    base = dict(
        prompt=PROMPT, width=512, height=320, frame_rate=24.0, overlap_frames=2,
        clips=[{"num_frames": 25}, {"num_frames": 25}],
    )
    base.update(overrides)
    return GenerateChainRequest(**base)


# --------------------------------------------------------------------------- #
# the fields and the tables
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("adapter", ADAPTERS, ids=ADAPTER_IDS)
def test_the_field_is_honoured_on_both_paths_and_not_a_limitation(adapter):
    assert FIELD in adapter.HONOURED_FIELDS
    assert FIELD in getattr(adapter, "CHAIN_HONOURED_FIELDS", adapter.HONOURED_FIELDS)
    assert FIELD not in {f for f, _feat, _p in adapter.REJECT_TABLE}
    assert FIELD not in {f for f, _feat, _p in adapter.CHAIN_REJECT_TABLE}
    assert FIELD not in adapter.IGNORED_FIELDS
    assert FIELD not in adapter.GOVERNED_FIELDS
    assert FIELD not in adapter.UNSUPPORTED_FEATURES
    assert FIELD not in getattr(adapter, "H3_ONLY_REFUSALS", {})


@pytest.mark.parametrize("adapter", ADAPTERS, ids=ADAPTER_IDS)
def test_a_true_request_is_accepted_single_and_chain(adapter):
    adapter.reject_unsupported(GenerateRequest(prompt=PROMPT, keep_text_conditioning=True))
    adapter.reject_chain(_chain_request23(keep_text_conditioning=True))


def test_unsupported_features_do_not_name_it():
    for family in ("ltx", "ltx25", "h3"):
        assert FIELD not in engines.unsupported_features(family)


def test_the_fields_default_to_false_and_the_clip_copy_keeps_it():
    assert GenerateRequest.model_fields[FIELD].default is False
    assert GenerateChainRequest.model_fields[FIELD].default is False
    chain = _chain_request23(keep_text_conditioning=True)
    assert chain.to_clip_request(0).keep_text_conditioning is True
    assert _chain_request23().to_clip_request(0).keep_text_conditioning is False


# --------------------------------------------------------------------------- #
# LTX 2.3: the worker payloads and the record
# --------------------------------------------------------------------------- #


def _backend23(captured: list[dict], event: dict):
    from services.ltx_runner import _RealBackend

    be = _RealBackend.__new__(_RealBackend)
    be._proc = types.SimpleNamespace(poll=lambda: None)  # type: ignore[attr-defined]
    be._lock = threading.Lock()

    def _send(msg: dict) -> None:
        captured.append(msg)
        out = Path(msg["output_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"\x00" * 16)

    be._send = _send  # type: ignore[attr-defined]
    be._read_worker_events = lambda cb, chain, prefix, **kw: dict(event)  # type: ignore[attr-defined]
    return be


_DONE23 = {"event": "done", "seed_used": 7, "peak_vram_mb": 100}


def test_ltx23_single_payload_has_the_key_only_when_true(tmp_path):
    captured: list[dict] = []
    be = _backend23(captured, _DONE23)
    be.generate(_request23(), tmp_path / "omitted", seed=7)
    be.generate(_request23(keep_text_conditioning=False), tmp_path / "off", seed=7)
    be.generate(_request23(keep_text_conditioning=True), tmp_path / "on", seed=7)
    omitted, off, on = captured
    assert FIELD not in omitted and FIELD not in off
    assert list(off) == list(omitted)
    assert on[FIELD] is True
    assert [k for k in on if k != FIELD] == list(omitted)


def test_ltx23_chain_payload_has_the_key_only_when_true(tmp_path):
    captured: list[dict] = []
    be = _backend23(captured, {"event": "done", "seed_used": 7, "chain": {}})
    be.generate_chain(_chain_request23(), tmp_path / "omitted", seed=7)
    be.generate_chain(_chain_request23(keep_text_conditioning=False), tmp_path / "off", seed=7)
    be.generate_chain(_chain_request23(keep_text_conditioning=True), tmp_path / "on", seed=7)
    omitted, off, on = captured
    assert FIELD not in omitted and FIELD not in off
    assert list(off) == list(omitted)
    assert on[FIELD] is True
    assert [k for k in on if k != FIELD] == list(omitted)


def test_ltx23_single_outcome_relays_the_record_verbatim(tmp_path):
    be = _backend23([], {**_DONE23, "ltx23": LTX23_RECORD})
    outcome = be.generate(_request23(keep_text_conditioning=True), tmp_path / "on", seed=7)
    assert outcome.ltx23 == LTX23_RECORD
    be = _backend23([], _DONE23)
    assert be.generate(_request23(), tmp_path / "off", seed=7).ltx23 is None


def test_ltx23_chain_outcome_carries_the_record_inside_chain_metadata(tmp_path):
    be = _backend23([], {"event": "done", "seed_used": 7, "chain": {"ltx23": LTX23_CHAIN_RECORD}})
    outcome = be.generate_chain(_chain_request23(keep_text_conditioning=True), tmp_path / "on", seed=7)
    assert outcome.chain_metadata["ltx23"] == LTX23_CHAIN_RECORD
    assert outcome.ltx23 is None  # a chain's facts ride chain_metadata, as ltx25's do


# --------------------------------------------------------------------------- #
# LTX 2.5: the worker payloads
# --------------------------------------------------------------------------- #


def test_ltx25_single_payload_has_the_key_only_when_true(tmp_path):
    captured: list[dict] = []
    be = _capturing_backend25(captured)
    be.generate(_request25(seed=5), tmp_path / "omitted")
    be.generate(_request25(seed=5, keep_text_conditioning=False), tmp_path / "off")
    be.generate(_request25(seed=5, keep_text_conditioning=True), tmp_path / "on")
    omitted, off, on = captured
    assert list(omitted) == GOLDEN_GENERATE_KEYS_25 + GOLDEN_ACCEL_KEYS_25
    assert FIELD not in off
    assert off == omitted | {"output_path": off["output_path"]}
    assert list(on) == GOLDEN_GENERATE_KEYS_25 + GOLDEN_ACCEL_KEYS_25 + [FIELD]
    assert on[FIELD] is True


def test_ltx25_chain_payload_has_the_key_only_when_true(tmp_path):
    captured: list[dict] = []
    be = _capturing_chain_backend25(captured)
    be.generate_chain(_chain_request25(), output_dir=tmp_path / "omitted")
    be.generate_chain(_chain_request25(keep_text_conditioning=False), output_dir=tmp_path / "off")
    be.generate_chain(_chain_request25(keep_text_conditioning=True), output_dir=tmp_path / "on")
    omitted, off, on = captured
    assert list(omitted) == GOLDEN_CHAIN_KEYS_25 + GOLDEN_ACCEL_KEYS_25
    assert FIELD not in off
    assert off == omitted | {"output_path": off["output_path"]}
    assert list(on) == GOLDEN_CHAIN_KEYS_25 + GOLDEN_ACCEL_KEYS_25 + [FIELD]
    assert on[FIELD] is True


# --------------------------------------------------------------------------- #
# over HTTP (the mock backend): accepted, listed nowhere as a limitation, and
# LTX 2.3's record reaches metadata.json
# --------------------------------------------------------------------------- #

BASE_REQUEST = {"prompt": PROMPT, "width": 320, "height": 320, "num_frames": 9}
CHAIN_BODY = {
    "prompt": PROMPT,
    "width": 320,
    "height": 320,
    "frame_rate": 24.0,
    "overlap_frames": 2,
    "clips": [{"num_frames": 25}, {"num_frames": 25}],
}


def _activate(client, base_model: str) -> None:
    r = client.post("/api/v1/pipeline/load", json={"base_model": base_model})
    assert r.status_code == 200, r.text


def _metadata(client, job_id: str) -> dict:
    assert client.get(f"/api/v1/jobs/{job_id}").json()["status"] == "completed"
    path = client.app_context.config.output_dir / job_id / "metadata.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("base_model", ["LTX23", "LTX25"])
def test_true_is_accepted_over_http_single_and_chain(three_family_client, base_model):
    _activate(three_family_client, base_model)
    r = three_family_client.post("/api/v1/generate", json={**BASE_REQUEST, FIELD: True})
    assert r.status_code == 202, r.text
    assert _metadata(three_family_client, r.json()["job_id"])  # runs to the end
    r = three_family_client.post("/api/v1/generate/chain", json={**CHAIN_BODY, FIELD: True})
    assert r.status_code == 202, r.text
    assert _metadata(three_family_client, r.json()["job_id"])


def test_models_does_not_publish_it_for_any_family(three_family_client):
    listing = three_family_client.get("/api/v1/models").json()
    for entry in listing["base_models"]:
        assert FIELD not in entry["unsupported_features"], entry["id"]


def test_ltx23_single_record_reaches_metadata_json_and_is_absent_otherwise(three_family_client):
    _activate(three_family_client, "LTX23")
    pm = three_family_client.app_context.pipeline_manager
    orig = pm.runner.generate
    attach = {"record": True}

    def wrapped(*a, **k):
        outcome = orig(*a, **k)
        if attach["record"]:
            outcome.ltx23 = LTX23_RECORD
        return outcome

    pm.runner.generate = wrapped  # type: ignore[method-assign]
    r = three_family_client.post("/api/v1/generate", json={**BASE_REQUEST, FIELD: True})
    assert r.status_code == 202, r.text
    assert _metadata(three_family_client, r.json()["job_id"])["ltx23"] == LTX23_RECORD

    attach["record"] = False  # a job the worker did not annotate writes no key
    r = three_family_client.post("/api/v1/generate", json=BASE_REQUEST)
    assert r.status_code == 202, r.text
    assert "ltx23" not in _metadata(three_family_client, r.json()["job_id"])


def test_ltx23_chain_record_reaches_metadata_json_and_is_absent_otherwise(three_family_client):
    _activate(three_family_client, "LTX23")
    pm = three_family_client.app_context.pipeline_manager
    orig = pm.runner.generate_chain
    attach = {"record": True}

    def wrapped(*a, **k):
        outcome = orig(*a, **k)
        if attach["record"]:
            outcome.chain_metadata["ltx23"] = LTX23_CHAIN_RECORD
        return outcome

    pm.runner.generate_chain = wrapped  # type: ignore[method-assign]
    r = three_family_client.post("/api/v1/generate/chain", json={**CHAIN_BODY, FIELD: True})
    assert r.status_code == 202, r.text
    assert _metadata(three_family_client, r.json()["job_id"])["ltx23"] == LTX23_CHAIN_RECORD

    attach["record"] = False
    r = three_family_client.post("/api/v1/generate/chain", json=CHAIN_BODY)
    assert r.status_code == 202, r.text
    assert "ltx23" not in _metadata(three_family_client, r.json()["job_id"])


def test_a_plain_mock_job_writes_no_ltx23_key(three_family_client):
    # The mock never fabricates the engine's record (it is not the engine).
    for base_model in ("LTX23", "LTX25"):
        _activate(three_family_client, base_model)
        r = three_family_client.post("/api/v1/generate", json={**BASE_REQUEST, FIELD: True})
        assert r.status_code == 202, r.text
        assert "ltx23" not in _metadata(three_family_client, r.json()["job_id"])


# --------------------------------------------------------------------------- #
# Alpha Gen: the sixth relayed field (the relay into the internal request is
# read back from the stored request in tests/test_generate_alpha_api.py)
# --------------------------------------------------------------------------- #


def test_alpha_gen_request_has_the_same_field_as_the_generate_request():
    spec = AlphaGenRequest.model_fields[FIELD]
    base = GenerateRequest.model_fields[FIELD]
    assert spec.annotation is base.annotation is bool
    assert spec.default is base.default is False
    body = {"reference_video_id": "vid", "num_frames": 9}
    assert AlphaGenRequest(**body).keep_text_conditioning is False
    assert AlphaGenRequest(**body, keep_text_conditioning=True).keep_text_conditioning is True

