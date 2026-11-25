"""MiniMax H3 ``keep_reference_latents`` on the app side: the request fields, the
clip copy, the worker payloads (the key rides only when true), the field
classification and the LTX refusals. The engine's cache is not touched here.
"""

from __future__ import annotations

import pytest

from api.errors import APIError
from api.models import GenerateChainRequest, GenerateRequest
from services import engines
from services.engines.h3 import adapter as h3
from services.engines.ltx import adapter as ltx23
from services.engines.ltx25 import adapter as ltx25
from test_engine_h3_adapter import (
    CHAIN_DONE_EVENT,
    DONE_EVENT,
    GOLDEN_CHAIN_KEYS_H3,
    GOLDEN_GENERATE_KEYS_H3,
    PROMPT,
    _activate,
    _capturing_backend,
    _chain,
    _request,
    _validated,
    _validated_chain,
)

FIELD = "keep_reference_latents"
H3_ONLY_DETAIL = (
    "参照素材の符号化結果の保持（keep_reference_latents）は MiniMax H3 専用です。"
    "LTX では使えません。"
)


# --------------------------------------------------------------------------- #
# the fields and the tables
# --------------------------------------------------------------------------- #


def test_the_fields_default_to_false():
    assert GenerateRequest.model_fields[FIELD].default is False
    assert GenerateChainRequest.model_fields[FIELD].default is False
    assert _request(keep_reference_latents=True).keep_reference_latents is True
    assert _chain(keep_reference_latents=True).keep_reference_latents is True


@pytest.mark.parametrize("value", [False, True])
def test_to_clip_request_keeps_it(value):
    chain = _validated_chain(keep_reference_latents=value)
    assert chain.to_clip_request(0).keep_reference_latents is value


def test_h3_honours_it_and_does_not_publish_it():
    assert FIELD in h3.HONOURED_FIELDS and FIELD in h3.CHAIN_HONOURED_FIELDS
    assert FIELD not in h3.UNSUPPORTED_FEATURES
    h3.reject_unsupported(_request(keep_reference_latents=True))  # no raise
    h3.reject_chain(_chain(keep_reference_latents=True))  # no raise


def test_the_field_is_classified_exactly_once_on_both_paths():
    # The H3 adapter's own audit (test_engine_h3_adapter) pins the whole schema;
    # this one names the field: honoured, never refused / ignored here.
    for field in (FIELD,):
        assert field in h3.HONOURED_FIELDS
        assert field not in {f for f, _feat, _p in h3.REJECT_TABLE}
        assert field not in h3.IGNORED_FIELDS
        assert field in h3.CHAIN_HONOURED_FIELDS
        assert field not in {f for f, _feat, _p in h3.CHAIN_REJECT_TABLE}


# --------------------------------------------------------------------------- #
# the worker payloads
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("overrides", [{}, {FIELD: False}], ids=["omitted", "off"])
def test_off_payloads_have_no_key(tmp_path, overrides):
    captured: list[dict] = []
    be = _capturing_backend(captured, DONE_EVENT)
    be.generate(_validated(**overrides), tmp_path / "job", seed=42)
    assert list(captured[-1]) == GOLDEN_GENERATE_KEYS_H3
    assert FIELD not in captured[-1]
    be = _capturing_backend(captured, CHAIN_DONE_EVENT)
    be.generate_chain(_validated_chain(**overrides), tmp_path / "job", seed=42)
    assert list(captured[-1]) == GOLDEN_CHAIN_KEYS_H3
    assert FIELD not in captured[-1]


def test_single_payload_carries_true_after_sol_vae_position(tmp_path):
    captured: list[dict] = []
    be = _capturing_backend(captured, DONE_EVENT)
    be.generate(_validated(keep_reference_latents=True, keep_resident=True), tmp_path / "job", seed=42)
    (payload,) = captured
    assert payload[FIELD] is True
    keys = list(payload)
    assert keys[-3:] == [FIELD, "output_path", "keep_resident"]
    assert [k for k in keys if k not in (FIELD, "keep_resident")] == GOLDEN_GENERATE_KEYS_H3


def test_chain_payload_carries_true(tmp_path):
    captured: list[dict] = []
    be = _capturing_backend(captured, CHAIN_DONE_EVENT)
    be.generate_chain(
        _validated_chain(keep_reference_latents=True, keep_resident=True), tmp_path / "job", seed=42
    )
    assert captured[-1][FIELD] is True
    assert list(captured[-1]) == GOLDEN_CHAIN_KEYS_H3 + [FIELD, "keep_resident"]


# --------------------------------------------------------------------------- #
# LTX refuses it
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("adapter", [ltx23, ltx25], ids=["ltx23", "ltx25"])
def test_ltx_refuses_with_the_h3_only_sentence(adapter):
    for reject, request, off in (
        (
            adapter.reject_unsupported,
            GenerateRequest(prompt=PROMPT, keep_reference_latents=True),
            GenerateRequest(prompt=PROMPT),
        ),
        (adapter.reject_chain, _chain(keep_reference_latents=True), _chain()),
    ):
        with pytest.raises(APIError) as ei:
            reject(request)
        assert ei.value.code == "FEATURE_UNSUPPORTED" and ei.value.status_code == 422
        assert ei.value.detail == H3_ONLY_DETAIL
        reject(off)  # no raise
    assert FIELD in adapter.UNSUPPORTED_FEATURES
    # keep_text_conditioning's row came after this one.
    assert adapter.REJECT_TABLE[-2][1] == FIELD
    assert adapter.CHAIN_REJECT_TABLE[-2][1] == FIELD


@pytest.mark.parametrize("base_model", ["LTX23", "LTX25"])
def test_ltx_refuses_over_http(three_family_client, base_model):
    _activate(three_family_client, base_model)
    for path, body in (
        ("/api/v1/generate", {"prompt": PROMPT, "num_frames": 25}),
        ("/api/v1/generate/chain", {"prompt": PROMPT, "clips": [{"num_frames": 25}, {"num_frames": 25}]}),
    ):
        r = three_family_client.post(path, json={**body, FIELD: True})
        assert r.status_code == 422, r.text
        error = r.json()["error"]
        assert error["code"] == "FEATURE_UNSUPPORTED"
        assert error["detail"] == H3_ONLY_DETAIL


def test_models_publishes_it_for_ltx_only(three_family_client):
    listing = three_family_client.get("/api/v1/models").json()
    by_id = {b["id"]: b for b in listing["base_models"]}
    assert FIELD not in by_id["MiniMaxH3"]["unsupported_features"]
    assert FIELD in by_id["LTX23"]["unsupported_features"]
    assert FIELD in by_id["LTX25"]["unsupported_features"]
    assert FIELD not in engines.unsupported_features("h3")
