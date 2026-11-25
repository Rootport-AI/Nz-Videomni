"""MiniMax H3 ``keep_text_conditioning`` on the app side: the request fields, the
clip copy, the worker payloads (the key rides only when true), the field
classification. The engine's cache is not touched here.
"""

from __future__ import annotations

import pytest

from api.models import GenerateChainRequest, GenerateRequest
from services import engines
from services.engines.h3 import adapter as h3
from test_engine_h3_adapter import (
    CHAIN_DONE_EVENT,
    DONE_EVENT,
    GOLDEN_CHAIN_KEYS_H3,
    GOLDEN_GENERATE_KEYS_H3,
    _capturing_backend,
    _chain,
    _request,
    _validated,
    _validated_chain,
)

FIELD = "keep_text_conditioning"


# --------------------------------------------------------------------------- #
# the fields and the tables
# --------------------------------------------------------------------------- #


def test_the_fields_default_to_false():
    assert GenerateRequest.model_fields[FIELD].default is False
    assert GenerateChainRequest.model_fields[FIELD].default is False
    assert _request(keep_text_conditioning=True).keep_text_conditioning is True
    assert _chain(keep_text_conditioning=True).keep_text_conditioning is True


@pytest.mark.parametrize("value", [False, True])
def test_to_clip_request_keeps_it(value):
    chain = _validated_chain(keep_text_conditioning=value)
    assert chain.to_clip_request(0).keep_text_conditioning is value


def test_h3_honours_it_and_does_not_publish_it():
    assert FIELD in h3.HONOURED_FIELDS and FIELD in h3.CHAIN_HONOURED_FIELDS
    assert FIELD not in h3.UNSUPPORTED_FEATURES
    assert FIELD not in engines.unsupported_features("h3")
    h3.reject_unsupported(_request(keep_text_conditioning=True))  # no raise
    h3.reject_chain(_chain(keep_text_conditioning=True))  # no raise


def test_the_field_is_classified_exactly_once_on_both_paths():
    # The H3 adapter's own audit (test_engine_h3_adapter) pins the whole schema;
    # this one names the field: honoured, never refused / ignored here.
    assert FIELD in h3.HONOURED_FIELDS
    assert FIELD not in {f for f, _feat, _p in h3.REJECT_TABLE}
    assert FIELD not in h3.IGNORED_FIELDS
    assert FIELD in h3.CHAIN_HONOURED_FIELDS
    assert FIELD not in {f for f, _feat, _p in h3.CHAIN_REJECT_TABLE}


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


def test_single_payload_carries_true_just_before_output_path(tmp_path):
    captured: list[dict] = []
    be = _capturing_backend(captured, DONE_EVENT)
    be.generate(_validated(keep_text_conditioning=True, keep_resident=True), tmp_path / "job", seed=42)
    (payload,) = captured
    assert payload[FIELD] is True
    keys = list(payload)
    assert keys[-3:] == [FIELD, "output_path", "keep_resident"]
    assert [k for k in keys if k not in (FIELD, "keep_resident")] == GOLDEN_GENERATE_KEYS_H3


def test_chain_payload_carries_true(tmp_path):
    captured: list[dict] = []
    be = _capturing_backend(captured, CHAIN_DONE_EVENT)
    be.generate_chain(
        _validated_chain(keep_text_conditioning=True, keep_resident=True), tmp_path / "job", seed=42
    )
    assert captured[-1][FIELD] is True
    assert list(captured[-1]) == GOLDEN_CHAIN_KEYS_H3 + [FIELD, "keep_resident"]


# --------------------------------------------------------------------------- #
# independent of keep_reference_latents
# --------------------------------------------------------------------------- #


def test_it_rides_with_keep_reference_latents_independently(tmp_path):
    captured: list[dict] = []
    be = _capturing_backend(captured, DONE_EVENT)
    be.generate(_validated(keep_text_conditioning=True), tmp_path / "job", seed=42)
    assert captured[-1][FIELD] is True and "keep_reference_latents" not in captured[-1]
    be.generate(
        _validated(keep_text_conditioning=True, keep_reference_latents=True), tmp_path / "job", seed=42
    )
    assert captured[-1][FIELD] is True and captured[-1]["keep_reference_latents"] is True
    be = _capturing_backend(captured, CHAIN_DONE_EVENT)
    be.generate_chain(_validated_chain(keep_reference_latents=True), tmp_path / "job", seed=42)
    assert FIELD not in captured[-1] and captured[-1]["keep_reference_latents"] is True
