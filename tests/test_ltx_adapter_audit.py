"""LTX 2.3 engine adapter — the four-way GenerateRequest field audit.

The 2.3 adapter's ``REJECT_TABLE`` grew its SECOND row with LTX AlphaGen
第 1 弾 (``alpha_gen``), and its own comment promised that a second row brings
the 2.5 adapter's exhaustive classification audit over in the same change
(tests/test_ltx25_adapter.py §3b). This file is that audit for 2.3:

1. every ``GenerateRequest`` field has exactly one home in ``REJECT_TABLE``,
   ``IGNORED_FIELDS``, ``HONOURED_FIELDS`` or ``GOVERNED_FIELDS`` — so a new
   request field that nobody classified for 2.3 fails here, loudly;
2. the two refusals (``keep_resident_embeddings`` and ``alpha_gen``, both
   LTX 2.5-only things) raise 422 FEATURE_UNSUPPORTED naming LTX 2.5 as the
   base model to switch to, at the adapter and through HTTP;
3. ``UNSUPPORTED_FEATURES`` (what GET /models publishes) carries both, in table
   order.

No GPU, no weight file.
"""

from __future__ import annotations

import inspect

import pytest

from api.errors import APIError
from api.models import GenerateRequest
from services.engines.ltx import adapter as ltx23

_LORAS = [{"name": "style-a", "strength": 1.0}]


def _request(**overrides) -> GenerateRequest:
    return GenerateRequest(prompt="a quiet harbour at first light", **overrides)


#: One VALID request per refused field, at a non-default value. ``alpha_gen``
#: carries its schema companions (reference video, LoRA, 64-grid canvas), so
#: the body reaches the engine's ruling instead of a ValidationError.
REQUEST_OVERRIDES_23: dict[str, dict] = {
    "keep_resident_embeddings": {"keep_resident_embeddings": True},
    "alpha_gen": {
        "alpha_gen": {"working_width": 512, "working_height": 384},
        "reference_video_id": "vid-123",
        "loras": _LORAS,
        "width": 512,
        "height": 384,
    },
}

_CLASSIFICATIONS = {
    "422 (REJECT_TABLE)": lambda: {f for f, _feat, _p in ltx23.REJECT_TABLE},
    "ignored (IGNORED_FIELDS)": lambda: set(ltx23.IGNORED_FIELDS),
    "honoured (HONOURED_FIELDS)": lambda: set(ltx23.HONOURED_FIELDS),
    "governed by another field (GOVERNED_FIELDS)": lambda: set(ltx23.GOVERNED_FIELDS),
}


# --------------------------------------------------------------------------- #
# 1) the audit
# --------------------------------------------------------------------------- #


def test_the_four_tables_together_are_exactly_the_schema():
    classified: set[str] = set()
    for produce in _CLASSIFICATIONS.values():
        classified |= produce()
    fields = set(GenerateRequest.model_fields)
    assert not fields - classified, (
        "GenerateRequest gained field(s) the LTX 2.3 adapter says nothing about: "
        f"{sorted(fields - classified)}. Put each one in REJECT_TABLE, "
        "IGNORED_FIELDS, HONOURED_FIELDS or GOVERNED_FIELDS in "
        "services/engines/ltx/adapter.py (and the 2.5 adapter's tables), then re-run."
    )
    assert not classified - fields, (
        f"the LTX 2.3 tables name unknown field(s): {sorted(classified - fields)}"
    )


def test_the_four_tables_do_not_overlap():
    seen: dict[str, str] = {}
    for label, produce in _CLASSIFICATIONS.items():
        for field in produce():
            assert field not in seen, f"{field} is in both {seen[field]} and {label}"
            seen[field] = label


def test_every_governor_is_itself_refused():
    refused = {f for f, _feat, _p in ltx23.REJECT_TABLE}
    for field, governor in ltx23.GOVERNED_FIELDS.items():
        assert governor in refused, f"{field} is governed by {governor}, which is not refused"


#: ``field -> the text in _RealBackend.generate that proves it is read``.
#: ``loras`` / ``reference_video_id`` arrive as orchestrator-resolved keyword
#: arguments; ``embed_mp4_metadata`` is the app-side recipe embed and is not
#: read inside the backend at all.
_HONOURED_READS = {
    "loras": "lora_paths",
    "reference_video_id": "reference_video_path",
}
_APP_SIDE_HONOURED_FIELDS = {"embed_mp4_metadata"}


def test_honoured_fields_are_read_by_generate():
    source = inspect.getsource(ltx23._RealBackend.generate)
    for field in ltx23.HONOURED_FIELDS:
        if field in _APP_SIDE_HONOURED_FIELDS:
            continue
        needle = _HONOURED_READS.get(field, f"request.{field}")
        assert needle in source, f"{field} is declared honoured but never read"


def test_ignored_fields_are_really_not_read_by_generate():
    source = inspect.getsource(ltx23._RealBackend.generate)
    for field in ltx23.IGNORED_FIELDS:
        assert f"request.{field}" not in source, f"{field} is declared ignored but read"


# --------------------------------------------------------------------------- #
# 2) the refusals
# --------------------------------------------------------------------------- #


def test_the_fixture_covers_every_refused_field():
    assert set(REQUEST_OVERRIDES_23) == {f for f, _feat, _p in ltx23.REJECT_TABLE}


@pytest.mark.parametrize("field", list(REQUEST_OVERRIDES_23))
def test_every_refused_field_raises_feature_unsupported_naming_ltx25(field):
    request = _request(**REQUEST_OVERRIDES_23[field])
    with pytest.raises(APIError) as ei:
        ltx23.reject_unsupported(request)
    assert ei.value.code == "FEATURE_UNSUPPORTED" and ei.value.status_code == 422
    assert field in ei.value.detail
    assert "LTX 2.5" in ei.value.detail


def test_a_default_request_is_accepted():
    ltx23.reject_unsupported(_request())  # no raise


def test_unsupported_features_publishes_both_refusals_in_table_order():
    assert ltx23.UNSUPPORTED_FEATURES == ("keep_resident_embeddings", "alpha_gen")
    assert "alpha_gen" in ltx23.UNSUPPORTED_FEATURES


def _activate(client, base_model: str) -> None:
    r = client.post("/api/v1/pipeline/load", json={"base_model": base_model})
    assert r.status_code == 200, r.text
    assert client.app_context.pipeline_manager.active_base_model == base_model


@pytest.mark.parametrize("field", list(REQUEST_OVERRIDES_23))
def test_ltx23_refuses_over_http_before_any_lookup(two_family_client, field):
    """The same refusals through POST /generate on LTX 2.3: 422
    FEATURE_UNSUPPORTED (not ALPHA_GEN_INVALID, and not a 404 for the missing
    reference video — the engine scope is answered first), no job created."""
    _activate(two_family_client, "LTX23")
    before = len(two_family_client.get("/api/v1/jobs").json())
    body = {"prompt": "a quiet harbour at first light", "num_frames": 9}
    body.update(REQUEST_OVERRIDES_23[field])
    r = two_family_client.post("/api/v1/generate", json=body)
    assert r.status_code == 422, r.text
    error = r.json()["error"]
    assert error["code"] == "FEATURE_UNSUPPORTED"
    assert "LTX 2.5" in error["detail"]
    assert len(two_family_client.get("/api/v1/jobs").json()) == before
