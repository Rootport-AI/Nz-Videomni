"""MiniMax H3 Reference (Ref2VA, ``references``) over the mock backend.

Pins the API half of §1-107 in one place: the request type (one entry per kind
and slot, no ``max_sec`` on a picture), the family's counts and slot ranges
(422), the engine scope (both LTX families refuse it with the H3-only
sentence), the 404 for an id its kind's store does not hold,
``metadata["references"]`` (the mock's counts + the app's ``inputs``), and the
real worker payload (``references`` sits right after ``source_audio``; a
request without it is today's golden, byte for byte). The four-way audit that
places ``references`` in each adapter's tables is the adapters' own suites'.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from api.models import GenerateRequest
from services.engines.h3 import adapter as h3
from test_engine_h3_adapter import (
    GOLDEN_GENERATE_KEYS_H3,
    PROMPT,
    _activate,
    _capturing_backend,
    _metadata,
    _validated,
)
from test_h3_a2v_api import _upload as _upload_audio

H3_ONLY_DETAIL = "参照素材（Ref2VA）は MiniMax H3 専用です。"
BODY = {"prompt": PROMPT, "width": 640, "height": 384, "num_frames": 22}
FAKE_MP4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64


def _upload_image(client, png_bytes) -> str:
    r = client.post("/api/v1/upload/image", files={"file": ("ref.png", png_bytes, "image/png")})
    assert r.status_code == 200, r.text
    return r.json()["image_id"]


def _upload_video(client) -> str:
    r = client.post("/api/v1/upload/video", files={"file": ("ref.mp4", FAKE_MP4, "video/mp4")})
    assert r.status_code == 200, r.text
    return r.json()["video_id"]


def _refs(kind: str, count: int, ref_id: str = "x") -> list[dict]:
    return [{"kind": kind, "id": ref_id, "slot": slot} for slot in range(1, count + 1)]


def _post(client, references, **extra):
    return client.post("/api/v1/generate", json={**BODY, "references": references, **extra})


# --------------------------------------------------------------------------- #
# the four-way classification
# --------------------------------------------------------------------------- #


def test_references_is_honoured_on_h3_and_refused_on_both_ltx():
    from services.engines.ltx import adapter as ltx23
    from services.engines.ltx25 import adapter as ltx25

    assert "references" in h3.HONOURED_FIELDS
    assert "references" not in {f for f, _feat, _p in h3.REJECT_TABLE}
    assert "references" not in h3.UNSUPPORTED_FEATURES
    for adapter in (ltx23, ltx25):
        assert ("references", "references") in {(f, feat) for f, feat, _p in adapter.REJECT_TABLE}
        assert "references" in adapter.UNSUPPORTED_FEATURES
    assert ltx23.H3_ONLY_REFUSALS["references"] == H3_ONLY_DETAIL


# --------------------------------------------------------------------------- #
# the request type
# --------------------------------------------------------------------------- #


def test_the_default_is_no_references():
    assert GenerateRequest(prompt=PROMPT).references == []


def test_a_kind_and_slot_pair_appears_once():
    with pytest.raises(ValidationError, match="two picture entries for slot 2"):
        GenerateRequest(
            prompt=PROMPT,
            references=[
                {"kind": "picture", "id": "a", "slot": 2},
                {"kind": "picture", "id": "b", "slot": 2},
            ],
        )
    # the same number on two kinds is two different cards
    GenerateRequest(
        prompt=PROMPT,
        references=[{"kind": "picture", "id": "a", "slot": 1}, {"kind": "video", "id": "b", "slot": 1}],
    )


def test_a_picture_takes_no_max_sec():
    with pytest.raises(ValidationError, match="picture reference takes no max_sec"):
        GenerateRequest(prompt=PROMPT, references=[{"kind": "picture", "id": "a", "slot": 1, "max_sec": 2.0}])
    GenerateRequest(prompt=PROMPT, references=[{"kind": "audio", "id": "a", "slot": 1, "max_sec": 2.0}])


@pytest.mark.parametrize(
    "entry",
    [
        {"kind": "picture", "id": "", "slot": 1},
        {"kind": "picture", "id": "a", "slot": 0},
        {"kind": "video", "id": "a", "slot": 1, "max_sec": 0},
        {"kind": "text", "id": "a", "slot": 1},
        {"kind": "picture", "path": "C:/x.png", "slot": 1},
    ],
)
def test_malformed_entries_are_refused(entry):
    with pytest.raises(ValidationError):
        GenerateRequest(prompt=PROMPT, references=[entry])


def test_an_infinite_or_nan_max_sec_is_422_over_http(three_family_client):
    """``Infinity`` / ``NaN`` are valid to Starlette's ``json.loads``; the field
    refuses them up front (422) rather than leaving the worker to fail the job."""
    _activate(three_family_client, "MiniMaxH3")
    for literal in ("Infinity", "NaN"):
        entry = {"kind": "video", "id": "x", "slot": 1, "max_sec": "@"}
        raw = json.dumps({**BODY, "references": [entry]}).replace('"@"', literal)
        r = three_family_client.post(
            "/api/v1/generate", content=raw, headers={"Content-Type": "application/json"}
        )
        assert r.status_code == 422, (literal, r.text)
        assert r.json()["error"]["code"] == "VALIDATION_ERROR"


def test_the_duplicate_and_picture_length_rules_are_422_over_http(three_family_client):
    _activate(three_family_client, "MiniMaxH3")
    dup = _post(three_family_client, _refs("picture", 1) * 2)
    assert dup.status_code == 422, dup.text
    assert dup.json()["error"]["code"] == "VALIDATION_ERROR"
    long_picture = _post(three_family_client, [{"kind": "picture", "id": "x", "slot": 1, "max_sec": 1.0}])
    assert long_picture.status_code == 422, long_picture.text
    assert long_picture.json()["error"]["code"] == "VALIDATION_ERROR"


# --------------------------------------------------------------------------- #
# the family's counts and slots (422, before any upload is looked up)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "references, message",
    [
        (_refs("picture", 10), "at most 9 picture references are supported"),
        (_refs("video", 4), "at most 3 video references are supported"),
        (_refs("audio", 4), "at most 3 audio references are supported"),
        (
            _refs("picture", 9) + _refs("video", 3) + _refs("audio", 1),
            "at most 12 references are supported in all",
        ),
        ([{"kind": "picture", "id": "x", "slot": 10}], "the picture reference slot must be between 1 and 9"),
        ([{"kind": "video", "id": "x", "slot": 4}], "the video reference slot must be between 1 and 3"),
        ([{"kind": "audio", "id": "x", "slot": 4}], "the audio reference slot must be between 1 and 3"),
    ],
)
def test_counts_and_slots_over_the_limits_are_422(three_family_client, references, message):
    _activate(three_family_client, "MiniMaxH3")
    r = _post(three_family_client, references)
    assert r.status_code == 422, r.text
    error = r.json()["error"]
    assert error["code"] == "VALIDATION_ERROR"
    assert message in r.text


def test_the_limits_are_the_published_ones():
    limits = h3.REQUEST_RULES.reference
    assert (limits.max_images, limits.max_videos, limits.max_audios, limits.max_total) == (9, 3, 3, 12)
    full = _validated(references=_refs("picture", 9) + _refs("video", 3))
    assert h3.REQUEST_RULES.violation(full) is None


# --------------------------------------------------------------------------- #
# the engine scope and the uploads
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("base_model", ["LTX23", "LTX25"])
def test_ltx_refuses_references_with_the_h3_only_sentence(three_family_client, png_bytes, base_model):
    _activate(three_family_client, base_model)
    image_id = _upload_image(three_family_client, png_bytes)
    r = three_family_client.post(
        "/api/v1/generate",
        json={"prompt": PROMPT, "num_frames": 25, "references": [{"kind": "picture", "id": image_id, "slot": 1}]},
    )
    assert r.status_code == 422, r.text
    error = r.json()["error"]
    assert error["code"] == "FEATURE_UNSUPPORTED"
    assert error["detail"] == H3_ONLY_DETAIL


@pytest.mark.parametrize("kind", ["picture", "video", "audio"])
def test_an_unknown_id_is_404(three_family_client, kind):
    _activate(three_family_client, "MiniMaxH3")
    before = len(three_family_client.get("/api/v1/jobs").json())
    r = _post(three_family_client, [{"kind": kind, "id": "nope", "slot": 1}])
    assert r.status_code == 404, r.text
    error = r.json()["error"]
    assert error["code"] == "REFERENCE_NOT_FOUND"
    assert error["message"] == f"references[0].id not found: nope ({kind} slot 1)"
    assert len(three_family_client.get("/api/v1/jobs").json()) == before


def test_an_id_of_another_kind_is_404(three_family_client, png_bytes):
    _activate(three_family_client, "MiniMaxH3")
    image_id = _upload_image(three_family_client, png_bytes)
    r = _post(three_family_client, [{"kind": "video", "id": image_id, "slot": 1}])
    assert r.status_code == 404, r.text
    assert r.json()["error"]["code"] == "REFERENCE_NOT_FOUND"


# --------------------------------------------------------------------------- #
# metadata.json
# --------------------------------------------------------------------------- #


def test_mock_metadata_carries_the_counts_and_the_inputs(three_family_client, tmp_path, png_bytes):
    _activate(three_family_client, "MiniMaxH3")
    image_id = _upload_image(three_family_client, png_bytes)
    video_id = _upload_video(three_family_client)
    audio_id = _upload_audio(three_family_client, tmp_path, 2.0)
    references = [
        {"kind": "picture", "id": image_id, "slot": 1},
        {"kind": "picture", "id": image_id, "slot": 3},
        {"kind": "video", "id": video_id, "slot": 1, "max_sec": 4.5},
        {"kind": "audio", "id": audio_id, "slot": 2},
    ]
    r = _post(three_family_client, references)
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    assert three_family_client.get(f"/api/v1/jobs/{job_id}").json()["status"] == "completed"
    meta = _metadata(three_family_client, job_id)
    assert meta["references"] == {
        "pictures": 2,
        "videos": 1,
        "audios": 1,
        "inputs": [
            {"kind": "picture", "slot": 1, "id": image_id, "max_sec": None},
            {"kind": "picture", "slot": 3, "id": image_id, "max_sec": None},
            {"kind": "video", "slot": 1, "id": video_id, "max_sec": 4.5},
            {"kind": "audio", "slot": 2, "id": audio_id, "max_sec": None},
        ],
    }
    assert "h3" not in meta  # the counts were lifted out; the mock has nothing else


def test_a_job_without_references_has_no_references_record(three_family_client):
    _activate(three_family_client, "MiniMaxH3")
    r = three_family_client.post("/api/v1/generate", json=BODY)
    assert r.status_code == 202, r.text
    meta = _metadata(three_family_client, r.json()["job_id"])
    assert "references" not in meta and "h3" not in meta
    assert meta["request"]["references"] == []


# --------------------------------------------------------------------------- #
# the real worker payload
# --------------------------------------------------------------------------- #


def test_real_payload_without_references_is_the_golden(tmp_path):
    captured: list[dict] = []
    _capturing_backend(captured).generate(_validated(), tmp_path / "job", seed=1, reference_paths=[])
    (payload,) = captured
    assert list(payload) == GOLDEN_GENERATE_KEYS_H3
    assert "references" not in payload


def test_real_payload_places_references_after_source_audio(tmp_path):
    audio = tmp_path / "voice.wav"
    pic = tmp_path / "a.png"
    clip = tmp_path / "b.mp4"
    captured: list[dict] = []
    req = _validated(
        source_audio={"audio_id": "aud-1"},
        references=[
            {"kind": "picture", "id": "img-1", "slot": 2},
            {"kind": "video", "id": "vid-1", "slot": 1, "max_sec": 3},
        ],
    )
    _capturing_backend(captured).generate(
        req, tmp_path / "job", seed=1, source_audio_path=audio, reference_paths=[pic, clip]
    )
    (payload,) = captured
    assert payload["references"] == [
        {"kind": "picture", "path": str(pic), "slot": 2},
        {"kind": "video", "path": str(clip), "slot": 1, "max_sec": 3.0},
    ]
    keys = list(payload)
    assert keys[-3:] == ["source_audio", "references", "output_path"]
    assert [k for k in keys if k not in ("source_audio", "references")] == GOLDEN_GENERATE_KEYS_H3


def test_real_payload_refuses_unresolved_references(tmp_path):
    req = _validated(references=[{"kind": "picture", "id": "img-1", "slot": 1}])
    with pytest.raises(RuntimeError, match="1 reference\\(s\\) requested but 0 resolved"):
        _capturing_backend([]).generate(req, tmp_path / "job", seed=1)
