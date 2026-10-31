"""MiniMax H3 Reference (Ref2VA) per chain clip (``clips[i].references``, 台帳 §1-109).

Pins the app half of the chain's references in one place: the schema (one
entry per kind and slot within a clip, ``clips[i]: `` in the message), the
family's per-clip counts and slots (422 ``clips[i]: at most ...``), the engine
scope (both LTX families refuse the field with the H3-only sentence, from the
last row of their ``CHAIN_REJECT_TABLE``), the 404 naming
``clips[i].references[j]``, the orchestration (``run_chain_job`` resolves each
clip's paths and hands them to ``H3Runner.generate_chain`` as
``clip_reference_paths`` only when a clip has references), the mock's per-clip
counts, ``metadata["references"].clips`` and the real worker payload (a clip
with references carries them; a clip without any carries no key, so the
chain golden stays byte for byte). Also the H3 twin of LTX 2.5's nested
``ChainClip`` audit.
"""

from __future__ import annotations

import inspect

import pytest
from pydantic import ValidationError

from api.models import ChainClip, GenerateChainRequest
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
from test_h3_ref_api import H3_ONLY_DETAIL, _refs, _upload_image, _upload_video

#: The smallest H3 chain the timeline admits (clip 0 >= 22, clip 1 >= 39).
CHAIN_BODY = {"prompt": PROMPT, "width": 640, "height": 384}


def _clips(*references: list[dict]) -> list[dict]:
    """One clip per argument, clip 0 of 22 frames and the rest of 39."""
    return [
        {"num_frames": 22 if i == 0 else 39, "references": refs}
        for i, refs in enumerate(references)
    ]


def _post_chain(client, clips, **extra):
    return client.post("/api/v1/generate/chain", json={**CHAIN_BODY, "clips": clips, **extra})


def _job_count(client) -> int:
    return len(client.get("/api/v1/jobs").json())


# --------------------------------------------------------------------------- #
# the schema and the four-way classification
# --------------------------------------------------------------------------- #


def test_a_clip_has_no_references_by_default():
    chain = GenerateChainRequest(prompt=PROMPT, clips=[{"num_frames": 22}, {"num_frames": 39}])
    assert chain.clips[0].references == [] and chain.clips[1].references == []
    assert chain.model_dump()["clips"][0]["references"] == []


def test_a_kind_and_slot_pair_appears_once_per_clip():
    with pytest.raises(ValidationError, match=r"clips\[1\]: references has two picture entries for slot 2"):
        GenerateChainRequest(prompt=PROMPT, clips=_clips([], _refs("picture", 2)[1:] * 2))
    # the same card number in two clips is two different cards
    GenerateChainRequest(prompt=PROMPT, clips=_clips(_refs("picture", 1), _refs("picture", 1)))


def test_to_clip_request_copies_the_clips_references():
    chain = GenerateChainRequest(prompt=PROMPT, clips=_clips(_refs("picture", 2, "a"), _refs("audio", 1, "b")))
    assert [r.id for r in chain.to_clip_request(0).references] == ["a", "a"]
    assert [r.kind for r in chain.to_clip_request(1).references] == ["audio"]


def test_both_ltx_chain_tables_refuse_clip_references_last():
    from services.engines.ltx import adapter as ltx23
    from services.engines.ltx25 import adapter as ltx25

    with_refs = GenerateChainRequest(prompt=PROMPT, clips=_clips([], _refs("picture", 1)))
    without = GenerateChainRequest(prompt=PROMPT, clips=_clips([], []))
    for adapter in (ltx23, ltx25):
        field, feature, predicate = adapter.CHAIN_REJECT_TABLE[-1]
        assert (field, feature) == ("clips", "references")
        assert predicate(with_refs) and not predicate(without)
    # H3 honours it: no chain row of its own names the feature.
    assert "references" not in {feat for _f, feat, _p in h3.CHAIN_REJECT_TABLE}


def test_the_nested_chain_clip_fields_are_all_accounted_for_on_h3():
    """The H3 twin of LTX 2.5's nested audit: the chain audit sees only the
    top level, and ``clips`` is classified as honoured (and refused for a start
    image), which is a promise about ``ChainClip``'s own fields too."""
    assert set(ChainClip.model_fields) == {"prompt", "num_frames", "conditioning_images", "references"}
    source = inspect.getsource(h3._RealBackendH3.generate_chain)
    assert "chain.clip_prompt(i)" in source  # prompt (the shared one + the clip's own)
    assert "num_frames" in source
    assert "_clip_reference_payloads(chain, clip_reference_paths)" in source  # references
    # conditioning_images is the refused one (``chain_start_image``), never read.
    assert ("clips", "chain_start_image") == h3.CHAIN_REJECT_TABLE[0][:2]
    assert "conditioning_images" not in source


# --------------------------------------------------------------------------- #
# the family's per-clip counts and slots (422, before any upload is looked up)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "references, message",
    [
        (_refs("picture", 10), "clips[1]: at most 9 picture references are supported"),
        (_refs("video", 4), "clips[1]: at most 3 video references are supported"),
        (
            _refs("picture", 9) + _refs("video", 3) + _refs("audio", 1),
            "clips[1]: at most 12 references are supported in all",
        ),
        ([{"kind": "audio", "id": "x", "slot": 4}], "clips[1]: the audio reference slot must be between 1 and 3"),
    ],
)
def test_per_clip_counts_and_slots_over_the_limits_are_422(three_family_client, references, message):
    _activate(three_family_client, "MiniMaxH3")
    r = _post_chain(three_family_client, _clips([], references))
    assert r.status_code == 422, r.text
    assert r.json()["error"]["code"] == "VALIDATION_ERROR"
    assert message in r.text


def test_the_limits_are_per_clip_not_per_chain():
    # 9 pictures in each of two clips: 18 in the chain, 9 per clip.
    _validated_chain(clips=_clips(_refs("picture", 9), _refs("picture", 9)))


def test_a_duplicate_within_a_clip_is_422_over_http(three_family_client):
    _activate(three_family_client, "MiniMaxH3")
    r = _post_chain(three_family_client, _clips(_refs("picture", 1) * 2, []))
    assert r.status_code == 422, r.text
    assert r.json()["error"]["code"] == "VALIDATION_ERROR"
    assert "clips[0]: references has two picture entries for slot 1" in r.text


# --------------------------------------------------------------------------- #
# the engine scope and the uploads
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("base_model", ["LTX23", "LTX25"])
def test_ltx_refuses_clip_references_with_the_h3_only_sentence(three_family_client, png_bytes, base_model):
    _activate(three_family_client, base_model)
    image_id = _upload_image(three_family_client, png_bytes)
    before = _job_count(three_family_client)
    r = three_family_client.post(
        "/api/v1/generate/chain",
        json={
            "prompt": PROMPT,
            "width": 384,
            "height": 256,
            "overlap_frames": 2,
            "clips": [
                {"num_frames": 25},
                {"num_frames": 25, "references": [{"kind": "picture", "id": image_id, "slot": 1}]},
            ],
        },
    )
    assert r.status_code == 422, r.text
    error = r.json()["error"]
    assert error["code"] == "FEATURE_UNSUPPORTED"
    assert "'references'" in error["message"]
    assert error["detail"] == H3_ONLY_DETAIL
    assert _job_count(three_family_client) == before


@pytest.mark.parametrize("kind", ["picture", "video", "audio"])
def test_an_unknown_id_is_404_naming_the_clip(three_family_client, png_bytes, kind):
    _activate(three_family_client, "MiniMaxH3")
    image_id = _upload_image(three_family_client, png_bytes)
    before = _job_count(three_family_client)
    clips = _clips(
        [{"kind": "picture", "id": image_id, "slot": 1}],
        [{"kind": "picture", "id": image_id, "slot": 1}, {"kind": kind, "id": "nope", "slot": 2}],
    )
    r = _post_chain(three_family_client, clips)
    assert r.status_code == 404, r.text
    error = r.json()["error"]
    assert error["code"] == "REFERENCE_NOT_FOUND"
    assert error["message"] == f"clips[1].references[1].id not found: nope ({kind} slot 2)"
    assert _job_count(three_family_client) == before


def test_the_single_404_message_is_unchanged(three_family_client):
    _activate(three_family_client, "MiniMaxH3")
    r = three_family_client.post(
        "/api/v1/generate",
        json={"prompt": PROMPT, "width": 640, "height": 384, "num_frames": 22,
              "references": [{"kind": "picture", "id": "nope", "slot": 1}]},
    )
    assert r.status_code == 404, r.text
    assert r.json()["error"]["message"] == "references[0].id not found: nope (picture slot 1)"


# --------------------------------------------------------------------------- #
# orchestration, the mock and metadata.json
# --------------------------------------------------------------------------- #


def _spy_runner(client) -> list[dict]:
    pm = client.app_context.pipeline_manager
    orig = pm.runner.generate_chain
    calls: list[dict] = []

    def wrapped(*a, **k):
        calls.append(dict(k))
        return orig(*a, **k)

    pm.runner.generate_chain = wrapped  # type: ignore[method-assign]
    return calls


def test_mock_chain_with_references_records_each_clip(three_family_client, tmp_path, png_bytes):
    _activate(three_family_client, "MiniMaxH3")
    image_id = _upload_image(three_family_client, png_bytes)
    video_id = _upload_video(three_family_client)
    audio_id = _upload_audio(three_family_client, tmp_path, 2.0)
    calls = _spy_runner(three_family_client)
    clips = _clips(
        [{"kind": "picture", "id": image_id, "slot": 1}],
        [],
        [
            {"kind": "picture", "id": image_id, "slot": 2},
            {"kind": "video", "id": video_id, "slot": 1, "max_sec": 4.5},
            {"kind": "audio", "id": audio_id, "slot": 1},
        ],
    )
    r = _post_chain(three_family_client, clips)
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    assert three_family_client.get(f"/api/v1/jobs/{job_id}").json()["status"] == "completed"

    # run_chain_job resolved one path list per clip, in each clip's order.
    ctx = three_family_client.app_context
    (call,) = calls
    assert call["clip_reference_paths"] == [
        [ctx.upload_store.path_for(image_id)],
        [],
        [
            ctx.upload_store.path_for(image_id),
            ctx.video_upload_store.path_for(video_id),
            ctx.audio_upload_store.path_for(audio_id),
        ],
    ]

    meta = _metadata(three_family_client, job_id)
    assert meta["references"] == {
        "clips": [
            {
                "pictures": 1, "videos": 0, "audios": 0,
                "inputs": [{"kind": "picture", "slot": 1, "id": image_id, "max_sec": None}],
            },
            None,
            {
                "pictures": 1, "videos": 1, "audios": 1,
                "inputs": [
                    {"kind": "picture", "slot": 2, "id": image_id, "max_sec": None},
                    {"kind": "video", "slot": 1, "id": video_id, "max_sec": 4.5},
                    {"kind": "audio", "slot": 1, "id": audio_id, "max_sec": None},
                ],
            },
        ]
    }
    # The mock's done.chain.clips[i]: the counts only on a clip with references.
    engine_clips = meta["h3"]["clips"]
    assert engine_clips[0]["references"] == {"pictures": 1, "videos": 0, "audios": 0}
    assert "references" not in engine_clips[1]
    assert engine_clips[2]["references"] == {"pictures": 1, "videos": 1, "audios": 1}


def test_a_chain_without_references_passes_no_extra_keyword(three_family_client):
    _activate(three_family_client, "MiniMaxH3")
    calls = _spy_runner(three_family_client)
    r = _post_chain(three_family_client, _clips([], []))
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    assert three_family_client.get(f"/api/v1/jobs/{job_id}").json()["status"] == "completed"
    (call,) = calls
    assert "clip_reference_paths" not in call
    meta = _metadata(three_family_client, job_id)
    assert "references" not in meta
    assert all("references" not in c for c in meta["h3"]["clips"])
    assert [c["references"] for c in meta["request"]["clips"]] == [[], []]


def test_the_mock_refuses_unresolved_clip_references(tmp_path):
    chain = _validated_chain(clips=_clips([], _refs("picture", 1)))
    mock = h3._MockBackendH3.__new__(h3._MockBackendH3)
    mock._loaded = True
    with pytest.raises(RuntimeError, match="1 reference\\(s\\) requested but 0 resolved"):
        mock.generate_chain(chain, tmp_path / "job", seed=1, clip_reference_paths=[[], []])


def test_h3_runner_forwards_clip_reference_paths():
    seen: dict = {}

    class _Backend:
        loaded = True

        def generate_chain(self, chain_request, **kwargs):
            seen.update(kwargs)
            return "outcome"

    runner = h3.H3Runner.__new__(h3.H3Runner)
    runner._backend = _Backend()
    paths = [["a.png"], []]
    assert runner.generate_chain("chain", output_dir="out", seed=3, clip_reference_paths=paths) == "outcome"
    assert seen["clip_reference_paths"] is paths
    assert seen["seed"] == 3


def test_h3_runner_chain_signature_is_ltx_runners_plus_clip_reference_paths():
    """The shared thirteen keywords are copied, not inherited: if
    ``run_chain_job`` ever passes a new one to every runner, this fails rather
    than an H3 chain dying with a TypeError at run time."""
    from services.engines.ltx.adapter import LTXRunner

    ltx = list(inspect.signature(LTXRunner.generate_chain).parameters)
    ours = list(inspect.signature(h3.H3Runner.generate_chain).parameters)
    assert ours == ltx + ["clip_reference_paths"]
    for backend in (h3._RealBackendH3, h3._MockBackendH3):
        assert list(inspect.signature(backend.generate_chain).parameters) == ours


# --------------------------------------------------------------------------- #
# the real worker payload
# --------------------------------------------------------------------------- #


def test_real_chain_payload_without_references_is_the_golden(tmp_path):
    captured: list[dict] = []
    be = _capturing_backend(captured, CHAIN_DONE_EVENT)
    be.generate_chain(_validated_chain(), tmp_path / "job", seed=42, clip_reference_paths=None)
    (payload,) = captured
    assert list(payload) == GOLDEN_CHAIN_KEYS_H3
    assert all(list(clip) == ["prompt", "num_frames"] for clip in payload["clips"])


def test_real_chain_payload_carries_references_only_on_their_clip(tmp_path):
    pic, clip_mp4, voice = tmp_path / "a.png", tmp_path / "b.mp4", tmp_path / "c.wav"
    captured: list[dict] = []
    chain = _validated_chain(
        clips=_clips(
            [],
            [
                {"kind": "picture", "id": "img-1", "slot": 2},
                {"kind": "video", "id": "vid-1", "slot": 1, "max_sec": 3},
                {"kind": "audio", "id": "aud-1", "slot": 1, "max_sec": 2.5},
            ],
            [],
        )
    )
    be = _capturing_backend(captured, CHAIN_DONE_EVENT)
    be.generate_chain(chain, tmp_path / "job", seed=42, clip_reference_paths=[[], [pic, clip_mp4, voice], []])
    (payload,) = captured
    assert list(payload) == GOLDEN_CHAIN_KEYS_H3
    assert "references" not in payload
    first, second, third = payload["clips"]
    assert list(first) == ["prompt", "num_frames"] and list(third) == ["prompt", "num_frames"]
    assert list(second) == ["prompt", "num_frames", "references"]
    # picture -> video -> audio, the clip's order; max_sec only where set.
    assert second["references"] == [
        {"kind": "picture", "path": str(pic), "slot": 2},
        {"kind": "video", "path": str(clip_mp4), "slot": 1, "max_sec": 3.0},
        {"kind": "audio", "path": str(voice), "slot": 1, "max_sec": 2.5},
    ]


@pytest.mark.parametrize(
    "clip_reference_paths, message",
    [
        (None, "1 reference\\(s\\) requested but 0 resolved"),
        ([[], []], "1 reference\\(s\\) requested but 0 resolved"),
        ([[], ["a.png"], []], "2 clip\\(s\\) in the chain but 3 reference path list\\(s\\) resolved"),
    ],
)
def test_real_chain_payload_refuses_unresolved_references(tmp_path, clip_reference_paths, message):
    chain = _validated_chain(clips=_clips([], _refs("picture", 1)))
    captured: list[dict] = []
    with pytest.raises(RuntimeError, match=message):
        _capturing_backend(captured, CHAIN_DONE_EVENT).generate_chain(
            chain, tmp_path / "job", seed=1, clip_reference_paths=clip_reference_paths
        )
    assert captured == []
