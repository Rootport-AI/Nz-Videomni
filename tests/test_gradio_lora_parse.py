"""Unit tests for the prompt-embedded ``<lora:name:weight>`` parsing (S2).

Two layers, both server-free:
  * ``parse_prompt_loras`` in isolation (weight defaults / case / clamp / unknown
    / dedup / token stripping / whitespace collapse);
  * ``make_generate_handler`` driven over an ``httpx.MockTransport`` to prove the
    payload contract — byte-identical on the token-free path (and NO GET /loras
    call there), style-only loras with no reference_video_id, unknown-name abort
    with zero generate/upload calls, and merge with the adapter dropdown.
"""

from __future__ import annotations

import json

import httpx
import pytest

from gradio_ui.adapters import ADAPTER_NONE
from gradio_ui.handlers import (
    LORA_AUDIO_WEIGHT_MAX,
    LORA_WEIGHT_MAX,
    LORA_WEIGHT_MIN,
    make_generate_handler,
    parse_prompt_loras,
)
from gradio_ui.api_client import ApiClient
from gradio_ui.presets import KF_MAX_SLOTS

KNOWN = ["neon-city", "AnimeStyle", "portrait"]


def _make_client(handler, *, api_key: str | None = "secret") -> ApiClient:
    transport = httpx.MockTransport(handler)
    hc = httpx.Client(transport=transport)
    return ApiClient("http://test", api_key=api_key, client=hc)


def _kf_args(*slots):
    """The whole keyframe grid as generate()'s single ``kf_slots`` argument:
    a KF_MAX_SLOTS-long list of (enabled, image, frame_idx, strength) tuples,
    padded with empty slots (same helper as tests/test_gradio_handlers.py)."""
    filled = list(slots) + [(False, None, 0, 0.8)] * (KF_MAX_SLOTS - len(slots))
    return filled[:KF_MAX_SLOTS]


def _adapter_args(adapter=ADAPTER_NONE, strength=1.0, ref_path=None, config=None,
                  control_adherence=1.0, reference_strength=1.0):
    return [adapter, strength, control_adherence, reference_strength, ref_path, config]


def _run_until_job_started(gen):
    first = next(gen)
    gen.close()
    return first


# --------------------------------------------------------------------------- #
# parse_prompt_loras in isolation.
# --------------------------------------------------------------------------- #
def test_weight_omitted_defaults_to_one():
    cleaned, loras, err = parse_prompt_loras("hi <lora:neon-city> there", KNOWN)
    assert err is None
    assert loras == [{"name": "neon-city", "strength": 1.0}]
    assert cleaned == "hi there"


def test_keyword_is_case_insensitive():
    _cleaned, loras, err = parse_prompt_loras("x <LORA:neon-city:0.8>", KNOWN)
    assert err is None
    assert loras == [{"name": "neon-city", "strength": 0.8}]


def test_name_match_is_case_insensitive():
    _cleaned, loras, err = parse_prompt_loras("<lora:animestyle>", KNOWN)
    assert err is None
    # Resolves to the canonical registered name, not the lower-cased token.
    assert loras == [{"name": "AnimeStyle", "strength": 1.0}]


def test_exact_match_preferred_over_case_fold():
    known = ["Foo", "foo"]
    _c, loras, err = parse_prompt_loras("<lora:foo>", known)
    assert err is None
    assert loras == [{"name": "foo", "strength": 1.0}]  # exact wins


def test_weight_clamped_above_max():
    _c, loras, err = parse_prompt_loras("<lora:neon-city:3.5>", KNOWN)
    assert err is None
    assert loras == [{"name": "neon-city", "strength": LORA_WEIGHT_MAX}]


def test_weight_clamped_at_or_below_zero():
    _c, loras, err = parse_prompt_loras("<lora:neon-city:0>", KNOWN)
    assert err is None
    assert loras == [{"name": "neon-city", "strength": LORA_WEIGHT_MIN}]


def test_multiple_loras_in_order():
    _c, loras, err = parse_prompt_loras(
        "<lora:neon-city:0.5> and <lora:portrait:1.2>", KNOWN)
    assert err is None
    assert loras == [
        {"name": "neon-city", "strength": 0.5},
        {"name": "portrait", "strength": 1.2},
    ]


def test_duplicate_name_last_wins_single_entry():
    _c, loras, err = parse_prompt_loras(
        "<lora:neon-city:0.5> x <lora:neon-city:1.5>", KNOWN)
    assert err is None
    assert loras == [{"name": "neon-city", "strength": 1.5}]


def test_unknown_name_returns_error_and_leaves_prompt():
    prompt = "a <lora:does-not-exist> b"
    cleaned, loras, err = parse_prompt_loras(prompt, KNOWN)
    assert err is not None
    assert loras == []
    assert cleaned == prompt  # unchanged on abort


def test_token_removal_collapses_whitespace():
    cleaned, _loras, err = parse_prompt_loras("a  <lora:neon-city>  b", KNOWN)
    assert err is None
    assert cleaned == "a b"


def test_no_token_prompt_returned_unchanged():
    prompt = "just a normal prompt"
    cleaned, loras, err = parse_prompt_loras(prompt, KNOWN)
    assert err is None
    assert loras == []
    assert cleaned == prompt


# --------------------------------------------------------------------------- #
# audio_strength (3-arg tag) parsing.
# --------------------------------------------------------------------------- #
def test_three_arg_tag_parses_video_and_audio_strength():
    _c, loras, err = parse_prompt_loras("<lora:neon-city:0.8:0>", KNOWN)
    assert err is None
    assert loras == [{"name": "neon-city", "strength": 0.8, "audio_strength": 0.0}]


def test_two_arg_tag_has_no_audio_strength_key():
    _c, loras, err = parse_prompt_loras("<lora:neon-city:0.8>", KNOWN)
    assert err is None
    assert len(loras) == 1
    assert "audio_strength" not in loras[0]


def test_bare_tag_has_no_audio_strength_key():
    _c, loras, err = parse_prompt_loras("<lora:neon-city>", KNOWN)
    assert err is None
    assert loras == [{"name": "neon-city", "strength": 1.0}]
    assert "audio_strength" not in loras[0]


def test_audio_strength_clamped_above_max_fires_warning(monkeypatch):
    from gradio_ui import handlers as handlers_mod

    toasts: list[str] = []
    monkeypatch.setattr(
        handlers_mod.gr, "Warning",
        lambda message, *args, **kwargs: toasts.append(message),
    )

    _c, loras, err = parse_prompt_loras("<lora:neon-city:0.8:5.0>", KNOWN)
    assert err is None
    assert loras == [
        {"name": "neon-city", "strength": 0.8, "audio_strength": LORA_AUDIO_WEIGHT_MAX}
    ]
    assert len(toasts) == 1


def test_duplicate_name_audio_strength_last_wins():
    _c, loras, err = parse_prompt_loras(
        "<lora:neon-city:1.0:0> x <lora:neon-city:1.0>", KNOWN)
    assert err is None
    # The second (last) tag has no audio group, so it wins entirely -- the
    # merged entry carries NO audio_strength key at all.
    assert loras == [{"name": "neon-city", "strength": 1.0}]
    assert "audio_strength" not in loras[0]


def test_empty_video_slot_with_audio_is_unsupported_and_left_in_prompt():
    prompt = "a <lora:neon-city::0> b"
    cleaned, loras, err = parse_prompt_loras(prompt, KNOWN)
    assert err is None
    assert loras == []
    assert cleaned == prompt  # token didn't match at all -> untouched


# --------------------------------------------------------------------------- #
# make_generate_handler integration (payload contract).
# --------------------------------------------------------------------------- #
def test_token_free_payload_is_identical_and_makes_no_loras_call():
    paths: list[str] = []
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path.endswith("/generate"):
            captured.update(json.loads(request.content))
            return httpx.Response(200, json={"job_id": "job-1"})
        return httpx.Response(200, json={})

    api = _make_client(handler)
    generate = make_generate_handler(api)
    gen = generate(
        "A calm river", "blurry", _kf_args(),
        512, 320, False, 0, 0, 49, 24.0, -1,
    )
    _run_until_job_started(gen)
    assert "/api/v1/loras" not in paths  # no lookup when no token present
    assert "loras" not in captured
    assert captured["prompt"] == "A calm river"


def test_style_lora_only_payload_no_reference_video():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/loras"):
            return httpx.Response(200, json={"loras": [
                {"name": "neon-city", "kind": "style", "has_thumbnail": False,
                 "exists": True, "source": "scan"},
            ]})
        assert not request.url.path.endswith("/upload/video")
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"job_id": "job-2"})

    api = _make_client(handler)
    generate = make_generate_handler(api)
    gen = generate(
        "cat <lora:neon-city:0.7>", "", _kf_args(),
        512, 320, False, 0, 0, 49, 24.0, -1,
    )
    _run_until_job_started(gen)
    assert captured["loras"] == [{"name": "neon-city", "strength": 0.7}]
    assert "reference_video_id" not in captured
    assert captured["prompt"] == "cat"


def test_unknown_prompt_lora_aborts_with_zero_generate_calls():
    posts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/loras") and request.method == "GET":
            return httpx.Response(200, json={"loras": [
                {"name": "neon-city", "kind": "style", "has_thumbnail": False,
                 "exists": True, "source": "scan"},
            ]})
        if request.method == "POST":
            posts["n"] += 1
        return httpx.Response(200, json={"job_id": "x"})

    api = _make_client(handler)
    generate = make_generate_handler(api)
    out = list(generate(
        "<lora:missing-one>", "", _kf_args(),
        512, 320, False, 0, 0, 49, 24.0, -1,
    ))
    assert posts["n"] == 0  # no generate, no upload
    progress, job_id, video = out[-1]
    assert job_id == "" and video is None


def test_adapter_and_prompt_loras_merge(tmp_path):
    vid = tmp_path / "ref.mp4"
    vid.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/loras") and request.method == "GET":
            return httpx.Response(200, json={"loras": [
                {"name": "neon-city", "kind": "style", "has_thumbnail": False,
                 "exists": True, "source": "scan"},
            ]})
        if request.url.path.endswith("/upload/video"):
            return httpx.Response(200, json={"video_id": "vid-1"})
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"job_id": "job-3"})

    api = _make_client(handler)
    generate = make_generate_handler(api)
    gen = generate(
        "sky <lora:neon-city:0.6>", "", _kf_args(),
        1280, 768, False, 0, 0, 257, 24.0, -1,
        *_adapter_args("canny-control", 1.5, str(vid), {}),
    )
    for out in gen:
        if out[1]:
            gen.close()
            break
    # adapter first, then the prompt style lora.
    assert captured["loras"] == [
        {"name": "canny-control", "strength": 1.5},
        {"name": "neon-city", "strength": 0.6},
    ]
    assert captured["reference_video_id"] == "vid-1"
    assert captured["prompt"] == "sky"
