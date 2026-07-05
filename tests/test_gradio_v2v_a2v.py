"""GUI V2V/A2V exposure (S2) — chain-handler modes, join flow, api_client
additions, i18n coverage. MockTransport only (no server, no GPU); mirrors the
style of tests/test_gradio_handlers.py.
"""

from __future__ import annotations

import json

import httpx

from gradio_ui.api_client import ApiClient
from gradio_ui.handlers import (
    MODE_A2V,
    MODE_NONE,
    MODE_V2V,
    make_chain_handler,
    make_join_handler,
)
from gradio_ui.i18n import LABELS
from gradio_ui.validation import check_chain_total, check_v2v_context


def _make_client(handler, *, api_key: str | None = "secret") -> ApiClient:
    transport = httpx.MockTransport(handler)
    hc = httpx.Client(transport=transport)
    return ApiClient("http://test", api_key=api_key, client=hc)


# _chain_args mirrors tests/test_gradio_handlers.py, extended with the ADDITIVE
# trailing mode args (mode, src_video, context_frames, src_audio) appended after
# the S6 params — the same positional order ui.py's click inputs use.
def _chain_args(prompt="Base prompt", negative="", width=1280, height=768,
                crop_enabled=False, crop_w=0, crop_h=0, fps=24.0, seed=-1,
                overlap=3, overlap_strength=0.5, clips=None, config=None,
                mode=MODE_NONE, src_video=None, context_frames=73,
                src_audio=None):
    clips = list(clips or [])
    filled = clips + [None] * (8 - len(clips))
    args = [prompt, negative, width, height, crop_enabled, crop_w, crop_h, fps, seed,
            overlap, overlap_strength]
    for i, spec in enumerate(filled[:8]):
        spec = spec or {}
        enabled = spec.get("enabled", False)
        p = spec.get("prompt", "")
        frames = spec.get("frames", 121)
        if i == 0:  # slot 1 has image + strength
            args.extend([enabled, p, frames, spec.get("image"), spec.get("strength", 0.8)])
        else:
            args.extend([enabled, p, frames])
    # config, ui_lang, poll_interval, poll_timeout_min, then the mode args.
    args.extend([config, None, None, None, mode, src_video, context_frames, src_audio])
    return args


def _run_until_started(gen):
    outs = []
    for out in gen:
        outs.append(out)
        if out[1]:
            gen.close()
            break
    return outs


def _mp4(tmp_path, name="src.mp4", size=1024):
    p = tmp_path / name
    p.write_bytes(b"\x00" * size)
    return str(p)


def _wav(tmp_path, name="src.wav", size=1024):
    p = tmp_path / name
    p.write_bytes(b"\x00" * size)
    return str(p)


# --------------------------------------------------------------------------- #
# ApiClient additions: upload_audio / join_job / fetch_joined.
# --------------------------------------------------------------------------- #
def test_upload_audio_path_and_returns_id(tmp_path):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["method"] = request.method
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"audio_id": "aud-1"})

    api = _make_client(handler)
    audio_id = api.upload_audio(_wav(tmp_path))
    assert audio_id == "aud-1"
    assert seen["url"] == "http://test/api/v1/upload/audio"
    assert seen["method"] == "POST"
    assert seen["auth"] == "Bearer secret"


def test_join_job_path_and_empty_body():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["method"] = request.method
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"job_id": "j1", "join_mode": "fade_pair"})

    api = _make_client(handler)
    resp = api.join_job("j1")
    assert resp.status_code == 200
    assert seen["url"] == "http://test/api/v1/jobs/j1/join"
    assert seen["method"] == "POST"
    assert seen["body"] == {}  # default = server's smoothed join


def test_fetch_joined_downloads_to_tempfile():
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://test/api/v1/jobs/j1/joined"
        return httpx.Response(200, content=b"JOINEDMP4")

    api = _make_client(handler)
    path = api.fetch_joined("j1")
    with open(path, "rb") as fh:
        assert fh.read() == b"JOINEDMP4"
    assert path.endswith(".mp4")


# --------------------------------------------------------------------------- #
# Chain handler: V2V mode.
# --------------------------------------------------------------------------- #
def test_chain_v2v_single_clip_uploads_video_and_payload(tmp_path):
    captured = {}
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if request.url.path.endswith("/upload/video"):
            return httpx.Response(200, json={"video_id": "vid-9"})
        captured.update(json.loads(request.content))
        return httpx.Response(202, json={"job_id": "chain-v2v"})

    api = _make_client(handler)
    chain = make_chain_handler(api)
    gen = chain(*_chain_args(
        clips=[{"enabled": True, "frames": 121}],  # 1 clip is OK with V2V
        mode=MODE_V2V, src_video=_mp4(tmp_path), context_frames=73,
    ))
    outs = _run_until_started(gen)

    assert captured["source_video"] == {"video_id": "vid-9", "context_frames": 73}
    assert "source_audio" not in captured
    assert len(captured["clips"]) == 1
    assert calls[0].endswith("/upload/video")
    assert outs[-1][1] == "chain-v2v"


def test_chain_v2v_context_ge_clip_frames_errors_zero_calls(tmp_path):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(202, json={"job_id": "x"})

    api = _make_client(handler)
    chain = make_chain_handler(api)
    # context 73 >= clip frames 73 -> must leave a NEW tail (mirror of the 422).
    out = list(chain(*_chain_args(
        clips=[{"enabled": True, "frames": 73}],
        mode=MODE_V2V, src_video=_mp4(tmp_path), context_frames=73,
    )))
    assert calls["n"] == 0
    assert out[0][1] == "" and out[0][2] is None


def test_chain_v2v_missing_video_errors_zero_calls():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(202, json={"job_id": "x"})

    api = _make_client(handler)
    chain = make_chain_handler(api)
    out = list(chain(*_chain_args(
        clips=[{"enabled": True, "frames": 121}],
        mode=MODE_V2V, src_video=None,
    )))
    assert calls["n"] == 0
    assert len(out) == 1


def test_chain_v2v_clip0_image_conflict_errors_zero_calls(tmp_path):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(202, json={"job_id": "x"})

    api = _make_client(handler)
    chain = make_chain_handler(api)
    out = list(chain(*_chain_args(
        clips=[{"enabled": True, "frames": 121, "image": _mp4(tmp_path, "kf.png")}],
        mode=MODE_V2V, src_video=_mp4(tmp_path),
    )))
    assert calls["n"] == 0  # rejected BEFORE the keyframe upload
    assert len(out) == 1


def test_chain_v2v_bad_extension_errors_zero_calls(tmp_path):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(202, json={"job_id": "x"})

    api = _make_client(handler)
    chain = make_chain_handler(api)
    out = list(chain(*_chain_args(
        clips=[{"enabled": True, "frames": 121}],
        mode=MODE_V2V, src_video=_mp4(tmp_path, "src.txt"),
    )))
    assert calls["n"] == 0
    assert len(out) == 1


# --------------------------------------------------------------------------- #
# Chain handler: A2V mode.
# --------------------------------------------------------------------------- #
def test_chain_a2v_uploads_audio_and_payload(tmp_path):
    captured = {}
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path.endswith("/upload/audio"):
            return httpx.Response(200, json={"audio_id": "aud-7"})
        captured.update(json.loads(request.content))
        return httpx.Response(202, json={"job_id": "chain-a2v"})

    api = _make_client(handler)
    chain = make_chain_handler(api)
    gen = chain(*_chain_args(
        clips=[{"enabled": True, "frames": 121}],
        mode=MODE_A2V, src_audio=_wav(tmp_path),
    ))
    outs = _run_until_started(gen)

    assert captured["source_audio"] == {"audio_id": "aud-7"}
    assert "source_video" not in captured
    assert len(captured["clips"]) == 1
    assert outs[-1][1] == "chain-a2v"


def test_chain_a2v_start_image_allowed_and_conditioning_sent(tmp_path):
    """A2V + clip-1 start image combine (API allows it; pins the close-up
    composition). The image uploads first, then the audio, then the chain."""
    captured = {}
    calls = []
    img = tmp_path / "kf.png"
    img.write_bytes(b"PNG")

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path.endswith("/upload/image"):
            return httpx.Response(200, json={"image_id": "img-1"})
        if request.url.path.endswith("/upload/audio"):
            return httpx.Response(200, json={"audio_id": "aud-1"})
        captured.update(json.loads(request.content))
        return httpx.Response(202, json={"job_id": "chain-a2v-i"})

    api = _make_client(handler)
    chain = make_chain_handler(api)
    gen = chain(*_chain_args(
        clips=[{"enabled": True, "frames": 121, "image": str(img), "strength": 0.9}],
        mode=MODE_A2V, src_audio=_wav(tmp_path),
    ))
    _run_until_started(gen)

    assert calls[0].endswith("/upload/image")
    assert calls[1].endswith("/upload/audio")
    cond = captured["clips"][0]["conditioning_images"]
    assert cond == [{"image_id": "img-1", "frame_idx": 0, "strength": 0.9}]
    assert captured["source_audio"] == {"audio_id": "aud-1"}


def test_chain_a2v_two_clips_errors_zero_calls(tmp_path):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(202, json={"job_id": "x"})

    api = _make_client(handler)
    chain = make_chain_handler(api)
    out = list(chain(*_chain_args(
        clips=[{"enabled": True, "frames": 121}, {"enabled": True, "frames": 121}],
        mode=MODE_A2V, src_audio=_wav(tmp_path),
    )))
    assert calls["n"] == 0
    assert len(out) == 1


def test_chain_a2v_missing_audio_errors_zero_calls():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(202, json={"job_id": "x"})

    api = _make_client(handler)
    chain = make_chain_handler(api)
    out = list(chain(*_chain_args(
        clips=[{"enabled": True, "frames": 121}],
        mode=MODE_A2V, src_audio=None,
    )))
    assert calls["n"] == 0
    assert len(out) == 1


def test_chain_a2v_bad_extension_errors_zero_calls(tmp_path):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(202, json={"job_id": "x"})

    api = _make_client(handler)
    chain = make_chain_handler(api)
    out = list(chain(*_chain_args(
        clips=[{"enabled": True, "frames": 121}],
        mode=MODE_A2V, src_audio=_wav(tmp_path, "notes.txt"),
    )))
    assert calls["n"] == 0
    assert len(out) == 1


# --------------------------------------------------------------------------- #
# Chain handler: mode "none" regression — payload byte-shape unchanged.
# --------------------------------------------------------------------------- #
def test_chain_mode_none_payload_has_no_source_keys():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(202, json={"job_id": "chain-none"})

    api = _make_client(handler)
    chain = make_chain_handler(api)
    gen = chain(*_chain_args(clips=[
        {"enabled": True, "frames": 121},
        {"enabled": True, "frames": 121},
    ]))
    _run_until_started(gen)
    assert "source_video" not in captured
    assert "source_audio" not in captured


def test_chain_mode_none_single_clip_still_errors():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(202, json={"job_id": "x"})

    api = _make_client(handler)
    chain = make_chain_handler(api)
    out = list(chain(*_chain_args(clips=[{"enabled": True, "frames": 121}])))
    assert calls["n"] == 0
    assert len(out) == 1


# --------------------------------------------------------------------------- #
# Join handler.
# --------------------------------------------------------------------------- #
def test_join_happy_path_posts_then_fetches():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.method == "POST":
            return httpx.Response(200, json={
                "job_id": "j1", "join_mode": "handle_crossfade",
                "joined_path": "outputs/j1/joined.mp4", "source_normalized": True,
            })
        return httpx.Response(200, content=b"JOINED")

    api = _make_client(handler)
    join = make_join_handler(api)
    outs = list(join("j1", True))

    assert calls == [("POST", "/api/v1/jobs/j1/join"),
                     ("GET", "/api/v1/jobs/j1/joined")]
    msg, video = outs[-1]
    assert "handle_crossfade" in msg and "j1" in msg
    assert video and video.endswith(".mp4")


def test_join_smoothing_unchecked_no_api_call():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={})

    api = _make_client(handler)
    join = make_join_handler(api)
    outs = list(join("j1", False))
    assert calls["n"] == 0
    assert outs[-1][1] is None


def test_join_no_job_id_no_api_call():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={})

    api = _make_client(handler)
    join = make_join_handler(api)
    outs = list(join("", True))
    assert calls["n"] == 0
    assert outs[-1][1] is None


def test_join_error_envelope_localized_hint():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"error": {
            "code": "JOB_NOT_JOINABLE",
            "message": "job is not a V2V continuation job (nothing to join)",
        }})

    api = _make_client(handler)
    join = make_join_handler(api)
    outs = list(join("j1", True, 300, "ja"))
    msg, video = outs[-1]
    assert video is None
    assert msg == LABELS["ja"]["apierr_JOB_NOT_JOINABLE"]


# --------------------------------------------------------------------------- #
# validation.check_v2v_context (server 422 mirror).
# --------------------------------------------------------------------------- #
def test_check_v2v_context_valid_returns_none():
    assert check_v2v_context(73, 121) is None


def test_check_v2v_context_not_8n1():
    assert check_v2v_context(74, 121) is not None


def test_check_v2v_context_out_of_bounds():
    assert check_v2v_context(17, 121) is not None       # < 25
    assert check_v2v_context(153, 481) is not None      # > 145 (fallback max)


def test_check_v2v_context_respects_config_limits():
    cfg = {"limits": {"v2v_context_frames_min": 25, "v2v_context_frames_max": 105}}
    assert check_v2v_context(113, 481, config=cfg) is not None   # > config max
    assert check_v2v_context(105, 481, config=cfg) is None


def test_check_v2v_context_ge_clip0():
    assert check_v2v_context(73, 73) is not None


def test_check_chain_total_with_source_context_px():
    # 1-clip V2V layout is valid geometry through the shared chain_math mirror.
    assert check_chain_total([121], 24.0, 3, source_context_px=73) is None


# --------------------------------------------------------------------------- #
# i18n coverage: every new v2v_/a2v_/apierr key exists in BOTH languages.
# --------------------------------------------------------------------------- #
def test_i18n_v2v_a2v_keys_present_in_both_languages():
    new_keys = [k for k in LABELS["en"]
                if k.startswith(("v2v_", "a2v_"))
                or k in ("apierr_SOURCE_VIDEO_NOT_FOUND", "apierr_SOURCE_VIDEO_TOO_SHORT",
                         "apierr_SOURCE_AUDIO_NOT_FOUND", "apierr_SOURCE_AUDIO_TOO_SHORT",
                         "apierr_JOB_NOT_JOINABLE", "apierr_JOIN_FAILED",
                         "apierr_JOINED_NOT_READY")]
    assert new_keys, "expected the new v2v_/a2v_ label keys to exist"
    for key in new_keys:
        assert key in LABELS["ja"], f"missing ja translation: {key}"
        assert LABELS["ja"][key], f"empty ja translation: {key}"


# --------------------------------------------------------------------------- #
# F4: V2V usage guide — registered en/ja text carrying the four guidance points
# (same-scene continuation / no re-instructed dialogue / explicit music
# continuation / larger context is more stable).
# --------------------------------------------------------------------------- #
def test_v2v_guide_present_in_both_languages():
    for lang in ("en", "ja"):
        assert "v2v_guide" in LABELS[lang], lang

    en = LABELS["en"]["v2v_guide"]
    assert en.startswith("**Getting good results with V2V**")
    assert "same" in en and "scene" in en          # 1) same-scene continuation
    assert "dialogue" in en and "already" in en    # 2) no re-instructed dialogue
    assert "music continues" in en                 # 3) explicit music continuation
    assert "more stable" in en                     # 4) larger context stability

    ja = LABELS["ja"]["v2v_guide"]
    assert ja.startswith("**V2Vを使いこなすには**")
    assert "同じシーン" in ja
    assert "セリフ" in ja
    assert "音楽" in ja
    assert "安定" in ja


def test_v2v_guide_is_registered_for_language_switch():
    """build_ui reg()s the guide Markdown, so switch_language must emit an
    update carrying the Japanese guide text."""
    from gradio_ui import build_ui

    demo = build_ui("http://127.0.0.1:8000", api_key=None)
    updates = demo.switch_language("ja", None)
    assert any(
        isinstance(u.get("value"), str) and u["value"].startswith("**V2Vを使いこなすには**")
        for u in updates
    )


# --------------------------------------------------------------------------- #
# F5: crossfade length — JoinRequest default 300 ms, GUI Dropdown rides along
# as handle_crossfade_ms in the join body (None -> field omitted).
# --------------------------------------------------------------------------- #
def test_join_request_default_crossfade_is_300ms():
    from api.models import JoinRequest

    req = JoinRequest()  # empty body {}
    assert req.handle_crossfade_ms == 300
    assert req.audio_smoothing is True


def test_join_handler_sends_selected_crossfade_ms():
    import json

    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            bodies.append(json.loads(request.content.decode("utf-8")))
            return httpx.Response(200, json={
                "job_id": "j1", "join_mode": "handle_crossfade",
                "joined_path": "outputs/j1/joined.mp4", "source_normalized": False,
            })
        return httpx.Response(200, content=b"JOINED")

    api = _make_client(handler)
    join = make_join_handler(api)
    outs = list(join("j1", True, 500))
    assert bodies == [{"handle_crossfade_ms": 500}]
    assert outs[-1][1] and outs[-1][1].endswith(".mp4")


def test_join_handler_without_selection_sends_empty_body():
    import json

    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            bodies.append(json.loads(request.content.decode("utf-8")))
            return httpx.Response(200, json={
                "job_id": "j1", "join_mode": "handle_crossfade",
                "joined_path": "outputs/j1/joined.mp4", "source_normalized": False,
            })
        return httpx.Response(200, content=b"JOINED")

    api = _make_client(handler)
    join = make_join_handler(api)
    list(join("j1", True))          # legacy 2-arg call: no crossfade selection
    list(join("j1", True, "junk"))  # junk selection also falls back
    assert bodies == [{}, {}]


def test_v2v_crossfade_dropdown_default_300():
    """build_ui exposes the crossfade Dropdown with 150/300/500 and default 300."""
    import gradio as gr

    from gradio_ui import build_ui

    demo = build_ui("http://127.0.0.1:8000", api_key=None)
    dropdowns = [
        c for c in demo.blocks.values()
        if isinstance(c, gr.Dropdown)
        and getattr(c, "choices", None)
        and [v for _lbl, v in c.choices] == [150, 300, 500]
    ]
    assert dropdowns, "crossfade dropdown (150/300/500) not found"
    assert dropdowns[0].value == 300
