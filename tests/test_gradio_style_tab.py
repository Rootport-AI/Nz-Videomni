"""Style LoRA tab (S2): gallery build from GET /loras (style-only), reload, and
the gallery-select -> prompt append. Mock transport only; the demo closures are
exposed on the Blocks (mirrors demo.on_page_load) so they run without a server.
"""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import numpy as np

from gradio_ui import build_style_gallery, build_ui, style_lora_names
from gradio_ui.api_client import ApiClient
from gradio_ui.i18n import LABELS

BASE = "http://127.0.0.1:8000"

_LORAS = [
    {"name": "neon-city", "kind": "style", "has_thumbnail": True,
     "exists": True, "source": "scan"},
    {"name": "portrait", "kind": "style", "has_thumbnail": False,
     "exists": True, "source": "scan"},
    {"name": "canny-control", "kind": "control", "has_thumbnail": False,
     "exists": True, "source": "config"},
]


# --------------------------------------------------------------------------- #
# Pure builders.
# --------------------------------------------------------------------------- #
def test_build_style_gallery_style_only_with_thumbnail_and_placeholder():
    items = build_style_gallery(_LORAS, BASE)
    # Control lora is filtered out; two style entries remain, in order.
    assert len(items) == 2
    (img0, cap0), (img1, cap1) = items
    assert cap0 == "neon-city" and cap1 == "portrait"
    assert img0 == f"{BASE}/api/v1/loras/neon-city/thumbnail"
    assert isinstance(img1, np.ndarray)  # placeholder for no-thumbnail entry


def test_style_lora_names_matches_gallery_order():
    assert style_lora_names(_LORAS) == ["neon-city", "portrait"]


def test_build_style_gallery_strips_trailing_slash_in_base():
    items = build_style_gallery(
        [{"name": "neon-city", "kind": "style", "has_thumbnail": True}],
        "http://x:1/")
    assert items[0][0] == "http://x:1/api/v1/loras/neon-city/thumbnail"


def test_build_style_gallery_empty_and_none():
    assert build_style_gallery([], BASE) == []
    assert build_style_gallery(None, BASE) == []


# --------------------------------------------------------------------------- #
# Demo closures over a mock transport.
# --------------------------------------------------------------------------- #
def _mock_demo(handler):
    demo = build_ui(BASE, api_key=None)
    demo.api._client = httpx.Client(transport=httpx.MockTransport(handler))
    return demo


def test_load_style_gallery_builds_from_list():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/loras"
        return httpx.Response(200, json={"loras": _LORAS})

    demo = _mock_demo(handler)
    gallery_update, names = demo.load_style_gallery("en", [])
    assert names == ["neon-city", "portrait"]
    value = gallery_update["value"]
    assert [cap for _img, cap in value] == ["neon-city", "portrait"]


def test_on_style_reload_calls_reload_then_list():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path))
        if request.url.path.endswith("/loras/reload"):
            return httpx.Response(200, json={"total": 3, "styles": 2, "controls": 1})
        return httpx.Response(200, json={"loras": _LORAS})

    demo = _mock_demo(handler)
    gallery_update, names = demo.on_style_reload("en", [])
    assert ("POST", "/api/v1/loras/reload") in seen
    assert ("GET", "/api/v1/loras") in seen
    assert names == ["neon-city", "portrait"]
    assert [cap for _img, cap in gallery_update["value"]] == ["neon-city", "portrait"]


def test_on_style_reload_keeps_state_on_failure():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"code": "BOOM"}})

    demo = _mock_demo(handler)
    _gallery_update, names = demo.on_style_reload("en", ["prev"])
    assert names == ["prev"]  # unchanged on failure


def test_on_style_select_appends_token_to_prompt():
    demo = build_ui(BASE, api_key=None)
    evt = SimpleNamespace(index=1)
    upd = demo.on_style_select("a scene", ["neon-city", "portrait"], "en", evt)
    assert upd["value"] == "a scene <lora:portrait:1.0>"


def test_on_style_select_from_empty_prompt():
    demo = build_ui(BASE, api_key=None)
    evt = SimpleNamespace(index=0)
    upd = demo.on_style_select("", ["neon-city"], "en", evt)
    assert upd["value"] == "<lora:neon-city:1.0>"


def test_on_style_select_out_of_range_is_noop():
    demo = build_ui(BASE, api_key=None)
    evt = SimpleNamespace(index=5)
    upd = demo.on_style_select("keep", ["neon-city"], "en", evt)
    # gr.update() with no value change -> no "value" key set.
    assert upd.get("value") is None


# --------------------------------------------------------------------------- #
# i18n coverage + language switch registration.
# --------------------------------------------------------------------------- #
def test_style_i18n_keys_present_both_langs():
    keys = ["tab_style_lora", "style_gallery_label", "style_reload_btn",
            "style_note", "style_added", "style_reload_done",
            "style_reload_failed", "style_list_failed",
            "lora_msg_unknown", "lora_warn_weight_clamp", "lora_msg_list_failed"]
    for k in keys:
        assert LABELS["en"].get(k), f"missing EN: {k}"
        assert LABELS["ja"].get(k), f"missing JA: {k}"


def test_style_note_is_registered_for_language_switch():
    demo = build_ui(BASE, api_key=None)
    updates = demo.switch_language("ja", None)
    note_ja = LABELS["ja"]["style_note"]
    assert any(u.get("value") == note_ja for u in updates)
