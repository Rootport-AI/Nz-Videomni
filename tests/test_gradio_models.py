"""Model-management S3 GUI unit tests (mock transport; no server, no GPU).

Covers the new Settings-tab "Models" pieces in isolation: dropdown choice
building from GET /models, the ApiClient methods, the load/fetch handlers
(exact request bodies + localized messages), i18n en/ja parity for the new
``model_`` keys, and a build-time smoke of the assembled Blocks (no HTTP is
performed at build time).

Deliberately imports from the submodules (not gradio_ui.__init__) so the
shared package surface stays untouched by this slice.
"""

from __future__ import annotations

import json

import httpx

from gradio_ui.adapters import (
    MODEL_CATEGORIES,
    MODEL_DEFAULT,
    build_model_choices,
    model_active_value,
)
from gradio_ui.api_client import ApiClient
from gradio_ui.handlers import fetch_models_safe, load_selected_models
from gradio_ui.i18n import LABELS
from gradio_ui.ui import build_ui


def _make_client(handler, *, api_key: str | None = "secret") -> ApiClient:
    transport = httpx.MockTransport(handler)
    return ApiClient("http://test", api_key=api_key, client=httpx.Client(transport=transport))


def _entry(name, exists=True, is_default=False, source="config"):
    return {"name": name, "path": f"models/{name}", "is_default": is_default,
            "exists": exists, "source": source}


SAMPLE_MODELS = {
    "categories": {
        "transformer": {
            "default": "default",
            "active": "alt",
            "entries": [
                _entry("default", is_default=True),
                _entry("alt", source="scan"),
                _entry("ghost", exists=False),
            ],
        },
        "text_encoder": {"default": "default", "active": "default",
                         "entries": [_entry("default", is_default=True)]},
        "video_vae": {"default": "default", "active": "default",
                      "entries": [_entry("default", is_default=True)]},
        "audio": {"default": "default", "active": "default",
                  "entries": [_entry("default", is_default=True)]},
    }
}


# --------------------------------------------------------------------------- #
# adapters: choices + active value
# --------------------------------------------------------------------------- #

def test_build_model_choices_values_are_names():
    choices = build_model_choices(SAMPLE_MODELS, "transformer")
    assert [value for _label, value in choices] == ["default", "alt", "ghost"]
    labels = {value: label for label, value in choices}
    assert labels["default"] == "default"
    assert labels["alt"] == "alt"
    # Missing-on-disk entry stays selectable but is labeled.
    assert LABELS["en"]["model_missing"] in labels["ghost"]


def test_build_model_choices_missing_label_localizes():
    choices = build_model_choices(SAMPLE_MODELS, "transformer", lang="ja")
    labels = {value: label for label, value in choices}
    assert LABELS["ja"]["model_missing"] in labels["ghost"]


def test_build_model_choices_empty_fallback():
    for bad in (None, {}, {"categories": {}}):
        assert build_model_choices(bad, "transformer") == [(MODEL_DEFAULT, MODEL_DEFAULT)]


def test_model_active_value():
    assert model_active_value(SAMPLE_MODELS, "transformer") == "alt"
    assert model_active_value(SAMPLE_MODELS, "audio") == MODEL_DEFAULT
    assert model_active_value(None, "transformer") == MODEL_DEFAULT


# --------------------------------------------------------------------------- #
# ApiClient: paths, method, auth, body
# --------------------------------------------------------------------------- #

def test_get_models_path_and_auth():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json=SAMPLE_MODELS)

    api = _make_client(handler)
    body = api.get_models()
    assert seen == {"method": "GET", "path": "/api/v1/models", "auth": "Bearer secret"}
    assert body == SAMPLE_MODELS


def test_load_pipeline_models_posts_models_block():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["json"] = json.loads(request.content)
        return httpx.Response(200, json={"pipeline_loaded": True, "state": "ready",
                                         "models": {c: MODEL_DEFAULT for c in MODEL_CATEGORIES}})

    api = _make_client(handler)
    api.load_pipeline_models({"transformer": "alt"})
    assert seen["method"] == "POST"
    assert seen["path"] == "/api/v1/pipeline/load"
    assert seen["json"] == {"models": {"transformer": "alt"}}


# --------------------------------------------------------------------------- #
# handlers: fetch_models_safe / load_selected_models
# --------------------------------------------------------------------------- #

def test_fetch_models_safe_success_and_failure():
    api_ok = _make_client(lambda _req: httpx.Response(200, json=SAMPLE_MODELS))
    models, err = fetch_models_safe(api_ok)
    assert err is None and models == SAMPLE_MODELS

    api_down = _make_client(lambda _req: httpx.Response(503, json={"error": {"code": "X"}}))
    models, err = fetch_models_safe(api_down)
    assert models is None
    assert err is not None  # localized warning, never raises


def test_load_selected_models_success_message_and_body():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["json"] = json.loads(request.content)
        return httpx.Response(200, json={
            "pipeline_loaded": True, "pipeline_type": "distilled", "state": "ready",
            "models": {"transformer": "alt", "text_encoder": "default",
                       "video_vae": "default", "audio": "default"},
        })

    msg = load_selected_models(_make_client(handler), "alt", "default", "default", "default")
    assert seen["json"] == {"models": {"transformer": "alt", "text_encoder": "default",
                                       "video_vae": "default", "audio": "default"}}
    assert "transformer=alt" in msg
    assert "text_encoder=default" in msg


def test_load_selected_models_empty_values_default():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["json"] = json.loads(request.content)
        return httpx.Response(200, json={"pipeline_loaded": True, "state": "ready"})

    load_selected_models(_make_client(handler), None, "", None, None)
    assert seen["json"] == {"models": {c: MODEL_DEFAULT for c in MODEL_CATEGORIES}}


def test_load_selected_models_422_incompatible_friendly_error():
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"error": {
            "code": "MODEL_INCOMPATIBLE",
            "message": "selected model 'transformer/bad' failed the compatibility precheck",
            "detail": "not a GGUF file (magic b'XXXX')",
        }})

    msg = load_selected_models(_make_client(handler), "bad", None, None, None)
    assert LABELS["en"]["apierr_MODEL_INCOMPATIBLE"] in msg
    assert "not a GGUF file" in msg  # detail is surfaced


def test_load_selected_models_409_busy_localized_ja():
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(409, json={"error": {
            "code": "JOB_BUSY",
            "message": "A job is already running",
            "detail": "cannot swap models while a job is running",
        }})

    msg = load_selected_models(_make_client(handler), "alt", None, None, None, lang="ja")
    assert LABELS["ja"]["apierr_JOB_BUSY"] in msg


def test_load_selected_models_connection_error_does_not_raise():
    def handler(_req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    msg = load_selected_models(_make_client(handler), None, None, None, None)
    assert "boom" in msg


# --------------------------------------------------------------------------- #
# i18n parity + UI build smoke
# --------------------------------------------------------------------------- #

def test_i18n_model_keys_en_ja_parity():
    en_keys = {k for k in LABELS["en"] if k.startswith(("model_", "apierr_MODEL_"))}
    ja_keys = {k for k in LABELS["ja"] if k.startswith(("model_", "apierr_MODEL_"))}
    assert en_keys == ja_keys
    assert "model_section_title" in en_keys
    for cat in MODEL_CATEGORIES:
        assert f"model_cat_{cat}" in en_keys


def test_build_ui_smoke_with_models_section():
    """build_ui assembles the Blocks graph (including the new Models section
    and its independent listeners) without performing any HTTP."""
    demo = build_ui("http://127.0.0.1:1", api_key=None)
    assert demo is not None
    # The label registry (language switch) picked up the new model_ keys.
    keys = {key for _c, key, _a in demo.label_registry}
    assert {"model_section_title", "model_btn_load", "model_btn_refresh",
            "model_cat_transformer", "model_cat_audio"} <= keys
