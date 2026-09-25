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
    active_base_model,
    active_unsupported_features,
    build_base_model_choices,
    build_model_choices,
    model_active_value,
)
from gradio_ui.api_client import ApiClient
from gradio_ui.feature_scope import GATED_CONTROLS, RESET_VALUES, hidden_controls
from gradio_ui.handlers import fetch_models_safe, load_selected_models
from gradio_ui.i18n import LABELS
from gradio_ui.ui import build_ui


def _make_client(handler, *, api_key: str | None = "secret") -> ApiClient:
    transport = httpx.MockTransport(handler)
    return ApiClient("http://test", api_key=api_key, client=httpx.Client(transport=transport))


def _entry(name, exists=True, is_default=False, source="config", path=None):
    return {"name": name, "path": path if path is not None else f"models/{name}",
            "is_default": is_default, "exists": exists, "source": source}


#: Filename backing the "default" transformer entry in SAMPLE_MODELS, used to
#: assert the new "default — <filename>" label formatting below.
_DEFAULT_TRANSFORMER_FILENAME = "LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf"

SAMPLE_MODELS = {
    "categories": {
        "transformer": {
            "default": "default",
            "active": "alt",
            "entries": [
                _entry("default", is_default=True,
                       path=f"models/transformer/{_DEFAULT_TRANSFORMER_FILENAME}"),
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
    # Default entry's label is decorated with its path's filename, but the
    # VALUE (what gets sent back to the server) must stay the bare "default".
    assert labels["default"] == f"default — {_DEFAULT_TRANSFORMER_FILENAME}"
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


def test_build_model_choices_default_label_includes_filename():
    """The default entry's label surfaces which file config points at, so the
    user isn't stuck staring at an uninformative bare "default" in the
    Settings-tab Models dropdown."""
    models_json = {"categories": {"transformer": {
        "default": "default", "active": "default",
        "entries": [_entry("default", is_default=True,
                            path="models/transformer/some-model.gguf")],
    }}}
    choices = build_model_choices(models_json, "transformer")
    assert choices == [("default — some-model.gguf", "default")]


def test_build_model_choices_default_label_windows_path_separator():
    """Path separators from a Windows-hosted backend (backslash) resolve to
    the same filename-only label as forward-slash paths."""
    models_json = {"categories": {"transformer": {
        "default": "default", "active": "default",
        "entries": [_entry("default", is_default=True,
                            path=r"models\transformer\some-model.gguf")],
    }}}
    choices = build_model_choices(models_json, "transformer")
    assert choices == [("default — some-model.gguf", "default")]


def test_build_model_choices_default_label_falls_back_without_path():
    """When the default entry has no path (empty string or missing key), the
    label falls back to the plain "default" text used before this change."""
    for bad_path in ("", None):
        entry = _entry("default", is_default=True, path=bad_path)
        if bad_path is None:
            del entry["path"]
        models_json = {"categories": {"transformer": {
            "default": "default", "active": "default", "entries": [entry],
        }}}
        choices = build_model_choices(models_json, "transformer")
        assert choices == [("default", "default")]


def test_build_model_choices_default_missing_on_disk_keeps_filename_and_flag():
    """A default entry that is both filename-labeled AND missing-on-disk gets
    both pieces of information in its label; the value is still "default"."""
    models_json = {"categories": {"transformer": {
        "default": "default", "active": "default",
        "entries": [_entry("default", is_default=True, exists=False,
                            path="models/transformer/some-model.gguf")],
    }}}
    choices = build_model_choices(models_json, "transformer")
    label, value = choices[0]
    assert value == "default"
    assert "some-model.gguf" in label
    assert LABELS["en"]["model_missing"] in label


def test_model_active_value():
    assert model_active_value(SAMPLE_MODELS, "transformer") == "alt"
    assert model_active_value(SAMPLE_MODELS, "audio") == MODEL_DEFAULT
    assert model_active_value(None, "transformer") == MODEL_DEFAULT


# --------------------------------------------------------------------------- #
# adapters: base-model layer (Docs/PENDING_TASKS_CLOSED.md's old §1-25,
# closed 2026-09-01). GET /models keeps its legacy top-level ``categories``
# block (the ACTIVE base model) and adds ``base_models[]`` with a per-base
# listing; the Settings tab's base dropdown reads the latter.
# --------------------------------------------------------------------------- #

#: SAMPLE_MODELS plus the multi-engine layer: LTX23 active, LTX25 listed at its
#: own defaults (the server sends an empty ``active`` for a non-loaded base).
SAMPLE_MODELS_MULTI = dict(SAMPLE_MODELS, **{
    "active_base_model": "LTX23",
    "base_models": [
        {"id": "LTX23", "display_name": "LTX 2.3", "active": True,
         "categories": SAMPLE_MODELS["categories"]},
        {"id": "LTX25", "display_name": "LTX 2.5", "active": False,
         "categories": {
             "transformer": {"default": "default", "active": "",
                             "entries": [_entry("default", is_default=True,
                                                path="models/LTX25/ltx25.gguf"),
                                         _entry("ltx25-alt", source="scan")]},
             "text_encoder": {"default": "default", "active": "",
                              "entries": [_entry("default", is_default=True)]},
             "video_vae": {"default": "default", "active": "",
                           "entries": [_entry("default", is_default=True)]},
             "audio": {"default": "default", "active": "",
                       "entries": [_entry("default", is_default=True)]},
         }},
    ],
})


def test_build_base_model_choices_labels_are_display_names():
    choices = build_base_model_choices(SAMPLE_MODELS_MULTI)
    assert choices == [("LTX 2.3", "LTX23"), ("LTX 2.5", "LTX25")]
    assert active_base_model(SAMPLE_MODELS_MULTI) == "LTX23"
    # A response without the multi-engine layer (or none at all) -> no choices,
    # so the caller leaves its dropdown untouched instead of blanking it.
    assert build_base_model_choices(SAMPLE_MODELS) == []
    assert build_base_model_choices(None) == []
    assert active_base_model(SAMPLE_MODELS) == ""


def test_build_model_choices_reads_the_requested_base_model():
    """Selecting a base model that is not loaded lists ITS entries, and (since
    the server sends no live selection for it) pre-selects "default"."""
    choices = build_model_choices(SAMPLE_MODELS_MULTI, "transformer",
                                  base_model="LTX25")
    assert [value for _label, value in choices] == ["default", "ltx25-alt"]
    assert model_active_value(SAMPLE_MODELS_MULTI, "transformer",
                              base_model="LTX25") == MODEL_DEFAULT
    # Without base_model the legacy (active-base) block is read, unchanged.
    assert [v for _l, v in build_model_choices(SAMPLE_MODELS_MULTI, "transformer")] \
        == ["default", "alt", "ghost"]
    # An unknown base id degrades to the lone "default" choice.
    assert build_model_choices(SAMPLE_MODELS_MULTI, "transformer",
                               base_model="nope") == [(MODEL_DEFAULT, MODEL_DEFAULT)]


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


def test_load_selected_models_sends_the_base_model():
    """Docs/PENDING_TASKS_CLOSED.md's old §1-25 (closed 2026-09-01): the base
    dropdown's id rides along on POST /pipeline/load. An empty/absent
    selection omits the key entirely (byte-identical to the pre-multi-engine
    request)."""
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["json"] = json.loads(request.content)
        return httpx.Response(200, json={"pipeline_loaded": True, "state": "ready"})

    load_selected_models(_make_client(handler), "alt", None, None, None,
                         base_model="LTX25")
    assert seen["json"]["base_model"] == "LTX25"
    assert seen["json"]["models"]["transformer"] == "alt"

    load_selected_models(_make_client(handler), "alt", None, None, None)
    assert "base_model" not in seen["json"]


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


# --------------------------------------------------------------------------- #
# Feature scope: the LOADED base model's ``unsupported_features`` closes the
# controls that would 422 on it. The pure table first, then the /models reader,
# then the refresh closure that joins them -- the same closure all three
# /models-pulling events (Refresh button, page load, post-Load re-pull) are
# wired to, so this covers every path into the gating.
# --------------------------------------------------------------------------- #

def test_hidden_controls_maps_known_names_and_ignores_the_rest():
    assert hidden_controls(["prune_vaed"]) == {"accel_vae"}
    assert hidden_controls(["keep_resident_embeddings"]) == {
        "accel_keep_resident_embeddings"}
    # A mixed list closes the names this build knows and skips the rest --
    # two_stage_hq has no control of its own (the quality radio already falls
    # back to distilled).
    assert hidden_controls(["two_stage_hq", "keep_resident_embeddings",
                            "something_new"]) == {"accel_keep_resident_embeddings"}
    # A backend newer than this build names things it has never heard of; that
    # is the ordinary case, not an error.
    assert hidden_controls(["two_stage_hq", "something_new"]) == frozenset()
    assert hidden_controls([]) == frozenset()
    # Every gated control has a reset value, or hiding it would leave a
    # rejected value riding along on every request.
    assert set(GATED_CONTROLS) <= set(RESET_VALUES)


def test_active_unsupported_features_reads_the_active_base_model():
    models_json = {"base_models": [
        {"id": "LTX23", "active": False,
         "unsupported_features": ["keep_resident_embeddings"]},
        {"id": "LTX25", "active": True,
         "unsupported_features": ["two_stage_hq", "prune_vaed"]},
    ]}
    assert active_unsupported_features(models_json) == ["two_stage_hq", "prune_vaed"]
    # No active entry / no multi-engine layer / nothing at all -> nothing closes.
    assert active_unsupported_features(
        {"base_models": [{"id": "LTX23", "active": False}]}) == []
    assert active_unsupported_features(SAMPLE_MODELS) == []
    assert active_unsupported_features(None) == []


def _models_json_with(active_id: str, unsupported: list[str]) -> dict:
    """SAMPLE_MODELS_MULTI with one base model active and declaring
    ``unsupported``."""
    base_models = []
    for entry in SAMPLE_MODELS_MULTI["base_models"]:
        is_active = entry["id"] == active_id
        base_models.append(dict(entry, active=is_active,
                                unsupported_features=unsupported if is_active else []))
    return dict(SAMPLE_MODELS_MULTI, active_base_model=active_id,
                base_models=base_models)


def _refresh_with(models_json: dict):
    """Drive ``refresh_model_dropdowns`` (the closure every /models event is
    wired to) against a mock transport, the way the on_page_load tests in
    tests/test_gradio_handlers.py drive theirs."""
    demo = build_ui("http://127.0.0.1:8000", api_key=None)
    demo.api._client = httpx.Client(transport=httpx.MockTransport(
        lambda _req: httpx.Response(200, json=models_json)))
    return demo.refresh_model_dropdowns("en", warn=False)


def _gated_update(updates, control: str):
    """The one update in ``refresh_model_dropdowns``'s return that belongs to
    ``control``.

    The gated updates are the TAIL of that tuple, in GATED_CONTROLS order (the
    same order ui.py appends the components to its output list), so the
    position is derived from the table rather than written down here -- adding
    a control to gradio_ui/feature_scope.py must not silently re-point these
    assertions at a neighbour."""
    tail = updates[-len(GATED_CONTROLS):]
    return tail[GATED_CONTROLS.index(control)]


def test_refresh_hides_and_resets_the_vae_radio_for_an_engine_without_it():
    updates = _refresh_with(_models_json_with("LTX25",
                                              ["two_stage_hq", "prune_vaed"]))
    # base dropdown + one per category + the Clip Chain engine State
    # (§3-165) + one per gated control.
    assert len(updates) == 1 + len(MODEL_CATEGORIES) + 1 + len(GATED_CONTROLS)
    vae_update = _gated_update(updates, "accel_vae")
    assert vae_update["visible"] is False
    # Hiding alone is not enough: an invisible component still SENDS its value.
    assert vae_update["value"] == RESET_VALUES["accel_vae"] == "default"
    # The engine that lacks the VAE is the one that HAS the embeddings
    # processor, so the other gated control goes the other way in the same pull.
    kre_update = _gated_update(updates, "accel_keep_resident_embeddings")
    assert kre_update["visible"] is True
    assert "value" not in kre_update


def test_refresh_shows_the_vae_radio_for_an_engine_that_supports_it():
    updates = _refresh_with(_models_json_with("LTX23",
                                              ["keep_resident_embeddings"]))
    vae_update = _gated_update(updates, "accel_vae")
    assert vae_update["visible"] is True
    # A shown control keeps whatever the user picked -- no value is written.
    assert "value" not in vae_update


def test_refresh_hides_and_resets_keep_resident_embeddings_on_an_engine_without_it():
    # The engine with no embeddings processor names the field in
    # unsupported_features; sending true there is a 422, so the checkbox is
    # hidden AND written back to the server default in the same update.
    updates = _refresh_with(_models_json_with("LTX23",
                                              ["keep_resident_embeddings"]))
    update = _gated_update(updates, "accel_keep_resident_embeddings")
    assert update["visible"] is False
    assert update["value"] == RESET_VALUES["accel_keep_resident_embeddings"] is False


def test_refresh_shows_keep_resident_embeddings_on_an_engine_that_supports_it():
    updates = _refresh_with(_models_json_with("LTX25",
                                              ["two_stage_hq", "prune_vaed"]))
    update = _gated_update(updates, "accel_keep_resident_embeddings")
    assert update["visible"] is True
    # Shown, and the user's own choice is left alone.
    assert "value" not in update


def test_refresh_failure_leaves_every_output_untouched():
    demo = build_ui("http://127.0.0.1:8000", api_key=None)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    demo.api._client = httpx.Client(transport=httpx.MockTransport(handler))
    updates = demo.refresh_model_dropdowns("en", warn=False)
    # Same arity as the success path (Gradio matches outputs positionally), and
    # every one of them a bare no-op update (the "+ 1" is the Clip Chain
    # engine State, §3-165; a bare update leaves a State as it is).
    assert len(updates) == 1 + len(MODEL_CATEGORIES) + 1 + len(GATED_CONTROLS)
    assert all("value" not in u and "visible" not in u for u in updates)
