"""機能スコープのAPI層ガード(§3-98 Phase 5・ゲートG7-1/G7-4)。

アダプタ側の判断そのものは tests/test_ltx25_adapter.py が固定している。
ここで確かめるのは**その判断がHTTPの入口に本当に挿さっているか**であり、
見るべきことは3つに絞れる:

1. **挿入点の全網羅**。対応表の422系フィールド一つひとつについて、LTX 2.5が
   アクティブなら422/FEATURE_UNSUPPORTED、既定値なら通り、LTX 2.3がアクティブ
   なら**どれも**FEATURE_UNSUPPORTEDにならない。「2.5で落ちる」だけを見ても
   ガードが正しいことにはならない——2.3で落ちないことと対で初めて意味を持つ;
2. **chain系の一括422**。Chained・Retake・End source・V2V・A2Vはすべて
   POST /generate/chain という一つの入口から来るので、拒否も一つで足りる;
3. **GET /modelsの互換性**。``unsupported_features`` は加算のみ。2.3運用の
   既存クライアントから見た応答は1バイトも変わっていない。

ガードは**ジョブを作る前**に効く。効いていなければ、拒否したはずのリクエスト
がジョブ台帳に残り、同時1ジョブの枠を食い、次の正当なリクエストが409になる——
だから「422が返る」だけでなく「ジョブが増えていない」ところまで見る。

GPUも重みも要らない(両系統ともmockバックエンド)。
"""

from __future__ import annotations

import pytest

from api.models import GenerateRequest
from services.engines.ltx25 import adapter as ltx25

#: 422系フィールドの入力表は**アダプタのテストが正本**(rootdir挿入により
#: tests/ は sys.path 上にあるので、モジュール名で直接引ける)。ここで書き直す
#: と、対応表が育ったときにどちらか片方だけが更新される。
from test_ltx25_adapter import REQUEST_OVERRIDES  # noqa: E402

#: 最小のリクエスト。320x320/9フレームなのは、mockでも本当に生成が走るテスト
#: が混ざっているため小さくしたい一方、outpaintの残し領域は各辺256px以上と
#: スキーマが要求するから(それ未満だとガードに届く前に422 VALIDATION_ERROR
#: になり、何を確かめているのか判らないテストになる)。
BASE_REQUEST: dict = {
    "prompt": "a quiet harbour at first light",
    "width": 320,
    "height": 320,
    "num_frames": 9,
}


def _activate(client, base_model: str) -> None:
    r = client.post("/api/v1/pipeline/load", json={"base_model": base_model})
    assert r.status_code == 200, r.text
    assert client.app_context.pipeline_manager.active_base_model == base_model


def _job_count(client) -> int:
    return len(client.get("/api/v1/jobs").json())


def _chain_body(**overrides) -> dict:
    body = {
        "prompt": "a quiet harbour at first light",
        "width": 320,
        "height": 320,
        "frame_rate": 24.0,
        "overlap_frames": 2,
        "clips": [{"num_frames": 25}, {"num_frames": 25}],
    }
    body.update(overrides)
    return body


# --------------------------------------------------------------------------- #
# 1) POST /generate — 対応表の422系フィールド全件
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("field", sorted(REQUEST_OVERRIDES))
def test_ltx25_refuses_every_field_in_the_reject_table(two_family_client, field):
    _activate(two_family_client, "LTX25")
    before = _job_count(two_family_client)

    r = two_family_client.post(
        "/api/v1/generate", json={**BASE_REQUEST, **REQUEST_OVERRIDES[field]}
    )

    assert r.status_code == 422, r.text
    error = r.json()["error"]
    assert error["code"] == "FEATURE_UNSUPPORTED"
    # メッセージは次の一手を言う(どの機能が・どうすれば使えるのか)。
    assert "LTX 2.3" in error["detail"]
    # ジョブは作られていない。作られていたら同時1ジョブの枠を無駄に食う。
    assert _job_count(two_family_client) == before


@pytest.mark.parametrize("field", sorted(REQUEST_OVERRIDES))
def test_ltx23_accepts_every_field_the_2_5_engine_refuses(two_family_client, field):
    """対の検証。ガードが系統を見ずに効いていたら、ここが落ちる。

    ステータスコードそのものは問わない——``loras`` や ``reference_video_id`` は
    存在しない素材を指しているので404が正しい。見るのは「機能が無いから拒否
    された」という**別の理由**で落ちていないことである。
    """
    _activate(two_family_client, "LTX23")

    r = two_family_client.post(
        "/api/v1/generate", json={**BASE_REQUEST, **REQUEST_OVERRIDES[field]}
    )

    if r.status_code >= 400:
        assert r.json().get("error", {}).get("code") != "FEATURE_UNSUPPORTED", r.text


@pytest.mark.parametrize(
    "field,value",
    [("keep_resident", True), ("attention_backend", "sage"), ("vae_mode", "prune_vaed")],
)
def test_ltx23_really_runs_the_jobs_2_5_refuses(two_family_client, field, value):
    """前のテストは「FEATURE_UNSUPPORTEDでない」しか言っていない。素材を必要と
    しない3件については、2.3で本当に202まで通ることを見ておく。"""
    _activate(two_family_client, "LTX23")
    r = two_family_client.post("/api/v1/generate", json={**BASE_REQUEST, field: value})
    assert r.status_code == 202, r.text


def test_a_plain_request_passes_the_guard_on_ltx25(two_family_client):
    """既定値のフィールドは「利用者が選んだこと」ではない。フロントエンドは
    毎回スキーマ全体を送るので、既定値で422にしたら素のT2Vが一度も通らない。"""
    _activate(two_family_client, "LTX25")
    r = two_family_client.post("/api/v1/generate", json=BASE_REQUEST)
    assert r.status_code == 202, r.text


def test_crop_output_is_not_refused_on_ltx25(two_family_client):
    """crop_outputは仕上げのffmpegセンタークロップで、エンジンに依存しない。
    黙って消さない・拒否しない、が対応表の裁定である。"""
    _activate(two_family_client, "LTX25")
    r = two_family_client.post(
        "/api/v1/generate",
        json={**BASE_REQUEST, "crop_output": {"width": 192, "height": 192}},
    )
    assert r.status_code == 202, r.text


def test_ignored_fields_do_not_fail_a_job_on_ltx25(two_family_client):
    """無視+ログ側は**通す**。block_swap_prefetch等は既定でONなので、ここで
    拒否したら素のT2Vが通らなくなる。"""
    _activate(two_family_client, "LTX25")
    body = {
        **BASE_REQUEST,
        "negative_prompt": "blurry, low quality",
        "block_swap_prefetch": False,
        "fused_gguf_dequant_kernel": False,
        "neg_method": "vsf",
        "vsf_scale": 2.0,
    }
    r = two_family_client.post("/api/v1/generate", json=body)
    assert r.status_code == 202, r.text


def test_the_guard_runs_before_the_upload_and_lora_lookups(two_family_client):
    """機能の可否はサーバの性質で、リクエストの中身の話ではない。存在しない
    参照動画を指した2.5のリクエストに「その動画は無い」と答えたら、利用者は
    直せないものを直しに行く。"""
    _activate(two_family_client, "LTX25")
    r = two_family_client.post(
        "/api/v1/generate",
        json={
            **BASE_REQUEST,
            "reference_video_id": "does-not-exist",
            "loras": [{"name": "no-such-lora", "strength": 1.0}],
        },
    )
    assert r.status_code == 422, r.text
    assert r.json()["error"]["code"] == "FEATURE_UNSUPPORTED"


def test_the_reject_table_and_this_suite_cover_the_same_fields():
    """パラメトライズはその入力表と同じ精度しか持たない。表が育ったら、
    ここが落ちてこのファイルの更新を強制する。"""
    assert set(REQUEST_OVERRIDES) == {f for f, _feat, _p in ltx25.REJECT_TABLE}
    assert set(REQUEST_OVERRIDES) <= set(GenerateRequest.model_fields)


# --------------------------------------------------------------------------- #
# 2) POST /generate/chain — chain系の一括422
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "label,body",
    [
        ("chained", _chain_body()),
        (
            "v2v",
            _chain_body(
                clips=[{"num_frames": 49}, {"num_frames": 25}],
                source_video={"video_id": "vid-1", "context_frames": 25},
            ),
        ),
        ("a2v", _chain_body(source_audio={"audio_id": "aud-1"})),
        (
            "retake",
            _chain_body(
                clips=[{"num_frames": 73}],
                retake={"video_id": "vid-1", "window_start_sec": 0.0},
            ),
        ),
        (
            "end_source",
            _chain_body(
                clips=[{"num_frames": 73}],
                end_source={"image_id": "img-1", "context_frames": 24},
            ),
        ),
    ],
)
def test_ltx25_refuses_the_whole_chain_family(two_family_client, label, body):
    """5つの機能はすべてこの一つの入口から来るので、拒否も一つで足りる。
    素材のidは実在しない——ガードがそれらの解決より前に効くことも同時に見える
    (先に404を返していたら、この表の4件は404で落ちる)。"""
    _activate(two_family_client, "LTX25")
    before = _job_count(two_family_client)

    r = two_family_client.post("/api/v1/generate/chain", json=body)

    assert r.status_code == 422, f"{label}: {r.text}"
    error = r.json()["error"]
    assert error["code"] == "FEATURE_UNSUPPORTED"
    assert "LTX 2.3" in error["detail"]
    assert _job_count(two_family_client) == before


def test_ltx23_still_chains(two_family_client):
    _activate(two_family_client, "LTX23")
    r = two_family_client.post("/api/v1/generate/chain", json=_chain_body())
    assert r.status_code == 202, r.text


def test_switching_back_to_2_3_lifts_the_chain_refusal(two_family_client):
    """ガードは系統に追随する。往復して初めて「系統を見ている」と言える。"""
    _activate(two_family_client, "LTX25")
    assert two_family_client.post("/api/v1/generate/chain", json=_chain_body()).status_code == 422
    _activate(two_family_client, "LTX23")
    assert two_family_client.post("/api/v1/generate/chain", json=_chain_body()).status_code == 202
    _activate(two_family_client, "LTX25")
    assert two_family_client.post("/api/v1/generate/chain", json=_chain_body()).status_code == 422


def test_the_active_family_follows_the_runner(two_family_client):
    pm = two_family_client.app_context.pipeline_manager
    _activate(two_family_client, "LTX25")
    assert pm.active_engine_family == "ltx25"
    _activate(two_family_client, "LTX23")
    assert pm.active_engine_family == "ltx"


# --------------------------------------------------------------------------- #
# 3) GET /models — unsupported_features は加算のみ
# --------------------------------------------------------------------------- #


def test_models_publishes_unsupported_features_per_base_model(two_family_client):
    body = two_family_client.get("/api/v1/models").json()
    by_id = {b["id"]: b for b in body["base_models"]}

    assert by_id["LTX23"]["unsupported_features"] == []
    features = by_id["LTX25"]["unsupported_features"]
    assert {"chain", "retake", "end_source", "v2v", "a2v"} <= set(features)
    assert set(features) == set(ltx25.UNSUPPORTED_FEATURES)
    assert isinstance(features, list), "JSONの配列であること(順序が保たれる)"


def test_the_published_list_matches_what_the_endpoints_actually_refuse(two_family_client):
    """機能名は飾りではない——フロントエンドがこの名前でコントロールを潰す。
    422側が名乗る機能名がこの一覧に無ければ、灰色にならないコントロールが
    残ることになる。"""
    body = two_family_client.get("/api/v1/models").json()
    published = set(
        next(b for b in body["base_models"] if b["id"] == "LTX25")["unsupported_features"]
    )
    assert {feat for _f, feat, _p in ltx25.REJECT_TABLE} <= published


def test_the_legacy_response_shape_is_untouched(two_family_client):
    """2.3運用の既存クライアント(gradio_ui/adapters.py)から見えている範囲は
    一切変わっていない。増えたのは base_models[] の中のキー1つだけである。"""
    body = two_family_client.get("/api/v1/models").json()
    assert set(body) == {"categories", "active_base_model", "base_models"}
    for block in body["categories"].values():
        assert set(block) == {"default", "active", "entries"}
    assert "unsupported_features" not in body["categories"]
    assert "unsupported_features" not in body


def test_every_base_model_entry_carries_the_key(two_family_client):
    """欠けているキーと空リストは別物。クライアントが「知らない=制限なし」と
    「制限なしと宣言された」を区別できるよう、常に在る。"""
    body = two_family_client.get("/api/v1/models").json()
    for entry in body["base_models"]:
        assert "unsupported_features" in entry, entry["id"]
