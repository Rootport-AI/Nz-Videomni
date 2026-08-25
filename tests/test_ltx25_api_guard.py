"""機能スコープのAPI層ガード(§3-98 Phase 5・ゲートG7-1/G7-4)。

アダプタ側の判断そのものは tests/test_ltx25_adapter.py が固定している。
ここで確かめるのは**その判断がHTTPの入口に本当に挿さっているか**であり、
見るべきことは3つに絞れる:

1. **挿入点の全網羅**。対応表の422系フィールド一つひとつについて、LTX 2.5が
   アクティブなら422/FEATURE_UNSUPPORTED、既定値なら通り、LTX 2.3がアクティブ
   なら**どれも**FEATURE_UNSUPPORTEDにならない。「2.5で落ちる」だけを見ても
   ガードが正しいことにはならない——2.3で落ちないことと対で初めて意味を持つ;
2. **chain系のフィールド単位の422**(§3-102で一括422から変わった)。Chained・
   Retake・End source・V2V・A2Vはすべて POST /generate/chain という一つの入口
   から来るが、**素のChainedは2.5でも走る**ので拒否は一つでは足りない。表の各行が
   422になり、素のChainedは202から completed まで行くこと、の両方を見る;
3. **GET /modelsの互換性**。応答の形として増えたのは ``unsupported_features``
   というキー1つだけで、2.3運用の既存クライアントから見た応答は1バイトも
   変わっていない(中身の機能名は、エンジンができることが増えれば減る——
   §3-102で ``chain`` が外れた)。

ガードは**ジョブを作る前**に効く。効いていなければ、拒否したはずのリクエスト
がジョブ台帳に残り、同時1ジョブの枠を食い、次の正当なリクエストが409になる——
だから「422が返る」だけでなく「ジョブが増えていない」ところまで見る。

GPUも重みも要らない(両系統ともmockバックエンド)。
"""

from __future__ import annotations

import argparse
import json
import struct
import wave

import pytest
import yaml
from fastapi.testclient import TestClient
from PIL import Image

import chain_math
import main
from api.models import GenerateChainRequest, GenerateRequest
from services import video_io
from services.engines.ltx25 import adapter as ltx25

#: 422系フィールドの入力表は**アダプタのテストが正本**(rootdir挿入により
#: tests/ は sys.path 上にあるので、モジュール名で直接引ける)。ここで書き直す
#: と、対応表が育ったときにどちらか片方だけが更新される。単発の
#: ``GenerateRequest`` 用とchain用の2枚あり、扱う機能が違う。
from test_ltx25_adapter import (  # noqa: E402
    CHAIN_ACCEPTED_KEEP_RESIDENT,
    CHAIN_ACCEPTED_LORAS,
    CHAIN_ACCEPTED_RETAKE,
    CHAIN_ACCEPTED_SAGE,
    CHAIN_ACCEPTED_SOURCES,
    CHAIN_OVERRIDES,
    REQUEST_ACCEPTED_KEEP_RESIDENT,
    REQUEST_ACCEPTED_LORAS,
    REQUEST_ACCEPTED_SAGE,
    REQUEST_OVERRIDES,
)

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


def _make_args(config_path: str) -> argparse.Namespace:
    """conftestの同名ヘルパと同じ引数束(そちらは非公開名なので写した)。"""
    return argparse.Namespace(
        listen=False, port=None, api_key=None, allow_all_cors=False,
        config=config_path, te_offload=None, dit_cpu_load=None,
    )


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
    [("vae_mode", "prune_vaed")],
)
def test_ltx23_really_runs_the_jobs_2_5_refuses(two_family_client, field, value):
    """前のテストは「FEATURE_UNSUPPORTEDでない」しか言っていない。素材を必要と
    しない件については、2.3で本当に202まで通ることを見ておく。

    ``keep_resident`` は高速化第2弾でこの一覧を**外れた**。2.5でも受理される
    ようになったので、「2.5が拒否する仕事」という前提そのものが成り立たない
    ——2.3側の受理は下の肯定テストが2系統まとめて見る。``attention_backend``
    は高速化第3弾で同じ理由で外れ、残るのは ``vae_mode`` 1件だけになった
    (PrunaVAEDは2.5に該当する経路が無く、据え置きである)。"""
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
    """無視+ログ側は**通す**。素のT2Vでも負のプロンプト系の項目は一式送られて
    くるので、ここで拒否したら通常の生成が一度も通らない。

    block_swap_prefetch と fused_gguf_dequant_kernel は高速化第1弾で
    「無視+ログ」から**honoured（実際に効く）**へ移った。ただし本文で明示的に
    False を送る形は変えていない。どちらの分類でも202で受理されること自体は
    変わらず、この一本は「受理される」という利用者から見た約束を守る番人で
    あって、内部の分類を写した鏡ではないからである。"""
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


def test_the_acceleration_echoes_reach_metadata_json_on_ltx25(two_family_client):
    """高速化第1弾の配線を、mockで端から端まで1本通す。

    2.5のアダプタは done イベントの
    ``block_swap_prefetch_used`` / ``fused_gguf_dequant_kernel_used`` を
    そのまま中継し、pipeline_manager が metadata.json へ書く。mockバックエンド
    はGPUを持たず何もエコーしないので、正しい答えは**null**である——「実際に
    何が起きたか」を報告する欄に、誰も報告していないのに "off" と書いたら、
    それはmockが実機のふりをしたことになる。実機での "on"/"on->off" は
    ゲートG5で確かめる。

    欄そのものが metadata.json に存在することを見るのがこの一本の値打ちで、
    キーごと消えていれば実機で開通しても利用者には何も見えない。"""
    _activate(two_family_client, "LTX25")
    r = two_family_client.post("/api/v1/generate", json=BASE_REQUEST)
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    assert two_family_client.get(f"/api/v1/jobs/{job_id}").json()["status"] == "completed"

    ctx = two_family_client.app_context
    meta = json.loads(
        (ctx.config.output_dir / job_id / "metadata.json").read_text(encoding="utf-8")
    )
    assert meta["backend"] == ltx25.MOCK_BACKEND_25
    assert "block_swap_prefetch_used" in meta and "fused_gguf_dequant_kernel_used" in meta
    assert meta["block_swap_prefetch_used"] is None
    assert meta["fused_gguf_dequant_kernel_used"] is None
    # 高速化第2弾の3本目。理由も答えも第1弾の2本と同じで、mockは何も報告しない
    # のだから null が正しい。実機での "on"/"off" はゲートG5で確かめる。
    assert "keep_resident_used" in meta
    assert meta["keep_resident_used"] is None


def test_the_guard_runs_before_the_upload_and_lora_lookups(two_family_client):
    """機能の可否はサーバの性質で、リクエストの中身の話ではない。存在しない
    素材を指した2.5のリクエストに「その画像は無い」と答えたら、利用者は
    直せないものを直しに行く。

    題材のフィールドは**NAG**である——第3段でLoRAと参照動画が通るように
    なったので、それらで見たらガードではなく404側が正しい答えになってしまう。
    2.5に残る拒否フィールドでなければ、この順序は確かめられない。"""
    _activate(two_family_client, "LTX25")
    r = two_family_client.post(
        "/api/v1/generate",
        json={
            **BASE_REQUEST,
            "nag_enabled": True,
            "negative_prompt": "blurry, low quality",
            "conditioning_images": [{"image_id": "does-not-exist"}],
        },
    )
    assert r.status_code == 422, r.text
    assert r.json()["error"]["code"] == "FEATURE_UNSUPPORTED"


@pytest.mark.parametrize("case", sorted(REQUEST_ACCEPTED_LORAS))
def test_ltx25_no_longer_refuses_style_and_reference(two_family_client, case):
    """第3段の逆転(表駆動)。素材のidとアダプタ名は実在しないので404で構わない
    ——見るのは「機能が無いから拒否された」で落ちていないことだけである。
    第2段まではこの2件がまさにFEATURE_UNSUPPORTEDだった。"""
    _activate(two_family_client, "LTX25")
    r = two_family_client.post(
        "/api/v1/generate", json={**BASE_REQUEST, **REQUEST_ACCEPTED_LORAS[case]}
    )
    if r.status_code >= 400:
        assert r.json().get("error", {}).get("code") != "FEATURE_UNSUPPORTED", r.text


@pytest.mark.parametrize("case", sorted(REQUEST_ACCEPTED_KEEP_RESIDENT))
def test_ltx25_no_longer_refuses_keep_resident(two_family_client, case):
    """高速化第2弾の逆転。ここは404で妥協しない——**202まで**を見る。

    このフィールドは素材を一つも要らないので、通るなら最後まで通るはずで
    あり、途中で止まる理由があるとしたらそれはガードだけである。そして
    直前まで、まさにここが422だった:2.5を選んだまま設定パネルで常駐を
    ONにすると、以後の**すべての**ジョブが422になる——利用者から見れば
    「LTX 2.5が壊れた」としか見えない罠で、それが消えたことをこの一本が
    見張る。"""
    _activate(two_family_client, "LTX25")
    r = two_family_client.post(
        "/api/v1/generate", json={**BASE_REQUEST, **REQUEST_ACCEPTED_KEEP_RESIDENT[case]}
    )
    assert r.status_code == 202, r.text


@pytest.mark.parametrize("case", sorted(REQUEST_ACCEPTED_SAGE))
def test_ltx25_no_longer_refuses_sage_attention(two_family_client, case):
    """高速化第3弾の逆転。``keep_resident`` と同じくここも404で妥協せず
    **202まで**見る——sageは素材を一つも要らないので、途中で止まる理由がある
    としたらそれはガードだけである。

    そして罠も同じ形だった:2.5を選んだまま設定パネルで注意機構をsageにすると、
    以後の**すべての**ジョブが422になる。設定は残るので、利用者が設定パネルを
    開き直すまで直らない——「LTX 2.5が壊れた」としか見えない。これが消えたこと
    をこの一本が見張る。

    mockバックエンドなのでsageのカーネルそのものは走らない(GPUも重みも無い)。
    ここで見るのは**入口の裁定**だけで、本当にカーネルが効くことは実機ゲート
    (G5)の担当である。"""
    _activate(two_family_client, "LTX25")
    r = two_family_client.post(
        "/api/v1/generate", json={**BASE_REQUEST, **REQUEST_ACCEPTED_SAGE[case]}
    )
    assert r.status_code == 202, r.text


def test_outpaint_is_still_refused_by_name_on_ltx25(two_family_client):
    """Outpaintは第3段でも対象外のままである。表の中で``outpaint``の行は外れた
    2行より**前**に居るので、スキーマが要求する道連れ(参照動画+IC-LoRA)に
    メッセージを奪われることもない——利用者は本当にできないものを知らされる。"""
    _activate(two_family_client, "LTX25")
    r = two_family_client.post(
        "/api/v1/generate", json={**BASE_REQUEST, **REQUEST_OVERRIDES["outpaint"]}
    )
    assert r.status_code == 422, r.text
    error = r.json()["error"]
    assert error["code"] == "FEATURE_UNSUPPORTED"
    assert "outpaint" in error["detail"]


def test_the_reject_table_and_this_suite_cover_the_same_fields():
    """パラメトライズはその入力表と同じ精度しか持たない。表が育ったら、
    ここが落ちてこのファイルの更新を強制する。"""
    assert set(REQUEST_OVERRIDES) == {f for f, _feat, _p in ltx25.REJECT_TABLE}
    assert set(REQUEST_OVERRIDES) <= set(GenerateRequest.model_fields)


# --------------------------------------------------------------------------- #
# 2) POST /generate/chain — chain系のフィールド単位の422と、素のChainedの完走
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("field", sorted(CHAIN_OVERRIDES))
def test_ltx25_refuses_every_field_in_the_chain_reject_table(two_family_client, field):
    """chain系の拒否は**フィールド単位**である(§3-102)。V2V・A2V・Retake・
    End source・LoRA・参照動画・NAG・加速系はどれもこの一つの入口から来るので、
    表の各行がHTTPの入口で本当に効いていることを一件ずつ見る。

    素材のidは実在しない——ガードがそれらの解決より前に効くことも同時に見える
    (先に404を返していたら、素材を指す4件は404で落ちる)。"""
    _activate(two_family_client, "LTX25")
    before = _job_count(two_family_client)

    body = _chain_body(**CHAIN_OVERRIDES[field])
    r = two_family_client.post("/api/v1/generate/chain", json=body)

    assert r.status_code == 422, f"{field}: {r.text}"
    error = r.json()["error"]
    assert error["code"] == "FEATURE_UNSUPPORTED"
    assert "LTX 2.3" in error["detail"]
    assert _job_count(two_family_client) == before


@pytest.mark.parametrize("field", sorted(CHAIN_OVERRIDES))
def test_ltx23_accepts_every_chain_field_the_2_5_engine_refuses(two_family_client, field):
    """対の検証。ガードが系統を見ずに効いていたら、ここが落ちる。

    ステータスコードそのものは問わない——素材のidは実在しないので404が正しい。
    見るのは「機能が無いから拒否された」という**別の理由**で落ちていないこと。"""
    _activate(two_family_client, "LTX23")

    body = _chain_body(**CHAIN_OVERRIDES[field])
    r = two_family_client.post("/api/v1/generate/chain", json=body)

    if r.status_code >= 400:
        assert r.json().get("error", {}).get("code") != "FEATURE_UNSUPPORTED", r.text


def test_the_chain_reject_table_and_this_suite_cover_the_same_fields():
    """パラメトライズはその入力表と同じ精度しか持たない。表が育ったら、
    ここが落ちてこのファイルの更新を強制する。"""
    assert set(CHAIN_OVERRIDES) == {f for f, _feat, _p in ltx25.CHAIN_REJECT_TABLE}
    assert set(CHAIN_OVERRIDES) <= set(GenerateChainRequest.model_fields)


def test_ltx25_runs_a_plain_chained_job(two_family_client):
    """§3-102の見出し。素のChainedは2.5でも202で受理され、mockバックエンドで
    最後まで走り切る(mockは2系統で共有なので、422を外せば完走する)。"""
    _activate(two_family_client, "LTX25")
    r = two_family_client.post("/api/v1/generate/chain", json=_chain_body())
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]

    job = two_family_client.get(f"/api/v1/jobs/{job_id}").json()
    assert job["status"] == "completed", job

    ctx = two_family_client.app_context
    out = ctx.config.output_dir / job_id / "output.mp4"
    assert out.exists() and out.stat().st_size > 0
    meta = json.loads((ctx.config.output_dir / job_id / "metadata.json").read_text("utf-8"))
    assert meta["kind"] == "chain"
    # metadata.json が「どちらのエンジンのmockが走ったか」を名乗る。GPUの無い
    # 環境で2.3↔2.5の往復を確かめられるのは、この一行があるからである。
    assert meta["backend"] == ltx25.MOCK_BACKEND_25


def test_a_plain_chain_survives_the_optional_knobs_on_ltx25(two_family_client):
    """動作フィールド側の対。crop_output・chunked_upsample・stage2_window・
    クリップ毎プロンプトは黙って消さない・拒否しない、が対応表の裁定である。"""
    _activate(two_family_client, "LTX25")
    r = two_family_client.post(
        "/api/v1/generate/chain",
        json=_chain_body(
            crop_output={"width": 256, "height": 192},
            chunked_upsample=True,
            stage2_window="high_resolution",
            clips=[{"num_frames": 25, "prompt": "夕暮れ"}, {"num_frames": 25}],
        ),
    )
    assert r.status_code == 202, r.text
    job = two_family_client.get(f"/api/v1/jobs/{r.json()['job_id']}").json()
    assert job["status"] == "completed", job


def test_the_ignored_chain_fields_do_not_fail_a_job_on_ltx25(two_family_client):
    """無視+ログ側は**通す**。素のChainedでも負のプロンプト系の項目は一式
    送られてくるので、ここで拒否したら連結生成が一度も通らない。

    単発側の双子と同じく、block_swap_prefetch と fused_gguf_dequant_kernel は
    高速化第1弾でhonouredへ移った。202で受理されるという約束は分類が変わっても
    同じで、それを見張るのがこの一本である。"""
    _activate(two_family_client, "LTX25")
    r = two_family_client.post(
        "/api/v1/generate/chain",
        json=_chain_body(
            negative_prompt="blurry, low quality",
            block_swap_prefetch=False,
            fused_gguf_dequant_kernel=False,
            neg_method="vsf",
            vsf_scale=2.0,
        ),
    )
    assert r.status_code == 202, r.text


def test_the_chain_guard_runs_before_the_upload_lookups(two_family_client):
    """機能の可否はサーバの性質で、リクエストの中身の話ではない。存在しない
    素材を指した2.5の連結生成に「その画像は無い」と答えたら、利用者は直せない
    ものを直しに行く。ガードは ``upload_store.path_for`` や LoRA の解決に伴う
    404より**先**に居る。

    題材のフィールドは**NAG**である——第2段でV2V・A2Vが、第3段でLoRAと参照動画が
    通るようになったので、2.5に残る拒否フィールドで見なければ、この順序は
    確かめられない。"""
    _activate(two_family_client, "LTX25")
    r = two_family_client.post(
        "/api/v1/generate/chain",
        json=_chain_body(
            clips=[
                {"num_frames": 25, "conditioning_images": [{"image_id": "does-not-exist"}]},
                {"num_frames": 25},
            ],
            nag_enabled=True,
            negative_prompt="blurry, low quality",
        ),
    )
    assert r.status_code == 422, r.text
    assert r.json()["error"]["code"] == "FEATURE_UNSUPPORTED"


@pytest.mark.parametrize("case", sorted(CHAIN_ACCEPTED_LORAS))
def test_ltx25_no_longer_refuses_chain_style_and_reference(two_family_client, case):
    """chain側の逆転(表駆動)。素材のidは実在しないので404で構わない——見るのは
    「機能が無いから拒否された」で落ちていないことだけである。"""
    _activate(two_family_client, "LTX25")
    r = two_family_client.post(
        "/api/v1/generate/chain", json=_chain_body(**CHAIN_ACCEPTED_LORAS[case])
    )
    if r.status_code >= 400:
        assert r.json().get("error", {}).get("code") != "FEATURE_UNSUPPORTED", r.text


@pytest.mark.parametrize("case", sorted(CHAIN_ACCEPTED_KEEP_RESIDENT))
def test_ltx25_no_longer_refuses_chain_keep_resident(two_family_client, case):
    """単発側の双子(高速化第2弾)。連結生成にも素材は要らないので、ここも
    **202まで**見る。設定パネルの常駐がONのままだと連結生成も一つ残らず422に
    なっていた——タブが違うだけで罠は同じものだった。"""
    _activate(two_family_client, "LTX25")
    r = two_family_client.post(
        "/api/v1/generate/chain", json=_chain_body(**CHAIN_ACCEPTED_KEEP_RESIDENT[case])
    )
    assert r.status_code == 202, r.text


@pytest.mark.parametrize("case", sorted(CHAIN_ACCEPTED_SAGE))
def test_ltx25_no_longer_refuses_chain_sage_attention(two_family_client, case):
    """単発側の双子(高速化第3弾)。設定パネルの注意機構がsageのままだと連結生成
    も一つ残らず422になっていた——タブが違うだけで罠は同じものだった。ここも
    **202まで**見る。"""
    _activate(two_family_client, "LTX25")
    r = two_family_client.post(
        "/api/v1/generate/chain", json=_chain_body(**CHAIN_ACCEPTED_SAGE[case])
    )
    assert r.status_code == 202, r.text


# --------------------------------------------------------------------------- #
# 2b) V2V・A2V・長尺A2Vがmockで往復すること(§3-102 第2段)
#
# 422が外れただけでは「使える」とは言えない。素材のアップロードから
# metadata.json まで、素のChainedと同じ道を最後まで通ることを見る。mockの
# バックエンドクラスは2系統で共有なので(設計裁定)、通れば2.5でも同じ形の
# 出力とメタデータが出る——GPUの無い環境で確かめられるのはここまでで、
# 実際の映像・音声の一致は実機ゲート(G5)の担当である。
# --------------------------------------------------------------------------- #


def _make_source_mp4(path, n_frames, fps, size=(96, 64)):
    frames = [Image.new("RGB", size, (i * 4 % 256, 90, 160)) for i in range(n_frames)]
    video_io.encode_frames_to_mp4(frames, path, frame_rate=fps)
    return path


def _make_wav(path, *, seconds, sr=16000, channels=1):
    n = int(round(seconds * sr))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(struct.pack("<%dh" % (n * channels), *([0] * (n * channels))))
    return path


def _upload_video(client, path) -> str:
    r = client.post(
        "/api/v1/upload/video", files={"file": ("src.mp4", path.read_bytes(), "video/mp4")}
    )
    assert r.status_code == 200, r.text
    return r.json()["video_id"]


def _upload_audio(client, path) -> str:
    r = client.post(
        "/api/v1/upload/audio", files={"file": ("voice.wav", path.read_bytes(), "audio/wav")}
    )
    assert r.status_code == 200, r.text
    return r.json()["audio_id"]


def _run_to_completion(client, body: dict) -> str:
    r = client.post("/api/v1/generate/chain", json=body)
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    job = client.get(f"/api/v1/jobs/{job_id}").json()
    assert job["status"] == "completed", job
    return job_id


def _metadata(client, job_id: str) -> dict:
    job_dir = client.app_context.config.output_dir / job_id
    return json.loads((job_dir / "metadata.json").read_text(encoding="utf-8"))


def test_ltx25_runs_a_v2v_chain_and_joins_it(two_family_client, tmp_path):
    """V2V往復の全長。アップロード→202→completed→メタの ``v2v`` ブロック→
    **Join 200**まで。Joinはこの機能の出口そのもの(元動画と続きを繋いで
    1本にする)なので、ここまで見て初めて「2.5でV2Vが使える」と言える。"""
    _activate(two_family_client, "LTX25")
    src = _make_source_mp4(tmp_path / "v2v_src.mp4", n_frames=50, fps=24.0)
    vid = _upload_video(two_family_client, src)

    job_id = _run_to_completion(
        two_family_client,
        _chain_body(
            clips=[{"num_frames": 49}],
            source_video={"video_id": vid, "context_frames": 25},
        ),
    )

    job_dir = two_family_client.app_context.config.output_dir / job_id
    # アプリが要求fpsへ切り出した末尾mp4。エンジンは再エンコードしない約束なので、
    # これが無いままworkerが呼ばれていたらV2Vは成立していない。
    assert (job_dir / "_source_tail.mp4").exists()
    out = job_dir / "output.mp4"
    assert out.exists() and out.stat().st_size > 0

    layout = chain_math.compute_chain_layout([49], 24.0, kv=2, source_context_px=25)
    assert video_io.frame_count(out) == layout.new_frames_px  # 文脈分は切り落とし済み

    meta = _metadata(two_family_client, job_id)
    assert meta["backend"] == ltx25.MOCK_BACKEND_25
    v2v = meta["v2v"]
    assert v2v["source_video_id"] == vid
    assert v2v["context_frames"] == 25
    assert v2v["new_frames_px"] == layout.new_frames_px
    assert v2v["v2v_context_junction_px"] == layout.v2v_context_junction_px
    # Joinが要求するサイドカー(切り落とし前の全長波形)も揃っている。
    assert (job_dir / v2v["audio_handle_filename"]).exists()

    r = two_family_client.post(f"/api/v1/jobs/{job_id}/join", json={})
    assert r.status_code == 200, r.text
    assert r.json()["joined_path"] == f"outputs/{job_id}/joined.mp4"
    assert (job_dir / "joined.mp4").exists()


def test_ltx25_runs_a_retake_chain(two_family_client, tmp_path):
    """Retake往復の全長。アップロード→202→completed→アプリが切り出した窓→
    メタの ``retake`` ブロックまで。422が外れただけでは「使える」とは言えない
    ので、素材の切り出しから metadata.json まで、2.3と同じ道を最後まで通る
    ことを見る。

    見どころは**納品が窓まるごと**であること。V2Vが文脈分を切り落とすのと
    逆で、撮り直しは糊代(前後の凍結帯)を付けたまま返す——タイムラインは
    それを元の映像に重ねて置くので、繋ぎ目が窓の外縁に来る。だから
    ``frame_count == 73`` であり、V2Vのような音声サイドカーも出ない。"""
    _activate(two_family_client, "LTX25")
    src = _make_source_mp4(tmp_path / "retake_src.mp4", n_frames=90, fps=24.0)
    vid = _upload_video(two_family_client, src)

    job_id = _run_to_completion(
        two_family_client,
        _chain_body(
            clips=[{"num_frames": 73}],
            retake={"video_id": vid, "window_start_sec": 0.0},
        ),
    )

    job_dir = two_family_client.app_context.config.output_dir / job_id
    # アプリが切り出した窓。エンジンは切りも再サンプルもしない約束なので、
    # これが無いままworkerが呼ばれていたら撮り直しは成立していない。
    assert (job_dir / "_retake_window.mp4").exists()
    assert video_io.frame_count(job_dir / "_retake_window.mp4") == 73

    out = job_dir / "output.mp4"
    assert out.exists() and out.stat().st_size > 0
    assert video_io.frame_count(out) == 73          # 窓まるごと・無トリム
    assert not list(job_dir.glob("*_audio_handle.wav"))   # V2Vと違い出さない

    meta = _metadata(two_family_client, job_id)
    assert meta["backend"] == ltx25.MOCK_BACKEND_25
    rt = meta["retake"]
    # 幾何は chain_math が唯一の出所——アプリの検証もエンジンもここを読む。
    layout = chain_math.compute_chain_layout(
        [73], 24.0, kv=2, retake_glue_px=(25, 24)
    )
    assert rt["window_px"] == layout.to_dict()["retake"]["window_px"] == 73
    assert (rt["head_px"], rt["tail_px"]) == (25, 24)
    assert (rt["n_head_v"], rt["n_tail_v"]) == (
        layout.to_dict()["retake"]["n_head_v"],
        layout.to_dict()["retake"]["n_tail_v"],
    )
    assert rt["free_middle_px"] == [25, 49]
    # 実行時の側。
    assert rt["regenerate_audio"] is True
    assert rt["decoded_frames_px"] == 73
    assert rt["retake_video_id"] == vid
    # mockは凍結の証明を捏造しない(潜在を持たないので出しようがない)。
    assert "freeze_proof" not in rt


def test_ltx25_runs_a_single_tab_a2v_chain(two_family_client, tmp_path):
    """SingleタブのA2V=フロントエンドが ``stage2_window="full_length"`` 固定の
    1クリップChainedを投げる形。A2Vが開通すればSingle A2Vも同時に開通する、
    という設計の裏取りなので、窓の指定込みで往復させる。"""
    _activate(two_family_client, "LTX25")
    wav = _make_wav(tmp_path / "single.wav", seconds=6.0)
    aid = _upload_audio(two_family_client, wav)

    job_id = _run_to_completion(
        two_family_client,
        _chain_body(
            clips=[{"num_frames": 121}],
            source_audio={"audio_id": aid},
            stage2_window="full_length",
        ),
    )

    meta = _metadata(two_family_client, job_id)
    assert meta["backend"] == ltx25.MOCK_BACKEND_25
    assert meta["chain"]["stage2_window"] == "full_length"
    assert meta["chain"]["n_tiles"] == 1
    a2v = meta["a2v"]
    assert a2v["source_audio_id"] == aid
    assert a2v["a_total"] == chain_math.compute_chain_layout([121], 24.0, kv=2).a_total
    # A2Vの約束: ボコーダーを通さず、入力の原波形をそのままmuxする。
    assert a2v["muxed_original_waveform"] is True
    assert a2v["vocoder_skipped"] is True


def test_ltx25_runs_a_long_a2v_chain(two_family_client, tmp_path):
    """長尺A2V: 1本の音声が3クリップを駆動する。クリップ毎の音声は無く、
    エンジンが1本の潜在をstage-1のセグメントへ割り付ける——だからメタの
    ``a_total`` は組み上がったタイムライン全体の長さになる。"""
    _activate(two_family_client, "LTX25")
    layout = chain_math.compute_chain_layout([121, 121, 121], 24.0, kv=2)
    wav = _make_wav(tmp_path / "long.wav", seconds=16.0)
    aid = _upload_audio(two_family_client, wav)

    job_id = _run_to_completion(
        two_family_client,
        _chain_body(
            clips=[{"num_frames": 121}, {"num_frames": 121}, {"num_frames": 121}],
            source_audio={"audio_id": aid},
        ),
    )

    meta = _metadata(two_family_client, job_id)
    assert meta["backend"] == ltx25.MOCK_BACKEND_25
    assert meta["chain"]["num_clips"] == 3
    assert meta["a2v"]["a_total"] == layout.a_total
    assert meta["a2v"]["source_audio_id"] == aid


@pytest.mark.parametrize("case", sorted(CHAIN_ACCEPTED_SOURCES))
def test_ltx25_no_longer_refuses_the_two_source_modes(two_family_client, case):
    """対の検証(表駆動)。素材のidは実在しないので404で構わない——見るのは
    「機能が無いから拒否された」で落ちていないことだけである。第1段まではこの
    2件がまさにFEATURE_UNSUPPORTEDだったので、ここが逆転の証拠になる。"""
    _activate(two_family_client, "LTX25")
    r = two_family_client.post(
        "/api/v1/generate/chain", json=_chain_body(**CHAIN_ACCEPTED_SOURCES[case])
    )
    if r.status_code >= 400:
        assert r.json().get("error", {}).get("code") != "FEATURE_UNSUPPORTED", r.text


def test_the_still_refused_chain_mode_is_exactly_end_source():
    """各段で外れたのは、その段が実装した分だけである。走らないモードまで
    一緒に落ちていたら、それが422にならず、mockでは通ってしまう——実機で
    初めて「エンジンにその道が無い」と判ることになる。"""
    assert {f for f, _feat, _p in ltx25.CHAIN_REJECT_TABLE} >= {"end_source"}
    assert not {"source_video", "source_audio"} & {
        f for f, _feat, _p in ltx25.CHAIN_REJECT_TABLE
    }
    # 第3段で外れたのも2件だけである。
    assert not {"loras", "reference_video_id"} & {
        f for f, _feat, _p in ltx25.CHAIN_REJECT_TABLE
    }
    # Retake段で外れたのは1件だけである。
    assert "retake" not in {f for f, _feat, _p in ltx25.CHAIN_REJECT_TABLE}


@pytest.mark.parametrize("case", sorted(CHAIN_ACCEPTED_RETAKE))
def test_ltx25_no_longer_refuses_retake(two_family_client, case):
    """Retake段の対の検証(表駆動)。素材のidは実在しないので404で構わない——
    見るのは「機能が無いから拒否された」で落ちていないことだけである。前段まで
    この3件はまさにFEATURE_UNSUPPORTEDだったので、ここが逆転の証拠になる。"""
    _activate(two_family_client, "LTX25")
    r = two_family_client.post(
        "/api/v1/generate/chain", json=_chain_body(**CHAIN_ACCEPTED_RETAKE[case])
    )
    if r.status_code >= 400:
        assert r.json().get("error", {}).get("code") != "FEATURE_UNSUPPORTED", r.text


def test_ltx23_still_chains(two_family_client):
    _activate(two_family_client, "LTX23")
    r = two_family_client.post("/api/v1/generate/chain", json=_chain_body())
    assert r.status_code == 202, r.text


def test_switching_back_to_2_3_lifts_the_chain_mode_refusals(two_family_client):
    """ガードは系統に追随する。往復して初めて「系統を見ている」と言える。
    題材は**End source**である——素のChainedもV2V・A2Vも、Retake段を経たいまは
    撮り直しも、両系統で通るようになった。2.5に残る拒否モードで見る必要が
    あり、残っているのはこれ1件である(この行の題材が枯れたら、それは2.5が
    chain系の全モードを持ったということなので、このテスト自体を畳んでよい)。"""
    body = _chain_body(**CHAIN_OVERRIDES["end_source"])
    _activate(two_family_client, "LTX25")
    assert two_family_client.post("/api/v1/generate/chain", json=body).status_code == 422
    _activate(two_family_client, "LTX23")
    # 2.3では機能の可否では落ちない(素材が無いので404)。
    r = two_family_client.post("/api/v1/generate/chain", json=body)
    assert r.json().get("error", {}).get("code") != "FEATURE_UNSUPPORTED", r.text
    _activate(two_family_client, "LTX25")
    assert two_family_client.post("/api/v1/generate/chain", json=body).status_code == 422


def test_the_active_family_follows_the_runner(two_family_client):
    pm = two_family_client.app_context.pipeline_manager
    _activate(two_family_client, "LTX25")
    assert pm.active_engine_family == "ltx25"
    _activate(two_family_client, "LTX23")
    assert pm.active_engine_family == "ltx"


# --------------------------------------------------------------------------- #
# 2c) Style LoRA・IC-LoRA・長尺IC-LoRAがmockで往復すること(§3-102 第3段)
#
# 422が外れただけでは「使える」とは言えない。アダプタの登録から
# metadata.json まで、2.3と同じ道を最後まで通ることを見る。mockのバックエンド
# クラスは2系統で共有なので(設計裁定)、通れば2.5でも同じ形の出力とメタデータが
# 出る——実際に絵柄が変わることや制御に追従することは実機ゲート(G5)の担当で、
# GPUの無い環境で確かめられるのはここまでである。
#
# **参照付きは全て%128の解像度**(512×384)。参照は出力の半分の解像度で64格子に
# 乗るので、api層が width/height % 128 != 0 を先に422にする(2.3と同じ規則)。
# --------------------------------------------------------------------------- #

#: 登録するアダプタ。名前は実機ゲートB9の題材(Pixar_Toon)と、2.3側のテストが
#: 使うcanny/depth/deblurの論理名に揃える。
STYLE_LORA = "Pixar_Toon"
CANNY_LORA = "canny-control"
DEPTH_LORA = "depth-control"
DEBLUR_LORA = "deblur"

#: 参照付きリクエストの基寸。512も384も128の倍数である。
REF_SIZE = {"width": 512, "height": 384}


def _write_safetensors(path, metadata=None):
    """ヘッダだけが意味を持つ最小のsafetensors(tests/test_lora_registry.py と同形)。

    ``reference_downscale_factor`` を持つヘッダは、preprocessが``none``でも
    **control**アダプタとして解決される——deblur(前処理不要の制御アダプタ)を
    再現するにはこれが要る。
    """
    header: dict = {}
    if metadata is not None:
        header["__metadata__"] = {k: str(v) for k, v in metadata.items()}
    blob = json.dumps(header).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        fh.write(bytes(16))
    return path


@pytest.fixture()
def lora_two_family_client(tmp_path):
    """``two_family_client`` と同じ2系統の世界に、IC-LoRAの登録を足したもの。

    共有フィクスチャを書き換えず別に建てるのは、``ic_loras`` を足すと
    ``GET /loras`` の応答が変わり、登録を前提にしていない既存テストの前提まで
    動いてしまうからである。組み立ての部品(記述子・重み・GGUFヘッダ)は
    conftestのものをそのまま使うので、二つの世界が食い違うことはない。
    """
    from conftest import (
        base_model_descriptor,
        build_model_layout,
        write_gguf_with_kv,
        write_model_file,
    )

    ltx23 = base_model_descriptor()
    ltx25_descriptor = base_model_descriptor("LTX25")
    ltx25_descriptor["display_name"] = "LTX 2.5"
    ltx25_descriptor["engine_family"] = "ltx25"

    fragment = build_model_layout(tmp_path, [ltx23, ltx25_descriptor])
    models_dir = tmp_path / "models"
    for category, spec in ltx25_descriptor["categories"].items():
        target = models_dir / spec["default_file"]
        if category == "transformer":
            write_gguf_with_kv(
                target, **{"general.architecture": "ltxv", "model_version": "2.5.0"}
            )
        else:
            write_model_file(target)
    write_gguf_with_kv(
        models_dir / ltx23["categories"]["transformer"]["default_file"],
        **{"general.architecture": "ltxv", "model_version": "2.3.0"},
    )

    adapters = tmp_path / "adapters"
    style = _write_safetensors(adapters / "Pixar_Toon.safetensors")
    union = _write_safetensors(
        adapters / "union-control.safetensors", {"reference_downscale_factor": "2"}
    )
    deblur = _write_safetensors(
        adapters / "deblur.safetensors", {"reference_downscale_factor": "1"}
    )

    cfg = {
        "server": {"log_dir": (tmp_path / "logs").as_posix()},
        "model": {
            "backend": "mock",
            **fragment,
            # スタイルLoRAの走査先も必ずtmpへ。既定のままだと開発機の実物の
            # StyleLoRAフォルダを読み、結果がその機械の持ち物に依存する。
            "lora_dir": (tmp_path / "style_scan").as_posix(),
            "ic_loras": {
                STYLE_LORA: style.as_posix(),
                CANNY_LORA: {"path": union.as_posix(), "preprocess": "canny"},
                DEPTH_LORA: {"path": union.as_posix(), "preprocess": "depth"},
                # 前処理不要の制御アダプタ(文字列形式)。controlになるのは
                # ヘッダの reference_downscale_factor による。
                DEBLUR_LORA: deblur.as_posix(),
            },
        },
        "output": {"dir": (tmp_path / "outputs").as_posix()},
        "upload": {"dir": (tmp_path / "uploads").as_posix()},
        "state_file": (tmp_path / "state.json").as_posix(),
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    app = main.build_app(_make_args(cfg_path.as_posix()))
    with TestClient(app) as client:
        client.app_context = app.state.context  # type: ignore[attr-defined]
        yield client


#: 参照動画として上げるバイト列。ストアは拡張子と大きさしか見ず、mockの
#: バックエンドはこれを開かない(実尺の計測はアプリ側でbest-effort)。
FAKE_MP4 = b"\x00\x00\x00\x18ftypmp42" + bytes(64)


def _upload_reference(client) -> str:
    r = client.post(
        "/api/v1/upload/video", files={"file": ("ref.mp4", FAKE_MP4, "video/mp4")}
    )
    assert r.status_code == 200, r.text
    return r.json()["video_id"]


def test_ltx25_runs_a_style_lora_single_job(lora_two_family_client):
    """①Style LoRA単発。フロントエンドはプロンプトのタグ
    ``<lora:Pixar_Toon:1.0>`` で選ばせるが、HTTPの契約は ``loras`` フィールド
    そのものなので、ここではフィールドで直接指定する(タグの解釈はフロント側の
    話である)。202→completed→metadataの ``ic_lora`` ブロックまで見る。
    第2段まではこれがFEATURE_UNSUPPORTEDだった。"""
    _activate(lora_two_family_client, "LTX25")
    r = lora_two_family_client.post(
        "/api/v1/generate",
        json={**BASE_REQUEST, "loras": [{"name": STYLE_LORA, "strength": 1.0}]},
    )
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    job = lora_two_family_client.get(f"/api/v1/jobs/{job_id}").json()
    assert job["status"] == "completed", job

    meta = _metadata(lora_two_family_client, job_id)
    assert meta["backend"] == ltx25.MOCK_BACKEND_25
    assert meta["ic_lora"]["loras"] == [
        {"name": STYLE_LORA, "strength": 1.0, "preprocess": "none"}
    ]
    # スタイル単体は参照動画を要らない(kind-awareな裁定。2.3と同じ)。
    assert meta["ic_lora"]["reference_video_id"] is None


def test_ltx25_runs_a_style_lora_chain(lora_two_family_client):
    """②Style LoRA×Chained。連結の全クリップに同じアダプタが一様に効く形で、
    参照動画は無い——なので ``ic_lora`` ブロックは**出ない**のが正しい(2.3の
    裁定そのまま)。要求そのものは ``request`` ダンプに残る。"""
    _activate(lora_two_family_client, "LTX25")
    job_id = _run_to_completion(
        lora_two_family_client,
        _chain_body(loras=[{"name": STYLE_LORA, "strength": 0.8}]),
    )
    meta = _metadata(lora_two_family_client, job_id)
    assert meta["backend"] == ltx25.MOCK_BACKEND_25
    assert meta["request"]["loras"] == [
        {"name": STYLE_LORA, "strength": 0.8, "audio_strength": None}
    ]
    assert "ic_lora" not in meta


def test_ltx25_runs_a_control_lora_with_a_reference_video(lora_two_family_client):
    """③制御IC-LoRA+参照動画の単発。解像度は**512×384**——参照は出力の半分の
    解像度で64格子に乗るので、%128でなければアプリが先に422にする。"""
    _activate(lora_two_family_client, "LTX25")
    vid = _upload_reference(lora_two_family_client)

    r = lora_two_family_client.post(
        "/api/v1/generate",
        json={
            **BASE_REQUEST,
            **REF_SIZE,
            "loras": [{"name": CANNY_LORA, "strength": 1.0}],
            "reference_video_id": vid,
            "conditioning_attention_strength": 0.7,
        },
    )
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    job = lora_two_family_client.get(f"/api/v1/jobs/{job_id}").json()
    assert job["status"] == "completed", job

    meta = _metadata(lora_two_family_client, job_id)
    assert meta["backend"] == ltx25.MOCK_BACKEND_25
    ic_lora = meta["ic_lora"]
    assert ic_lora["reference_video_id"] == vid
    assert ic_lora["loras"][0]["name"] == CANNY_LORA
    assert ic_lora["loras"][0]["preprocess"] == "canny"
    assert ic_lora["conditioning_attention_strength"] == 0.7


def test_ltx25_refuses_a_non_128_resolution_for_a_reference_job(lora_two_family_client):
    """③の対。参照付きで%128でない解像度は、機能の可否ではなく**寸法**の理由で
    422になる——2.3と同じ符号でなければ、利用者は直せないものを直しに行く。"""
    _activate(lora_two_family_client, "LTX25")
    vid = _upload_reference(lora_two_family_client)
    r = lora_two_family_client.post(
        "/api/v1/generate",
        json={
            **BASE_REQUEST,  # 320x320: 128で割り切れない
            "loras": [{"name": CANNY_LORA, "strength": 1.0}],
            "reference_video_id": vid,
        },
    )
    assert r.status_code == 422, r.text
    assert r.json()["error"]["code"] == "REFERENCE_RESOLUTION_INVALID"


def test_ltx25_still_refuses_a_depth_adapter_on_a_multi_clip_chain(lora_two_family_client):
    """④depth前処理×多クリップは、**エンジン非依存**の制限として残る
    (LORA_DEPTH_CHAIN_UNSUPPORTED)。Video-Depth-Anythingがクリップ全体を
    一括で見る設計だからで、2.5だから駄目なのではない——だから
    FEATURE_UNSUPPORTEDではなく、こちらの符号で落ちなければならない。"""
    _activate(lora_two_family_client, "LTX25")
    vid = _upload_reference(lora_two_family_client)

    r = lora_two_family_client.post(
        "/api/v1/generate/chain",
        json=_chain_body(
            **REF_SIZE,
            loras=[{"name": DEPTH_LORA, "strength": 1.0}],
            reference_video_id=vid,
        ),
    )
    assert r.status_code == 422, r.text
    assert r.json()["error"]["code"] == "LORA_DEPTH_CHAIN_UNSUPPORTED"
    # 1クリップなら通る(制限はクリップ数の側にある)。
    r1 = lora_two_family_client.post(
        "/api/v1/generate/chain",
        json=_chain_body(
            **REF_SIZE,
            clips=[{"num_frames": 25}],
            loras=[{"name": DEPTH_LORA, "strength": 1.0}],
            reference_video_id=vid,
        ),
    )
    assert r1.status_code == 202, r1.text


def test_ltx25_runs_a_long_ic_lora_chain(lora_two_family_client):
    """⑤長尺IC-LoRA(§3-78の幾何)。1本の参照動画が2クリップを駆動し、
    metadataの ``reference_segment_windows`` が ``chain_math`` の直接呼び出しと
    一致する。窓はアプリ側が再計算して書くので、mockでもここまで確かめられる
    ——実機ゲートB12はこの一致を実物の映像で見る側の担当である。

    題材はdeblur(前処理不要の制御アダプタ)。depthと違い多クリップに乗る。"""
    _activate(lora_two_family_client, "LTX25")
    vid = _upload_reference(lora_two_family_client)
    clip_frames = [25, 25]

    job_id = _run_to_completion(
        lora_two_family_client,
        _chain_body(
            **REF_SIZE,
            clips=[{"num_frames": f} for f in clip_frames],
            loras=[{"name": DEBLUR_LORA, "strength": 1.0}],
            reference_video_id=vid,
        ),
    )

    meta = _metadata(lora_two_family_client, job_id)
    assert meta["backend"] == ltx25.MOCK_BACKEND_25
    ic_lora = meta["ic_lora"]
    assert ic_lora["reference_video_id"] == vid
    assert ic_lora["loras"][0]["name"] == DEBLUR_LORA

    layout = chain_math.compute_chain_layout(clip_frames, 24.0, kv=2)
    expected = [list(w) for w in chain_math.video_segment_windows(layout)]
    assert ic_lora["reference_segment_windows"] == expected
    assert len(expected) == len(clip_frames)
    assert ic_lora["reference_frames_needed"] == layout.total_px


def test_the_2_3_engine_runs_the_same_lora_jobs(lora_two_family_client):
    """対の検証。2.5で通るようになったからといって2.3が壊れていない——同じ
    リクエストが同じ登録の上で両系統とも202になる。"""
    for base_model in ("LTX23", "LTX25"):
        _activate(lora_two_family_client, base_model)
        r = lora_two_family_client.post(
            "/api/v1/generate",
            json={**BASE_REQUEST, "loras": [{"name": STYLE_LORA, "strength": 1.0}]},
        )
        assert r.status_code == 202, f"{base_model}: {r.text}"


# --------------------------------------------------------------------------- #
# 3) GET /models — unsupported_features は加算のみ
# --------------------------------------------------------------------------- #


def test_models_publishes_unsupported_features_per_base_model(two_family_client):
    body = two_family_client.get("/api/v1/models").json()
    by_id = {b["id"]: b for b in body["base_models"]}

    assert by_id["LTX23"]["unsupported_features"] == []
    features = by_id["LTX25"]["unsupported_features"]
    assert {"end_source"} <= set(features)
    # Retake段で ``retake`` が外れた——残っていればEditタブの「撮り直し」
    # サブタブも、タイムラインの右クリックからそこへ入る導線も灰色のままに
    # なる(どちらも動くようになった)。
    assert "retake" not in features
    # "chain" は§3-102で一覧から外れた。素のChainedが走るようになった以上、
    # ここに残っていればフロントエンドが動くタブを灰色にしてしまう。同じ理屈で
    # "v2v"/"a2v" も第2段で外れた——残っていればChainedタブのソース欄も、
    # Singleタブのa2vアコーディオンも、BatchタブのA2V行も灰色のままになる。
    assert "chain" not in features
    assert "v2v" not in features and "a2v" not in features
    # 第3段で ``loras`` / ``reference_video`` も外れた——残っていればLoRAチップも
    # 参照動画のパネルも灰色のままになる。``outpaint`` は残る。
    assert "loras" not in features and "reference_video" not in features
    assert "outpaint" in features
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
    # chain側の表も同じ理屈で公開されていなければならない(§3-102)。
    assert {feat for _f, feat, _p in ltx25.CHAIN_REJECT_TABLE} <= published


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
