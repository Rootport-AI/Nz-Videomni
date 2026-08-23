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

import json
import struct
import wave

import pytest
from PIL import Image

import chain_math
from api.models import GenerateChainRequest, GenerateRequest
from services import video_io
from services.engines.ltx25 import adapter as ltx25

#: 422系フィールドの入力表は**アダプタのテストが正本**(rootdir挿入により
#: tests/ は sys.path 上にあるので、モジュール名で直接引ける)。ここで書き直す
#: と、対応表が育ったときにどちらか片方だけが更新される。単発の
#: ``GenerateRequest`` 用とchain用の2枚あり、扱う機能が違う。
from test_ltx25_adapter import (  # noqa: E402
    CHAIN_ACCEPTED_SOURCES,
    CHAIN_OVERRIDES,
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
    """無視+ログ側は**通す**。block_swap_prefetch等は既定でONなので、ここで
    拒否したら素のChainedが一度も通らない。"""
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

    題材のフィールドは**LoRA**である——第2段でV2V・A2Vが通るようになったので、
    2.5に残る拒否フィールドで見なければ、この順序は確かめられない。"""
    _activate(two_family_client, "LTX25")
    r = two_family_client.post(
        "/api/v1/generate/chain",
        json=_chain_body(
            clips=[
                {"num_frames": 25, "conditioning_images": [{"image_id": "does-not-exist"}]},
                {"num_frames": 25},
            ],
            loras=[{"name": "no-such-lora", "strength": 1.0}],
        ),
    )
    assert r.status_code == 422, r.text
    assert r.json()["error"]["code"] == "FEATURE_UNSUPPORTED"


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


def test_the_still_refused_chain_modes_are_exactly_retake_and_end_source():
    """第2段で外れたのは2件だけである。V2V/A2Vと一緒にRetakeやEnd sourceの
    行まで落ちていたら、走らないモードが422にならず、mockでは通ってしまう。"""
    assert {f for f, _feat, _p in ltx25.CHAIN_REJECT_TABLE} >= {"retake", "end_source"}
    assert not {"source_video", "source_audio"} & {
        f for f, _feat, _p in ltx25.CHAIN_REJECT_TABLE
    }


def test_ltx23_still_chains(two_family_client):
    _activate(two_family_client, "LTX23")
    r = two_family_client.post("/api/v1/generate/chain", json=_chain_body())
    assert r.status_code == 202, r.text


def test_switching_back_to_2_3_lifts_the_chain_mode_refusals(two_family_client):
    """ガードは系統に追随する。往復して初めて「系統を見ている」と言える。
    題材は**Retake**である——素のChainedもV2V・A2Vもいまや両系統で通るので、
    それらで往復させても何も動かない。2.5に残る拒否モードで見る必要がある。"""
    body = _chain_body(**CHAIN_OVERRIDES["retake"])
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
# 3) GET /models — unsupported_features は加算のみ
# --------------------------------------------------------------------------- #


def test_models_publishes_unsupported_features_per_base_model(two_family_client):
    body = two_family_client.get("/api/v1/models").json()
    by_id = {b["id"]: b for b in body["base_models"]}

    assert by_id["LTX23"]["unsupported_features"] == []
    features = by_id["LTX25"]["unsupported_features"]
    assert {"retake", "end_source"} <= set(features)
    # "chain" は§3-102で一覧から外れた。素のChainedが走るようになった以上、
    # ここに残っていればフロントエンドが動くタブを灰色にしてしまう。同じ理屈で
    # "v2v"/"a2v" も第2段で外れた——残っていればChainedタブのソース欄も、
    # Singleタブのa2vアコーディオンも、BatchタブのA2V行も灰色のままになる。
    assert "chain" not in features
    assert "v2v" not in features and "a2v" not in features
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
