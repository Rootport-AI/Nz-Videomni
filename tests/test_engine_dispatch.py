"""エンジン系統のディスパッチ(§3-98 P3c)。

ベースモデルが2つになった時点で必要になった判断——「この重みは、どのアダプタ
が走らせるのか」——を固定する。ここで押さえるのは4点:

1. **系統×KVの4組全表**(設計M1)。正しい組み合わせ2つは通り、交差した2つは
   明示的に422になる。判定の主体はKV(ファイルの中身)であって、記述子は
   照合相手である;
2. **遅延import**。``services.engines`` を import しただけではどのアダプタも
   読み込まれない。これは性能の話ではなく構造の話で、dictにクラスを1つ置けば
   壊れる性質なので、別プロセスで実測する;
3. **契約**。出荷されている記述子のカテゴリ集合は、その記述子が宣言する系統の
   ``SELECTION_FIELDS`` と一致していなければならない;
4. **系統跨ぎの入れ替え**。ベースモデルを2.3↔2.5で往復させると、Runnerの
   オブジェクトそのものが差し替わり、その前に必ず旧Runnerのunloadが走る。

GPUも重みも要らない。ここでの「LTX 2.5のファイル」は ``model_version 2.5.0``
を名乗る手製のGGUFヘッダで、判定が読むのはそれだけである。
"""

from __future__ import annotations

import logging
import subprocess
import sys

import pytest

from api.errors import APIError
from config import PROJECT_ROOT, AppConfig
from services import engines
from services.base_models import BaseModelDescriptor, CategoryDescriptor, load_base_models

LTX23_KV = {"general.architecture": "ltxv", "model_version": "2.3.0"}
LTX25_KV = {"general.architecture": "ltxv", "model_version": "2.5.0"}


def _descriptor(family: str, display_name: str) -> BaseModelDescriptor:
    return BaseModelDescriptor(
        id=display_name.replace(" ", "").replace(".", ""),
        display_name=display_name,
        engine_family=family,
        categories={"transformer": CategoryDescriptor("transformer", ("x",), (".gguf",))},
    )


LTX23_DESCRIPTOR = _descriptor("ltx", "LTX 2.3")
LTX25_DESCRIPTOR = _descriptor("ltx25", "LTX 2.5")


# --------------------------------------------------------------------------- #
# 1) 系統 x KV の4組全表(M1)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "descriptor,kv",
    [(LTX23_DESCRIPTOR, LTX23_KV), (LTX25_DESCRIPTOR, LTX25_KV)],
    ids=["ltx23-descriptor+2.3-weights", "ltx25-descriptor+2.5-weights"],
)
def test_matching_family_and_kv_pass(descriptor, kv):
    engines.check_kv(descriptor, "transformer", "default", kv)  # no raise


@pytest.mark.parametrize(
    "descriptor,kv,wanted",
    [
        (LTX23_DESCRIPTOR, LTX25_KV, "LTX 2.5"),
        (LTX25_DESCRIPTOR, LTX23_KV, "LTX 2.3"),
    ],
    ids=["ltx23-descriptor+2.5-weights", "ltx25-descriptor+2.3-weights"],
)
def test_crossed_family_and_kv_are_refused_naming_the_base_model_to_pick(descriptor, kv, wanted):
    """交差した2組は「壊れたファイル」ではない——正しいファイルを、間違った
    ベースモデルで開こうとしている。だからメッセージは次の一手を言う。"""
    with pytest.raises(APIError) as ei:
        engines.check_kv(descriptor, "transformer", "weights", kv)
    assert ei.value.code == "MODEL_INCOMPATIBLE" and ei.value.status_code == 422
    detail = ei.value.detail
    assert f"ltxv {kv['model_version']}" in detail
    assert f"「{wanted}」を選んでください" in detail
    # 選択中のベースモデルの名前も出る(どちらが今なのかを言わないと直せない)。
    assert descriptor.display_name in detail


@pytest.mark.parametrize("category", ["text_encoder", "video_vae", "audio"])
def test_only_the_transformer_is_matched_against_the_family(category):
    """世代を定義するのはtransformerだけ。GGUF Gemmaは自前の
    ``general.architecture`` を持っており、それを「エンジン違い」と読んでは
    ならない。"""
    engines.check_kv(LTX25_DESCRIPTOR, category, "default", LTX23_KV)
    engines.check_kv(LTX23_DESCRIPTOR, category, "default", {"general.architecture": "gemma3"})


def test_a_foreign_architecture_is_still_refused_by_the_family_adapter():
    """系統表からは引けない(``wan`` は無い)ので照合は見送られ、最終判断は
    アダプタが下す。"""
    with pytest.raises(APIError) as ei:
        engines.check_kv(
            LTX23_DESCRIPTOR, "transformer", "wan",
            {"general.architecture": "wan", "model_version": "2.2.0"},
        )
    assert "'wan'系のモデルです" in ei.value.detail


def test_missing_kv_keys_do_not_trigger_a_family_mismatch(caplog):
    """キー欠損は§2.5からの一貫した方針どおり警告どまり。系統照合も見送る
    (決められないものを決めたことにしない)。"""
    with caplog.at_level(logging.WARNING):
        engines.check_kv(LTX23_DESCRIPTOR, "transformer", "homebrew", {})
    assert engines.family_by_kv({}) is None
    assert engines.family_by_kv({"model_version": "2.5.0"}) is None
    assert engines.family_by_kv(LTX25_KV) == "ltx25"


def test_a_patch_respin_still_maps_to_its_family():
    assert engines.family_by_kv({**LTX25_KV, "model_version": "2.5.11"}) == "ltx25"


def test_an_unknown_generation_is_left_to_the_family_adapter():
    """表に無いminorはNone——「この配布物が知らない世代」であって、系統が
    違うという判断ではない。"""
    kv = {"general.architecture": "ltxv", "model_version": "9.9.0"}
    assert engines.family_by_kv(kv) is None
    with pytest.raises(APIError) as ei:
        engines.check_kv(LTX25_DESCRIPTOR, "transformer", "future", kv)
    assert ei.value.code == "MODEL_INCOMPATIBLE"


def test_an_unknown_engine_family_fails_loud():
    unknown = _descriptor("wanx", "WAN 2.2")
    with pytest.raises(RuntimeError, match="未知のエンジン系統"):
        engines.check_kv(unknown, "transformer", "default", LTX23_KV)
    with pytest.raises(RuntimeError, match="未知のエンジン系統"):
        engines.runner_class_for("wanx")


def test_the_three_family_tables_agree():
    assert set(FAMILY_IDS := set(engines.FAMILY_BY_ID)) == set(engines.FAMILY_DISPLAY)
    assert set(engines.FAMILY_BY_KV.values()) <= FAMILY_IDS
    assert engines.FAMILY_BY_KV == {("ltxv", "2.3"): "ltx", ("ltxv", "2.5"): "ltx25"}


# --------------------------------------------------------------------------- #
# 2) 遅延import
# --------------------------------------------------------------------------- #


def _in_subprocess(code: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_importing_the_dispatcher_loads_no_adapter():
    """新プロセスで実測する。このテスト自身のプロセスはconftestがmainを読んだ
    時点で全アダプタを抱えており、``sys.modules`` を見ても何も判らないため。"""
    loaded = _in_subprocess(
        "import sys, services.engines;"
        "print(sorted(m for m in sys.modules if m.startswith('services.engines.')))"
    )
    assert loaded == "[]"


def test_asking_for_one_family_loads_only_that_family():
    loaded = _in_subprocess(
        "import sys, services.engines as e;"
        "e.runner_class_for('ltx');"
        "print(int(any(m.startswith('services.engines.ltx25') for m in sys.modules)))"
    )
    assert loaded == "0"


def test_the_dispatcher_does_not_import_base_models():
    """記述子は「エンジン層が読む純粋なデータ」。逆向きのimportを作らない、が
    services/base_models.py 冒頭の規約。"""
    loaded = _in_subprocess(
        "import sys, services.engines;print(int('services.base_models' in sys.modules))"
    )
    assert loaded == "0"


# --------------------------------------------------------------------------- #
# 3) Runnerクラスと機能表
# --------------------------------------------------------------------------- #


def test_runner_class_for_returns_each_familys_facade():
    from services.engines.ltx.adapter import LTXRunner
    from services.engines.ltx25.adapter import LTX25Runner

    assert engines.runner_class_for("ltx") is LTXRunner
    assert engines.runner_class_for("ltx25") is LTX25Runner


def test_unsupported_features_per_family():
    # §3-114で ``ltx`` 側にも宣言ができた。**向きが逆の1件**である——
    # keep_resident_embeddings が指す埋め込み処理器は LTX 2.5 にしか無い部品
    # なので、断るのは 2.3 の側になる。ここが空タプルへ戻ったら、2.3を選んだ
    # ままこのつまみをONにできてしまい、フロントエンドは灰色にしない。
    assert engines.unsupported_features("ltx") == ("keep_resident_embeddings",)
    features = engines.unsupported_features("ltx25")
    # 系統ごとの一覧は「その系統のアダプタが宣言したものだけ」である。それは
    # **残っている名前**でも**外れた名前**でも同じように確かめられ、いま2.5に
    # 残るのはエンジン側の機能名だけなので、代表として1つ在ることを見る。
    assert "two_stage_hq" in features
    # "chain" は§3-102の第1段で、"v2v"/"a2v" は第2段で、"retake" は
    # Retake段で、"end_source" は End source段で外れた(素のChained・V2V・A2V・
    # 撮り直し・素材(末尾)が2.5でも走る)。
    assert "chain" not in features
    assert "v2v" not in features and "a2v" not in features
    assert "retake" not in features
    assert "end_source" not in features
    # そして "outpaint" が Outpainting段で外れた——**種類を問わず最後のモード
    # 名**である。ここに残っていれば、フロントエンドはEditタブの「画角拡張」
    # サブタブを、動くモードのために灰色のままにしてしまう。
    assert "outpaint" not in features


def test_shipped_descriptors_match_their_familys_selection_fields():
    """契約(両記述子)。カテゴリをマニフェストに増やしても、その系統の
    ``SELECTION_FIELDS`` に入口が無ければGET /modelsに並ぶだけで何もしない。"""
    import importlib

    for base_id, descriptor in load_base_models(AppConfig().manifest_dir).items():
        module_path, _runner = engines.FAMILY_BY_ID[descriptor.engine_family]
        fields = importlib.import_module(module_path).SELECTION_FIELDS
        assert set(descriptor.categories) == set(fields), base_id


# --------------------------------------------------------------------------- #
# 4) 系統跨ぎの往復(mock)
# --------------------------------------------------------------------------- #


#: ``two_family_client`` (LTX 2.3 + LTX 2.5 installed side by side, both on the
#: mock backend) now lives in conftest.py — the API feature-guard suite needs
#: the same world, and two copies of it would be two worlds that drift.


def test_switching_family_replaces_the_runner_and_unloads_the_old_one(two_family_client):
    from services.engines.ltx.adapter import LTXRunner
    from services.engines.ltx25.adapter import LTX25Runner

    pm = two_family_client.app_context.pipeline_manager
    assert two_family_client.post("/api/v1/pipeline/load").status_code == 200
    old = pm.runner
    assert type(old) is LTXRunner and old.loaded

    # 旧Runnerのunloadが本当に呼ばれたことを見る(呼ばれなければ旧ワーカーの
    # プロセスが残り、VRAMを抱えたままになる — R14)。
    unloads: list[int] = []
    original_unload = old.unload
    old.unload = lambda: (unloads.append(1), original_unload())[1]  # type: ignore[method-assign]

    r = two_family_client.post("/api/v1/pipeline/load", json={"base_model": "LTX25"})
    assert r.status_code == 200, r.text
    assert r.json()["base_model"] == "LTX25"
    assert unloads, "系統を跨ぐ切り替えで旧Runnerのunloadが呼ばれていない"
    assert type(pm.runner) is LTX25Runner
    assert pm.runner is not old
    assert not old.loaded
    assert pm.runner.descriptor.id == "LTX25"


def test_switching_back_restores_the_2_3_runner(two_family_client):
    from services.engines.ltx.adapter import LTXRunner
    from services.engines.ltx25.adapter import LTX25Runner

    pm = two_family_client.app_context.pipeline_manager
    two_family_client.post("/api/v1/pipeline/load")
    two_family_client.post("/api/v1/pipeline/load", json={"base_model": "LTX25"})
    assert type(pm.runner) is LTX25Runner
    ltx25_runner = pm.runner

    r = two_family_client.post("/api/v1/pipeline/load", json={"base_model": "LTX23"})
    assert r.status_code == 200, r.text
    assert type(pm.runner) is LTXRunner
    assert pm.runner is not ltx25_runner
    assert not ltx25_runner.loaded
    assert pm.runner.descriptor.id == "LTX23"
    # 2周目も同じであること(1回だけ動く入れ替えは入れ替えではない)。
    assert two_family_client.post(
        "/api/v1/pipeline/load", json={"base_model": "LTX25"}
    ).status_code == 200
    assert type(pm.runner) is LTX25Runner


def test_the_mock_outcome_says_which_engine_ran(two_family_client):
    """metadata.jsonのbackendが系統ごとに違う——GPU無しでも往復が確かめられる、
    唯一の手掛かりである。"""
    from services.engines.ltx25.adapter import MOCK_BACKEND_25

    pm = two_family_client.app_context.pipeline_manager
    two_family_client.post("/api/v1/pipeline/load", json={"base_model": "LTX25"})
    pm.runner.load()
    assert pm.runner._backend.backend_label == MOCK_BACKEND_25

    two_family_client.post("/api/v1/pipeline/load", json={"base_model": "LTX23"})
    pm.runner.load()
    assert pm.runner._backend.backend_label == "mock"


def test_a_within_family_base_change_keeps_the_same_runner_object(tmp_path):
    """系統が同じなら差し替える理由が無い。set_descriptorがバックエンドを
    捨てるので、次のloadは新しいベースモデルのパスで組み直される。"""
    from services.engines.ltx.adapter import LTXRunner
    from services.low_vram import build_low_vram_settings
    from services.pipeline_manager import PipelineManager
    from services.job_store import JobStore
    from services.upload_store import UploadStore

    cfg = AppConfig.model_validate({"model": {"backend": "mock"}})
    second = BaseModelDescriptor(
        id="LTX23B", display_name="LTX 2.3 (second)", engine_family="ltx",
        categories=LTX23_DESCRIPTOR.categories,
    )

    class _Registry:
        def descriptor(self, base_model):
            return second

        def set_active_base_model(self, base_model):
            pass

    pm = PipelineManager(
        cfg, JobStore(), UploadStore(cfg),
        descriptor=LTX23_DESCRIPTOR, model_registry=_Registry(),
        active_base_model=LTX23_DESCRIPTOR.id,
    )
    assert type(pm.runner) is LTXRunner
    runner = pm.runner
    pm._point_runner_at(second)
    assert pm.runner is runner
    assert pm.runner.descriptor is second
    assert build_low_vram_settings(cfg) is not None  # 構築が成立していることの確認
