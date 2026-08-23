"""エンジン系統のディスパッチ層(§3-98 P3c)。

ベースモデルが2つになった時点で「どの重みを、どのアダプタが走らせるのか」を
決める場所が必要になった。ここがその一箇所である。

**すべて遅延import**。この モジュールを import しただけでは、どのアダプタも
読み込まれない。理由は二つある。(1) LTXアダプタは2000行を超え、PIL・
chain_math・api.models を芋づるに引き込む。ベースモデルを列挙したいだけの
プロセス(インストーラ、MCP、起動時のマニフェスト検査)がその代金を払う理由は
ない。(2) エンジンが増えるたびに、無関係な系統まで必ず import される構造には
したくない。dictを1つ置いた瞬間にそうなるため、ここでは**モジュールのパスを
文字列で持ち**、呼ばれた系統だけを importlib で解決する。

``services.base_models`` も import しない。記述子は「エンジン層が読む純粋な
データ」であり、記述子→アダプタ→記述子 の循環importを作らないための規約
(services/base_models.py 冒頭の INVARIANT)がある。型注釈は
``from __future__ import annotations`` により文字列のままなので、
TYPE_CHECKING ガードの下で書ける。

**判定の主体はKV、記述子は照合相手**(設計原則F-1)。ユーザーが選んだ
ベースモデルが何を名乗っていようと、実際に走るのはファイルの中身である。
:func:`check_kv` は「KVが指す系統」と「記述子が宣言する系統」が食い違ったら
明示的に422で止める——黙ってどちらかに合わせると、LTX 2.3のワーカーに2.5の
重みが渡り、遥かに読みにくい失敗として現れる。
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import TYPE_CHECKING

from api.errors import model_incompatible

if TYPE_CHECKING:  # 実行時にはimportしない(上のdocstringの規約)
    from api.models import GenerateChainRequest, GenerateRequest
    from services.base_models import BaseModelDescriptor

#: (``general.architecture``, ``model_version``のminor) -> エンジン系統id。
#: GGUFヘッダから読み取った事実だけで引ける表。ここに無い組み合わせは
#: 「この配布物が知らない世代」であり、系統の照合はせず各アダプタ自身の
#: ``check_kv`` に最終判断を委ねる(未知のminorを名乗る自作GGUFなど)。
FAMILY_BY_KV: dict[tuple[str, str], str] = {
    ("ltxv", "2.3"): "ltx",
    ("ltxv", "2.5"): "ltx25",
}

#: エンジン系統id -> (アダプタのモジュールパス, Runnerクラス名)。
#: **クラスそのものではなく文字列**であることがこのモジュールの遅延性の実体。
FAMILY_BY_ID: dict[str, tuple[str, str]] = {
    "ltx": ("services.engines.ltx.adapter", "LTXRunner"),
    "ltx25": ("services.engines.ltx25.adapter", "LTX25Runner"),
}

#: エンジン系統id -> 画面に出る名前。系統不一致の422で「どちらを選べば
#: よいのか」を日本語で言うために使う(エラーメッセージが利用者向けの
#: 手順書になっている、という既存の流儀に合わせる)。
FAMILY_DISPLAY: dict[str, str] = {
    "ltx": "LTX 2.3",
    "ltx25": "LTX 2.5",
}


def _minor(version: str) -> str:
    """``"2.5.0"`` -> ``"2.5"``。パッチ番号は系統を選ばない。

    各アダプタの ``_minor_version`` と同じ規則をここにも置いてある。1行の
    文字列処理のためにアダプタ(2000行超)をimportしては、このモジュールが
    遅延importである意味が消えるため。
    """
    return ".".join(version.strip().split(".")[:2])


def _require_family(family: str) -> str:
    if family not in FAMILY_BY_ID:
        raise RuntimeError(
            f"未知のエンジン系統 '{family}' です。この配布物が持つのは "
            f"{sorted(FAMILY_BY_ID)} です(マニフェストの engine_family を確認してください)。"
        )
    return family


def _adapter(family: str) -> ModuleType:
    """系統のアダプタモジュール。**ここで初めて**実際のimportが走る。"""
    module_path, _runner = FAMILY_BY_ID[_require_family(family)]
    return importlib.import_module(module_path)


def family_by_kv(kv: dict[str, str]) -> str | None:
    """GGUFのKVヘッダだけから系統を引く。判らないときは None。

    Noneは「不正」ではなく「この表からは決められない」。アーキテクチャ名か
    世代のどちらかを欠くGGUF(自作・第三者製)はそれに当たり、系統の照合を
    見送って各アダプタの ``check_kv`` に最終判断を委ねる——欠損キーを拒否
    しないのは§2.5からの一貫した方針である。
    """
    architecture = (kv.get("general.architecture") or "").strip()
    version = (kv.get("model_version") or "").strip()
    if not architecture or not version:
        return None
    return FAMILY_BY_KV.get((architecture, _minor(version)))


def check_kv(
    descriptor: BaseModelDescriptor,
    category: str,
    name: str,
    kv: dict[str, str],
) -> None:
    """読み込もうとしている重みのKVを、選択中のベースモデルと突き合わせる。

    二段構え:

    1. **系統の照合**(このモジュールの仕事)。KVが指す系統と、記述子が宣言
       する系統が食い違えば422。判定の主体はKV——ファイルの中身が本当のこと
       を言っているのであって、記述子は「ユーザーが何を選んだか」でしかない。
       ここで止めれば、利用者へのメッセージは「LTX 2.5を選んでください」と
       いう次の一手になる。
    2. **系統内の判定**(各アダプタの仕事)。アーキテクチャが ``ltxv`` かどうか、
       扱える世代かどうか、キー欠損の警告。系統が一致していても、その系統が
       走らせられない世代というものはあり得る。

    transformer以外のカテゴリは1も2も素通りする(世代を定義するのは
    transformerだけで、VAEやテキストエンコーダはその刻印を持たない)。
    """
    family = _require_family(descriptor.engine_family)
    if category == "transformer":
        kv_family = family_by_kv(kv)
        if kv_family is not None and kv_family != family:
            version = (kv.get("model_version") or "").strip()
            raise model_incompatible(
                category,
                name,
                detail=(
                    f"このtransformerはltxv {version}（{FAMILY_DISPLAY[kv_family]}）です。"
                    f"選択中のベースモデル「{descriptor.display_name}」は"
                    f"{FAMILY_DISPLAY[family]}用のため読み込めません。"
                    f"ベースモデルに「{FAMILY_DISPLAY[kv_family]}」を選んでください。"
                ),
            )
    _adapter(family).check_kv(category, name, kv)


def runner_class_for(family: str) -> type:
    """系統のRunnerクラス(``LTXRunner`` / ``LTX25Runner``)。

    ``PipelineManager`` がベースモデルを切り替えるときに、Runnerオブジェクト
    そのものを作り直すために使う。系統が変わればワーカーのプロセスも、venvも、
    プロトコルのペイロードも別物になるため、記述子を差し替えるだけでは足りない。
    """
    module_path, runner_name = FAMILY_BY_ID[_require_family(family)]
    return getattr(importlib.import_module(module_path), runner_name)


def unsupported_features(family: str) -> tuple[str, ...]:
    """この系統が扱えない機能名(GET /modelsが公開する。§3-98 Phase 5)。

    宣言していない系統は空タプル——「制限なし」が既定であって、機能表を
    持たないことが「全部だめ」を意味してはならない。
    """
    return tuple(getattr(_adapter(family), "UNSUPPORTED_FEATURES", ()))


def reject_unsupported(family: str, request: GenerateRequest) -> None:
    """この系統が走らせられないリクエストなら、その場で422にする(§3-98 P5)。

    **判断はアダプタが持ち、ここは取り次ぐだけ**である。「LTX 2.5(v1)では
    outpaintができない」というのはエンジンの事実であって、エンドポイントの
    事情ではない。api/generate.py が系統名で分岐して機能表を持ち始めた瞬間に、
    同じ表が2箇所に生まれて必ずずれる。

    宣言していない系統(``ltx``)は素通り。``getattr`` で見に行くのは、
    「制限を宣言しない」が既定であるという :func:`unsupported_features` と
    同じ規約による——新しい系統を足した人が、この関数の存在を知らなくても
    正しく動く側に倒れる。

    エンドポイントから**関数として**呼ぶ(FastAPIの ``Depends`` にはしない)。
    Dependsは引数の解決順に依存するため、リクエスト本文の検証と機能の可否の
    どちらが先に効くかが読めなくなる。
    """
    guard = getattr(_adapter(family), "reject_unsupported", None)
    if guard is not None:
        guard(request)


def reject_chain(family: str, request: GenerateChainRequest) -> None:
    """この系統が走らせられない連結生成リクエストなら、その場で422にする。

    :func:`reject_unsupported` と対になる、chain系エンドポイント用の入口。
    :func:`reject_unsupported` と同じく、**判断はアダプタが持ち、ここは
    取り次ぐだけ**である。

    かつては引数にリクエストを取らなかった。「chain系はまるごと扱えるか扱えない
    かのどちらかで、中身を見ても答えが変わらない」からだった——LTX 2.5が連結生成
    を一切できなかった頃の話である。いまは**系統によって中身で答えが変わる**:
    素のChainedは2.5でも走り、V2V・A2V・Retake・End source・LoRA・参照動画は
    走らない。だからリクエストを渡す。

    宣言していない系統(``ltx``)は素通り、という :func:`reject_unsupported` と
    同じ規約。
    """
    guard = getattr(_adapter(family), "reject_chain", None)
    if guard is not None:
        guard(request)
