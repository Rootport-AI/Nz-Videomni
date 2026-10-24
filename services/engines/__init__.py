"""エンジン系統のディスパッチ層。

「どの重みを、どのアダプタが走らせるのか」を決める場所はここ一箇所である。

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

**判定の主体はKV、記述子は照合相手**。ユーザーが選んだ
ベースモデルが何を名乗っていようと、実際に走るのはファイルの中身である。
:func:`check_kv` は「KVが指す系統」と「記述子が宣言する系統」が食い違ったら
明示的に422で止める——黙ってどちらかに合わせると、LTX 2.3のワーカーに2.5の
重みが渡り、遥かに読みにくい失敗として現れる。
"""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Callable

from api.errors import model_incompatible, request_validation_failed

if TYPE_CHECKING:  # 実行時にはimportしない(上のdocstringの規約)
    from api.models import GenerateChainRequest, GenerateRequest
    from services.base_models import BaseModelDescriptor

#: (``general.architecture``, ``model_version``のminor) -> エンジン系統id。
#: 重みファイルのヘッダ(GGUFのKV、または量子化safetensorsのヘッダから読み替えた
#: KV)から読み取った事実だけで引ける表。ここに無い組み合わせは
#: 「この配布物が知らない世代」であり、系統の照合はせず各アダプタ自身の
#: ``check_kv`` に最終判断を委ねる(未知のminorを名乗る自作の重みなど)。
#:
#: MiniMax H3(``h3``)は**載せない**。H3 は GGUF を持たず、safetensors の
#: テンソル名の指紋で判定する(``h3_format``。事前検査
#: ``services.model_registry.precheck_model_file`` が系統ごとに呼び分ける)。
#: H3 の事前検査は KV を返さないので、:func:`check_kv` の系統の照合は
#: H3 では何もしない。
FAMILY_BY_KV: dict[tuple[str, str], str] = {
    ("ltxv", "2.3"): "ltx",
    ("ltxv", "2.5"): "ltx25",
}

#: エンジン系統id -> (アダプタのモジュールパス, Runnerクラス名)。
#: **クラスそのものではなく文字列**であることがこのモジュールの遅延性の実体。
FAMILY_BY_ID: dict[str, tuple[str, str]] = {
    "ltx": ("services.engines.ltx.adapter", "LTXRunner"),
    "ltx25": ("services.engines.ltx25.adapter", "LTX25Runner"),
    "h3": ("services.engines.h3.adapter", "H3Runner"),
}

#: エンジン系統id -> 画面に出る名前。系統不一致の422で「どちらを選べば
#: よいのか」を日本語で言うために使う(エラーメッセージが利用者向けの
#: 手順書になっている、という既存の流儀に合わせる)。
FAMILY_DISPLAY: dict[str, str] = {
    "ltx": "LTX 2.3",
    "ltx25": "LTX 2.5",
    "h3": "MiniMax H3",
}


def _minor(version: str) -> str:
    """``"2.5.0"`` -> ``"2.5"``。パッチ番号は系統を選ばない。

    LTX 2.3 アダプタの ``_minor_version``(LTX 2.5 アダプタもこれを import して
    使う)と同じ規則をここにも置いてある。1行の文字列処理のためにアダプタ
    (2000行超)をimportしては、このモジュールが遅延importである意味が消えるため。
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
    """KV(GGUFのKVヘッダ、または量子化safetensorsのヘッダから読み替えた辞書)
    だけから系統を引く。判らないときは None。

    Noneは「不正」ではなく「この表からは決められない」。アーキテクチャ名か
    世代のどちらかを欠く重み(自作・第三者製のGGUF、``model_version`` を持たない
    量子化safetensorsなど)はそれに当たり、系統の照合を見送って各アダプタの
    ``check_kv`` に最終判断を委ねる。欠損キーを拒否せず警告で通す方針の正本は
    MULTI_ENGINE_DESIGN.md §2.2。
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
    """系統のRunnerクラス(対応は ``FAMILY_BY_ID``)。

    ``PipelineManager`` がベースモデルを切り替えるときに、Runnerオブジェクト
    そのものを作り直すために使う。系統が変わればワーカーのプロセスも、venvも、
    プロトコルのペイロードも別物になるため、記述子を差し替えるだけでは足りない。
    """
    module_path, runner_name = FAMILY_BY_ID[_require_family(family)]
    return getattr(importlib.import_module(module_path), runner_name)


def unsupported_features(family: str) -> tuple[str, ...]:
    """この系統が扱えない機能名(GET /modelsが公開する)。

    宣言していない系統は空タプル——「制限なし」が既定であって、機能表を
    持たないことが「全部だめ」を意味してはならない。新しい系統を足した人が
    この関数を知らなくても、正しく動く側に倒れる。
    """
    return tuple(getattr(_adapter(family), "UNSUPPORTED_FEATURES", ()))


def reject_unsupported(family: str, request: GenerateRequest) -> None:
    """この系統が走らせられないリクエストなら、その場で422にする。

    **判断はアダプタが持ち、ここは取り次ぐだけ**である。どの系統がどの
    フィールドを断るかはエンジンの事実(各アダプタの ``REJECT_TABLE`` が正本)
    であって、エンドポイントの事情ではない。api/generate.py が系統名で分岐して
    機能表を持ち始めた瞬間に、同じ表が2箇所に生まれて必ずずれる。

    宣言していない系統は素通り。``getattr`` で見に行くのは、「制限を宣言
    しない」が既定であるという :func:`unsupported_features` と同じ規約に
    よる——新しい系統を足した人が、この関数の存在を知らなくても正しく動く側に
    倒れる。

    エンドポイントから**関数として**呼ぶ(FastAPIの ``Depends`` にはしない)。
    Dependsは引数の解決順に依存するため、リクエスト本文の検証と機能の可否の
    どちらが先に効くかが読めなくなる。
    """
    guard = getattr(_adapter(family), "reject_unsupported", None)
    if guard is not None:
        guard(request)


def request_rules(family: str):
    """この系統の数値の規約表(アダプタの ``REQUEST_RULES``)。宣言なしは None。

    表の型は ``services.engines.ltx.request_rules.RequestRules``。
    """
    return getattr(_adapter(family), "REQUEST_RULES", None)


def request_constraints(family: str) -> dict | None:
    """GET /models が公開する ``constraints``。規約表から機械的に作る。

    **正本はアダプタの表1か所**で、操作パネル・Gradio・MCP はこれを読む。
    表を宣言しない系統は None(「規約を公開しない」)。
    """
    rules = request_rules(family)
    return None if rules is None else rules.constraints()


def count_prompt_tokens(
    family: str,
    texts: list[str],
    *,
    assets: Callable[[str], Path],
    text_encoder: Callable[[], Path],
) -> list[int] | None:
    """各プロンプトのトークン数(POST /utils/prompt-tokens)。宣言なしは None。

    **数え方はアダプタの ``count_prompt_tokens`` が持ち、ここは取り次ぐだけ**。
    トークナイザーの種類・資産の在りか・前処理はエンジンの知識だからである
    (上限は規則表の ``prompt_token_limit``)。``assets`` は記述子の資産キー→
    絶対パス、``text_encoder`` は選択中のテキストエンコーダーの絶対パスを返す
    関数で、使うアダプタが呼んだときだけ解決される(使わない系統は、その解決の
    失敗に巻き込まれない)。数えられないときの例外はそのまま上げる。
    """
    counter = getattr(_adapter(family), "count_prompt_tokens", None)
    if counter is None:
        return None
    return counter(texts, assets=assets, text_encoder=text_encoder)


def validate_for_family(family: str, request: GenerateRequest) -> None:
    """系統の数値の規約で ``request`` を検査する(違反は 422 VALIDATION_ERROR)。

    **``None`` の解決もここ1か所**。``num_inference_steps``(と H3 の shift)は
    スキーマの既定が ``None``＝「系統の既定値」で、ここで系統の既定値
    (各アダプタの ``REQUEST_RULES``)を要求に**書き戻す**。以後の読み手
    (mock・ジョブ開始ログ・アダプタ・metadata.json・mp4 のレシピ)は
    常に数値を見る。

    順序は ``fill_defaults`` → ``violation`` → ``normalize_keyframes``。
    キーフレームの位置を系統の格子へ寄せるのは、``num_frames`` が正しいと
    分かってから(寄せる計算がフレーム数を使う)。同じ段で、省略された
    キーフレームの ``strength`` に系統の既定(LTX は 0.8。H3 は既定なし＝
    ``None`` のまま)を書き戻す。位置の重複も同じ 422 になる。

    呼ぶ場所は api/generate.py の :func:`reject_unsupported` の**後**と、
    api/generate_alpha.py(内部で組んだ要求の既定値の書き戻しのため)。
    機能の可否(FEATURE_UNSUPPORTED)が数値の違反(VALIDATION_ERROR)より
    先に返る。422 の形は pydantic の検証の失敗と同じ
    (``api.errors.request_validation_failed``)。

    宣言していない系統は素通り(:func:`reject_unsupported` と同じ規約)。
    """
    rules = request_rules(family)
    if rules is None:
        return
    rules.fill_defaults(request)
    message = rules.violation(request)
    if message is None:
        message = rules.normalize_keyframes(request.conditioning_images, request.num_frames)
    if message is not None:
        raise request_validation_failed(message)


def reject_chain(family: str, request: GenerateChainRequest) -> None:
    """この系統が走らせられない連結生成リクエストなら、その場で422にする。

    :func:`reject_unsupported` と対になる、chain系エンドポイント用の入口。
    :func:`reject_unsupported` と同じく、**判断はアダプタが持ち、ここは
    取り次ぐだけ**である。

    **系統によってリクエストの中身で答えが変わる**ので、リクエストを渡す。
    **どのフィールドを断るかの正本は各アダプタの ``CHAIN_REJECT_TABLE``
    であり、ここには書き写さない**——正本は1枚ではなく系統ごとに1枚である。

    何も宣言していない系統は素通り、という :func:`reject_unsupported` と
    同じ規約。
    """
    guard = getattr(_adapter(family), "reject_chain", None)
    if guard is not None:
        guard(request)
