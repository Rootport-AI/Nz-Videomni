"""System-level tools: backend reachability, config, and pipeline/model
management (6 tools total, per the approved plan's module list).

Note: ``reload_loras`` was cut per the plan's review item 6 -- ``GET /loras``
already rescans on every call (``api/loras.py``), so a separate reload tool
would be redundant.
"""

from __future__ import annotations

from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

from mcp_server.client import get_client

_WAIT_SEC_MIN = 5
_WAIT_SEC_MAX = 45


async def backend_status() -> dict[str, Any]:
    """バックエンドの疎通状況を確認します。

    他のどのツールよりも先に呼ぶことを想定しています。バックエンドが起動して
    いない・接続できない場合でも例外を投げず、``reachable: false`` を返します
    （それ以外のツールはエラー時に例外を投げます）。

    Returns:
        reachable: バックエンドに接続できたか。
        base_url: 接続を試みたベースURL（例: http://127.0.0.1:18620）。
        api_key_configured: Bearer認証キーが設定されているか。
        pipeline_load_in_flight: このMCPサーバーがload_pipelineの応答待ち中か。
        status: 接続できた場合のみ、GET /status の内容。主なフィールドは
            ``status.state``（パイプラインの状態。``unloaded`` / ``loading``
            / ``ready`` / ``running`` / ``error``）と ``status.base_model``
            （現在選択中のベースモデルのid。例: ``"LTX23"`` / ``"LTX25"``）。
        error: 接続できなかった場合のみ、エラーの説明。
    """
    client = get_client()
    result: dict[str, Any] = {
        "reachable": False,
        "base_url": client.base_url,
        "api_key_configured": client.api_key_configured,
        "pipeline_load_in_flight": client.pipeline_load_in_flight,
    }
    try:
        status = await client.get_json("/status")
    except Exception as exc:  # noqa: BLE001 -- this tool must NEVER raise
        result["error"] = str(exc)
        return result
    result["reachable"] = True
    result["status"] = status
    return result


async def get_config() -> dict[str, Any]:
    """バックエンドの実効設定を取得します（GET /config）。

    解像度・フレーム数の上限、アップロード可能な拡張子、V2V/A2Vの範囲など、
    他のツールを呼ぶ前に確認しておくと良い情報が含まれます。

    Returns:
        設定オブジェクト全体（server / model / vram / limits / upload / output
        などのセクション）。
    """
    client = get_client()
    return await client.get_json("/config")


async def list_models() -> dict[str, Any]:
    """選択可能なモデル（transformer / text_encoder / video_vae / audio）を
    カテゴリ別に一覧します（GET /models）。

    各カテゴリに現在有効な名前（active）と選択肢（entries）が含まれます。
    load_pipeline の ``models`` 引数にはここで得られる名前を渡してください。

    ベースモデル（LTX 2.3 / LTX 2.5 など、推論エンジンごと入れ替わる大枠）も
    このツールで確認します。``categories`` は**現在選択中のベースモデルのもの**
    であり、他のベースモデルの選択肢は ``base_models[]`` の各要素が持つ同名の
    ブロックにあります。

    Returns:
        categories: カテゴリ名 -> {default, active, entries} の辞書
            （現在選択中のベースモデルのもの）。
        active_base_model: 現在選択中のベースモデルのid（例: ``"LTX23"``）。
        base_models: 宣言されている全ベースモデルのリスト。主なフィールドは
            ``id``（load_pipeline の ``base_model`` 引数に渡す値）、
            ``display_name``（表示名）、``active``（現在選択中か）、
            ``installed``（全カテゴリの既定ファイルが揃っているか）、
            ``missing_categories``（欠けているカテゴリ名のリスト）、
            ``unsupported_features``（そのベースモデルの推論エンジンが実行
            できない機能名のリスト）。ほかに ``engine_family`` / ``present``
            / ``category_order`` / ``categories`` を含みます。
            ``unsupported_features`` に載っている機能を submit_generate /
            submit_chain で使うと 422 FEATURE_UNSUPPORTED になります。
    """
    client = get_client()
    return await client.get_json("/models")


async def load_pipeline(
    models: dict[str, str] | None = None,
    wait_sec: int = 45,
    base_model: str | None = None,
) -> dict[str, Any]:
    """パイプライン（推論モデル一式）を読み込みます（POST /pipeline/load）。

    ジョブ実行中は 409 JOB_BUSY になります（同時1ジョブ制約）。読み込みには
    数十秒〜数分かかることがあるため、``wait_sec``（既定45秒、5〜45秒にクランプ）
    だけ応答を待ち、それでも終わらない場合はタイムアウトを検出して
    ``finished: false`` を返します（エラーにはしません -- バックエンド側の処理
    は接続が切れても続行します）。

    Args:
        models: カテゴリ名 -> モデル名 の辞書（省略時は現在の選択のまま読み込み）。
            list_models で得られる名前を指定してください。
        wait_sec: 応答を待つ秒数（5〜45にクランプ）。
        base_model: 切り替えたいベースモデルのid（``list_models`` の
            ``base_models[].id``。例: ``"LTX23"`` / ``"LTX25"``）。省略時は
            現在のベースモデルを維持します。切り替えはワーカーの載せ替えを
            伴うため数秒〜十数秒かかります。エラーは、ジョブ実行中なら
            409 JOB_BUSY、未知のidなら 404 MODEL_NOT_FOUND、重みが未導入なら
            422 MODEL_FILE_MISSING、重みとベースモデルの系統が食い違うなら
            422 MODEL_INCOMPATIBLE、既に読み込み処理中なら 409 PIPELINE_LOADING
            です。**切り替え直後の1本目の生成はキャッシュが冷えているため
            通常の約2倍かかります**（``wait_for_job`` がタイムアウトしても
            失敗ではないので、複数回呼び直してください）。

    Returns:
        no_op branch: pipeline_loaded=True かつ models・base_model の両方を
            省略したとき、何もせず現状を返す（``no_op: true`` 付き）。
        in_flight branch: 既に読み込み中の場合、POSTを送らず
            ``in_flight: true`` を返す。
        timeout branch: 応答待ちがタイムアウトした場合、
            ``started: true, finished: false, hint: ...`` を返す。
        それ以外: バックエンドの応答（pipeline_loaded / pipeline_type / state
            / models / base_model）に ``finished: true`` を加えたもの。
            ``base_model`` と ``models`` は ``models`` か ``base_model`` の
            いずれかを指定したときだけ応答に載ります（引数なしの読み込みの
            応答形は従来どおり3キーのままです）。
    """
    client = get_client()
    wait_sec = max(_WAIT_SEC_MIN, min(_WAIT_SEC_MAX, wait_sec))

    if models is None and base_model is None:
        status = await client.get_json("/status")
        if status.get("pipeline_loaded"):
            return {
                "no_op": True,
                "pipeline_loaded": True,
                "pipeline_type": status.get("pipeline_type"),
            }

    if client.pipeline_load_in_flight:
        return {
            "in_flight": True,
            "started": False,
            "finished": False,
            "hint": (
                "既に読み込み処理が進行中です。しばらくしてから backend_status で "
                "pipeline_loaded を確認してください。"
            ),
        }

    client.pipeline_load_in_flight = True
    try:
        body: dict[str, Any] = {}
        if models:
            body["models"] = models
        if base_model:
            body["base_model"] = base_model
        try:
            result = await client.post_json("/pipeline/load", json=body, timeout=wait_sec)
        except httpx.ReadTimeout:
            return {
                "started": True,
                "finished": False,
                "hint": (
                    "応答がタイムアウトしましたが、バックエンド側では読み込みが継続している"
                    "可能性があります。backend_status で pipeline_loaded を確認してください。"
                ),
            }
        result["finished"] = True
        return result
    finally:
        client.pipeline_load_in_flight = False


async def unload_pipeline() -> dict[str, Any]:
    """パイプラインをメモリから解放します（POST /pipeline/unload）。

    ジョブ実行中は 409 JOB_BUSY になります（同時1ジョブ制約）。

    選択中のベースモデルは解放後も保持されます（次に load_pipeline を
    ``base_model`` なしで呼べば、同じベースモデルが読み込まれます）。

    Returns:
        pipeline_loaded: 解放後の状態（False のはず）。
        state: パイプラインマネージャの状態文字列。
    """
    client = get_client()
    return await client.post_json("/pipeline/unload")


async def list_loras() -> dict[str, Any]:
    """選択可能な IC-LoRA アダプタを一覧します（GET /loras）。

    スタイル系（style）と制御系（control、参照動画が必須）の両方を含みます。
    呼ぶたびにサーバー側でフォルダを再スキャンするので、追加した直後の
    アダプタも再起動なしで見えます。

    Returns:
        loras: 各アダプタの {name, kind, has_thumbnail, source, ...} のリスト。
    """
    client = get_client()
    return await client.get_json("/loras")


def register(mcp: FastMCP) -> None:
    mcp.tool()(backend_status)
    mcp.tool()(get_config)
    mcp.tool()(list_models)
    mcp.tool()(load_pipeline)
    mcp.tool()(unload_pipeline)
    mcp.tool()(list_loras)
