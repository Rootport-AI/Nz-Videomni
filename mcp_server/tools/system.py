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
        status: 接続できた場合のみ、GET /status の内容。
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

    Returns:
        categories: カテゴリ名 -> {default, active, entries} の辞書。
    """
    client = get_client()
    return await client.get_json("/models")


async def load_pipeline(models: dict[str, str] | None = None, wait_sec: int = 45) -> dict[str, Any]:
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

    Returns:
        no_op branch: pipeline_loaded=True かつ models 省略時、何もせず現状を返す
            （``no_op: true`` 付き）。
        in_flight branch: 既に読み込み中の場合、POSTを送らず
            ``in_flight: true`` を返す。
        timeout branch: 応答待ちがタイムアウトした場合、
            ``started: true, finished: false, hint: ...`` を返す。
        それ以外: バックエンドの応答（pipeline_loaded / pipeline_type / state
            / models）に ``finished: true`` を加えたもの。
    """
    client = get_client()
    wait_sec = max(_WAIT_SEC_MIN, min(_WAIT_SEC_MAX, wait_sec))

    if models is None:
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
