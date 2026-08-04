"""ジョブ管理ツール（7本）。

``job_status`` / ``list_jobs`` / ``wait_for_job`` / ``cancel_job`` /
``delete_job`` / ``purge_terminal_jobs`` / ``join_job``。

状態ガードの方針（計画D9）: ``cancel_job`` / ``delete_job`` は呼ぶ前に必ず
GETで現在の状態を確認し、意図と食い違う場合はエラーにする（1エンドポイント
2動作の ``DELETE /jobs/{id}`` を暗黙に踏ませない）。
"""

from __future__ import annotations

from typing import Any

import anyio
import httpx
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations

from mcp_server.client import get_client

_TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
_ACTIVE_STATUSES = {"queued", "running"}

_WAIT_TIMEOUT_MIN = 1.0
_WAIT_TIMEOUT_MAX = 45.0
_WAIT_POLL_MIN = 0.5
_WAIT_POLL_MAX = 10.0
_JOIN_WAIT_MIN = 5.0
_JOIN_WAIT_MAX = 45.0


def _clamp_wait_timeout_sec(value: float) -> float:
    """``wait_for_job`` の ``timeout_sec`` を [1, 45] にクランプする純関数。"""
    return max(_WAIT_TIMEOUT_MIN, min(_WAIT_TIMEOUT_MAX, value))


def _clamp_poll_interval_sec(value: float) -> float:
    """``wait_for_job`` の ``poll_interval_sec`` を [0.5, 10] にクランプする純関数。"""
    return max(_WAIT_POLL_MIN, min(_WAIT_POLL_MAX, value))


def _clamp_join_wait_sec(value: float) -> float:
    """``join_job`` の ``wait_sec``（読み取りタイムアウト秒）を [5, 45] にクランプする純関数。"""
    return max(_JOIN_WAIT_MIN, min(_JOIN_WAIT_MAX, value))


async def job_status(job_id: str) -> dict[str, Any]:
    """1件のジョブの詳細を取得します（GET /jobs/{job_id}、全文）。

    リクエスト本文（プロンプト等）まで含めた完全な情報が必要なときはこちらを
    使ってください。一覧で概要だけ見たい場合は ``list_jobs`` を使ってください。

    Args:
        job_id: submit_generate / submit_chain が返した job_id。

    Returns:
        JobResponse相当のフィールド一式（status, progress, stage, clip,
        clip_count, is_v2v, joined, created_at, started_at, completed_at,
        error, request, result）。
    """
    client = get_client()
    return await client.get_json(f"/jobs/{job_id}")


async def list_jobs() -> dict[str, Any]:
    """全ジョブの一覧を要約付きで取得します（GET /jobs、射影あり）。

    各ジョブの ``request``（プロンプト全文など）はここには含めません --
    ジョブ数が多いと合計トークン量が膨らむため。全文が必要な場合は
    ``job_status(job_id)`` を個別に呼んでください。

    Returns:
        jobs: 各ジョブの job_id / status / progress / stage / clip /
            clip_count / is_v2v / joined / created_at / error / has_result
            の辞書のリスト。
        counts: status別の件数（queued/running/completed/failed/cancelled）。
    """
    client = get_client()
    records: list[dict[str, Any]] = await client.get_json("/jobs")  # type: ignore[assignment]

    counts: dict[str, int] = {
        "queued": 0, "running": 0, "completed": 0, "failed": 0, "cancelled": 0,
    }
    summaries: list[dict[str, Any]] = []
    for r in records:
        status = r.get("status")
        if status in counts:
            counts[status] += 1
        summaries.append(
            {
                "job_id": r.get("job_id"),
                "status": status,
                "progress": r.get("progress"),
                "stage": r.get("stage"),
                "clip": r.get("clip"),
                "clip_count": r.get("clip_count"),
                "is_v2v": r.get("is_v2v"),
                "joined": r.get("joined"),
                "created_at": r.get("created_at"),
                "error": r.get("error"),
                "has_result": r.get("result") is not None,
            }
        )
    return {"jobs": summaries, "counts": counts}


async def wait_for_job(
    job_id: str, timeout_sec: float = 45, poll_interval_sec: float = 2.0
) -> dict[str, Any]:
    """ジョブが終端状態（completed/failed/cancelled）になるまで待ちます。

    ``timeout_sec`` は1〜45秒にクランプされます（MCPのツール呼び出し自体が
    長時間ブロックし続けるのを避けるため）。タイムアウトしてもエラーには
    せず、その時点のジョブ状況に ``timed_out: true`` を付けて返します --
    同じ引数でもう一度呼べば続きから待てます。

    Args:
        job_id: 待ちたいジョブのID。
        timeout_sec: 最大待機秒数（1〜45にクランプ）。
        poll_interval_sec: ポーリング間隔秒数（0.5〜10にクランプ）。

    Returns:
        job_status と同じフィールド一式に、timed_out（bool）・waited_sec
        （実際に待った秒数）を追加したもの。タイムアウト時は hint も追加。
    """
    client = get_client()
    timeout_sec = _clamp_wait_timeout_sec(timeout_sec)
    poll_interval_sec = _clamp_poll_interval_sec(poll_interval_sec)

    start = anyio.current_time()
    while True:
        status = await client.get_json(f"/jobs/{job_id}")
        elapsed = anyio.current_time() - start
        if status.get("status") in _TERMINAL_STATUSES:
            status["timed_out"] = False
            status["waited_sec"] = elapsed
            return status
        if elapsed >= timeout_sec:
            status["timed_out"] = True
            status["waited_sec"] = elapsed
            status["hint"] = "同じ引数でもう一度呼んでください"
            return status
        await anyio.sleep(poll_interval_sec)


async def cancel_job(job_id: str) -> dict[str, Any]:
    """ジョブをキャンセルします（DELETE /jobs/{job_id}、未終了ジョブ専用）。

    既に終端状態（completed/failed/cancelled）のジョブに対しては呼べません
    （エラーになります）-- 終了済みジョブの出力を消したい場合は
    ``delete_job`` を使ってください。

    queued（未着手）のジョブは即座にキャンセルされます。running（実行中）の
    ジョブはベストエフォートです -- Phase 1では推論を中断できないため、
    ``cancel_requested`` フラグが立つだけで、実際に停止する保証はありません。

    Args:
        job_id: キャンセルしたいジョブのID。

    Returns:
        バックエンドの応答（job_id, cancelled/cancel_requested, status）に
        挙動説明の note を追加したもの。
    """
    client = get_client()
    current = await client.get_json(f"/jobs/{job_id}")
    status = current.get("status")
    if status in _TERMINAL_STATUSES:
        raise ToolError(
            f"JOB_ALREADY_TERMINAL: job {job_id} is already {status} — "
            "delete_job を使ってください"
        )
    result = await client.delete_json(f"/jobs/{job_id}")
    result["note"] = (
        "queued だったジョブは即座にキャンセルされます。running だったジョブは"
        "ベストエフォート（cancel_requested フラグのみ）で、実際に停止する保証は"
        "ありません。"
    )
    return result


async def delete_job(job_id: str) -> dict[str, Any]:
    """終了済みジョブの記録と出力フォルダを削除します（DELETE /jobs/{job_id}）。

    まだ実行中/待機中（queued/running）のジョブに対しては呼べません（エラー
    になります）-- 実行中のジョブを止めたい場合は先に ``cancel_job`` を呼んで
    ください。

    Args:
        job_id: 削除したいジョブのID（終端状態である必要があります）。

    Returns:
        バックエンドの応答（job_id, deleted: true）。
    """
    client = get_client()
    current = await client.get_json(f"/jobs/{job_id}")
    status = current.get("status")
    if status in _ACTIVE_STATUSES:
        raise ToolError(
            f"JOB_STILL_ACTIVE: job {job_id} is still {status} — "
            "cancel_job を先に呼んでください"
        )
    return await client.delete_json(f"/jobs/{job_id}")


async def purge_terminal_jobs(dry_run: bool = False) -> dict[str, Any]:
    """終了済み（completed/failed/cancelled）ジョブをまとめて削除します。

    実行中/待機中のジョブには一切触れません。個別の削除が失敗しても処理を
    止めず、失敗したジョブだけ ``failed`` に記録して続行します。

    Args:
        dry_run: True の場合、実際には削除せず対象件数だけ確認します。

    Returns:
        attempted（対象件数）, deleted（削除できた件数）, failed（{job_id,
        error} のリスト）, dry_run。
    """
    client = get_client()
    records: list[dict[str, Any]] = await client.get_json("/jobs")  # type: ignore[assignment]
    candidates = [r["job_id"] for r in records if r.get("status") in _TERMINAL_STATUSES]

    if dry_run:
        return {"attempted": len(candidates), "deleted": 0, "failed": [], "dry_run": True}

    deleted = 0
    failed: list[dict[str, Any]] = []
    for job_id in candidates:
        try:
            await client.delete_json(f"/jobs/{job_id}")
            deleted += 1
        except ToolError as exc:
            failed.append({"job_id": job_id, "error": str(exc)})
    return {"attempted": len(candidates), "deleted": deleted, "failed": failed, "dry_run": False}


async def join_job(
    job_id: str,
    audio_smoothing: bool = True,
    handle_crossfade_ms: int = 300,
    source_tail_seconds: float = 5.0,
    wait_sec: float = 45,
) -> dict[str, Any]:
    """V2V継続ジョブの音声を元動画に繋ぎ直します（POST /jobs/{job_id}/join）。

    V2V（``submit_chain`` の ``source_video`` を使ったジョブ）専用です。
    それ以外のジョブ（単発生成・A2V・source_videoなしのチェーン）に対して
    呼ぶとエラーになります（``job_status`` の ``is_v2v`` で事前に確認できます）。

    GPU不要（ffmpegのみ）で、他のジョブと並行して実行できます。応答が
    ``wait_sec`` でタイムアウトした場合もエラーにはせず、``finished: false``
    を返します（バックエンド側の処理は接続が切れても続行するので、
    ``job_status`` の ``joined`` で完了を確認できます）。

    Args:
        job_id: V2Vジョブのjob_id。
        audio_smoothing: True（既定）ならクロスフェード、False なら単純結合。
        handle_crossfade_ms: クロスフェード長（ミリ秒、0〜2000）。
        source_tail_seconds: 元動画の末尾から何秒だけ残して繋ぐか（0で全長）。
        wait_sec: 応答を待つ秒数（5〜45にクランプ）。

    Returns:
        finished: true のとき、JoinResponse相当（job_id, joined_path,
            join_mode, source_normalized 等）。
        finished: false のとき、タイムアウトを示す hint のみ。
    """
    client = get_client()
    wait_sec = _clamp_join_wait_sec(wait_sec)
    body = {
        "audio_smoothing": audio_smoothing,
        "handle_crossfade_ms": handle_crossfade_ms,
        "source_tail_seconds": source_tail_seconds,
    }
    try:
        result = await client.post_json(f"/jobs/{job_id}/join", json=body, timeout=wait_sec)
    except httpx.ReadTimeout:
        return {
            "finished": False,
            "hint": (
                "応答がタイムアウトしましたが、バックエンド側では処理が継続している"
                "可能性があります。job_status の joined を確認してください。"
            ),
        }
    result["finished"] = True
    return result


def register(mcp: FastMCP) -> None:
    mcp.tool()(job_status)
    mcp.tool()(list_jobs)
    mcp.tool()(wait_for_job)
    mcp.tool()(cancel_job)
    mcp.tool(annotations=ToolAnnotations(destructiveHint=True))(delete_job)
    mcp.tool(annotations=ToolAnnotations(destructiveHint=True))(purge_terminal_jobs)
    mcp.tool()(join_job)
