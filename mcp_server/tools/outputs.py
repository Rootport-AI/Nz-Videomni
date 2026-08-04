"""出力動画の実パス取得・ローカル保存ツール（3本、W5）。

``get_job_video_path`` / ``get_joined_video_path`` は ``GET /jobs/{id}`` で
状態だけを確認し、実パスは常に ``mcp_server.client.BackendClient.output_dir``
（ローカル ``config.yaml`` 由来）から ``mcp_server.paths`` で組み立てる ――
レスポンスの ``result.output_path`` は相対ハードコードで信用できないため
（``mcp_server/paths.py`` の docstring参照）、絶対に使わない。

``save_job_video`` はHTTPを一切使わない（バックエンドへ問い合わせず、ローカル
``output_dir`` から直接コピーする）。ブロッキングI/O（``Path.exists`` /
``Path.stat`` / ``Path.mkdir`` / ``shutil.copy2``）は ``anyio.to_thread.run_sync``
経由で呼ぶ（計画D3）。
"""

from __future__ import annotations

import functools
import shutil
from pathlib import Path
from typing import Any, Literal

import anyio
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from mcp_server.client import get_client
from mcp_server.paths import job_joined_path, job_output_path, unique_dest


async def _stat_if_exists(path: Path) -> tuple[bool, int | None]:
    exists = await anyio.to_thread.run_sync(path.exists)
    if not exists:
        return False, None
    size = (await anyio.to_thread.run_sync(path.stat)).st_size
    return True, size


async def get_job_video_path(job_id: str) -> dict[str, Any]:
    """ジョブの出力動画（``output.mp4``）のローカル絶対パスを返します。

    ``GET /jobs/{job_id}`` でジョブの状態だけを確認し、パス自体はローカルの
    ``output_dir``（MCPサーバーの ``config.yaml``）から組み立てます。ジョブが
    未完了・失敗している場合でも例外は投げず、``exists: false`` を返します
    （ファイルが物理的に存在するかどうかは常に実際にチェックします）。

    Args:
        job_id: submit_generate / submit_chain が返した job_id。

    Returns:
        path: 動画の絶対パス（存在するかどうかに関わらず組み立てられる）。
        exists: そのパスに実際にファイルがあるか。
        size_bytes: 存在する場合のみファイルサイズ、それ以外は None。
        status: ジョブの現在の状態。
        note: status が completed でない場合のみ付与される注意書き。
    """
    client = get_client()
    job = await client.get_json(f"/jobs/{job_id}")
    status = job.get("status")

    path = job_output_path(client.output_dir, job_id)
    exists, size_bytes = await _stat_if_exists(path)

    result: dict[str, Any] = {
        "path": str(path),
        "exists": exists,
        "size_bytes": size_bytes,
        "status": status,
    }
    if status != "completed":
        result["note"] = (
            f"ジョブがまだ completed ではありません（status={status}）。"
            "動画はまだ存在しないか、不完全な可能性があります。"
        )
    return result


async def get_joined_video_path(job_id: str) -> dict[str, Any]:
    """V2V継続ジョブの結合済み動画（``joined.mp4``）のローカル絶対パスを返します。

    ``get_job_video_path`` と同じ流儀ですが、``joined.mp4``（``join_job`` が
    生成する）を対象にします。あわせて ``GET /jobs/{job_id}`` の ``joined``
    フラグ（結合が実行済みか）も返します。

    Args:
        job_id: V2Vジョブの job_id。

    Returns:
        path, exists, size_bytes, status に加えて joined（bool）。status が
        completed でない場合のみ note が付与されます。
    """
    client = get_client()
    job = await client.get_json(f"/jobs/{job_id}")
    status = job.get("status")

    path = job_joined_path(client.output_dir, job_id)
    exists, size_bytes = await _stat_if_exists(path)

    result: dict[str, Any] = {
        "path": str(path),
        "exists": exists,
        "size_bytes": size_bytes,
        "status": status,
        "joined": job.get("joined"),
    }
    if status != "completed":
        result["note"] = (
            f"ジョブがまだ completed ではありません（status={status}）。"
            "結合済み動画はまだ存在しない可能性があります。"
        )
    return result


async def save_job_video(
    job_id: str,
    dest_dir: str,
    filename: str | None = None,
    no_clobber: bool = True,
    which: Literal["output", "joined"] = "output",
) -> dict[str, Any]:
    """出力動画をローカルの任意フォルダへコピーします（バックエンドへは問い合わせません）。

    ``which="output"`` なら ``output.mp4``（既定ファイル名 ``{job_id}.mp4``）、
    ``which="joined"`` なら ``joined.mp4``（既定ファイル名
    ``{job_id}_joined.mp4``）をコピーします――どちらも
    ``api/jobs.py`` の ``FileResponse`` が付けるダウンロードファイル名と同じ
    命名です。コピー元が存在しない場合（ジョブ未完了 / join未実行など）は
    例外になります。

    Args:
        job_id: 対象ジョブの job_id。
        dest_dir: コピー先フォルダ（存在しなければ作成します）。
        filename: 保存ファイル名（省略時は既定ファイル名）。
        no_clobber: True（既定）なら同名ファイルがあった場合 ``_2`` ``_3`` ...
            と連番を振って衝突を避けます。False なら上書きします。
        which: "output"（通常の生成結果）か "joined"（V2V結合結果）。

    Returns:
        saved_path: 実際に保存された絶対パス。
        source_path: コピー元の絶対パス。
        size_bytes: 保存後のファイルサイズ。
        clobbered: no_clobber=False で既存ファイルを上書きしたか。
    """
    client = get_client()
    if which == "output":
        source_path = job_output_path(client.output_dir, job_id)
        default_filename = f"{job_id}.mp4"
    elif which == "joined":
        source_path = job_joined_path(client.output_dir, job_id)
        default_filename = f"{job_id}_joined.mp4"
    else:
        raise ToolError(f"INVALID_WHICH: which は 'output' か 'joined' である必要があります（{which!r}）")

    source_exists = await anyio.to_thread.run_sync(source_path.exists)
    if not source_exists:
        raise ToolError(
            f"VIDEO_NOT_FOUND: {source_path} が見つかりません "
            f"(which={which}、ジョブが未完了か join未実行の可能性があります)"
        )

    dest_dir_path = Path(dest_dir)
    await anyio.to_thread.run_sync(
        functools.partial(dest_dir_path.mkdir, parents=True, exist_ok=True)
    )

    fname = filename or default_filename
    if no_clobber:
        dest_path = await anyio.to_thread.run_sync(unique_dest, dest_dir_path, fname)
        clobbered = False
    else:
        dest_path = dest_dir_path / fname
        clobbered = await anyio.to_thread.run_sync(dest_path.exists)

    await anyio.to_thread.run_sync(shutil.copy2, source_path, dest_path)
    size_bytes = (await anyio.to_thread.run_sync(dest_path.stat)).st_size

    return {
        "saved_path": str(dest_path),
        "source_path": str(source_path),
        "size_bytes": size_bytes,
        "clobbered": clobbered,
    }


def register(mcp: FastMCP) -> None:
    mcp.tool()(get_job_video_path)
    mcp.tool()(get_joined_video_path)
    mcp.tool()(save_job_video)
