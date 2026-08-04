"""バッチA2V計画ツール（1本、W6）。

``plan_a2v_batch`` はHTTPを一切使わない純ローカル計画ツールです ―― wavフォルダ
を走査して行ごとの提案フレーム数等を返すだけで、アップロードやジョブ投入は
一切行いません（部品＋エージェントループ方式。長時間ブロックする複合ツールは
作らない、計画の設計方針）。実際の生成は ``next_steps`` の案内に従い、行ごとに
``upload_audio`` → ``submit_chain`` → ``wait_for_job`` を直列で繰り返してください。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anyio
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from mcp_server import batch_planning

_NEXT_STEPS = (
    "各行を1件ずつ直列で処理してください（同時に実行できるジョブは1本だけです）: "
    "(1) skip_reason が付いている行はスキップする。 "
    "(2) upload_audio(wav_path) で audio_id を得る。 "
    "(3) submit_chain(prompt=..., clips=[{\"num_frames\": suggested_num_frames}], "
    "source_audio_id=audio_id, chunked_upsample=True)（image_path がある行は "
    "先に upload_image して clips[0].conditioning_images に渡す）でジョブを投げる。 "
    "(4) wait_for_job(job_id) が completed になるまで繰り返す。 "
    "(5) save_job_video(job_id, dest_dir) で保存する。 "
    "409 JOB_BUSY が出た場合は前のジョブの完了を待ってから同じ行を再試行してください。"
)


async def plan_a2v_batch(
    wav_dir: str,
    fps: float = 24.0,
    image_dir: str | None = None,
    max_frames: int = 481,
) -> dict[str, Any]:
    """A2Vバッチの実行計画を立てます（wavフォルダを走査するだけ、HTTP不使用）。

    パネルの「Batch A2V」タブと同じ走査規約です: 音声拡張子（wav/mp3/m4a/aac/
    flac/ogg）を候補にし、マニフェストファイル自身は除外、更新日時（mtime）
    昇順に並べます。ただし **wav以外はフレーム数を測定できません**
    （stdlib の ``wave`` モジュールしか使わないため、mp3/m4a等は長さを読めず、
    ``skip_reason="wav-only-alpha"`` の可視Skip行になります――生成できない
    という意味ではなく、このツールが事前に長さを提案できないだけです。それ
    でも行として一覧には残ります）。1クリップの尺が481フレームを超える長さの
    wavは ``skip_reason="over-481f"`` になります。

    ``image_dir`` を指定すると、各wavと同じstem（拡張子を除くファイル名、
    大小無視）の画像ファイルを1件だけ自動で紐付けます（同stemが複数拡張子で
    あれば先勝ち）。一致しない・``image_dir`` 未指定の行は ``image_path: null``
    です。

    Args:
        wav_dir: 音声ファイルが入ったローカルフォルダの絶対パス。
        fps: フレームレート（フレーム数提案の計算に使う）。
        image_dir: 同stem画像を探すローカルフォルダの絶対パス（省略可）。
        max_frames: 1クリップの最大フレーム数（既定481、サーバーの上限と同じ）。

    Returns:
        fps, wav_dir, rows（index/wav_path/filename/duration_seconds/
        suggested_num_frames/image_path/skip_reason のリスト）, counts
        （total/plannable/skipped）, next_steps（次に行うべき手順の案内文）。
    """
    wav_path = Path(wav_dir)
    if not await anyio.to_thread.run_sync(wav_path.is_dir):
        raise ToolError(f"WAV_DIR_NOT_FOUND: {wav_dir}")

    img_path: Path | None = None
    if image_dir:
        img_path = Path(image_dir)
        if not await anyio.to_thread.run_sync(img_path.is_dir):
            raise ToolError(f"IMAGE_DIR_NOT_FOUND: {image_dir}")

    rows = await anyio.to_thread.run_sync(
        batch_planning.plan_rows, wav_path, fps, img_path, max_frames
    )

    total = len(rows)
    skipped = sum(1 for r in rows if r["skip_reason"])
    plannable = total - skipped

    return {
        "fps": fps,
        "wav_dir": str(wav_path),
        "rows": rows,
        "counts": {"total": total, "plannable": plannable, "skipped": skipped},
        "next_steps": _NEXT_STEPS,
    }


def register(mcp: FastMCP) -> None:
    mcp.tool()(plan_a2v_batch)
