"""出力ファイルの実パス解決（純関数、W5）。

出力実パスの正本は ``config.output_dir/{job_id}/output.mp4``
（``api/jobs.py``）・``joined.mp4``（``services/join_manager.py``）―― どちらも
サーバーの ``GET /jobs/{id}`` レスポンス（``JobResponse`` / ``JobResult``）には
含まれない（``JobResult.output_path`` は相対ハードコードで信用できない、計画の
「重要な事実」節参照）。よってこのモジュールはHTTPレスポンスを一切見ず、
``mcp_server.client.BackendClient.output_dir``（= ``Settings.output_dir`` =
ローカル ``config.load_config().output_dir``）だけを入力に取る。

``job_output_path`` / ``job_joined_path`` は純粋なパス組み立てのみ（I/O無し）。
``unique_dest`` だけがファイルシステムを読む（``Path.exists``）ため、呼び出し側
（``tools/outputs.py``）は必ず ``anyio.to_thread.run_sync`` 経由で呼ぶこと
（計画D3、ブロッキングI/Oはスレッドへ）。
"""

from __future__ import annotations

from pathlib import Path


def job_output_path(output_dir: Path | str, job_id: str) -> Path:
    """1本生成/チェーン共通の出力動画パス（``output_dir/{job_id}/output.mp4``）。"""
    return Path(output_dir) / job_id / "output.mp4"


def job_joined_path(output_dir: Path | str, job_id: str) -> Path:
    """V2V join済み動画パス（``output_dir/{job_id}/joined.mp4``）。"""
    return Path(output_dir) / job_id / "joined.mp4"


def unique_dest(dest_dir: Path | str, filename: str) -> Path:
    """``dest_dir/filename`` が既に存在する場合、衝突しない連番名を返す。

    ``video.mp4`` が存在すれば ``video_2.mp4``、それも存在すれば ``video_3.mp4``
    ……と続ける。``dest_dir`` 自体の作成はここでは行わない（呼び出し側の責務）。
    """
    dest_dir = Path(dest_dir)
    candidate = dest_dir / filename
    if not candidate.exists():
        return candidate

    stem = Path(filename).stem
    suffix = Path(filename).suffix
    n = 2
    while True:
        candidate = dest_dir / f"{stem}_{n}{suffix}"
        if not candidate.exists():
            return candidate
        n += 1
