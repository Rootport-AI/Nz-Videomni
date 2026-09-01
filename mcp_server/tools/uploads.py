"""アップロード系ツール（画像・参照/継続動画・音声、3本）。

ローカルの絶対/相対パスを受け取り、(1) 存在チェック、(2) ローカル
``config.yaml`` の ``upload:`` セクション由来の拡張子・サイズの事前チェック、
(3) マルチパートPOST、の順で処理する。事前チェックはサーバー側のストア
（``services/{upload,video_upload,audio_upload}_store.py``）と同じ
``config.upload`` を読んでいるだけで、別ルールを新設しているわけではない
（正本はあくまでサーバー -- config.yaml を書き換えてMCPサーバーを再起動し
忘れた場合など、事前チェックを通過してもサーバー側の400で弾かれ得る）。

``Path.is_file`` / ``Path.stat`` / ``Path.read_bytes`` はブロッキングI/Oなので
``anyio.to_thread.run_sync`` 経由で呼ぶ（計画D3）。
"""

from __future__ import annotations

import functools
import mimetypes
from pathlib import Path
from typing import Any

import anyio
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from mcp_server.client import get_client

# 計画の指定タイムアウト（秒）: 画像60 / 動画300 / 音声120。
_IMAGE_TIMEOUT = 60.0
_VIDEO_TIMEOUT = 300.0
_AUDIO_TIMEOUT = 120.0


def _precheck(path: Path, *, allowed_ext: set[str], max_bytes: int, kind: str) -> None:
    """存在・拡張子・サイズをチェックする（ブロッキング、スレッドで呼ぶこと）。"""
    if not path.is_file():
        raise ToolError(f"FILE_NOT_FOUND: {path}")
    ext = path.suffix.lower()
    if ext not in allowed_ext:
        raise ToolError(
            f"UPLOAD_INVALID_TYPE: {kind} の拡張子 '{ext or '(なし)'}' は許可されていません "
            f"(許可: {sorted(allowed_ext)})"
        )
    size = path.stat().st_size
    if size > max_bytes:
        raise ToolError(
            f"UPLOAD_TOO_LARGE: {kind} のサイズが上限を超えています "
            f"({size / 1024 / 1024:.1f}MB > {max_bytes / 1024 / 1024:.0f}MB)"
        )


async def _upload(
    file_path: str,
    *,
    allowed_ext: set[str],
    max_bytes: int,
    kind: str,
    endpoint: str,
    timeout: float,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path = Path(file_path)
    await anyio.to_thread.run_sync(
        functools.partial(_precheck, path, allowed_ext=allowed_ext, max_bytes=max_bytes, kind=kind)
    )
    data = await anyio.to_thread.run_sync(path.read_bytes)
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"

    client = get_client()
    return await client.post_file(
        endpoint,
        files={"file": (path.name, data, content_type)},
        params=params,
        timeout=timeout,
    )


async def upload_image(file_path: str) -> dict[str, Any]:
    """ローカルの画像ファイルをアップロードします（POST /upload/image）。

    ``submit_generate`` の ``conditioning_images`` に渡す ``image_id`` を得る
    ための前段ツールです（最小I2V/キーフレーム条件付け用）。許可される拡張子・
    最大サイズはバックエンドの設定（``config.yaml`` の ``upload:``）に従います
    （``get_config`` の ``upload`` セクションでも確認できます）。

    Args:
        file_path: MCPサーバーを動かしているマシン上のローカルファイルパス。

    Returns:
        image_id / original_filename / stored_path / width / height / content_type。
    """
    client = get_client()
    upload_cfg = client.upload
    return await _upload(
        file_path,
        allowed_ext={e.lower() for e in upload_cfg.allowed_image_extensions},
        max_bytes=upload_cfg.max_image_size_mb * 1024 * 1024,
        kind="画像",
        endpoint="/upload/image",
        timeout=_IMAGE_TIMEOUT,
    )


async def upload_video(file_path: str, max_frames: int | None = None) -> dict[str, Any]:
    """ローカルの動画ファイルをアップロードします（POST /upload/video）。

    4通りの用途で使う参照/継続動画のアップロードです:
    (1) ``submit_generate`` / ``submit_chain`` の ``reference_video_id``
        （IC-LoRA制御アダプタの参照動画。画角拡張〔outpaint〕の元動画も
        この口です）、
    (2) ``submit_chain`` の ``source_video``（V2V継続の元動画）、
    (3) ``submit_chain`` の ``end_source_video_id``（素材（末尾））、
    (4) ``submit_chain`` の ``retake_video_id``（撮り直しの元動画）。
    許可される拡張子・最大サイズはバックエンドの設定に従います。

    **``max_frames`` を渡すと、尺（``frame_count``）と ``fps`` が実測されて
    返ります。撮り直し（``submit_chain`` の ``retake_video_id``）の
    ``retake_window_start_sec`` を決めるための下調べにはこれを使ってください。**
    渡さない場合、``frame_count`` と ``fps`` は両方 ``null`` で返ります
    （通常のアップロードに余計な ffprobe を払わせないための仕様であり、
    エラーではありません）。実測に失敗した場合も ``null`` になります。
    **``max_frames`` は「先頭 N フレームだけ残して切り詰める」引数でもあります**
    ——尺を測るためだけに渡すときは、元の尺より確実に大きい値
    （例: 100000）を渡してください。小さい値を渡すと**動画そのものが切り
    詰められて保存されます**。
    なお解像度（width / height）はこの応答からは分かりません。画角拡張の
    キャンバス（128の倍数）を逆算するには、アップロード前にローカルで
    ffprobe 等により解像度を確認してください。

    Args:
        file_path: MCPサーバーを動かしているマシン上のローカルファイルパス。
        max_frames: 先頭から残すフレーム数の上限（省略時は切り詰めなし）。
            指定すると保存後のファイルの ``frame_count`` / ``fps`` が実測され
            て返ります。

    Returns:
        video_id / original_filename / stored_path / content_type / size_bytes /
        trimmed / frame_count / fps（``frame_count`` と ``fps`` は
        ``max_frames`` を渡したときだけ実測値、それ以外は ``null``）。
    """
    client = get_client()
    upload_cfg = client.upload
    return await _upload(
        file_path,
        allowed_ext={e.lower() for e in upload_cfg.allowed_video_extensions},
        max_bytes=upload_cfg.max_video_size_mb * 1024 * 1024,
        kind="動画",
        endpoint="/upload/video",
        timeout=_VIDEO_TIMEOUT,
        # クエリ引数（api/uploads.py:57 の ``max_frames: int | None = Query(None)``）。
        # 未指定のときはキーごと送らない -- 従来のアップロードのリクエストを
        # 1バイトも変えないため（「Noneまたは空は送らない」のペイロード契約）。
        params=None if max_frames is None else {"max_frames": max_frames},
    )


async def upload_audio(file_path: str) -> dict[str, Any]:
    """ローカルの音声ファイルをアップロードします（POST /upload/audio）。

    ``submit_chain`` の ``source_audio``（A2V: アップロードした音声に映像を
    同期させて生成する機能、W4で公開）に渡す ``audio_id`` を得るための前段
    ツールです。許可される拡張子・最大サイズはバックエンドの設定に従います。
    コーデックの妥当性チェックはアップロード時ではなく生成時に行われます。

    Args:
        file_path: MCPサーバーを動かしているマシン上のローカルファイルパス。

    Returns:
        audio_id / original_filename / stored_path / content_type / size_bytes。
    """
    client = get_client()
    upload_cfg = client.upload
    return await _upload(
        file_path,
        allowed_ext={e.lower() for e in upload_cfg.allowed_audio_extensions},
        max_bytes=upload_cfg.max_audio_size_mb * 1024 * 1024,
        kind="音声",
        endpoint="/upload/audio",
        timeout=_AUDIO_TIMEOUT,
    )


def register(mcp: FastMCP) -> None:
    mcp.tool()(upload_image)
    mcp.tool()(upload_video)
    mcp.tool()(upload_audio)
