"""バッチA2V計画のための純ローカルロジック（HTTP不使用、W6）。

このモジュールは ``gradio_ui`` を一切 import しない ―― ``gradio_ui`` パッケージ
は ``__init__.py`` 経由で最終的に ``gradio`` を import してしまう
（重い・stdout汚染リスクがある、計画D2/敵対的レビュー事実5）。代わりに以下の
2ファイルのロジックを1対1で写経する:

* ``gradio_ui/handlers.py`` の ``_wav_duration_seconds`` / ``suggest_frames_for_audio``
  （フレーム数提案の正本 ―― stdlib ``wave`` で長さを取得し、
  ``chain_math.audio_latents_required`` と突き合わせて8刻みで縮める）。
* ``gradio_ui/manifest.py`` の ``scan_wav_folder`` 走査規約（全音声拡張子を
  候補にし、manifest/autosave/*.tmp を除外、mtime昇順、非wav・読めないwavは
  可視Skip行 ``skip_reason="wav-only-alpha"``、実効上限（``min(max_frames,
  481)``）超は ``"over-cap"``）。

写経ロジックの乖離を防ぐため、``tests/test_mcp_batch_planning.py`` が本家
（``gradio_ui.handlers.suggest_frames_for_audio``）との総当たりパリティで
固定している。**ロジックを変更する場合は本家側とこのファイルとパリティ
テストの3点を必ず同時に更新すること。**

wav以外のファイルは stdlib ``wave`` で長さを測定できないため、走査結果では
可視のSkip行（``skip_reason="wav-only-alpha"``）になります（v1の既知の制限、
パネルの Batch A2V と同じ）。

画像フォルダの同stemマッチング（``plan_rows`` の ``image_dir`` 引数）は
MCP専用の追加規約です ―― パネルの Batch A2V は行ごとに手動でドロップダウンから
画像を選ぶ方式（``gradio_ui/ui.py::batch_image_choices``）で、自動の
同stemマッチングに相当する既存ロジックは存在しません。エージェントが非対話で
計画を立てられるよう、ここで新設しています。
"""

from __future__ import annotations

import math
import wave
from pathlib import Path
from typing import Any

# gradio_ui/manifest.py の同名定数と同一（写経 -- 変更時は両方更新）。
_MANIFEST_NAME = "batch_a2v_manifest.csv"
_AUTOSAVE_NAME = "batch_a2v_manifest.autosave.csv"
_ALLOWED_AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg")

# gradio_ui/ui.py の _BATCH_IMAGE_EXTS と同一 -- 画像フォルダの同stemマッチング
# （MCP専用の追加規約、モジュールdocstring参照）が対象とする拡張子集合。
_ALLOWED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")

_DEFAULT_MAX_FRAMES = 481


# --------------------------------------------------------------------------- #
# フレーム数提案（gradio_ui/handlers.py 写経）
# --------------------------------------------------------------------------- #
def _wav_duration_seconds(path: Path) -> float | None:
    """.wav の長さ（秒）を stdlib ``wave`` で取得する。読めない/不正なら None。

    ``gradio_ui/manifest.py::_wav_duration_seconds`` と同一ロジック（呼び出し
    側で拡張子を .wav に絞ってから呼ぶ規約も含めて写経）。
    """
    try:
        with wave.open(str(path), "rb") as w:
            frames = w.getnframes()
            rate = w.getframerate()
    except Exception:
        return None
    if not rate or rate <= 0:
        return None
    return frames / float(rate)


def _resolve_fps(fps: Any) -> float:
    """``gradio_ui/handlers.py::_resolve_fps`` の写経。"""
    try:
        fps_v = float(fps) if fps else 24.0
    except (TypeError, ValueError):
        fps_v = 24.0
    return fps_v or 24.0


def raw_frame_count(dur: float, fps: Any) -> int:
    """``gradio_ui/manifest.py::raw_frame_count`` の写経（未クランプの8n+1）。"""
    return ((math.floor(dur * _resolve_fps(fps)) - 1) // 8) * 8 + 1


def over_frame_limit(dur: float, fps: Any, max_frames: int = _DEFAULT_MAX_FRAMES) -> bool:
    """``gradio_ui/manifest.py::over_frame_limit`` の写経。実効上限は
    ``min(max_frames, 481)``（WebView2フロントエンドの ``Math.min(cap, 481)``
    と同型）。未指定/0の ``max_frames``（空欄相当）はハード上限へフォールバック
    する。"""
    cap = min(int(max_frames or _DEFAULT_MAX_FRAMES), _DEFAULT_MAX_FRAMES)
    return raw_frame_count(dur, fps) > cap


def suggest_frames_for_audio(dur: float, fps: Any) -> int:
    """``gradio_ui/handlers.py::suggest_frames_for_audio`` の写経（**パリティ
    テストで総当たり固定** ―― ロジックを変更する場合は本家・写経・テストの
    3点を同時に更新すること）。

    最大の8n+1フレーム数から出発し、``chain_math.audio_latents_required``
    （``kv=3``、A2Vチェーンの固定 ``overlap_frames`` を模す）が要求する潜在
    フレーム数を、実際にエンコードされる音声潜在フレーム数
    （``round(dur * chain_math.AUDIO_LATENTS_PER_SEC)``）が下回る間、8ずつ
    縮める。最後に ``[9, 481]`` にクランプする。
    """
    import chain_math

    fps_v = _resolve_fps(fps)
    nf = ((math.floor(dur * fps_v) - 1) // 8) * 8 + 1
    nf = max(nf, 9)
    while nf > 9:
        try:
            required = chain_math.audio_latents_required([nf], fps_v, kv=3)
        except ValueError:
            # nf が小さすぎて固定 kv=3 のオーバーラップすら計算できない
            # （サーバー側の同じ算術がぶつかる境界と同一）-- 縮小を止める。
            break
        available = round(dur * chain_math.AUDIO_LATENTS_PER_SEC)
        if available >= required:
            break
        nf -= 8
    return max(9, min(481, nf))


# --------------------------------------------------------------------------- #
# 画像フォルダの同stemマッチング（MCP専用の追加規約 -- モジュールdocstring参照）
# --------------------------------------------------------------------------- #
def _index_images_by_stem(image_dir: Path) -> dict[str, str]:
    """``image_dir`` 内の画像ファイルを stem（拡張子除くファイル名、小文字化）
    をキーに1件だけ索引する。同stemが複数拡張子で存在する場合は
    ``sorted(image_dir.iterdir())`` 順で先勝ち。"""
    index: dict[str, str] = {}
    try:
        entries = sorted(image_dir.iterdir())
    except OSError:
        return index
    for f in entries:
        if not f.is_file():
            continue
        if f.suffix.lower() not in _ALLOWED_IMAGE_EXTENSIONS:
            continue
        key = f.stem.lower()
        if key not in index:
            index[key] = str(f)
    return index


# --------------------------------------------------------------------------- #
# 走査（gradio_ui/manifest.py::scan_wav_folder の規約を1対1で再現）
# --------------------------------------------------------------------------- #
def plan_rows(
    wav_dir: Path | str,
    fps: Any,
    image_dir: Path | str | None = None,
    max_frames: int = _DEFAULT_MAX_FRAMES,
) -> list[dict[str, Any]]:
    """``wav_dir`` を走査し、行の辞書リストを返す（``tools/batch.py`` から
    ``anyio.to_thread.run_sync`` 経由で呼ばれるブロッキング関数）。

    ``gradio_ui/manifest.py::scan_wav_folder`` と同じ規約:
    全音声拡張子（``_ALLOWED_AUDIO_EXTENSIONS``）を候補にし、manifestファイル
    自身とautosave・``*.tmp`` は除外、mtime昇順。非wav・読めないwavは
    ``skip_reason="wav-only-alpha"`` の可視Skip行（除外はしない）。実効上限
    （``min(max_frames, 481)``）超は ``skip_reason="over-cap"``。それ以外は
    ``suggest_frames_for_audio`` でフレーム数を提案する。
    """
    wav_dir = Path(wav_dir)
    reserved = {_MANIFEST_NAME, _AUTOSAVE_NAME}

    candidates: list[tuple[float, Path]] = []
    for p in wav_dir.iterdir():
        if not p.is_file():
            continue
        if p.name in reserved or p.name.endswith(".tmp"):
            continue
        if p.suffix.lower() not in _ALLOWED_AUDIO_EXTENSIONS:
            continue
        try:
            mtime = p.stat().st_mtime
        except OSError:
            continue
        candidates.append((mtime, p))
    candidates.sort(key=lambda t: t[0])

    image_index = _index_images_by_stem(Path(image_dir)) if image_dir else {}

    rows: list[dict[str, Any]] = []
    for i, (_mtime, p) in enumerate(candidates, start=1):
        is_wav = p.suffix.lower() == ".wav"
        dur = _wav_duration_seconds(p) if is_wav else None
        image_path = image_index.get(p.stem.lower())

        if dur is None:
            rows.append({
                "index": i,
                "wav_path": str(p),
                "filename": p.name,
                "duration_seconds": 0.0,
                "suggested_num_frames": None,
                "image_path": image_path,
                "skip_reason": "wav-only-alpha",
            })
            continue

        if over_frame_limit(dur, fps, max_frames):
            rows.append({
                "index": i,
                "wav_path": str(p),
                "filename": p.name,
                "duration_seconds": dur,
                "suggested_num_frames": None,
                "image_path": image_path,
                "skip_reason": "over-cap",
            })
            continue

        rows.append({
            "index": i,
            "wav_path": str(p),
            "filename": p.name,
            "duration_seconds": dur,
            "suggested_num_frames": suggest_frames_for_audio(dur, fps),
            "image_path": image_path,
            "skip_reason": "",
        })

    return rows
