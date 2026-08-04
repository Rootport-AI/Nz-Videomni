# 保存領域の方針と実構造（STORAGE_POLICY）

作成: 2026-08-01（オーナーディスカッションによる方針確定）

## 0. 設計原則（最重要・新機能を作る人へ）

> **Outputsは宝物、Uploadsは事実上の一時ファイル置き場。**

- `outputs/` に置くもの: ユーザーが失いたくない**成果物**（完成動画・そのレシピであるmetadata.json）。バックアップの対象はここだけ。
- `uploads/` に置くもの: 生成のために一時的に預かった**入力素材**。いつ消えても成果物の閲覧・再ダウンロードには影響しない、キャッシュとして扱う。
- **新しい機能を追加するとき、「消すことのできないもの」「後から必要になる唯一のコピー」をuploads/へ保存してはならない。** 永続させたいものはoutputs/（ジョブに紐づくならそのジョブのディレクトリ）へ。この原則はSD WebUIの思想（outputsだけが永続で、入力・中間物——例えばControlNetのプリプロセッサ抽出画像——は既定では保存されない）に合わせたもの。

## 1. 実構造（2026-08-01実測）

### uploads/（入力素材・キャッシュ）

| 系統 | レイアウト | ID形式 | 実装 |
|---|---|---|---|
| 画像 | `uploads/{image_id}/input.png`（PNGへ正規化・EXIF回転補正） | UUID | `services/upload_store.py` |
| 動画 | `uploads/videos/{video_id}/input{ext}`（リボン範囲トリム指定時はffmpegで切り出し`input.mp4`へ置換） | UUID | `services/video_upload_store.py` |
| 音声 | `uploads/audios/{audio_id}/input{ext}` | UUID | `services/audio_upload_store.py` |

- TTL・自動削除・一覧/削除APIは**存在しない**（`api/uploads.py`にはPOST3本のみ）。
- 実測（2026-08-01）: 画像243件/約281MB・動画137件/710MB・音声154件/127MB、合計534件・約1.1GB（最古は約1ヶ月前）。

### outputs/（成果物）

- `outputs/{job_id}/` に `output.mp4`・`metadata.json`（設定で有効時）・V2V結合後は`joined.mp4`。job_idはジョブ作成時に発行されるUUID。
- `DELETE /jobs/{job_id}`（終了済みジョブのみ）が`outputs/{job_id}`を削除する。これが唯一のディスク削除API。
- 実測（2026-08-01）: 433件・約1.5GB。

## 2. UploadsとOutputsの紐づけ

- **命名上の一貫性は無い**（両方UUIDだが独立に発行される）。
- **ジョブ→アップロードの追跡は可能**: `outputs/{job_id}/metadata.json`の`request`にリクエスト全体が保存されており、使用した`image_id`（conditioning_images）・`reference_video_id`・`source_video.video_id`・`source_audio.audio_id`がそのまま読める。チェーンジョブは`v2v.source_video_id`/`a2v.source_audio_id`の明示ブロックも持つ。
- **アップロード→ジョブの逆引きは不可能**（参照カウント・使用履歴の類は無い）。突き合わせるには全metadata.jsonのID全文検索が必要。

## 3. ユーザー向けのディスク整理ルール

- **バックアップすべきはoutputs/だけ**。整理は`DELETE /jobs/{id}`またはジョブディレクトリ削除で1ジョブ単位。
- **uploads/はバックエンド停止中なら丸ごと削除して安全**。完了済みジョブの閲覧・再ダウンロード・Joinには影響しない（成果物と結合用の素材末尾はoutputs側に保存済み）。アップロードIDを掴んでいるのは実行中のUIセッションだけなので、次に使うときは自動で再アップロードされる。
- **稼働中のuploads/削除は避ける**: バッチ実行中のキャッシュ（同一画像/参照動画のIDを複数行が使い回す）や、画面のスロット引き継ぎが保持しているIDが404になり、以後のGenerateが失敗する。

## 4. 将来の改善候補（台帳: フロントエンド`Docs/PENDING_TASKS.md` §4-23）

- OutputsとUploadsの一本化案（「同じジョブのものは同じディレクトリへ」）は2026-08-01の議論で**スコープ外**とした。理由: アップロードはジョブ誕生前に起きる／1アップロードを複数ジョブが使い回すためコピーは重複を生む／SD WebUIの思想とも方向が異なる。
- 軽い改善として「**起動時に古いuploadsを自動掃除**（年齢ベースGC。例: 7日超を削除）」が有力。metadata.jsonの紐づけを使えば「どのジョブからも参照されない孤児のみ削除」も実装可能だが、キャッシュと割り切るなら年齢ベースで十分。
