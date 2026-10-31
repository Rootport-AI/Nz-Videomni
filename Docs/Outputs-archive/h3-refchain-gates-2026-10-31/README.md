# §1-109 MiniMax H3 の参照付き Chained の GPU の門 G-1・G-1b・G-2（2026-10-31）

§1-109 の GPU の門の台本です。**2026-10-31 に実行済みで、G-1・G-1b・G-2 に合格しました（下の「結果」）。** 製品のコードは変更しません。製品の入口（HTTP の `POST /generate` と `POST /generate/chain`）を呼び、製品が `metadata.json` に残す記録で合否を決めます。数値は記録簿にだけ書きます（ここには書きません）。

## 中身

| ファイル | 内容 |
|---|---|
| `g_refchain.py` | 門の台本（HTTP クライアント。製品を import しない）。門の名前 `G-1`・`G-1b`・`G-2`、後片付けの `cleanup`、GPU を使わない `--selfcheck` を受ける。前回の `h3-chain-gates-2026-10-29\g_chain.py` の複製拡張 |
| `run_gates.ps1` | 全体の流れ。前処理の確認 → 自己検査 → バックエンドの起動 → 3 つの門 → 後片付け |
| `run_common.ps1` | 起動の共通部。前回の複写（1 条件 = 1 プロセス・時間切れでプロセスの木ごと停止・コンソールを `<名前>.console.log` へ） |
| `README.md` | このファイル |

実行後に増えたもの: `g_selfcheck.*`、`g_1.*`・`g_1b.*`・`g_2.*`（各門の JSON とコンソール）、`g_cleanup.*`、`backend.console.log`、`run_gates.runner.log`。動画と `metadata.json` は `outputs\<ジョブ番号>\`（git 管理外）に残ります。G-2 の動画はオーナーの目視用です。

## 結果（2026-10-31）

- **G-1・G-1b・G-2 のすべてに合格しました。** 終了コードはいずれも 0 で、`g_1.json`・`g_1b.json`・`g_2.json` の `ok` は真、後片付け（`g_cleanup.json`）も成功しました。自己検査（`g_selfcheck.json`）も合格しています。
- 合否の項目ごとの内訳・mp4 と潜在の sha256・`ref_video` の行数・`metadata.references.clips`・所要・VRAM・主記憶・アップロードの保存ファイルの扱いは、**記録簿 [`VERIFICATION_LOG.md`](../../VERIFICATION_LOG.md) §186.6** が正本です（ここには写しません）。
- 1 回目の起動は PowerShell の変数名の衝突で自己検査の段で停止しました（GPU は使っていません）。ループ変数を改名して再起動し、`run_gates.runner.log` は 2 回目のものです。経緯は §186.6。
- G-3（LTX 2.3 の 422）はこの台本に入れず、アプリの単体テストで担保しました（§186.2・§186.7）。
- オーナーの実機確認 G-4 にも合格し、CLOSED 3-264 で閉じました（§186.8）。

## 走らせ方

リポジトリ直下で（GPU を使うので AviUtl2 を閉じ、ほかに GPU を使うプロセスが無い状態で）:

```
powershell -NoProfile -ExecutionPolicy Bypass -File Docs\Outputs-archive\h3-refchain-gates-2026-10-31\run_gates.ps1
```

流れ（`run_gates.ps1` が順に行う）:

1. 前処理の確認: `nvidia-smi` の表示、AviUtl2 の起動検査、重み（FL2VA・Ref2VA の 2 本、テキストエンコーダー、2 つの VAE、tokenizer）、2 つの仮想環境、`config.yaml` の backend が mock でないこと、参照の画像 2 枚と比較対象の記録、GPU の使用量、コミット上限（物理メモリ＋ページファイル）、指定ポートが空いていること。
2. `g_refchain.py --selfcheck`（GPU も HTTP も使わない）。
3. バックエンドを別のポート（既定 18629）で起動（`state.json` は先に控えて、最後に戻す）。
4. G-1 → G-1b → G-2。各門は 1 プロセスで、自分で重みを読み込む（G-1 は FL2VA、G-1b と G-2 は Ref2VA）。0 以外で終わったら止まる（`-ContinueOnMismatch` は終了コード 2「基準の不一致」のときだけ先へ進む。1「失敗」と時間切れでは進まない）。
5. 後片付け（アンロード → バックエンド停止 → `state.json` を戻す）。途中で止まったときもここは必ず通る。

主な引数: `-SkipPreflight`（AviUtl2・GPU 使用量・コミット上限の検査を飛ばす）、`-Port`、`-Gates G-1b,G-2`（門の絞り込み。各門は単独で走る）、`-UseRunningBackend`（起動済みのバックエンドを使う。実物であること。`state.json` には触れない）、`-ContinueOnMismatch`。

自己検査だけ（GPU も HTTP も使わない）:

```
.venv-engine-h3\Scripts\python.exe Docs\Outputs-archive\h3-refchain-gates-2026-10-31\g_refchain.py --selfcheck
```

## 門の表

| 門 | 重み | 画角・注意の計算 | 呼ぶ API | 内容 |
|---|---|---|---|---|
| G-1 回帰 | FL2VA（登録名 `default`） | 640×384・sdpa | `POST /generate`、続けて `POST /generate/chain` | (a) 単発の T2V（前回の指紋と同じ条件・黄色いレインコートの文）の mp4 の sha256 が基準の指紋 `0da6111c…` と一致。(b) 参照なしの連結（G-C0 と同じ条件・同じ文）の mp4 の sha256 が G-C0 の値 `4a98be08…` と一致 |
| G-1b 一致 | Ref2VA（登録名は拡張子を除いたファイル名 `minimax_h3_ref2va_pruned_int8_convrot`） | 640×384・sdpa | `POST /upload/image`、`POST /generate/chain` | 3 クリップ・各クリップに老紳士 1 枚（slot 1）。映像・音声の潜在の sha256 が `h3-refchain-2026-10-31\g_ref2va_gate.json`（計測）の各クリップの値と一致 |
| G-2 完走 | Ref2VA | 1344×768・sage | `POST /upload/image`（2 枚）、`POST /generate/chain` | 3 クリップ。クリップ 0 は老紳士 1 枚、クリップ 1・2 は老紳士（slot 1）＋少女（slot 2）。完走と参照の記録の形（下の基準）。継ぎ目の連続は目視（G-4） |

共通の条件: seed 42・124 フレーム×3 クリップ（重なり 22）・20 ステップ・shift 12／3・VRAM 16GB mode オン。G-1 と G-1b は `embed_mp4_metadata=false`（mp4 の sha256 を比べるため）。G-2 は既定のまま。連結の要求は `clips[i].references = [{kind: "picture", id, slot}]`（計画 §3.1）。`id` は `POST /upload/image` の答えの `image_id`（仕様書 §6 の UploadImageResponse）。

プロンプトは「全クリップの文のうち、空白で終わる最長の共通の先頭部分」を共通文（`prompt`）に、残りを各クリップの `prompt`（追記）にして送ります。製品の合成（`compose_clip_prompt`＝共通文＋空白＋追記）が元の文字列に戻ることは台本が確かめます。G-1・G-1b ではクリップ 0 の文がそのまま共通文になります。G-2 の `prompts_c3.json` は 2 クリップ目以降で文の途中に一文が入るため、共通文は先頭の `<Picture 1> is the elderly gentleman.` だけで、各クリップが残りを追記として持ちます（合成後の文字列は同じ）。

## 合格基準（計画 §7）

全門に共通: ジョブが完了・バックエンドが実物の H3・進捗とクリップ番号が単調・最後の進捗が 1.0・2 本目以降の `head_equals_prev_tail` がすべて真・mp4 が全フレーム再生できてフレーム数が期待どおり・種が 42＋番号・注意の計算・VRAM 16GB mode・ステップ数と shift が条件どおり・合成後のプロンプトがエンジンへそのまま届く・LoRA なし・読み込んだ重みの `transformer_role` が期待どおり（`GET /models`。FL2VA は `fl2va`、Ref2VA は `ref2va`）。

- **G-1**: (a) 単発の mp4 の sha256 が `0da6111c…` と一致、(b) 連結の mp4 の sha256 が G-C0 の `4a98be08…` と一致（参照を足した改修が既存の経路を変えていないこと）。連結に参照の記録（`metadata.references`・クリップごとの `references`・`sequence.ref_video`）が出ないこと。記録のみ: 前回の G-C0 の製品の記録（`g_c0.json`）との潜在の一致。
- **G-1b**: 3 クリップの映像・音声の潜在の sha256（クリップごとの記録の `latents_sha256`）が `g_ref2va_gate.json` と一致（第 1 基準）。種も一致。加えて `metadata.references.clips` の形（各クリップ pictures 1・videos 0・audios 0 と、入力の `kind`・`slot`・`id`）、クリップごとの `references` の件数、`h3.phases.clips[i].denoise.sequence.ref_video`（画像 1 枚の行数）。mp4 の sha256 は記録のみ。
- **G-2**: 完走。`metadata.references.clips`（クリップ 0 は pictures 1、クリップ 1・2 は pictures 2）、クリップごとの `references` の件数、`sequence.ref_video` が各クリップの参照の行数の合計と一致（画像 1 枚の行数は画角で決まる定数。台本の `PICTURE_ROWS`）、`head_equals_prev_tail` が真。VRAM の山・所要・主記憶は記録のみ。

クリップごとの記録の置き場: 製品はこれを `metadata["h3"]["clips"]`（前回の G-C0 と同じ）に書きます。計画の `done.chain.clips` と同じものを指すはずですが、台本は `h3.clips` を先に、無ければ `chain.clips` を探し、どちらにあったかを `clip_record_source` に残します（実行の結果、3 門とも `h3.clips` で見つかりました）。

記録のみの項目: アップロード後に保管された画像（`uploads/<image_id>/input.png`）が元のファイルとバイト単位で同じか。違うと G-1b の潜在が計測と一致しない原因になるので、不一致のときの切り分けの材料です。

終了コード（前回と同じ）: 0＝すべての基準に合格、1＝走行の失敗（例外・ジョブの失敗・時間切れ・モックのバックエンド）、2＝完走したが基準に落ちた。

## G-3（LTX 2.3 で 422）の扱い

この台本には入れていません。LTX 2.3 の連結へ `clips[0].references` を送ると 422 `FEATURE_UNSUPPORTED`（機能名 `references`）になることは、**アプリの単体テスト（計画の M2）で担保**します。理由は、422 を HTTP で確かめるには動かしているバックエンドを LTX 2.3 に切り替える（`POST /pipeline/load` で LTX 2.3 の重みを GPU に載せる）必要があり、H3 の門と同じ台本の中で切り替えと戻しを重ねると手順が複雑になり、短時間で済む保証も無いためです。GPU を使って確かめたい場合は、LTX 2.3 を読み込んだバックエンドへ、実在しない画像の `id` を入れた参照付きの連結の要求を送れば足ります（422 は存在の確認の前に出ます）。

## 注意

- **途中で取り消せません。** 連結のジョブには協調的な停止が無く、終わった時点で `cancelled` になるだけです。始めたら終わるまで待つか、バックエンドごと止めるしかありません。
- G-2 は 1344×768 で参照を持つ 3 クリップのため、3 つの門のうちで一番長く、主記憶（コミット）も大きくなります。ページファイルが小さいと失敗します（前処理の確認でコミット上限が足りなければ止まります）。
- 台本は転送の失敗に強くしてあります（`GET /jobs/{id}` の一時的な失敗は 12 回連続まで許す）。バックエンドが落ちたら門は失敗として終わります。
- `POST /pipeline/load` は時間切れなし（初回の Ref2VA の読み込みは重みの sha256 の計算で数十秒余計にかかり、ルート直下の `weight_identity.json` が増えることがあります。戻しません）。ほかの呼び出しは 30 秒（アップロードは 120 秒）。
- `state.json` は先に控えて最後に戻します（実機の選択が Ref2VA／FL2VA のどちらでも、台本は重みを登録名で明示して読み込みます）。
- `backend.console.log` は大きくなりえます。保管庫に入れる前にサイズを確認してください。
