# MiniMax H3 の Inpainting のスパイク（エンジン直接の台本）

`h3_inpaint_spike.py` は、MiniMax H3 の Inpainting（マスクで指定した部分だけを描き直す機能。台帳 `Docs/PENDING_TASKS.md` §1-124）が成り立つかを 1 本の GPU 実験で確かめた使い捨ての台本です。経緯・条件・結果・判定は記録簿 `Docs/VERIFICATION_LOG.md` §207 にあります。

この台本はサーバーも MCP（AI エージェント向けの道具の口）も通さず、エンジンの仮想環境で `engine_h3.pipeline_h3.H3Pipeline` を直接呼びます（写し元は `Docs/Outputs-archive/h3-outpaint-2026-11-10/h3_outpaint_gate.py`）。製品コードは変えません。この保管庫には台本・この説明書・結果の JSON・画像 2 枚だけを置き、動画・素材・ログは置きません。数値はここに書きません。条件と数値は `result.json` と Hugging Face に上げた説明書（下の「遠隔目視」）を参照してください。

## 目的

- Outpainting で作ったトークン（32×32 画素の升目）単位の固定の仕組み（`PENDING_TASKS_CLOSED.md` 1-125）に、利用者のマスク動画から作った値を差し込めば、マスクの中だけがプロンプトに従って描き直され、マスクの外の潜在（VAE〔映像を圧縮する符号化器と復号器〕が作る圧縮表現）が元のまま保たれるかを確かめます。
- 合否はオーナーの目視です。台本は数値（自己検査・トークン数・マスクの内外の平均絶対差・所要秒数・VRAM〔GPU の作業用メモリ〕の山）を取るだけです。

## 前提

- GPU を使う実行は、毎回オーナーの了承を得てから行います。`--dry-run` と `--post-only` は GPU を隠して（`CUDA_VISIBLE_DEVICES=-1`）CPU だけで動きます。
- バックエンド（サーバー）と AviUtl2 は閉じておきます。この台本は自分のプロセスでモデルを読み込みます。
- 使う python はエンジンの仮想環境 `.venv-engine-h3`。リポジトリの根を `PYTHONPATH` に入れ、`PYTHONUTF8=1` を設定します。
- 台本は自分の置き場所から 2 階層上をリポジトリの根とみなし、出力も自分の置き場所に書きます。回すときは `outputs\h3-inpaint-spike-2026-11-11\` に置いた元のものを使います（この保管庫の写しをそのまま動かすと根の位置がずれます）。
- 製品の関数を 2 か所だけ、このプロセスの中で差し替えます（モジュール属性への代入。try で入れ、finally で戻す）。
  - `engine_h3.outpaint_h3.outpaint_fix` を包み、トークンの値を「幾何の値（下の 1 行だけ 1）とマスクの値の大きいほう」にする。
  - `engine_h3.outpaint_geometry.composite` を素通しにする（元動画への合成〔マスクの外へ元の画素を戻す処理〕をしない）。
- 終了コード: 0＝完走、1＝失敗（`result.json` に記録）、2＝引数・素材の誤り。

## コマンド

Git Bash から（リポジトリの根で）:

```
export PYTHONPATH="$PWD" PYTHONUTF8=1
./.venv-engine-h3/Scripts/python.exe outputs/h3-inpaint-spike-2026-11-11/h3_inpaint_spike.py --dry-run
./.venv-engine-h3/Scripts/python.exe outputs/h3-inpaint-spike-2026-11-11/h3_inpaint_spike.py
./.venv-engine-h3/Scripts/python.exe outputs/h3-inpaint-spike-2026-11-11/h3_inpaint_spike.py --post-only
```

- `--dry-run`: モデルを読まずに、素材の切り出し・マスクのトークン化・幾何・差し替えの到達・`video_mask` の形・固定領域の自己検査の関数・`validate_request` を確かめ、`dry_run.json` に書きます。
- 引数なし（本番）: 読み込み → 生成 → 後処理まで 1 回で行い、`result.json` に書きます。
- `--post-only`: 既にある `output.mp4` から後処理（上 768 行の切り出し・平均絶対差・一覧の画像・Hugging Face に上げるフォルダ）だけをやり直します。
- `--dry-run` と `--post-only` は同時に指定できません。`--prompt` で文を差し替えられます（既定は `result.json` の `conditions.prompt`）。

## 素材

- 元の動画: `uploads\h3-inpaint-development\inpaint-base.mp4`（オーナーが用意したテスト素材。git の追跡外）。台本が先頭 73 枚を切り出して `_material\src_1280x768_73f.mp4` を作ります（ffmpeg の命令は `result.json` の `material.ffmpeg_cmd`）。
- マスク動画: `uploads\h3-inpaint-development\mask-768p.mp4`（同じ素材の LTX 2.5 の Inpainting のジョブで AviUtl2 のプラグインが描いたマスクを、オーナーが 768 版に縮小・位置調整したもの）。先頭 73 枚を読みます。
- 素材とマスクの ffprobe の結果は `result.json` の `material`・`mask` にあります。

## 条件

`result.json` の `conditions`（台本の固定値）と `request`（エンジンへ渡した要求そのまま）が正本です。マスクの規則（赤の 128 以上を白・32×32 のブロックに 1 画素でも白があれば生成・時間は 17 枚の塊ごとの和集合）は `mask.rule` にあります。

## 記録するもの

| ファイル | 内容 |
|---|---|
| `dry_run.json` | `--dry-run` の各検査（`checks`）と全体の合否（`all_passed`）、期待する潜在フレームごとの生成トークン数 |
| `result.json` | 本番の記録。`conditions`・`geometry`・`material`・`mask`（塊ごとのトークン数・白の面積率）・`request`・`load`・`spike_valid`（差し替えが 1 回ずつ効いたか）・`self_check`（固定した潜在が元とビット一致したか）・`tokens_actual`（台本が数えた固定と生成のトークン数）・`engine`（エンジンの戻り値。所要秒数・VRAM の山・段ごとの記録）・`post.metrics`（マスクの内外の平均絶対差） |
| `sheet.png` | フレーム 0・36・72 の 3 行 ×「元」「出力の上 768 行」「出力にマスクを赤で重ねたもの」の 3 列 |
| `token_overlay_f036.png` | フレーム 36 の元の映像に、その塊のマスクのトークン（赤）と 32 画素の格子（白）を重ねたもの |
| `h3_inpaint_spike.py` | 台本の写し |

## 出力の置き場

すべて `outputs\h3-inpaint-spike-2026-11-11\`（git の追跡外）。

| ファイル | 内容 |
|---|---|
| `_material\` | 切り出した元の動画 |
| `output.mp4` | 出力そのもの（キャンバスの寸法） |
| `output_crop768.mp4` | 出力の上 768 行（元の動画と同じ寸法。見比べ用） |
| `token_mask_73.npy` | トークン化したマスク |
| `dry_run.json`・`result.json`・`run.log`・`console.log` | 記録とログ |
| `sheet.png`・`token_overlay_f036.png` | 一覧の画像 |
| `hf-round14\` | Hugging Face に上げたフォルダ（動画 3 本・画像 2 枚・`README.md`） |

監督は実行後に、上の「記録するもの」の表のファイルをこの保管庫へ写します。

## 遠隔目視

非公開の Hugging Face データセット `Rootport/Nz-LTX23-eval-clips` の `round14-h3-inpaint/`（https://huggingface.co/datasets/Rootport/Nz-LTX23-eval-clips/tree/main/round14-h3-inpaint ）。上げる元は `outputs\h3-inpaint-spike-2026-11-11\hf-round14\` です。条件の表と数値の表は、そこに置いた `README.md` にあります。
