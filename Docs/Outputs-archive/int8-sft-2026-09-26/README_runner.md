# int8 実機ランナー（`runner.py`）の使い方

§3-168「ComfyUI 標準の int8 safetensors を置くだけで使えるようにする」の実機
検証（計画 `floofy-enchanting-sundae.md` 第 4 節 G3〜G8）を、MCP 経由の客観検査
で行うためのツール。**REST 直叩きではなく MCP ツール呼び出し**を使う点だけが
`stage3\stage3_runner.py`（§3-167 B-1・LTX 2.3）／
`stage3_25\stage3_runner25.py`（§3-167 B-2・LTX 2.5）との違いで、副命令の構成・
結果 JSON の形・VRAM／コミット計測・`psnr-ssim` はほぼそのまま踏襲している。

作業ディレクトリはどこでも良い（`runner.py` はスクリプト自身の場所から
`results/` と `RESULTS.md` を解決する）。

```
cd "C:\Users\PHENOT~1\AppData\Local\Temp\claude\s--OriginalApps-12-Nz-LTX23-AviUtl2\4f4f90b0-d37e-4ed8-b2f7-5ad52bc95cee\scratchpad\int8"
python runner.py <subcommand> ...
```

`python` はリポジトリの `.venv\Scripts\python.exe`（`mcp==1.28.1` 導入済み）を
直接使うこと。このスクリプトが自分でも `PY = ROOT/.venv/Scripts/python.exe` を
`-m mcp_server` の起動に使うので、実行する側の Python は何でもよい（標準ライブ
ラリ＋`mcp` パッケージだけで動く。ランナー自身は `.venv` の python で動かすのが
簡単）。

## 重要: `single` / `iclora` / `chain` はベースモデル／transformer を切り替えない

**`single`・`iclora`・`chain` は現在ロード済みのパイプラインへそのまま投げる
だけで、`--base-model`/`--transformer` 引数は記録用のラベルに過ぎない。**
実際にどのベースモデル・どの transformer で生成されるかは、直前に
`load --base-model X --transformer N` で何をロードしたかだけで決まる。

**実例（2026-09-26）**: `load --base-model LTX25 ...` を挟まずに
`single --base-model LTX25 --tag c0_25_fp8 ...` を呼んだところ、実際には
直前にロードされていた LTX23 の fp8（`sulphur_distil_fp8mixed.safetensors`）
で生成された（`single` の引数 `--base-model LTX25` は記録に残るだけで、
実行には一切効かない）。この無効な記録は `results/invalid_c0_25_fp8_ran_on_
ltx23.json` に改名し、`report` の集計から除外している（`report` は
`invalid_` 接頭辞のファイルを無視する）。

**必ず `single`/`iclora`/`chain` の前に `load --base-model X --transformer N`
を呼び、`list_models`（または直前の `load` の応答）で選択が意図どおりである
ことを確認してから生成に進むこと。**

## MCP の呼び方（確立済みの経路）

`Docs/VERIFICATION_LOG.md` §39.5・§39.6 と同じ経路。`mcp_server.py` を
`.venv\Scripts\python.exe -m mcp_server` として stdio トランスポートの子プロ
セスで起動し、`mcp` パッケージのクライアントで
`initialize → tools/list → tools/call` する。**1 コマンドごとにセッションを
張って閉じる**（`Docs/Outputs-archive/start-end-bridge-2026-09-07/mcp_call.py`
と同じパターンを流用）。`PYTHONUTF8=1` を子プロセスの環境変数に必ず付ける
（`mcp_server/__main__.py` の docstring が警告している通り、stdio は
JSON-RPC 専用なので迷子の1バイトも許されない）。

エラーは全て `mcp_server/client.py` の `_raise_for_error` が
`ToolError("CODE: message — detail")` にする（`tests/test_mcp_errors.py`）。
MCP プロトコル上は例外ではなく `isError:true` の通常応答として返ってくるので、
`Mcp.call()` がそれを検出して Python 例外 `RunnerToolError` に変換している。

**注意（anyio の ExceptionGroup）**: `stdio_client`/`ClientSession` は anyio の
TaskGroup 上に乗っているため、セッション内で `RunnerToolError` を送出すると
素のままでは `BaseExceptionGroup`（ネストしうる）に包まれて出てくる。
`run_async()` が `except* RunnerToolError` で受けて再帰的に剥がしてから
再送出しているので、呼び出し側は普通に `except RunnerToolError` で書ける。

## 副命令

| 副命令 | 内容 |
|---|---|
| `status` | `backend_status`・`list_models`・`list_jobs` を表示。バックエンド未稼働でも落ちない（`list_models`/`list_jobs` は疎通時のみ呼ぶ） |
| `models --base-model LTX23\|LTX25 [--transformer NAME ...]` | 指定ベースモデルの transformer 一覧を表示し、名前が存在するか確認 |
| `load --base-model X --transformer NAME` | `load_pipeline` → 応答（`finished`）と `list_models` の両方で選択が一致することを確認。`wait_sec` の上限（45秒）を超える場合は `backend_status` をポーリングして完了を待つ |
| `restore --base-model X --transformer NAME` | `load` と同じ実装（名前だけ変えたエイリアス）。使い終わったら元の選択（LTX23=fp8・LTX25=default 等、状況に応じて呼び出し側が指定）へ戻すのに使う |
| `single --base-model X --transformer NAME --tag T [--width --height --frames --seed --prompt --lora NAME:STRENGTH ... --reference-video PATH --keep-resident]` | `submit_generate` → `wait_for_job` ループ → `get_job_video_path` → `get_mp4_info`（完了時のみ）。VRAM／コミットを1秒間隔でサンプリング |
| `iclora --base-model X --transformer NAME --tag T --reference PATH --ic-lora NAME [--strength --width --height --frames --seed --prompt]` | `upload_video` → `submit_generate`（`reference_video_id` + `loras=[{name: ic-lora}]`）。幅・高さは128の倍数が必須（既定512×384） |
| `chain --base-model X --transformer NAME --tag T [--width --height --clip-frames N N --seed --prompt --lora ...]` | `submit_chain` で2クリップ（既定512×320・[49,49]・24fps・seed12345＝§116.9と同条件） |
| `keep2 --base-model X --transformer NAME --tag T [...]` | 同条件で `keep_resident=True` の single を2本連続実行し、mp4のSHA-256一致と `psnr-ssim` を記録 |
| `bad-header --base-model X --weights-dir PATH [--case NAME] [--cleanup]` | 偽 safetensors ヘッダを `weights-dir` に作り `load_pipeline` が拒否される（`ToolError`）ことと、選択が不変であることを確認。ケースは `BAD_HEADER_CASES` 辞書1箇所にまとまっており追加しやすい（初期ケース `fp8_per_row_scale`＝前回ランナーと同一バイト列の per-row 倍率 fp8） |
| `psnr-ssim --a X --b Y [--tag T]` | 同梱 `tools/ffmpeg` の `psnr`/`ssim` フィルタで2本のmp4を比較。`--a`/`--b` は絶対/ROOT相対パス、または `results/<tag>.json` の `<tag>` 名 |
| `save --job-id ID --dest PATH [--filename --which output\|joined --overwrite]` | `save_job_video` |
| `record --job-id ID --tag T [--kind single\|chain\|iclora] [--from-outputs]` | 既存ジョブから `single` と同じ形の記録を事後復元する（GPU不使用）。既定は MCP 経由（`job_status`・`get_job_video_path`・`get_mp4_info`）。`--from-outputs` はバックエンドすら使わず、ローカルの `outputs/<job-id>/{metadata.json,output.mp4}` を直接読む（バックエンド停止中でも使える）。`measure`/`samples` は null、`recovered` に `true`（MCP経由）または `"from_outputs"`（直読み）を付ける |
| `report` | `results/*.json`（`_`／`invalid_` 接頭辞を除く）を集計して `RESULTS.md` を書く |

## 結果ファイル

`results/<tag>.json`（＋ 生成系コマンドは `results/<tag>.samples.jsonl` に
VRAM/コミットの生サンプルも保存）。`report` がこれらを集計して同じ場所の
`RESULTS.md` を書く。**`invalid_` 接頭辞のファイルは `report` が無視する**
（無効と判定された記録を残しつつ集計から外すための運用。上の「重要」節参照）。

## 2026-09-27 の改修2点（実機投入中の別チェーンからの依頼）

このランナーは実機投入中の別チェーン（バックグラウンド）から並行して呼ばれて
いたため、**編集は `runner_new.py` に全文書いてから `os.replace()` で
`runner.py` へ原子的に置き換える**手順で行った（`python -c "import os;
os.replace('runner_new.py','runner.py')"`。書き込み途中の状態を実行中の
プロセスに読ませないため）。

1. **`_wait_job` の待ち受け耐性**: `wait_for_job` の呼び出し自体が
   `RunnerToolError`（`BACKEND_UNREACHABLE` やタイムアウト系の文言）になっても
   即座に失敗にせず、5秒待って再試行する（連続6回＝約30秒まで。それでも
   駄目なら諦めて例外を再送出）。**実例**: `single --tag c0_23_fp8_pixar`
   （2.3 fp8＋Pixar_Toon 0.8）で、投入直後の `wait_for_job` が
   `BACKEND_UNREACHABLE: バックエンドに接続できません…` になりランナーが
   クラッシュしたが、サーバー側ではジョブ `4f04197d-e67e-40a6-a4af-
   16565f89db93` が82秒で `completed` していた（LoRA読み込み中にサーバーが
   一時的に応答しなかったと推測）。**このケースは1回でも成功すれば連続失敗
   カウントをリセットする**ので、断続的な応答不良にも耐える。
2. **`record` 副命令の新設**（上の副命令表参照）。バックエンド停止中でも使える
   `--from-outputs` を含む。上記のクラッシュで記録できなかった
   `c0_23_fp8_pixar` は、この副命令で事後復元した（下記「実施結果」）。

### 実施結果: `c0_23_fp8_pixar` と `c0_25_fp8` の事後復元（`--from-outputs`）

いずれもバックエンド停止中に `record --from-outputs` で復元。SHA-256 の先頭
（オーナー参考値）と実ファイルが一致することを確認済み。

| tag | job_id | status | transformer.file | LoRA | 生成秒 | mp4 SHA-256(先頭) |
|---|---|---|---|---|---|---|
| `c0_23_fp8_pixar` | `4f04197d-e67e-40a6-a4af-16565f89db93` | completed | `sulphur_distil_fp8mixed.safetensors`（LTX23 fp8） | `Pixar_Toon:0.8` | 82.63 | `9b385f9f…`（一致） |
| `c0_25_fp8` | `7d09d4de-ecf6-4632-9134-29f3fa32f048` | completed | `ltx25_uncensored_v1.1-fp8_scaled.safetensors`（**LTX25** fp8） | なし | 52.06 | `e862e95d…`（一致） |

`c0_25_fp8` は `models.base_model` が `"LTX25"` であることも `metadata.json`
で確認済み（旧 `results/c0_25_fp8.json` は LTX23 で走った無効な記録だった。
`results/invalid_c0_25_fp8_ran_on_ltx23.json` に改名して集計から除外し、
正しい LTX25 の記録を同じ `c0_25_fp8` タグで新規保存した）。

## GPU を使う副命令の実行状況

タスクの指示により、GPU を使う操作（ジョブ投入・`load_pipeline`・
`unload_pipeline`）はこの作成セッションから一度も実行していない。
**実行して現物確認したのは `status`・`models`・`report`・`record`（GPU不使用、
上記参照）の4つ**（下記「動作確認の結果」）。`load` / `single` / `iclora` /
`chain` / `keep2` / `bad-header` の load 部分 / `restore` は、コードレビュー
（構文チェック・`--help` の表示確認・`py_compile`）のみで、実機での動作は
この作成セッションからは未検証（別チェーンが実機投入で使用中）。

## 動作確認の結果（2026-09-27 実施）

- `status`: MCP の stdio 往復（`initialize → tools/list → tools/call`）は
  正常に動作した。**この時点でバックエンドは稼働していなかった**
  （`backend_status` が `reachable: false`、`error` は
  `BACKEND_UNREACHABLE: バックエンドに接続できません。run.bat でバックエンド
  を起動してください (base_url=http://127.0.0.1:18620) — All connection
  attempts failed`）。この場合 `list_models`/`list_jobs` は呼ばずスキップし、
  exit code 1 を返すことを確認済み（クラッシュしない）。
- `models --base-model LTX23`: バックエンド未稼働のため `list_models` が
  `BACKEND_UNREACHABLE` の `ToolError` になり、`ToolError: ...` の1行を出して
  exit code 1 で正常終了することを確認済み（`RunnerToolError` への変換と
  `except* ... raise` によるアンラップが機能している）。
- `report`: `results/status.json`（上記 `status` 実行の記録）だけの状態で
  実行し、`RESULTS.md` が例外なく書けることを確認済み（該当する表が無いだけ
  の空レポート）。
- それ以外の副命令は `--help` の表示と `py_compile` のみ確認済み（実行はして
  いない）。

---

# G3〜G8 連鎖台本（`gates.py`）

計画 `floofy-enchanting-sundae.md` 第4節（G3〜G8）を、サーバー起動後に監督が
一気に回せるようにした薄いオーケストレーター。`runner.py` の副命令を
サブプロセスとして正しい順序（`load`→…→`restore`）で呼ぶだけで、MCP には
直接触れない。

```
python gates.py --dry-run              # 全コマンドを表示するだけ（実行しない）
python gates.py                        # G3〜G8を通しで実行
python gates.py --only a,b             # 変換器(a)(b)だけ
python gates.py --stage g6             # G6だけ実行
python gates.py --skip g7              # G7を飛ばす
```

## 対象の変換器5本と基準

| キー | base | 候補（登録名＝拡張子を除いた語） | 配布元 | 基準GGUF | restoreの戻し先 |
|---|---|---|---|---|---|
| (a) | LTX23 | `ltx-2.3-22b-distilled-1.1_transformer_only_int8_convrot` | Kijai | `LTX-2.3-22B-distilled-1.1-Q4_K_M` | `sulphur_distil_fp8mixed` |
| (b) | LTX23 | `ltx-2.3-22b-distilled-1.1_int8mixedtensorwise` | silveroxides | 同上 | 同上 |
| (c) | LTX23 | `ltx-2.3-22b-distilled-1.1_w4a8` | JoaoZaokk | 同上 | 同上 |
| (d) | LTX25 | `ltx-2.5-22b-distilled-transformer-comfy-int8-convrot` | — | `default`（2.5公式GGUF） | `default` |
| (e) | LTX25 | `redgraftLTX25Fast2K_ltx25RedgraftNSFW` | CivitAI 3250230 | `redgraftLTX25Fast2K_ltx25RedgraftNSFW-Q6_K` | `default` |

各本とも `models` 副命令で候補・基準GGUFの両方の存在を実行時に確認してから
進む（`runner.py` の `models` は名前が見つからなければ exit 1 を返すので、
未配置なら自動的にそこで止まる。下記「現状の配置状況」参照）。「元」は
計画の閉じ方の慣習どおり LTX23=fp8（`sulphur_distil_fp8mixed`）・
LTX25=`default`。

### 現状の配置状況（2026-09-27 時点、`ls` で確認・ダウンロードは別チェーンが進行中）

(a)(b)(c) は `models/LTX23/Weights/` に完全配置済み。(d)
`ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors.part`・(e)
`redgraftLTX25Fast2K_ltx25RedgraftNSFW.safetensors.part` はまだ `.part`
（ダウンロード中）。**(d)(e) は完了するまで `models` 確認の時点で止まる**
（これは想定どおりの安全な挙動）。

## 各本の手順（G3+G4+G5を1本ぶんの連続実行としてまとめた）

```
models確認(候補) → load(候補) → single(512×320×49f seed12345, tag g3_<key>_single)
→ single --lora Pixar_Toon:0.8 (g3_<key>_pixar)
→ [(a)のみ] single --prefetch off (g3_<key>_single_prefetch_off)
→ iclora deblur (g3_<key>_iclora)
→ chain 2クリップ (g3_<key>_chain)
→ keep2 (g4_<key>)
→ models確認(基準GGUF) → load(基準GGUF, g5_<key>_load_ref)
→ single(基準GGUF, g5_<key>_ref)
→ psnr-ssim: g3_<key>_single 対 g5_<key>_ref (g5_<key>_vs_ref)
→ psnr-ssim: g3_<key>_pixar 対 g3_<key>_single (g3_<key>_pixar_vs_single)
→ restore(元へ)                      ※ try/finally: 途中で失敗しても実行を試みる
```

**途中で失敗したら（exit code != 0）その場で `GateFailure` を送出し、台本
全体を打ち切る**（次の変換器へは進まない）。ただし `restore` だけは
`try/finally` で失敗時にも実行を試みる（選択を汚したまま止まらないための
安全策。実機（backend停止中）で `models確認` をわざと失敗させ、`restore` が
それでも呼ばれることを確認済み。下記「動作確認」参照）。

## IC-LoRA deblur の参照動画

前回の実機確認（`Docs/VERIFICATION_LOG.md` §117.8・
`stage3/results/fp8_iclora.json`）と同じ動画を使い回している:
`outputs/99951e90-6b90-4dc7-af40-a78f9d536177/output.mp4`（ROOT相対、
2026-09-27時点で現存確認済み）。IC-LoRA名 `deblur`・強さ1.0・512×384
（128の倍数が必須のため。§117.8と同条件）。**(d)(e)＝LTX25でも同じ
参照動画/IC-LoRA名を使う設計にしているが、LTX25側に `deblur` が実際に
登録されているかは未確認**（前回のstage3_25調査時点＝2026-09-25では
LTX25用のIC-LoRAは0件だった。README_runner25.mdの`fp8_iclora`参照。
バックエンド起動後、`list_loras` で確認してから(d)(e)の`iclora`ステップを
走らせること。登録が無ければ422で失敗し、台本はそこで正しく止まる）。

## G3「先読みオフ（同期経路）」

`mcp_server/tools/generate.py` の `submit_generate` を確認したところ、
**加速設定の引数は実在する**: `block_swap_prefetch: bool`（既定True＝先読み。
Falseで同期スワップ経路）と `attention_backend: str`（既定"sdpa"、"sage"も
選べる）。今回必要なのは前者なので、`runner.py` の `single` に
`--prefetch {on,off}` を追加した（`--prefetch off` で
`block_swap_prefetch: false` をペイロードに乗せる）。(a) だけ
`g3_a_single_prefetch_off` として1本追加している。

## G6（fp8回帰）

2.3fp8（`sulphur_distil_fp8mixed`、LoRAなし/あり）・2.5fp8
（`ltx25_uncensored_v1.1-fp8_scaled`）を C-0 の基準と完全に同じ設定
（512×320×49f seed12345、LoRAは`Pixar_Toon:0.8`のみ）で再生成し、
**C-0 の基準mp4そのもの（`scratchpad\int8\baseline\*.mp4`）と映像/音声
ストリームMD5を比較する**（`results/g6_compare_<tag>.json` に記録。判定
基準の変更は下の「2026-09-27の修正3点」参照）。3本の対応: `g6_23_fp8`↔
`baseline\c0_23_fp8.mp4`・`g6_23_fp8_pixar`↔`baseline\c0_23_fp8_pixar.mp4`・
`g6_25_fp8`↔`baseline\c0_25_fp8.mp4`。映像/音声ストリームMD5一致が期待値。

## G7（偽ヘッダ422）

`runner.py` の `BAD_HEADER_CASES` に、既存の `fp8_per_row_scale` に加えて
5ケースを追加した（計画§2.1の拒否条件表に対応。すべて LTX23 の Weights に
置く。ヘッダ構造は `g1_census.py` で読めることを事前に確認済み＝下記
「動作確認」）:

| ケース名 | 想定する拒否理由（計画§2.1） |
|---|---|
| `fp8_per_row_scale` | （既存）per-row 倍率の fp8 |
| `nvfp4_format` | format が受理集合の外（nvfp4） |
| `i8_no_marker` | I8 の重みに印が無い |
| `w4a8_no_codebook` | asym_w4a8_int8 の印はあるが `weight_codebook` が無い（補助の不足） |
| `quanto_data_placement` | I8・U8 の置き場所違反（optimum-quanto の `_data` 風） |
| `weight_scale_wrong_shape` | 倍率の形違い（`[o,2]`。受理は `()`/`(1,)`/`(o,1)` のみ） |

**含めていないケース＝計画§2.1「paramsネストのconvrot（受理側）」。** これは
拒否ではなく**受理**を確かめるケースなので、`bad-header`→`load_pipeline`→
422 の枠組みに乗らない（`bad-header` は偽の重みデータしか持たないファイルを
本当にロードさせようとするため、たとえヘッダのマーカー解決が正しく
「受理」されても、後続の実際の重み読み込みで別の理由により失敗しうる）。
**代替の確認手順**: 実装後の worktree（`_wt_3168` 等）で
`sft_quant_format.layer_schemes(path, header, prefix)` を直接呼び、
`{"format": "int8_tensorwise", "params": {"convrot": true}}` のような
params入れ子マーカーを持つ層が `convrot=True` として正しく解決される
（＝flattenしてから読む§2.1の規則どおり）ことを単体で確認する。これは
`g1_census.py`（変換ツールの `parse_quant_marker` を使う）と同じ発想だが、
**製品側の `sft_quant_format.py` を対象にする点が異なる**（変換ツールと
製品側は別実装なので、製品側のparams平坦化を確認するには製品側を直接
呼ぶ必要がある）。

## G8（資源）

`report`（既存）が生成秒・peak_vram_mb・コミット最大を全 single/iclora/
chain/keep2 記録から集計する。追加で `gates.py` が (a)(d) の
`g3_<key>_single`（ConvRot）と対応する `g6_*` fp8 同条件の
`generation_time_seconds` の比を計算し、`results/g8_convrot_cost_ratio.json`
に保存する。

**計画の元のG8にある「connectorの復元時間とジョブごとの再発の有無（2.5は
EP常駐オン・オフ）」はこの台本では自動化していない**——現状の
`metadata.json` にconnector復元時間の専用フィールドが無く、EP常駐
オン/オフの比較には `keep_resident_embeddings` を切り替えた複数本の
追加実行と、そのための計測項目の拡張が要る（下記「未解決点」）。

## 所要時間の見込み

- ベースモデル切り替え（`load`。特に候補↔基準GGUFの往復）: 約6〜8分
  （重み数十GB級の読み込み。初回ロード後の1本目は約2倍かかる旨が
  `load_pipeline` のdocstringに明記されている）。
- `single` 1本: 約1〜1.5分（512×320×49f。§117.8実測の57〜70秒台と同程度）。
- 変換器1本ぶん（G3+G4+G5）: load×2（候補・基準）＋single系5本＋chain1本＋
  keep2(2本)＋psnr-ssim2回＋restore ≈ 12〜18分。**5本合計で概算1〜1.5時間**
  （(a)は先読みオフ分+1〜1.5分）。
- G6: load×3＋single×3 ≈ 10〜15分。G7: 6ケース×（数秒〜1分程度、ロードは
  422で即終わる想定）。G8: report は数秒。

## 2026-09-27の修正3点（実機投入で発覚した合否基準の誤り＋resume）

**発端**: 実機投入で (a) の `keep2` が MISMATCH で止まった。r1（job
`0c153e4b…`）と r2（job `f6e908c6…`）は PSNR∞・SSIM 1.0 で、同梱ffmpegの
`-map 0:v:0 -c copy -f md5 -`（映像ストリームMD5、両方 `a109d2d8…`）も
`-map 0:a:0 …`（音声ストリームMD5、両方 `cfc3cd3b…`）も完全一致していたが、
ファイル全体のSHA-256だけが不一致だった。**原因は mp4 コンテナのメタデータ**
（§3-164 で埋め込む job_id 等が2本で異なる）——`keep_resident` の同一性判定に
ファイル全体SHA-256を使っていたのが誤りだった。

### (1) `keep2` の合否基準を変更（`runner.py`）

`_stream_md5(path, map_selector)` を新設（`-map <selector> -c copy -f md5 -`
でコンテナメタデータの影響を受けない、対象ストリームだけのMD5を取る。
ストリームが無ければ `None`）。`cmd_keep2` の合否を
**「映像ストリームMD5一致 かつ 音声ストリームMD5一致 かつ PSNR∞」**に変更
（ファイル全体SHA-256は`sha256_r1`/`sha256_r2`/`sha256_match`として記録する
だけで合否には使わない）。実際に (a) の既存記録（`results/g4_a_r1.json`/
`_r2.json`、job `0c153e4b…`/`f6e908c6…`）に対して新基準を適用したところ
**映像MD5=`a109d2d8…`・音声MD5=`cfc3cd3b…`（ともに一致）・PSNR y=inf・
判定=OK** となることを確認済み（コーディネーター報告のハッシュ先頭と完全
一致）。

### (2) G6の比較もストリームMD5に変更（`gates.py`）

`_compare_g6` を、`results/c0_*.json` 経由のファイル全体SHA-256比較から、
**新しい生成 対 `baseline\*.mp4` のストリームMD5比較**に変更した
（`G6_BASELINE_MP4` で `g6_23_fp8`→`c0_23_fp8.mp4` 等に対応付け）。基準側の
ストリームMD5は `python gates.py --baseline-md5`（GPU不使用・ローカルの
`baseline\*.mp4` だけを読む）で事前計算し、`baseline\stream_md5.json` に
保存する設計にした。**この節の作業で実際に実行済み**（下記の値）:

```json
{
  "g6_23_fp8": {"file": "c0_23_fp8.mp4",
    "video_md5": "cef1129a83e76df7144f26c3aa832070",
    "audio_md5": "607c55471f289bd42952eec0e91ba591"},
  "g6_23_fp8_pixar": {"file": "c0_23_fp8_pixar.mp4",
    "video_md5": "8d9a2c25c6a38b977f42d8f27e8dfb76",
    "audio_md5": "88af419e95cd0283790ecde9bca14d91"},
  "g6_25_fp8": {"file": "c0_25_fp8.mp4",
    "video_md5": "042cfe644e72f2ff64da17a0eb9fd28c",
    "audio_md5": "0ffd125508f36a511f0531752456228a"}
}
```

`_compare_g6`のmatch/mismatch両方のロジックを、既存の実ファイル（`c0_23_fp8`
本人のmp4を`g6_23_fp8`として渡す＝一致するはず／`c0_23_fp8_pixar`のmp4を
`g6_23_fp8`として渡す＝不一致するはず、という2通り）で検証し、想定どおり
動作することを確認済み（テストで作った偽の`results/g6_compare_g6_23_fp8.json`
は確認後に削除済み）。

### (3) `gates.py --resume`

`--resume` を追加した。**single/iclora/chain**は `results/<tag>.json` が
`status=="completed"` なら再実行せずスキップする。**keep2**は
`<tag>_r1.json`/`_r2.json` の両方が `completed` なら `keep2 --reuse`
（生成をやり直さず、新基準の再判定だけ行う）を呼ぶ。**models確認・load・
restore・psnr-ssimは安価なので `--resume` でも毎回実行する**（この4種類に
スキップ判定は実装していない——GPU/ジョブ投入を伴わない・数秒で終わる操作
のため、都度実行しても実害が無い）。G6の`load`/`single`も同様の規律
（single はresume可、比較は毎回実行）。

**`gates.py --resume --dry-run --only a` の実行結果**（(a)は既に
single/pixar/先読みオフ/iclora/chain が `completed`、keep2のr1/r2も
`completed`だった実際の状態に対して実行）:

```
(a) models確認（候補）                                    → 実行（常時）
(a) load 候補                                              → 実行（常時）
(a) single 512x320x49f seed12345                          → [resume: スキップ]
(a) single + Pixar_Toon:0.8                                → [resume: スキップ]
(a) single 先読みオフ（同期経路）                          → [resume: スキップ]
(a) iclora deblur                                          → [resume: スキップ]
(a) chain 2クリップ                                        → [resume: スキップ]
(a) keep2（G4）（--resume: 既存のr1/r2から新基準で再判定のみ・生成なし）
                                                            → 実行（--reuse付き。GPU不使用）
(a) models確認（基準GGUF）                                 → 実行（常時）
(a) load 基準GGUF                                          → 実行（新規）
(a) single 基準GGUF（同条件・G5）                          → 実行（新規、未完了のため）
(a) psnr-ssim: int8 single 対 基準GGUF                     → 実行（常時）
(a) psnr-ssim: Pixar 対 single（LoRA有無の差）             → 実行（常時）
(a) restore 元へ（sulphur_distil_fp8mixed）                → 実行（常時）
```

**確認できた点**: models確認/load/keep2(--reuse)/psnr-ssim/restoreは
「安価なので毎回実行する」設計どおり常に表示されるが、単発生成5本
（single/pixar/先読みオフ/iclora/chain）は`--resume`で正しくスキップされ、
**残る実質的な未完了作業は「models確認基準GGUF→load基準GGUF→single基準
→psnr-ssim×2→restore」の系列に絞られる**（コーディネーター確認依頼の
とおり）。全stage・全converterでの`--resume --dry-run`も実行し、306行中
5件の`[resume: スキップ]`（すべて(a)の分。他の変換器はまだ記録が無いので
スキップ対象なし）が出ることも確認済み。

## 動作確認（2026-09-27・GPU/ジョブ投入なし）

- `gates.py --dry-run`（全stage・全converter）を実行し、**306行の全コマンド
  列が期待どおりの順序・引数で印字される**ことを確認済み（`--only`/
  `--stage`/`--skip` の絞り込みも動作確認済み）。
- `BAD_HEADER_CASES` の新5ケースは、実際にバイト列を組み立てて
  `g1_census.py` のヘッダリーダーで読み、**48ブロック・正しいテンソル数・
  想定どおりの印解決**（`nvfp4_format`は`parse_quant_marker`が
  `UnknownQuantFormatError`で拒否、`w4a8_no_codebook`と
  `weight_scale_wrong_shape`は印自体は正しく解決される＝欠落/形違いの
  検出はより上位の層の仕事、`i8_no_marker`と`quanto_data_placement`は
  印が一切検出されない＝狙いどおり）であることを確認済み。
- `run_step`/`GateFailure`/`run_converter_gate`の`try/finally`は、実際に
  失敗するコマンドと成功するコマンドの両方で確認済み。**さらに実機
  （バックエンド停止中）に対して`models確認`ステップを実行し、実際に
  `BACKEND_UNREACHABLE`で失敗→`GateFailure`送出→`finally`の`restore`
  ステップが（モック関数で）それでも呼ばれることを確認済み**
  （GPU・ジョブ投入は一切発生していない。`models`はbackend_status/
  list_modelsと同じ読み取り専用カテゴリ）。
- `single`の`--prefetch`引数は`--help`の表示と`_single_payload()`の
  ロジック確認のみ（実際に`block_swap_prefetch: false`が
  `submit_generate`へ届くところまでは未検証。実機G3で確認されるはず）。

## 未解決点

1. **(d)(e)のダウンロード未完了。** `.part`が外れるまで`models確認`で
   止まる（想定どおりの安全な挙動だが、実行前に確認すること）。
2. **LTX25のIC-LoRA `deblur`登録の有無が未確認。** 前回調査時点
   （2026-09-25）ではLTX25用IC-LoRAは0件だった。バックエンド起動後
   `list_loras`で確認すること。未登録なら(d)(e)の`iclora`ステップは
   422で失敗し、台本はそこで正しく止まる（次の変換器へは進まない）。
3. **`--prefetch off`が実際にsubmit_generateのペイロードに正しく届き、
   同期スワップ経路が実際に使われるかは実機未確認。** `block_swap_
   prefetch_used`が`metadata.json`に`"off"`と記録されることをG3実施時に
   確認すること。
4. **G7の新5ケースの拒否理由の文言。** 現行の稼働中サーバーが
   §3-168のどの実装段階のコードを積んでいるか（旧`sft_fp8_format.py`の
   ままか、`sft_quant_format.py`へ移行済みか）はこのセッションからは
   確認できていない。どちらでも「拒否される」こと自体は期待できるが、
   拒否理由の文言が計画§2.1の想定と一致するかはサーバーの実装段階次第。
5. **G8「connector復元時間・EP常駐オン/オフの再発」は自動化していない。**
   計測フィールドの追加実装が要る（上記G8節）。
6. **G6の`load`はG3〜G5の各変換器のloadと独立している。** つまりG6は
   G3〜G5より後に走る設計だが、仮にG3〜G5のどこかで失敗して台本が
   止まった場合、G6は実行されない（`--stage g6`で独立に再実行可能）。

---

# G1 突き合わせ台本（`g1_census.py` / `g1_equiv.py`）

計画（`floofy-enchanting-sundae.md`）第4節 G1「NumPy 突き合わせ」用。両方とも
標準ライブラリ＋numpy（＋torch は `g1_equiv.py` のみ）で動き、変換ツール
`Nz-GGUF-Converter-LTX23\src\converter\comfy_dequant.py`（編集しない・
`sys.path` に `src` を足して import するだけ）の公開関数を使う。

## `g1_census.py`（G1 (ii)）

```
python g1_census.py <safetensors のパス> [--json OUT.json] [--product]
```

- ヘッダ（先頭8バイトの長さ＋JSON）と `comfy_quant` テンソルの数十〜数百バイト
  だけを seek + read で読む（mmap・全読み禁止）。
- 層ごとの印の解決は計画 §2.1 のとおり: `__metadata__._quantization_metadata
  .layers` にあればそれ、無ければ `<層>.comfy_quant`（両方の経路を実装・
  検証済み。下記「動作確認」参照）。
- 印は変換ツールの `parse_quant_marker` にそのまま渡して解釈する（このスクリ
  プト自身は format の妥当性を判定しない）。受理された層は
  (scheme, format, convrot) 別の本数、拒否された層は raw format 別の本数を
  表にする。
- 重み(`.weight`)の dtype 分布、`weight_scale`/`weight_s_channel`/
  `weight_s_rel`/`weight_codebook` の shape 分布も表示する。
- `--product`: 製品側の判定（`sft_quant_format.layer_schemes`、§3-168 で
  これから実装される）とのつき合わせのスタブ。今はモジュールが無いので
  「import 失敗（未実装・想定内）」の1行を出すだけで正常終了する
  （`compare_with_product()`／`product_layer_schemes()`）。実装後は
  `sft_quant_format.layer_schemes(path, header, prefix) -> dict[str,str]`
  （計画 §3 C-1b の公開API）をそのまま呼べるようになっている。

### 動作確認（2026-09-27・実ファイル1本＋自作の合成ファイル1本）

- **実ファイル**: `models/LTX23/Weights/sulphur_distil_fp8mixed.safetensors`
  （現行の fp8 混合精度ファイル、29,161,842,846 バイト）に対して実行し、
  `_quantization_metadata.layers` 経由で1,232層を正しく検出、全層が
  `float8_e4m3fn`（int8/w4a8 ではない）として `parse_quant_marker` に正しく
  拒否されることを確認した（fail_counts に `float8_e4m3fn: 1232` として
  集計される）。重み dtype 分布 `F8_E4M3: 1232`、`weight_scale` の shape 分布
  `(): 1232` も手動検証済みの実測値と一致した。
- **合成ファイル**: `comfy_quant` テンソル経由（`_quantization_metadata` を
  持たない REDGraft 型のファイルを模した）で、`int8_tensorwise`（convrot
  あり／なし）と `asym_w4a8_int8` の3層＋未対応format（`nvfp4`）1層を含む
  最小 safetensors を自作して実行し、scheme_counts が
  `int8:1 / int8_convrot:1 / w4a8:1`、fail_counts が `nvfp4:1` と正しく
  出ることを確認した（`weight_scale`/`weight_s_channel`/`weight_s_rel`/
  `weight_codebook` の shape 分布も期待どおり）。合成ファイルはテスト後に
  削除済み（リポジトリにもscratchpadにも残していない）。

## `g1_equiv.py`（G1 (i)(iii)）

**engine 側の venv（torch入り）で実行すること**:

```
S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\.venv-engine\Scripts\python.exe g1_equiv.py [--device cpu|cuda] [--seed N] [--worktree PATH]
S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\.venv-engine-ltx25\Scripts\python.exe g1_equiv.py ...
python g1_equiv.py --file <safetensors> --layer <接頭辞込みの層名>
```

torch の場所は2箇所（2026-09-27 実測、どちらも `torch==2.9.1+cu128`・
`torch.cuda.is_available()==True`）:

| venv | 用途 |
|---|---|
| `.venv-engine\Scripts\python.exe` | LTX 2.3 エンジン |
| `.venv-engine-ltx25\Scripts\python.exe` | LTX 2.5 エンジン |

（`run.ps1` はアプリ本体を `.venv\Scripts\python.exe` で起動するだけで、
推論エンジン自体は別プロセス・別venvとして `engine/worker.py` 等から
spawn される。engine venv の場所は `services/engines/ltx/adapter.py`・
`services/engines/ltx25/adapter.py` の docstring 中のコメントから確認した。）

`--device` を省略すると cpu と（使えれば）cuda の両方を自動で実行する。
`torch` 自体が無いインタプリタ（アプリ用 `.venv` 等）で実行しても、torch側の
比較を丸ごとスキップして NumPy 側の健全性（有限値・形状）だけ確認し、
クラッシュしない。

**製品側 `engine.sft_quant.dequant`（`dequantize`／`normalize_aux`）は
§3-168 の実装用 worktree `S:\OriginalApps\12_Nz-LTX23-AviUtl2\_wt_3168` に
実装済み**（2026-09-27 時点、`SCHEME_TABLE` は `fp8`/`fp8_scaled`/`int8`/
`int8_convrot` の4方式。`w4a8` はまだ無い＝C-3でこれから）。
`--worktree`（既定が上記パス）を sys.path の先頭に足してから
`from engine.sft_quant.dequant import dequantize, normalize_aux` する。
**補助テンソルの正規化は必ず製品の `normalize_aux(scheme, leaf, raw, o, i)`
を通す**（スクリプト側で (o,1) 等へ手で整形しない。`raw_aux` には実ファイルに
入っているとおりの生の形 -- スカラーは真に shape `()` -- を渡す。NumPy 側の
変換ツールだけは自分では広げてくれないので、`_numpy_ready_scale()` で
別途 (o,1) へブロードキャストして渡す）。`w4a8` は `normalize_aux`/
`dequantize` の両方が `SCHEME_TABLE`/`if-elif` に無い方式として例外を出す
ので、比較は失敗して当然 -- `torch_error` に記録するだけで異常終了しない。

### 判定式（2026-09-27 診断→修正: bf16 ulp に絶対の床つき OR 判定）

**症状**: 乱数ケースを CPU/CUDA で実行したところ、`int8_scalar`／
`int8_rowwise`（ConvRot なし）は `fp32 |Δ|max=0`・`bf16 ulp_max=0` で完全
一致だったが、`int8_convrot_scalar`／`int8_convrot_rowwise`（ConvRot あり）
は `fp32 |Δ|max` が 2e-6〜8e-6（許容 6e-5〜8e-5 の内側）と小さいのに
`bf16 ulp_max` が 27,000 前後になり、当初の「両方満たす」判定では
MISMATCH と誤判定していた。

**仮説（0 近傍の符号反転）**: ConvRot はアダマール回転（+1/-1 の線形結合）
なので、係数がほぼ打ち消し合う成分は真値が 0 近傍（|W| ~ 1e-7〜1e-8）になる。
この領域は bf16 の指数部が非常に小さく隣接値の間隔が極小なので、NumPy の
`np.matmul`（BLAS）と torch の `torch.matmul` の**加算順の違いによる
~1e-7 オーダーの丸め誤差だけで符号が反転**し、ulp 距離（0 を挟んだ全順序
距離）が数万に達する。

**診断結果（仮説を確認）**: 乱数ケース・実ファイル層のどちらでも、
`ulp>1` の要素は**例外なく** `|W|`（NumPy側の復元値の大きさ）が
`1e-6`〜`6e-8` のオーダー、`|Δ|` も `1e-6`〜`1e-8` のオーダーで、
テンソル全体の絶対の床（`1e-5 * max|W|`、`5e-7`〜`8e-5`）の**内側**に
収まっていた（詳細は下記「実行結果」の `[ulp>1: ...]` 診断）。仮説どおり。

**修正**: 判定を要素ごとの OR に変更した（`compare_arrays()`）:
```
(bf16 ulp <= 1) OR (fp32 |Δ| <= 1e-5 * max|W|)
```
両方とも偽の要素だけを「真の不一致」（`n_bad_or`）として数える。`ulp>1` の
要素数・その中の `|W|`／`|Δ|` の範囲（診断用）は常に表示する。修正後、乱数
ケース・実ファイル層とも **`n_bad_or` は全ケースで0**（下記「実行結果」）。

**`torch.backends.cuda.matmul.allow_tf32`**: 実行環境では `False`
（TF32 は無効。今回の不一致は TF32 起因ではなく、上記の 0 近傍の丸め誤差の
みが原因と判断できる）。

### 実行結果（2026-09-27・`.venv-engine`・CPU と CUDA・修正後）

**(i) 乱数ケース**（`PYTHONPATH` は使わず `--worktree` 既定値で worktree を
解決。shape 3種 × 5ケース × cpu/cuda = 30通り、`--seed 12345`）:

| ケース | fp32 \|Δ\|max | bf16 ulp_max | ulp>1 の要素数（診断） | \|W\|範囲（診断） | n_bad_or |
|---|---|---|---|---|---|
| int8_scalar | 0 | 0 | — | — | **0** |
| int8_rowwise | 0 | 0 | — | — | **0** |
| int8_convrot_scalar | 3.8e-6〜8.1e-6 | 26,945〜27,381 | 22,273/67,108,864（大形状）・54/131,072（小形状） | ≤2.15e-6 | **0** |
| int8_convrot_rowwise | 4.8e-6〜6.7e-6 | 26,961〜27,322 | 22,200/67,108,864・53/131,072 | ≤1.43e-6 | **0** |
| w4a8 | — | — | — | — | 比較スキップ（`KeyError: 'w4a8'` = 製品側未実装。想定内） |

3形状（[16384,4096]・[4096,16384]・[32,4096]）とも同じ傾向（int8系は完全
一致・convrot系はulp>1だが全て床の内側）。CPU/CUDA間の差はごく僅か
（[32,4096] の convrot で CPU 27,047 vs CUDA 26,945 等、TF32無効なのでほぼ
BLAS実装差のノイズ）。exit code **0**（全通過）。

**(iii) 実ファイルの層1本**（Kijai `ltx-2.3-22b-distilled-1.1_transformer_
only_int8_convrot.safetensors`＝int8_convrot・行ごと、silveroxides
`ltx-2.3-22b-distilled-1.1_int8mixedtensorwise.safetensors`＝int8・
スカラー・印はメタデータ側。層名はヘッダから実在するものを選んだ）:

| ファイル | 層 | scheme | shape | fp32 \|Δ\|max | bf16 ulp_max | ulp>1要素/全体 | n_bad_or |
|---|---|---|---|---|---|---|---|
| Kijai | `transformer_blocks.10.attn1.to_q` | int8_convrot | (4096,4096) | 3.576e-07 | 25,983 | 12,873/16,777,216 | **0** |
| Kijai | `video_embeddings_connector.transformer_1d_blocks.0.attn1.to_k` | int8_convrot | (4096,4096) | 6.706e-08 | 25,645 | 10,121/16,777,216 | **0** |
| silveroxides | `transformer_blocks.10.attn1.to_q` | int8 | (4096,4096) | 0 | 0 | — | **0** |

CPU/CUDAで数値は完全に同一（この3層はいずれも数値差が出なかった＝BLAS実装
差が出るケースと出ないケースがある）。3層とも exit code **0**。connector側
（`video_embeddings_connector`）も transformer側と同じ傾向で、layer の場所
による違いは無い。

### 動作確認の要点（再掲）

- 乱数ケースは shape [16384,4096]・[4096,16384]・[32,4096] × 5ケース ×
  cpu/cuda の全通りで NumPy 側が有限値・想定形状を返すことを確認済み。
- `--file/--layer` モードは実ファイルのヘッダオフセットからの seek 読み →
  dtype 復元（`F8_E4M3`/`F8_E5M2` は生の `U8` バイト列として読む。
  `_np_array_from_tensor()`。`.copy()` で書き込み可能にしてから
  `torch.from_numpy` に渡す）→ `dequantize_layer` 呼び出しの一連が正しく
  動くことを確認済み（合成ファイルでの単体確認に加え、今回は実ファイル3層
  でも確認）。
- `bf16_ulp_diff()`（旧 `max_bf16_ulp_diff()`。要素ごとの配列を返すように
  変更）の正しさは、既知の隣接bf16値のペア（`1.0` とその次に大きい表現可能
  値 `≈1.0078125`）で ulp差がちょうど `1` になることを単体で確認済み
  （同一配列は ulp=0・fp32一致、大きく異なる配列は大きな ulp になることも
  確認済み）。

## 確かめられなかった点（推測で埋めなかった箇所）

1. **`get_mp4_info` の `comment` フィールドの実際のJSON構造。** ドキュメント
   （`mcp_server/tools/outputs.py`・README §8）から `metadata.json` と同じ
   JSON が返ることは分かるが、実際に生成ジョブを1本も走らせていないため
   `_meta_summary()` の抽出ロジック（`models.selection.transformer.{name,file}`
   等のキー名）は前回ランナー（`stage3_runner25.py`）の `outputs/<job_id>/
   metadata.json` 直接読みのロジックを移植しただけで、`get_mp4_info` 経由でも
   同じキー名が届くという想定に基づく（`embed_mp4_metadata` の説明文には
   「`metadata.json` と同じ JSON」と明記されているので、キー名が変わる可能性
   は低いと見ている）。
2. **`load_pipeline` の45秒タイムアウト後のポーリングが実際に収束するか。**
   `backend_status` を2秒間隔でポーリングして `state` が `ready`/`error` に
   なるまで待つ実装にしたが、ベースモデル切替を伴う読み込み（数十GB級の
   int8/w4a8 ファイル）でどのくらいの回数ポーリングが必要かは実測していない。
3. **`bad-header` のケースが実際に `ToolError` として弾かれる際の文言。**
   現行コードの検査モジュールは `sft_fp8_format.py`（文言接頭辞「fp8
   safetensors の検査に不合格: 」）のままで、計画が予定している改称後の
   `sft_quant_format.py`（文言接頭辞「量子化 safetensors の検査に不合格: 」）
   はまだ実装されていない。どちらの文言が返ってきても `rec["tool_error"]` に
   そのまま記録されるだけなので実装には影響しないはずだが、実機では未確認。
4. **`iclora`/`chain`/`keep2` の実引数の組み合わせが実際のバックエンドの
   受理条件（128グリッド・8n+1フレーム等）を満たすか。** ツールの docstring
   から読み取れる制約はコードに反映した（IC-LoRA は128の倍数、等）が、実際に
   投げて確認してはいない。
5. **`save_job_video` の `no_clobber` と `--overwrite` の対応関係。** ツールの
   引数は `no_clobber: bool`（既定True）なので、CLI側は `--overwrite` を
   指定したときだけ `no_clobber=False` を送る設計にしたが、実際の連番衝突時の
   挙動（`_2`, `_3` ...）は前回ランナーの `stage3_runner25.py` には無かった
   副命令のため実機で未確認。
6. **（2026-09-27 解消）`g1_equiv.py` の aux 組み立てが製品側の実際の実装と
   一致するか。** §3-168 実装用 worktree `_wt_3168` に着地した
   `engine.sft_quant.dequant.normalize_aux` を実際に呼び出す形へ書き換えて
   検証した（`int8`/`int8_convrot` の乱数ケース30通り＋実ファイル層3本の
   すべてで NumPy 側と一致、詳細は上の「G1 突き合わせ台本」節）。**残るのは
   `w4a8` のみ**（`SCHEME_TABLE` にまだ無い＝C-3スコープ、`normalize_aux`/
   `dequantize` とも `KeyError`/`RuntimeError` で失敗するのが正しい。C-3で
   `w4a8` の行が追加されたら再実行するだけでよい）。
7. **`_quantization_metadata` を持つファイルで、その層に対応する
   `<層>.comfy_quant` テンソルが同時に存在するとき、両者の中身が食い違う
   実例が存在するか。** 実測した1ファイル（`sulphur_distil_fp8mixed
   .safetensors`）では両方が存在する1,232層すべてで中身が完全一致していた
   （`{"format": "float8_e4m3fn"}` のみ）が、これが一般的な保証かは未確認
   （計画どおり「層ごとにメタデータ優先」で実装済みなので、食い違いがあって
   も規則には従う）。
