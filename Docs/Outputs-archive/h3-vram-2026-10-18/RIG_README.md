# MiniMax H3 の VRAM 溢れの計測台（2026-10-18）

台帳 §1-101 の第 1 段（計測）。計画の正本は監督の `PLAN_measure.md`（scratchpad の `h3vram\`。後で `Docs/Outputs-archive/h3-vram-2026-10-18/` に写す）。このフォルダは git 管理外（`.gitignore` の `outputs/*`）。

**製品コードは 1 行も変えない。** 台本はセルフテストと同じ合成メッセージで `engine_h3.worker._do_load`／`_do_generate` を呼び、割り込みはワーカーの関数（`_emit`）とパイプラインの属性（`_PIPE._progress`）を包むだけで行う。

## 置いてあるもの

| ファイル | 役目 |
|---|---|
| `measure_h3_vram.py` | 腕（計測条件の組）1 本を 1 プロセスで回す台本。`--dry-run` で試運転 |
| `vram_sampler2.ps1` | WDDM（Windows の GPU メモリ管理）の専用／共有メモリの採取器。`outputs/comfort-calib-2026-10-04/` からの写し（変更なし） |
| `snapshot_report.py` | メモリの確保と解放の履歴（スナップショット）の集計。割り当てが最大だった瞬間を復元する |
| `sdpa_int32_probe.py` | T6 の合成試験（SDPA の mem-efficient 経路の int32 の掛け算の溢れ。モデルを使わない） |
| `runs/<label>/` | 腕ごとの採取物（下記） |
| `dryrun.json` | 最後に行った試運転の結果（`runs/dryrun/dryrun.json` と同じ） |

## 条件（計画 §2 の要約）

- 機械: 開発機（RTX 4070 Ti SUPER 16GB・主メモリ 64GB）。python は `.venv-engine-h3`、環境変数 `PYTHONUTF8=1`。
- **実行前にオーナーのバックエンド（`run.bat`）を止めるか、H3 の常駐を解放してもらう**（常駐中は主メモリ約 36GB を占め、計測のコミット量の山〔オーナーの I2V の実測で 44.6GiB〕と重なる）。動画再生やブラウザの GPU 加速など他の GPU 利用は控える。
- 画角 A＝T2V 960×1152・124 フレーム（台本の既定）。画角 B＝T2V 640×384・124 フレーム（基準点）。
- ステップ数 2・seed 42・プロンプトはセルフテストの既定・先読み（ブロックの先回り転送）on・deterministic off。復号も通す。
- **1 腕 1 プロセス**。台本は実際に効いた値（常駐ブロック数・attention・先読み・受け皿〔先読み用の GPU 上の置き場〕の本数・割り当て器の上限）を `run.json` に書き、引数と合わなければ `effective_mismatch` を立てて終了コード 2 で終わる。
- 時間の代表は **2 ステップ目**＝「progress の index 2 の時刻 − index 1 の割り込みを終えた時刻」（`run.json` の `timing.step2_seconds`。時刻の前に `torch.cuda.synchronize()`）。index 1→2 の `num_alloc_retries`（確保の失敗からの再試行の回数）・`num_device_alloc`・`num_device_free` の増分は `timing.counts_step1_to_step2` にある。腕の間でこの回数が 2 倍以上違うとき、また履歴ありの腕と無しの腕の間の時間の比較は「参考」扱い（計画 §2）。
- 共有メモリの基準＝T0 のノイズ除去の窓の共有メモリの中央値。判定は COMFORT_LIMIT_TABLE §4.2 の v3 の閾値（+300MB）を借りるが v3 そのものではない（基準の画角が違う）。

## 腕の表（計画 §3。この順で、GPU は同時に 1 本）

| 腕 | ラベル | 画角 | attention | 常駐 | 上限 | 履歴 | 目的 |
|---|---|---|---|---|---|---|---|
| T0 | `T0_B_sdpa_b8_nocap_hist` | B | sdpa | 8 | なし | あり | 基準・台本の動作確認（門 G-M2） |
| T1 | `T1_A_sage_b8_nocap_hist` | A | sage | 8 | なし | あり | §166.9 の再現・ピーク時の隙間の内訳（H1a） |
| T2 | `T2_A_sdpa_b8_nocap` | A | sdpa | 8 | なし | なし | H2: sdpa だけで溢れを免れるか |
| T3 | `T3_A_sage_b8_cap085_hist` | A | sage | 8 | 0.85 | あり | H1b: キャッシュを返させれば溢れが消えるか |
| T4 | `T4_A_sdpa_b8_cap085` | A | sdpa | 8 | 0.85 | なし | H3: 溢れの無い条件での sdpa 対 sage |
| T5a | `T5a_A_<att>_b4_cap085` | A | T3／T4 で速かった方 | 4 | 0.85 | なし | H4 |
| T5b | `T5b_A_<att>_b2_cap085` | A | 同上 | 2 | 0.85 | なし | H4 |
| T6 | `T6_probe` | — | sdpa | — | — | — | H5（合成試験。別プロセスで最後に） |
| F0 | `F0_I2V_896x1152_sage_b8_nocap_keep_det` | I2V 896×1152×124 | sage | 8 | なし | なし | 対照（計測台の検証）。オーナーの 1 回目の実行（`outputs/dc9491a2-…`）と同じ画像・プロンプト・常駐の保持・deterministic・3 ステップで、操作パネルからの実行と同じ減速が出るか |

上限 0.85 ≈ 13,920MiB（物理 16,376MiB × 0.85）。実行時の条件: 読み込み前の `mem_get_info` の空きから torch 以外の持ち分の見込み（約 0.5〜0.8GB）を引いた値より下にあること。`run.json` の `memory_cap.cap_exceeds_preload_free_minus_800mib` が true なら監督に知らせる。T1 の実測の割り当てが上限を超えていたら、上限を「割り当て＋800MiB」に合わせ直してよい（値を記録する）。

## 起動コマンド（PowerShell。リポジトリのルートで）

```powershell
$env:PYTHONUTF8 = "1"
$py = ".\.venv-engine-h3\Scripts\python.exe"
$m  = ".\outputs\h3-vram-2026-10-18\measure_h3_vram.py"

# 試運転（モデルを読まない。GPU は 64MiB の確保 1 回だけ）
& $py $m --dry-run --label dryrun

# T0〜T5b
& $py $m --label T0_B_sdpa_b8_nocap_hist  --width 640 --height 384 --attention-backend sdpa --blocks-on-gpu 8 --record-history
& $py $m --label T1_A_sage_b8_nocap_hist  --attention-backend sage --blocks-on-gpu 8 --record-history
& $py $m --label T2_A_sdpa_b8_nocap       --attention-backend sdpa --blocks-on-gpu 8
& $py $m --label T3_A_sage_b8_cap085_hist --attention-backend sage --blocks-on-gpu 8 --memory-fraction 0.85 --record-history
& $py $m --label T4_A_sdpa_b8_cap085      --attention-backend sdpa --blocks-on-gpu 8 --memory-fraction 0.85
# <att> は T3／T4 の 2 ステップ目の秒数で速かった方（sdpa か sage）
& $py $m --label T5a_A_<att>_b4_cap085    --attention-backend <att> --blocks-on-gpu 4 --memory-fraction 0.85
& $py $m --label T5b_A_<att>_b2_cap085    --attention-backend <att> --blocks-on-gpu 2 --memory-fraction 0.85

# F0（対照。I2V・常駐の保持・deterministic・3 ステップ。画像とプロンプトはオーナーの 1 回目の実行と同じ）
& $py $m --label F0_I2V_896x1152_sage_b8_nocap_keep_det --width 896 --height 1152 --num-frames 124 --attention-backend sage --blocks-on-gpu 8 --no-record-history --steps 3 --image "S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\uploads\71c540dc-970b-470e-aa0e-856dce76dfdc\input.png" --prompt-file <UTF-8 のプロンプトのファイル> --keep-resident --deterministic

# T6（S ごとに別プロセス。結果は probe_result.json に書き足す）
& $py .\outputs\h3-vram-2026-10-18\sdpa_int32_probe.py --seq 98304
& $py .\outputs\h3-vram-2026-10-18\sdpa_int32_probe.py --seq 100352

# スナップショットの集計（履歴ありの腕のあと）
& $py .\outputs\h3-vram-2026-10-18\snapshot_report.py .\outputs\h3-vram-2026-10-18\runs\T1_A_sage_b8_nocap_hist\snapshot_step1.pickle
```

F0 のために足した引数（既定では従来の振る舞いのまま）: `--image`（I2V の先頭フレーム）・`--prompt-file`（UTF-8 のプロンプト）・`--keep-resident`・`--deterministic`。進捗の採取は `stage1_denoise` の index 1 以上の全部で行い（`memory_stats_step<k>.json`）、`run.json` の `timing.step_seconds` に各ステップの秒数を配列で入れる。効いた値の検査に `keep_resident_used`・`images_used`・`build_report` の `deterministic` を足した。

台本の既定: 画角 A（960×1152×124）・`--steps 2`・`--seed 42`・`--idle-seconds 10`・`--history-entries 2000000`・モデルのパスは `models/MiniMaxH3/` の配布ファイル（`scripts/manifests/40-minimaxh3.json` が正本。名前は `engine_h3/worker.py` のセルフテストと同じ `--transformer`／`--text-encoder`／`--video-vae`／`--audio-vae`／`--tokenizer-dir`）。

## 採取物（`runs/<label>/`）

| ファイル | 中身 |
|---|---|
| `run.json` | 引数・実際に効いた値（`effective`）・食い違い（`effective_mismatch`）・sdpa の経路・全イベント（`events`）・progress の時刻（単調時計と壁時計）・index 1／2 の採取（`samples.step1`／`step2`）・`done`・`timing`・torch 以外の持ち分の見積もり 2 通り・WDDM の要約と待機中の中央値・`error`（OOM など） |
| `memory_stats_step1.json`／`_step2.json` | `memory_stats()`（allocated／reserved／inactive_split の current と peak・再試行・OOM・確保と解放の回数）と `mem_get_info()`。step1 には受け皿の本数とプロセスごとの GPU メモリの要約、履歴ありならスナップショットの書き出し秒数 |
| `snapshot_step1.pickle` | 履歴ありの腕。index 1 の時点の履歴とセグメント（`snapshot_report.py` で集計） |
| `snapshot_error.pickle` | 履歴ありの腕で例外が出たとき。OOM の瞬間の状態を復元する唯一の手段 |
| `wddm.csv` | 専用／共有メモリ（MB）。約 1 秒刻み |
| `dmon.txt` | `nvidia-smi dmon -s ut -o DT -d 1`（使用率と PCIe の受信・送信 MB/s） |
| `gpu_process_memory_preload.csv`／`_step1.csv` | プロセスごとの専用／共有（`\GPU Process Memory(*)`）。`is_self`＝この台本 |
| `luid.json` | この回に取り直した LUID（アダプタの識別子）。**記録であってキャッシュではない**（毎回取り直す） |
| `stdout.log` | 台本の標準出力と標準エラーの写し（ワーカーのログ `[h3_worker]` を含む） |
| `out.mp4` | 生成物 |

終了コード: 0＝完了（OOM などの例外も「結果」として `error` に記録して 0）／2＝実際に効いた値が引数と合わない／1＝台本の失敗（モデルが無い・採取器が動かない等）。

## 再実行の仕方

- 同じラベルで走らせると `runs/<label>/` の各ファイルを上書きする（`stdout.log` だけは追記）。やり直す前に、残したい場合はフォルダごと `runs/<label>_try1/` などへ改名する。
- 腕ごとに台本を起動し直す（1 腕 1 プロセス）。同じプロセスで 2 本目を回す仕組みは持たせていない。
- 終了コード 2 のときは `run.json` の `effective_mismatch` を読み、原因（例: sage が使えず sdpa に落ちた＝`attention_used` が `sage->sdpa`）を監督へ報告する。勝手に条件を変えて再実行しない。

## クールダウンの手順（腕と腕の間）

1. 台本が終わったら **30 秒以上** 待つ。
2. 次の腕の前に、共有メモリとコミット量（Windows が約束した仮想メモリの総量）が待機の水準へ戻ったことを確かめる。目安は前の腕の `run.json` の `wddm.idle`（待機中の専用・共有の中央値）。タスクマネージャーの「パフォーマンス → GPU」の「共有 GPU メモリ」と「メモリ → コミット済み」でも見られる。
3. 次の腕の台本は、読み込み前に 10 秒の待機を採取して `run.json` の `wddm.idle` と `mem_info_preload` に書くので、戻っていなかった場合も後から分かる。戻り切らないまま走らせた腕は、RESULTS で「参考」扱いにする。

## 注意（試運転で分かったこと）

- **LUID は起動のたびに変わる。** 10-04 の較正台の `luid.json` は `luid_0x00000000_0x0000f29f_phys_0`、今日（10-18）の試運転では `luid_0x00000000_0x0000f2a9_phys_0` だった。古い値を使うと採取器は例外にならず 0 を書き続ける。台本は毎回取り直す。
- **プロセスごとの GPU メモリのカウンタは合計がアダプタ全体を超える。** 試運転では dwm（画面合成）だけで専用 4,591MB、他のプロセスの合計 5,292MB に対し、アダプタ全体の専用は 1,147MB だった（共有の面を重ねて数えていると見られる。推測）。そのため torch 以外の持ち分は 2 通りで見積もる: `non_torch_estimate_mib`（計画どおり、他プロセス＝プロセスごとのカウンタの合計）と `non_torch_estimate_mib_adapter_idle`（他プロセス＝待機中〔このプロセスが CUDA を使い始める前〕のアダプタ全体の専用の中央値）。後者のほうが筋がよいと見る。
- 履歴の `alloc` の `size` は要求した大きさ（丸める前）。`snapshot_report.py` は割り当て器と同じく 512 バイトの倍数に切り上げて数える。
