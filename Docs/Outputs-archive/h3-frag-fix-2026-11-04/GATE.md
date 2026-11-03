> **このファイルは `outputs/h3-frag-fix-2026-11-04/GATE.md` の複写です（2026-11-04）。**

# 実機ゲートの記録（台帳 §1-270・§1-271）— 2026-11-04

- 対象: `Nz-Videomni` の dev・HEAD `bcf0caa`（コミット 1 `c5b1095`＝§1-270、コミット 2 `bcf0caa`＝§1-271）。計画は `enchanted-watching-deer.md` §4。
- 実施: 2026-11-04 01:10〜01:36。GPU を使う処理は 1 本ずつ流し、各工程の前に使用量が 1,200MiB 未満であることを確かめてから起動しました（時刻と待ち明けの使用量は `arms.log`）。
- 結論: **4-0・4b・4c・4d はすべて合格です。4e は記録のみです**（警告が出て、`expandable_segments` は新名・旧名とも false）。

## 工程ごとの合否

| 工程 | 結果 | 要点 |
|---|---|---|
| 4-0 前提 | 合格 | dev・`bcf0caa`・`git status --short` は空・GPU 610MiB・python のプロセス 0 本・ワーカーの孤児なし（`preconditions.txt`） |
| 4b GPU のテスト | 合格 | `187 passed in 34.35s`（スキップ 0・失敗 0）。`4b_gpu_tests.txt` |
| 4c 指紋 | 合格 | off・on とも sha256 が `0da6111cdca853ce718531d14f6970c78c524d01744b22461c7a0cb482f098d9` で、`outputs/h3-vram16-gates/g3_on.mp4` と全桁一致 |
| 4d r1080_277 | 合格 | OOM なしで完走（`done` あり・`error` なし・`mismatch` なし・終了コード 0）。境目の (i)〜(iv) はすべて条件どおり |
| 4e expandable segments | 記録のみ | 新名・旧名とも `UserWarning: expandable_segments not supported on this platform`、値は false |

## 4c 指紋（off と on）

条件は `Docs/Outputs-archive/h3-vram16-2026-10-18/gates/g3.sh` と同じです（FL2VA の重み・黄色いレインコートのプロンプト・selftest の既定 640×384・124 フレーム・seed 42・20 ステップ・sdpa）。変えたのは出力先（`g3/`）だけです。1 ステップの時間を読むため、ログを行ごとの時刻つきで `g3_<arm>.ts.log` にも写し、そのために `PYTHONUNBUFFERED=1` を足しました（数値には関わりません。元のログ `g3_<arm>.log` の書式は前回と同じです）。前回のログには時刻が無いので、前回の 1 ステップの時間はノイズ除去の段の秒÷20 だけで比べます。

| 項目 | 前回 off | 今回 off | 前回 on | 今回 on |
|---|---|---|---|---|
| sha256（先頭 16 桁） | 0da6111cdca853ce | 0da6111cdca853ce | 0da6111cdca853ce | 0da6111cdca853ce |
| 大きさ（バイト） | 854,334 | 854,334 | 854,334 | 854,334 |
| `rounds` の数 | 1 | 1 | 1 | 1 |
| 所要時間（arm の壁時計。秒） | 146 | 156 | 148 | 150 |
| `done.seconds` | 142.93 | 142.70 | 144.72 | 136.81 |
| TE の段の秒 | 20.47 | 22.40 | 17.23 | 17.87 |
| ノイズ除去の段の秒（DiT の読み込みを含む） | 110.31 | 108.36 | 115.48 | 107.19 |
| 同 ÷20 | 5.516 | 5.418 | 5.774 | 5.359 |
| 1 ステップの秒（時刻つきログ。ステップ 1〜19 の平均、最小〜最大） | — | 4.763（4.73〜4.79） | — | 4.732（4.71〜4.77） |
| ステップ 0 の秒 | — | 3.98 | — | 4.79 |
| 先読みの待ち（1 周の平均 ms） | 3,168.7 | 3,055.9 | 3,504.3 | 3,182.4 |
| 復号の段の秒 | 12.02 | 11.81 | 11.87 | 11.61 |
| 全体のピーク 割り当て／予約（MiB） | 6,677／7,590 | 6,677／7,590 | 5,065／5,382 | 5,065／5,382 |
| TE の段のピーク 割り当て／予約（MiB） | 1,971／1,988 | 1,493／1,510 | 1,971／1,988 | 1,493／1,510 |
| ノイズ除去の段のピーク 割り当て／予約（MiB） | 6,677／7,590 | 6,677／7,590 | 3,779／3,790 | 3,779／3,790 |
| `vram16.num_alloc_retries`・`num_ooms` | — | — | 0・0 | 0・0 |
| `vram16.inactive_split_peak_mib` | — | — | 450 | 147 |

- on の境目の `empty_cache` 2 回による時間の増加は、この小さい形（S＝9,335）では見えません。on のノイズ除去の段は前回より 8.3 秒短く、1 ステップは off と同じ水準（約 4.73 秒）です。
- TE の段のピークが on・off とも 478MiB 下がりました（1,971→1,493MiB）。ほかの段のピークは前回と同じです。
- `vram16` の報告に `qkv_quant_shared: true` の鍵が増えています（前回の報告には無い鍵です）。
- 比較の全表は `g3/compare.txt` にあります。

## 4d r1080_277（メモリ確保の履歴つき・1 本）

### 完走と時間

- 完走しました。`run.json` は `done` あり・`error` は None・`mismatch` は None・`effective_mismatch` は None です（今回は `done` が来たので、受け皿の期待数も 6 で一致しました）。計測台の終了コードは 0 です。
- 所要時間: 計測台全体 1,065.79 秒（17 分 46 秒）、`done.seconds` 1,044.82 秒。段ごとには TE 18.58 秒・ノイズ除去 861.89 秒・復号 164.22 秒です。

| ステップ | 今回（`timing.step_seconds`） | 履歴つきの再現（`r1080_277_hist`） | 較正（r1080_277／r1080_277b） |
|---|---|---|---|
| 0（先読みの立ち上がりを含む） | 212.347 | 212.127 | 211.0／215.0 |
| 1 | 211.882 | （ステップ 1 で OOM） | （OOM） |
| 2 | 211.766 | — | — |
| 3 | 211.615 | — | — |

### メモリの数値

| 項目 | 今回 | 修正前（`r1080_277_hist`） |
|---|---|---|
| ノイズ除去の段のピーク 割り当て／予約（MiB） | 10,449／13,090 | 割り当て 10,448.8／予約 13,056（OOM の時点の `memory_stats` のピーク） |
| TE の段のピーク 割り当て／予約（MiB） | 1,492／1,514 | — |
| 復号の段のピーク 割り当て／予約（MiB） | 8,241／10,564 | — |
| 上限（`vram16.cap_mib`） | 14,025 | 14,025 |
| `num_alloc_retries` | 4（各ステップ 1 回。境目ごとの累計 1・2・3・4） | ステップ 1 の境目で 2、OOM の時点で 3 |
| `inactive_split` のピーク（MiB） | 2,883.4 | 3,717.9（ステップ 1 の境目の時点） |
| 境目の予約／割り当て／予約済みの空き（MiB） | **2,750.0／2,434.8／315.2** | 10,734.0／2,434.4／8,299.6 |
| 境目の `inactive_split` の現在値（MiB） | 33.2 | 1,435.6 |
| 境目の GPU 全体の空き（`mem_get_info`。MiB） | 12,279.0 | 4,295.0 |
| 境目の他のプロセスの専用メモリ（MiB） | 2,757.8 | 1,594.4 |

- 注: 計画とご依頼にある `h3.phase_peak_vram_mb`・`h3.phase_peak_vram_reserved_mb` の鍵は、今回の `run.json` にはありませんでした（`h3` の鍵は `phases` と `quant_backends` だけです）。上の段ごとのピークは `h3.phases.<段>.peak_vram_mb`・`peak_vram_reserved_mb` から読みました。
- ステップ 2〜4 の境目（`memory_stats_step2〜4.json`）も、予約 2,738／2,738／2,736MiB・割り当て 2,434.8MiB で、ステップ 1 の境目とほぼ同じです。

### 境目のセグメント（`snapshot_step1.pickle`。判定は `analysis/r1080_277_fix_step1.json` の `segments[]` から）

全セグメント（アドレス順。16 個）:

| # | アドレス | 大きさ MiB | 使用中 MiB | 空き MiB | 最大の連続した空き MiB | 使用中の塊 | 起点（`origin_key`） |
|---|---|---|---|---|---|---|---|
| 0 | 0x1303a00000 | 20.0 | 14.6 | 5.4 | 5.4 | 5 | （フレームなし） |
| 1 | 0x130a000000 | 2.0 | 0.4 | 1.6 | 0.9 | 2 | `pipeline_h3.py:1490 _denoise` |
| 2 | 0x130a200000 | 2.0 | 0.6 | 1.4 | 1.4 | 1 | `dit.py:268 forward` |
| 3 | 0x130b200000 | 2.0 | 1.9 | 0.1 | 0.1 | 12 | （フレームなし） |
| 4 | 0x1318400000 | 84.0 | 0.0 | 84.0 | 84.0 | 0 | `lowvram.py:232 head` |
| 5 | 0x131d800000 | 62.0 | 0.0 | 62.0 | 62.0 | 0 | `dit.py:188 unpatchify_video` |
| 6 | 0x1323800000 | 74.0 | 0.0 | 74.0 | 74.0 | 0 | `sft_load.py:141 move_module_tree` |
| 7 | 0x1328200000 | 62.0 | 0.0 | 62.0 | 62.0 | 0 | `scheduler.py:51 euler_step` |
| 8 | 0x132c000000 | 62.0 | 62.0 | 0.0 | 0.0 | 1 | `scheduler.py:51 euler_step` |
| 9 | 0x133a800000 | 148.0 | 123.2 | 24.8 | 24.8 | 2 | `sft_load.py:141 move_module_tree` |
| 10 | 0x1371e00000 | 372.0 | 372.0 | 0.0 | 0.0 | 1 | `block_swap_prefetch.py:492 _validate` |
| 11〜15 | 0x1389200000 ほか 4 個 | 372.0×5 | 372.0×5 | 0 | 0 | 各 1 | `block_swap_prefetch.py:407 prepare`（受け皿） |
| 合計 | | 2,750.0 | 2,434.8 | 315.2 | | | |

判定:

- **(i) 合格。** 3,300〜3,500MiB のセグメントはありません（修正前は `dit.py:551` と `lowvram.py:203` の 3,432MiB が 2 つあり、どちらも丸ごと空きでした）。
- **(ii) 合格。** `origin_key` にも `origin_chain` にも `sft_load.py:174`・`text_encoder.py` を含むセグメントはありません（修正前は `text_encoder.py:821` の 2MiB・`sft_load.py:174` の 1,484MiB・`sft_load.py:148` の 2MiB の 3 つがありました）。
- **(iii) 合格。** 20MiB のセグメント（0x1303a00000、起点なし）の先頭（オフセット 0）に、8,519,680 バイトの使用中の塊が 1 つあり、起点はありません（フレームなし）。同じセグメントの残りは、DiT のブロック以外の重み 2.0MiB×2（`sft_load.py:141`）、`modality_rows`・`time_rows` の 1.3MiB×2、空き 5.4MiB です。
- **(iv) 合格。** 最大の連続した空きが 100MiB を超えるセグメントは **1 つもありません**（最大は #4 の 84MiB）。S·U 級（1,700〜1,760MiB）のセグメントは境目に存在しないので、そこに小さな塊が居座ることもありません。映像の状態（`scheduler.py:51`、61.3MiB 相当）は 62MiB のセグメント #8 に単独で入り、RoPE の表（`rope.py:317` の 61.6MiB×2）は DiT のブロック以外のセグメント #9 に入っています。

### 気づいたこと

事実:

- 境目に、使用中の塊が 0 の丸ごと空いたセグメントが 4 つ残っています（#4 84MiB・#5 62MiB・#6 74MiB・#7 62MiB、計 282MiB）。最後にそこへ入っていたのは、それぞれ `lowvram.py:233 head`（61.3MiB）・`unpatchify_video`（61.3MiB）・`pipeline_h3.py:1512 _denoise`（61.3MiB）・`euler_step`（61.3MiB）です。
- RoPE の表の塊が 61.6MiB で 2 つあります（#9）。修正前の境目では、TE の段の 1,484MiB のセグメントの中に 61.6MiB が 1 つ見えていました。
- 各ステップで `num_alloc_retries` が 1 回ずつ増えています（上限に当たって割り当て器がキャッシュを返した回数）。OOM はありません。
- 境目の他のプロセスの専用メモリは、修正前の 1,594.4MiB に対して今回 2,757.8MiB でした（同居するアプリの差）。
- 4c の on の最中（01:15〜01:17）に、私の書き込みではない変更が作業ツリーに入りました。`Docs/COMFORT_LIMIT_TABLE.md`・`Docs/MINIMAX_H3_DESIGN.md`・`Docs/VERIFICATION_LOG.md` の変更と、新しいフォルダ `Docs/Outputs-archive/h3-frag-fix-2026-11-04/`（`README.md`・`research_1-270_1-271.md`）です。製品コードの差分は無く、HEAD は `bcf0caa` のままです（`postconditions.txt`）。

推定:

- 丸ごと空いた 4 つのセグメントは、境目の `empty_cache`（計画 2.2 の 2 か所）より後の `unpatchify_video`・`euler_step` などで作られたため、その時点では返されずに残っていると見られます。次のステップの頭で使い回されるか返されるかは、今回の記録からは決められません。量は 282MiB です。
- RoPE の表が 2 つ見えるのは、修正前に TE の段のセグメントに入っていた方が、そのセグメントが無くなったため DiT のブロック以外のセグメントの空きに入ったものと見られます（並びの変化で、量の増加ではない可能性が高いと考えます）。
- TE の段のピークが 478MiB 下がったのは、§1-271 の修正（作業領域の先取りと `text_hidden` の置き場所）によるものと見られます。

## 4e expandable segments の確認

torch 2.13.0+cu130（CUDA 13.0）です。どちらも終了コードは 0 でした。

| 環境変数 | 標準エラーの警告 | 印字された `expandable_segments` |
|---|---|---|
| `PYTORCH_ALLOC_CONF=expandable_segments:True`（新名） | `<string>:1: UserWarning: expandable_segments not supported on this platform (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\c10/cuda/CUDAAllocatorConfig.h:40.)` | false |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（旧名） | 同じ文 | false |

- どちらの場合も、印字された設定の中では、設定の文字列が `"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"` という鍵の名前で出ています（新名で設定しても鍵の名前は旧名です）。
- 判断はしていません。

## ファイル一覧（すべて `outputs/h3-frag-fix-2026-11-04/`）

| ファイル | 中身 |
|---|---|
| `GATE.md` | この記録 |
| `preconditions.txt`・`postconditions.txt` | 前と後の状態（git・GPU・python のプロセス） |
| `arms.log` | 各工程の開始・終了の時刻、終了コード、待ち明けの GPU の使用量 |
| `4b_gpu_tests.txt` | GPU のテストの出力 |
| `g3/g3_arm.sh` | 4c で流したコマンド（g3.sh と同じ引数、出力先のみ変更） |
| `g3/g3_off.*`・`g3/g3_on.*` | mp4・報告の json・ログ・時刻つきログ |
| `g3/compare_g3.py`・`g3/step_times.py`・`g3/compare.txt` | 前回との比較とステップの秒 |
| `console_r1080_277_fix.txt` | 4d の計測台の標準出力・標準エラー |
| `runs/r1080_277_fix/` | 4d の計測台の出力（`run.json`・`snapshot_step1.pickle`・`memory_stats_step1〜4.json`・`out.mp4` ほか） |
| `analysis/r1080_277_fix_step1.json`・`.md`・`analyze_console.txt` | `analyze_snapshot.py` の集計 |
| `analysis/judge_segments.py`・`judge_r1080_277_fix_step1.txt` | (i)〜(iv) の判定（json の `segments[]` から） |
| `analysis/memory_stats_summary.txt` | 各ステップの境目の `memory_stats` の要点 |
| `4e_new_name.txt`・`4e_old_name.txt` | 4e の出力 |
