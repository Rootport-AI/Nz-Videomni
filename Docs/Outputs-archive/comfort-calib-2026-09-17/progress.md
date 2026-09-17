> **このファイルは実機の `outputs/comfort-calib-2026-09-17/progress.md`（git追跡外）のスナップショットである（2026-09-17複写）。** 正本は実機側にあり、実機側が更新された場合はこの複写も更新する。複写の目的は、git cloneした読者が参照を辿れるようにすること。

# progress.md — 第5次（2026-09-17・Sulphur-2 Q6_K）P1〜P2 実行ログ

計測担当（Opus）の作業日誌です。**結論は書きません**（一次記録の正本は
`RESULTS_comfort-calib-2026-09-17.md`、規約の正本は 2026-09-04 の README 第5〜7節）。
今回の担当範囲は **P1（着手前検査・読み込み・待機時の床）と P2（Tier A の 10 点）だけ**です。
Tier B 以降は親が判断します。

---

## P1 着手前検査（H2）— 2026-09-17 16:43〜16:45

| # | 検査 | 結果 |
|---|---|---|
| 1a | `nvidia-smi --query-compute-apps` の python 計算プロセス | **1 件のみ**: PID 39164 `.python\cpython-3.12.9-windows-x86_64-none\python.exe -u -m engine.worker`。親 PID 30932（`.venv-engine` の `engine.worker`）→ その親が **PID 27300**（オーナーのサーバー）。他の python は無し |
| 1b | `nvidia-smi --query-gpu=memory.used` | **1,045 MiB / 16,376 MiB**（待機値 約 1,050 MiB と一致） |
| 2 | `GET /api/v1/status` | **200**・`state: ready`・`queue.pending=0`・`queue.running=0`（`completed=27`・`failed=0`）・`base_model: LTX23`・`pipeline_loaded: true` |
| 3 | `GET /api/v1/models` の LTX23 transformer 一覧 | `default`(=公式 Q4_K_M)・`10Eros-v1.2-Q4_K_M`・**`Sulphur-2-base-distil-Q4_K_M`**・**`Sulphur-2-base-distil-Q6_K`** —— 4 件すべて `exists: true`。**稼働中の選択（active）= `Sulphur-2-base-distil-Q4_K_M`**（`active_base_model: LTX23`） |
| 4 | `git -C <repo> status --short` | **空**（HEAD `e3151fc`・ブランチ `dev`） |
| 5 | `calib.py server start`（`--pid` 無し） | `server already listening on 18620; not starting a second one` → `GET /status -> 200`・rc=0。**`server.pid` は作られていない**。`original_state.json` は P0 のまま（16:30:21・`record_original_state` は既存なら即 return）。`netstat -ano \| findstr :18620` は実行前後とも **LISTENING は PID 27300 の 1 行だけ** |

- `original_state.json` の中身: `active_base_model: LTX23`／`selections.LTX23.transformer: Sulphur-2-base-distil-Q4_K_M`／`selections.LTX25.transformer: redgraftLTX25Fast2K_ltx25RedgraftNSFW-Q6_K`。

### 再走の仕組みについての事前確認（副命令 9 の前提検査）

`calib.py` の `cmd_run` には **`manifest.jsonl` を見て完了済みラベルを飛ばす経路がありません**
（`grep -n "def taken" calib.py` は 0 件。`taken()` は `mkplan.py` 92 行の関数で、
「実行ディレクトリが既にあるラベルを新しい計画ファイルに書かない」ためのもの）。
`cmd_run` が飛ばすのは **base_model 違い**（2116〜2121 行）と **transformer 違い**（2130〜2138 行）の 2 つだけです。
したがって**同じ計画ファイルを再実行すると完了済みの点も全部走り直します**。
副命令 9 の但し書きどおり、**点が失敗した場合は再走せず停止して親に報告します**。

## P1 読み込みと待機時の床

- **16:45:35〜16:45:40**　`calib.py load --base-model LTX23 --transformer Sulphur-2-base-distil-Q6_K`
  → **4.7 秒**（`real 4.892s`）で `200`・`state=ready`・rc=0。
  二重照合も一致: 読み込み応答の `models.transformer` = `Sulphur-2-base-distil-Q6_K`、
  `GET /models` の `active` = `Sulphur-2-base-distil-Q6_K`、要求 = 同じ。
  ワーカーは作り直されました（読み込み前 PID 30932/39164 → 読み込み後 **PID 4788/40348**・
  どちらも生成時刻 16:45:36・親は PID 27300）。直後の GPU 使用量は 990 MiB
  （前のワーカーの分が解放され、新しいワーカーはまだ重みを載せていない状態）。
  ※第4次の読み込みは 65.8 秒でしたが、それは LTX 2.5 への切替でした。今回は同じ LTX 2.3 の中での
  変換器の付け替えなので、ワーカー再生成だけで済んでいます（重みの読み込みは最初のジョブまで遅延）。

- **16:45:56〜16:49:01**　`calib.py idle-calib --minutes 3`（rc=0）。
  LUID は `luid_0x00000000_0x0000f07d_phys_0`（第4次の `…0x0000f18b…` とは別）。
  出力先 `calib/idle_LTX23_Sulphur_2_base_distil_Q6_K/`、`calib/calibration.json` を作成。

| 量 | 中央値 | 95%点 | 最大 |
|---|---|---|---|
| 専有GPUメモリ | 990.9 MB | 991.2 MB | 994.8 MB |
| 共有GPUメモリ | **137.0 MB** | 161.4 MB | 162.4 MB |
| PCIe 受信 | 0.0 | 2.0 | 382.0 |
| SM 使用率 | 0.0 | 3.0 | — |

台地の判定に使う共有メモリの上限 `shared_plateau_limit_mb` = **461.4 MB**。
WDDM 標本 182 本（間隔の中央値 1,003 ミリ秒）・dmon 標本 175 本。

---

## P2 Tier A ①　`plans/plan_A1_single_720_1080_q6.json`（7 点）

コマンド（キャンペーンフォルダから実行・冷却は既定の 150 秒）:

```
.venv\Scripts\python.exe calib.py run --plan plans\plan_A1_single_720_1080_q6.json --job-timeout 2400
```

**16:49:39 開始 → 17:22:46 終了（33 分 07 秒）・rc=0・7 点すべて `status: completed`。**
実行直前の `GET /status` は `state: ready`・`queue.pending=0`・`queue.running=0`。

| ラベル | 幾何 | トークン | 秒 | 除去窓 共有中央値 / 基準（差） | 台地 | 復元窓 共有中央値 / 基準（差） | 台地 | **規則v3（2窓）** | `verdict_v3prime`（5窓複合） |
|---|---|---|---|---|---|---|---|---|---|
| q6_s_720_b185 | 1280x768/185f | 23,040 | 136.4 | 1174.6 / 基準点 | — | 1174.3 / 基準点 | — | **基準点** | baseline |
| q6_s_720_361 | 1280x768/361f | 44,160 | 164.5 | 2472.7 / 1174.6 (**+1298.1**) | **Y** | 1128.9 / 1174.3 (−45.4) | N | **台地** | plateau |
| q6_s_720_369 | 1280x768/369f | 45,120 | 160.5 | 1286.3 / 1174.6 (+111.7) | N | 1142.7 / 1174.3 (−31.6) | N | **快適** | clean |
| q6_s_720_377 | 1280x768/377f | 46,080 | 162.6 | 1346.2 / 1174.6 (+171.6) | N | 1153.9 / 1174.3 (−20.4) | N | **快適** | clean |
| q6_s_1080_b89 | 1920x1088/89f | 24,480 | 84.3 | 1178.8 / 基準点 | — | 1179.3 / 基準点 | — | **基準点** | baseline |
| q6_s_1080_169 | 1920x1088/169f | 44,880 | 158.6 | 1129.7 / 1178.8 (−49.1) | N | 1125.8 / 1179.3 (−53.5) | N | **快適** | clean |
| q6_s_1080_177 | 1920x1088/177f | 46,920 | 186.5 | 1380.5 / 1178.8 (+201.7) | N | 1146.0 / 1179.3 (−33.3) | N | **快適** | clean |

補助 3 窓（記録のみ・線の決定には使わない）と専有ピーク・ジョブピーク:

| ラベル | s23_stage1 Δ | s23_gap_mid Δ | s23_gap_tail Δ | WDDM専有ピーク ジョブ/除去/復号 (MB) | `peak_vram` (MB) | 予約ピーク (MB) | 変換器 |
|---|---|---|---|---|---|---|---|
| q6_s_720_b185 | 1158.1（基準） | 1174.6（基準） | 1174.3（基準） | 11788.7 / 11788.4 / 11788.4 | 8,685 | 10,960 | Sulphur-2-base-distil-Q6_K |
| q6_s_720_361 | +29.7 | +13.1 | −45.4 | 15660.2 / 15660.2 / 15655.3 | 13,106 | **16,142** | 同上 |
| q6_s_720_369 | +6.3 | +117.8 | −31.6 | 15701.6 / 15700.9 / 9938.9 | 13,304 | 15,094 | 同上 |
| q6_s_720_377 | +16.6 | +171.6 | −20.4 | 15622.3 / 15622.3 / 9701.2 | 13,503 | 15,350 | 同上 |
| q6_s_1080_b89 | 1180.5（基準） | 1180.5（基準） | 1179.3（基準） | 11934.4 / 11932.1 / 10470.0 | 8,988 | 11,428 | 同上 |
| q6_s_1080_169 | −3.7 | +4.8 | −53.5 | 15522.5 / 15522.5 / 11086.3 | 13,239 | 15,004 | 同上 |
| q6_s_1080_177 | −16.8 | +226.5 | −33.3 | 15889.5 / 15889.5 / 14577.7 | 13,666 | 15,620 | 同上 |

- 専有ピークは **WDDM のカード全体の値**で、デスクトップぶんを含みます（README 第4節）。`peak_vram` は
  `server_slice.log` の `Job … peak_vram=…MB`（＝`metadata.json` の `vram_optimization.peak_vram_mb`）と一致しました。
- 全 7 点で `used_matches_request.ok = true`・不一致 0 件。`metadata.json` の
  `models.selection.transformer.name` は 7 点とも **`Sulphur-2-base-distil-Q6_K`**、`models.base_model` は `LTX23`。
- 加速のこだま: `attention_used=sage`・`block_swap_prefetch_used=on`・`keep_resident_used=on`・
  `fused_gguf_dequant_kernel_used=on`・**`vae_mode_used=on`**（送った値は `vae_mode: prune_vaed`。
  LTX 2.3 のこだまは `on`／`off` の語彙で、計測台の期待値も `on`）・
  **`keep_resident_embeddings_used=null`**（送った値は `false`。LTX 2.3 はこの項目を `null` で返す。
  計測台の期待は `"off"` だが不一致には数えられず `ok=true`）。
- `server_slice.log`（各点 14 行）に `Traceback`・`out of memory`・`OOM`・`ERROR`・`WARNING`・`fail` は **0 件**。
- 失敗・待ち時間切れ・422・`undecidable`（基準点を除く）は **ゼロ**。再走は行っていません。

**逆行の観測（事実のみ）**: 1280×768 の腕は 44,160（361 コマ）が台地、その上の 45,120（369 コマ）と
46,080（377 コマ）が快適です。台地が出た点は除去窓の共有中央値が 2,472.7 MB（基準比 +1,298.1 MB）で、
予約ピークも 7 点中最大の 16,142 MB でした。計画のゲート H3 が言う「逆行なし」は成立していません。

---

## P2 Tier A ②　`plans/plan_A2_single_p1280_q6.json`（3 点）

計画ファイルの境目には冷却が入らないので、**手で 150 秒空けました**（17:23:11〜17:25:41）。
その直前（17:23:05）の `GET /status` は `state: ready`・`queue.pending=0`・`queue.running=0`、
`GET /models` の `active` は `Sulphur-2-base-distil-Q6_K`。

```
.venv\Scripts\python.exe calib.py run --plan plans\plan_A2_single_p1280_q6.json --job-timeout 2400
```

**17:25:41 開始 → 17:37:29 終了（11 分 48 秒）・rc=0・3 点すべて `status: completed`。**

| ラベル | 幾何 | トークン | 秒 | 除去窓 共有中央値 / 基準（差） | 台地 | 復元窓 共有中央値 / 基準（差） | 台地 | **規則v3（2窓）** | `verdict_v3prime` |
|---|---|---|---|---|---|---|---|---|---|
| q6_s_p1280_b185 | 768x1280/185f | 23,040 | 78.3 | 1189.6 / 基準点 | — | 1165.5 / 基準点 | — | **基準点** | baseline |
| q6_s_p1280_361 | 768x1280/361f | 44,160 | 156.5 | 1858.2 / 1189.6 (**+668.6**) | **Y** | 1125.8 / 1165.5 (−39.7) | N | **台地** | plateau |
| q6_s_p1280_369 | 768x1280/369f | 45,120 | 158.5 | 1154.2 / 1189.6 (−35.4) | N | 1154.0 / 1165.5 (−11.5) | N | **快適** | clean |

| ラベル | s23_stage1 Δ | s23_gap_mid Δ | s23_gap_tail Δ | WDDM専有ピーク ジョブ/除去/復号 (MB) | `peak_vram` (MB) | 予約ピーク (MB) | 変換器 |
|---|---|---|---|---|---|---|---|
| q6_s_p1280_b185 | 1189.6（基準） | 1189.6（基準） | 1165.5（基準） | 11413.1 / 11413.1 / 9466.9 | 8,688 | 10,896 | Sulphur-2-base-distil-Q6_K |
| q6_s_p1280_361 | −27.5 | −27.3 | −39.7 | 15960.5 / 15960.5 / 9715.0 | 13,106 | **16,142** | 同上 |
| q6_s_p1280_369 | −0.5 | −63.8 | −11.5 | 15615.4 / 15615.4 / 9721.5 | 13,304 | 15,094 | 同上 |

- 3 点とも `used_matches_request.ok = true`・不一致 0 件、`metadata.json` の
  `models.selection.transformer.name` は `Sulphur-2-base-distil-Q6_K`・`models.base_model` は `LTX23`。
  こだまは A1 と同じ（`sage`／`on`／`on`／`on`／`vae_mode_used=on`／`keep_resident_embeddings_used=null`）。
- `server_slice.log` に `Traceback`・`out of memory`・`OOM`・`ERROR`・`WARNING`・`fail` は **0 件**。

**README 第3節の予告との違い（事実のみ）**: 「`q6_s_p1280_369` は規則 v3 では台地になる見込み」と
書かれていましたが、実測は除去窓 **−35.4 MB**・復元窓 **−11.5 MB** で**台地ではありません**。
代わりに 1 段下の `q6_s_p1280_361`（44,160）が台地になりました。

---

## Tier A のまとめ（10 点・機械抽出の値のみ）

- **10 点すべて `status: completed`**（`manifest.jsonl` の 10 レコードとも `completed`）。
  基準点 3 点は `verdict_v3prime = baseline`（基準が無いので 5 窓とも `decidable: false`・
  理由は `no baseline for this window`）。残り 7 点は **5 窓すべて `decidable: true`**で、
  `undecidable` はゼロです。
- **規則 v3（除去窓・復元窓の 2 窓）と 5 窓複合 `verdict_v3prime` は 10 点すべてで一致**しました。
  台地に立ったのは**どの点でも除去窓だけ**で、復元窓・`s23_stage1`・`s23_gap_mid`・`s23_gap_tail` は
  7 点すべてで +300 MB 未満です（最大でも `s23_gap_mid` の +226.5 MB＝`q6_s_1080_177`）。

| 腕（幾何） | 快適側の最大（台地なし） | 最初の台地 | 備考 |
|---|---|---|---|
| 単発 1280×768 | **46,080**（377 コマ・測った最上段） | **44,160**（361 コマ） | 逆行あり。44,160 が台地で、その上の 45,120・46,080 が快適 |
| 単発 1920×1088 | **46,920**（177 コマ・測った最上段） | **なし** | 測った 2 点とも快適（44,880 は −49.1 MB、46,920 は +201.7 MB） |
| 単発 768×1280 | **45,120**（369 コマ・測った最上段） | **44,160**（361 コマ） | 逆行あり。1280×768 と同じ形 |

- 「快適側の最大」はいずれも**梯子の最上段**であって、上限が見つかったという意味ではありません。
- 2 つの台地点（`q6_s_720_361` と `q6_s_p1280_361`）は**同じ 44,160 トークン・同じ 46 潜在フレーム・
  同じ復号チャンク数 9** で、`peak_vram`（13,106 MB）も予約ピーク（16,142 MB）も**同じ値**でした。
  どちらも各計画ファイルの**2 点目**（基準点の直後）です。以上は観測した事実で、解釈はしません。

## 実行したコマンド（時刻順・すべてキャンペーンフォルダから）

```
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
git -C S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni status --short
netstat -ano | findstr :18620
.venv\Scripts\python.exe calib.py server start
.venv\Scripts\python.exe calib.py load --base-model LTX23 --transformer Sulphur-2-base-distil-Q6_K
.venv\Scripts\python.exe calib.py idle-calib --minutes 3
.venv\Scripts\python.exe calib.py run --plan plans\plan_A1_single_720_1080_q6.json --job-timeout 2400
（手動で 150 秒の冷却）
.venv\Scripts\python.exe calib.py run --plan plans\plan_A2_single_p1280_q6.json --job-timeout 2400
.venv\Scripts\python.exe batch_table.py plans\plan_A1_single_720_1080_q6.json > tables\plan_A1_single_720_1080_q6.txt
.venv\Scripts\python.exe batch_table.py plans\plan_A2_single_p1280_q6.json  > tables\plan_A2_single_p1280_q6.txt
.venv\Scripts\python.exe calib.py report > report_output.txt
```

（`.venv\Scripts\python.exe` は `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\.venv\Scripts\python.exe` の略です。
`--pid` は一度も付けていません。`server stop`・`restore-state`・`reanalyse` は打っていません。）

## Tier A 終了時の状態（17:38）

| 項目 | 値 |
|---|---|
| `GET /status` | `state: ready`・`queue.pending=0`・`queue.running=0`・`completed=37`・`failed=0`・`base_model: LTX23` |
| `GET /models` | `active` = **`Sulphur-2-base-distil-Q6_K`**（Tier B のために載せたまま） |
| ワーカー | PID 4788 / 40348（16:45:36 生成・親 27300）が生存 |
| GPU 使用量 | 540〜544 MiB（開始時の待機値 1,045 MiB・`idle-calib` の専有中央値 990.9 MB より低い） |
| `git status --short`（Nz-Videomni） | **空**（HEAD `e3151fc`） |
| `server.pid` | **作られていません** |

生成物: `report_output.txt`（10 行）・`tables\plan_A1_single_720_1080_q6.txt`・
`tables\plan_A2_single_p1280_q6.txt`・`runs\<ラベル>\`（10 個）・`manifest.jsonl`（10 行）・
`calib\calibration.json`・`calib\idle_LTX23_Sulphur_2_base_distil_Q6_K\`・`run_A1.log`・`run_A2.log`。

**Tier B 以降は親の判断待ちです。担当は点を増やしていません。**

---

# 追記 — Tier B 以降（監督役のセッションが直接実行）

**ここから下は、監督役のセッションが較正台を直接叩いた記録です。** 計測担当のエージェントの側で計画ファイルを作る命令が道具の許可判定に繰り返し拒まれたため、監督役が Tier A の点の雛形から計画ファイルを起こし、そのまま `calib.py run` を実行しました。**較正台・点の設計・判定の規則は変えていません。** 結論はここには書きません（正本は `RESULTS_comfort-calib-2026-09-17.md`）。

実行そのものの記録は `run_B_D1.log`・`run_B2_swap_C.log`・`run_C2.log` です。下の表の時刻と戻り値はすべてその3本から取っています。

## 実行の一覧（時刻順）

| # | 段 | 計画ファイル | 点 | 開始 | 終了 | 戻り値 | ログ |
|---|---|---|---:|---|---|---|---|
| 1 | Tier B | `plans\plan_B_single_q6.json` | 8 | 18:22:55 | 19:05:08 | `rc_B=0` | `run_B_D1.log` |
| 2 | Tier D-1 | `plans\plan_D1_chain_q6.json` | 3 | 19:05:08 | 19:16:58 | `rc_D1=0` | `run_B_D1.log` |
| 3 | Tier B2 | `plans\plan_B2_single_q6.json` | 4 | 19:17:26 | 19:38:18 | `rc_B2=0` | `run_B2_swap_C.log` |
| 4 | 変換器の切り替え | （`load`） | — | 19:38:18 | 19:38:26 | `rc_load=0` | `run_B2_swap_C.log` |
| 5 | 待機時の床の取り直し | （`idle-calib --minutes 3`） | — | 19:38:26 | 19:41:29 | `rc_idle=0` | `run_B2_swap_C.log` |
| 6 | Tier C | `plans\plan_C_single_q4.json` | 9 | 19:41:29 | 20:23:42 | `rc_C=0` | `run_B2_swap_C.log` |
| 7 | Tier C2 | `plans\plan_C2_single_q4.json` | 3 | 20:23:52 | 20:36:55 | `rc_C2=0` | `run_C2.log` |

**合計27点。すべて `status: completed`・戻り値はすべて 0・再走した点はありません。**

## 各段で走らせた点

| 段 | ラベル |
|---|---|
| Tier B（8点） | `q6_s_720_377_r2`・`q6_s_720_361_r2`・`q6_s_720_385`・`q6_s_1080_185`・`q6_s_1080_177_r2`・`q6_s_p1280_377`・`q6_s_p1280_361_r2`・`q6_s_p1280_369_r2` |
| Tier D-1（3点） | `q6_c_base`（基準点）・`q6_c_1856x1024`・`q6_c_1920x1088` |
| Tier B2（4点） | `q6_s_1080_169_r2`・`q6_s_720_369_r2`・`q6_s_720_385_r2`・`q6_s_1080_185_r2` |
| Tier C（9点） | `q4_s_720_b185`（基準点）・`q4_s_720_361`・`q4_s_720_369`・`q4_s_1080_b89`（基準点）・`q4_s_1080_169`・`q4_s_1080_177`・`q4_s_p1280_b185`（基準点）・`q4_s_p1280_361`・`q4_s_p1280_369` |
| Tier C2（3点） | `q4_s_1080_169_r2`・`q4_s_720_361_r2`・`q4_s_p1280_361_r2` |

- Tier A の終わり（17:37）から Tier B の開始（18:22:55）まで約45分空いているのは、Tier A の結果を読んで Tier B の計画ファイルを起こしていたためです（断片は `plans\fragments_B\`）。
- **Tier B2 は Tier B の続きです。** Tier B を組んだ時点では Tier A の 1920×1088・169コマの再走と 1280×768・369コマの再走が入っておらず、2回一致を揃えるために足しました。
- **Tier D-2（Q4_K_M の連結）は実行していません。** Tier D-1 の連結2点がどちらも台地にならなかったため、対照を取る相手がありませんでした。

## 変換器の切り替え（1回きり）

```
POST /pipeline/load base_model=LTX23 transformer=Sulphur-2-base-distil-Q4_K_M
-> 200 state=ready base_model=LTX23 pipeline_loaded=true   （8.0 秒）
transformer check: 読み込み応答='Sulphur-2-base-distil-Q4_K_M'
                   GET /models の active='Sulphur-2-base-distil-Q4_K_M'
                   要求='Sulphur-2-base-distil-Q4_K_M'
```

続けて `idle-calib --minutes 3` を実行し、`calib\idle_LTX23_Sulphur_2_base_distil_Q4_K_M\` と `calib\calibration.json` を作りました。書き出された待機時の床は専有の中央値 564.5 MB・共有の中央値 133.0 MB・`shared_plateau_limit_mb` 441.9 MB（WDDM 標本182本・dmon 標本175本）です。**切り替えのあとに `reanalyse` は打っていません。**

## 計画ファイルの境目の冷却（逸脱として記録）

**`cmd_run` の冷却150秒は1つの計画ファイルの中の2点目以降にだけ効きます**（README 第5節の7番）。Tier A の2本の境目では手で150秒空けましたが、**それ以降の境目は次のとおり短いまま続けています。**

| 境目 | 空いた時間 |
|---|---|
| Tier B → Tier D-1 | 0秒（同じ実行の中で続けて呼んだため） |
| Tier D-1 → Tier B2 | 約28秒 |
| Tier B2 → 切り替え → Tier C | 切り替えと `idle-calib` で約3分11秒 |
| Tier C → Tier C2 | 約10秒 |

**観測した事実**（解釈はしません）: 境目の直後に走った点は `q6_c_base`（Tier D-1 の基準点・除去窓の共有中央値 1,172.8 MB）・`q6_s_1080_169_r2`（Tier B2 の1点目・基準比 −22.8）・`q4_s_1080_169_r2`（Tier C2 の1点目・基準比 −27.2）の3点です。`q6_c_base` の値は、150秒の冷却を置いて測った他の Q6_K の基準点3点（1,174.6／1,178.8／1,189.6 MB）と同じ帯に入っています。

## 後片づけ（20:36〜20:37）

| 項目 | 結果 |
|---|---|
| 変換器の選択 | `state.json` の `selections.LTX23.transformer` が `Sulphur-2-base-distil-Q4_K_M`（開始時の控え `original_state.json` と一致）。`selections.LTX25.transformer` は前のキャンペーンの選択のまま無傷。**`restore-state` の実行そのものの出力は `restore_state_output.txt` にあります**——実行時に標準出力をファイルへ流していなかったため、監督役のセッションの記録からの逐語の写しです。切り替えが1回きりで最後の段が Q4_K_M だったため、選択は自然に元へ戻る順序になっています |
| GPU の占有 | **638 メビバイト**（`git_state_end.txt`。開始時の待機値は 1,043 メビバイト） |
| 待ち受け | ポート 18620 は PID 27300 のまま LISTENING。サーバーは止めていません |
| `git status --short`（Nz-Videomni） | **空**（HEAD `e3151fc`） |
| `server.pid` | 最後まで作られていません |

## 生成物（20:36〜20:44）

```
.venv\Scripts\python.exe batch_table.py plans\<各計画> > tables\<同名>.txt   （7本）
.venv\Scripts\python.exe calib.py report > report_output.txt                （37行＋見出し2行）
.venv\Scripts\python.exe make_results.py > make_results_output.md
.venv\Scripts\python.exe calib.py preflight …                              （全7計画・37点・失敗0）
```

`preflight_output.txt` はキャンペーン後に全7計画ぶんを取り直したものです（着手前の記録は Tier A と連結の3計画・13点でした）。
