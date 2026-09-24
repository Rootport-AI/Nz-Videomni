> **このファイルは実機の `outputs/comfort-calib-2026-09-25/make_results_output.md`（git追跡外）のスナップショットである（2026-09-25複写）。** 正本は実機側にあり、実機側が更新された場合はこの複写も更新する。複写の目的は、git cloneした読者が参照を辿れるようにすること。

### 全計測点

| ラベル | 系統 | 幾何 | トークン | 状態 | v1 | v2 | **v3′** | 加速照合 |
|---|---|---|---:|---|---|---|---|---|
| `a_b1216` | LTX23 | 1216x704/722f | 18,392 | completed | organic | baseline | **baseline** | ok |
| `a_w46` | LTX23 | 1216x704/722f | 38,456 | completed | organic | organic | **plateau** | ok |
| `a_w46_r2` | LTX23 | 1216x704/722f | 38,456 | completed | organic | organic | **plateau** | ok |
| `a_b1152` | LTX23 | 1152x576/722f | 14,256 | completed | organic | baseline | **baseline** | ok |
| `a_w61` | LTX23 | 1152x576/722f | 39,528 | completed | organic | organic | **clean** | ok |
| `a_w61_r2` | LTX23 | 1152x576/722f | 39,528 | completed | organic | organic | **clean** | ok |
| `a_w40` | LTX23 | 1216x704/722f | 33,440 | completed | organic | organic | **clean** | ok |
| `a_w43` | LTX23 | 1216x704/722f | 35,948 | completed | organic | organic | **clean** | ok |
| `a_b1088` | LTX23 | 1088x576/722f | 13,464 | completed | organic | baseline | **baseline** | ok |
| `a_w46lo` | LTX23 | 1088x576/722f | 28,152 | completed | organic | organic | **clean** | ok |
| `a_b1280` | LTX23 | 1280x768/722f | 21,120 | completed | organic | baseline | **baseline** | ok |
| `a_o46` | LTX23 | 1280x768/722f | 44,160 | completed | organic | organic | **clean** | ok |
| `a4_b1216` | LTX23 | 1216x704/722f | 18,392 | completed | organic | baseline | **baseline** | ok |
| `a4_w46` | LTX23 | 1216x704/722f | 38,456 | completed | organic | organic | **clean** | ok |
| `b_b1280` | LTX25 | 1280x768/722f | 21,120 | completed | organic | baseline | **baseline** | ok |
| `b_w46` | LTX25 | 1280x768/722f | 44,160 | completed | organic | organic | **clean** | ok |
| `b_w46_r2` | LTX25 | 1280x768/722f | 44,160 | completed | organic | organic | **clean** | ok |
| `b_b1152` | LTX25 | 1152x640/722f | 15,840 | completed | organic | baseline | **baseline** | ok |
| `b_w61` | LTX25 | 1152x640/722f | 43,920 | completed | organic | organic | **clean** | ok |
| `b_w61_r2` | LTX25 | 1152x640/722f | 43,920 | completed | organic | organic | **clean** | ok |
| `b_o52` | LTX25 | 1280x768/722f | 49,920 | completed | organic | organic | **plateau** | ok |

### 加速が実際に効いたか（`metadata.json` の `*_used`）

| ラベル | transformer | attention | block_swap_prefetch | keep_resident | fused_gguf_dequant_kernel | vae_mode | keep_resident_embeddings | ok |
|---|---|---|---|---|---|---|---|---|
| `a_b1216` | `Sulphur-2-base-distil-Q6_K` | `sage` | `on` | `on` | `on` | `on` | *null* | ok |
| `a_w46` | `Sulphur-2-base-distil-Q6_K` | `sage` | `on` | `on` | `on` | `on` | *null* | ok |
| `a_w46_r2` | `Sulphur-2-base-distil-Q6_K` | `sage` | `on` | `on` | `on` | `on` | *null* | ok |
| `a_b1152` | `Sulphur-2-base-distil-Q6_K` | `sage` | `on` | `on` | `on` | `on` | *null* | ok |
| `a_w61` | `Sulphur-2-base-distil-Q6_K` | `sage` | `on` | `on` | `on` | `on` | *null* | ok |
| `a_w61_r2` | `Sulphur-2-base-distil-Q6_K` | `sage` | `on` | `on` | `on` | `on` | *null* | ok |
| `a_w40` | `Sulphur-2-base-distil-Q6_K` | `sage` | `on` | `on` | `on` | `on` | *null* | ok |
| `a_w43` | `Sulphur-2-base-distil-Q6_K` | `sage` | `on` | `on` | `on` | `on` | *null* | ok |
| `a_b1088` | `Sulphur-2-base-distil-Q6_K` | `sage` | `on` | `on` | `on` | `on` | *null* | ok |
| `a_w46lo` | `Sulphur-2-base-distil-Q6_K` | `sage` | `on` | `on` | `on` | `on` | *null* | ok |
| `a_b1280` | `Sulphur-2-base-distil-Q6_K` | `sage` | `on` | `on` | `on` | `on` | *null* | ok |
| `a_o46` | `Sulphur-2-base-distil-Q6_K` | `sage` | `on` | `on` | `on` | `on` | *null* | ok |
| `a4_b1216` | `Sulphur-2-base-distil-Q4_K_M` | `sage` | `on` | `on` | `on` | `on` | *null* | ok |
| `a4_w46` | `Sulphur-2-base-distil-Q4_K_M` | `sage` | `on` | `on` | `on` | `on` | *null* | ok |
| `b_b1280` | `default` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `b_w46` | `default` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `b_w46_r2` | `default` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `b_b1152` | `default` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `b_w61` | `default` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `b_w61_r2` | `default` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `b_o52` | `default` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |

`keep_resident_embeddings` が 2.3 の点で *null* なのは仕様です（`services/engines/ltx/adapter.py:481`。2.3 に埋め込み処理器が無い）。

### v3′ の窓ごとの判定（基準比・+300MB が線）

**`a_b1216`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 66 | × no baseline for this window | — | — |
| s23_gap_mid | 10 | × no baseline for this window | — | — |
| s23_gap_tail | 66 | × no baseline for this window | — | — |
| s23_stage1 | 67 | × no baseline for this window | — | — |
| stage2 | 139 | × no baseline for this window | — | — |

**`a_w46`** — v3′=`plateau`（基準 `a_b1216`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 65 | ○ | -0.1 | 下 |
| s23_gap_mid | 43 | ○ | 975.9 | **超** |
| s23_gap_tail | 65 | ○ | -0.1 | 下 |
| s23_stage1 | 67 | ○ | 20.9 | 下 |
| stage2 | 214 | ○ | 945.3 | **超** |

**`a_w46_r2`** — v3′=`plateau`（基準 `a_b1216`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 68 | ○ | -1.9 | 下 |
| s23_gap_mid | 43 | ○ | 938.6 | **超** |
| s23_gap_tail | 68 | ○ | -1.9 | 下 |
| s23_stage1 | 66 | ○ | 23.9 | 下 |
| stage2 | 212 | ○ | 914.8 | **超** |

**`a_b1152`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 56 | × no baseline for this window | — | — |
| s23_gap_mid | 7 | × fewer than 10 WDDM samples | — | — |
| s23_gap_tail | 56 | × no baseline for this window | — | — |
| s23_stage1 | 53 | × no baseline for this window | — | — |
| stage2 | 89 | × no baseline for this window | — | — |

**`a_w61`** — v3′=`clean`（基準 `a_b1152`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 55 | ○ | -0.1 | 下 |
| s23_gap_mid | 22 | ○ | 17.5 | 下 |
| s23_gap_tail | 55 | ○ | -0.1 | 下 |
| s23_stage1 | 52 | ○ | 16.6 | 下 |
| stage2 | 71 | ○ | 16.4 | 下 |

**`a_w61_r2`** — v3′=`clean`（基準 `a_b1152`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 54 | ○ | 0.2 | 下 |
| s23_gap_mid | 22 | ○ | -7.9 | 下 |
| s23_gap_tail | 54 | ○ | 0.2 | 下 |
| s23_stage1 | 52 | ○ | 8.6 | 下 |
| stage2 | 71 | ○ | -0.2 | 下 |

**`a_w40`** — v3′=`clean`（基準 `a_b1216`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 63 | ○ | -16.1 | 下 |
| s23_gap_mid | 18 | ○ | -7.5 | 下 |
| s23_gap_tail | 63 | ○ | -16.1 | 下 |
| s23_stage1 | 67 | ○ | 23.2 | 下 |
| stage2 | 105 | ○ | -6.8 | 下 |

**`a_w43`** — v3′=`clean`（基準 `a_b1216`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 63 | ○ | 1.0 | 下 |
| s23_gap_mid | 20 | ○ | 16.9 | 下 |
| s23_gap_tail | 63 | ○ | 1.0 | 下 |
| s23_stage1 | 67 | ○ | 7.0 | 下 |
| stage2 | 105 | ○ | -14.7 | 下 |

**`a_b1088`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 52 | × no baseline for this window | — | — |
| s23_gap_mid | 7 | × fewer than 10 WDDM samples | — | — |
| s23_gap_tail | 52 | × no baseline for this window | — | — |
| s23_stage1 | 48 | × no baseline for this window | — | — |
| stage2 | 83 | × no baseline for this window | — | — |

**`a_w46lo`** — v3′=`clean`（基準 `a_b1088`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 52 | ○ | -0.2 | 下 |
| s23_gap_mid | 15 | ○ | -8.1 | 下 |
| s23_gap_tail | 52 | ○ | -0.2 | 下 |
| s23_stage1 | 49 | ○ | -15.7 | 下 |
| stage2 | 75 | ○ | -15.8 | 下 |

**`a_b1280`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 68 | × no baseline for this window | — | — |
| s23_gap_mid | 11 | × no baseline for this window | — | — |
| s23_gap_tail | 68 | × no baseline for this window | — | — |
| s23_stage1 | 77 | × no baseline for this window | — | — |
| stage2 | 133 | × no baseline for this window | — | — |

**`a_o46`** — v3′=`clean`（基準 `a_b1280`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 68 | ○ | 4.1 | 下 |
| s23_gap_mid | 26 | ○ | 185.7 | 下 |
| s23_gap_tail | 68 | ○ | 4.1 | 下 |
| s23_stage1 | 77 | ○ | -16.6 | 下 |
| stage2 | 133 | ○ | 201.2 | 下 |

**`a4_b1216`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 64 | × no baseline for this window | — | — |
| s23_gap_mid | 10 | × no baseline for this window | — | — |
| s23_gap_tail | 64 | × no baseline for this window | — | — |
| s23_stage1 | 67 | × no baseline for this window | — | — |
| stage2 | 114 | × no baseline for this window | — | — |

**`a4_w46`** — v3′=`clean`（基準 `a4_b1216`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 62 | ○ | 24.8 | 下 |
| s23_gap_mid | 22 | ○ | -6.8 | 下 |
| s23_gap_tail | 62 | ○ | 24.8 | 下 |
| s23_stage1 | 67 | ○ | -0.5 | 下 |
| stage2 | 106 | ○ | 8.3 | 下 |

**`b_b1280`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 54 | × no baseline for this window | — | — |
| restore | 54 | × no baseline for this window | — | — |
| stage2 | 139 | × no baseline for this window | — | — |

**`b_w46`** — v3′=`clean`（基準 `b_b1280`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 53 | ○ | -49.4 | 下 |
| restore | 53 | ○ | -49.4 | 下 |
| stage2 | 132 | ○ | -49.5 | 下 |

**`b_w46_r2`** — v3′=`clean`（基準 `b_b1280`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 53 | ○ | -24.9 | 下 |
| restore | 53 | ○ | -24.9 | 下 |
| stage2 | 132 | ○ | -32.9 | 下 |

**`b_b1152`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 41 | × no baseline for this window | — | — |
| restore | 41 | × no baseline for this window | — | — |
| stage2 | 104 | × no baseline for this window | — | — |

**`b_w61`** — v3′=`clean`（基準 `b_b1152`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 41 | ○ | -16.4 | 下 |
| restore | 41 | ○ | -16.4 | 下 |
| stage2 | 84 | ○ | -33.6 | 下 |

**`b_w61_r2`** — v3′=`clean`（基準 `b_b1152`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 40 | ○ | -2.4 | 下 |
| restore | 40 | ○ | -2.4 | 下 |
| stage2 | 84 | ○ | -33.2 | 下 |

**`b_o52`** — v3′=`plateau`（基準 `b_b1280`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 52 | ○ | -24.4 | 下 |
| restore | 52 | ○ | -24.4 | 下 |
| stage2 | 226 | ○ | 1551.3 | **超** |

### 速度と共有メモリ

| ラベル | 生成秒 | stage2秒 | 秒/トークン | 復元秒 | 予約ピークMB | 共有ピーク(job)MB | 参照符号化 予約MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| `a_b1216` | 347.6 | 130.9 | 0.007120 | 66.1 | 11,610 | 1157.9 | — |
| `a_w46` | 407.4 | 255.8 | 0.006650 | 65.0 | 15,564 | 2362.5 | — |
| `a_w46_r2` | 399.4 | 253.7 | 0.006596 | 68.2 | 15,564 | 2331.5 | — |
| `a_b1152` | 213.3 | 94.8 | 0.006650 | 56.3 | 9,996 | 1233.8 | — |
| `a_w61` | 207.9 | 92.4 | 0.002338 | 54.6 | 13,468 | 1241.5 | — |
| `a_w61_r2` | 207.4 | 92.2 | 0.002334 | 54.4 | 13,468 | 1159.0 | — |
| `a_w40` | 262.0 | 122.8 | 0.003674 | 62.8 | 13,862 | 1175.4 | — |
| `a_w43` | 263.9 | 124.8 | 0.003472 | 62.3 | 14,710 | 1175.0 | — |
| `a_b1088` | 198.1 | 88.7 | 0.006584 | 52.5 | 9,676 | 1174.8 | — |
| `a_w46lo` | 198.2 | 88.5 | 0.003144 | 52.6 | 11,964 | 1159.0 | — |
| `a_b1280` | 298.2 | 142.7 | 0.006754 | 68.1 | 12,962 | 1175.5 | — |
| `a_o46` | 313.3 | 159.9 | 0.003621 | 68.6 | 16,222 | 2493.5 | — |
| `a4_b1216` | 311.0 | 123.2 | 0.006696 | 63.5 | 11,598 | 645.6 | — |
| `a4_w46` | 269.7 | 127.0 | 0.003304 | 62.4 | 13,106 | 661.2 | — |
| `b_b1280` | 322.0 | 146.6 | 0.006939 | 53.8 | 8,576 | 1163.9 | — |
| `b_w46` | 300.1 | 154.1 | 0.003488 | 52.7 | 14,612 | 1122.2 | — |
| `b_w46_r2` | 300.1 | 154.1 | 0.003488 | 52.9 | 14,612 | 1122.3 | — |
| `b_b1152` | 222.0 | 108.6 | 0.006856 | 40.9 | 6,906 | 1147.6 | — |
| `b_w61` | 218.4 | 107.5 | 0.002449 | 40.9 | 14,552 | 1131.0 | — |
| `b_w61_r2` | 218.6 | 107.5 | 0.002449 | 40.8 | 14,522 | 1122.9 | — |
| `b_o52` | 444.9 | 298.4 | 0.005977 | 52.8 | 16,566 | 2244.7 | — |

### 所要時間（`manifest.jsonl` の実測）

| ラベル | 開始 | 終了 | 秒 |
|---|---|---|---:|
| `a_b1216` | 2026-09-25 00:36:11.000 | 2026-09-25 00:41:59.863 | 348.9 |
| `a_w46` | 2026-09-25 00:44:34.198 | 2026-09-25 00:51:22.851 | 408.6 |
| `a_w46_r2` | 2026-09-25 00:53:57.379 | 2026-09-25 01:00:38.112 | 400.7 |
| `a_b1152` | 2026-09-25 01:03:13.054 | 2026-09-25 01:06:47.479 | 214.4 |
| `a_w61` | 2026-09-25 01:09:22.073 | 2026-09-25 01:12:50.557 | 208.5 |
| `a_w61_r2` | 2026-09-25 01:15:24.813 | 2026-09-25 01:18:53.422 | 208.6 |
| `a_w40` | 2026-09-25 01:22:01.050 | 2026-09-25 01:26:23.882 | 262.8 |
| `a_w43` | 2026-09-25 01:28:58.921 | 2026-09-25 01:33:23.684 | 264.8 |
| `a_b1088` | 2026-09-25 01:35:58.792 | 2026-09-25 01:39:17.356 | 198.6 |
| `a_w46lo` | 2026-09-25 01:41:51.569 | 2026-09-25 01:45:10.171 | 198.6 |
| `a_b1280` | 2026-09-25 01:48:57.824 | 2026-09-25 01:53:56.706 | 298.9 |
| `a_o46` | 2026-09-25 01:56:31.722 | 2026-09-25 02:01:46.610 | 314.9 |
| `a4_b1216` | 2026-09-25 02:06:30.438 | 2026-09-25 02:11:43.205 | 312.8 |
| `a4_w46` | 2026-09-25 02:14:18.402 | 2026-09-25 02:18:49.110 | 270.7 |
| `b_b1280` | 2026-09-25 02:23:09.022 | 2026-09-25 02:28:31.993 | 323.0 |
| `b_w46` | 2026-09-25 02:31:07.001 | 2026-09-25 02:36:07.596 | 300.6 |
| `b_w46_r2` | 2026-09-25 02:38:41.937 | 2026-09-25 02:43:42.641 | 300.7 |
| `b_b1152` | 2026-09-25 02:46:16.859 | 2026-09-25 02:49:59.516 | 222.7 |
| `b_w61` | 2026-09-25 02:52:33.693 | 2026-09-25 02:56:12.369 | 218.7 |
| `b_w61_r2` | 2026-09-25 02:58:47.484 | 2026-09-25 03:02:28.174 | 220.7 |
| `b_o52` | 2026-09-25 03:05:03.563 | 2026-09-25 03:12:28.825 | 445.3 |

- 計測点の実行時間の合計: **99.1 分**
- 最初の点の開始から最後の点の終了まで（冷却込み）: **156.3 分**
- 実行した点の数（再走を含む延べ）: **21**

