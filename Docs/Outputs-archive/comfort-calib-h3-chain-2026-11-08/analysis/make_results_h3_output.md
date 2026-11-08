> **このファイルは実機の `outputs/comfort-calib-h3-chain-2026-11-08/analysis/make_results_h3_output.md`（git追跡外）のスナップショットである（2026-11-08複写）。** 正本は実機側にあり、実機側が更新された場合はこの複写も更新する。複写の目的は、git cloneした読者が参照を辿れるようにすること。

### MiniMax H3 の全計測点（判定 v3″）

| label | transformer | attn | geometry | K | steps | loras | status | v3'' | reasons | oom | oom vram16 | prefetch degr | video+cond | S total | text | cap MiB | s/step meta | s/step log | refs | vcr | S | S src | ref_video | ref_audio | oom at | oom stage | refenc s | refenc peak MB | refenc res MB | refenc commit GiB | enc peak MB | te tokens | denoise s | dit read s | decode s | den Δ MB | den n | dec Δ MB | dec n | enc med MB | enc n | den peak MB | den res MB | dec res MB | inact split MiB | alloc retries | commit peak GiB | used ok | baseline |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| warm | minimax_h3_ref2va_pruned_int8_convrot | sage | 640x384/56f | 0 | 4 | - | completed | smoke | - | False | False | False | 4080 | 4296 | 30 | 14025 | 1.962 | 1.725 | - | 4080 | 4296 | metadata | 0 | 0 | — | — | — | — | — | — | 1492 | 30 | 20.11 | 12.26 | 9.18 | — | 7 | — | 9 | 182.3 | 35 | 3776 | 3788 | 5370 | 107 | 0 | 30.69 | True | (smoke) |
| warm_rec1108030159 | minimax_h3_ref2va_pruned_int8_convrot | sage | 640x384/56f | 0 | 4 | - | completed | smoke | - | False | False | False | 4080 | 4296 | 30 | 14025 | 1.835 | 1.625 | - | 4080 | 4296 | metadata | 0 | 0 | — | — | — | — | — | — | 1492 | 30 | 17.53 | 10.19 | 8.71 | — | 6 | — | 9 | 149.5 | 27 | 3776 | 3788 | 5370 | 107 | 0 | 30.69 | True | (smoke) |
| b1080 | minimax_h3_ref2va_pruned_int8_convrot | sage | 1920x1088/56f | 1 | 4 | - | completed | baseline | - | False | False | False | 36720 | 38984 | 2078 | 13999 | 19.843 | 19.675 | - | 36720 | 38984 | metadata | 0 | 0 | — | — | — | — | — | — | 2687 | 2078 | 90.0 | 10.63 | 34.67 | — | 79 | — | 34 | 1168.8 | 15 | 4340 | 5866 | 10486 | 616 | 0 | 42.21 | True | (baseline) |
| D-K3 | minimax_h3_ref2va_pruned_int8_convrot | sage | 1920x1088/345f | 3 | 4 | - | completed | comfortable | - | False | False | False | 214200 | 219476 | 4126 | 13999 | 341.295 | 341.075 | - | 214200 | 219476 | metadata | 0 | 0 | — | — | — | — | — | — | 2731 | 4126 | 1376.72 | 11.54 | 206.07 | -40.5 | 1359 | -27.1 | 205 | 1177.7 | 26 | 12990 | 13970 | 10548 | 5056 | 9 | 42.21 | True | b1080 |
| D-K3_r2 | minimax_h3_ref2va_pruned_int8_convrot | sage | 1920x1088/345f | 3 | 4 | - | completed | comfortable | - | False | False | False | 214200 | 219476 | 4126 | 13999 | 341.277 | 341.05 | - | 214200 | 219476 | metadata | 0 | 0 | — | — | — | — | — | — | 2731 | 4126 | 1376.62 | 11.51 | 205.96 | -40.7 | 1360 | -35.6 | 205 | 1178.6 | 27 | 12990 | 13970 | 10548 | 5056 | 9 | 42.21 | True | b1080 |

### 行数の軸（vc・vcr・S。完了点は sequence の実数、OOM 点は worker の text 行＋式）

| label | status | v3″ | vc | vcr | S | S の出どころ | text | ref_video | ref_audio | audio | predict S | predict text | OOM の場所 | OOM の段 | 線に数える | cap MiB |
|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---|---|---:|
| warm | completed | smoke | 4,080 | 4,080 | 4,296 | metadata | 30 | 0 | 0 | 186 | 4,296 | 30 | — | — | — | 14,025 |
| warm_rec1108030159 | completed | smoke | 4,080 | 4,080 | 4,296 | metadata | 30 | 0 | 0 | 186 | 4,296 | 30 | — | — | — | 14,025 |
| b1080 | completed | baseline | 36,720 | 36,720 | 38,984 | metadata | 2,078 | 0 | 0 | 186 | 38,984 | 2,078 | — | — | — | 13,999 |
| D-K3 | completed | comfortable | 214,200 | 214,200 | 219,476 | metadata | 4,126 | 0 | 0 | 1,150 | 219,476 | 4,126 | — | — | — | 13,999 |
| D-K3_r2 | completed | comfortable | 214,200 | 214,200 | 219,476 | metadata | 4,126 | 0 | 0 | 1,150 | 219,476 | 4,126 | — | — | — | 13,999 |

### 参照の符号化の段と TE の段（記録だけ。§1-118 の材料）

| label | refenc 秒 | refenc 山 MB | refenc 予約 MB | refenc コミット GiB | 参照ごと (kind slot used_frames pairs rows 秒) | encode 秒 | encode 山 MB | encode コミット GiB | te tokens |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| warm | — | — | — | — | — | 34.6 | 1,492 | 5.20 | 30 |
| warm_rec1108030159 | — | — | — | — | — | 26.7 | 1,492 | 5.20 | 30 |
| b1080 | — | — | — | — | — | 12.8 | 2,687 | 42.21 | 2,078 |
| D-K3 | — | — | — | — | — | 20.5 | 2,731 | 42.21 | 4,126 |
| D-K3_r2 | — | — | — | — | — | 20.8 | 2,731 | 42.21 | 4,126 |

### MiniMax H3 の窓ごとの判定（基準比・+300MB が線。encode は記録だけ）

**`warm`** — v3″=`smoke`（基準 `—`・理由 —・照合 ok）

| 窓 | 判定に使う | 標本 | 中央値 MB | 基準 MB | 基準比 MB | 判定可 | 天井 |
|---|---|---:|---:|---:|---:|---|---|
| denoise | はい | 7 | 1211.9 | — | — | × fewer than 10 WDDM samples | — |
| decode | はい | 9 | 1211.1 | — | — | × fewer than 10 WDDM samples | — |
| encode | いいえ | 35 | 182.3 | — | — | × recorded only (plan §2.2) | — |

**`warm_rec1108030159`** — v3″=`smoke`（基準 `—`・理由 —・照合 ok）

| 窓 | 判定に使う | 標本 | 中央値 MB | 基準 MB | 基準比 MB | 判定可 | 天井 |
|---|---|---:|---:|---:|---:|---|---|
| denoise | はい | 6 | 1175.5 | — | — | × fewer than 10 WDDM samples | — |
| decode | はい | 9 | 1174.7 | — | — | × fewer than 10 WDDM samples | — |
| encode | いいえ | 27 | 149.5 | — | — | × recorded only (plan §2.2) | — |

**`b1080`** — v3″=`baseline`（基準 `—`・理由 —・照合 ok）

| 窓 | 判定に使う | 標本 | 中央値 MB | 基準 MB | 基準比 MB | 判定可 | 天井 |
|---|---|---:|---:|---:|---:|---|---|
| denoise | はい | 79 | 1176.9 | — | — | × baseline point | — |
| decode | はい | 34 | 1184.3 | — | — | × baseline point | — |
| encode | いいえ | 15 | 1168.8 | — | — | × recorded only (plan §2.2) | — |

**`D-K3`** — v3″=`comfortable`（基準 `b1080`・理由 —・照合 ok）

| 窓 | 判定に使う | 標本 | 中央値 MB | 基準 MB | 基準比 MB | 判定可 | 天井 |
|---|---|---:|---:|---:|---:|---|---|
| denoise | はい | 1,359 | 1136.4 | 1176.9 | -40.5 | ○ | 下 |
| decode | はい | 205 | 1157.2 | 1184.3 | -27.1 | ○ | 下 |
| encode | いいえ | 26 | 1177.7 | 1168.8 | 8.9 | × recorded only (plan §2.2) | — |

**`D-K3_r2`** — v3″=`comfortable`（基準 `b1080`・理由 —・照合 ok）

| 窓 | 判定に使う | 標本 | 中央値 MB | 基準 MB | 基準比 MB | 判定可 | 天井 |
|---|---|---:|---:|---:|---:|---|---|
| denoise | はい | 1,360 | 1136.2 | 1176.9 | -40.7 | ○ | 下 |
| decode | はい | 205 | 1148.7 | 1184.3 | -35.6 | ○ | 下 |
| encode | いいえ | 27 | 1178.6 | 1168.8 | 9.8 | × recorded only (plan §2.2) | — |

### MiniMax H3 の連結（Chained）の全計測点（判定 v3″ の連結版・クリップごと）

| label | clip | F | K | refs | S | S src | text | cap MiB | dit src | prefetch | s/step meta | s/step log | denoise s | den peak MB | den res MB | den Δ MB | den n | enc med MB | enc n | status | v3'' | reasons | causes | oom clip | oom stage | dec s | dec frames | dec res MB | dec Δ MB | dec n | commit peak GiB | used ok | baseline |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| smoke_c640 | 0 | 56 | 1 | - | 4784 | metadata | 278 | 13999 | file | on | 1.84 | 1.625 | 19.44 | 3816 | 3848 | — | 7 | 1257.6 | 33 | completed | smoke | - | - | — | — | 9.84 | 90 | 5370 | — | 9 | 32.15 | True | (smoke) |
|  | 1 | 56 | 0 | - | 4296 | metadata | 30 | 13999 | resident | on | 1.715 | 1.525 | 6.86 | 3776 | 3788 | — | 6 | 1191.0 | 21 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| cb1080 | 0 | 56 | 1 | - | 38984 | metadata | 2078 | 13999 | file | on | 20.125 | 19.925 | 91.5 | 4340 | 5866 | — | 79 | 1204.2 | 28 | completed | baseline | - | - | — | — | 55.41 | 90 | 10492 | — | 55 | 34.11 | True | (baseline) |
|  | 1 | 56 | 0 | - | 34896 | metadata | 30 | 13999 | resident | on | 17.797 | 17.65 | 71.19 | 4168 | 5624 | — | 71 | 1225.2 | 22 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| A1 | 1 | 362 | 0 | - | 219516 | metadata | 30 | 13999 | resident | on | 341.087 | 340.85 | 1364.35 | 13110 | 13876 | -82.3 | 1360 | 1138.8 | 22 | completed | comfortable | - | - | — | — | 401.99 | 685 | 10694 | -48.4 | 401 | 42.19 | True | cb1080 |
|  | 0 | 345 | 1 | - | 213348 | metadata | 2078 | 13999 | file | on | 336.168 | 335.95 | 1356.41 | 12669 | 13486 | -30.9 | 1340 | 1211.8 | 29 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| A-U | 0 | 362 | 1 | - | 223604 | worker_log | 2078 | — | — | — | — | 352.9 | — | — | — | -74.2 | 352 | 1180.0 | 26 | failed | oom | oom_vram16:clip0 | denoise | 0 | denoise | — | — | — | — | 0 | — | — | cb1080 |
|  | 1 | 39 | 0 | - | 24640 | estimate | 30 | — | — | — | — | — | — | — | — | — | 0 | — | 0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| cb1440 | 0 | 56 | 1 | - | 70144 | metadata | 3718 | 13999 | file | on | 48.388 | 48.175 | 205.12 | 5709 | 8214 | — | 192 | 1183.2 | 36 | completed | baseline | - | - | — | — | 92.4 | 90 | 14822 | — | 92 | 36.44 | True | (baseline) |
|  | 1 | 56 | 0 | - | 62776 | metadata | 30 | 13999 | resident | on | 40.595 | 40.4 | 162.38 | 5369 | 7834 | — | 161 | 1189.3 | 22 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| B1 | 0 | 192 | 1 | - | 217798 | metadata | 3718 | 13999 | file | on | 337.585 | 337.35 | 1361.26 | 12905 | 13762 | -38.3 | 1345 | 1173.3 | 32 | completed | comfortable | - | - | — | — | 372.01 | 362 | 15004 | 5.5 | 371 | 42.0 | True | cb1440 |
|  | 1 | 192 | 0 | - | 210430 | metadata | 30 | 13999 | resident | on | 316.86 | 316.675 | 1267.44 | 12662 | 13532 | -16.0 | 1263 | 1160.0 | 21 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| A1_r2 | 1 | 362 | 0 | - | 219516 | metadata | 30 | 13999 | resident | on | 341.723 | 341.5 | 1366.89 | 13110 | 13876 | -87.6 | 1362 | 1158.7 | 22 | completed | comfortable | - | - | — | — | 408.32 | 685 | 10694 | -64.5 | 408 | 42.21 | True | cb1080 |
|  | 0 | 345 | 1 | - | 213348 | metadata | 2078 | 13999 | file | on | 325.402 | 325.2 | 1312.29 | 12669 | 13486 | -65.2 | 1296 | 1168.1 | 27 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| B1_r2 | 0 | 192 | 1 | - | 217798 | metadata | 3718 | 13999 | file | on | 337.61 | 337.375 | 1360.99 | 12905 | 13762 | -55.3 | 1345 | 1170.2 | 31 | completed | comfortable | - | - | — | — | 371.45 | 362 | 15004 | 9.1 | 371 | 42.21 | True | cb1440 |
|  | 1 | 192 | 0 | - | 210430 | metadata | 30 | 13999 | resident | on | 316.837 | 316.625 | 1267.35 | 12662 | 13532 | -22.6 | 1264 | 1134.9 | 21 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| cb720 | 0 | 56 | 1 | - | 18464 | metadata | 998 | 13999 | file | on | 7.375 | 7.2 | 40.22 | 3932 | 4336 | — | 29 | 1169.5 | 24 | completed | baseline | - | - | — | — | 27.88 | 90 | 7496 | — | 28 | 42.21 | True | (baseline) |
|  | 1 | 56 | 0 | - | 16536 | metadata | 30 | 13999 | resident | on | 6.455 | 6.25 | 25.82 | 3776 | 4124 | — | 25 | 1168.5 | 20 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| C1 | 0 | 481 | 1 | - | 139882 | metadata | 998 | 13999 | file | on | 152.92 | 152.7 | 622.34 | 9080 | 11352 | -5.9 | 609 | 1170.6 | 23 | completed | comfortable | - | - | — | — | 263.88 | 940 | 7666 | -16.7 | 263 | 42.21 | True | cb720 |
|  | 1 | 481 | 0 | - | 137954 | metadata | 30 | 13999 | resident | on | 149.015 | 148.85 | 596.06 | 9076 | 11456 | -19.7 | 593 | 1160.2 | 22 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| E | 1 | 345 | 0 | v1 | 217167 | metadata | 15281 | 13999 | resident | on | 335.428 | 335.125 | 1341.71 | 13059 | 13516 | -20.2 | 1336 | 1160.9 | 306 | completed | comfortable | - | - | — | — | 188.7 | 668 | 7612 | -16.3 | 188 | 42.21 | True | cb720 |
|  | 0 | 345 | 1 | - | 101028 | metadata | 998 | 13999 | file | on | 87.62 | 87.4 | 361.86 | 7185 | 10872 | -5.5 | 349 | 1185.1 | 29 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| cb1080v | 0 | 56 | 0 | - | 34896 | metadata | 30 | 13999 | file | on | 17.17 | 16.975 | 79.61 | 4168 | 5624 | — | 68 | 1168.8 | 65 | completed | baseline | - | - | — | — | 54.45 | 68 | 10492 | — | 55 | 42.21 | True | (baseline) |
|  | 1 | 56 | 0 | - | 34896 | metadata | 30 | 13999 | resident | on | 17.135 | 16.975 | 68.54 | 4168 | 5624 | — | 68 | 1168.6 | 21 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| F | 0 | 362 | 0 | - | 219516 | metadata | 30 | 13999 | file | on | 341.522 | 341.3 | 1376.74 | 13110 | 13876 | -32.5 | 1362 | 1184.4 | 65 | completed | comfortable | - | - | — | — | 417.94 | 680 | 10698 | -6.4 | 417 | 42.27 | True | cb1080v |
|  | 1 | 362 | 0 | - | 219516 | metadata | 30 | 13999 | resident | on | 343.767 | 343.55 | 1375.07 | 13110 | 13876 | -29.3 | 1371 | 1135.3 | 22 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

### 連結の窓ごとの判定（基準点の同じ窓との比較・+300MB が線。clip<i>.encode は記録だけ）

**`smoke_c640`** — v3″=`smoke`（基準 `—`・理由 —・原因 —・OOM のクリップ —・照合 ok）

| 窓 | 判定に使う | 標本 | 中央値 MB | 基準 MB | 基準比 MB | 判定可 | 天井 |
|---|---|---:|---:|---:|---:|---|---|
| clip0.encode | いいえ | 33 | 1257.6 | — | — | × recorded only (plan §4.1) | — |
| clip0.denoise | はい | 7 | 1257.2 | — | — | × fewer than 10 WDDM samples | — |
| clip1.encode | いいえ | 21 | 1191.0 | — | — | × recorded only (plan §4.1) | — |
| clip1.denoise | はい | 6 | 1191.1 | — | — | × fewer than 10 WDDM samples | — |
| decode | はい | 9 | 1191.0 | — | — | × fewer than 10 WDDM samples | — |

**`cb1080`** — v3″=`baseline`（基準 `—`・理由 —・原因 —・OOM のクリップ —・照合 ok）

| 窓 | 判定に使う | 標本 | 中央値 MB | 基準 MB | 基準比 MB | 判定可 | 天井 |
|---|---|---:|---:|---:|---:|---|---|
| clip0.encode | いいえ | 28 | 1204.2 | — | — | × recorded only (plan §4.1) | — |
| clip0.denoise | はい | 79 | 1222.4 | — | — | × baseline point | — |
| clip1.encode | いいえ | 22 | 1225.2 | — | — | × recorded only (plan §4.1) | — |
| clip1.denoise | はい | 71 | 1222.7 | — | — | × baseline point | — |
| decode | はい | 55 | 1216.4 | — | — | × baseline point | — |

**`A1`** — v3″=`comfortable`（基準 `cb1080`・理由 —・原因 —・OOM のクリップ —・照合 ok）

| 窓 | 判定に使う | 標本 | 中央値 MB | 基準 MB | 基準比 MB | 判定可 | 天井 |
|---|---|---:|---:|---:|---:|---|---|
| clip0.encode | いいえ | 29 | 1211.8 | 1204.2 | 7.6 | × recorded only (plan §4.1) | — |
| clip0.denoise | はい | 1,340 | 1191.5 | 1222.4 | -30.9 | ○ | 下 |
| clip1.encode | いいえ | 22 | 1138.8 | 1225.2 | -86.4 | × recorded only (plan §4.1) | — |
| clip1.denoise | はい | 1,360 | 1140.4 | 1222.7 | -82.3 | ○ | 下 |
| decode | はい | 401 | 1168.0 | 1216.4 | -48.4 | ○ | 下 |

**`A-U`** — v3″=`oom`（基準 `cb1080`・理由 oom_vram16:clip0・原因 denoise・OOM のクリップ 0・照合 対象外）

| 窓 | 判定に使う | 標本 | 中央値 MB | 基準 MB | 基準比 MB | 判定可 | 天井 |
|---|---|---:|---:|---:|---:|---|---|
| clip0.encode | いいえ | 26 | 1180.0 | 1204.2 | -24.2 | × recorded only (plan §4.1) | — |
| clip0.denoise | はい | 352 | 1148.2 | 1222.4 | -74.2 | ○ | 下 |
| clip1.encode | いいえ | 0 | — | 1225.2 | — | × window not found | — |
| clip1.denoise | はい | 0 | — | 1222.7 | — | × window not found | — |
| decode | はい | 0 | — | 1216.4 | — | × window not found | — |

**`cb1440`** — v3″=`baseline`（基準 `—`・理由 —・原因 —・OOM のクリップ —・照合 ok）

| 窓 | 判定に使う | 標本 | 中央値 MB | 基準 MB | 基準比 MB | 判定可 | 天井 |
|---|---|---:|---:|---:|---:|---|---|
| clip0.encode | いいえ | 36 | 1183.2 | — | — | × recorded only (plan §4.1) | — |
| clip0.denoise | はい | 192 | 1191.6 | — | — | × baseline point | — |
| clip1.encode | いいえ | 22 | 1189.3 | — | — | × recorded only (plan §4.1) | — |
| clip1.denoise | はい | 161 | 1179.2 | — | — | × baseline point | — |
| decode | はい | 92 | 1256.0 | — | — | × baseline point | — |

**`B1`** — v3″=`comfortable`（基準 `cb1440`・理由 —・原因 —・OOM のクリップ —・照合 ok）

| 窓 | 判定に使う | 標本 | 中央値 MB | 基準 MB | 基準比 MB | 判定可 | 天井 |
|---|---|---:|---:|---:|---:|---|---|
| clip0.encode | いいえ | 32 | 1173.3 | 1183.2 | -9.9 | × recorded only (plan §4.1) | — |
| clip0.denoise | はい | 1,345 | 1153.3 | 1191.6 | -38.3 | ○ | 下 |
| clip1.encode | いいえ | 21 | 1160.0 | 1189.3 | -29.3 | × recorded only (plan §4.1) | — |
| clip1.denoise | はい | 1,263 | 1163.2 | 1179.2 | -16.0 | ○ | 下 |
| decode | はい | 371 | 1261.5 | 1256.0 | 5.5 | ○ | 下 |

**`A1_r2`** — v3″=`comfortable`（基準 `cb1080`・理由 —・原因 —・OOM のクリップ —・照合 ok）

| 窓 | 判定に使う | 標本 | 中央値 MB | 基準 MB | 基準比 MB | 判定可 | 天井 |
|---|---|---:|---:|---:|---:|---|---|
| clip0.encode | いいえ | 27 | 1168.1 | 1204.2 | -36.1 | × recorded only (plan §4.1) | — |
| clip0.denoise | はい | 1,296 | 1157.2 | 1222.4 | -65.2 | ○ | 下 |
| clip1.encode | いいえ | 22 | 1158.7 | 1225.2 | -66.5 | × recorded only (plan §4.1) | — |
| clip1.denoise | はい | 1,362 | 1135.1 | 1222.7 | -87.6 | ○ | 下 |
| decode | はい | 408 | 1151.9 | 1216.4 | -64.5 | ○ | 下 |

**`B1_r2`** — v3″=`comfortable`（基準 `cb1440`・理由 —・原因 —・OOM のクリップ —・照合 ok）

| 窓 | 判定に使う | 標本 | 中央値 MB | 基準 MB | 基準比 MB | 判定可 | 天井 |
|---|---|---:|---:|---:|---:|---|---|
| clip0.encode | いいえ | 31 | 1170.2 | 1183.2 | -13.0 | × recorded only (plan §4.1) | — |
| clip0.denoise | はい | 1,345 | 1136.3 | 1191.6 | -55.3 | ○ | 下 |
| clip1.encode | いいえ | 21 | 1134.9 | 1189.3 | -54.4 | × recorded only (plan §4.1) | — |
| clip1.denoise | はい | 1,264 | 1156.6 | 1179.2 | -22.6 | ○ | 下 |
| decode | はい | 371 | 1265.1 | 1256.0 | 9.1 | ○ | 下 |

**`cb720`** — v3″=`baseline`（基準 `—`・理由 —・原因 —・OOM のクリップ —・照合 ok）

| 窓 | 判定に使う | 標本 | 中央値 MB | 基準 MB | 基準比 MB | 判定可 | 天井 |
|---|---|---:|---:|---:|---:|---|---|
| clip0.encode | いいえ | 24 | 1169.5 | — | — | × recorded only (plan §4.1) | — |
| clip0.denoise | はい | 29 | 1168.5 | — | — | × baseline point | — |
| clip1.encode | いいえ | 20 | 1168.5 | — | — | × recorded only (plan §4.1) | — |
| clip1.denoise | はい | 25 | 1176.9 | — | — | × baseline point | — |
| decode | はい | 28 | 1176.9 | — | — | × baseline point | — |

**`C1`** — v3″=`comfortable`（基準 `cb720`・理由 —・原因 —・OOM のクリップ —・照合 ok）

| 窓 | 判定に使う | 標本 | 中央値 MB | 基準 MB | 基準比 MB | 判定可 | 天井 |
|---|---|---:|---:|---:|---:|---|---|
| clip0.encode | いいえ | 23 | 1170.6 | 1169.5 | 1.1 | × recorded only (plan §4.1) | — |
| clip0.denoise | はい | 609 | 1162.6 | 1168.5 | -5.9 | ○ | 下 |
| clip1.encode | いいえ | 22 | 1160.2 | 1168.5 | -8.3 | × recorded only (plan §4.1) | — |
| clip1.denoise | はい | 593 | 1157.2 | 1176.9 | -19.7 | ○ | 下 |
| decode | はい | 263 | 1160.2 | 1176.9 | -16.7 | ○ | 下 |

**`E`** — v3″=`comfortable`（基準 `cb720`・理由 —・原因 —・OOM のクリップ —・照合 ok）

| 窓 | 判定に使う | 標本 | 中央値 MB | 基準 MB | 基準比 MB | 判定可 | 天井 |
|---|---|---:|---:|---:|---:|---|---|
| clip0.encode | いいえ | 29 | 1185.1 | 1169.5 | 15.6 | × recorded only (plan §4.1) | — |
| clip0.denoise | はい | 349 | 1163.0 | 1168.5 | -5.5 | ○ | 下 |
| clip1.encode | いいえ | 306 | 1160.9 | 1168.5 | -7.6 | × recorded only (plan §4.1) | — |
| clip1.denoise | はい | 1,336 | 1156.7 | 1176.9 | -20.2 | ○ | 下 |
| decode | はい | 188 | 1160.6 | 1176.9 | -16.3 | ○ | 下 |

**`cb1080v`** — v3″=`baseline`（基準 `—`・理由 —・原因 —・OOM のクリップ —・照合 ok）

| 窓 | 判定に使う | 標本 | 中央値 MB | 基準 MB | 基準比 MB | 判定可 | 天井 |
|---|---|---:|---:|---:|---:|---|---|
| clip0.encode | いいえ | 65 | 1168.8 | — | — | × recorded only (plan §4.1) | — |
| clip0.denoise | はい | 68 | 1168.5 | — | — | × baseline point | — |
| clip1.encode | いいえ | 21 | 1168.6 | — | — | × recorded only (plan §4.1) | — |
| clip1.denoise | はい | 68 | 1168.5 | — | — | × baseline point | — |
| decode | はい | 55 | 1168.5 | — | — | × baseline point | — |

**`F`** — v3″=`comfortable`（基準 `cb1080v`・理由 —・原因 —・OOM のクリップ —・照合 ok）

| 窓 | 判定に使う | 標本 | 中央値 MB | 基準 MB | 基準比 MB | 判定可 | 天井 |
|---|---|---:|---:|---:|---:|---|---|
| clip0.encode | いいえ | 65 | 1184.4 | 1168.8 | 15.6 | × recorded only (plan §4.1) | — |
| clip0.denoise | はい | 1,362 | 1136.0 | 1168.5 | -32.5 | ○ | 下 |
| clip1.encode | いいえ | 22 | 1135.3 | 1168.6 | -33.3 | × recorded only (plan §4.1) | — |
| clip1.denoise | はい | 1,371 | 1139.2 | 1168.5 | -29.3 | ○ | 下 |
| decode | はい | 417 | 1162.1 | 1168.5 | -6.4 | ○ | 下 |

