> **このファイルは実機の `outputs/comfort-calib-2026-09-26/make_results_output.md`（git追跡外）のスナップショットである（2026-09-26複写）。** 正本は実機側にあり、実機側が更新された場合はこの複写も更新する。複写の目的は、git cloneした読者が参照を辿れるようにすること。

### 全計測点

| ラベル | 系統 | 幾何 | トークン | 状態 | v1 | v2 | **v3′** | 加速照合 |
|---|---|---|---:|---|---|---|---|---|
| `f_warm` | LTX23 | 768x512/121f | 6,144 | completed | organic | baseline | **baseline** | ok |
| `f_b1216` | LTX23 | 1216x704/722f | 18,392 | completed | organic | baseline | **baseline** | ok |
| `f_w46` | LTX23 | 1216x704/722f | 38,456 | completed | organic | organic | **plateau** | ok |
| `f_s_1080_b89` | LTX23 | 1920x1088/89f | 24,480 | completed | organic | baseline | **baseline** | ok |
| `f_s_1080_161` | LTX23 | 1920x1088/161f | 42,840 | completed | harmful | harmful | **plateau** | ok |
| `f_s_1080_b89_d` | LTX23 | 1920x1088/89f | 24,480 | completed | organic | baseline | **baseline** | ok |
| `f_s_1080_161_d` | LTX23 | 1920x1088/161f | 42,840 | completed | harmful | harmful | **plateau** | ok |
| `g_warm` | LTX25 | 768x512/121f | 6,144 | completed | organic | baseline | **baseline** | ok |
| `g_s_1080_b89` | LTX25 | 1920x1088/89f | 24,480 | completed | organic | baseline | **baseline** | ok |
| `g_s_1080_161` | LTX25 | 1920x1088/161f | 42,840 | completed | harmful | harmful | **plateau** | ok |
| `g_s_1080_169` | LTX25 | 1920x1088/169f | 44,880 | completed | harmful | harmful | **plateau** | ok |
| `g_s_720_b169` | LTX25 | 1280x768/169f | 21,120 | completed | organic | baseline | **baseline** | ok |
| `g_s_720_361` | LTX25 | 1280x768/361f | 44,160 | completed | harmful | harmful | **plateau** | ok |
| `g_s_720_369` | LTX25 | 1280x768/369f | 45,120 | completed | harmful | harmful | **plateau** | ok |
| `g_s_896_b161` | LTX25 | 896x1152/161f | 21,168 | completed | organic | baseline | **baseline** | ok |
| `g_s_896_345` | LTX25 | 896x1152/345f | 44,352 | completed | harmful | harmful | **plateau** | ok |
| `g_s_896_353` | LTX25 | 896x1152/353f | 45,360 | completed | harmful | harmful | **plateau** | ok |
| `g_c_b1280x768` | LTX25 | 1280x768/186f | 21,120 | completed | organic | baseline | **baseline** | ok |
| `g_c_1856x1024` | LTX25 | 1856x1024/186f | 40,832 | completed | harmful | harmful | **plateau** | ok |
| `g_c_1920x1088` | LTX25 | 1920x1088/186f | 44,880 | completed | harmful | harmful | **plateau** | ok |
| `g_s_1080_153` | LTX25 | 1920x1088/153f | 40,800 | completed | harmful | harmful | **plateau** | ok |
| `g_s_1080_145` | LTX25 | 1920x1088/145f | 38,760 | completed | harmful | organic | **clean** | ok |
| `g_s_720_353` | LTX25 | 1280x768/353f | 43,200 | completed | harmful | harmful | **plateau** | ok |
| `g_s_720_345` | LTX25 | 1280x768/345f | 42,240 | completed | harmful | harmful | **plateau** | ok |
| `g_s_720_329` | LTX25 | 1280x768/329f | 40,320 | completed | harmful | organic | **clean** | ok |
| `g_s_896_329` | LTX25 | 896x1152/329f | 42,336 | completed | harmful | harmful | **plateau** | ok |
| `g_s_896_313` | LTX25 | 896x1152/313f | 40,320 | completed | harmful | harmful | **plateau** | ok |
| `g_s_896_297` | LTX25 | 896x1152/297f | 38,304 | completed | organic | organic | **clean** | ok |
| `g_c_1792x1024` | LTX25 | 1792x1024/186f | 39,424 | completed | harmful | organic | **clean** | ok |
| `g_c_1728x1024` | LTX25 | 1728x1024/186f | 38,016 | completed | harmful | organic | **clean** | ok |
| `g_c_1664x960` | LTX25 | 1664x960/186f | 34,320 | completed | harmful | organic | **clean** | ok |
| `g_s_1080_145_r2` | LTX25 | 1920x1088/145f | 38,760 | completed | harmful | organic | **clean** | ok |
| `g_s_1080_153_r2` | LTX25 | 1920x1088/153f | 40,800 | completed | harmful | harmful | **plateau** | ok |
| `g_s_720_329_r2` | LTX25 | 1280x768/329f | 40,320 | completed | harmful | harmful | **plateau** | ok |
| `g_s_720_313` | LTX25 | 1280x768/313f | 38,400 | completed | harmful | organic | **clean** | ok |
| `g_s_896_297_r2` | LTX25 | 896x1152/297f | 38,304 | completed | organic | organic | **clean** | ok |
| `g_s_896_313_r2` | LTX25 | 896x1152/313f | 40,320 | completed | harmful | harmful | **plateau** | ok |
| `g_c_1792x1024_r2` | LTX25 | 1792x1024/186f | 39,424 | completed | harmful | organic | **clean** | ok |
| `g_c_1856x1024_r2` | LTX25 | 1856x1024/186f | 40,832 | completed | harmful | harmful | **plateau** | ok |
| `g_s_720_313_r2` | LTX25 | 1280x768/313f | 38,400 | completed | harmful | organic | **clean** | **NG {"transformer_used": {"requested": "ltx-2.5-22b-distilled-transformer-fp8_e4m3fn", "used": "default"}}** |
| `b_warm` | LTX25 | 768x512/121f | 6,144 | completed | clean | baseline | **baseline** | ok |
| `b_s_1080_b89` | LTX25 | 1920x1088/89f | 24,480 | completed | clean | baseline | **baseline** | ok |
| `b_s_1080_161` | LTX25 | 1920x1088/161f | 42,840 | completed | clean | clean | **clean** | ok |
| `b_s_1080_169` | LTX25 | 1920x1088/169f | 44,880 | completed | clean | clean | **clean** | ok |
| `a_warm` | LTX23 | 768x512/121f | 6,144 | completed | organic | baseline | **baseline** | ok |
| `a_s_1080_b89` | LTX23 | 1920x1088/89f | 24,480 | completed | organic | baseline | **baseline** | ok |
| `a_s_1080_161` | LTX23 | 1920x1088/161f | 42,840 | completed | harmful | organic | **clean** | ok |
| `a_s_1080_169` | LTX23 | 1920x1088/169f | 44,880 | completed | harmful | organic | **clean** | ok |
| `f_s_1080_153_d` | LTX23 | 1920x1088/153f | 40,800 | completed | harmful | harmful | **plateau** | ok |
| `f_s_1080_145_d` | LTX23 | 1920x1088/145f | 38,760 | completed | harmful | harmful | **plateau** | ok |
| `f_s_1080_137_d` | LTX23 | 1920x1088/137f | 36,720 | completed | inconclusive | harmful | **plateau** | ok |
| `g_warm2` | LTX25 | 768x512/121f | 6,144 | completed | organic | baseline | **baseline** | ok |
| `g_s_720_313_r2b` | LTX25 | 1280x768/313f | 38,400 | completed | harmful | organic | **clean** | ok |
| `f_s_1080_121_d` | LTX23 | 1920x1088/121f | 32,640 | completed | inconclusive | organic | **clean** | ok |
| `f_s_1080_105_d` | LTX23 | 1920x1088/105f | 28,560 | completed | inconclusive | organic | **clean** | ok |
| `f_s_1080_129_d` | LTX23 | 1920x1088/129f | 34,680 | completed | organic | organic | **plateau** | ok |
| `f_s_1080_121_d_r2` | LTX23 | 1920x1088/121f | 32,640 | completed | inconclusive | organic | **clean** | ok |

### 加速が実際に効いたか（`metadata.json` の `*_used`）

| ラベル | transformer | attention | block_swap_prefetch | keep_resident | fused_gguf_dequant_kernel | vae_mode | keep_resident_embeddings | ok |
|---|---|---|---|---|---|---|---|---|
| `f_warm` | `sulphur_distil_fp8mixed` | `sage` | `on` | `on` | `on` | `on` | *null* | ok |
| `f_b1216` | `sulphur_distil_fp8mixed` | `sage` | `on` | `on` | `on` | `on` | *null* | ok |
| `f_w46` | `sulphur_distil_fp8mixed` | `sage` | `on` | `on` | `on` | `on` | *null* | ok |
| `f_s_1080_b89` | `sulphur_distil_fp8mixed` | `sage` | `on` | `on` | `on` | `on` | *null* | ok |
| `f_s_1080_161` | `sulphur_distil_fp8mixed` | `sage` | `on` | `on` | `on` | `on` | *null* | ok |
| `f_s_1080_b89_d` | `sulphur_distil_fp8mixed` | `sdpa` | `on` | `off` | `on` | `off` | *null* | ok |
| `f_s_1080_161_d` | `sulphur_distil_fp8mixed` | `sdpa` | `on` | `off` | `on` | `off` | *null* | ok |
| `g_warm` | `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `g_s_1080_b89` | `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `g_s_1080_161` | `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `g_s_1080_169` | `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `g_s_720_b169` | `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `g_s_720_361` | `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `g_s_720_369` | `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `g_s_896_b161` | `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `g_s_896_345` | `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `g_s_896_353` | `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `g_c_b1280x768` | `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `g_c_1856x1024` | `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `g_c_1920x1088` | `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `g_s_1080_153` | `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `g_s_1080_145` | `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `g_s_720_353` | `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `g_s_720_345` | `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `g_s_720_329` | `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `g_s_896_329` | `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `g_s_896_313` | `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `g_s_896_297` | `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `g_c_1792x1024` | `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `g_c_1728x1024` | `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `g_c_1664x960` | `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `g_s_1080_145_r2` | `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `g_s_1080_153_r2` | `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `g_s_720_329_r2` | `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `g_s_720_313` | `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `g_s_896_297_r2` | `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `g_s_896_313_r2` | `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `g_c_1792x1024_r2` | `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `g_c_1856x1024_r2` | `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `g_s_720_313_r2` | `default` | `sage` | `on` | `on` | `on` | `conv` | `on` | **NG** |
| `b_warm` | `default` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `b_s_1080_b89` | `default` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `b_s_1080_161` | `default` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `b_s_1080_169` | `default` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `a_warm` | `Sulphur-2-base-distil-Q6_K` | `sage` | `on` | `on` | `on` | `on` | *null* | ok |
| `a_s_1080_b89` | `Sulphur-2-base-distil-Q6_K` | `sage` | `on` | `on` | `on` | `on` | *null* | ok |
| `a_s_1080_161` | `Sulphur-2-base-distil-Q6_K` | `sage` | `on` | `on` | `on` | `on` | *null* | ok |
| `a_s_1080_169` | `Sulphur-2-base-distil-Q6_K` | `sage` | `on` | `on` | `on` | `on` | *null* | ok |
| `f_s_1080_153_d` | `sulphur_distil_fp8mixed` | `sdpa` | `on` | `off` | `on` | `off` | *null* | ok |
| `f_s_1080_145_d` | `sulphur_distil_fp8mixed` | `sdpa` | `on` | `off` | `on` | `off` | *null* | ok |
| `f_s_1080_137_d` | `sulphur_distil_fp8mixed` | `sdpa` | `on` | `off` | `on` | `off` | *null* | ok |
| `g_warm2` | `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `g_s_720_313_r2b` | `ltx-2.5-22b-distilled-transformer-fp8_e4m3fn` | `sage` | `on` | `on` | `on` | `conv` | `on` | ok |
| `f_s_1080_121_d` | `sulphur_distil_fp8mixed` | `sdpa` | `on` | `off` | `on` | `off` | *null* | ok |
| `f_s_1080_105_d` | `sulphur_distil_fp8mixed` | `sdpa` | `on` | `off` | `on` | `off` | *null* | ok |
| `f_s_1080_129_d` | `sulphur_distil_fp8mixed` | `sdpa` | `on` | `off` | `on` | `off` | *null* | ok |
| `f_s_1080_121_d_r2` | `sulphur_distil_fp8mixed` | `sdpa` | `on` | `off` | `on` | `off` | *null* | ok |

`keep_resident_embeddings` が 2.3 の点で *null* なのは仕様です（`services/engines/ltx/adapter.py:481`。2.3 に埋め込み処理器が無い）。

### v3′ の窓ごとの判定（基準比・+300MB が線）

**`f_warm`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 6 | × fewer than 10 WDDM samples | — | — |
| s23_gap_mid | 4 | × fewer than 10 WDDM samples | — | — |
| s23_gap_tail | 6 | × fewer than 10 WDDM samples | — | — |
| s23_stage1 | 10 | × no baseline for this window | — | — |
| stage2 | 7 | × fewer than 10 WDDM samples | — | — |

**`f_b1216`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 64 | × no baseline for this window | — | — |
| s23_gap_mid | 10 | × no baseline for this window | — | — |
| s23_gap_tail | 64 | × no baseline for this window | — | — |
| s23_stage1 | 74 | × no baseline for this window | — | — |
| stage2 | 121 | × no baseline for this window | — | — |

**`f_w46`** — v3′=`plateau`（基準 `f_b1216`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 65 | ○ | -18.7 | 下 |
| s23_gap_mid | 43 | ○ | 872.5 | **超** |
| s23_gap_tail | 65 | ○ | -18.7 | 下 |
| s23_stage1 | 74 | ○ | 13.4 | 下 |
| stage2 | 220 | ○ | 2126.7 | **超** |

**`f_s_1080_b89`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 18 | × no baseline for this window | — | — |
| s23_gap_mid | 13 | × no baseline for this window | — | — |
| s23_gap_tail | 18 | × no baseline for this window | — | — |
| s23_stage1 | 23 | × no baseline for this window | — | — |
| stage2 | 25 | × no baseline for this window | — | — |

**`f_s_1080_161`** — v3′=`plateau`（基準 `f_s_1080_b89`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 36 | ○ | -14.2 | 下 |
| s23_gap_mid | 68 | ○ | 2538.1 | **超** |
| s23_gap_tail | 36 | ○ | -14.2 | 下 |
| s23_stage1 | 37 | ○ | -16.3 | 下 |
| stage2 | 277 | ○ | 3864.8 | **超** |

**`f_s_1080_b89_d`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 27 | × no baseline for this window | — | — |
| s23_gap_mid | 20 | × no baseline for this window | — | — |
| s23_gap_tail | 27 | × no baseline for this window | — | — |
| s23_stage1 | 26 | × no baseline for this window | — | — |
| stage2 | 39 | × no baseline for this window | — | — |

**`f_s_1080_161_d`** — v3′=`plateau`（基準 `f_s_1080_b89_d`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 57 | ○ | -42.5 | 下 |
| s23_gap_mid | 86 | ○ | 2233.4 | **超** |
| s23_gap_tail | 57 | ○ | -42.5 | 下 |
| s23_stage1 | 47 | ○ | -3.2 | 下 |
| stage2 | 147 | ○ | 3529.5 | **超** |

**`g_warm`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 3 | × fewer than 10 WDDM samples | — | — |
| restore | 3 | × fewer than 10 WDDM samples | — | — |
| stage2 | 6 | × fewer than 10 WDDM samples | — | — |

**`g_s_1080_b89`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 13 | × no baseline for this window | — | — |
| restore | 13 | × no baseline for this window | — | — |
| stage2 | 24 | × no baseline for this window | — | — |

**`g_s_1080_161`** — v3′=`plateau`（基準 `g_s_1080_b89`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 24 | ○ | -17.9 | 下 |
| restore | 24 | ○ | -17.9 | 下 |
| stage2 | 111 | ○ | 920.2 | **超** |

**`g_s_1080_169`** — v3′=`plateau`（基準 `g_s_1080_b89`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 25 | ○ | -8.9 | 下 |
| restore | 25 | ○ | -8.9 | 下 |
| stage2 | 145 | ○ | 1574.7 | **超** |

**`g_s_720_b169`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 12 | × no baseline for this window | — | — |
| restore | 12 | × no baseline for this window | — | — |
| stage2 | 21 | × no baseline for this window | — | — |

**`g_s_720_361`** — v3′=`plateau`（基準 `g_s_720_b169`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 26 | ○ | -31.6 | 下 |
| restore | 26 | ○ | -31.6 | 下 |
| stage2 | 139 | ○ | 1303.5 | **超** |

**`g_s_720_369`** — v3′=`plateau`（基準 `g_s_720_b169`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 26 | ○ | -13.8 | 下 |
| restore | 26 | ○ | -13.8 | 下 |
| stage2 | 145 | ○ | 1623.4 | **超** |

**`g_s_896_b161`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 11 | × no baseline for this window | — | — |
| restore | 11 | × no baseline for this window | — | — |
| stage2 | 21 | × no baseline for this window | — | — |

**`g_s_896_345`** — v3′=`plateau`（基準 `g_s_896_b161`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 25 | ○ | 0.1 | 下 |
| restore | 25 | ○ | 0.1 | 下 |
| stage2 | 141 | ○ | 1376.4 | **超** |

**`g_s_896_353`** — v3′=`plateau`（基準 `g_s_896_b161`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 26 | ○ | -26.7 | 下 |
| restore | 26 | ○ | -26.7 | 下 |
| stage2 | 146 | ○ | 1692.2 | **超** |

**`g_c_b1280x768`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 12 | × no baseline for this window | — | — |
| restore | 12 | × no baseline for this window | — | — |
| stage2 | 21 | × no baseline for this window | — | — |

**`g_c_1856x1024`** — v3′=`plateau`（基準 `g_c_b1280x768`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 24 | ○ | -60.9 | 下 |
| restore | 24 | ○ | -60.9 | 下 |
| stage2 | 79 | ○ | 533.7 | **超** |

**`g_c_1920x1088`** — v3′=`plateau`（基準 `g_c_b1280x768`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 26 | ○ | -60.8 | 下 |
| restore | 26 | ○ | -60.8 | 下 |
| stage2 | 145 | ○ | 1527.5 | **超** |

**`g_s_1080_153`** — v3′=`plateau`（基準 `g_s_1080_b89`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 23 | ○ | -35.6 | 下 |
| restore | 23 | ○ | -35.6 | 下 |
| stage2 | 78 | ○ | 536.5 | **超** |

**`g_s_1080_145`** — v3′=`clean`（基準 `g_s_1080_b89`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 22 | ○ | -9.1 | 下 |
| restore | 22 | ○ | -9.1 | 下 |
| stage2 | 42 | ○ | -35.9 | 下 |

**`g_s_720_353`** — v3′=`plateau`（基準 `g_s_720_b169`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 24 | ○ | -4.3 | 下 |
| restore | 24 | ○ | -4.3 | 下 |
| stage2 | 116 | ○ | 1001.4 | **超** |

**`g_s_720_345`** — v3′=`plateau`（基準 `g_s_720_b169`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 24 | ○ | -13.3 | 下 |
| restore | 24 | ○ | -13.3 | 下 |
| stage2 | 100 | ○ | 785.4 | **超** |

**`g_s_720_329`** — v3′=`clean`（基準 `g_s_720_b169`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 23 | ○ | -4.7 | 下 |
| restore | 23 | ○ | -4.7 | 下 |
| stage2 | 61 | ○ | 297.0 | 下 |

**`g_s_896_329`** — v3′=`plateau`（基準 `g_s_896_b161`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 24 | ○ | -26.4 | 下 |
| restore | 24 | ○ | -26.4 | 下 |
| stage2 | 101 | ○ | 776.6 | **超** |

**`g_s_896_313`** — v3′=`plateau`（基準 `g_s_896_b161`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 23 | ○ | -0.1 | 下 |
| restore | 23 | ○ | -0.1 | 下 |
| stage2 | 66 | ○ | 302.3 | **超** |

**`g_s_896_297`** — v3′=`clean`（基準 `g_s_896_b161`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 21 | ○ | -8.9 | 下 |
| restore | 21 | ○ | -8.9 | 下 |
| stage2 | 41 | ○ | -35.8 | 下 |

**`g_c_1792x1024`** — v3′=`clean`（基準 `g_c_b1280x768`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 24 | ○ | -53.2 | 下 |
| restore | 24 | ○ | -53.2 | 下 |
| stage2 | 43 | ○ | 88.7 | 下 |

**`g_c_1728x1024`** — v3′=`clean`（基準 `g_c_b1280x768`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 22 | ○ | -64.7 | 下 |
| restore | 22 | ○ | -64.7 | 下 |
| stage2 | 41 | ○ | -37.6 | 下 |

**`g_c_1664x960`** — v3′=`clean`（基準 `g_c_b1280x768`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 20 | ○ | -78.2 | 下 |
| restore | 20 | ○ | -78.2 | 下 |
| stage2 | 36 | ○ | -21.4 | 下 |

**`g_s_1080_145_r2`** — v3′=`clean`（基準 `g_s_1080_b89`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 22 | ○ | -33.9 | 下 |
| restore | 22 | ○ | -33.9 | 下 |
| stage2 | 42 | ○ | -35.2 | 下 |

**`g_s_1080_153_r2`** — v3′=`plateau`（基準 `g_s_1080_b89`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 23 | ○ | -10.6 | 下 |
| restore | 23 | ○ | -10.6 | 下 |
| stage2 | 78 | ○ | 536.6 | **超** |

**`g_s_720_329_r2`** — v3′=`plateau`（基準 `g_s_720_b169`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 23 | ○ | -13.0 | 下 |
| restore | 23 | ○ | -13.0 | 下 |
| stage2 | 61 | ○ | 305.7 | **超** |

**`g_s_720_313`** — v3′=`clean`（基準 `g_s_720_b169`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 22 | ○ | -17.8 | 下 |
| restore | 22 | ○ | -17.8 | 下 |
| stage2 | 41 | ○ | -40.6 | 下 |

**`g_s_896_297_r2`** — v3′=`clean`（基準 `g_s_896_b161`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 21 | ○ | -3.3 | 下 |
| restore | 21 | ○ | -3.3 | 下 |
| stage2 | 41 | ○ | -35.4 | 下 |

**`g_s_896_313_r2`** — v3′=`plateau`（基準 `g_s_896_b161`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 23 | ○ | 0.2 | 下 |
| restore | 23 | ○ | 0.2 | 下 |
| stage2 | 61 | ○ | 303.2 | **超** |

**`g_c_1792x1024_r2`** — v3′=`clean`（基準 `g_c_b1280x768`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 23 | ○ | -78.5 | 下 |
| restore | 23 | ○ | -78.5 | 下 |
| stage2 | 44 | ○ | 40.0 | 下 |

**`g_c_1856x1024_r2`** — v3′=`plateau`（基準 `g_c_b1280x768`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 24 | ○ | -78.2 | 下 |
| restore | 24 | ○ | -78.2 | 下 |
| stage2 | 78 | ○ | 512.5 | **超** |

**`g_s_720_313_r2`** — v3′=`clean`（基準 `g_s_720_b169`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 22 | ○ | -525.4 | 下 |
| restore | 22 | ○ | -525.4 | 下 |
| stage2 | 42 | ○ | -500.8 | 下 |

**`b_warm`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 3 | × fewer than 10 WDDM samples | — | — |
| restore | 3 | × fewer than 10 WDDM samples | — | — |
| stage2 | 5 | × fewer than 10 WDDM samples | — | — |

**`b_s_1080_b89`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 13 | × no baseline for this window | — | — |
| restore | 13 | × no baseline for this window | — | — |
| stage2 | 24 | × no baseline for this window | — | — |

**`b_s_1080_161`** — v3′=`clean`（基準 `b_s_1080_b89`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 24 | ○ | -43.7 | 下 |
| restore | 24 | ○ | -43.7 | 下 |
| stage2 | 47 | ○ | -0.6 | 下 |

**`b_s_1080_169`** — v3′=`clean`（基準 `b_s_1080_b89`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 25 | ○ | -43.6 | 下 |
| restore | 25 | ○ | -43.6 | 下 |
| stage2 | 50 | ○ | -43.3 | 下 |

**`a_warm`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 6 | × fewer than 10 WDDM samples | — | — |
| s23_gap_mid | 4 | × fewer than 10 WDDM samples | — | — |
| s23_gap_tail | 6 | × fewer than 10 WDDM samples | — | — |
| s23_stage1 | 7 | × fewer than 10 WDDM samples | — | — |
| stage2 | 6 | × fewer than 10 WDDM samples | — | — |

**`a_s_1080_b89`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 18 | × no baseline for this window | — | — |
| s23_gap_mid | 12 | × no baseline for this window | — | — |
| s23_gap_tail | 18 | × no baseline for this window | — | — |
| s23_stage1 | 20 | × no baseline for this window | — | — |
| stage2 | 24 | × no baseline for this window | — | — |

**`a_s_1080_161`** — v3′=`clean`（基準 `a_s_1080_b89`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 35 | ○ | -17.3 | 下 |
| s23_gap_mid | 24 | ○ | 1.4 | 下 |
| s23_gap_tail | 35 | ○ | -17.3 | 下 |
| s23_stage1 | 35 | ○ | 1.0 | 下 |
| stage2 | 46 | ○ | 1.2 | 下 |

**`a_s_1080_169`** — v3′=`clean`（基準 `a_s_1080_b89`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 37 | ○ | 0.3 | 下 |
| s23_gap_mid | 26 | ○ | -18.5 | 下 |
| s23_gap_tail | 37 | ○ | 0.3 | 下 |
| s23_stage1 | 36 | ○ | -23.1 | 下 |
| stage2 | 48 | ○ | -23.8 | 下 |

**`f_s_1080_153_d`** — v3′=`plateau`（基準 `f_s_1080_b89_d`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 54 | ○ | 145.0 | 下 |
| s23_gap_mid | 80 | ○ | 1664.5 | **超** |
| s23_gap_tail | 54 | ○ | 145.0 | 下 |
| s23_stage1 | 44 | ○ | 17.0 | 下 |
| stage2 | 156 | ○ | 2909.0 | **超** |

**`f_s_1080_145_d`** — v3′=`plateau`（基準 `f_s_1080_b89_d`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 55 | ○ | -42.0 | 下 |
| s23_gap_mid | 59 | ○ | 798.0 | **超** |
| s23_gap_tail | 55 | ○ | -42.0 | 下 |
| s23_stage1 | 42 | ○ | -17.2 | 下 |
| stage2 | 116 | ○ | 2001.9 | **超** |

**`f_s_1080_137_d`** — v3′=`plateau`（基準 `f_s_1080_b89_d`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 45 | ○ | -16.9 | 下 |
| s23_gap_mid | 37 | ○ | 206.0 | 下 |
| s23_gap_tail | 45 | ○ | -16.9 | 下 |
| s23_stage1 | 40 | ○ | 16.7 | 下 |
| stage2 | 72 | ○ | 1316.7 | **超** |

**`g_warm2`** — v3′=`baseline`（基準 `—`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 3 | × fewer than 10 WDDM samples | — | — |
| restore | 3 | × fewer than 10 WDDM samples | — | — |
| stage2 | 6 | × fewer than 10 WDDM samples | — | — |

**`g_s_720_313_r2b`** — v3′=`clean`（基準 `g_s_720_b169`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| p25_30_decode_encode | 22 | ○ | -14.1 | 下 |
| restore | 22 | ○ | -14.1 | 下 |
| stage2 | 41 | ○ | -41.4 | 下 |

**`f_s_1080_121_d`** — v3′=`clean`（基準 `f_s_1080_b89_d`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 40 | ○ | -42.2 | 下 |
| s23_gap_mid | 31 | ○ | -3.2 | 下 |
| s23_gap_tail | 40 | ○ | -42.2 | 下 |
| s23_stage1 | 34 | ○ | -3.1 | 下 |
| stage2 | 58 | ○ | -42.2 | 下 |

**`f_s_1080_105_d`** — v3′=`clean`（基準 `f_s_1080_b89_d`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 38 | ○ | -17.2 | 下 |
| s23_gap_mid | 26 | ○ | -17.5 | 下 |
| s23_gap_tail | 38 | ○ | -17.2 | 下 |
| s23_stage1 | 30 | ○ | 16.8 | 下 |
| stage2 | 48 | ○ | -15.8 | 下 |

**`f_s_1080_129_d`** — v3′=`plateau`（基準 `f_s_1080_b89_d`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 42 | ○ | -42.4 | 下 |
| s23_gap_mid | 34 | ○ | -17.0 | 下 |
| s23_gap_tail | 42 | ○ | -42.4 | 下 |
| s23_stage1 | 38 | ○ | -0.2 | 下 |
| stage2 | 65 | ○ | 483.3 | **超** |

**`f_s_1080_121_d_r2`** — v3′=`clean`（基準 `f_s_1080_b89_d`）

| 窓 | 標本 | 判定可 | 基準比 MB | 天井 |
|---|---:|---|---:|---|
| restore | 41 | ○ | -24.1 | 下 |
| s23_gap_mid | 31 | ○ | 21.0 | 下 |
| s23_gap_tail | 41 | ○ | -24.1 | 下 |
| s23_stage1 | 35 | ○ | 19.8 | 下 |
| stage2 | 58 | ○ | -24.1 | 下 |

### 速度と共有メモリ

| ラベル | 生成秒 | stage2秒 | 秒/トークン | 復元秒 | 予約ピークMB | 共有ピーク(job)MB | 参照符号化 予約MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| `f_warm` | 68.1 | 9.9 | 0.001611 | 6.3 | 9,498 | 2196.0 | — |
| `f_b1216` | 280.0 | 130.2 | 0.007079 | 63.5 | 14,310 | 2212.0 | — |
| `f_w46` | 412.0 | 261.3 | 0.006795 | 64.8 | 17,700 | 4318.3 | — |
| `f_s_1080_b89` | 87.2 | 37.0 | 0.001513 | 18.6 | 12,412 | 2204.3 | — |
| `f_s_1080_161` | 428.4 | 415.9 | 0.009709 | 36.1 | 19,310 | 6078.7 | — |
| `f_s_1080_b89_d` | 169.0 | 57.8 | 0.002359 | 27.5 | 12,320 | 2202.2 | — |
| `f_s_1080_161_d` | 390.9 | 221.2 | 0.005165 | 57.0 | 18,952 | 5749.5 | — |
| `g_warm` | 53.8 | 9.0 | 0.001465 | 3.0 | 7,180 | 1694.5 | — |
| `g_s_1080_b89` | 81.5 | 36.8 | 0.001501 | 13.1 | 11,326 | 1508.0 | — |
| `g_s_1080_161` | 240.2 | 166.9 | 0.003897 | 24.1 | 16,414 | 2119.9 | — |
| `g_s_1080_169` | 293.6 | 217.3 | 0.004843 | 25.0 | 16,982 | 2756.4 | — |
| `g_s_720_b169` | 70.6 | 30.9 | 0.001463 | 11.3 | 10,414 | 1637.5 | — |
| `g_s_720_361` | 278.9 | 208.8 | 0.004728 | 26.0 | 16,774 | 2525.0 | — |
| `g_s_720_369` | 296.8 | 219.0 | 0.004854 | 26.4 | 17,048 | 2828.5 | — |
| `g_s_896_b161` | 71.4 | 31.4 | 0.001481 | 11.4 | 10,424 | 1657.7 | — |
| `g_s_896_345` | 283.2 | 212.1 | 0.004782 | 25.1 | 16,833 | 2567.1 | — |
| `g_s_896_353` | 297.1 | 219.9 | 0.004848 | 25.6 | 17,116 | 2908.9 | — |
| `g_c_b1280x768` | 79.5 | 30.9 | 0.001463 | 12.4 | 10,414 | 1672.4 | — |
| `g_c_1856x1024` | 195.4 | 117.8 | 0.002884 | 23.9 | 15,870 | 1744.4 | — |
| `g_c_1920x1088` | 301.4 | 217.3 | 0.004843 | 25.9 | 16,994 | 2756.5 | — |
| `g_s_1080_153` | 187.1 | 117.8 | 0.002886 | 23.2 | 15,856 | 1752.6 | — |
| `g_s_1080_145` | 129.1 | 63.1 | 0.001629 | 22.2 | 15,286 | 1639.7 | — |
| `g_s_720_353` | 248.3 | 174.6 | 0.004042 | 24.6 | 16,496 | 2209.1 | — |
| `g_s_720_345` | 223.1 | 150.6 | 0.003565 | 24.0 | 16,226 | 1976.8 | — |
| `g_s_720_329` | 161.3 | 91.7 | 0.002273 | 23.1 | 15,708 | 1667.2 | — |
| `g_s_896_329` | 224.2 | 151.5 | 0.003579 | 24.3 | 16,272 | 1975.9 | — |
| `g_s_896_313` | 167.3 | 98.0 | 0.002429 | 23.4 | 15,706 | 1667.0 | — |
| `g_s_896_297` | 127.5 | 62.0 | 0.001617 | 21.4 | 15,142 | 1640.0 | — |
| `g_c_1792x1024` | 139.7 | 64.5 | 0.001636 | 23.3 | 15,452 | 1689.4 | — |
| `g_c_1728x1024` | 135.1 | 61.2 | 0.001610 | 22.5 | 15,080 | 1483.0 | — |
| `g_c_1664x960` | 122.3 | 54.5 | 0.001587 | 20.6 | 14,224 | 1683.6 | — |
| `g_s_1080_145_r2` | 129.0 | 63.0 | 0.001625 | 22.2 | 15,297 | 1667.3 | — |
| `g_s_1080_153_r2` | 186.6 | 117.6 | 0.002882 | 23.2 | 15,846 | 1736.4 | — |
| `g_s_720_329_r2` | 161.0 | 91.7 | 0.002273 | 23.1 | 15,708 | 1658.4 | — |
| `g_s_720_313` | 129.5 | 62.1 | 0.001617 | 22.3 | 15,158 | 1667.1 | — |
| `g_s_896_297_r2` | 127.4 | 62.0 | 0.001617 | 21.4 | 15,142 | 1658.4 | — |
| `g_s_896_313_r2` | 161.1 | 91.7 | 0.002273 | 23.4 | 15,706 | 1694.9 | — |
| `g_c_1792x1024_r2` | 139.2 | 65.2 | 0.001655 | 23.2 | 15,452 | 1640.2 | — |
| `g_c_1856x1024_r2` | 195.9 | 117.8 | 0.002884 | 23.9 | 15,870 | 1752.5 | — |
| `g_s_720_313_r2` | 156.4 | 62.2 | 0.001621 | 22.5 | 13,060 | 1145.5 | — |
| `b_warm` | 26.8 | 8.7 | 0.001416 | 3.0 | 6,796 | 1155.3 | — |
| `b_s_1080_b89` | 80.2 | 36.6 | 0.001495 | 13.1 | 9,206 | 1139.7 | — |
| `b_s_1080_161` | 143.1 | 70.8 | 0.001653 | 24.0 | 14,290 | 1128.5 | — |
| `b_s_1080_169` | 150.2 | 75.2 | 0.001674 | 24.9 | 14,830 | 1096.4 | — |
| `a_warm` | 84.1 | 8.4 | 0.001367 | 6.1 | 9,064 | 1168.5 | — |
| `a_s_1080_b89` | 85.2 | 35.7 | 0.001458 | 18.1 | 11,428 | 1179.0 | — |
| `a_s_1080_161` | 149.1 | 69.3 | 0.001618 | 35.5 | 14,420 | 1180.2 | — |
| `a_s_1080_169` | 155.7 | 73.3 | 0.001634 | 36.3 | 15,004 | 1159.7 | — |
| `f_s_1080_153_d` | 373.8 | 234.4 | 0.005746 | 54.7 | 18,244 | 5112.4 | — |
| `f_s_1080_145_d` | 330.9 | 174.4 | 0.004501 | 54.2 | 17,478 | 4231.3 | — |
| `f_s_1080_137_d` | 249.7 | 108.0 | 0.002941 | 44.2 | 16,674 | 3541.9 | — |
| `g_warm2` | 52.2 | 8.8 | 0.001440 | 3.0 | 7,180 | 1675.2 | — |
| `g_s_720_313_r2b` | 129.5 | 62.1 | 0.001617 | 22.3 | 15,158 | 1657.4 | — |
| `f_s_1080_121_d` | 203.4 | 87.5 | 0.002679 | 40.4 | 15,220 | 2208.5 | — |
| `f_s_1080_105_d` | 192.1 | 72.0 | 0.002521 | 38.2 | 13,874 | 2204.8 | — |
| `f_s_1080_129_d` | 219.7 | 98.0 | 0.002824 | 41.8 | 15,968 | 2695.6 | — |
| `f_s_1080_121_d_r2` | 219.0 | 87.3 | 0.002675 | 41.0 | 15,190 | 2208.5 | — |

### 所要時間（`manifest.jsonl` の実測）

| ラベル | 開始 | 終了 | 秒 |
|---|---|---|---:|
| `f_warm` | 2026-09-26 02:10:05.863 | 2026-09-26 02:11:16.031 | 70.2 |
| `f_b1216` | 2026-09-26 02:12:46.045 | 2026-09-26 02:17:26.782 | 280.7 |
| `f_w46` | 2026-09-26 02:20:02.634 | 2026-09-26 02:26:55.559 | 412.9 |
| `f_s_1080_b89` | 2026-09-26 02:28:01.581 | 2026-09-26 02:29:29.765 | 88.2 |
| `f_s_1080_161` | 2026-09-26 02:32:04.786 | 2026-09-26 02:39:13.660 | 428.9 |
| `f_s_1080_b89_d` | 2026-09-26 02:41:14.691 | 2026-09-26 02:44:05.039 | 170.3 |
| `f_s_1080_161_d` | 2026-09-26 02:46:40.252 | 2026-09-26 02:53:11.235 | 391.0 |
| `g_warm` | 2026-09-26 02:57:46.677 | 2026-09-26 02:58:40.832 | 54.1 |
| `g_s_1080_b89` | 2026-09-26 03:01:15.899 | 2026-09-26 03:02:38.054 | 82.2 |
| `g_s_1080_161` | 2026-09-26 03:05:13.129 | 2026-09-26 03:09:13.809 | 240.7 |
| `g_s_1080_169` | 2026-09-26 03:11:49.563 | 2026-09-26 03:16:44.352 | 294.8 |
| `g_s_720_b169` | 2026-09-26 03:19:20.144 | 2026-09-26 03:20:32.343 | 72.2 |
| `g_s_720_361` | 2026-09-26 03:23:07.337 | 2026-09-26 03:27:48.030 | 280.7 |
| `g_s_720_369` | 2026-09-26 03:30:23.880 | 2026-09-26 03:35:22.662 | 298.8 |
| `g_s_896_b161` | 2026-09-26 03:37:58.482 | 2026-09-26 03:39:10.704 | 72.2 |
| `g_s_896_345` | 2026-09-26 03:41:45.660 | 2026-09-26 03:46:30.412 | 284.8 |
| `g_s_896_353` | 2026-09-26 03:49:06.180 | 2026-09-26 03:54:04.947 | 298.8 |
| `g_c_b1280x768` | 2026-09-26 03:56:40.793 | 2026-09-26 03:58:01.039 | 80.2 |
| `g_c_1856x1024` | 2026-09-26 04:00:35.997 | 2026-09-26 04:03:52.424 | 196.4 |
| `g_c_1920x1088` | 2026-09-26 04:06:27.460 | 2026-09-26 04:11:30.144 | 302.7 |
| `g_s_1080_153` | 2026-09-26 04:14:22.416 | 2026-09-26 04:17:30.848 | 188.4 |
| `g_s_1080_145` | 2026-09-26 04:20:05.803 | 2026-09-26 04:22:16.150 | 130.3 |
| `g_s_720_353` | 2026-09-26 04:24:51.058 | 2026-09-26 04:28:59.712 | 248.7 |
| `g_s_720_345` | 2026-09-26 04:31:35.503 | 2026-09-26 04:35:20.018 | 224.5 |
| `g_s_720_329` | 2026-09-26 04:37:54.929 | 2026-09-26 04:40:37.357 | 162.4 |
| `g_s_896_329` | 2026-09-26 04:43:24.845 | 2026-09-26 04:47:09.436 | 224.6 |
| `g_s_896_313` | 2026-09-26 04:49:45.288 | 2026-09-26 04:52:33.761 | 168.5 |
| `g_s_896_297` | 2026-09-26 04:55:09.600 | 2026-09-26 04:57:17.974 | 128.4 |
| `g_c_1792x1024` | 2026-09-26 04:59:52.896 | 2026-09-26 05:02:13.348 | 140.4 |
| `g_c_1728x1024` | 2026-09-26 05:04:48.164 | 2026-09-26 05:07:04.542 | 136.4 |
| `g_c_1664x960` | 2026-09-26 05:09:39.472 | 2026-09-26 05:11:43.796 | 124.3 |
| `g_s_1080_145_r2` | 2026-09-26 05:14:24.580 | 2026-09-26 05:16:34.910 | 130.3 |
| `g_s_1080_153_r2` | 2026-09-26 05:19:09.858 | 2026-09-26 05:22:18.416 | 188.6 |
| `g_s_720_329_r2` | 2026-09-26 05:25:01.284 | 2026-09-26 05:27:43.770 | 162.5 |
| `g_s_720_313` | 2026-09-26 05:30:19.583 | 2026-09-26 05:32:29.975 | 130.4 |
| `g_s_896_297_r2` | 2026-09-26 05:35:15.409 | 2026-09-26 05:37:23.712 | 128.3 |
| `g_s_896_313_r2` | 2026-09-26 05:39:58.647 | 2026-09-26 05:42:40.997 | 162.3 |
| `g_c_1792x1024_r2` | 2026-09-26 05:45:24.937 | 2026-09-26 05:47:45.367 | 140.4 |
| `g_c_1856x1024_r2` | 2026-09-26 05:50:20.188 | 2026-09-26 05:53:36.719 | 196.5 |
| `g_s_720_313_r2` | 2026-09-26 05:56:18.282 | 2026-09-26 05:58:56.695 | 158.4 |
| `b_warm` | 2026-09-26 05:59:23.195 | 2026-09-26 05:59:51.269 | 28.1 |
| `b_s_1080_b89` | 2026-09-26 06:02:26.283 | 2026-09-26 06:03:48.544 | 82.3 |
| `b_s_1080_161` | 2026-09-26 06:06:23.481 | 2026-09-26 06:08:47.877 | 144.4 |
| `b_s_1080_169` | 2026-09-26 06:11:22.900 | 2026-09-26 06:13:53.412 | 150.5 |
| `a_warm` | 2026-09-26 06:19:41.372 | 2026-09-26 06:21:07.673 | 86.3 |
| `a_s_1080_b89` | 2026-09-26 06:23:42.630 | 2026-09-26 06:25:08.828 | 86.2 |
| `a_s_1080_161` | 2026-09-26 06:27:43.950 | 2026-09-26 06:30:14.348 | 150.4 |
| `a_s_1080_169` | 2026-09-26 06:32:50.257 | 2026-09-26 06:35:26.684 | 156.4 |
| `f_s_1080_153_d` | 2026-09-26 06:41:12.764 | 2026-09-26 06:47:27.640 | 374.9 |
| `f_s_1080_145_d` | 2026-09-26 06:50:03.476 | 2026-09-26 06:55:36.234 | 332.8 |
| `f_s_1080_137_d` | 2026-09-26 06:58:11.254 | 2026-09-26 07:02:21.922 | 250.7 |
| `g_warm2` | 2026-09-26 07:07:52.010 | 2026-09-26 07:08:46.153 | 54.1 |
| `g_s_720_313_r2b` | 2026-09-26 07:11:21.197 | 2026-09-26 07:13:31.626 | 130.4 |
| `f_s_1080_121_d` | 2026-09-26 07:19:19.359 | 2026-09-26 07:22:43.877 | 204.5 |
| `f_s_1080_105_d` | 2026-09-26 07:25:19.778 | 2026-09-26 07:28:32.200 | 192.4 |
| `f_s_1080_129_d` | 2026-09-26 07:34:20.503 | 2026-09-26 07:38:01.080 | 220.6 |
| `f_s_1080_121_d_r2` | 2026-09-26 07:40:35.949 | 2026-09-26 07:44:16.589 | 220.6 |

- 計測点の実行時間の合計: **175.7 分**
- 最初の点の開始から最後の点の終了まで（冷却込み）: **334.2 分**
- 実行した点の数（再走を含む延べ）: **57**

