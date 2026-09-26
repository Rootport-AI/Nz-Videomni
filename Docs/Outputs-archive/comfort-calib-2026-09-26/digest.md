> **このファイルは実機の `outputs/comfort-calib-2026-09-26/digest.md`（git追跡外）のスナップショットである（2026-09-26複写）。** 正本は実機側にあり、実機側が更新された場合はこの複写も更新する。複写の目的は、git cloneした読者が参照を辿れるようにすること。

| # | label | base | transformer_used | accel(used) | geometry | stage2 tokens | v3' | stage2 Δ MB | restore Δ MB | used-check | commit peak |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | f_warm | LTX23 | sulphur_distil_fp8mixed | attn=sage kr=on vae=on | 768x512/121f | 6144 | baseline |  |  | ok |  |
| 2 | f_b1216 | LTX23 | sulphur_distil_fp8mixed | attn=sage kr=on vae=on | 1216x704 clips=[{'num_frames': 361}, {'num_frames': 361}] win=standard | 18392 | baseline |  |  | ok |  |
| 3 | f_w46 | LTX23 | sulphur_distil_fp8mixed | attn=sage kr=on vae=on | 1216x704 clips=[{'num_frames': 361}, {'num_frames': 361}] win=w46 | 38456 | plateau | +2126.7 | -18.7 | ok |  |
| 4 | f_s_1080_b89 | LTX23 | sulphur_distil_fp8mixed | attn=sage kr=on vae=on | 1920x1088/89f | 24480 | baseline |  |  | ok | 103.53 GiB (91.0%) |
| 5 | f_s_1080_161 | LTX23 | sulphur_distil_fp8mixed | attn=sage kr=on vae=on | 1920x1088/161f | 42840 | plateau | +3864.8 | -14.2 | ok | 110.59 GiB (97.2%) |
| 6 | f_s_1080_b89_d | LTX23 | sulphur_distil_fp8mixed | attn=sdpa kr=off vae=off | 1920x1088/89f | 24480 | baseline |  |  | ok | 94.80 GiB (83.3%) |
| 7 | f_s_1080_161_d | LTX23 | sulphur_distil_fp8mixed | attn=sdpa kr=off vae=off | 1920x1088/161f | 42840 | plateau | +3529.5 | -42.5 | ok | 97.44 GiB (85.6%) |
| 8 | g_warm | LTX25 | ltx-2.5-22b-distilled-transformer-fp8_e4m3fn | attn=sage kr=on vae=conv | 768x512/121f | 6144 | baseline |  |  | ok | 79.20 GiB (69.6%) |
| 9 | g_s_1080_b89 | LTX25 | ltx-2.5-22b-distilled-transformer-fp8_e4m3fn | attn=sage kr=on vae=conv | 1920x1088/89f | 24480 | baseline |  |  | ok | 94.91 GiB (83.4%) |
| 10 | g_s_1080_161 | LTX25 | ltx-2.5-22b-distilled-transformer-fp8_e4m3fn | attn=sage kr=on vae=conv | 1920x1088/161f | 42840 | plateau | +920.2 | -17.9 | ok | 100.04 GiB (87.9%) |
| 11 | g_s_1080_169 | LTX25 | ltx-2.5-22b-distilled-transformer-fp8_e4m3fn | attn=sage kr=on vae=conv | 1920x1088/169f | 44880 | plateau | +1574.7 | -8.9 | ok | 100.59 GiB (88.4%) |
| 12 | g_s_720_b169 | LTX25 | ltx-2.5-22b-distilled-transformer-fp8_e4m3fn | attn=sage kr=on vae=conv | 1280x768/169f | 21120 | baseline |  |  | ok | 93.99 GiB (82.6%) |
| 13 | g_s_720_361 | LTX25 | ltx-2.5-22b-distilled-transformer-fp8_e4m3fn | attn=sage kr=on vae=conv | 1280x768/361f | 44160 | plateau | +1303.5 | -31.6 | ok | 100.35 GiB (88.2%) |
| 14 | g_s_720_369 | LTX25 | ltx-2.5-22b-distilled-transformer-fp8_e4m3fn | attn=sage kr=on vae=conv | 1280x768/369f | 45120 | plateau | +1623.4 | -13.8 | ok | 100.65 GiB (88.4%) |
| 15 | g_s_896_b161 | LTX25 | ltx-2.5-22b-distilled-transformer-fp8_e4m3fn | attn=sage kr=on vae=conv | 896x1152/161f | 21168 | baseline |  |  | ok | 94.04 GiB (82.6%) |
| 16 | g_s_896_345 | LTX25 | ltx-2.5-22b-distilled-transformer-fp8_e4m3fn | attn=sage kr=on vae=conv | 896x1152/345f | 44352 | plateau | +1376.4 | +0.1 | ok | 100.51 GiB (88.3%) |
| 17 | g_s_896_353 | LTX25 | ltx-2.5-22b-distilled-transformer-fp8_e4m3fn | attn=sage kr=on vae=conv | 896x1152/353f | 45360 | plateau | +1692.2 | -26.7 | ok | 100.88 GiB (88.6%) |
| 18 | g_c_b1280x768 | LTX25 | ltx-2.5-22b-distilled-transformer-fp8_e4m3fn | attn=sage kr=on vae=conv | 1280x768 clips=[{'num_frames': 25}, {'num_frames': 161}] win=standard | 21120 | baseline |  |  | ok | 94.23 GiB (82.8%) |
| 19 | g_c_1856x1024 | LTX25 | ltx-2.5-22b-distilled-transformer-fp8_e4m3fn | attn=sage kr=on vae=conv | 1856x1024 clips=[{'num_frames': 25}, {'num_frames': 161}] win=standard | 40832 | plateau | +533.7 | -60.9 | ok | 99.67 GiB (87.6%) |
| 20 | g_c_1920x1088 | LTX25 | ltx-2.5-22b-distilled-transformer-fp8_e4m3fn | attn=sage kr=on vae=conv | 1920x1088 clips=[{'num_frames': 25}, {'num_frames': 161}] win=standard | 44880 | plateau | +1527.5 | -60.8 | ok | 100.83 GiB (88.6%) |
| 21 | g_s_1080_153 | LTX25 | ltx-2.5-22b-distilled-transformer-fp8_e4m3fn | attn=sage kr=on vae=conv | 1920x1088/153f | 40800 | plateau | +536.5 | -35.6 | ok | 99.54 GiB (87.5%) |
| 22 | g_s_1080_145 | LTX25 | ltx-2.5-22b-distilled-transformer-fp8_e4m3fn | attn=sage kr=on vae=conv | 1920x1088/145f | 38760 | clean | -35.9 | -9.1 | ok | 98.91 GiB (86.9%) |
| 23 | g_s_720_353 | LTX25 | ltx-2.5-22b-distilled-transformer-fp8_e4m3fn | attn=sage kr=on vae=conv | 1280x768/353f | 43200 | plateau | +1001.4 | -4.3 | ok | 100.32 GiB (88.1%) |
| 24 | g_s_720_345 | LTX25 | ltx-2.5-22b-distilled-transformer-fp8_e4m3fn | attn=sage kr=on vae=conv | 1280x768/345f | 42240 | plateau | +785.4 | -13.3 | ok | 100.05 GiB (87.9%) |
| 25 | g_s_720_329 | LTX25 | ltx-2.5-22b-distilled-transformer-fp8_e4m3fn | attn=sage kr=on vae=conv | 1280x768/329f | 40320 | clean | +297.0 | -4.7 | ok | 99.55 GiB (87.5%) |
| 26 | g_s_896_329 | LTX25 | ltx-2.5-22b-distilled-transformer-fp8_e4m3fn | attn=sage kr=on vae=conv | 896x1152/329f | 42336 | plateau | +776.6 | -26.4 | ok | 100.12 GiB (88.0%) |
| 27 | g_s_896_313 | LTX25 | ltx-2.5-22b-distilled-transformer-fp8_e4m3fn | attn=sage kr=on vae=conv | 896x1152/313f | 40320 | plateau | +302.3 | -0.1 | ok | 99.59 GiB (87.5%) |
| 28 | g_s_896_297 | LTX25 | ltx-2.5-22b-distilled-transformer-fp8_e4m3fn | attn=sage kr=on vae=conv | 896x1152/297f | 38304 | clean | -35.8 | -8.9 | ok | 98.90 GiB (86.9%) |
| 29 | g_c_1792x1024 | LTX25 | ltx-2.5-22b-distilled-transformer-fp8_e4m3fn | attn=sage kr=on vae=conv | 1792x1024 clips=[{'num_frames': 25}, {'num_frames': 161}] win=standard | 39424 | clean | +88.7 | -53.2 | ok | 99.39 GiB (87.3%) |
| 30 | g_c_1728x1024 | LTX25 | ltx-2.5-22b-distilled-transformer-fp8_e4m3fn | attn=sage kr=on vae=conv | 1728x1024 clips=[{'num_frames': 25}, {'num_frames': 161}] win=standard | 38016 | clean | -37.6 | -64.7 | ok | 98.87 GiB (86.9%) |
| 31 | g_c_1664x960 | LTX25 | ltx-2.5-22b-distilled-transformer-fp8_e4m3fn | attn=sage kr=on vae=conv | 1664x960 clips=[{'num_frames': 25}, {'num_frames': 161}] win=standard | 34320 | clean | -21.4 | -78.2 | ok | 98.13 GiB (86.2%) |
| 32 | g_s_1080_145_r2 | LTX25 | ltx-2.5-22b-distilled-transformer-fp8_e4m3fn | attn=sage kr=on vae=conv | 1920x1088/145f | 38760 | clean | -35.2 | -33.9 | ok | 99.09 GiB (87.1%) |
| 33 | g_s_1080_153_r2 | LTX25 | ltx-2.5-22b-distilled-transformer-fp8_e4m3fn | attn=sage kr=on vae=conv | 1920x1088/153f | 40800 | plateau | +536.6 | -10.6 | ok | 99.79 GiB (87.7%) |
| 34 | g_s_720_329_r2 | LTX25 | ltx-2.5-22b-distilled-transformer-fp8_e4m3fn | attn=sage kr=on vae=conv | 1280x768/329f | 40320 | plateau | +305.7 | -13.0 | ok | 99.61 GiB (87.5%) |
| 35 | g_s_720_313 | LTX25 | ltx-2.5-22b-distilled-transformer-fp8_e4m3fn | attn=sage kr=on vae=conv | 1280x768/313f | 38400 | clean | -40.6 | -17.8 | ok | 98.98 GiB (87.0%) |
| 36 | g_s_896_297_r2 | LTX25 | ltx-2.5-22b-distilled-transformer-fp8_e4m3fn | attn=sage kr=on vae=conv | 896x1152/297f | 38304 | clean | -35.4 | -3.3 | ok | 99.42 GiB (87.3%) |
| 37 | g_s_896_313_r2 | LTX25 | ltx-2.5-22b-distilled-transformer-fp8_e4m3fn | attn=sage kr=on vae=conv | 896x1152/313f | 40320 | plateau | +303.2 | +0.2 | ok | 100.13 GiB (88.0%) |
| 38 | g_c_1792x1024_r2 | LTX25 | ltx-2.5-22b-distilled-transformer-fp8_e4m3fn | attn=sage kr=on vae=conv | 1792x1024 clips=[{'num_frames': 25}, {'num_frames': 161}] win=standard | 39424 | clean | +40.0 | -78.5 | ok | 99.25 GiB (87.2%) |
| 39 | g_c_1856x1024_r2 | LTX25 | ltx-2.5-22b-distilled-transformer-fp8_e4m3fn | attn=sage kr=on vae=conv | 1856x1024 clips=[{'num_frames': 25}, {'num_frames': 161}] win=standard | 40832 | plateau | +512.5 | -78.2 | ok | 99.78 GiB (87.7%) |
| 40 | g_s_720_313_r2 | LTX25 | default | attn=sage kr=on vae=conv | 1280x768/313f | 38400 | clean | -500.8 | -525.4 | MISMATCH transformer_used=default | 74.60 GiB (65.5%) |
| 41 | b_warm | LTX25 | default | attn=sage kr=on vae=conv | 768x512/121f | 6144 | baseline |  |  | ok | 66.07 GiB (58.0%) |
| 42 | b_s_1080_b89 | LTX25 | default | attn=sage kr=on vae=conv | 1920x1088/89f | 24480 | baseline |  |  | ok | 70.90 GiB (62.3%) |
| 43 | b_s_1080_161 | LTX25 | default | attn=sage kr=on vae=conv | 1920x1088/161f | 42840 | clean | -0.6 | -43.7 | ok | 75.89 GiB (66.7%) |
| 44 | b_s_1080_169 | LTX25 | default | attn=sage kr=on vae=conv | 1920x1088/169f | 44880 | clean | -43.3 | -43.6 | ok | 76.60 GiB (67.3%) |
| 45 | a_warm | LTX23 | Sulphur-2-base-distil-Q6_K | attn=sage kr=on vae=on | 768x512/121f | 6144 | baseline |  |  | ok | 83.61 GiB (73.5%) |
| 46 | a_s_1080_b89 | LTX23 | Sulphur-2-base-distil-Q6_K | attn=sage kr=on vae=on | 1920x1088/89f | 24480 | baseline |  |  | ok | 87.19 GiB (76.6%) |
| 47 | a_s_1080_161 | LTX23 | Sulphur-2-base-distil-Q6_K | attn=sage kr=on vae=on | 1920x1088/161f | 42840 | clean | +1.2 | -17.3 | ok | 90.07 GiB (79.1%) |
| 48 | a_s_1080_169 | LTX23 | Sulphur-2-base-distil-Q6_K | attn=sage kr=on vae=on | 1920x1088/169f | 44880 | clean | -23.8 | +0.3 | ok | 90.62 GiB (79.6%) |
| 49 | f_s_1080_153_d | LTX23 | sulphur_distil_fp8mixed | attn=sdpa kr=off vae=off | 1920x1088/153f | 40800 | plateau | +2909.0 | +145.0 | ok | 81.49 GiB (71.6%) |
| 50 | f_s_1080_145_d | LTX23 | sulphur_distil_fp8mixed | attn=sdpa kr=off vae=off | 1920x1088/145f | 38760 | plateau | +2001.9 | -42.0 | ok | 89.25 GiB (78.4%) |
| 51 | f_s_1080_137_d | LTX23 | sulphur_distil_fp8mixed | attn=sdpa kr=off vae=off | 1920x1088/137f | 36720 | plateau | +1316.7 | -16.9 | ok | 88.50 GiB (77.8%) |
| 52 | g_warm2 | LTX25 | ltx-2.5-22b-distilled-transformer-fp8_e4m3fn | attn=sage kr=on vae=conv | 768x512/121f | 6144 | baseline |  |  | ok | 79.32 GiB (69.7%) |
| 53 | g_s_720_313_r2b | LTX25 | ltx-2.5-22b-distilled-transformer-fp8_e4m3fn | attn=sage kr=on vae=conv | 1280x768/313f | 38400 | clean | -41.4 | -14.1 | ok | 98.80 GiB (86.8%) |
| 54 | f_s_1080_121_d | LTX23 | sulphur_distil_fp8mixed | attn=sdpa kr=off vae=off | 1920x1088/121f | 32640 | clean | -42.2 | -42.2 | ok | 78.61 GiB (69.1%) |
| 55 | f_s_1080_105_d | LTX23 | sulphur_distil_fp8mixed | attn=sdpa kr=off vae=off | 1920x1088/105f | 28560 | clean | -15.8 | -17.2 | ok | 87.90 GiB (77.2%) |
| 56 | f_s_1080_129_d | LTX23 | sulphur_distil_fp8mixed | attn=sdpa kr=off vae=off | 1920x1088/129f | 34680 | plateau | +483.3 | -42.4 | ok | 79.27 GiB (69.6%) |
| 57 | f_s_1080_121_d_r2 | LTX23 | sulphur_distil_fp8mixed | attn=sdpa kr=off vae=off | 1920x1088/121f | 32640 | clean | -24.1 | -24.1 | ok | 87.88 GiB (77.2%) |
