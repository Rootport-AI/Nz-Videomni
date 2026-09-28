> **このファイルは実機の `outputs/comfort-calib-2026-09-27/digest.md`（git追跡外）のスナップショットである（2026-09-28複写）。** 正本は実機側にあり、実機側が更新された場合はこの複写も更新する。複写の目的は、git cloneした読者が参照を辿れるようにすること。

| # | label | base | transformer_used | accel(used) | geometry | stage2 tokens | v3' | stage2 Δ MB | restore Δ MB | used-check | commit peak |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | u_warm | LTX25 | ltx25_uncensored_v1.1-fp8_scaled | attn=sage kr=on vae=conv | 768x512/121f | 6144 | baseline |  |  | ok | 75.58 GiB (67.4%) |
| 2 | u_s_1080_b89 | LTX25 | ltx25_uncensored_v1.1-fp8_scaled | attn=sage kr=on vae=conv | 1920x1088/89f | 24480 | baseline |  |  | ok | 90.38 GiB (80.6%) |
| 3 | u_s_1080_145 | LTX25 | ltx25_uncensored_v1.1-fp8_scaled | attn=sage kr=on vae=conv | 1920x1088/145f | 38760 | clean | +82.6 | -31.6 | ok | 93.31 GiB (83.2%) |
| 4 | u_s_1080_153 | LTX25 | ltx25_uncensored_v1.1-fp8_scaled | attn=sage kr=on vae=conv | 1920x1088/153f | 40800 | plateau | +903.3 | -39.1 | ok | 93.92 GiB (83.8%) |
| 5 | u_s_1080_145_r2 | LTX25 | ltx25_uncensored_v1.1-fp8_scaled | attn=sage kr=on vae=conv | 1920x1088/145f | 38760 | clean | +113.6 | -31.8 | ok | 93.43 GiB (83.3%) |
| 6 | u_s_1080_153_r2 | LTX25 | ltx25_uncensored_v1.1-fp8_scaled | attn=sage kr=on vae=conv | 1920x1088/153f | 40800 | plateau | +622.5 | -31.8 | ok | 93.94 GiB (83.8%) |
| 7 | r_warm | LTX25 | redgraftLTX25Fast2K_ltx25RedgraftNSFW | attn=sage kr=on vae=conv | 768x512/121f | 6144 | baseline |  |  | ok | 67.06 GiB (59.8%) |
| 8 | r_s_1080_b89 | LTX25 | redgraftLTX25Fast2K_ltx25RedgraftNSFW | attn=sage kr=on vae=conv | 1920x1088/89f | 24480 | baseline |  |  | ok | 82.80 GiB (73.9%) |
| 9 | r_s_1080_153 | LTX25 | redgraftLTX25Fast2K_ltx25RedgraftNSFW | attn=sage kr=on vae=conv | 1920x1088/153f | 40800 | plateau | +558.2 | -26.7 | ok | 86.77 GiB (77.4%) |
| 10 | r_s_1080_161 | LTX25 | redgraftLTX25Fast2K_ltx25RedgraftNSFW | attn=sage kr=on vae=conv | 1920x1088/161f | 42840 | plateau | +1191.0 | -33.6 | ok | 87.45 GiB (78.0%) |
| 11 | r_s_1080_145 | LTX25 | redgraftLTX25Fast2K_ltx25RedgraftNSFW | attn=sage kr=on vae=conv | 1920x1088/145f | 38760 | clean | +58.0 | -35.7 | ok | 86.30 GiB (77.0%) |
| 12 | r_s_1080_145_r2 | LTX25 | redgraftLTX25Fast2K_ltx25RedgraftNSFW | attn=sage kr=on vae=conv | 1920x1088/145f | 38760 | clean | -35.1 | -47.3 | ok | 86.31 GiB (77.0%) |
| 13 | r_s_1080_153_r2 | LTX25 | redgraftLTX25Fast2K_ltx25RedgraftNSFW | attn=sage kr=on vae=conv | 1920x1088/153f | 40800 | plateau | +618.5 | -36.4 | ok | 86.90 GiB (77.5%) |
| 14 | r_s_720_b169 | LTX25 | redgraftLTX25Fast2K_ltx25RedgraftNSFW | attn=sage kr=on vae=conv | 1280x768/169f | 21120 | baseline |  |  | ok | 82.05 GiB (73.2%) |
| 15 | r_s_720_313 | LTX25 | redgraftLTX25Fast2K_ltx25RedgraftNSFW | attn=sage kr=on vae=conv | 1280x768/313f | 38400 | clean | -34.0 | -23.7 | ok | 86.22 GiB (76.9%) |
| 16 | r_s_720_313_r2 | LTX25 | redgraftLTX25Fast2K_ltx25RedgraftNSFW | attn=sage kr=on vae=conv | 1280x768/313f | 38400 | clean | -25.5 | -15.4 | ok | 86.26 GiB (76.9%) |
| 17 | r_s_720_337 | LTX25 | redgraftLTX25Fast2K_ltx25RedgraftNSFW | attn=sage kr=on vae=conv | 1280x768/337f | 41280 | plateau | +678.5 | -41.4 | ok | 87.15 GiB (77.7%) |
| 18 | h_warm | LTX25 | ltx-2.5-22b-distilled-transformer-comfy-int8-convrot | attn=sage kr=on vae=conv | 768x512/121f | 6144 | baseline |  |  | ok | 74.26 GiB (66.2%) |
| 19 | h_s_1080_b89 | LTX25 | ltx-2.5-22b-distilled-transformer-comfy-int8-convrot | attn=sage kr=on vae=conv | 1920x1088/89f | 24480 | baseline |  |  | ok | 90.04 GiB (80.3%) |
| 20 | h_s_1080_145 | LTX25 | ltx-2.5-22b-distilled-transformer-comfy-int8-convrot | attn=sage kr=on vae=conv | 1920x1088/145f | 38760 | clean | +153.6 | -7.0 | ok | 93.37 GiB (83.3%) |
| 21 | h_s_1080_153 | LTX25 | ltx-2.5-22b-distilled-transformer-comfy-int8-convrot | attn=sage kr=on vae=conv | 1920x1088/153f | 40800 | plateau | +814.2 | -36.6 | ok | 94.09 GiB (83.9%) |
| 22 | h_s_1080_145_r2 | LTX25 | ltx-2.5-22b-distilled-transformer-comfy-int8-convrot | attn=sage kr=on vae=conv | 1920x1088/145f | 38760 | clean | +140.3 | -31.4 | ok | 93.57 GiB (83.5%) |
| 23 | h_s_1080_153_r2 | LTX25 | ltx-2.5-22b-distilled-transformer-comfy-int8-convrot | attn=sage kr=on vae=conv | 1920x1088/153f | 40800 | plateau | +695.0 | -15.4 | ok | 94.25 GiB (84.1%) |
| 24 | f_warm_d | LTX23 | sulphur_distil_fp8mixed | attn=sdpa kr=off vae=off | 768x512/121f | 6144 | baseline |  |  | ok | 73.29 GiB (65.4%) |
| 25 | f_s_1080_b89_d | LTX23 | sulphur_distil_fp8mixed | attn=sdpa kr=off vae=off | 1920x1088/89f | 24480 | baseline |  |  | ok | 81.79 GiB (73.0%) |
| 26 | f_s_1080_121_d | LTX23 | sulphur_distil_fp8mixed | attn=sdpa kr=off vae=off | 1920x1088/121f | 32640 | clean | +108.3 | +6.9 | ok | 82.39 GiB (73.5%) |
| 27 | f_s_1080_129_d | LTX23 | sulphur_distil_fp8mixed | attn=sdpa kr=off vae=off | 1920x1088/129f | 34680 | plateau | +726.3 | -27.2 | ok | 82.55 GiB (73.6%) |
| 28 | f_s_1080_121_d_r2 | LTX23 | sulphur_distil_fp8mixed | attn=sdpa kr=off vae=off | 1920x1088/121f | 32640 | clean | +28.6 | -0.1 | ok | 82.46 GiB (73.6%) |
| 29 | f_s_1080_129_d_r2 | LTX23 | sulphur_distil_fp8mixed | attn=sdpa kr=off vae=off | 1920x1088/129f | 34680 | plateau | +747.4 | -0.1 | ok | 82.48 GiB (73.6%) |
| 30 | x_warm_d | LTX23 | ltx-2.3-22b-distilled-1.1_int8mixedtensorwise | attn=sdpa kr=off vae=off | 768x512/121f | 6144 | baseline |  |  | ok | 73.11 GiB (65.2%) |
| 31 | x_s_1080_b89_d | LTX23 | ltx-2.3-22b-distilled-1.1_int8mixedtensorwise | attn=sdpa kr=off vae=off | 1920x1088/89f | 24480 | baseline |  |  | ok | 81.59 GiB (72.8%) |
| 32 | x_s_1080_121_d | LTX23 | ltx-2.3-22b-distilled-1.1_int8mixedtensorwise | attn=sdpa kr=off vae=off | 1920x1088/121f | 32640 | clean | +21.2 | -18.6 | ok | 82.14 GiB (73.3%) |
| 33 | x_s_1080_129_d | LTX23 | ltx-2.3-22b-distilled-1.1_int8mixedtensorwise | attn=sdpa kr=off vae=off | 1920x1088/129f | 34680 | plateau | +728.8 | -18.1 | ok | 82.89 GiB (73.9%) |
| 34 | x_s_1080_121_d_r2 | LTX23 | ltx-2.3-22b-distilled-1.1_int8mixedtensorwise | attn=sdpa kr=off vae=off | 1920x1088/121f | 32640 | clean | +46.8 | +0.7 | ok | 82.83 GiB (73.9%) |
| 35 | x_s_1080_129_d_r2 | LTX23 | ltx-2.3-22b-distilled-1.1_int8mixedtensorwise | attn=sdpa kr=off vae=off | 1920x1088/129f | 34680 | plateau | +748.5 | +1.7 | ok | 83.00 GiB (74.0%) |
| 36 | j_warm | LTX23 | ltx-2.3-22b-distilled-1.1_w4a8 | attn=sage kr=on vae=on | 768x512/121f | 6144 | baseline |  |  | ok | 67.09 GiB (59.8%) |
| 37 | j_s_1080_b89 | LTX23 | ltx-2.3-22b-distilled-1.1_w4a8 | attn=sage kr=on vae=on | 1920x1088/89f | 24480 | baseline |  |  | ok | 78.44 GiB (70.0%) |
| 38 | j_s_1080_145 | LTX23 | ltx-2.3-22b-distilled-1.1_w4a8 | attn=sage kr=on vae=on | 1920x1088/145f | 38760 | clean | +19.0 | -9.4 | ok | 81.31 GiB (72.5%) |
| 39 | j_s_1080_161 | LTX23 | ltx-2.3-22b-distilled-1.1_w4a8 | attn=sage kr=on vae=on | 1920x1088/161f | 42840 | clean | -3.9 | -7.8 | ok | 82.57 GiB (73.7%) |
| 40 | j_s_1080_169 | LTX23 | ltx-2.3-22b-distilled-1.1_w4a8 | attn=sage kr=on vae=on | 1920x1088/169f | 44880 | clean | +27.1 | -26.7 | ok | 83.11 GiB (74.1%) |
| 41 | j_s_1080_169_r2 | LTX23 | ltx-2.3-22b-distilled-1.1_w4a8 | attn=sage kr=on vae=on | 1920x1088/169f | 44880 | clean | +12.8 | -16.0 | ok | 83.32 GiB (74.3%) |
| 42 | k_warm_d | LTX23 | ltx-2.3-22b-distilled-1.1_transformer_only_int8_convrot | attn=sdpa kr=off vae=off | 768x512/121f | 6144 | baseline |  |  | ok | 69.66 GiB (62.1%) |
| 43 | k_s_1080_b89_d | LTX23 | ltx-2.3-22b-distilled-1.1_transformer_only_int8_convrot | attn=sdpa kr=off vae=off | 1920x1088/89f | 24480 | baseline |  |  | ok | 82.28 GiB (73.4%) |
| 44 | k_s_1080_121_d | LTX23 | ltx-2.3-22b-distilled-1.1_transformer_only_int8_convrot | attn=sdpa kr=off vae=off | 1920x1088/121f | 32640 | clean | -8.3 | -8.1 | ok | 82.88 GiB (73.9%) |
| 45 | k_s_1080_129_d | LTX23 | ltx-2.3-22b-distilled-1.1_transformer_only_int8_convrot | attn=sdpa kr=off vae=off | 1920x1088/129f | 34680 | clean | -41.6 | -8.2 | ok | 82.92 GiB (74.0%) |
| 46 | k_s_1080_137_d | LTX23 | ltx-2.3-22b-distilled-1.1_transformer_only_int8_convrot | attn=sdpa kr=off vae=off | 1920x1088/137f | 36720 | clean | -37.9 | -33.2 | ok | 82.62 GiB (73.7%) |
| 47 | k_s_1080_137_d_r2 | LTX23 | ltx-2.3-22b-distilled-1.1_transformer_only_int8_convrot | attn=sdpa kr=off vae=off | 1920x1088/137f | 36720 | clean | -31.4 | -27.8 | ok | 82.96 GiB (74.0%) |
