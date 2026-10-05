| # | label | base | transformer_used | accel(used) | geometry | stage2 tokens | v3' | stage2 Δ MB | restore Δ MB | used-check | commit peak |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | sa_warm | LTX23 | default | attn=sage kr=on vae=on | 768x512/121f | 6144 | baseline |  |  | ok | 75.04 GiB (52.4%) |
| 2 | sa_c_b1280x768 | LTX23 | default | attn=sage kr=on vae=on | 1280x768 clips=[{'num_frames': 361}, {'num_frames': 361}] win=standard | 21120 | baseline |  |  | ok | 78.00 GiB (54.4%) |
| 3 | sa_c_1920x1024 | LTX23 | default | attn=sage kr=on vae=on | 1920x1024 clips=[{'num_frames': 361}, {'num_frames': 361}] win=standard | 42240 | clean | -43.7 | -72.3 | ok | 83.29 GiB (58.1%) |
| 4 | sa_c_1984x1024 | LTX23 | default | attn=sage kr=on vae=on | 1984x1024 clips=[{'num_frames': 361}, {'num_frames': 361}] win=standard | 43648 | plateau | +711.7 | -66.6 | ok | 81.10 GiB (56.6%) |
| 5 | sa_c_1920x1024_r2 | LTX23 | default | attn=sage kr=on vae=on | 1920x1024 clips=[{'num_frames': 361}, {'num_frames': 361}] win=standard | 42240 | clean | -64.6 | -71.3 | ok | 80.24 GiB (56.0%) |
| 6 | sa_c_1984x1024_r2 | LTX23 | default | attn=sage kr=on vae=on | 1984x1024 clips=[{'num_frames': 361}, {'num_frames': 361}] win=standard | 43648 | plateau | +553.4 | -82.4 | ok | 80.87 GiB (56.4%) |
| 7 | qa_warm | LTX23 | Sulphur-2-base-distil-Q6_K | attn=sage kr=on vae=on | 768x512/121f | 6144 | baseline |  |  | ok | 76.36 GiB (53.3%) |
| 8 | qa_s_720_b89 | LTX23 | Sulphur-2-base-distil-Q6_K | attn=sage kr=on vae=on | 1280x768/89f | 11520 | baseline |  |  | ok | 75.92 GiB (53.0%) |
| 9 | qa_s_720_353 | LTX23 | Sulphur-2-base-distil-Q6_K | attn=sage kr=on vae=on | 1280x768/353f | 43200 | clean | +10.6 | +2.5 | ok | 82.83 GiB (57.8%) |
| 10 | qa_s_720_353_r2 | LTX23 | Sulphur-2-base-distil-Q6_K | attn=sage kr=on vae=on | 1280x768/353f | 43200 | clean | +3.3 | +9.0 | ok | 83.07 GiB (58.0%) |
| 11 | qa_s_720_361 | LTX23 | Sulphur-2-base-distil-Q6_K | attn=sage kr=on vae=on | 1280x768/361f | 44160 | plateau | +947.2 | -21.4 | ok | 85.24 GiB (59.5%) |
| 12 | qa_s_1080_b89 | LTX23 | Sulphur-2-base-distil-Q6_K | attn=sage kr=on vae=on | 1920x1088/89f | 24480 | baseline |  |  | ok | 80.63 GiB (56.3%) |
| 13 | qa_s_1080_161 | LTX23 | Sulphur-2-base-distil-Q6_K | attn=sage kr=on vae=on | 1920x1088/161f | 42840 | clean | -0.3 | -8.5 | ok | 83.62 GiB (58.4%) |
| 14 | qa_s_1080_161_r2 | LTX23 | Sulphur-2-base-distil-Q6_K | attn=sage kr=on vae=on | 1920x1088/161f | 42840 | clean | -25.5 | -25.4 | ok | 83.54 GiB (58.3%) |
| 15 | qa_c_b1280x768 | LTX23 | Sulphur-2-base-distil-Q6_K | attn=sage kr=on vae=on | 1280x768 clips=[{'num_frames': 361}, {'num_frames': 361}] win=standard | 21120 | baseline |  |  | ok | 79.48 GiB (55.5%) |
| 16 | qa_c_1920x1024 | LTX23 | Sulphur-2-base-distil-Q6_K | attn=sage kr=on vae=on | 1920x1024 clips=[{'num_frames': 361}, {'num_frames': 361}] win=standard | 42240 | plateau | +351.3 | -15.9 | ok | 84.49 GiB (59.0%) |
| 17 | qa_c_1856x1024 | LTX23 | Sulphur-2-base-distil-Q6_K | attn=sage kr=on vae=on | 1856x1024 clips=[{'num_frames': 361}, {'num_frames': 361}] win=standard | 40832 | clean | -0.4 | -10.5 | ok | 84.21 GiB (58.8%) |
| 18 | qa_c_1856x1024_r2 | LTX23 | Sulphur-2-base-distil-Q6_K | attn=sage kr=on vae=on | 1856x1024 clips=[{'num_frames': 361}, {'num_frames': 361}] win=standard | 40832 | clean | +63.2 | -7.1 | ok | 84.21 GiB (58.8%) |
| 19 | qa_c_1920x1024_r2 | LTX23 | Sulphur-2-base-distil-Q6_K | attn=sage kr=on vae=on | 1920x1024 clips=[{'num_frames': 361}, {'num_frames': 361}] win=standard | 42240 | plateau | +351.3 | -24.1 | ok | 84.68 GiB (59.1%) |
| 20 | sb_warm | LTX25 | default | attn=sage kr=on vae=conv | 768x512/121f | 6144 | baseline |  |  | ok | 59.30 GiB (41.4%) |
| 21 | sb_c_b1280x768 | LTX25 | default | attn=sage kr=on vae=conv | 1280x768 clips=[{'num_frames': 361}, {'num_frames': 361}] win=standard | 21120 | baseline |  |  | ok | 63.37 GiB (44.2%) |
| 22 | sb_c_2048x1088 | LTX25 | default | attn=sage kr=on vae=conv | 2048x1088 clips=[{'num_frames': 361}, {'num_frames': 361}] win=standard | 47872 | plateau | +412.0 | -3.0 | ok | 71.15 GiB (49.7%) |
| 23 | sb_c_1984x1088 | LTX25 | default | attn=sage kr=on vae=conv | 1984x1088 clips=[{'num_frames': 361}, {'num_frames': 361}] win=standard | 46376 | clean | +37.0 | +5.2 | ok | 70.40 GiB (49.1%) |
| 24 | sb_c_1984x1088_r2 | LTX25 | default | attn=sage kr=on vae=conv | 1984x1088 clips=[{'num_frames': 361}, {'num_frames': 361}] win=standard | 46376 | clean | -4.8 | -3.0 | ok | 70.28 GiB (49.1%) |
| 25 | sb_c_2048x1088_r2 | LTX25 | default | attn=sage kr=on vae=conv | 2048x1088 clips=[{'num_frames': 361}, {'num_frames': 361}] win=standard | 47872 | clean | +97.5 | -11.3 | ok | 70.65 GiB (49.3%) |
| 26 | sd_warm | LTX25 | default | attn=sdpa kr=off vae=conv | 768x512/121f | 6144 | baseline |  |  | ok | 55.28 GiB (38.6%) |
| 27 | sd_c_b1280x768 | LTX25 | default | attn=sdpa kr=off vae=conv | 1280x768 clips=[{'num_frames': 361}, {'num_frames': 361}] win=standard | 21120 | baseline |  |  | ok | 51.22 GiB (35.8%) |
| 28 | sd_c_1984x1088 | LTX25 | default | attn=sdpa kr=off vae=conv | 1984x1088 clips=[{'num_frames': 361}, {'num_frames': 361}] win=standard | 46376 | clean | +10.2 | -6.9 | ok | 58.22 GiB (40.6%) |
| 29 | qb_warm | LTX25 | redgraftLTX25Fast2K_ltx25RedgraftNSFW-Q6_K | attn=sage kr=on vae=conv | 768x512/121f | 6144 | baseline |  |  | ok | 64.72 GiB (45.2%) |
| 30 | qb_s_896_b89 | LTX25 | redgraftLTX25Fast2K_ltx25RedgraftNSFW-Q6_K | attn=sage kr=on vae=conv | 896x1152/89f | 12096 | baseline |  |  | ok | 66.96 GiB (46.7%) |
| 31 | qb_s_896_337 | LTX25 | redgraftLTX25Fast2K_ltx25RedgraftNSFW-Q6_K | attn=sage kr=on vae=conv | 896x1152/337f | 43344 | clean | -9.9 | -42.8 | ok | 75.36 GiB (52.6%) |
| 32 | qb_s_896_337_r2 | LTX25 | redgraftLTX25Fast2K_ltx25RedgraftNSFW-Q6_K | attn=sage kr=on vae=conv | 896x1152/337f | 43344 | clean | -25.9 | -49.9 | ok | 75.40 GiB (52.6%) |
| 33 | qb_s_720_b89 | LTX25 | redgraftLTX25Fast2K_ltx25RedgraftNSFW-Q6_K | attn=sage kr=on vae=conv | 1280x768/89f | 11520 | baseline |  |  | ok | 67.02 GiB (46.8%) |
| 34 | qb_s_720_353 | LTX25 | redgraftLTX25Fast2K_ltx25RedgraftNSFW-Q6_K | attn=sage kr=on vae=conv | 1280x768/353f | 43200 | clean | -32.7 | -32.7 | ok | 75.42 GiB (52.6%) |
| 35 | qb_s_720_353_r2 | LTX25 | redgraftLTX25Fast2K_ltx25RedgraftNSFW-Q6_K | attn=sage kr=on vae=conv | 1280x768/353f | 43200 | clean | -32.7 | -32.7 | ok | 75.43 GiB (52.6%) |
| 36 | qb_s_720_361 | LTX25 | redgraftLTX25Fast2K_ltx25RedgraftNSFW-Q6_K | attn=sage kr=on vae=conv | 1280x768/361f | 44160 | clean | +243.5 | -48.5 | ok | 75.75 GiB (52.9%) |
| 37 | qb_s_720_361_r2 | LTX25 | redgraftLTX25Fast2K_ltx25RedgraftNSFW-Q6_K | attn=sage kr=on vae=conv | 1280x768/361f | 44160 | clean | +287.3 | -32.5 | ok | 75.77 GiB (52.9%) |
| 38 | qb_s_1080_b89 | LTX25 | redgraftLTX25Fast2K_ltx25RedgraftNSFW-Q6_K | attn=sage kr=on vae=conv | 1920x1088/89f | 24480 | baseline |  |  | ok | 70.64 GiB (49.3%) |
| 39 | qb_s_1080_161 | LTX25 | redgraftLTX25Fast2K_ltx25RedgraftNSFW-Q6_K | attn=sage kr=on vae=conv | 1920x1088/161f | 42840 | clean | -41.0 | -9.3 | ok | 75.42 GiB (52.6%) |
| 40 | qb_s_1080_161_r2 | LTX25 | redgraftLTX25Fast2K_ltx25RedgraftNSFW-Q6_K | attn=sage kr=on vae=conv | 1920x1088/161f | 42840 | clean | -41.0 | -18.6 | ok | 75.38 GiB (52.6%) |
| 41 | qb_c_b1280x768 | LTX25 | redgraftLTX25Fast2K_ltx25RedgraftNSFW-Q6_K | attn=sage kr=on vae=conv | 1280x768 clips=[{'num_frames': 361}, {'num_frames': 361}] win=standard | 21120 | baseline |  |  | ok | 69.83 GiB (48.7%) |
| 42 | qb_c_1920x1024 | LTX25 | redgraftLTX25Fast2K_ltx25RedgraftNSFW-Q6_K | attn=sage kr=on vae=conv | 1920x1024 clips=[{'num_frames': 361}, {'num_frames': 361}] win=standard | 42240 | clean | -21.7 | +0.0 | ok | 75.26 GiB (52.5%) |
| 43 | qb_c_1984x1024 | LTX25 | redgraftLTX25Fast2K_ltx25RedgraftNSFW-Q6_K | attn=sage kr=on vae=conv | 1984x1024 clips=[{'num_frames': 361}, {'num_frames': 361}] win=standard | 43648 | clean | +42.3 | +8.5 | ok | 75.68 GiB (52.8%) |
| 44 | qb_c_2048x1024 | LTX25 | redgraftLTX25Fast2K_ltx25RedgraftNSFW-Q6_K | attn=sage kr=on vae=conv | 2048x1024 clips=[{'num_frames': 361}, {'num_frames': 361}] win=standard | 45056 | plateau | +514.1 | +0.0 | ok | 76.55 GiB (53.4%) |
| 45 | qb_c_1984x1024_r2 | LTX25 | redgraftLTX25Fast2K_ltx25RedgraftNSFW-Q6_K | attn=sage kr=on vae=conv | 1984x1024 clips=[{'num_frames': 361}, {'num_frames': 361}] win=standard | 43648 | clean | +122.1 | +9.6 | ok | 75.75 GiB (52.9%) |
| 46 | qb_c_2048x1024_r2 | LTX25 | redgraftLTX25Fast2K_ltx25RedgraftNSFW-Q6_K | attn=sage kr=on vae=conv | 2048x1024 clips=[{'num_frames': 361}, {'num_frames': 361}] win=standard | 45056 | plateau | +608.4 | -7.7 | ok | 76.10 GiB (53.1%) |
| 47 | ra_warm | LTX25 | redgraftLTX25Fast2K_ltx25RedgraftNSFW | attn=sdpa kr=off vae=conv | 768x512/121f | 6144 | baseline |  |  | ok | 51.79 GiB (36.2%) |
| 48 | ra_s_1080_b89 | LTX25 | redgraftLTX25Fast2K_ltx25RedgraftNSFW | attn=sdpa kr=off vae=conv | 1920x1088/89f | 24480 | baseline |  |  | ok | 67.41 GiB (47.1%) |
| 49 | ra_s_1080_145 | LTX25 | redgraftLTX25Fast2K_ltx25RedgraftNSFW | attn=sdpa kr=off vae=conv | 1920x1088/145f | 38760 | clean | -16.5 | -16.5 | ok | 70.78 GiB (49.4%) |
| 50 | ra_s_1080_153 | LTX25 | redgraftLTX25Fast2K_ltx25RedgraftNSFW | attn=sdpa kr=off vae=conv | 1920x1088/153f | 40800 | plateau | +338.2 | -42.7 | ok | 71.40 GiB (49.8%) |
| 51 | ra_s_1080_145_r2 | LTX25 | redgraftLTX25Fast2K_ltx25RedgraftNSFW | attn=sdpa kr=off vae=conv | 1920x1088/145f | 38760 | clean | -16.2 | -16.5 | ok | 71.20 GiB (49.7%) |
| 52 | ra_s_1080_153_r2 | LTX25 | redgraftLTX25Fast2K_ltx25RedgraftNSFW | attn=sdpa kr=off vae=conv | 1920x1088/153f | 40800 | plateau | +346.5 | -45.7 | ok | 71.54 GiB (49.9%) |
| 53 | ra_s_720_b89 | LTX25 | redgraftLTX25Fast2K_ltx25RedgraftNSFW | attn=sdpa kr=off vae=conv | 1280x768/89f | 11520 | baseline |  |  | ok | 64.09 GiB (44.7%) |
| 54 | ra_s_720_313 | LTX25 | redgraftLTX25Fast2K_ltx25RedgraftNSFW | attn=sdpa kr=off vae=conv | 1280x768/313f | 38400 | clean | -41.1 | -8.7 | ok | 70.92 GiB (49.5%) |
| 55 | ra_s_720_313_r2 | LTX25 | redgraftLTX25Fast2K_ltx25RedgraftNSFW | attn=sdpa kr=off vae=conv | 1280x768/313f | 38400 | clean | -28.7 | -17.2 | ok | 70.97 GiB (49.5%) |
| 56 | ra_s_720_337 | LTX25 | redgraftLTX25Fast2K_ltx25RedgraftNSFW | attn=sdpa kr=off vae=conv | 1280x768/337f | 41280 | plateau | +367.2 | -0.2 | ok | 71.80 GiB (50.1%) |
| 57 | ra_c_b1280x768 | LTX25 | redgraftLTX25Fast2K_ltx25RedgraftNSFW | attn=sdpa kr=off vae=conv | 1280x768 clips=[{'num_frames': 361}, {'num_frames': 361}] win=standard | 21120 | baseline |  |  | ok | 66.84 GiB (46.7%) |
| 58 | ra_c_1728x1024 | LTX25 | redgraftLTX25Fast2K_ltx25RedgraftNSFW | attn=sdpa kr=off vae=conv | 1728x1024 clips=[{'num_frames': 361}, {'num_frames': 361}] win=standard | 38016 | clean | -5.7 | +15.9 | ok | 71.25 GiB (49.7%) |
| 59 | ra_c_1792x1024 | LTX25 | redgraftLTX25Fast2K_ltx25RedgraftNSFW | attn=sdpa kr=off vae=conv | 1792x1024 clips=[{'num_frames': 361}, {'num_frames': 361}] win=standard | 39424 | clean | -21.5 | +13.7 | ok | 71.46 GiB (49.9%) |
| 60 | ra_c_1856x1024 | LTX25 | redgraftLTX25Fast2K_ltx25RedgraftNSFW | attn=sdpa kr=off vae=conv | 1856x1024 clips=[{'num_frames': 361}, {'num_frames': 361}] win=standard | 40832 | plateau | +421.1 | +0.0 | ok | 72.01 GiB (50.3%) |
| 61 | ra_c_1792x1024_r2 | LTX25 | redgraftLTX25Fast2K_ltx25RedgraftNSFW | attn=sdpa kr=off vae=conv | 1792x1024 clips=[{'num_frames': 361}, {'num_frames': 361}] win=standard | 39424 | clean | -5.7 | +0.0 | ok | 71.67 GiB (50.0%) |
| 62 | ra_c_1856x1024_r2 | LTX25 | redgraftLTX25Fast2K_ltx25RedgraftNSFW | attn=sdpa kr=off vae=conv | 1856x1024 clips=[{'num_frames': 361}, {'num_frames': 361}] win=standard | 40832 | clean | +230.5 | -15.7 | ok | 72.46 GiB (50.6%) |
| 63 | rb_warm | LTX25 | redgraftLTX25Fast2K_ltx25RedgraftNSFW | attn=sage kr=on vae=conv | 768x512/121f | 6144 | baseline |  |  | ok | 71.92 GiB (50.2%) |
| 64 | rb_c_b1280x768 | LTX25 | redgraftLTX25Fast2K_ltx25RedgraftNSFW | attn=sage kr=on vae=conv | 1280x768 clips=[{'num_frames': 361}, {'num_frames': 361}] win=standard | 21120 | baseline |  |  | ok | 81.94 GiB (57.2%) |
| 65 | rb_c_1792x1024 | LTX25 | redgraftLTX25Fast2K_ltx25RedgraftNSFW | attn=sage kr=on vae=conv | 1792x1024 clips=[{'num_frames': 361}, {'num_frames': 361}] win=standard | 39424 | clean | -14.7 | +15.9 | ok | 86.09 GiB (60.1%) |
| 66 | ua_warm | LTX25 | ltx25_uncensored_v1.1-fp8_scaled | attn=sdpa kr=off vae=conv | 768x512/121f | 6144 | baseline |  |  | ok | 59.22 GiB (41.3%) |
| 67 | ua_s_1080_b89 | LTX25 | ltx25_uncensored_v1.1-fp8_scaled | attn=sdpa kr=off vae=conv | 1920x1088/89f | 24480 | baseline |  |  | ok | 74.98 GiB (52.3%) |
| 68 | ua_s_1080_145 | LTX25 | ltx25_uncensored_v1.1-fp8_scaled | attn=sdpa kr=off vae=conv | 1920x1088/145f | 38760 | clean | -9.0 | -33.7 | ok | 78.13 GiB (54.5%) |
| 69 | ua_s_1080_153 | LTX25 | ltx25_uncensored_v1.1-fp8_scaled | attn=sdpa kr=off vae=conv | 1920x1088/153f | 40800 | plateau | +634.9 | -9.0 | ok | 78.51 GiB (54.8%) |
| 70 | ha_warm | LTX23 | sulphur_distil_fp8mixed | attn=sage kr=on vae=on | 768x512/121f | 6144 | baseline |  |  | ok | 84.47 GiB (59.0%) |
| 71 | ha_s_1080_b89 | LTX23 | sulphur_distil_fp8mixed | attn=sage kr=on vae=on | 1920x1088/89f | 24480 | baseline |  |  | ok | 97.40 GiB (68.0%) |
| 72 | ha_s_1080_129 | LTX23 | sulphur_distil_fp8mixed | attn=sage kr=on vae=on | 1920x1088/129f | 34680 | plateau | +929.6 | +0.0 | ok | 101.16 GiB (70.6%) |
| 73 | ha_s_1080_121 | LTX23 | sulphur_distil_fp8mixed | attn=sage kr=on vae=on | 1920x1088/121f | 32640 | clean | +127.0 | +10.6 | ok | 100.41 GiB (70.1%) |
| 74 | ha_s_1080_121_r2 | LTX23 | sulphur_distil_fp8mixed | attn=sage kr=on vae=on | 1920x1088/121f | 32640 | clean | +124.0 | +9.5 | ok | 100.39 GiB (70.1%) |
| 75 | ha_s_1080_129_r2 | LTX23 | sulphur_distil_fp8mixed | attn=sage kr=on vae=on | 1920x1088/129f | 34680 | plateau | +753.9 | +1.5 | ok | 101.25 GiB (70.7%) |
| 76 | ha_s_720_b89 | LTX23 | sulphur_distil_fp8mixed | attn=sage kr=on vae=on | 1280x768/89f | 11520 | baseline |  |  | ok | 94.88 GiB (66.2%) |
| 77 | ha_s_720_265 | LTX23 | sulphur_distil_fp8mixed | attn=sage kr=on vae=on | 1280x768/265f | 32640 | clean | -29.3 | -25.6 | ok | 100.36 GiB (70.1%) |
| 78 | ha_s_720_265_r2 | LTX23 | sulphur_distil_fp8mixed | attn=sage kr=on vae=on | 1280x768/265f | 32640 | clean | -27.4 | -0.1 | ok | 100.44 GiB (70.1%) |
| 79 | ha_s_720_289 | LTX23 | sulphur_distil_fp8mixed | attn=sage kr=on vae=on | 1280x768/289f | 35520 | plateau | +1063.6 | +0.9 | ok | 101.66 GiB (71.0%) |
| 80 | ha_c_b1280x768 | LTX23 | sulphur_distil_fp8mixed | attn=sage kr=on vae=on | 1280x768 clips=[{'num_frames': 361}, {'num_frames': 361}] win=standard | 21120 | baseline |  |  | ok | 97.76 GiB (68.2%) |
| 81 | ha_c_1472x1024 | LTX23 | sulphur_distil_fp8mixed | attn=sage kr=on vae=on | 1472x1024 clips=[{'num_frames': 361}, {'num_frames': 361}] win=standard | 32384 | clean | +2.7 | -13.6 | ok | 100.69 GiB (70.3%) |
| 82 | ha_c_1536x1024 | LTX23 | sulphur_distil_fp8mixed | attn=sage kr=on vae=on | 1536x1024 clips=[{'num_frames': 361}, {'num_frames': 361}] win=standard | 33792 | plateau | +410.3 | -15.9 | ok | 101.21 GiB (70.6%) |
| 83 | ha_c_1472x1024_r2 | LTX23 | sulphur_distil_fp8mixed | attn=sage kr=on vae=on | 1472x1024 clips=[{'num_frames': 361}, {'num_frames': 361}] win=standard | 32384 | clean | -21.5 | -21.6 | ok | 100.72 GiB (70.3%) |
| 84 | ha_c_1536x1024_r2 | LTX23 | sulphur_distil_fp8mixed | attn=sage kr=on vae=on | 1536x1024 clips=[{'num_frames': 361}, {'num_frames': 361}] win=standard | 33792 | plateau | +418.3 | -5.5 | ok | 101.24 GiB (70.7%) |
