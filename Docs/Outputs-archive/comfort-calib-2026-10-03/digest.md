| # | label | base | transformer_used | accel(used) | geometry | stage2 tokens | v3' | stage2 Δ MB | restore Δ MB | used-check | commit peak |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | a1_warm | LTX23 | default | attn=sage kr=on vae=on | 768x512/121f | 6144 | baseline |  |  | ok | 79.30 GiB (71.7%) |
| 2 | a1_s_1080_b89 | LTX23 | default | attn=sage kr=on vae=on | 1920x1088/89f | 24480 | baseline |  |  | ok | 81.86 GiB (74.0%) |
| 3 | a1_s_1080_161 | LTX23 | default | attn=sage kr=on vae=on | 1920x1088/161f | 42840 | clean | -9.5 | -0.2 | ok | 85.74 GiB (77.5%) |
| 4 | a1_s_1080_169 | LTX23 | default | attn=sage kr=on vae=on | 1920x1088/169f | 44880 | plateau | +1130.6 | -8.7 | ok | 87.67 GiB (79.2%) |
| 5 | a1_s_1080_161_r2 | LTX23 | default | attn=sage kr=on vae=on | 1920x1088/161f | 42840 | clean | +0.5 | -24.4 | ok | 85.90 GiB (77.6%) |
| 6 | a1_s_1080_169_r2 | LTX23 | default | attn=sage kr=on vae=on | 1920x1088/169f | 44880 | plateau | +1195.8 | +1.0 | ok | 87.85 GiB (79.4%) |
| 7 | a1_s_720_b89 | LTX23 | default | attn=sage kr=on vae=on | 1280x768/89f | 11520 | baseline |  |  | ok | 79.03 GiB (71.4%) |
| 8 | a1_s_720_345 | LTX23 | default | attn=sage kr=on vae=on | 1280x768/345f | 42240 | clean | +16.0 | -33.2 | ok | 85.75 GiB (77.5%) |
| 9 | a1_s_720_345_r2 | LTX23 | default | attn=sage kr=on vae=on | 1280x768/345f | 42240 | clean | -33.1 | -32.6 | ok | 85.82 GiB (77.5%) |
| 10 | a1_s_720_369 | LTX23 | default | attn=sage kr=on vae=on | 1280x768/369f | 45120 | clean | +87.5 | -24.9 | ok | 86.66 GiB (78.3%) |
| 11 | a2_c_b1280x768 | LTX23 | default | attn=sage kr=on vae=on | 1280x768 clips=[{'num_frames': 361}, {'num_frames': 361}] win=standard | 21120 | baseline |  |  | ok | 81.99 GiB (74.1%) |
| 12 | a2_c_1792x1024 | LTX23 | default | attn=sage kr=on vae=on | 1792x1024 clips=[{'num_frames': 361}, {'num_frames': 361}] win=standard | 39424 | plateau | -0.9 | -0.5 | ok | 92.22 GiB (83.3%) |
| 13 | a2_c_1792x1024_r2 | LTX23 | default | attn=sage kr=on vae=on | 1792x1024 clips=[{'num_frames': 361}, {'num_frames': 361}] win=standard | 39424 | clean | -22.7 | +21.2 | ok | 91.73 GiB (82.9%) |
| 14 | a2_c_1856x1024 | LTX23 | default | attn=sage kr=on vae=on | 1856x1024 clips=[{'num_frames': 361}, {'num_frames': 361}] win=standard | 40832 | clean | -25.0 | -15.1 | ok | 92.42 GiB (83.5%) |
| 15 | a2_c_1856x1024_r2 | LTX23 | default | attn=sage kr=on vae=on | 1856x1024 clips=[{'num_frames': 361}, {'num_frames': 361}] win=standard | 40832 | clean | -8.4 | -0.1 | ok | 92.46 GiB (83.5%) |
| 16 | a2_c_1920x1024 | LTX23 | default | attn=sage kr=on vae=on | 1920x1024 clips=[{'num_frames': 361}, {'num_frames': 361}] win=standard | 42240 | plateau | -24.0 | -6.6 | ok | 93.19 GiB (84.2%) |
| 17 | a2_c_1920x1024_r2 | LTX23 | default | attn=sage kr=on vae=on | 1920x1024 clips=[{'num_frames': 361}, {'num_frames': 361}] win=standard | 42240 | plateau | -7.1 | +1.3 | ok | 93.22 GiB (84.2%) |
| 18 | b1_warm | LTX25 | default | attn=sage kr=on vae=conv | 768x512/121f | 6144 | baseline |  |  | ok | 64.61 GiB (58.4%) |
| 19 | b1_s_1080_b89 | LTX25 | default | attn=sage kr=on vae=conv | 1920x1088/89f | 24480 | baseline |  |  | ok | 70.05 GiB (63.3%) |
| 20 | b1_s_1080_169 | LTX25 | default | attn=sage kr=on vae=conv | 1920x1088/169f | 44880 | clean | -24.4 | -10.1 | ok | 75.44 GiB (68.2%) |
| 21 | b1_s_1080_177 | LTX25 | default | attn=sage kr=on vae=conv | 1920x1088/177f | 46920 | clean | +157.4 | -5.2 | ok | 76.07 GiB (68.7%) |
| 22 | b1_s_1080_185 | LTX25 | default | attn=sage kr=on vae=conv | 1920x1088/185f | 48960 | plateau | +797.5 | -4.9 | ok | 76.72 GiB (69.3%) |
| 23 | b1_s_1080_177_r2 | LTX25 | default | attn=sage kr=on vae=conv | 1920x1088/177f | 46920 | clean | +166.9 | +3.4 | ok | 76.03 GiB (68.7%) |
| 24 | b1_s_1080_185_r2 | LTX25 | default | attn=sage kr=on vae=conv | 1920x1088/185f | 48960 | plateau | +699.9 | -4.8 | ok | 76.71 GiB (69.3%) |
| 25 | b1_s_720_b89 | LTX25 | default | attn=sage kr=on vae=conv | 1280x768/89f | 11520 | baseline |  |  | ok | 66.56 GiB (60.1%) |
| 26 | b1_s_720_377 | LTX25 | default | attn=sage kr=on vae=conv | 1280x768/377f | 46080 | clean | +15.6 | -9.3 | ok | 75.95 GiB (68.6%) |
| 27 | b1_s_720_377_r2 | LTX25 | default | attn=sage kr=on vae=conv | 1280x768/377f | 46080 | clean | -0.5 | -17.4 | ok | 75.99 GiB (68.7%) |
| 28 | b1_s_720_401 | LTX25 | default | attn=sage kr=on vae=conv | 1280x768/401f | 48960 | plateau | +728.3 | -32.7 | ok | 76.75 GiB (69.3%) |
| 29 | b2_warm | LTX25 | default | attn=sdpa kr=off vae=conv | 768x512/121f | 6144 | baseline |  |  | ok | 61.05 GiB (55.2%) |
| 30 | b2_s_1080_b89 | LTX25 | default | attn=sdpa kr=off vae=conv | 1920x1088/89f | 24480 | baseline |  |  | ok | 57.72 GiB (52.2%) |
| 31 | b2_s_1080_177 | LTX25 | default | attn=sdpa kr=off vae=conv | 1920x1088/177f | 46920 | clean | +76.2 | -44.2 | ok | 63.78 GiB (57.6%) |
| 32 | b2_s_1080_185 | LTX25 | default | attn=sdpa kr=off vae=conv | 1920x1088/185f | 48960 | plateau | +630.5 | -68.8 | ok | 64.44 GiB (58.2%) |
| 33 | b2_s_1080_177_r2 | LTX25 | default | attn=sdpa kr=off vae=conv | 1920x1088/177f | 46920 | clean | +135.1 | -68.8 | ok | 63.88 GiB (57.7%) |
| 34 | b2_s_1080_185_r2 | LTX25 | default | attn=sdpa kr=off vae=conv | 1920x1088/185f | 48960 | plateau | +647.2 | -52.1 | ok | 64.36 GiB (58.2%) |
| 35 | b2_s_720_b89 | LTX25 | default | attn=sdpa kr=off vae=conv | 1280x768/89f | 11520 | baseline |  |  | ok | 55.69 GiB (50.3%) |
| 36 | b2_s_720_377 | LTX25 | default | attn=sdpa kr=off vae=conv | 1280x768/377f | 46080 | clean | -50.3 | -42.1 | ok | 63.54 GiB (57.4%) |
| 37 | b2_s_720_401 | LTX25 | default | attn=sdpa kr=off vae=conv | 1280x768/401f | 48960 | plateau | +658.4 | -16.7 | ok | 64.35 GiB (58.1%) |
| 38 | b2_s_720_377_r2 | LTX25 | default | attn=sdpa kr=off vae=conv | 1280x768/377f | 46080 | clean | -50.0 | -42.1 | ok | 63.58 GiB (57.4%) |
| 39 | b2_s_720_401_r2 | LTX25 | default | attn=sdpa kr=off vae=conv | 1280x768/401f | 48960 | plateau | +705.7 | -41.8 | ok | 64.42 GiB (58.2%) |
| 40 | b3_warm | LTX25 | default | attn=sage kr=on vae=conv | 768x512/121f | 6144 | baseline |  |  | ok | 65.29 GiB (59.0%) |
| 41 | b3_c_b1280x768 | LTX25 | default | attn=sage kr=on vae=conv | 1280x768 clips=[{'num_frames': 361}, {'num_frames': 361}] win=standard | 21120 | baseline |  |  | ok | 69.25 GiB (62.6%) |
| 42 | b3_c_1984x1088 | LTX25 | default | attn=sage kr=on vae=conv | 1984x1088 clips=[{'num_frames': 361}, {'num_frames': 361}] win=standard | 46376 | clean | +181.0 | -8.6 | ok | 76.51 GiB (69.1%) |
| 43 | b3_c_2048x1088 | LTX25 | default | attn=sage kr=on vae=conv | 2048x1088 clips=[{'num_frames': 361}, {'num_frames': 361}] win=standard | 47872 | clean | +92.8 | +6.9 | ok | 76.59 GiB (69.2%) |
| 44 | b3_c_1984x1088_r2 | LTX25 | default | attn=sage kr=on vae=conv | 1984x1088 clips=[{'num_frames': 361}, {'num_frames': 361}] win=standard | 46376 | clean | +200.1 | -16.5 | ok | 77.06 GiB (69.6%) |
| 45 | b3_c_2048x1088_r2 | LTX25 | default | attn=sage kr=on vae=conv | 2048x1088 clips=[{'num_frames': 361}, {'num_frames': 361}] win=standard | 47872 | clean | +183.3 | -8.4 | ok | 76.63 GiB (69.2%) |
| 46 | b3_c_2112x1088 | LTX25 | default | attn=sage kr=on vae=conv | 2112x1088 clips=[{'num_frames': 361}, {'num_frames': 361}] win=standard | 49368 | plateau | +670.2 | +7.5 | ok | 77.26 GiB (69.8%) |
| 47 | b3_c_2112x1088_r2 | LTX25 | default | attn=sage kr=on vae=conv | 2112x1088 clips=[{'num_frames': 361}, {'num_frames': 361}] win=standard | 49368 | plateau | +1121.3 | -12.6 | ok | 79.73 GiB (72.0%) |
