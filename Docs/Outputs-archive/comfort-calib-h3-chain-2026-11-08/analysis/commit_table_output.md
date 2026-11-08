> **このファイルは実機の `outputs/comfort-calib-h3-chain-2026-11-08/analysis/commit_table_output.md`（git追跡外）のスナップショットである（2026-11-08複写）。** 正本は実機側にあり、実機側が更新された場合はこの複写も更新する。複写の目的は、git cloneした読者が参照を辿れるようにすること。

| plan | label | v3' (H3: v3'') | stage2 Δ MB (H3: denoise) | restore Δ MB (H3: decode) | used | commit peak GiB | % of limit | H3 encode-window commit peak GiB | H3 denoise-window commit peak GiB |
|---|---|---|---|---|---|---|---|---|---|
| p_A-U | A-U | oom | -74.2 |  | n/a | 63.79 | 46.1 | 63.79 | 63.72 |
| p_A1 | A1 | comfortable | -30.9 | -48.4 | ok | 66.78 | 48.3 | 54.28 | 66.78 |
| p_A1_r2 | A1_r2 | comfortable | -65.2 | -64.5 | ok | 63.81 | 46.1 | 62.95 | 63.81 |
| p_B1 | B1 | comfortable | -16.0 | +5.5 | ok | 63.85 | 46.1 | 61.27 | 63.85 |
| p_B1_r2 | B1_r2 | comfortable | -22.6 | +9.1 | ok | 63.86 | 46.2 | 53.83 | 63.86 |
| p_C1 | C1 | comfortable | -5.9 | -16.7 | ok | 61.43 | 44.4 | 53.26 | 61.43 |
| p_D-K3 | D-K3 | comfortable | -40.5 | -27.1 | ok | 62.06 | 44.9 | 31.13 | 62.06 |
| p_D-K3_r2 | D-K3_r2 | comfortable | -40.7 | -35.6 | ok | 62.11 | 44.9 | 31.28 | 62.11 |
| p_E | E | comfortable | -5.5 | -16.3 | ok | 63.64 | 46.0 | 61.30 | 63.64 |
| p_F | F | comfortable | -29.3 | -6.4 | ok | 66.73 | 48.2 | 53.09 | 66.73 |
| p_b1080 | b1080 | baseline |  |  | ok | 54.35 | 39.3 | 29.36 | 54.35 |
| p_cb1080 | cb1080 | baseline |  |  | ok | 59.21 | 42.8 | 59.21 | 59.21 |
| p_cb1080v | cb1080v | baseline |  |  | ok | 55.57 | 40.2 | 54.84 | 55.57 |
| p_cb1440 | cb1440 | baseline |  |  | ok | 58.30 | 42.1 | 56.46 | 58.30 |
| p_cb720 | cb720 | baseline |  |  | ok | 53.76 | 38.9 | 53.53 | 53.76 |
| p_smoke_c640 | smoke_c640 | smoke |  |  | ok | 55.55 | 40.2 | 55.53 | 55.55 |
| p_warm | warm | smoke |  |  | ok | 54.39 | 39.3 | 29.88 | 54.39 |
