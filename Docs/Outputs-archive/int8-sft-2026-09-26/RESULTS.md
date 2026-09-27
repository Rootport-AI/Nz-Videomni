# §3-168 実機検証の結果（MCP 実機ランナー）

作成 2026-09-27 10:10:02。VRAM は nvidia-smi の専有（MiB）、コミットは `\Memory\Committed Bytes`（GiB）、共有は WDDM の `GPU Adapter Memory\Shared Usage`（参考値）。`invalid_*.json` は集計から除外している（無効と判定された記録。ファイル名の接頭辞で判定）。

## ロード

| tag | transformer | finished | 所要秒 | コミット開始前→最大（増分） | 上限 | VRAM 最大 | active_after |
|---|---|---|---|---|---|---|---|
| g3_a_load | ltx-2.3-22b-distilled-1.1_transformer_only_int8_convrot | True | 4.4 | 28.79→28.85（+0.06） | 113.08 | 1064.0 | ltx-2.3-22b-distilled-1.1_transformer_only_int8_convrot |
| g3_b_load | ltx-2.3-22b-distilled-1.1_int8mixedtensorwise | True | 4.5 | 29.27→29.27（+0.0） | 113.08 | 1046.0 | ltx-2.3-22b-distilled-1.1_int8mixedtensorwise |
| g3_c_load | ltx-2.3-22b-distilled-1.1_w4a8 | True | 4.6 | 29.17→29.17（+0.0） | 113.08 | 1052.0 | ltx-2.3-22b-distilled-1.1_w4a8 |
| g3_d_load | ltx-2.5-22b-distilled-transformer-comfy-int8-convrot | True | 8.0 | 29.45→29.45（+0.0） | 113.08 | 1040.0 | ltx-2.5-22b-distilled-transformer-comfy-int8-convrot |
| g3_e_load | redgraftLTX25Fast2K_ltx25RedgraftNSFW | True | 5.9 | 52.46→52.46（+0.0） | 113.08 | 1151.0 | redgraftLTX25Fast2K_ltx25RedgraftNSFW |
| g5_a_load_ref | default | True | 4.3 | 28.93→28.93（+0.0） | 113.08 | 1104.0 | default |
| g5_b_load_ref | default | True | 11.2 | 95.01→95.01（+0.0） | 113.08 | 1107.0 | default |
| g5_c_load_ref | default | True | 9.1 | 77.03→77.03（+0.0） | 113.08 | 1077.0 | default |
| g5_d_load_ref | default | True | 8.3 | 80.3→80.3（+0.0） | 113.08 | 1159.0 | default |
| g5_e_load_ref | redgraftLTX25Fast2K_ltx25RedgraftNSFW-Q6_K | True | 8.0 | 73.28→73.28（+0.0） | 113.08 | 1159.0 | redgraftLTX25Fast2K_ltx25RedgraftNSFW-Q6_K |
| g6_load_23 | sulphur_distil_fp8mixed | True | 7.2 | 30.09→30.16（+0.07） | 113.08 | 1166.0 | sulphur_distil_fp8mixed |
| g6_load_23_pixar | sulphur_distil_fp8mixed | True | 0.4 | 67.06→67.07（+0.01） | 113.08 | 1224.0 | sulphur_distil_fp8mixed |
| g6_load_25 | ltx25_uncensored_v1.1-fp8_scaled | True | 10.6 | 73.26→73.26（+0.0） | 113.08 | 1213.0 | ltx25_uncensored_v1.1-fp8_scaled |
| load_ltx25_uncensored_v1.1-fp8_scaled | ltx25_uncensored_v1.1-fp8_scaled | True | 26.2 | 28.41→28.56（+0.15） | 113.08 | 1036.0 | ltx25_uncensored_v1.1-fp8_scaled |
| restore | default | True | 7.3 | 61.24→61.24（+0.0） | 113.08 | 1213.0 | default |

## 生成（single / iclora / chain / keep2）

| tag | 種別 | 状態 | transformer file | 生成秒 | 経過秒 | peak_vram_mb | smi 最大 | コミット最大 | 復元 |
|---|---|---|---|---|---|---|---|---|---|
| c0_23_fp8 | single | completed | sulphur_distil_fp8mixed.safetensors | 64.79 | 70.2 | 8442 | 8536.0 | 73.65 |  |
| c0_23_fp8_pixar | single | completed | sulphur_distil_fp8mixed.safetensors | 82.63 | 82.0 | 8445 | — | — | from_outputs |
| c0_25_fp8 | single | completed | ltx25_uncensored_v1.1-fp8_scaled.safetensors | 52.06 | 52.0 | 6952 | — | — | from_outputs |
| g3_a_chain | chain | completed | ltx-2.3-22b-distilled-1.1_transformer_only_int8_convrot.safetensors | 79.9 | 80.2 | 8442 | 9868.0 | 87.58 |  |
| g3_a_iclora | iclora | completed | ltx-2.3-22b-distilled-1.1_transformer_only_int8_convrot.safetensors | 67.97 | 70.0 | 6828 | 10080.0 | 85.34 |  |
| g3_a_pixar | single | completed | ltx-2.3-22b-distilled-1.1_transformer_only_int8_convrot.safetensors | 69.93 | 70.2 | 8442 | 8389.0 | 87.25 |  |
| g3_a_single | single | completed | ltx-2.3-22b-distilled-1.1_transformer_only_int8_convrot.safetensors | 60.33 | 65.2 | 8442 | 9917.0 | 67.16 |  |
| g3_a_single_prefetch_off | single | completed | ltx-2.3-22b-distilled-1.1_transformer_only_int8_convrot.safetensors | 100.6 | 105.2 | 8442 | 8934.0 | 87.35 |  |
| g3_b_chain | chain | completed | ltx-2.3-22b-distilled-1.1_int8mixedtensorwise.safetensors | 77.43 | 80.2 | 8445 | 10389.0 | 89.16 |  |
| g3_b_iclora | iclora | completed | ltx-2.3-22b-distilled-1.1_int8mixedtensorwise.safetensors | 66.92 | 70.1 | 7196 | 10572.0 | 89.04 |  |
| g3_b_pixar | single | completed | ltx-2.3-22b-distilled-1.1_int8mixedtensorwise.safetensors | 64.66 | 65.2 | 8445 | 9443.0 | 89.26 |  |
| g3_b_single | single | completed | ltx-2.3-22b-distilled-1.1_int8mixedtensorwise.safetensors | 53.26 | 55.2 | 8442 | 8367.0 | 73.07 |  |
| g3_c_chain | chain | completed | ltx-2.3-22b-distilled-1.1_w4a8.safetensors | 90.33 | 95.3 | 8445 | 9930.0 | 84.38 |  |
| g3_c_iclora | iclora | completed | ltx-2.3-22b-distilled-1.1_w4a8.safetensors | 66.54 | 70.0 | 6666 | 8422.0 | 79.15 |  |
| g3_c_pixar | single | completed | ltx-2.3-22b-distilled-1.1_w4a8.safetensors | 74.71 | 75.2 | 8445 | 10019.0 | 83.66 |  |
| g3_c_single | single | completed | ltx-2.3-22b-distilled-1.1_w4a8.safetensors | 60.64 | 65.2 | 8442 | 8913.0 | 71.65 |  |
| g3_d_chain | chain | completed | ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors | 48.52 | 55.3 | 6606 | 6710.0 | 80.85 |  |
| g3_d_iclora | iclora | completed | ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors | 47.03 | 50.0 | 6606 | 6773.0 | 81.18 |  |
| g3_d_pixar | single | completed | ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors | 41.31 | 45.2 | 6606 | 6629.0 | 80.31 |  |
| g3_d_single | single | completed | ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors | 54.29 | 55.2 | 6952 | 8240.0 | 67.7 |  |
| g3_e_chain | chain | completed | redgraftLTX25Fast2K_ltx25RedgraftNSFW.safetensors | 54.24 | 55.2 | 6607 | 6591.0 | 75.05 |  |
| g3_e_iclora | iclora | completed | redgraftLTX25Fast2K_ltx25RedgraftNSFW.safetensors | 46.22 | 50.0 | 6607 | 6502.0 | 74.94 |  |
| g3_e_pixar | single | completed | redgraftLTX25Fast2K_ltx25RedgraftNSFW.safetensors | 44.9 | 45.1 | 6607 | 6367.0 | 70.9 |  |
| g3_e_single | single | completed | redgraftLTX25Fast2K_ltx25RedgraftNSFW.safetensors | 49.79 | 50.2 | 6952 | 8300.0 | 60.07 |  |
| g4_a | keep2 | None | — | — | — | — | — | — |  |
| g4_a_r1 | keep2 | completed | ltx-2.3-22b-distilled-1.1_transformer_only_int8_convrot.safetensors | 63.41 | 65.2 | 8442 | 8297.0 | 92.9 |  |
| g4_a_r2 | keep2 | completed | ltx-2.3-22b-distilled-1.1_transformer_only_int8_convrot.safetensors | 25.75 | 30.1 | 8442 | 8954.0 | 96.76 |  |
| g4_b | keep2 | None | — | — | — | — | — | — |  |
| g4_b_r1 | keep2 | completed | ltx-2.3-22b-distilled-1.1_int8mixedtensorwise.safetensors | 61.46 | 65.2 | 8442 | 9433.0 | 96.8 |  |
| g4_b_r2 | keep2 | completed | ltx-2.3-22b-distilled-1.1_int8mixedtensorwise.safetensors | 32.97 | 35.0 | 8442 | 9432.0 | 103.07 |  |
| g4_c | keep2 | None | — | — | — | — | — | — |  |
| g4_c_r1 | keep2 | completed | ltx-2.3-22b-distilled-1.1_w4a8.safetensors | 60.64 | 65.2 | 8442 | 8991.0 | 79.23 |  |
| g4_c_r2 | keep2 | completed | ltx-2.3-22b-distilled-1.1_w4a8.safetensors | 32.0 | 40.0 | 8442 | 8864.0 | 83.25 |  |
| g4_d | keep2 | None | — | — | — | — | — | — |  |
| g4_d_r1 | keep2 | completed | ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors | 34.61 | 35.2 | 6606 | 6559.0 | 86.08 |  |
| g4_d_r2 | keep2 | completed | ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors | 27.09 | 30.0 | 6606 | 7914.0 | 85.77 |  |
| g4_e | keep2 | None | — | — | — | — | — | — |  |
| g4_e_r1 | keep2 | completed | redgraftLTX25Fast2K_ltx25RedgraftNSFW.safetensors | 36.32 | 40.2 | 6605 | 6308.0 | 78.34 |  |
| g4_e_r2 | keep2 | completed | redgraftLTX25Fast2K_ltx25RedgraftNSFW.safetensors | 31.7 | 35.0 | 6607 | 7928.0 | 78.23 |  |
| g5_a_ref | single | completed | LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf | 59.19 | 60.2 | 8442 | 10092.0 | 64.5 |  |
| g5_b_ref | single | completed | LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf | 59.36 | 60.2 | 8442 | 9649.0 | 67.08 |  |
| g5_c_ref | single | completed | LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf | 59.88 | 60.2 | 8442 | 8933.0 | 64.19 |  |
| g5_d_ref | single | completed | LTX-2.5-22B-distilled-transformer.gguf | 45.15 | 50.2 | 6952 | 8288.0 | 56.36 |  |
| g5_e_ref | single | completed | redgraftLTX25Fast2K_ltx25RedgraftNSFW-Q6_K.gguf | 52.69 | 55.2 | 6952 | 8291.0 | 62.79 |  |
| g6_23_fp8 | single | completed | sulphur_distil_fp8mixed.safetensors | 57.81 | 60.2 | 8442 | 10152.0 | 75.24 |  |
| g6_23_fp8_pixar | single | completed | sulphur_distil_fp8mixed.safetensors | 71.18 | 75.2 | 8445 | 9055.0 | 90.09 |  |
| g6_25_fp8 | single | completed | ltx25_uncensored_v1.1-fp8_scaled.safetensors | 48.09 | 50.2 | 6952 | 8351.0 | 66.37 |  |

失敗 `g4_a`: —

失敗 `g4_b`: —

失敗 `g4_c`: —

失敗 `g4_d`: —

失敗 `g4_e`: —

## keep_resident 2本の同一性（`g4_a`）（--reuse で既存の記録から再判定）

- 判定（映像/音声ストリームMD5一致＋PSNR∞）: **True**
- 映像ストリームMD5一致: True（r1=a109d2d82258… r2=a109d2d82258…）
- 音声ストリームMD5一致: True（r1=cfc3cd3bdab2… r2=cfc3cd3bdab2…）
- 参考（合否には使わない）: ファイル全体SHA256一致: False（r1=348d7b65a99c… r2=293cedf050e5…。mp4コンテナのメタデータ（§3-164のjob_id等）の違いで不一致になり得る）

- PSNR y=inf average=inf / SSIM Y=1.0 All=1.0

## keep_resident 2本の同一性（`g4_b`）

- 判定（映像/音声ストリームMD5一致＋PSNR∞）: **True**
- 映像ストリームMD5一致: True（r1=d219a83c688c… r2=d219a83c688c…）
- 音声ストリームMD5一致: True（r1=8b2fd077b6a8… r2=8b2fd077b6a8…）
- 参考（合否には使わない）: ファイル全体SHA256一致: False（r1=45e213ea638e… r2=83635e75ef27…。mp4コンテナのメタデータ（§3-164のjob_id等）の違いで不一致になり得る）

- PSNR y=inf average=inf / SSIM Y=1.0 All=1.0

## keep_resident 2本の同一性（`g4_c`）

- 判定（映像/音声ストリームMD5一致＋PSNR∞）: **True**
- 映像ストリームMD5一致: True（r1=1e077e69c3e2… r2=1e077e69c3e2…）
- 音声ストリームMD5一致: True（r1=4df6f4dfe8c9… r2=4df6f4dfe8c9…）
- 参考（合否には使わない）: ファイル全体SHA256一致: False（r1=0acdc6d2f0be… r2=a58dba1dbe5d…。mp4コンテナのメタデータ（§3-164のjob_id等）の違いで不一致になり得る）

- PSNR y=inf average=inf / SSIM Y=1.0 All=1.0

## keep_resident 2本の同一性（`g4_d`）

- 判定（映像/音声ストリームMD5一致＋PSNR∞）: **True**
- 映像ストリームMD5一致: True（r1=34eac718c59f… r2=34eac718c59f…）
- 音声ストリームMD5一致: True（r1=45814024615a… r2=45814024615a…）
- 参考（合否には使わない）: ファイル全体SHA256一致: False（r1=e5800f855601… r2=1fb7190de27a…。mp4コンテナのメタデータ（§3-164のjob_id等）の違いで不一致になり得る）

- PSNR y=inf average=inf / SSIM Y=1.0 All=1.0

## keep_resident 2本の同一性（`g4_e`）

- 判定（映像/音声ストリームMD5一致＋PSNR∞）: **True**
- 映像ストリームMD5一致: True（r1=f891d0d81e14… r2=f891d0d81e14…）
- 音声ストリームMD5一致: True（r1=f883f713a4ce… r2=f883f713a4ce…）
- 参考（合否には使わない）: ファイル全体SHA256一致: False（r1=9ddfc240849c… r2=4a28a740ae60…。mp4コンテナのメタデータ（§3-164のjob_id等）の違いで不一致になり得る）

- PSNR y=inf average=inf / SSIM Y=1.0 All=1.0

## 不合格ファイル（ケース `fp8_per_row_scale`）

- ファイル: `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\models\LTX23\Weights\zz_bad_fp8_per_row_scale.safetensors`（10854 バイト）・登録名: zz_bad_fp8_per_row_scale
- active: 前=default 後=default（拒否なら不変のはず）
- ToolError: Error executing tool load_pipeline: MODEL_INCOMPATIBLE: selected model 'transformer/zz_bad_fp8_per_row_scale' failed the compatibility precheck — 量子化 safetensors の検査に不合格: 補助テンソル 'model.diffusion_model.transformer_blocks.0.attn1.to_q.weight_scale' が F32[64] です（方式 fp8_scaled で受理するのは F32 の []／[1] のみ）

## 不合格ファイル（ケース `i8_no_marker`）

- ファイル: `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\models\LTX23\Weights\zz_bad_i8_no_marker.safetensors`（10462 バイト）・登録名: zz_bad_i8_no_marker
- active: 前=default 後=default（拒否なら不変のはず）
- ToolError: Error executing tool load_pipeline: MODEL_INCOMPATIBLE: selected model 'transformer/zz_bad_i8_no_marker' failed the compatibility precheck — 量子化 safetensors の検査に不合格: I8 の重み 'model.diffusion_model.transformer_blocks.0.attn1.to_q.weight' に量子化の印がありません（comfy_quant も __metadata__._quantization_metadata もありません）

## 不合格ファイル（ケース `nvfp4_format`）

- ファイル: `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\models\LTX23\Weights\zz_bad_nvfp4_format.safetensors`（10593 バイト）・登録名: zz_bad_nvfp4_format
- active: 前=default 後=default（拒否なら不変のはず）
- ToolError: Error executing tool load_pipeline: MODEL_INCOMPATIBLE: selected model 'transformer/zz_bad_nvfp4_format' failed the compatibility precheck — 量子化 safetensors の検査に不合格: 量子化の印 format='nvfp4' は未対応です（受理: float8_e4m3fn・float8_e5m2・int8_tensorwise・asym_w4a8_int8・'model.diffusion_model.transformer_blocks.0.attn1.to_q.comfy_quant'）

## 不合格ファイル（ケース `quanto_data_placement`）

- ファイル: `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\models\LTX23\Weights\zz_bad_quanto_data_placement.safetensors`（10454 バイト）・登録名: zz_bad_quanto_data_placement
- active: 前=default 後=default（拒否なら不変のはず）
- ToolError: Error executing tool load_pipeline: MODEL_INCOMPATIBLE: selected model 'transformer/zz_bad_quanto_data_placement' failed the compatibility precheck — 量子化 safetensors の検査に不合格: I8 を置けるのは .comfy_quant・.weight だけです（'model.diffusion_model.transformer_blocks.0.attn1.to_q._data'）

## 不合格ファイル（ケース `w4a8_no_codebook`）

- ファイル: `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\models\LTX23\Weights\zz_bad_w4a8_no_codebook.safetensors`（16266 バイト）・登録名: zz_bad_w4a8_no_codebook
- active: 前=default 後=default（拒否なら不変のはず）
- ToolError: Error executing tool load_pipeline: MODEL_INCOMPATIBLE: selected model 'transformer/zz_bad_w4a8_no_codebook' failed the compatibility precheck — 量子化 safetensors の検査に不合格: 'model.diffusion_model.transformer_blocks.0.attn1.to_q'（方式 w4a8）に補助テンソル ['weight_codebook'] がありません

## 不合格ファイル（ケース `weight_scale_wrong_shape`）

- ファイル: `S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\models\LTX23\Weights\zz_bad_weight_scale_wrong_shape.safetensors`（11277 バイト）・登録名: zz_bad_weight_scale_wrong_shape
- active: 前=default 後=default（拒否なら不変のはず）
- ToolError: Error executing tool load_pipeline: MODEL_INCOMPATIBLE: selected model 'transformer/zz_bad_weight_scale_wrong_shape' failed the compatibility precheck — 量子化 safetensors の検査に不合格: 補助テンソル 'model.diffusion_model.transformer_blocks.0.attn1.to_q.weight_scale' が F32[64, 2] です（方式 int8 で受理するのは F32 の []／[1]／[64, 1] のみ）

## PSNR／SSIM（単独実行分）

| tag | a | b | PSNR y | PSNR average | SSIM Y | SSIM All |
|---|---|---|---|---|---|---|
| g3_a_pixar_vs_single | S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\outputs\2d25092a-fb92-4e49-b68e-e1663cac53a6\output.mp4 | S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\outputs\d05677f0-ba8c-49b0-b0fe-132ea766f25a\output.mp4 | 16.116453 | 17.817485 | 0.631603 | 0.725316 |
| g3_b_pixar_vs_single | S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\outputs\77e0eb60-9975-4b3a-b445-fdbc94478bd7\output.mp4 | S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\outputs\5af6f005-4888-413c-91cb-19aea1052247\output.mp4 | 17.163508 | 18.85809 | 0.656705 | 0.745134 |
| g3_c_pixar_vs_single | S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\outputs\5494ca2a-ee13-4b34-8b97-baa338f6ef25\output.mp4 | S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\outputs\681dd22a-9675-4395-a588-91046e488318\output.mp4 | 15.337913 | 17.033764 | 0.611523 | 0.7101 |
| g3_d_pixar_vs_single | S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\outputs\fc25c393-71c7-4562-b994-95d24a837cc8\output.mp4 | S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\outputs\0f7b128c-dc1f-40b7-9f21-ce753b634e4c\output.mp4 | 20.411083 | 21.82168 | 0.733454 | 0.788663 |
| g3_e_pixar_vs_single | S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\outputs\1de121be-dba8-4f12-96c2-2b19e61ac8ea\output.mp4 | S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\outputs\8eaf613c-c77e-4a9f-a903-55958d82cc6d\output.mp4 | 18.363675 | 20.065444 | 0.732593 | 0.803401 |
| g4_a_psnr_ssim | S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\outputs\0c153e4b-c2e7-4726-bd7e-cf8cbc45e0a8\output.mp4 | S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\outputs\f6e908c6-b926-430a-aea5-9237ecab9dfe\output.mp4 | inf | inf | 1.0 | 1.0 |
| g4_b_psnr_ssim | S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\outputs\4a0616b3-0df1-435f-b03a-a8e93600a187\output.mp4 | S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\outputs\ed6e8ccb-1b4d-4217-a45f-62a08e7f173c\output.mp4 | inf | inf | 1.0 | 1.0 |
| g4_c_psnr_ssim | S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\outputs\e808534c-7b5f-4bc4-91cc-a8561a649d5d\output.mp4 | S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\outputs\3d39a734-a870-424a-af40-769a246056e8\output.mp4 | inf | inf | 1.0 | 1.0 |
| g4_d_psnr_ssim | S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\outputs\9aa26caa-63c1-436a-aaa6-d4f88b9b1568\output.mp4 | S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\outputs\02ea2fbf-77a5-41e0-b9de-4a8ea1f1442b\output.mp4 | inf | inf | 1.0 | 1.0 |
| g4_e_psnr_ssim | S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\outputs\dc21e661-4c31-4315-9782-5114144d98dd\output.mp4 | S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\outputs\d02cd292-a578-4fe8-b4d9-fd207d348fa1\output.mp4 | inf | inf | 1.0 | 1.0 |
| g5_a_vs_ref | S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\outputs\d05677f0-ba8c-49b0-b0fe-132ea766f25a\output.mp4 | S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\outputs\8dbc7ce8-6fc1-4380-9e4e-dd3376b60741\output.mp4 | 20.813182 | 22.535913 | 0.749469 | 0.81662 |
| g5_b_vs_ref | S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\outputs\5af6f005-4888-413c-91cb-19aea1052247\output.mp4 | S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\outputs\7dd19030-2d45-4103-b64e-903d961d8b89\output.mp4 | 20.084792 | 21.799525 | 0.713577 | 0.789666 |
| g5_c_vs_ref | S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\outputs\681dd22a-9675-4395-a588-91046e488318\output.mp4 | S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\outputs\6969dd75-49c6-4f37-b90a-54e9c211e0fa\output.mp4 | 18.752408 | 20.47684 | 0.692464 | 0.775226 |
| g5_d_vs_ref | S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\outputs\0f7b128c-dc1f-40b7-9f21-ce753b634e4c\output.mp4 | S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\outputs\693bff44-a189-4943-8878-3a0d54b2dbf7\output.mp4 | 23.693864 | 25.266228 | 0.804847 | 0.848697 |
| g5_e_vs_ref | S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\outputs\8eaf613c-c77e-4a9f-a903-55958d82cc6d\output.mp4 | S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni\outputs\e6698dad-c17a-410a-b9fa-2dbc90aa88fe\output.mp4 | 30.68107 | 32.380374 | 0.939425 | 0.956013 |

