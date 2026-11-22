# MiniMax H3 の混在 int8 の門（記録）

台本 `outputs/h3-mixed-int8-gate-2026-11-22/h3_mixed_int8_gate_api.py`。
R0 は修正の前、R1・X・Y・Z・YL は修正の後に回す。

共通: T2V・1280×768・73 枚・24fps・seed 1234・注意機構 sage・VRAM 16GB mode on・先読み on・常駐 off。

**目安は機械の指標で、合否ではない。最終の合否は人が動画を見て決める。**

| 条件 | 状態 | 自己検査 | 目安 |
|---|---|---|---|
| R0 | completed | 合格 | — |
| R1 | completed | 合格 | — |
| X | completed | 合格 | 要確認 |
| Y | completed | 合格 | 要確認 |
| Z | completed | 合格 | 範囲内 |
| YL | completed | 合格 | — |

## R0

公式 int8 ConvRot（G1 と同じ）。修正の前に回す

- 重み: `minimax_h3_fl2va_pruned_int8_convrot.safetensors`
- ステップ／shift: 20／12・3
- LoRA: なし
- 状態: completed
- git: `c773dd6d27dd`（作業ツリーの変更 0 件）
- 読み込み秒: 17.4
- ジョブ秒: 250.1
- 秒／ステップ: 9.200
- 量子化層: 200（期待 200）
- 置き場: 6 × 371.1 MB
- ノイズ除去の VRAM の山: 3780／4536 MiB（割り当て／予約）
- 映像の MD5: `dc4d54641a648eb359dc74001be3bf5c`
- 音声の MD5: `0717000179fb6bb4cb0f2800d824f490`
- バイト／フレーム: 17297
- ラプラシアンの平均: 5.66
- フレーム差の平均: 3.34
- 音声の RMS: -41.2 dBFS

自己検査:

- status_completed: 合格
- mp4_frames_73_audio: 合格
- transformer_file: 合格
- steps_shift: 合格
- vram16_used: 合格
- quant_backends_flags: 合格
- load_report_found: 合格
- fp8_mode_none: 合格
- quant_layers_expected: 合格
- layers_by_format_vs_header: 合格
- block_swap_ring_logged: 合格
- no_model_incompatible: 合格

## R1

R0 と同じ条件。修正の後に回し、R0 とストリームの MD5 を比べる

- 重み: `minimax_h3_fl2va_pruned_int8_convrot.safetensors`
- ステップ／shift: 20／12・3
- LoRA: なし
- 状態: completed
- git: `c773dd6d27dd`（作業ツリーの変更 4 件）
- 読み込み秒: 2.8
- ジョブ秒: 235.1
- 秒／ステップ: 8.760
- 量子化層: 200（期待 200）
- 置き場: 6 × 371.1 MB
- ノイズ除去の VRAM の山: 3780／4536 MiB（割り当て／予約）
- 映像の MD5: `dc4d54641a648eb359dc74001be3bf5c`
- 音声の MD5: `0717000179fb6bb4cb0f2800d824f490`
- バイト／フレーム: 17297
- ラプラシアンの平均: 5.66
- フレーム差の平均: 3.34
- 音声の RMS: -41.2 dBFS

自己検査:

- status_completed: 合格
- mp4_frames_73_audio: 合格
- transformer_file: 合格
- steps_shift: 合格
- vram16_used: 合格
- quant_backends_flags: 合格
- load_report_found: 合格
- fp8_mode_none: 合格
- quant_layers_expected: 合格
- layers_by_format_vs_header: 合格
- block_swap_ring_logged: 合格
- no_model_incompatible: 合格
- stream_md5_equals_R0: 合格

## X

s3_raw_int8（adaln_proj も int8。最初の関門）

- 重み: `s3_raw_int8.safetensors`
- ステップ／shift: 20／12・3
- LoRA: なし
- 状態: completed
- git: `c773dd6d27dd`（作業ツリーの変更 4 件）
- 読み込み秒: 3.3
- ジョブ秒: 250.1
- 秒／ステップ: 9.645
- 量子化層: 250（期待 250）
- 置き場: 6 × 369.0 MB
- ノイズ除去の VRAM の山: 3768／4616 MiB（割り当て／予約）
- 映像の MD5: `6a0442db2105149ab3ee98f8f156b98f`
- 音声の MD5: `8aa9e65bdbfb7d72ecb46a71c3ab3ebe`
- バイト／フレーム: 32129
- ラプラシアンの平均: 8.93
- フレーム差の平均: 9.24
- 音声の RMS: -37.5 dBFS
- 基準 R1 との比: バイト 1.86・ラプラシアン 1.58・フレーム差 2.76
- 目安: 要確認
  - 要確認: フレーム差が基準の 2.76 倍（目安 2 倍以下）

自己検査:

- status_completed: 合格
- mp4_frames_73_audio: 合格
- transformer_file: 合格
- steps_shift: 合格
- vram16_used: 合格
- quant_backends_flags: 合格
- load_report_found: 合格
- fp8_mode_none: 合格
- quant_layers_expected: 合格
- layers_by_format_vs_header: 合格
- block_swap_ring_logged: 合格
- no_model_incompatible: 合格

## Y

Sulphur-3 keepB0-1-49（ブロック 0・1・49 が BF16）

- 重み: `Sulphur-3_Ref2VA-hybrid_mid_int8_convrot_keepB0-1-49.safetensors`
- ステップ／shift: 20／12・3
- LoRA: なし
- 状態: completed
- git: `c773dd6d27dd`（作業ツリーの変更 4 件）
- 読み込み秒: 3.3
- ジョブ秒: 250.1
- 秒／ステップ: 9.490
- 量子化層: 235（期待 235）
- 置き場: 6 × 738.3 MB
- ノイズ除去の VRAM の山: 5978／6834 MiB（割り当て／予約）
- 映像の MD5: `4efd68cee37bb9075231c604dab1d744`
- 音声の MD5: `9df2d489f86e0963e4e376942751ef46`
- バイト／フレーム: 41344
- ラプラシアンの平均: 10.15
- フレーム差の平均: 10.42
- 音声の RMS: -35.6 dBFS
- 基準 R1 との比: バイト 2.39・ラプラシアン 1.79・フレーム差 3.12
- 目安: 要確認
  - 要確認: 1 フレームあたりのバイト数が基準の 2.39 倍（目安 2 倍以下）
  - 要確認: フレーム差が基準の 3.12 倍（目安 2 倍以下）

自己検査:

- status_completed: 合格
- mp4_frames_73_audio: 合格
- transformer_file: 合格
- steps_shift: 合格
- vram16_used: 合格
- quant_backends_flags: 合格
- load_report_found: 合格
- fp8_mode_none: 合格
- quant_layers_expected: 合格
- layers_by_format_vs_header: 合格
- block_swap_ring_logged: 合格
- no_model_incompatible: 合格

## Z

10Eros 8stepTURBO int8HQ（ブロック 0・1・47〜49 が BF16）

- 重み: `10Eros_Max_h3_8stepTURBO_beta7_int8HQ.safetensors`
- ステップ／shift: 8／10・3
- LoRA: なし
- 状態: completed
- git: `c773dd6d27dd`（作業ツリーの変更 4 件）
- 読み込み秒: 3.3
- ジョブ秒: 140.0
- 秒／ステップ: 9.762
- 量子化層: 180（期待 180）
- 置き場: 6 × 738.3 MB
- ノイズ除去の VRAM の山: 5978／6834 MiB（割り当て／予約）
- 映像の MD5: `703d0dbc69123706eff6bbd3a2f73e23`
- 音声の MD5: `bac88ca0c9aa99bc47a92cf4b678c751`
- バイト／フレーム: 20166
- ラプラシアンの平均: 6.58
- フレーム差の平均: 3.82
- 音声の RMS: -26.6 dBFS
- 基準 R1 との比: バイト 1.17・ラプラシアン 1.16・フレーム差 1.14
- 目安: 範囲内

自己検査:

- status_completed: 合格
- mp4_frames_73_audio: 合格
- transformer_file: 合格
- steps_shift: 合格
- vram16_used: 合格
- quant_backends_flags: 合格
- load_report_found: 合格
- fp8_mode_none: 合格
- quant_layers_expected: 合格
- layers_by_format_vs_header: 合格
- block_swap_ring_logged: 合格
- no_model_incompatible: 合格

## YL

Y に Turbo LoRA（強さ 1.0・4 ステップ。G6 と同じ）

- 重み: `Sulphur-3_Ref2VA-hybrid_mid_int8_convrot_keepB0-1-49.safetensors`
- ステップ／shift: 4／12・3
- LoRA: `minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16`（強さ 1.0）
- 状態: completed
- git: `c773dd6d27dd`（作業ツリーの変更 4 件）
- 読み込み秒: 3.4
- ジョブ秒: 110.0
- 秒／ステップ: 12.800
- 量子化層: 235（期待 235）
- 置き場: 6 × 774.2 MB
- ノイズ除去の VRAM の山: 6277／7248 MiB（割り当て／予約）
- 映像の MD5: `6415c4ca1ffea214341c37b0bcfe7059`
- 音声の MD5: `7ee1026b2b8d91c3b881af9b8945712d`
- バイト／フレーム: 43221
- ラプラシアンの平均: 9.79
- フレーム差の平均: 12.76
- 音声の RMS: -35.9 dBFS

自己検査:

- status_completed: 合格
- mp4_frames_73_audio: 合格
- transformer_file: 合格
- steps_shift: 合格
- vram16_used: 合格
- quant_backends_flags: 合格
- load_report_found: 合格
- fp8_mode_none: 合格
- quant_layers_expected: 合格
- layers_by_format_vs_header: 合格
- block_swap_ring_logged: 合格
- lora_applied: 合格
- no_model_incompatible: 合格
