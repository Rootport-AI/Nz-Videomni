# MiniMax H3 の fp8・w4a8 の GPU の門（G1〜G6。API 経由。記録）

台本 `outputs/h3-fp8-w4a8-gate-2026-11-12/h3_quant_gate_api.py`。計画 `imperative-crafting-lightning.md` §5.2。G7（Settings の拒否の理由文）は手で確かめる。

| 門 | 条件 | 状態 | 読み込み秒 | ジョブ秒（壁） | 生成秒 | 秒／ステップ（中央値） | スパイク S3 の秒／ステップ | VRAM の山 割り当て／予約（MiB） | fp8_mode | 形式ごとの層の数 | 自己検査 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G1 | `int8` | completed | 16.1 | 235.1 | 234.4 | 8.760（8.700） | 8.785 | 3780／4536 | — | {"int8_tensorwise": 200} | すべて合格 |
| G2 | `fp8` | completed | 3.2 | 290.1 | 288.1 | 11.725（11.700） | 11.753 | 3773／4684 | scaled | {"float8_e4m3fn": 200} | すべて合格 |
| G3 | `w4a8_kijai` | completed | 2.9 | 245.0 | 241.0 | 9.625（9.600） | 9.652 | 2810／3638 | — | {"asym_w4a8_int8": 200} | すべて合格 |
| G4 | `w4a8_eros` | completed | 2.9 | 125.0 | 120.5 | 9.575（9.500） | 9.608 | 3780／4536 | — | {"asym_w4a8_int8": 180, "int8_tensorwise": 20} | すべて合格 |
| G5 | `int8_oldmarker` | completed | 2.9 | 125.0 | 120.6 | 8.762（8.700） | — | 3780／4536 | — | {"int8_tensorwise": 200} | すべて合格 |
| G5b | `int8_oldmarker_s20` | completed | 3.0 | 235.1 | 231.7 | 8.815（8.800） | — | 3780／4536 | — | {"int8_tensorwise": 200} | すべて合格 |
| G6 | `fp8_turbo` | completed | 3.0 | 120.0 | 116.0 | 15.125（15.100） | — | 4072／5096 | scaled | {"float8_e4m3fn": 200} | すべて合格 |

秒／ステップはサーバーのログの `denoise k/N steps (..., Ts elapsed)` から（N ステップ目の経過秒 ÷ N。スパイクの StepClock と同じ定義）。fp8_mode と層の数はワーカーのログの `LOAD_OK` の行（`build_report`）から。

## G1 `int8`

| 検査 | 値 | 期待 | 合否 |
|---|---|---|---|
| status_completed | `"completed"` | — | 合格 |
| mp4_frames_73_audio | `[1280, 768, 73, "aac"]` | — | 合格 |
| transformer_file | `"minimax_h3_fl2va_pruned_int8_convrot.safetensors"` | `"minimax_h3_fl2va_pruned_int8_convrot.safetensors"` | 合格 |
| steps_shift | `[20, [12.0, 3.0]]` | `[20, [12.0, 3.0]]` | 合格 |
| quant_backends_flags | `{"available": true, "disabled": false, "int8_linear": true, "w4a8_int8_linear": true, "quantize_per_tensor_fp8": true}` | — | 合格 |
| load_report_found | `true` | — | 合格 |
| fp8_mode | `null` | `null` | 合格 |
| layers_by_format | `{"int8_tensorwise": 200}` | `{"int8_tensorwise": 200}` | 合格 |

## G2 `fp8`

| 検査 | 値 | 期待 | 合否 |
|---|---|---|---|
| status_completed | `"completed"` | — | 合格 |
| mp4_frames_73_audio | `[1280, 768, 73, "aac"]` | — | 合格 |
| transformer_file | `"minimax_h3_fl2va_pruned_fp8_scaled.safetensors"` | `"minimax_h3_fl2va_pruned_fp8_scaled.safetensors"` | 合格 |
| steps_shift | `[20, [12.0, 3.0]]` | `[20, [12.0, 3.0]]` | 合格 |
| quant_backends_flags | `{"available": true, "disabled": false, "int8_linear": true, "w4a8_int8_linear": true, "quantize_per_tensor_fp8": true}` | — | 合格 |
| load_report_found | `true` | — | 合格 |
| fp8_mode | `"scaled"` | `"scaled"` | 合格 |
| layers_by_format | `{"float8_e4m3fn": 200}` | `{"float8_e4m3fn": 200}` | 合格 |
| no_model_incompatible | `false` | — | 合格 |

## G3 `w4a8_kijai`

| 検査 | 値 | 期待 | 合否 |
|---|---|---|---|
| status_completed | `"completed"` | — | 合格 |
| mp4_frames_73_audio | `[1280, 768, 73, "aac"]` | — | 合格 |
| transformer_file | `"minimax_h3_fl2va_pruned_w4a8_mixed.safetensors"` | `"minimax_h3_fl2va_pruned_w4a8_mixed.safetensors"` | 合格 |
| steps_shift | `[20, [12.0, 3.0]]` | `[20, [12.0, 3.0]]` | 合格 |
| quant_backends_flags | `{"available": true, "disabled": false, "int8_linear": true, "w4a8_int8_linear": true, "quantize_per_tensor_fp8": true}` | — | 合格 |
| load_report_found | `true` | — | 合格 |
| fp8_mode | `null` | `null` | 合格 |
| layers_by_format | `{"asym_w4a8_int8": 200}` | `{"asym_w4a8_int8": 200}` | 合格 |
| no_model_incompatible | `false` | — | 合格 |

## G4 `w4a8_eros`

| 検査 | 値 | 期待 | 合否 |
|---|---|---|---|
| status_completed | `"completed"` | — | 合格 |
| mp4_frames_73_audio | `[1280, 768, 73, "aac"]` | — | 合格 |
| transformer_file | `"10Eros_Max_h3_TURBO-hybrid_beta5_w4a8_14gb_optimized.safetensors"` | `"10Eros_Max_h3_TURBO-hybrid_beta5_w4a8_14gb_optimized.safetensors"` | 合格 |
| steps_shift | `[8, [10.0, 3.0]]` | `[8, [10.0, 3.0]]` | 合格 |
| quant_backends_flags | `{"available": true, "disabled": false, "int8_linear": true, "w4a8_int8_linear": true, "quantize_per_tensor_fp8": true}` | — | 合格 |
| load_report_found | `true` | — | 合格 |
| fp8_mode | `null` | `null` | 合格 |
| layers_by_format | `{"asym_w4a8_int8": 180, "int8_tensorwise": 20}` | `{"asym_w4a8_int8": 180, "int8_tensorwise": 20}` | 合格 |
| no_model_incompatible | `false` | — | 合格 |

## G5 `int8_oldmarker`

| 検査 | 値 | 期待 | 合否 |
|---|---|---|---|
| status_completed | `"completed"` | — | 合格 |
| mp4_frames_73_audio | `[1280, 768, 73, "aac"]` | — | 合格 |
| transformer_file | `"h3ErosMax_beta5_3185154.safetensors"` | `"h3ErosMax_beta5_3185154.safetensors"` | 合格 |
| steps_shift | `[8, [10.0, 3.0]]` | `[8, [10.0, 3.0]]` | 合格 |
| quant_backends_flags | `{"available": true, "disabled": false, "int8_linear": true, "w4a8_int8_linear": true, "quantize_per_tensor_fp8": true}` | — | 合格 |
| load_report_found | `true` | — | 合格 |
| fp8_mode | `null` | `null` | 合格 |
| layers_by_format | `{"int8_tensorwise": 200}` | `{"int8_tensorwise": 200}` | 合格 |
| no_model_incompatible | `false` | — | 合格 |

## G5b `int8_oldmarker_s20`

| 検査 | 値 | 期待 | 合否 |
|---|---|---|---|
| status_completed | `"completed"` | — | 合格 |
| mp4_frames_73_audio | `[1280, 768, 73, "aac"]` | — | 合格 |
| transformer_file | `"h3ErosMax_beta5_3185154.safetensors"` | `"h3ErosMax_beta5_3185154.safetensors"` | 合格 |
| steps_shift | `[20, [12.0, 3.0]]` | `[20, [12.0, 3.0]]` | 合格 |
| quant_backends_flags | `{"available": true, "disabled": false, "int8_linear": true, "w4a8_int8_linear": true, "quantize_per_tensor_fp8": true}` | — | 合格 |
| load_report_found | `true` | — | 合格 |
| fp8_mode | `null` | `null` | 合格 |
| layers_by_format | `{"int8_tensorwise": 200}` | `{"int8_tensorwise": 200}` | 合格 |
| no_model_incompatible | `false` | — | 合格 |

## G6 `fp8_turbo`

| 検査 | 値 | 期待 | 合否 |
|---|---|---|---|
| status_completed | `"completed"` | — | 合格 |
| mp4_frames_73_audio | `[1280, 768, 73, "aac"]` | — | 合格 |
| transformer_file | `"minimax_h3_fl2va_pruned_fp8_scaled.safetensors"` | `"minimax_h3_fl2va_pruned_fp8_scaled.safetensors"` | 合格 |
| steps_shift | `[4, [12.0, 3.0]]` | `[4, [12.0, 3.0]]` | 合格 |
| quant_backends_flags | `{"available": true, "disabled": false, "int8_linear": true, "w4a8_int8_linear": true, "quantize_per_tensor_fp8": true}` | — | 合格 |
| load_report_found | `true` | — | 合格 |
| fp8_mode | `"scaled"` | `"scaled"` | 合格 |
| layers_by_format | `{"float8_e4m3fn": 200}` | `{"float8_e4m3fn": 200}` | 合格 |
| lora_applied | `[{"name": "minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors", "strength": 1.0, "layout": "ab", "dtype": "BF16", "pairs_applied": 208, "skipped": []}]` | — | 合格 |
| no_model_incompatible | `false` | — | 合格 |
