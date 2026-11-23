| 項目 | 製品の既定（出どころ） | この実験の値（a / b / c / d / e） | 違う場合の理由 |
|---|---|---|---|
| seed | -1（config.yaml generation_defaults.seed（-1 = 毎回ランダム）） | a: 1234 / b: 1234 / c: 1234 / d: 1234 / e: 1234 | 同じ潜在をすべての復号器で比べるため、種を固定する（pi-H3 の実験と同じ 1234） |
| steps | 20（engine_h3/constants.py DEFAULT_STEPS・services/engines/h3/adapter.py H3_DEFAULT_STEPS） | a: 4 / b: 4 / c: 20 / d: 4 / e: 4 | 計画 5.3: Turbo LoRA の対照の条件（記録簿 §223.12）は 4 ステップ。c だけ操作パネルの既定の 20 |
| shift_video | 12.0（engine_h3/constants.py DEFAULT_SHIFT_VIDEO） | a: 12.0 / b: 12.0 / c: 12.0 / d: 12.0 / e: 12.0 | 同じ |
| shift_audio | 3.0（engine_h3/constants.py DEFAULT_SHIFT_AUDIO） | a: 3.0 / b: 3.0 / c: 3.0 / d: 3.0 / e: 3.0 | 同じ |
| width | 1280（config.yaml generation_defaults.width） | a: 1280 / b: 1280 / c: 1280 / d: 896 / e: 640 | 計画 5.3: d は素材のイラスト（896×1152）の大きさ、e は連結の実験の 640×384 |
| height | 768（config.yaml generation_defaults.height） | a: 768 / b: 768 / c: 768 / d: 1152 / e: 384 | 計画 5.3: d は素材のイラスト（896×1152）の大きさ、e は連結の実験の 640×384 |
| num_frames | 345（config.yaml generation_defaults.num_frames） | a: 124 / b: 124 / c: 345 / d: 124 / e: 124 x 2 | 計画 5.3: ComfyUI の実測の条件（124）と比べやすくし GPU の時間を抑える。c は既定の 345 |
| frame_rate | 24（engine_h3/constants.py FRAME_RATE） | a: 24 / b: 24 / c: 24 / d: 24 / e: 24 | 同じ |
| loras | なし（操作パネルの既定（LoRA なし）） | a: Turbo LoRA minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors 0.8 / b: Turbo LoRA minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors 0.8 / c: なし / d: Turbo LoRA minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors 0.8 / e: Turbo LoRA minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors 0.8 | 計画 5.3: a・b・d・e は Turbo LoRA 0.8（記録簿 §223.12 の対照の条件）。c は既定のなし |
| negative_prompt | （engine_h3/pipeline_h3.py H3GenerateRequest） | a:  / b:  / c:  / d:  / e:  | 同じ |
| attention_backend | sdpa（webui shell/accelerationSettings.ts ATTENTION_BACKEND_DEFAULT・api/models.py） | a: sdpa / b: sdpa / c: sdpa / d: sdpa / e: sdpa | 同じ |
| vram16_mode | True（webui VRAM16_MODE_SERVER_DEFAULT・api/models.py VRAM16_MODE_DEFAULT_H3） | a: True / b: True / c: True / d: True / e: True | 同じ |
| block_swap_prefetch | True（webui BLOCK_SWAP_PREFETCH_SERVER_DEFAULT・api/models.py） | a: True / b: True / c: True / d: True / e: True | 同じ |
| keep_resident | False（webui KEEP_RESIDENT_SERVER_DEFAULT・api/models.py） | a: False / b: False / c: False / d: False / e: False | 同じ |
| blocks_on_gpu | 8（services/engines/h3/adapter.py DEFAULT_BLOCKS_ON_GPU） | a: 8 / b: 8 / c: 8 / d: 8 / e: 8 | 同じ |
| deterministic | True（services/engines/h3/adapter.py（読み込みの指示の deterministic）） | a: True / b: True / c: True / d: True / e: True | 同じ |
| 重み・テキストエンコーダ・動画VAE・音声VAE | マニフェストの default_file（scripts/manifests/40-minimaxh3.json） | transformer: minimax_h3_fl2va_pruned_int8_convrot.safetensors / text_encoder: qwen3vl_32b_minimax_h3_int8_convrot.safetensors / video_vae: minimax_h3_video_vae_fp16.safetensors / audio_vae: minimax_h3_audio_vae_fp32.safetensors / tokenizer_dir: qwen3vl_tokenizer | 同じ（計画 5.3: FL2VA 版の INT8 ConvRot） |
| 生成の種類・入力 | T2V（操作パネル） | a: t2v / b: t2v / c: t2v / d: i2v（先頭の画像 00043-2168133585.jpg） / e: chain | 計画 5.3: d は I2V（記録簿 §224.12 の縦長のイラスト）、e は連結 2 本（共通のプロンプト） |
