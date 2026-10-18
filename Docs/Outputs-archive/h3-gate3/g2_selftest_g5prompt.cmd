@echo off
rem 3-G2 retake: H3 selftest T2V with the G5 prompt (the prompt the reference
rem fingerprint 0da6111c... was taken with). Run from the repository root.
set PYTHONPATH=%CD%
.venv-engine-h3\Scripts\python.exe -m engine_h3.worker --selftest-generate ^
  --transformer models/MiniMaxH3/Weights/minimax_h3_fl2va_pruned_int8_convrot.safetensors ^
  --text-encoder models/MiniMaxH3/TextEncoder/qwen3vl_32b_minimax_h3_int8_convrot.safetensors ^
  --video-vae models/MiniMaxH3/VAE/minimax_h3_video_vae_fp16.safetensors ^
  --audio-vae models/MiniMaxH3/VAE/minimax_h3_audio_vae_fp32.safetensors ^
  --tokenizer-dir models/MiniMaxH3/TextEncoder/qwen3vl_tokenizer ^
  --prompt "A woman in a yellow raincoat stands on a wet city street at night, neon signs reflecting in puddles. She looks at the camera and says <d>[English] It finally stopped raining.</d>" ^
  --output outputs\h3-gate3\g2_t2v_g5prompt.mp4 --report outputs\h3-gate3\g2_t2v_g5prompt.json
