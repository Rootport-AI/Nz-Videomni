@echo off
rem 2-G1: g5_mid_only (middle key frame), product defaults (vram16 on, prefetch on, sdpa)
rem Run from the repository root (Nz-Videomni). GPU. One at a time.
set PYTHONPATH=%CD%
set PYTHONUTF8=1
.venv-engine-h3\Scripts\python.exe -m engine_h3.worker --selftest-generate ^
  --transformer models/MiniMaxH3/Weights/minimax_h3_fl2va_pruned_int8_convrot.safetensors ^
  --text-encoder models/MiniMaxH3/TextEncoder/qwen3vl_32b_minimax_h3_int8_convrot.safetensors ^
  --video-vae models/MiniMaxH3/VAE/minimax_h3_video_vae_fp16.safetensors ^
  --audio-vae models/MiniMaxH3/VAE/minimax_h3_audio_vae_fp32.safetensors ^
  --tokenizer-dir models/MiniMaxH3/TextEncoder/qwen3vl_tokenizer ^
  --prompt "A woman in a yellow raincoat stands on a wet city street at night, neon signs reflecting in puddles. She looks at the camera and says <d>[English] It finally stopped raining.</d>" ^
  --keyframe 62=outputs/h3-keyframes-2026-10-19/kf_mid62.png --attention-backend sdpa --prefetch --vram16 ^
  --output outputs/h3-keyframes-2026-10-19/g5_mid_only.mp4 --report outputs/h3-keyframes-2026-10-19/g5_mid_only.json
