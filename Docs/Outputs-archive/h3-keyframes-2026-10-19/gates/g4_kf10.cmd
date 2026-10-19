@echo off
rem 1-G4: 10 key frames (VRAM peak record), frames of g5_t2v_after_review.mp4 at the same positions, product defaults
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
  --keyframe 0=outputs/h3-keyframes-2026-10-19/kf10_000.png --keyframe 13=outputs/h3-keyframes-2026-10-19/kf10_013.png --keyframe 27=outputs/h3-keyframes-2026-10-19/kf10_027.png --keyframe 41=outputs/h3-keyframes-2026-10-19/kf10_041.png --keyframe 55=outputs/h3-keyframes-2026-10-19/kf10_055.png ^
  --keyframe 68=outputs/h3-keyframes-2026-10-19/kf10_068.png --keyframe 82=outputs/h3-keyframes-2026-10-19/kf10_082.png --keyframe 96=outputs/h3-keyframes-2026-10-19/kf10_096.png --keyframe 110=outputs/h3-keyframes-2026-10-19/kf10_110.png --keyframe 123=outputs/h3-keyframes-2026-10-19/kf10_123.png ^
  --attention-backend sdpa --prefetch --vram16 ^
  --output outputs/h3-keyframes-2026-10-19/g4_kf10.mp4 --report outputs/h3-keyframes-2026-10-19/g4_kf10.json
