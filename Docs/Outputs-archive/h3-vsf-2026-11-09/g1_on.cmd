@echo off
rem Ledger 1-127 gate G1 (VSF on, small shape), VRAM 16GB mode ON.
rem Same selftest as g0_on.cmd plus --width 512 --height 288 --num-frames 56
rem --negative-prompt "blurry, low quality" --vsf-scale 3 (the two flags are added to engine_h3.worker by this task).
rem pass = g1_on and g1_off give the same mp4 sha256 and the same rounds[0].done.h3.vsf (m_neg / m_text).
rem Run from the repository root (AviUtl2 closed, no other GPU process).
set PYTHONPATH=%CD%
set PYTHONUTF8=1
if not exist Docs\Outputs-archive\h3-vsf-2026-11-09\g1_on mkdir Docs\Outputs-archive\h3-vsf-2026-11-09\g1_on
.venv-engine-h3\Scripts\python.exe -m engine_h3.worker --selftest-generate ^
  --transformer models/MiniMaxH3/Weights/minimax_h3_fl2va_pruned_int8_convrot.safetensors ^
  --text-encoder models/MiniMaxH3/TextEncoder/qwen3vl_32b_minimax_h3_int8_convrot.safetensors ^
  --video-vae models/MiniMaxH3/VAE/minimax_h3_video_vae_fp16.safetensors ^
  --audio-vae models/MiniMaxH3/VAE/minimax_h3_audio_vae_fp32.safetensors ^
  --tokenizer-dir models/MiniMaxH3/TextEncoder/qwen3vl_tokenizer ^
  --width 512 --height 288 --num-frames 56 --seed 42 --steps 20 ^
  --shift-video 12 --shift-audio 3 --attention-backend sdpa --vram16 ^
  --negative-prompt "blurry, low quality" --vsf-scale 3 ^
  --prompt "A woman in a yellow raincoat stands on a wet city street at night, neon signs reflecting in puddles. She looks at the camera and says <d>[English] It finally stopped raining.</d>" ^
  --output Docs\Outputs-archive\h3-vsf-2026-11-09\g1_on\g1_on.mp4 ^
  --report Docs\Outputs-archive\h3-vsf-2026-11-09\g1_on\g1_on.json ^
  > Docs\Outputs-archive\h3-vsf-2026-11-09\g1_on\g1_on.console.log 2>&1
echo exit=%ERRORLEVEL%
