@echo off
rem Ledger 1-127 gate G0 (VSF off regression), VRAM 16GB mode ON.
rem Copy of Docs\Outputs-archive\h3-te-vram-2026-10-29\g_i5_t2v_fingerprint.cmd with only the
rem --output / --report / redirect pointed at this folder (g0_on\). Conditions unchanged:
rem fl2va weight, yellow-raincoat prompt, 640x384, 124 frames, 20 steps, seed 42, shift 12/3, sdpa, --vram16.
rem pass = exit 0 and mp4 sha256 0da6111cdca853ce718531d14f6970c78c524d01744b22461c7a0cb482f098d9.
rem Run from the repository root (about 2.5 minutes; AviUtl2 closed, no other GPU process).
set PYTHONPATH=%CD%
set PYTHONUTF8=1
if not exist Docs\Outputs-archive\h3-vsf-2026-11-09\g0_on mkdir Docs\Outputs-archive\h3-vsf-2026-11-09\g0_on
.venv-engine-h3\Scripts\python.exe -m engine_h3.worker --selftest-generate ^
  --transformer models/MiniMaxH3/Weights/minimax_h3_fl2va_pruned_int8_convrot.safetensors ^
  --text-encoder models/MiniMaxH3/TextEncoder/qwen3vl_32b_minimax_h3_int8_convrot.safetensors ^
  --video-vae models/MiniMaxH3/VAE/minimax_h3_video_vae_fp16.safetensors ^
  --audio-vae models/MiniMaxH3/VAE/minimax_h3_audio_vae_fp32.safetensors ^
  --tokenizer-dir models/MiniMaxH3/TextEncoder/qwen3vl_tokenizer ^
  --width 640 --height 384 --num-frames 124 --seed 42 --steps 20 ^
  --shift-video 12 --shift-audio 3 --attention-backend sdpa --vram16 ^
  --prompt "A woman in a yellow raincoat stands on a wet city street at night, neon signs reflecting in puddles. She looks at the camera and says <d>[English] It finally stopped raining.</d>" ^
  --output Docs\Outputs-archive\h3-vsf-2026-11-09\g0_on\g0_on.mp4 ^
  --report Docs\Outputs-archive\h3-vsf-2026-11-09\g0_on\g0_on.json ^
  > Docs\Outputs-archive\h3-vsf-2026-11-09\g0_on\g0_on.console.log 2>&1
echo exit=%ERRORLEVEL%
