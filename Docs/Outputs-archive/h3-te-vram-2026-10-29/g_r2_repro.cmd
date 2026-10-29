@echo off
rem Ledger 1-116 GPU gate G-I4: G-R2 (Ref2VA, 3 pictures + 1 audio, VERIFICATION_LOG 180) rerun with
rem the product TE (blocked math attention + early drop). Same arguments as G-R2 (recorded in VERIFICATION_LOG 182.8);
rem pass = exit 0, no OOM, mp4 sha256 fba02ba1cb473caeaf3df875f548ab6d14864f9e287c80dfc3ef0a79a96ad109.
rem Run from the repository root (about 22 minutes; AviUtl2 closed, no other GPU process).
set PYTHONPATH=%CD%
set PYTHONUTF8=1
.venv-engine-h3\Scripts\python.exe -m engine_h3.worker --selftest-generate ^
  --transformer models/MiniMaxH3/Weights/minimax_h3_ref2va_pruned_int8_convrot.safetensors ^
  --text-encoder models/MiniMaxH3/TextEncoder/qwen3vl_32b_minimax_h3_int8_convrot.safetensors ^
  --video-vae models/MiniMaxH3/VAE/minimax_h3_video_vae_fp16.safetensors ^
  --audio-vae models/MiniMaxH3/VAE/minimax_h3_audio_vae_fp32.safetensors ^
  --tokenizer-dir models/MiniMaxH3/TextEncoder/qwen3vl_tokenizer ^
  --width 1344 --height 768 --num-frames 124 --seed 42 --steps 20 ^
  --shift-video 12 --shift-audio 3 --attention-backend sdpa --vram16 ^
  --ref-image Docs\Outputs-archive\h3-ref2va-2026-10-28\17fde0c1-image.png ^
  --ref-image Docs\Outputs-archive\h3-ref2va-2026-10-28\d80320dd-image.png ^
  --ref-image Docs\Outputs-archive\h3-ref2va-2026-10-28\31d79f30-image.png ^
  --ref-audio outputs\h3-ref2va-2026-10-28\ref_audio_raincoat_speech.wav ^
  --prompt "<Picture 1> is an elderly gentleman. <Picture 2> is a girl in a black gothic dress holding a red paper umbrella. They stand together on the bridge of the spaceship shown in <Picture 3>, looking out at the planet. <Audio 1> is the voice of the gentleman speaking." ^
  --output outputs\h3-te-vram-2026-10-29\g_r2_repro.mp4 ^
  --report Docs\Outputs-archive\h3-te-vram-2026-10-29\g_r2_repro.json ^
  > Docs\Outputs-archive\h3-te-vram-2026-10-29\g_r2_repro.console.log 2>&1
echo exit=%ERRORLEVEL%
