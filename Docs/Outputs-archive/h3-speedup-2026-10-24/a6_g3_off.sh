#!/usr/bin/env bash
# A-6 (R3): outputs/h3-vram16-gates/g3.sh の写し。off だけ・出力先を r3/ に変更。
cd "S:/OriginalApps/12_Nz-LTX23-AviUtl2/Nz-Videomni"
G=outputs/h3-speedup-2026-10-24/r3
P="A woman in a yellow raincoat stands on a wet city street at night, neon signs reflecting in puddles. She looks at the camera and says <d>[English] It finally stopped raining.</d>"
for arm in off; do
  sleep 30
  for i in $(seq 1 60); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' \r'); [ "$used" -lt 1200 ] && break; sleep 5; done
  extra=(); [ $arm = on ] && extra=(--vram16)
  bash $G/precheck.sh a6_g3_$arm || { echo "python process left"; exit 9; }
  echo "[g3] $arm idle_used_mib=$used $(date +%T)" | tee -a $G/r3.log
  PYTHONUTF8=1 PYTHONPATH=$PWD ./.venv-engine-h3/Scripts/python.exe -m engine_h3.worker --selftest-generate \
    --transformer models/MiniMaxH3/Weights/minimax_h3_fl2va_pruned_int8_convrot.safetensors \
    --text-encoder models/MiniMaxH3/TextEncoder/qwen3vl_32b_minimax_h3_int8_convrot.safetensors \
    --video-vae models/MiniMaxH3/VAE/minimax_h3_video_vae_fp16.safetensors \
    --audio-vae models/MiniMaxH3/VAE/minimax_h3_audio_vae_fp32.safetensors \
    --tokenizer-dir models/MiniMaxH3/TextEncoder/qwen3vl_tokenizer \
    --prompt "$P" "${extra[@]}" \
    --output $G/a6_g3_$arm.mp4 --report $G/a6_g3_$arm.json > $G/a6_g3_$arm.log 2>&1
  echo "[g3] $arm exit=$? $(date +%T) sha256=$(sha256sum $G/a6_g3_$arm.mp4 | cut -c1-64)" | tee -a $G/r3.log
done
