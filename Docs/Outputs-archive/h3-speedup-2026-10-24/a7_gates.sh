#!/usr/bin/env bash
# A-7 (R3, 台帳 §1-105 ①): Docs/Outputs-archive/h3-lora-2026-10-24/gates/gates.sh の写し。変えたのは OUT（出力先）だけ。
# Each run is the H3 worker selftest (sdpa, deterministic off as in the reference, seed 42).
set -euo pipefail
cd "S:/OriginalApps/12_Nz-LTX23-AviUtl2/Nz-Videomni"
export PYTHONPATH="$PWD"
PY=.venv-engine-h3/Scripts/python.exe
OUT=outputs/h3-speedup-2026-10-24/r3/a7_gates
mkdir -p "$OUT"
LORA_DIR=models/MiniMaxH3/LoRA
TURBO="$LORA_DIR/minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors"
STYLE="$LORA_DIR/3dsrx_minimax_h3.safetensors"
MOTION="$LORA_DIR/camera_motion_h3_lora_v1_3000_pruned.safetensors"
G5_PROMPT="A woman in a yellow raincoat stands on a wet city street at night, neon signs reflecting in puddles. She looks at the camera and says <d>[English] It finally stopped raining.</d>"

base() {  # base <name> <extra args...>
  local name="$1"; shift
  local t0=$(date +%s)
  "$PY" -m engine_h3.worker --selftest-generate \
    --transformer models/MiniMaxH3/Weights/minimax_h3_fl2va_pruned_int8_convrot.safetensors \
    --text-encoder models/MiniMaxH3/TextEncoder/qwen3vl_32b_minimax_h3_int8_convrot.safetensors \
    --video-vae models/MiniMaxH3/VAE/minimax_h3_video_vae_fp16.safetensors \
    --audio-vae models/MiniMaxH3/VAE/minimax_h3_audio_vae_fp32.safetensors \
    --tokenizer-dir models/MiniMaxH3/TextEncoder/qwen3vl_tokenizer \
    --prompt "$G5_PROMPT" \
    --output "$OUT/$name.mp4" --report "$OUT/$name.json" "$@" > "$OUT/$name.console.log" 2>&1 || { echo "FAILED $name (see $OUT/$name.console.log)"; tail -30 "$OUT/$name.console.log"; return 1; }
  local t1=$(date +%s)
  echo "$name: $((t1-t0)) s wall; sha256=$(sha256sum "$OUT/$name.mp4" | cut -c1-16)"
}

case "${1:-}" in
  g1) base g1_nolora_off ;;                                   # fingerprint 0da6111c... expected
  g2) base g2_nolora_on --vram16 ;;                           # same fingerprint expected
  g4a) base g4_turbo_640_off  --lora "$TURBO" --steps 4 --shift-video 6 --shift-audio 3 ;;
  g4b) base g4_turbo_640_on   --lora "$TURBO" --steps 4 --shift-video 6 --shift-audio 3 --vram16 ;;
  g4c) base g4_turbo_768p_on  --lora "$TURBO" --steps 4 --shift-video 6 --shift-audio 3 --vram16 --width 1344 --height 768 ;;
  g5a) base g5_style_off      --lora "$STYLE" ;;
  g5b) base g5_style_on       --lora "$STYLE" --vram16 ;;
  g5c) base g5_motion_off     --lora "$MOTION" ;;
  g5d) base g5_motion_on      --lora "$MOTION" --vram16 ;;
  g5e) base g5_mixed_off      --lora "$STYLE" --lora "$MOTION" ;;
  g5f) base g5_mixed_on       --lora "$STYLE" --lora "$MOTION" --vram16 ;;
  g5g) base g5_style_on_noprefetch --lora "$STYLE" --vram16 --no-prefetch ;;
  *) echo "usage: bash gates.sh g1|g2|g4a|g4b|g4c|g5a..g5g"; exit 2 ;;
esac
