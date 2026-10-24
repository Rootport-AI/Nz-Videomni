#!/usr/bin/env bash
# Retake of gate 5 at 768p (owner request 2026-10-24): LoRA-off baseline, Style 0.8, Motion 1.0, mixed 0.8+0.8.
# 1344x768 x 56 frames (2.3 s), 20 steps, shift 12/3, sdpa, VRAM 16GB mode on. Same prompt for all four
# (trigger words at the start: "camera motion" (Motion LoRA) and "3dsrx" (Style LoRA)) so the only difference is the LoRA.
set -euo pipefail
cd "S:/OriginalApps/12_Nz-LTX23-AviUtl2/Nz-Videomni"
export PYTHONPATH="$PWD"
PY=.venv-engine-h3/Scripts/python.exe
OUT=outputs/h3-gate-lora/retake768
mkdir -p "$OUT"
LORA_DIR=models/MiniMaxH3/LoRA
STYLE="$LORA_DIR/3dsrx_minimax_h3.safetensors"
MOTION="$LORA_DIR/camera_motion_h3_lora_v1_3000_pruned.safetensors"
PROMPT="camera motion, slow push-in. 3dsrx, a woman in a yellow raincoat stands on a wet city street at night, neon signs reflecting in puddles, rain has just stopped, she looks around calmly, premium stylized cinematic 3D animation, polished animated-film quality"

base() {  # base <name> <extra args...>
  local name="$1"; shift
  local t0=$(date +%s)
  "$PY" -m engine_h3.worker --selftest-generate \
    --transformer models/MiniMaxH3/Weights/minimax_h3_fl2va_pruned_int8_convrot.safetensors \
    --text-encoder models/MiniMaxH3/TextEncoder/qwen3vl_32b_minimax_h3_int8_convrot.safetensors \
    --video-vae models/MiniMaxH3/VAE/minimax_h3_video_vae_fp16.safetensors \
    --audio-vae models/MiniMaxH3/VAE/minimax_h3_audio_vae_fp32.safetensors \
    --tokenizer-dir models/MiniMaxH3/TextEncoder/qwen3vl_tokenizer \
    --prompt "$PROMPT" --width 1344 --height 768 --num-frames 56 --vram16 \
    --output "$OUT/$name.mp4" --report "$OUT/$name.json" "$@" > "$OUT/$name.console.log" 2>&1 || { echo "FAILED $name"; tail -30 "$OUT/$name.console.log"; return 1; }
  local t1=$(date +%s)
  echo "$name: $((t1-t0)) s wall; sha256=$(sha256sum "$OUT/$name.mp4" | cut -c1-16)"
}

case "${1:-}" in
  none)   base r768_none ;;
  style)  base r768_style_0.8  --lora "$STYLE"  --lora-strength 0.8 ;;
  motion) base r768_motion_1.0 --lora "$MOTION" --lora-strength 1.0 ;;
  mixed)  base r768_mixed_0.8  --lora "$STYLE" --lora "$MOTION" --lora-strength 0.8 ;;
  *) echo "usage: bash gates2.sh none|style|motion|mixed"; exit 2 ;;
esac
