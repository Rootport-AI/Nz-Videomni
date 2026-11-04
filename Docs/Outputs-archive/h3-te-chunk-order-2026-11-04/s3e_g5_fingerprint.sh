#!/usr/bin/env bash
# S3-e (plan §5.3): the G5 T2V regression fingerprint, the --vram16 arm only, arguments copied from
# Docs/Outputs-archive/h3-te-vram-2026-10-29/g_i5_t2v_fingerprint.cmd (output / report moved here).
# Pass = exit 0 and mp4 sha256 0da6111cdca853ce718531d14f6970c78c524d01744b22461c7a0cb482f098d9.
source "/s/OriginalApps/12_Nz-LTX23-AviUtl2/Nz-Videomni/outputs/h3-te-equal-chunks-2026-11-04/common.sh"
EXPECT=0da6111cdca853ce718531d14f6970c78c524d01744b22461c7a0cb482f098d9
P="A woman in a yellow raincoat stands on a wet city street at night, neon signs reflecting in puddles. She looks at the camera and says <d>[English] It finally stopped raining.</d>"
run_logged s3e_g5_fingerprint "$PY" -m engine_h3.worker --selftest-generate \
  --transformer models/MiniMaxH3/Weights/minimax_h3_fl2va_pruned_int8_convrot.safetensors \
  --text-encoder models/MiniMaxH3/TextEncoder/qwen3vl_32b_minimax_h3_int8_convrot.safetensors \
  --video-vae models/MiniMaxH3/VAE/minimax_h3_video_vae_fp16.safetensors \
  --audio-vae models/MiniMaxH3/VAE/minimax_h3_audio_vae_fp32.safetensors \
  --tokenizer-dir models/MiniMaxH3/TextEncoder/qwen3vl_tokenizer \
  --width 640 --height 384 --num-frames 124 --seed 42 --steps 20 \
  --shift-video 12 --shift-audio 3 --attention-backend sdpa --vram16 \
  --prompt "$P" \
  --output "$(wp g5_on.mp4)" --report "$(wp g5_on.json)"
rc=$?
got=$(sha256sum "$W/g5_on.mp4" 2>/dev/null | cut -d' ' -f1)
{ echo "g5_on.mp4 sha256=$got"; echo "expected        =$EXPECT"
  [ "$got" = "$EXPECT" ] && echo "G5 MATCH" || echo "G5 MISMATCH (stop, plan §9)"; } | tee "$W/g5_on.mp4.sha256.txt" | tee -a "$LOGS/s3e_g5_fingerprint.meta.log"
[ $rc = 0 ] && [ "$got" = "$EXPECT" ]
