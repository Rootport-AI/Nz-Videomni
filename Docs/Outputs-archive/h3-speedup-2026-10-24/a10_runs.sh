#!/usr/bin/env bash
# A-10（R3・門の外の記録）: 長尺 T2V 1280x768x311（S=89,388）。outputs/h3-vram16-gates/g4.sh の 2 行目の写しで、違いは --steps 3 と前の腕の --no-qkv-share だけ。
cd "S:/OriginalApps/12_Nz-LTX23-AviUtl2/Nz-Videomni/outputs/h3-speedup-2026-10-24"
COMMON=(--width 1280 --height 768 --num-frames 311 --attention-backend sage --blocks-on-gpu 8 --steps 3 --keep-resident --deterministic --vram16)
for tag in ${@:-pre post}; do
  extra=(); case $tag in pre*) extra=(--no-qkv-share);; esac
  ./arm_r3.sh R3_A10_T2V1280x768x311_${tag} "${COMMON[@]}" "${extra[@]}" || exit $?
done
