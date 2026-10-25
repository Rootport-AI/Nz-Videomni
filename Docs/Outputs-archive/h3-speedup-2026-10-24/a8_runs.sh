#!/usr/bin/env bash
# A-8（R3）: 前（--no-qkv-share）→後→前→後。条件は arm_n3.sh と同じ（I2V 896x1152x124・sage・vram16・常駐・決定性・先読み・blocks-on-gpu 8・3 ステップ・seed 42・f0_prompt.txt）。
cd "S:/OriginalApps/12_Nz-LTX23-AviUtl2/Nz-Videomni/outputs/h3-speedup-2026-10-24"
COMMON=(--width 896 --height 1152 --num-frames 124 --attention-backend sage --blocks-on-gpu 8 --steps 3
  --image "S:/OriginalApps/12_Nz-LTX23-AviUtl2/Nz-Videomni/uploads/71c540dc-970b-470e-aa0e-856dce76dfdc/input.png"
  --prompt-file "S:/OriginalApps/12_Nz-LTX23-AviUtl2/Nz-Videomni/outputs/h3-vram16-gates/f0_prompt.txt"
  --keep-resident --deterministic --vram16 --vram16-head-group 8 --vram16-row-chunk 4096 --vram16-blocks 4)
for tag in ${@:-pre1 post1 pre2 post2}; do
  extra=(); case $tag in pre*) extra=(--no-qkv-share);; esac
  ./arm_r3.sh R3_A8_${tag}_h8_r4096_b4 "${COMMON[@]}" "${extra[@]}" || exit $?
done
