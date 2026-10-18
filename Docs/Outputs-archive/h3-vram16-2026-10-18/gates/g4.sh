#!/usr/bin/env bash
cd "S:/OriginalApps/12_Nz-LTX23-AviUtl2/Nz-Videomni/outputs/h3-vram16-gates"
./arm.sh G4_F0eq_I2V896x1152x124_sage_keep_det_vram16 --width 896 --height 1152 --num-frames 124 --attention-backend sage --blocks-on-gpu 8 --steps 3 \
  --image "S:/OriginalApps/12_Nz-LTX23-AviUtl2/Nz-Videomni/uploads/71c540dc-970b-470e-aa0e-856dce76dfdc/input.png" \
  --prompt-file "S:/OriginalApps/12_Nz-LTX23-AviUtl2/Nz-Videomni/outputs/h3-vram16-gates/f0_prompt.txt" --keep-resident --deterministic --vram16
./arm.sh G4_T2V1280x768x311_sage_keep_det_vram16 --width 1280 --height 768 --num-frames 311 --attention-backend sage --blocks-on-gpu 8 --steps 2 --keep-resident --deterministic --vram16
./arm.sh G4_T2V1920x1088x141_sage_keep_det_vram16 --width 1920 --height 1088 --num-frames 141 --attention-backend sage --blocks-on-gpu 8 --steps 2 --keep-resident --deterministic --vram16
