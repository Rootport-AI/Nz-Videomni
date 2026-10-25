#!/usr/bin/env bash
# R3: A-6 off を 1 回、続けて A-7 の on 側 6 本を 1 本ずつ（各回の前に Python プロセスの確認）
cd "S:/OriginalApps/12_Nz-LTX23-AviUtl2/Nz-Videomni/outputs/h3-speedup-2026-10-24"
bash a6_g3_off.sh || exit $?
for g in g2 g4b g5b g5d g5f g5g; do
  for i in $(seq 1 60); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' \r'); [ "$used" -lt 1200 ] && break; sleep 5; done
  bash r3/precheck.sh a7_$g || { echo "python process left before $g" | tee -a r3/r3.log; exit 9; }
  echo "[a7] $g start $(date +%T)" | tee -a r3/r3.log
  bash a7_gates.sh $g 2>&1 | tee -a r3/r3.log || exit 1
done
echo "[a67] done $(date +%T)" | tee -a r3/r3.log
