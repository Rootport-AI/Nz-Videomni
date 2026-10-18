#!/usr/bin/env bash
# One arm: cool down (>=30 s, GPU back to idle), then run the bench once. Usage: arm.sh LABEL [bench args...]
set -u
cd "S:/OriginalApps/12_Nz-LTX23-AviUtl2/Nz-Videomni"
G=outputs/h3-vram16-gates
label="$1"; shift
sleep 30
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' \r')
  [ "$used" -lt 1200 ] && break
  sleep 5
done
echo "[arm] $label idle_used_mib=$used $(date +%T)" | tee -a $G/arms.log
PYTHONUTF8=1 ./.venv-engine-h3/Scripts/python.exe outputs/h3-vram-2026-10-18/measure_h3_vram.py --label "$label" --runs-dir $G/runs "$@" > /dev/null 2>&1
rc=$?
echo "[arm] $label exit=$rc $(date +%T)" | tee -a $G/arms.log
exit $rc
