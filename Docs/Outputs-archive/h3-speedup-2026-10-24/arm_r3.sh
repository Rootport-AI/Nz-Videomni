#!/usr/bin/env bash
# R3（台帳 §1-105 ① の門 A-8〜A-10）: arm.sh の写し。冷まし方は同じ。足したのは回す直前の Python プロセスの確認（残っていれば止まる）と r3/r3.log への記録だけ。
# 元: outputs/h3-vram16-gates/arm.sh の写し（台帳 §1-105）。違いは計測台・出力先・標準出力の行き先（/dev/null ではなくログ）だけ。
# One arm: cool down (>=30 s, GPU back to idle), then run the bench once. Usage: arm.sh LABEL [bench args...]
set -u
cd "S:/OriginalApps/12_Nz-LTX23-AviUtl2/Nz-Videomni"
G=outputs/h3-speedup-2026-10-24
label="$1"; shift
mkdir -p $G/runs
sleep 30
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' \r')
  [ "$used" -lt 1200 ] && break
  sleep 5
done
echo "[arm] $label idle_used_mib=$used $(date +%T)" | tee -a $G/arms.log $G/r3/r3.log
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > $G/runs/$label.precheck_compute_apps.csv 2>&1
py=$(grep -ci python $G/runs/$label.precheck_compute_apps.csv)
echo "[arm] $label python_procs=$py" | tee -a $G/r3/r3.log
[ "$py" -eq 0 ] || { echo "[arm] $label STOP: a python process is on the GPU" | tee -a $G/r3/r3.log; exit 9; }
PYTHONUTF8=1 ./.venv-engine-h3/Scripts/python.exe $G/measure_h3_speedup.py --label "$label" --runs-dir $G/runs "$@" > $G/runs/$label.console.log 2>&1
rc=$?
echo "[arm] $label exit=$rc $(date +%T)" | tee -a $G/arms.log $G/r3/r3.log
exit $rc
