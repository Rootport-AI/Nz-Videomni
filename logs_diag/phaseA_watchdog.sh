#!/usr/bin/env bash
LOG="logs_diag/phaseA_parity_stderr.log"
DEADLINE_GAP=1800
last_ts=$(date +%s)
last_line=""
while true; do
  if grep -qE "RUN_DONE|RUN_FAILED" "$LOG" 2>/dev/null; then
    echo "TERMINAL_STATE_REACHED $(grep -E 'RUN_DONE|RUN_FAILED' "$LOG" | tail -n1)"
    break
  fi
  cur_line=$(tail -n 1 "$LOG" 2>/dev/null)
  if [ "$cur_line" != "$last_line" ]; then
    last_line="$cur_line"
    last_ts=$(date +%s)
  fi
  now=$(date +%s)
  gap=$((now - last_ts))
  if [ "$gap" -gt "$DEADLINE_GAP" ]; then
    echo "WATCHDOG_TIMEOUT gap=${gap}s stalled_at=$last_line"
    pids=$(wmic process where "Name='python.exe' and CommandLine like '%run_spike.py%'" get ProcessId 2>/dev/null | tr -d ' \r' | grep -E '^[0-9]+$')
    for p in $pids; do
      echo "killing PID $p"
      taskkill //PID "$p" //F //T 2>/dev/null
    done
    break
  fi
  sleep 15
done
