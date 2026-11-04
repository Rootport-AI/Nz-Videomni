#!/usr/bin/env bash
# S0 (plan §5.3): read-only checks before S1.  Writes git_state_pre.txt (and logs/s0_precheck.log).
# Exit 0 = every check OK; 1 = at least one NG (stop and report to the supervisor).
source "/s/OriginalApps/12_Nz-LTX23-AviUtl2/Nz-Videomni/outputs/h3-te-equal-chunks-2026-11-04/common.sh"
OUT=$W/git_state_pre.txt
EXPECT_HEAD=7669e01
ng=0
check() { if [ "$1" = 0 ]; then echo "OK  $2"; else echo "NG  $2"; ng=1; fi; }
{
  echo "S0 precheck $(date '+%F %T')"
  # WDDM lists desktop apps that only draw (used_memory [N/A]); the supervisor's rule (2026-11-04):
  # no python.exe in the list, no process with a numeric used_memory, and the whole GPU below 1,200 MiB (as g3_arm.sh).
  echo "--- nvidia-smi compute-apps list (WDDM: drawing-only apps appear with [N/A])"
  apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | tr -d '\r')
  echo "${apps:-<none>}"
  ! printf '%s\n' "$apps" | grep -qi 'python'; check $? "no python.exe in the compute-apps list"
  ! printf '%s\n' "$apps" | awk -F', ' '{print $NF}' | grep -q '[0-9]'; check $? "no process with a numeric used_memory"
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' \r')
  echo "gpu_used_mib=$used"
  [ "$used" -lt 1200 ]; check $? "GPU memory.used < 1200 MiB (idle level)"
  echo "--- aviutl2 (expect none)"
  av=$(tasklist | tr -d '\r' | grep -i aviutl2)
  echo "${av:-<none>}"
  [ -z "$av" ]; check $? "aviutl2 not running"
  echo "--- git"
  head=$(git rev-parse HEAD); echo "HEAD=$head"
  case $head in $EXPECT_HEAD*) r=0 ;; *) r=1 ;; esac; check $r "HEAD is $EXPECT_HEAD"
  por=$(git status --porcelain); echo "porcelain_lines=$(printf '%s' "$por" | grep -c .)"; [ -n "$por" ] && echo "$por"
  [ -z "$por" ]; check $? "git status --porcelain is empty"
  echo "--- engine_h3/text_encoder.py (pre-fix = same as HEAD)"
  te_state
  echo "sha256_head_blob=$(git show HEAD:engine_h3/text_encoder.py | sha256sum | cut -d' ' -f1)"
  echo "note: core.autocrlf=$(git config --get core.autocrlf); the working tree has CRLF, so sha256_raw != sha256_head_blob even when unchanged; judged by sha256_lf and hash_object"
  [ "$(tr -d '\r' < engine_h3/text_encoder.py | sha256sum | cut -d' ' -f1)" = "$(git show HEAD:engine_h3/text_encoder.py | sha256sum | cut -d' ' -f1)" ]
  check $? "text_encoder.py sha256 (CR removed) == sha256 of git show HEAD:engine_h3/text_encoder.py"
  [ "$(git hash-object engine_h3/text_encoder.py)" = "$(git rev-parse HEAD:engine_h3/text_encoder.py)" ]
  check $? "git hash-object text_encoder.py == HEAD blob"
  echo "--- the Japanese prompt of S1-c/S3-d reaches Python unchanged (CPU, argv round trip)"
  want_hex=$(printf '%s' "$I1080_PROMPT" | od -An -tx1 | tr -d '[:space:]')
  got_hex=$("$PY" -c 'import sys; print(sys.argv[1].encode("utf-8").hex())' "$I1080_PROMPT" | tr -d '[:space:]')
  echo "prompt_chars_python=$("$PY" -c 'import sys; print(len(sys.argv[1]))' "$I1080_PROMPT" | tr -d '[:space:]')"
  [ "$want_hex" = "$got_hex" ]; check $? "prompt bytes identical on the Python side"
  echo "--- fix_v2.patch"
  if [ -f "$W/fix_v2.patch" ]; then
    got=$(sha256sum "$W/fix_v2.patch" | cut -d' ' -f1)
    want=$(grep -oE '[0-9a-fA-F]{64}' "$W/fix_v2.patch.sha256.txt" 2>/dev/null | head -1 | tr 'A-F' 'a-f')
    echo "fix_v2.patch sha256=$got"; echo "fix_v2.patch.sha256.txt=${want:-<missing>}"
    [ -n "$want" ] && [ "$got" = "$want" ]; check $? "fix_v2.patch sha256 matches fix_v2.patch.sha256.txt"
    echo "--- git apply --check fix_v2.patch (read only)"
    git apply --check -v "$(wp fix_v2.patch)" 2>&1; check $? "git apply --check fix_v2.patch"
  else
    check 1 "fix_v2.patch present"
  fi
  echo "S0 result: $([ $ng = 0 ] && echo ALL_OK || echo NG)"
} 2>&1 | tee "$OUT"
cp "$OUT" "$LOGS/s0_precheck_$(date '+%m%d_%H%M%S').log"
grep -q '^S0 result: ALL_OK' "$OUT"
