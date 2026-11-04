#!/usr/bin/env bash
# Shared settings for the ledger 1-275 gate scripts (plan smooth-watching-lobster.md §5).  Sourced, not run.
# Paths: bash side /s/..., the Python side gets the Windows form S:\... (absolute, inside this folder).
ROOT=/s/OriginalApps/12_Nz-LTX23-AviUtl2/Nz-Videomni
ROOT_WIN=$(cygpath -w "$ROOT")
W=$ROOT/outputs/h3-te-equal-chunks-2026-11-04
WW=$(cygpath -w "$W")
PY=$ROOT/.venv-engine-h3/Scripts/python.exe
ARCH=$ROOT/Docs/Outputs-archive/h3-te-vram-2026-10-29
T2=$ARCH/t2_te_ab.py
T4=$ARCH/t4_tail_block.py
LOGS=$W/logs
# wp <file>: the Windows absolute path (S:\...) of a file in this folder, for the Python side.
wp() { cygpath -w "$W/$1"; }
# PYTHONUNBUFFERED=1 (as g3_arm.sh) so the tee'd console shows the encode line while the run is going.
export PYTHONPATH="$ROOT_WIN" PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
mkdir -p "$LOGS"
cd "$ROOT" || exit 2

# The state of engine_h3/text_encoder.py: raw sha256 of the working-tree file (CRLF under core.autocrlf=true),
# sha256 with CR removed, git hash-object (clean filter applied) and the HEAD blob id.
# "pre-fix" = hash-object == HEAD blob (the raw sha256 never equals `git show HEAD:` because of CRLF).
te_state() {
  local f=engine_h3/text_encoder.py
  echo "text_encoder.py sha256_raw=$(sha256sum "$f" | cut -d' ' -f1) sha256_lf=$(tr -d '\r' < "$f" | sha256sum | cut -d' ' -f1) hash_object=$(git hash-object "$f") head_blob=$(git rev-parse HEAD:$f)"
}

# run_logged <name> <command...>: one condition = one process.  Console -> logs/<name>.console.log (tee);
# start/end time, the command, the text_encoder.py state and the exit code -> logs/<name>.meta.log;
# one line per run -> logs/runs.log.
run_logged() {
  local name=$1; shift
  local meta=$LOGS/$name.meta.log con=$LOGS/$name.console.log t0 t1 rc
  t0=$(date +%s)
  {
    echo "name=$name"
    echo "start=$(date '+%F %T')"
    echo "cwd=$PWD"
    printf 'cmd='; printf '[%s] ' "$@"; echo
    te_state
    echo "gpu_used_mib_before=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' \r')"
  } > "$meta"
  "$@" 2>&1 | tee "$con"
  rc=${PIPESTATUS[0]}
  t1=$(date +%s)
  { echo "end=$(date '+%F %T')"; echo "wall_s=$((t1-t0))"; echo "exit=$rc"; } >> "$meta"
  echo "[$(date '+%F %T')] $name exit=$rc wall_s=$((t1-t0))" | tee -a "$LOGS/runs.log"
  return $rc
}

# The worker command of S1-c and S3-d (plan §5.3 S1-c; TE / VAE / tokenizer as g_i5_t2v_fingerprint.cmd).
# --prefetch is the parser default (BooleanOptionalAction, default True) and is passed explicitly as in the plan.
# --ref-image is action="append": nine times = <Picture 1>..<Picture 9>.
I1080_PROMPT="コスプレイヤーの日本人女性が、こちらに向かって歩いてくる。彼女はこちらに片手を伸ばして微笑む。"
I1080_REF=outputs/comfort-calib-h3-2026-11-04/materials/large-image.jpg
i1080_cmd() {  # i1080_cmd <pre|post>: sets the array CMD
  local tag=$1 refs=() i
  for i in 1 2 3 4 5 6 7 8 9; do refs+=(--ref-image "$I1080_REF"); done
  CMD=("$PY" -m engine_h3.worker --selftest-generate
    --transformer models/MiniMaxH3/Weights/minimax_h3_ref2va_pruned_int8_convrot.safetensors
    --text-encoder models/MiniMaxH3/TextEncoder/qwen3vl_32b_minimax_h3_int8_convrot.safetensors
    --video-vae models/MiniMaxH3/VAE/minimax_h3_video_vae_fp16.safetensors
    --audio-vae models/MiniMaxH3/VAE/minimax_h3_audio_vae_fp32.safetensors
    --tokenizer-dir models/MiniMaxH3/TextEncoder/qwen3vl_tokenizer
    --width 1920 --height 1088 --num-frames 56 --steps 4 --seed 12345
    --attention-backend sdpa --vram16 --prefetch
    --prompt "$I1080_PROMPT" "${refs[@]}"
    --rounds 1 --output "$(wp ${tag}_i1080.mp4)" --report "$(wp ${tag}_i1080.json)")
}

# After a worker run: the tokens line and the mp4 sha256 (sha256sum, beside the report's own sha256).
i1080_after() {  # i1080_after <pre|post> <logname>
  local tag=$1 name=$2 f=$W/${1}_i1080.mp4
  echo "--- grep 'encode: N text tokens' in logs/$name.console.log"
  grep -n 'encode: [0-9]* text tokens' "$LOGS/$name.console.log" || echo "<no encode line>"
  if grep -q 'encode: 18462 text tokens' "$LOGS/$name.console.log"; then echo "TOKENS_OK 18462"
  else echo "TOKENS_NG (not 18462 -> stop, plan §9)"; fi
  echo "sha256sum ${tag}_i1080.mp4 = $(sha256sum "$f" 2>/dev/null | cut -d' ' -f1)" | tee "$W/${tag}_i1080.mp4.sha256.txt"
}
