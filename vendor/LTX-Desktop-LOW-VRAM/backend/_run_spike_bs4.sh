#!/usr/bin/env bash
set -u
export PWD_ROOT="s:/OriginalApps/12_Nz-LTX23-backend"
export UV_PYTHON_INSTALL_DIR="$PWD_ROOT/.python"
export UV_CACHE_DIR="$PWD_ROOT/.uv_cache"
export HF_HOME="$PWD_ROOT/hf_home"
export SPIKE_CPU_TEXT_ENCODE=1
export SPIKE_BLOCKS_ON_GPU=4

cd "$PWD_ROOT/vendor/LTX-Desktop-LOW-VRAM/backend"
LOG="$PWD_ROOT/outputs/forkenv_spike_bs4_cpuenc.log"
echo "=== launching spike bs4 cpuenc at $(date) ===" > "$LOG"
.venv/Scripts/python.exe -u _spike_gguf_measured.py >> "$LOG" 2>&1
echo "=== spike process exited rc=$? at $(date) ===" >> "$LOG"
