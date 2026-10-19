@echo off
rem 1-G2: key-frame parts vs DiffSynth (GPU, small). Run from the repository root.
set PYTHONPATH=%CD%
set PYTHONUTF8=1
.venv-engine-h3\Scripts\python.exe outputs/h3-keyframes-2026-10-19/g2_keyframe_parts.py
