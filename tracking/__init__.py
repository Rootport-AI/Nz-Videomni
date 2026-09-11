"""Utility-AI worker package (object tracking).

Runs on the dedicated CPU-only `.venv-utils` interpreter, NOT on the app venv and
NOT on either engine venv -- see README.md and tracking/utils-venv-pyproject.toml.
The only module here that imports torch is `uetrack_runtime`.
"""
