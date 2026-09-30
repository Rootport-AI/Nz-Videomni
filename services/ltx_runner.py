"""互換shim: 実装はservices/engines/ltx/adapter.pyにある。

outputs/配下の実機ドライバとtests/のテストがこのパスに依存するため互換維持。
"""

from __future__ import annotations

from services.engines.ltx.adapter import (
    MOCK_BACKEND,
    REAL_BACKEND,
    GenerationOutcome,
    LTXRunner,
    ProgressCallback,
    _MockBackend,
    _RealBackend,
    _resolve_reference_preprocess,
    resolve_seed,
)
from services.engines.ltx.adapter import gpu_info, video_io

__all__ = [
    "LTXRunner",
    "resolve_seed",
    "GenerationOutcome",
    "ProgressCallback",
    "MOCK_BACKEND",
    "REAL_BACKEND",
    "_RealBackend",
    "_MockBackend",
    "_resolve_reference_preprocess",
    "video_io",
    "gpu_info",
]
