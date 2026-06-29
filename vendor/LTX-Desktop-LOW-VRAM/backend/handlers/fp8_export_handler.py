"""Handler for exporting a pre-quantized fp8 transformer checkpoint."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from enum import Enum
from threading import RLock
from typing import TYPE_CHECKING

from handlers.base import StateHandlerBase
from runtime_config.model_download_specs import resolve_model_path
from services.fp8_export_service import (
    fp8_transformer_exists,
    fp8_transformer_path,
    export_fp8_transformer,
)

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig
    from state.app_state_types import AppState

logger = logging.getLogger(__name__)


class Fp8ExportStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


@dataclass
class Fp8ExportState:
    status: Fp8ExportStatus = Fp8ExportStatus.IDLE
    progress: float = 0.0
    error: str | None = None


class Fp8ExportHandler(StateHandlerBase):
    """Manages the one-time export of a pre-quantized fp8 transformer file."""

    def __init__(self, state: AppState, lock: RLock, config: RuntimeConfig) -> None:
        super().__init__(state, lock, config)
        self._export_state = Fp8ExportState()
        self._export_lock = threading.Lock()

    def get_status(self) -> Fp8ExportState:
        with self._export_lock:
            return Fp8ExportState(
                status=self._export_state.status,
                progress=self._export_state.progress,
                error=self._export_state.error,
            )

    def fp8_file_exists(self) -> bool:
        return fp8_transformer_exists(self.models_dir)

    def fp8_file_path(self) -> str:
        return str(fp8_transformer_path(self.models_dir))

    def start_export(self) -> bool:
        """Start export in background thread.  Returns False if already running."""
        with self._export_lock:
            if self._export_state.status == Fp8ExportStatus.RUNNING:
                return False
            self._export_state = Fp8ExportState(
                status=Fp8ExportStatus.RUNNING, progress=0.0
            )

        thread = threading.Thread(target=self._run_export, daemon=True)
        thread.start()
        return True

    def _run_export(self) -> None:
        checkpoint_path = str(
            resolve_model_path(
                self.models_dir, self.config.model_download_specs, "checkpoint"
            )
        )
        output_path = str(fp8_transformer_path(self.models_dir))

        def _progress(fraction: float) -> None:
            with self._export_lock:
                self._export_state.progress = fraction

        try:
            logger.info("FP8 export started: %s -> %s", checkpoint_path, output_path)
            export_fp8_transformer(
                checkpoint_path=checkpoint_path,
                output_path=output_path,
                progress_cb=_progress,
            )
            with self._export_lock:
                self._export_state = Fp8ExportState(
                    status=Fp8ExportStatus.DONE, progress=1.0
                )
            logger.info("FP8 export complete: %s", output_path)
        except Exception as exc:
            logger.error("FP8 export failed: %s", exc, exc_info=True)
            with self._export_lock:
                self._export_state = Fp8ExportState(
                    status=Fp8ExportStatus.ERROR,
                    progress=0.0,
                    error=str(exc),
                )
