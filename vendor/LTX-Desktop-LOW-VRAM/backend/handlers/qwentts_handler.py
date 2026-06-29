"""Qwen3-TTS handler — in-process TTS inference via the qwen-tts package.

Install the package into the backend venv before use:
    uv pip install qwen-tts

No WSL, no flash-attn required.  Uses PyTorch SDPA so it works identically
on Windows and Linux.
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

from _routes._errors import HTTPError
from api_types import QwenTTSGenerateRequest, QwenTTSProgressResponse

logger = logging.getLogger(__name__)

QwenTTSStatus = Literal["idle", "running", "complete", "error", "cancelled"]

_MODEL_IDS = {
    ("custom_voice", "1.7b"): "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    ("custom_voice", "0.6b"): "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    ("voice_clone",  "1.7b"): "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    ("voice_clone",  "0.6b"): "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
}

# Built-in speakers for CustomVoice mode (language: [speaker, ...])
BUILTIN_SPEAKERS: dict[str, list[str]] = {
    "English":  ["Ryan", "Aiden"],
    "Chinese":  ["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric"],
    "Japanese": ["Ono_Anna"],
    "Korean":   ["Sohee"],
}
ALL_SPEAKERS: list[str] = [s for speakers in BUILTIN_SPEAKERS.values() for s in speakers]

SUPPORTED_LANGUAGES = [
    "English", "Chinese", "Japanese", "Korean",
    "German", "French", "Russian", "Portuguese", "Spanish", "Italian",
]


@dataclass
class _QwenTTSJob:
    status: QwenTTSStatus = "idle"
    output_path: str | None = None
    error: str | None = None
    log_lines: list[str] = field(default_factory=list)
    cancel_event: threading.Event = field(default_factory=threading.Event)


class QwenTTSHandler:
    """In-process Qwen3-TTS inference with lazy model loading."""

    def __init__(self, outputs_dir: Path) -> None:
        self._outputs_dir = outputs_dir
        self._lock = threading.Lock()
        self._job = _QwenTTSJob()
        # Cached model keyed by (mode, model_size) to avoid reloading
        self._model: object | None = None
        self._loaded_key: tuple[str, str] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_progress(self) -> QwenTTSProgressResponse:
        with self._lock:
            j = self._job
            return QwenTTSProgressResponse(
                status=j.status,
                output_path=j.output_path,
                error=j.error,
                log_tail="\n".join(j.log_lines[-50:]),
            )

    def generate(self, req: QwenTTSGenerateRequest) -> QwenTTSProgressResponse:
        with self._lock:
            if self._job.status == "running":
                raise HTTPError(409, "Qwen TTS generation already in progress")
            self._job = _QwenTTSJob(status="running")

        thread = threading.Thread(target=self._run, args=(req,), daemon=True)
        thread.start()
        return self.get_progress()

    def cancel(self) -> None:
        with self._lock:
            if self._job.status == "running":
                self._job.cancel_event.set()
                self._job.status = "cancelled"

    def unload(self) -> None:
        """Release the cached model to free VRAM."""
        with self._lock:
            if self._model is not None:
                logger.info("[qwentts] Unloading model")
                self._model = None
                self._loaded_key = None
                try:
                    import gc
                    import torch
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

    def get_speakers(self) -> dict[str, list[str]]:
        return BUILTIN_SPEAKERS

    def get_languages(self) -> list[str]:
        return SUPPORTED_LANGUAGES

    # ------------------------------------------------------------------
    # Background worker
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        logger.info("[qwentts] %s", msg)
        with self._lock:
            self._job.log_lines.append(msg)

    def _run(self, req: QwenTTSGenerateRequest) -> None:
        try:
            import torch
            try:
                from qwen_tts import Qwen3TTSModel  # type: ignore[import]
            except ImportError:
                raise RuntimeError(
                    "qwen-tts package not installed. "
                    "Run: uv pip install qwen-tts  (inside the backend venv)"
                )

            model_key = (req.mode, req.model_size)
            model_id = _MODEL_IDS.get(model_key)
            if model_id is None:
                raise ValueError(f"Unknown mode/size combination: {model_key}")

            # Load or reuse cached model
            with self._lock:
                cached = self._model if self._loaded_key == model_key else None

            if cached is None:
                self._log(f"Loading {model_id} (SDPA, bfloat16)…")
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = Qwen3TTSModel.from_pretrained(
                    model_id,
                    device_map=device,
                    dtype=torch.bfloat16,
                    attn_implementation="sdpa",
                )
                with self._lock:
                    self._model = model
                    self._loaded_key = model_key
                self._log("Model loaded.")
            else:
                model = cached
                self._log("Using cached model.")

            # Check for cancellation before heavy inference
            with self._lock:
                if self._job.cancel_event.is_set():
                    return

            self._log(f"Synthesising {len(req.text)} chars ({req.language})…")

            if req.mode == "custom_voice":
                wavs, sr = model.generate_custom_voice(
                    text=req.text,
                    language=req.language,
                    speaker=req.speaker or "Ryan",
                    instruct=req.instruct or "",
                )
            else:
                # voice_clone — requires ref_audio_path + ref_text
                if not req.ref_audio_path:
                    raise ValueError("voice_clone mode requires ref_audio_path")
                wavs, sr = model.generate_voice_clone(
                    text=req.text,
                    language=req.language,
                    ref_audio=req.ref_audio_path,
                    ref_text=req.ref_text or "",
                )

            with self._lock:
                if self._job.cancel_event.is_set():
                    return

            # Write WAV output
            import numpy as np
            from scipy.io import wavfile  # type: ignore[import]

            stem = f"tts_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
            out_path = self._outputs_dir / f"{stem}.wav"

            audio = wavs[0]
            # scipy expects int16 PCM; convert float32 → int16
            if audio.dtype in (np.float32, np.float64):
                audio = np.clip(audio, -1.0, 1.0)
                audio = (audio * 32767).astype(np.int16)

            wavfile.write(str(out_path), sr, audio)
            self._log(f"✓ Output: {out_path}")

            with self._lock:
                self._job.status = "complete"
                self._job.output_path = str(out_path)

        except Exception as exc:
            logger.exception("[qwentts] Unexpected error")
            with self._lock:
                self._job.status = "error"
                self._job.error = str(exc)
