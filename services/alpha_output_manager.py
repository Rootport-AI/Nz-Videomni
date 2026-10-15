"""Alpha Gen exit (第 3 弾): POST /jobs/{job_id}/alpha/rgba | chroma.

Composites a completed Alpha Gen job's window (``_alpha_window.mp4``, with the
source's audio) and its matte (``matte.mkv``) into the deliverables of
Docs/ALPHAGEN_DESIGN.md §10.1, next to the job output:

* ``rgba.mov`` -- the transparent video (ProRes 4444) and ``alpha_preview.mp4``
  -- the same picture over a checkerboard (640 wide). One pair per job.
* ``chroma_<color>.mp4`` -- the chroma-key video filled with one of
  ``api.models.ALPHA_CHROMA_COLORS``. One per colour.

Modelled on :class:`services.join_manager.JoinManager`: synchronous, GPU-free
(ffmpeg only) and independent of the single-GPU-job guard; every material is
checked before any ffmpeg work; ``metadata.json`` is only read; each output is
written under a temporary name and moved into place with ``os.replace``.

Reuse: an output made with the same thresholds (and colour) whose file still
exists is returned as is (``reused: True``). Otherwise its record entry is
dropped first, the file is made again, and the entry is written back last, so
a failure in between never leaves a record that promises the old thresholds.
The record is ``services/alpha_outputs_record.py``.

Two cards of the same job (the job ledger and the history) can POST at the
same time, and the temporary names are fixed, so the work is serialized per
job with a lock.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from api.errors import (
    alpha_output_failed,
    alpha_output_not_ready,
    job_not_alpha,
    job_not_found,
    video_not_ready,
)
from api.models import (
    ALPHA_CHROMA_COLORS,
    AlphaChromaRequest,
    AlphaOutputResponse,
    AlphaRgbaRequest,
    JobStatus,
)
from config import AppConfig
from services import alpha_outputs_record as record_io
from services import video_io
from services.alpha_outputs_record import (
    PREVIEW_FILENAME,
    RGBA_FILENAME,
    AlphaThresholds,
    chroma_filename,
)
from services.job_store import JobStore
from services.recipe_paths import relativize_recipe_paths

WINDOW_FILENAME = "_alpha_window.mp4"
MATTE_FILENAME = "matte.mkv"


class AlphaOutputManager:
    def __init__(self, config: AppConfig, job_store: JobStore) -> None:
        self.config = config
        self.job_store = job_store
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    # ------------------------------------------------------------- resolve

    def _job_dir(self, job_id: str) -> Path:
        return self.config.output_dir / job_id

    def _lock_for(self, job_id: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(job_id, threading.Lock())

    def _existing_file(self, job_id: str, filename: str) -> Path:
        """An output file for the GET routes; raises if the job or file is absent."""
        if self.job_store.get(job_id) is None:
            raise job_not_found(job_id)
        path = self._job_dir(job_id) / filename
        if not path.exists():
            raise alpha_output_not_ready(job_id, detail=f"{filename} not found")
        return path

    def rgba_path(self, job_id: str) -> Path:
        return self._existing_file(job_id, RGBA_FILENAME)

    def preview_path(self, job_id: str) -> Path:
        return self._existing_file(job_id, PREVIEW_FILENAME)

    def chroma_path(self, job_id: str, color: str) -> Path:
        return self._existing_file(job_id, chroma_filename(color))

    def _resolve_materials(self, job_id: str) -> tuple[Path, Path, str, dict]:
        """Validate the job + collect (window, matte, output_fps, metadata).

        Every reject happens before any ffmpeg work (the JoinManager rule).
        """
        record = self.job_store.get(job_id)
        if record is None:
            raise job_not_found(job_id)
        if record.status != JobStatus.completed:
            raise video_not_ready(job_id)

        job_dir = self._job_dir(job_id)
        meta_path = job_dir / "metadata.json"
        if not meta_path.exists():
            raise job_not_alpha(job_id, detail="metadata.json not found for job")
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise job_not_alpha(job_id, detail=f"metadata.json unreadable: {exc}")

        alpha_gen = metadata.get("alpha_gen")
        if not isinstance(alpha_gen, dict):
            raise job_not_alpha(job_id, detail="job has no alpha_gen block")
        output_fps = alpha_gen.get("output_fps")
        if not isinstance(output_fps, str) or not output_fps:
            raise job_not_alpha(job_id, detail="alpha_gen block has no output_fps")

        window = job_dir / WINDOW_FILENAME
        matte = job_dir / MATTE_FILENAME
        if not window.exists() or not matte.exists():
            raise job_not_found(job_id)

        return window, matte, output_fps, metadata

    # -------------------------------------------------------------- outputs

    def rgba(self, job_id: str, request: AlphaRgbaRequest) -> AlphaOutputResponse:
        window, matte, fps, _metadata = self._resolve_materials(job_id)
        wanted = AlphaThresholds(request.transparent_threshold, request.opaque_threshold)
        job_dir = self._job_dir(job_id)

        with self._lock_for(job_id):
            reused = record_io.existing(job_dir).rgba == wanted
            if not reused:
                mov_tmp = job_dir / "_alpha_rgba.tmp.mov"
                preview_tmp = job_dir / "_alpha_preview.tmp.mp4"
                try:
                    current = record_io.read(job_dir)
                    current.rgba = None
                    record_io.write(job_dir, current)
                    video_io.compose_alpha_rgba(
                        window,
                        matte,
                        transparent=wanted.transparent,
                        opaque=wanted.opaque,
                        fps=fps,
                        rgba_mov=mov_tmp,
                        preview_mp4=preview_tmp,
                    )
                    os.replace(mov_tmp, job_dir / RGBA_FILENAME)
                    os.replace(preview_tmp, job_dir / PREVIEW_FILENAME)
                    current.rgba = wanted
                    record_io.write(job_dir, current)
                except (video_io.FFmpegError, OSError) as exc:
                    raise alpha_output_failed(job_id, detail=str(exc)[-1000:])
                finally:
                    mov_tmp.unlink(missing_ok=True)
                    preview_tmp.unlink(missing_ok=True)

        return AlphaOutputResponse(
            job_id=job_id,
            kind="rgba",
            color=None,
            transparent_threshold=wanted.transparent,
            opaque_threshold=wanted.opaque,
            path=f"outputs/{job_id}/{RGBA_FILENAME}",
            reused=reused,
        )

    def chroma(self, job_id: str, request: AlphaChromaRequest) -> AlphaOutputResponse:
        window, matte, fps, metadata = self._resolve_materials(job_id)
        color = request.color
        wanted = AlphaThresholds(request.transparent_threshold, request.opaque_threshold)
        job_dir = self._job_dir(job_id)
        filename = chroma_filename(color)

        with self._lock_for(job_id):
            reused = record_io.existing(job_dir).chroma.get(color) == wanted
            if not reused:
                out_tmp = job_dir / f"_chroma_{color}.tmp.mp4"
                try:
                    current = record_io.read(job_dir)
                    current.chroma.pop(color, None)
                    record_io.write(job_dir, current)
                    video_io.compose_alpha_chroma(
                        window,
                        matte,
                        color_hex=ALPHA_CHROMA_COLORS[color],
                        transparent=wanted.transparent,
                        opaque=wanted.opaque,
                        fps=fps,
                        out_mp4=out_tmp,
                    )
                    # The job's recipe plus what this file is, as its ``comment``
                    # tag (Docs/ALPHAGEN_DESIGN.md §10.1). Same rule as the Join:
                    # unless the job opted out, best effort.
                    if metadata.get("request", {}).get("embed_mp4_metadata", True):
                        recipe = relativize_recipe_paths(
                            {
                                **metadata,
                                "alpha_output": {
                                    "kind": "chroma",
                                    "color": color,
                                    "transparent_threshold": wanted.transparent,
                                    "opaque_threshold": wanted.opaque,
                                },
                            },
                            output_dir=self.config.output_dir,
                            upload_dir=self.config.upload_dir,
                        )
                        video_io.try_embed_comment_tag(out_tmp, video_io.recipe_text(recipe))
                    os.replace(out_tmp, job_dir / filename)
                    current.chroma[color] = wanted
                    record_io.write(job_dir, current)
                except (video_io.FFmpegError, OSError) as exc:
                    raise alpha_output_failed(job_id, detail=str(exc)[-1000:])
                finally:
                    out_tmp.unlink(missing_ok=True)

        return AlphaOutputResponse(
            job_id=job_id,
            kind="chroma",
            color=color,
            transparent_threshold=wanted.transparent,
            opaque_threshold=wanted.opaque,
            path=f"outputs/{job_id}/{filename}",
            reused=reused,
        )
