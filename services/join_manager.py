"""V2V server-side join orchestration (POST /jobs/{job_id}/join).

Joins a completed V2V job's continuation (``output.mp4`` — the NEW part only,
the engine trims the context off) back onto the user's uploaded source video,
producing ``outputs/{job_id}/joined.mp4``. GPU-free (ffmpeg only) and fully
independent of the single-GPU-job guard, so a join can run while another
generation is in flight.

Design (Docs/mockups/JOIN_API_PROPOSAL.md, approved 2026-07-05):

* materials — the uploads store's original source video
  (``video_upload_store.path_for``), the job's ``output.mp4``, and (when the
  source had audio) the engine's ``<stem>_audio_handle.wav`` sidecar recorded in
  ``metadata.json``'s ``v2v`` block.
* audio_smoothing=True (default) — :func:`services.video_io.join_v2v`: a true
  overlapped equal-power crossfade via the handle sidecar when present, else
  the no-handle fade-pair. audio_smoothing=False — hard concat
  (:func:`services.video_io.concat_mp4s`, no fades; API parity only, the GUI
  does not expose it).
* R3 normalization (smoke-verified 2026-07-05) — ``join_v2v``/``concat`` demand
  matching resolution+fps, but the full uploaded source generally differs from
  the delivered continuation (only the consumed context TAIL was fps-aligned).
  When they differ the source is first re-encoded to the continuation's
  geometry via :func:`services.video_io.normalize_clip` (scale-to-cover +
  centered crop + ``setsar=1`` + fps resample; one extra encode generation is
  the accepted quality cost). The temp file lives in the job dir and is removed
  after the join.
"""

from __future__ import annotations

import json
from pathlib import Path

from api.errors import (
    APIError,
    job_not_found,
    job_not_joinable,
    join_failed,
    joined_not_ready,
    source_video_not_found,
    video_not_ready,
)
from api.models import JobStatus, JoinRequest, JoinResponse
from config import AppConfig
from services import video_io
from services.job_store import JobStore
from services.video_upload_store import VideoUploadStore

JOINED_FILENAME = "joined.mp4"


class JoinManager:
    def __init__(
        self,
        config: AppConfig,
        job_store: JobStore,
        video_upload_store: VideoUploadStore,
    ) -> None:
        self.config = config
        self.job_store = job_store
        self.video_upload_store = video_upload_store

    # ------------------------------------------------------------- resolve

    def _job_dir(self, job_id: str) -> Path:
        return self.config.output_dir / job_id

    def joined_path(self, job_id: str) -> Path:
        """Path of a previously-joined output; raises if absent (GET route)."""
        record = self.job_store.get(job_id)
        if record is None:
            raise job_not_found(job_id)
        path = self._job_dir(job_id) / JOINED_FILENAME
        if not path.exists():
            raise joined_not_ready(job_id)
        return path

    def _resolve_materials(self, job_id: str) -> tuple[Path, Path, dict]:
        """Validate the job + collect (source_video, continuation, v2v_meta).

        Up-front-failure discipline (mirrors api/generate_chain.py): every
        reject happens before any ffmpeg work.
        """
        record = self.job_store.get(job_id)
        if record is None:
            raise job_not_found(job_id)
        if record.status != JobStatus.completed:
            raise video_not_ready(job_id)

        job_dir = self._job_dir(job_id)
        continuation = job_dir / "output.mp4"
        if not continuation.exists():
            raise job_not_found(job_id)

        meta_path = job_dir / "metadata.json"
        if not meta_path.exists():
            raise job_not_joinable(job_id, detail="metadata.json not found for job")
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise job_not_joinable(job_id, detail=f"metadata.json unreadable: {exc}")

        v2v = metadata.get("v2v")
        if not isinstance(v2v, dict):
            raise job_not_joinable(
                job_id,
                detail="job has no v2v block (plain chain / A2V / single generate)",
            )
        source_video_id = v2v.get("source_video_id")
        if not source_video_id:
            raise job_not_joinable(job_id, detail="v2v block has no source_video_id")

        try:
            source = self.video_upload_store.path_for(str(source_video_id))
        except APIError:
            # The upload was purged since generation — its own stable code so
            # the client can tell "re-upload the source" from "wrong job".
            raise source_video_not_found(str(source_video_id))

        return source, continuation, v2v

    # ---------------------------------------------------------------- join

    def join(self, job_id: str, request: JoinRequest) -> JoinResponse:
        source, continuation, v2v = self._resolve_materials(job_id)
        job_dir = self._job_dir(job_id)
        out = job_dir / JOINED_FILENAME

        # R3: normalize the source to the continuation's geometry when needed.
        cont_res = video_io.probe_resolution(continuation)
        cont_fps = video_io.probe_fps(continuation)
        if cont_res is None or cont_fps is None:
            raise join_failed(job_id, detail="could not probe continuation resolution/fps")
        src_res = video_io.probe_resolution(source)
        src_fps = video_io.probe_fps(source)
        needs_norm = src_res != cont_res or (
            src_fps is None or abs(src_fps - cont_fps) > 1e-3
        )

        # Handle sidecar (true crossfade) — only meaningful for the smoothed
        # path and only when the engine actually emitted one (source had audio).
        handle: Path | None = None
        if request.audio_smoothing:
            handle_name = v2v.get("audio_handle_filename")
            if handle_name:
                candidate = job_dir / str(handle_name)
                if candidate.exists():
                    handle = candidate

        norm_tmp = job_dir / "_join_source_norm.mp4"
        source_use = source
        try:
            if needs_norm:
                video_io.normalize_clip(
                    source, norm_tmp, cont_res[0], cont_res[1], cont_fps
                )
                source_use = norm_tmp

            if request.audio_smoothing:
                info = video_io.join_v2v(
                    source_use,
                    continuation,
                    out,
                    handle_audio=handle,
                    handle_crossfade_ms=request.handle_crossfade_ms,
                )
            else:
                # Hard concat (no fades). API parity with §24.7's OFF meaning;
                # not exposed by the GUI.
                video_io.concat_mp4s([source_use, continuation], out, cont_fps)
                info = {"join_mode": "hard_concat"}
        except video_io.FFmpegError as exc:
            raise join_failed(job_id, detail=str(exc)[-1000:])
        finally:
            norm_tmp.unlink(missing_ok=True)

        return JoinResponse(
            job_id=job_id,
            joined_path=f"outputs/{job_id}/{JOINED_FILENAME}",
            join_mode=str(info.get("join_mode", "hard_concat")),
            source_normalized=needs_norm,
            source_lufs=info.get("source_lufs"),
            continuation_lufs_before=info.get("continuation_lufs_before"),
            fade_ms_applied=int(info.get("fade_ms_applied", 0) or 0),
            handle_crossfade_ms_applied=int(info.get("handle_crossfade_ms_applied", 0) or 0),
            handle_context_seconds=info.get("handle_context_seconds"),
            loudness_matched=bool(info.get("loudness_matched", False)),
        )
