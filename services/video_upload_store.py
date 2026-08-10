"""Upload store for reference videos (Phase B, POST /upload/video).

Mirrors :mod:`services.upload_store` (the image store) but for the reference
video used by the Pixel-Spatial-Upscaler IC-LoRA. Videos are stored AS-IS (no
re-encode) under ``uploads/videos/{video_id}/input{ext}`` — the engine's
ffmpeg-based video IO reads the container directly. Validation is minimal
(extension + size), consistent with the image store's allow-list pattern.

Videos live in a dedicated ``videos/`` subdir so a ``video_id`` can never be
confused with an image ``image_id`` (both are UUIDs, but the subdir keeps the
two path spaces disjoint and makes ``path_for`` unambiguous).
"""

from __future__ import annotations

import logging
import math
import os
import uuid
from pathlib import Path

from api.errors import reference_video_not_found, upload_invalid_type, upload_too_large
from config import AppConfig
from services.video_io import FFmpegError, cut_range_mp4, frame_count

logger = logging.getLogger("ltx.video_upload_store")

# Map allowed extensions to the content types we report back.
_CONTENT_TYPES = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
    ".mkv": "video/x-matroska",
}

# Trim work file. MUST start with "_" so it never matches ``path_for``'s
# ``glob("input.*")`` even if a crash leaves it behind.
_TRIM_TMP_NAME = "_input.tmp.mp4"


def _trim_window(start_sec: float | None, duration_sec: float | None) -> tuple[float, float] | None:
    """Return a usable ``(start, duration)`` window, or None to store as-is.

    Anything unusable -- either value missing, non-numeric, NaN/inf, a negative
    start or a non-positive duration -- returns None so the caller falls through
    to the plain (byte-identical) store path. Deliberately NOT an error: the
    trim arguments are an optional refinement of an existing endpoint, so bad
    values must never invent a new 4xx/5xx response for an upload that would
    otherwise have succeeded.
    """
    if start_sec is None or duration_sec is None:
        return None
    try:
        start = float(start_sec)
        duration = float(duration_sec)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(start) and math.isfinite(duration)):
        return None
    if start < 0 or duration <= 0:
        return None
    return start, duration


class StoredVideo:
    def __init__(
        self,
        video_id: str,
        path: Path,
        content_type: str,
        original_filename: str,
        size_bytes: int,
        trimmed: bool = False,
    ):
        self.video_id = video_id
        self.path = path
        self.content_type = content_type
        self.original_filename = original_filename
        self.size_bytes = size_bytes
        self.trimmed = trimmed


class VideoUploadStore:
    def __init__(self, config: AppConfig):
        self.config = config
        self.video_dir = config.upload_dir / "videos"
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.allowed = {ext.lower() for ext in config.upload.allowed_video_extensions}
        self.max_bytes = config.upload.max_video_size_mb * 1024 * 1024

    def save(
        self,
        *,
        data: bytes,
        filename: str,
        trim_start_sec: float | None = None,
        trim_duration_sec: float | None = None,
        max_frames: int | None = None,
    ) -> StoredVideo:
        """Store an uploaded video, optionally keeping only one time window of it.

        With ``trim_start_sec``/``trim_duration_sec`` omitted (or unusable, see
        :func:`_trim_window`) this is the original store: the received bytes are
        written verbatim and nothing else runs -- the no-trim path below is
        byte-for-byte the pre-trim behavior and must stay that way.

        With a usable window, the file is still written verbatim FIRST, then cut
        into a sibling temp file and swapped in only on success. So any ffmpeg
        failure (missing binary, unprobeable source, bad window) leaves the
        original upload untouched and simply returns it with ``trimmed=False``
        -- the endpoint never gains a new error response because of trimming.

        ``max_frames`` (§1-15: the 11544f chain-reference upload ceiling) is a
        SEPARATE, lower-priority mechanism: an explicit trim window always wins
        (the V2V source-video upload path is unchanged by this parameter). When
        no trim window applies and ``max_frames`` is given, the stored file's
        frame count is measured (:func:`services.video_io.frame_count`) and, ONLY
        if it exceeds ``max_frames``, the first ``max_frames`` frames are cut out
        the same tmp-then-``os.replace`` way the trim path does. At or under the
        limit nothing runs -- no re-encode, ``trimmed=False``, byte-identical to
        the plain store. A failed probe or cut (ffprobe/ffmpeg missing, unreadable
        source) degrades the same way the trim path does: the untouched original
        comes back with ``trimmed=False`` rather than a new error response.
        """
        ext = Path(filename).suffix.lower()
        if ext not in self.allowed:
            raise upload_invalid_type(
                detail=f"allowed video types: {sorted(self.allowed)}, got: {ext or '(none)'}"
            )
        if not data:
            raise upload_invalid_type(detail="empty file")
        if len(data) > self.max_bytes:
            raise upload_too_large(
                detail=f"max {self.config.upload.max_video_size_mb} MB, got {len(data) / 1024 / 1024:.1f} MB"
            )

        video_id = str(uuid.uuid4())  # UUID -> no traversal
        dest_dir = self.video_dir / video_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"input{ext}"
        dest.write_bytes(data)

        window = _trim_window(trim_start_sec, trim_duration_sec)
        if window is None:
            if max_frames is not None and int(max_frames) > 0:
                return self._apply_max_frames(
                    video_id=video_id,
                    dest_dir=dest_dir,
                    dest=dest,
                    filename=filename,
                    data=data,
                    max_frames=int(max_frames),
                )
            return StoredVideo(
                video_id=video_id,
                path=dest,
                content_type=_CONTENT_TYPES.get(ext, "application/octet-stream"),
                original_filename=filename,
                size_bytes=len(data),
            )

        # ---- trim path ------------------------------------------------------
        # ``dest`` (the verbatim upload) is never touched until the cut has
        # succeeded, so every failure below degrades to the result above.
        start_sec, duration_sec = window
        tmp = dest_dir / _TRIM_TMP_NAME
        final = dest
        content_type = _CONTENT_TYPES.get(ext, "application/octet-stream")
        size_bytes = len(data)
        trimmed = False
        try:
            info = cut_range_mp4(dest, tmp, start_sec, duration_sec)
            target = dest_dir / "input.mp4"
            os.replace(tmp, target)
            if dest != target:
                dest.unlink(missing_ok=True)  # e.g. the original input.mkv
            final = target
            content_type = "video/mp4"
            size_bytes = final.stat().st_size
            trimmed = True
            logger.info(
                "trimmed uploaded video %s to [%.3fs, +%.3fs) -> frames %s..%s of %s",
                video_id, start_sec, duration_sec,
                info["start_frame"], info["end_frame"], info["total_frames"],
            )
        except (FFmpegError, OSError) as exc:
            logger.warning(
                "video trim failed for %s (start=%.3f duration=%.3f); storing the "
                "untrimmed upload instead: %s",
                video_id, start_sec, duration_sec, exc,
            )
        finally:
            # Best effort: a leftover temp file is harmless (its "_" prefix keeps
            # it out of path_for's glob) and must never fail the upload.
            try:
                tmp.unlink(missing_ok=True)
            except OSError as exc:  # pragma: no cover - defensive
                logger.warning("could not remove trim temp file %s: %s", tmp, exc)

        return StoredVideo(
            video_id=video_id,
            path=final,
            content_type=content_type,
            original_filename=filename,
            size_bytes=size_bytes,
            trimmed=trimmed,
        )

    def _apply_max_frames(
        self,
        *,
        video_id: str,
        dest_dir: Path,
        dest: Path,
        filename: str,
        data: bytes,
        max_frames: int,
    ) -> StoredVideo:
        """§1-15 upload-time frame-count ceiling: keep only the first
        ``max_frames`` frames of ``dest`` when it has more than that.

        Mirrors the trim path's tmp-then-``os.replace`` swap and its
        degrade-to-the-untouched-original failure handling (:func:`save`'s
        docstring), but measures with :func:`services.video_io.frame_count`
        instead of taking a caller-given window, and does nothing at all
        (no probe result needed beyond the one comparison, no re-encode) when
        the upload is already at or under the limit.
        """
        ext = dest.suffix.lower()
        content_type = _CONTENT_TYPES.get(ext, "application/octet-stream")
        final = dest
        size_bytes = len(data)
        trimmed = False
        tmp = dest_dir / _TRIM_TMP_NAME
        try:
            total = frame_count(dest)
            if total > max_frames:
                info = cut_range_mp4(
                    dest, tmp, 0.0, 0.0, start_frame=0, num_frames=max_frames
                )
                target = dest_dir / "input.mp4"
                os.replace(tmp, target)
                if dest != target:
                    dest.unlink(missing_ok=True)
                final = target
                content_type = "video/mp4"
                size_bytes = final.stat().st_size
                trimmed = True
                logger.info(
                    "max_frames-trimmed uploaded video %s to the first %d frames "
                    "(had %d) -> frames %s..%s",
                    video_id, max_frames, total, info["start_frame"], info["end_frame"],
                )
        except (FFmpegError, OSError) as exc:
            logger.warning(
                "max_frames trim failed for %s (max_frames=%s); storing the "
                "untrimmed upload instead: %s",
                video_id, max_frames, exc,
            )
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except OSError as exc:  # pragma: no cover - defensive
                logger.warning("could not remove trim temp file %s: %s", tmp, exc)

        return StoredVideo(
            video_id=video_id,
            path=final,
            content_type=content_type,
            original_filename=filename,
            size_bytes=size_bytes,
            trimmed=trimmed,
        )

    def stored_relpath(self, stored: StoredVideo) -> str:
        """Project-relative path used in the API response."""
        return f"uploads/videos/{stored.video_id}/{stored.path.name}"

    def path_for(self, video_id: str) -> Path:
        """Resolve the stored reference-video path, guarding against traversal."""
        if "/" in video_id or "\\" in video_id or video_id in ("", ".", ".."):
            raise reference_video_not_found(video_id)
        base = self.video_dir.resolve()
        d = (self.video_dir / video_id).resolve()
        if d.parent != base or not d.is_dir():
            raise reference_video_not_found(video_id)
        matches = sorted(d.glob("input.*"))
        if not matches:
            raise reference_video_not_found(video_id)
        return matches[0]
