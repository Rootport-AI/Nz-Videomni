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

import uuid
from pathlib import Path

from api.errors import reference_video_not_found, upload_invalid_type, upload_too_large
from config import AppConfig

# Map allowed extensions to the content types we report back.
_CONTENT_TYPES = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
    ".mkv": "video/x-matroska",
}


class StoredVideo:
    def __init__(self, video_id: str, path: Path, content_type: str, original_filename: str, size_bytes: int):
        self.video_id = video_id
        self.path = path
        self.content_type = content_type
        self.original_filename = original_filename
        self.size_bytes = size_bytes


class VideoUploadStore:
    def __init__(self, config: AppConfig):
        self.config = config
        self.video_dir = config.upload_dir / "videos"
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.allowed = {ext.lower() for ext in config.upload.allowed_video_extensions}
        self.max_bytes = config.upload.max_video_size_mb * 1024 * 1024

    def save(self, *, data: bytes, filename: str) -> StoredVideo:
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

        return StoredVideo(
            video_id=video_id,
            path=dest,
            content_type=_CONTENT_TYPES.get(ext, "application/octet-stream"),
            original_filename=filename,
            size_bytes=len(data),
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
