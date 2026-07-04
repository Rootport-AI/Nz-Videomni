"""Upload store for source audio (A2V, POST /upload/audio).

Mirrors :mod:`services.video_upload_store` but for the uploaded audio track that
drives audio-to-video generation. Audio is stored AS-IS (no re-encode) under
``uploads/audios/{audio_id}/input{ext}`` — the engine's PyAV-based decode
(``decode_audio_from_file``) reads the container directly (wav/mp3/m4a/…).
Validation is minimal (extension + size), consistent with the image/video
store allow-list pattern; codec validity is verified later at preflight
(ffprobe) rather than on upload.

Audio lives in a dedicated ``audios/`` subdir so an ``audio_id`` can never be
confused with a ``video_id`` or an image ``image_id`` (all are UUIDs, but the
subdir keeps the path spaces disjoint and makes ``path_for`` unambiguous).
"""

from __future__ import annotations

import uuid
from pathlib import Path

from api.errors import source_audio_not_found, upload_invalid_type, upload_too_large
from config import AppConfig

# Map allowed extensions to the content types we report back.
_CONTENT_TYPES = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
}


class StoredAudio:
    def __init__(self, audio_id: str, path: Path, content_type: str, original_filename: str, size_bytes: int):
        self.audio_id = audio_id
        self.path = path
        self.content_type = content_type
        self.original_filename = original_filename
        self.size_bytes = size_bytes


class AudioUploadStore:
    def __init__(self, config: AppConfig):
        self.config = config
        self.audio_dir = config.upload_dir / "audios"
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.allowed = {ext.lower() for ext in config.upload.allowed_audio_extensions}
        self.max_bytes = config.upload.max_audio_size_mb * 1024 * 1024

    def save(self, *, data: bytes, filename: str) -> StoredAudio:
        ext = Path(filename).suffix.lower()
        if ext not in self.allowed:
            raise upload_invalid_type(
                detail=f"allowed audio types: {sorted(self.allowed)}, got: {ext or '(none)'}"
            )
        if not data:
            raise upload_invalid_type(detail="empty file")
        if len(data) > self.max_bytes:
            raise upload_too_large(
                detail=f"max {self.config.upload.max_audio_size_mb} MB, got {len(data) / 1024 / 1024:.1f} MB"
            )

        audio_id = str(uuid.uuid4())  # UUID -> no traversal
        dest_dir = self.audio_dir / audio_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"input{ext}"
        dest.write_bytes(data)

        return StoredAudio(
            audio_id=audio_id,
            path=dest,
            content_type=_CONTENT_TYPES.get(ext, "application/octet-stream"),
            original_filename=filename,
            size_bytes=len(data),
        )

    def stored_relpath(self, stored: StoredAudio) -> str:
        """Project-relative path used in the API response."""
        return f"uploads/audios/{stored.audio_id}/{stored.path.name}"

    def path_for(self, audio_id: str) -> Path:
        """Resolve the stored source-audio path, guarding against traversal."""
        if "/" in audio_id or "\\" in audio_id or audio_id in ("", ".", ".."):
            raise source_audio_not_found(audio_id)
        base = self.audio_dir.resolve()
        d = (self.audio_dir / audio_id).resolve()
        if d.parent != base or not d.is_dir():
            raise source_audio_not_found(audio_id)
        matches = sorted(d.glob("input.*"))
        if not matches:
            raise source_audio_not_found(audio_id)
        return matches[0]
