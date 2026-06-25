"""Upload store for minimal-I2V input images (spec 7.1 / 0.4).

Saves an uploaded image, normalizing it to ``uploads/{image_id}/input.png``:
EXIF orientation applied, converted to RGB. Returns metadata used by the API.
"""

from __future__ import annotations

import io
import uuid
from pathlib import Path

from PIL import Image, ImageOps

from api.errors import image_not_found, upload_invalid_type, upload_too_large
from config import AppConfig

# Map allowed extensions to content types we report back.
_CONTENT_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


class StoredImage:
    def __init__(self, image_id: str, path: Path, width: int, height: int, content_type: str, original_filename: str):
        self.image_id = image_id
        self.path = path
        self.width = width
        self.height = height
        self.content_type = content_type
        self.original_filename = original_filename


class UploadStore:
    def __init__(self, config: AppConfig):
        self.config = config
        self.upload_dir = config.upload_dir
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.allowed = {ext.lower() for ext in config.upload.allowed_image_extensions}
        self.max_bytes = config.upload.max_image_size_mb * 1024 * 1024

    def save(self, *, data: bytes, filename: str) -> StoredImage:
        ext = Path(filename).suffix.lower()
        if ext not in self.allowed:
            raise upload_invalid_type(
                detail=f"allowed: {sorted(self.allowed)}, got: {ext or '(none)'}"
            )
        if len(data) > self.max_bytes:
            raise upload_too_large(
                detail=f"max {self.config.upload.max_image_size_mb} MB, got {len(data) / 1024 / 1024:.1f} MB"
            )

        try:
            image = Image.open(io.BytesIO(data))
            image = ImageOps.exif_transpose(image)  # honor EXIF orientation
            image = image.convert("RGB")  # RGBA/P -> RGB
        except Exception as exc:
            raise upload_invalid_type(detail=f"cannot decode image: {exc}")

        image_id = str(uuid.uuid4())
        dest_dir = self.upload_dir / image_id  # image_id is a UUID -> no traversal
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / "input.png"
        image.save(dest, format="PNG")

        return StoredImage(
            image_id=image_id,
            path=dest,
            width=image.width,
            height=image.height,
            content_type=_CONTENT_TYPES.get(ext, "image/png"),
            original_filename=filename,
        )

    def stored_relpath(self, image_id: str) -> str:
        """Project-relative path used in API responses (spec 7.1)."""
        return f"uploads/{image_id}/input.png"

    def path_for(self, image_id: str) -> Path:
        """Resolve the stored image path, guarding against traversal."""
        # Reject anything that isn't a bare UUID-like segment.
        if "/" in image_id or "\\" in image_id or image_id in ("", ".", ".."):
            raise image_not_found(image_id)
        path = (self.upload_dir / image_id / "input.png").resolve()
        if self.upload_dir.resolve() not in path.parents:
            raise image_not_found(image_id)
        if not path.exists():
            raise image_not_found(image_id)
        return path
