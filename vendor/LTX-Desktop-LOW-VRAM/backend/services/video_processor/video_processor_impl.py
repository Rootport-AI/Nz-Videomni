"""Video processing (cv2) service for IC-LoRA operations."""

from __future__ import annotations

import logging
from typing import cast

from services.depth_processor_pipeline.depth_processor_pipeline import DepthProcessorPipeline
from services.pose_processor_pipeline.pose_processor_pipeline import PoseProcessorPipeline
from services.video_processor.video_processor import VideoInfoPayload
from services.services_utils import FrameArray, VideoCaptureLike, VideoWriterLike

logger = logging.getLogger(__name__)


class VideoProcessorImpl:
    """Wraps cv2 operations for IC-LoRA processing."""

    def open_video(self, path: str) -> VideoCaptureLike:
        import cv2

        return cast(VideoCaptureLike, cv2.VideoCapture(path))

    def get_video_info(self, cap: VideoCaptureLike) -> VideoInfoPayload:
        import cv2

        return {
            "fps": float(cap.get(cv2.CAP_PROP_FPS) or 24),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        }

    def read_frame(self, cap: VideoCaptureLike, frame_idx: int | None = None) -> FrameArray | None:
        import cv2

        if frame_idx is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            return None
        return cast(FrameArray, frame)

    def apply_canny(self, frame: FrameArray) -> FrameArray:
        import cv2
        import numpy as np

        img = frame.copy()

        # Pad for compatibility with training flow.
        H, W = img.shape[:2]
        H_pad = int(np.ceil(H / 64.0) * 64) - H
        W_pad = int(np.ceil(W / 64.0) * 64) - W
        if H_pad > 0 or W_pad > 0:
            img = np.pad(img, [[0, H_pad], [0, W_pad], [0, 0]], mode="edge")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)

        # Remove padding
        edges = edges[:H, :W]

        # HWC3: convert single-channel to 3-channel
        edges_3ch = np.concatenate([edges[:, :, None]] * 3, axis=2)

        return cast(FrameArray, edges_3ch)

    def apply_depth(self, frame: FrameArray, depth_pipeline: DepthProcessorPipeline) -> FrameArray:
        return depth_pipeline.apply(frame)

    def apply_pose(self, frame: FrameArray, pose_pipeline: PoseProcessorPipeline) -> FrameArray:
        return pose_pipeline.apply(frame)

    def encode_frame_jpeg(self, frame: FrameArray, quality: int = 85) -> bytes:
        import cv2

        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError("Failed to encode frame")
        return bytes(buf)

    def create_writer(self, path: str, fourcc: str, fps: float, size: tuple[int, int]) -> VideoWriterLike:
        import cv2

        code = cv2.VideoWriter.fourcc(*fourcc)
        return cast(VideoWriterLike, cv2.VideoWriter(path, code, fps, size))

    def release(self, cap_or_writer: VideoCaptureLike | VideoWriterLike) -> None:
        try:
            cap_or_writer.release()
        except Exception:
            logger.warning("Failed to release video resource", exc_info=True)
