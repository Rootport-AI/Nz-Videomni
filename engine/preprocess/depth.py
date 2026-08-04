"""Video-Depth-Anything depth-map control-signal processor (IC-LoRA depth).

Produces the grayscale relative-depth video the Union-Control IC-LoRA consumes
as its "depth" control signal: near = white, far = black, normalised ONCE over
the whole clip.

Unlike Canny and DWPose this is a WHOLE-VIDEO processor, not a frame processor:
Video-Depth-Anything's official ``infer_video_depth`` runs a 32-frame sliding
window with a 10-frame overlap and rescales each window onto the previous one,
which is exactly what keeps the depth temporally stable. Feeding it one frame at
a time would throw that away, and the min-max normalisation is global by
definition. Hence the ``VideoProcessor`` protocol and the video-level branch in
``driver.preprocess_video``.

Fixed parameters (all from the official ComfyUI Union-Control workflow, and
re-measured in the G0 smoke on the engine venv):

  * ``vits`` / "Small" checkpoint  -- ~4 GB VRAM, ~15 fps at 1280x768
  * ``input_size=518``            -- the network's working resolution
  * ``fp32=True``                 -- autocast disabled
  * ``max_res=960``               -- inference input is downscaled so the long
    edge is <= 960 (aspect preserved); the OUTPUT is resized back to the source
    resolution, because the driver preserves resolution end-to-end.

The heavy imports (the vendored ``vda`` tree, which needs torchvision) happen
inside ``_ensure_loaded`` so that a canny/pose-only job never touches them and a
failure there cannot take canny/pose down with it — the same lazy-load contract
DWPose uses. ``release()`` frees the weights before the generation denoise.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

# Project root = engine/preprocess/depth.py -> parents[2]. The checkpoint lives
# in its OWN directory (sibling of models/preprocessors/, not a child) so the
# installer's per-directory size check cannot be fooled by the other's files.
# Resolved from this file (not cwd) so the path holds regardless of chdir.
_MODELS_DIR = Path(__file__).resolve().parents[2] / "models" / "preprocessors-vda"
_CHECKPOINT_PATH = _MODELS_DIR / "video_depth_anything_vits.pth"

# ``vits`` architecture constants — these MUST match the checkpoint (loaded with
# strict=True) and are the upstream ``model_configs['vits']`` values.
_ENCODER = "vits"
_FEATURES = 64
_OUT_CHANNELS = [48, 96, 192, 384]

_INPUT_SIZE = 518
_MAX_RES = 960
_FP32 = True


class DepthProcessor:
    """BGR video -> grayscale relative-depth (3-channel BGR) video converter.

    Stateful: holds the ~29M-parameter VDA network on the GPU once loaded.
    Loaded lazily on the first ``process_video`` call and freed by ``release()``.
    """

    def __init__(self) -> None:
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model: Any = None

    # ---- lifecycle ----------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Build the vits network and load the checkpoint onto the GPU."""
        if self._model is not None:
            return
        if not _CHECKPOINT_PATH.exists():
            raise RuntimeError(
                "Video-Depth-Anything checkpoint not found under "
                f"{_MODELS_DIR} (expected {_CHECKPOINT_PATH.name})"
            )
        # Deferred: the vendored tree pulls in torchvision + the DINOv2 stack.
        from engine.preprocess.vda.video_depth_anything.video_depth import (
            VideoDepthAnything,
        )

        model = VideoDepthAnything(
            encoder=_ENCODER, features=_FEATURES, out_channels=_OUT_CHANNELS
        )
        state_dict = torch.load(str(_CHECKPOINT_PATH), map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        del state_dict
        self._model = model.to(self._device).eval()

    def release(self) -> None:
        """Evict the network from the GPU and free the CUDA blocks.

        Called by the driver once the control video is written so the depth
        weights are not resident during the 16 GB-tight generation denoise. A
        subsequent ``process_video`` transparently reloads via ``_ensure_loaded``.
        """
        self._model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- VideoProcessor contract -------------------------------------------

    @torch.inference_mode()
    def process_video(self, frames_bgr: list[np.ndarray]) -> list[np.ndarray]:
        """Convert a whole BGR clip to its grayscale depth clip (same size)."""
        if not frames_bgr:
            raise RuntimeError("depth: received an empty clip")
        self._ensure_loaded()

        height, width = frames_bgr[0].shape[:2]
        rgb = self._to_inference_input(frames_bgr)

        # ``target_fps`` is echoed straight back out by infer_video_depth and is
        # not used in the computation; the driver owns FPS.
        depths, _ = self._model.infer_video_depth(
            rgb,
            0.0,
            input_size=_INPUT_SIZE,
            device=str(self._device),
            fp32=_FP32,
        )

        return self._to_control_frames(depths, width, height)

    # ---- input / output conversion -----------------------------------------

    def _to_inference_input(self, frames_bgr: list[np.ndarray]) -> np.ndarray:
        """BGR frame list -> uint8 RGB array [N, H, W, 3], long edge <= _MAX_RES.

        Downscaling is what the official workflow does (``max_res``) and is what
        keeps VRAM at ~4 GB; the depth map is resized back to the source
        resolution on the way out. No padding: the network's own transform
        handles its 14-multiple alignment.
        """
        height, width = frames_bgr[0].shape[:2]
        target = self._inference_size(width, height)
        out = []
        for frame in frames_bgr:
            if target is not None:
                frame = cv2.resize(frame, target, interpolation=cv2.INTER_AREA)
            out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return np.stack(out, axis=0)

    def _inference_size(self, width: int, height: int) -> tuple[int, int] | None:
        """(w, h) to downscale to, or None when the source is already <= _MAX_RES.

        Dimensions are forced even, matching the official ComfyUI node.
        """
        longest = max(width, height)
        if longest <= _MAX_RES:
            return None
        scale = _MAX_RES / float(longest)
        new_w = int(round(width * scale))
        new_h = int(round(height * scale))
        new_w += new_w % 2
        new_h += new_h % 2
        return (new_w, new_h)

    def _to_control_frames(
        self, depths: np.ndarray, width: int, height: int
    ) -> list[np.ndarray]:
        """Depth array [N, H, W] -> list of uint8 BGR frames at (width, height).

        Normalisation is min-max over the WHOLE clip (the official node's
        ``postprocess_gray``), so the mapping from depth to gray is constant
        across the clip and does not flicker. VDA outputs relative inverse depth
        (near = large), so this is already near = white: no inversion.
        """
        d_min = float(depths.min())
        d_max = float(depths.max())
        normalised = (depths - d_min) / (d_max - d_min + 1e-6)
        gray = np.rint(np.clip(normalised, 0.0, 1.0) * 255.0).astype(np.uint8)

        frames = []
        for i in range(gray.shape[0]):
            control = cv2.cvtColor(gray[i], cv2.COLOR_GRAY2BGR)
            if control.shape[1] != width or control.shape[0] != height:
                control = cv2.resize(
                    control, (width, height), interpolation=cv2.INTER_LINEAR
                )
            frames.append(control)
        return frames
