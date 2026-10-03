"""Type aliases and device helpers shared by the pipeline modules."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, TypeAlias

import torch

if TYPE_CHECKING:
    from ltx_core.model.video_vae import TilingConfig
    from ltx_core.types import Audio as AudioType

    TilingConfigType: TypeAlias = TilingConfig
else:
    TilingConfigType: TypeAlias = object
    AudioType: TypeAlias = object

AudioOrNone: TypeAlias = AudioType | None

logger = logging.getLogger(__name__)


def get_device_type(device: str | torch.device | object | None) -> str:
    if device is None:
        return "cpu"

    device_type = getattr(device, "type", None)
    if isinstance(device_type, str):
        return device_type

    if isinstance(device, str):
        try:
            return str(torch.device(device).type)
        except Exception:
            logger.warning("Could not parse device string '%s', using it as-is", device, exc_info=True)
            return device

    return "cpu"


def device_supports_fp8(device: str | torch.device | object | None) -> bool:
    return get_device_type(device) == "cuda"
