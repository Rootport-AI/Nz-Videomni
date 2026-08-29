"""Outpainting building blocks: canvas geometry and Laplacian seam blending.

``canvas`` is torch-free at import time; ``pyramid_blend`` is not. Importing a
submodule always executes its package ``__init__``, so an eager
``from .pyramid_blend import ...`` here would make ``engine.outpaint.canvas``
unimportable in the app ``.venv`` (which has no torch) and take the geometry
unit tests down with it. The torch-backed names are therefore re-exported
lazily through PEP 562 ``__getattr__``: ``from engine.outpaint import
blend_video_u8`` still works wherever torch is installed, and merely importing
the package costs nothing.
"""

from __future__ import annotations

from typing import Any

from .canvas import (
    CANVAS_MULTIPLE,
    GREEN_RGB,
    MIN_INNER_SIDE,
    OutpaintGeometry,
    build_blend_mask,
    fill_pad_with_generated_,
)

__all__ = [
    "CANVAS_MULTIPLE",
    "GREEN_RGB",
    "MIN_INNER_SIDE",
    "OutpaintGeometry",
    "blend_video_u8",
    "build_blend_mask",
    "fill_pad_with_generated_",
    "laplacian_pyramid_blend",
]

_LAZY_EXPORTS = {
    "blend_video_u8": ".pyramid_blend",
    "laplacian_pyramid_blend": ".pyramid_blend",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value  # cache: subsequent lookups skip __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
