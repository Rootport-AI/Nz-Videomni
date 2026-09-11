"""Minimal inference-only vendoring of UETrack (https://github.com/kangben258/UETrack).

Only what a single forward pass of the Base tracker touches is here; training,
evaluation, the CLIP text branch, the distillation teacher and the yacs config
layer are all left upstream. tracking/VENDOR_NOTICE.md lists the provenance and
every modification.
"""

from .uetrack import UETrack, build_uetrack_inference

__all__ = ["UETrack", "build_uetrack_inference"]
