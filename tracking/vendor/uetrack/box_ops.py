# Vendored from https://github.com/kangben258/UETrack @ fd13b0ea
# Source file: lib/utils/box_ops.py  (the two helpers inference reaches, only)
#
# NZ: upstream's box_ops.py also carries `box_iou` / `generalized_box_iou`, which
# are TRAINING losses and are the sole reason that file imports
# `torchvision.ops.boxes.box_area`. Dropping them drops torchvision from this
# interpreter entirely (see tracking/utils-venv-pyproject.toml). The two
# functions below are byte-for-byte upstream's and depend on nothing but torch.

import torch


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def clip_box(box: list, H, W, margin=0):
    x1, y1, w, h = box
    x2, y2 = x1 + w, y1 + h
    x1 = min(max(0, x1), W-margin)
    x2 = min(max(margin, x2), W)
    y1 = min(max(0, y1), H-margin)
    y2 = min(max(margin, y2), H)
    w = max(margin, x2-x1)
    h = max(margin, y2-y1)
    return [x1, y1, w, h]
