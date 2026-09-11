"""UETrack Base, as this project uses it: one object, forwards only, on the CPU.

This is the ONLY module in `tracking/` that imports torch, and it runs only on
the `.venv-utils` interpreter (see tracking/utils-venv-pyproject.toml). The
public surface is three calls:

    tracker = load_tracker(checkpoint_path, device="cpu")
    tracker.initialize(frame_rgb, (x, y, w, h), search_factor=4.0)
    (x, y, w, h), score = tracker.track(next_frame_rgb)

`frame_rgb` is an HxWx3 uint8 numpy array in RGB order (NOT BGR: upstream's
evaluation loader hands the model RGB, and the normalisation constants below are
ImageNet's, which are stated in RGB order).

WHAT IS FIXED HERE AND WHY
--------------------------
Upstream reads its geometry from a yacs/easydict config assembled out of
experiments/uetrack/uetrack_base.yaml plus lib/config/uetrack/config.py's
defaults. Carrying that machinery would mean carrying yacs, easydict, a YAML
parser and a file whose values must never change -- so the resolved Base values
are folded into `_BASE_CFG` below instead, and the yaml is not vendored. Every
number in it is transcribed from those two upstream files; the comment on each
line says which.

The one knob this project exposes is `search_factor` (how far around the last
box the model looks). Template size 112, search size 224, TEMPLATE_FACTOR 2.0
and the Hanning window are internal, per the owner's ruling.

MULTI_MODAL_LANGUAGE is pinned False: the CLIP text branch is not vendored, and
tracking by a text description is not a feature of this project. Its weights are
therefore absent from the state dict this module loads -- `initialize()` passes
`text_src=None`, which upstream's encoder already handles.

MULTI_MODAL_VISION stays True. It is not a "second modality" here: the tracker's
patch embedding has 6 input channels because it was trained to take RGB plus a
depth/thermal/event partner, and for an RGB-only input upstream duplicates the
RGB into both halves. Turning it off would not match the trained weights.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Sequence, Tuple

import numpy as np
import torch

from .vendor.uetrack import build_uetrack_inference
from .vendor.uetrack.box_ops import clip_box
from .vendor.uetrack.hann import hann2d
from .vendor.uetrack.preprocess import Preprocessor, sample_target, transform_image_to_crop

__all__ = ["load_tracker", "Tracker", "BASE_SEARCH_FACTOR"]

# TEST.SEARCH_FACTOR in experiments/uetrack/uetrack_base.yaml. The webui default
# comes from the design document, not from here; this is the value upstream
# benchmarks with, and the one `initialize()` falls back to.
BASE_SEARCH_FACTOR = 4.0

# Every value transcribed from experiments/uetrack/uetrack_base.yaml, falling back
# to lib/config/uetrack/config.py for keys the yaml does not override (marked
# "default"). Keys upstream reads only on the training or CLIP paths are omitted;
# the ones that remain are exactly what build_uetrack_inference() dereferences.
_BASE_CFG = {
    "MODEL": {
        "TASK_NUM": 5,                       # yaml MODEL.TASK_NUM
        "ENCODER": {
            "TYPE": "fastitpnt_layer6",      # yaml -- the 6-layer student, ~13M params
            "STRIDE": 16,                    # yaml
            "PRETRAIN_TYPE": "",             # yaml (empty: no ImageNet backbone to load)
            "PATCHEMBED_INIT": "halfcopy",   # yaml
            "TOKEN_TYPE_INDICATE": True,     # yaml
            "NUM_EXPERT": 8,                 # yaml -- mixture-of-experts width
            "MOE_LAYER": [5],                # yaml -- which block is the MoE one
            "CLASS_TOKEN": True,             # default
            "POS_TYPE": "index",             # default
        },
        "DECODER": {
            "TYPE": "CENTER",                # yaml
            "NUM_CHANNELS": 256,             # yaml
            "CONV_TYPE": "normal",           # yaml
            "XAVIER_INIT": True,             # yaml
        },
        "TASK_DECODER": {
            "NUM_CHANNELS": 256,             # yaml
            "FEATURE_TYPE": "average",       # yaml
        },
    },
    "TRAIN": {
        # Read by build_encoder to decide requires_grad. Inference calls eval()
        # and wraps every forward in no_grad, so these only affect flags.
        "ENCODER_MULTIPLIER": 0.1,           # yaml
        "FREEZE_ENCODER": False,             # default
        "ENCODER_OPEN": [],                  # default
        "DISTILL_LAYER_S": [6],              # yaml -- which student layer is tapped
    },
    "DATA": {
        "MULTI_MODAL_VISION": True,          # yaml -- see module docstring
        "MULTI_MODAL_LANGUAGE": False,       # NZ: yaml says True; pinned False here
        "SEARCH": {"SIZE": 224, "NUMBER": 1},        # yaml
        "TEMPLATE": {"SIZE": 112, "NUMBER": 1},      # yaml
    },
    "TEST": {
        "SEARCH_FACTOR": BASE_SEARCH_FACTOR,  # yaml
        "SEARCH_SIZE": 224,                   # yaml
        "TEMPLATE_FACTOR": 2.0,               # yaml
        "TEMPLATE_SIZE": 112,                 # yaml
        "WINDOW": True,                       # yaml -- Hanning penalty, fixed on
        "NUM_TEMPLATES": 1,                   # default
    },
}


def _namespace(obj):
    """dict -> attribute access, so the vendored `cfg.MODEL.ENCODER.TYPE` reads work."""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _namespace(v) for k, v in obj.items()})
    return obj


class _UpstreamStub(dict):
    """Stand-in for an upstream class the official archive pickled by reference."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.update(state)


class _UpstreamStubFinder:
    """Resolves `lib.*` / `easydict` / `yacs` imports to throwaway stub modules.

    Needed only to read the OFFICIAL .tar. That archive is a training snapshot,
    so besides the tensors it pickles five upstream objects BY REFERENCE --
    measured on uetrack_base.tar (2026-09-11):

        easydict.EasyDict                       (the yacs/easydict config)
        lib.train.admin.settings.Settings       (the run's settings object)
        lib.train.admin.local.EnvironmentSettings
        lib.train.admin.stats.AverageMeter      (training loss meters)
        lib.train.admin.stats.StatValue

    None of them is a tensor and none is reachable from `blob["net"]`, but the
    unpickler resolves every class in the file before handing anything back, so
    without stubs `torch.load` dies with ModuleNotFoundError before the weights
    appear. Importing the real upstream package instead would mean vendoring its
    training tree and adding easydict/yacs to this interpreter, to read five
    objects that are then thrown away.

    This is exactly the hazard the distributed .safetensors removes: unpickling
    is code execution, so the .tar branch is for the one-off trimming step and
    for local A/B checks against the original -- never for a user download.
    """

    _ROOTS = ("lib", "easydict", "yacs")

    def __init__(self):
        # Exactly the names this finder put into sys.modules, so the cleanup
        # below removes only its own leavings and never a real module that
        # happened to share a root name.
        self.created = []

    def find_spec(self, name, path=None, target=None):
        import importlib.machinery
        import sys
        if name.split(".")[0] not in self._ROOTS:
            return None
        if name in sys.modules:
            return None
        return importlib.machinery.ModuleSpec(name, self, is_package=True)

    def create_module(self, spec):
        import types
        module = types.ModuleType(spec.name)
        module.__path__ = []
        module.__getattr__ = lambda attr: type(attr, (_UpstreamStub,), {})
        self.created.append(spec.name)
        return module

    def exec_module(self, module):
        pass


def _load_state_dict(checkpoint_path: str):
    """Return the tracker state dict from either weight format.

    ONE rule, keyed on the extension:

    * `.safetensors` -- what this project distributes
      (models/UETrack/uetrack_base.safetensors, produced by
      scripts/trim_uetrack_checkpoint.py). Flat tensor map, no pickle.
    * anything else -- the official `uetrack_base.tar` from
      huggingface.co/kangben258/UETrack: a torch pickle whose 'net' key holds the
      state dict. `weights_only=False` is required for it (and the stub finder
      above with it); see that class for why, and why this is not the shipped
      format.
    """
    if checkpoint_path.lower().endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(checkpoint_path, device="cpu")

    import sys
    finder = _UpstreamStubFinder()
    sys.meta_path.insert(0, finder)
    try:
        blob = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    finally:
        sys.meta_path.remove(finder)
        for name in finder.created:
            sys.modules.pop(name, None)
    if isinstance(blob, dict) and "net" in blob:
        return blob["net"]
    return blob


class Tracker:
    """One tracked object. Not thread-safe; the worker owns exactly one."""

    def __init__(self, network, device: str = "cpu"):
        self.device = device
        self.cfg = _namespace(_BASE_CFG)
        self.network = network
        self.network.eval()
        self.preprocessor = Preprocessor(device=device)
        self.state = None

        self.template_factor = _BASE_CFG["TEST"]["TEMPLATE_FACTOR"]
        self.template_size = _BASE_CFG["TEST"]["TEMPLATE_SIZE"]
        self.search_size = _BASE_CFG["TEST"]["SEARCH_SIZE"]
        self.search_factor = BASE_SEARCH_FACTOR
        self.num_template = _BASE_CFG["TEST"]["NUM_TEMPLATES"]
        self.multi_modal_vision = _BASE_CFG["DATA"]["MULTI_MODAL_VISION"]

        self.fx_sz = self.search_size // _BASE_CFG["MODEL"]["ENCODER"]["STRIDE"]
        self.output_window = None
        if _BASE_CFG["TEST"]["WINDOW"]:
            self.output_window = hann2d(
                torch.tensor([self.fx_sz, self.fx_sz]).long(), centered=True
            ).to(device)

        # None = "no task hint". Upstream picks a task index per BENCHMARK
        # (LaSOT/GOT10k -> 0, TNL2K -> 1, DepthTrack -> 2, ...); there is no
        # benchmark here, and the encoder treats None as "add no task token".
        self.task_index_batch = None
        self.text_src = None
        self.template_list = []
        self.template_anno_list = []
        self.frame_id = 0
        # Filled in by load_tracker(); defined here so a directly constructed
        # Tracker still answers the same questions.
        self.missing_keys = []
        self.unexpected_keys = []

    # -- upstream lib/test/tracker/uetrack.py: initialize() ------------------
    def initialize(self, frame_rgb: np.ndarray, box_xywh: Sequence[float], *,
                   search_factor: float = BASE_SEARCH_FACTOR) -> None:
        """Seed the tracker with the first frame and the box to follow."""
        self.search_factor = float(search_factor)
        init_bbox = [float(v) for v in box_xywh]
        if init_bbox[2] <= 0 or init_bbox[3] <= 0:
            raise ValueError(f"box must have positive width and height, got {init_bbox}")

        z_patch_arr, resize_factor = sample_target(
            frame_rgb, init_bbox, self.template_factor, output_sz=self.template_size)
        template = self.preprocessor.process(z_patch_arr)
        if self.multi_modal_vision and (template.size(1) == 3):
            template = torch.cat((template, template), axis=1)
        self.template_list = [template] * self.num_template

        self.state = init_bbox
        prev_box_crop = transform_image_to_crop(
            torch.tensor(init_bbox),
            torch.tensor(init_bbox),
            resize_factor,
            torch.Tensor([self.template_size, self.template_size]),
            normalize=True)
        self.template_anno_list = [prev_box_crop.to(template.device).unsqueeze(0)]
        self.frame_id = 0
        self.text_src = None

    # -- upstream lib/test/tracker/uetrack.py: track() -----------------------
    def track(self, frame_rgb: np.ndarray) -> Tuple[Tuple[float, float, float, float], float]:
        """Advance one frame. Returns ((x, y, w, h), best_score)."""
        if self.state is None:
            raise RuntimeError("initialize() must be called before track()")

        H, W, _ = frame_rgb.shape
        self.frame_id += 1
        x_patch_arr, resize_factor = sample_target(
            frame_rgb, self.state, self.search_factor, output_sz=self.search_size)
        search = self.preprocessor.process(x_patch_arr)
        if self.multi_modal_vision and (search.size(1) == 3):
            search = torch.cat((search, search), axis=1)
        search_list = [search]

        with torch.no_grad():
            enc_opt, feature_list, _ = self.network.forward_encoder(
                self.template_list, search_list, self.template_anno_list,
                self.text_src, self.task_index_batch)
            out_dict = self.network.forward_decoder(feature=enc_opt)

        pred_score_map = out_dict["score_map"]
        if self.output_window is not None:
            response = self.output_window * pred_score_map
        else:
            response = pred_score_map

        if "size_map" in out_dict.keys():
            pred_boxes, conf_score = self.network.decoder.cal_bbox(
                response, out_dict["size_map"], out_dict["offset_map"], return_score=True)
        else:
            pred_boxes, conf_score = self.network.decoder.cal_bbox(
                response, out_dict["offset_map"], return_score=True)
        pred_boxes = pred_boxes.view(-1, 4)
        pred_box = (pred_boxes.mean(dim=0) * self.search_size / resize_factor).tolist()
        self.state = clip_box(self._map_box_back(pred_box, resize_factor), H, W, margin=10)

        return tuple(self.state), float(conf_score)

    def _map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]


def load_tracker(checkpoint_path: str, device: str = "cpu") -> Tracker:
    """Build the Base tracker and load `checkpoint_path` into it.

    `strict=False` is upstream's own call (lib/test/tracker/uetrack.py), and it
    is load-bearing rather than sloppy in this port: the CLIP text encoder and
    the `interface_text_proj` seam are not built here, so their names are
    expected to be missing from the module when the OFFICIAL checkpoint is read.
    Anything else missing means the vendored geometry has drifted from the
    weights, so the caller is handed the two key lists to check.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"UETrack checkpoint not found: {checkpoint_path}")

    cfg = _namespace(_BASE_CFG)
    network = build_uetrack_inference(cfg)
    state_dict = _load_state_dict(checkpoint_path)
    result = network.load_state_dict(state_dict, strict=False)
    network = network.to(device)

    tracker = Tracker(network, device=device)
    tracker.missing_keys = list(result.missing_keys)
    tracker.unexpected_keys = list(result.unexpected_keys)
    return tracker
