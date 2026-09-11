# Replacement for the four `timm` symbols the vendored UETrack encoder imports.
#
# Vendoring these instead of depending on timm is deliberate: timm's only job in
# this tree is `to_2tuple`, `drop_path`, `trunc_normal_` and the `register_model`
# decorator, and three of the four never touch a trained weight --
#
#   register_model  upstream uses timm's global model registry only so its
#                   factories can be looked up by name from a config string. This
#                   port calls `fastitpnt_layer6` directly, so the decorator has
#                   nothing to do and is a pass-through.
#   to_2tuple       a three-line shape helper.
#   drop_path       stochastic depth. It is a no-op whenever `training` is False
#                   or the rate is 0, and the tracker runs under eval() with
#                   nothing else, so it never actually drops anything here. The
#                   body is still the real one, so a future training use is not
#                   silently wrong.
#   trunc_normal_   weight INITIALISATION only. Every tensor it touches is
#                   overwritten by load_state_dict moments later. torch has
#                   carried its own implementation since 1.9, and timm's is a
#                   thin wrapper over the same algorithm.
#
# -- so the dependency bought nothing but its own tail (torchvision,
# huggingface-hub, pyyaml, safetensors) in an interpreter that is meant to stay
# small. See tracking/utils-venv-pyproject.toml.

import collections.abc
import itertools

import torch
from torch.nn.init import trunc_normal_  # re-exported; timm's is a wrapper over this


def register_model(fn):
    """No-op stand-in for timm's model registry decorator (see module docstring)."""
    return fn


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(itertools.repeat(x, n))
    return parse


to_2tuple = _ntuple(2)


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Stochastic depth, per sample (timm's `drop_path`, verbatim in behaviour)."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    return x.div(keep_prob) * random_tensor


__all__ = ["register_model", "to_2tuple", "drop_path", "trunc_normal_"]
