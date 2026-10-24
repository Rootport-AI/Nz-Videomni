"""Base-model descriptor parsing: the optional ``lora_dir`` (台帳 §1-104).

``lora_dir`` names a base model's own LoRA directory, ``models_dir``-relative
like every other descriptor path, so an absolute path or a ``..`` escape is a
build defect that fails loud at startup. Absent -> ``None`` (the family shares
``config.model.lora_dir``, LTX unchanged).
"""

from __future__ import annotations

import pytest

from config import AppConfig
from conftest import base_model_descriptor, write_manifests
from services.base_models import load_base_models


def _load(tmp_path, **extra):
    descriptor = {**base_model_descriptor(), **extra}
    return load_base_models(write_manifests(tmp_path / "manifests", [descriptor]))


def test_lora_dir_absent_is_none(tmp_path):
    (descriptor,) = _load(tmp_path).values()
    assert descriptor.lora_dir is None


def test_lora_dir_is_parsed(tmp_path):
    (descriptor,) = _load(tmp_path, lora_dir="MiniMaxH3/LoRA").values()
    assert descriptor.lora_dir == "MiniMaxH3/LoRA"


@pytest.mark.parametrize("value", ["../outside", "MiniMaxH3/../../x", "C:/abs/LoRA"])
def test_lora_dir_outside_models_dir_fails_loud(tmp_path, value):
    with pytest.raises(RuntimeError, match="lora_dir must be a relative path"):
        _load(tmp_path, lora_dir=value)


@pytest.mark.parametrize("value", ["", "  ", 3])
def test_lora_dir_must_be_a_non_empty_string(tmp_path, value):
    with pytest.raises(RuntimeError, match="'lora_dir' must be a non-empty string"):
        _load(tmp_path, lora_dir=value)


def test_shipped_descriptors_lora_dir():
    shipped = load_base_models(AppConfig().manifest_dir)
    assert shipped["MiniMaxH3"].lora_dir == "MiniMaxH3/LoRA"
    assert shipped["LTX23"].lora_dir is None
    assert shipped["LTX25"].lora_dir is None
