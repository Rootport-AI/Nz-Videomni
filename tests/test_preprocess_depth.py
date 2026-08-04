"""IC-LoRA depth — registration/plumbing tests (app venv, no cv2/torch/GPU).

Scope is deliberately the layer the APP venv can reach: the ``preprocess:
depth`` config value, the registry resolving it, the runner's single-kind /
conflict handling for it, and the shipped ``config.yaml.example`` actually
registering ``depth-control`` + ``deblur``.

The engine-side pieces (``engine/preprocess/driver.py``'s whole-clip branch and
``frame_cap``, the ``_FACTORIES`` entry, and ``DepthProcessor``'s pure array
helpers) need cv2/torch, which only ``.venv-engine`` has — and ``.venv-engine``
has no fastapi, so ``tests/conftest.py`` cannot even be collected there. Those
checks therefore live in ``engine/preprocess/preprocess_selfcheck.py``, the same
split the block-swap prefetch work used. Real VDA inference (GPU) is the G1 gate,
not a test.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from config import AppConfig, IcLoraEntry
from services.lora_registry import LoraRegistry
from services.ltx_runner import _resolve_reference_preprocess

_REPO_ROOT = Path(__file__).resolve().parents[1]

DEPTH_LORA = "depth-control"
DEBLUR_LORA = "deblur"


def test_ic_lora_entry_accepts_depth():
    entry = IcLoraEntry(path="./x.safetensors", preprocess="depth")
    assert entry.preprocess == "depth"


def test_ic_lora_entry_rejects_unknown_preprocess():
    with pytest.raises(Exception):
        IcLoraEntry(path="./x.safetensors", preprocess="deth")


def test_registry_resolves_depth_preprocess(tmp_path):
    control_file = tmp_path / "union-control.safetensors"
    control_file.write_bytes(b"\x00" * 8)
    config = AppConfig.model_validate(
        {
            "model": {
                "ic_loras": {
                    DEPTH_LORA: {"path": control_file.as_posix(), "preprocess": "depth"},
                }
            }
        }
    )
    registry = LoraRegistry(config)
    path, strength, preprocess, audio_strength = registry.resolve(DEPTH_LORA, 1.0)
    assert path == control_file.resolve()
    assert preprocess == "depth"
    assert audio_strength is None


def test_resolve_reference_preprocess_depth_single_kind():
    assert (
        _resolve_reference_preprocess(
            [(Path("a"), 1.0, "depth"), (Path("b"), 1.0, "none")]
        )
        == "depth"
    )


def test_resolve_reference_preprocess_depth_conflicts_with_canny():
    with pytest.raises(Exception):
        _resolve_reference_preprocess([(Path("a"), 1.0, "depth"), (Path("b"), 1.0, "canny")])


def test_config_example_registers_depth_control_and_deblur():
    """The shipped example is what a fresh install copies — it must carry both."""
    cfg = yaml.safe_load((_REPO_ROOT / "config.yaml.example").read_text(encoding="utf-8"))
    ic_loras = cfg["model"]["ic_loras"]

    depth = IcLoraEntry(**ic_loras[DEPTH_LORA])
    assert depth.preprocess == "depth"
    # Same file as the other Union-Control names: one adapter, three control kinds.
    assert depth.path == ic_loras["canny-control"]["path"]

    # Deblur needs no preprocessing, so it is a plain string entry (preprocess
    # defaults to "none") pointing at its own dedicated adapter file.
    assert isinstance(ic_loras[DEBLUR_LORA], str)
    assert IcLoraEntry(path=ic_loras[DEBLUR_LORA]).preprocess == "none"
    assert ic_loras[DEBLUR_LORA].endswith(".safetensors")
    assert "deblur" in ic_loras[DEBLUR_LORA]
