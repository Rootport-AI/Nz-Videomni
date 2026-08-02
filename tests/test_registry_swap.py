"""Structural guard for ``LTXFastVideoPipeline._swap_registry`` (§48).

The keep-resident feature turns the cross-job CPU-skeleton cache on and off by
re-pointing ONE registry object in two places at once:

  * ``ledger.registry`` — read live by ``ModelLedger._target_device()``, so it
    decides CPU-vs-GPU build placement;
  * every ``ledger.*_builder.registry`` — read by
    ``SingleGPUModelBuilder.load_sd``, so it decides cache HIT/MISS.

Miss either half and the failure is SILENT: submodels that still hold the old
registry keep caching into (or building on) the wrong thing, which shows up
only as "slower and heavier than expected". Hence this test asserts both halves
move together, that ``_target_device()`` actually follows, that
``dataclasses.replace`` preserved every other builder field (the reason
``build_model_builders()`` is not used — it would throw away the GGUF loaders
and component-file paths the install group wrote), and that a missing builder
attribute trips the assert instead of being skipped.

Uses the REAL ``ModelLedger`` (with ``checkpoint_path=None`` so it creates no
builders and needs no weights) plus hand-built ``SingleGPUModelBuilder``
instances — a fake ledger would only prove the test's own assumptions.

Run with ``.venv-engine`` and ``--noconftest``.
"""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("ltx_core")

from ltx_core.loader.registry import DummyRegistry, StateDictRegistry  # noqa: E402
from ltx_core.loader.sd_ops import SDOps  # noqa: E402
from ltx_core.loader.single_gpu_model_builder import (  # noqa: E402
    SingleGPUModelBuilder,
)
from ltx_pipelines.utils.model_ledger import ModelLedger  # noqa: E402

from engine.pipeline.fast_video_pipeline import (  # noqa: E402
    _LEDGER_BUILDER_ATTRS,
    LTXFastVideoPipeline,
)


class _Configurator:
    """Stand-in for a ModelConfigurator: never called (nothing is built)."""

    @staticmethod
    def from_config(config: dict) -> object:  # pragma: no cover - never invoked
        raise AssertionError("no model is ever built in this test")


class _MarkerLoader:
    """Stand-in for a state-dict loader, used as an identity marker.

    Its survival across the swap is what proves ``dataclasses.replace`` (and
    not ``build_model_builders()``) was used: the real pipeline replaces
    ``model_loader`` with the GGUF loaders, and rebuilding the builders would
    silently put the 46GB-monolith safetensors loader back.
    """

    def __init__(self, tag: str) -> None:
        self.tag = tag


def _builder(tag: str, registry) -> SingleGPUModelBuilder:
    return SingleGPUModelBuilder(
        model_class_configurator=_Configurator,
        model_path=f"/nonexistent/{tag}.safetensors",
        model_sd_ops=SDOps(name=f"OPS_{tag}"),
        model_loader=_MarkerLoader(tag),
        registry=registry,
    )


def _pipeline_with_ledger():
    """A bare pipeline object wired to a real, weight-free ModelLedger.

    ``LTXFastVideoPipeline.__init__`` would build a DistilledPipeline (46GB of
    weights), so the instance is made with ``__new__`` and given only the two
    attributes ``_swap_registry`` touches. The LEDGER is the real thing.
    """
    ledger = ModelLedger(
        dtype=torch.bfloat16,
        device=torch.device("cuda:0"),
        checkpoint_path=None,          # -> no builders, no file access
        gemma_root_path=None,
        spatial_upsampler_path=None,
    )
    start = ledger.registry
    for attr in _LEDGER_BUILDER_ATTRS:
        setattr(ledger, attr, _builder(attr, start))

    pipe = LTXFastVideoPipeline.__new__(LTXFastVideoPipeline)
    pipe.pipeline = types.SimpleNamespace(model_ledger=ledger)
    pipe._keep_resident_enabled = False
    pipe._keep_resident_registry = None
    return pipe, ledger


def test_ledger_and_every_builder_registry_move_together():
    pipe, ledger = _pipeline_with_ledger()
    assert isinstance(ledger.registry, DummyRegistry)

    pipe._swap_registry(True)

    assert isinstance(ledger.registry, StateDictRegistry)
    for attr in _LEDGER_BUILDER_ATTRS:
        builder = getattr(ledger, attr)
        assert builder.registry is ledger.registry, (
            f"{attr} kept the old registry — it would cache into a registry "
            "nothing else reads"
        )
    assert pipe._keep_resident_enabled is True

    # ...and back off: the ledger and all 8 builders return to a Dummy.
    pipe._swap_registry(False)
    assert isinstance(ledger.registry, DummyRegistry)
    for attr in _LEDGER_BUILDER_ATTRS:
        assert getattr(ledger, attr).registry is ledger.registry
        assert isinstance(getattr(ledger, attr).registry, DummyRegistry)
    assert pipe._keep_resident_enabled is False


def test_target_device_follows_the_swap():
    # _target_device() is the wheel's own live read of ledger.registry. Non-
    # Dummy -> submodels build on CPU (cacheable); Dummy -> straight to the GPU.
    pipe, ledger = _pipeline_with_ledger()
    assert ledger._target_device() == ledger.device

    pipe._swap_registry(True)
    assert ledger._target_device() == torch.device("cpu")

    pipe._swap_registry(False)
    assert ledger._target_device() == ledger.device


def test_other_builder_fields_are_preserved():
    # The whole reason for dataclasses.replace over build_model_builders():
    # the install group has already rewritten model_path / model_loader /
    # model_sd_ops on these builders, and losing any of them means the engine
    # quietly reads the wrong weights.
    pipe, ledger = _pipeline_with_ledger()
    before = {
        attr: getattr(ledger, attr) for attr in _LEDGER_BUILDER_ATTRS
    }

    pipe._swap_registry(True)

    for attr, old in before.items():
        new = getattr(ledger, attr)
        assert new is not old                       # replaced, not mutated
        assert new.model_path == old.model_path
        assert new.model_sd_ops is old.model_sd_ops
        assert new.model_loader is old.model_loader
        assert new.model_loader.tag == attr
        assert new.model_class_configurator is old.model_class_configurator


def test_on_reuses_the_same_registry_instance():
    # ON -> OFF -> ON keeps ONE StateDictRegistry object (cleared in between),
    # which is why the documented cost of a round trip is "re-pay the 50-70s
    # warm-up once" rather than "leak a second 20GB cache".
    pipe, ledger = _pipeline_with_ledger()
    pipe._swap_registry(True)
    first = ledger.registry
    pipe._swap_registry(False)
    pipe._swap_registry(True)
    assert ledger.registry is first


def test_off_clears_the_cache():
    pipe, ledger = _pipeline_with_ledger()
    pipe._swap_registry(True)
    registry = ledger.registry
    registry.add(["/nonexistent/x.safetensors"], None, {"w": torch.zeros(2)})
    assert registry.get(["/nonexistent/x.safetensors"], None) is not None

    pipe._swap_registry(False)
    assert registry.get(["/nonexistent/x.safetensors"], None) is None


@pytest.mark.parametrize("missing", _LEDGER_BUILDER_ATTRS)
def test_missing_builder_attribute_trips_the_assert(missing):
    # Every one of the 8 is load-bearing. A getattr(..., None) + silent skip
    # would turn a typo (or a wheel rename) into "that submodel alone is
    # CPU-built with no cache" — slower and heavier, with nothing in any log.
    pipe, ledger = _pipeline_with_ledger()
    delattr(ledger, missing)
    with pytest.raises(AssertionError, match=missing):
        pipe._swap_registry(True)


def test_ledger_builder_attrs_covers_every_builder_the_wheel_makes():
    # Guards the OTHER direction: if the wheel starts creating a NEW builder,
    # _LEDGER_BUILDER_ATTRS must grow with it or that submodel is left behind.
    # checkpoint_path/gemma_root/upsampler are all set here so
    # build_model_builders() creates its full set (no file is opened — the
    # builders are lazy).
    ledger = ModelLedger(
        dtype=torch.bfloat16,
        device=torch.device("cuda:0"),
        checkpoint_path="/nonexistent/checkpoint.safetensors",
        gemma_root_path=None,   # the engine builds text_encoder_builder itself
        spatial_upsampler_path="/nonexistent/upsampler.safetensors",
    )
    wheel_made = {
        name for name in vars(ledger)
        if name.endswith("_builder")
    }
    # text_encoder_builder is ours (engine/gemma/gguf_quant_service.py builds
    # the shardless one because we pass gemma_root=None), so it is the only
    # entry allowed to be in our list without the wheel making it here.
    assert wheel_made <= set(_LEDGER_BUILDER_ATTRS)
    assert set(_LEDGER_BUILDER_ATTRS) - wheel_made == {"text_encoder_builder"}
