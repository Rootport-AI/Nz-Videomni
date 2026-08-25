"""The LTX 2.5 ``keep_resident`` switch, against the REAL official registry.

Covers ``engine25.pipeline25._swap_keep_resident`` -- the whole of the feature's
mechanism -- and the no-op guard in ``Ltx25Pipeline.set_acceleration_job`` that
decides when it is allowed to run.

Everything here is device-free and I/O-free. The registry is the official
``ModelRegistry`` (the point: a wheel that changed how ``_cache_weights`` is
read has to break these, not just the docs), the "builder" is a two-attribute
stand-in -- ``model_path`` and ``model_sd_ops`` are all the swap ever reads off
it -- and the cached "weights" are a ``StateDict`` holding no tensors at all,
because only its ``size`` is looked at.

The four cases are the four things that can go wrong:

1. OFF -> ON makes ``add`` start caching (the ON path is a flag write and
   nothing else, so if the flag is not the thing ``add`` reads, nothing works);
2. re-arming the SAME setting does not touch the cache (an unguarded swap would
   pop the entry the previous job just paid 7.7 GiB to fill -- the feature would
   look armed and hit nothing);
3. ON -> OFF really releases (``get`` does not consult the flag, so clearing it
   without the ``pop`` would leak the state dict forever -- risk R-2);
4. the release uses ``pop`` and not ``clear``, so the cached model SHELLS
   survive (they are structural, cost nothing, and dropping them would make
   every subsequent build reconstruct the module tree for no reason).

Run with ``.venv-engine-ltx25`` and ``--noconftest`` (the app conftest builds a
FastAPI app that venv does not have).
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("ltx_core")

import torch  # noqa: E402
from torch import nn  # noqa: E402

from engine.transformer.sage_attention_service import SageState  # noqa: E402
from engine25.ltxcore_compat import ModelRegistry, StateDict  # noqa: E402
from engine25.pipeline25 import Ltx25Pipeline, _swap_keep_resident  # noqa: E402

#: A path that need not exist: the registry only hashes it (``Path.resolve`` on
#: a missing name is not an error), and nothing here opens a file.
FAKE_TE_GGUF = "S:/nonexistent/LTX-2.5-gemma4-text-encoder.gguf"

#: Stands in for the 7.7 GiB the real text encoder retains.
FAKE_SIZE = 9_876_543_210


class _DummyBuilder:
    """The two attributes ``_swap_keep_resident`` reads off the TE builder.

    Deliberately NOT a ``SingleGPUModelBuilder``: the swap must key the pop the
    same way ``load_state_dict`` keys the add, and the only thing that makes
    that true is these two values. Anything else on a real builder is noise for
    this question.
    """

    def __init__(self, model_path: str = FAKE_TE_GGUF, model_sd_ops: object = None) -> None:
        self.model_path = model_path
        self.model_sd_ops = model_sd_ops


def _state_dict() -> StateDict:
    return StateDict(sd={}, device=torch.device("cpu"), size=FAKE_SIZE, dtype=set())


def _cache(registry: ModelRegistry, builder: _DummyBuilder) -> StateDict:
    """Fill the cache the way ``ltx_core.loader.helpers.load_state_dict`` does."""
    return registry.add([builder.model_path], builder.model_sd_ops, _state_dict())


def _cached(registry: ModelRegistry, builder: _DummyBuilder) -> StateDict | None:
    return registry.get([builder.model_path], builder.model_sd_ops)


def test_arming_on_makes_the_registry_start_caching_weights() -> None:
    """OFF -> ON: ``add`` retains from the next build onwards."""
    registry = ModelRegistry(cache_weights=False, cache_models=True)
    builder = _DummyBuilder()

    # The state the pipeline starts in: the TE builder's registry is built with
    # cache_weights=False, so a build caches nothing.
    _cache(registry, builder)
    assert _cached(registry, builder) is None

    released = _swap_keep_resident(registry, builder, True)
    assert released == 0  # nothing to release when turning it ON

    _cache(registry, builder)
    retained = _cached(registry, builder)
    assert retained is not None
    assert retained.size == FAKE_SIZE


def test_rearming_the_same_setting_leaves_the_cache_intact() -> None:
    """ON -> ON must not pop: the guard in ``set_acceleration_job`` is the point.

    Driven through the real method (bound to a bare instance carrying only the
    attributes it touches) rather than through ``_swap_keep_resident``, because
    the guard -- not the swap -- is what makes a run of identical jobs hit the
    cache instead of rebuilding every time.
    """
    registry = ModelRegistry(cache_weights=False, cache_models=True)
    builder = _DummyBuilder()

    pipeline = Ltx25Pipeline.__new__(Ltx25Pipeline)
    pipeline._te_registry = registry
    pipeline._te_builder = builder
    pipeline._keep_resident_enabled = False
    pipeline._keep_resident_requested = False
    pipeline._keep_resident_used = "off"
    pipeline._block_swap_prefetch_requested = False
    # The bare instance carries only what the two methods under test touch, and
    # the sage arm/reset is now among them: ``set_acceleration_job`` starts by
    # setting the backend on this object and ``reset_acceleration_job`` ends the
    # job by snapshotting it. A real ``SageState`` rather than a stub -- it is
    # four attribute writes with no dependencies, so a stub could only be a
    # place for the two to disagree.
    pipeline._sage = SageState()

    def arm(keep_resident: bool) -> None:
        pipeline.set_acceleration_job(
            block_swap_prefetch=False,
            fused_gguf_dequant_kernel=False,
            keep_resident=keep_resident,
            # Stated explicitly because the parameter has no default: every
            # caller of this method declares its attention backend, so a new
            # entry point cannot inherit one by accident.
            attention_backend="sdpa",
        )

    arm(True)  # job 1 turns it on
    _cache(registry, builder)  # job 1 builds the text encoder
    assert _cached(registry, builder) is not None

    arm(True)  # job 2 asks for the same thing
    assert _cached(registry, builder) is not None, "the re-arm destroyed the cache it should hit"
    assert pipeline._keep_resident_enabled is True

    # ...and the echo the ``done`` event carries is frozen at reset, not here.
    pipeline.reset_acceleration_job()
    assert pipeline.keep_resident_used() == "on"
    assert pipeline._keep_resident_requested is False
    # The asymmetry: the reset froze the echo and left the weights alone.
    assert _cached(registry, builder) is not None
    assert pipeline._keep_resident_enabled is True


def test_arming_off_pops_the_retained_state_dict() -> None:
    """ON -> OFF: the flag alone is not enough, the entry has to be popped."""
    registry = ModelRegistry(cache_weights=False, cache_models=True)
    builder = _DummyBuilder()

    _swap_keep_resident(registry, builder, True)
    _cache(registry, builder)
    assert _cached(registry, builder) is not None

    released = _swap_keep_resident(registry, builder, False)

    assert released == FAKE_SIZE, "the release has to report the bytes it actually gave back"
    assert _cached(registry, builder) is None, (
        "`get` never consults _cache_weights, so clearing the flag without the pop "
        "would keep serving -- and holding -- the retained weights forever"
    )
    # Turning it off twice is harmless and reports an honest zero.
    assert _swap_keep_resident(registry, builder, False) == 0


def test_the_release_keeps_the_cached_model_shells() -> None:
    """``pop``, never ``clear``: the structural shells are not the 7.7 GiB."""
    registry = ModelRegistry(cache_weights=False, cache_models=True)
    builder = _DummyBuilder()
    shell = nn.Linear(2, 2)
    registry.add_model("te-shell", shell)

    _swap_keep_resident(registry, builder, True)
    _cache(registry, builder)
    _swap_keep_resident(registry, builder, False)

    assert registry.get_model("te-shell") is shell
    assert registry._models, "clear() would have taken the shells with the weights"
