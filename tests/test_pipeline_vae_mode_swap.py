"""Per-job video VAE decoder switching (PrunaVAED, §3-50) — gate G4's main half.

``LTXFastVideoPipeline._set_vae_mode_job`` re-points ONE ledger builder per job.
Three properties have to hold, and all three are provable in process — which is
why G4's primary evidence lives here rather than in a three-job real-device run
(a real run can only show "it did not get slower", never "the registry held both
state dicts"):

  1. **Owner's hard requirement 0-3** — with ``keep_resident`` on, the stock and
     the pruned state dict COEXIST in one ``StateDictRegistry`` (their cache keys
     differ because their PATHS differ), so a default -> pruned -> default batch
     re-reads nothing from disk.
  2. **Registry freshness (§4.4)** — ``self._default_vae_builder`` is a snapshot
     taken once in ``__init__``, and the ``registry`` reference inside it is
     frozen at that moment. Every assignment must therefore re-inject
     ``ledger.registry``, on BOTH branches, or a later keep_resident toggle
     leaves the decoder caching into a registry nothing else reads.
  3. **Per-job existence check (§8 E5)** — a missing weight file degrades THIS
     job to the stock decoder and reports ``"on->off"``, with no worker restart
     needed to notice the file coming back.

Built like ``test_registry_swap.py``: the real ``ModelLedger`` (weight-free) plus
hand-made ``SingleGPUModelBuilder`` instances, and the pipeline object made with
``__new__`` because ``__init__`` would load 46GB of weights. The state-dict
loader is a stub that COUNTS calls — that counter is the direct measurement of
"did this job go to disk".

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
from engine.vae.pruned_video_decoder import PrunedVideoDecoderConfigurator  # noqa: E402


class _Configurator:
    """Stand-in for the STOCK VideoDecoderConfigurator: never called."""

    @staticmethod
    def from_config(config: dict) -> object:  # pragma: no cover - never invoked
        raise AssertionError("no model is ever built in this test")


class _CountingLoader:
    """State-dict loader stub that records every disk read it is asked for.

    ``SingleGPUModelBuilder.load_sd`` consults the registry first and only calls
    this on a MISS, so ``calls`` is exactly "how many times this configuration
    went to disk" — the quantity owner requirement 0-3 is about.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], str | None]] = []

    def load(self, paths, sd_ops=None, device=None):
        self.calls.append((tuple(paths), None if sd_ops is None else sd_ops.name))
        return {"marker": tuple(paths)}

    def metadata(self, path):  # pragma: no cover - not exercised here
        raise AssertionError("no metadata is read in this test")


def _builder(tag: str, registry, loader=None, model_path: str | None = None):
    return SingleGPUModelBuilder(
        model_class_configurator=_Configurator,
        model_path=model_path or f"/nonexistent/{tag}.safetensors",
        model_sd_ops=SDOps(name=f"OPS_{tag}"),
        model_loader=loader or _CountingLoader(),
        registry=registry,
    )


def _pipeline(tmp_path, *, pruned_exists: bool):
    """A bare pipeline wired to a real, weight-free ledger.

    ``stock`` and ``pruned`` are distinct paths (that difference is the whole
    reason the two cache keys cannot collide). The pruned file is created only
    when the test wants it to exist — nothing ever opens it.
    """
    stock = tmp_path / "LTX23_video_vae_bf16.safetensors"
    stock.write_bytes(b"\0")
    pruned = tmp_path / "prunavaed" / "PrunaVAED-decoder-bf16.safetensors"
    if pruned_exists:
        pruned.parent.mkdir(parents=True, exist_ok=True)
        pruned.write_bytes(b"\0")

    ledger = ModelLedger(
        dtype=torch.bfloat16,
        device=torch.device("cuda:0"),
        checkpoint_path=None,          # -> no builders, no file access
        gemma_root_path=None,
        spatial_upsampler_path=None,
    )
    loader = _CountingLoader()
    for attr in _LEDGER_BUILDER_ATTRS:
        setattr(
            ledger,
            attr,
            _builder(
                attr,
                ledger.registry,
                loader=loader if attr == "vae_decoder_builder" else None,
                model_path=str(stock) if attr == "vae_decoder_builder" else None,
            ),
        )

    pipe = LTXFastVideoPipeline.__new__(LTXFastVideoPipeline)
    pipe.pipeline = types.SimpleNamespace(model_ledger=ledger)
    pipe._keep_resident_enabled = False
    pipe._keep_resident_registry = None
    pipe._component_video_vae_pruned_path = str(pruned)
    pipe._vae_mode_used = "off"
    # Same moment as the real __init__ takes it: AFTER the install group and the
    # optional _swap_registry(True), and exactly once.
    pipe._default_vae_builder = ledger.vae_decoder_builder
    return pipe, ledger, loader, str(stock), str(pruned)


def _build_once(ledger):
    """Do what ``ledger.video_decoder()`` does to the state dict, and no more.

    ``SingleGPUModelBuilder.build`` would also construct the model; this is the
    ONE line of it that touches the registry.
    """
    b = ledger.vae_decoder_builder
    return b.load_sd(
        [b.model_path], registry=b.registry, device=None, sd_ops=b.model_sd_ops
    )


# --------------------------------------------------------------------------- #
# builder swapping
# --------------------------------------------------------------------------- #

def test_vae_decoder_builder_is_covered_by_ledger_builder_attrs():
    # Regression guard: if vae_decoder_builder ever drops out of this tuple, the
    # decoder alone stops following keep_resident's registry swap — a silent
    # partial failure (fast_video_pipeline.py's own warning comment).
    assert "vae_decoder_builder" in _LEDGER_BUILDER_ATTRS


def test_default_mode_keeps_the_stock_builder(tmp_path):
    pipe, ledger, _loader, stock, _pruned = _pipeline(tmp_path, pruned_exists=True)
    pipe._set_vae_mode_job("default")

    builder = ledger.vae_decoder_builder
    assert builder.model_path == stock
    assert builder.model_class_configurator is _Configurator
    assert builder.model_sd_ops is not None      # the stock key-filter chain survives
    assert pipe.vae_mode_used() == "off"


def test_pruned_mode_swaps_path_configurator_and_drops_sd_ops(tmp_path):
    pipe, ledger, _loader, _stock, pruned = _pipeline(tmp_path, pruned_exists=True)
    pipe._set_vae_mode_job("prune_vaed")

    builder = ledger.vae_decoder_builder
    assert builder.model_path == pruned
    assert builder.model_class_configurator is PrunedVideoDecoderConfigurator
    # model_sd_ops MUST be None, not a freshly named SDOps: an SDOps with no
    # matcher drops EVERY key (sd_ops.py's any([]) is False), which strict=False
    # would then hide behind a single warning.
    assert builder.model_sd_ops is None
    assert pipe.vae_mode_used() == "on"


def test_missing_weight_file_degrades_this_job_only(tmp_path):
    pipe, ledger, _loader, stock, pruned = _pipeline(tmp_path, pruned_exists=False)
    pipe._set_vae_mode_job("prune_vaed")

    assert ledger.vae_decoder_builder.model_path == stock
    assert ledger.vae_decoder_builder.model_class_configurator is _Configurator
    assert pipe.vae_mode_used() == "on->off"

    # The check is per JOB, so dropping the file in makes the very next job pick
    # it up — no worker restart, which is the point of A-3's fix.
    import os

    os.makedirs(os.path.dirname(pruned), exist_ok=True)
    with open(pruned, "wb") as fh:
        fh.write(b"\0")
    pipe._set_vae_mode_job("prune_vaed")
    assert ledger.vae_decoder_builder.model_path == pruned
    assert pipe.vae_mode_used() == "on"


def test_unset_path_degrades(tmp_path):
    pipe, ledger, _loader, stock, _pruned = _pipeline(tmp_path, pruned_exists=True)
    pipe._component_video_vae_pruned_path = ""
    pipe._set_vae_mode_job("prune_vaed")
    assert ledger.vae_decoder_builder.model_path == stock
    assert pipe.vae_mode_used() == "on->off"


def test_switching_back_to_default_restores_the_stock_builder(tmp_path):
    # A -> B -> A. There is no _reset_vae_mode_job by design: every job sets the
    # mode explicitly, so nothing can leak from the previous one (§8 E7).
    pipe, ledger, _loader, stock, pruned = _pipeline(tmp_path, pruned_exists=True)
    pipe._set_vae_mode_job("default")
    pipe._set_vae_mode_job("prune_vaed")
    assert ledger.vae_decoder_builder.model_path == pruned
    pipe._set_vae_mode_job("default")
    assert ledger.vae_decoder_builder.model_path == stock
    assert ledger.vae_decoder_builder.model_sd_ops is not None
    assert pipe.vae_mode_used() == "off"


def test_the_snapshot_itself_is_never_mutated(tmp_path):
    # Builders are frozen dataclasses and the pruned one is derived per job, so
    # the held snapshot must still describe the STOCK decoder afterwards.
    pipe, _ledger, _loader, stock, _pruned = _pipeline(tmp_path, pruned_exists=True)
    before = pipe._default_vae_builder
    pipe._set_vae_mode_job("prune_vaed")
    assert pipe._default_vae_builder is before
    assert pipe._default_vae_builder.model_path == stock
    assert pipe._default_vae_builder.model_class_configurator is _Configurator


# --------------------------------------------------------------------------- #
# registry freshness (§4.4)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("mode", ["default", "prune_vaed"])
def test_registry_is_reinjected_on_both_branches(tmp_path, mode):
    # The snapshot was taken while the ledger still had a DummyRegistry. After
    # keep_resident arms a StateDictRegistry, assigning the snapshot verbatim
    # would hand the decoder a registry nothing else reads. Both branches must
    # re-inject — which is why even the "default" branch goes through
    # dataclasses.replace instead of assigning the snapshot directly.
    pipe, ledger, _loader, _stock, _pruned = _pipeline(tmp_path, pruned_exists=True)
    assert isinstance(pipe._default_vae_builder.registry, DummyRegistry)

    pipe._swap_registry(True)
    assert isinstance(ledger.registry, StateDictRegistry)

    pipe._set_vae_mode_job(mode)
    assert ledger.vae_decoder_builder.registry is ledger.registry

    # ...and the same on the way back down.
    pipe._swap_registry(False)
    pipe._set_vae_mode_job(mode)
    assert ledger.vae_decoder_builder.registry is ledger.registry
    assert isinstance(ledger.vae_decoder_builder.registry, DummyRegistry)


# --------------------------------------------------------------------------- #
# owner requirement 0-3: both state dicts coexist, no re-read on the way back
# --------------------------------------------------------------------------- #

def test_both_decoders_coexist_in_one_registry_and_the_third_job_hits_cache(tmp_path):
    pipe, ledger, loader, stock, pruned = _pipeline(tmp_path, pruned_exists=True)
    pipe._swap_registry(True)                      # keep_resident ON
    registry = ledger.registry

    pipe._set_vae_mode_job("default")              # job A
    _build_once(ledger)
    pipe._set_vae_mode_job("prune_vaed")           # job B
    _build_once(ledger)
    pipe._set_vae_mode_job("default")              # job C
    _build_once(ledger)

    # Job C read nothing: two disk reads for three jobs.
    assert [paths for paths, _ops in loader.calls] == [(stock,), (pruned,)]
    assert len(loader.calls) == 2

    # Both entries are still there, under two DIFFERENT ids. This is the direct
    # proof of requirement 0-3 — and of §4.4's claim that differing PATHS alone
    # keep the keys apart (the pruned side carries no sd_ops name at all).
    stock_id = registry._generate_id([stock], SDOps(name="OPS_vae_decoder_builder"))
    pruned_id = registry._generate_id([pruned], None)
    assert stock_id != pruned_id
    assert set(registry._state_dicts) == {stock_id, pruned_id}
    assert registry.get([stock], SDOps(name="OPS_vae_decoder_builder")) is not None
    assert registry.get([pruned], None) is not None


def test_keep_resident_off_releases_both(tmp_path):
    # Turning keep_resident off clears the ONE registry that holds both, which
    # is the intended behaviour (§4.4): the pruned decoder adds ~690MB to the
    # resident set and leaves with everything else.
    pipe, ledger, _loader, stock, pruned = _pipeline(tmp_path, pruned_exists=True)
    pipe._swap_registry(True)
    registry = ledger.registry
    pipe._set_vae_mode_job("default")
    _build_once(ledger)
    pipe._set_vae_mode_job("prune_vaed")
    _build_once(ledger)
    assert len(registry._state_dicts) == 2

    pipe._swap_registry(False)
    assert registry._state_dicts == {}
