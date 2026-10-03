"""What the selftest's ``install_ran`` relies on (台帳 §1-47).

``engine25.gguf_transformer._selftest`` records, per round, how much the build
moved ``Ltx25DiffusionStage._prefetch_builds`` (``install_ran``), and holds
every swap-path round to ``install_ran == 1``. That check is only worth
something if:

* an install that actually runs returns True and adds exactly one to
  ``_prefetch_builds``;
* an install SKIPPED because the blocks still carry the previous build's marker
  returns False and adds nothing;
* ``last_prefetch_used`` still reads the previous build's value after the skip
  -- i.e. ``prefetch_used`` alone cannot see the skip, which is why the count
  is needed at all.

The REAL ``BlockSwapService`` and the REAL stage methods run here, bound to a
``SimpleNamespace`` self, on a tiny CPU model. Only the two prefetch-engine
hooks that would touch CUDA are replaced. ``_selftest`` itself needs a GPU and a
real GGUF, so it is not called.

NO GPU. Run with ``.venv-engine-ltx25`` and ``--noconftest``::

    .venv-engine-ltx25/Scripts/python.exe \\
        Docs/Outputs-archive/start-end-bridge-2026-09-07/implA_engine_runner/run_ltx25_pytest.py \\
        tests/test_ltx25_prefetch_builds_count.py
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("ltx_core")

from torch import nn  # noqa: E402

from engine.transformer.block_swap_service import BlockSwapService  # noqa: E402
from engine25 import gguf_transformer as gt  # noqa: E402

Stage = gt.Ltx25DiffusionStage


class _Inner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer_blocks = nn.ModuleList([nn.Linear(2, 2) for _ in range(6)])


@pytest.fixture()
def rig():
    """A 6-block CPU model, a real swap service keeping 2 on the device, and a
    stand-in stage with prefetch requested."""
    model = nn.Module()
    model.velocity_model = _Inner()

    svc = BlockSwapService(blocks_on_gpu=2, device=torch.device("cpu"))
    # The two hooks that would build a CUDA prefetch engine; the stand-in
    # patch leaves the same marker the real one does.
    svc._build_prefetch_engine = lambda blocks: SimpleNamespace(teardown=lambda: None)
    svc._patch_block_prefetch = (
        lambda block, idx, engine: setattr(block, gt._BLOCK_SWAP_ATTR, block.forward)
    )

    stage = SimpleNamespace(
        _swap_service=svc,
        _prefetch_requested=False,
        _prefetch_builds=0,
        _prefetch_engaged=0,
    )
    Stage.set_block_swap_prefetch(stage, True)
    return stage, svc, model


def test_an_install_that_runs_counts_once(rig):
    stage, svc, model = rig
    assert Stage.ensure_block_swap_installed(stage, model) is True
    assert stage._prefetch_builds == 1
    assert svc.last_prefetch_used == "on"


def test_a_skipped_install_does_not_count(rig):
    stage, svc, model = rig
    Stage.ensure_block_swap_installed(stage, model)
    before = stage._prefetch_builds

    # Next "build" without stripping the marker: what a regression in
    # ``_place_transformer``'s ``_unpatch_block_swap`` would look like.
    svc.teardown_prefetch()
    assert Stage.ensure_block_swap_installed(stage, model) is False
    assert stage._prefetch_builds == before  # install_ran == 0 for this round


def test_prefetch_used_alone_cannot_see_the_skip(rig):
    """The property the count exists for: after a skipped install,
    ``last_prefetch_used`` still reads the previous build's "on"."""
    stage, svc, model = rig
    Stage.ensure_block_swap_installed(stage, model)
    svc.teardown_prefetch()  # the build's own first step; does not reset the value
    Stage.ensure_block_swap_installed(stage, model)
    assert svc.last_prefetch_used == "on"


def test_a_stripped_rebuild_counts_again(rig):
    """The normal path: the marker is stripped, so the second build installs
    and ``_prefetch_builds`` moves by one again."""
    stage, svc, model = rig
    Stage.ensure_block_swap_installed(stage, model)
    svc.teardown_prefetch()
    Stage._unpatch_block_swap(stage, model)
    assert Stage.ensure_block_swap_installed(stage, model) is True
    assert stage._prefetch_builds == 2
