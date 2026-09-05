"""§3-105 F2: ``BlockSwapService.release_installed()`` actually frees its hold.

Covers the keep-latest leak fix's counterpart: ``install()`` retains the job's
transformer in ``_installed_transformers`` until the NEXT job's install() call
(see the comment there), which is the reference this feature drops early, at
end-of-job, via ``release_installed()``.

The test is a weakref round-trip on one dummy transformer: after install(),
the service is the transformer's only external strong reference, so deleting
the local variable and running ``gc.collect()`` must NOT free it (the leak,
observed directly) — and after ``release_installed()``, the same del + gc
sequence must free it (the fix). Anything in between (block content, device)
does not matter for this; a CPU-only dummy keeps the test GPU-free.

Guards against an empty pass: ``install()`` has three early returns
(``blocks_on_gpu == 0`` / no blocks found / ``blocks_on_gpu >= total``) that
would all skip appending to ``_installed_transformers`` — the test asserts
the list actually holds the transformer before proceeding.

Run with ``.venv-engine`` (module only needs torch, no ltx_core / CUDA).
"""

from __future__ import annotations

import gc
import weakref

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402
from torch import nn  # noqa: E402

from engine.transformer.block_swap_service import BlockSwapService  # noqa: E402


def _dummy_transformer(num_blocks: int = 8) -> nn.Module:
    """Minimal stand-in exposing ``transformer_blocks`` — the first attribute
    name ``BlockSwapService._get_blocks`` tries."""
    transformer = nn.Module()
    transformer.transformer_blocks = nn.ModuleList(
        nn.Linear(4, 4) for _ in range(num_blocks)
    )
    return transformer


def test_release_installed_lets_the_transformer_be_collected():
    svc = BlockSwapService(blocks_on_gpu=2, device=torch.device("cpu"))
    transformer = _dummy_transformer()
    svc.install(transformer)

    # Precondition: install() actually kept the reference (not one of its
    # early-return paths), or the assertions below would pass vacuously.
    assert len(svc._installed_transformers) == 1

    ref = weakref.ref(transformer)
    del transformer
    gc.collect()
    assert ref() is not None, "service should still hold the keep-latest reference"

    svc.release_installed()
    assert svc._installed_transformers == []

    gc.collect()
    assert ref() is None, "release_installed() should let gc reclaim the transformer"
