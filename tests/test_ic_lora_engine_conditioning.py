"""IC-LoRA conditioning_attention_strength — engine wiring unit tests (CPU-only).

Exercise the reference-conditioning wrapper branch added to
``LTXFastVideoPipeline._reference_conditioning_for_stage`` (fast_video_pipeline.py):
at attention_strength == 1.0 (or default) the reference conditioning is a bare
``VideoConditionByReferenceLatent`` (structurally byte-identical to before);
below 1.0 it is wrapped in ``ConditioningItemAttentionStrengthWrapper`` with the
scalar strength as its ``attention_mask`` (upstream ``iclora_utils`` parity).

Unit-level: no GPU, no model load, no LTX wheel. The pipeline module imports the
``ltx_core`` conditioning classes + the ``ltx_pipelines`` media loader LAZILY
inside the method under test, so we inject tiny stand-in modules into
``sys.modules`` and drive the real method end-to-end. This runs under the
canonical ``.venv`` pytest runner (torch present, ltx wheel absent) exactly like
the torch-only tests in ``test_ic_lora_forward.py``.
"""

from __future__ import annotations

import sys
import types

import pytest

torch = pytest.importorskip("torch")

from engine.pipeline.fast_video_pipeline import LTXFastVideoPipeline  # noqa: E402


# ── Stand-in conditioning classes mirroring the ltx_core signatures ──────────
class _StubVideoConditionByReferenceLatent:
    def __init__(self, latent, downscale_factor=1, strength=1.0):
        self.latent = latent
        self.downscale_factor = downscale_factor
        self.strength = strength


class _StubAttentionStrengthWrapper:
    def __init__(self, conditioning, attention_mask):
        self.conditioning = conditioning
        self.attention_mask = attention_mask


@pytest.fixture
def stub_ltx(monkeypatch):
    """Inject stub ``ltx_core.conditioning`` + ``ltx_pipelines.utils.media_io``.

    The method under test does ``from ltx_core.conditioning import (...)`` and
    ``from ltx_pipelines.utils.media_io import load_video_conditioning`` at call
    time, so pre-seeding sys.modules with these stubs (parents included) makes the
    imports resolve without the real wheel. monkeypatch restores sys.modules.
    """
    cond_mod = types.ModuleType("ltx_core.conditioning")
    cond_mod.VideoConditionByReferenceLatent = _StubVideoConditionByReferenceLatent
    cond_mod.ConditioningItemAttentionStrengthWrapper = _StubAttentionStrengthWrapper

    media_mod = types.ModuleType("ltx_pipelines.utils.media_io")
    media_mod.load_video_conditioning = lambda **kw: torch.zeros(
        1, 3, int(kw["frame_cap"]), int(kw["height"]), int(kw["width"])
    )

    for name, mod in (
        ("ltx_core", types.ModuleType("ltx_core")),
        ("ltx_core.conditioning", cond_mod),
        ("ltx_pipelines", types.ModuleType("ltx_pipelines")),
        ("ltx_pipelines.utils", types.ModuleType("ltx_pipelines.utils")),
        ("ltx_pipelines.utils.media_io", media_mod),
    ):
        monkeypatch.setitem(sys.modules, name, mod)
    return cond_mod


class _FakePipe:
    """Minimal object exposing only the attributes the method reads."""

    def __init__(self, attn_strength: float, scale: int = 2) -> None:
        self._ic_reference = ("ref.mp4", 1.0)
        self._ic_reference_downscale_factor = scale
        self._ic_attention_strength = attn_strength


def _call(attn_strength: float, scale: int = 2):
    """Invoke the real (unbound) method against a fake pipe + stage-1 cond_kwargs."""
    full_height = 768
    cond_height = full_height // 2  # 384 -> stage-1 gate passes
    cond_width = 640  # divisible by scale=2
    cond_kwargs = {
        "height": cond_height,
        "width": cond_width,
        "video_encoder": lambda vid: torch.zeros(1, 128, 3, 4, 5),
        "dtype": torch.float32,
        "device": torch.device("cpu"),
    }
    fake = _FakePipe(attn_strength, scale=scale)
    method = LTXFastVideoPipeline._reference_conditioning_for_stage
    return method(fake, full_height=full_height, num_frames=9, cond_kwargs=cond_kwargs)


def test_reference_conditioning_no_wrap_at_one(stub_ltx):
    """strength 1.0 -> bare VideoConditionByReferenceLatent (no wrapper)."""
    conds = _call(1.0)
    assert len(conds) == 1
    assert isinstance(conds[0], _StubVideoConditionByReferenceLatent)
    assert not isinstance(conds[0], _StubAttentionStrengthWrapper)
    assert conds[0].downscale_factor == 2
    assert conds[0].strength == 1.0


def test_reference_conditioning_wraps_below_one(stub_ltx):
    """strength 0.6 -> ConditioningItemAttentionStrengthWrapper(mask == 0.6)."""
    conds = _call(0.6)
    assert len(conds) == 1
    wrapped = conds[0]
    assert isinstance(wrapped, _StubAttentionStrengthWrapper)
    assert wrapped.attention_mask == pytest.approx(0.6)
    assert isinstance(wrapped.conditioning, _StubVideoConditionByReferenceLatent)
    # underlying reference item is unchanged (strength + downscale preserved)
    assert wrapped.conditioning.strength == 1.0
    assert wrapped.conditioning.downscale_factor == 2
