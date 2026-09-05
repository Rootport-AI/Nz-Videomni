"""LTX 2.5 reference VAE-encode: which branch a reference takes, and why (§3-76).

``engine25.reference25.reference_conditioning_from_pixels`` has two encode
paths, and until §3-76 the choice was spelled ``scale == 1`` in three separate
places (the ``channels_last_3d`` decision, the branch itself, and the log line).
It is now one predicate::

    tiled = scale == 1 or ref_tokens > REFERENCE_ENCODE_TILE_TOKEN_BUDGET

and the ``channels_last_3d`` re-layout follows ``tiled`` rather than ``scale``.
That pairing is the load-bearing part: tiling with contiguous ``Conv3d`` weights
was measured OOM in this repository, so a future edit that lets tiling happen
without the layout switch would reintroduce a failure mode nothing else here
would catch.

What this file pins, all of it CPU-only and model-free:

1. A factor-2 reference at or below the token budget still takes the ONE-SHOT
   encode -- the byte-identity guarantee for every output shipped so far.
2. Above the budget it takes ``tiled_encode``, AND the tiling config the caller
   threaded in arrives there. The three 2.5 call sites (``pipeline25``,
   ``chain25``, ``outpaint25``) all pass ``AUTO_TILING``, so a ``None`` reaching
   the encoder would mean the config was dropped in transit, not that a caller
   omitted it.
3. Factor 1 (deblur) is tiled regardless of length -- the token budget must not
   be able to turn the factor-1 path OFF, which is the one direction that OOMs.
4. The budget's edge sits exactly where the arithmetic says it does.

This is the FIRST test over the 2.5 reference encode, so the fixtures are built
from scratch rather than borrowed: a fake encoder that records which of its two
entry points was called, and a ``meta`` reference tensor so a 665-frame probe
costs no memory. Both the tensor AND the ``device`` argument are ``meta``:
``meta -> cpu`` is not a legal ``.to()``, so a cpu device against a meta tensor
would fail the untiled branch for the wrong reason, and a meta device keeps the
``cuda``-gated re-layout branch off at the same time.

Run with ``.venv-engine-ltx25`` and ``--noconftest`` (the app conftest builds a
FastAPI app that venv does not have); in the app ``.venv`` the importorskip
below skips the module whole::

    $env:PYTHONPATH="."; .\\.venv-engine-ltx25\\Scripts\\python.exe -m pytest tests\\test_ltx25_reference_encode.py --noconftest --rootdir . -q

``.venv-engine-ltx25`` has no ``pytest`` of its own, so that command needs the
runner from VERIFICATION_LOG §76.4: a two-line script that APPENDS the app
venv's ``site-packages`` to ``sys.path`` (appends, so ``torch``/``ltx_core``
still resolve to the engine venv) and then calls ``pytest.main``.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("ltx_core")

from chain_math import (  # noqa: E402
    REFERENCE_ENCODE_TILE_TOKEN_BUDGET,
    reference_encode_tokens,
)
from engine25 import reference25  # noqa: E402
from engine25.reference25 import reference_conditioning_from_pixels  # noqa: E402

#: Reference pixel dims for every probe below: (320 // 32) * (192 // 32) == 60
#: latent patches per video latent frame, which puts the budget's edge on a
#: whole latent frame and keeps the frame counts short enough to read.
REF_H = 192
REF_W = 320
PATCHES = (REF_W // 32) * (REF_H // 32)

#: Pixel frames whose reference token count lands just below / just above
#: :data:`REFERENCE_ENCODE_TILE_TOKEN_BUDGET` (see ``test_budget_edge``).
UNDER_FRAMES = 657   # 83 video latent frames -> 4,980 tokens
OVER_FRAMES = 665    # 84 video latent frames -> 5,040 tokens

#: Stands in for the ``TilingConfig`` the three callers thread through.
_TILING_SENTINEL = object()


class _FakeVideoEncoder:
    """Records which entry point ran, with what, and with which tiling config.

    The real ``VideoEncoder`` is callable (plain encode) AND carries
    ``tiled_encode``; the function under test picks one, so both have to exist
    and both have to be observable.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.tiling_configs: list[object] = []
        self.inputs: list[torch.Tensor] = []

    def __call__(self, video):
        self.calls.append("plain")
        self.inputs.append(video)
        return torch.zeros(1, 128, 3, 4, 5)

    def tiled_encode(self, video, tiling_config=None):
        self.calls.append("tiled")
        self.tiling_configs.append(tiling_config)
        self.inputs.append(video)
        return torch.zeros(1, 128, 3, 4, 5)


@pytest.fixture(autouse=True)
def _no_cleanup(monkeypatch):
    """Neutralise ``cleanup_memory``.

    It is ``gc.collect`` + ``torch.cuda.empty_cache`` + ``synchronize``; on a
    box with a GPU the real one would initialise a CUDA context for a test that
    has nothing to do with the GPU.
    """
    monkeypatch.setattr(reference25, "cleanup_memory", lambda: None)


def _encode(frames: int, scale: int = 2, encoder=None) -> _FakeVideoEncoder:
    """Drive the real function over a ``meta`` reference; return the encoder."""
    enc = encoder if encoder is not None else _FakeVideoEncoder()
    video = torch.zeros(1, 3, frames, REF_H, REF_W, device="meta")
    conds = reference_conditioning_from_pixels(
        video,
        video_encoder=enc,
        device=torch.device("meta"),
        tiling_config=_TILING_SENTINEL,
        scale=scale,
        strength=1.0,
    )
    # The conditioning itself is unaffected by the branch: same one item, same
    # declared factor, whichever encode produced the latent.
    assert len(conds) == 1
    assert conds[0].downscale_factor == scale
    return enc


def test_factor_two_under_budget_stays_one_shot():
    """657 frames -> 4,980 tokens, at or below the budget: ONE-SHOT encode.

    This is the byte-identity guarantee. Every factor-2 reference the Chained
    screen's stage-1 comfort banner stays silent about must keep taking the
    branch it has always taken, so already-shipped outputs do not move.
    """
    assert reference_encode_tokens(REF_W, REF_H, UNDER_FRAMES) == 4_980
    assert reference_encode_tokens(REF_W, REF_H, UNDER_FRAMES) <= REFERENCE_ENCODE_TILE_TOKEN_BUDGET

    enc = _encode(UNDER_FRAMES, scale=2)
    assert enc.calls == ["plain"]
    assert enc.tiling_configs == []


def test_factor_two_over_budget_tiles_with_the_callers_config():
    """665 frames -> 5,040 tokens, above the budget: ``tiled_encode``.

    And the tiling config it is handed is the caller's own object, not
    ``None``: all three 2.5 call sites pass ``AUTO_TILING``, so a ``None``
    arriving here would mean the config was dropped between them and the
    encoder -- silently falling back to whatever default the encoder picks.
    """
    assert reference_encode_tokens(REF_W, REF_H, OVER_FRAMES) == 5_040
    assert reference_encode_tokens(REF_W, REF_H, OVER_FRAMES) > REFERENCE_ENCODE_TILE_TOKEN_BUDGET

    enc = _encode(OVER_FRAMES, scale=2)
    assert enc.calls == ["tiled"]
    assert enc.tiling_configs == [_TILING_SENTINEL]
    assert enc.tiling_configs[0] is not None


def test_factor_one_is_tiled_however_short():
    """Factor 1 (deblur) is tiled at ANY length -- the budget cannot switch it off.

    9 frames is 2 video latent frames, 120 tokens: two orders of magnitude
    under the budget. A factor-1 reference carries 4x the pixels of a factor-2
    one at the same output size and OOM'd a 16GB card untiled, so ``scale == 1``
    has to short-circuit the token comparison rather than be compared alongside
    it.
    """
    enc = _encode(9, scale=1)
    assert enc.calls == ["tiled"]
    assert enc.tiling_configs == [_TILING_SENTINEL]
    # ...and the tiled path does NOT move the video: tiled_encode streams tiles
    # to the device itself, which is why the untiled branch is the only one
    # carrying a ``.to(device)``.
    assert enc.inputs[0].device.type == "meta"


def test_budget_edge():
    """The last one-shot frame count and the first tiled one, spelled out.

    ``reference_encode_tokens(w, h, F) == (w//32) * (h//32) * v_latent_frames(F)``
    with ``v_latent_frames(F) == (F - 1)//8 + 1``. At 320x192 that is 60 latent
    patches per latent frame, so the budget of 5,000 lands mid-frame --
    ``5000 / 60 == 83.33`` -- and the switch happens between latent frame 83
    (60 x 83 = 4,980) and 84 (60 x 84 = 5,040). Latent frame 83 spans pixel
    frames 657..664 and 84 spans 665..672, so 664 is the LAST one-shot count
    and 665 the first tiled one. The whole 8-frame group flips together
    because a latent frame is indivisible, not because of any rounding here.
    """
    assert reference_encode_tokens(REF_W, REF_H, 664) == PATCHES * 83 == 4_980
    assert reference_encode_tokens(REF_W, REF_H, 665) == PATCHES * 84 == 5_040
    assert 4_980 <= REFERENCE_ENCODE_TILE_TOKEN_BUDGET < 5_040

    assert _encode(664, scale=2).calls == ["plain"]
    assert _encode(665, scale=2).calls == ["tiled"]
