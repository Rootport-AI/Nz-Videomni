"""UETrack itself: load the shipped weights and track one frame. CPU only.

RUN THIS ON ``.venv-utils``, NOT on the app venv::

    .venv-utils\\Scripts\\python.exe -m pytest --noconftest tests/test_tracking_runtime_smoke.py

Two reasons, the same pair as ``tests/test_worker_vae_mode_resolve.py:23-34``:
importing ``tracking.uetrack_runtime`` pulls in torch, numpy and cv2, which the
app venv does not have (there the whole module skips); and ``tests/conftest.py``
builds a FastAPI app out of packages ``.venv-utils`` does not have, so the app
conftest must not be collected.

On the app venv this file contributes exactly one SKIP -- that is the expected
state of an ordinary ``pytest`` run, not a gap.

What it is for: everything else in the suite runs against the mock backend, so
this is the only automatic check that the vendored model geometry still matches
the distributed checkpoint. A drift there shows up here as missing keys, and
nowhere else until someone tries to track something.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("cv2")

import numpy as np  # noqa: E402

from tracking.geometry import rgba_bytes_to_rgb  # noqa: E402
from tracking.uetrack_runtime import BASE_SEARCH_FACTOR, load_tracker  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = PROJECT_ROOT / "models" / "UETrack" / "uetrack_base.safetensors"

#: Small on purpose: this is a wiring check, not a benchmark. The tracker's own
#: crops are 112 and 224 square regardless of the frame it is handed.
WIDTH, HEIGHT = 320, 240
BOX = (120.0, 90.0, 60.0, 50.0)


pytestmark = pytest.mark.skipif(
    not CHECKPOINT.is_file(),
    reason=f"UETrack weights not installed ({CHECKPOINT}); run install-UETrack.bat",
)


def _rgba_frame(offset: int) -> bytes:
    """A dark frame with one bright square, moved ``offset`` px down-right.

    Synthetic rather than a fixture image so the repository carries no test
    video, and because a high-contrast blob on a flat field is the one thing a
    correlation tracker cannot plausibly fail to follow -- which is what makes a
    failure here mean "the port is broken" and not "the clip was hard".
    """
    frame = np.zeros((HEIGHT, WIDTH, 4), dtype=np.uint8)
    frame[..., 3] = 255
    x, y, w, h = (int(v) for v in BOX)
    frame[y + offset : y + h + offset, x + offset : x + w + offset, 0] = 230
    frame[y + offset : y + h + offset, x + offset : x + w + offset, 1] = 180
    return frame.tobytes()


@pytest.fixture(scope="module")
def tracker():
    return load_tracker(str(CHECKPOINT), device="cpu")


def test_the_shipped_checkpoint_fills_every_tensor(tracker):
    """No MISSING keys.

    ``load_tracker`` loads non-strictly because the official .tar carries a CLIP
    text branch this port does not build -- so UNEXPECTED keys are legitimate
    and are not asserted on. MISSING ones never are: they would mean the
    vendored geometry has drifted away from the weights, and the tracker would
    then run on partly random parameters rather than fail.
    """
    assert tracker.missing_keys == []


def test_initialize_then_track_returns_a_box_and_a_score(tracker):
    tracker.initialize(
        rgba_bytes_to_rgb(_rgba_frame(0), WIDTH, HEIGHT),
        BOX,
        search_factor=BASE_SEARCH_FACTOR,
    )
    box, score = tracker.track(rgba_bytes_to_rgb(_rgba_frame(6), WIDTH, HEIGHT))

    assert len(box) == 4
    assert all(isinstance(v, float) for v in box)
    assert 0.0 <= score <= 1.0
    # It followed the square rather than sitting still: the blob moved 6 px
    # down-right, so the box must have moved that way too (generously, because
    # the point is direction, not precision).
    assert box[0] > BOX[0] + 1.0
    assert box[1] > BOX[1] + 1.0
    assert box[0] < BOX[0] + 20.0
    assert box[1] < BOX[1] + 20.0
    # And it is still the same object, not a collapse onto a corner.
    assert 0.4 * BOX[2] < box[2] < 2.0 * BOX[2]
    assert 0.4 * BOX[3] < box[3] < 2.0 * BOX[3]
    # A clean, high-contrast target should score well; a weak score here would
    # make the plugin's default lost threshold meaningless.
    assert score > 0.5


def test_track_before_initialize_is_refused():
    """The worker relies on this: ``track`` before ``init`` must raise, not
    return a box computed from whatever state was left by the last session."""
    fresh = load_tracker(str(CHECKPOINT), device="cpu")
    with pytest.raises(RuntimeError):
        fresh.track(rgba_bytes_to_rgb(_rgba_frame(0), WIDTH, HEIGHT))


def test_a_degenerate_seed_box_is_refused(tracker):
    with pytest.raises(ValueError):
        tracker.initialize(
            rgba_bytes_to_rgb(_rgba_frame(0), WIDTH, HEIGHT), (10.0, 10.0, 0.0, 10.0)
        )
