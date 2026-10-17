r"""G5 regression bisect: rerun the after-review G5 with the video VAE's OLD cast order.

Before REVIEW_code1 m2 the video VAE was cast whole to bf16 and the float32
islands were then put back with ``.float()`` (F16 -> bf16 -> f32, lossy). If the
after-review run differs from the pre-review digest ONLY because m2 now casts
the islands F16 -> f32 directly, this run (old order, everything else current)
reproduces the pre-review sha256 2a09ee8ed86d2e94...

    .venv-engine-h3\Scripts\python.exe outputs\h3-gate1\g5_bisect_old_vae_cast.py <selftest args...>
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine_h3 import video_vae, worker  # noqa: E402


def old_apply_precision(self, dtype):
    mean, std = self.latents_mean.clone(), self.latents_std.clone()  # float32, as the old code rebuilt them
    self.to(dtype=dtype)
    for m in self.float32_islands():
        m.float()
    self.latents_mean, self.latents_std = mean, std


video_vae.H3VideoVAE.apply_precision = old_apply_precision
raise SystemExit(worker._selftest_generate(sys.argv[1:]))
