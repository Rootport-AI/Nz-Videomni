"""MiniMax H3 engine constants -- the ONE place each number lives.

Every module under ``engine_h3`` (and the tests) imports these instead of
repeating the literal, because the same handful of numbers -- the frame grid,
the latent strides, the two schedule shifts -- show up in the noise shapes,
the RoPE grid, the VAE and the worker's defaults, and a second copy that
drifted would be a silent shape bug rather than a loud one.

Sources (Docs/MINIMAX_H3_RESEARCH_NOTES.md, scratchpad R2 report):
  * steps 20 / Euler is the supervisor's first-gate choice (plan section 1);
    the reference Euler default is 50, the Comfy template's 20 is res_multistep.
  * shift 12 (video) / 3 (audio): the official scheduler_config.json pair.
  * 17n+5 frames, T' = 5n+2 latent frames: the video VAE's clip length 17,
    temporal stride 4 and token_drop 3 (R2 F).
  * 40 audio latents per second at 32 kHz, hop 800 (R2 F).

torch-free on purpose: the worker imports this module before torch is loaded.
"""

from __future__ import annotations

DEFAULT_STEPS = 20
DEFAULT_SHIFT_VIDEO = 12.0
DEFAULT_SHIFT_AUDIO = 3.0
FRAME_RATE = 24                 # fixed: the model was trained at 24 fps only
SPATIAL_FACTOR = 32             # one DiT token = 32 px square (VAE 16 x patch 2)
VAE_SPATIAL = 16
VAE_TEMPORAL = 4
VAE_CLIP_LENGTH = 17
VAE_TOKEN_DROP = 3
AUDIO_LATENTS_PER_SECOND = 40
AUDIO_SAMPLE_RATE = 32000
AUDIO_HOP = 800                 # one audio latent = 800 samples
KEYFRAME_NOISE_AUG = 0.999      # used from the second stage on (I2V condition rows)
MODALITY_VIDEO, MODALITY_TEXT, MODALITY_AUDIO = 0, 1, 2
LOG_PREFIX = "[h3_worker]"
#: The attention backends the DiT accepts (the one list: worker, pipeline and dit read it).
ATTENTION_BACKENDS = ("sdpa", "sage")


def latent_frames(num_frames: int) -> int:
    """``F = 17n + 5`` pixel frames -> ``T' = 5n + 2`` latent frames.

    Any other F is refused rather than rounded: the decoder emits exactly
    17n+5 frames, so a request off that grid would come back a different
    length than it asked for (R2 F, "17n+5 の導出").
    """
    f = int(num_frames)
    if f < 5 or (f - 5) % VAE_CLIP_LENGTH != 0:
        raise ValueError(
            f"フレーム数 {f} は 17n+5 の形ではありません（例: 124・192・243・345）"
        )
    return (f - 5) // VAE_CLIP_LENGTH * 5 + 2


def audio_latents(num_frames: int) -> int:
    """Audio latents per channel: ``round(F / 24 * 40)`` (Python round, as DiffSynth)."""
    return round(int(num_frames) / FRAME_RATE * AUDIO_LATENTS_PER_SECOND)
