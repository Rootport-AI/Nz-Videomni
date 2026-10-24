"""MiniMax H3 engine constants -- the ONE place each number lives.

Every module under ``engine_h3`` (and the tests) imports these instead of
repeating the literal, because the same handful of numbers -- the frame grid,
the latent strides, the two schedule shifts -- show up in the noise shapes,
the RoPE grid, the VAE and the worker's defaults, and a second copy that
drifted would be a silent shape bug rather than a loud one.

Sources (Docs/MINIMAX_H3_RESEARCH_NOTES.md; "R2" below is the model report
Docs/Outputs-archive/h3-stage1-reports-2026-10-17/R2_model.md):
  * steps 20 / Euler is the app's default (Docs/MINIMAX_H3_DESIGN.md §9; a job
    may send its own ``num_inference_steps``); the reference Euler default is
    50, the Comfy template's 20 is res_multistep.
  * shift 12 (video) / 3 (audio): the official scheduler_config.json pair (the
    defaults; a job may send its own ``shift_video`` / ``shift_audio``).
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
KEYFRAME_NOISE_AUG = 0.999      # I2V condition rows: 0.999 * clean + 0.001 * noise, t >= 0.999
MODALITY_VIDEO, MODALITY_TEXT, MODALITY_AUDIO = 0, 1, 2
LOG_PREFIX = "[h3_worker]"
#: The attention backends the DiT accepts (the one list: worker, pipeline and dit read it).
ATTENTION_BACKENDS = ("sdpa", "sage")

# VRAM 16GB mode (engine_h3.lowvram; 台帳 §1-101, Docs/MINIMAX_H3_DESIGN.md).
# Split units and the swap window come from the shape alone (裁定 51). Fixed by
# gate A-G5 (2026-10-18, RTX 4070 Ti SUPER 16GB, I2V 896x1152x124f = S 39,764,
# sage, 2 steps; outputs/h3-vram16-gates/): the pre-registered rule = among the
# arms within +10% of the fastest 2nd step, the smallest allocation peak; a tie
# goes to the larger unit.
#: Attention heads per group (56 heads -> 7 groups). A-G5: groups of 28 / 14 / 8
#: gave 20.54 / 20.99 / 21.74 s (all within +10%) and allocation peaks of
#: 6,257 / 4,965 / 4,411 MiB -> 8.
VRAM16_HEAD_GROUP = 8
#: Rows per chunk for the FFN, the AdaLN modulation and the final layer
#: (split into ceil(S / 4096) EQUAL chunks; S <= 4096 is not split). A-G5 with
#: groups of 8: 8,192 / 4,096 / 2,048 rows gave 21.75 / 21.74 / 21.28 s and
#: 4,814 / 4,411 / 4,411 MiB -> 4,096 (tie with 2,048 -> the larger chunk).
VRAM16_ROW_CHUNK = 4096
#: DiT blocks kept on the GPU by the swap window in this mode (at most; always,
#: whatever the sequence length). A-G5: 4 vs 8 blocks took 21.74 vs 21.68 s at
#: I2V 896x1152 and 3.40 vs 3.41 s at T2V 640x384 (no difference), for 1,488 /
#: 1,488 MiB less allocation -> the length threshold of the plan was deleted.
VRAM16_BLOCKS_ON_GPU = 4
#: Headroom left below the allocator cap, MiB.
VRAM16_CAP_MARGIN_MIB = 1024

# LoRA (engine_h3.lora, 台帳 §1-104).
#: Output columns per chunk of the LoRA's B stage. Equals VRAM16_HEAD_GROUP x
#: head_dim (128) = the q / k / v rows of one head group, which is what VRAM 16GB
#: mode's ``forward_rows`` hands over; the off path cuts every target the same
#: way, so cuBLAS sees the same shapes on both paths (bit identity by
#: construction). Pinned by a CPU test.
LORA_COL_CHUNK = 1024


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
