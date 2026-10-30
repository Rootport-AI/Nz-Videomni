"""The chain checks specific to LTX (2.3 and 2.5); applied by
``services.engines.validate_chain_for_family``.

The messages and their order are the ones clients parse (pinned by
tests/test_chain.py ``test_ltx_chain_422_body_is_pinned``); only a request
breaking two rules at once can see a schema check or a ``FEATURE_UNSUPPORTED``
first.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import chain_math
from api import models
from services.engines.ltx.request_rules import LTX_DISTILLED_STEPS, LTX_REQUEST_RULES

if TYPE_CHECKING:  # pragma: no cover - typing only
    from api.models import GenerateChainRequest


def validate_ltx_chain(request: "GenerateChainRequest") -> None:
    """Raise ``ValueError`` with the message of the first LTX rule ``request`` breaks.

    In order: the 64 multiple, the distilled pipeline's 8 steps / CFG 1.0,
    then per clip the 8n+1 grid, the overlap against the clip's stage-1 latent
    frames and the keyframes snapped onto LTX's grid (written back in place),
    then the whole timeline's geometry (``chain_math.compute_chain_layout``)
    and its frame cap.
    """
    if request.width % 64 != 0:
        raise ValueError("width must be a multiple of 64")
    if request.height % 64 != 0:
        raise ValueError("height must be a multiple of 64")
    if request.pipeline == "distilled":
        if request.num_inference_steps != LTX_DISTILLED_STEPS:
            raise ValueError(
                f"distilled pipeline requires num_inference_steps={LTX_DISTILLED_STEPS}"
            )
        if request.guidance_scale != 1.0:
            raise ValueError("distilled pipeline requires guidance_scale=1.0")

    for i, clip in enumerate(request.clips):
        if (clip.num_frames - 1) % 8 != 0:
            raise ValueError(f"clips[{i}].num_frames must be 8n+1")
        # overlap_frames (K_v LATENT) must fit inside EVERY clip's stage-1
        # latent-frame count = (num_frames - 1)//8 + 1 (each segment either
        # provides or receives the K_v-frame overlap).
        stage1_frames = (clip.num_frames - 1) // 8 + 1
        if request.overlap_frames >= stage1_frames:
            raise ValueError(
                f"overlap_frames ({request.overlap_frames}) must be < "
                f"clips[{i}] stage-1 latent frames ({stage1_frames})"
            )
        # Snap onto LTX's keyframe grid and refuse duplicates (the count cap is
        # not a family fact and stays on the schema).
        message = LTX_REQUEST_RULES.normalize_keyframes(
            clip.conditioning_images, clip.num_frames, where=f"clips[{i}]: "
        )
        if message is not None:
            raise ValueError(message)

    # Total-timeline geometry: sum of pixel frames minus the shared overlaps.
    # Delegated to the shared pure-Python chain_math so the validator, the
    # engine and the metadata agree; it also raises on a degenerate audio
    # overlap (clips too short for a continuous crossfade). The stage-2 window
    # preset is resolved to the (v_tile, v_adv) the engine will actually tile
    # with and passed EXPLICITLY, so this geometry matches the engine's even
    # when the request opted into a non-default window.
    stage2_v_tile, stage2_v_adv = chain_math.resolve_stage2_window(request.stage2_window)
    layout = chain_math.compute_chain_layout(
        [c.num_frames for c in request.clips], request.frame_rate,
        kv=request.overlap_frames,
        v_tile=stage2_v_tile,
        v_adv=stage2_v_adv,
        source_context_px=(
            request.source_video.context_frames if request.source_video else None
        ),
        # Window length + glue-band geometry (8n+1 window in
        # [chain_math.RETAKE_WINDOW_MIN_PX, retake_max_window_px(v_tile)],
        # head/tail grids, a free middle in BOTH latent domains) is validated
        # THERE, so the validator, the engine and the mock cannot disagree.
        retake_glue_px=(
            None if request.retake is None
            else (request.retake.head_px, request.retake.tail_px)
        ),
        # The frozen tail band's own geometry (multiple of 8, >= 8, and free
        # stage-1 latents left in the FINAL clip) is validated THERE for the
        # same reason. NO MODE IS PASSED: the clip count plus
        # ``source_context_px`` are the whole input to that decision.
        end_context_px=(
            None if request.end_source is None
            else request.end_source.context_frames
        ),
    )
    # The cap is on the USER'S OWN CLIPS, not on the delivered timeline
    # (``ChainLayout.clips_total_px``, the same property ``to_dict`` publishes).
    # The cap is read at call time (``models.MAX_CHAIN_TOTAL_PIXEL_FRAMES``, the
    # one definition) rather than bound at import.
    cap = models.MAX_CHAIN_TOTAL_PIXEL_FRAMES
    clips_total_px = layout.clips_total_px
    if clips_total_px > cap:
        raise ValueError(
            f"chain total timeline {clips_total_px} pixel frames exceeds the "
            f"cap {cap} (reduce clip count or lengths)"
        )
