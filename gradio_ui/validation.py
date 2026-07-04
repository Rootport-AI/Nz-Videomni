"""Clip-chain total-timeline precheck (S5). Mirrors the SAME arithmetic the API
validator uses (api/models.py GenerateChainRequest.validate_chain_constraints):
it delegates to the shared pure-Python ``chain_math.compute_chain_layout`` and
compares against MAX_CHAIN_TOTAL_PIXEL_FRAMES = 8 * 481 (= 3848), so the GUI and
the server agree byte-for-byte. Returns a localized error string on violation,
else None. Note: because a single clip is capped at 481 frames and a chain at 8
clips, the 3848 cap is unreachable through the 8-slot UI (max 8×481 = 3841 total
pixel frames after overlap); the check is defence-in-depth that mirrors the
server and also surfaces chain_math's degenerate-geometry ValueError (e.g. clips
too short for a continuous audio cross-fade) before any API call.
"""

from __future__ import annotations

from .i18n import L, _DEFAULT_LANG

MAX_CHAIN_TOTAL_PIXEL_FRAMES = 8 * 481  # 3848; mirrors api/models.py


def check_chain_total(clip_frames, fps, overlap_frames, lang: str = _DEFAULT_LANG):
    import chain_math
    try:
        layout = chain_math.compute_chain_layout(
            [int(f) for f in clip_frames], float(fps), kv=int(overlap_frames),
        )
    except ValueError as exc:
        return L("msg_chain_geometry", lang).format(err=exc)
    if layout.total_px > MAX_CHAIN_TOTAL_PIXEL_FRAMES:
        return L("msg_chain_total_frames", lang).format(
            total=layout.total_px, cap=MAX_CHAIN_TOTAL_PIXEL_FRAMES,
        )
    return None
