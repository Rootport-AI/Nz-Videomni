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


def check_chain_total(clip_frames, fps, overlap_frames, lang: str = _DEFAULT_LANG,
                      source_context_px=None):
    """``source_context_px`` (ADDITIVE, V2V): forwarded to
    ``compute_chain_layout`` so a V2V chain's precheck runs the same frozen-head
    geometry the server does (incl. the stage-2 tile-fit invariant raise, which
    surfaces here as the localized geometry message). ``None`` keeps the
    pre-V2V arithmetic byte-identical."""
    import chain_math
    try:
        kwargs = {}
        if source_context_px is not None:
            kwargs["source_context_px"] = int(source_context_px)
        layout = chain_math.compute_chain_layout(
            [int(f) for f in clip_frames], float(fps), kv=int(overlap_frames), **kwargs,
        )
    except ValueError as exc:
        return L("msg_chain_geometry", lang).format(err=exc)
    if layout.total_px > MAX_CHAIN_TOTAL_PIXEL_FRAMES:
        return L("msg_chain_total_frames", lang).format(
            total=layout.total_px, cap=MAX_CHAIN_TOTAL_PIXEL_FRAMES,
        )
    return None


# V2V context_frames bounds — MIRRORS api/models.py SourceVideoSpec
# (validate_context_frames: 8n+1, [min, max]) + the request-level cross-check
# (context_frames < clips[0].num_frames). The fallback bounds match
# config.LimitsConfig defaults; the live values come from /config limits.
V2V_CONTEXT_MIN_FALLBACK = 25
V2V_CONTEXT_MAX_FALLBACK = 145


def check_v2v_context(context_frames, clip0_frames, lang: str = _DEFAULT_LANG,
                      config: dict | None = None):
    """Localized precheck for ``source_video.context_frames`` (zero API calls on
    violation). Mirrors the server's 422s: 8n+1, within [min, max] (from /config
    ``limits`` when available), and strictly smaller than clip 1's frames so a
    NEW tail remains to generate. Returns an error string or None."""
    limits = (config or {}).get("limits") or {}
    cf_min = int(limits.get("v2v_context_frames_min", V2V_CONTEXT_MIN_FALLBACK))
    cf_max = int(limits.get("v2v_context_frames_max", V2V_CONTEXT_MAX_FALLBACK))
    try:
        cf = int(context_frames)
    except (TypeError, ValueError):
        return L("v2v_msg_bad_context", lang).format(mincf=cf_min, maxcf=cf_max)
    if cf < cf_min or cf > cf_max or (cf - 1) % 8 != 0:
        return L("v2v_msg_bad_context", lang).format(mincf=cf_min, maxcf=cf_max)
    try:
        clip0 = int(clip0_frames)
    except (TypeError, ValueError):
        clip0 = 0
    if cf >= clip0:
        return L("v2v_msg_context_ge_clip", lang).format(cf=cf, clip=clip0)
    return None
