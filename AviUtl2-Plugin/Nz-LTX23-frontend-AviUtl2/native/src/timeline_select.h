// timeline_select.h - pure timeline cut-out planning for Nz-LTX23.
//
// Given a user selection (an explicit range and/or a selected object span) this
// module decides "which layer, which start frame, and how many frames" to cut
// out of the timeline for a generation request, WITHOUT touching the AviUtl2
// SDK. It is the selection-side companion to timeline_math (which answers how
// long the RESULT will be); here we answer how much of the SOURCE timeline to
// feed in.
//
// Two request families need opposite anchoring when no explicit range is given:
//   - vid2vid continuation cuts the TAIL of the selected object (the last N
//     project frames become the frozen source context), so anchor == kTail.
//   - Aud2Vid / IC-LoRA cut the HEAD (the first N project frames), so
//     anchor == kHead.
//
// desired_project_frames is an UPPER BOUND: the maximum number of project-fps
// frames the request may consume (e.g. IC-LoRA's / Aud2Vid's input-length limit
// already converted from generation frames to project frames via
// timeline_math::ProjectFramesForPixels). It is a safety cap that must never be
// exceeded, even when the user drags an explicit range longer than it.
//
// No AviUtl2 SDK / WebView2 / Win32 dependency: unit-tested with doctest in
// isolation (native/tests/test_timeline_select.cpp). Invalid input is reported
// through CutoutPlan::error (never thrown). All comments are ASCII/English.
//
// NOTE on the error type: the design brief suggested reusing timeline_math's
// TimelineError, extending it if needed. Extending that enum would mean editing
// timeline_math.h, which is an existing, frozen file for this change, so this
// module defines its own small CutoutError enum instead (same bridge_core-style
// result-struct discipline: kNone == success, numeric fields 0 on error).
#pragma once

namespace nzltx {

// Which end of the object span to keep when there is no explicit range.
enum class CutoutAnchor {
    kTail,  // vid2vid: keep the LAST desired_project_frames of the span
    kHead,  // Aud2Vid / IC-LoRA: keep the FIRST desired_project_frames
};

// Error codes for CutoutPlan (kNone == success).
enum class CutoutError {
    kNone = 0,
    kInvalidObjectSpan,   // obj_layer/obj_start negative, or obj_end <= obj_start
    kInvalidDesired,      // desired_project_frames <= 0 (no valid upper bound)
    kRangeOutsideObject,  // explicit range does not intersect the object span
    kEmptyCutout,         // explicit range has zero/negative length (range_end <= range_start)
};

// Stable ASCII message for an error code (empty string for kNone).
const char* CutoutErrorMessage(CutoutError e);

// The planned cut-out. On error the numeric fields are 0 and error != kNone.
struct CutoutPlan {
    int layer = 0;
    int frame_start = 0;   // first project frame of the cut-out
    int frame_count = 0;   // number of project frames (> 0 on success)
    CutoutError error = CutoutError::kNone;
};

// Plan a cut-out from a selection.
//
// Intervals are HALF-OPEN, measured in project frames: the object occupies
// [obj_start, obj_end) and an explicit range covers [range_start, range_end), so
// frame_count == end - start throughout. (Documenting this explicitly avoids the
// off-by-one ambiguity of "[a, b]" notation; the caller must pass an exclusive
// end for both the span and the range.)
//
// Behaviour:
//   * has_range == true  -> the user's explicit range is honoured, but clamped
//     to the object span and then, ONLY IF it is longer than
//     desired_project_frames, trimmed back to that cap from the anchor side
//     (kTail keeps the range's tail, kHead keeps its head). Rationale: the range
//     is the user's intent and is respected as-is whenever it fits, yet the
//     desired upper bound is a hard safety limit (IC-LoRA / Aud2Vid input cap)
//     that must always hold, so it wins over an over-long range.
//       - range_end <= range_start          -> kEmptyCutout
//       - range does not intersect the span  -> kRangeOutsideObject
//   * has_range == false -> take min(desired_project_frames, span length) frames
//     from the anchor end (kTail: from obj_end backwards; kHead: from obj_start).
//
// Validity: obj_layer >= 0, obj_start >= 0, obj_end > obj_start
// (else kInvalidObjectSpan); desired_project_frames > 0 (else kInvalidDesired).
// On success frame_count > 0, frame_start >= obj_start, and
// frame_start + frame_count <= obj_end.
CutoutPlan PlanCutout(bool has_range, int range_start, int range_end,
                      int obj_layer, int obj_start, int obj_end,
                      CutoutAnchor anchor, int desired_project_frames);

}  // namespace nzltx
