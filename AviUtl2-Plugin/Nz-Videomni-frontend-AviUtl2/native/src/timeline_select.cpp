// timeline_select.cpp - implementation. See timeline_select.h for the contract
// (half-open intervals, anchor semantics, desired_project_frames as a hard cap).
// ASCII-only source.
#include "timeline_select.h"

namespace nzvideomni {

const char* CutoutErrorMessage(CutoutError e) {
    switch (e) {
        case CutoutError::kNone:
            return "";
        case CutoutError::kInvalidObjectSpan:
            return "object span invalid (need layer>=0, start>=0, end>start)";
        case CutoutError::kInvalidDesired:
            return "desired_project_frames must be >= 1";
        case CutoutError::kRangeOutsideObject:
            return "selection range does not overlap the object";
        case CutoutError::kEmptyCutout:
            return "selection range is empty (end <= start)";
    }
    return "unknown error";
}

CutoutPlan PlanCutout(bool has_range, int range_start, int range_end,
                      int obj_layer, int obj_start, int obj_end,
                      CutoutAnchor anchor, int desired_project_frames) {
    CutoutPlan p;

    // --- validity of the object span and the upper bound --------------------
    if (obj_layer < 0 || obj_start < 0 || obj_end <= obj_start) {
        p.error = CutoutError::kInvalidObjectSpan;
        return p;
    }
    if (desired_project_frames <= 0) {
        // The cap is required: without a positive upper bound there is no
        // well-defined amount to cut (and an explicit range could not be capped).
        p.error = CutoutError::kInvalidDesired;
        return p;
    }

    if (has_range) {
        // Degenerate / inverted range: nothing to cut.
        if (range_end <= range_start) {
            p.error = CutoutError::kEmptyCutout;
            return p;
        }
        // Clamp the range to the object span (half-open intersection).
        const int s = range_start > obj_start ? range_start : obj_start;
        const int e = range_end < obj_end ? range_end : obj_end;
        if (e <= s) {
            // The range lies entirely before/after the object span.
            p.error = CutoutError::kRangeOutsideObject;
            return p;
        }
        int start = s;
        int count = e - s;
        // Respect the user's range, but never exceed the hard upper bound: only
        // when the (clamped) range is longer than the cap do we trim it back to
        // the cap from the anchor side.
        if (count > desired_project_frames) {
            if (anchor == CutoutAnchor::kTail) {
                start = e - desired_project_frames;  // keep the tail
            }
            // kHead keeps the head: start stays at s.
            count = desired_project_frames;
        }
        p.layer = obj_layer;
        p.frame_start = start;
        p.frame_count = count;
        return p;
    }

    // --- no explicit range: cut from the anchor end -------------------------
    const int span_len = obj_end - obj_start;
    const int count = span_len < desired_project_frames ? span_len
                                                        : desired_project_frames;
    p.layer = obj_layer;
    p.frame_count = count;
    p.frame_start =
        (anchor == CutoutAnchor::kTail) ? (obj_end - count) : obj_start;
    return p;
}

}  // namespace nzvideomni
