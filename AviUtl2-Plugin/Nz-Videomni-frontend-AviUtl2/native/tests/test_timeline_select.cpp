// test_timeline_select.cpp - unit tests for the pure cut-out planner.
// Intervals are half-open [start, end); desired_project_frames is a hard cap.
#include "doctest.h"

#include <string>

#include "timeline_select.h"

using namespace nzvideomni;

namespace {

// Convenience: object span [100, 200) on layer 3.
constexpr int kLayer = 3;
constexpr int kObjStart = 100;
constexpr int kObjEnd = 200;  // span length 100

}  // namespace

// ---------------------------------------------------------------------------
// No explicit range: anchor + cap behaviour
// ---------------------------------------------------------------------------

TEST_CASE("no range, tail anchor takes the last N frames of the span") {
    // desired 40 < span 100 -> last 40 frames: [160, 200).
    const CutoutPlan p =
        PlanCutout(false, 0, 0, kLayer, kObjStart, kObjEnd, CutoutAnchor::kTail, 40);
    CHECK(p.error == CutoutError::kNone);
    CHECK(p.layer == kLayer);
    CHECK(p.frame_start == 160);
    CHECK(p.frame_count == 40);
    CHECK(p.frame_start + p.frame_count == kObjEnd);
}

TEST_CASE("no range, head anchor takes the first N frames of the span") {
    const CutoutPlan p =
        PlanCutout(false, 0, 0, kLayer, kObjStart, kObjEnd, CutoutAnchor::kHead, 40);
    CHECK(p.error == CutoutError::kNone);
    CHECK(p.frame_start == kObjStart);
    CHECK(p.frame_count == 40);
}

TEST_CASE("no range, desired larger than the span clamps to the whole span") {
    // desired 500 > span 100 -> whole span, regardless of anchor.
    const CutoutPlan tail =
        PlanCutout(false, 0, 0, kLayer, kObjStart, kObjEnd, CutoutAnchor::kTail, 500);
    CHECK(tail.error == CutoutError::kNone);
    CHECK(tail.frame_start == kObjStart);
    CHECK(tail.frame_count == 100);

    const CutoutPlan head =
        PlanCutout(false, 0, 0, kLayer, kObjStart, kObjEnd, CutoutAnchor::kHead, 500);
    CHECK(head.error == CutoutError::kNone);
    CHECK(head.frame_start == kObjStart);
    CHECK(head.frame_count == 100);
}

TEST_CASE("no range, desired exactly the span length") {
    const CutoutPlan p =
        PlanCutout(false, 0, 0, kLayer, kObjStart, kObjEnd, CutoutAnchor::kTail, 100);
    CHECK(p.error == CutoutError::kNone);
    CHECK(p.frame_start == kObjStart);
    CHECK(p.frame_count == 100);
}

// ---------------------------------------------------------------------------
// Explicit range: honoured, clamped to span, capped at desired
// ---------------------------------------------------------------------------

TEST_CASE("explicit range within span and under the cap is used verbatim") {
    // range [120, 160) length 40, cap 80 -> used as-is.
    const CutoutPlan p = PlanCutout(true, 120, 160, kLayer, kObjStart, kObjEnd,
                                    CutoutAnchor::kTail, 80);
    CHECK(p.error == CutoutError::kNone);
    CHECK(p.layer == kLayer);
    CHECK(p.frame_start == 120);
    CHECK(p.frame_count == 40);
}

TEST_CASE("explicit range is clamped to the object span") {
    // range [80, 250) clamps to span [100, 200) length 100; cap 200 (no trim).
    const CutoutPlan p = PlanCutout(true, 80, 250, kLayer, kObjStart, kObjEnd,
                                    CutoutAnchor::kHead, 200);
    CHECK(p.error == CutoutError::kNone);
    CHECK(p.frame_start == kObjStart);
    CHECK(p.frame_count == 100);
}

TEST_CASE("over-long explicit range is trimmed to the cap from the tail anchor") {
    // clamped range [100, 200) length 100, cap 30, tail -> keep tail [170, 200).
    const CutoutPlan p = PlanCutout(true, 100, 200, kLayer, kObjStart, kObjEnd,
                                    CutoutAnchor::kTail, 30);
    CHECK(p.error == CutoutError::kNone);
    CHECK(p.frame_start == 170);
    CHECK(p.frame_count == 30);
    CHECK(p.frame_start + p.frame_count == 200);
}

TEST_CASE("over-long explicit range is trimmed to the cap from the head anchor") {
    // clamped range [100, 200) length 100, cap 30, head -> keep head [100, 130).
    const CutoutPlan p = PlanCutout(true, 100, 200, kLayer, kObjStart, kObjEnd,
                                    CutoutAnchor::kHead, 30);
    CHECK(p.error == CutoutError::kNone);
    CHECK(p.frame_start == kObjStart);
    CHECK(p.frame_count == 30);
}

// ---------------------------------------------------------------------------
// Error paths
// ---------------------------------------------------------------------------

TEST_CASE("explicit range entirely outside the span is rejected") {
    const CutoutPlan before = PlanCutout(true, 0, 90, kLayer, kObjStart, kObjEnd,
                                         CutoutAnchor::kTail, 40);
    CHECK(before.error == CutoutError::kRangeOutsideObject);
    CHECK(before.frame_count == 0);

    const CutoutPlan after = PlanCutout(true, 210, 260, kLayer, kObjStart, kObjEnd,
                                        CutoutAnchor::kHead, 40);
    CHECK(after.error == CutoutError::kRangeOutsideObject);
}

TEST_CASE("range touching the span edge only (half-open) is outside") {
    // [200, 260) shares just the exclusive edge with [100,200) -> no overlap.
    const CutoutPlan p = PlanCutout(true, 200, 260, kLayer, kObjStart, kObjEnd,
                                    CutoutAnchor::kTail, 40);
    CHECK(p.error == CutoutError::kRangeOutsideObject);
}

TEST_CASE("zero/negative-length explicit range is empty") {
    CHECK(PlanCutout(true, 150, 150, kLayer, kObjStart, kObjEnd,
                     CutoutAnchor::kTail, 40)
              .error == CutoutError::kEmptyCutout);
    CHECK(PlanCutout(true, 160, 150, kLayer, kObjStart, kObjEnd,
                     CutoutAnchor::kTail, 40)
              .error == CutoutError::kEmptyCutout);
}

TEST_CASE("invalid object span is rejected") {
    CHECK(PlanCutout(false, 0, 0, kLayer, 100, 100, CutoutAnchor::kTail, 40).error ==
          CutoutError::kInvalidObjectSpan);  // zero-length span
    CHECK(PlanCutout(false, 0, 0, kLayer, 200, 100, CutoutAnchor::kTail, 40).error ==
          CutoutError::kInvalidObjectSpan);  // inverted span
    CHECK(PlanCutout(false, 0, 0, -1, kObjStart, kObjEnd, CutoutAnchor::kTail, 40)
              .error == CutoutError::kInvalidObjectSpan);  // negative layer
    CHECK(PlanCutout(false, 0, 0, kLayer, -5, kObjEnd, CutoutAnchor::kTail, 40)
              .error == CutoutError::kInvalidObjectSpan);  // negative start
}

TEST_CASE("non-positive desired cap is rejected (both branches)") {
    CHECK(PlanCutout(false, 0, 0, kLayer, kObjStart, kObjEnd, CutoutAnchor::kTail, 0)
              .error == CutoutError::kInvalidDesired);
    CHECK(PlanCutout(true, 120, 160, kLayer, kObjStart, kObjEnd, CutoutAnchor::kHead,
                     -3)
              .error == CutoutError::kInvalidDesired);
}

TEST_CASE("error messages are stable and non-empty for errors only") {
    CHECK(std::string(CutoutErrorMessage(CutoutError::kNone)).empty());
    CHECK_FALSE(std::string(CutoutErrorMessage(CutoutError::kInvalidObjectSpan)).empty());
    CHECK_FALSE(std::string(CutoutErrorMessage(CutoutError::kInvalidDesired)).empty());
    CHECK_FALSE(std::string(CutoutErrorMessage(CutoutError::kRangeOutsideObject)).empty());
    CHECK_FALSE(std::string(CutoutErrorMessage(CutoutError::kEmptyCutout)).empty());
}
