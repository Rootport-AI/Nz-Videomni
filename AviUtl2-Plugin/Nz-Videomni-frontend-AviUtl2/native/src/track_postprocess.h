// track_postprocess.h - pure post-processing of an object-tracking run.
//
// The tracking backend returns nothing but a raw box and a confidence score per
// frame (Docs\OBJECT_TRACKING_DESIGN.md section 4.1); EVERY judgement about what
// to do with those numbers lives here, in a function that touches neither the
// AviUtl2 SDK nor the network. That is what lets the whole of the "lost /
// smoothing / size-follow / keyframe thinning" contract be unit-tested with
// doctest (native/tests/test_track_postprocess.cpp).
//
// The six stages run in a FIXED order (design section 5.5):
//   1. mark a sample "lost" when score < lost_score_threshold
//   2. hold_on_lost -> refill the lost samples' boxes from the last good one
//   3. exponential moving average over x/y/w/h with alpha = 1 - smoothing
//   4. follow_size == false -> pin w/h to the SEED (the first raw sample)
//   5. keep every keyframe_stride-th sample, ALWAYS including first and last
//   6. collapse runs of lost samples into inclusive ranges
//
// Frame numbers in and out are 0-based offsets RELATIVE to the object's start
// frame, because that is the coordinate system the alias' own "frame=" header
// uses (Docs\SDK_REFERENCE.md section 16 (h)).
//
// ASCII-only source; comments in English, matching the rest of native/src.
#pragma once

#include <vector>

namespace nzvideomni {

// One frame's raw result as reported by the backend. `frame` is the 0-based
// offset inside the object; x/y/w/h are pixels with the ORIGIN AT THE TOP-LEFT
// of the rendered frame (the tracker's own convention, not AviUtl2's centre
// origin - see RectToPartialFilter in alias_util.h for the conversion).
struct TrackSample {
    int frame = 0;
    double x = 0.0;
    double y = 0.0;
    double w = 0.0;
    double h = 0.0;
    double score = 0.0;
};

// The five user-facing knobs that affect post-processing (design section 3.1;
// the sixth knob, the search factor, is consumed by the backend and the seventh
// is the model radio, so neither reaches this function).
struct TrackPostOptions {
    // A sample whose score is STRICTLY below this is "lost". 0 disables the
    // judgement entirely (a score is never negative).
    double lost_score_threshold = 0.0;
    // true  = "stop at the last position": lost samples reuse the last good box.
    // false = "keep following": lost samples pass through with the raw box.
    bool hold_on_lost = true;
    // 0 = no smoothing (pass-through), 1 = fully frozen at the first value.
    // Values outside [0, 1] are clamped into it.
    double smoothing = 0.0;
    // false pins w/h to the seed's size for every keyframe.
    bool follow_size = true;
    // Keep one sample out of every `keyframe_stride`. Anything below 1 is read
    // as 1 (keep them all).
    int keyframe_stride = 1;
};

// A run of consecutive lost frames, INCLUSIVE at both ends, in the same 0-based
// relative numbering as TrackSample::frame. Shown to the user as a list in the
// Toolbox panel; it never changes the written alias.
struct TrackRange {
    int start = 0;
    int end = 0;
};

// One keyframe to write into the partial filter. `frame_offset` is 0-based and
// relative to the object's start frame.
struct TrackKeyframe {
    int frame_offset = 0;
    double x = 0.0;
    double y = 0.0;
    double w = 0.0;
    double h = 0.0;
};

struct TrackPostResult {
    std::vector<TrackKeyframe> keyframes;
    std::vector<TrackRange> lost_ranges;
};

// Run the six stages above. An empty input yields an empty result (no keyframes,
// no ranges) - the caller treats that as "nothing to write back".
//
// Notes that the stage list alone does not pin down:
//   * Stage 2 needs a box to hold, and only ever looks BACKWARDS: a lost sample
//     with no good sample before it - a leading lost run, or an all-lost run -
//     keeps its raw box. A leading lost run cannot actually occur, because the
//     first sample is the seed box the caller handed the tracker and the backend
//     answers it with a score of 1.0; the rule is stated this way so there is no
//     second case to reason about.
//   * Stage 4's "seed" is the first RAW sample's w/h, captured before any other
//     stage runs - it is the box the user aimed in AviUtl2's preview, which is
//     also what was sent to the tracker as the initialisation box.
//   * Stage 5 thins by POSITION IN THE SAMPLE LIST, not by frame number, so a
//     gap in the input cannot shift the phase of the kept keyframes. The last
//     sample is kept even when the stride would have skipped it, and a
//     single-sample input produces exactly one keyframe (first == last).
//   * Stage 6 scans ALL samples, not just the kept ones: the user is told every
//     frame the tracker was unsure about, even between keyframes.
TrackPostResult PostProcessTrack(const std::vector<TrackSample>& samples,
                                 const TrackPostOptions& opt);

}  // namespace nzvideomni
