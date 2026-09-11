// track_postprocess.cpp - implementation. See track_postprocess.h for the fixed
// six-stage contract and the notes that the stage list alone does not pin down.
// ASCII-only source, no SDK and no I/O.
#include "track_postprocess.h"

#include <cstddef>

namespace nzvideomni {

namespace {

// The working copy a stage reads and the next stage overwrites.
struct WorkBox {
    double x = 0.0;
    double y = 0.0;
    double w = 0.0;
    double h = 0.0;
};

double Clamp01(double v) {
    if (!(v >= 0.0)) {  // also catches NaN, which fails every comparison
        return 0.0;
    }
    return v > 1.0 ? 1.0 : v;
}

}  // namespace

TrackPostResult PostProcessTrack(const std::vector<TrackSample>& samples,
                                 const TrackPostOptions& opt) {
    TrackPostResult result;
    if (samples.empty()) {
        return result;
    }
    const size_t n = samples.size();

    // The seed size (stage 4) is read from the RAW first sample, before stage 2
    // can overwrite it with a later box.
    const double seed_w = samples[0].w;
    const double seed_h = samples[0].h;

    // --- Stage 1: lost judgement -------------------------------------------
    std::vector<bool> lost(n, false);
    for (size_t i = 0; i < n; ++i) {
        lost[i] = samples[i].score < opt.lost_score_threshold;
    }

    std::vector<WorkBox> box(n);
    for (size_t i = 0; i < n; ++i) {
        box[i].x = samples[i].x;
        box[i].y = samples[i].y;
        box[i].w = samples[i].w;
        box[i].h = samples[i].h;
    }

    // --- Stage 2: hold the last good box over the lost stretches ------------
    // "Keep following" (hold_on_lost == false) leaves the raw boxes alone, which
    // is the whole point of that setting: the user wants to see where the
    // tracker drifted to, not a frozen box.
    // One rule, no special cases: a lost sample takes the last good box SEEN SO
    // FAR, and a lost sample with no good box before it keeps its raw one. The
    // leading run therefore stays raw - which in practice never happens, since
    // the first sample is the seed the caller handed the tracker and the backend
    // answers it with score 1.0.
    if (opt.hold_on_lost) {
        bool have_held = false;
        WorkBox held;
        for (size_t i = 0; i < n; ++i) {
            if (!lost[i]) {
                held = box[i];
                have_held = true;
            } else if (have_held) {
                box[i] = held;
            }
        }
    }

    // --- Stage 3: exponential moving average --------------------------------
    // alpha == 1 (smoothing 0) reproduces the input exactly, so the "no
    // smoothing" case needs no special path; it is spelled out anyway because
    // skipping the loop is both cheaper and immune to rounding.
    const double smoothing = Clamp01(opt.smoothing);
    if (smoothing > 0.0) {
        const double alpha = 1.0 - smoothing;
        for (size_t i = 1; i < n; ++i) {
            box[i].x = alpha * box[i].x + smoothing * box[i - 1].x;
            box[i].y = alpha * box[i].y + smoothing * box[i - 1].y;
            box[i].w = alpha * box[i].w + smoothing * box[i - 1].w;
            box[i].h = alpha * box[i].h + smoothing * box[i - 1].h;
        }
    }

    // --- Stage 4: pin the size to the seed ----------------------------------
    if (!opt.follow_size) {
        for (size_t i = 0; i < n; ++i) {
            box[i].w = seed_w;
            box[i].h = seed_h;
        }
    }

    // --- Stage 5: thin to every stride-th sample, keeping both ends ---------
    const int stride_raw = opt.keyframe_stride;
    const size_t stride = stride_raw < 1 ? 1u : static_cast<size_t>(stride_raw);
    const size_t last = n - 1;
    for (size_t i = 0; i < n; i += stride) {
        TrackKeyframe kf;
        kf.frame_offset = samples[i].frame;
        kf.x = box[i].x;
        kf.y = box[i].y;
        kf.w = box[i].w;
        kf.h = box[i].h;
        result.keyframes.push_back(kf);
    }
    // The stride lands on the last sample only when (n - 1) % stride == 0; force
    // it in otherwise, so the written keyframes always span the whole object.
    if (last % stride != 0) {
        TrackKeyframe kf;
        kf.frame_offset = samples[last].frame;
        kf.x = box[last].x;
        kf.y = box[last].y;
        kf.w = box[last].w;
        kf.h = box[last].h;
        result.keyframes.push_back(kf);
    }

    // --- Stage 6: collapse runs of lost samples -----------------------------
    size_t i = 0;
    while (i < n) {
        if (!lost[i]) {
            ++i;
            continue;
        }
        size_t j = i;
        while (j + 1 < n && lost[j + 1]) {
            ++j;
        }
        TrackRange range;
        range.start = samples[i].frame;
        range.end = samples[j].frame;
        result.lost_ranges.push_back(range);
        i = j + 1;
    }

    return result;
}

}  // namespace nzvideomni
