// timeline_math.cpp - implementation. See timeline_math.h for the contract and
// the backend cross-reference (chain_math.py / api/models.py). ASCII-only source.
#include "timeline_math.h"

#include <cmath>

namespace nzvideomni {

namespace {

// Python round() semantics (round half to even). Inputs here are non-negative.
long long PyRoundHalfEven(double x) {
    const double f = std::floor(x);
    const double diff = x - f;
    long long lo = static_cast<long long>(f);
    if (diff < 0.5) {
        return lo;
    }
    if (diff > 0.5) {
        return lo + 1;
    }
    return (lo % 2 == 0) ? lo : lo + 1;  // exact .5 -> nearest even
}

}  // namespace

const char* TimelineErrorMessage(TimelineError e) {
    switch (e) {
        case TimelineError::kNone:
            return "";
        case TimelineError::kNumFramesRange:
            return "num_frames must be within [9, 481]";
        case TimelineError::kNumFramesNotAligned:
            return "num_frames must be 8n+1 (e.g. 9, 17, ..., 481)";
        case TimelineError::kFrameRateRange:
            return "frame_rate must be within [1, 60]";
        case TimelineError::kNoClips:
            return "chain requires at least one clip";
        case TimelineError::kTooManyClips:
            return "chain supports at most 8 clips";
        case TimelineError::kOverlapRange:
            return "overlap_frames must be within [1, 8]";
        case TimelineError::kOverlapTooLarge:
            return "overlap_frames must be < every clip's stage-1 latent frames";
        case TimelineError::kAudioDegenerate:
            return "clips too short for a continuous audio crossfade";
        case TimelineError::kChainTooLong:
            return "chain total timeline exceeds 3848 pixel frames";
        case TimelineError::kSourceContextRange:
            return "source_context_px must be >= 1";
        case TimelineError::kSourceContextNotAligned:
            return "source_context_px must be 8n+1";
        case TimelineError::kSourceContextTooLong:
            return "source_context_px frozen head exceeds stage-2 tile 0 (169 px max)";
        case TimelineError::kSourceContextNotLessThanClip0:
            return "source_context_px must be < clip 0's frame count";
    }
    return "unknown error";
}

bool IsValidNumFrames(int num_frames) {
    if (num_frames < kMinNumFrames || num_frames > kMaxNumFrames) {
        return false;
    }
    return (num_frames - 1) % kVideoTimeFactor == 0;
}

int NearestValidNumFrames(int num_frames) {
    // n index on the 8n+1 grid; ties round up via + 0.5 floor.
    const double n = std::floor((num_frames - 1) / 8.0 + 0.5);
    int idx = static_cast<int>(n);
    if (idx < 1) {
        idx = 1;  // -> kMinNumFrames (9)
    } else if (idx > 60) {
        idx = 60;  // -> kMaxNumFrames (481)
    }
    return idx * kVideoTimeFactor + 1;
}

int VLatentFrames(int pixel_frames) {
    return (pixel_frames - 1) / kVideoTimeFactor + 1;
}

int PxFromVLatent(int n_latent) {
    return (n_latent - 1) * kVideoTimeFactor + 1;
}

int AudioLatentFrames(int pixel_frames, double fps) {
    if (fps <= 0.0) {
        return 0;
    }
    return static_cast<int>(
        PyRoundHalfEven(pixel_frames / fps * kAudioLatentsPerSec));
}

int ProjectFramesForPixels(int pixel_frames, double gen_fps, double project_fps) {
    if (gen_fps <= 0.0) {
        return 0;
    }
    const double v = static_cast<double>(pixel_frames) * project_fps / gen_fps;
    return static_cast<int>(std::floor(v + 0.5));  // round half up
}

int ProjectFramesForSeconds(double seconds, double project_fps) {
    // Blanket-reject anything that isn't a strictly-positive input. The
    // !(x > 0.0) form also catches NaN (every comparison with NaN is false),
    // so NaN seconds / NaN fps fall through to 0. It does NOT reject +inf
    // (inf > 0.0 is true); that case is handled below by the clamp.
    if (!(project_fps > 0.0) || !(seconds > 0.0)) {
        return 0;
    }
    const double v = std::floor(seconds * project_fps + 0.5);  // round half up
    if (v < 1.0) {
        return 0;  // sub-half durations (e.g. 0.02s @ 24fps -> 0.48) round to 0
    }
    // Clamp into the int-safe domain before the cast: an absurd total_time
    // (e.g. 1e18) or +inf makes `v` exceed INT_MAX, and static_cast<int> of an
    // out-of-range double is undefined behavior. floor(inf) == inf and
    // inf > kMaxFrames is true, so +inf clamps here rather than reaching the
    // UB cast below.
    constexpr double kMaxFrames = 2000000000.0;  // int-safe upper bound
    if (v > kMaxFrames) {
        return static_cast<int>(kMaxFrames);
    }
    return static_cast<int>(v);
}

bool SpansOverlap(int a_start, int a_end, int b_start, int b_end) {
    if (a_start > a_end || b_start > b_end) {
        return false;  // a degenerate (empty) span never overlaps anything
    }
    // Two inclusive spans overlap iff neither ends before the other starts.
    return a_start <= b_end && b_start <= a_end;
}

SingleClipResult ComputeSingleClip(int num_frames, double frame_rate) {
    SingleClipResult r;
    if (num_frames < kMinNumFrames || num_frames > kMaxNumFrames) {
        r.error = TimelineError::kNumFramesRange;
        return r;
    }
    if ((num_frames - 1) % kVideoTimeFactor != 0) {
        r.error = TimelineError::kNumFramesNotAligned;
        return r;
    }
    if (frame_rate < kMinFrameRate || frame_rate > kMaxFrameRate) {
        r.error = TimelineError::kFrameRateRange;
        return r;
    }
    r.pixel_frames = num_frames;
    r.seconds = num_frames / frame_rate;
    return r;
}

ChainResult ComputeChain(const ChainInput& in) {
    ChainResult r;
    const int n = static_cast<int>(in.clip_frames.size());
    if (n < 1) {
        r.error = TimelineError::kNoClips;
        return r;
    }
    if (n > kMaxChainClips) {
        r.error = TimelineError::kTooManyClips;
        return r;
    }
    if (in.frame_rate < kMinFrameRate || in.frame_rate > kMaxFrameRate) {
        r.error = TimelineError::kFrameRateRange;
        return r;
    }
    if (in.overlap_frames < kMinOverlapFrames ||
        in.overlap_frames > kMaxOverlapFrames) {
        r.error = TimelineError::kOverlapRange;
        return r;
    }
    // Each clip must be a valid single-clip count (range + 8n+1). chain_math
    // itself assumes this (models.py validates it upstream); we re-check so the
    // pre-computation never runs on a malformed clip list.
    for (int f : in.clip_frames) {
        if (f < kMinNumFrames || f > kMaxNumFrames) {
            r.error = TimelineError::kNumFramesRange;
            return r;
        }
        if ((f - 1) % kVideoTimeFactor != 0) {
            r.error = TimelineError::kNumFramesNotAligned;
            return r;
        }
    }

    // vid2vid frozen-context invariants (chain_math compute_chain_layout, order
    // preserved: context checks precede the seg-latent/overlap checks).
    if (in.has_source_context) {
        const int px = in.source_context_px;
        if (px < 1) {
            r.error = TimelineError::kSourceContextRange;
            return r;
        }
        if ((px - 1) % kVideoTimeFactor != 0) {
            r.error = TimelineError::kSourceContextNotAligned;
            return r;
        }
        if (px >= in.clip_frames[0]) {
            r.error = TimelineError::kSourceContextNotLessThanClip0;
            return r;
        }
        if (VLatentFrames(px) > kStage2VTile) {
            r.error = TimelineError::kSourceContextTooLong;
            return r;
        }
    }

    // seg_latent = v_latent_frames per clip; K_v must fit inside every clip.
    long long sum_latent = 0;
    for (int f : in.clip_frames) {
        const int seg = VLatentFrames(f);
        if (in.overlap_frames >= seg) {
            r.error = TimelineError::kOverlapTooLarge;
            return r;
        }
        sum_latent += seg;
    }

    const long long f_total = sum_latent - static_cast<long long>(n - 1) * in.overlap_frames;
    const int total_px = PxFromVLatent(static_cast<int>(f_total));

    // Audio crossfade degeneracy check (chain_math): sum_ka = sum(seg_audio) -
    // a_total must be >= number of joins so each join gets >= 1 overlap frame.
    if (n > 1) {
        long long sum_audio = 0;
        for (int f : in.clip_frames) {
            sum_audio += AudioLatentFrames(f, in.frame_rate);
        }
        const long long a_total = AudioLatentFrames(total_px, in.frame_rate);
        const long long sum_ka = sum_audio - a_total;
        if (sum_ka < static_cast<long long>(n - 1)) {
            r.error = TimelineError::kAudioDegenerate;
            return r;
        }
    }

    if (total_px > kMaxChainTotalPixelFrames) {
        r.error = TimelineError::kChainTooLong;
        return r;
    }

    r.f_total_latent = static_cast<int>(f_total);
    r.total_px = total_px;
    // vid2vid: the delivered mp4 is the NEW part only - the frozen context
    // (source_context_px pixel frames) is trimmed off the front (chain_math
    // trim_px == source_context_px).
    const int trim = in.has_source_context ? in.source_context_px : 0;
    r.delivered_px = total_px - trim;
    r.total_seconds = total_px / in.frame_rate;
    r.delivered_seconds = r.delivered_px / in.frame_rate;
    return r;
}

bool IsValidHandleCrossfadeMs(int ms) {
    return ms >= 0 && ms <= kMaxHandleCrossfadeMs;
}

JoinLengthEstimate EstimateJoinLength(int source_frames, int continuation_frames,
                                      double fps) {
    JoinLengthEstimate e;
    const int s = source_frames > 0 ? source_frames : 0;
    const int c = continuation_frames > 0 ? continuation_frames : 0;
    e.joined_frames = s + c;
    e.joined_seconds = fps > 0.0 ? e.joined_frames / fps : 0.0;
    return e;
}

}  // namespace nzvideomni
