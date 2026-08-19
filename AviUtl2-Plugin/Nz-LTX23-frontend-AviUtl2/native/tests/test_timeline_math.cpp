// test_timeline_math.cpp - unit tests for the generation-length pre-computation.
// Expected values were cross-checked against Nz-LTX23-backend/chain_math.py.
#include "doctest.h"

#include "timeline_math.h"

#include <limits>

using namespace nzltx;

// ---------------------------------------------------------------------------
// 8n+1 validation / snapping (models.py GenerateRequest bounds)
// ---------------------------------------------------------------------------

TEST_CASE("IsValidNumFrames boundaries and alignment") {
    CHECK(IsValidNumFrames(9));
    CHECK(IsValidNumFrames(481));
    CHECK(IsValidNumFrames(49));
    CHECK_FALSE(IsValidNumFrames(8));    // below range
    CHECK_FALSE(IsValidNumFrames(482));  // above range
    CHECK_FALSE(IsValidNumFrames(50));   // not 8n+1
}

TEST_CASE("NearestValidNumFrames clamps and snaps (ties round up)") {
    CHECK(NearestValidNumFrames(5) == 9);      // clamp low
    CHECK(NearestValidNumFrames(500) == 481);  // clamp high
    CHECK(NearestValidNumFrames(9) == 9);
    CHECK(NearestValidNumFrames(481) == 481);
    CHECK(NearestValidNumFrames(50) == 49);    // 6.625 -> 6 -> 49
    CHECK(NearestValidNumFrames(53) == 57);    // 7.0 tie -> up -> 57
    CHECK(NearestValidNumFrames(100) == 97);   // 12.875 -> 12 -> 97
}

// ---------------------------------------------------------------------------
// latent geometry (chain_math.v_latent_frames / px_from_v_latent)
// ---------------------------------------------------------------------------

TEST_CASE("v_latent round trip for 8n+1 pixel counts") {
    for (int p : {9, 17, 49, 89, 177, 481}) {
        CHECK(PxFromVLatent(VLatentFrames(p)) == p);
    }
    CHECK(VLatentFrames(49) == 7);
    CHECK(PxFromVLatent(7) == 49);
    CHECK(VLatentFrames(481) == 61);
}

TEST_CASE("AudioLatentFrames matches a_frames_for_px (round half to even)") {
    CHECK(AudioLatentFrames(49, 24.0) == 51);   // round(51.0416)
    CHECK(AudioLatentFrames(81, 24.0) == 84);   // round(84.375)
    CHECK(AudioLatentFrames(9, 24.0) == 9);     // round(9.375)
    CHECK(AudioLatentFrames(24, 24.0) == 25);   // exact 25
    CHECK(AudioLatentFrames(10, 0.0) == 0);     // guard: fps <= 0
}

// ---------------------------------------------------------------------------
// generation fps <-> project fps
// ---------------------------------------------------------------------------

TEST_CASE("ProjectFramesForPixels preserves duration (round half up)") {
    CHECK(ProjectFramesForPixels(49, 24.0, 24.0) == 49);   // same fps
    CHECK(ProjectFramesForPixels(49, 24.0, 30.0) == 61);   // 61.25 -> 61
    CHECK(ProjectFramesForPixels(81, 24.0, 60.0) == 203);  // 202.5 -> 203 (up)
    CHECK(ProjectFramesForPixels(100, 0.0, 30.0) == 0);    // guard
}

TEST_CASE("ProjectFramesForSeconds sizes media to real length (round half up)") {
    CHECK(ProjectFramesForSeconds(2.0, 24.0) == 48);        // exact
    CHECK(ProjectFramesForSeconds(2.0, 30.0) == 60);        // exact
    CHECK(ProjectFramesForSeconds(1.0, 23.976) == 24);      // 23.976 -> 24
    CHECK(ProjectFramesForSeconds(1.0 / 48.0, 24.0) == 1);  // 0.5 tie -> up
    CHECK(ProjectFramesForSeconds(0.0, 30.0) == 0);         // guard: still image
    CHECK(ProjectFramesForSeconds(3.0, 0.0) == 0);          // guard: fps <= 0
}

TEST_CASE("ProjectFramesForSeconds clamps out-of-range products (no UB cast)") {
    // A corrupt/huge total_time makes seconds*fps exceed INT_MAX; the raw
    // static_cast<int>(double) would be undefined behavior, so it clamps.
    CHECK(ProjectFramesForSeconds(1e18, 30.0) == 2000000000);
    // +inf is NOT rejected by the !(x > 0) guard (inf > 0 is true); floor(inf)
    // is inf and inf > kMaxFrames, so it reaches the same clamp, not the cast.
    CHECK(ProjectFramesForSeconds(std::numeric_limits<double>::infinity(), 30.0) == 2000000000);
    // NaN is rejected by the !(x > 0) guard (every NaN comparison is false).
    CHECK(ProjectFramesForSeconds(std::numeric_limits<double>::quiet_NaN(), 30.0) == 0);
    CHECK(ProjectFramesForSeconds(2.0, std::numeric_limits<double>::quiet_NaN()) == 0);
    // Negative seconds -> 0.
    CHECK(ProjectFramesForSeconds(-2.0, 30.0) == 0);
    // Round-half-up lower edge: 0.02 * 24 = 0.48 -> floor(0.98) = 0.
    CHECK(ProjectFramesForSeconds(0.02, 24.0) == 0);
}

// ---------------------------------------------------------------------------
// span occupancy (D2 provisional collision pre-check kernel)
// ---------------------------------------------------------------------------

TEST_CASE("SpansOverlap detects inclusive-span intersection and its boundaries") {
    // Disjoint, well clear of each other.
    CHECK_FALSE(SpansOverlap(0, 4, 10, 14));
    CHECK_FALSE(SpansOverlap(10, 14, 0, 4));
    // Adjacent but NOT touching (a ends at 9, b starts at 10) -> free.
    CHECK_FALSE(SpansOverlap(0, 9, 10, 19));
    // Touching at exactly one frame (a ends where b starts) -> occupied.
    CHECK(SpansOverlap(0, 10, 10, 19));
    // The requested span [100,124] (frame 100, length 25) against an existing
    // object: exactly-enough gap before it (ends at 99) is free; one frame short
    // (ends at 100) collides.
    CHECK_FALSE(SpansOverlap(0, 99, 100, 124));
    CHECK(SpansOverlap(0, 100, 100, 124));
    // Full containment, either way round.
    CHECK(SpansOverlap(50, 60, 0, 100));
    CHECK(SpansOverlap(0, 100, 50, 60));
    // A degenerate (empty) span never overlaps, even when nominally "inside".
    CHECK_FALSE(SpansOverlap(5, 4, 0, 100));
    CHECK_FALSE(SpansOverlap(0, 100, 5, 4));
}

// ---------------------------------------------------------------------------
// single clip (txt2vid / img2vid)
// ---------------------------------------------------------------------------

TEST_CASE("ComputeSingleClip valid and error paths") {
    SingleClipResult ok = ComputeSingleClip(49, 24.0);
    CHECK(ok.error == TimelineError::kNone);
    CHECK(ok.pixel_frames == 49);
    CHECK(ok.seconds == doctest::Approx(49.0 / 24.0));

    SingleClipResult cap = ComputeSingleClip(481, 24.0);
    CHECK(cap.error == TimelineError::kNone);
    CHECK(cap.seconds == doctest::Approx(481.0 / 24.0));

    CHECK(ComputeSingleClip(50, 24.0).error == TimelineError::kNumFramesNotAligned);
    CHECK(ComputeSingleClip(8, 24.0).error == TimelineError::kNumFramesRange);
    CHECK(ComputeSingleClip(482, 24.0).error == TimelineError::kNumFramesRange);
    CHECK(ComputeSingleClip(49, 0.5).error == TimelineError::kFrameRateRange);
    CHECK(ComputeSingleClip(49, 61.0).error == TimelineError::kFrameRateRange);
}

// ---------------------------------------------------------------------------
// chain (chain_math.compute_chain_layout)
// ---------------------------------------------------------------------------

static ChainInput MakeChain(std::vector<int> clips, int kv = 3, double fps = 24.0) {
    ChainInput in;
    in.clip_frames = std::move(clips);
    in.overlap_frames = kv;
    in.frame_rate = fps;
    return in;
}

TEST_CASE("ComputeChain single clip equals its own frame count") {
    ChainResult r = ComputeChain(MakeChain({49}));
    CHECK(r.error == TimelineError::kNone);
    CHECK(r.f_total_latent == 7);
    CHECK(r.total_px == 49);
    CHECK(r.delivered_px == 49);
    CHECK(r.delivered_seconds == doctest::Approx(49.0 / 24.0));
}

TEST_CASE("ComputeChain two clips with default overlap") {
    ChainResult r = ComputeChain(MakeChain({49, 49}, 3));
    CHECK(r.error == TimelineError::kNone);
    CHECK(r.f_total_latent == 11);  // 7 + 7 - 3
    CHECK(r.total_px == 81);        // px_from_v_latent(11)
    CHECK(r.delivered_px == 81);
    CHECK(r.delivered_seconds == doctest::Approx(81.0 / 24.0));  // 3.375
}

TEST_CASE("larger overlap shrinks the timeline") {
    ChainResult a = ComputeChain(MakeChain({49, 49}, 3));
    ChainResult b = ComputeChain(MakeChain({49, 49}, 6));
    CHECK(b.error == TimelineError::kNone);
    CHECK(b.total_px == 57);      // 14 - 6 -> 8 latents -> 57 px
    CHECK(b.total_px < a.total_px);
}

TEST_CASE("ComputeChain input validation") {
    CHECK(ComputeChain(MakeChain({}, 3)).error == TimelineError::kNoClips);
    CHECK(ComputeChain(MakeChain({49, 49, 49, 49, 49, 49, 49, 49, 49}, 3)).error ==
          TimelineError::kTooManyClips);
    CHECK(ComputeChain(MakeChain({49, 49}, 0)).error == TimelineError::kOverlapRange);
    CHECK(ComputeChain(MakeChain({49, 49}, 9)).error == TimelineError::kOverlapRange);
    // clip of 9 frames -> 2 stage-1 latents; K_v=3 cannot fit.
    CHECK(ComputeChain(MakeChain({9, 49}, 3)).error == TimelineError::kOverlapTooLarge);
    // malformed clip counts.
    CHECK(ComputeChain(MakeChain({50, 49}, 3)).error == TimelineError::kNumFramesNotAligned);
    CHECK(ComputeChain(MakeChain({8, 49}, 3)).error == TimelineError::kNumFramesRange);
}

TEST_CASE("degenerate audio overlap is rejected") {
    // Two 9-frame clips: seg_audio 9+9=18, a_total(17px)=18 -> sum_ka 0 < 1 join.
    CHECK(ComputeChain(MakeChain({9, 9}, 1)).error == TimelineError::kAudioDegenerate);
}

TEST_CASE("near-maximum chain stays within the total cap") {
    // 8 clips of 481 with K_v=1 -> 3841 px, below the 3848 sanity cap.
    ChainResult r = ComputeChain(MakeChain({481, 481, 481, 481, 481, 481, 481, 481}, 1));
    CHECK(r.error == TimelineError::kNone);
    CHECK(r.total_px == 3841);
    CHECK(r.total_px <= kMaxChainTotalPixelFrames);
}

// ---------------------------------------------------------------------------
// vid2vid continuation (source_video)
// ---------------------------------------------------------------------------

TEST_CASE("ComputeChain vid2vid trims the frozen context from delivery") {
    ChainInput in = MakeChain({89}, 3);
    in.has_source_context = true;
    in.source_context_px = 25;  // 8n+1, < 89, v_latent 4 <= tile 22
    ChainResult r = ComputeChain(in);
    CHECK(r.error == TimelineError::kNone);
    CHECK(r.total_px == 89);           // full decoded timeline (context included)
    CHECK(r.delivered_px == 64);       // 89 - 25 (context trimmed off the front)
    CHECK(r.delivered_seconds == doctest::Approx(64.0 / 24.0));
    CHECK(r.total_seconds == doctest::Approx(89.0 / 24.0));
}

TEST_CASE("vid2vid context invariants") {
    auto ctx = [](int px, int clip0) {
        ChainInput in = MakeChain({clip0}, 3);
        in.has_source_context = true;
        in.source_context_px = px;
        return ComputeChain(in).error;
    };
    CHECK(ctx(24, 89) == TimelineError::kSourceContextNotAligned);  // 24 not 8n+1
    CHECK(ctx(0, 89) == TimelineError::kSourceContextRange);        // < 1
    CHECK(ctx(57, 49) == TimelineError::kSourceContextNotLessThanClip0);
    CHECK(ctx(177, 481) == TimelineError::kSourceContextTooLong);   // v_latent 23 > 22
}

// ---------------------------------------------------------------------------
// join (approximate, audio crossfade is audio-only)
// ---------------------------------------------------------------------------

TEST_CASE("EstimateJoinLength is a hard concat of frame counts") {
    JoinLengthEstimate e = EstimateJoinLength(100, 64, 24.0);
    CHECK(e.joined_frames == 164);
    CHECK(e.joined_seconds == doctest::Approx(164.0 / 24.0));
    CHECK(EstimateJoinLength(-5, 64, 24.0).joined_frames == 64);  // negative -> 0
}

TEST_CASE("IsValidHandleCrossfadeMs range") {
    CHECK(IsValidHandleCrossfadeMs(0));
    CHECK(IsValidHandleCrossfadeMs(kDefaultHandleCrossfadeMs));
    CHECK(IsValidHandleCrossfadeMs(2000));
    CHECK_FALSE(IsValidHandleCrossfadeMs(2001));
    CHECK_FALSE(IsValidHandleCrossfadeMs(-1));
}
