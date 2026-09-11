// test_track_postprocess.cpp - unit tests for the object-tracking post-process
// (section 3-54). The six stages and their order are the contract; these cases
// pin each stage separately and then the couplings between them, because the
// order is what the design fixes (Docs\OBJECT_TRACKING_DESIGN.md section 5.5).
#include "doctest.h"

#include <vector>

#include "track_postprocess.h"

using namespace nzvideomni;

namespace {

TrackSample S(int frame, double x, double y, double w, double h, double score) {
    TrackSample s;
    s.frame = frame;
    s.x = x;
    s.y = y;
    s.w = w;
    s.h = h;
    s.score = score;
    return s;
}

// Neutral options: half-way threshold, hold on lost, no smoothing, size follows,
// every frame kept. Each case flips only what it is about.
TrackPostOptions Opt() {
    TrackPostOptions o;
    o.lost_score_threshold = 0.5;
    o.hold_on_lost = true;
    o.smoothing = 0.0;
    o.follow_size = true;
    o.keyframe_stride = 1;
    return o;
}

}  // namespace

// ---------------------------------------------------------------------------
// Degenerate inputs
// ---------------------------------------------------------------------------

TEST_CASE("an empty sample list yields an empty result") {
    const TrackPostResult r = PostProcessTrack(std::vector<TrackSample>(), Opt());
    CHECK(r.keyframes.empty());
    CHECK(r.lost_ranges.empty());
}

TEST_CASE("a single sample yields exactly one keyframe") {
    std::vector<TrackSample> in{S(0, 10, 20, 30, 40, 0.9)};
    const TrackPostResult r = PostProcessTrack(in, Opt());
    REQUIRE(r.keyframes.size() == 1);
    CHECK(r.keyframes[0].frame_offset == 0);
    CHECK(r.keyframes[0].x == doctest::Approx(10.0));
    CHECK(r.keyframes[0].y == doctest::Approx(20.0));
    CHECK(r.keyframes[0].w == doctest::Approx(30.0));
    CHECK(r.keyframes[0].h == doctest::Approx(40.0));
    CHECK(r.lost_ranges.empty());
}

TEST_CASE("a single sample is not duplicated by the always-keep-last rule") {
    // The first and the last sample are the same one, so forcing the last in
    // must not emit it twice - including under a stride that skips it.
    std::vector<TrackSample> in{S(7, 1, 2, 3, 4, 0.9)};
    TrackPostOptions o = Opt();
    o.keyframe_stride = 9;
    const TrackPostResult r = PostProcessTrack(in, o);
    REQUIRE(r.keyframes.size() == 1);
    CHECK(r.keyframes[0].frame_offset == 7);
}

// ---------------------------------------------------------------------------
// Stage 1: the lost judgement
// ---------------------------------------------------------------------------

TEST_CASE("the lost threshold is strict - a score equal to it is good") {
    std::vector<TrackSample> in{S(0, 0, 0, 10, 10, 0.5), S(1, 0, 0, 10, 10, 0.4999)};
    const TrackPostResult r = PostProcessTrack(in, Opt());
    REQUIRE(r.lost_ranges.size() == 1);
    CHECK(r.lost_ranges[0].start == 1);
    CHECK(r.lost_ranges[0].end == 1);
}

TEST_CASE("a threshold of 0 marks nothing lost") {
    std::vector<TrackSample> in{S(0, 0, 0, 10, 10, 0.0), S(1, 0, 0, 10, 10, 0.01)};
    TrackPostOptions o = Opt();
    o.lost_score_threshold = 0.0;
    const TrackPostResult r = PostProcessTrack(in, o);
    CHECK(r.lost_ranges.empty());
    CHECK(r.keyframes.size() == 2);
}

// ---------------------------------------------------------------------------
// Stage 2: hold vs keep following
// ---------------------------------------------------------------------------

TEST_CASE("holding refills a lost stretch from the preceding good box") {
    std::vector<TrackSample> in{S(0, 100, 0, 10, 10, 0.9),
                                S(1, 900, 0, 10, 10, 0.1),
                                S(2, 950, 0, 10, 10, 0.1),
                                S(3, 200, 0, 10, 10, 0.9)};
    const TrackPostResult r = PostProcessTrack(in, Opt());
    REQUIRE(r.keyframes.size() == 4);
    CHECK(r.keyframes[1].x == doctest::Approx(100.0));
    CHECK(r.keyframes[2].x == doctest::Approx(100.0));
    CHECK(r.keyframes[3].x == doctest::Approx(200.0));
}

TEST_CASE("keep-following leaves the raw boxes of a lost stretch alone") {
    std::vector<TrackSample> in{S(0, 100, 0, 10, 10, 0.9),
                                S(1, 900, 0, 10, 10, 0.1),
                                S(2, 200, 0, 10, 10, 0.9)};
    TrackPostOptions o = Opt();
    o.hold_on_lost = false;
    const TrackPostResult r = PostProcessTrack(in, o);
    REQUIRE(r.keyframes.size() == 3);
    CHECK(r.keyframes[1].x == doctest::Approx(900.0));
    // The range is still reported - the user is told, the box is just not moved.
    REQUIRE(r.lost_ranges.size() == 1);
    CHECK(r.lost_ranges[0].start == 1);
}

TEST_CASE("a lost run at the head borrows the first good box") {
    // There is no preceding good box, so the earliest good one is projected
    // backwards rather than leaving a wild box at frame 0.
    std::vector<TrackSample> in{S(0, 900, 0, 10, 10, 0.1),
                                S(1, 950, 0, 10, 10, 0.1),
                                S(2, 300, 0, 10, 10, 0.9)};
    const TrackPostResult r = PostProcessTrack(in, Opt());
    REQUIRE(r.keyframes.size() == 3);
    CHECK(r.keyframes[0].x == doctest::Approx(300.0));
    CHECK(r.keyframes[1].x == doctest::Approx(300.0));
}

TEST_CASE("a lost run at the tail holds the last good box") {
    std::vector<TrackSample> in{S(0, 100, 0, 10, 10, 0.9),
                                S(1, 400, 0, 10, 10, 0.9),
                                S(2, 900, 0, 10, 10, 0.1),
                                S(3, 950, 0, 10, 10, 0.1)};
    const TrackPostResult r = PostProcessTrack(in, Opt());
    REQUIRE(r.keyframes.size() == 4);
    CHECK(r.keyframes[2].x == doctest::Approx(400.0));
    CHECK(r.keyframes[3].x == doctest::Approx(400.0));
    REQUIRE(r.lost_ranges.size() == 1);
    CHECK(r.lost_ranges[0].start == 2);
    CHECK(r.lost_ranges[0].end == 3);
}

TEST_CASE("with every sample lost the raw boxes pass through even when holding") {
    // Nothing good to hold: the alternative would be to invent a box.
    std::vector<TrackSample> in{S(0, 10, 0, 4, 4, 0.1), S(1, 20, 0, 4, 4, 0.2),
                                S(2, 30, 0, 4, 4, 0.0)};
    const TrackPostResult r = PostProcessTrack(in, Opt());
    REQUIRE(r.keyframes.size() == 3);
    CHECK(r.keyframes[0].x == doctest::Approx(10.0));
    CHECK(r.keyframes[1].x == doctest::Approx(20.0));
    CHECK(r.keyframes[2].x == doctest::Approx(30.0));
    REQUIRE(r.lost_ranges.size() == 1);
    CHECK(r.lost_ranges[0].start == 0);
    CHECK(r.lost_ranges[0].end == 2);
}

// ---------------------------------------------------------------------------
// Stage 3: the exponential moving average
// ---------------------------------------------------------------------------

TEST_CASE("smoothing 0 passes the boxes through untouched") {
    std::vector<TrackSample> in{S(0, 0, 0, 10, 10, 0.9), S(1, 10, 20, 30, 40, 0.9)};
    const TrackPostResult r = PostProcessTrack(in, Opt());
    REQUIRE(r.keyframes.size() == 2);
    CHECK(r.keyframes[1].x == doctest::Approx(10.0));
    CHECK(r.keyframes[1].y == doctest::Approx(20.0));
    CHECK(r.keyframes[1].w == doctest::Approx(30.0));
    CHECK(r.keyframes[1].h == doctest::Approx(40.0));
}

TEST_CASE("smoothing 1 freezes every keyframe at the first box") {
    std::vector<TrackSample> in{S(0, 5, 6, 7, 8, 0.9), S(1, 500, 600, 700, 800, 0.9),
                                S(2, 900, 900, 900, 900, 0.9)};
    TrackPostOptions o = Opt();
    o.smoothing = 1.0;
    const TrackPostResult r = PostProcessTrack(in, o);
    REQUIRE(r.keyframes.size() == 3);
    for (size_t i = 0; i < r.keyframes.size(); ++i) {
        CHECK(r.keyframes[i].x == doctest::Approx(5.0));
        CHECK(r.keyframes[i].y == doctest::Approx(6.0));
        CHECK(r.keyframes[i].w == doctest::Approx(7.0));
        CHECK(r.keyframes[i].h == doctest::Approx(8.0));
    }
}

TEST_CASE("smoothing 0.5 halves the remaining distance each frame") {
    // alpha = 1 - smoothing = 0.5: out[i] = 0.5*in[i] + 0.5*out[i-1].
    std::vector<TrackSample> in{S(0, 0, 0, 10, 10, 0.9), S(1, 10, 0, 10, 10, 0.9),
                                S(2, 20, 0, 10, 10, 0.9)};
    TrackPostOptions o = Opt();
    o.smoothing = 0.5;
    const TrackPostResult r = PostProcessTrack(in, o);
    REQUIRE(r.keyframes.size() == 3);
    CHECK(r.keyframes[0].x == doctest::Approx(0.0));
    CHECK(r.keyframes[1].x == doctest::Approx(5.0));
    CHECK(r.keyframes[2].x == doctest::Approx(12.5));
}

TEST_CASE("smoothing applies to the size as well as the position") {
    std::vector<TrackSample> in{S(0, 0, 0, 100, 200, 0.9), S(1, 0, 0, 200, 400, 0.9)};
    TrackPostOptions o = Opt();
    o.smoothing = 0.5;
    const TrackPostResult r = PostProcessTrack(in, o);
    REQUIRE(r.keyframes.size() == 2);
    CHECK(r.keyframes[1].w == doctest::Approx(150.0));
    CHECK(r.keyframes[1].h == doctest::Approx(300.0));
}

TEST_CASE("a smoothing value outside 0..1 is clamped into it") {
    std::vector<TrackSample> in{S(0, 0, 0, 10, 10, 0.9), S(1, 10, 0, 10, 10, 0.9)};
    TrackPostOptions lo = Opt();
    lo.smoothing = -3.0;  // clamps to 0 -> pass-through
    TrackPostOptions hi = Opt();
    hi.smoothing = 4.0;  // clamps to 1 -> frozen
    const TrackPostResult a = PostProcessTrack(in, lo);
    const TrackPostResult b = PostProcessTrack(in, hi);
    CHECK(a.keyframes[1].x == doctest::Approx(10.0));
    CHECK(b.keyframes[1].x == doctest::Approx(0.0));
}

TEST_CASE("holding feeds the smoothing - the stages run in order") {
    // Same input, same smoothing: holding first removes the excursion, so the
    // smoothed result differs from the keep-following one. This is what makes
    // the fixed stage order observable.
    std::vector<TrackSample> in{S(0, 0, 0, 10, 10, 0.9), S(1, 1000, 0, 10, 10, 0.1)};
    TrackPostOptions hold = Opt();
    hold.smoothing = 0.5;
    TrackPostOptions cont = hold;
    cont.hold_on_lost = false;
    CHECK(PostProcessTrack(in, hold).keyframes[1].x == doctest::Approx(0.0));
    CHECK(PostProcessTrack(in, cont).keyframes[1].x == doctest::Approx(500.0));
}

// ---------------------------------------------------------------------------
// Stage 4: pinning the size to the seed
// ---------------------------------------------------------------------------

TEST_CASE("follow_size off pins w and h to the seed for every keyframe") {
    std::vector<TrackSample> in{S(0, 0, 0, 100, 50, 0.9), S(1, 10, 10, 300, 700, 0.9),
                                S(2, 20, 20, 1, 1, 0.9)};
    TrackPostOptions o = Opt();
    o.follow_size = false;
    const TrackPostResult r = PostProcessTrack(in, o);
    REQUIRE(r.keyframes.size() == 3);
    for (size_t i = 0; i < r.keyframes.size(); ++i) {
        CHECK(r.keyframes[i].w == doctest::Approx(100.0));
        CHECK(r.keyframes[i].h == doctest::Approx(50.0));
    }
    // The position is still free to move.
    CHECK(r.keyframes[2].x == doctest::Approx(20.0));
}

TEST_CASE("the pinned size is the RAW first sample, not the held one") {
    // Frame 0 is lost and gets refilled from frame 1 by stage 2; stage 4 must
    // still use the seed the user aimed in the preview, captured beforehand.
    std::vector<TrackSample> in{S(0, 0, 0, 11, 22, 0.1), S(1, 0, 0, 500, 600, 0.9)};
    TrackPostOptions o = Opt();
    o.follow_size = false;
    const TrackPostResult r = PostProcessTrack(in, o);
    REQUIRE(r.keyframes.size() == 2);
    CHECK(r.keyframes[0].w == doctest::Approx(11.0));
    CHECK(r.keyframes[1].w == doctest::Approx(11.0));
    CHECK(r.keyframes[1].h == doctest::Approx(22.0));
}

TEST_CASE("follow_size on lets the size move") {
    std::vector<TrackSample> in{S(0, 0, 0, 100, 50, 0.9), S(1, 0, 0, 300, 700, 0.9)};
    const TrackPostResult r = PostProcessTrack(in, Opt());
    REQUIRE(r.keyframes.size() == 2);
    CHECK(r.keyframes[1].w == doctest::Approx(300.0));
    CHECK(r.keyframes[1].h == doctest::Approx(700.0));
}

// ---------------------------------------------------------------------------
// Stage 5: keyframe thinning
// ---------------------------------------------------------------------------

TEST_CASE("stride 1 keeps every sample") {
    std::vector<TrackSample> in;
    for (int i = 0; i < 5; ++i) {
        in.push_back(S(i, i * 10.0, 0, 10, 10, 0.9));
    }
    const TrackPostResult r = PostProcessTrack(in, Opt());
    REQUIRE(r.keyframes.size() == 5);
    for (int i = 0; i < 5; ++i) {
        CHECK(r.keyframes[static_cast<size_t>(i)].frame_offset == i);
    }
}

TEST_CASE("stride 3 keeps every third sample and still keeps the last") {
    // 5 samples: the stride lands on 0 and 3, and 4 is forced in as the end.
    std::vector<TrackSample> in;
    for (int i = 0; i < 5; ++i) {
        in.push_back(S(i, i * 10.0, 0, 10, 10, 0.9));
    }
    TrackPostOptions o = Opt();
    o.keyframe_stride = 3;
    const TrackPostResult r = PostProcessTrack(in, o);
    REQUIRE(r.keyframes.size() == 3);
    CHECK(r.keyframes[0].frame_offset == 0);
    CHECK(r.keyframes[1].frame_offset == 3);
    CHECK(r.keyframes[2].frame_offset == 4);
    CHECK(r.keyframes[2].x == doctest::Approx(40.0));
}

TEST_CASE("a stride that already lands on the last sample does not duplicate it") {
    std::vector<TrackSample> in;
    for (int i = 0; i < 7; ++i) {  // last index 6 = 2 * 3
        in.push_back(S(i, i * 10.0, 0, 10, 10, 0.9));
    }
    TrackPostOptions o = Opt();
    o.keyframe_stride = 3;
    const TrackPostResult r = PostProcessTrack(in, o);
    REQUIRE(r.keyframes.size() == 3);
    CHECK(r.keyframes[0].frame_offset == 0);
    CHECK(r.keyframes[1].frame_offset == 3);
    CHECK(r.keyframes[2].frame_offset == 6);
}

TEST_CASE("a huge stride still keeps both ends") {
    std::vector<TrackSample> in;
    for (int i = 0; i < 10; ++i) {
        in.push_back(S(i, i * 10.0, 0, 10, 10, 0.9));
    }
    TrackPostOptions o = Opt();
    o.keyframe_stride = 100000;
    const TrackPostResult r = PostProcessTrack(in, o);
    REQUIRE(r.keyframes.size() == 2);
    CHECK(r.keyframes[0].frame_offset == 0);
    CHECK(r.keyframes[1].frame_offset == 9);
}

TEST_CASE("a stride below 1 is read as 1") {
    std::vector<TrackSample> in;
    for (int i = 0; i < 4; ++i) {
        in.push_back(S(i, 0, 0, 10, 10, 0.9));
    }
    TrackPostOptions zero = Opt();
    zero.keyframe_stride = 0;
    TrackPostOptions negative = Opt();
    negative.keyframe_stride = -5;
    CHECK(PostProcessTrack(in, zero).keyframes.size() == 4);
    CHECK(PostProcessTrack(in, negative).keyframes.size() == 4);
}

TEST_CASE("thinning does not shift the phase when the frames are not 0-based") {
    // frame_offset echoes the sample's own relative number; the stride counts
    // positions in the list, so a list that starts at 100 keeps 100 and 103.
    std::vector<TrackSample> in;
    for (int i = 0; i < 5; ++i) {
        in.push_back(S(100 + i, 0, 0, 10, 10, 0.9));
    }
    TrackPostOptions o = Opt();
    o.keyframe_stride = 3;
    const TrackPostResult r = PostProcessTrack(in, o);
    REQUIRE(r.keyframes.size() == 3);
    CHECK(r.keyframes[0].frame_offset == 100);
    CHECK(r.keyframes[1].frame_offset == 103);
    CHECK(r.keyframes[2].frame_offset == 104);
}

// ---------------------------------------------------------------------------
// Stage 6: collapsing lost runs
// ---------------------------------------------------------------------------

TEST_CASE("a single lost frame is its own one-frame range") {
    std::vector<TrackSample> in{S(0, 0, 0, 10, 10, 0.9), S(1, 0, 0, 10, 10, 0.1),
                                S(2, 0, 0, 10, 10, 0.9)};
    const TrackPostResult r = PostProcessTrack(in, Opt());
    REQUIRE(r.lost_ranges.size() == 1);
    CHECK(r.lost_ranges[0].start == 1);
    CHECK(r.lost_ranges[0].end == 1);
}

TEST_CASE("two lost runs separated by one good frame stay separate") {
    std::vector<TrackSample> in{S(0, 0, 0, 10, 10, 0.1), S(1, 0, 0, 10, 10, 0.1),
                                S(2, 0, 0, 10, 10, 0.9), S(3, 0, 0, 10, 10, 0.1)};
    const TrackPostResult r = PostProcessTrack(in, Opt());
    REQUIRE(r.lost_ranges.size() == 2);
    CHECK(r.lost_ranges[0].start == 0);
    CHECK(r.lost_ranges[0].end == 1);
    CHECK(r.lost_ranges[1].start == 3);
    CHECK(r.lost_ranges[1].end == 3);
}

TEST_CASE("a lost run that reaches the last sample is closed at it") {
    std::vector<TrackSample> in{S(10, 0, 0, 10, 10, 0.9), S(11, 0, 0, 10, 10, 0.1),
                                S(12, 0, 0, 10, 10, 0.1)};
    const TrackPostResult r = PostProcessTrack(in, Opt());
    REQUIRE(r.lost_ranges.size() == 1);
    CHECK(r.lost_ranges[0].start == 11);
    CHECK(r.lost_ranges[0].end == 12);
}

TEST_CASE("lost ranges are reported over ALL samples, not just the kept ones") {
    // A stride of 5 keeps only frames 0 and 5, but the lost frames 2..3 in
    // between must still be listed for the user.
    std::vector<TrackSample> in;
    for (int i = 0; i < 6; ++i) {
        const double score = (i == 2 || i == 3) ? 0.1 : 0.9;
        in.push_back(S(i, 0, 0, 10, 10, score));
    }
    TrackPostOptions o = Opt();
    o.keyframe_stride = 5;
    const TrackPostResult r = PostProcessTrack(in, o);
    CHECK(r.keyframes.size() == 2);
    REQUIRE(r.lost_ranges.size() == 1);
    CHECK(r.lost_ranges[0].start == 2);
    CHECK(r.lost_ranges[0].end == 3);
}
