// test_provisional.cpp - unit tests for the provisional-object state machine.
// Includes a round trip against alias_util::BuildProvisionalTextAlias so the
// tag format stays symmetric with the placeholder builder.
#include "doctest.h"

#include <set>
#include <string>
#include <vector>

#include "alias_util.h"
#include "provisional.h"

using namespace nzltx;

namespace {

// The AviUtl2 "text" effect name (also its text-item key), UTF-8 byte escapes,
// matching alias_util.cpp / provisional.cpp.
const char* kTxt = "\xe3\x83\x86\xe3\x82\xad\xe3\x82\xb9\xe3\x83\x88";

// Build a minimal text-object alias whose visible text carries the "[#job]"
// marker, as BuildProvisionalTextAlias does, but with controllable text.
std::string TextAlias(const std::string& text) {
    std::string a;
    a += "[Object]\r\n";
    a += "[Object.0]\r\n";
    a += "effect.name=";
    a += kTxt;
    a += "\r\n";
    a += kTxt;  // text-content item key
    a += "=";
    a += text;
    a += "\r\n";
    return a;
}

ScannedObject Obj(int layer, int fs, int fe, const std::string& alias) {
    ScannedObject o;
    o.layer = layer;
    o.frame_start = fs;
    o.frame_end = fe;
    o.alias = alias;
    return o;
}

}  // namespace

// ---------------------------------------------------------------------------
// AliasMatchesJob - exact whole-token matching
// ---------------------------------------------------------------------------

TEST_CASE("AliasMatchesJob matches the text marker exactly") {
    const std::string a = TextAlias("generating [#job-abc]");
    CHECK(AliasMatchesJob(a, "job-abc"));
    CHECK_FALSE(AliasMatchesJob(a, "job-ab"));    // prefix must not match
    CHECK_FALSE(AliasMatchesJob(a, "job-abcd"));  // superset must not match
    CHECK_FALSE(AliasMatchesJob(a, ""));          // empty never matches
}

TEST_CASE("AliasMatchesJob matches the object_name token with a boundary") {
    // A raw alias carrying a serialized object_name token.
    const std::string a = "[Object]\r\nname=NzLTX23#abc\r\n";
    CHECK(AliasMatchesJob(a, "abc"));
    CHECK_FALSE(AliasMatchesJob(a, "ab"));  // "NzLTX23#abc" is not "NzLTX23#ab"
    // A longer neighbouring id must not satisfy a shorter query.
    const std::string b = "[Object]\r\nname=NzLTX23#abcd\r\n";
    CHECK_FALSE(AliasMatchesJob(b, "abc"));
    CHECK(AliasMatchesJob(b, "abcd"));
}

TEST_CASE("round trip: BuildProvisionalTextAlias output matches its job id") {
    const ProvisionalTextAlias built =
        BuildProvisionalTextAlias("my prompt text", "7f3a-JOB-01");
    CHECK(AliasMatchesJob(built.alias, "7f3a-JOB-01"));
    CHECK_FALSE(AliasMatchesJob(built.alias, "7f3a-JOB-0"));  // prefix
    CHECK(built.object_name == "NzLTX23#7f3a-JOB-01");
    // The object_name token form also resolves.
    const std::string named = "name=" + built.object_name + "\r\n";
    CHECK(AliasMatchesJob(named, "7f3a-JOB-01"));
}

// ---------------------------------------------------------------------------
// FindProvisionalIndex - exact match then reserved-position fallback
// ---------------------------------------------------------------------------

TEST_CASE("FindProvisionalIndex finds by exact alias match") {
    std::vector<ScannedObject> scan = {
        Obj(1, 0, 30, TextAlias("x [#other]")),
        Obj(2, 10, 40, TextAlias("y [#target]")),
        Obj(3, 5, 25, TextAlias("z [#target-2]")),
    };
    CHECK(FindProvisionalIndex(scan, "target", -1, 0) == 1);
    CHECK(FindProvisionalIndex(scan, "target-2", -1, 0) == 2);
    CHECK(FindProvisionalIndex(scan, "missing", -1, 0) == -1);
}

TEST_CASE("FindProvisionalIndex falls back to the reserved covering frame") {
    // No alias carries the marker (user edited the text away).
    std::vector<ScannedObject> scan = {
        Obj(1, 0, 30, TextAlias("edited")),
        Obj(2, 10, 40, TextAlias("edited too")),  // layer 2 covers frame 20
    };
    CHECK(FindProvisionalIndex(scan, "target", 2, 20) == 1);
    // Reserved frame not covered on that layer -> no fallback.
    CHECK(FindProvisionalIndex(scan, "target", 2, 100) == -1);
    // Negative reserved layer disables the fallback entirely.
    CHECK(FindProvisionalIndex(scan, "target", -1, 20) == -1);
}

TEST_CASE("FindProvisionalIndex prefers exact match over the fallback") {
    std::vector<ScannedObject> scan = {
        Obj(2, 10, 40, TextAlias("edited")),           // would satisfy fallback
        Obj(5, 0, 100, TextAlias("real [#target]")),   // exact match
    };
    CHECK(FindProvisionalIndex(scan, "target", 2, 20) == 1);
}

// ---------------------------------------------------------------------------
// DecideResolve - replace vs reserved-insert branch
// ---------------------------------------------------------------------------

TEST_CASE("DecideResolve replaces in place when the placeholder was found") {
    Reservation r;
    r.job_id = "j";
    r.layer = 2;
    r.frame = 10;
    const ProvisionalDecision d = DecideResolve(true, r, 7, 123);
    CHECK(d.next == ProvisionalState::kResolvedReplaced);
    CHECK(d.replace);
    CHECK(d.layer == 7);    // found position, not the reservation
    CHECK(d.frame == 123);
}

TEST_CASE("DecideResolve inserts at the reservation when not found") {
    Reservation r;
    r.job_id = "j";
    r.layer = 2;
    r.frame = 10;
    const ProvisionalDecision d = DecideResolve(false, r, 7, 123);
    CHECK(d.next == ProvisionalState::kResolvedReserved);
    CHECK_FALSE(d.replace);
    CHECK(d.layer == 2);   // reservation position
    CHECK(d.frame == 10);
}

// ---------------------------------------------------------------------------
// DetectOrphans - stale placeholders, foreign objects ignored
// ---------------------------------------------------------------------------

TEST_CASE("DetectOrphans returns provisional objects with no active job") {
    std::vector<ScannedObject> scan = {
        Obj(1, 0, 30, TextAlias("a [#alive]")),
        Obj(2, 40, 90, TextAlias("b [#stale]")),
        Obj(3, 5, 25, "[Object]\r\n[Object.0]\r\neffect.name=foo\r\n"),  // foreign
    };
    std::set<std::string> active = {"alive"};
    const std::vector<Reservation> orphans = DetectOrphans(scan, active);
    REQUIRE(orphans.size() == 1);
    CHECK(orphans[0].job_id == "stale");
    CHECK(orphans[0].object_name == "NzLTX23#stale");
    CHECK(orphans[0].layer == 2);
    CHECK(orphans[0].frame == 40);
    CHECK(orphans[0].length_frames == 50);  // 90 - 40
}

TEST_CASE("DetectOrphans with no active jobs orphans every provisional object") {
    std::vector<ScannedObject> scan = {
        Obj(1, 0, 30, TextAlias("a [#one]")),
        Obj(2, 40, 90, TextAlias("b [#two]")),
    };
    const std::vector<Reservation> orphans = DetectOrphans(scan, {});
    REQUIRE(orphans.size() == 2);
    CHECK(orphans[0].job_id == "one");
    CHECK(orphans[1].job_id == "two");
}

TEST_CASE("DetectOrphans ignores foreign objects even when no jobs are active") {
    std::vector<ScannedObject> scan = {
        Obj(3, 5, 25, "[Object]\r\n[Object.0]\r\neffect.name=foo\r\nbar=1\r\n"),
    };
    CHECK(DetectOrphans(scan, {}).empty());
}

TEST_CASE("DetectOrphans extracts ids from a built provisional alias") {
    const ProvisionalTextAlias built =
        BuildProvisionalTextAlias("prompt", "round-trip-id");
    std::vector<ScannedObject> scan = {Obj(4, 0, 12, built.alias)};
    const std::vector<Reservation> orphans = DetectOrphans(scan, {});
    REQUIRE(orphans.size() == 1);
    CHECK(orphans[0].job_id == "round-trip-id");
    // A still-active job of that id is not an orphan.
    CHECK(DetectOrphans(scan, {"round-trip-id"}).empty());
}
