// test_alias_util.cpp - unit tests for the AviUtl2 alias string builders.
// The Japanese effect / item names are UTF-8 byte escapes (ASCII-only source),
// matching alias_util.cpp and the aviutl2_sdk WindowClient.cpp sample.
#include "doctest.h"

#include <string>
#include <vector>

#include "alias_util.h"

using namespace nzvideomni;

namespace {

const char* kTxt = "\xe3\x83\x86\xe3\x82\xad\xe3\x82\xb9\xe3\x83\x88";  // text effect + item
const char* kStd = "\xe6\xa8\x99\xe6\xba\x96\xe6\x8f\x8f\xe7\x94\xbb";  // standard draw
const char* kSize = "\xe3\x82\xb5\xe3\x82\xa4\xe3\x82\xba";            // size item
const char* kAlign = "\xe6\x96\x87\xe5\xad\x97\xe6\x8f\x83\xe3\x81\x88";  // text-alignment item
const char* kAlignCenterMid =
    "\xe4\xb8\xad\xe5\xa4\xae\xe6\x8f\x83\xe3\x81\x88\x5b\xe4\xb8\xad\x5d";  // center [mid]
const char* kVidFile =
    "\xe5\x8b\x95\xe7\x94\xbb\xe3\x83\x95\xe3\x82\xa1\xe3\x82\xa4\xe3\x83\xab";  // video file effect
const char* kFileJp = "\xe3\x83\x95\xe3\x82\xa1\xe3\x82\xa4\xe3\x83\xab";      // file item
const char* kPlayJp = "\xe5\x86\x8d\xe7\x94\x9f\xe4\xbd\x8d\xe7\xbd\xae";      // playback item
// Section 3-140 (BuildMediaObjectAlias). A THIRD copy of these names on purpose:
// the test must fail if alias_util.cpp's bytes ever drift, so it cannot share
// them. The last four are the items the builder deliberately does NOT write.
const char* kVidPlay = "\xe6\x98\xa0\xe5\x83\x8f\xe5\x86\x8d\xe7\x94\x9f";  // video playback
const char* kHasAudio = "\xe9\x9f\xb3\xe5\xa3\xb0\xe4\xbb\x98\xe3\x81\x8d";  // audio present
const char* kPlayRange = "\xe5\x86\x8d\xe7\x94\x9f\xe7\xaf\x84\xe5\x9b\xb2";  // playback range
const char* kPlaySpeed =
    "\xe5\x86\x8d\xe7\x94\x9f\xe9\x80\x9f\xe5\xba\xa6";  // playback speed (omitted)
const char* kTrack = "\xe3\x83\x88\xe3\x83\xa9\xe3\x83\x83\xe3\x82\xaf";  // track (omitted)
const char* kLoopPlay =
    "\xe3\x83\xab\xe3\x83\xbc\xe3\x83\x97\xe5\x86\x8d\xe7\x94\x9f";  // loop playback (omitted)
// Section 3-54 (object tracking). Independent copies again, for the same
// reason: the test has to fail if alias_util.cpp's bytes ever drift.
const char* kPartialFilter =
    "\xe9\x83\xa8\xe5\x88\x86\xe3\x83\x95\xe3\x82\xa3\xe3\x83\xab\xe3\x82\xbf";  // partial filter
const char* kAspect = "\xe7\xb8\xa6\xe6\xa8\xaa\xe6\xaf\x94";  // aspect ratio item

bool Contains(const std::string& hay, const std::string& needle) {
    return hay.find(needle) != std::string::npos;
}

}  // namespace

// ---------------------------------------------------------------------------
// StripUtf8Bom
// ---------------------------------------------------------------------------

TEST_CASE("StripUtf8Bom removes a leading BOM only") {
    CHECK(StripUtf8Bom("\xEF\xBB\xBF[Object]") == "[Object]");
    CHECK(StripUtf8Bom("[Object]") == "[Object]");
    CHECK(StripUtf8Bom("") == "");
    // A BOM only at the start is stripped; interior bytes are untouched.
    CHECK(StripUtf8Bom("x\xEF\xBB\xBF") == "x\xEF\xBB\xBF");
}

// ---------------------------------------------------------------------------
// NormalizeAliasObjectFrameHeader (the most important helper)
// ---------------------------------------------------------------------------

TEST_CASE("frame line is rewritten in place") {
    const std::string in = "[Object]\nframe=0,100\n[Object.0]\neffect.name=x\n";
    const std::string out = NormalizeAliasObjectFrameHeader(in, 200);
    CHECK(Contains(out, "frame=0,199"));  // 200 frames, inclusive end
    CHECK_FALSE(Contains(out, "frame=0,100"));  // the input value is gone
}

TEST_CASE("frame line is inserted right after [Object] when missing") {
    const std::string in = "[Object]\n[Object.0]\neffect.name=x\n";
    const std::string out = NormalizeAliasObjectFrameHeader(in, 50);
    CHECK(Contains(out, "[Object]\nframe=0,49\n[Object.0]"));  // 50 frames
}

TEST_CASE("a frame line inside another section is not touched") {
    const std::string in = "[Object]\n[Object.0]\neffect.name=x\nframe=99,99\n";
    const std::string out = NormalizeAliasObjectFrameHeader(in, 10);
    CHECK(Contains(out, "frame=0,9"));     // inserted into [Object] (10 frames)
    CHECK(Contains(out, "frame=99,99"));   // [Object.0]'s own line survives
}

TEST_CASE("CRLF line endings are preserved and BOM stripped") {
    const std::string in = "\xEF\xBB\xBF[Object]\r\nframe=0,1\r\n[Object.0]\r\n";
    const std::string out = NormalizeAliasObjectFrameHeader(in, 7);
    CHECK(Contains(out, "frame=0,6\r\n"));  // 7 frames
    CHECK_FALSE(Contains(out, "\xEF\xBB\xBF"));
}

TEST_CASE("no [Object] section leaves the alias unchanged") {
    const std::string in = "[Object.0]\neffect.name=x\n";
    CHECK(NormalizeAliasObjectFrameHeader(in, 5) == in);
}

TEST_CASE("the pinned header matches what the host itself serializes") {
    // Regression pin for the off-by-one fixed on 2026-09-04. The header's second
    // value is an INCLUSIVE end frame, so a 121-frame clip must read
    // "frame=0,120" - byte-identical to what AviUtl2 writes for the SAME 121-frame
    // material dropped onto the timeline (real-device capture
    // data\Alias\R4_d_and_d.object; Docs\SDK_REFERENCE.md section 16 (h)).
    // Writing "frame=0,121" - the old behaviour, captured in R4_insert.object -
    // produced a 122-frame object, one frame longer than the video.
    const std::string in = "[Object]\n[Object.0]\neffect.name=x\n";
    const std::string out = NormalizeAliasObjectFrameHeader(in, 121);
    CHECK(Contains(out, "frame=0,120"));
    CHECK_FALSE(Contains(out, "frame=0,121"));
}

TEST_CASE("a one-frame length pins the degenerate inclusive range") {
    // Boundary: 1 frame occupies frame 0 only, so start == end.
    const std::string in = "[Object]\n[Object.0]\neffect.name=x\n";
    CHECK(Contains(NormalizeAliasObjectFrameHeader(in, 1), "frame=0,0"));
}

// ---------------------------------------------------------------------------
// PatchAliasReplaceVideoFilePath
// ---------------------------------------------------------------------------

static std::string VideoAlias(const std::string& file_key,
                              const std::string& play_key) {
    std::string a = "[Object]\n[Object.0]\neffect.name=";
    a += kVidFile;
    a += "\n";
    a += file_key + "=C:\\old\\clip.mp4\n";
    a += play_key + "=12\n";
    return a;
}

TEST_CASE("video path replaced and playback removed for each candidate key") {
    struct Case { const char* file_key; const char* play_key; };
    const Case cases[] = {
        {kFileJp, kPlayJp}, {"File", "Playback"}, {"file", "Playback"},
        {"path", "Playback"}, {"Path", "Playback"},
    };
    for (const Case& c : cases) {
        const std::string in = VideoAlias(c.file_key, c.play_key);
        const std::string out = PatchAliasReplaceVideoFilePath(in, "D:\\new\\v.mp4");
        CHECK(Contains(out, std::string(c.file_key) + "=D:\\new\\v.mp4"));
        CHECK_FALSE(Contains(out, "C:\\old\\clip.mp4"));
        CHECK_FALSE(Contains(out, std::string(c.play_key) + "="));
    }
}

TEST_CASE("video-file effect is found past an earlier effect") {
    std::string in = "[Object]\n[Object.0]\neffect.name=";
    in += kStd;
    in += "\n[Object.1]\neffect.name=";
    in += kVidFile;
    in += "\nFile=old.mp4\n";
    const std::string out = PatchAliasReplaceVideoFilePath(in, "new.mp4");
    CHECK(Contains(out, "File=new.mp4"));
    CHECK_FALSE(Contains(out, "old.mp4"));
}

TEST_CASE("no video-file effect leaves the alias unchanged") {
    const std::string in = "[Object]\n[Object.0]\neffect.name=x\nFile=old.mp4\n";
    CHECK(PatchAliasReplaceVideoFilePath(in, "new.mp4") == in);
}

// ---------------------------------------------------------------------------
// BuildProvisionalTextAlias
// ---------------------------------------------------------------------------

TEST_CASE("provisional text alias has the expected effect structure and job id") {
    ProvisionalTextAlias p = BuildProvisionalTextAlias("Hello there", "job-123");
    CHECK(Contains(p.alias, std::string("effect.name=") + kTxt));
    CHECK(Contains(p.alias, std::string("effect.name=") + kStd));
    CHECK(Contains(p.alias, std::string(kSize) + "=34"));
    CHECK(Contains(p.alias, "[#job-123]"));       // job id embedded in the text
    CHECK(p.object_name == "NzVideomni#job-123");    // and in the search name
}

TEST_CASE("provisional text alias centers the label with the alignment item") {
    ProvisionalTextAlias p = BuildProvisionalTextAlias("Hello there", "job-c");
    CHECK(Contains(p.alias, std::string(kAlign) + "=" + kAlignCenterMid));
    // The alignment item belongs to the text effect (readable via ParseAliasItemValue).
    std::string value;
    CHECK(ParseAliasItemValue(p.alias, kTxt, kAlign, &value));
    CHECK(value == kAlignCenterMid);
}

TEST_CASE("provisional alias round-trips through the frame normalizer") {
    ProvisionalTextAlias p = BuildProvisionalTextAlias("x", "j1");
    const std::string pinned = NormalizeAliasObjectFrameHeader(p.alias, 42);
    CHECK(Contains(pinned, "frame=0,41"));  // 42 frames, inclusive end
}

TEST_CASE("provisional text alias honors an explicit textPrefix (spec 5-5)") {
    // The default ASCII fallback prefix, mirrored from alias_util.cpp.
    const std::string kGeneratingPrefix = "Generating: ";

    // Default (no/empty prefix) still uses the generating prefix and keeps the
    // [#job] marker + object_name double tag.
    const ProvisionalTextAlias def = BuildProvisionalTextAlias("body", "jP");
    std::string value;
    REQUIRE(ParseAliasItemValue(def.alias, kTxt, kTxt, &value));
    CHECK(value.rfind(kGeneratingPrefix, 0) == 0);  // begins with the default prefix
    CHECK(Contains(value, "[#jP]"));
    CHECK(def.object_name == "NzVideomni#jP");
    // An empty explicit prefix behaves identically to omitting it.
    CHECK(BuildProvisionalTextAlias("body", "jP", "").alias == def.alias);

    // An explicit (ASCII) prefix replaces the default but the marker/name survive.
    const ProvisionalTextAlias pre = BuildProvisionalTextAlias("body", "jP", "RSV: ");
    REQUIRE(ParseAliasItemValue(pre.alias, kTxt, kTxt, &value));
    CHECK(value.rfind("RSV: body", 0) == 0);
    CHECK_FALSE(Contains(value, kGeneratingPrefix));
    CHECK(Contains(value, "[#jP]"));
    CHECK(pre.object_name == "NzVideomni#jP");
}

// ---------------------------------------------------------------------------
// BuildMediaObjectAlias (section 3-140: the "audio present" flag)
// ---------------------------------------------------------------------------

// The exact "playback position" value the builder must emit for a given
// 3-decimal duration string: start, source duration, range keyword, flag.
static std::string ExpectedPlayback(const std::string& seconds_3dp) {
    return std::string("0.000,") + seconds_3dp + "," + kPlayRange + ",0";
}

TEST_CASE("media object alias has the two-effect video structure and no frame line") {
    const std::string a = BuildMediaObjectAlias("C:\\v\\clip.mp4", 10.0416666, true);
    REQUIRE_FALSE(a.empty());
    CHECK(Contains(a, "[Object]"));
    CHECK(Contains(a, std::string("effect.name=") + kVidFile));
    CHECK(Contains(a, "[Object.1]"));
    CHECK(Contains(a, std::string("effect.name=") + kVidPlay));
    // No frame= line: the length is pinned by NormalizeAliasObjectFrameHeader
    // (as the inclusive "frame=0,<length-1>"), because an alias-local frame range
    // would OVERRIDE the requested length.
    CHECK_FALSE(Contains(a, "frame="));
    CHECK(Contains(a, "\r\n"));  // CRLF, like BuildProvisionalTextAlias
}

TEST_CASE("media object alias items belong to the video-file effect section") {
    const std::string with_audio = BuildMediaObjectAlias("C:\\v\\a.mp4", 10.0416666, true);
    const std::string no_audio = BuildMediaObjectAlias("C:\\v\\a.mp4", 10.0416666, false);
    std::string value;

    // The whole point of 3-140: "audio present" must be readable as an item of
    // the [Object.0] video-file effect, and must follow has_audio.
    REQUIRE(ParseAliasItemValue(with_audio, kVidFile, kHasAudio, &value));
    CHECK(value == "1");
    REQUIRE(ParseAliasItemValue(no_audio, kVidFile, kHasAudio, &value));
    CHECK(value == "0");

    // The other two per-material items round-trip out of the same section.
    REQUIRE(ParseAliasItemValue(with_audio, kVidFile, kFileJp, &value));
    CHECK(value == "C:\\v\\a.mp4");
    REQUIRE(ParseAliasItemValue(with_audio, kVidFile, kPlayJp, &value));
    CHECK(value == ExpectedPlayback("10.042"));

    // They are NOT items of the [Object.1] video-playback effect.
    CHECK_FALSE(ParseAliasItemValue(with_audio, kVidPlay, kHasAudio, &value));
}

TEST_CASE("playback position always carries exactly three decimals") {
    struct Case { double seconds; const char* text; };
    const Case cases[] = {
        {10.0416666, "10.042"},  // rounds up
        {10.0, "10.000"},        // integral seconds still get 3 decimals
        {0.5, "0.500"},          // sub-second, trailing zeros kept
        {1234.5678, "1234.568"}, // long clip, rounds up
    };
    for (const Case& c : cases) {
        const std::string a = BuildMediaObjectAlias("C:\\v\\a.mp4", c.seconds, true);
        REQUIRE_FALSE(a.empty());
        std::string value;
        REQUIRE(ParseAliasItemValue(a, kVidFile, kPlayJp, &value));
        CHECK(value == ExpectedPlayback(c.text));
    }
}

TEST_CASE("an '=' inside the path stays part of the value") {
    const std::string a = BuildMediaObjectAlias("C:\\a=b\\v.mp4", 2.0, true);
    REQUIRE_FALSE(a.empty());
    std::string value;
    REQUIRE(ParseAliasItemValue(a, kVidFile, kFileJp, &value));
    CHECK(value == "C:\\a=b\\v.mp4");  // everything after the FIRST '=' is the value
}

TEST_CASE("unbuildable inputs return an empty alias (caller falls back)") {
    CHECK(BuildMediaObjectAlias("", 10.0, true).empty());               // empty path
    CHECK(BuildMediaObjectAlias("C:\\a\nb.mp4", 10.0, true).empty());   // LF in path
    CHECK(BuildMediaObjectAlias("C:\\a\rb.mp4", 10.0, true).empty());   // CR in path
    CHECK(BuildMediaObjectAlias("C:\\v\\a.mp4", 0.0, true).empty());    // still image
    CHECK(BuildMediaObjectAlias("C:\\v\\a.mp4", -1.0, true).empty());   // nonsense duration
}

TEST_CASE("media object alias pins its length through the frame normalizer") {
    // This is the production shape: build, then normalize with the resolved
    // project-frame length before handing it to create_object_from_alias.
    const std::string a = BuildMediaObjectAlias("C:\\v\\a.mp4", 10.0416666, true);
    REQUIRE_FALSE(a.empty());
    const std::string pinned = NormalizeAliasObjectFrameHeader(a, 240);
    CHECK(Contains(pinned, "[Object]\r\nframe=0,239\r\n[Object.0]"));  // 240 frames
    std::string value;  // normalizing must not disturb the items
    REQUIRE(ParseAliasItemValue(pinned, kVidFile, kHasAudio, &value));
    CHECK(value == "1");
}

TEST_CASE("items left at their static defaults are not written") {
    // Design pin: only per-material values are emitted; the host fills the rest
    // in from its own defaults (the aviutl2_sdk sample omits them likewise).
    const std::string a = BuildMediaObjectAlias("C:\\v\\a.mp4", 10.0416666, true);
    REQUIRE_FALSE(a.empty());
    CHECK_FALSE(Contains(a, kPlaySpeed));
    CHECK_FALSE(Contains(a, kTrack));
    CHECK_FALSE(Contains(a, kLoopPlay));
    CHECK_FALSE(Contains(a, "YUV"));
}

// ---------------------------------------------------------------------------
// ParseAliasItemValue
// ---------------------------------------------------------------------------

TEST_CASE("ParseAliasItemValue reads effect items and reports misses") {
    ProvisionalTextAlias p = BuildProvisionalTextAlias("Hello", "job-9");
    std::string value;
    CHECK(ParseAliasItemValue(p.alias, kTxt, kSize, &value));
    CHECK(value == "34");
    CHECK(ParseAliasItemValue(p.alias, kTxt, kTxt, &value));
    CHECK(Contains(value, "[#job-9]"));
    // The standard-draw effect has no size item.
    CHECK_FALSE(ParseAliasItemValue(p.alias, kStd, kSize, &value));
}

// ---------------------------------------------------------------------------
// Object tracking (section 3-54): RectToPartialFilter / PartialFilterToRect /
// ParsePartialFilterValues / FirstEffectName.
// ---------------------------------------------------------------------------

namespace {

// A partial-filter alias in the normal layout: the "[Object]" meta section
// first, then the effect blocks. `x`/`y`/`size`/`aspect` are inserted verbatim
// so a case can feed a keyframed list or deliberate junk. An EMPTY string omits
// that line entirely.
std::string MakePartialFilterAlias(const std::string& x, const std::string& y,
                                   const std::string& size,
                                   const std::string& aspect,
                                   const std::string& eol = "\n") {
    std::string a;
    a += "[Object]";
    a += eol;
    a += "frame=0,120";
    a += eol;
    a += "[Object.0]";
    a += eol;
    a += "effect.name=";
    a += kPartialFilter;
    a += eol;
    if (!x.empty()) {
        a += "X=" + x + eol;
    }
    if (!y.empty()) {
        a += "Y=" + y + eol;
    }
    if (!size.empty()) {
        a += std::string(kSize) + "=" + size + eol;
    }
    if (!aspect.empty()) {
        a += std::string(kAspect) + "=" + aspect + eol;
    }
    a += "[Object.1]";
    a += eol;
    a += "effect.name=";
    a += kStd;
    a += eol;
    return a;
}

// Round-trip helper: rect -> AviUtl2 values -> rect, all four numbers back.
void RoundTrip(double rx, double ry, double rw, double rh, int sw, int sh) {
    const PartialFilterValues v = RectToPartialFilter(rx, ry, rw, rh, sw, sh);
    double ox = 0.0, oy = 0.0, ow = 0.0, oh = 0.0;
    REQUIRE(PartialFilterToRect(v, sw, sh, &ox, &oy, &ow, &oh));
    CHECK(ox == doctest::Approx(rx));
    CHECK(oy == doctest::Approx(ry));
    CHECK(ow == doctest::Approx(rw));
    CHECK(oh == doctest::Approx(rh));
}

}  // namespace

TEST_CASE("a landscape rectangle converts to a negative aspect and back") {
    const PartialFilterValues v = RectToPartialFilter(100, 200, 200, 100, 1920, 1080);
    CHECK(v.x == doctest::Approx(100 + 100 - 960));  // centre 200 -> -760
    CHECK(v.y == doctest::Approx(200 + 50 - 540));   // centre 250 -> -290
    CHECK(v.size == doctest::Approx(200.0));         // the LONGER side
    CHECK(v.aspect == doctest::Approx(-50.0));       // height is half the width
    RoundTrip(100, 200, 200, 100, 1920, 1080);
}

TEST_CASE("a portrait rectangle converts to a positive aspect and back") {
    const PartialFilterValues v = RectToPartialFilter(0, 0, 100, 200, 1920, 1080);
    CHECK(v.size == doctest::Approx(200.0));
    CHECK(v.aspect == doctest::Approx(50.0));
    RoundTrip(0, 0, 100, 200, 1920, 1080);
}

TEST_CASE("a square centred on the screen is the origin with aspect 0") {
    const PartialFilterValues v = RectToPartialFilter(910, 490, 100, 100, 1920, 1080);
    CHECK(v.x == doctest::Approx(0.0));
    CHECK(v.y == doctest::Approx(0.0));
    CHECK(v.size == doctest::Approx(100.0));
    CHECK(v.aspect == doctest::Approx(0.0));
    RoundTrip(910, 490, 100, 100, 1920, 1080);
}

TEST_CASE("a rectangle at the top-left corner is negative on both axes") {
    // AviUtl2's Y grows DOWNWARDS, so the top-left corner is the most negative
    // corner on both axes - the sign convention that would silently flip the
    // whole track if it were wrong.
    const PartialFilterValues v = RectToPartialFilter(0, 0, 100, 100, 1920, 1080);
    CHECK(v.x == doctest::Approx(50 - 960));
    CHECK(v.y == doctest::Approx(50 - 540));
    RoundTrip(0, 0, 100, 100, 1920, 1080);
    // ... and the bottom-right corner is positive on both.
    const PartialFilterValues br =
        RectToPartialFilter(1820, 980, 100, 100, 1920, 1080);
    CHECK(br.x == doctest::Approx(910.0));
    CHECK(br.y == doctest::Approx(490.0));
}

TEST_CASE("an extreme aspect clamps at +/-99.99") {
    // 0.5 x 10000 would be +/-99.995; at +/-100 the short side would vanish.
    const PartialFilterValues wide = RectToPartialFilter(0, 0, 10000, 0.5, 1920, 1080);
    CHECK(wide.aspect == doctest::Approx(-99.99));
    const PartialFilterValues tall = RectToPartialFilter(0, 0, 0.5, 10000, 1920, 1080);
    CHECK(tall.aspect == doctest::Approx(99.99));
    // Just inside the clamp the round trip is still exact.
    RoundTrip(0, 0, 1000, 10, 1920, 1080);
    RoundTrip(0, 0, 10, 1000, 1920, 1080);
}

TEST_CASE("a degenerate rectangle yields aspect 0 rather than a NaN") {
    // The caller rejects these; the guard only keeps a NaN out of the alias.
    const PartialFilterValues v = RectToPartialFilter(10, 20, 0, 50, 1920, 1080);
    CHECK(v.aspect == doctest::Approx(0.0));
    CHECK(v.size == doctest::Approx(50.0));
}

TEST_CASE("PartialFilterToRect rejects a non-positive size") {
    PartialFilterValues v;
    v.x = 1.0;
    v.y = 2.0;
    v.size = 0.0;
    v.aspect = 0.0;
    double rx = -1.0, ry = -1.0, rw = -1.0, rh = -1.0;
    CHECK_FALSE(PartialFilterToRect(v, 1920, 1080, &rx, &ry, &rw, &rh));
    CHECK(rx == -1.0);  // outputs untouched
    v.size = -10.0;
    CHECK_FALSE(PartialFilterToRect(v, 1920, 1080, &rx, &ry, &rw, &rh));
}

TEST_CASE("PartialFilterToRect accepts null output pointers") {
    PartialFilterValues v;
    v.x = 0.0;
    v.y = 0.0;
    v.size = 100.0;
    v.aspect = 0.0;
    double rw = 0.0;
    CHECK(PartialFilterToRect(v, 1920, 1080, nullptr, nullptr, &rw, nullptr));
    CHECK(rw == doctest::Approx(100.0));
}

TEST_CASE("ParsePartialFilterValues reads a plain, keyframe-less box") {
    const std::string a = MakePartialFilterAlias("-760", "-290", "200", "-50");
    PartialFilterValues v;
    REQUIRE(ParsePartialFilterValues(a, &v));
    CHECK(v.x == doctest::Approx(-760.0));
    CHECK(v.y == doctest::Approx(-290.0));
    CHECK(v.size == doctest::Approx(200.0));
    CHECK(v.aspect == doctest::Approx(-50.0));
}

TEST_CASE("ParsePartialFilterValues takes the FIRST token of a keyframed value") {
    // The seed is the box at the object's first frame, so only the head of the
    // comma-separated list matters. The tail shapes below are deliberately
    // different from each other: until the real keyframed format is captured on
    // the device (design section 5.7), the contract is "first token, whatever
    // follows".
    const std::string a =
        MakePartialFilterAlias("-760,120,480", "-290,-100,PLACEHOLDER,0",
                               "200,340", "-50,0,X,0");
    PartialFilterValues v;
    REQUIRE(ParsePartialFilterValues(a, &v));
    CHECK(v.x == doctest::Approx(-760.0));
    CHECK(v.y == doctest::Approx(-290.0));
    CHECK(v.size == doctest::Approx(200.0));
    CHECK(v.aspect == doctest::Approx(-50.0));
}

TEST_CASE("ParsePartialFilterValues needs the size line and nothing else") {
    // A missing X / Y / aspect line means "still on its default", which is 0 -
    // a square box whose aspect ratio the user never touched is the case this
    // exists for. Only the size has no usable default.
    PartialFilterValues v;
    REQUIRE(ParsePartialFilterValues(
        MakePartialFilterAlias("", "-290", "200", "-50"), &v));
    CHECK(v.x == doctest::Approx(0.0));
    CHECK(v.y == doctest::Approx(-290.0));
    REQUIRE(ParsePartialFilterValues(
        MakePartialFilterAlias("-760", "", "200", "-50"), &v));
    CHECK(v.y == doctest::Approx(0.0));
    REQUIRE(ParsePartialFilterValues(
        MakePartialFilterAlias("-760", "-290", "200", ""), &v));
    CHECK(v.aspect == doctest::Approx(0.0));
    CHECK(v.size == doctest::Approx(200.0));
    // All three at once: the bare square box.
    REQUIRE(ParsePartialFilterValues(
        MakePartialFilterAlias("", "", "200", ""), &v));
    CHECK(v.x == doctest::Approx(0.0));
    CHECK(v.y == doctest::Approx(0.0));
    CHECK(v.aspect == doctest::Approx(0.0));
    CHECK(v.size == doctest::Approx(200.0));

    // No size line: there is no box, and the output is left alone.
    v.size = 4242.0;  // sentinel: a failed parse must not touch the output
    CHECK_FALSE(ParsePartialFilterValues(
        MakePartialFilterAlias("-760", "-290", "", "-50"), &v));
    CHECK(v.size == doctest::Approx(4242.0));
}

TEST_CASE("ParsePartialFilterValues ignores a different effect's items") {
    // Same four item names, wrong effect: the X/Y/size/aspect of some other
    // effect must never be mistaken for the partial filter's box.
    std::string a = "[Object]\nframe=0,120\n[Object.0]\neffect.name=";
    a += kStd;
    a += "\nX=1\nY=2\n";
    a += std::string(kSize) + "=3\n";
    a += std::string(kAspect) + "=4\n";
    PartialFilterValues v;
    CHECK_FALSE(ParsePartialFilterValues(a, &v));
}

TEST_CASE("ParsePartialFilterValues rejects a value it cannot fully read") {
    PartialFilterValues v;
    // Trailing junk inside the first token, not a separate token. A line that
    // IS there and does not parse is a malformed alias - it does not fall back
    // to the default the way an absent line does.
    CHECK_FALSE(ParsePartialFilterValues(
        MakePartialFilterAlias("-760px", "-290", "200", "-50"), &v));
    // An empty first token.
    CHECK_FALSE(ParsePartialFilterValues(
        MakePartialFilterAlias("-760", ",-290", "200", "-50"), &v));
    // Same for the aspect ratio, which is otherwise the most defaultable item.
    CHECK_FALSE(ParsePartialFilterValues(
        MakePartialFilterAlias("-760", "-290", "200", "zero"), &v));
}

TEST_CASE("ParsePartialFilterValues ignores value lines ahead of effect.name") {
    // The parser only starts collecting once it has seen the effect.name that
    // says which effect the block belongs to, so anything above that line is
    // not attributed to the partial filter. This case pins that behaviour:
    // the X above effect.name is dropped (X falls back to its default 0) and
    // the one below it is the value that counts.
    std::string a = "[Object]\nframe=0,120\n[Object.0]\nX=777\n";
    a += std::string(kSize) + "=999\n";
    a += "effect.name=";
    a += kPartialFilter;
    a += "\nY=-290\n";
    a += std::string(kSize) + "=200\n";
    a += std::string(kAspect) + "=-50\n";
    PartialFilterValues v;
    REQUIRE(ParsePartialFilterValues(a, &v));
    CHECK(v.x == doctest::Approx(0.0));
    CHECK(v.y == doctest::Approx(-290.0));
    CHECK(v.size == doctest::Approx(200.0));
    CHECK(v.aspect == doctest::Approx(-50.0));
}

TEST_CASE("ParsePartialFilterValues survives a BOM and CRLF line endings") {
    const std::string a =
        "\xEF\xBB\xBF" + MakePartialFilterAlias("10", "20", "30", "0", "\r\n");
    PartialFilterValues v;
    REQUIRE(ParsePartialFilterValues(a, &v));
    CHECK(v.x == doctest::Approx(10.0));
    CHECK(v.size == doctest::Approx(30.0));
}

TEST_CASE("the seed pipeline reads an alias straight back into a rectangle") {
    // The production path: alias -> four values -> top-left-origin rectangle.
    const PartialFilterValues seed =
        RectToPartialFilter(640, 360, 320, 180, 1920, 1080);
    const std::string a = MakePartialFilterAlias(
        std::to_string(seed.x), std::to_string(seed.y),
        std::to_string(seed.size), std::to_string(seed.aspect));
    PartialFilterValues read;
    REQUIRE(ParsePartialFilterValues(a, &read));
    double rx = 0.0, ry = 0.0, rw = 0.0, rh = 0.0;
    REQUIRE(PartialFilterToRect(read, 1920, 1080, &rx, &ry, &rw, &rh));
    CHECK(rx == doctest::Approx(640.0).epsilon(0.001));
    CHECK(ry == doctest::Approx(360.0).epsilon(0.001));
    CHECK(rw == doctest::Approx(320.0).epsilon(0.001));
    CHECK(rh == doctest::Approx(180.0).epsilon(0.001));
}

TEST_CASE("FirstEffectName returns the first effect block's name") {
    const std::string a = MakePartialFilterAlias("1", "2", "3", "0");
    CHECK(FirstEffectName(a) == kPartialFilter);
}

TEST_CASE("FirstEffectName skips the [Object] meta section") {
    // Pathological but decisive: a meta section carrying an effect.name-shaped
    // line must not be mistaken for effect 0.
    std::string a = "[Object]\neffect.name=NOT_AN_EFFECT\nframe=0,10\n[Object.0]\n";
    a += "effect.name=";
    a += kPartialFilter;
    a += "\n";
    CHECK(FirstEffectName(a) == kPartialFilter);
}

TEST_CASE("FirstEffectName handles a BOM, CRLF and surrounding spaces") {
    std::string a = "\xEF\xBB\xBF[Object]\r\nframe=0,10\r\n[Object.0]\r\n";
    a += "effect.name=  ";
    a += kPartialFilter;
    a += "  \r\n";
    CHECK(FirstEffectName(a) == kPartialFilter);
}

TEST_CASE("FirstEffectName returns empty when there is nothing to name") {
    CHECK(FirstEffectName("") == "");
    CHECK(FirstEffectName("[Object]\nframe=0,10\n") == "");  // no effect section
    // A first effect block with no effect.name: the answer is "unknown", not
    // the name of the SECOND block.
    std::string a = "[Object.0]\nX=1\n[Object.1]\neffect.name=";
    a += kStd;
    a += "\n";
    CHECK(FirstEffectName(a) == "");
}

TEST_CASE("FirstEffectName names whatever effect actually comes first") {
    // Not every alias starts with a partial filter; the caller compares the
    // returned name itself, so a non-matching name must come back verbatim.
    const std::string a = BuildProvisionalTextAlias("Hello", "job-1").alias;
    CHECK(FirstEffectName(a) == kTxt);
}

// ---------------------------------------------------------------------------
// PatchAliasPartialFilterKeyframes (section 3-54). Every case below is built on
// the two .object files the owner saved from AviUtl2 v2.0.54 on 2026-09-11 -
// one partial filter left at its defaults, one carrying two midpoints - which
// are the only primary source for the keyframed format. The expected outputs
// are spelled out BYTE FOR BYTE (same helper, same CRLF) so an untouched line
// that drifts fails the case just as loudly as a wrong value.
// ---------------------------------------------------------------------------

namespace {

// Item names the capture carries that no other case needs. Independent copies
// of the bytes, like the constants at the top of this file: the test has to
// fail if alias_util.cpp's own escapes ever drift.
const char* kRotate = "\xe5\x9b\x9e\xe8\xbb\xa2";                  // rotation
const char* kBlur = "\xe3\x81\xbc\xe3\x81\x8b\xe3\x81\x97";        // blur
const char* kMaskKind =
    "\xe3\x83\x9e\xe3\x82\xb9\xe3\x82\xaf\xe3\x81\xae\xe7\xa8\xae\xe9\xa1\x9e";  // mask kind
const char* kCircle = "\xe5\x86\x86";                              // circle
const char* kMatchScene =
    "\xe3\x82\xb7\xe3\x83\xbc\xe3\x83\xb3\xe3\x81\xae\xe9\x95\xb7\xe3\x81\x95"
    "\xe3\x82\x92\xe5\x90\x88\xe3\x82\x8f\xe3\x81\x9b\xe3\x82\x8b";  // match scene length
const char* kInvertMask =
    "\xe3\x83\x9e\xe3\x82\xb9\xe3\x82\xaf\xe3\x81\xae\xe5\x8f\x8d\xe8\xbb\xa2";  // invert mask
const char* kMoveLinear =
    "\xe7\x9b\xb4\xe7\xb7\x9a\xe7\xa7\xbb\xe5\x8b\x95";            // linear move
const char* kMosaic = "\xe3\x83\xa2\xe3\x82\xb6\xe3\x82\xa4\xe3\x82\xaf";  // mosaic

// The captured partial-filter object, line for line, with the values that vary
// substituted. Called with the capture's own strings it reproduces the file;
// called with the expected strings it spells out the expected output.
std::string CapturedPartialFilter(const std::string& frame, const std::string& x,
                                  const std::string& y, const std::string& size,
                                  const std::string& aspect,
                                  const std::string& rotate = "0.00",
                                  const std::string& eol = "\r\n") {
    std::string a;
    a += "[Object]";
    a += eol;
    a += "frame=" + frame;
    a += eol;
    a += "[Object.0]";
    a += eol;
    a += "effect.name=";
    a += kPartialFilter;
    a += eol;
    a += "X=" + x;
    a += eol;
    a += "Y=" + y;
    a += eol;
    a += "Group=1";
    a += eol;
    a += std::string(kRotate) + "=" + rotate;
    a += eol;
    a += std::string(kSize) + "=" + size;
    a += eol;
    a += std::string(kAspect) + "=" + aspect;
    a += eol;
    a += std::string(kBlur) + "=0";
    a += eol;
    a += std::string(kMaskKind) + "=" + kCircle;
    a += eol;
    a += std::string(kMatchScene) + "=0";
    a += eol;
    a += std::string(kInvertMask) + "=0";
    a += eol;
    return a;
}

// "<values>,<linear move>,0" - the capture's keyframed value line.
std::string Kf(const std::string& values) {
    return values + "," + kMoveLinear + ",0";
}

// The three boxes case 1 walks through, in scene coordinates of 1920x1080:
//   (860,470,200x140) -> X=0  Y=0  size=200 aspect=-30.00
//   (900,500,200x140) -> X=40 Y=30 size=200 aspect=-30.00
//   (960,540,100x100) -> X=50 Y=50 size=100 aspect=  0.00
TrackKeyframe BoxA(int at) { return TrackKeyframe{at, 860.0, 470.0, 200.0, 140.0}; }
TrackKeyframe BoxB(int at) { return TrackKeyframe{at, 900.0, 500.0, 200.0, 140.0}; }
TrackKeyframe BoxC(int at) { return TrackKeyframe{at, 960.0, 540.0, 100.0, 100.0}; }

}  // namespace

TEST_CASE("PatchAliasPartialFilterKeyframes keyframes the captured defaults object") {
    // Case 1: the "dropped in and left alone" capture, three tracked boxes,
    // an 11-frame object. Boundaries become the keyframe offsets, the four
    // value lines take the "<values>,<linear move>,0" form, and every other
    // line - Group, rotation, blur, mask kind, the CRLF endings - is untouched.
    const std::string in = CapturedPartialFilter("23,33", "0", "0", "100", "0.00");
    std::vector<TrackKeyframe> kfs;
    kfs.push_back(BoxA(0));
    kfs.push_back(BoxB(5));
    kfs.push_back(BoxC(10));
    const std::string want =
        CapturedPartialFilter("0,5,10", Kf("0,40,50"), Kf("0,30,50"),
                              Kf("200,200,100"), Kf("-30.00,-30.00,0.00"));
    CHECK(PatchAliasPartialFilterKeyframes(in, kfs, 1920, 1080, 11) == want);
}

TEST_CASE("PatchAliasPartialFilterKeyframes rewrites the captured midpoints object") {
    // Case 2: the capture with two midpoints (four boundaries) tracked again
    // with only two keyframes - the boundary list SHRINKS to two, and the four
    // value lines shrink with it. Everything else still matches byte for byte.
    const std::string in = CapturedPartialFilter(
        "24,54,80,119", Kf("76,89,89,89"), Kf("-169,-153,-153,-153"),
        Kf("77,231,207,207"), "0.00");
    std::vector<TrackKeyframe> kfs;
    kfs.push_back(BoxA(0));
    kfs.push_back(BoxB(95));
    const std::string want = CapturedPartialFilter(
        "0,95", Kf("0,40"), Kf("0,30"), Kf("200,200"), Kf("-30.00,-30.00"));
    CHECK(PatchAliasPartialFilterKeyframes(in, kfs, 1920, 1080, 96) == want);
}

TEST_CASE("PatchAliasPartialFilterKeyframes holds the last box to the object's end") {
    // Case 3: a stopped run - the last keyframe is at 4 but the object runs to
    // frame 10. An extra boundary at 10 REPEATS the last value, so the box
    // stands still from 4 to the end instead of the object losing its tail.
    const std::string in = CapturedPartialFilter("23,33", "0", "0", "100", "0.00");
    std::vector<TrackKeyframe> kfs;
    kfs.push_back(BoxA(0));
    kfs.push_back(BoxB(4));
    const std::string want = CapturedPartialFilter(
        "0,4,10", Kf("0,40,40"), Kf("0,30,30"), Kf("200,200,200"),
        Kf("-30.00,-30.00,-30.00"));
    CHECK(PatchAliasPartialFilterKeyframes(in, kfs, 1920, 1080, 11) == want);
}

TEST_CASE("PatchAliasPartialFilterKeyframes writes a single keyframe as a constant") {
    // Case 4: one keyframe is a box that never moves, so the values are SINGLE
    // - no move method, no trailing 0 - exactly the shape the defaults capture
    // has, and the frame line is the plain start/end pair.
    const std::string in = CapturedPartialFilter("23,33", "0", "0", "100", "0.00");
    std::vector<TrackKeyframe> kfs;
    kfs.push_back(BoxA(0));
    const std::string want =
        CapturedPartialFilter("0,10", "0", "0", "200", "-30.00");
    CHECK(PatchAliasPartialFilterKeyframes(in, kfs, 1920, 1080, 11) == want);
}

TEST_CASE("PatchAliasPartialFilterKeyframes collapses another item's keyframes") {
    // Case 5: the boundary count is changing, so an item the user had keyframed
    // himself - here the rotation - would be left with a list that no longer
    // matches. It is frozen at its FIRST value instead.
    const std::string in = CapturedPartialFilter("23,33,60,90", "0", "0", "100",
                                                 "0.00", Kf("1,2,3,4"));
    std::vector<TrackKeyframe> kfs;
    kfs.push_back(BoxA(0));
    const std::string want =
        CapturedPartialFilter("0,10", "0", "0", "200", "-30.00", "1");
    CHECK(PatchAliasPartialFilterKeyframes(in, kfs, 1920, 1080, 11) == want);
}

TEST_CASE("PatchAliasPartialFilterKeyframes leaves a following effect block alone") {
    // Case 6: a mosaic effect after the partial filter has a "size" item of its
    // own. Only the partial filter's is rewritten - the two must not be
    // confused - and the whole second block survives byte for byte.
    std::string mosaic;
    mosaic += "[Object.1]\r\n";
    mosaic += "effect.name=" + std::string(kMosaic) + "\r\n";
    mosaic += std::string(kSize) + "=12\r\n";
    const std::string in =
        CapturedPartialFilter("23,33", "0", "0", "100", "0.00") + mosaic;
    std::vector<TrackKeyframe> kfs;
    kfs.push_back(BoxA(0));
    kfs.push_back(BoxC(10));
    const std::string want =
        CapturedPartialFilter("0,10", Kf("0,50"), Kf("0,50"), Kf("200,100"),
                              Kf("-30.00,0.00")) +
        mosaic;
    CHECK(PatchAliasPartialFilterKeyframes(in, kfs, 1920, 1080, 11) == want);
}

TEST_CASE("PatchAliasPartialFilterKeyframes returns empty when there is nothing to write") {
    // Case 7 and the rest of the "leave the timeline alone" answers: the caller
    // has exactly ONE result to test against.
    const std::string ok = CapturedPartialFilter("23,33", "0", "0", "100", "0.00");
    std::vector<TrackKeyframe> kfs;
    kfs.push_back(BoxA(0));
    kfs.push_back(BoxC(10));
    // No partial-filter block at all.
    std::string other = "[Object]\r\nframe=0,10\r\n[Object.0]\r\neffect.name=";
    other += kStd;
    other += "\r\n";
    CHECK(PatchAliasPartialFilterKeyframes(other, kfs, 1920, 1080, 11) == "");
    // No "[Object]" meta section to own the frame line.
    std::string headless = "[Object.0]\r\neffect.name=";
    headless += kPartialFilter;
    headless += "\r\nX=0\r\n";
    CHECK(PatchAliasPartialFilterKeyframes(headless, kfs, 1920, 1080, 11) == "");
    CHECK(PatchAliasPartialFilterKeyframes("", kfs, 1920, 1080, 11) == "");
    CHECK(PatchAliasPartialFilterKeyframes(ok, {}, 1920, 1080, 11) == "");
    CHECK(PatchAliasPartialFilterKeyframes(ok, kfs, 1920, 1080, 0) == "");
    CHECK(PatchAliasPartialFilterKeyframes(ok, kfs, 0, 1080, 11) == "");
    CHECK(PatchAliasPartialFilterKeyframes(ok, kfs, 1920, 0, 11) == "");
}

TEST_CASE("PatchAliasPartialFilterKeyframes keeps a BOM") {
    // Case 8: an alias that arrived with a BOM goes back out with it.
    const std::string in =
        "\xEF\xBB\xBF" + CapturedPartialFilter("23,33", "0", "0", "100", "0.00");
    std::vector<TrackKeyframe> kfs;
    kfs.push_back(BoxA(0));
    const std::string want =
        "\xEF\xBB\xBF" + CapturedPartialFilter("0,10", "0", "0", "200", "-30.00");
    CHECK(PatchAliasPartialFilterKeyframes(in, kfs, 1920, 1080, 11) == want);
}

TEST_CASE("PatchAliasPartialFilterKeyframes clamps the aspect to two decimals") {
    // Case 9: a box that is all but flat would want +/-100, where the short
    // side collapses to nothing. RectToPartialFilter clamps at +/-99.99 and the
    // line carries two decimals, the way the capture writes the aspect.
    const std::string in = CapturedPartialFilter("23,33", "0", "0", "100", "0.00");
    std::vector<TrackKeyframe> kfs;
    kfs.push_back(TrackKeyframe{0, 0.0, 0.0, 200.0, 0.001});   // landscape
    kfs.push_back(TrackKeyframe{10, 0.0, 0.0, 0.001, 200.0});  // portrait
    const std::string want = CapturedPartialFilter(
        "0,10", Kf("-860,-960"), Kf("-540,-440"), Kf("200,200"),
        Kf("-99.99,99.99"));
    CHECK(PatchAliasPartialFilterKeyframes(in, kfs, 1920, 1080, 11) == want);
}

TEST_CASE("PatchAliasPartialFilterKeyframes rounds X, Y and size to whole numbers") {
    // Case 10: the capture has no decimals on X / Y / size, so they are rounded
    // half-away-from-zero. -479.5 -> -480 and -59.5 -> -60 pin the direction.
    const std::string in = CapturedPartialFilter("23,33", "0", "0", "100", "0.00");
    std::vector<TrackKeyframe> kfs;
    kfs.push_back(TrackKeyframe{0, 100.2, 200.7, 50.4, 50.4});   // -834.6, -314.1, 50.4
    kfs.push_back(TrackKeyframe{10, 0.0, 0.0, 961.0, 961.0});    // -479.5,  -59.5, 961
    const std::string want =
        CapturedPartialFilter("0,10", Kf("-835,-480"), Kf("-314,-60"),
                              Kf("50,961"), Kf("0.00,0.00"));
    CHECK(PatchAliasPartialFilterKeyframes(in, kfs, 1920, 1080, 11) == want);
}
