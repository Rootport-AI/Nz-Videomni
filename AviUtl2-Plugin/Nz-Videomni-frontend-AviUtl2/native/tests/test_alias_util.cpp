// test_alias_util.cpp - unit tests for the AviUtl2 alias string builders.
// The Japanese effect / item names are UTF-8 byte escapes (ASCII-only source),
// matching alias_util.cpp and the aviutl2_sdk WindowClient.cpp sample.
#include "doctest.h"

#include <string>

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
    CHECK(Contains(out, "frame=0,200"));
    CHECK_FALSE(Contains(out, "frame=0,100"));
}

TEST_CASE("frame line is inserted right after [Object] when missing") {
    const std::string in = "[Object]\n[Object.0]\neffect.name=x\n";
    const std::string out = NormalizeAliasObjectFrameHeader(in, 50);
    CHECK(Contains(out, "[Object]\nframe=0,50\n[Object.0]"));
}

TEST_CASE("a frame line inside another section is not touched") {
    const std::string in = "[Object]\n[Object.0]\neffect.name=x\nframe=99,99\n";
    const std::string out = NormalizeAliasObjectFrameHeader(in, 10);
    CHECK(Contains(out, "frame=0,10"));    // inserted into [Object]
    CHECK(Contains(out, "frame=99,99"));   // [Object.0]'s own line survives
}

TEST_CASE("CRLF line endings are preserved and BOM stripped") {
    const std::string in = "\xEF\xBB\xBF[Object]\r\nframe=0,1\r\n[Object.0]\r\n";
    const std::string out = NormalizeAliasObjectFrameHeader(in, 7);
    CHECK(Contains(out, "frame=0,7\r\n"));
    CHECK_FALSE(Contains(out, "\xEF\xBB\xBF"));
}

TEST_CASE("no [Object] section leaves the alias unchanged") {
    const std::string in = "[Object.0]\neffect.name=x\n";
    CHECK(NormalizeAliasObjectFrameHeader(in, 5) == in);
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
    CHECK(Contains(pinned, "frame=0,42"));
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
