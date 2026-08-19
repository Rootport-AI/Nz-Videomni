// test_strconv.cpp - unit tests for the UTF-8 <-> UTF-16 conversion helpers.
#include "doctest.h"

#include "strconv.h"

using nzvideomni::Utf8ToWide;
using nzvideomni::WideToUtf8;

TEST_CASE("empty strings round-trip to empty") {
    CHECK(Utf8ToWide(std::string()).empty());
    CHECK(WideToUtf8(std::wstring()).empty());
}

TEST_CASE("ascii round-trips unchanged") {
    const std::string ascii = "Nz-Videomni hello world 12345";
    const std::wstring wide = Utf8ToWide(ascii);
    CHECK(wide == std::wstring(L"Nz-Videomni hello world 12345"));
    CHECK(WideToUtf8(wide) == ascii);
}

TEST_CASE("utf-8 japanese round-trips") {
    // "こんにちは" encoded as explicit UTF-8 bytes (no source-encoding reliance).
    const std::string utf8 =
        "\xE3\x81\x93\xE3\x82\x93\xE3\x81\xAB\xE3\x81\xA1\xE3\x81\xAF";
    const std::wstring wide = Utf8ToWide(utf8);
    // 5 code points, each in the BMP -> 5 UTF-16 code units.
    CHECK(wide.size() == 5);
    CHECK(wide[0] == static_cast<wchar_t>(0x3053)); // U+3053 KO
    CHECK(wide[4] == static_cast<wchar_t>(0x306F)); // U+306F HA
    CHECK(WideToUtf8(wide) == utf8);
}

TEST_CASE("utf-8 emoji (surrogate pair) round-trips") {
    // U+1F600 GRINNING FACE -> UTF-8 F0 9F 98 80, UTF-16 surrogate pair.
    const std::string utf8 = "\xF0\x9F\x98\x80";
    const std::wstring wide = Utf8ToWide(utf8);
    CHECK(wide.size() == 2); // high + low surrogate
    CHECK(wide[0] == static_cast<wchar_t>(0xD83D));
    CHECK(wide[1] == static_cast<wchar_t>(0xDE00));
    CHECK(WideToUtf8(wide) == utf8);
}

TEST_CASE("embedded NUL is preserved") {
    std::string utf8("a\0b", 3);
    const std::wstring wide = Utf8ToWide(utf8);
    CHECK(wide.size() == 3);
    CHECK(wide[1] == L'\0');
    CHECK(WideToUtf8(wide) == utf8);
}
