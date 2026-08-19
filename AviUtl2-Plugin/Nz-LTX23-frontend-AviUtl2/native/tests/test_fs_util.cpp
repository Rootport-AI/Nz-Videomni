// test_fs_util.cpp - unit tests for the pure path/numbering/BOM helpers.
#include "doctest.h"

#include <set>
#include <string>

#include "fs_util.h"

using namespace nzltx;

// ---------------------------------------------------------------------------
// ExtensionLower / MatchesAnyExtension
// ---------------------------------------------------------------------------

TEST_CASE("ExtensionLower lowercases and handles no-extension cases") {
    CHECK(ExtensionLower("clip.WAV") == ".wav");
    CHECK(ExtensionLower("clip.mp4") == ".mp4");
    CHECK(ExtensionLower("archive.tar.gz") == ".gz");
    CHECK(ExtensionLower("README") == "");
    CHECK(ExtensionLower("") == "");
    // A leading-dot-only name (dotfile) is treated as extensionless.
    CHECK(ExtensionLower(".gitignore") == "");
}

TEST_CASE("MatchesAnyExtension is case-insensitive and honors an empty filter") {
    CHECK(MatchesAnyExtension("a.WAV", {".wav", ".csv"}));
    CHECK(MatchesAnyExtension("a.wav", {".WAV"}));
    CHECK_FALSE(MatchesAnyExtension("a.mp4", {".wav", ".csv"}));
    // Empty filter list: no filtering, everything matches.
    CHECK(MatchesAnyExtension("a.mp4", {}));
    CHECK(MatchesAnyExtension("README", {}));
    // Extensionless names never match a non-empty filter.
    CHECK_FALSE(MatchesAnyExtension("README", {".wav"}));
}

// ---------------------------------------------------------------------------
// MakeNumberedName
// ---------------------------------------------------------------------------

TEST_CASE("MakeNumberedName inserts _n before the extension") {
    CHECK(MakeNumberedName("clip.mp4", 2) == "clip_2.mp4");
    CHECK(MakeNumberedName("clip.mp4", 3) == "clip_3.mp4");
    CHECK(MakeNumberedName("archive.tar.gz", 2) == "archive.tar_2.gz");
}

TEST_CASE("MakeNumberedName leaves the name unchanged for n <= 1") {
    CHECK(MakeNumberedName("clip.mp4", 1) == "clip.mp4");
    CHECK(MakeNumberedName("clip.mp4", 0) == "clip.mp4");
    CHECK(MakeNumberedName("clip.mp4", -1) == "clip.mp4");
}

TEST_CASE("MakeNumberedName handles an extensionless base name") {
    CHECK(MakeNumberedName("README", 2) == "README_2");
    CHECK(MakeNumberedName(".gitignore", 2) == ".gitignore_2");
}

// ---------------------------------------------------------------------------
// NextAvailableName (noClobber dedupe, Docs/BRIDGE_CONTRACT.md Sec.4.20)
// ---------------------------------------------------------------------------

TEST_CASE("NextAvailableName: no collision returns the desired name as-is") {
    const std::set<std::string> existing = {"other.mp4"};
    CHECK(NextAvailableName(existing, "clip.mp4") == "clip.mp4");
}

TEST_CASE("NextAvailableName: single collision numbers to _2") {
    const std::set<std::string> existing = {"clip.mp4"};
    CHECK(NextAvailableName(existing, "clip.mp4") == "clip_2.mp4");
}

TEST_CASE("NextAvailableName: consecutive collisions walk up the sequence") {
    const std::set<std::string> existing = {"clip.mp4", "clip_2.mp4", "clip_3.mp4"};
    CHECK(NextAvailableName(existing, "clip.mp4") == "clip_4.mp4");
}

TEST_CASE("NextAvailableName: a gap in the sequence is still skipped over in order") {
    // _2 is free but _3 is taken too far ahead to matter: dedupe always starts
    // at _2 and walks up, it never "fills gaps" out of order.
    const std::set<std::string> existing = {"clip.mp4", "clip_3.mp4"};
    CHECK(NextAvailableName(existing, "clip.mp4") == "clip_2.mp4");
}

TEST_CASE("NextAvailableName: extensionless base name") {
    const std::set<std::string> existing = {"README", "README_2"};
    CHECK(NextAvailableName(existing, "README") == "README_3");
}

// ---------------------------------------------------------------------------
// UTF-8 BOM helpers
// ---------------------------------------------------------------------------

TEST_CASE("HasUtf8Bom detects a leading BOM only") {
    CHECK(HasUtf8Bom("\xEF\xBB\xBFhello"));
    CHECK_FALSE(HasUtf8Bom("hello"));
    CHECK_FALSE(HasUtf8Bom(""));
    CHECK_FALSE(HasUtf8Bom("\xEF\xBB"));  // too short
}

TEST_CASE("AddUtf8Bom is idempotent (never doubles the BOM)") {
    const std::string once = AddUtf8Bom("hello");
    CHECK(once == std::string("\xEF\xBB\xBF") + "hello");
    const std::string twice = AddUtf8Bom(once);
    CHECK(twice == once);  // not doubled
    CHECK(HasUtf8Bom(twice));
}

TEST_CASE("RemoveUtf8Bom strips exactly one leading BOM") {
    CHECK(RemoveUtf8Bom(std::string("\xEF\xBB\xBF") + "hello") == "hello");
    CHECK(RemoveUtf8Bom("hello") == "hello");  // unchanged, no BOM present
    CHECK(RemoveUtf8Bom("") == "");
}

TEST_CASE("AddUtf8Bom then RemoveUtf8Bom round-trips") {
    const std::string original = "content without a bom";
    CHECK(RemoveUtf8Bom(AddUtf8Bom(original)) == original);
}

// ---------------------------------------------------------------------------
// Path helpers
// ---------------------------------------------------------------------------

TEST_CASE("JoinPath inserts a single backslash") {
    CHECK(JoinPath("C:\\Users\\mock", "clip.mp4") == "C:\\Users\\mock\\clip.mp4");
}

TEST_CASE("JoinPath does not double a trailing separator") {
    CHECK(JoinPath("C:\\Users\\mock\\", "clip.mp4") == "C:\\Users\\mock\\clip.mp4");
    CHECK(JoinPath("C:/Users/mock/", "clip.mp4") == "C:/Users/mock/clip.mp4");
}

TEST_CASE("JoinPath with an empty dir returns the name unchanged") {
    CHECK(JoinPath("", "clip.mp4") == "clip.mp4");
}

TEST_CASE("ParentDirOf extracts the containing folder") {
    CHECK(ParentDirOf("C:\\Users\\mock\\clip.mp4") == "C:\\Users\\mock");
    CHECK(ParentDirOf("C:/Users/mock/clip.mp4") == "C:/Users/mock");
}

TEST_CASE("ParentDirOf ignores a trailing separator") {
    CHECK(ParentDirOf("C:\\Users\\mock\\") == "C:\\Users");
}

TEST_CASE("ParentDirOf with no separator returns empty") {
    CHECK(ParentDirOf("clip.mp4") == "");
    CHECK(ParentDirOf("") == "");
}
