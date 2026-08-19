// test_settings.cpp - unit tests for persistent settings (contract v4).
//
// Covers the pure validation/serialization helpers (NormalizeBaseUrl,
// ParseSettingsJson, BuildSettingsJson) and a SettingsStore round trip against
// a temporary file (never touches the real %LOCALAPPDATA% location).
//
// ASCII-only source.
#include "doctest.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <string>

#include "settings.h"

using nzltx::BuildSettingsJson;
using nzltx::NormalizeBaseUrl;
using nzltx::ParseSettingsJson;
using nzltx::SettingsStore;

namespace {

// A fresh temp file path for this test run; never actually created by
// GetTempFileNameW/CreateFile here, just a name under %TEMP% the test owns.
std::wstring TempSettingsPath(const wchar_t* name) {
    wchar_t dir[MAX_PATH] = {};
    const DWORD n = ::GetTempPathW(MAX_PATH, dir);
    std::wstring path(dir, n);
    path += L"nzltx23_test_settings_";
    path += name;
    path += L".json";
    return path;
}

// Write raw UTF-8 bytes to a file, overwriting it. Used to seed a
// malformed/edge-case settings.json ahead of a Load() call.
void WriteRaw(const std::wstring& path, const std::string& content) {
    HANDLE h = ::CreateFileW(path.c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr,
                             CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    REQUIRE(h != INVALID_HANDLE_VALUE);
    DWORD written = 0;
    if (!content.empty()) {
        ::WriteFile(h, content.data(), static_cast<DWORD>(content.size()), &written,
                    nullptr);
    }
    ::CloseHandle(h);
}

// Read raw UTF-8 bytes back from a file (empty string if it does not exist).
std::string ReadRaw(const std::wstring& path) {
    HANDLE h = ::CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                             OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        return std::string();
    }
    LARGE_INTEGER size = {};
    ::GetFileSizeEx(h, &size);
    std::string content(static_cast<size_t>(size.QuadPart), '\0');
    DWORD read = 0;
    if (!content.empty()) {
        ::ReadFile(h, content.data(), static_cast<DWORD>(content.size()), &read, nullptr);
    }
    ::CloseHandle(h);
    content.resize(read);
    return content;
}

}  // namespace

// ---------------------------------------------------------------------------
// NormalizeBaseUrl
// ---------------------------------------------------------------------------

TEST_CASE("NormalizeBaseUrl accepts http/https and strips a trailing slash") {
    std::string out;
    std::string err;
    REQUIRE(NormalizeBaseUrl("http://127.0.0.1:19999/", &out, &err));
    CHECK(out == "http://127.0.0.1:19999");

    REQUIRE(NormalizeBaseUrl("https://backend.example.com", &out, &err));
    CHECK(out == "https://backend.example.com");
}

TEST_CASE("NormalizeBaseUrl strips repeated trailing slashes") {
    std::string out;
    std::string err;
    REQUIRE(NormalizeBaseUrl("http://127.0.0.1:19999///", &out, &err));
    CHECK(out == "http://127.0.0.1:19999");
}

TEST_CASE("NormalizeBaseUrl leaves an already-normalized URL unchanged") {
    std::string out;
    std::string err;
    REQUIRE(NormalizeBaseUrl("http://127.0.0.1:18620", &out, &err));
    CHECK(out == "http://127.0.0.1:18620");
}

TEST_CASE("NormalizeBaseUrl rejects a missing/unsupported scheme") {
    std::string out;
    std::string err;
    CHECK_FALSE(NormalizeBaseUrl("127.0.0.1:19999", &out, &err));
    CHECK_FALSE(err.empty());
    CHECK_FALSE(NormalizeBaseUrl("ftp://127.0.0.1:19999", &out, &err));
    CHECK_FALSE(NormalizeBaseUrl("", &out, &err));
}

TEST_CASE("NormalizeBaseUrl rejects a scheme with an empty host") {
    std::string out;
    std::string err;
    CHECK_FALSE(NormalizeBaseUrl("http://", &out, &err));
    CHECK_FALSE(NormalizeBaseUrl("http:///path", &out, &err));
    CHECK_FALSE(NormalizeBaseUrl("https://", &out, &err));
}

// ---------------------------------------------------------------------------
// ParseSettingsJson / BuildSettingsJson
// ---------------------------------------------------------------------------

TEST_CASE("BuildSettingsJson then ParseSettingsJson round-trips a valid baseUrl") {
    const std::string doc = BuildSettingsJson("http://127.0.0.1:19999");
    CHECK(ParseSettingsJson(doc) == "http://127.0.0.1:19999");
}

TEST_CASE("ParseSettingsJson falls back to the default on malformed JSON") {
    CHECK(ParseSettingsJson("not json at all") == nzltx::kBackendBaseUrl);
    CHECK(ParseSettingsJson("") == nzltx::kBackendBaseUrl);
    CHECK(ParseSettingsJson("[1,2,3]") == nzltx::kBackendBaseUrl);
}

TEST_CASE("ParseSettingsJson falls back to the default on a missing/invalid baseUrl") {
    CHECK(ParseSettingsJson("{}") == nzltx::kBackendBaseUrl);
    CHECK(ParseSettingsJson(R"({"baseUrl": 123})") == nzltx::kBackendBaseUrl);
    CHECK(ParseSettingsJson(R"({"baseUrl": "not-a-url"})") == nzltx::kBackendBaseUrl);
}

// ---------------------------------------------------------------------------
// SettingsStore
// ---------------------------------------------------------------------------

TEST_CASE("SettingsStore defaults to kBackendBaseUrl before Load()") {
    const std::wstring path = TempSettingsPath(L"defaults");
    ::DeleteFileW(path.c_str());
    SettingsStore store(path);
    CHECK(store.GetBaseUrl() == nzltx::kBackendBaseUrl);
}

TEST_CASE("SettingsStore Load() with no file on disk keeps the default") {
    const std::wstring path = TempSettingsPath(L"missing");
    ::DeleteFileW(path.c_str());
    SettingsStore store(path);
    store.Load();
    CHECK(store.GetBaseUrl() == nzltx::kBackendBaseUrl);
}

TEST_CASE("SettingsStore SetBaseUrl persists and a fresh instance reloads it") {
    const std::wstring path = TempSettingsPath(L"roundtrip");
    ::DeleteFileW(path.c_str());

    SettingsStore writer(path);
    std::string normalized;
    std::string err;
    REQUIRE(writer.SetBaseUrl("http://127.0.0.1:19999/", &normalized, &err));
    CHECK(normalized == "http://127.0.0.1:19999");
    CHECK(writer.GetBaseUrl() == "http://127.0.0.1:19999");

    // A second, independent instance pointed at the same file should reload
    // the persisted value.
    SettingsStore reader(path);
    reader.Load();
    CHECK(reader.GetBaseUrl() == "http://127.0.0.1:19999");

    ::DeleteFileW(path.c_str());
}

TEST_CASE("SettingsStore SetBaseUrl rejects an invalid candidate without writing") {
    const std::wstring path = TempSettingsPath(L"reject");
    ::DeleteFileW(path.c_str());

    SettingsStore store(path);
    std::string normalized;
    std::string err;
    REQUIRE(store.SetBaseUrl("http://127.0.0.1:18620", &normalized, &err));
    const std::string before = ReadRaw(path);
    REQUIRE_FALSE(before.empty());

    const bool ok = store.SetBaseUrl("not-a-url", &normalized, &err);
    CHECK_FALSE(ok);
    CHECK_FALSE(err.empty());
    // The in-memory value and the on-disk file are both unchanged.
    CHECK(store.GetBaseUrl() == "http://127.0.0.1:18620");
    CHECK(ReadRaw(path) == before);

    ::DeleteFileW(path.c_str());
}

TEST_CASE("SettingsStore Load() self-heals from a malformed file without rewriting it") {
    const std::wstring path = TempSettingsPath(L"corrupt");
    WriteRaw(path, "{ this is not valid json");

    SettingsStore store(path);
    store.Load();
    CHECK(store.GetBaseUrl() == nzltx::kBackendBaseUrl);
    // Load() must not have overwritten the corrupt file.
    CHECK(ReadRaw(path) == "{ this is not valid json");

    ::DeleteFileW(path.c_str());
}

TEST_CASE("SettingsStore Load() self-heals from an invalid baseUrl value") {
    const std::wstring path = TempSettingsPath(L"invalid_value");
    WriteRaw(path, R"({"baseUrl": "not-a-url"})");

    SettingsStore store(path);
    store.Load();
    CHECK(store.GetBaseUrl() == nzltx::kBackendBaseUrl);

    ::DeleteFileW(path.c_str());
}

TEST_CASE("SettingsStore with an empty file path never persists") {
    // Named variable (not a temporary) to avoid a most-vexing-parse ambiguity
    // with `SettingsStore store(std::wstring());`.
    const std::wstring empty_path;
    SettingsStore store(empty_path);
    std::string normalized;
    std::string err;
    REQUIRE(store.SetBaseUrl("http://127.0.0.1:19999", &normalized, &err));
    CHECK(store.GetBaseUrl() == "http://127.0.0.1:19999");
    // Nothing to assert on disk (no path was given); just confirm Load() is
    // safe to call and does not crash / does not clear the in-memory value
    // set above (it has nothing to read, so it resets to the default).
    store.Load();
    CHECK(store.GetBaseUrl() == nzltx::kBackendBaseUrl);
}
