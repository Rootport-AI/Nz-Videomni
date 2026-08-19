// test_http_client.cpp - unit tests for CrackUrl() plus integration tests for
// the WinHTTP client.
//
// The CrackUrl() cases below run in the *default* doctest suite (no
// TEST_SUITE wrapper), so they always execute -- including under
// `scripts/build.ps1 -RunTests`, which runs
// NzVideomni_tests.exe --test-suite-exclude=integration with no live backend.
//
// Everything else here exercises real HTTP against the mock LTX23 backend
// expected at http://127.0.0.1:18620 (kBackendBaseUrl), so it is grouped in
// the "integration" test suite and filtered out by that same flag.
//
// ASCII-only source.
#include "doctest.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <algorithm>
#include <chrono>
#include <future>
#include <string>
#include <thread>

#include "json.hpp"

#include "bridge_core.h"
#include "http_client.h"

using json = nlohmann::json;

namespace {

std::string ApiBase() {
    return std::string(nzvideomni::kBackendBaseUrl) + nzvideomni::kBackendApiPrefix;
}

std::wstring TempMp4Path(const std::wstring& name) {
    wchar_t dir[MAX_PATH] = {};
    const DWORD n = ::GetTempPathW(MAX_PATH, dir);
    std::wstring path(dir, n);
    path += L"nzvideomni_test_";
    path += name;
    return path;
}

}  // namespace

// CrackUrl() is a pure function (no sockets), so these run unconditionally in
// the default suite. They pin down the WinHttpCrackUrl behaviour this module
// depends on: with lpszExtraInfo=NULL/dwExtraInfoLength=0, the query string
// comes back already attached to lpszUrlPath, so CrackUrl() must NOT
// re-attach it a second time (that was the bug behind the double-query
// "?max_frames=11544?max_frames=11544" upload failures).

TEST_CASE("CrackUrl: single-key query is attached to the path exactly once") {
    const nzvideomni::detail::UrlParts parts = nzvideomni::detail::CrackUrl(
        L"http://127.0.0.1:18620/api/v1/upload/video?max_frames=11544");
    REQUIRE(parts.ok);
    CHECK(parts.host == L"127.0.0.1");
    CHECK(parts.port == 18620);
    CHECK_FALSE(parts.secure);
    CHECK(parts.path == L"/api/v1/upload/video?max_frames=11544");
    CHECK(std::count(parts.path.begin(), parts.path.end(), L'?') == 1);
}

TEST_CASE("CrackUrl: multi-key query is preserved verbatim") {
    const nzvideomni::detail::UrlParts parts =
        nzvideomni::detail::CrackUrl(L"http://127.0.0.1:18620/api/v1/status?a=1&b=2");
    REQUIRE(parts.ok);
    CHECK(parts.host == L"127.0.0.1");
    CHECK(parts.port == 18620);
    CHECK(parts.path == L"/api/v1/status?a=1&b=2");
    CHECK(std::count(parts.path.begin(), parts.path.end(), L'?') == 1);
}

TEST_CASE("CrackUrl: no query string leaves the path unchanged") {
    const nzvideomni::detail::UrlParts parts =
        nzvideomni::detail::CrackUrl(L"http://127.0.0.1:18620/api/v1/status");
    REQUIRE(parts.ok);
    CHECK(parts.host == L"127.0.0.1");
    CHECK(parts.port == 18620);
    CHECK(parts.path == L"/api/v1/status");
    CHECK(parts.path.find(L'?') == std::wstring::npos);
}

TEST_CASE("CrackUrl: no path defaults to \"/\"") {
    const nzvideomni::detail::UrlParts parts =
        nzvideomni::detail::CrackUrl(L"http://127.0.0.1:18620");
    REQUIRE(parts.ok);
    CHECK(parts.host == L"127.0.0.1");
    CHECK(parts.port == 18620);
    CHECK(parts.path == L"/");
}

TEST_SUITE("integration") {

TEST_CASE("GET /status returns 200 with a running server body") {
    nzvideomni::HttpClient client;
    const nzvideomni::HttpResponse resp =
        client.RequestSync(ApiBase() + "/status", "GET", "", nzvideomni::kDefaultRequestTimeoutMs);
    REQUIRE(resp.transport == nzvideomni::TransportError::kNone);
    CHECK(resp.status == 200);
    const json body = json::parse(resp.body, nullptr, false);
    REQUIRE_FALSE(body.is_discarded());
    CHECK(body.contains("server"));
}

TEST_CASE("an unknown job id returns a 404 as a normal (non-transport) result") {
    nzvideomni::HttpClient client;
    const nzvideomni::HttpResponse resp = client.RequestSync(
        ApiBase() + "/jobs/does-not-exist", "GET", "", nzvideomni::kDefaultRequestTimeoutMs);
    REQUIRE(resp.transport == nzvideomni::TransportError::kNone);
    CHECK(resp.status == 404);
}

TEST_CASE("RequestAsync delivers the response on a worker thread") {
    nzvideomni::HttpClient client;
    std::promise<nzvideomni::HttpResponse> promise;
    auto future = promise.get_future();
    client.RequestAsync(ApiBase() + "/status", "GET", "",
                        nzvideomni::kDefaultRequestTimeoutMs,
                        [&promise](nzvideomni::HttpResponse r) { promise.set_value(std::move(r)); });
    REQUIRE(future.wait_for(std::chrono::seconds(30)) == std::future_status::ready);
    const nzvideomni::HttpResponse resp = future.get();
    CHECK(resp.transport == nzvideomni::TransportError::kNone);
    CHECK(resp.status == 200);
}

TEST_CASE("full generate -> poll -> download flow saves a non-empty mp4") {
    nzvideomni::HttpClient client;

    // 1. Kick off a tiny T2V generation.
    const std::string gen_body =
        R"({"prompt":"integration test clip","width":512,"height":320,)"
        R"("num_frames":9,"frame_rate":24.0})";
    const nzvideomni::HttpResponse gen =
        client.RequestSync(ApiBase() + "/generate", "POST", gen_body,
                           nzvideomni::kDefaultRequestTimeoutMs);
    REQUIRE(gen.transport == nzvideomni::TransportError::kNone);
    REQUIRE(gen.status == 202);
    const json gen_json = json::parse(gen.body, nullptr, false);
    REQUIRE_FALSE(gen_json.is_discarded());
    const std::string job_id = gen_json.value("job_id", std::string());
    REQUIRE_FALSE(job_id.empty());

    // 2. Poll until the job completes (mock finishes quickly; allow generous slack).
    std::string status;
    for (int i = 0; i < 120; ++i) {
        const nzvideomni::HttpResponse poll = client.RequestSync(
            ApiBase() + "/jobs/" + job_id, "GET", "", nzvideomni::kDefaultRequestTimeoutMs);
        REQUIRE(poll.transport == nzvideomni::TransportError::kNone);
        REQUIRE(poll.status == 200);
        const json pj = json::parse(poll.body, nullptr, false);
        REQUIRE_FALSE(pj.is_discarded());
        status = pj.value("status", std::string());
        if (status == "completed" || status == "failed" || status == "cancelled") {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    REQUIRE(status == "completed");

    // 3. Download the finished video to disk.
    const std::wstring dest = TempMp4Path(std::wstring(job_id.begin(), job_id.end()) + L".mp4");
    ::DeleteFileW(dest.c_str());
    const nzvideomni::DownloadResult dl = client.DownloadSync(
        ApiBase() + "/jobs/" + job_id + "/video", dest, nzvideomni::kDefaultDownloadTimeoutMs);
    REQUIRE(dl.transport == nzvideomni::TransportError::kNone);
    CHECK(dl.status == 200);
    CHECK(dl.size_bytes > 0);

    // Verify the on-disk file matches the reported size.
    const HANDLE h = ::CreateFileW(dest.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                                   OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    REQUIRE(h != INVALID_HANDLE_VALUE);
    LARGE_INTEGER size = {};
    ::GetFileSizeEx(h, &size);
    ::CloseHandle(h);
    CHECK(size.QuadPart == dl.size_bytes);
    ::DeleteFileW(dest.c_str());
}

TEST_CASE("downloading a non-existent job reports a non-200 and writes no file") {
    nzvideomni::HttpClient client;
    const std::wstring dest = TempMp4Path(L"missing.mp4");
    ::DeleteFileW(dest.c_str());
    const nzvideomni::DownloadResult dl = client.DownloadSync(
        ApiBase() + "/jobs/does-not-exist/video", dest, nzvideomni::kDefaultDownloadTimeoutMs);
    REQUIRE(dl.transport == nzvideomni::TransportError::kNone);
    CHECK(dl.status != 200);
    CHECK(dl.size_bytes == 0);
    CHECK(::GetFileAttributesW(dest.c_str()) == INVALID_FILE_ATTRIBUTES);
}

}  // TEST_SUITE("integration")
