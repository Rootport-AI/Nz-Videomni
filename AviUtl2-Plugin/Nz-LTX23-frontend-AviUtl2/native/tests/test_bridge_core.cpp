// test_bridge_core.cpp - unit tests for the RPC dispatch logic (contract v2).
#include "doctest.h"

#include "json.hpp"

#include "bridge_core.h"

using nzltx::BackendRequest;
using nzltx::CaptureFrameRequest;
using nzltx::DownloadVideoRequest;
using nzltx::UploadFileRequest;
using nzltx::EditInfoResult;
using nzltx::HandleRequestJson;
using nzltx::InsertMediaParams;
using nzltx::InsertMediaResult;
using nzltx::MakeThumbnailRequest;
using nzltx::PickFileFilterEntry;
using nzltx::RequestContext;
using json = nlohmann::json;

namespace {

// Provider that reports an available edit info snapshot.
EditInfoResult AvailableInfo() {
    EditInfoResult info;
    info.available = true;
    info.width = 1920;
    info.height = 1080;
    info.rate = 30;
    info.scale = 1;
    info.sample_rate = 48000;
    info.frame = 42;
    info.layer = 3;
    info.frame_max = 100;
    info.layer_max = 10;
    return info;
}

// A context whose edit_info is available and which records the last insert
// request while returning a caller-chosen status.
RequestContext MakeContext(nzltx::EditInfoProvider edit,
                           InsertMediaResult::Status insert_status,
                           InsertMediaParams* captured = nullptr) {
    RequestContext ctx;
    ctx.edit_info = std::move(edit);
    ctx.insert_media = [insert_status, captured](const InsertMediaParams& p)
        -> InsertMediaResult {
        if (captured != nullptr) {
            *captured = p;
        }
        InsertMediaResult r;
        r.status = insert_status;
        // Echo back a resolved position (mimicking the cursor fallback).
        r.layer = p.has_layer ? p.layer : 7;
        r.frame = p.has_frame ? p.frame : 99;
        return r;
    };
    return ctx;
}

RequestContext AvailableCtx() {
    return MakeContext(AvailableInfo, InsertMediaResult::Status::kOk);
}

}  // namespace

// ---------------------------------------------------------------------------
// Envelope / contract v1 carry-over
// ---------------------------------------------------------------------------

TEST_CASE("malformed JSON yields BAD_REQUEST with null id") {
    const std::string resp = HandleRequestJson("not json at all", AvailableCtx());
    const json j = json::parse(resp);
    CHECK(j["ok"] == false);
    CHECK(j["id"].is_null());
    CHECK(j["error"]["code"] == "BAD_REQUEST");
    CHECK(j["error"]["message"].is_string());
}

TEST_CASE("valid JSON that is not an object yields BAD_REQUEST") {
    const std::string resp = HandleRequestJson("[1,2,3]", AvailableCtx());
    const json j = json::parse(resp);
    CHECK(j["ok"] == false);
    CHECK(j["error"]["code"] == "BAD_REQUEST");
}

TEST_CASE("missing method field yields BAD_REQUEST and echoes id") {
    const std::string resp =
        HandleRequestJson(R"({"id": 7, "params": {}})", AvailableCtx());
    const json j = json::parse(resp);
    CHECK(j["ok"] == false);
    CHECK(j["id"] == 7);
    CHECK(j["error"]["code"] == "BAD_REQUEST");
}

TEST_CASE("unknown method yields UNKNOWN_METHOD and echoes id") {
    const std::string resp =
        HandleRequestJson(R"({"id": 11, "method": "frobnicate"})", AvailableCtx());
    const json j = json::parse(resp);
    CHECK(j["ok"] == false);
    CHECK(j["id"] == 11);
    CHECK(j["error"]["code"] == "UNKNOWN_METHOD");
}

TEST_CASE("ping returns pong and plugin version") {
    const std::string resp =
        HandleRequestJson(R"({"id": 1, "method": "ping", "params": {}})", AvailableCtx());
    const json j = json::parse(resp);
    CHECK(j["id"] == 1);
    CHECK(j["ok"] == true);
    CHECK(j["result"]["pong"] == true);
    CHECK(j["result"]["pluginVersion"] == "1.0.0-rc1");
}

TEST_CASE("getEditInfo returns the six mandatory fields") {
    const std::string resp =
        HandleRequestJson(R"({"id": 2, "method": "getEditInfo"})", AvailableCtx());
    const json j = json::parse(resp);
    CHECK(j["id"] == 2);
    CHECK(j["ok"] == true);
    const json& r = j["result"];
    CHECK(r["width"] == 1920);
    CHECK(r["height"] == 1080);
    CHECK(r["rate"] == 30);
    CHECK(r["scale"] == 1);
    CHECK(r["sampleRate"] == 48000);
    CHECK(r["frame"] == 42);
    CHECK(r.contains("layer"));
    CHECK(r.contains("frameMax"));
    CHECK(r.contains("layerMax"));
}

TEST_CASE("getEditInfo without an edit handle yields NO_EDIT_HANDLE") {
    RequestContext ctx = MakeContext([]() { return EditInfoResult{}; },
                                     InsertMediaResult::Status::kOk);
    const std::string resp =
        HandleRequestJson(R"({"id": 3, "method": "getEditInfo"})", ctx);
    const json j = json::parse(resp);
    CHECK(j["id"] == 3);
    CHECK(j["ok"] == false);
    CHECK(j["error"]["code"] == "NO_EDIT_HANDLE");
}

TEST_CASE("an object with neither id nor method produces no response") {
    const std::string resp =
        HandleRequestJson(R"({"event": "something", "data": {}})", AvailableCtx());
    CHECK(resp.empty());
}

TEST_CASE("string id is not echoed as a number (defaults to null)") {
    const std::string resp =
        HandleRequestJson(R"({"id": "abc", "method": "ping"})", AvailableCtx());
    const json j = json::parse(resp);
    CHECK(j["ok"] == true);
    CHECK(j["id"].is_null());
}

// ---------------------------------------------------------------------------
// backend.getBaseUrl
// ---------------------------------------------------------------------------

TEST_CASE("backend.getBaseUrl returns the configured base URL") {
    const std::string resp =
        HandleRequestJson(R"({"id": 5, "method": "backend.getBaseUrl"})", AvailableCtx());
    const json j = json::parse(resp);
    CHECK(j["id"] == 5);
    CHECK(j["ok"] == true);
    CHECK(j["result"]["baseUrl"] == "http://127.0.0.1:18620");
}

// ---------------------------------------------------------------------------
// timeline.insertMedia
// ---------------------------------------------------------------------------

TEST_CASE("insertMedia without filePath yields BAD_REQUEST") {
    const std::string resp = HandleRequestJson(
        R"({"id": 6, "method": "timeline.insertMedia", "params": {}})", AvailableCtx());
    const json j = json::parse(resp);
    CHECK(j["ok"] == false);
    CHECK(j["error"]["code"] == "BAD_REQUEST");
}

TEST_CASE("insertMedia success echoes resolved layer/frame") {
    InsertMediaParams captured;
    RequestContext ctx =
        MakeContext(AvailableInfo, InsertMediaResult::Status::kOk, &captured);
    const std::string resp = HandleRequestJson(
        R"({"id": 6, "method": "timeline.insertMedia",
            "params": {"filePath": "C:\\tmp\\a.mp4", "layer": 2, "frame": 100}})",
        ctx);
    const json j = json::parse(resp);
    CHECK(j["ok"] == true);
    CHECK(j["result"]["inserted"] == true);
    CHECK(j["result"]["layer"] == 2);
    CHECK(j["result"]["frame"] == 100);
    CHECK(captured.file_path == "C:\\tmp\\a.mp4");
    CHECK(captured.has_layer == true);
    CHECK(captured.has_frame == true);
}

TEST_CASE("insertMedia falls back to cursor position when layer/frame omitted") {
    InsertMediaParams captured;
    RequestContext ctx =
        MakeContext(AvailableInfo, InsertMediaResult::Status::kOk, &captured);
    const std::string resp = HandleRequestJson(
        R"({"id": 6, "method": "timeline.insertMedia",
            "params": {"filePath": "C:\\tmp\\a.mp4"}})",
        ctx);
    const json j = json::parse(resp);
    CHECK(j["ok"] == true);
    CHECK(captured.has_layer == false);
    CHECK(captured.has_frame == false);
    // Provider's cursor fallback values.
    CHECK(j["result"]["layer"] == 7);
    CHECK(j["result"]["frame"] == 99);
}

TEST_CASE("insertMedia maps provider statuses to error codes") {
    struct Case {
        InsertMediaResult::Status status;
        const char* code;
    };
    const Case cases[] = {
        {InsertMediaResult::Status::kFileNotFound, "FILE_NOT_FOUND"},
        {InsertMediaResult::Status::kNoEditHandle, "NO_EDIT_HANDLE"},
        {InsertMediaResult::Status::kInsertFailed, "INSERT_FAILED"},
    };
    for (const Case& c : cases) {
        RequestContext ctx = MakeContext(AvailableInfo, c.status);
        const std::string resp = HandleRequestJson(
            R"({"id": 6, "method": "timeline.insertMedia",
                "params": {"filePath": "C:\\tmp\\a.mp4"}})",
            ctx);
        const json j = json::parse(resp);
        CHECK(j["ok"] == false);
        CHECK(j["error"]["code"] == c.code);
    }
}

// ---------------------------------------------------------------------------
// backend.request parsing / URL building / result formatting
// ---------------------------------------------------------------------------

TEST_CASE("ParseBackendRequest accepts a minimal GET") {
    BackendRequest out;
    std::string err;
    const json params = json::parse(R"({"method": "GET", "path": "/status"})");
    CHECK(nzltx::ParseBackendRequest(params, &out, &err));
    CHECK(out.http_method == "GET");
    CHECK(out.path == "/status");
    CHECK(out.query.empty());
    CHECK(out.has_body == false);
}

TEST_CASE("ParseBackendRequest url-encodes query and serializes a body") {
    BackendRequest out;
    std::string err;
    const json params = json::parse(
        R"({"method": "POST", "path": "/generate",
            "query": {"a b": "x/y", "k": "v"},
            "body": {"prompt": "hi"}})");
    CHECK(nzltx::ParseBackendRequest(params, &out, &err));
    CHECK(out.http_method == "POST");
    // Object key order is preserved by nlohmann for insertion... but query uses
    // the (sorted) object; assert both pairs are present and encoded.
    CHECK(out.query.find("a%20b=x%2Fy") != std::string::npos);
    CHECK(out.query.find("k=v") != std::string::npos);
    CHECK(out.has_body == true);
    CHECK(json::parse(out.body)["prompt"] == "hi");
}

TEST_CASE("ParseBackendRequest rejects bad method / path / query / body") {
    std::string err;
    BackendRequest out;
    CHECK_FALSE(nzltx::ParseBackendRequest(
        json::parse(R"({"method": "PUT", "path": "/x"})"), &out, &err));
    CHECK_FALSE(nzltx::ParseBackendRequest(
        json::parse(R"({"method": "GET", "path": "no-slash"})"), &out, &err));
    CHECK_FALSE(nzltx::ParseBackendRequest(
        json::parse(R"({"method": "GET", "path": "/x", "query": {"k": 1}})"), &out,
        &err));
    CHECK_FALSE(nzltx::ParseBackendRequest(
        json::parse(R"({"method": "GET", "path": "/x", "body": "nope"})"), &out, &err));
    CHECK_FALSE(nzltx::ParseBackendRequest(json::parse(R"({"path": "/x"})"), &out, &err));
}

TEST_CASE("ParseBackendRequest defaults timeoutMs to the built-in default") {
    BackendRequest out;
    std::string err;
    const json params = json::parse(R"({"method": "GET", "path": "/status"})");
    CHECK(nzltx::ParseBackendRequest(params, &out, &err));
    CHECK(out.timeout_ms == nzltx::kDefaultBackendTimeoutMs);
}

TEST_CASE("ParseBackendRequest accepts an in-range timeoutMs (WebUI join 120000)") {
    BackendRequest out;
    std::string err;
    const json params = json::parse(
        R"({"method": "POST", "path": "/jobs/abc/join", "timeoutMs": 120000})");
    CHECK(nzltx::ParseBackendRequest(params, &out, &err));
    CHECK(out.timeout_ms == 120000);
}

TEST_CASE("ParseBackendRequest clamps timeoutMs to [1000, 600000]") {
    BackendRequest out;
    std::string err;
    // Below the floor -> clamped up to 1000.
    CHECK(nzltx::ParseBackendRequest(
        json::parse(R"({"method": "GET", "path": "/x", "timeoutMs": 5})"), &out, &err));
    CHECK(out.timeout_ms == nzltx::kMinBackendTimeoutMs);
    // Above the ceiling -> clamped down to 600000.
    CHECK(nzltx::ParseBackendRequest(
        json::parse(R"({"method": "GET", "path": "/x", "timeoutMs": 5000000})"), &out,
        &err));
    CHECK(out.timeout_ms == nzltx::kMaxBackendTimeoutMs);
    // A gigantic value that would overflow a naive int cast is still clamped.
    CHECK(nzltx::ParseBackendRequest(
        json::parse(R"({"method": "GET", "path": "/x", "timeoutMs": 1e12})"), &out,
        &err));
    CHECK(out.timeout_ms == nzltx::kMaxBackendTimeoutMs);
    // Exact bounds pass through unchanged.
    CHECK(nzltx::ParseBackendRequest(
        json::parse(R"({"method": "GET", "path": "/x", "timeoutMs": 1000})"), &out,
        &err));
    CHECK(out.timeout_ms == 1000);
    // A fractional value truncates toward zero.
    CHECK(nzltx::ParseBackendRequest(
        json::parse(R"({"method": "GET", "path": "/x", "timeoutMs": 1500.9})"), &out,
        &err));
    CHECK(out.timeout_ms == 1500);
}

TEST_CASE("ParseBackendRequest rejects a non-numeric timeoutMs as BAD_REQUEST") {
    BackendRequest out;
    std::string err;
    CHECK_FALSE(nzltx::ParseBackendRequest(
        json::parse(R"({"method": "GET", "path": "/x", "timeoutMs": "soon"})"), &out,
        &err));
    CHECK_FALSE(err.empty());
    CHECK_FALSE(nzltx::ParseBackendRequest(
        json::parse(R"({"method": "GET", "path": "/x", "timeoutMs": true})"), &out,
        &err));
    // A null timeoutMs is treated as absent (default applies).
    CHECK(nzltx::ParseBackendRequest(
        json::parse(R"({"method": "GET", "path": "/x", "timeoutMs": null})"), &out,
        &err));
    CHECK(out.timeout_ms == nzltx::kDefaultBackendTimeoutMs);
}

TEST_CASE("BuildBackendUrl appends the already-prefixed path and query") {
    BackendRequest req;
    req.http_method = "GET";
    req.path = "/api/v1/jobs/123";
    req.query = "a=1&b=2";
    CHECK(nzltx::BuildBackendUrl("http://127.0.0.1:18620", req) ==
          "http://127.0.0.1:18620/api/v1/jobs/123?a=1&b=2");
    req.query.clear();
    CHECK(nzltx::BuildBackendUrl("http://127.0.0.1:18620", req) ==
          "http://127.0.0.1:18620/api/v1/jobs/123");
    // Regression guard for the double-prefix bug: req.path already carries
    // /api/v1, so BuildBackendUrl must not prepend it again.
    CHECK(nzltx::BuildBackendUrl("http://127.0.0.1:18620", req)
              .find("/api/v1/api/v1") == std::string::npos);
}

TEST_CASE("MakeBackendRequestResult parses JSON bodies and passes through status") {
    json r = nzltx::MakeBackendRequestResult(202, R"({"job_id": "abc"})");
    CHECK(r["status"] == 202);
    CHECK(r["body"]["job_id"] == "abc");

    json empty = nzltx::MakeBackendRequestResult(204, "");
    CHECK(empty["status"] == 204);
    CHECK(empty["body"].is_null());

    json garbage = nzltx::MakeBackendRequestResult(500, "<<not json>>");
    CHECK(garbage["status"] == 500);
    CHECK(garbage["body"].is_null());
}

// ---------------------------------------------------------------------------
// backend.downloadVideo parsing
// ---------------------------------------------------------------------------

TEST_CASE("ParseDownloadVideo accepts a job id and derives path + file name") {
    DownloadVideoRequest out;
    std::string err;
    CHECK(nzltx::ParseDownloadVideo(json::parse(R"({"jobId": "abc-123"})"), &out, &err));
    CHECK(out.job_id == "abc-123");
    CHECK(out.joined == false);
    CHECK(nzltx::DownloadVideoPath(out) == "/jobs/abc-123/video");
    CHECK(nzltx::DownloadFileName(out) == "abc-123.mp4");
}

TEST_CASE("ParseDownloadVideo honours joined=true") {
    DownloadVideoRequest out;
    std::string err;
    CHECK(nzltx::ParseDownloadVideo(json::parse(R"({"jobId": "j", "joined": true})"),
                                    &out, &err));
    CHECK(out.joined == true);
    CHECK(nzltx::DownloadVideoPath(out) == "/jobs/j/joined");
    CHECK(nzltx::DownloadFileName(out) == "j_joined.mp4");
}

TEST_CASE("ParseDownloadVideo rejects missing id and path-traversal ids") {
    DownloadVideoRequest out;
    std::string err;
    CHECK_FALSE(nzltx::ParseDownloadVideo(json::parse(R"({})"), &out, &err));
    CHECK_FALSE(
        nzltx::ParseDownloadVideo(json::parse(R"({"jobId": "../x"})"), &out, &err));
    CHECK_FALSE(
        nzltx::ParseDownloadVideo(json::parse(R"({"jobId": "a/b"})"), &out, &err));
    CHECK_FALSE(nzltx::ParseDownloadVideo(
        json::parse(R"({"jobId": "a\\b"})"), &out, &err));
    CHECK_FALSE(nzltx::ParseDownloadVideo(
        json::parse(R"({"jobId": "j", "joined": "yes"})"), &out, &err));
}

// --- backend.downloadVideo contract v6 extension (destDir/fileName/noClobber) -

TEST_CASE("ParseDownloadVideo with only legacy params leaves the v6 fields unset "
         "(backward compatibility)") {
    DownloadVideoRequest out;
    std::string err;
    CHECK(nzltx::ParseDownloadVideo(json::parse(R"({"jobId": "abc-123"})"), &out, &err));
    CHECK(out.has_dest_dir == false);
    CHECK(out.has_file_name == false);
    CHECK(out.no_clobber == false);
}

TEST_CASE("ParseDownloadVideo accepts destDir/fileName/noClobber") {
    DownloadVideoRequest out;
    std::string err;
    CHECK(nzltx::ParseDownloadVideo(
        json::parse(R"({"jobId": "abc-123", "destDir": "C:\\out",
                        "fileName": "clip.mp4", "noClobber": true})"),
        &out, &err));
    CHECK(out.has_dest_dir == true);
    CHECK(out.dest_dir == "C:\\out");
    CHECK(out.has_file_name == true);
    CHECK(out.file_name == "clip.mp4");
    CHECK(out.no_clobber == true);
}

TEST_CASE("ParseDownloadVideo rejects empty/wrong-typed destDir, fileName, noClobber") {
    DownloadVideoRequest out;
    std::string err;
    CHECK_FALSE(nzltx::ParseDownloadVideo(
        json::parse(R"({"jobId": "j", "destDir": ""})"), &out, &err));
    CHECK_FALSE(nzltx::ParseDownloadVideo(
        json::parse(R"({"jobId": "j", "destDir": 1})"), &out, &err));
    CHECK_FALSE(nzltx::ParseDownloadVideo(
        json::parse(R"({"jobId": "j", "fileName": ""})"), &out, &err));
    CHECK_FALSE(nzltx::ParseDownloadVideo(
        json::parse(R"({"jobId": "j", "fileName": 1})"), &out, &err));
    CHECK_FALSE(nzltx::ParseDownloadVideo(
        json::parse(R"({"jobId": "j", "noClobber": "yes"})"), &out, &err));
}

TEST_CASE("ParseDownloadVideo treats null destDir/fileName/noClobber as absent") {
    DownloadVideoRequest out;
    std::string err;
    CHECK(nzltx::ParseDownloadVideo(
        json::parse(R"({"jobId": "j", "destDir": null, "fileName": null,
                        "noClobber": null})"),
        &out, &err));
    CHECK(out.has_dest_dir == false);
    CHECK(out.has_file_name == false);
    CHECK(out.no_clobber == false);
}

TEST_CASE("ParseDownloadVideo with only legacy params leaves reuseIfPresent unset") {
    DownloadVideoRequest out;
    std::string err;
    CHECK(nzltx::ParseDownloadVideo(json::parse(R"({"jobId": "abc-123"})"), &out, &err));
    CHECK(out.reuse_if_present == false);
}

TEST_CASE("ParseDownloadVideo honours reuseIfPresent=true and treats null/absent as false") {
    DownloadVideoRequest out;
    std::string err;
    CHECK(nzltx::ParseDownloadVideo(
        json::parse(R"({"jobId": "j", "reuseIfPresent": true})"), &out, &err));
    CHECK(out.reuse_if_present == true);
    CHECK(nzltx::ParseDownloadVideo(
        json::parse(R"({"jobId": "j", "reuseIfPresent": null})"), &out, &err));
    CHECK(out.reuse_if_present == false);
}

TEST_CASE("ParseDownloadVideo rejects a wrong-typed reuseIfPresent") {
    DownloadVideoRequest out;
    std::string err;
    CHECK_FALSE(nzltx::ParseDownloadVideo(
        json::parse(R"({"jobId": "j", "reuseIfPresent": "yes"})"), &out, &err));
}

// ---------------------------------------------------------------------------
// backend.uploadFile parsing / content-type / multipart framing
// ---------------------------------------------------------------------------

TEST_CASE("ParseUploadFile accepts each valid kind and a file path") {
    for (const char* kind : {"image", "video", "audio"}) {
        UploadFileRequest out;
        std::string err;
        const json params = {{"kind", kind}, {"filePath", "C:\\tmp\\x.png"}};
        CHECK(nzltx::ParseUploadFile(params, &out, &err));
        CHECK(out.kind == kind);
        CHECK(out.file_path == "C:\\tmp\\x.png");
    }
}

TEST_CASE("ParseUploadFile rejects bad kind / missing or empty file path") {
    UploadFileRequest out;
    std::string err;
    CHECK_FALSE(nzltx::ParseUploadFile(
        json::parse(R"({"kind": "thumbnail", "filePath": "a.png"})"), &out, &err));
    CHECK_FALSE(
        nzltx::ParseUploadFile(json::parse(R"({"filePath": "a.png"})"), &out, &err));
    CHECK_FALSE(
        nzltx::ParseUploadFile(json::parse(R"({"kind": "image"})"), &out, &err));
    CHECK_FALSE(nzltx::ParseUploadFile(
        json::parse(R"({"kind": "image", "filePath": ""})"), &out, &err));
    CHECK_FALSE(nzltx::ParseUploadFile(json::parse(R"([])"), &out, &err));
}

// ---------------------------------------------------------------------------
// Contract v10: the shared query helpers + backend.uploadFile's 'query'.
//
// The overriding requirement is that an upload WITHOUT a query keeps producing
// the exact URL it produced before v10 existed (D4 + D8), because that is what
// makes "no trim requested" byte-equivalent to the pre-trim behaviour.
// ---------------------------------------------------------------------------

TEST_CASE("D1 ParseUploadFile accepts a query object") {
    UploadFileRequest out;
    std::string err;
    const json params = json::parse(
        R"({"kind": "video", "filePath": "C:\\tmp\\a.mp4",
            "query": {"trim_start_sec": "1.500"}})");
    CHECK(nzltx::ParseUploadFile(params, &out, &err));
    CHECK(out.kind == "video");
    CHECK(out.file_path == "C:\\tmp\\a.mp4");
    CHECK(out.query == "trim_start_sec=1.500");
}

TEST_CASE("D2 a non-object query is rejected") {
    std::string err;
    std::string q;
    CHECK_FALSE(nzltx::BuildQueryString(json::parse(R"("a=b")"), &q, &err));
    CHECK(err == "'query' must be an object");
    CHECK(q.empty());
    CHECK_FALSE(nzltx::BuildQueryString(json::parse(R"([1,2])"), &q, &err));
    CHECK_FALSE(nzltx::BuildQueryString(json(nullptr), &q, &err));

    UploadFileRequest out;
    CHECK_FALSE(nzltx::ParseUploadFile(
        json::parse(R"({"kind": "video", "filePath": "a.mp4", "query": "x=1"})"),
        &out, &err));
}

TEST_CASE("D3 a non-string query value is rejected") {
    std::string err;
    std::string q;
    CHECK_FALSE(nzltx::BuildQueryString(json::parse(R"({"k": 1})"), &q, &err));
    CHECK(err == "'query' values must be strings");
    CHECK_FALSE(nzltx::BuildQueryString(json::parse(R"({"k": null})"), &q, &err));
    CHECK_FALSE(nzltx::BuildQueryString(json::parse(R"({"k": true})"), &q, &err));

    UploadFileRequest out;
    CHECK_FALSE(nzltx::ParseUploadFile(
        json::parse(R"({"kind": "video", "filePath": "a.mp4", "query": {"k": 2}})"),
        &out, &err));
}

TEST_CASE("D4 an upload without a query leaves UploadFileRequest.query empty") {
    UploadFileRequest out;
    std::string err;
    // Pre-dirty the output so a missing explicit clear would be caught.
    out.query = "stale=1";
    const json params = json::parse(R"({"kind": "video", "filePath": "a.mp4"})");
    CHECK(nzltx::ParseUploadFile(params, &out, &err));
    CHECK(out.query.empty());
    // ... and therefore the assembled URL is exactly the pre-v10 one.
    const std::string base = "http://127.0.0.1:18620/api/v1/upload/video";
    CHECK(nzltx::AppendQueryToUrl(base, out.query) == base);
}

TEST_CASE("D5 BuildQueryString percent-encodes both keys and values") {
    std::string err;
    std::string q;
    CHECK(nzltx::BuildQueryString(json::parse(R"({"a b": "x/y"})"), &q, &err));
    CHECK(q == "a%20b=x%2Fy");
    CHECK(nzltx::BuildQueryString(json::parse(R"({"k": "v"})"), &q, &err));
    CHECK(q == "k=v");
    // Unreserved characters pass through untouched; '&' and '=' do not.
    CHECK(nzltx::BuildQueryString(json::parse(R"({"a-b_c.d~e": "1&2=3"})"), &q, &err));
    CHECK(q == "a-b_c.d~e=1%262%3D3");
    // An empty object yields an empty (URL-neutral) query.
    CHECK(nzltx::BuildQueryString(json::parse(R"({})"), &q, &err));
    CHECK(q.empty());
}

TEST_CASE("D6 a null 'query' is treated as no query at all") {
    UploadFileRequest up;
    std::string err;
    up.query = "stale=1";
    CHECK(nzltx::ParseUploadFile(
        json::parse(R"({"kind": "video", "filePath": "a.mp4", "query": null})"), &up,
        &err));
    CHECK(up.query.empty());

    BackendRequest req;
    CHECK(nzltx::ParseBackendRequest(
        json::parse(R"({"method": "GET", "path": "/x", "query": null})"), &req, &err));
    CHECK(req.query.empty());
    CHECK(nzltx::BuildBackendUrl("http://h", req) == "http://h/x");
}

TEST_CASE("D7 query rejections keep their historical method-prefixed wording") {
    std::string err;
    BackendRequest req;
    CHECK_FALSE(nzltx::ParseBackendRequest(
        json::parse(R"({"method": "GET", "path": "/x", "query": "a=b"})"), &req, &err));
    CHECK(err == "backend.request 'query' must be an object");
    CHECK_FALSE(nzltx::ParseBackendRequest(
        json::parse(R"({"method": "GET", "path": "/x", "query": {"k": 1}})"), &req,
        &err));
    CHECK(err == "backend.request 'query' values must be strings");

    UploadFileRequest up;
    CHECK_FALSE(nzltx::ParseUploadFile(
        json::parse(R"({"kind": "video", "filePath": "a.mp4", "query": 5})"), &up,
        &err));
    CHECK(err == "backend.uploadFile 'query' must be an object");
    CHECK_FALSE(nzltx::ParseUploadFile(
        json::parse(R"({"kind": "video", "filePath": "a.mp4", "query": {"k": []}})"),
        &up, &err));
    CHECK(err == "backend.uploadFile 'query' values must be strings");
}

TEST_CASE("D8 AppendQueryToUrl returns the URL completely unchanged on an empty query") {
    const char* urls[] = {"http://127.0.0.1:18620/api/v1/upload/video",
                          "http://h/x?already=here", "", "/relative/path"};
    for (const char* u : urls) {
        CHECK(nzltx::AppendQueryToUrl(u, "") == std::string(u));
    }
}

TEST_CASE("D9 AppendQueryToUrl joins with '?' and BuildQueryString joins with '&'") {
    CHECK(nzltx::AppendQueryToUrl("http://h/upload/video", "a=1") ==
          "http://h/upload/video?a=1");
    std::string err;
    std::string q;
    CHECK(nzltx::BuildQueryString(
        json::parse(R"({"trim_start_sec": "1.500", "trim_duration_sec": "2.000"})"), &q,
        &err));
    // nlohmann's default object is sorted by key, so the order is deterministic.
    CHECK(q == "trim_duration_sec=2.000&trim_start_sec=1.500");
    CHECK(nzltx::AppendQueryToUrl("http://h/api/v1/upload/video", q) ==
          "http://h/api/v1/upload/video?trim_duration_sec=2.000&trim_start_sec=1.500");
}

TEST_CASE("UploadPath maps a kind to its endpoint tail") {
    CHECK(nzltx::UploadPath("image") == "/upload/image");
    CHECK(nzltx::UploadPath("video") == "/upload/video");
    CHECK(nzltx::UploadPath("audio") == "/upload/audio");
}

TEST_CASE("FileNameFromPath returns the last component for both separators") {
    CHECK(nzltx::FileNameFromPath("C:\\a\\b\\frame_10.png") == "frame_10.png");
    CHECK(nzltx::FileNameFromPath("/var/tmp/clip.mp4") == "clip.mp4");
    CHECK(nzltx::FileNameFromPath("bare.wav") == "bare.wav");
    CHECK(nzltx::FileNameFromPath("").empty());
}

TEST_CASE("ContentTypeForExtension covers the v3 mapping (case-insensitive)") {
    CHECK(nzltx::ContentTypeForExtension("a.png") == "image/png");
    CHECK(nzltx::ContentTypeForExtension("a.PNG") == "image/png");
    CHECK(nzltx::ContentTypeForExtension("a.jpg") == "image/jpeg");
    CHECK(nzltx::ContentTypeForExtension("a.jpeg") == "image/jpeg");
    CHECK(nzltx::ContentTypeForExtension("a.webp") == "image/webp");
    CHECK(nzltx::ContentTypeForExtension("a.mp4") == "video/mp4");
    CHECK(nzltx::ContentTypeForExtension("a.mov") == "video/quicktime");
    CHECK(nzltx::ContentTypeForExtension("a.webm") == "video/webm");
    CHECK(nzltx::ContentTypeForExtension("a.mkv") == "video/x-matroska");
    CHECK(nzltx::ContentTypeForExtension("a.wav") == "audio/wav");
    CHECK(nzltx::ContentTypeForExtension("a.mp3") == "audio/mpeg");
    CHECK(nzltx::ContentTypeForExtension("a.m4a") == "audio/mp4");
    CHECK(nzltx::ContentTypeForExtension("a.aac") == "audio/aac");
    CHECK(nzltx::ContentTypeForExtension("a.flac") == "audio/flac");
    CHECK(nzltx::ContentTypeForExtension("a.ogg") == "audio/ogg");
    CHECK(nzltx::ContentTypeForExtension("a.bin") == "application/octet-stream");
    CHECK(nzltx::ContentTypeForExtension("noext") == "application/octet-stream");
    // A dot only in a directory component is not an extension.
    CHECK(nzltx::ContentTypeForExtension("C:\\a.dir\\file") ==
          "application/octet-stream");
}

TEST_CASE("BuildMultipartHeader/Footer produce the exact RFC-7578 framing") {
    const std::string boundary = "TestBoundary123";
    const std::string header = nzltx::BuildMultipartHeader(boundary, "file",
                                                           "frame_10.png", "image/png");
    CHECK(header ==
          "--TestBoundary123\r\n"
          "Content-Disposition: form-data; name=\"file\"; "
          "filename=\"frame_10.png\"\r\n"
          "Content-Type: image/png\r\n\r\n");
    CHECK(nzltx::BuildMultipartFooter(boundary) == "\r\n--TestBoundary123--\r\n");
    CHECK(nzltx::MultipartContentType(boundary) ==
          "multipart/form-data; boundary=TestBoundary123");
}

// ---------------------------------------------------------------------------
// timeline.captureFrame parsing
// ---------------------------------------------------------------------------

TEST_CASE("ParseCaptureFrame treats an absent or empty params as no frame") {
    CaptureFrameRequest out;
    std::string err;
    CHECK(nzltx::ParseCaptureFrame(json::object(), &out, &err));
    CHECK(out.has_frame == false);
    CHECK(nzltx::ParseCaptureFrame(json::parse("null"), &out, &err));
    CHECK(out.has_frame == false);
}

TEST_CASE("ParseCaptureFrame reads an explicit integer frame") {
    CaptureFrameRequest out;
    std::string err;
    CHECK(nzltx::ParseCaptureFrame(json::parse(R"({"frame": 42})"), &out, &err));
    CHECK(out.has_frame == true);
    CHECK(out.frame == 42);
}

TEST_CASE("ParseCaptureFrame rejects a non-integer frame") {
    CaptureFrameRequest out;
    std::string err;
    CHECK_FALSE(nzltx::ParseCaptureFrame(json::parse(R"({"frame": 1.5})"), &out, &err));
    CHECK_FALSE(
        nzltx::ParseCaptureFrame(json::parse(R"({"frame": "10"})"), &out, &err));
}

TEST_CASE("CaptureFileName combines the frame number with a unique suffix") {
    CHECK(nzltx::CaptureFileName(10, "20260707_120000_001") ==
          "frame_10_20260707_120000_001.png");
}

// ---------------------------------------------------------------------------
// ui.pickFile parsing / filter spec
// ---------------------------------------------------------------------------

TEST_CASE("ParsePickFileKind accepts image/video/audio/imageOrVideo and rejects others") {
    std::string kind;
    std::string err;
    for (const char* k : {"image", "video", "audio", "imageOrVideo"}) {
        const json params = {{"kind", k}};
        CHECK(nzltx::ParsePickFileKind(params, &kind, &err));
        CHECK(kind == k);
    }
    CHECK_FALSE(nzltx::ParsePickFileKind(json::parse(R"({"kind": "document"})"),
                                         &kind, &err));
    CHECK_FALSE(nzltx::ParsePickFileKind(json::parse(R"({})"), &kind, &err));
    CHECK_FALSE(nzltx::ParsePickFileKind(json::parse(R"({"kind": 3})"), &kind, &err));
    CHECK_FALSE(nzltx::ParsePickFileKind(json::parse(R"([])"), &kind, &err));
}

TEST_CASE("PickFileFilter matches the backend allow-list and appends All files") {
    const std::vector<PickFileFilterEntry> img = nzltx::PickFileFilter("image");
    REQUIRE(img.size() == 2);
    CHECK(img[0].pattern == "*.png;*.jpg;*.jpeg;*.webp");
    CHECK(img[0].label.find("*.png;*.jpg;*.jpeg;*.webp") != std::string::npos);
    CHECK(img[1].pattern == "*.*");
    CHECK(img[1].label == "All files (*.*)");

    const std::vector<PickFileFilterEntry> vid = nzltx::PickFileFilter("video");
    REQUIRE(vid.size() == 2);
    CHECK(vid[0].pattern == "*.mp4;*.mov;*.webm;*.mkv");
    CHECK(vid[1].pattern == "*.*");

    const std::vector<PickFileFilterEntry> aud = nzltx::PickFileFilter("audio");
    REQUIRE(aud.size() == 2);
    CHECK(aud[0].pattern == "*.wav;*.mp3;*.m4a;*.aac;*.flac;*.ogg");
    CHECK(aud[1].pattern == "*.*");

    // Unknown kind still yields the All-files catch-all only.
    const std::vector<PickFileFilterEntry> other = nzltx::PickFileFilter("other");
    REQUIRE(other.size() == 1);
    CHECK(other[0].pattern == "*.*");
}

TEST_CASE("PickFileFilter for imageOrVideo leads with a combined filter, then image-only/video-only/All files") {
    const std::vector<PickFileFilterEntry> combo = nzltx::PickFileFilter("imageOrVideo");
    REQUIRE(combo.size() == 4);
    CHECK(combo[0].pattern == "*.png;*.jpg;*.jpeg;*.webp;*.mp4;*.mov;*.webm;*.mkv");
    CHECK(combo[0].label.find(combo[0].pattern) != std::string::npos);
    CHECK(combo[1].pattern == "*.png;*.jpg;*.jpeg;*.webp");
    CHECK(combo[2].pattern == "*.mp4;*.mov;*.webm;*.mkv");
    CHECK(combo[3].pattern == "*.*");
    CHECK(combo[3].label == "All files (*.*)");
}

// ---------------------------------------------------------------------------
// ui.makeThumbnail parsing / sizing / base64 / data URL
// ---------------------------------------------------------------------------

TEST_CASE("ParseMakeThumbnail requires a filePath and defaults maxDim to 256") {
    MakeThumbnailRequest out;
    std::string err;
    CHECK(nzltx::ParseMakeThumbnail(json::parse(R"({"filePath": "C:\\a\\x.png"})"),
                                    &out, &err));
    CHECK(out.file_path == "C:\\a\\x.png");
    CHECK(out.max_dim == 256);

    CHECK_FALSE(nzltx::ParseMakeThumbnail(json::parse(R"({})"), &out, &err));
    CHECK_FALSE(
        nzltx::ParseMakeThumbnail(json::parse(R"({"filePath": ""})"), &out, &err));
    CHECK_FALSE(nzltx::ParseMakeThumbnail(json::parse(R"([])"), &out, &err));
}

TEST_CASE("ParseMakeThumbnail honours maxDim and clamps into [16, 1024]") {
    MakeThumbnailRequest out;
    std::string err;
    CHECK(nzltx::ParseMakeThumbnail(
        json::parse(R"({"filePath": "a.png", "maxDim": 128})"), &out, &err));
    CHECK(out.max_dim == 128);

    CHECK(nzltx::ParseMakeThumbnail(
        json::parse(R"({"filePath": "a.png", "maxDim": 4})"), &out, &err));
    CHECK(out.max_dim == 16);  // clamped up to the minimum

    CHECK(nzltx::ParseMakeThumbnail(
        json::parse(R"({"filePath": "a.png", "maxDim": 5000})"), &out, &err));
    CHECK(out.max_dim == 1024);  // clamped down to the maximum

    CHECK_FALSE(nzltx::ParseMakeThumbnail(
        json::parse(R"({"filePath": "a.png", "maxDim": "big"})"), &out, &err));
}

TEST_CASE("ComputeThumbnailSize preserves aspect ratio and never upscales") {
    int w = 0;
    int h = 0;
    // Landscape downscale: long edge -> max_dim.
    nzltx::ComputeThumbnailSize(1920, 1080, 256, &w, &h);
    CHECK(w == 256);
    CHECK(h == 144);
    // Portrait downscale.
    nzltx::ComputeThumbnailSize(1080, 1920, 256, &w, &h);
    CHECK(w == 144);
    CHECK(h == 256);
    // Already small enough: 1:1, no upscale.
    nzltx::ComputeThumbnailSize(100, 80, 256, &w, &h);
    CHECK(w == 100);
    CHECK(h == 80);
    // Square exactly at the limit.
    nzltx::ComputeThumbnailSize(256, 256, 256, &w, &h);
    CHECK(w == 256);
    CHECK(h == 256);
    // Extreme aspect ratio keeps the short edge at least 1.
    nzltx::ComputeThumbnailSize(1000, 3, 256, &w, &h);
    CHECK(w == 256);
    CHECK(h == 1);
    // Degenerate inputs -> 0x0.
    nzltx::ComputeThumbnailSize(0, 10, 256, &w, &h);
    CHECK(w == 0);
    CHECK(h == 0);
}

TEST_CASE("Base64Encode matches known vectors with correct padding") {
    auto enc = [](const std::string& s) {
        return nzltx::Base64Encode(
            reinterpret_cast<const unsigned char*>(s.data()), s.size());
    };
    CHECK(enc("") == "");
    CHECK(enc("f") == "Zg==");
    CHECK(enc("fo") == "Zm8=");
    CHECK(enc("foo") == "Zm9v");
    CHECK(enc("foob") == "Zm9vYg==");
    CHECK(enc("fooba") == "Zm9vYmE=");
    CHECK(enc("foobar") == "Zm9vYmFy");
    // A byte with the high bit set exercises the full 8-bit range.
    const unsigned char bytes[] = {0x00, 0xFF, 0x10};
    CHECK(nzltx::Base64Encode(bytes, sizeof(bytes)) == "AP8Q");
    CHECK(nzltx::Base64Encode(nullptr, 0) == "");
}

TEST_CASE("MakeDataUrl builds the data: prefix with the mime and base64") {
    CHECK(nzltx::MakeDataUrl("image/jpeg", "Zm9v") == "data:image/jpeg;base64,Zm9v");
    CHECK(std::string(nzltx::kThumbnailMimeType) == "image/jpeg");
}

// ---------------------------------------------------------------------------
// settings.get / settings.set (contract v4)
//
// bridge_core.h keeps HandleRequestJson independent of the real SettingsStore
// (settings.{h,cpp}) via the SettingsGetProvider / SettingsSetProvider
// injection points, so these tests exercise the dispatch logic with a small
// in-memory fake store instead of touching disk.
// ---------------------------------------------------------------------------

namespace {

// A minimal in-memory stand-in for SettingsStore: accepts any candidate
// starting with "http://" (rejects everything else), tracks whether
// settings_set was invoked, and remembers the last accepted value.
struct FakeSettingsStore {
    std::string base_url = "http://127.0.0.1:18620";
    bool set_called = false;

    nzltx::SettingsGetProvider Getter() {
        return [this]() -> std::string { return base_url; };
    }

    nzltx::SettingsSetProvider Setter() {
        return [this](const std::string& candidate) -> nzltx::SettingsSetOutcome {
            set_called = true;
            nzltx::SettingsSetOutcome outcome;
            if (candidate.rfind("http://", 0) != 0) {
                outcome.ok = false;
                outcome.err_message = "baseUrl must start with 'http://'";
                return outcome;
            }
            base_url = candidate;
            outcome.ok = true;
            outcome.base_url = candidate;
            return outcome;
        };
    }
};

}  // namespace

TEST_CASE("settings.get returns the value from the injected provider") {
    FakeSettingsStore fake;
    fake.base_url = "http://192.168.1.5:18620";
    RequestContext ctx = AvailableCtx();
    ctx.settings_get = fake.Getter();

    const std::string resp =
        HandleRequestJson(R"({"id": 30, "method": "settings.get"})", ctx);
    const json j = json::parse(resp);
    CHECK(j["ok"] == true);
    CHECK(j["result"]["baseUrl"] == "http://192.168.1.5:18620");
}

TEST_CASE("settings.get falls back to ctx.base_url when no provider is injected") {
    RequestContext ctx = AvailableCtx();
    ctx.base_url = "http://127.0.0.1:18620";
    const std::string resp =
        HandleRequestJson(R"({"id": 31, "method": "settings.get"})", ctx);
    const json j = json::parse(resp);
    CHECK(j["ok"] == true);
    CHECK(j["result"]["baseUrl"] == "http://127.0.0.1:18620");
}

TEST_CASE("settings.set with a valid baseUrl applies it and echoes the new value") {
    FakeSettingsStore fake;
    RequestContext ctx = AvailableCtx();
    ctx.settings_get = fake.Getter();
    ctx.settings_set = fake.Setter();

    const std::string resp = HandleRequestJson(
        R"({"id": 32, "method": "settings.set", "params": {"baseUrl": "http://127.0.0.1:19999"}})",
        ctx);
    const json j = json::parse(resp);
    CHECK(j["ok"] == true);
    CHECK(j["result"]["baseUrl"] == "http://127.0.0.1:19999");
    CHECK(fake.set_called);
    CHECK(fake.base_url == "http://127.0.0.1:19999");
}

TEST_CASE("settings.set with an invalid baseUrl yields BAD_REQUEST and leaves the value") {
    FakeSettingsStore fake;
    RequestContext ctx = AvailableCtx();
    ctx.settings_get = fake.Getter();
    ctx.settings_set = fake.Setter();

    const std::string resp = HandleRequestJson(
        R"({"id": 33, "method": "settings.set", "params": {"baseUrl": "not-a-url"}})", ctx);
    const json j = json::parse(resp);
    CHECK(j["ok"] == false);
    CHECK(j["error"]["code"] == "BAD_REQUEST");
    CHECK(fake.set_called);
    // The store's value must be unchanged after a rejected candidate.
    CHECK(fake.base_url == "http://127.0.0.1:18620");
}

TEST_CASE("settings.set with a non-string baseUrl yields BAD_REQUEST without calling the provider") {
    FakeSettingsStore fake;
    RequestContext ctx = AvailableCtx();
    ctx.settings_get = fake.Getter();
    ctx.settings_set = fake.Setter();

    const std::string resp = HandleRequestJson(
        R"({"id": 34, "method": "settings.set", "params": {"baseUrl": 123}})", ctx);
    const json j = json::parse(resp);
    CHECK(j["ok"] == false);
    CHECK(j["error"]["code"] == "BAD_REQUEST");
    CHECK_FALSE(fake.set_called);
}

TEST_CASE("settings.set with baseUrl omitted is a no-op that echoes the current value") {
    FakeSettingsStore fake;
    fake.base_url = "http://127.0.0.1:18620";
    RequestContext ctx = AvailableCtx();
    ctx.settings_get = fake.Getter();
    ctx.settings_set = fake.Setter();

    const std::string resp =
        HandleRequestJson(R"({"id": 35, "method": "settings.set", "params": {}})", ctx);
    const json j = json::parse(resp);
    CHECK(j["ok"] == true);
    CHECK(j["result"]["baseUrl"] == "http://127.0.0.1:18620");
    CHECK_FALSE(fake.set_called);
}

TEST_CASE("settings.set with baseUrl explicitly null is also a no-op") {
    FakeSettingsStore fake;
    RequestContext ctx = AvailableCtx();
    ctx.settings_get = fake.Getter();
    ctx.settings_set = fake.Setter();

    const std::string resp = HandleRequestJson(
        R"({"id": 36, "method": "settings.set", "params": {"baseUrl": null}})", ctx);
    const json j = json::parse(resp);
    CHECK(j["ok"] == true);
    CHECK_FALSE(fake.set_called);
}

// ---------------------------------------------------------------------------
// Contract v5 timeline.* methods
//
// Sync methods (getSelection / insertProvisional / resolveProvisional /
// updateProvisionalText / scanProvisionals) are dispatched by HandleRequestJson
// through injected providers; the two async methods (cutoutRange / extractAudio)
// expose only ParseXxx / MakeXxxResult here (the bridge dispatches them, like
// captureFrame). These tests inject in-memory mock providers so no AviUtl2 SDK
// is required.
// ---------------------------------------------------------------------------

using nzltx::AliasMatchesJob;
using nzltx::CutoutRangeRequest;
using nzltx::ExtractAudioRequest;
using nzltx::InsertProvisionalOutcome;
using nzltx::InsertProvisionalParams;
using nzltx::ProvisionalTextAlias;
using nzltx::ReplaceObjectRequest;
using nzltx::ResolveProvisionalParams;
using nzltx::ScannedObject;
using nzltx::SelectionItem;
using nzltx::SelectionSnapshot;
using nzltx::UpdateObjectTextRequest;
using nzltx::UpdateProvisionalTextParams;
// I3 (timeline.updateProvisionalReservation + placement geometry).
using nzltx::PlacementResolveInput;
using nzltx::PlacementResolveResult;
using nzltx::ProvisionalPlacement;
using nzltx::ResolveProvisionalPlacement;
using nzltx::UpdateProvisionalReservationParams;
using nzltx::UpdateReservationOutcome;
using nzltx::UpdateReservationRequest;
// I13 (timeline.deleteProvisionalByJob).
using nzltx::DeleteProvisionalRequest;
// Replace-insert (timeline.insertMediaForJob).
using nzltx::ReplaceMediaForJobOutcome;
using nzltx::ReplaceMediaForJobRequest;

// --- timeline.getSelection --------------------------------------------------

TEST_CASE("getSelection without an edit handle yields NO_EDIT_HANDLE") {
    RequestContext ctx = AvailableCtx();
    ctx.get_selection = []() { return SelectionSnapshot{}; };  // available == false
    const std::string resp =
        HandleRequestJson(R"({"id": 40, "method": "timeline.getSelection"})", ctx);
    const json j = json::parse(resp);
    CHECK(j["ok"] == false);
    CHECK(j["error"]["code"] == "NO_EDIT_HANDLE");
}

TEST_CASE("getSelection formats the snapshot with nullable filePath/objectName") {
    RequestContext ctx = AvailableCtx();
    ctx.get_selection = []() {
        SelectionSnapshot s;
        s.available = true;
        s.has_range = true;
        s.range_start = 10;
        s.range_end = 90;
        s.cursor_frame = 42;
        s.cursor_layer = 3;
        s.rate = 30;
        s.scale = 1;
        s.sample_rate = 48000;
        SelectionItem a;  // a media object: filePath + objectName present
        a.layer = 2;
        a.frame_start = 10;
        a.frame_end = 50;
        a.effect_name = "video file";
        a.has_file_path = true;
        a.file_path = "C:\\tmp\\clip.mp4";
        a.has_object_name = true;
        a.object_name = "NzLTX23#job1";
        a.media_width = 1920;
        a.media_height = 1080;
        SelectionItem b;  // a bare object: both nullable fields absent
        b.layer = 3;
        b.frame_start = 20;
        b.frame_end = 40;
        b.effect_name = "text";
        // media_width/media_height left at their 0 default (no get_media_info hit)
        s.selected = {a, b};
        return s;
    };
    const std::string resp =
        HandleRequestJson(R"({"id": 41, "method": "timeline.getSelection"})", ctx);
    const json j = json::parse(resp);
    REQUIRE(j["ok"] == true);
    const json& r = j["result"];
    CHECK(r["hasRange"] == true);
    CHECK(r["rangeStart"] == 10);
    CHECK(r["rangeEnd"] == 90);
    CHECK(r["cursorFrame"] == 42);
    CHECK(r["cursorLayer"] == 3);
    CHECK(r["rate"] == 30);
    CHECK(r["scale"] == 1);
    CHECK(r["sampleRate"] == 48000);
    REQUIRE(r["selected"].is_array());
    REQUIRE(r["selected"].size() == 2);
    CHECK(r["selected"][0]["layer"] == 2);
    CHECK(r["selected"][0]["frameStart"] == 10);
    CHECK(r["selected"][0]["frameEnd"] == 50);
    CHECK(r["selected"][0]["effectName"] == "video file");
    CHECK(r["selected"][0]["filePath"] == "C:\\tmp\\clip.mp4");
    CHECK(r["selected"][0]["objectName"] == "NzLTX23#job1");
    CHECK(r["selected"][0]["mediaWidth"] == 1920);
    CHECK(r["selected"][0]["mediaHeight"] == 1080);
    // The bare object's nullable fields serialize as JSON null.
    CHECK(r["selected"][1]["filePath"].is_null());
    CHECK(r["selected"][1]["objectName"].is_null());
    // mediaWidth/mediaHeight are NOT nullable: unknown resolution serializes
    // as plain 0, not JSON null.
    CHECK(r["selected"][1]["mediaWidth"] == 0);
    CHECK(r["selected"][1]["mediaHeight"] == 0);
    CHECK_FALSE(r["selected"][1]["mediaWidth"].is_null());
}

TEST_CASE("getSelection defaults mediaWidth/mediaHeight to 0 when unset") {
    RequestContext ctx = AvailableCtx();
    ctx.get_selection = []() {
        SelectionSnapshot s;
        s.available = true;
        SelectionItem it;  // media_width/media_height left default-constructed
        s.selected = {it};
        return s;
    };
    const std::string resp =
        HandleRequestJson(R"({"id": 42, "method": "timeline.getSelection"})", ctx);
    const json j = json::parse(resp);
    REQUIRE(j["ok"] == true);
    const json& r = j["result"];
    REQUIRE(r["selected"].size() == 1);
    CHECK(r["selected"][0]["mediaWidth"] == 0);
    CHECK(r["selected"][0]["mediaHeight"] == 0);
}

TEST_CASE("getSelection serializes textContent (nullable) and mediaDurationSec") {
    RequestContext ctx = AvailableCtx();
    ctx.get_selection = []() {
        SelectionSnapshot s;
        s.available = true;
        SelectionItem media;  // a video: no text body, real duration
        media.effect_name = "video file";
        media.has_file_path = true;
        media.file_path = "C:\\tmp\\clip.mp4";
        media.media_width = 1920;
        media.media_height = 1080;
        media.media_duration_sec = 4.5;
        SelectionItem text;  // a text object: body present, no duration
        text.effect_name = "text";
        text.has_text_content = true;
        text.text_content = "hello world";
        s.selected = {media, text};
        return s;
    };
    const std::string resp =
        HandleRequestJson(R"({"id": 43, "method": "timeline.getSelection"})", ctx);
    const json j = json::parse(resp);
    REQUIRE(j["ok"] == true);
    const json& r = j["result"];
    REQUIRE(r["selected"].size() == 2);
    // Media object: textContent null (has_text_content false), duration present.
    CHECK(r["selected"][0]["textContent"].is_null());
    CHECK(r["selected"][0]["mediaDurationSec"] == 4.5);
    // Text object: textContent carries the body, mediaDurationSec is a plain 0
    // (never null) since there is no media file to probe.
    CHECK(r["selected"][1]["textContent"] == "hello world");
    CHECK(r["selected"][1]["mediaDurationSec"] == 0);
    CHECK_FALSE(r["selected"][1]["mediaDurationSec"].is_null());
}

TEST_CASE("getSelection defaults textContent to null and mediaDurationSec to 0") {
    RequestContext ctx = AvailableCtx();
    ctx.get_selection = []() {
        SelectionSnapshot s;
        s.available = true;
        SelectionItem it;  // all new fields left default-constructed
        s.selected = {it};
        return s;
    };
    const std::string resp =
        HandleRequestJson(R"({"id": 44, "method": "timeline.getSelection"})", ctx);
    const json j = json::parse(resp);
    REQUIRE(j["ok"] == true);
    const json& r = j["result"];
    REQUIRE(r["selected"].size() == 1);
    CHECK(r["selected"][0]["textContent"].is_null());
    CHECK(r["selected"][0]["mediaDurationSec"] == 0);
    CHECK_FALSE(r["selected"][0]["mediaDurationSec"].is_null());
}

// --- contract v10: the playback range / speed / loop / section fields --------
// The raw-value shapes below are the ones actually captured on real hardware on
// 2026-08-01 (Docs/V2V_RIBBON_TRIM_WORKORDER.md section 4): a 4-field CSV whose
// first two fields are SECONDS on the source axis, with 3 decimals.

TEST_CASE("ParsePlaybackRange reads the first two fields of the real 4-field CSV") {
    double start = -1.0;
    double end = -1.0;

    // An untouched ribbon: the whole source, written out explicitly.
    REQUIRE(nzltx::ParsePlaybackRange("0.000,10.700,\xe5\x86\x8d\xe7\x94\x9f\xe7\xaf\x84\xe5\x9b\xb2,0",
                                      &start, &end));
    CHECK(start == doctest::Approx(0.0));
    CHECK(end == doctest::Approx(10.7));

    // Two seconds trimmed off the head: the FIRST field moves.
    REQUIRE(nzltx::ParsePlaybackRange("2.000,10.700,\xe5\x86\x8d\xe7\x94\x9f\xe7\xaf\x84\xe5\x9b\xb2,0",
                                      &start, &end));
    CHECK(start == doctest::Approx(2.0));
    CHECK(end == doctest::Approx(10.7));

    // The R4 case: the window is SHORTER than the ribbon (start moved, end did
    // not), which is what forces the webui's min(spanSec, end - start) clamp.
    REQUIRE(nzltx::ParsePlaybackRange("3.333,10.700,\xe5\x86\x8d\xe7\x94\x9f\xe7\xaf\x84\xe5\x9b\xb2,0",
                                      &start, &end));
    CHECK(start == doctest::Approx(3.333));
    CHECK(end == doctest::Approx(10.7));
}

TEST_CASE("ParsePlaybackRange ignores everything past the second field") {
    double start = 0.0;
    double end = 0.0;
    // Exactly two fields is enough...
    REQUIRE(nzltx::ParsePlaybackRange("1.500,4.250", &start, &end));
    CHECK(start == doctest::Approx(1.5));
    CHECK(end == doctest::Approx(4.25));
    // ...and so is any number of trailing fields, whatever the mode is called.
    // (The third field is a LOCALIZED name; a future/English build must not
    // break the parse.)
    REQUIRE(nzltx::ParsePlaybackRange("1.500,4.250,play range,0", &start, &end));
    CHECK(start == doctest::Approx(1.5));
    CHECK(end == doctest::Approx(4.25));
    REQUIRE(nzltx::ParsePlaybackRange("1.500,4.250,mode,0,extra,fields", &start, &end));
    CHECK(start == doctest::Approx(1.5));
    CHECK(end == doctest::Approx(4.25));
    // Surrounding whitespace is tolerated (a value read straight off an alias
    // line can carry a trailing space / CR).
    REQUIRE(nzltx::ParsePlaybackRange(" 1.500 , 4.250 ,mode,0\r", &start, &end));
    CHECK(start == doctest::Approx(1.5));
    CHECK(end == doctest::Approx(4.25));
}

TEST_CASE("ParsePlaybackRange rejects anything it cannot read with certainty") {
    double start = 111.0;
    double end = 222.0;
    const auto rejected = [&](const char* raw) {
        CHECK_FALSE(nzltx::ParsePlaybackRange(raw, &start, &end));
        // The outputs must be untouched, so a failed parse can never leak a
        // half-written window into SelectionItem.
        CHECK(start == doctest::Approx(111.0));
        CHECK(end == doctest::Approx(222.0));
    };
    rejected("");                     // empty
    rejected("2.000");                // one field only
    rejected("2.000,");               // second field empty
    rejected(",10.700");              // first field empty
    rejected("abc,10.700");           // non-numeric
    rejected("2.000,xyz");            // non-numeric
    rejected("2.0x,10.700");          // trailing junk (never silently truncated)
    rejected("-1.000,10.700");        // negative start
    rejected("10.700,2.000");         // end before start
    CHECK_FALSE(nzltx::ParsePlaybackRange("1.0,2.0", nullptr, &end));
    CHECK_FALSE(nzltx::ParsePlaybackRange("1.0,2.0", &start, nullptr));
}

TEST_CASE("ParsePlaybackRange accepts a zero-length window (end == start)") {
    double start = 0.0;
    double end = 0.0;
    REQUIRE(nzltx::ParsePlaybackRange("4.000,4.000,mode,0", &start, &end));
    CHECK(start == doctest::Approx(4.0));
    CHECK(end == doctest::Approx(4.0));
}

TEST_CASE("ParsePlaybackSpeedPercent normalizes the percentage to a 1.0 scale") {
    double speed = 0.0;
    REQUIRE(nzltx::ParsePlaybackSpeedPercent("100.00", &speed));
    CHECK(speed == doctest::Approx(1.0));
    REQUIRE(nzltx::ParsePlaybackSpeedPercent("200.00", &speed));
    CHECK(speed == doctest::Approx(2.0));
    REQUIRE(nzltx::ParsePlaybackSpeedPercent("50.00", &speed));
    CHECK(speed == doctest::Approx(0.5));

    speed = 1.0;
    CHECK_FALSE(nzltx::ParsePlaybackSpeedPercent("", &speed));
    CHECK_FALSE(nzltx::ParsePlaybackSpeedPercent("fast", &speed));
    CHECK_FALSE(nzltx::ParsePlaybackSpeedPercent("100%", &speed));
    CHECK(speed == doctest::Approx(1.0));  // untouched on failure
}

TEST_CASE("getSelection serializes the v10 playback fields") {
    RequestContext ctx = AvailableCtx();
    ctx.get_selection = []() {
        SelectionSnapshot s;
        s.available = true;
        SelectionItem trimmed;  // a ribbon that plays 2.0s..10.7s of its source
        trimmed.effect_name = "video file";
        trimmed.playback_start_sec = 2.0;
        trimmed.playback_end_sec = 10.7;
        trimmed.has_playback_range = true;
        trimmed.playback_speed = 2.0;
        trimmed.loop_play = true;
        trimmed.section_count = 3;
        s.selected = {trimmed};
        return s;
    };
    const std::string resp =
        HandleRequestJson(R"({"id": 45, "method": "timeline.getSelection"})", ctx);
    const json j = json::parse(resp);
    REQUIRE(j["ok"] == true);
    const json& e = j["result"]["selected"][0];
    CHECK(e["playbackStartSec"] == 2.0);
    CHECK(e["playbackEndSec"] == 10.7);
    CHECK(e["hasPlaybackRange"] == true);
    CHECK(e["playbackSpeed"] == 2.0);
    CHECK(e["loopPlay"] == true);
    CHECK(e["sectionCount"] == 3);
}

TEST_CASE("getSelection's v10 defaults are the conservative no-trim ones") {
    RequestContext ctx = AvailableCtx();
    ctx.get_selection = []() {
        SelectionSnapshot s;
        s.available = true;
        SelectionItem it;  // every v10 field left default-constructed
        s.selected = {it};
        return s;
    };
    const std::string resp =
        HandleRequestJson(R"({"id": 46, "method": "timeline.getSelection"})", ctx);
    const json j = json::parse(resp);
    REQUIRE(j["ok"] == true);
    const json& e = j["result"]["selected"][0];
    // hasPlaybackRange false is what makes the webui upload the WHOLE file, i.e.
    // behave exactly as it did before v10.
    CHECK(e["hasPlaybackRange"] == false);
    CHECK(e["playbackStartSec"] == 0.0);
    CHECK(e["playbackEndSec"] == 0.0);
    // Neutral speed / no loop / one section: an object we could not fully
    // inspect must never look like a reason to skip on its own.
    CHECK(e["playbackSpeed"] == 1.0);
    CHECK(e["loopPlay"] == false);
    CHECK(e["sectionCount"] == 1);
    // None of them are nullable (the has-flag carries the "unknown" meaning).
    CHECK_FALSE(e["playbackStartSec"].is_null());
    CHECK_FALSE(e["sectionCount"].is_null());
}

// --- timeline.insertProvisional ---------------------------------------------

TEST_CASE("ParseInsertProvisional accepts a full request and rejects bad params") {
    InsertProvisionalParams out;
    std::string err;
    CHECK(nzltx::ParseInsertProvisional(
        json::parse(R"({"jobId": "j1", "displayText": "hi", "numFrames": 40,
                        "genFps": 30, "placement": "C", "cursorLayer": 2,
                        "cursorFrame": 100})"),
        &out, &err));
    CHECK(out.job_id == "j1");
    CHECK(out.display_text == "hi");
    CHECK(out.num_frames == 40);
    CHECK(out.gen_fps == doctest::Approx(30.0));
    CHECK(out.placement == ProvisionalPlacement::kCursor);
    CHECK(out.has_cursor_frame);
    CHECK(out.cursor_frame == 100);

    // Missing/empty jobId, non-string displayText, non-positive numFrames,
    // missing genFps, and missing placement all fail.
    CHECK_FALSE(nzltx::ParseInsertProvisional(
        json::parse(R"({"displayText": "x", "numFrames": 1, "genFps": 30,
                        "placement": "C", "cursorLayer": 0, "cursorFrame": 0})"),
        &out, &err));
    CHECK_FALSE(nzltx::ParseInsertProvisional(
        json::parse(R"({"jobId": "", "displayText": "x", "numFrames": 1,
                        "genFps": 30, "placement": "C", "cursorLayer": 0,
                        "cursorFrame": 0})"),
        &out, &err));
    CHECK_FALSE(nzltx::ParseInsertProvisional(
        json::parse(R"({"jobId": "j", "displayText": 5, "numFrames": 1,
                        "genFps": 30, "placement": "C", "cursorLayer": 0,
                        "cursorFrame": 0})"),
        &out, &err));
    CHECK_FALSE(nzltx::ParseInsertProvisional(
        json::parse(R"({"jobId": "j", "displayText": "x", "numFrames": 0,
                        "genFps": 30, "placement": "C", "cursorLayer": 0,
                        "cursorFrame": 0})"),
        &out, &err));  // non-positive numFrames
    CHECK_FALSE(nzltx::ParseInsertProvisional(
        json::parse(R"({"jobId": "j", "displayText": "x", "numFrames": 1,
                        "placement": "C", "cursorLayer": 0, "cursorFrame": 0})"),
        &out, &err));  // missing genFps
    CHECK_FALSE(nzltx::ParseInsertProvisional(
        json::parse(R"({"jobId": "j", "displayText": "x", "numFrames": 1,
                        "genFps": 30})"),
        &out, &err));  // missing placement
    CHECK_FALSE(nzltx::ParseInsertProvisional(json::parse(R"([])"), &out, &err));
}

TEST_CASE("ParseInsertProvisional handles the optional textPrefix (spec 5-5)") {
    InsertProvisionalParams out;
    std::string err;
    // Absent -> default (empty) prefix.
    REQUIRE(nzltx::ParseInsertProvisional(
        json::parse(R"({"jobId": "j1", "displayText": "hi", "numFrames": 40,
                        "genFps": 30, "placement": "C", "cursorLayer": 2,
                        "cursorFrame": 100})"),
        &out, &err));
    CHECK(out.text_prefix.empty());
    // Present string -> captured verbatim.
    REQUIRE(nzltx::ParseInsertProvisional(
        json::parse(R"({"jobId": "j1", "displayText": "hi", "numFrames": 40,
                        "genFps": 30, "placement": "C", "cursorLayer": 2,
                        "cursorFrame": 100, "textPrefix": "RSV: "})"),
        &out, &err));
    CHECK(out.text_prefix == "RSV: ");
    // Present but non-string -> parse failure.
    CHECK_FALSE(nzltx::ParseInsertProvisional(
        json::parse(R"({"jobId": "j1", "displayText": "hi", "numFrames": 40,
                        "genFps": 30, "placement": "C", "cursorLayer": 2,
                        "cursorFrame": 100, "textPrefix": 5})"),
        &out, &err));
}

TEST_CASE("ParseUpdateProvisionalReservation handles the optional textPrefix") {
    UpdateProvisionalReservationParams out;
    std::string err;
    REQUIRE(nzltx::ParseUpdateProvisionalReservation(
        json::parse(R"({"oldJobId": "", "newJobId": "n1", "displayText": "hi",
                        "numFrames": 40, "genFps": 30, "placement": "A",
                        "materialLayer": 2, "materialFrameEnd": 130})"),
        &out, &err));
    CHECK(out.text_prefix.empty());
    REQUIRE(nzltx::ParseUpdateProvisionalReservation(
        json::parse(R"({"oldJobId": "", "newJobId": "n1", "displayText": "hi",
                        "numFrames": 40, "genFps": 30, "placement": "A",
                        "materialLayer": 2, "materialFrameEnd": 130,
                        "textPrefix": "RSV: "})"),
        &out, &err));
    CHECK(out.text_prefix == "RSV: ");
    CHECK_FALSE(nzltx::ParseUpdateProvisionalReservation(
        json::parse(R"({"oldJobId": "", "newJobId": "n1", "displayText": "hi",
                        "numFrames": 40, "genFps": 30, "placement": "A",
                        "materialLayer": 2, "materialFrameEnd": 130,
                        "textPrefix": 5})"),
        &out, &err));
}

TEST_CASE("BuildProvisionalPlaceholder embeds the job id and pins the length") {
    const ProvisionalTextAlias built =
        nzltx::BuildProvisionalPlaceholder("job-XYZ", "a nice prompt", 37);
    // object_name is the exact-match search key.
    CHECK(built.object_name == "NzLTX23#job-XYZ");
    // The alias round-trips through the provisional re-discovery predicate.
    CHECK(AliasMatchesJob(built.alias, "job-XYZ"));
    CHECK_FALSE(AliasMatchesJob(built.alias, "job-XY"));  // no prefix collision
    // The top [Object] frame header is pinned to frame=0,<length>.
    CHECK(built.alias.find("frame=0,37") != std::string::npos);

    // With an explicit stage-1 (ASCII) prefix, the label leads with it yet the
    // job id remains embedded/re-discoverable (double-tag preserved, spec 5-5).
    const ProvisionalTextAlias reserved =
        nzltx::BuildProvisionalPlaceholder("job-XYZ", "a nice prompt", 37, "RSV: ");
    CHECK(reserved.alias.find("RSV: a nice prompt") != std::string::npos);
    CHECK(reserved.object_name == "NzLTX23#job-XYZ");
    CHECK(AliasMatchesJob(reserved.alias, "job-XYZ"));
}

TEST_CASE("insertProvisional builds the alias, calls the provider and echoes name") {
    std::string captured_alias;
    std::string captured_name;
    int captured_layer = -1;
    int captured_frame = -1;
    int captured_length = -1;
    RequestContext ctx = AvailableCtx();
    ctx.insert_provisional = [&](const std::string& alias, const std::string& name,
                                 int layer, int frame,
                                 int length) -> InsertProvisionalOutcome {
        captured_alias = alias;
        captured_name = name;
        captured_layer = layer;
        captured_frame = frame;
        captured_length = length;
        InsertProvisionalOutcome o;
        o.ok = true;
        o.layer = layer;
        o.frame = frame;
        return o;
    };
    const std::string resp = HandleRequestJson(
        R"({"id": 42, "method": "timeline.insertProvisional",
            "params": {"jobId": "j9", "displayText": "hello", "numFrames": 24,
                       "genFps": 30, "placement": "C", "cursorLayer": 5,
                       "cursorFrame": 200}})",
        ctx);
    const json j = json::parse(resp);
    REQUIRE(j["ok"] == true);
    CHECK(j["result"]["inserted"] == true);
    CHECK(j["result"]["layer"] == 5);
    CHECK(j["result"]["frame"] == 200);
    CHECK(j["result"]["objectName"] == "NzLTX23#j9");
    // The provider received a job-tagged, length-pinned alias.
    CHECK(captured_name == "NzLTX23#j9");
    CHECK(captured_layer == 5);
    CHECK(captured_frame == 200);
    // D2: the resolved length is forwarded to the provider (explicit create len).
    CHECK(captured_length == 24);
    CHECK(AliasMatchesJob(captured_alias, "j9"));
    CHECK(captured_alias.find("frame=0,24") != std::string::npos);
}

TEST_CASE("insertProvisional maps a provider failure to PROVISIONAL_FAILED") {
    RequestContext ctx = AvailableCtx();
    ctx.insert_provisional = [](const std::string&, const std::string&, int, int,
                                int) -> InsertProvisionalOutcome {
        return InsertProvisionalOutcome{};  // ok == false
    };
    const std::string resp = HandleRequestJson(
        R"({"id": 43, "method": "timeline.insertProvisional",
            "params": {"jobId": "j", "displayText": "x", "numFrames": 1,
                       "genFps": 30, "placement": "C", "cursorLayer": 0,
                       "cursorFrame": 0}})",
        ctx);
    const json j = json::parse(resp);
    CHECK(j["ok"] == false);
    CHECK(j["error"]["code"] == "PROVISIONAL_FAILED");
}

TEST_CASE("insertProvisional with bad params yields BAD_REQUEST") {
    const std::string resp = HandleRequestJson(
        R"({"id": 44, "method": "timeline.insertProvisional", "params": {}})",
        AvailableCtx());
    const json j = json::parse(resp);
    CHECK(j["ok"] == false);
    CHECK(j["error"]["code"] == "BAD_REQUEST");
}

// --- timeline.resolveProvisional --------------------------------------------

TEST_CASE("ParseResolveProvisional accepts a full request and rejects bad params") {
    ResolveProvisionalParams out;
    std::string err;
    CHECK(nzltx::ParseResolveProvisional(
        json::parse(R"({"jobId": "j", "videoFilePath": "C:\\v.mp4",
                        "reservedLayer": 2, "reservedFrame": 100, "lengthFrames": 24})"),
        &out, &err));
    CHECK(out.job_id == "j");
    CHECK(out.video_file_path == "C:\\v.mp4");
    CHECK(out.reserved_layer == 2);
    CHECK(out.reserved_frame == 100);
    CHECK(out.length_frames == 24);

    CHECK_FALSE(nzltx::ParseResolveProvisional(
        json::parse(R"({"jobId": "j", "reservedLayer": 2, "reservedFrame": 1,
                        "lengthFrames": 1})"),
        &out, &err));  // missing videoFilePath
    CHECK_FALSE(nzltx::ParseResolveProvisional(
        json::parse(R"({"jobId": "j", "videoFilePath": "v", "reservedLayer": 2,
                        "reservedFrame": 1, "lengthFrames": 0})"),
        &out, &err));  // non-positive lengthFrames
}

TEST_CASE("resolveProvisional replaces a present placeholder (mode replaced)") {
    // The scan returns a placeholder whose alias exactly matches the job.
    const ProvisionalTextAlias built =
        nzltx::BuildProvisionalTextAlias("resume", "jobR");
    ReplaceObjectRequest captured;
    bool replace_called = false;
    bool insert_called = false;
    RequestContext ctx = AvailableCtx();
    ctx.scan_objects = [built]() {
        ScannedObject o;
        o.layer = 5;
        o.frame_start = 100;
        o.frame_end = 200;
        o.alias = built.alias;
        return std::vector<ScannedObject>{o};
    };
    ctx.replace_object = [&](const ReplaceObjectRequest& r) {
        replace_called = true;
        captured = r;
        return true;
    };
    ctx.insert_media = [&](const InsertMediaParams&) {
        insert_called = true;
        return InsertMediaResult{};
    };
    const std::string resp = HandleRequestJson(
        R"({"id": 45, "method": "timeline.resolveProvisional",
            "params": {"jobId": "jobR", "videoFilePath": "C:\\out.mp4",
                       "reservedLayer": 9, "reservedFrame": 999, "lengthFrames": 24}})",
        ctx);
    const json j = json::parse(resp);
    REQUIRE(j["ok"] == true);
    CHECK(j["result"]["mode"] == "replaced");
    // Replacement acts at the FOUND position, not the reservation.
    CHECK(j["result"]["layer"] == 5);
    CHECK(j["result"]["frame"] == 100);
    CHECK(replace_called);
    CHECK_FALSE(insert_called);
    CHECK(captured.job_id == "jobR");
    CHECK(captured.layer == 5);
    CHECK(captured.frame == 100);
    CHECK(captured.video_file_path == "C:\\out.mp4");
    CHECK(captured.length_frames == 24);
}

TEST_CASE("resolveProvisional inserts at the reservation when the placeholder is gone") {
    InsertMediaParams captured;
    bool replace_called = false;
    RequestContext ctx = AvailableCtx();
    ctx.scan_objects = []() { return std::vector<ScannedObject>{}; };  // nothing found
    ctx.replace_object = [&](const ReplaceObjectRequest&) {
        replace_called = true;
        return true;
    };
    ctx.insert_media = [&](const InsertMediaParams& p) {
        captured = p;
        InsertMediaResult r;
        r.status = InsertMediaResult::Status::kOk;
        r.layer = p.layer;
        r.frame = p.frame;
        return r;
    };
    const std::string resp = HandleRequestJson(
        R"({"id": 46, "method": "timeline.resolveProvisional",
            "params": {"jobId": "gone", "videoFilePath": "C:\\out.mp4",
                       "reservedLayer": 4, "reservedFrame": 150, "lengthFrames": 24}})",
        ctx);
    const json j = json::parse(resp);
    REQUIRE(j["ok"] == true);
    CHECK(j["result"]["mode"] == "insertedReserved");
    CHECK(j["result"]["layer"] == 4);
    CHECK(j["result"]["frame"] == 150);
    CHECK_FALSE(replace_called);
    CHECK(captured.file_path == "C:\\out.mp4");
    CHECK(captured.has_layer == true);
    CHECK(captured.layer == 4);
    CHECK(captured.has_frame == true);
    CHECK(captured.frame == 150);
}

TEST_CASE("resolveProvisional maps a replace failure to PROVISIONAL_FAILED") {
    const ProvisionalTextAlias built =
        nzltx::BuildProvisionalTextAlias("x", "jobF");
    RequestContext ctx = AvailableCtx();
    ctx.scan_objects = [built]() {
        ScannedObject o;
        o.layer = 1;
        o.frame_start = 0;
        o.frame_end = 10;
        o.alias = built.alias;
        return std::vector<ScannedObject>{o};
    };
    ctx.replace_object = [](const ReplaceObjectRequest&) { return false; };
    const std::string resp = HandleRequestJson(
        R"({"id": 47, "method": "timeline.resolveProvisional",
            "params": {"jobId": "jobF", "videoFilePath": "C:\\v.mp4",
                       "reservedLayer": 1, "reservedFrame": 0, "lengthFrames": 8}})",
        ctx);
    const json j = json::parse(resp);
    CHECK(j["ok"] == false);
    CHECK(j["error"]["code"] == "PROVISIONAL_FAILED");
}

TEST_CASE("resolveProvisional maps a reserved-insert file-not-found to FILE_NOT_FOUND") {
    RequestContext ctx = AvailableCtx();
    ctx.scan_objects = []() { return std::vector<ScannedObject>{}; };
    ctx.insert_media = [](const InsertMediaParams&) {
        InsertMediaResult r;
        r.status = InsertMediaResult::Status::kFileNotFound;
        return r;
    };
    const std::string resp = HandleRequestJson(
        R"({"id": 48, "method": "timeline.resolveProvisional",
            "params": {"jobId": "gone", "videoFilePath": "C:\\missing.mp4",
                       "reservedLayer": 1, "reservedFrame": 0, "lengthFrames": 8}})",
        ctx);
    const json j = json::parse(resp);
    CHECK(j["ok"] == false);
    CHECK(j["error"]["code"] == "FILE_NOT_FOUND");
}

// --- timeline.updateProvisionalText -----------------------------------------

TEST_CASE("ParseUpdateProvisionalText accepts a full request and rejects bad params") {
    UpdateProvisionalTextParams out;
    std::string err;
    CHECK(nzltx::ParseUpdateProvisionalText(
        json::parse(R"({"jobId": "j", "layer": 2, "frame": 100, "text": "50%"})"),
        &out, &err));
    CHECK(out.job_id == "j");
    CHECK(out.layer == 2);
    CHECK(out.frame == 100);
    CHECK(out.text == "50%");

    CHECK_FALSE(nzltx::ParseUpdateProvisionalText(
        json::parse(R"({"layer": 2, "frame": 1, "text": "x"})"), &out, &err));
    CHECK_FALSE(nzltx::ParseUpdateProvisionalText(
        json::parse(R"({"jobId": "j", "frame": 1, "text": "x"})"), &out, &err));
    CHECK_FALSE(nzltx::ParseUpdateProvisionalText(
        json::parse(R"({"jobId": "j", "layer": 2, "frame": 1, "text": 9})"), &out,
        &err));
}

TEST_CASE("updateProvisionalText updates a matched placeholder") {
    const ProvisionalTextAlias built = nzltx::BuildProvisionalTextAlias("p", "jobU");
    UpdateObjectTextRequest captured;
    RequestContext ctx = AvailableCtx();
    ctx.scan_objects = [built]() {
        ScannedObject o;
        o.layer = 3;
        o.frame_start = 50;
        o.frame_end = 90;
        o.alias = built.alias;
        return std::vector<ScannedObject>{o};
    };
    ctx.update_object_text = [&](const UpdateObjectTextRequest& r) {
        captured = r;
        return true;
    };
    const std::string resp = HandleRequestJson(
        R"({"id": 49, "method": "timeline.updateProvisionalText",
            "params": {"jobId": "jobU", "layer": 3, "frame": 60, "text": "80%"}})",
        ctx);
    const json j = json::parse(resp);
    REQUIRE(j["ok"] == true);
    CHECK(j["result"]["updated"] == true);
    CHECK(captured.job_id == "jobU");
    CHECK(captured.layer == 3);      // the scanned position, not the request's
    CHECK(captured.frame == 50);
    CHECK(captured.text == "80%");
}

TEST_CASE("updateProvisionalText reports updated=false when no object matches") {
    bool provider_called = false;
    RequestContext ctx = AvailableCtx();
    ctx.scan_objects = []() { return std::vector<ScannedObject>{}; };
    ctx.update_object_text = [&](const UpdateObjectTextRequest&) {
        provider_called = true;
        return true;
    };
    const std::string resp = HandleRequestJson(
        R"({"id": 50, "method": "timeline.updateProvisionalText",
            "params": {"jobId": "nope", "layer": -1, "frame": 0, "text": "x"}})",
        ctx);
    const json j = json::parse(resp);
    REQUIRE(j["ok"] == true);
    CHECK(j["result"]["updated"] == false);
    CHECK_FALSE(provider_called);
}

// --- I13: timeline.deleteProvisionalByJob (spec 5-10) -----------------------

TEST_CASE("ParseDeleteProvisionalByJob accepts a jobId and rejects missing/empty") {
    nzltx::DeleteProvisionalByJobParams out;
    std::string err;
    REQUIRE(nzltx::ParseDeleteProvisionalByJob(
        json::parse(R"({"jobId": "job-77"})"), &out, &err));
    CHECK(out.job_id == "job-77");
    // Missing jobId.
    CHECK_FALSE(nzltx::ParseDeleteProvisionalByJob(json::parse("{}"), &out, &err));
    CHECK_FALSE(err.empty());
    // Empty jobId.
    CHECK_FALSE(nzltx::ParseDeleteProvisionalByJob(
        json::parse(R"({"jobId": ""})"), &out, &err));
    // Non-string jobId.
    CHECK_FALSE(nzltx::ParseDeleteProvisionalByJob(
        json::parse(R"({"jobId": 5})"), &out, &err));
    // Non-object params.
    CHECK_FALSE(nzltx::ParseDeleteProvisionalByJob(json::parse("[]"), &out, &err));
}

TEST_CASE("deleteProvisionalByJob deletes the matched placeholder (deleted=true)") {
    const ProvisionalTextAlias built =
        nzltx::BuildProvisionalTextAlias("done", "jobD");
    DeleteProvisionalRequest captured;
    bool provider_called = false;
    RequestContext ctx = AvailableCtx();
    ctx.scan_objects = [built]() {
        ScannedObject o;
        o.layer = 6;
        o.frame_start = 120;
        o.frame_end = 180;
        o.alias = built.alias;
        return std::vector<ScannedObject>{o};
    };
    ctx.delete_provisional = [&](const DeleteProvisionalRequest& r) {
        provider_called = true;
        captured = r;
        return true;
    };
    const std::string resp = HandleRequestJson(
        R"({"id": 60, "method": "timeline.deleteProvisionalByJob",
            "params": {"jobId": "jobD"}})",
        ctx);
    const json j = json::parse(resp);
    REQUIRE(j["ok"] == true);
    CHECK(j["result"]["ok"] == true);
    CHECK(j["result"]["deleted"] == true);
    CHECK(provider_called);
    // The delete acts at the FOUND position (from the scan).
    CHECK(captured.job_id == "jobD");
    CHECK(captured.layer == 6);
    CHECK(captured.frame == 120);
}

TEST_CASE("deleteProvisionalByJob is an idempotent no-op when nothing matches (deleted=false)") {
    bool provider_called = false;
    RequestContext ctx = AvailableCtx();
    ctx.scan_objects = []() { return std::vector<ScannedObject>{}; };  // nothing found
    ctx.delete_provisional = [&](const DeleteProvisionalRequest&) {
        provider_called = true;
        return true;
    };
    const std::string resp = HandleRequestJson(
        R"({"id": 61, "method": "timeline.deleteProvisionalByJob",
            "params": {"jobId": "absent"}})",
        ctx);
    const json j = json::parse(resp);
    REQUIRE(j["ok"] == true);
    CHECK(j["result"]["ok"] == true);
    CHECK(j["result"]["deleted"] == false);
    CHECK_FALSE(provider_called);  // never called: no placeholder to delete
}

TEST_CASE("deleteProvisionalByJob reports deleted=false when the provider fails") {
    const ProvisionalTextAlias built =
        nzltx::BuildProvisionalTextAlias("done", "jobF");
    RequestContext ctx = AvailableCtx();
    ctx.scan_objects = [built]() {
        ScannedObject o;
        o.layer = 2;
        o.frame_start = 0;
        o.frame_end = 30;
        o.alias = built.alias;
        return std::vector<ScannedObject>{o};
    };
    ctx.delete_provisional = [](const DeleteProvisionalRequest&) { return false; };
    const std::string resp = HandleRequestJson(
        R"({"id": 62, "method": "timeline.deleteProvisionalByJob",
            "params": {"jobId": "jobF"}})",
        ctx);
    const json j = json::parse(resp);
    REQUIRE(j["ok"] == true);
    CHECK(j["result"]["deleted"] == false);
}

TEST_CASE("deleteProvisionalByJob with a missing jobId yields BAD_REQUEST") {
    const std::string resp = HandleRequestJson(
        R"({"id": 63, "method": "timeline.deleteProvisionalByJob", "params": {}})",
        AvailableCtx());
    const json j = json::parse(resp);
    CHECK(j["ok"] == false);
    CHECK(j["error"]["code"] == "BAD_REQUEST");
}

// --- timeline.insertMediaForJob (replace-insert 🎞) --------------------------

TEST_CASE("ParseInsertMediaForJob accepts jobId+filePath and rejects missing/empty/non-string") {
    nzltx::InsertMediaForJobParams out;
    std::string err;
    REQUIRE(nzltx::ParseInsertMediaForJob(
        json::parse(R"({"jobId": "job-9", "filePath": "C:\\a.mp4"})"), &out, &err));
    CHECK(out.job_id == "job-9");
    CHECK(out.file_path == "C:\\a.mp4");
    // Missing jobId.
    CHECK_FALSE(nzltx::ParseInsertMediaForJob(
        json::parse(R"({"filePath": "C:\\a.mp4"})"), &out, &err));
    CHECK_FALSE(err.empty());
    // Empty jobId.
    CHECK_FALSE(nzltx::ParseInsertMediaForJob(
        json::parse(R"({"jobId": "", "filePath": "C:\\a.mp4"})"), &out, &err));
    // Non-string jobId.
    CHECK_FALSE(nzltx::ParseInsertMediaForJob(
        json::parse(R"({"jobId": 5, "filePath": "C:\\a.mp4"})"), &out, &err));
    // Missing filePath.
    CHECK_FALSE(nzltx::ParseInsertMediaForJob(
        json::parse(R"({"jobId": "job-9"})"), &out, &err));
    // Empty filePath.
    CHECK_FALSE(nzltx::ParseInsertMediaForJob(
        json::parse(R"({"jobId": "job-9", "filePath": ""})"), &out, &err));
    // Non-object params.
    CHECK_FALSE(nzltx::ParseInsertMediaForJob(json::parse("[]"), &out, &err));
}

TEST_CASE("insertMediaForJob with a missing jobId yields BAD_REQUEST") {
    const std::string resp = HandleRequestJson(
        R"({"id": 70, "method": "timeline.insertMediaForJob",
            "params": {"filePath": "C:\\a.mp4"}})",
        AvailableCtx());
    const json j = json::parse(resp);
    CHECK(j["ok"] == false);
    CHECK(j["error"]["code"] == "BAD_REQUEST");
}

TEST_CASE("insertMediaForJob replaces in place when the job's marker is found") {
    const ProvisionalTextAlias built =
        nzltx::BuildProvisionalTextAlias("done", "jobR");
    ReplaceMediaForJobRequest captured;
    bool replace_called = false;
    bool insert_called = false;
    RequestContext ctx = AvailableCtx();
    ctx.scan_objects = [built]() {
        ScannedObject o;
        o.layer = 4;
        o.frame_start = 90;
        o.frame_end = 150;
        o.alias = built.alias;
        return std::vector<ScannedObject>{o};
    };
    ctx.insert_media = [&](const InsertMediaParams&) {
        insert_called = true;
        return InsertMediaResult{};
    };
    ctx.replace_media_for_job =
        [&](const ReplaceMediaForJobRequest& r) -> ReplaceMediaForJobOutcome {
        replace_called = true;
        captured = r;
        ReplaceMediaForJobOutcome out;
        out.ok = true;
        out.layer = r.layer;
        out.frame = r.frame;
        out.used_fallback = false;
        return out;
    };
    const std::string resp = HandleRequestJson(
        R"({"id": 71, "method": "timeline.insertMediaForJob",
            "params": {"jobId": "jobR", "filePath": "C:\\out.mp4"}})",
        ctx);
    const json j = json::parse(resp);
    REQUIRE(j["ok"] == true);
    CHECK(j["result"]["ok"] == true);
    CHECK(j["result"]["mode"] == "replaced");
    CHECK(j["result"]["layer"] == 4);
    CHECK(j["result"]["frame"] == 90);
    CHECK(j["result"]["usedFallback"] == false);
    CHECK(replace_called);
    CHECK_FALSE(insert_called);  // the replace branch never touches insert_media
    // The replace acts at the marker's scanned position with the given file.
    CHECK(captured.job_id == "jobR");
    CHECK(captured.layer == 4);
    CHECK(captured.frame == 90);
    CHECK(captured.file_path == "C:\\out.mp4");
}

TEST_CASE("insertMediaForJob reports usedFallback when the replace provider fell back") {
    const ProvisionalTextAlias built =
        nzltx::BuildProvisionalTextAlias("done", "jobFB");
    RequestContext ctx = AvailableCtx();
    ctx.scan_objects = [built]() {
        ScannedObject o;
        o.layer = 1;
        o.frame_start = 0;
        o.frame_end = 30;
        o.alias = built.alias;
        return std::vector<ScannedObject>{o};
    };
    ctx.replace_media_for_job =
        [](const ReplaceMediaForJobRequest&) -> ReplaceMediaForJobOutcome {
        ReplaceMediaForJobOutcome out;
        out.ok = true;
        out.layer = 11;  // landed on the layer_max+1 retreat
        out.frame = 0;
        out.used_fallback = true;
        return out;
    };
    const std::string resp = HandleRequestJson(
        R"({"id": 72, "method": "timeline.insertMediaForJob",
            "params": {"jobId": "jobFB", "filePath": "C:\\out.mp4"}})",
        ctx);
    const json j = json::parse(resp);
    REQUIRE(j["ok"] == true);
    CHECK(j["result"]["mode"] == "replaced");
    CHECK(j["result"]["layer"] == 11);
    CHECK(j["result"]["usedFallback"] == true);
}

TEST_CASE("insertMediaForJob reports INSERT_FAILED when the replace provider fails") {
    const ProvisionalTextAlias built =
        nzltx::BuildProvisionalTextAlias("done", "jobX");
    RequestContext ctx = AvailableCtx();
    ctx.scan_objects = [built]() {
        ScannedObject o;
        o.layer = 2;
        o.frame_start = 10;
        o.frame_end = 40;
        o.alias = built.alias;
        return std::vector<ScannedObject>{o};
    };
    ctx.replace_media_for_job =
        [](const ReplaceMediaForJobRequest&) { return ReplaceMediaForJobOutcome{}; };
    const std::string resp = HandleRequestJson(
        R"({"id": 73, "method": "timeline.insertMediaForJob",
            "params": {"jobId": "jobX", "filePath": "C:\\out.mp4"}})",
        ctx);
    const json j = json::parse(resp);
    CHECK(j["ok"] == false);
    CHECK(j["error"]["code"] == "INSERT_FAILED");
}

TEST_CASE("insertMediaForJob falls back to a plain insert when no marker is found") {
    InsertMediaParams captured;
    bool replace_called = false;
    RequestContext ctx =
        MakeContext(AvailableInfo, InsertMediaResult::Status::kOk, &captured);
    // No provisional objects on the timeline -> the not-found (insert) branch.
    ctx.scan_objects = []() { return std::vector<ScannedObject>{}; };
    ctx.replace_media_for_job =
        [&](const ReplaceMediaForJobRequest&) -> ReplaceMediaForJobOutcome {
        replace_called = true;
        return ReplaceMediaForJobOutcome{};
    };
    const std::string resp = HandleRequestJson(
        R"({"id": 74, "method": "timeline.insertMediaForJob",
            "params": {"jobId": "absent", "filePath": "C:\\out.mp4"}})",
        ctx);
    const json j = json::parse(resp);
    REQUIRE(j["ok"] == true);
    CHECK(j["result"]["mode"] == "inserted");
    // The insert_media provider's cursor fallback (AvailableCtx: layer 7 / frame 99).
    CHECK(j["result"]["layer"] == 7);
    CHECK(j["result"]["frame"] == 99);
    CHECK(j["result"]["usedFallback"] == false);
    CHECK_FALSE(replace_called);  // the replace provider is never consulted here
    // insert_media was called with no explicit layer/frame (cursor fallback) and
    // the given file path.
    CHECK(captured.file_path == "C:\\out.mp4");
    CHECK(captured.has_layer == false);
    CHECK(captured.has_frame == false);
}

TEST_CASE("insertMediaForJob maps a not-found insert failure to its error code") {
    RequestContext ctx = MakeContext(AvailableInfo, InsertMediaResult::Status::kFileNotFound);
    ctx.scan_objects = []() { return std::vector<ScannedObject>{}; };
    const std::string resp = HandleRequestJson(
        R"({"id": 75, "method": "timeline.insertMediaForJob",
            "params": {"jobId": "absent", "filePath": "C:\\gone.mp4"}})",
        ctx);
    const json j = json::parse(resp);
    CHECK(j["ok"] == false);
    CHECK(j["error"]["code"] == "FILE_NOT_FOUND");
}

// --- I3: provisional placement geometry (pure) ------------------------------

TEST_CASE("ResolveProvisionalPlacement (A) places directly after the material") {
    PlacementResolveInput in;
    in.placement = ProvisionalPlacement::kAfterMaterial;
    in.material_layer = 2;
    in.material_frame_start = 30;
    in.material_frame_end = 90;  // inclusive last frame
    in.layer_max = 10;
    const PlacementResolveResult r = ResolveProvisionalPlacement(in);
    REQUIRE(r.ok);
    CHECK(r.layer == 2);       // same layer as the material
    CHECK(r.frame == 91);      // end + 1 (inclusive end)
}

TEST_CASE("ResolveProvisionalPlacement (B) shares the start on layer_max+1") {
    PlacementResolveInput in;
    in.placement = ProvisionalPlacement::kSameStartFront;
    in.material_frame_start = 30;
    in.material_frame_end = 90;
    in.layer_max = 7;
    const PlacementResolveResult r = ResolveProvisionalPlacement(in);
    REQUIRE(r.ok);
    CHECK(r.layer == 8);       // layer_max + 1
    CHECK(r.frame == 30);      // same start frame as the material
}

TEST_CASE("ResolveProvisionalPlacement (C) uses the cursor position") {
    PlacementResolveInput in;
    in.placement = ProvisionalPlacement::kCursor;
    in.cursor_layer = 4;
    in.cursor_frame = 123;
    const PlacementResolveResult r = ResolveProvisionalPlacement(in);
    REQUIRE(r.ok);
    CHECK(r.layer == 4);
    CHECK(r.frame == 123);
}

TEST_CASE("ResolveProvisionalPlacement rejects missing/negative required inputs") {
    // (A) needs materialLayer + materialFrameEnd.
    PlacementResolveInput a;
    a.placement = ProvisionalPlacement::kAfterMaterial;
    a.material_layer = -1;  // missing
    a.material_frame_end = 90;
    CHECK_FALSE(ResolveProvisionalPlacement(a).ok);
    // (B) needs materialFrameStart.
    PlacementResolveInput b;
    b.placement = ProvisionalPlacement::kSameStartFront;
    b.material_frame_start = -1;  // missing
    b.layer_max = 5;
    CHECK_FALSE(ResolveProvisionalPlacement(b).ok);
    // (C) needs cursorLayer + cursorFrame.
    PlacementResolveInput c;
    c.placement = ProvisionalPlacement::kCursor;
    c.cursor_layer = 2;
    c.cursor_frame = -3;  // negative
    CHECK_FALSE(ResolveProvisionalPlacement(c).ok);
    // kLegacy is not a resolvable placement.
    PlacementResolveInput legacy;
    legacy.placement = ProvisionalPlacement::kLegacy;
    CHECK_FALSE(ResolveProvisionalPlacement(legacy).ok);
}

// --- insertProvisional length/placement resolution --------------------------

TEST_CASE("insertProvisional resolves length from numFrames+genFps (project fps)") {
    // AvailableInfo() reports rate 30 / scale 1 -> project fps 30. A 49-frame
    // clip at genFps 30 keeps its 49-frame duration on a 30-fps timeline.
    std::string captured_alias;
    int captured_length = -1;
    RequestContext ctx = AvailableCtx();
    ctx.insert_provisional = [&](const std::string& alias, const std::string&, int layer,
                                 int frame, int length) -> InsertProvisionalOutcome {
        captured_alias = alias;
        captured_length = length;
        InsertProvisionalOutcome o;
        o.ok = true;
        o.layer = layer;
        o.frame = frame;
        return o;
    };
    const std::string resp = HandleRequestJson(
        R"({"id": 60, "method": "timeline.insertProvisional",
            "params": {"jobId": "jN", "displayText": "hi", "numFrames": 49,
                       "genFps": 30, "placement": "C", "cursorLayer": 1,
                       "cursorFrame": 5}})",
        ctx);
    const json j = json::parse(resp);
    REQUIRE(j["ok"] == true);
    // The alias frame header is pinned to the resolved length (49); D2: the same
    // resolved length is forwarded to create.
    CHECK(captured_alias.find("frame=0,49") != std::string::npos);
    CHECK(captured_length == 49);
    CHECK(j["result"]["usedFallback"] == false);
    CHECK(j["result"]["placedLayer"] == 1);
    CHECK(j["result"]["placedFrame"] == 5);
}

TEST_CASE("insertProvisional rejects numFrames with a non-positive genFps") {
    const std::string resp = HandleRequestJson(
        R"({"id": 62, "method": "timeline.insertProvisional",
            "params": {"jobId": "jZ", "displayText": "hi", "numFrames": 49,
                       "genFps": 0, "placement": "C", "cursorLayer": 1,
                       "cursorFrame": 5}})",
        AvailableCtx());
    const json j = json::parse(resp);
    CHECK(j["ok"] == false);
    CHECK(j["error"]["code"] == "BAD_REQUEST");
}

TEST_CASE("insertProvisional resolves placement B to layer_max+1 at the material start") {
    int captured_layer = -1;
    int captured_frame = -1;
    RequestContext ctx = AvailableCtx();  // AvailableInfo layer_max == 10
    ctx.insert_provisional = [&](const std::string&, const std::string&, int layer,
                                 int frame, int) -> InsertProvisionalOutcome {
        captured_layer = layer;
        captured_frame = frame;
        InsertProvisionalOutcome o;
        o.ok = true;
        o.layer = layer;
        o.frame = frame;
        return o;
    };
    const std::string resp = HandleRequestJson(
        R"({"id": 63, "method": "timeline.insertProvisional",
            "params": {"jobId": "jB", "displayText": "hi", "numFrames": 24,
                       "genFps": 30, "placement": "B", "materialLayer": 2,
                       "materialFrameStart": 40, "materialFrameEnd": 88}})",
        ctx);
    const json j = json::parse(resp);
    REQUIRE(j["ok"] == true);
    CHECK(captured_layer == 11);  // layer_max (10) + 1
    CHECK(captured_frame == 40);  // material start
}

TEST_CASE("insertProvisional surfaces the provider's usedFallback flag") {
    RequestContext ctx = AvailableCtx();
    ctx.insert_provisional = [](const std::string&, const std::string&, int, int,
                                int) -> InsertProvisionalOutcome {
        InsertProvisionalOutcome o;
        o.ok = true;
        o.layer = 11;  // relocated to layer_max+1 by the (mocked) fallback
        o.frame = 5;
        o.used_fallback = true;
        return o;
    };
    const std::string resp = HandleRequestJson(
        R"({"id": 64, "method": "timeline.insertProvisional",
            "params": {"jobId": "jF", "displayText": "hi", "numFrames": 24,
                       "genFps": 30, "placement": "C", "cursorLayer": 1,
                       "cursorFrame": 5}})",
        ctx);
    const json j = json::parse(resp);
    REQUIRE(j["ok"] == true);
    CHECK(j["result"]["usedFallback"] == true);
    CHECK(j["result"]["placedLayer"] == 11);
}

TEST_CASE("insertProvisional rejects an invalid placement value") {
    const std::string resp = HandleRequestJson(
        R"({"id": 65, "method": "timeline.insertProvisional",
            "params": {"jobId": "jX", "displayText": "hi", "numFrames": 24,
                       "genFps": 30, "placement": "Z"}})",
        AvailableCtx());
    const json j = json::parse(resp);
    CHECK(j["ok"] == false);
    CHECK(j["error"]["code"] == "BAD_REQUEST");
}

// --- I3: timeline.updateProvisionalReservation ------------------------------

TEST_CASE("ParseUpdateProvisionalReservation accepts a full request") {
    UpdateProvisionalReservationParams out;
    std::string err;
    CHECK(nzltx::ParseUpdateProvisionalReservation(
        json::parse(R"({"oldJobId": "old", "newJobId": "new", "displayText": "t",
                        "numFrames": 49, "genFps": 24, "placement": "A",
                        "materialLayer": 2, "materialFrameStart": 10,
                        "materialFrameEnd": 60})"),
        &out, &err));
    CHECK(out.old_job_id == "old");
    CHECK(out.new_job_id == "new");
    CHECK(out.num_frames == 49);
    CHECK(out.gen_fps == doctest::Approx(24.0));
    CHECK(out.placement == ProvisionalPlacement::kAfterMaterial);
    CHECK(out.has_material_frame_end);
    CHECK(out.material_frame_end == 60);
}

TEST_CASE("ParseUpdateProvisionalReservation allows an empty oldJobId (create-only)") {
    UpdateProvisionalReservationParams out;
    std::string err;
    CHECK(nzltx::ParseUpdateProvisionalReservation(
        json::parse(R"({"oldJobId": "", "newJobId": "new", "displayText": "t",
                        "numFrames": 49, "genFps": 24, "placement": "C",
                        "cursorLayer": 3, "cursorFrame": 7})"),
        &out, &err));
    CHECK(out.old_job_id.empty());
    CHECK(out.placement == ProvisionalPlacement::kCursor);
}

TEST_CASE("ParseUpdateProvisionalReservation rejects bad params") {
    UpdateProvisionalReservationParams out;
    std::string err;
    // Missing newJobId.
    CHECK_FALSE(nzltx::ParseUpdateProvisionalReservation(
        json::parse(R"({"oldJobId": "o", "displayText": "t", "numFrames": 49,
                        "genFps": 24, "placement": "A"})"),
        &out, &err));
    // numFrames wrong type.
    CHECK_FALSE(nzltx::ParseUpdateProvisionalReservation(
        json::parse(R"({"oldJobId": "o", "newJobId": "n", "displayText": "t",
                        "numFrames": "49", "genFps": 24, "placement": "A"})"),
        &out, &err));
    // genFps non-positive.
    CHECK_FALSE(nzltx::ParseUpdateProvisionalReservation(
        json::parse(R"({"oldJobId": "o", "newJobId": "n", "displayText": "t",
                        "numFrames": 49, "genFps": 0, "placement": "A"})"),
        &out, &err));
    // Invalid placement value.
    CHECK_FALSE(nzltx::ParseUpdateProvisionalReservation(
        json::parse(R"({"oldJobId": "o", "newJobId": "n", "displayText": "t",
                        "numFrames": 49, "genFps": 24, "placement": "X"})"),
        &out, &err));
    // Missing placement entirely (required on this RPC).
    CHECK_FALSE(nzltx::ParseUpdateProvisionalReservation(
        json::parse(R"({"oldJobId": "o", "newJobId": "n", "displayText": "t",
                        "numFrames": 49, "genFps": 24})"),
        &out, &err));
}

TEST_CASE("updateProvisionalReservation resolves length/placement and reports deletedOld") {
    UpdateReservationRequest captured;
    RequestContext ctx = AvailableCtx();  // rate 30 / scale 1, layer_max 10
    ctx.update_reservation = [&](const UpdateReservationRequest& r) -> UpdateReservationOutcome {
        captured = r;
        UpdateReservationOutcome o;
        o.ok = true;
        o.deleted_old = !r.old_job_id.empty();
        o.placed_layer = r.layer;
        o.placed_frame = r.frame;
        o.used_fallback = false;
        return o;
    };
    const std::string resp = HandleRequestJson(
        R"({"id": 66, "method": "timeline.updateProvisionalReservation",
            "params": {"oldJobId": "pending-1", "newJobId": "job-9",
                       "displayText": "Generating", "numFrames": 49, "genFps": 30,
                       "placement": "A", "materialLayer": 2,
                       "materialFrameStart": 10, "materialFrameEnd": 60}})",
        ctx);
    const json j = json::parse(resp);
    REQUIRE(j["ok"] == true);
    CHECK(j["result"]["ok"] == true);
    CHECK(j["result"]["deletedOld"] == true);
    // Placement (A): layer == materialLayer, frame == materialFrameEnd + 1.
    CHECK(j["result"]["placedLayer"] == 2);
    CHECK(j["result"]["placedFrame"] == 61);
    CHECK(j["result"]["usedFallback"] == false);
    // The provider received the built alias pinned to the resolved length (49).
    CHECK(captured.new_job_id == "job-9");
    CHECK(captured.old_job_id == "pending-1");
    CHECK(AliasMatchesJob(captured.alias, "job-9"));
    CHECK(captured.alias.find("frame=0,49") != std::string::npos);
    CHECK(captured.layer == 2);
    CHECK(captured.frame == 61);
    // D2: the resolved length rides on the request (explicit create len + pre-check).
    CHECK(captured.length_frames == 49);
}

TEST_CASE("updateProvisionalReservation reports deletedOld=false for an empty oldJobId") {
    RequestContext ctx = AvailableCtx();
    ctx.update_reservation = [](const UpdateReservationRequest& r) -> UpdateReservationOutcome {
        UpdateReservationOutcome o;
        o.ok = true;
        o.deleted_old = !r.old_job_id.empty();  // create-only -> false
        o.placed_layer = r.layer;
        o.placed_frame = r.frame;
        return o;
    };
    const std::string resp = HandleRequestJson(
        R"({"id": 67, "method": "timeline.updateProvisionalReservation",
            "params": {"oldJobId": "", "newJobId": "job-x", "displayText": "t",
                       "numFrames": 49, "genFps": 30, "placement": "C",
                       "cursorLayer": 4, "cursorFrame": 200}})",
        ctx);
    const json j = json::parse(resp);
    REQUIRE(j["ok"] == true);
    CHECK(j["result"]["deletedOld"] == false);
    CHECK(j["result"]["placedLayer"] == 4);
    CHECK(j["result"]["placedFrame"] == 200);
}

TEST_CASE("updateProvisionalReservation maps a provider failure to PROVISIONAL_FAILED") {
    RequestContext ctx = AvailableCtx();
    ctx.update_reservation = [](const UpdateReservationRequest&) -> UpdateReservationOutcome {
        return UpdateReservationOutcome{};  // ok == false
    };
    const std::string resp = HandleRequestJson(
        R"({"id": 68, "method": "timeline.updateProvisionalReservation",
            "params": {"oldJobId": "", "newJobId": "job-x", "displayText": "t",
                       "numFrames": 49, "genFps": 30, "placement": "C",
                       "cursorLayer": 4, "cursorFrame": 200}})",
        ctx);
    const json j = json::parse(resp);
    CHECK(j["ok"] == false);
    CHECK(j["error"]["code"] == "PROVISIONAL_FAILED");
}

TEST_CASE("updateProvisionalReservation with bad params yields BAD_REQUEST") {
    const std::string resp = HandleRequestJson(
        R"({"id": 69, "method": "timeline.updateProvisionalReservation", "params": {}})",
        AvailableCtx());
    const json j = json::parse(resp);
    CHECK(j["ok"] == false);
    CHECK(j["error"]["code"] == "BAD_REQUEST");
}

// --- timeline.scanProvisionals ----------------------------------------------

TEST_CASE("scanProvisionals returns every NzLTX23 placeholder and ignores foreign objects") {
    const ProvisionalTextAlias a = nzltx::BuildProvisionalTextAlias("clip A", "jobA");
    const ProvisionalTextAlias b = nzltx::BuildProvisionalTextAlias("clip B", "jobB");
    RequestContext ctx = AvailableCtx();
    ctx.scan_objects = [a, b]() {
        ScannedObject oa;
        oa.layer = 1;
        oa.frame_start = 100;
        oa.frame_end = 140;
        oa.alias = a.alias;
        ScannedObject foreign;  // a non-NzLTX23 object -> ignored
        foreign.layer = 2;
        foreign.frame_start = 0;
        foreign.frame_end = 30;
        foreign.alias = "[Object]\r\n[Object.0]\r\neffect.name=whatever\r\n";
        ScannedObject ob;
        ob.layer = 3;
        ob.frame_start = 200;
        ob.frame_end = 260;
        ob.alias = b.alias;
        return std::vector<ScannedObject>{oa, foreign, ob};
    };
    const std::string resp =
        HandleRequestJson(R"({"id": 51, "method": "timeline.scanProvisionals"})", ctx);
    const json j = json::parse(resp);
    REQUIRE(j["ok"] == true);
    const json& orphans = j["result"]["orphans"];
    REQUIRE(orphans.is_array());
    REQUIRE(orphans.size() == 2);
    CHECK(orphans[0]["jobId"] == "jobA");
    CHECK(orphans[0]["layer"] == 1);
    CHECK(orphans[0]["frame"] == 100);
    CHECK(orphans[1]["jobId"] == "jobB");
    CHECK(orphans[1]["layer"] == 3);
    CHECK(orphans[1]["frame"] == 200);
}

TEST_CASE("scanProvisionals returns an empty list when no scan provider is set") {
    RequestContext ctx = AvailableCtx();  // scan_objects unset
    const std::string resp =
        HandleRequestJson(R"({"id": 52, "method": "timeline.scanProvisionals"})", ctx);
    const json j = json::parse(resp);
    REQUIRE(j["ok"] == true);
    CHECK(j["result"]["orphans"].is_array());
    CHECK(j["result"]["orphans"].empty());
}

// --- timeline.cutoutRange (async: Parse + Make only) ------------------------

TEST_CASE("ParseCutoutRange accepts a full request and rejects bad params") {
    CutoutRangeRequest out;
    std::string err;
    CHECK(nzltx::ParseCutoutRange(
        json::parse(R"({"layer": 2, "frameStart": 10, "frameCount": 48,
                        "withAudio": true, "audioMode": "solo",
                        "soloKeepLayers": [2, 3]})"),
        &out, &err));
    CHECK(out.layer == 2);
    CHECK(out.frame_start == 10);
    CHECK(out.frame_count == 48);
    CHECK(out.with_audio == true);
    CHECK(out.audio_mode == "solo");
    REQUIRE(out.solo_keep_layers.size() == 2);
    CHECK(out.solo_keep_layers[0] == 2);
    CHECK(out.solo_keep_layers[1] == 3);

    // frameCount must be > 0.
    CHECK_FALSE(nzltx::ParseCutoutRange(
        json::parse(R"({"layer": 0, "frameStart": 0, "frameCount": 0,
                        "withAudio": false, "audioMode": "mix", "soloKeepLayers": []})"),
        &out, &err));
    // audioMode must be mix|solo.
    CHECK_FALSE(nzltx::ParseCutoutRange(
        json::parse(R"({"layer": 0, "frameStart": 0, "frameCount": 1,
                        "withAudio": false, "audioMode": "loud", "soloKeepLayers": []})"),
        &out, &err));
    // withAudio must be boolean.
    CHECK_FALSE(nzltx::ParseCutoutRange(
        json::parse(R"({"layer": 0, "frameStart": 0, "frameCount": 1,
                        "withAudio": "yes", "audioMode": "mix", "soloKeepLayers": []})"),
        &out, &err));
    // soloKeepLayers must be an integer array.
    CHECK_FALSE(nzltx::ParseCutoutRange(
        json::parse(R"({"layer": 0, "frameStart": 0, "frameCount": 1,
                        "withAudio": false, "audioMode": "mix",
                        "soloKeepLayers": ["x"]})"),
        &out, &err));
    CHECK_FALSE(nzltx::ParseCutoutRange(json::parse(R"([])"), &out, &err));
}

TEST_CASE("MakeCutoutResult mirrors the contract v5 result shape") {
    const json r = nzltx::MakeCutoutResult("C:\\tmp\\cut.mp4", 1280, 720, 48, true);
    CHECK(r["filePath"] == "C:\\tmp\\cut.mp4");
    CHECK(r["width"] == 1280);
    CHECK(r["height"] == 720);
    CHECK(r["frameCount"] == 48);
    CHECK(r["hasAudio"] == true);
}

// --- timeline.extractAudio (async: Parse + Make only) -----------------------

TEST_CASE("ParseExtractAudio accepts a full request and rejects bad params") {
    ExtractAudioRequest out;
    std::string err;
    CHECK(nzltx::ParseExtractAudio(
        json::parse(R"({"layer": 1, "frameStart": 5, "frameCount": 30,
                        "audioMode": "mix", "soloKeepLayers": []})"),
        &out, &err));
    CHECK(out.layer == 1);
    CHECK(out.frame_start == 5);
    CHECK(out.frame_count == 30);
    CHECK(out.audio_mode == "mix");
    CHECK(out.solo_keep_layers.empty());

    CHECK_FALSE(nzltx::ParseExtractAudio(
        json::parse(R"({"layer": 1, "frameStart": 5, "frameCount": -1,
                        "audioMode": "mix", "soloKeepLayers": []})"),
        &out, &err));
    CHECK_FALSE(nzltx::ParseExtractAudio(
        json::parse(R"({"layer": 1, "frameStart": 5, "frameCount": 1,
                        "soloKeepLayers": []})"),
        &out, &err));  // missing audioMode
}

TEST_CASE("MakeExtractAudioResult mirrors the contract v5 result shape") {
    const json r = nzltx::MakeExtractAudioResult("C:\\tmp\\a.wav", 1.25, 48000, true);
    CHECK(r["filePath"] == "C:\\tmp\\a.wav");
    CHECK(r["durationSec"] == doctest::Approx(1.25));
    CHECK(r["sampleRate"] == 48000);
    CHECK(r["hasAudioStream"] == true);
}

TEST_CASE("IsAudioStreamPresent flags silent / empty ranges as no audio") {
    CHECK(nzltx::IsAudioStreamPresent(1000, 0.5));
    CHECK(nzltx::IsAudioStreamPresent(1, nzltx::kAudioSilenceThreshold));
    CHECK_FALSE(nzltx::IsAudioStreamPresent(0, 1.0));      // no samples
    CHECK_FALSE(nzltx::IsAudioStreamPresent(1000, 0.0));   // pure silence
    CHECK_FALSE(nzltx::IsAudioStreamPresent(1000, nzltx::kAudioSilenceThreshold / 2));
}

// ---------------------------------------------------------------------------
// Contract v6 (batch A2V + fs bridge): ui.pickFolder / fs.* pure parsing +
// result formatting. Win32 dispatch (IFileOpenDialog, FindFirstFileW,
// ReplaceFileW/MoveFileExW, ProbeWavFile) lives only in bridge.cpp and is
// deliberately NOT exercised here - see bridge_core.h's per-method comments.
// ---------------------------------------------------------------------------

// --- ui.pickFolder -----------------------------------------------------------

TEST_CASE("ParsePickFolder treats absent params / absent-or-null title as no title") {
    nzltx::PickFolderRequest out;
    std::string err;
    CHECK(nzltx::ParsePickFolder(json::object(), &out, &err));
    CHECK(out.has_title == false);
    CHECK(nzltx::ParsePickFolder(json::parse(R"({})"), &out, &err));
    CHECK(out.has_title == false);
    CHECK(nzltx::ParsePickFolder(json::parse(R"({"title": null})"), &out, &err));
    CHECK(out.has_title == false);
}

TEST_CASE("ParsePickFolder reads an explicit title and rejects a non-string one") {
    nzltx::PickFolderRequest out;
    std::string err;
    CHECK(nzltx::ParsePickFolder(json::parse(R"({"title": "Pick a folder"})"), &out, &err));
    CHECK(out.has_title == true);
    CHECK(out.title == "Pick a folder");
    CHECK_FALSE(nzltx::ParsePickFolder(json::parse(R"({"title": 5})"), &out, &err));
}

// --- fs.listFiles --------------------------------------------------------------

TEST_CASE("ParseListFiles requires a non-empty folderPath and defaults filters") {
    nzltx::ListFilesRequest out;
    std::string err;
    CHECK(nzltx::ParseListFiles(json::parse(R"({"folderPath": "C:\\wavs"})"), &out, &err));
    CHECK(out.folder_path == "C:\\wavs");
    CHECK(out.extensions.empty());
    CHECK(out.with_audio_duration == false);

    CHECK_FALSE(nzltx::ParseListFiles(json::parse(R"({})"), &out, &err));
    CHECK_FALSE(nzltx::ParseListFiles(json::parse(R"({"folderPath": ""})"), &out, &err));
    CHECK_FALSE(nzltx::ParseListFiles(json::parse(R"([])"), &out, &err));
}

TEST_CASE("ParseListFiles parses the extensions array and withAudioDuration") {
    nzltx::ListFilesRequest out;
    std::string err;
    CHECK(nzltx::ParseListFiles(
        json::parse(R"({"folderPath": "C:\\wavs", "extensions": [".wav", ".WAV", ".csv"],
                        "withAudioDuration": true})"),
        &out, &err));
    REQUIRE(out.extensions.size() == 3);
    CHECK(out.extensions[0] == ".wav");
    CHECK(out.extensions[1] == ".WAV");
    CHECK(out.extensions[2] == ".csv");
    CHECK(out.with_audio_duration == true);
}

TEST_CASE("ParseListFiles rejects a non-array extensions and a non-boolean withAudioDuration") {
    nzltx::ListFilesRequest out;
    std::string err;
    CHECK_FALSE(nzltx::ParseListFiles(
        json::parse(R"({"folderPath": "C:\\wavs", "extensions": ".wav"})"), &out, &err));
    CHECK_FALSE(nzltx::ParseListFiles(
        json::parse(R"({"folderPath": "C:\\wavs", "extensions": [1]})"), &out, &err));
    CHECK_FALSE(nzltx::ParseListFiles(
        json::parse(R"({"folderPath": "C:\\wavs", "withAudioDuration": "yes"})"), &out,
        &err));
}

TEST_CASE("MakeListFilesResult formats each entry and preserves input order") {
    std::vector<nzltx::ListedFile> files;
    nzltx::ListedFile a;
    a.name = "b.wav";
    a.path = "C:\\wavs\\b.wav";
    a.size_bytes = 12345;
    a.mtime_ms = 1700000000000.0;
    a.duration_sec = 2.5;
    nzltx::ListedFile b;
    b.name = "a.wav";
    b.path = "C:\\wavs\\a.wav";
    b.size_bytes = 999;
    b.mtime_ms = 1600000000000.0;
    b.duration_sec = 0.0;
    files.push_back(a);
    files.push_back(b);

    const json r = nzltx::MakeListFilesResult(files);
    REQUIRE(r["files"].is_array());
    REQUIRE(r["files"].size() == 2);
    CHECK(r["files"][0]["name"] == "b.wav");
    CHECK(r["files"][0]["path"] == "C:\\wavs\\b.wav");
    CHECK(r["files"][0]["sizeBytes"] == 12345);
    CHECK(r["files"][0]["mtimeMs"] == doctest::Approx(1700000000000.0));
    CHECK(r["files"][0]["durationSec"] == doctest::Approx(2.5));
    CHECK(r["files"][1]["name"] == "a.wav");
    CHECK(r["files"][1]["durationSec"] == doctest::Approx(0.0));
}

TEST_CASE("MakeListFilesResult on an empty input yields an empty files array") {
    const json r = nzltx::MakeListFilesResult({});
    CHECK(r["files"].is_array());
    CHECK(r["files"].empty());
}

// --- fs.probeAudioDuration -------------------------------------------------------

TEST_CASE("ParseProbeAudioDuration requires a non-empty filePath") {
    nzltx::ProbeAudioDurationRequest out;
    std::string err;
    CHECK(nzltx::ParseProbeAudioDuration(json::parse(R"({"filePath": "C:\\a.wav"})"), &out,
                                         &err));
    CHECK(out.file_path == "C:\\a.wav");

    CHECK_FALSE(nzltx::ParseProbeAudioDuration(json::parse(R"({})"), &out, &err));
    CHECK_FALSE(
        nzltx::ParseProbeAudioDuration(json::parse(R"({"filePath": ""})"), &out, &err));
    CHECK_FALSE(
        nzltx::ParseProbeAudioDuration(json::parse(R"({"filePath": 5})"), &out, &err));
}

// --- fs.probeMediaInfo -----------------------------------------------------

TEST_CASE("ParseProbeMediaInfo requires a non-empty filePath") {
    nzltx::ProbeMediaInfoRequest out;
    std::string err;
    CHECK(nzltx::ParseProbeMediaInfo(json::parse(R"({"filePath": "C:\\a.mp4"})"), &out,
                                     &err));
    CHECK(out.file_path == "C:\\a.mp4");

    CHECK_FALSE(nzltx::ParseProbeMediaInfo(json::parse(R"({})"), &out, &err));
    CHECK_FALSE(
        nzltx::ParseProbeMediaInfo(json::parse(R"({"filePath": ""})"), &out, &err));
    CHECK_FALSE(
        nzltx::ParseProbeMediaInfo(json::parse(R"({"filePath": 5})"), &out, &err));
}

TEST_CASE("MakeMediaInfoResult reports all-zero for an unavailable snapshot") {
    nzltx::MediaInfoSnapshot snap;  // available == false; other fields untouched
    snap.duration_sec = 9.0;  // should be ignored because available == false
    snap.width = 1920;
    snap.height = 1080;
    const json r = nzltx::MakeMediaInfoResult(snap);
    CHECK(r["durationSec"] == 0.0);
    CHECK(r["width"] == 0);
    CHECK(r["height"] == 0);
}

TEST_CASE("MakeMediaInfoResult passes through the probed values when available") {
    nzltx::MediaInfoSnapshot snap;
    snap.available = true;
    snap.duration_sec = 4.5;
    snap.width = 1280;
    snap.height = 720;
    const json r = nzltx::MakeMediaInfoResult(snap);
    CHECK(r["durationSec"] == 4.5);
    CHECK(r["width"] == 1280);
    CHECK(r["height"] == 720);
}

TEST_CASE("fs.probeMediaInfo succeeds with an all-zero result when no provider is set") {
    RequestContext ctx = AvailableCtx();  // probe_media_info left unset
    const std::string resp = HandleRequestJson(
        R"({"id": 60, "method": "fs.probeMediaInfo", "params": {"filePath": "C:\\a.mp4"}})",
        ctx);
    const json j = json::parse(resp);
    REQUIRE(j["ok"] == true);
    const json& r = j["result"];
    CHECK(r["durationSec"] == 0.0);
    CHECK(r["width"] == 0);
    CHECK(r["height"] == 0);
}

TEST_CASE("fs.probeMediaInfo passes a provider's values through and forwards the path") {
    RequestContext ctx = AvailableCtx();
    std::string captured_path;
    ctx.probe_media_info = [&captured_path](const std::string& path)
        -> nzltx::MediaInfoSnapshot {
        captured_path = path;
        nzltx::MediaInfoSnapshot snap;
        snap.available = true;
        snap.duration_sec = 12.5;
        snap.width = 3840;
        snap.height = 2160;
        return snap;
    };
    const std::string resp = HandleRequestJson(
        R"({"id": 61, "method": "fs.probeMediaInfo", "params": {"filePath": "C:\\clip.mp4"}})",
        ctx);
    CHECK(captured_path == "C:\\clip.mp4");
    const json j = json::parse(resp);
    REQUIRE(j["ok"] == true);
    const json& r = j["result"];
    CHECK(r["durationSec"] == 12.5);
    CHECK(r["width"] == 3840);
    CHECK(r["height"] == 2160);
}

TEST_CASE("fs.probeMediaInfo rejects a missing/empty filePath as BAD_REQUEST") {
    RequestContext ctx = AvailableCtx();
    const std::string resp = HandleRequestJson(
        R"({"id": 62, "method": "fs.probeMediaInfo", "params": {}})", ctx);
    const json j = json::parse(resp);
    REQUIRE(j["ok"] == false);
    CHECK(j["error"]["code"] == "BAD_REQUEST");
}

// ---------------------------------------------------------------------------
// ui.resolveDroppedFiles (contract v7)
// ---------------------------------------------------------------------------

TEST_CASE("ParseResolveDroppedFiles reads __droppedPaths in order") {
    const nzltx::ResolveDroppedFilesRequest out = nzltx::ParseResolveDroppedFiles(
        json::parse(R"({"__droppedPaths": ["C:\\a\\one.png", "C:\\b\\two.mp4"]})"));
    REQUIRE(out.file_paths.size() == 2);
    CHECK(out.file_paths[0] == "C:\\a\\one.png");
    CHECK(out.file_paths[1] == "C:\\b\\two.mp4");
}

TEST_CASE("ParseResolveDroppedFiles yields an empty result when the key is absent, empty, "
          "or the wrong shape") {
    // No key at all.
    CHECK(nzltx::ParseResolveDroppedFiles(json::parse(R"({})")).file_paths.empty());
    // Explicit empty array.
    CHECK(nzltx::ParseResolveDroppedFiles(json::parse(R"({"__droppedPaths": []})"))
              .file_paths.empty());
    // Not an array at all.
    CHECK(nzltx::ParseResolveDroppedFiles(json::parse(R"({"__droppedPaths": "C:\\a.png"})"))
              .file_paths.empty());
    CHECK(nzltx::ParseResolveDroppedFiles(json::parse(R"({"__droppedPaths": null})"))
              .file_paths.empty());
    // params itself not an object.
    CHECK(nzltx::ParseResolveDroppedFiles(json::parse(R"([])")).file_paths.empty());
}

TEST_CASE("ParseResolveDroppedFiles silently skips non-string and empty-string elements") {
    const nzltx::ResolveDroppedFilesRequest out = nzltx::ParseResolveDroppedFiles(json::parse(
        R"({"__droppedPaths": ["C:\\ok.png", 5, null, {}, [], "", "C:\\also_ok.mp4"]})"));
    REQUIRE(out.file_paths.size() == 2);
    CHECK(out.file_paths[0] == "C:\\ok.png");
    CHECK(out.file_paths[1] == "C:\\also_ok.mp4");
}

TEST_CASE("MakeResolveDroppedFilesResult pairs each path with its derived file name") {
    nzltx::ResolveDroppedFilesRequest req;
    req.file_paths = {"C:\\dir\\clip.mp4", "clip_no_dir.wav"};
    const json result = nzltx::MakeResolveDroppedFilesResult(req);
    REQUIRE(result["files"].size() == 2);
    CHECK(result["files"][0]["filePath"] == "C:\\dir\\clip.mp4");
    CHECK(result["files"][0]["fileName"] == "clip.mp4");
    CHECK(result["files"][1]["filePath"] == "clip_no_dir.wav");
    CHECK(result["files"][1]["fileName"] == "clip_no_dir.wav");
}

TEST_CASE("MakeResolveDroppedFilesResult yields an empty files array for no dropped paths") {
    const nzltx::ResolveDroppedFilesRequest req;  // default-constructed, empty
    const json result = nzltx::MakeResolveDroppedFilesResult(req);
    CHECK(result["files"].is_array());
    CHECK(result["files"].empty());
}

TEST_CASE("HandleRequestJson dispatches ui.resolveDroppedFiles end-to-end") {
    RequestContext ctx;
    const std::string resp = HandleRequestJson(
        R"({"id": 7, "method": "ui.resolveDroppedFiles",
            "params": {"__droppedPaths": ["C:\\a\\x.png"]}})",
        ctx);
    const json j = json::parse(resp);
    REQUIRE(j["ok"] == true);
    REQUIRE(j["result"]["files"].size() == 1);
    CHECK(j["result"]["files"][0]["filePath"] == "C:\\a\\x.png");
    CHECK(j["result"]["files"][0]["fileName"] == "x.png");
}

TEST_CASE("HandleRequestJson resolves ui.resolveDroppedFiles to an empty list with no params") {
    RequestContext ctx;
    const std::string resp =
        HandleRequestJson(R"({"id": 8, "method": "ui.resolveDroppedFiles"})", ctx);
    const json j = json::parse(resp);
    REQUIRE(j["ok"] == true);
    CHECK(j["result"]["files"].empty());
}

// ---------------------------------------------------------------------------
// InjectDroppedPathsIntoRequest (contract v7 BLOCKER fix): the pure half of
// webview_host.cpp's __droppedPaths injection. webview_host.cpp itself only
// decides WHETHER to call this (AdditionalObjects presence / the
// "__droppedPaths" substring pre-check - not testable here, WebView2-
// dependent); this function's own behavior once called IS fully testable.
// ---------------------------------------------------------------------------

TEST_CASE("InjectDroppedPathsIntoRequest injects into an EMPTY params object "
          "(regression test for the real-drop-never-injected BLOCKER)") {
    // This is exactly the shape a real drop sends: `useFileDrop.ts` posts
    // `requestWithFiles("ui.resolveDroppedFiles", {}, [file])` - empty
    // params, no "__droppedPaths" substring anywhere in the raw JSON. The
    // original webview_host.cpp implementation checked that substring FIRST
    // and returned early when absent, so a real drop's AdditionalObjects
    // were resolved but then silently discarded. This must inject regardless
    // of what the raw string contains.
    const std::string resp = nzltx::InjectDroppedPathsIntoRequest(
        R"({"id": 1, "method": "ui.resolveDroppedFiles", "params": {}})",
        {"C:\\Users\\me\\Desktop\\clip.mp4"});
    const json j = json::parse(resp);
    REQUIRE(j["params"]["__droppedPaths"].size() == 1);
    CHECK(j["params"]["__droppedPaths"][0] == "C:\\Users\\me\\Desktop\\clip.mp4");
    // Everything else about the request is preserved verbatim.
    CHECK(j["id"] == 1);
    CHECK(j["method"] == "ui.resolveDroppedFiles");
}

TEST_CASE("InjectDroppedPathsIntoRequest creates 'params' when it is missing entirely") {
    const std::string resp = nzltx::InjectDroppedPathsIntoRequest(
        R"({"id": 2, "method": "ui.resolveDroppedFiles"})", {"C:\\a.png"});
    const json j = json::parse(resp);
    REQUIRE(j.contains("params"));
    REQUIRE(j["params"]["__droppedPaths"].size() == 1);
    CHECK(j["params"]["__droppedPaths"][0] == "C:\\a.png");
}

TEST_CASE("InjectDroppedPathsIntoRequest overwrites a page-authored __droppedPaths with "
          "an empty array when there are no real paths (anti-spoofing)") {
    const std::string resp = nzltx::InjectDroppedPathsIntoRequest(
        R"({"id": 3, "method": "ui.resolveDroppedFiles",
            "params": {"__droppedPaths": ["C:\\Windows\\System32\\config\\SAM"]}})",
        {});
    const json j = json::parse(resp);
    REQUIRE(j["params"]["__droppedPaths"].is_array());
    CHECK(j["params"]["__droppedPaths"].empty());
}

TEST_CASE("InjectDroppedPathsIntoRequest overwrites a page-authored __droppedPaths with the "
          "real resolved paths, never leaving the spoofed value") {
    const std::string resp = nzltx::InjectDroppedPathsIntoRequest(
        R"({"id": 4, "method": "ui.resolveDroppedFiles",
            "params": {"__droppedPaths": ["C:\\spoofed\\path.txt"]}})",
        {"C:\\real\\clip.mp4"});
    const json j = json::parse(resp);
    REQUIRE(j["params"]["__droppedPaths"].size() == 1);
    CHECK(j["params"]["__droppedPaths"][0] == "C:\\real\\clip.mp4");
}

TEST_CASE("InjectDroppedPathsIntoRequest passes a non-object / unparsable request through unchanged") {
    CHECK(nzltx::InjectDroppedPathsIntoRequest("not json at all", {"C:\\a.png"}) ==
          "not json at all");
    CHECK(nzltx::InjectDroppedPathsIntoRequest("[]", {"C:\\a.png"}) == "[]");
    CHECK(nzltx::InjectDroppedPathsIntoRequest(R"("just a string")", {"C:\\a.png"}) ==
          R"("just a string")");
}

TEST_CASE("InjectDroppedPathsIntoRequest preserves a non-object 'params' by replacing it "
          "with a fresh object carrying only __droppedPaths") {
    // 'params' present but the wrong type (e.g. a stray string) - still an
    // object overall, so this is NOT the passthrough case; 'params' is
    // replaced with an object so the key has somewhere to live.
    const std::string resp = nzltx::InjectDroppedPathsIntoRequest(
        R"({"id": 5, "method": "ui.resolveDroppedFiles", "params": "oops"})", {"C:\\a.png"});
    const json j = json::parse(resp);
    REQUIRE(j["params"].is_object());
    REQUIRE(j["params"]["__droppedPaths"].size() == 1);
    CHECK(j["params"]["__droppedPaths"][0] == "C:\\a.png");
}
