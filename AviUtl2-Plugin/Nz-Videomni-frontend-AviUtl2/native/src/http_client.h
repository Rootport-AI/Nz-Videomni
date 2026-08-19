// http_client.h - a small WinHTTP wrapper with a worker-thread job queue.
//
// The bridge must never block the UI thread on a synchronous HTTP call, so this
// module owns a pool of worker threads. Requests are submitted with a
// completion callback that is invoked *on a worker thread*; the caller is
// responsible for marshalling any UI work back to the UI thread.
//
// Two request shapes are supported:
//   * RequestAsync  - GET/POST(JSON)/DELETE, response body buffered in memory.
//   * DownloadAsync - GET streamed straight to a file on disk.
//
// Synchronous variants (RequestSync / DownloadSync) run on the calling thread
// and exist mainly so the behaviour can be exercised directly by integration
// tests against a live backend.
//
// All strings crossing this boundary are UTF-8, except the on-disk destination
// path which is a native wide (UTF-16) path.
//
// ASCII-only source.
#pragma once

#include <functional>
#include <string>

namespace nzvideomni {

// Implementation details exposed only so the doctest unit tests can exercise
// the URL-cracking logic directly (see native/tests/test_http_client.cpp).
// Not part of the public HttpClient contract.
namespace detail {

// Split a full URL into scheme flags, host, port and the path+query tail.
struct UrlParts {
    bool ok = false;
    bool secure = false;
    std::wstring host;
    unsigned short port = 0;  // same width as WinHTTP's INTERNET_PORT
    std::wstring path;        // path plus query, e.g. L"/api/v1/status?x=1"
};

// Wraps WinHttpCrackUrl(). Defined in http_client.cpp.
UrlParts CrackUrl(const std::wstring& url);

}  // namespace detail

// Transport-level error classification. kNone means the HTTP exchange itself
// completed (an HTTP status was received); a non-zero HTTP status is not a
// transport error and is reported through HttpResponse::status.
enum class TransportError {
    kNone = 0,
    kUnreachable,  // connect/resolve/connection reset -> BACKEND_UNREACHABLE
    kTimeout,      // WinHTTP timed out               -> BACKEND_TIMEOUT
};

// Result of a buffered request.
struct HttpResponse {
    TransportError transport = TransportError::kNone;
    std::string transport_message;  // human-readable detail on transport failure
    int status = 0;                 // HTTP status code (valid when transport==kNone)
    std::string body;               // response body, UTF-8 (valid when transport==kNone)
};

// Result of a streamed download.
struct DownloadResult {
    TransportError transport = TransportError::kNone;
    std::string transport_message;
    int status = 0;                    // HTTP status code (valid when transport==kNone)
    std::wstring file_path;            // destination path (set when status==200 and saved)
    long long size_bytes = 0;          // bytes written to disk (status==200)
    std::string response_summary;      // short body text for a non-200 response
};

// Default timeouts (milliseconds).
inline constexpr int kDefaultRequestTimeoutMs = 30000;    // 30 s
inline constexpr int kDefaultDownloadTimeoutMs = 300000;  // 300 s
inline constexpr int kUploadTimeoutMs = 120000;           // 120 s

class HttpClient {
public:
    using RequestCallback = std::function<void(HttpResponse)>;
    using DownloadCallback = std::function<void(DownloadResult)>;

    // Starts the worker pool. Must be constructed after DLL load (never during
    // static initialisation / DllMain, to avoid the loader lock).
    HttpClient();
    ~HttpClient();

    HttpClient(const HttpClient&) = delete;
    HttpClient& operator=(const HttpClient&) = delete;

    // Submit a buffered request. url is a full "http://host:port/path?query".
    // method is "GET" | "POST" | "DELETE". A non-empty body_json is sent with
    // Content-Type application/json. on_done runs on a worker thread.
    void RequestAsync(std::string url, std::string method, std::string body_json,
                      int timeout_ms, RequestCallback on_done);

    // Submit a streamed GET download to dest_path (native wide path). Parent
    // directories are created as needed. on_done runs on a worker thread.
    void DownloadAsync(std::string url, std::wstring dest_path, int timeout_ms,
                       DownloadCallback on_done);

    // Submit a multipart/form-data POST that streams file_path (native wide path)
    // as a single form field. The file is read from disk in chunks so large media
    // uploads do not buffer in memory. on_done runs on a worker thread.
    void UploadFileAsync(std::string url, std::wstring file_path,
                         std::string field_name, std::string filename,
                         std::string content_type, int timeout_ms,
                         RequestCallback on_done);

    // Run an arbitrary job on the worker pool. Used by the bridge to drive the
    // frame-capture render/encode off the UI thread without a dedicated thread.
    void Post(std::function<void()> job);

    // Synchronous forms (run on the calling thread).
    HttpResponse RequestSync(const std::string& url, const std::string& method,
                             const std::string& body_json, int timeout_ms);
    DownloadResult DownloadSync(const std::string& url, const std::wstring& dest_path,
                                int timeout_ms);
    HttpResponse UploadFileSync(const std::string& url, const std::wstring& file_path,
                                const std::string& field_name,
                                const std::string& filename,
                                const std::string& content_type, int timeout_ms);

private:
    struct Impl;
    Impl* impl_ = nullptr;
};

}  // namespace nzvideomni
