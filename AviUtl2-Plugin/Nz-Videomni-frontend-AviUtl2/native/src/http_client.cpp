// http_client.cpp - WinHTTP implementation of the async HTTP client.
// See http_client.h for the contract. ASCII-only source.
#include "http_client.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <winhttp.h>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

#include "bridge_core.h"  // multipart framing helpers (BuildMultipartHeader, ...)
#include "log.h"
#include "strconv.h"

namespace nzvideomni {

namespace detail {

UrlParts CrackUrl(const std::wstring& url) {
    UrlParts parts;
    URL_COMPONENTS comp = {};
    comp.dwStructSize = sizeof(comp);
    wchar_t host[256] = {};
    wchar_t path[2048] = {};
    comp.lpszHostName = host;
    comp.dwHostNameLength = static_cast<DWORD>(std::size(host));
    comp.lpszUrlPath = path;
    comp.dwUrlPathLength = static_cast<DWORD>(std::size(path));
    // We do not split the query separately; keep it attached to the path.
    comp.lpszExtraInfo = nullptr;
    comp.dwExtraInfoLength = 0;

    if (!::WinHttpCrackUrl(url.c_str(), 0, 0, &comp)) {
        return parts;
    }

    parts.host.assign(comp.lpszHostName, comp.dwHostNameLength);
    std::wstring tail(comp.lpszUrlPath, comp.dwUrlPathLength);
    // Per WinHttpCrackUrl's documented Remarks: when lpszExtraInfo is NULL and
    // dwExtraInfoLength is 0, the query string (and fragment) is NOT split out
    // separately -- WinHTTP leaves it attached to lpszUrlPath instead. So the
    // path buffer above already contains "/path?query" in full; there is
    // nothing left to re-attach here.
    if (tail.empty()) {
        tail = L"/";
    }
    parts.path = tail;
    parts.port = comp.nPort;
    parts.secure = (comp.nScheme == INTERNET_SCHEME_HTTPS);
    parts.ok = true;
    return parts;
}

}  // namespace detail

namespace {

// Number of worker threads. Status polling, a video download and a (blocking,
// up to 10 s) frame-capture render can all be in flight at once, so four leaves
// headroom for concurrent HTTP work while a capture job occupies one thread.
constexpr unsigned kWorkerCount = 4;

// RAII wrapper for a WinHTTP HINTERNET handle.
class WinHttpHandle {
public:
    WinHttpHandle() = default;
    explicit WinHttpHandle(HINTERNET h) : h_(h) {}
    ~WinHttpHandle() { reset(); }

    WinHttpHandle(const WinHttpHandle&) = delete;
    WinHttpHandle& operator=(const WinHttpHandle&) = delete;

    WinHttpHandle(WinHttpHandle&& other) noexcept : h_(other.h_) { other.h_ = nullptr; }

    void reset(HINTERNET h = nullptr) {
        if (h_ != nullptr) {
            ::WinHttpCloseHandle(h_);
        }
        h_ = h;
    }

    HINTERNET get() const { return h_; }
    explicit operator bool() const { return h_ != nullptr; }

private:
    HINTERNET h_ = nullptr;
};

// Map a WinHTTP GetLastError() value to our transport classification.
TransportError ClassifyError(DWORD err) {
    switch (err) {
        case ERROR_WINHTTP_TIMEOUT:
            return TransportError::kTimeout;
        case ERROR_WINHTTP_NAME_NOT_RESOLVED:
        case ERROR_WINHTTP_CANNOT_CONNECT:
        case ERROR_WINHTTP_CONNECTION_ERROR:
        default:
            return TransportError::kUnreachable;
    }
}

std::string TransportDetail(const char* stage, DWORD err) {
    char buffer[128];
    ::_snprintf_s(buffer, sizeof(buffer), _TRUNCATE, "%s failed (winhttp error %lu)",
                  stage, static_cast<unsigned long>(err));
    return std::string(buffer);
}

// Open connect + request handles for a cracked URL and apply timeouts. Returns
// false with *err set on failure.
bool OpenRequest(HINTERNET session, const detail::UrlParts& parts,
                 const std::wstring& method, int timeout_ms, WinHttpHandle* connect,
                 WinHttpHandle* request, DWORD* err) {
    connect->reset(::WinHttpConnect(session, parts.host.c_str(), parts.port, 0));
    if (!*connect) {
        *err = ::GetLastError();
        return false;
    }
    const DWORD flags = parts.secure ? WINHTTP_FLAG_SECURE : 0;
    request->reset(::WinHttpOpenRequest(connect->get(), method.c_str(), parts.path.c_str(),
                                        nullptr, WINHTTP_NO_REFERER,
                                        WINHTTP_DEFAULT_ACCEPT_TYPES, flags));
    if (!*request) {
        *err = ::GetLastError();
        return false;
    }
    // Resolve gets a fixed budget; connect/send/receive share the caller's.
    ::WinHttpSetTimeouts(request->get(), 10000, timeout_ms, timeout_ms, timeout_ms);
    return true;
}

// Generate a unique multipart boundary: a fixed ASCII prefix plus a hex token
// derived from a tick count, a monotonically increasing counter and the current
// thread id. Only characters that never need escaping are used.
std::string MakeBoundary() {
    static std::atomic<unsigned long long> counter{0};
    const unsigned long long token =
        (static_cast<unsigned long long>(::GetTickCount64()) << 16) ^
        (counter.fetch_add(1) * 0x9E3779B97F4A7C15ULL) ^
        (static_cast<unsigned long long>(::GetCurrentThreadId()));
    char buffer[64];
    ::_snprintf_s(buffer, sizeof(buffer), _TRUNCATE, "----NzVideomniFormBoundary%016llX",
                  token);
    return std::string(buffer);
}

// Read the whole small response body of an in-flight request into memory. On a
// transport error sets *err and returns false.
bool ReadResponseBody(HINTERNET request, std::string* body, DWORD* err) {
    for (;;) {
        DWORD available = 0;
        if (!::WinHttpQueryDataAvailable(request, &available)) {
            *err = ::GetLastError();
            return false;
        }
        if (available == 0) {
            break;
        }
        std::string chunk(available, '\0');
        DWORD read = 0;
        if (!::WinHttpReadData(request, chunk.data(), available, &read)) {
            *err = ::GetLastError();
            return false;
        }
        if (read == 0) {
            break;
        }
        body->append(chunk.data(), read);
    }
    return true;
}

int QueryStatusCode(HINTERNET request) {
    DWORD status = 0;
    DWORD size = sizeof(status);
    if (!::WinHttpQueryHeaders(request,
                               WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                               WINHTTP_HEADER_NAME_BY_INDEX, &status, &size,
                               WINHTTP_NO_HEADER_INDEX)) {
        return 0;
    }
    return static_cast<int>(status);
}

// Diagnostic-only: on a non-2xx response, log the exact path (query included)
// that was sent, so a request-shape bug like the CrackUrl double-query one
// this replaces shows up in plugin.log without needing to reproduce with a
// debugger attached. Silent on 2xx to avoid log noise on the (dominant)
// success path.
void LogNonSuccessPath(const detail::UrlParts& parts, int status) {
    if (status >= 200 && status < 300) {
        return;
    }
    LogWarn(L"HttpClient: non-2xx status=" + std::to_wstring(status) +
            L" path=" + parts.path);
}

}  // namespace

struct HttpClient::Impl {
    HINTERNET session = nullptr;
    std::vector<std::thread> workers;
    std::deque<std::function<void()>> jobs;
    std::mutex mutex;
    std::condition_variable cv;
    bool stopping = false;

    void Enqueue(std::function<void()> job) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            jobs.push_back(std::move(job));
        }
        cv.notify_one();
    }

    void WorkerLoop() {
        for (;;) {
            std::function<void()> job;
            {
                std::unique_lock<std::mutex> lock(mutex);
                cv.wait(lock, [this] { return stopping || !jobs.empty(); });
                if (stopping && jobs.empty()) {
                    return;
                }
                job = std::move(jobs.front());
                jobs.pop_front();
            }
            job();
        }
    }
};

HttpClient::HttpClient() {
    impl_ = new Impl();
    impl_->session = ::WinHttpOpen(L"Nz-Videomni/0.2 (WinHTTP)",
                                   WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY,
                                   WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
    if (impl_->session == nullptr) {
        LogError(L"HttpClient: WinHttpOpen failed");
    }
    for (unsigned i = 0; i < kWorkerCount; ++i) {
        impl_->workers.emplace_back([this] { impl_->WorkerLoop(); });
    }
    LogInfo(L"HttpClient: worker pool started");
}

HttpClient::~HttpClient() {
    if (impl_ == nullptr) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        impl_->stopping = true;
    }
    impl_->cv.notify_all();
    for (auto& t : impl_->workers) {
        if (t.joinable()) {
            t.join();
        }
    }
    if (impl_->session != nullptr) {
        ::WinHttpCloseHandle(impl_->session);
        impl_->session = nullptr;
    }
    delete impl_;
    impl_ = nullptr;
}

HttpResponse HttpClient::RequestSync(const std::string& url, const std::string& method,
                                     const std::string& body_json, int timeout_ms,
                                     const std::string& content_type) {
    HttpResponse resp;
    if (impl_ == nullptr || impl_->session == nullptr) {
        resp.transport = TransportError::kUnreachable;
        resp.transport_message = "HTTP session is not available";
        return resp;
    }

    const detail::UrlParts parts = detail::CrackUrl(Utf8ToWide(url));
    if (!parts.ok) {
        resp.transport = TransportError::kUnreachable;
        resp.transport_message = "Malformed URL";
        return resp;
    }

    WinHttpHandle connect;
    WinHttpHandle request;
    DWORD err = 0;
    if (!OpenRequest(impl_->session, parts, Utf8ToWide(method), timeout_ms, &connect,
                     &request, &err)) {
        resp.transport = ClassifyError(err);
        resp.transport_message = TransportDetail("open request", err);
        return resp;
    }

    LPCWSTR headers = WINHTTP_NO_ADDITIONAL_HEADERS;
    DWORD header_len = 0;
    // Content-Type comes from the caller (default: JSON - see http_client.h).
    // Section 3-54 sends "application/octet-stream" here for a raw RGBA frame.
    const std::wstring ct_header =
        L"Content-Type: " + Utf8ToWide(content_type) + L"\r\n";
    if (!body_json.empty() && !content_type.empty()) {
        headers = ct_header.c_str();
        header_len = static_cast<DWORD>(ct_header.size());
    }

    if (!::WinHttpSendRequest(request.get(), headers, header_len,
                              body_json.empty() ? WINHTTP_NO_REQUEST_DATA
                                                : const_cast<char*>(body_json.data()),
                              static_cast<DWORD>(body_json.size()),
                              static_cast<DWORD>(body_json.size()), 0)) {
        err = ::GetLastError();
        resp.transport = ClassifyError(err);
        resp.transport_message = TransportDetail("send", err);
        return resp;
    }

    if (!::WinHttpReceiveResponse(request.get(), nullptr)) {
        err = ::GetLastError();
        resp.transport = ClassifyError(err);
        resp.transport_message = TransportDetail("receive", err);
        return resp;
    }

    resp.status = QueryStatusCode(request.get());
    LogNonSuccessPath(parts, resp.status);

    // Read the whole body into memory.
    for (;;) {
        DWORD available = 0;
        if (!::WinHttpQueryDataAvailable(request.get(), &available)) {
            err = ::GetLastError();
            resp.transport = ClassifyError(err);
            resp.transport_message = TransportDetail("read", err);
            resp.status = 0;
            resp.body.clear();
            return resp;
        }
        if (available == 0) {
            break;
        }
        std::string chunk(available, '\0');
        DWORD read = 0;
        if (!::WinHttpReadData(request.get(), chunk.data(), available, &read)) {
            err = ::GetLastError();
            resp.transport = ClassifyError(err);
            resp.transport_message = TransportDetail("read", err);
            resp.status = 0;
            resp.body.clear();
            return resp;
        }
        if (read == 0) {
            break;
        }
        resp.body.append(chunk.data(), read);
    }

    return resp;
}

DownloadResult HttpClient::DownloadSync(const std::string& url,
                                        const std::wstring& dest_path, int timeout_ms) {
    DownloadResult result;
    result.file_path = dest_path;
    if (impl_ == nullptr || impl_->session == nullptr) {
        result.transport = TransportError::kUnreachable;
        result.transport_message = "HTTP session is not available";
        return result;
    }

    const detail::UrlParts parts = detail::CrackUrl(Utf8ToWide(url));
    if (!parts.ok) {
        result.transport = TransportError::kUnreachable;
        result.transport_message = "Malformed URL";
        return result;
    }

    WinHttpHandle connect;
    WinHttpHandle request;
    DWORD err = 0;
    if (!OpenRequest(impl_->session, parts, L"GET", timeout_ms, &connect, &request, &err)) {
        result.transport = ClassifyError(err);
        result.transport_message = TransportDetail("open request", err);
        return result;
    }

    if (!::WinHttpSendRequest(request.get(), WINHTTP_NO_ADDITIONAL_HEADERS, 0,
                              WINHTTP_NO_REQUEST_DATA, 0, 0, 0)) {
        err = ::GetLastError();
        result.transport = ClassifyError(err);
        result.transport_message = TransportDetail("send", err);
        return result;
    }
    if (!::WinHttpReceiveResponse(request.get(), nullptr)) {
        err = ::GetLastError();
        result.transport = ClassifyError(err);
        result.transport_message = TransportDetail("receive", err);
        return result;
    }

    result.status = QueryStatusCode(request.get());

    // On a non-200 response, buffer the (small) error body for the caller's
    // failure summary and do not create the destination file.
    if (result.status != 200) {
        for (;;) {
            DWORD available = 0;
            if (!::WinHttpQueryDataAvailable(request.get(), &available) || available == 0) {
                break;
            }
            std::string chunk(available, '\0');
            DWORD read = 0;
            if (!::WinHttpReadData(request.get(), chunk.data(), available, &read) ||
                read == 0) {
                break;
            }
            result.response_summary.append(chunk.data(), read);
            if (result.response_summary.size() > 512) {
                result.response_summary.resize(512);
                break;
            }
        }
        result.file_path.clear();
        return result;
    }

    // Ensure the destination directory exists, then stream the body to disk.
    const size_t slash = dest_path.find_last_of(L"\\/");
    if (slash != std::wstring::npos) {
        const std::wstring dir = dest_path.substr(0, slash);
        for (size_t i = 0; i < dir.size(); ++i) {
            if (dir[i] == L'\\' || dir[i] == L'/') {
                if (i > 0) {
                    ::CreateDirectoryW(dir.substr(0, i).c_str(), nullptr);
                }
            }
        }
        ::CreateDirectoryW(dir.c_str(), nullptr);
    }

    HANDLE file = ::CreateFileW(dest_path.c_str(), GENERIC_WRITE, 0, nullptr,
                                CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        result.transport = TransportError::kUnreachable;  // treated as failure upstream
        result.transport_message = "Could not open destination file";
        result.status = 0;
        result.file_path.clear();
        return result;
    }

    for (;;) {
        DWORD available = 0;
        if (!::WinHttpQueryDataAvailable(request.get(), &available)) {
            err = ::GetLastError();
            ::CloseHandle(file);
            ::DeleteFileW(dest_path.c_str());
            result.transport = ClassifyError(err);
            result.transport_message = TransportDetail("read", err);
            result.status = 0;
            result.size_bytes = 0;
            result.file_path.clear();
            return result;
        }
        if (available == 0) {
            break;
        }
        std::string chunk(available, '\0');
        DWORD read = 0;
        if (!::WinHttpReadData(request.get(), chunk.data(), available, &read)) {
            err = ::GetLastError();
            ::CloseHandle(file);
            ::DeleteFileW(dest_path.c_str());
            result.transport = ClassifyError(err);
            result.transport_message = TransportDetail("read", err);
            result.status = 0;
            result.size_bytes = 0;
            result.file_path.clear();
            return result;
        }
        if (read == 0) {
            break;
        }
        DWORD written = 0;
        if (!::WriteFile(file, chunk.data(), read, &written, nullptr) || written != read) {
            ::CloseHandle(file);
            ::DeleteFileW(dest_path.c_str());
            result.transport = TransportError::kUnreachable;
            result.transport_message = "Could not write destination file";
            result.status = 0;
            result.size_bytes = 0;
            result.file_path.clear();
            return result;
        }
        result.size_bytes += written;
    }

    ::CloseHandle(file);
    return result;
}

HttpResponse HttpClient::UploadFileSync(const std::string& url,
                                        const std::wstring& file_path,
                                        const std::string& field_name,
                                        const std::string& filename,
                                        const std::string& content_type,
                                        int timeout_ms) {
    HttpResponse resp;
    if (impl_ == nullptr || impl_->session == nullptr) {
        resp.transport = TransportError::kUnreachable;
        resp.transport_message = "HTTP session is not available";
        return resp;
    }

    // Open the source file and learn its size (a missing file is normally caught
    // earlier by the bridge, but guard here too).
    HANDLE file = ::CreateFileW(file_path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                                nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        resp.transport = TransportError::kUnreachable;
        resp.transport_message = "Could not open upload source file";
        return resp;
    }
    LARGE_INTEGER file_size = {};
    if (!::GetFileSizeEx(file, &file_size)) {
        ::CloseHandle(file);
        resp.transport = TransportError::kUnreachable;
        resp.transport_message = "Could not size upload source file";
        return resp;
    }

    const std::string boundary = MakeBoundary();
    const std::string header =
        BuildMultipartHeader(boundary, field_name, filename, content_type);
    const std::string footer = BuildMultipartFooter(boundary);
    const unsigned long long total =
        static_cast<unsigned long long>(header.size()) +
        static_cast<unsigned long long>(file_size.QuadPart) +
        static_cast<unsigned long long>(footer.size());

    const detail::UrlParts parts = detail::CrackUrl(Utf8ToWide(url));
    if (!parts.ok) {
        ::CloseHandle(file);
        resp.transport = TransportError::kUnreachable;
        resp.transport_message = "Malformed URL";
        return resp;
    }

    WinHttpHandle connect;
    WinHttpHandle request;
    DWORD err = 0;
    if (!OpenRequest(impl_->session, parts, L"POST", timeout_ms, &connect, &request,
                     &err)) {
        ::CloseHandle(file);
        resp.transport = ClassifyError(err);
        resp.transport_message = TransportDetail("open request", err);
        return resp;
    }

    const std::wstring ct_header =
        L"Content-Type: " + Utf8ToWide(MultipartContentType(boundary)) + L"\r\n";

    // Send the headers with the known total length, then stream header + file +
    // footer with WinHttpWriteData so the payload never buffers fully in memory.
    if (!::WinHttpSendRequest(request.get(), ct_header.c_str(),
                              static_cast<DWORD>(ct_header.size()),
                              WINHTTP_NO_REQUEST_DATA, 0,
                              static_cast<DWORD>(total), 0)) {
        err = ::GetLastError();
        ::CloseHandle(file);
        resp.transport = ClassifyError(err);
        resp.transport_message = TransportDetail("send", err);
        return resp;
    }

    auto write_all = [&request, &err](const char* data, size_t size) -> bool {
        size_t sent = 0;
        while (sent < size) {
            DWORD written = 0;
            const DWORD chunk =
                static_cast<DWORD>((size - sent) > 0x100000 ? 0x100000 : (size - sent));
            if (!::WinHttpWriteData(request.get(), data + sent, chunk, &written)) {
                err = ::GetLastError();
                return false;
            }
            sent += written;
        }
        return true;
    };

    bool write_ok = write_all(header.data(), header.size());
    if (write_ok) {
        std::string buffer(0x100000, '\0');  // 1 MiB streaming buffer
        for (;;) {
            DWORD read = 0;
            if (!::ReadFile(file, buffer.data(), static_cast<DWORD>(buffer.size()),
                            &read, nullptr)) {
                err = ERROR_READ_FAULT;
                write_ok = false;
                break;
            }
            if (read == 0) {
                break;  // end of file
            }
            if (!write_all(buffer.data(), read)) {
                write_ok = false;
                break;
            }
        }
    }
    if (write_ok) {
        write_ok = write_all(footer.data(), footer.size());
    }
    ::CloseHandle(file);
    if (!write_ok) {
        resp.transport = ClassifyError(err);
        resp.transport_message = TransportDetail("upload write", err);
        return resp;
    }

    if (!::WinHttpReceiveResponse(request.get(), nullptr)) {
        err = ::GetLastError();
        resp.transport = ClassifyError(err);
        resp.transport_message = TransportDetail("receive", err);
        return resp;
    }

    resp.status = QueryStatusCode(request.get());
    LogNonSuccessPath(parts, resp.status);
    if (!ReadResponseBody(request.get(), &resp.body, &err)) {
        resp.transport = ClassifyError(err);
        resp.transport_message = TransportDetail("read", err);
        resp.status = 0;
        resp.body.clear();
    }
    return resp;
}

void HttpClient::RequestAsync(std::string url, std::string method, std::string body_json,
                              int timeout_ms, RequestCallback on_done) {
    if (impl_ == nullptr) {
        return;
    }
    impl_->Enqueue([this, url = std::move(url), method = std::move(method),
                    body = std::move(body_json), timeout_ms,
                    cb = std::move(on_done)]() mutable {
        HttpResponse resp = RequestSync(url, method, body, timeout_ms);
        if (cb) {
            cb(std::move(resp));
        }
    });
}

void HttpClient::DownloadAsync(std::string url, std::wstring dest_path, int timeout_ms,
                               DownloadCallback on_done) {
    if (impl_ == nullptr) {
        return;
    }
    impl_->Enqueue([this, url = std::move(url), dest = std::move(dest_path), timeout_ms,
                    cb = std::move(on_done)]() mutable {
        DownloadResult result = DownloadSync(url, dest, timeout_ms);
        if (cb) {
            cb(std::move(result));
        }
    });
}

void HttpClient::UploadFileAsync(std::string url, std::wstring file_path,
                                 std::string field_name, std::string filename,
                                 std::string content_type, int timeout_ms,
                                 RequestCallback on_done) {
    if (impl_ == nullptr) {
        return;
    }
    impl_->Enqueue([this, url = std::move(url), file = std::move(file_path),
                    field = std::move(field_name), name = std::move(filename),
                    ct = std::move(content_type), timeout_ms,
                    cb = std::move(on_done)]() mutable {
        HttpResponse resp = UploadFileSync(url, file, field, name, ct, timeout_ms);
        if (cb) {
            cb(std::move(resp));
        }
    });
}

void HttpClient::Post(std::function<void()> job) {
    if (impl_ == nullptr || !job) {
        return;
    }
    impl_->Enqueue(std::move(job));
}

}  // namespace nzvideomni
