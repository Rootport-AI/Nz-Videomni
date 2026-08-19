// settings.cpp - implementation of persistent plugin settings. See settings.h.
// ASCII-only source.
#include "settings.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <utility>

#include "json.hpp"
#include "log.h"

namespace nzltx {

namespace {

using json = nlohmann::json;

// Match a scheme prefix ("http://" / "https://"). On success fills *rest with
// the remainder of the string after the scheme.
bool HasScheme(const std::string& s, const char* scheme, std::string* rest) {
    const size_t len = std::char_traits<char>::length(scheme);
    if (s.size() < len || s.compare(0, len, scheme) != 0) {
        return false;
    }
    *rest = s.substr(len);
    return true;
}

// Recursively create every component of a directory path. Best effort; the
// same approach as log.cpp's EnsureDir, duplicated here so this translation
// unit does not need to reach into log.cpp's anonymous namespace.
void EnsureDir(const std::wstring& path) {
    if (path.empty()) {
        return;
    }
    for (size_t i = 0; i < path.size(); ++i) {
        if (path[i] == L'\\' || path[i] == L'/') {
            if (i > 0) {
                ::CreateDirectoryW(path.substr(0, i).c_str(), nullptr);
            }
        }
    }
    ::CreateDirectoryW(path.c_str(), nullptr);
}

std::wstring ParentDir(const std::wstring& file_path) {
    const size_t slash = file_path.find_last_of(L"\\/");
    if (slash == std::wstring::npos) {
        return std::wstring();
    }
    return file_path.substr(0, slash);
}

// Read an entire file as raw bytes into a std::string (the bytes are UTF-8
// text; no transcoding happens here). Returns false if the file does not
// exist or cannot be fully read. A generous 16 MiB cap guards against reading
// something unexpected at this path.
bool ReadFileUtf8(const std::wstring& path, std::string* out) {
    HANDLE h = ::CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                             OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        return false;
    }
    LARGE_INTEGER size = {};
    if (!::GetFileSizeEx(h, &size) || size.QuadPart < 0 ||
        size.QuadPart > (16LL * 1024 * 1024)) {
        ::CloseHandle(h);
        return false;
    }
    std::string content(static_cast<size_t>(size.QuadPart), '\0');
    BOOL ok = TRUE;
    DWORD read = 0;
    if (!content.empty()) {
        ok = ::ReadFile(h, content.data(), static_cast<DWORD>(content.size()), &read,
                        nullptr);
    }
    ::CloseHandle(h);
    if (!ok || static_cast<size_t>(read) != content.size()) {
        return false;
    }
    *out = std::move(content);
    return true;
}

// Overwrite (or create) a file with raw UTF-8 bytes, creating the parent
// directory first if needed. Returns false on any I/O failure.
bool WriteFileUtf8(const std::wstring& path, const std::string& content) {
    EnsureDir(ParentDir(path));
    HANDLE h = ::CreateFileW(path.c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr,
                             CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        return false;
    }
    DWORD written = 0;
    BOOL ok = TRUE;
    if (!content.empty()) {
        ok = ::WriteFile(h, content.data(), static_cast<DWORD>(content.size()), &written,
                         nullptr);
    }
    ::CloseHandle(h);
    return ok && static_cast<size_t>(written) == content.size();
}

}  // namespace

bool NormalizeBaseUrl(const std::string& candidate, std::string* out_normalized,
                      std::string* err_message) {
    std::string rest;
    const bool has_scheme =
        HasScheme(candidate, "http://", &rest) || HasScheme(candidate, "https://", &rest);
    if (!has_scheme) {
        *err_message = "baseUrl must start with 'http://' or 'https://'";
        return false;
    }
    const size_t slash = rest.find('/');
    const std::string host = (slash == std::string::npos) ? rest : rest.substr(0, slash);
    if (host.empty()) {
        *err_message = "baseUrl is missing a host";
        return false;
    }
    std::string normalized = candidate;
    while (!normalized.empty() && normalized.back() == '/') {
        normalized.pop_back();
    }
    *out_normalized = std::move(normalized);
    return true;
}

std::string ParseSettingsJson(const std::string& json_utf8) {
    const json doc = json::parse(json_utf8, nullptr, /*allow_exceptions=*/false);
    if (doc.is_discarded() || !doc.is_object() || !doc.contains("baseUrl") ||
        !doc["baseUrl"].is_string()) {
        return kBackendBaseUrl;
    }
    std::string normalized;
    std::string err;
    if (!NormalizeBaseUrl(doc["baseUrl"].get<std::string>(), &normalized, &err)) {
        return kBackendBaseUrl;
    }
    return normalized;
}

std::string BuildSettingsJson(const std::string& base_url) {
    json doc;
    doc["baseUrl"] = base_url;
    return doc.dump(2);
}

SettingsStore::SettingsStore(std::wstring file_path)
    : file_path_(std::move(file_path)), base_url_(kBackendBaseUrl) {}

void SettingsStore::Load() {
    std::string loaded = kBackendBaseUrl;
    if (!file_path_.empty()) {
        std::string content;
        if (ReadFileUtf8(file_path_, &content)) {
            loaded = ParseSettingsJson(content);
        }
    }
    std::lock_guard<std::mutex> lock(mutex_);
    base_url_ = loaded;
}

std::string SettingsStore::GetBaseUrl() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return base_url_;
}

bool SettingsStore::SetBaseUrl(const std::string& candidate, std::string* out_normalized,
                               std::string* err_message) {
    std::string normalized;
    if (!NormalizeBaseUrl(candidate, &normalized, err_message)) {
        return false;
    }
    if (!file_path_.empty()) {
        if (!WriteFileUtf8(file_path_, BuildSettingsJson(normalized))) {
            LogWarn(std::wstring(L"SettingsStore: failed to write ") + file_path_);
        }
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        base_url_ = normalized;
    }
    *out_normalized = normalized;
    return true;
}

}  // namespace nzltx
