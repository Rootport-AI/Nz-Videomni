// log.cpp - implementation of the logging facade.
//
// Two sinks are driven from every LogInfo/LogWarn/LogError call:
//   1. The host LOG_HANDLE (visible in the AviUtl2 log window). Limited to
//      1024 wide characters, so those messages are clamped.
//   2. A UTF-8 file log at %LOCALAPPDATA%\NzLTX23\logs\plugin.log with a
//      timestamp on each line. AviUtl2 never writes its own log to disk, so
//      this file is the only channel automated verification can inspect.
//
// File I/O failures are swallowed on purpose: logging must never interfere
// with the plugin's real work. All comments are ASCII/English so this
// translation unit compiles cleanly under any default code page.
#include "log.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <cstdlib>
#include <mutex>
#include <string>

#include "logger2.h"  // LOG_HANDLE definition (SDK header, Shift-JIS comments)
#include "strconv.h"  // WideToUtf8

namespace nzltx {

namespace {

LOG_HANDLE* g_log_handle = nullptr;
std::mutex  g_file_mutex;

// Truncate to the host limit (leaving room for the terminating NUL).
std::wstring Clamp(const std::wstring& message) {
    if (message.size() < kMaxLogChars) {
        return message;
    }
    return message.substr(0, kMaxLogChars - 1);
}

using LogFn = void (*)(LOG_HANDLE*, LPCWSTR);

// Read an environment variable into a wide string ("" if unset). Uses the
// Win32 API (not _wgetenv) to avoid the CRT deprecation warning under /W4.
std::wstring EnvVar(const wchar_t* name) {
    const DWORD needed = ::GetEnvironmentVariableW(name, nullptr, 0);
    if (needed == 0) {
        return std::wstring();
    }
    std::wstring value(needed, L'\0');
    const DWORD written = ::GetEnvironmentVariableW(name, value.data(), needed);
    value.resize(written);
    return value;
}

// Recursively create every component of a directory path. Best effort.
void EnsureDir(const std::wstring& path) {
    if (path.empty()) {
        return;
    }
    for (size_t i = 0; i < path.size(); ++i) {
        if (path[i] == L'\\' || path[i] == L'/') {
            if (i > 0) {
                const std::wstring part = path.substr(0, i);
                ::CreateDirectoryW(part.c_str(), nullptr);
            }
        }
    }
    ::CreateDirectoryW(path.c_str(), nullptr);
}

// "2026-07-07 12:34:56.789 " local-time prefix.
std::string TimestampPrefix() {
    SYSTEMTIME st = {};
    ::GetLocalTime(&st);
    char buffer[64];
    ::_snprintf_s(buffer, sizeof(buffer), _TRUNCATE,
                  "%04d-%02d-%02d %02d:%02d:%02d.%03d ",
                  st.wYear, st.wMonth, st.wDay,
                  st.wHour, st.wMinute, st.wSecond, st.wMilliseconds);
    return std::string(buffer);
}

// Append one line (already UTF-8, no trailing newline) to the file log.
void WriteFileLog(const char* level, const std::wstring& message) {
    const std::wstring dir = LogDir();
    if (dir.empty()) {
        return;
    }
    std::lock_guard<std::mutex> guard(g_file_mutex);

    EnsureDir(dir);
    const std::wstring file = dir + L"\\plugin.log";

    HANDLE h = ::CreateFileW(file.c_str(),
                             FILE_APPEND_DATA,
                             FILE_SHARE_READ | FILE_SHARE_WRITE,
                             nullptr,
                             OPEN_ALWAYS,
                             FILE_ATTRIBUTE_NORMAL,
                             nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        return;
    }

    std::string line = TimestampPrefix();
    line += "[";
    line += level;
    line += "] ";
    line += WideToUtf8(message);
    line += "\r\n";

    DWORD written = 0;
    ::WriteFile(h, line.data(), static_cast<DWORD>(line.size()), &written, nullptr);
    ::CloseHandle(h);
}

void Emit(const char* level, LogFn fn, const std::wstring& message) {
    const std::wstring text = Clamp(message);

    ::OutputDebugStringW(L"[Nz-LTX23] ");
    ::OutputDebugStringW(text.c_str());
    ::OutputDebugStringW(L"\n");

    if (g_log_handle != nullptr && fn != nullptr) {
        fn(g_log_handle, text.c_str());
    }

    WriteFileLog(level, message);
}

}  // namespace

void SetLogHandle(LOG_HANDLE* handle) {
    g_log_handle = handle;
}

std::wstring AppDataDir() {
    const std::wstring base = EnvVar(L"LOCALAPPDATA");
    if (base.empty()) {
        return std::wstring();
    }
    return base + L"\\NzLTX23";
}

std::wstring LogDir() {
    const std::wstring base = AppDataDir();
    if (base.empty()) {
        return std::wstring();
    }
    return base + L"\\logs";
}

void LogInfo(const std::wstring& message) {
    Emit("INFO", g_log_handle ? g_log_handle->info : nullptr, message);
}

void LogWarn(const std::wstring& message) {
    Emit("WARN", g_log_handle ? g_log_handle->warn : nullptr, message);
}

void LogError(const std::wstring& message) {
    Emit("ERROR", g_log_handle ? g_log_handle->error : nullptr, message);
}

}  // namespace nzltx
