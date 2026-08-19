// log.h - thin logging facade over the AviUtl2 LOG_HANDLE.
// The host limits log messages to 1024 characters, so all helpers truncate.
// All comments are ASCII/English on purpose.
#pragma once

#include <string>

struct LOG_HANDLE; // forward declared; defined in logger2.h

namespace nzvideomni {

// Maximum number of wide characters the host accepts per log line (incl. NUL).
inline constexpr size_t kMaxLogChars = 1024;

// Store the host logger handle (may be nullptr before InitializeLogger runs).
void SetLogHandle(LOG_HANDLE* handle);

// Return the per-user data directory for this plugin:
//   %LOCALAPPDATA%\NzVideomni
// Returns an empty string if the environment variable is not available.
std::wstring AppDataDir();

// Return the directory the file log is written to:
//   %LOCALAPPDATA%\NzVideomni\logs
std::wstring LogDir();

// Log at the given level. Messages longer than kMaxLogChars-1 are truncated.
// Each call also mirrors to OutputDebugStringW so logs are visible in a
// debugger even when the host handle is not yet available.
void LogInfo(const std::wstring& message);
void LogWarn(const std::wstring& message);
void LogError(const std::wstring& message);

} // namespace nzvideomni
