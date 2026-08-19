// settings.h - persistent plugin settings (contract v4: settings.get/settings.set).
//
// Stores a small UTF-8 JSON document at %LOCALAPPDATA%\NzVideomni\settings.json
// (see log.h::AppDataDir() - the same parent directory as logs/downloads/
// captures). Currently the only setting is the backend connection base URL,
// which lets the connection target be changed at runtime instead of being
// compiled in (see bridge_core.h::kBackendBaseUrl, still used as the built-in
// default value).
//
// Load() self-heals: a missing file, malformed JSON, or an invalid/missing
// 'baseUrl' field in the file all fall back to the built-in default silently,
// without ever writing to the file - only a successful SetBaseUrl() persists.
// SettingsStore is thread-safe (a mutex guards the in-memory value; the disk
// write in SetBaseUrl runs while holding it too, so concurrent Load/Save calls
// cannot interleave and corrupt the file).
//
// ASCII-only source.
#pragma once

#include <mutex>
#include <string>

#include "bridge_core.h"  // kBackendBaseUrl: the built-in default base URL

namespace nzvideomni {

// Validate and normalize a candidate backend base URL:
//   - must start with "http://" or "https://" (case-sensitive)
//   - the host component (everything up to the next '/', or the end of the
//     string, after the scheme) must be non-empty
//   - trailing '/' characters are stripped
// This is deliberately lightweight (no full RFC 3986 parse) - it only guards
// against the obviously-wrong inputs a settings UI could submit.
// Returns true and fills *out_normalized on success; false and fills
// *err_message (suitable for a BAD_REQUEST response) on failure.
bool NormalizeBaseUrl(const std::string& candidate, std::string* out_normalized,
                      std::string* err_message);

// Extract a base URL from the UTF-8 contents of settings.json. Malformed JSON,
// a non-object document, a missing/non-string 'baseUrl', or a 'baseUrl' that
// fails NormalizeBaseUrl all yield kBackendBaseUrl. Never throws.
std::string ParseSettingsJson(const std::string& json_utf8);

// Build the UTF-8 contents of settings.json for a given (already-normalized)
// base URL.
std::string BuildSettingsJson(const std::string& base_url);

// Thread-safe persistent store for the plugin's settings (currently just the
// backend base URL). One instance is normally shared for the plugin's
// lifetime (owned by Bridge); the file path is injected so tests can point it
// at a temporary file instead of the real %LOCALAPPDATA% location.
class SettingsStore {
public:
    // file_path: full path to settings.json. An empty path disables
    // persistence entirely (Load() leaves the built-in default in place,
    // SetBaseUrl() still validates/applies the value in memory but writes
    // nothing to disk) - used when %LOCALAPPDATA% is not available.
    explicit SettingsStore(std::wstring file_path);

    SettingsStore(const SettingsStore&) = delete;
    SettingsStore& operator=(const SettingsStore&) = delete;

    // Read settings.json if present and apply a valid baseUrl from it;
    // otherwise (missing file, malformed JSON, invalid value) leave the
    // built-in default in place. Never writes. Call once at startup.
    void Load();

    // The current base URL (thread-safe).
    std::string GetBaseUrl() const;

    // Validate `candidate` (see NormalizeBaseUrl). On success, persists the
    // normalized value to disk (best-effort: a write failure is logged via
    // log.h but does not fail the call) and applies it as the current value,
    // filling *out_normalized. On validation failure, returns false, fills
    // *err_message, and leaves both the current value and the file untouched.
    bool SetBaseUrl(const std::string& candidate, std::string* out_normalized,
                    std::string* err_message);

private:
    mutable std::mutex mutex_;
    std::wstring file_path_;
    std::string base_url_;
};

}  // namespace nzvideomni
