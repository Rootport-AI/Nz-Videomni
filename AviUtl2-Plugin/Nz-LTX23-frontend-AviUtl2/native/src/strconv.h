// strconv.h - UTF-8 <-> UTF-16 conversion helpers for Nz-LTX23.
// All comments are ASCII/English on purpose so this translation unit
// compiles cleanly under any default code page.
#pragma once

#include <string>

namespace nzltx {

// Convert a UTF-8 encoded narrow string to a UTF-16 (wide) string.
// Invalid byte sequences are dropped by the underlying Win32 conversion.
std::wstring Utf8ToWide(const std::string& utf8);

// Convert a UTF-16 (wide) string to a UTF-8 encoded narrow string.
std::string WideToUtf8(const std::wstring& wide);

} // namespace nzltx
