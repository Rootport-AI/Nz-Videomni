// fs_util.cpp - implementation. See fs_util.h for the contract.
#include "fs_util.h"

#include <cctype>

namespace nzvideomni {

namespace {

// ASCII-only lowercasing (extensions in practice are always ASCII, e.g.
// ".wav"/".csv"/".mp4"); non-ASCII bytes (UTF-8 continuation bytes, always
// >= 0x80) pass through untouched.
std::string ToLowerAscii(const std::string& s) {
    std::string out = s;
    for (char& c : out) {
        if (c >= 'A' && c <= 'Z') {
            c = static_cast<char>(c - 'A' + 'a');
        }
    }
    return out;
}

constexpr const char* kUtf8Bom = "\xEF\xBB\xBF";
constexpr size_t kUtf8BomLen = 3;

}  // namespace

// ---------------------------------------------------------------------------
// Extension matching
// ---------------------------------------------------------------------------

std::string ExtensionLower(const std::string& file_name) {
    const size_t dot = file_name.find_last_of('.');
    if (dot == std::string::npos || dot == 0) {
        return std::string();
    }
    return ToLowerAscii(file_name.substr(dot));
}

bool MatchesAnyExtension(const std::string& file_name, const std::vector<std::string>& extensions) {
    if (extensions.empty()) {
        return true;
    }
    const std::string ext = ExtensionLower(file_name);
    for (const std::string& candidate : extensions) {
        if (ext == ToLowerAscii(candidate)) {
            return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// Collision-avoiding numbering
// ---------------------------------------------------------------------------

std::string MakeNumberedName(const std::string& base_name, int n) {
    if (n <= 1) {
        return base_name;
    }
    const size_t dot = base_name.find_last_of('.');
    const std::string stem = (dot == std::string::npos || dot == 0) ? base_name : base_name.substr(0, dot);
    const std::string ext = (dot == std::string::npos || dot == 0) ? std::string() : base_name.substr(dot);
    return stem + "_" + std::to_string(n) + ext;
}

std::string NextAvailableName(const std::set<std::string>& existing_names,
                              const std::string& desired_name) {
    if (existing_names.find(desired_name) == existing_names.end()) {
        return desired_name;
    }
    int n = 2;
    std::string candidate = MakeNumberedName(desired_name, n);
    while (existing_names.find(candidate) != existing_names.end()) {
        ++n;
        candidate = MakeNumberedName(desired_name, n);
    }
    return candidate;
}

// ---------------------------------------------------------------------------
// UTF-8 BOM helpers
// ---------------------------------------------------------------------------

bool HasUtf8Bom(const std::string& s) {
    return s.size() >= kUtf8BomLen && s.compare(0, kUtf8BomLen, kUtf8Bom) == 0;
}

std::string AddUtf8Bom(const std::string& s) {
    if (HasUtf8Bom(s)) {
        return s;
    }
    return std::string(kUtf8Bom) + s;
}

std::string RemoveUtf8Bom(const std::string& s) {
    if (!HasUtf8Bom(s)) {
        return s;
    }
    return s.substr(kUtf8BomLen);
}

// ---------------------------------------------------------------------------
// Path helpers
// ---------------------------------------------------------------------------

std::string JoinPath(const std::string& dir, const std::string& name) {
    if (dir.empty()) {
        return name;
    }
    const char back = dir.back();
    const bool has_sep = (back == '\\' || back == '/');
    return has_sep ? (dir + name) : (dir + "\\" + name);
}

std::string ParentDirOf(const std::string& path) {
    std::string p = path;
    while (!p.empty() && (p.back() == '\\' || p.back() == '/')) {
        p.pop_back();
    }
    const size_t pos = p.find_last_of("\\/");
    if (pos == std::string::npos) {
        return std::string();
    }
    return p.substr(0, pos);
}

}  // namespace nzvideomni
