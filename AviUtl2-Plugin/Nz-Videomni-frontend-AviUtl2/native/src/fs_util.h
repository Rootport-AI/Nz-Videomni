// fs_util.h - pure path/numbering/BOM string helpers for Nz-Videomni (contract v6
// fs.* bridge methods: fs.listFiles, and backend.downloadVideo's
// destDir/fileName/noClobber extension). The UTF-8 BOM helpers were originally
// added for the fs.readTextFile / fs.writeTextFileAtomic methods, which were
// removed on 2026-07-18 (batch A2V dropped CSV state management); the helpers
// themselves are retained as general-purpose string utilities.
//
// Every function here is a pure string transform (no filesystem access, no
// Win32 API) so it is fully unit-testable with doctest (see
// native/tests/test_fs_util.cpp). Paths and file names are plain UTF-8
// std::string throughout, matching the JSON strings crossing the bridge.
// Deliberately NOT implemented via std::filesystem::path: constructing a
// std::filesystem::path from a narrow std::string interprets the bytes in the
// process's native (system ANSI) code page on Windows, which would corrupt any
// non-ASCII UTF-8 byte in a path. Plain byte-wise scanning for '\\'/'/'
// separators is safe on UTF-8 because every continuation byte is >= 0x80 and
// can never be mistaken for an ASCII separator - the same reasoning
// settings.cpp's (wstring) ParentDir already relies on.
//
// All comments are ASCII/English.
#pragma once

#include <set>
#include <string>
#include <vector>

namespace nzvideomni {

// --- Extension matching (fs.listFiles' `extensions` filter) ----------------

// Lowercased extension of `file_name`, including the leading '.', or "" if
// there is none. A name whose ONLY '.' is the leading character (e.g.
// ".gitignore") is treated as extensionless (matches
// webui/src/bridge/mockBridge.ts's extnameLower: `idx > 0`, not `idx >= 0`).
std::string ExtensionLower(const std::string& file_name);

// True when file_name's extension (see ExtensionLower) case-insensitively
// equals one of `extensions` (each entry compared the same way, so callers
// may pass mixed-case entries with or without a leading '.' being pre-cased -
// only the leading '.' itself is significant, e.g. "wav" without a dot never
// matches). An empty `extensions` list matches everything (no filter).
bool MatchesAnyExtension(const std::string& file_name, const std::vector<std::string>& extensions);

// --- Collision-avoiding numbering (backend.downloadVideo's noClobber) ------

// Returns base_name unchanged when n <= 1; otherwise inserts "_<n>" right
// before the extension (extension = the same split ExtensionLower uses, i.e.
// the substring from the last '.' when that '.' is not the first character).
// A base_name with no extension gets the suffix appended directly, e.g.
// MakeNumberedName("clip.mp4", 2) == "clip_2.mp4";
// MakeNumberedName("README", 3)   == "README_3".
std::string MakeNumberedName(const std::string& base_name, int n);

// Finds the first name that is NOT in `existing_names`: desired_name itself
// if free, otherwise MakeNumberedName(desired_name, 2), then _3, _4, ... until
// a free name is found. Mirrors mockBridge.ts's dedupeFileName exactly (same
// numbering scheme), which is the noClobber behaviour documented in
// Docs/BRIDGE_CONTRACT.md Sec.4.20.
std::string NextAvailableName(const std::set<std::string>& existing_names,
                              const std::string& desired_name);

// --- UTF-8 BOM helpers (general-purpose; originally for the now-removed
// fs.readTextFile / fs.writeTextFileAtomic methods) ------------------------

// True when s begins with the 3-byte UTF-8 BOM (EF BB BF).
bool HasUtf8Bom(const std::string& s);

// Prepends a UTF-8 BOM, unless s already has one (idempotent - never produces
// a doubled BOM).
std::string AddUtf8Bom(const std::string& s);

// Strips a leading UTF-8 BOM if present; returns s unchanged otherwise.
std::string RemoveUtf8Bom(const std::string& s);

// --- Path helpers ------------------------------------------------------------

// Joins `dir` and `name` with a single backslash, unless dir is empty or
// already ends with a path separator ('\\' or '/', which is left as-is).
// Mirrors webui/src/bridge/mockBridge.ts's joinMockPath.
std::string JoinPath(const std::string& dir, const std::string& name);

// Returns the parent directory of `path` (everything before the last '\\' or
// '/'), or "" if path has no separator. Trailing separators in `path` are
// ignored first (so "C:\\a\\b\\" and "C:\\a\\b" both yield "C:\\a").
std::string ParentDirOf(const std::string& path);

}  // namespace nzvideomni
