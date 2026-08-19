// alias_util.h - pure UTF-8 string builders/patchers for AviUtl2 object aliases.
//
// An AviUtl2 object alias is an INI-style UTF-8 document:
//   [Object]                 <- top meta section (holds the "frame=" range)
//   [Object.0]               <- effect 0
//   effect.name=<effect>     <- first line of every [Object.N] section
//   <item>=<value>           <- effect parameters
//   [Object.1]
//   effect.name=<effect>
//   ...
// These helpers ONLY build / rewrite that text - they never call the AviUtl2
// SDK. They exist so the bridge can (a) pin a created object's length to exactly
// the generated video (NormalizeAliasObjectFrameHeader), (b) repoint a video
// object at the downloaded mp4 (PatchAliasReplaceVideoFilePath), and (c) drop a
// provisional "generating..." text placeholder (BuildProvisionalTextAlias).
//
// The Japanese effect/item names AviUtl2 uses (from aviutl2_sdk WindowClient.cpp
// and the object property sheet) are embedded below as UTF-8 byte escapes so
// this translation unit stays strictly ASCII - the build deliberately compiles
// without /utf-8 (system ANSI == Shift-JIS on the target host), and \xHH escapes
// are inserted as literal bytes regardless of code page. Unit-tested with
// doctest (native/tests/test_alias_util.cpp). All comments are ASCII/English.
#pragma once

#include <string>

namespace nzltx {

// Remove a leading UTF-8 BOM (EF BB BF) if present; otherwise return s as-is.
// Call before parsing an alias whose origin might have added a BOM.
std::string StripUtf8Bom(const std::string& s);

// Rewrite the top "[Object]" section's "frame=" line to "frame=0,<length>",
// inserting it right after the "[Object]" header when absent. This is the MOST
// IMPORTANT helper: create_object_from_alias lets the alias' own frame range
// override the requested length, so pinning frame=0,<length> is what forces a
// created object to the intended duration. A leading BOM is stripped first; the
// input's line-ending style (CRLF vs LF) is preserved. If the alias has no
// "[Object]" section the input is returned unchanged.
std::string NormalizeAliasObjectFrameHeader(const std::string& alias, int length);

// Inside the video-file effect block, replace the file-path line's value with
// new_path_utf8 and delete the playback-position line. The path key is matched
// case-permissively against the known candidates (the Japanese file key plus
// File / file / path / Path); when no such line exists one is inserted after
// effect.name. The playback-position line (Japanese key or "Playback") is
// removed so the repointed clip plays from its start. A leading BOM is stripped
// and the line-ending style is preserved. If the video-file effect block is
// absent the input is returned unchanged.
std::string PatchAliasReplaceVideoFilePath(const std::string& alias,
                                           const std::string& new_path_utf8);

// Result of BuildProvisionalTextAlias: the alias text plus a name string the
// caller can pass to set_object_name for later exact-match lookup by job id.
struct ProvisionalTextAlias {
    std::string alias;        // ready for create_object_from_alias
    std::string object_name;  // e.g. "NzLTX23#<job_id>" (exact-match search key)
};

// Build a provisional text-object alias (effects: text + standard-draw, mirroring
// the aviutl2_sdk WindowClient.cpp sample) that shows a "<prefix><lead-in>..."
// label for display_text and embeds job_id both in the visible text (trailing
// "[#<job_id>]" marker) and in object_name so the finished object can be found
// again by an exact match. display_text is truncated (codepoint-aware) to a short
// lead-in; job_id is used verbatim. text_prefix selects the 4-stage label prefix
// (spec section 5-5): an EMPTY text_prefix uses the default "generating:" prefix
// (stage 2); a non-empty one (e.g. the stage-1 "reserved:" prefix) is used
// verbatim. The "[#<job_id>]" marker and object_name are emitted regardless of the
// prefix, so the double-tag re-discovery structure is preserved either way.
ProvisionalTextAlias BuildProvisionalTextAlias(const std::string& display_text,
                                               const std::string& job_id,
                                               const std::string& text_prefix = std::string());

// Read the value of effect's item from an alias, e.g. the text-effect's text
// line or a video effect's file line. Returns true and fills *out_value (raw,
// everything after the first '=') on the first match; false if the effect/item
// is not present. A leading BOM is stripped first.
bool ParseAliasItemValue(const std::string& alias, const std::string& effect,
                         const std::string& item, std::string* out_value);

}  // namespace nzltx
