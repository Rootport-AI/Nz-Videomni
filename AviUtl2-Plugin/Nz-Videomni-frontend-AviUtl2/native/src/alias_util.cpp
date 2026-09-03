// alias_util.cpp - implementation. See alias_util.h for the alias format and the
// aviutl2_sdk WindowClient.cpp cross-reference. ASCII-only source: the Japanese
// effect / item names are UTF-8 byte escapes so no /utf-8 flag is required.
#include "alias_util.h"

#include <vector>

namespace nzvideomni {

namespace {

// UTF-8 byte escapes for the AviUtl2 Japanese effect / item names and the
// provisional-label decorations. The trailing comment gives the literal string.
const char kEffectText[] =
    "\xe3\x83\x86\xe3\x82\xad\xe3\x82\xb9\xe3\x83\x88";  // "text" effect + text item
const char kEffectStdDraw[] =
    "\xe6\xa8\x99\xe6\xba\x96\xe6\x8f\x8f\xe7\x94\xbb";  // "standard draw" effect
const char kItemSize[] = "\xe3\x82\xb5\xe3\x82\xa4\xe3\x82\xba";  // "size" item
const char kItemAlign[] =
    "\xe6\x96\x87\xe5\xad\x97\xe6\x8f\x83\xe3\x81\x88";  // "text alignment" item
const char kAlignCenterMid[] =
    "\xe4\xb8\xad\xe5\xa4\xae\xe6\x8f\x83\xe3\x81\x88\x5b\xe4\xb8\xad\x5d";  // "center [mid]"
const char kEffectVideoFile[] =
    "\xe5\x8b\x95\xe7\x94\xbb\xe3\x83\x95\xe3\x82\xa1\xe3\x82\xa4\xe3\x83\xab";  // "video file" effect
const char kItemFileJp[] =
    "\xe3\x83\x95\xe3\x82\xa1\xe3\x82\xa4\xe3\x83\xab";  // "file" item (path)
const char kItemPlaybackJp[] =
    "\xe5\x86\x8d\xe7\x94\x9f\xe4\xbd\x8d\xe7\xbd\xae";  // "playback position" item
// Section 3-140: the three names a drag-and-drop-equivalent video object needs.
// No \xHH escape here is followed by a hex digit, so none of them can swallow a
// neighbouring character; "<playback range>,0" is assembled by concatenation.
const char kEffectVideoPlayJp[] =
    "\xe6\x98\xa0\xe5\x83\x8f\xe5\x86\x8d\xe7\x94\x9f";  // "video playback" effect
const char kItemHasAudioJp[] =
    "\xe9\x9f\xb3\xe5\xa3\xb0\xe4\xbb\x98\xe3\x81\x8d";  // "audio present" item
const char kPlaybackRangeJp[] =
    "\xe5\x86\x8d\xe7\x94\x9f\xe7\xaf\x84\xe5\x9b\xb2";  // "playback range" (3rd value)
// ASCII fallback prefix. The webui always passes an explicit, localized prefix
// (spec 5-5's 4-stage labels), so this default is only reached on an off-nominal
// path with no caller prefix; keep it ASCII so no tofu box can ever appear.
const char kGeneratingPrefix[] = "Generating: ";
const char kEllipsis[] = "\xe2\x80\xa6";  // horizontal ellipsis

// Provisional text placeholder tuning.
const char kProvisionalTextSize[] = "34";
const char kProvisionalNamePrefix[] = "NzVideomni#";  // object_name = prefix + job_id
constexpr size_t kProvisionalDisplayMaxCodepoints = 16;

std::string Trim(const std::string& s) {
    size_t b = 0;
    size_t e = s.size();
    while (b < e && (s[b] == ' ' || s[b] == '\t')) {
        ++b;
    }
    while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t')) {
        --e;
    }
    return s.substr(b, e - b);
}

std::string DetectEol(const std::string& s) {
    return s.find("\r\n") != std::string::npos ? std::string("\r\n")
                                               : std::string("\n");
}

// Split into line contents (trailing '\r' stripped). *trailing_newline records
// whether the input ended with a newline, so Join can round-trip it.
std::vector<std::string> SplitLines(const std::string& s, bool* trailing_newline) {
    std::vector<std::string> lines;
    if (s.empty()) {
        *trailing_newline = false;
        return lines;
    }
    std::string cur;
    for (char c : s) {
        if (c == '\n') {
            if (!cur.empty() && cur.back() == '\r') {
                cur.pop_back();
            }
            lines.push_back(cur);
            cur.clear();
        } else {
            cur += c;
        }
    }
    if (s.back() == '\n') {
        *trailing_newline = true;
    } else {
        if (!cur.empty() && cur.back() == '\r') {
            cur.pop_back();
        }
        lines.push_back(cur);
        *trailing_newline = false;
    }
    return lines;
}

std::string Join(const std::vector<std::string>& lines, const std::string& eol,
                 bool trailing_newline) {
    std::string out;
    for (size_t i = 0; i < lines.size(); ++i) {
        out += lines[i];
        if (i + 1 < lines.size()) {
            out += eol;
        }
    }
    if (trailing_newline) {
        out += eol;
    }
    return out;
}

bool IsSectionHeader(const std::string& line) {
    const std::string t = Trim(line);
    return t.size() >= 2 && t.front() == '[' && t.back() == ']';
}

// Trimmed key before the first '=' (empty if the line has no '=').
std::string KeyOf(const std::string& line) {
    const size_t eq = line.find('=');
    if (eq == std::string::npos) {
        return std::string();
    }
    return Trim(line.substr(0, eq));
}

bool IsFileKey(const std::string& key) {
    return key == kItemFileJp || key == "File" || key == "file" ||
           key == "path" || key == "Path";
}

bool IsPlaybackKey(const std::string& key) {
    return key == kItemPlaybackJp || key == "Playback";
}

// Format a double with EXACTLY three decimals, e.g. 10.0416666 -> "10.042",
// 10.0 -> "10.000", 0.5 -> "0.500". Deliberately NOT snprintf("%.3f"): that
// honours the C locale's decimal point, and the host process may have called
// setlocale, which would emit "10,042" and corrupt the comma-separated
// "playback position" value. Integer arithmetic has no such dependency.
// Rounding is half-away-from-zero, matching AviUtl2's own 3-decimal output.
std::string Fixed3(double value) {
    const bool negative = value < 0.0;
    double magnitude = negative ? -value : value;
    // Keep the scaled value far inside the 64-bit range (an out-of-range
    // double -> long long conversion is undefined). Also catches NaN, which
    // fails every comparison and so lands on 0.
    const double kMaxMagnitude = 1.0e12;
    if (!(magnitude < kMaxMagnitude)) {
        magnitude = magnitude > kMaxMagnitude ? kMaxMagnitude : 0.0;
    }
    const long long scaled = static_cast<long long>(magnitude * 1000.0 + 0.5);
    const long long whole = scaled / 1000;
    const long long frac = scaled % 1000;
    std::string out;
    if (negative) {
        out += '-';
    }
    out += std::to_string(whole);
    out += '.';
    if (frac < 100) {
        out += '0';
    }
    if (frac < 10) {
        out += '0';
    }
    out += std::to_string(frac);
    return out;
}

// First codepoints of a UTF-8 string (<= max_cp). *truncated is set when bytes
// remained. Never splits a multi-byte sequence.
std::string Utf8Head(const std::string& s, size_t max_cp, bool* truncated) {
    size_t i = 0;
    size_t cp = 0;
    while (i < s.size() && cp < max_cp) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        size_t adv = 1;
        if (c >= 0xF0) {
            adv = 4;
        } else if (c >= 0xE0) {
            adv = 3;
        } else if (c >= 0xC0) {
            adv = 2;
        }
        if (i + adv > s.size()) {
            adv = s.size() - i;
        }
        i += adv;
        ++cp;
    }
    *truncated = i < s.size();
    return s.substr(0, i);
}

}  // namespace

std::string StripUtf8Bom(const std::string& s) {
    if (s.size() >= 3 && static_cast<unsigned char>(s[0]) == 0xEF &&
        static_cast<unsigned char>(s[1]) == 0xBB &&
        static_cast<unsigned char>(s[2]) == 0xBF) {
        return s.substr(3);
    }
    return s;
}

std::string NormalizeAliasObjectFrameHeader(const std::string& alias, int length) {
    // Guard, not a code path: both production callers (BuildProvisionalPlaceholder
    // in bridge_core.cpp and CreateMediaObject in bridge.cpp) already require
    // length >= 1. A length below 1 has no representable inclusive end frame, so
    // the alias is returned untouched - and an alias with no "frame=" line is
    // created at the HOST's automatic duration, not at the requested one.
    if (length < 1) {
        return alias;
    }
    const std::string src = StripUtf8Bom(alias);
    const std::string eol = DetectEol(src);
    bool trailing = false;
    std::vector<std::string> lines = SplitLines(src, &trailing);

    int obj_idx = -1;
    for (size_t i = 0; i < lines.size(); ++i) {
        if (Trim(lines[i]) == "[Object]") {
            obj_idx = static_cast<int>(i);
            break;
        }
    }
    if (obj_idx < 0) {
        return alias;  // no top meta section: nothing to pin
    }

    // Section spans until the next header (or end).
    size_t sec_end = lines.size();
    for (size_t j = static_cast<size_t>(obj_idx) + 1; j < lines.size(); ++j) {
        if (IsSectionHeader(lines[j])) {
            sec_end = j;
            break;
        }
    }

    // "frame=a,b" is 0-based and INCLUSIVE at BOTH ends, whereas `length` counts
    // frames, so N frames are pinned as "frame=0,N-1" - writing N would produce
    // an object one frame too long. Measured on the real device 2026-09-04: a
    // 121-frame clip dropped onto the timeline serializes as "frame=0,120"
    // (data\Alias\R4_d_and_d.object); see Docs\SDK_REFERENCE.md section 16 (h).
    const std::string frame_line = "frame=0," + std::to_string(length - 1);
    int frame_idx = -1;
    for (size_t j = static_cast<size_t>(obj_idx) + 1; j < sec_end; ++j) {
        if (KeyOf(lines[j]) == "frame") {
            frame_idx = static_cast<int>(j);
            break;
        }
    }
    if (frame_idx >= 0) {
        lines[static_cast<size_t>(frame_idx)] = frame_line;
    } else {
        lines.insert(lines.begin() + obj_idx + 1, frame_line);
    }
    return Join(lines, eol, trailing);
}

std::string PatchAliasReplaceVideoFilePath(const std::string& alias,
                                           const std::string& new_path_utf8) {
    const std::string src = StripUtf8Bom(alias);
    const std::string eol = DetectEol(src);
    bool trailing = false;
    std::vector<std::string> lines = SplitLines(src, &trailing);

    // Locate the video-file effect section [sec_start, sec_end).
    int sec_start = -1;
    size_t sec_end = lines.size();
    size_t i = 0;
    while (i < lines.size()) {
        if (!IsSectionHeader(lines[i])) {
            ++i;
            continue;
        }
        size_t e = lines.size();
        for (size_t k = i + 1; k < lines.size(); ++k) {
            if (IsSectionHeader(lines[k])) {
                e = k;
                break;
            }
        }
        bool match = false;
        for (size_t k = i + 1; k < e; ++k) {
            const size_t eq = lines[k].find('=');
            if (eq == std::string::npos) {
                continue;
            }
            if (Trim(lines[k].substr(0, eq)) == "effect.name" &&
                Trim(lines[k].substr(eq + 1)) == kEffectVideoFile) {
                match = true;
                break;
            }
        }
        if (match) {
            sec_start = static_cast<int>(i);
            sec_end = e;
            break;
        }
        i = e;
    }
    if (sec_start < 0) {
        return alias;  // no video-file effect: nothing to repoint
    }

    const size_t body_begin = static_cast<size_t>(sec_start) + 1;
    std::vector<std::string> head(lines.begin(),
                                  lines.begin() + static_cast<long>(body_begin));
    std::vector<std::string> body(lines.begin() + static_cast<long>(body_begin),
                                  lines.begin() + static_cast<long>(sec_end));
    std::vector<std::string> tail(lines.begin() + static_cast<long>(sec_end),
                                  lines.end());

    std::vector<std::string> new_body;
    bool replaced = false;
    for (const std::string& line : body) {
        const std::string key = KeyOf(line);
        if (IsPlaybackKey(key)) {
            continue;  // drop the playback-position line
        }
        if (!replaced && IsFileKey(key)) {
            const size_t eq = line.find('=');
            new_body.push_back(line.substr(0, eq + 1) + new_path_utf8);
            replaced = true;
            continue;
        }
        new_body.push_back(line);
    }
    if (!replaced) {
        int at = -1;
        for (size_t j = 0; j < new_body.size(); ++j) {
            if (KeyOf(new_body[j]) == "effect.name") {
                at = static_cast<int>(j);
                break;
            }
        }
        const size_t ins_at = at >= 0 ? static_cast<size_t>(at) + 1 : 0;
        new_body.insert(new_body.begin() + static_cast<long>(ins_at),
                        std::string(kItemFileJp) + "=" + new_path_utf8);
    }

    std::vector<std::string> out;
    out.reserve(head.size() + new_body.size() + tail.size());
    out.insert(out.end(), head.begin(), head.end());
    out.insert(out.end(), new_body.begin(), new_body.end());
    out.insert(out.end(), tail.begin(), tail.end());
    return Join(out, eol, trailing);
}

ProvisionalTextAlias BuildProvisionalTextAlias(const std::string& display_text,
                                               const std::string& job_id,
                                               const std::string& text_prefix) {
    bool truncated = false;
    const std::string head =
        Utf8Head(display_text, kProvisionalDisplayMaxCodepoints, &truncated);

    // Visible label: <prefix> + lead-in + ellipsis, with the job id embedded (in
    // [# ...]) so the finished object is exact-match findable. text_prefix lets the
    // caller pick the 4-stage label prefix (spec section 5-5): the default (empty
    // text_prefix) is the "generating:" prefix used for stage 2; the reservation
    // insert path passes the stage-1 "reserved:" prefix instead. The trailing
    // "[#<job_id>]" marker and the object_name below are ALWAYS emitted regardless
    // of the prefix, keeping the double-tag re-discovery structure intact.
    const std::string prefix = text_prefix.empty() ? std::string(kGeneratingPrefix) : text_prefix;
    std::string text = prefix + head + kEllipsis;
    text += " [#" + job_id + "]";

    const std::string eol = "\r\n";
    std::string alias;
    alias += "[Object]";
    alias += eol;
    alias += "[Object.0]";
    alias += eol;
    alias += "effect.name=";
    alias += kEffectText;
    alias += eol;
    alias += kItemSize;
    alias += "=";
    alias += kProvisionalTextSize;
    alias += eol;
    alias += kItemAlign;  // center the provisional label
    alias += "=";
    alias += kAlignCenterMid;
    alias += eol;
    alias += kEffectText;  // text-content item key ("text")
    alias += "=";
    alias += text;
    alias += eol;
    alias += "[Object.1]";
    alias += eol;
    alias += "effect.name=";
    alias += kEffectStdDraw;
    alias += eol;

    ProvisionalTextAlias out;
    out.alias = alias;
    out.object_name = std::string(kProvisionalNamePrefix) + job_id;
    return out;
}

std::string BuildMediaObjectAlias(const std::string& file_path_utf8,
                                  double total_time_sec, bool has_audio) {
    // Guards (see the header): every one of these means "the caller must use the
    // legacy create_object_from_media_file path instead".
    if (file_path_utf8.empty()) {
        return std::string();
    }
    // The format is line-based with no escape mechanism, so a CR or LF inside
    // the path would silently split the alias into bogus lines. '=', '[' and ']'
    // are harmless: everything after the first '=' is the value, verbatim.
    if (file_path_utf8.find('\r') != std::string::npos ||
        file_path_utf8.find('\n') != std::string::npos) {
        return std::string();
    }
    // Written as !(x > 0) rather than (x <= 0) so a NaN duration is rejected too.
    if (!(total_time_sec > 0.0)) {
        return std::string();
    }

    const std::string eol = "\r\n";
    std::string alias;
    // No frame= line here: NormalizeAliasObjectFrameHeader pins the length by
    // writing "frame=0,<length-1>" (the header's end frame is inclusive).
    alias += "[Object]";
    alias += eol;
    alias += "[Object.0]";
    alias += eol;
    alias += "effect.name=";
    alias += kEffectVideoFile;
    alias += eol;
    // "playback position" = start,source-duration,<playback range>,0. The comma
    // separator is why Fixed3 (not the locale-aware snprintf) formats the value.
    alias += kItemPlaybackJp;
    alias += "=0.000,";
    alias += Fixed3(total_time_sec);
    alias += ",";
    alias += kPlaybackRangeJp;
    alias += ",";
    alias += "0";
    alias += eol;
    alias += kItemFileJp;
    alias += "=";
    alias += file_path_utf8;
    alias += eol;
    alias += kItemHasAudioJp;  // the flag create_object_from_media_file never sets
    alias += "=";
    alias += has_audio ? "1" : "0";
    alias += eol;
    alias += "[Object.1]";
    alias += eol;
    alias += "effect.name=";
    alias += kEffectVideoPlayJp;
    alias += eol;
    return alias;
}

bool ParseAliasItemValue(const std::string& alias, const std::string& effect,
                         const std::string& item, std::string* out_value) {
    const std::string src = StripUtf8Bom(alias);
    bool trailing = false;
    const std::vector<std::string> lines = SplitLines(src, &trailing);

    std::string cur_effect;
    for (const std::string& line : lines) {
        if (IsSectionHeader(line)) {
            cur_effect.clear();
            continue;
        }
        const size_t eq = line.find('=');
        if (eq == std::string::npos) {
            continue;
        }
        const std::string key = Trim(line.substr(0, eq));
        if (key == "effect.name") {
            cur_effect = Trim(line.substr(eq + 1));
            continue;
        }
        if (cur_effect == effect && key == item) {
            *out_value = line.substr(eq + 1);
            return true;
        }
    }
    return false;
}

}  // namespace nzvideomni
