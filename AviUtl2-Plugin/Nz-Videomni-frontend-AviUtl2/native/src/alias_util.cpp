// alias_util.cpp - implementation. See alias_util.h for the alias format and the
// aviutl2_sdk WindowClient.cpp cross-reference. ASCII-only source: the Japanese
// effect / item names are UTF-8 byte escapes so no /utf-8 flag is required.
#include "alias_util.h"

#include <charconv>  // std::from_chars (locale-independent number parsing)
#include <cmath>
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
// Section 3-54 (object tracking): the partial filter's effect name and the one
// item name it does not share with the text effect. Its X / Y items are ASCII
// and its size item is the SAME "size" string as kItemSize above, reused here.
const char kEffectPartialFilterJp[] =
    "\xe9\x83\xa8\xe5\x88\x86\xe3\x83\x95\xe3\x82\xa3\xe3\x83\xab\xe3\x82\xbf";  // "partial filter"
const char kItemAspectJp[] =
    "\xe7\xb8\xa6\xe6\xa8\xaa\xe6\xaf\x94";  // "aspect ratio" item
// The move method written into every keyframed line (the capture's own choice;
// see PatchAliasPartialFilterKeyframes in alias_util.h).
const char kMoveLinearJp[] =
    "\xe7\x9b\xb4\xe7\xb7\x9a\xe7\xa7\xbb\xe5\x8b\x95";  // "linear move"
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

// True for "[Object.<digits>]" - an EFFECT section - and false for the
// "[Object]" meta section that normally precedes them. Used by FirstEffectName
// and ParsePartialFilterValues to walk effect blocks without mistaking the meta
// section for effect 0.
bool IsEffectSectionHeader(const std::string& line) {
    const std::string t = Trim(line);
    const std::string kPrefix = "[Object.";
    if (t.size() <= kPrefix.size() || t.back() != ']') {
        return false;
    }
    if (t.compare(0, kPrefix.size(), kPrefix) != 0) {
        return false;
    }
    const size_t digits_end = t.size() - 1;  // index of ']'
    if (digits_end <= kPrefix.size()) {
        return false;  // "[Object.]" - no index
    }
    for (size_t i = kPrefix.size(); i < digits_end; ++i) {
        if (t[i] < '0' || t[i] > '9') {
            return false;
        }
    }
    return true;
}

// Read the FIRST comma-separated token of an alias value line as a double,
// locale-independently. Keyframed values look like "100,250,<move>,0", so the
// first token is the value at the object's first frame; a value with no
// keyframes is just the single number. Surrounding spaces/tabs (and a stray CR)
// are stripped, but trailing junk inside the token is rejected, so a value whose
// shape we do not fully understand can never be silently truncated into a
// plausible-looking number. Mirrors ParseWholeDoubleField in bridge_core.cpp -
// duplicated rather than shared because that one lives in its own translation
// unit's anonymous namespace, the same way Trim is duplicated here.
bool ParseFirstValueToken(const std::string& raw, double* out) {
    const size_t comma = raw.find(',');
    const std::string field =
        comma == std::string::npos ? raw : raw.substr(0, comma);
    size_t b = 0;
    size_t e = field.size();
    while (b < e && (field[b] == ' ' || field[b] == '\t')) {
        ++b;
    }
    while (e > b && (field[e - 1] == ' ' || field[e - 1] == '\t' ||
                     field[e - 1] == '\r')) {
        --e;
    }
    if (b >= e) {
        return false;
    }
    double value = 0.0;
    const char* first = field.data() + b;
    const char* last = field.data() + e;
    const std::from_chars_result r = std::from_chars(first, last, value);
    if (r.ec != std::errc() || r.ptr != last) {
        return false;
    }
    if (!std::isfinite(value)) {
        return false;
    }
    *out = value;
    return true;
}

// The aspect percentage is clamped just short of +/-100, where the short side
// would collapse to zero (see RectToPartialFilter's contract).
constexpr double kAspectLimit = 99.99;

// Format a double with EXACTLY `digits` decimals (0 to 3), e.g. 10.0416666 ->
// "10.042" / "10.04" / "10". Deliberately NOT snprintf("%.*f"): that honours the
// C locale's decimal point, and the host process may have called setlocale,
// which would emit "10,042" and corrupt every comma-separated value this file
// writes. Integer arithmetic has no such dependency. Rounding is
// half-away-from-zero, matching AviUtl2's own output; a value that rounds to
// zero loses its sign, so a box a hair left of centre is "0", never "-0".
std::string FixedDecimals(double value, int digits) {
    long long scale = 1;
    for (int i = 0; i < digits; ++i) {
        scale *= 10;
    }
    const bool negative = value < 0.0;
    double magnitude = negative ? -value : value;
    // Keep the scaled value far inside the 64-bit range (an out-of-range
    // double -> long long conversion is undefined). Also catches NaN, which
    // fails every comparison and so lands on 0.
    const double kMaxMagnitude = 1.0e12;
    if (!(magnitude < kMaxMagnitude)) {
        magnitude = magnitude > kMaxMagnitude ? kMaxMagnitude : 0.0;
    }
    const long long scaled =
        static_cast<long long>(magnitude * static_cast<double>(scale) + 0.5);
    const long long whole = scaled / scale;
    const long long frac = scaled % scale;
    std::string out;
    if (negative && scaled != 0) {
        out += '-';
    }
    out += std::to_string(whole);
    if (digits > 0) {
        out += '.';
        std::string f = std::to_string(frac);
        while (static_cast<int>(f.size()) < digits) {
            f.insert(f.begin(), '0');
        }
        out += f;
    }
    return out;
}

// The three-decimal form the "playback position" item wants.
std::string Fixed3(double value) { return FixedDecimals(value, 3); }

// Split a value on commas WITHOUT trimming - the callers trim the pieces they
// actually inspect, and a collapsed value is handed back with its original
// spelling intact.
std::vector<std::string> SplitCommas(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : s) {
        if (c == ',') {
            out.push_back(cur);
            cur.clear();
        } else {
            cur += c;
        }
    }
    out.push_back(cur);
    return out;
}

// True when the whole token (spaces, tabs and a stray CR aside) is a finite
// number, read locale-independently like ParseFirstValueToken above.
bool IsNumericToken(const std::string& tok) {
    size_t b = 0;
    size_t e = tok.size();
    while (b < e && (tok[b] == ' ' || tok[b] == '\t')) {
        ++b;
    }
    while (e > b && (tok[e - 1] == ' ' || tok[e - 1] == '\t' || tok[e - 1] == '\r')) {
        --e;
    }
    if (b >= e) {
        return false;
    }
    double value = 0.0;
    const char* first = tok.data() + b;
    const char* last = tok.data() + e;
    const std::from_chars_result r = std::from_chars(first, last, value);
    return r.ec == std::errc() && r.ptr == last && std::isfinite(value);
}

// Recognise a keyframed item value - "<v0>,...,<vN-1>,<move method>,<0>" - and
// hand back its FIRST value. Shape is the only signal the alias format offers:
// one or more leading numbers, then a non-numeric token (the move method's
// name), then a number. See PatchAliasPartialFilterKeyframes in alias_util.h
// for why these lines have to be collapsed and what that costs.
bool CollapseKeyframedValue(const std::string& raw, std::string* out) {
    const std::vector<std::string> toks = SplitCommas(raw);
    if (toks.size() < 3) {
        return false;  // too short to hold values + method + trailing number
    }
    const size_t method = toks.size() - 2;
    if (!IsNumericToken(toks.back()) || IsNumericToken(toks[method]) ||
        Trim(toks[method]).empty()) {
        return false;
    }
    for (size_t i = 0; i < method; ++i) {
        if (!IsNumericToken(toks[i])) {
            return false;
        }
    }
    *out = toks[0];
    return true;
}

// Index one past the section that starts at `header_idx`, i.e. the next section
// header or the end of the document.
size_t SectionEnd(const std::vector<std::string>& lines, size_t header_idx) {
    for (size_t j = header_idx + 1; j < lines.size(); ++j) {
        if (IsSectionHeader(lines[j])) {
            return j;
        }
    }
    return lines.size();
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

// ---------------------------------------------------------------------------
// Object tracking (section 3-54). Docs\OBJECT_TRACKING_DESIGN.md 5.6 / 5.7.
// ---------------------------------------------------------------------------

PartialFilterValues RectToPartialFilter(double rx, double ry, double rw, double rh,
                                        int scene_w, int scene_h) {
    PartialFilterValues v;
    v.x = (rx + rw / 2.0) - static_cast<double>(scene_w) / 2.0;
    v.y = (ry + rh / 2.0) - static_cast<double>(scene_h) / 2.0;
    v.size = rw >= rh ? rw : rh;
    if (rw > 0.0 && rh > 0.0) {
        // A landscape box keeps its width and squashes its height, so the
        // percentage is reported negative; a portrait box is the mirror image.
        v.aspect = rw >= rh ? -100.0 * (1.0 - rh / rw) : 100.0 * (1.0 - rw / rh);
        if (v.aspect > kAspectLimit) {
            v.aspect = kAspectLimit;
        } else if (v.aspect < -kAspectLimit) {
            v.aspect = -kAspectLimit;
        }
    } else {
        // Degenerate input: the caller rejects these (see the header). Emitting
        // 0 rather than dividing by zero keeps a NaN out of the written alias.
        v.aspect = 0.0;
    }
    return v;
}

bool PartialFilterToRect(const PartialFilterValues& v, int scene_w, int scene_h,
                         double* rx, double* ry, double* rw, double* rh) {
    if (!(v.size > 0.0)) {  // also catches NaN
        return false;
    }
    double w = v.size;
    double h = v.size;
    if (v.aspect < 0.0) {
        h = v.size * (1.0 + v.aspect / 100.0);
    } else if (v.aspect > 0.0) {
        w = v.size * (1.0 - v.aspect / 100.0);
    }
    if (rw != nullptr) {
        *rw = w;
    }
    if (rh != nullptr) {
        *rh = h;
    }
    if (rx != nullptr) {
        *rx = v.x + static_cast<double>(scene_w) / 2.0 - w / 2.0;
    }
    if (ry != nullptr) {
        *ry = v.y + static_cast<double>(scene_h) / 2.0 - h / 2.0;
    }
    return true;
}

bool ParsePartialFilterValues(const std::string& alias, PartialFilterValues* out) {
    const std::string src = StripUtf8Bom(alias);
    bool trailing = false;
    const std::vector<std::string> lines = SplitLines(src, &trailing);

    bool in_partial_filter = false;
    bool seen_partial_filter = false;
    double x = 0.0;
    double y = 0.0;
    double size = 0.0;
    double aspect = 0.0;
    // "seen" is "the line was there at all"; "have" is "and its first token
    // parsed". A line that is present but unreadable is a malformed alias and
    // fails; a line that is simply absent falls back to its default (see the
    // header).
    bool seen_x = false;
    bool seen_y = false;
    bool seen_aspect = false;
    bool have_x = false;
    bool have_y = false;
    bool have_size = false;
    bool have_aspect = false;

    for (const std::string& line : lines) {
        if (IsSectionHeader(line)) {
            // The first partial-filter block is the one that owns the box; stop
            // at its end rather than letting a later block overwrite the values.
            if (seen_partial_filter) {
                break;
            }
            in_partial_filter = false;
            continue;
        }
        const size_t eq = line.find('=');
        if (eq == std::string::npos) {
            continue;
        }
        const std::string key = Trim(line.substr(0, eq));
        const std::string raw = line.substr(eq + 1);
        if (key == "effect.name") {
            in_partial_filter = Trim(raw) == kEffectPartialFilterJp;
            if (in_partial_filter) {
                seen_partial_filter = true;
            }
            continue;
        }
        if (!in_partial_filter) {
            continue;
        }
        if (key == "X") {
            seen_x = true;
            have_x = ParseFirstValueToken(raw, &x);
        } else if (key == "Y") {
            seen_y = true;
            have_y = ParseFirstValueToken(raw, &y);
        } else if (key == kItemSize) {
            have_size = ParseFirstValueToken(raw, &size);
        } else if (key == kItemAspectJp) {
            seen_aspect = true;
            have_aspect = ParseFirstValueToken(raw, &aspect);
        }
    }

    // X, Y and the aspect ratio all default to 0. The 2026-09-11 capture shows
    // AviUtl2 writing every item out, defaults included, so all four lines are
    // in practice always present - but treating an absent one as "not a partial
    // filter" would refuse to track a perfectly good box, so an absent line
    // takes its default and only an unreadable one fails. The size has no
    // useful default (0 is no box at all), so its line stays mandatory.
    if ((seen_x && !have_x) || (seen_y && !have_y) ||
        (seen_aspect && !have_aspect) || !have_size) {
        return false;
    }
    out->x = x;
    out->y = y;
    out->size = size;
    out->aspect = aspect;
    return true;
}

std::string FirstEffectName(const std::string& alias) {
    const std::string src = StripUtf8Bom(alias);
    bool trailing = false;
    const std::vector<std::string> lines = SplitLines(src, &trailing);

    bool in_effect = false;
    for (const std::string& line : lines) {
        if (IsSectionHeader(line)) {
            if (in_effect) {
                return std::string();  // first effect block had no effect.name
            }
            in_effect = IsEffectSectionHeader(line);
            continue;
        }
        if (!in_effect) {
            continue;
        }
        const size_t eq = line.find('=');
        if (eq == std::string::npos) {
            continue;
        }
        if (Trim(line.substr(0, eq)) == "effect.name") {
            return Trim(line.substr(eq + 1));
        }
    }
    return std::string();
}

std::string PatchAliasPartialFilterKeyframes(const std::string& alias,
                                             const std::vector<TrackKeyframe>& kfs,
                                             int scene_w, int scene_h, int length) {
    // Every one of these means "there is nothing safe to write"; the caller
    // reads the empty string as "leave the timeline alone".
    if (kfs.empty() || length < 1 || scene_w <= 0 || scene_h <= 0) {
        return std::string();
    }
    const std::string src = StripUtf8Bom(alias);
    const bool had_bom = src.size() != alias.size();
    const std::string eol = DetectEol(src);
    bool trailing = false;
    const std::vector<std::string> lines = SplitLines(src, &trailing);

    // --- 1. locate the "[Object]" meta section ------------------------------
    int obj_idx = -1;
    for (size_t i = 0; i < lines.size(); ++i) {
        if (Trim(lines[i]) == "[Object]") {
            obj_idx = static_cast<int>(i);
            break;
        }
    }
    if (obj_idx < 0) {
        return std::string();  // no "frame=" to own: not an object alias
    }
    const size_t obj_end = SectionEnd(lines, static_cast<size_t>(obj_idx));

    // --- 2. locate the FIRST partial-filter effect block --------------------
    // Only "[Object.<n>]" sections are considered, so a "[Object]" meta section
    // carrying an effect.name-shaped line cannot be mistaken for effect 0 (the
    // same rule FirstEffectName follows).
    int pf_start = -1;
    size_t pf_end = lines.size();
    size_t i = 0;
    while (i < lines.size()) {
        if (!IsEffectSectionHeader(lines[i])) {
            ++i;
            continue;
        }
        const size_t e = SectionEnd(lines, i);
        for (size_t k = i + 1; k < e; ++k) {
            const size_t eq = lines[k].find('=');
            if (eq == std::string::npos) {
                continue;
            }
            if (Trim(lines[k].substr(0, eq)) == "effect.name" &&
                Trim(lines[k].substr(eq + 1)) == kEffectPartialFilterJp) {
                pf_start = static_cast<int>(i);
                pf_end = e;
                break;
            }
        }
        if (pf_start >= 0) {
            break;
        }
        i = e;
    }
    if (pf_start < 0) {
        return std::string();  // nothing to keyframe
    }

    // --- 3. boundaries and the value at each one ----------------------------
    // `kfs` arrives ascending, 0-based and inside [0, length - 1] (the worker
    // builds it that way from its own sample offsets). The two skips below are
    // guards, not code paths: a boundary outside the object or a repeat of the
    // one before it cannot be written as a frame list.
    std::vector<int> bounds;
    std::vector<PartialFilterValues> vals;
    for (const TrackKeyframe& k : kfs) {
        if (k.frame_offset < 0 || k.frame_offset > length - 1) {
            continue;
        }
        if (!bounds.empty() && k.frame_offset <= bounds.back()) {
            continue;
        }
        bounds.push_back(k.frame_offset);
        vals.push_back(RectToPartialFilter(k.x, k.y, k.w, k.h, scene_w, scene_h));
    }
    if (bounds.empty()) {
        return std::string();
    }
    const bool keyframed = bounds.size() >= 2;
    if (keyframed) {
        if (bounds.front() != 0) {
            // Cannot happen from the worker (its first sample is the seed box
            // at offset 0); a run that somehow started late holds its first box
            // from the object's start rather than leaving the head unpinned.
            const PartialFilterValues first = vals.front();  // copy: vals grows
            bounds.insert(bounds.begin(), 0);
            vals.insert(vals.begin(), first);
        }
        if (bounds.back() != length - 1) {
            // A stopped or truncated run: hold the last box to the very end.
            const PartialFilterValues last = vals.back();  // copy: vals grows
            bounds.push_back(length - 1);
            vals.push_back(last);
        }
    } else {
        // One keyframe is a constant box: single values, and the plain
        // start/end pair the "left alone" capture writes.
        bounds.assign(1, 0);
        bounds.push_back(length - 1);
    }

    // --- 4. the replacement lines -------------------------------------------
    std::string frame_val;
    for (size_t n = 0; n < bounds.size(); ++n) {
        if (n != 0) {
            frame_val += ',';
        }
        frame_val += std::to_string(bounds[n]);
    }
    std::string x_val;
    std::string y_val;
    std::string size_val;
    std::string aspect_val;
    for (size_t n = 0; n < vals.size(); ++n) {
        if (n != 0) {
            x_val += ',';
            y_val += ',';
            size_val += ',';
            aspect_val += ',';
        }
        // X / Y / size are whole numbers and the aspect carries two decimals,
        // exactly as the capture spells them.
        x_val += FixedDecimals(vals[n].x, 0);
        y_val += FixedDecimals(vals[n].y, 0);
        size_val += FixedDecimals(vals[n].size, 0);
        aspect_val += FixedDecimals(vals[n].aspect, 2);
    }
    if (keyframed) {
        const std::string tail = std::string(",") + kMoveLinearJp + ",0";
        x_val += tail;
        y_val += tail;
        size_val += tail;
        aspect_val += tail;
    }

    // --- 5. which of the four lines are already there? ----------------------
    bool have_frame = false;
    for (size_t j = static_cast<size_t>(obj_idx) + 1; j < obj_end; ++j) {
        if (KeyOf(lines[j]) == "frame") {
            have_frame = true;
            break;
        }
    }
    bool have_x = false;
    bool have_y = false;
    bool have_size = false;
    bool have_aspect = false;
    for (size_t j = static_cast<size_t>(pf_start) + 1; j < pf_end; ++j) {
        const std::string key = KeyOf(lines[j]);
        if (key == "X") {
            have_x = true;
        } else if (key == "Y") {
            have_y = true;
        } else if (key == kItemSize) {
            have_size = true;
        } else if (key == kItemAspectJp) {
            have_aspect = true;
        }
    }

    // --- 6. rebuild the document in one pass --------------------------------
    std::vector<std::string> out;
    out.reserve(lines.size() + 5);
    // The capture writes all four, so this only ever fires for an alias built
    // by something other than AviUtl2.
    const auto append_missing = [&]() {
        if (!have_x) {
            out.push_back("X=" + x_val);
        }
        if (!have_y) {
            out.push_back("Y=" + y_val);
        }
        if (!have_size) {
            out.push_back(std::string(kItemSize) + "=" + size_val);
        }
        if (!have_aspect) {
            out.push_back(std::string(kItemAspectJp) + "=" + aspect_val);
        }
    };
    for (size_t j = 0; j < lines.size(); ++j) {
        if (j == pf_end) {
            append_missing();  // end of the block: just before the next header
        }
        const std::string& line = lines[j];
        const size_t eq = line.find('=');
        if (eq == std::string::npos || IsSectionHeader(line)) {
            out.push_back(line);
        } else {
            const std::string key = Trim(line.substr(0, eq));
            const bool in_obj = j > static_cast<size_t>(obj_idx) && j < obj_end;
            const bool in_pf = j > static_cast<size_t>(pf_start) && j < pf_end;
            std::string collapsed;
            if (in_obj && key == "frame") {
                out.push_back("frame=" + frame_val);
            } else if (in_pf && key == "X") {
                out.push_back("X=" + x_val);
            } else if (in_pf && key == "Y") {
                out.push_back("Y=" + y_val);
            } else if (in_pf && key == kItemSize) {
                out.push_back(std::string(kItemSize) + "=" + size_val);
            } else if (in_pf && key == kItemAspectJp) {
                out.push_back(std::string(kItemAspectJp) + "=" + aspect_val);
            } else if (j > static_cast<size_t>(pf_start) && key != "effect.name" &&
                       CollapseKeyframedValue(line.substr(eq + 1), &collapsed)) {
                // Another item still keyframed against the OLD boundaries: it
                // is frozen at its first value (see alias_util.h).
                out.push_back(line.substr(0, eq + 1) + collapsed);
            } else {
                out.push_back(line);
            }
        }
        if (j == static_cast<size_t>(obj_idx) && !have_frame) {
            out.push_back("frame=" + frame_val);
        }
    }
    if (pf_end >= lines.size()) {
        append_missing();  // the partial filter was the last block
    }

    std::string text = Join(out, eol, trailing);
    if (had_bom) {
        text.insert(0, "\xEF\xBB\xBF");
    }
    return text;
}

}  // namespace nzvideomni
