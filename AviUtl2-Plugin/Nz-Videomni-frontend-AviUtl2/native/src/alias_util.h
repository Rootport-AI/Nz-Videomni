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
// the generated video - by writing the INCLUSIVE frame range the header wants
// (NormalizeAliasObjectFrameHeader), (b) repoint a video
// object at the downloaded mp4 (PatchAliasReplaceVideoFilePath), (c) drop a
// provisional "generating..." text placeholder (BuildProvisionalTextAlias), and
// (d) place a finished media file as a VIDEO OBJECT (BuildMediaObjectAlias) -
// the drag-and-drop-equivalent alias that carries the "audio present" flag.
//
// The Japanese effect/item names AviUtl2 uses (from aviutl2_sdk WindowClient.cpp
// and the object property sheet) are embedded below as UTF-8 byte escapes so
// this translation unit stays strictly ASCII - the build deliberately compiles
// without /utf-8 (system ANSI == Shift-JIS on the target host), and \xHH escapes
// are inserted as literal bytes regardless of code page. Unit-tested with
// doctest (native/tests/test_alias_util.cpp). All comments are ASCII/English.
#pragma once

#include <string>
#include <vector>

#include "track_postprocess.h"  // TrackKeyframe (section 3-54 write-back)

namespace nzvideomni {

// Remove a leading UTF-8 BOM (EF BB BF) if present; otherwise return s as-is.
// Call before parsing an alias whose origin might have added a BOM.
std::string StripUtf8Bom(const std::string& s);

// Rewrite the top "[Object]" section's "frame=" line to "frame=0,<length-1>",
// inserting it right after the "[Object]" header when absent. This is the MOST
// IMPORTANT helper: create_object_from_alias lets the alias' own frame range
// override the requested length, so pinning the frame range is what forces a
// created object to the intended duration.
//
// The header's "frame=a,b" is 0-based and INCLUSIVE at both ends, while `length`
// is a FRAME COUNT, hence the -1: N frames are written as "frame=0,N-1". (Real
// device, 2026-09-04: a 121-frame clip dropped onto the timeline serializes as
// "frame=0,120"; see Docs\SDK_REFERENCE.md section 16 (h). Writing "frame=0,N"
// - what this helper did before that measurement - made every created object one
// frame too long.) A length below 1 has no representable inclusive end, so the
// alias is returned unchanged and the host picks the duration itself.
//
// A leading BOM is stripped first; the input's line-ending style (CRLF vs LF) is
// preserved. If the alias has no "[Object]" section the input is returned
// unchanged.
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
    std::string object_name;  // e.g. "NzVideomni#<job_id>" (exact-match search key)
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

// Build the alias for a finished media file placed as a VIDEO OBJECT, i.e. the
// text AviUtl2 itself serializes for a drag-and-dropped clip (section 3-140).
// Two items make this different from create_object_from_media_file, which never
// writes either of them:
//   * "audio present" = has_audio ? 1 : 0 - the flag that decides the ribbon's
//     two-tone (video+audio) look AND whether the object's context menu offers
//     "separate audio". This is THE reason this builder exists.
//   * "playback position" = 0.000,<total_time_sec>,<playback range>,0 - the
//     source duration, so a trimmed ribbon reports a real source span.
// Only the items that vary per material are written; "playback speed", "track",
// "loop playback" and "YUV" are deliberately omitted and left to the host's
// defaults (the aviutl2_sdk sample omits every default likewise). No "frame="
// line is written - pass the result through NormalizeAliasObjectFrameHeader to
// pin the length (it writes the inclusive "frame=0,<length-1>"), exactly like
// BuildProvisionalPlaceholder does.
//
// Returns an EMPTY string when the alias cannot be built safely - an empty
// path, a path containing CR or LF (it would break the line-based format; there
// is no escape mechanism), or total_time_sec <= 0 (a still image or an
// unreadable file). The caller treats an empty result as "fall back to the
// legacy create_object_from_media_file path".
std::string BuildMediaObjectAlias(const std::string& file_path_utf8,
                                  double total_time_sec, bool has_audio);

// Read the value of effect's item from an alias, e.g. the text-effect's text
// line or a video effect's file line. Returns true and fills *out_value (raw,
// everything after the first '=') on the first match; false if the effect/item
// is not present. A leading BOM is stripped first.
bool ParseAliasItemValue(const std::string& alias, const std::string& effect,
                         const std::string& item, std::string* out_value);

// ---------------------------------------------------------------------------
// Object tracking (section 3-54): the partial-filter box <-> alias conversions.
// Docs\OBJECT_TRACKING_DESIGN.md sections 5.6 and 5.7 are the canonical text.
// ---------------------------------------------------------------------------

// AviUtl2's own four numbers for a partial filter's box, in the units the alias
// stores them in:
//   x, y   - offset of the box's CENTRE from the SCREEN's centre, in pixels,
//            with Y growing DOWNWARDS (AviUtl2's coordinate convention)
//   size   - the box's longer side, in pixels
//   aspect - signed percentage: negative squashes the height (a landscape box),
//            positive squashes the width (a portrait box), 0 is a square
// The tracker instead speaks in top-left-origin (x, y, w, h) rectangles, so the
// two functions below are the only place those two conventions meet.
struct PartialFilterValues {
    double x = 0.0;
    double y = 0.0;
    double size = 0.0;
    double aspect = 0.0;
};

// Rectangle (top-left origin, pixels) -> AviUtl2's four values, against a frame
// of scene_w x scene_h:
//   x      = (rx + rw / 2) - scene_w / 2
//   y      = (ry + rh / 2) - scene_h / 2
//   size   = max(rw, rh)
//   aspect = rw >= rh ? -100 * (1 - rh / rw) : +100 * (1 - rw / rh)
// aspect is CLAMPED to +/-99.99: at +/-100 the short side would be zero, which
// is not a box AviUtl2 can show. A degenerate input (rw <= 0 or rh <= 0) is the
// caller's job to reject - it never reaches here from the tracking worker, whose
// seed box comes from PartialFilterToRect and whose per-frame boxes come from a
// tracker that only emits positive extents. The guard below exists solely so a
// stray zero yields 0 rather than a NaN that would poison the written alias.
PartialFilterValues RectToPartialFilter(double rx, double ry, double rw, double rh,
                                        int scene_w, int scene_h);

// The exact inverse, for reading the user's seed box out of the alias. Returns
// false (leaving the outputs untouched) when v.size <= 0, i.e. when there is no
// box to speak of. Any of the four out-pointers may be null.
//   aspect < 0 -> rw = size,                 rh = size * (1 + aspect / 100)
//   aspect > 0 -> rh = size,                 rw = size * (1 - aspect / 100)
//   aspect = 0 -> rw = rh = size
//   rx = v.x + scene_w / 2 - rw / 2, ry = v.y + scene_h / 2 - rh / 2
// Round-trips with RectToPartialFilter exactly for any rectangle whose aspect
// lands inside the +/-99.99 clamp.
bool PartialFilterToRect(const PartialFilterValues& v, int scene_w, int scene_h,
                         double* rx, double* ry, double* rw, double* rh);

// Read the four values out of the first "partial filter" effect block of an
// alias. Each value line may already carry keyframes, in which case the value is
// a comma-separated list ("100,250,<interpolation>,0"); THE FIRST NUMERIC TOKEN
// is taken, because that is the value at the object's first frame - the frame
// whose box seeds the tracker. The Japanese "size" line MUST be there and MUST
// parse; X, Y and the Japanese "aspect ratio" line fall back to their default
// of 0 when the line is absent, and fail when the line is there but its first
// token is not a finite number. On failure *out is untouched. A leading BOM is
// stripped and both CRLF and LF inputs are accepted.
//
// REALDEVICE-VERIFY: confirmed by capture (2026-09-11). AviUtl2 writes EVERY
// item out, defaults included, so in practice all four lines are always there
// and the lenient default is never reached. It is kept anyway: reading a
// missing line as its default costs nothing, while refusing to track the box
// over it would be the worse failure. The same capture confirms the "first
// token" rule - a keyframed line reads "<v0>,...,<vN-1>,<move method>,0", so
// the head is still the value at the object's first frame (the whole format is
// written out above PatchAliasPartialFilterKeyframes).
//
// Parsing is locale-independent (std::from_chars), like ParsePlaybackRange in
// bridge_core: the host process may have called setlocale, and a comma decimal
// point would silently mis-read every value.
bool ParsePartialFilterValues(const std::string& alias, PartialFilterValues* out);

// The effect name of an alias' FIRST "[Object.N]" section, trimmed; an empty
// string when the alias has no such section or the section has no effect.name.
// This is how a selection is recognised as a partial filter (design section
// 5.1): the webui's classifySelectionKind compares the returned name against
// the Japanese "partial filter" string. A leading BOM is stripped, CRLF and LF
// are both accepted, and a "[Object]" meta section ahead of the effects (the
// normal layout) is skipped rather than mistaken for effect 0.
std::string FirstEffectName(const std::string& alias);

// Rewrite a partial filter's alias so its box follows `kfs`: the four value
// lines (X / Y / the Japanese "size" and "aspect ratio") become keyframed
// lists, and the "[Object]" section's "frame=" line becomes the matching list
// of 0-based inclusive keyframe boundaries. `length` is the object's frame
// count, so the last boundary is length - 1. Everything else in the alias -
// effects the user added, the mask kind, item order, line endings, a BOM - is
// preserved byte for byte (design section 5.7). Returns an EMPTY string when
// the alias cannot be rewritten safely; the caller then abandons the write-back
// with the timeline untouched.
//
// THE FORMAT comes from two .object files the owner saved on the real device
// (AviUtl2 v2.0.54, 2026-09-11): one partial filter dropped in and left alone,
// one carrying two midpoints whose position and size differ per interval.
// Side by side, with the Japanese item names spelled out in <angle brackets>:
//
//   [Object]                       [Object]
//   frame=23,33                    frame=24,54,80,119
//   [Object.0]                     [Object.0]
//   effect.name=<partial filter>   effect.name=<partial filter>
//   X=0                            X=76,89,89,89,<linear move>,0
//   Y=0                            Y=-169,-153,-153,-153,<linear move>,0
//   Group=1                        Group=1
//   <rotation>=0.00                <rotation>=0.00
//   <size>=100                     <size>=77,231,207,207,<linear move>,0
//   <aspect>=0.00                  <aspect>=0.00
//   <blur>=0                       <blur>=0
//   <mask kind>=<circle>           <mask kind>=<circle>
//   <match scene length>=0         <match scene length>=0
//   <invert mask>=0                <invert mask>=0
//
// Four rules fall straight out of that pair:
//   1. "frame=" lists the BOUNDARY frames, not the intervals: the first is the
//      start, the last is the end, both inclusive. Two midpoints -> four
//      boundaries.
//   2. A keyframed item line is "<v0>,...,<vN-1>,<move method>,0" - exactly as
//      many values as there are boundaries, then the move method's NAME, then a
//      trailing 0. An item the user never keyframed stays a SINGLE value even
//      when the object has four boundaries (the <aspect> column above), so the
//      single form always reads as "constant for the whole object".
//   3. Every item is written out, defaults included. X / Y / <size> are whole
//      numbers; <rotation> and <aspect> carry two decimals.
//   4. The capture holds ABSOLUTE timeline numbers because that is what a save
//      produces. This function writes 0-BASED RELATIVE ones instead
//      ("frame=0,...,<length-1>"), which is what create_object_from_alias
//      wants - the same convention NormalizeAliasObjectFrameHeader relies on.
// The trailing 0 of a keyframed line has no documented meaning; it is written
// verbatim, the way the capture has it.
//
// WHAT IS REWRITTEN, and nothing else:
//   * the "[Object]" section's "frame=" line (inserted after the header when
//     absent, like NormalizeAliasObjectFrameHeader does);
//   * the X / Y / <size> / <aspect> lines of the FIRST partial-filter block. A
//     missing one - rule 3 says that never happens, this is the safe side - is
//     appended at the END of that block, just before the next section header.
// Every other line, the BOM and the CRLF/LF style survive byte for byte, with
// ONE deliberate exception:
//   * any OTHER item line that still carries a keyframe list is COLLAPSED to
//     its first value ("<rotation>=1,2,3,4,<linear move>,0" -> "<rotation>=1"),
//     across the partial-filter block and every block after it. The boundary
//     count is changing, so a list sized for the OLD boundaries would no longer
//     match it; keeping the first value freezes that item at the value it had
//     on the object's first frame. The collapse is decided by SHAPE - one or
//     more leading numbers, then a non-numeric token, then a number - because
//     that is the only signal the format offers: an item whose value happens to
//     have that shape without being keyframed would be collapsed too.
//
// BOUNDARIES are built from `kfs` (ascending, 0-based, object-relative):
//   * two or more keyframes -> "frame=0,<k1>,...,<length-1>", and the four
//     lines take the keyframed form. A last keyframe EARLIER than length - 1 (a
//     stopped or truncated run) gets one extra boundary at length - 1 repeating
//     the last value, so the box holds its final position to the object's end.
//   * exactly one keyframe -> "frame=0,<length-1>" and four SINGLE values: a
//     constant box, written the way the defaults capture writes it.
//
// Returns an EMPTY string - the caller's "do not touch the timeline" - when
// `kfs` is empty, `length` < 1, scene_w or scene_h is <= 0, the alias has no
// "[Object]" section, or it has no partial-filter effect block.
std::string PatchAliasPartialFilterKeyframes(const std::string& alias,
                                             const std::vector<TrackKeyframe>& kfs,
                                             int scene_w, int scene_h, int length);

}  // namespace nzvideomni
