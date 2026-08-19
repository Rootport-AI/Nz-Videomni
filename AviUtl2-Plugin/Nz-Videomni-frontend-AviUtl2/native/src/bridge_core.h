// bridge_core.h - WebView2-independent JSON-RPC dispatch logic.
//
// This module contains the pure request-parsing, validation and dispatch logic
// for the Web UI <-> native bridge (RPC contract v2). It has no dependency on
// WebView2, WinHTTP or any Win32 UI type, so it can be unit-tested in isolation
// with doctest. HTTP transport and AviUtl2 SDK calls are injected through
// std::function providers, so they can be mocked in tests.
//
// RPC contract (envelope):
//   Request (WebUI -> native):
//     {"id": <number>, "method": <string>, "params": <object>}
//   Success (native -> WebUI):
//     {"id": <number>, "ok": true,  "result": <object>}
//   Failure:
//     {"id": <number>, "ok": false, "error": {"code": <string>,
//                                              "message": <string>}}
//
// Synchronous methods (handled here, returning a response string directly):
//   ping, getEditInfo, backend.getBaseUrl, timeline.insertMedia,
//   settings.get, settings.set (contract v4: persistent connection settings,
//   see settings.h for the SettingsStore backing these two), and contract
//   v7's ui.resolveDroppedFiles (pure in-memory work, no I/O - see its own
//   section below for the params.__droppedPaths reserved-key discipline).
// Asynchronous methods (backend.request, backend.downloadVideo,
// backend.uploadFile, timeline.captureFrame, and contract v6's
// fs.listFiles / fs.probeAudioDuration) are NOT handled by HandleRequestJson: the bridge
// validates their params with the ParseXxx helpers below, performs the HTTP /
// render / file I/O work off the UI thread, and formats the response with
// MakeSuccessResponse / MakeErrorResponse. ui.pickFolder (contract v6) is also
// dispatched directly by the bridge (mirrors ui.pickFile: modal dialog on the
// UI thread), not by HandleRequestJson.
//
// Error codes: BAD_REQUEST, UNKNOWN_METHOD, NO_EDIT_HANDLE, FILE_NOT_FOUND,
// INSERT_FAILED, BACKEND_UNREACHABLE, BACKEND_TIMEOUT, DOWNLOAD_FAILED,
// CAPTURE_FAILED, CANCELLED, DIALOG_FAILED, THUMBNAIL_FAILED (contract v3:
// timeline.captureFrame, backend.uploadFile, ui.pickFile, ui.makeThumbnail).
//
// All comments are ASCII/English so this translation unit compiles cleanly
// under any default code page.
#pragma once

#include <functional>
#include <string>
#include <vector>

#include "json.hpp"

// Contract v5 timeline methods reuse these pure, SDK-free modules: the
// provisional placeholder alias builders (alias_util) and the provisional
// re-discovery / resolve / orphan logic + ScannedObject POD (provisional).
#include "alias_util.h"
#include "provisional.h"

namespace nzvideomni {

// JSON type used throughout the bridge. Backend responses (e.g. config.yaml
// driven key order) and RPC result objects must preserve insertion order, so
// this is nlohmann::ordered_json rather than the default std::map-backed
// nlohmann::json (which silently re-sorts object keys alphabetically on
// parse/dump). All bridge_core / bridge / plugin JSON handling should go
// through this alias instead of nlohmann::json directly.
using json_t = nlohmann::ordered_json;

// Plugin version string reported by the "ping" method.
inline constexpr char kPluginVersion[] = "1.0.0-rc1";

// Built-in default backend base URL (host + scheme, no trailing slash) and API
// path prefix. The base URL is now a runtime setting (contract v4: see
// settings.h / SettingsStore) - Bridge reads the current value from the
// settings store and passes it through RequestContext::base_url /
// SettingsGetProvider below. kBackendBaseUrl remains the built-in default that
// SettingsStore falls back to and is what a fresh install starts with. The
// "/api/v1" prefix is not user-configurable and stays a compile-time constant.
inline constexpr char kBackendBaseUrl[] = "http://127.0.0.1:18620";
inline constexpr char kBackendApiPrefix[] = "/api/v1";

// ---------------------------------------------------------------------------
// getEditInfo
// ---------------------------------------------------------------------------

// Snapshot of the host EDIT_INFO used to answer "getEditInfo".
struct EditInfoResult {
    bool available = false;  // false -> respond NO_EDIT_HANDLE
    int width = 0;
    int height = 0;
    int rate = 0;
    int scale = 0;
    int sample_rate = 0;
    int frame = 0;
    int layer = 0;
    int frame_max = 0;
    int layer_max = 0;
};

// Supplies a fresh EditInfoResult at dispatch time (reads the host EDIT_INFO).
using EditInfoProvider = std::function<EditInfoResult()>;

// ---------------------------------------------------------------------------
// timeline.insertMedia
// ---------------------------------------------------------------------------

// Parameters passed to the insert-media provider. When has_layer / has_frame is
// false the provider must fall back to the current edit cursor position.
struct InsertMediaParams {
    std::string file_path;  // UTF-8 absolute path
    bool has_layer = false;
    int layer = 0;
    bool has_frame = false;
    int frame = 0;
};

// Outcome of an insert attempt. The provider runs the actual SDK call inside a
// call_edit_section_param callback (on the UI/main thread).
struct InsertMediaResult {
    enum class Status { kOk, kFileNotFound, kNoEditHandle, kInsertFailed };
    Status status = Status::kInsertFailed;
    int layer = 0;  // resolved layer actually used
    int frame = 0;  // resolved frame actually used
};

using InsertMediaProvider = std::function<InsertMediaResult(const InsertMediaParams&)>;

// ---------------------------------------------------------------------------
// settings.get / settings.set (contract v4) - persistent connection settings.
//
// The validation and disk persistence live in settings.{h,cpp} (SettingsStore
// / NormalizeBaseUrl); this module only wires the two RPC methods onto that
// store through injected providers, mirroring the edit_info / insert_media
// pattern above so HandleRequestJson stays testable without real file I/O.
// ---------------------------------------------------------------------------

// Outcome of a settings.set attempt against the store.
struct SettingsSetOutcome {
    bool ok = false;
    std::string base_url;     // normalized value (valid only when ok)
    std::string err_message;  // BAD_REQUEST detail (valid only when !ok)
};

// Returns the current persisted base URL.
using SettingsGetProvider = std::function<std::string()>;

// Validates, persists and applies a candidate base URL (see
// SettingsStore::SetBaseUrl). Only called when the request supplies a
// 'baseUrl'; settings.set treats an absent/null 'baseUrl' as a no-op so future
// settings fields can be updated independently of this one.
using SettingsSetProvider = std::function<SettingsSetOutcome(const std::string&)>;

// ---------------------------------------------------------------------------
// timeline.* synchronous provider-backed methods (contract v5).
//
// These mirror the insertMedia pattern: bridge_core does the pure parsing,
// alias construction (BuildProvisionalPlaceholder) and result formatting
// (MakeSelectionResult), then calls an injected std::function that performs the
// AviUtl2 SDK work. Tests inject mock providers so no SDK is required. The
// bridge implements the real providers (see bridge.cpp handoff I6, I9-I11).
// ---------------------------------------------------------------------------

// --- timeline.getSelection --------------------------------------------------

// One currently-selected timeline object. filePath / objectName are nullable in
// the JSON: has_file_path / has_object_name == false -> the key is emitted null.
struct SelectionItem {
    int layer = 0;
    int frame_start = 0;
    int frame_end = 0;
    std::string effect_name;
    bool has_file_path = false;
    std::string file_path;    // UTF-8 (valid only when has_file_path)
    bool has_object_name = false;
    std::string object_name;  // e.g. "NzVideomni#<job>" (valid only when has_object_name)
    // Text-object body, for #8 ("append to the main prompt"). Filled only for a
    // text object (effect_name == "text"): has_text_content / text_content
    // mirror the has_file_path / file_path nullable pattern -- has_text_content
    // == false -> the JSON key is emitted null.
    bool has_text_content = false;
    std::string text_content;  // UTF-8 (valid only when has_text_content)
    // Real resolution of the underlying media file, from get_media_info.
    // 0 when unavailable (no file_path, unsupported format, or SDK lookup
    // failure) -- unlike filePath/objectName these are NOT nullable in the
    // JSON; the webui treats 0 as "unknown" itself.
    int media_width = 0;
    int media_height = 0;
    // Real media duration in seconds, from get_media_info's total_time. 0 for a
    // still image, an audio-less object, or a lookup that failed -- NOT nullable
    // in the JSON (like media_width/media_height), the webui treats 0 as
    // "unknown". Consumed by the length guard for #1/#2 (see the redesign spec
    // section 4-5).
    double media_duration_sec = 0.0;
    // --- contract v10 (source trim, section 1-6) ---------------------------
    // The window of the BACKING FILE this object plays, from AviUtl2's
    // "playback position" item. Both are SECONDS on the SOURCE time axis; the
    // unit was established on real hardware on 2026-08-01 (the item's raw value
    // is a 4-field CSV "<start>,<end>,<mode>,<flag>" whose first two fields are
    // seconds with 3 decimals, independent of the project fps). See
    // ParsePlaybackRange below and Docs/V2V_RIBBON_TRIM_WORKORDER.md section 4.
    //
    // Like media_width/media_height these are NOT nullable in the JSON: the
    // explicit has_playback_range flag says whether they were really read,
    // because 0.0 is both a perfectly normal start ("plays from the beginning")
    // and the natural "not available" default.
    double playback_start_sec = 0.0;
    double playback_end_sec = 0.0;
    bool has_playback_range = false;
    // AviUtl2's "playback speed" item, normalized to a 1.0 scale (the raw value
    // is a percentage string like "100.00"). 1.0 = neutral, which is also the
    // default when the item could not be read -- the webui only treats a
    // REPORTED non-neutral speed as a reason to skip the trim.
    double playback_speed = 1.0;
    // AviUtl2's "loop playback" item (raw "0"/"1"). True makes the object's
    // time mapping non-monotonic, so the trim is skipped.
    bool loop_play = false;
    // Number of intermediate-point ("nakama-ten") sections, from
    // get_object_section_num. >= 2 means the time mapping is piecewise and a
    // single start/duration pair cannot describe it. Defaults to 1 (the
    // single-section case) when the SDK entry point is missing or fails.
    int section_count = 1;
};

// Snapshot of the current selection / cursor plus the project rate fields, which
// must equal getEditInfo's rate / scale / sampleRate for frame<->time math.
struct SelectionSnapshot {
    bool available = false;  // false -> respond NO_EDIT_HANDLE
    bool has_range = false;
    int range_start = 0;
    int range_end = 0;
    std::vector<SelectionItem> selected;
    int cursor_frame = 0;
    int cursor_layer = 0;
    int rate = 0;
    int scale = 0;
    int sample_rate = 0;
};

using SelectionProvider = std::function<SelectionSnapshot()>;

// --- timeline.insertProvisional ---------------------------------------------

// Outcome of inserting a provisional placeholder.
struct InsertProvisionalOutcome {
    bool ok = false;  // false -> PROVISIONAL_FAILED
    int layer = 0;    // resolved position actually used
    int frame = 0;
    // True when the requested layer/frame collided with an existing object and
    // the provider fell back to layer_max+1 (redesign spec section 5-4). The
    // webui surfaces this as usedFallback so it can show the fallback note.
    bool used_fallback = false;
};

// Inserts a placeholder from a ready alias + object_name at layer/frame, sized
// to `length` project frames. bridge_core builds the alias
// (BuildProvisionalPlaceholder) and resolves the length so this provider only
// runs the SDK create_object_from_alias + set_object_name. D2: `length` is
// forwarded as the explicit create_object_from_alias length arg (and drives the
// room pre-check) so a partial free gap can never silently truncate the
// placeholder - see bridge.cpp HasRoomForLength / InsertProvisionalEditProc.
using InsertProvisionalProvider = std::function<InsertProvisionalOutcome(
    const std::string& alias, const std::string& object_name, int layer, int frame,
    int length)>;

// --- timeline.resolveProvisional / updateProvisionalText / scanProvisionals --

// Supplies a fresh scan of the timeline's objects (position + alias text) so the
// pure provisional.h logic can re-find placeholders by exact job-id match.
using ScanObjectsProvider = std::function<std::vector<ScannedObject>()>;

// Replaces the still-present placeholder for job_id with the finished video,
// pinned to length_frames project frames.
struct ReplaceObjectRequest {
    std::string job_id;           // re-find key (exact alias match)
    int layer = 0;                // found position (from the scan)
    int frame = 0;
    std::string video_file_path;  // UTF-8 absolute path of the finished mp4
    int length_frames = 0;
};
using ReplaceObjectProvider = std::function<bool(const ReplaceObjectRequest&)>;

// Updates the visible text of a still-pending placeholder.
struct UpdateObjectTextRequest {
    std::string job_id;
    int layer = 0;  // found position (from the scan)
    int frame = 0;
    std::string text;
};
using UpdateObjectTextProvider = std::function<bool(const UpdateObjectTextRequest&)>;

// --- timeline.updateProvisionalReservation (I3) -----------------------------
//
// Placement systems for a provisional object, from the right-click redesign
// spec section 5-9. insertProvisional / updateProvisionalReservation resolve
// the target layer/frame from one of these using the selected material's frame
// range (A/B) or the cursor (C). kLegacy means "no placement given; use the
// explicit layer/frame params" (the backward-compatible pre-I3 path).
enum class ProvisionalPlacement {
    kLegacy = 0,      // use the explicit layer/frame (pre-I3 behavior)
    kAfterMaterial,   // (A) directly after material: material_layer, end+1
    kSameStartFront,  // (B) same start frame, layer_max+1
    kCursor,          // (C) the right-click cursor position
};

// Inputs for the pure placement resolver. Fields not used by a given placement
// are ignored; missing/negative required fields are rejected (ok == false).
struct PlacementResolveInput {
    ProvisionalPlacement placement = ProvisionalPlacement::kLegacy;
    int material_layer = -1;
    int material_frame_start = -1;
    int material_frame_end = -1;  // inclusive (see ResolveProvisionalPlacement)
    int cursor_layer = -1;
    int cursor_frame = -1;
    int layer_max = 0;  // from EDIT_INFO, for (B) and the collision fallback
};

struct PlacementResolveResult {
    bool ok = false;
    int layer = 0;
    int frame = 0;
    std::string err;  // BAD_REQUEST detail when !ok
};

// Resolve a provisional object's target layer/frame from a placement system
// (spec section 5-9). Pure integer geometry, doctest-tested:
//   (A) kAfterMaterial:  layer=material_layer, frame=material_frame_end+1.
//       material_frame_end is treated as INCLUSIVE (the object's last covered
//       frame), matching FindObjectByJob's next=end+1 scan step. Real-device
//       confirmation is spec section 8 check #7.
//   (B) kSameStartFront: layer=layer_max+1, frame=material_frame_start.
//   (C) kCursor:         layer=cursor_layer, frame=cursor_frame.
// kLegacy is rejected (its caller uses the explicit layer/frame instead of
// calling this). Missing/negative required inputs are rejected.
PlacementResolveResult ResolveProvisionalPlacement(const PlacementResolveInput& in);

// Atomically re-places (delete old + create new) a provisional reservation in
// one edit section. old_job_id empty -> skip the delete search (create-only,
// e.g. self-heal after the user removed the placeholder). bridge_core supplies
// the ready alias / object_name / resolved target; the provider only runs the
// SDK delete_object + create_object_from_alias (with a layer_max+1 fallback).
struct UpdateReservationRequest {
    std::string old_job_id;   // "" -> no delete search
    std::string new_job_id;   // non-empty
    std::string alias;        // BuildProvisionalPlaceholder(new_job_id, ...).alias
    std::string object_name;  // "NzVideomni#<new_job_id>"
    int layer = 0;            // resolved target (placement or legacy)
    int frame = 0;
    // D2: resolved length (project frames). Forwarded as the explicit
    // create_object_from_alias length arg and used for the room pre-check so a
    // partial free gap can never silently truncate the placeholder.
    int length_frames = 0;
};

struct UpdateReservationOutcome {
    bool ok = false;             // false -> PROVISIONAL_FAILED
    bool deleted_old = false;    // an old placeholder was found and deleted
    int placed_layer = 0;        // where the new placeholder actually landed
    int placed_frame = 0;
    bool used_fallback = false;  // collided -> created at layer_max+1
};

using UpdateReservationProvider =
    std::function<UpdateReservationOutcome(const UpdateReservationRequest&)>;

// --- timeline.deleteProvisionalByJob (spec 5-10) ----------------------------
//
// Deletes the still-present provisional placeholder for a job id, as the
// cleanup tied to the user's explicit successful timeline.insertMedia (the panel
// 🎞 button). Only the FIRST match is removed (duplicates are left as harmless
// orphan text); a missing placeholder is a no-op success (idempotent). Runs the
// SDK delete_object in one edit section (one undo step) after bridge_core has
// located the placeholder via ctx.scan_objects + FindProvisionalIndex.
struct DeleteProvisionalRequest {
    std::string job_id;  // re-find key (exact alias match)
    int layer = 0;       // found position (from the scan)
    int frame = 0;
};
using DeleteProvisionalProvider = std::function<bool(const DeleteProvisionalRequest&)>;

// --- timeline.insertMediaForJob (replace-insert) ----------------------------
//
// The "place & replace" 🎞 insert tied to a job's provisional marker. When a
// still-present provisional placeholder for job_id is found (ctx.scan_objects +
// FindProvisionalIndex, EXACT job-id match across all layers - never the
// position fallback), the finished media REPLACES it in place: the provider's
// EditProc re-finds the placeholder (FindObjectByJob), deletes it, and creates
// the media at the SAME layer/frame, sized to the media's REAL length
// (get_media_info -> ProjectFramesForSeconds, exactly like InsertMediaEditProc).
// When no marker is found, bridge_core does NOT call this provider at all - it
// falls back to the plain insert_media provider (current selection / cursor,
// real length), i.e. behaviour identical to timeline.insertMedia (owner
// decision: normal-generation 🎞 must stay unchanged).
//
// Deliberately NO HasRoomForLength pre-check (unlike insertProvisional /
// updateReservation): the marker we are deleting occupies the very slot we then
// create in, so a room pre-check would false-positive against our own
// about-to-be-deleted object; the SDK's collision behaviour at the slot is
// observed on the real device (gate G3). On a null create at the marker slot
// (a genuine failure - broken file / truly-blocked slot) the provider retries
// ONCE at layer_max+1 / same frame / same explicit length (used_fallback), and
// on a second null reports ok == false (a single undo then restores the just-
// deleted marker as the user's retreat).
struct ReplaceMediaForJobRequest {
    std::string job_id;     // re-find key (exact alias match)
    int layer = 0;          // the found placeholder position (from the scan)
    int frame = 0;
    std::string file_path;  // UTF-8 absolute path of the finished media
};

struct ReplaceMediaForJobOutcome {
    bool ok = false;             // false -> INSERT_FAILED
    int layer = 0;               // where the media actually landed
    int frame = 0;
    bool used_fallback = false;  // marker-slot create collided -> retried at layer_max+1
};

using ReplaceMediaForJobProvider =
    std::function<ReplaceMediaForJobOutcome(const ReplaceMediaForJobRequest&)>;

// --- fs.probeMediaInfo (synchronous) ----------------------------------------
//
// A best-effort probe of a media file's real duration / resolution via the SDK's
// get_media_info (inside a call_edit_section_param callback). This NEVER fails
// from the caller's point of view: a missing edit handle, an unavailable
// get_media_info, an unsupported/unreadable file, or an unset provider all
// resolve to an all-zero success result ({durationSec:0,width:0,height:0}) - the
// webui treats 0 as "unknown". Only param validation (filePath) can error.
struct MediaInfoSnapshot {
    bool available = false;   // false -> emit an all-zero result
    double duration_sec = 0.0;
    int width = 0;
    int height = 0;
};

// Probes a file's media info (UTF-8 path). Returns an unavailable snapshot on
// any lookup failure (never throws / never signals an error to the caller).
using MediaInfoProvider = std::function<MediaInfoSnapshot(const std::string& file_path)>;

// Bundle of everything HandleRequestJson needs to answer the synchronous methods.
struct RequestContext {
    EditInfoProvider edit_info;
    InsertMediaProvider insert_media;
    SettingsGetProvider settings_get;
    SettingsSetProvider settings_set;
    std::string base_url = kBackendBaseUrl;  // reported by backend.getBaseUrl
                                             // (fallback when settings_get is unset)
    // Contract v5 timeline providers - appended only (existing field order and
    // the settings/insert providers above are unchanged).
    SelectionProvider get_selection;
    InsertProvisionalProvider insert_provisional;
    ScanObjectsProvider scan_objects;
    ReplaceObjectProvider replace_object;
    UpdateObjectTextProvider update_object_text;
    // I3 (appended at the end to preserve existing field order): the atomic
    // delete-old + create-new provisional re-placement for
    // timeline.updateProvisionalReservation.
    UpdateReservationProvider update_reservation;
    // I13 (appended at the end): the delete of the ✅ provisional placeholder on
    // a successful 🎞 insert (timeline.deleteProvisionalByJob, spec 5-10).
    DeleteProvisionalProvider delete_provisional;
    // Replace-insert (appended at the end): the atomic delete-placeholder +
    // create-media-at-the-marker EditProc for timeline.insertMediaForJob's
    // "found a marker" branch. The "not found" branch reuses insert_media above.
    ReplaceMediaForJobProvider replace_media_for_job;
    // fs.probeMediaInfo (appended at the end): best-effort media duration /
    // resolution probe. Unset -> fs.probeMediaInfo still succeeds with an
    // all-zero result (see MediaInfoProvider / fs.probeMediaInfo below).
    MediaInfoProvider probe_media_info;
};

// Parse a UTF-8 JSON request and return the UTF-8 JSON response for the
// synchronous methods (ping, getEditInfo, backend.getBaseUrl,
// timeline.insertMedia). Never throws.
//
// - Malformed JSON / missing method -> BAD_REQUEST (id null when unknown).
// - Unknown method -> UNKNOWN_METHOD. Note the async backend.* methods are
//   NOT recognised here; the bridge intercepts them before calling this.
// - An object with neither id nor method -> empty string (no response).
std::string HandleRequestJson(const std::string& request_json,
                              const RequestContext& ctx);

// ---------------------------------------------------------------------------
// Response envelope builders (shared with the bridge for async responses).
// ---------------------------------------------------------------------------

std::string MakeSuccessResponse(const json_t& id, json_t result);
std::string MakeErrorResponse(const json_t& id, const std::string& code,
                              const std::string& message);

// Echoable numeric id from a parsed request object (null if absent/non-number).
json_t ExtractId(const json_t& req);

// ---------------------------------------------------------------------------
// backend.request (async) - pure parsing / result formatting.
// ---------------------------------------------------------------------------

// backend.request timeout bounds (contract v4.1). An optional 'timeoutMs' in the
// params overrides the default per-request timeout; it is clamped into
// [kMinBackendTimeoutMs, kMaxBackendTimeoutMs]. A present-but-non-numeric value
// is rejected as BAD_REQUEST.
inline constexpr int kDefaultBackendTimeoutMs = 30000;   // 30 s (matches http_client)
inline constexpr int kMinBackendTimeoutMs = 1000;        // 1 s
inline constexpr int kMaxBackendTimeoutMs = 600000;      // 600 s

// ---------------------------------------------------------------------------
// Shared URL query helpers (contract v10).
//
// Both backend.request and backend.uploadFile accept an optional 'query'
// object; these two pure functions are the single implementation behind them.
// ---------------------------------------------------------------------------

// Percent-encode a JSON object of string values into a URL query component
// ("a=b&c=d"). *out is cleared first, so an empty object yields an empty
// string. Returns false and sets *err_message when `value` is not an object or
// when any of its values is not a string.
//
// IMPORTANT: *err_message carries NO method-name prefix (e.g.
// "'query' must be an object"); the caller prepends its own method name, which
// is what keeps the user-visible strings of backend.request and
// backend.uploadFile byte-identical to their historical wording.
bool BuildQueryString(const json_t& value, std::string* out,
                      std::string* err_message);

// Append an already-encoded query component to a URL. An EMPTY query returns
// `url` completely unchanged - no '?' is appended. That property is contractual:
// it is what guarantees that a request carrying no query is byte-for-byte the
// request that was sent before the 'query' field existed.
std::string AppendQueryToUrl(const std::string& url, const std::string& query);

struct BackendRequest {
    std::string http_method;  // "GET" | "POST" | "DELETE"
    std::string path;         // starts with '/', /api/v1 prefix included, e.g. "/api/v1/status"
    std::string query;        // URL-encoded "a=b&c=d", or empty
    bool has_body = false;
    std::string body;         // serialized JSON body (valid when has_body)
    int timeout_ms = kDefaultBackendTimeoutMs;  // clamped request timeout (v4.1)
};

// Validate the params of backend.request. On success returns true and fills
// *out. On failure returns false and sets *err_message (code is always
// BAD_REQUEST for parse failures). An optional numeric 'timeoutMs' is clamped
// into [kMinBackendTimeoutMs, kMaxBackendTimeoutMs]; a present non-numeric
// 'timeoutMs' is a parse failure.
bool ParseBackendRequest(const json_t& params, BackendRequest* out,
                         std::string* err_message);

// Assemble the full URL for a parsed BackendRequest against a base URL.
// req.path already includes the /api/v1 prefix (the contract requires the
// caller to pass a prefix-included full path), so this does NOT prepend
// kBackendApiPrefix; it only concatenates base_url + req.path (+ query).
std::string BuildBackendUrl(const std::string& base_url, const BackendRequest& req);

// Build the result object for backend.request from an HTTP status and raw body.
// The body is parsed as JSON; a non-object / empty / invalid body becomes null.
json_t MakeBackendRequestResult(int status, const std::string& body_utf8);

// ---------------------------------------------------------------------------
// backend.downloadVideo (async) - pure parsing.
// ---------------------------------------------------------------------------

struct DownloadVideoRequest {
    std::string job_id;
    bool joined = false;
    // Contract v6 additions (all optional / additive): omitting all three from
    // params reproduces the exact pre-v6 destination and never-clobber-check
    // behavior (see bridge.cpp's HandleMessage for how these are applied).
    bool has_dest_dir = false;
    std::string dest_dir;   // UTF-8 (valid only when has_dest_dir)
    bool has_file_name = false;
    std::string file_name;  // UTF-8, includes extension (valid only when has_file_name)
    bool no_clobber = false;
    // When true and the resolved (non-noClobber) destination already exists
    // with size > 0, the download is skipped and the existing path/size is
    // returned. Immutable per-clip outputs only (see the WebUI contract);
    // ignored under no_clobber, which already never overwrites.
    bool reuse_if_present = false;
};

// Validate params of backend.downloadVideo. Rejects a job id containing path
// separators or "..". Contract v6: also accepts optional 'destDir' (non-empty
// string), 'fileName' (non-empty string) and 'noClobber' (boolean); a present
// but wrongly-typed/empty value is a parse failure, an absent one leaves the
// corresponding has_* flag false. Also accepts optional 'reuseIfPresent'
// (boolean) with the same rules. Returns false + *err_message on failure.
bool ParseDownloadVideo(const json_t& params, DownloadVideoRequest* out,
                        std::string* err_message);

// Relative video path for a download request, e.g. "/jobs/<id>/video" or
// ".../joined". Combine with kBackendApiPrefix and the base URL to form the URL.
std::string DownloadVideoPath(const DownloadVideoRequest& req);

// Bare download file name, e.g. "<jobId>.mp4" or "<jobId>_joined.mp4".
std::string DownloadFileName(const DownloadVideoRequest& req);

// ---------------------------------------------------------------------------
// backend.uploadFile (async, contract v3) - pure parsing / multipart framing.
// ---------------------------------------------------------------------------

// Upload timeout: large media files can take a while to stream.
inline constexpr int kDefaultUploadTimeoutMs = 120000;  // 120 s

struct UploadFileRequest {
    std::string kind;       // "image" | "video" | "audio"
    std::string file_path;  // UTF-8 absolute path on disk
    // Contract v10: URL-encoded "a=b&c=d" built from the optional 'query'
    // object, or EMPTY when the caller sent no query. Empty means the upload
    // URL is assembled exactly as it was before v10 (see AppendQueryToUrl).
    std::string query;
};

// Validate params of backend.uploadFile. Requires a 'kind' of image/video/audio
// and a non-empty 'filePath'. Contract v10 also accepts an optional 'query'
// object of string values (absent or null leaves out->query empty); a
// non-object query, or a non-string value inside it, is a parse failure.
// Returns false + *err_message on failure.
bool ParseUploadFile(const json_t& params, UploadFileRequest* out,
                     std::string* err_message);

// Relative upload path for a kind, e.g. "/upload/image". Combine with
// kBackendApiPrefix and the base URL to form the full URL.
std::string UploadPath(const std::string& kind);

// Return the last path component of a UTF-8 file path (after the final '/' or
// '\\'). Empty input yields an empty string.
std::string FileNameFromPath(const std::string& path);

// Map a file's extension (case-insensitive) to a MIME content type per the v3
// contract. Unknown / missing extensions yield "application/octet-stream".
std::string ContentTypeForExtension(const std::string& path);

// Build the leading bytes of one multipart/form-data part (everything up to and
// including the blank line that precedes the raw file payload):
//   --<boundary>\r\n
//   Content-Disposition: form-data; name="<field>"; filename="<filename>"\r\n
//   Content-Type: <content_type>\r\n
//   \r\n
std::string BuildMultipartHeader(const std::string& boundary,
                                 const std::string& field_name,
                                 const std::string& filename,
                                 const std::string& content_type);

// Build the trailing bytes of a single-part multipart/form-data body:
//   \r\n--<boundary>--\r\n
std::string BuildMultipartFooter(const std::string& boundary);

// The full "multipart/form-data; boundary=<boundary>" request Content-Type.
std::string MultipartContentType(const std::string& boundary);

// ---------------------------------------------------------------------------
// timeline.captureFrame (async, contract v3) - pure parsing.
// ---------------------------------------------------------------------------

struct CaptureFrameRequest {
    bool has_frame = false;  // false -> caller uses the current cursor frame
    int frame = 0;
};

// Validate params of timeline.captureFrame. 'frame' is optional; when present it
// must be an integer. Returns false + *err_message on failure.
bool ParseCaptureFrame(const json_t& params, CaptureFrameRequest* out,
                       std::string* err_message);

// Bare capture file name for a frame, e.g. "frame_<frame>_<unique>.png".
std::string CaptureFileName(int frame, const std::string& unique);

// ---------------------------------------------------------------------------
// ui.pickFile (contract v3) - pure kind validation / open-dialog filter spec.
//
// The bridge shows the native Open dialog on the UI thread; this module only
// validates the 'kind' and builds the human-readable/pattern filter entries so
// the wiring can be unit-tested without any Win32 UI dependency.
// ---------------------------------------------------------------------------

// One entry of a native file-open dialog filter.
struct PickFileFilterEntry {
    std::string label;    // e.g. "Image files (*.png;*.jpg;*.jpeg;*.webp)"
    std::string pattern;  // e.g. "*.png;*.jpg;*.jpeg;*.webp"
};

// Validate params of ui.pickFile. Requires a 'kind' of image/video/audio/
// imageOrVideo — the last is Chain's unified source-input picker (task brief
// "Chainのソース入力欄一本化"), whose caller routes the picked file to the
// image or video flow by extension after the fact instead of picking a kind
// up front.
// Returns false + *err_message on failure (mapped to BAD_REQUEST by the bridge).
bool ParsePickFileKind(const json_t& params, std::string* kind_out,
                       std::string* err_message);

// Filter entries for a validated kind, in dialog order. "image"/"video"/
// "audio" each yield their single-type entry followed by an "All files (*.*)"
// catch-all; "imageOrVideo" leads with a combined image+video filter, then the
// same image-only and video-only entries, then All files. The extensions match
// the backend allow-list. An unrecognised kind yields just the All-files entry.
std::vector<PickFileFilterEntry> PickFileFilter(const std::string& kind);

// ---------------------------------------------------------------------------
// ui.makeThumbnail (contract v3) - pure parsing / sizing / base64 / data URL.
//
// The decode/scale/encode itself is done with WIC off the UI thread (see
// wic_png.h); everything testable without a real image lives here.
// ---------------------------------------------------------------------------

inline constexpr int kDefaultThumbnailMaxDim = 256;
inline constexpr int kMinThumbnailMaxDim = 16;
inline constexpr int kMaxThumbnailMaxDim = 1024;

// MIME type of the thumbnail image the bridge produces (JPEG, see wic_png.cpp).
inline constexpr char kThumbnailMimeType[] = "image/jpeg";

struct MakeThumbnailRequest {
    std::string file_path;                      // UTF-8 absolute path on disk
    int max_dim = kDefaultThumbnailMaxDim;      // clamped into [16, 1024]
};

// Validate params of ui.makeThumbnail. Requires a non-empty string 'filePath';
// 'maxDim' is optional (default 256) and, when present, must be a number - it is
// clamped into [16, 1024]. Returns false + *err_message (BAD_REQUEST) on failure.
bool ParseMakeThumbnail(const json_t& params, MakeThumbnailRequest* out,
                        std::string* err_message);

// Compute the scaled thumbnail dimensions that preserve the source aspect ratio
// so the long edge is at most max_dim. When the source already fits (long edge
// <= max_dim) the source dimensions are returned unchanged (1:1). Each output is
// at least 1. Non-positive inputs yield 0x0.
void ComputeThumbnailSize(int src_w, int src_h, int max_dim, int* out_w,
                          int* out_h);

// Base64-encode raw bytes with the standard alphabet and '=' padding, no line
// breaks. An empty input yields an empty string.
std::string Base64Encode(const unsigned char* data, size_t len);

// Build a data URL "data:<mime>;base64,<base64>".
std::string MakeDataUrl(const std::string& mime, const std::string& base64);

// ---------------------------------------------------------------------------
// timeline.getSelection - pure result formatting.
// ---------------------------------------------------------------------------

// Build the JSON result object for timeline.getSelection from a snapshot. Maps
// has_file_path / has_object_name == false to a JSON null for filePath /
// objectName (contract v5's nullable fields).
json_t MakeSelectionResult(const SelectionSnapshot& snap);

// Parse the raw value of AviUtl2's "playback position" item (contract v10).
//
// Real-hardware finding (2026-08-01, Docs/V2V_RIBBON_TRIM_WORKORDER.md section
// 4): the raw value is a comma-separated list whose FIRST TWO fields are the
// start and end of the played window, in SECONDS on the source's own time axis
// with 3 decimals and a '.' decimal point - e.g. "2.000,10.700,<mode>,0". The
// third field is a localized mode name and the fourth a flag; both are ignored
// here on purpose, so a future AviUtl2 build that renames the mode, or adds /
// drops trailing fields, cannot break the parse. Nothing but the leading two
// numbers is required.
//
// Returns false (and leaves the outputs untouched) for anything it cannot read
// with certainty: fewer than two fields, a field that is not entirely a number,
// a non-finite value, a negative start, or an end before the start. Every false
// makes the caller fall back to "no playback range reported", which the WebUI
// turns into an untrimmed, pre-v10 upload - the conservative direction.
//
// Locale-independent: uses std::from_chars, never strtod/atof.
bool ParsePlaybackRange(const std::string& raw, double* start_sec, double* end_sec);

// Parse the raw value of AviUtl2's "playback speed" item (contract v10), a
// percentage string with 2 decimals ("100.00" = normal speed), into a 1.0-scale
// multiplier (1.0 = neutral). Returns false for an unparseable or non-finite
// value, in which case the caller keeps the neutral default rather than
// guessing. Locale-independent (std::from_chars).
bool ParsePlaybackSpeedPercent(const std::string& raw, double* speed);

// ---------------------------------------------------------------------------
// timeline.insertProvisional - pure parsing + alias construction.
// ---------------------------------------------------------------------------

struct InsertProvisionalParams {
    std::string job_id;
    std::string display_text;
    // Optional 4-stage label prefix (spec section 5-5). Empty -> the default
    // "generating:" prefix; the stage-1 reservation insert passes the "reserved:"
    // prefix instead. Forwarded to BuildProvisionalPlaceholder.
    std::string text_prefix;
    // Length via generation frames + generation fps (required): the length is
    // resolved natively with ProjectFramesForPixels(num_frames, gen_fps,
    // project_fps) at dispatch time (project_fps from EDIT_INFO rate/scale).
    int num_frames = 0;   // > 0
    double gen_fps = 0.0;  // > 0
    // Placement system (section 5-9), required (one of "A"/"B"/"C").
    ProvisionalPlacement placement = ProvisionalPlacement::kCursor;
    bool has_material_layer = false;
    int material_layer = 0;
    bool has_material_frame_start = false;
    int material_frame_start = 0;
    bool has_material_frame_end = false;
    int material_frame_end = 0;
    bool has_cursor_layer = false;
    int cursor_layer = 0;
    bool has_cursor_frame = false;
    int cursor_frame = 0;
};

// Validate params of timeline.insertProvisional. Requires a non-empty string
// 'jobId' and a string 'displayText'. The length is the pair 'numFrames'
// (positive int) + 'genFps' (positive number), both required (resolved to a
// project-frame length natively). The position is the required 'placement'
// ("A"/"B"/"C") plus the relevant material_*/cursor_* fields; a missing or
// invalid 'placement' is rejected. An optional 'textPrefix' string (spec section
// 5-5) selects the label prefix; absent/null leaves the default, a present
// non-string is rejected. Returns false + *err_message (BAD_REQUEST) on failure.
bool ParseInsertProvisional(const json_t& params, InsertProvisionalParams* out,
                            std::string* err_message);

// Build the provisional placeholder alias + object_name for a job, pinned to
// length_frames project frames. Pure wrapper over BuildProvisionalTextAlias +
// NormalizeAliasObjectFrameHeader; the produced alias round-trips through
// provisional::AliasMatchesJob(alias, job_id). text_prefix picks the 4-stage
// label prefix (spec section 5-5): empty = the default "generating:" prefix
// (stage 2); a non-empty value (the stage-1 "reserved:" prefix) is used verbatim.
ProvisionalTextAlias BuildProvisionalPlaceholder(const std::string& job_id,
                                                 const std::string& display_text,
                                                 int length_frames,
                                                 const std::string& text_prefix = std::string());

// ---------------------------------------------------------------------------
// timeline.resolveProvisional - pure parsing.
// ---------------------------------------------------------------------------

struct ResolveProvisionalParams {
    std::string job_id;
    std::string video_file_path;
    int reserved_layer = 0;
    int reserved_frame = 0;
    int length_frames = 0;  // > 0
};

// Validate params of timeline.resolveProvisional. Requires a non-empty string
// 'jobId', a non-empty string 'videoFilePath', integer 'reservedLayer' /
// 'reservedFrame', and a positive integer 'lengthFrames'.
bool ParseResolveProvisional(const json_t& params, ResolveProvisionalParams* out,
                             std::string* err_message);

// ---------------------------------------------------------------------------
// timeline.updateProvisionalText - pure parsing.
// ---------------------------------------------------------------------------

struct UpdateProvisionalTextParams {
    std::string job_id;
    int layer = 0;
    int frame = 0;
    std::string text;
};

// Validate params of timeline.updateProvisionalText. Requires a non-empty string
// 'jobId', integer 'layer' / 'frame', and a string 'text'.
bool ParseUpdateProvisionalText(const json_t& params,
                                UpdateProvisionalTextParams* out,
                                std::string* err_message);

// ---------------------------------------------------------------------------
// timeline.updateProvisionalReservation (I3) - pure parsing.
// ---------------------------------------------------------------------------

struct UpdateProvisionalReservationParams {
    std::string old_job_id;   // may be empty (skip the delete search)
    std::string new_job_id;   // required non-empty
    std::string display_text;
    // Optional 4-stage label prefix (spec section 5-5), same contract as
    // InsertProvisionalParams::text_prefix. Empty -> the default "generating:"
    // prefix (stage 2, used by the Generate-time bind); the "reserved:" prefix is
    // passed when a still-unbound reservation is MOVED (stage 1 preserved).
    std::string text_prefix;
    // Length is always via numFrames+genFps for this RPC (no legacy lengthFrames).
    int num_frames = 0;   // > 0
    double gen_fps = 0.0;  // > 0
    // Placement is required (one of "A"/"B"/"C"); never kLegacy here.
    ProvisionalPlacement placement = ProvisionalPlacement::kAfterMaterial;
    bool has_material_layer = false;
    int material_layer = 0;
    bool has_material_frame_start = false;
    int material_frame_start = 0;
    bool has_material_frame_end = false;
    int material_frame_end = 0;
    bool has_cursor_layer = false;
    int cursor_layer = 0;
    bool has_cursor_frame = false;
    int cursor_frame = 0;
};

// Validate params of timeline.updateProvisionalReservation. Requires a string
// 'oldJobId' (may be empty -> create-only), a non-empty string 'newJobId', a
// string 'displayText', a positive integer 'numFrames', a positive number
// 'genFps', and a 'placement' of "A"/"B"/"C" (plus the material_*/cursor_*
// fields the placement needs). An optional 'textPrefix' string (spec section
// 5-5) selects the label prefix (same rules as insertProvisional). Returns false
// + *err_message (BAD_REQUEST) on failure.
bool ParseUpdateProvisionalReservation(const json_t& params,
                                       UpdateProvisionalReservationParams* out,
                                       std::string* err_message);

// ---------------------------------------------------------------------------
// timeline.deleteProvisionalByJob (spec 5-10) - pure parsing.
// ---------------------------------------------------------------------------

struct DeleteProvisionalByJobParams {
    std::string job_id;  // required non-empty
};

// Validate params of timeline.deleteProvisionalByJob. Requires a non-empty
// string 'jobId'. Returns false + *err_message (BAD_REQUEST) on failure.
bool ParseDeleteProvisionalByJob(const json_t& params,
                                 DeleteProvisionalByJobParams* out,
                                 std::string* err_message);

// ---------------------------------------------------------------------------
// timeline.insertMediaForJob (replace-insert) - pure parsing.
// ---------------------------------------------------------------------------

struct InsertMediaForJobParams {
    std::string job_id;     // required non-empty
    std::string file_path;  // required non-empty (UTF-8 absolute path)
};

// Validate params of timeline.insertMediaForJob. Requires a non-empty string
// 'jobId' and a non-empty string 'filePath'. Returns false + *err_message
// (BAD_REQUEST) on failure.
bool ParseInsertMediaForJob(const json_t& params, InsertMediaForJobParams* out,
                            std::string* err_message);

// ---------------------------------------------------------------------------
// timeline.cutoutRange (async, contract v5) - pure parsing / result formatting.
//
// Dispatch stays in the bridge (mirrors timeline.captureFrame): the bridge
// validates with ParseCutoutRange, renders/muxes the range to a temp mp4 off the
// UI thread (see bridge.cpp handoff I7), and formats the reply with
// MakeCutoutResult. HandleRequestJson does NOT handle this method.
// ---------------------------------------------------------------------------

struct CutoutRangeRequest {
    int layer = 0;
    int frame_start = 0;
    int frame_count = 0;  // > 0
    bool with_audio = false;
    std::string audio_mode;  // "mix" | "solo"
    std::vector<int> solo_keep_layers;
};

// Validate params of timeline.cutoutRange. Requires integer 'layer' /
// 'frameStart', a positive integer 'frameCount', a boolean 'withAudio', an
// 'audioMode' of "mix" | "solo", and an integer array 'soloKeepLayers'.
bool ParseCutoutRange(const json_t& params, CutoutRangeRequest* out,
                      std::string* err_message);

// Build the JSON result object for timeline.cutoutRange.
json_t MakeCutoutResult(const std::string& file_path, int width, int height,
                                int frame_count, bool has_audio);

// ---------------------------------------------------------------------------
// timeline.extractAudio (async, contract v5) - pure parsing / result formatting.
//
// Dispatch stays in the bridge (mirrors timeline.captureFrame): the bridge
// validates with ParseExtractAudio, extracts/encodes the range's audio off the
// UI thread (see bridge.cpp handoff I8), and formats the reply with
// MakeExtractAudioResult. HandleRequestJson does NOT handle this method.
// ---------------------------------------------------------------------------

struct ExtractAudioRequest {
    int layer = 0;
    int frame_start = 0;
    int frame_count = 0;  // > 0
    std::string audio_mode;  // "mix" | "solo"
    std::vector<int> solo_keep_layers;
};

// Validate params of timeline.extractAudio. Requires integer 'layer' /
// 'frameStart', a positive integer 'frameCount', an 'audioMode' of "mix" |
// "solo", and an integer array 'soloKeepLayers'.
bool ParseExtractAudio(const json_t& params, ExtractAudioRequest* out,
                       std::string* err_message);

// Build the JSON result object for timeline.extractAudio.
json_t MakeExtractAudioResult(const std::string& file_path, double duration_sec,
                                      int sample_rate, bool has_audio_stream);

// Amplitude (normalized to [0, 1]) below which a captured audio range counts as
// silent: one LSB of 16-bit PCM. The bridge's audio extractor (I8) computes the
// peak absolute amplitude while muxing and passes it to IsAudioStreamPresent.
inline constexpr double kAudioSilenceThreshold = 1.0 / 32768.0;

// True when a captured audio range carries a real (non-silent) stream: at least
// one sample and a peak amplitude at or above kAudioSilenceThreshold. The bridge
// uses this to decide hasAudioStream (and, when false for a range that was meant
// to carry audio, to raise EXTRACT_FAILED).
bool IsAudioStreamPresent(int sample_count, double peak_abs_amplitude);

// ---------------------------------------------------------------------------
// ui.pickFolder (contract v6) - pure params parsing.
//
// Dispatch stays in the bridge (mirrors ui.pickFile): the bridge shows a modal
// IFileOpenDialog with FOS_PICKFOLDERS on the UI thread, reusing ui.pickFile's
// re-entrancy guard and CANCELLED/DIALOG_FAILED error convention.
// HandleRequestJson does NOT handle this method.
// ---------------------------------------------------------------------------

struct PickFolderRequest {
    bool has_title = false;
    std::string title;  // UTF-8 (valid only when has_title)
};

// Validate params of ui.pickFolder. Params (and 'title' within them) are both
// optional; an absent/null 'title' just means "use native's default title". A
// present-but-non-string 'title' is a parse failure. Returns false +
// *err_message (BAD_REQUEST) on failure.
bool ParsePickFolder(const json_t& params, PickFolderRequest* out,
                     std::string* err_message);

// ---------------------------------------------------------------------------
// fs.listFiles (contract v6) - pure params parsing / result formatting.
//
// Dispatch stays in the bridge (mirrors timeline.captureFrame): the bridge
// enumerates the folder off the UI thread with FindFirstFileW/FindNextFileW
// (directories and hidden/system entries skipped, no recursion), filters with
// fs_util's MatchesAnyExtension, and - when requested - probes each matching
// .wav's duration with wav_probe's ProbeWavFile. This module only validates
// params and formats the JSON result from the pre-collected entries.
// HandleRequestJson does NOT handle this method.
// ---------------------------------------------------------------------------

struct ListFilesRequest {
    std::string folder_path;
    std::vector<std::string> extensions;  // empty -> no filter
    bool with_audio_duration = false;
};

// Validate params of fs.listFiles. Requires a non-empty string 'folderPath'.
// 'extensions', when present, must be an array of strings (each compared via
// fs_util::MatchesAnyExtension - case-insensitive, leading '.' significant).
// 'withAudioDuration', when present, must be a boolean. Returns false +
// *err_message (BAD_REQUEST) on failure.
bool ParseListFiles(const json_t& params, ListFilesRequest* out,
                    std::string* err_message);

// One folder entry pre-collected by the bridge's Win32 enumeration.
struct ListedFile {
    std::string name;         // UTF-8 file name (no path)
    std::string path;         // UTF-8 absolute path (folder_path + name)
    long long size_bytes = 0;
    double mtime_ms = 0;      // milliseconds since the Unix epoch
    double duration_sec = 0;  // 0 when not requested / not a readable .wav
};

// Build the JSON result object for fs.listFiles from the pre-collected entries.
// Native's enumeration order is unspecified (contract note: callers must sort
// 'files' themselves), so this simply preserves the input order.
json_t MakeListFilesResult(const std::vector<ListedFile>& files);

// ---------------------------------------------------------------------------
// fs.probeAudioDuration (contract v6) - pure params parsing.
//
// Dispatch stays in the bridge (mirrors timeline.captureFrame): the bridge
// probes the file off the UI thread with wav_probe's ProbeWavFile (gated on a
// ".wav" extension) and formats {"durationSec", "isWav"} itself. A non-.wav
// file or an unreadable .wav never errors - it resolves {0, false} (see
// bridge.cpp). HandleRequestJson does NOT handle this method.
// ---------------------------------------------------------------------------

struct ProbeAudioDurationRequest {
    std::string file_path;  // UTF-8
};

// Validate params of fs.probeAudioDuration. Requires a non-empty string
// 'filePath'. Returns false + *err_message (BAD_REQUEST) on failure.
bool ParseProbeAudioDuration(const json_t& params, ProbeAudioDurationRequest* out,
                             std::string* err_message);

// ---------------------------------------------------------------------------
// fs.probeMediaInfo (synchronous) - pure params parsing / result formatting.
//
// Unlike fs.probeAudioDuration this IS dispatched synchronously by
// HandleRequestJson (the SDK get_media_info call is cheap and already runs on
// the UI thread via call_edit_section_param, so no async worker hand-off is
// needed - it mirrors timeline.getSelection). The probe never errors: any
// failure resolves the all-zero result (MediaInfoSnapshot::available == false).
// ---------------------------------------------------------------------------

struct ProbeMediaInfoRequest {
    std::string file_path;  // UTF-8
};

// Validate params of fs.probeMediaInfo. Requires a non-empty string 'filePath'.
// Returns false + *err_message (BAD_REQUEST) on failure.
bool ParseProbeMediaInfo(const json_t& params, ProbeMediaInfoRequest* out,
                         std::string* err_message);

// Build the JSON result object for fs.probeMediaInfo from a snapshot. When the
// snapshot is unavailable, every field is emitted as 0
// ({durationSec:0,width:0,height:0}); otherwise the probed values are passed
// through verbatim.
json_t MakeMediaInfoResult(const MediaInfoSnapshot& snap);

// ---------------------------------------------------------------------------
// ui.resolveDroppedFiles (contract v7) - drag-and-drop file resolution.
//
// Handled here (HandleRequestJson), like ping/getEditInfo - it is pure,
// synchronous, in-memory work with no I/O and no SDK dependency, so no async
// worker hand-off (bridge.cpp) is needed for it, unlike ui.pickFile/pickFolder
// (modal dialogs) or the fs.* methods (real disk I/O). The WebUI keeps this on
// its normal 10s local timeout (it is NOT added to NO_LOCAL_TIMEOUT_METHODS).
//
// 'params.__droppedPaths' is a RESERVED key the page never legitimately sets
// itself. webview_host.cpp's WebMessageReceived handler decides, per message,
// whether to force-overwrite it (see InjectDroppedPathsIntoRequest below for
// the pure half of that decision):
//   - this message carries >=1 AdditionalObject -> ALWAYS overwritten with
//     whatever resolved to a real file path (possibly still `[]`, if none of
//     the attached objects were files);
//   - 0 AdditionalObjects but the raw string mentions "__droppedPaths" -> also
//     overwritten, forced to `[]` (anti-spoofing: neutralizes a page-authored
//     value even though this message carried no real drop);
//   - 0 AdditionalObjects and no mention of the key at all -> left untouched
//     (cheap fast path; there is no key for HandleRequestJson to ever see, so
//     nothing to spoof or lose).
// Because that discipline is enforced one layer up, ParseResolveDroppedFiles
// itself never rejects a request - a missing/malformed key just means
// "nothing was dropped".
// ---------------------------------------------------------------------------

struct ResolveDroppedFilesRequest {
    std::vector<std::string> file_paths;  // UTF-8 absolute paths, in order
};

// Reads params.__droppedPaths. A missing key, a non-array value, and any
// non-string or empty-string element are all simply excluded (never
// BAD_REQUEST - see the module comment above for why). Never throws.
ResolveDroppedFilesRequest ParseResolveDroppedFiles(const json_t& params);

// Build the JSON result object for ui.resolveDroppedFiles: one
// {filePath, fileName} entry per path in req.file_paths, in the same order.
// fileName is the last path component (FileNameFromPath).
json_t MakeResolveDroppedFilesResult(const ResolveDroppedFilesRequest& req);

// ---------------------------------------------------------------------------
// InjectDroppedPathsIntoRequest (contract v7) - the pure, doctest-able half of
// webview_host.cpp's __droppedPaths injection. webview_host.cpp itself only
// decides WHETHER to call this (based on AdditionalObjects presence / the
// "__droppedPaths" substring pre-check, see the section comment above) and
// resolves `paths` from AdditionalObjects (WebView2-dependent, not testable
// here); everything about HOW the request string gets rewritten lives in this
// pure function so it can be unit-tested without any COM/WebView2 dependency.
//
// Parses `request_json`; if it parses to a JSON object, ensures its 'params'
// is an object (creating an empty one if 'params' was missing or not an
// object - this is what lets an empty-params request like
// `{"id":1,"method":"ui.resolveDroppedFiles","params":{}}` still receive the
// key: the caller never has to pre-populate 'params' with anything for a real
// drop to land) and unconditionally overwrites params.__droppedPaths with
// `paths` (as a JSON array of strings, possibly empty), then returns the
// re-serialized request. If `request_json` does not parse to a JSON object at
// all, it is returned UNCHANGED - the only allowed passthrough case (whoever
// calls this decides not to call it at all for the "no AdditionalObjects, no
// mention of the key" fast path; this function itself always injects once
// called, regardless of `paths` being empty).
std::string InjectDroppedPathsIntoRequest(const std::string& request_json,
                                          const std::vector<std::string>& paths);

}  // namespace nzvideomni
