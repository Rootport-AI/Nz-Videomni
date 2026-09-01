// bridge_core.cpp - implementation of the WebView2-independent RPC dispatch.
// See bridge_core.h for the RPC contract. ASCII-only source.
#include "bridge_core.h"

#include <charconv>  // std::from_chars (locale-independent number parsing, v10)
#include <cmath>     // std::isfinite (v10 playback-range validation)
#include <set>

#include "timeline_math.h"  // ProjectFramesForPixels (I3: numFrames+genFps length)

namespace nzvideomni {

namespace {

using json = json_t;

// I3: encode a missing optional integer as -1 so ResolveProvisionalPlacement's
// ">= 0" validation rejects it (all legitimate layer/frame values are >= 0).
int OrSentinel(bool has, int value) { return has ? value : -1; }

// Percent-encode a string for use in a URL query component (RFC 3986
// unreserved characters pass through; everything else is %XX).
std::string UrlEncode(const std::string& in) {
    static const char* kHex = "0123456789ABCDEF";
    std::string out;
    out.reserve(in.size() * 3);
    for (unsigned char c : in) {
        const bool unreserved = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
                                (c >= '0' && c <= '9') || c == '-' || c == '_' ||
                                c == '.' || c == '~';
        if (unreserved) {
            out.push_back(static_cast<char>(c));
        } else {
            out.push_back('%');
            out.push_back(kHex[(c >> 4) & 0xF]);
            out.push_back(kHex[c & 0xF]);
        }
    }
    return out;
}

// Read one whole field as a double, locale-independently. The field must consist
// of nothing but the number (surrounding spaces/tabs are tolerated and stripped);
// trailing junk like "1.5x" is rejected so a value we do not fully understand can
// never be silently truncated into a plausible-looking number.
bool ParseWholeDoubleField(const std::string& field, double* out) {
    size_t b = 0;
    size_t e = field.size();
    while (b < e && (field[b] == ' ' || field[b] == '\t')) ++b;
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

}  // namespace

std::string MakeSuccessResponse(const json& id, json result) {
    json out;
    out["id"] = id;
    out["ok"] = true;
    out["result"] = std::move(result);
    return out.dump();
}

std::string MakeErrorResponse(const json& id, const std::string& code,
                              const std::string& message) {
    json out;
    out["id"] = id;
    out["ok"] = false;
    out["error"] = {{"code", code}, {"message", message}};
    return out.dump();
}

json ExtractId(const json& req) {
    if (req.is_object() && req.contains("id") && req["id"].is_number()) {
        return req["id"];
    }
    return json(nullptr);
}

std::string HandleRequestJson(const std::string& request_json,
                              const RequestContext& ctx) {
    json req = json::parse(request_json, nullptr, /*allow_exceptions=*/false);

    if (req.is_discarded() || !req.is_object()) {
        return MakeErrorResponse(json(nullptr), "BAD_REQUEST", "Malformed JSON request");
    }

    const bool has_id = req.contains("id");
    const bool has_method = req.contains("method");
    if (!has_id && !has_method) {
        return std::string();
    }

    const json id = ExtractId(req);

    if (!has_method || !req["method"].is_string()) {
        return MakeErrorResponse(id, "BAD_REQUEST", "Missing or invalid 'method' field");
    }

    const std::string method = req["method"].get<std::string>();
    const json params =
        (req.contains("params") && req["params"].is_object()) ? req["params"] : json::object();

    if (method == "ping") {
        json result;
        result["pong"] = true;
        result["pluginVersion"] = kPluginVersion;
        return MakeSuccessResponse(id, std::move(result));
    }

    if (method == "getEditInfo") {
        const EditInfoResult info = ctx.edit_info ? ctx.edit_info() : EditInfoResult{};
        if (!info.available) {
            return MakeErrorResponse(id, "NO_EDIT_HANDLE", "Edit handle is not available");
        }
        json result;
        result["width"] = info.width;
        result["height"] = info.height;
        result["rate"] = info.rate;
        result["scale"] = info.scale;
        result["sampleRate"] = info.sample_rate;
        result["frame"] = info.frame;
        result["layer"] = info.layer;
        result["frameMax"] = info.frame_max;
        result["layerMax"] = info.layer_max;
        return MakeSuccessResponse(id, std::move(result));
    }

    if (method == "backend.getBaseUrl") {
        json result;
        result["baseUrl"] = ctx.base_url;
        return MakeSuccessResponse(id, std::move(result));
    }

    if (method == "timeline.insertMedia") {
        if (!params.contains("filePath") || !params["filePath"].is_string() ||
            params["filePath"].get<std::string>().empty()) {
            return MakeErrorResponse(id, "BAD_REQUEST",
                                     "insertMedia requires a non-empty 'filePath'");
        }
        InsertMediaParams p;
        p.file_path = params["filePath"].get<std::string>();
        if (params.contains("layer") && params["layer"].is_number_integer()) {
            p.has_layer = true;
            p.layer = params["layer"].get<int>();
        }
        if (params.contains("frame") && params["frame"].is_number_integer()) {
            p.has_frame = true;
            p.frame = params["frame"].get<int>();
        }
        const InsertMediaResult r =
            ctx.insert_media ? ctx.insert_media(p) : InsertMediaResult{};
        switch (r.status) {
            case InsertMediaResult::Status::kOk: {
                json result;
                result["inserted"] = true;
                result["layer"] = r.layer;
                result["frame"] = r.frame;
                return MakeSuccessResponse(id, std::move(result));
            }
            case InsertMediaResult::Status::kFileNotFound:
                return MakeErrorResponse(id, "FILE_NOT_FOUND",
                                         "Media file not found: " + p.file_path);
            case InsertMediaResult::Status::kNoEditHandle:
                return MakeErrorResponse(id, "NO_EDIT_HANDLE",
                                         "Edit handle is not available");
            case InsertMediaResult::Status::kInsertFailed:
            default:
                return MakeErrorResponse(id, "INSERT_FAILED",
                                         "create_object_from_media_file returned null "
                                         "(unsupported format or overlapping object)");
        }
    }

    // --- timeline.insertMediaForJob (replace-insert 🎞) ---------------------
    if (method == "timeline.insertMediaForJob") {
        InsertMediaForJobParams p;
        std::string err;
        if (!ParseInsertMediaForJob(params, &p, &err)) {
            return MakeErrorResponse(id, "BAD_REQUEST", err);
        }
        // Locate this job's provisional placeholder by EXACT job-id match across
        // all layers. Pass a -1 reservation position so FindProvisionalIndex uses
        // ONLY the job-id match (never a position fallback) - the replace must
        // never touch an object that does not carry this job's marker.
        const std::vector<ScannedObject> scanned =
            ctx.scan_objects ? ctx.scan_objects() : std::vector<ScannedObject>{};
        const int idx = FindProvisionalIndex(scanned, p.job_id, /*reserved_layer=*/-1,
                                             /*reserved_frame=*/-1);
        if (idx >= 0) {
            // Found -> REPLACE the placeholder in place at its own layer/frame.
            // The provider's EditProc re-finds the placeholder inside the section,
            // deletes it, and creates the media at the same slot sized to its real
            // length. No HasRoomForLength pre-check (see the header rationale): a
            // room check would false-positive against the marker we are deleting.
            ReplaceMediaForJobRequest rr;
            rr.job_id = p.job_id;
            rr.layer = scanned[static_cast<size_t>(idx)].layer;
            rr.frame = scanned[static_cast<size_t>(idx)].frame_start;
            rr.file_path = p.file_path;
            const ReplaceMediaForJobOutcome outcome =
                ctx.replace_media_for_job ? ctx.replace_media_for_job(rr)
                                          : ReplaceMediaForJobOutcome{};
            if (!outcome.ok) {
                return MakeErrorResponse(
                    id, "INSERT_FAILED",
                    "create_object_from_media_file returned null for the replace-"
                    "insert (unsupported/broken media or a blocked slot)");
            }
            json result;
            result["ok"] = true;
            result["mode"] = "replaced";
            result["layer"] = outcome.layer;
            result["frame"] = outcome.frame;
            // usedFallback is true only when the marker-slot create collided and
            // the provider retried on layer_max+1 (never true on the insert path).
            result["usedFallback"] = outcome.used_fallback;
            return MakeSuccessResponse(id, std::move(result));
        }
        // Not found -> behave EXACTLY like timeline.insertMedia (owner decision,
        // case 1: normal-generation 🎞 is unchanged). Reuse the same insert_media
        // provider with no explicit layer/frame so it uses the current selection
        // layer / cursor frame and the media's real length.
        InsertMediaParams ip;
        ip.file_path = p.file_path;
        const InsertMediaResult ir =
            ctx.insert_media ? ctx.insert_media(ip) : InsertMediaResult{};
        switch (ir.status) {
            case InsertMediaResult::Status::kOk: {
                json result;
                result["ok"] = true;
                result["mode"] = "inserted";
                result["layer"] = ir.layer;
                result["frame"] = ir.frame;
                result["usedFallback"] = false;
                return MakeSuccessResponse(id, std::move(result));
            }
            case InsertMediaResult::Status::kFileNotFound:
                return MakeErrorResponse(id, "FILE_NOT_FOUND",
                                         "Media file not found: " + p.file_path);
            case InsertMediaResult::Status::kNoEditHandle:
                return MakeErrorResponse(id, "NO_EDIT_HANDLE",
                                         "Edit handle is not available");
            case InsertMediaResult::Status::kInsertFailed:
            default:
                return MakeErrorResponse(id, "INSERT_FAILED",
                                         "create_object_from_media_file returned null "
                                         "(unsupported format or overlapping object)");
        }
    }

    if (method == "settings.get") {
        json result;
        result["baseUrl"] = ctx.settings_get ? ctx.settings_get() : ctx.base_url;
        return MakeSuccessResponse(id, std::move(result));
    }

    if (method == "settings.set") {
        if (params.contains("baseUrl") && !params["baseUrl"].is_null()) {
            if (!params["baseUrl"].is_string()) {
                return MakeErrorResponse(id, "BAD_REQUEST",
                                         "settings.set 'baseUrl' must be a string");
            }
            if (!ctx.settings_set) {
                return MakeErrorResponse(id, "BAD_REQUEST",
                                         "settings store is not available");
            }
            const SettingsSetOutcome outcome =
                ctx.settings_set(params["baseUrl"].get<std::string>());
            if (!outcome.ok) {
                return MakeErrorResponse(id, "BAD_REQUEST", outcome.err_message);
            }
        }
        json result;
        result["baseUrl"] = ctx.settings_get ? ctx.settings_get() : ctx.base_url;
        return MakeSuccessResponse(id, std::move(result));
    }

    // --- timeline.getSelection (contract v5) --------------------------------
    if (method == "timeline.getSelection") {
        const SelectionSnapshot snap =
            ctx.get_selection ? ctx.get_selection() : SelectionSnapshot{};
        if (!snap.available) {
            return MakeErrorResponse(id, "NO_EDIT_HANDLE", "Edit handle is not available");
        }
        return MakeSuccessResponse(id, MakeSelectionResult(snap));
    }

    // --- timeline.insertProvisional (contract v5; I3 staged migration) ------
    if (method == "timeline.insertProvisional") {
        InsertProvisionalParams p;
        std::string err;
        if (!ParseInsertProvisional(params, &p, &err)) {
            return MakeErrorResponse(id, "BAD_REQUEST", err);
        }
        // Both the numFrames+genFps length (ProjectFramesForPixels needs
        // project_fps) and the placement resolution (B needs layer_max) consult
        // EDIT_INFO, so load it once and require it to be available.
        const EditInfoResult info =
            ctx.edit_info ? ctx.edit_info() : EditInfoResult{};
        if (!info.available || info.scale == 0) {
            return MakeErrorResponse(
                id, "BAD_REQUEST",
                "timeline.insertProvisional needs an available edit handle with a "
                "non-zero rate/scale");
        }
        // Resolve the length from numFrames+genFps.
        const double project_fps =
            static_cast<double>(info.rate) / static_cast<double>(info.scale);
        const int length = ProjectFramesForPixels(p.num_frames, p.gen_fps, project_fps);
        if (length <= 0) {
            return MakeErrorResponse(
                id, "BAD_REQUEST",
                "timeline.insertProvisional could not resolve a positive length "
                "from numFrames/genFps");
        }
        // Resolve the position from the placement system (section 5-9).
        PlacementResolveInput pin;
        pin.placement = p.placement;
        pin.material_layer = OrSentinel(p.has_material_layer, p.material_layer);
        pin.material_frame_start =
            OrSentinel(p.has_material_frame_start, p.material_frame_start);
        pin.material_frame_end =
            OrSentinel(p.has_material_frame_end, p.material_frame_end);
        pin.cursor_layer = OrSentinel(p.has_cursor_layer, p.cursor_layer);
        pin.cursor_frame = OrSentinel(p.has_cursor_frame, p.cursor_frame);
        pin.layer_max = info.layer_max;
        const PlacementResolveResult pr = ResolveProvisionalPlacement(pin);
        if (!pr.ok) {
            return MakeErrorResponse(id, "BAD_REQUEST", pr.err);
        }
        const int layer = pr.layer;
        const int frame = pr.frame;
        // Alias construction is pure (unit-tested via BuildProvisionalPlaceholder):
        // the placeholder embeds the job id so it can be re-found on resolve, and
        // is pinned to the resolved length so it covers the reserved span exactly.
        const ProvisionalTextAlias built =
            BuildProvisionalPlaceholder(p.job_id, p.display_text, length, p.text_prefix);
        const InsertProvisionalOutcome outcome =
            ctx.insert_provisional
                ? ctx.insert_provisional(built.alias, built.object_name, layer, frame,
                                         length)
                : InsertProvisionalOutcome{};
        if (!outcome.ok) {
            return MakeErrorResponse(id, "PROVISIONAL_FAILED",
                                     "Failed to insert provisional placeholder");
        }
        json result;
        result["inserted"] = true;
        result["layer"] = outcome.layer;
        result["frame"] = outcome.frame;
        result["objectName"] = built.object_name;
        // I3: the placed position + collision-fallback flag. The webui shows the
        // fallback note off usedFallback and tracks the reservation seat by the
        // placed layer/frame.
        result["placedLayer"] = outcome.layer;
        result["placedFrame"] = outcome.frame;
        result["usedFallback"] = outcome.used_fallback;
        return MakeSuccessResponse(id, std::move(result));
    }

    // --- timeline.resolveProvisional (contract v5) --------------------------
    if (method == "timeline.resolveProvisional") {
        ResolveProvisionalParams p;
        std::string err;
        if (!ParseResolveProvisional(params, &p, &err)) {
            return MakeErrorResponse(id, "BAD_REQUEST", err);
        }
        const std::vector<ScannedObject> scanned =
            ctx.scan_objects ? ctx.scan_objects() : std::vector<ScannedObject>{};
        const int idx = FindProvisionalIndex(scanned, p.job_id, p.reserved_layer,
                                             p.reserved_frame);
        const bool found = idx >= 0;
        Reservation reservation;
        reservation.job_id = p.job_id;
        reservation.layer = p.reserved_layer;
        reservation.frame = p.reserved_frame;
        reservation.length_frames = p.length_frames;
        const int found_layer = found ? scanned[static_cast<size_t>(idx)].layer : 0;
        const int found_frame =
            found ? scanned[static_cast<size_t>(idx)].frame_start : 0;
        const ProvisionalDecision decision =
            DecideResolve(found, reservation, found_layer, found_frame);
        if (decision.replace) {
            ReplaceObjectRequest rr;
            rr.job_id = p.job_id;
            rr.layer = decision.layer;
            rr.frame = decision.frame;
            rr.video_file_path = p.video_file_path;
            rr.length_frames = p.length_frames;
            const bool ok = ctx.replace_object ? ctx.replace_object(rr) : false;
            if (!ok) {
                return MakeErrorResponse(id, "PROVISIONAL_FAILED",
                                         "Failed to replace provisional placeholder");
            }
            json result;
            result["mode"] = "replaced";
            result["layer"] = decision.layer;
            result["frame"] = decision.frame;
            return MakeSuccessResponse(id, std::move(result));
        }
        // insertedReserved: the placeholder is gone; insert the finished video at
        // the reserved position via the existing insert_media provider.
        InsertMediaParams ip;
        ip.file_path = p.video_file_path;
        ip.has_layer = true;
        ip.layer = decision.layer;
        ip.has_frame = true;
        ip.frame = decision.frame;
        const InsertMediaResult ir =
            ctx.insert_media ? ctx.insert_media(ip) : InsertMediaResult{};
        if (ir.status != InsertMediaResult::Status::kOk) {
            const char* code =
                ir.status == InsertMediaResult::Status::kFileNotFound  ? "FILE_NOT_FOUND"
                : ir.status == InsertMediaResult::Status::kNoEditHandle ? "NO_EDIT_HANDLE"
                                                                        : "PROVISIONAL_FAILED";
            return MakeErrorResponse(id, code,
                                     "Failed to insert reserved video: " + p.video_file_path);
        }
        json result;
        result["mode"] = "insertedReserved";
        result["layer"] = ir.layer;
        result["frame"] = ir.frame;
        return MakeSuccessResponse(id, std::move(result));
    }

    // --- timeline.updateProvisionalText (contract v5) -----------------------
    if (method == "timeline.updateProvisionalText") {
        UpdateProvisionalTextParams p;
        std::string err;
        if (!ParseUpdateProvisionalText(params, &p, &err)) {
            return MakeErrorResponse(id, "BAD_REQUEST", err);
        }
        const std::vector<ScannedObject> scanned =
            ctx.scan_objects ? ctx.scan_objects() : std::vector<ScannedObject>{};
        // layer/frame double as the reservation fallback for FindProvisionalIndex
        // when the user edited the text and destroyed the "[#...]" marker.
        const int idx = FindProvisionalIndex(scanned, p.job_id, p.layer, p.frame);
        bool updated = false;
        if (idx >= 0 && ctx.update_object_text) {
            UpdateObjectTextRequest ur;
            ur.job_id = p.job_id;
            ur.layer = scanned[static_cast<size_t>(idx)].layer;
            ur.frame = scanned[static_cast<size_t>(idx)].frame_start;
            ur.text = p.text;
            updated = ctx.update_object_text(ur);
        }
        json result;
        result["updated"] = updated;
        return MakeSuccessResponse(id, std::move(result));
    }

    // --- timeline.updateProvisionalReservation (I3) -------------------------
    if (method == "timeline.updateProvisionalReservation") {
        UpdateProvisionalReservationParams p;
        std::string err;
        if (!ParseUpdateProvisionalReservation(params, &p, &err)) {
            return MakeErrorResponse(id, "BAD_REQUEST", err);
        }
        const EditInfoResult info = ctx.edit_info ? ctx.edit_info() : EditInfoResult{};
        if (!info.available || info.scale == 0) {
            return MakeErrorResponse(
                id, "BAD_REQUEST",
                "timeline.updateProvisionalReservation needs an available edit "
                "handle with a non-zero rate/scale");
        }
        const double project_fps =
            static_cast<double>(info.rate) / static_cast<double>(info.scale);
        const int length = ProjectFramesForPixels(p.num_frames, p.gen_fps, project_fps);
        if (length <= 0) {
            return MakeErrorResponse(
                id, "BAD_REQUEST",
                "timeline.updateProvisionalReservation could not resolve a positive "
                "length from numFrames/genFps");
        }
        PlacementResolveInput pin;
        pin.placement = p.placement;
        pin.material_layer = OrSentinel(p.has_material_layer, p.material_layer);
        pin.material_frame_start =
            OrSentinel(p.has_material_frame_start, p.material_frame_start);
        pin.material_frame_end =
            OrSentinel(p.has_material_frame_end, p.material_frame_end);
        pin.cursor_layer = OrSentinel(p.has_cursor_layer, p.cursor_layer);
        pin.cursor_frame = OrSentinel(p.has_cursor_frame, p.cursor_frame);
        pin.layer_max = info.layer_max;
        const PlacementResolveResult pr = ResolveProvisionalPlacement(pin);
        if (!pr.ok) {
            return MakeErrorResponse(id, "BAD_REQUEST", pr.err);
        }
        const ProvisionalTextAlias built =
            BuildProvisionalPlaceholder(p.new_job_id, p.display_text, length, p.text_prefix);
        UpdateReservationRequest rr;
        rr.old_job_id = p.old_job_id;
        rr.new_job_id = p.new_job_id;
        rr.alias = built.alias;
        rr.object_name = built.object_name;
        rr.layer = pr.layer;
        rr.frame = pr.frame;
        rr.length_frames = length;  // D2: explicit length for the create + pre-check
        const UpdateReservationOutcome outcome =
            ctx.update_reservation ? ctx.update_reservation(rr)
                                   : UpdateReservationOutcome{};
        if (!outcome.ok) {
            return MakeErrorResponse(id, "PROVISIONAL_FAILED",
                                     "Failed to update provisional reservation");
        }
        json result;
        result["ok"] = true;
        result["deletedOld"] = outcome.deleted_old;
        result["placedLayer"] = outcome.placed_layer;
        result["placedFrame"] = outcome.placed_frame;
        result["usedFallback"] = outcome.used_fallback;
        return MakeSuccessResponse(id, std::move(result));
    }

    // --- timeline.deleteProvisionalByJob (I13, spec 5-10) -------------------
    if (method == "timeline.deleteProvisionalByJob") {
        DeleteProvisionalByJobParams p;
        std::string err;
        if (!ParseDeleteProvisionalByJob(params, &p, &err)) {
            return MakeErrorResponse(id, "BAD_REQUEST", err);
        }
        const std::vector<ScannedObject> scanned =
            ctx.scan_objects ? ctx.scan_objects() : std::vector<ScannedObject>{};
        // Locate the placeholder by EXACT job-id match across all layers. Pass a
        // -1 reservation position so FindProvisionalIndex uses only the job-id
        // match (never a position fallback) - this delete must never touch an
        // object that does not carry this job's marker. Only the first match is
        // acted on; duplicates are left as harmless orphan text (spec 5-10).
        const int idx = FindProvisionalIndex(scanned, p.job_id, /*reserved_layer=*/-1,
                                             /*reserved_frame=*/-1);
        bool deleted = false;
        if (idx >= 0 && ctx.delete_provisional) {
            DeleteProvisionalRequest dr;
            dr.job_id = p.job_id;
            dr.layer = scanned[static_cast<size_t>(idx)].layer;
            dr.frame = scanned[static_cast<size_t>(idx)].frame_start;
            deleted = ctx.delete_provisional(dr);
        }
        // Idempotent: a missing placeholder is still a success (deleted=false).
        json result;
        result["ok"] = true;
        result["deleted"] = deleted;
        return MakeSuccessResponse(id, std::move(result));
    }

    // --- timeline.scanProvisionals (contract v5) ----------------------------
    if (method == "timeline.scanProvisionals") {
        const std::vector<ScannedObject> scanned =
            ctx.scan_objects ? ctx.scan_objects() : std::vector<ScannedObject>{};
        // Design decision: the contract's params carry NO active-job set (see
        // types.ts - timeline.scanProvisionals takes Record<string, never>), so
        // active-job filtering is the WebUI's responsibility. We therefore run
        // DetectOrphans with an EMPTY active set, which returns every NzVideomni
        // placeholder found on the timeline as an orphan CANDIDATE; the WebUI drops
        // the ones whose job it still tracks. Foreign (non-NzVideomni) objects are
        // ignored by DetectOrphans regardless.
        const std::vector<Reservation> orphans =
            DetectOrphans(scanned, std::set<std::string>{});
        json arr = json::array();
        for (const Reservation& o : orphans) {
            json e;
            e["jobId"] = o.job_id;
            e["layer"] = o.layer;
            e["frame"] = o.frame;
            arr.push_back(std::move(e));
        }
        json result;
        result["orphans"] = std::move(arr);
        return MakeSuccessResponse(id, std::move(result));
    }

    // --- ui.resolveDroppedFiles (contract v7) -------------------------------
    if (method == "ui.resolveDroppedFiles") {
        const ResolveDroppedFilesRequest resolved = ParseResolveDroppedFiles(params);
        return MakeSuccessResponse(id, MakeResolveDroppedFilesResult(resolved));
    }

    // --- fs.probeMediaInfo (synchronous) ------------------------------------
    if (method == "fs.probeMediaInfo") {
        ProbeMediaInfoRequest p;
        std::string err;
        if (!ParseProbeMediaInfo(params, &p, &err)) {
            return MakeErrorResponse(id, "BAD_REQUEST", err);
        }
        // Best-effort: an unset provider (or any lookup failure inside it)
        // resolves the all-zero result rather than an error - only bad params
        // (handled above) can fail this method.
        const MediaInfoSnapshot snap =
            ctx.probe_media_info ? ctx.probe_media_info(p.file_path) : MediaInfoSnapshot{};
        return MakeSuccessResponse(id, MakeMediaInfoResult(snap));
    }

    return MakeErrorResponse(id, "UNKNOWN_METHOD", "Unknown method: " + method);
}

bool BuildQueryString(const json& value, std::string* out,
                      std::string* err_message) {
    out->clear();
    if (!value.is_object()) {
        // No method-name prefix here - the caller adds its own (see the header).
        *err_message = "'query' must be an object";
        return false;
    }
    std::string q;
    for (auto it = value.begin(); it != value.end(); ++it) {
        if (!it.value().is_string()) {
            *err_message = "'query' values must be strings";
            return false;
        }
        if (!q.empty()) {
            q.push_back('&');
        }
        q += UrlEncode(it.key());
        q.push_back('=');
        q += UrlEncode(it.value().get<std::string>());
    }
    *out = std::move(q);
    return true;
}

std::string AppendQueryToUrl(const std::string& url, const std::string& query) {
    if (query.empty()) {
        return url;  // contractual: an empty query never touches the URL
    }
    std::string out = url;
    out.push_back('?');
    out += query;
    return out;
}

bool ParseBackendRequest(const json& params, BackendRequest* out,
                         std::string* err_message) {
    if (!params.is_object()) {
        *err_message = "backend.request requires a params object";
        return false;
    }
    if (!params.contains("method") || !params["method"].is_string()) {
        *err_message = "backend.request requires a string 'method'";
        return false;
    }
    std::string method = params["method"].get<std::string>();
    if (method != "GET" && method != "POST" && method != "DELETE") {
        *err_message = "backend.request 'method' must be GET, POST or DELETE";
        return false;
    }
    if (!params.contains("path") || !params["path"].is_string()) {
        *err_message = "backend.request requires a string 'path'";
        return false;
    }
    std::string path = params["path"].get<std::string>();
    if (path.empty() || path.front() != '/') {
        *err_message = "backend.request 'path' must start with '/'";
        return false;
    }

    out->http_method = std::move(method);
    out->path = std::move(path);
    out->query.clear();
    out->has_body = false;
    out->body.clear();
    out->timeout_ms = kDefaultBackendTimeoutMs;

    if (params.contains("timeoutMs") && !params["timeoutMs"].is_null()) {
        if (!params["timeoutMs"].is_number()) {
            *err_message = "backend.request 'timeoutMs' must be a number";
            return false;
        }
        // Clamp on the double value first so an out-of-range (or overflowing)
        // request never wraps when narrowed to int. Truncates toward zero.
        const double v = params["timeoutMs"].get<double>();
        if (v < static_cast<double>(kMinBackendTimeoutMs)) {
            out->timeout_ms = kMinBackendTimeoutMs;
        } else if (v > static_cast<double>(kMaxBackendTimeoutMs)) {
            out->timeout_ms = kMaxBackendTimeoutMs;
        } else {
            out->timeout_ms = static_cast<int>(v);
        }
    }

    if (params.contains("query") && !params["query"].is_null()) {
        std::string q;
        std::string q_err;
        if (!BuildQueryString(params["query"], &q, &q_err)) {
            // Prepend the method name so the wording stays byte-identical to
            // the pre-refactor messages ("backend.request 'query' must be an
            // object" / "... values must be strings").
            *err_message = "backend.request " + q_err;
            return false;
        }
        out->query = std::move(q);
    }

    if (params.contains("body") && !params["body"].is_null()) {
        if (!params["body"].is_object() && !params["body"].is_array()) {
            *err_message = "backend.request 'body' must be a JSON object or array";
            return false;
        }
        out->has_body = true;
        out->body = params["body"].dump();
    }

    return true;
}

std::string BuildBackendUrl(const std::string& base_url, const BackendRequest& req) {
    // AppendQueryToUrl is the single place that knows "empty query => URL
    // unchanged"; this function is only the base_url + path concatenation.
    return AppendQueryToUrl(base_url + req.path, req.query);
}

json MakeBackendRequestResult(int status, const std::string& body_utf8) {
    json result;
    result["status"] = status;
    json body = json::parse(body_utf8, nullptr, /*allow_exceptions=*/false);
    if (body.is_discarded() || body_utf8.empty()) {
        result["body"] = nullptr;
    } else {
        result["body"] = std::move(body);
    }
    return result;
}

bool ParseDownloadVideo(const json& params, DownloadVideoRequest* out,
                        std::string* err_message) {
    if (!params.is_object()) {
        *err_message = "backend.downloadVideo requires a params object";
        return false;
    }
    if (!params.contains("jobId") || !params["jobId"].is_string() ||
        params["jobId"].get<std::string>().empty()) {
        *err_message = "backend.downloadVideo requires a non-empty string 'jobId'";
        return false;
    }
    std::string job_id = params["jobId"].get<std::string>();
    if (job_id.find('/') != std::string::npos || job_id.find('\\') != std::string::npos ||
        job_id.find("..") != std::string::npos) {
        *err_message = "backend.downloadVideo 'jobId' contains illegal path characters";
        return false;
    }
    bool joined = false;
    if (params.contains("joined")) {
        if (!params["joined"].is_boolean()) {
            *err_message = "backend.downloadVideo 'joined' must be a boolean";
            return false;
        }
        joined = params["joined"].get<bool>();
    }
    out->job_id = std::move(job_id);
    out->joined = joined;

    // Contract v6: destDir / fileName / noClobber. All three optional and
    // additive - leaving them all absent/null keeps has_dest_dir/has_file_name
    // false and no_clobber false, i.e. the exact pre-v6 defaults.
    out->has_dest_dir = false;
    out->dest_dir.clear();
    if (params.contains("destDir") && !params["destDir"].is_null()) {
        if (!params["destDir"].is_string() || params["destDir"].get<std::string>().empty()) {
            *err_message = "backend.downloadVideo 'destDir' must be a non-empty string";
            return false;
        }
        out->has_dest_dir = true;
        out->dest_dir = params["destDir"].get<std::string>();
    }
    out->has_file_name = false;
    out->file_name.clear();
    if (params.contains("fileName") && !params["fileName"].is_null()) {
        if (!params["fileName"].is_string() || params["fileName"].get<std::string>().empty()) {
            *err_message = "backend.downloadVideo 'fileName' must be a non-empty string";
            return false;
        }
        out->has_file_name = true;
        out->file_name = params["fileName"].get<std::string>();
    }
    out->no_clobber = false;
    if (params.contains("noClobber") && !params["noClobber"].is_null()) {
        if (!params["noClobber"].is_boolean()) {
            *err_message = "backend.downloadVideo 'noClobber' must be a boolean";
            return false;
        }
        out->no_clobber = params["noClobber"].get<bool>();
    }
    out->reuse_if_present = false;
    if (params.contains("reuseIfPresent") && !params["reuseIfPresent"].is_null()) {
        if (!params["reuseIfPresent"].is_boolean()) {
            *err_message = "backend.downloadVideo 'reuseIfPresent' must be a boolean";
            return false;
        }
        out->reuse_if_present = params["reuseIfPresent"].get<bool>();
    }
    return true;
}

std::string DownloadVideoPath(const DownloadVideoRequest& req) {
    std::string path = "/jobs/";
    path += req.job_id;
    path += req.joined ? "/joined" : "/video";
    return path;
}

std::string DownloadFileName(const DownloadVideoRequest& req) {
    std::string name = req.job_id;
    name += req.joined ? "_joined.mp4" : ".mp4";
    return name;
}

// ---------------------------------------------------------------------------
// backend.uploadFile
// ---------------------------------------------------------------------------

bool ParseUploadFile(const json& params, UploadFileRequest* out,
                     std::string* err_message) {
    if (!params.is_object()) {
        *err_message = "backend.uploadFile requires a params object";
        return false;
    }
    if (!params.contains("kind") || !params["kind"].is_string()) {
        *err_message = "backend.uploadFile requires a string 'kind'";
        return false;
    }
    const std::string kind = params["kind"].get<std::string>();
    if (kind != "image" && kind != "video" && kind != "audio") {
        *err_message = "backend.uploadFile 'kind' must be image, video or audio";
        return false;
    }
    if (!params.contains("filePath") || !params["filePath"].is_string() ||
        params["filePath"].get<std::string>().empty()) {
        *err_message = "backend.uploadFile requires a non-empty string 'filePath'";
        return false;
    }
    out->kind = kind;
    out->file_path = params["filePath"].get<std::string>();
    // Explicitly clear before the optional parse below (same discipline as
    // ParseBackendRequest), so a reused request object can never leak a query
    // from a previous call into an upload that did not ask for one.
    out->query.clear();

    // Contract v10: optional 'query'. Absent or null leaves it empty, which
    // makes the upload URL identical to the pre-v10 one (AppendQueryToUrl).
    if (params.contains("query") && !params["query"].is_null()) {
        std::string q;
        std::string q_err;
        if (!BuildQueryString(params["query"], &q, &q_err)) {
            *err_message = "backend.uploadFile " + q_err;
            return false;
        }
        out->query = std::move(q);
    }
    return true;
}

std::string UploadPath(const std::string& kind) {
    return "/upload/" + kind;
}

std::string FileNameFromPath(const std::string& path) {
    const size_t slash = path.find_last_of("/\\");
    if (slash == std::string::npos) {
        return path;
    }
    return path.substr(slash + 1);
}

std::string ContentTypeForExtension(const std::string& path) {
    const size_t dot = path.find_last_of('.');
    const size_t slash = path.find_last_of("/\\");
    // No extension, or the dot belongs to a directory component.
    if (dot == std::string::npos || (slash != std::string::npos && dot < slash)) {
        return "application/octet-stream";
    }
    std::string ext = path.substr(dot + 1);
    for (char& c : ext) {
        if (c >= 'A' && c <= 'Z') {
            c = static_cast<char>(c - 'A' + 'a');
        }
    }
    // Image
    if (ext == "png") return "image/png";
    if (ext == "jpg" || ext == "jpeg") return "image/jpeg";
    if (ext == "webp") return "image/webp";
    // Video
    if (ext == "mp4") return "video/mp4";
    if (ext == "mov") return "video/quicktime";
    if (ext == "webm") return "video/webm";
    if (ext == "mkv") return "video/x-matroska";
    // Audio
    if (ext == "wav") return "audio/wav";
    if (ext == "mp3") return "audio/mpeg";
    if (ext == "m4a") return "audio/mp4";
    if (ext == "aac") return "audio/aac";
    if (ext == "flac") return "audio/flac";
    if (ext == "ogg") return "audio/ogg";
    return "application/octet-stream";
}

std::string BuildMultipartHeader(const std::string& boundary,
                                 const std::string& field_name,
                                 const std::string& filename,
                                 const std::string& content_type) {
    std::string h = "--";
    h += boundary;
    h += "\r\n";
    h += "Content-Disposition: form-data; name=\"";
    h += field_name;
    h += "\"; filename=\"";
    h += filename;
    h += "\"\r\n";
    h += "Content-Type: ";
    h += content_type;
    h += "\r\n\r\n";
    return h;
}

std::string BuildMultipartFooter(const std::string& boundary) {
    std::string f = "\r\n--";
    f += boundary;
    f += "--\r\n";
    return f;
}

std::string MultipartContentType(const std::string& boundary) {
    return "multipart/form-data; boundary=" + boundary;
}

// ---------------------------------------------------------------------------
// timeline.captureFrame
// ---------------------------------------------------------------------------

bool ParseCaptureFrame(const json& params, CaptureFrameRequest* out,
                       std::string* err_message) {
    out->has_frame = false;
    out->frame = 0;
    if (!params.is_object()) {
        // An absent params object is allowed: capture the current cursor frame.
        return true;
    }
    if (params.contains("frame") && !params["frame"].is_null()) {
        if (!params["frame"].is_number_integer()) {
            *err_message = "timeline.captureFrame 'frame' must be an integer";
            return false;
        }
        out->has_frame = true;
        out->frame = params["frame"].get<int>();
    }
    return true;
}

std::string CaptureFileName(int frame, const std::string& unique) {
    std::string name = "frame_";
    name += std::to_string(frame);
    name += "_";
    name += unique;
    name += ".png";
    return name;
}

// ---------------------------------------------------------------------------
// ui.pickFile
// ---------------------------------------------------------------------------

bool ParsePickFileKind(const json& params, std::string* kind_out,
                       std::string* err_message) {
    if (!params.is_object()) {
        *err_message = "ui.pickFile requires a params object";
        return false;
    }
    if (!params.contains("kind") || !params["kind"].is_string()) {
        *err_message = "ui.pickFile requires a string 'kind'";
        return false;
    }
    const std::string kind = params["kind"].get<std::string>();
    if (kind != "image" && kind != "video" && kind != "audio" && kind != "imageOrVideo") {
        *err_message = "ui.pickFile 'kind' must be image, video, audio or imageOrVideo";
        return false;
    }
    *kind_out = kind;
    return true;
}

std::vector<PickFileFilterEntry> PickFileFilter(const std::string& kind) {
    std::vector<PickFileFilterEntry> entries;
    // Patterns mirror the backend upload allow-list (see ContentTypeForExtension).
    if (kind == "image") {
        entries.push_back(
            {"Image files (*.png;*.jpg;*.jpeg;*.webp)", "*.png;*.jpg;*.jpeg;*.webp"});
    } else if (kind == "video") {
        entries.push_back(
            {"Video files (*.mp4;*.mov;*.webm;*.mkv)", "*.mp4;*.mov;*.webm;*.mkv"});
    } else if (kind == "audio") {
        entries.push_back({"Audio files (*.wav;*.mp3;*.m4a;*.aac;*.flac;*.ogg)",
                           "*.wav;*.mp3;*.m4a;*.aac;*.flac;*.ogg"});
    } else if (kind == "imageOrVideo") {
        // Chain's unified source-input picker (task brief "Chainのソース入力欄
        // 一本化"): a combined filter leads so both image and video files show
        // by default, followed by single-type filters for anyone who wants to
        // narrow it, same extension lists as "image"/"video" above.
        entries.push_back({"Image/Video files (*.png;*.jpg;*.jpeg;*.webp;*.mp4;*.mov;*.webm;*.mkv)",
                           "*.png;*.jpg;*.jpeg;*.webp;*.mp4;*.mov;*.webm;*.mkv"});
        entries.push_back(
            {"Image files (*.png;*.jpg;*.jpeg;*.webp)", "*.png;*.jpg;*.jpeg;*.webp"});
        entries.push_back(
            {"Video files (*.mp4;*.mov;*.webm;*.mkv)", "*.mp4;*.mov;*.webm;*.mkv"});
    }
    entries.push_back({"All files (*.*)", "*.*"});
    return entries;
}

// ---------------------------------------------------------------------------
// ui.makeThumbnail
// ---------------------------------------------------------------------------

bool ParseMakeThumbnail(const json& params, MakeThumbnailRequest* out,
                        std::string* err_message) {
    if (!params.is_object()) {
        *err_message = "ui.makeThumbnail requires a params object";
        return false;
    }
    if (!params.contains("filePath") || !params["filePath"].is_string() ||
        params["filePath"].get<std::string>().empty()) {
        *err_message = "ui.makeThumbnail requires a non-empty string 'filePath'";
        return false;
    }
    int max_dim = kDefaultThumbnailMaxDim;
    if (params.contains("maxDim") && !params["maxDim"].is_null()) {
        if (!params["maxDim"].is_number()) {
            *err_message = "ui.makeThumbnail 'maxDim' must be a number";
            return false;
        }
        // Truncate toward zero, then clamp into the supported range.
        max_dim = static_cast<int>(params["maxDim"].get<double>());
        if (max_dim < kMinThumbnailMaxDim) {
            max_dim = kMinThumbnailMaxDim;
        } else if (max_dim > kMaxThumbnailMaxDim) {
            max_dim = kMaxThumbnailMaxDim;
        }
    }
    out->file_path = params["filePath"].get<std::string>();
    out->max_dim = max_dim;
    return true;
}

void ComputeThumbnailSize(int src_w, int src_h, int max_dim, int* out_w,
                          int* out_h) {
    if (src_w <= 0 || src_h <= 0 || max_dim <= 0) {
        *out_w = 0;
        *out_h = 0;
        return;
    }
    const int longest = src_w >= src_h ? src_w : src_h;
    if (longest <= max_dim) {
        // Already small enough: keep the source dimensions (1:1, no upscale).
        *out_w = src_w;
        *out_h = src_h;
        return;
    }
    // Scale the long edge down to max_dim and derive the short edge, rounding to
    // the nearest pixel and never dropping below 1.
    auto scaled = [max_dim, longest](int side) -> int {
        const long long v =
            (static_cast<long long>(side) * max_dim + longest / 2) / longest;
        return v < 1 ? 1 : static_cast<int>(v);
    };
    if (src_w >= src_h) {
        *out_w = max_dim;
        *out_h = scaled(src_h);
    } else {
        *out_w = scaled(src_w);
        *out_h = max_dim;
    }
}

std::string Base64Encode(const unsigned char* data, size_t len) {
    static const char kAlphabet[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    if (data == nullptr || len == 0) {
        return out;
    }
    out.reserve(((len + 2) / 3) * 4);
    size_t i = 0;
    for (; i + 3 <= len; i += 3) {
        const unsigned n = (static_cast<unsigned>(data[i]) << 16) |
                           (static_cast<unsigned>(data[i + 1]) << 8) |
                           static_cast<unsigned>(data[i + 2]);
        out.push_back(kAlphabet[(n >> 18) & 0x3F]);
        out.push_back(kAlphabet[(n >> 12) & 0x3F]);
        out.push_back(kAlphabet[(n >> 6) & 0x3F]);
        out.push_back(kAlphabet[n & 0x3F]);
    }
    const size_t rem = len - i;
    if (rem == 1) {
        const unsigned n = static_cast<unsigned>(data[i]) << 16;
        out.push_back(kAlphabet[(n >> 18) & 0x3F]);
        out.push_back(kAlphabet[(n >> 12) & 0x3F]);
        out.push_back('=');
        out.push_back('=');
    } else if (rem == 2) {
        const unsigned n = (static_cast<unsigned>(data[i]) << 16) |
                           (static_cast<unsigned>(data[i + 1]) << 8);
        out.push_back(kAlphabet[(n >> 18) & 0x3F]);
        out.push_back(kAlphabet[(n >> 12) & 0x3F]);
        out.push_back(kAlphabet[(n >> 6) & 0x3F]);
        out.push_back('=');
    }
    return out;
}

std::string MakeDataUrl(const std::string& mime, const std::string& base64) {
    std::string url = "data:";
    url += mime;
    url += ";base64,";
    url += base64;
    return url;
}

// ---------------------------------------------------------------------------
// timeline.getSelection
// ---------------------------------------------------------------------------

json MakeSelectionResult(const SelectionSnapshot& snap) {
    json selected = json::array();
    for (const SelectionItem& it : snap.selected) {
        json e;
        e["layer"] = it.layer;
        e["frameStart"] = it.frame_start;
        e["frameEnd"] = it.frame_end;
        e["effectName"] = it.effect_name;
        e["filePath"] = it.has_file_path ? json(it.file_path) : json(nullptr);
        e["objectName"] = it.has_object_name ? json(it.object_name) : json(nullptr);
        e["textContent"] = it.has_text_content ? json(it.text_content) : json(nullptr);
        // Not nullable: 0 means "unknown resolution" (webui treats 0 as null).
        e["mediaWidth"] = it.media_width;
        e["mediaHeight"] = it.media_height;
        // Not nullable either: 0 means "unknown duration" (still image / no
        // audio / lookup failure); the webui treats 0 as unknown.
        e["mediaDurationSec"] = it.media_duration_sec;
        // Contract v11 (material fps, section 3-13): the material's own frame
        // rate, RAW (29.97 stays 29.97 - the webui does the integer snap). Not
        // nullable either; 0 means "unknown", which 0 can safely mean because
        // it is never a real frame rate.
        e["mediaFps"] = it.media_fps;
        // Contract v10 (source trim, section 1-6). None of these are nullable:
        // hasPlaybackRange is the explicit "was it really read" flag, and the
        // other three carry conservative defaults (neutral speed, no loop, one
        // section) so an object we could not fully inspect still lands on the
        // untrimmed, pre-v10 upload path in the webui.
        e["playbackStartSec"] = it.playback_start_sec;
        e["playbackEndSec"] = it.playback_end_sec;
        e["hasPlaybackRange"] = it.has_playback_range;
        e["playbackSpeed"] = it.playback_speed;
        e["loopPlay"] = it.loop_play;
        e["sectionCount"] = it.section_count;
        selected.push_back(std::move(e));
    }
    json result;
    result["hasRange"] = snap.has_range;
    result["rangeStart"] = snap.range_start;
    result["rangeEnd"] = snap.range_end;
    result["selected"] = std::move(selected);
    result["cursorFrame"] = snap.cursor_frame;
    result["cursorLayer"] = snap.cursor_layer;
    result["rate"] = snap.rate;
    result["scale"] = snap.scale;
    result["sampleRate"] = snap.sample_rate;
    return result;
}

bool ParsePlaybackRange(const std::string& raw, double* start_sec, double* end_sec) {
    if (start_sec == nullptr || end_sec == nullptr) {
        return false;
    }
    // Take exactly the first two comma-separated fields; everything after the
    // second comma (the localized mode name and whatever follows it) is ignored
    // by design - see the header comment.
    const size_t c1 = raw.find(',');
    if (c1 == std::string::npos) {
        return false;  // fewer than two fields: not a range we understand
    }
    size_t c2 = raw.find(',', c1 + 1);
    if (c2 == std::string::npos) {
        c2 = raw.size();  // exactly two fields is fine too
    }
    double start = 0.0;
    double end = 0.0;
    if (!ParseWholeDoubleField(raw.substr(0, c1), &start)) {
        return false;
    }
    if (!ParseWholeDoubleField(raw.substr(c1 + 1, c2 - (c1 + 1)), &end)) {
        return false;
    }
    if (start < 0.0 || end < start) {
        return false;
    }
    *start_sec = start;
    *end_sec = end;
    return true;
}

bool ParsePlaybackSpeedPercent(const std::string& raw, double* speed) {
    if (speed == nullptr) {
        return false;
    }
    double percent = 0.0;
    if (!ParseWholeDoubleField(raw, &percent)) {
        return false;
    }
    *speed = percent / 100.0;
    return true;
}

// ---------------------------------------------------------------------------
// Provisional placement (I3) - pure geometry + shared param parsing
// ---------------------------------------------------------------------------

PlacementResolveResult ResolveProvisionalPlacement(const PlacementResolveInput& in) {
    PlacementResolveResult r;
    switch (in.placement) {
        case ProvisionalPlacement::kAfterMaterial:
            // (A) directly after the material. material_frame_end is INCLUSIVE
            // (the object's last covered frame): the next free frame is end+1,
            // matching FindObjectByJob's next=end+1 scan step (spec 8 #7).
            if (in.material_layer < 0 || in.material_frame_end < 0) {
                r.err = "placement \"A\" requires materialLayer and materialFrameEnd >= 0";
                return r;
            }
            r.layer = in.material_layer;
            r.frame = in.material_frame_end + 1;
            r.ok = true;
            return r;
        case ProvisionalPlacement::kSameStartFront:
            // (B) same start frame, on the guaranteed-empty frontmost layer.
            if (in.material_frame_start < 0) {
                r.err = "placement \"B\" requires materialFrameStart >= 0";
                return r;
            }
            if (in.layer_max < 0) {
                r.err = "placement \"B\" requires a valid layer_max";
                return r;
            }
            r.layer = in.layer_max + 1;
            r.frame = in.material_frame_start;
            r.ok = true;
            return r;
        case ProvisionalPlacement::kCursor:
            // (C) the right-click cursor position.
            if (in.cursor_layer < 0 || in.cursor_frame < 0) {
                r.err = "placement \"C\" requires cursorLayer and cursorFrame >= 0";
                return r;
            }
            r.layer = in.cursor_layer;
            r.frame = in.cursor_frame;
            r.ok = true;
            return r;
        case ProvisionalPlacement::kLegacy:
        default:
            r.err = "placement resolver called with no placement (kLegacy)";
            return r;
    }
}

namespace {

// Shared placeholder for the optional I3 placement + material_*/cursor_* fields.
struct PlacementFields {
    ProvisionalPlacement placement = ProvisionalPlacement::kLegacy;
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

// Parse one optional integer field. Absent/null leaves *has false; a present
// non-integer is a parse failure.
bool ParseOptInt(const json& params, const char* key, const char* method, bool* has,
                 int* out, std::string* err) {
    if (params.contains(key) && !params[key].is_null()) {
        if (!params[key].is_number_integer()) {
            *err = std::string(method) + " '" + key + "' must be an integer";
            return false;
        }
        *has = true;
        *out = params[key].get<int>();
    }
    return true;
}

// Parse the optional I3 placement + material_*/cursor_* fields. When
// placement_required is true, an absent 'placement' is rejected; otherwise an
// absent 'placement' leaves out->placement at kLegacy (the caller then uses the
// explicit layer/frame). A present-but-invalid 'placement' string is rejected.
bool ParsePlacementFields(const json& params, const char* method,
                          bool placement_required, PlacementFields* out,
                          std::string* err) {
    if (params.contains("placement") && !params["placement"].is_null()) {
        if (!params["placement"].is_string()) {
            *err = std::string(method) + " 'placement' must be a string";
            return false;
        }
        const std::string pv = params["placement"].get<std::string>();
        if (pv == "A") {
            out->placement = ProvisionalPlacement::kAfterMaterial;
        } else if (pv == "B") {
            out->placement = ProvisionalPlacement::kSameStartFront;
        } else if (pv == "C") {
            out->placement = ProvisionalPlacement::kCursor;
        } else {
            *err = std::string(method) + " 'placement' must be \"A\", \"B\" or \"C\"";
            return false;
        }
    } else if (placement_required) {
        *err = std::string(method) + " requires a 'placement' of \"A\", \"B\" or \"C\"";
        return false;
    }
    return ParseOptInt(params, "materialLayer", method, &out->has_material_layer,
                       &out->material_layer, err) &&
           ParseOptInt(params, "materialFrameStart", method,
                       &out->has_material_frame_start, &out->material_frame_start, err) &&
           ParseOptInt(params, "materialFrameEnd", method,
                       &out->has_material_frame_end, &out->material_frame_end, err) &&
           ParseOptInt(params, "cursorLayer", method, &out->has_cursor_layer,
                       &out->cursor_layer, err) &&
           ParseOptInt(params, "cursorFrame", method, &out->has_cursor_frame,
                       &out->cursor_frame, err);
}

// Copy the shared placement fields onto whichever params struct owns them.
template <typename T>
void ApplyPlacementFields(const PlacementFields& pf, T* out) {
    out->placement = pf.placement;
    out->has_material_layer = pf.has_material_layer;
    out->material_layer = pf.material_layer;
    out->has_material_frame_start = pf.has_material_frame_start;
    out->material_frame_start = pf.material_frame_start;
    out->has_material_frame_end = pf.has_material_frame_end;
    out->material_frame_end = pf.material_frame_end;
    out->has_cursor_layer = pf.has_cursor_layer;
    out->cursor_layer = pf.cursor_layer;
    out->has_cursor_frame = pf.has_cursor_frame;
    out->cursor_frame = pf.cursor_frame;
}

}  // namespace

// ---------------------------------------------------------------------------
// timeline.insertProvisional
// ---------------------------------------------------------------------------

bool ParseInsertProvisional(const json& params, InsertProvisionalParams* out,
                            std::string* err_message) {
    if (!params.is_object()) {
        *err_message = "timeline.insertProvisional requires a params object";
        return false;
    }
    if (!params.contains("jobId") || !params["jobId"].is_string() ||
        params["jobId"].get<std::string>().empty()) {
        *err_message = "timeline.insertProvisional requires a non-empty string 'jobId'";
        return false;
    }
    if (!params.contains("displayText") || !params["displayText"].is_string()) {
        *err_message = "timeline.insertProvisional requires a string 'displayText'";
        return false;
    }
    out->job_id = params["jobId"].get<std::string>();
    out->display_text = params["displayText"].get<std::string>();

    // Optional textPrefix (spec section 5-5): absent/null -> the default
    // "generating:" prefix; a present non-string is a parse failure.
    out->text_prefix.clear();
    if (params.contains("textPrefix") && !params["textPrefix"].is_null()) {
        if (!params["textPrefix"].is_string()) {
            *err_message = "timeline.insertProvisional 'textPrefix' must be a string";
            return false;
        }
        out->text_prefix = params["textPrefix"].get<std::string>();
    }

    // Length: numFrames (positive int) + genFps (positive number), both
    // required. genFps<=0 is rejected here so it never reaches the native
    // ProjectFramesForPixels conversion.
    if (!params.contains("numFrames") || !params["numFrames"].is_number_integer() ||
        params["numFrames"].get<int>() <= 0) {
        *err_message =
            "timeline.insertProvisional requires a positive integer 'numFrames'";
        return false;
    }
    out->num_frames = params["numFrames"].get<int>();
    if (!params.contains("genFps") || !params["genFps"].is_number()) {
        *err_message =
            "timeline.insertProvisional requires a numeric 'genFps'";
        return false;
    }
    out->gen_fps = params["genFps"].get<double>();
    if (out->gen_fps <= 0.0) {
        *err_message = "timeline.insertProvisional 'genFps' must be > 0";
        return false;
    }

    // Position: placement system (section 5-9), required.
    PlacementFields pf;
    if (!ParsePlacementFields(params, "timeline.insertProvisional",
                              /*placement_required=*/true, &pf, err_message)) {
        return false;
    }
    ApplyPlacementFields(pf, out);
    return true;
}

ProvisionalTextAlias BuildProvisionalPlaceholder(const std::string& job_id,
                                                 const std::string& display_text,
                                                 int length_frames,
                                                 const std::string& text_prefix) {
    ProvisionalTextAlias built =
        BuildProvisionalTextAlias(display_text, job_id, text_prefix);
    built.alias = NormalizeAliasObjectFrameHeader(built.alias, length_frames);
    return built;
}

// ---------------------------------------------------------------------------
// timeline.resolveProvisional
// ---------------------------------------------------------------------------

bool ParseResolveProvisional(const json& params, ResolveProvisionalParams* out,
                             std::string* err_message) {
    if (!params.is_object()) {
        *err_message = "timeline.resolveProvisional requires a params object";
        return false;
    }
    if (!params.contains("jobId") || !params["jobId"].is_string() ||
        params["jobId"].get<std::string>().empty()) {
        *err_message = "timeline.resolveProvisional requires a non-empty string 'jobId'";
        return false;
    }
    if (!params.contains("videoFilePath") || !params["videoFilePath"].is_string() ||
        params["videoFilePath"].get<std::string>().empty()) {
        *err_message =
            "timeline.resolveProvisional requires a non-empty string 'videoFilePath'";
        return false;
    }
    if (!params.contains("reservedLayer") ||
        !params["reservedLayer"].is_number_integer()) {
        *err_message =
            "timeline.resolveProvisional requires an integer 'reservedLayer'";
        return false;
    }
    if (!params.contains("reservedFrame") ||
        !params["reservedFrame"].is_number_integer()) {
        *err_message =
            "timeline.resolveProvisional requires an integer 'reservedFrame'";
        return false;
    }
    if (!params.contains("lengthFrames") || !params["lengthFrames"].is_number_integer() ||
        params["lengthFrames"].get<int>() <= 0) {
        *err_message =
            "timeline.resolveProvisional requires a positive integer 'lengthFrames'";
        return false;
    }
    out->job_id = params["jobId"].get<std::string>();
    out->video_file_path = params["videoFilePath"].get<std::string>();
    out->reserved_layer = params["reservedLayer"].get<int>();
    out->reserved_frame = params["reservedFrame"].get<int>();
    out->length_frames = params["lengthFrames"].get<int>();
    return true;
}

// ---------------------------------------------------------------------------
// timeline.updateProvisionalText
// ---------------------------------------------------------------------------

bool ParseUpdateProvisionalText(const json& params, UpdateProvisionalTextParams* out,
                                std::string* err_message) {
    if (!params.is_object()) {
        *err_message = "timeline.updateProvisionalText requires a params object";
        return false;
    }
    if (!params.contains("jobId") || !params["jobId"].is_string() ||
        params["jobId"].get<std::string>().empty()) {
        *err_message =
            "timeline.updateProvisionalText requires a non-empty string 'jobId'";
        return false;
    }
    if (!params.contains("layer") || !params["layer"].is_number_integer()) {
        *err_message = "timeline.updateProvisionalText requires an integer 'layer'";
        return false;
    }
    if (!params.contains("frame") || !params["frame"].is_number_integer()) {
        *err_message = "timeline.updateProvisionalText requires an integer 'frame'";
        return false;
    }
    if (!params.contains("text") || !params["text"].is_string()) {
        *err_message = "timeline.updateProvisionalText requires a string 'text'";
        return false;
    }
    out->job_id = params["jobId"].get<std::string>();
    out->layer = params["layer"].get<int>();
    out->frame = params["frame"].get<int>();
    out->text = params["text"].get<std::string>();
    return true;
}

// ---------------------------------------------------------------------------
// timeline.updateProvisionalReservation (I3)
// ---------------------------------------------------------------------------

bool ParseUpdateProvisionalReservation(const json& params,
                                       UpdateProvisionalReservationParams* out,
                                       std::string* err_message) {
    const char* kM = "timeline.updateProvisionalReservation";
    if (!params.is_object()) {
        *err_message = std::string(kM) + " requires a params object";
        return false;
    }
    // oldJobId: optional string, empty allowed (skip the delete search).
    out->old_job_id.clear();
    if (params.contains("oldJobId") && !params["oldJobId"].is_null()) {
        if (!params["oldJobId"].is_string()) {
            *err_message = std::string(kM) + " 'oldJobId' must be a string";
            return false;
        }
        out->old_job_id = params["oldJobId"].get<std::string>();
    }
    if (!params.contains("newJobId") || !params["newJobId"].is_string() ||
        params["newJobId"].get<std::string>().empty()) {
        *err_message = std::string(kM) + " requires a non-empty string 'newJobId'";
        return false;
    }
    out->new_job_id = params["newJobId"].get<std::string>();
    if (!params.contains("displayText") || !params["displayText"].is_string()) {
        *err_message = std::string(kM) + " requires a string 'displayText'";
        return false;
    }
    out->display_text = params["displayText"].get<std::string>();
    // Optional textPrefix (spec section 5-5), same rules as insertProvisional.
    out->text_prefix.clear();
    if (params.contains("textPrefix") && !params["textPrefix"].is_null()) {
        if (!params["textPrefix"].is_string()) {
            *err_message = std::string(kM) + " 'textPrefix' must be a string";
            return false;
        }
        out->text_prefix = params["textPrefix"].get<std::string>();
    }
    if (!params.contains("numFrames") || !params["numFrames"].is_number_integer() ||
        params["numFrames"].get<int>() <= 0) {
        *err_message = std::string(kM) + " requires a positive integer 'numFrames'";
        return false;
    }
    out->num_frames = params["numFrames"].get<int>();
    if (!params.contains("genFps") || !params["genFps"].is_number()) {
        *err_message = std::string(kM) + " requires a numeric 'genFps'";
        return false;
    }
    out->gen_fps = params["genFps"].get<double>();
    if (out->gen_fps <= 0.0) {
        *err_message = std::string(kM) + " 'genFps' must be > 0";
        return false;
    }
    PlacementFields pf;
    if (!ParsePlacementFields(params, kM, /*placement_required=*/true, &pf,
                              err_message)) {
        return false;
    }
    ApplyPlacementFields(pf, out);
    return true;
}

// ---------------------------------------------------------------------------
// timeline.deleteProvisionalByJob (spec 5-10)
// ---------------------------------------------------------------------------

bool ParseDeleteProvisionalByJob(const json& params, DeleteProvisionalByJobParams* out,
                                 std::string* err_message) {
    if (!params.is_object()) {
        *err_message = "timeline.deleteProvisionalByJob requires a params object";
        return false;
    }
    if (!params.contains("jobId") || !params["jobId"].is_string() ||
        params["jobId"].get<std::string>().empty()) {
        *err_message =
            "timeline.deleteProvisionalByJob requires a non-empty string 'jobId'";
        return false;
    }
    out->job_id = params["jobId"].get<std::string>();
    return true;
}

// ---------------------------------------------------------------------------
// timeline.insertMediaForJob (replace-insert)
// ---------------------------------------------------------------------------

bool ParseInsertMediaForJob(const json& params, InsertMediaForJobParams* out,
                            std::string* err_message) {
    if (!params.is_object()) {
        *err_message = "timeline.insertMediaForJob requires a params object";
        return false;
    }
    if (!params.contains("jobId") || !params["jobId"].is_string() ||
        params["jobId"].get<std::string>().empty()) {
        *err_message =
            "timeline.insertMediaForJob requires a non-empty string 'jobId'";
        return false;
    }
    if (!params.contains("filePath") || !params["filePath"].is_string() ||
        params["filePath"].get<std::string>().empty()) {
        *err_message =
            "timeline.insertMediaForJob requires a non-empty string 'filePath'";
        return false;
    }
    out->job_id = params["jobId"].get<std::string>();
    out->file_path = params["filePath"].get<std::string>();
    return true;
}

// ---------------------------------------------------------------------------
// timeline.cutoutRange / timeline.extractAudio (async - Parse/Make only)
// ---------------------------------------------------------------------------

namespace {

// Shared validation for the range-bounce methods' common fields. Fills the
// caller-owned range fields; the caller validates its method-specific extras.
bool ParseRangeCommon(const json& params, const char* method, int* layer,
                      int* frame_start, int* frame_count, std::string* audio_mode,
                      std::vector<int>* solo_keep_layers, std::string* err_message) {
    if (!params.is_object()) {
        *err_message = std::string(method) + " requires a params object";
        return false;
    }
    if (!params.contains("layer") || !params["layer"].is_number_integer()) {
        *err_message = std::string(method) + " requires an integer 'layer'";
        return false;
    }
    if (!params.contains("frameStart") || !params["frameStart"].is_number_integer()) {
        *err_message = std::string(method) + " requires an integer 'frameStart'";
        return false;
    }
    if (!params.contains("frameCount") || !params["frameCount"].is_number_integer() ||
        params["frameCount"].get<int>() <= 0) {
        *err_message = std::string(method) + " requires a positive integer 'frameCount'";
        return false;
    }
    if (!params.contains("audioMode") || !params["audioMode"].is_string()) {
        *err_message = std::string(method) + " requires a string 'audioMode'";
        return false;
    }
    const std::string mode = params["audioMode"].get<std::string>();
    if (mode != "mix" && mode != "solo") {
        *err_message = std::string(method) + " 'audioMode' must be \"mix\" or \"solo\"";
        return false;
    }
    if (!params.contains("soloKeepLayers") || !params["soloKeepLayers"].is_array()) {
        *err_message = std::string(method) + " requires an array 'soloKeepLayers'";
        return false;
    }
    std::vector<int> layers;
    for (const json& v : params["soloKeepLayers"]) {
        if (!v.is_number_integer()) {
            *err_message =
                std::string(method) + " 'soloKeepLayers' values must be integers";
            return false;
        }
        layers.push_back(v.get<int>());
    }
    *layer = params["layer"].get<int>();
    *frame_start = params["frameStart"].get<int>();
    *frame_count = params["frameCount"].get<int>();
    *audio_mode = mode;
    *solo_keep_layers = std::move(layers);
    return true;
}

}  // namespace

bool ParseCutoutRange(const json& params, CutoutRangeRequest* out,
                      std::string* err_message) {
    if (!ParseRangeCommon(params, "timeline.cutoutRange", &out->layer,
                          &out->frame_start, &out->frame_count, &out->audio_mode,
                          &out->solo_keep_layers, err_message)) {
        return false;
    }
    if (!params.contains("withAudio") || !params["withAudio"].is_boolean()) {
        *err_message = "timeline.cutoutRange requires a boolean 'withAudio'";
        return false;
    }
    out->with_audio = params["withAudio"].get<bool>();
    return true;
}

json MakeCutoutResult(const std::string& file_path, int width, int height,
                      int frame_count, bool has_audio) {
    json result;
    result["filePath"] = file_path;
    result["width"] = width;
    result["height"] = height;
    result["frameCount"] = frame_count;
    result["hasAudio"] = has_audio;
    return result;
}

bool ParseExtractAudio(const json& params, ExtractAudioRequest* out,
                       std::string* err_message) {
    return ParseRangeCommon(params, "timeline.extractAudio", &out->layer,
                            &out->frame_start, &out->frame_count, &out->audio_mode,
                            &out->solo_keep_layers, err_message);
}

json MakeExtractAudioResult(const std::string& file_path, double duration_sec,
                            int sample_rate, bool has_audio_stream) {
    json result;
    result["filePath"] = file_path;
    result["durationSec"] = duration_sec;
    result["sampleRate"] = sample_rate;
    result["hasAudioStream"] = has_audio_stream;
    return result;
}

bool IsAudioStreamPresent(int sample_count, double peak_abs_amplitude) {
    return sample_count > 0 && peak_abs_amplitude >= kAudioSilenceThreshold;
}

// ---------------------------------------------------------------------------
// ui.pickFolder
// ---------------------------------------------------------------------------

bool ParsePickFolder(const json& params, PickFolderRequest* out, std::string* err_message) {
    out->has_title = false;
    out->title.clear();
    if (!params.is_object()) {
        // An absent params object is allowed: use native's default title.
        return true;
    }
    if (params.contains("title") && !params["title"].is_null()) {
        if (!params["title"].is_string()) {
            *err_message = "ui.pickFolder 'title' must be a string";
            return false;
        }
        out->has_title = true;
        out->title = params["title"].get<std::string>();
    }
    return true;
}

// ---------------------------------------------------------------------------
// fs.listFiles
// ---------------------------------------------------------------------------

bool ParseListFiles(const json& params, ListFilesRequest* out, std::string* err_message) {
    if (!params.is_object()) {
        *err_message = "fs.listFiles requires a params object";
        return false;
    }
    if (!params.contains("folderPath") || !params["folderPath"].is_string() ||
        params["folderPath"].get<std::string>().empty()) {
        *err_message = "fs.listFiles requires a non-empty string 'folderPath'";
        return false;
    }
    out->folder_path = params["folderPath"].get<std::string>();

    out->extensions.clear();
    if (params.contains("extensions") && !params["extensions"].is_null()) {
        if (!params["extensions"].is_array()) {
            *err_message = "fs.listFiles 'extensions' must be an array of strings";
            return false;
        }
        for (const json& v : params["extensions"]) {
            if (!v.is_string()) {
                *err_message = "fs.listFiles 'extensions' values must be strings";
                return false;
            }
            out->extensions.push_back(v.get<std::string>());
        }
    }

    out->with_audio_duration = false;
    if (params.contains("withAudioDuration") && !params["withAudioDuration"].is_null()) {
        if (!params["withAudioDuration"].is_boolean()) {
            *err_message = "fs.listFiles 'withAudioDuration' must be a boolean";
            return false;
        }
        out->with_audio_duration = params["withAudioDuration"].get<bool>();
    }
    return true;
}

json MakeListFilesResult(const std::vector<ListedFile>& files) {
    json arr = json::array();
    for (const ListedFile& f : files) {
        json e;
        e["name"] = f.name;
        e["path"] = f.path;
        e["sizeBytes"] = f.size_bytes;
        e["mtimeMs"] = f.mtime_ms;
        e["durationSec"] = f.duration_sec;
        arr.push_back(std::move(e));
    }
    json result;
    result["files"] = std::move(arr);
    return result;
}

// ---------------------------------------------------------------------------
// fs.probeAudioDuration
// ---------------------------------------------------------------------------

bool ParseProbeAudioDuration(const json& params, ProbeAudioDurationRequest* out,
                             std::string* err_message) {
    if (!params.is_object()) {
        *err_message = "fs.probeAudioDuration requires a params object";
        return false;
    }
    if (!params.contains("filePath") || !params["filePath"].is_string() ||
        params["filePath"].get<std::string>().empty()) {
        *err_message = "fs.probeAudioDuration requires a non-empty string 'filePath'";
        return false;
    }
    out->file_path = params["filePath"].get<std::string>();
    return true;
}

// ---------------------------------------------------------------------------
// fs.probeMediaInfo
// ---------------------------------------------------------------------------

bool ParseProbeMediaInfo(const json& params, ProbeMediaInfoRequest* out,
                         std::string* err_message) {
    if (!params.is_object()) {
        *err_message = "fs.probeMediaInfo requires a params object";
        return false;
    }
    if (!params.contains("filePath") || !params["filePath"].is_string() ||
        params["filePath"].get<std::string>().empty()) {
        *err_message = "fs.probeMediaInfo requires a non-empty string 'filePath'";
        return false;
    }
    out->file_path = params["filePath"].get<std::string>();
    return true;
}

json MakeMediaInfoResult(const MediaInfoSnapshot& snap) {
    json result;
    // An unavailable snapshot is reported as all-zero (never an error); the
    // webui treats 0 as "unknown".
    result["durationSec"] = snap.available ? snap.duration_sec : 0.0;
    result["width"] = snap.available ? snap.width : 0;
    result["height"] = snap.available ? snap.height : 0;
    return result;
}

// ---------------------------------------------------------------------------
// ui.resolveDroppedFiles
// ---------------------------------------------------------------------------

ResolveDroppedFilesRequest ParseResolveDroppedFiles(const json& params) {
    ResolveDroppedFilesRequest out;
    if (!params.is_object() || !params.contains("__droppedPaths") ||
        !params["__droppedPaths"].is_array()) {
        return out;  // no key / not an array -> "nothing was dropped"
    }
    for (const json& v : params["__droppedPaths"]) {
        if (v.is_string() && !v.get<std::string>().empty()) {
            out.file_paths.push_back(v.get<std::string>());
        }
        // Non-string / empty-string entries are silently skipped rather than
        // rejected - see ParseResolveDroppedFiles's header comment.
    }
    return out;
}

json MakeResolveDroppedFilesResult(const ResolveDroppedFilesRequest& req) {
    json arr = json::array();
    for (const std::string& path : req.file_paths) {
        json e;
        e["filePath"] = path;
        e["fileName"] = FileNameFromPath(path);
        arr.push_back(std::move(e));
    }
    json result;
    result["files"] = std::move(arr);
    return result;
}

std::string InjectDroppedPathsIntoRequest(const std::string& request_json,
                                          const std::vector<std::string>& paths) {
    json req = json::parse(request_json, nullptr, /*allow_exceptions=*/false);
    if (req.is_discarded() || !req.is_object()) {
        return request_json;  // malformed / non-object - the only passthrough case
    }
    if (!req.contains("params") || !req["params"].is_object()) {
        req["params"] = json::object();  // e.g. {} or a missing 'params' entirely
    }
    json arr = json::array();
    for (const std::string& p : paths) {
        arr.push_back(p);
    }
    req["params"]["__droppedPaths"] = std::move(arr);  // always force-set, even if empty
    return req.dump();
}

}  // namespace nzvideomni
