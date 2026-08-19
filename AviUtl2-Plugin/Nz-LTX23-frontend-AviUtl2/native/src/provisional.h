// provisional.h - pure provisional-object state machine for Nz-LTX23.
//
// A "provisional object" is the placeholder text object dropped on the AviUtl2
// timeline while a generation job runs (see alias_util::BuildProvisionalTextAlias).
// We deliberately DO NOT keep an OBJECT_HANDLE for it: handles go stale across
// undo/redo, project reload and user edits. Instead every provisional object is
// tagged with its job id, and we re-discover it later by an EXACT job-id match in
// the object's alias text - the proven pattern from the reference implementation
// (FindObjectCoveringFrameByExactAlias). This module holds that discovery /
// resolve / orphan logic as pure string+integer code: no AviUtl2 SDK, no handles.
//
// The tag format is dictated by alias_util::BuildProvisionalTextAlias and this
// module stays strictly symmetric with it:
//   - the visible text ends with the self-delimiting marker  "[#<job_id>]"
//   - object_name is                                          "NzLTX23#<job_id>"
// AliasMatchesJob / the job-id extractor accept either form, so discovery works
// whether the scanner hands us the alias text (carrying the "[#...]" marker) or a
// serialized object_name line.
//
// Unit-tested with doctest (native/tests/test_provisional.cpp), including a
// round trip against BuildProvisionalTextAlias. All comments are ASCII/English.
#pragma once

#include <set>
#include <string>
#include <vector>

namespace nzltx {

// Lifecycle of a provisional object, from placement to final resolution.
enum class ProvisionalState {
    kPending,           // job running, placeholder on the timeline
    kCompleted,         // job finished OK (result ready to place)
    kFailed,            // job failed (placeholder should be cleared)
    kResolvedReplaced,  // placeholder re-found and replaced with the result
    kResolvedReserved,  // placeholder NOT found; result inserted at the reservation
    kOrphan,            // placeholder with no matching active job (stale)
};

// A reservation records where a provisional object was placed so the result can
// be inserted at the same spot if the placeholder cannot be re-found.
struct Reservation {
    std::string job_id;
    std::string object_name;   // "NzLTX23#<job_id>"
    int layer = 0;
    int frame = 0;
    int length_frames = 0;
};

// One object returned by a timeline scan (SDK-side; this module only sees the
// plain fields). frame span is half-open: [frame_start, frame_end).
struct ScannedObject {
    int layer = 0;
    int frame_start = 0;
    int frame_end = 0;
    std::string alias;
};

// True when the alias belongs to job_id: the job-id marker is embedded (exactly)
// in the alias' text-effect text item ("[#<job_id>]") or as the object_name
// token ("NzLTX23#<job_id>"). Both markers are self-delimiting, so matching is an
// exact whole-token comparison - never a prefix collision (job "abc" does not
// match a placeholder for "abcd"). An empty job_id never matches.
bool AliasMatchesJob(const std::string& alias, const std::string& job_id);

// Index of the provisional object for job_id in a scan, or -1 if none.
//
// Primary: the FIRST candidate whose alias exactly matches job_id (AliasMatchesJob).
// Fallback (mirrors the reference FindObjectCoveringFrameByExactAlias recovery
// path): if no alias matches - e.g. the user edited the placeholder text and
// destroyed the "[#...]" marker - recover the object the reservation still points
// at: the first candidate on reserved_layer whose [frame_start, frame_end) covers
// reserved_frame. The fallback is skipped when reserved_layer < 0 (no usable
// reservation), so a purely negative-layer sentinel disables it.
int FindProvisionalIndex(const std::vector<ScannedObject>& scanned,
                         const std::string& job_id, int reserved_layer,
                         int reserved_frame);

// Outcome of resolving a finished job against a fresh scan.
struct ProvisionalDecision {
    ProvisionalState next = ProvisionalState::kResolvedReserved;
    bool replace = false;  // true -> replace the found object; false -> insert new
    int layer = 0;         // where to act (found position or the reservation)
    int frame = 0;
};

// Decide how to place a finished result. found == (FindProvisionalIndex >= 0);
// found_layer/found_frame are that object's position (ignored when !found).
//   found  -> replace the placeholder in place  (kResolvedReplaced)
//   !found -> insert at the reservation position (kResolvedReserved)
ProvisionalDecision DecideResolve(bool found, const Reservation& r,
                                  int found_layer, int found_frame);

// Orphan detection: every scanned NzLTX23 provisional object whose embedded job
// id is NOT in active_job_ids is returned as a Reservation (so the caller can
// remove it). Objects with no NzLTX23 job-id marker (foreign objects) are
// ignored. The returned Reservation carries the object's scanned position/length.
std::vector<Reservation> DetectOrphans(const std::vector<ScannedObject>& scanned,
                                       const std::set<std::string>& active_job_ids);

}  // namespace nzltx
