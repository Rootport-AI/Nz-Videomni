// provisional.cpp - implementation. See provisional.h for the contract and the
// symmetry requirement with alias_util::BuildProvisionalTextAlias. ASCII-only
// source: the one Japanese name we need (the "text" effect/item name) is a UTF-8
// byte escape, exactly as in alias_util.cpp, so no /utf-8 flag is required.
#include "provisional.h"

#include "alias_util.h"

namespace nzltx {

namespace {

// Must stay identical to alias_util.cpp: the AviUtl2 "text" effect name, which is
// ALSO the key of its text-content item (BuildProvisionalTextAlias writes both).
const char kEffectText[] =
    "\xe3\x83\x86\xe3\x82\xad\xe3\x82\xb9\xe3\x83\x88";  // "text"
// Must stay identical to alias_util.cpp's kProvisionalNamePrefix.
const char kNamePrefix[] = "NzLTX23#";  // object_name = prefix + job_id

// Characters that may appear in a job id (so a name token can be delimited).
bool IsIdChar(char c) {
    const unsigned char u = static_cast<unsigned char>(c);
    return (u >= '0' && u <= '9') || (u >= 'A' && u <= 'Z') ||
           (u >= 'a' && u <= 'z') || u == '-' || u == '_';
}

// Extract the job id embedded in an alias, if this is an NzLTX23 provisional
// object. Returns true + *out on success. Two symmetric sources are tried:
//   1. the text-effect text item's trailing "[#<job_id>]" marker
//   2. the "NzLTX23#<job_id>" object_name token anywhere in the raw alias
// The marker in (1) is looked up with rfind so decorative text cannot shadow it.
bool ExtractJobId(const std::string& alias, std::string* out) {
    std::string value;
    if (ParseAliasItemValue(alias, kEffectText, kEffectText, &value)) {
        const size_t p = value.rfind("[#");
        if (p != std::string::npos) {
            const size_t q = value.find(']', p + 2);
            if (q != std::string::npos && q > p + 2) {
                *out = value.substr(p + 2, q - (p + 2));
                return true;
            }
        }
    }
    const std::string src = StripUtf8Bom(alias);
    const size_t p = src.find(kNamePrefix);
    if (p != std::string::npos) {
        const size_t s = p + (sizeof(kNamePrefix) - 1);
        size_t e = s;
        while (e < src.size() && IsIdChar(src[e])) {
            ++e;
        }
        if (e > s) {
            *out = src.substr(s, e - s);
            return true;
        }
    }
    return false;
}

}  // namespace

bool AliasMatchesJob(const std::string& alias, const std::string& job_id) {
    if (job_id.empty()) {
        return false;
    }
    // Primary: the "[#<job_id>]" marker inside the text-effect text item. The
    // closing ']' makes this an exact whole-token match under a plain substring
    // search (no prefix collision).
    const std::string text_marker = "[#" + job_id + "]";
    std::string value;
    if (ParseAliasItemValue(alias, kEffectText, kEffectText, &value)) {
        if (value.find(text_marker) != std::string::npos) {
            return true;
        }
    }
    // Fallback: the "NzLTX23#<job_id>" object_name token, if the scanner serialized
    // it into the alias. Delimit the end so "NzLTX23#abc" does not match "...abcd".
    const std::string src = StripUtf8Bom(alias);
    const std::string name_token = std::string(kNamePrefix) + job_id;
    size_t from = 0;
    for (;;) {
        const size_t at = src.find(name_token, from);
        if (at == std::string::npos) {
            return false;
        }
        const size_t after = at + name_token.size();
        if (after >= src.size() || !IsIdChar(src[after])) {
            return true;  // token boundary confirmed
        }
        from = at + 1;  // this was a longer id; keep looking
    }
}

int FindProvisionalIndex(const std::vector<ScannedObject>& scanned,
                         const std::string& job_id, int reserved_layer,
                         int reserved_frame) {
    // Primary: exact alias/job match.
    for (size_t i = 0; i < scanned.size(); ++i) {
        if (AliasMatchesJob(scanned[i].alias, job_id)) {
            return static_cast<int>(i);
        }
    }
    // Fallback: recover by the reservation position (only if one was supplied).
    if (reserved_layer >= 0) {
        for (size_t i = 0; i < scanned.size(); ++i) {
            const ScannedObject& o = scanned[i];
            if (o.layer == reserved_layer && o.frame_start <= reserved_frame &&
                reserved_frame < o.frame_end) {
                return static_cast<int>(i);
            }
        }
    }
    return -1;
}

ProvisionalDecision DecideResolve(bool found, const Reservation& r,
                                  int found_layer, int found_frame) {
    ProvisionalDecision d;
    if (found) {
        d.next = ProvisionalState::kResolvedReplaced;
        d.replace = true;
        d.layer = found_layer;
        d.frame = found_frame;
    } else {
        d.next = ProvisionalState::kResolvedReserved;
        d.replace = false;
        d.layer = r.layer;
        d.frame = r.frame;
    }
    return d;
}

std::vector<Reservation> DetectOrphans(
    const std::vector<ScannedObject>& scanned,
    const std::set<std::string>& active_job_ids) {
    std::vector<Reservation> orphans;
    for (const ScannedObject& o : scanned) {
        std::string jid;
        if (!ExtractJobId(o.alias, &jid)) {
            continue;  // foreign object (no NzLTX23 marker) -> ignore
        }
        if (active_job_ids.find(jid) != active_job_ids.end()) {
            continue;  // still an active job -> not an orphan
        }
        Reservation r;
        r.job_id = jid;
        r.object_name = std::string(kNamePrefix) + jid;
        r.layer = o.layer;
        r.frame = o.frame_start;
        r.length_frames = o.frame_end - o.frame_start;
        orphans.push_back(r);
    }
    return orphans;
}

}  // namespace nzltx
