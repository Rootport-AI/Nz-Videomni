import type { JobResponse } from "../api/types";

/**
 * True iff `job` is a V2V continuation of an uploaded source video, i.e. it
 * was submitted through `POST /generate/chain` with a `source_video` spec and
 * is therefore eligible for `POST /jobs/{id}/join` (Docs/API_REFERENCE.md
 * §3.17). This reads the server-provided `JobResponse.is_v2v` flag directly:
 * the echoed `JobResponse.request` is the whitelisted per-clip
 * `GenerateRequest` and never carries `source_video`, so the flag is the only
 * reliable signal. A2V/Clips chain jobs and plain `/generate` T2V/I2V jobs
 * report `is_v2v: false`. Pure and dependency-free so it's trivially
 * unit-testable.
 */
export function isV2VJob(job: JobResponse): boolean {
  return job.is_v2v === true;
}
