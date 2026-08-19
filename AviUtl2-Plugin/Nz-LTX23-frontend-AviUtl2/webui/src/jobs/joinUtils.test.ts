import { describe, expect, it } from "vitest";
import type { JobResponse } from "../api/types";
import { isV2VJob } from "./joinUtils";

/** Builds a completed job. `is_v2v` is the server-provided flag `isV2VJob`
 * reads; the echoed `request` is the whitelisted per-clip `GenerateRequest`
 * (never `source_video`), matching the real `GET /jobs` shape. */
function makeJob(overrides: Partial<JobResponse> = {}): JobResponse {
  return {
    job_id: "job-1",
    status: "completed",
    progress: 1,
    current_step: null,
    total_steps: null,
    stage: null,
    clip: null,
    clip_count: null,
    is_v2v: false,
    joined: false,
    created_at: "2026-01-01T00:00:00Z",
    started_at: "2026-01-01T00:00:00Z",
    completed_at: "2026-01-01T00:00:01Z",
    error: null,
    request: { prompt: "a cat", width: 384, height: 256, num_frames: 17 },
    result: null,
    ...overrides,
  };
}

describe("isV2VJob", () => {
  it("is true when the server marks the job is_v2v (a source_video continuation)", () => {
    // The echoed request never carries source_video (the real server whitelists
    // it out); the is_v2v flag is the only signal.
    const job = makeJob({ is_v2v: true });
    expect(isV2VJob(job)).toBe(true);
  });

  it("is false for a plain /generate T2V/I2V job (is_v2v false)", () => {
    const job = makeJob({ is_v2v: false });
    expect(isV2VJob(job)).toBe(false);
  });

  it("is false for an A2V chain job (source_audio, not a V2V continuation)", () => {
    const job = makeJob({ is_v2v: false });
    expect(isV2VJob(job)).toBe(false);
  });

  it("is false for a plain Clips chain job (neither source present)", () => {
    const job = makeJob({ is_v2v: false });
    expect(isV2VJob(job)).toBe(false);
  });
});
