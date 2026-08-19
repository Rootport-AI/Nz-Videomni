import { describe, expect, it } from "vitest";
import type { JobResponse, JobResult } from "../api/types";
import { jobSeed, latestJobSeed } from "./seedUtils";

function result(overrides: Partial<JobResult> = {}): JobResult {
  return {
    video_url: "/api/v1/jobs/x/video",
    duration_seconds: 2,
    resolution: "384x256",
    file_size_bytes: 1024,
    generation_time_seconds: 8,
    seed_used: 999,
    output_path: "/out/x.mp4",
    metadata_path: "/out/x.json",
    ...overrides,
  };
}

function job(overrides: Partial<JobResponse> = {}): JobResponse {
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
    created_at: "2026-07-17T00:00:00Z",
    started_at: null,
    completed_at: null,
    error: null,
    request: {},
    result: null,
    ...overrides,
  };
}

describe("jobSeed", () => {
  it("prefers result.seed_used when present", () => {
    expect(jobSeed(job({ request: { seed: 5 }, result: result({ seed_used: 123 }) }))).toBe(123);
  });

  it("uses result.seed_used even when it is 0", () => {
    expect(jobSeed(job({ result: result({ seed_used: 0 }) }))).toBe(0);
  });

  it("falls back to a non-negative request.seed when there is no result", () => {
    expect(jobSeed(job({ request: { seed: 42 }, result: null }))).toBe(42);
  });

  it("treats request.seed === -1 (random sentinel) as unknown", () => {
    expect(jobSeed(job({ request: { seed: -1 }, result: null }))).toBeNull();
  });

  it("returns null when neither seed_used nor a usable request.seed is present", () => {
    expect(jobSeed(job({ request: {}, result: null }))).toBeNull();
  });

  it("ignores a non-numeric request.seed", () => {
    expect(jobSeed(job({ request: { seed: "7" }, result: null }))).toBeNull();
  });
});

describe("latestJobSeed", () => {
  it("returns null for an empty list", () => {
    expect(latestJobSeed([])).toBeNull();
  });

  it("applies jobSeed to the newest job by created_at (descending)", () => {
    const older = job({ job_id: "old", created_at: "2026-07-17T00:00:00Z", result: result({ seed_used: 1 }) });
    const newer = job({ job_id: "new", created_at: "2026-07-17T05:00:00Z", result: result({ seed_used: 2 }) });
    // Deliberately out of order to prove it sorts rather than trusting input order.
    expect(latestJobSeed([older, newer])).toBe(2);
  });

  it("returns null when the newest job has no resolvable seed", () => {
    const newer = job({ job_id: "new", created_at: "2026-07-17T05:00:00Z", request: { seed: -1 }, result: null });
    const older = job({ job_id: "old", created_at: "2026-07-17T00:00:00Z", result: result({ seed_used: 1 }) });
    expect(latestJobSeed([older, newer])).toBeNull();
  });
});
