import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { ReactElement } from "react";
import type { JobResponse } from "../api/types";
import { createMockBridge } from "../bridge/mockBridge";
import { createApiClient } from "../api/client";
import { LanguageProvider } from "../i18n/LanguageContext";
import { JobCard } from "./JobCard";

/**
 * Regression guard for the "test-green / device-invisible" Join bug (increment
 * I4). On a real server, `GET /jobs` runs each record through
 * `to_clip_request`, so the echoed `JobResponse.request` NEVER carries
 * `source_video` — the old request-sniffing `isV2VJob` therefore returned
 * false on real hardware even though the mock (which used to echo the raw
 * body) made every test pass. These tests pin the current contract: the Join
 * control's visibility is driven exclusively by the server-provided
 * `is_v2v` flag, rendered through the real `JobCard` -> `JoinControls` path.
 */
function renderWithLanguage(ui: ReactElement) {
  return render(<LanguageProvider>{ui}</LanguageProvider>);
}

/** A completed job shaped exactly like the real `GET /jobs/{id}` response:
 * the echoed `request` is the whitelisted per-clip `GenerateRequest` (no
 * `source_video`), and V2V-ness is carried only by `is_v2v`. */
function realServerV2VJob(overrides: Partial<JobResponse> = {}): JobResponse {
  return {
    job_id: "job-v2v-1",
    status: "completed",
    progress: 1,
    current_step: null,
    total_steps: null,
    stage: null,
    clip: 1,
    clip_count: 1,
    is_v2v: true,
    joined: false,
    created_at: "2026-07-21T00:00:00Z",
    started_at: "2026-07-21T00:00:01Z",
    completed_at: "2026-07-21T00:00:09Z",
    error: null,
    // Whitelisted per-clip GenerateRequest — deliberately NO source_video.
    request: {
      prompt: "continue this video",
      negative_prompt: "",
      width: 512,
      height: 320,
      crop_output: null,
      num_frames: 49,
      frame_rate: 24,
      num_inference_steps: 8,
      guidance_scale: 1.0,
      seed: 7,
      pipeline: "distilled",
      conditioning_images: [],
      loras: [],
      reference_video_id: null,
      conditioning_attention_strength: null,
      reference_video_strength: null,
    },
    result: {
      video_url: "/api/v1/jobs/job-v2v-1/video",
      duration_seconds: 2,
      resolution: "512x320",
      file_size_bytes: 1024,
      generation_time_seconds: 8,
      seed_used: 7,
      output_path: "/out/job-v2v-1.mp4",
      metadata_path: "/out/job-v2v-1.json",
    },
    ...overrides,
  };
}

function renderCard(job: JobResponse) {
  const bridge = createMockBridge({ delayMs: 0 });
  const apiClient = createApiClient(bridge);
  return renderWithLanguage(
    <JobCard
      job={job}
      baseUrl="http://127.0.0.1:18620"
      isCancelling={false}
      isDeleting={false}
      onCancel={() => {}}
      onDelete={() => {}}
      nativeBridge={bridge}
      apiClient={apiClient}
    />,
  );
}

describe("Join control visibility (I4 regression guard)", () => {
  it("shows the Join button for a real-server-shaped V2V job (is_v2v:true, request has no source_video)", () => {
    renderCard(realServerV2VJob());
    // This is the exact case that was invisible on real hardware before I4.
    expect(screen.getByRole("button", { name: /join with source/i })).toBeInTheDocument();
  });

  it("does NOT show the Join button when is_v2v is false, even if request happened to carry source_video", () => {
    // Proves the gate is is_v2v-driven, not request-sniffing: a stray
    // source_video in the echoed request must never resurrect the old bug.
    const job = realServerV2VJob({
      is_v2v: false,
      request: { ...realServerV2VJob().request, source_video: { video_id: "vid-1", context_frames: 25 } },
    });
    renderCard(job);
    expect(screen.queryByRole("button", { name: /join with source/i })).not.toBeInTheDocument();
  });

  it("does NOT show the Join button for a non-completed V2V job", () => {
    renderCard(realServerV2VJob({ status: "running", result: null, completed_at: null, progress: 0.5 }));
    expect(screen.queryByRole("button", { name: /join with source/i })).not.toBeInTheDocument();
  });
});
