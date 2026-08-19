import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ReactElement } from "react";
import type { JobResponse } from "../api/types";
import type { NativeBridge } from "../bridge";
import { createMockBridge } from "../bridge/mockBridge";
import { createApiClient } from "../api/client";
import { LanguageProvider } from "../i18n/LanguageContext";
import { recordSourceLocation, resetSourceLocationMap } from "../timeline/sourceLocationMap";
import { markJobInserted, resetProvisionalReservation } from "../timeline/provisionalReservation";
import { JobCard } from "./JobCard";
import type { JobCardProps } from "./JobCard";

/** A native bridge stub for the Insert flow. `downloadVideo` returns a fake
 * path and `insertMediaForJob` a fake replaced position, unless `failDownload`
 * is set (to exercise the error state). Every call is recorded so tests can
 * assert the native side was / wasn't invoked. */
function stubBridge({ failDownload = false }: { failDownload?: boolean } = {}) {
  const request = vi.fn((method: string) => {
    if (method === "backend.downloadVideo") {
      return failDownload ? Promise.reject(new Error("disk full")) : Promise.resolve({ filePath: "C:/tmp/out.mp4" });
    }
    // The per-clip 🎞 insert now goes through the single replace-insert RPC.
    if (method === "timeline.insertMediaForJob")
      return Promise.resolve({ ok: true, mode: "replaced", layer: 1, frame: 0, usedFallback: false });
    if (method === "timeline.insertMedia") return Promise.resolve({ layer: 1, frame: 0 });
    return Promise.resolve({});
  });
  const bridge = { request, requestWithFiles: vi.fn(), on: vi.fn(() => () => {}) } as unknown as NativeBridge;
  return { bridge, request };
}

/** `JobCard` calls `useStrings()`, so every render needs a `LanguageProvider`
 * ancestor. */
function renderWithLanguage(ui: ReactElement) {
  return render(<LanguageProvider>{ui}</LanguageProvider>);
}

/** A minimal `completed` job with a result video, enough for the preview
 * to render. */
function completedJob(): JobResponse {
  return {
    job_id: "job-abcdef12",
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
    started_at: "2026-07-17T00:00:01Z",
    completed_at: "2026-07-17T00:00:09Z",
    error: null,
    request: {},
    result: {
      video_url: "/api/v1/jobs/job-abcdef12/video",
      duration_seconds: 2,
      resolution: "384x256",
      file_size_bytes: 1024,
      generation_time_seconds: 8,
      seed_used: 1,
      output_path: "/out/job-abcdef12.mp4",
      metadata_path: "/out/job-abcdef12.json",
    },
  };
}

/** A `running` counterpart to `completedJob()`, same `job_id`, for the
 * mount-running-then-rerender-completed transition tests below. */
function runningJob(): JobResponse {
  return {
    ...completedJob(),
    status: "running",
    progress: 0.5,
    started_at: "2026-07-17T00:00:01Z",
    completed_at: null,
    result: null,
  };
}

const noop = () => {};

function renderCard(props: Partial<JobCardProps> & { job: JobResponse }) {
  return renderWithLanguage(
    <JobCard baseUrl="http://127.0.0.1:18620" isCancelling={false} isDeleting={false} onCancel={noop} onDelete={noop} {...props} />,
  );
}

describe("JobCard autoOpenPreview (U-R1 MJ-4, fixed)", () => {
  it("shows the play-to-expand button (preview collapsed) by default", () => {
    renderCard({ job: completedJob() });
    // Collapsed: the play button is present, no <video> yet.
    expect(screen.getByRole("button", { name: /preview/i })).toBeInTheDocument();
    expect(document.querySelector("video")).toBeNull();
  });

  it("does NOT auto-open when the job is already completed at mount, even with autoOpenPreview set", () => {
    // Regression case: re-rendering/remounting a card for a job that
    // finished earlier (e.g. a ledger refresh) must not yank a
    // user-closed preview back open. Only an *observed transition* into
    // `completed` should open it (see the next test).
    renderCard({ job: completedJob(), autoOpenPreview: true });
    expect(document.querySelector("video")).toBeNull();
    expect(screen.getByRole("button", { name: /preview/i })).toBeInTheDocument();
  });

  it("auto-expands the preview only when the job transitions into completed while mounted", () => {
    const { rerender } = renderWithLanguage(
      <JobCard
        job={runningJob()}
        baseUrl="http://127.0.0.1:18620"
        isCancelling={false}
        isDeleting={false}
        onCancel={noop}
        onDelete={noop}
        autoOpenPreview
      />,
    );
    expect(document.querySelector("video")).toBeNull();

    rerender(
      <LanguageProvider>
        <JobCard
          job={completedJob()}
          baseUrl="http://127.0.0.1:18620"
          isCancelling={false}
          isDeleting={false}
          onCancel={noop}
          onDelete={noop}
          autoOpenPreview
        />
      </LanguageProvider>,
    );

    // Expanded once the transition is observed: the <video> renders.
    const video = document.querySelector("video");
    expect(video).not.toBeNull();
    expect(video?.getAttribute("src")).toBe("http://127.0.0.1:18620/api/v1/jobs/job-abcdef12/video");
    // preload="auto" is intentional (Chromium "metadata" leaves the first
    // frame undrawn); pin it so it can't be silently downgraded.
    expect(video).toHaveAttribute("preload", "auto");
  });
});

describe("JobCard header actions (emoji buttons)", () => {
  it("shows Insert + Delete in the header for a completed job", () => {
    renderCard({ job: completedJob() });
    const insert = screen.getByRole("button", { name: /^insert$/i });
    const del = screen.getByRole("button", { name: /^delete$/i });
    expect(insert).toBeInTheDocument();
    expect(del).toBeInTheDocument();
    // Both live in the header's action cluster, not a bottom action row.
    expect(insert.closest(".job-card-head-actions")).not.toBeNull();
    expect(del.closest(".job-card-head-actions")).not.toBeNull();
    // Cancel is not valid on a terminal job.
    expect(screen.queryByRole("button", { name: /^cancel$/i })).toBeNull();
  });

  it("shows only Cancel in the header for a running job", () => {
    renderCard({ job: runningJob() });
    const cancel = screen.getByRole("button", { name: /^cancel$/i });
    expect(cancel).toBeInTheDocument();
    expect(cancel.closest(".job-card-head-actions")).not.toBeNull();
    expect(screen.queryByRole("button", { name: /^insert$/i })).toBeNull();
    expect(screen.queryByRole("button", { name: /^delete$/i })).toBeNull();
  });

  it("shows only Delete in the header for a failed job", () => {
    const job: JobResponse = { ...completedJob(), status: "failed", result: null, error: "boom" };
    renderCard({ job });
    const del = screen.getByRole("button", { name: /^delete$/i });
    expect(del).toBeInTheDocument();
    expect(del.closest(".job-card-head-actions")).not.toBeNull();
    expect(screen.queryByRole("button", { name: /^insert$/i })).toBeNull();
    expect(screen.queryByRole("button", { name: /^cancel$/i })).toBeNull();
  });
});

describe("JobCard Insert cycle (③)", () => {
  it("after a successful insert, clicking the ✅ button resets to idle without calling the bridge again", async () => {
    const { bridge, request } = stubBridge();
    renderCard({ job: completedJob(), nativeBridge: bridge });

    fireEvent.click(screen.getByRole("button", { name: /^insert$/i }));

    // Settles into the `done`/"Inserted" state after the two bridge calls
    // (download + the single replace-insert `insertMediaForJob`, which subsumes
    // the old separate deleteProvisionalByJob marker cleanup).
    const inserted = await screen.findByRole("button", { name: /^inserted$/i });
    expect(request).toHaveBeenCalledTimes(2);
    expect(inserted).toHaveTextContent("✅");

    // Clicking ✅ is NOT a re-insert — it just resets to idle (🎞️) and never
    // touches the native bridge.
    fireEvent.click(inserted);
    const idle = screen.getByRole("button", { name: /^insert$/i });
    expect(idle).toHaveTextContent("🎞️");
    expect(request).toHaveBeenCalledTimes(2);
  });

  it("shows ⚠️ + the error message in the title on failure, and resets to idle on click without calling the bridge", async () => {
    const { bridge, request } = stubBridge({ failDownload: true });
    renderCard({ job: completedJob(), nativeBridge: bridge });

    fireEvent.click(screen.getByRole("button", { name: /^insert$/i }));

    // Error state: emoji ⚠️, aria-label still "Insert", title carries the
    // failure message, and the red hint row appears.
    await waitFor(() => {
      const btn = screen.getByRole("button", { name: /^insert$/i });
      expect(btn).toHaveTextContent("⚠️");
      expect(btn.getAttribute("title")).toContain("disk full");
    });
    expect(screen.getByText("disk full")).toBeInTheDocument();
    // Only the failed download call happened (the second insertMedia step was
    // never reached).
    expect(request).toHaveBeenCalledTimes(1);

    // Clicking ⚠️ is NOT a retry — it just resets to idle (🎞️) and never
    // touches the native bridge, exactly like the ✅ reset. The next 🎞️ click
    // is what would re-insert.
    fireEvent.click(screen.getByRole("button", { name: /^insert$/i }));
    const idle = screen.getByRole("button", { name: /^insert$/i });
    expect(idle).toHaveTextContent("🎞️");
    expect(request).toHaveBeenCalledTimes(1);
  });

  it("after a ✅ reset, re-inserting calls timeline.insertMediaForJob but NOT backend.downloadVideo (reuses the remembered path)", async () => {
    const { bridge, request } = stubBridge();
    renderCard({ job: completedJob(), nativeBridge: bridge });

    // First insert: download + the single replace-insert (two calls).
    fireEvent.click(screen.getByRole("button", { name: /^insert$/i }));
    const inserted = await screen.findByRole("button", { name: /^inserted$/i });
    expect(request).toHaveBeenCalledTimes(2);
    expect(request.mock.calls.filter((c) => c[0] === "backend.downloadVideo")).toHaveLength(1);

    // ✅ -> idle reset (no bridge call).
    fireEvent.click(inserted);
    expect(request).toHaveBeenCalledTimes(2);

    // Re-insert (🎞️): skips the download (remembered path reused) and only
    // calls timeline.insertMediaForJob, so AviUtl2's still-open file is never
    // re-downloaded over.
    fireEvent.click(screen.getByRole("button", { name: /^insert$/i }));
    await screen.findByRole("button", { name: /^inserted$/i });
    expect(request.mock.calls.filter((c) => c[0] === "backend.downloadVideo")).toHaveLength(1);
    expect(request.mock.calls.filter((c) => c[0] === "timeline.insertMediaForJob")).toHaveLength(2);
  });
});

describe("JobCard 🎞 insert -> ✅ marker cleanup (I13, spec §5-10)", () => {
  it("removes the job's provisional marker from the mock timeline table after a successful 🎞 insert", async () => {
    // A real mock bridge so the provisional table is observable via
    // scanProvisionals. Drive Generate -> completion on the SAME bridge the
    // JobCard inserts through, so backend.downloadVideo finds the job.
    const bridge = createMockBridge({ delayMs: 0 });

    const gen = await bridge.request("backend.request", {
      method: "POST",
      path: "/api/v1/generate",
      body: { prompt: "hi", width: 512, height: 320, num_frames: 49 },
    });
    const jobId = (gen.body as { job_id: string }).job_id;

    // Poll to completion (Generate -> 完了).
    let body: JobResponse | undefined;
    for (let i = 0; i < 12; i += 1) {
      const r = await bridge.request("backend.request", { method: "GET", path: `/api/v1/jobs/${jobId}` });
      body = r.body as unknown as JobResponse;
      if (body.status === "completed") break;
    }
    expect(body?.status).toBe("completed");

    // A ✅ provisional placeholder sits on the timeline for this job (as an
    // ✨/📷 reservation would after Generate bound it).
    await bridge.request("timeline.insertProvisional", {
      jobId,
      displayText: "done",
      numFrames: 49,
      genFps: 24,
      placement: "C",
      cursorLayer: 0,
      cursorFrame: 0,
    });
    const before = await bridge.request("timeline.scanProvisionals", {});
    expect(before.orphans.map((o) => o.jobId)).toContain(jobId);

    // 🎞 insert through the JobCard UI.
    renderCard({ job: body as JobResponse, nativeBridge: bridge });
    fireEvent.click(screen.getByRole("button", { name: /^insert$/i }));
    await screen.findByRole("button", { name: /^inserted$/i });

    // The ✅ marker is gone from the mock timeline table.
    const after = await bridge.request("timeline.scanProvisionals", {});
    expect(after.orphans.map((o) => o.jobId)).not.toContain(jobId);
  });

  it("a second 🎞 insert (marker already consumed) resolves mode:\"inserted\" instead of \"replaced\"", async () => {
    const bridge = createMockBridge({ delayMs: 0 });

    const gen = await bridge.request("backend.request", {
      method: "POST",
      path: "/api/v1/generate",
      body: { prompt: "hi", width: 512, height: 320, num_frames: 49 },
    });
    const jobId = (gen.body as { job_id: string }).job_id;
    let body: JobResponse | undefined;
    for (let i = 0; i < 12; i += 1) {
      const r = await bridge.request("backend.request", { method: "GET", path: `/api/v1/jobs/${jobId}` });
      body = r.body as unknown as JobResponse;
      if (body.status === "completed") break;
    }
    // A ✅ marker sits on the timeline for this job.
    await bridge.request("timeline.insertProvisional", {
      jobId,
      displayText: "done",
      numFrames: 49,
      genFps: 24,
      placement: "C",
      cursorLayer: 0,
      cursorFrame: 0,
    });

    const spy = vi.spyOn(bridge, "request");
    renderCard({ job: body as JobResponse, nativeBridge: bridge });

    // First 🎞: the marker is present -> replaced.
    fireEvent.click(screen.getByRole("button", { name: /^insert$/i }));
    const inserted = await screen.findByRole("button", { name: /^inserted$/i });
    const firstIdx = spy.mock.calls.findIndex((c) => c[0] === "timeline.insertMediaForJob");
    expect(await spy.mock.results[firstIdx]!.value).toMatchObject({ mode: "replaced" });

    // ✅ -> idle reset, then re-insert: the marker is gone now -> inserted.
    fireEvent.click(inserted);
    spy.mockClear();
    fireEvent.click(screen.getByRole("button", { name: /^insert$/i }));
    await screen.findByRole("button", { name: /^inserted$/i });
    const secondIdx = spy.mock.calls.findIndex((c) => c[0] === "timeline.insertMediaForJob");
    expect(await spy.mock.results[secondIdx]!.value).toMatchObject({ mode: "inserted" });
  });
});

describe("JobCard seed display (⑤)", () => {
  it("shows #<seed_used> from the result", () => {
    const job: JobResponse = { ...completedJob(), result: { ...completedJob().result!, seed_used: 123 } };
    renderCard({ job });
    expect(screen.getByText("#123")).toBeInTheDocument();
  });

  it("shows the random word when the seed is a -1 request with no result", () => {
    const job: JobResponse = { ...completedJob(), status: "failed", result: null, error: "x", request: { seed: -1 } };
    renderCard({ job });
    expect(screen.getByText(/random/i)).toBeInTheDocument();
  });
});

describe("JobCard preview toggle", () => {
  it("opens the preview on click, then closes it again on a second click", () => {
    renderCard({ job: completedJob() });
    const toggle = screen.getByRole("button", { name: /preview/i });

    fireEvent.click(toggle);
    expect(document.querySelector("video")).not.toBeNull();

    // The same button now advertises the close action and, clicked again,
    // collapses the preview back to the play state.
    const closeToggle = screen.getByRole("button", { name: /close preview/i });
    fireEvent.click(closeToggle);
    expect(document.querySelector("video")).toBeNull();
    expect(screen.getByRole("button", { name: /preview/i })).toBeInTheDocument();
  });
});

/** Drives the mock bridge through a real V2V chain job to completion so the
 * joined flow runs against a realistic `is_v2v: true` completed job. */
async function completedV2VJob(bridge: ReturnType<typeof createMockBridge>): Promise<JobResponse> {
  const apiClient = createApiClient(bridge);
  const accepted = await apiClient.generateChain({
    prompt: "continue this video",
    width: 512,
    height: 320,
    frame_rate: 24,
    seed: -1,
    overlap_frames: 3,
    overlap_strength: 0.5,
    clips: [{ num_frames: 49 }],
    source_video: { video_id: "mock-video-1", context_frames: 25 },
  });
  let job = await apiClient.getJob(accepted.job_id);
  for (let i = 0; i < 20 && job.status !== "completed"; i += 1) job = await apiClient.getJob(accepted.job_id);
  if (job.status !== "completed") throw new Error("mock V2V job never completed");
  return job;
}

describe("JobCard joined insert (I6, JOIN_FEATURE_RESEARCH.md §4.5)", () => {
  beforeEach(() => {
    resetSourceLocationMap();
  });

  it("joined 🎞 inserts at the source tail − trim on the frontmost layer, then removes the marker", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    const apiClient = createApiClient(bridge);
    const job = await completedV2VJob(bridge);
    // Source [0,299] @ projectFps 30 = 10.0s; join trims 7.0s (default 120f@24fps
    // -> 5.0s tail -> mock 12-5). kept 3.0s -> head = 300 - round(3*30) = 210.
    // Frontmost layer = getEditInfo.layerMax (100) + 1 = 101.
    recordSourceLocation(job.job_id, {
      layer: 2,
      frameStart: 0,
      frameEnd: 299,
      filePath: "C:\\src.mp4",
      rate: 30,
      scale: 1,
    });

    renderCard({ job, nativeBridge: bridge, apiClient });
    fireEvent.click(screen.getByRole("button", { name: /join with source/i }));
    await screen.findByRole("button", { name: /^unjoin$/i });
    // Wait for the JobCard's showingJoined state to propagate (the preview swaps
    // to /joined) before inserting, so the 🎞 click takes the joined branch.
    await waitFor(() =>
      expect(document.querySelector("video")?.getAttribute("src")).toContain("/joined"),
    );

    const spy = vi.spyOn(bridge, "request");
    fireEvent.click(screen.getByRole("button", { name: /^insert$/i }));
    await screen.findByRole("button", { name: /^inserted$/i });

    // Joined download (not the plain per-clip output).
    expect(
      spy.mock.calls.some((c) => c[0] === "backend.downloadVideo" && (c[1] as { joined?: boolean }).joined === true),
    ).toBe(true);
    // Positioned insertMedia at the resolved slot.
    const insertCall = spy.mock.calls.find((c) => c[0] === "timeline.insertMedia");
    expect(insertCall?.[1]).toMatchObject({ layer: 101, frame: 210 });
    // Provisional marker auto-removed, source location released.
    expect(
      spy.mock.calls.some(
        (c) => c[0] === "timeline.deleteProvisionalByJob" && (c[1] as { jobId?: string }).jobId === job.job_id,
      ),
    ).toBe(true);
    // The reservation-backed replace-insert is NEVER used for a joined clip.
    expect(spy.mock.calls.some((c) => c[0] === "timeline.insertMediaForJob")).toBe(false);
  });

  it("degrades to a position-less joined insert when no source location is recorded", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    const apiClient = createApiClient(bridge);
    const job = await completedV2VJob(bridge);
    // No recordSourceLocation -> the position can't be resolved.

    renderCard({ job, nativeBridge: bridge, apiClient });
    fireEvent.click(screen.getByRole("button", { name: /join with source/i }));
    await screen.findByRole("button", { name: /^unjoin$/i });
    await waitFor(() =>
      expect(document.querySelector("video")?.getAttribute("src")).toContain("/joined"),
    );

    const spy = vi.spyOn(bridge, "request");
    fireEvent.click(screen.getByRole("button", { name: /^insert$/i }));
    await screen.findByRole("button", { name: /^inserted$/i });

    const insertCall = spy.mock.calls.find((c) => c[0] === "timeline.insertMedia");
    expect(insertCall).toBeDefined();
    // Position-less: no layer/frame passed -> native's cursor placement.
    expect((insertCall?.[1] as { layer?: number }).layer).toBeUndefined();
    expect((insertCall?.[1] as { frame?: number }).frame).toBeUndefined();
  });

  it("swaps the single preview to the joined src while the joined view is active", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    const apiClient = createApiClient(bridge);
    const job = await completedV2VJob(bridge);

    renderCard({ job, nativeBridge: bridge, apiClient });
    fireEvent.click(screen.getByRole("button", { name: /join with source/i }));
    await screen.findByRole("button", { name: /^unjoin$/i });

    // Entering the joined view auto-opens the preview and points it at /joined
    // (the swap runs through JoinControls -> JobCard effects, so wait for it).
    await waitFor(() => {
      expect(document.querySelector("video")?.getAttribute("src")).toBe(
        `http://127.0.0.1:18620/api/v1/jobs/${job.job_id}/joined`,
      );
    });
  });

  it("body-view 🎞 still runs the plain per-clip replace-insert (no joined branch)", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    const apiClient = createApiClient(bridge);
    const job = await completedV2VJob(bridge);

    renderCard({ job, nativeBridge: bridge, apiClient });
    // Do NOT join: the card is still showing the body clip.
    const spy = vi.spyOn(bridge, "request");
    fireEvent.click(screen.getByRole("button", { name: /^insert$/i }));
    await screen.findByRole("button", { name: /^inserted$/i });

    expect(spy.mock.calls.some((c) => c[0] === "timeline.insertMediaForJob")).toBe(true);
    expect(spy.mock.calls.some((c) => c[0] === "timeline.insertMedia")).toBe(false);
    expect(spy.mock.calls.some((c) => c[0] === "timeline.deleteProvisionalByJob")).toBe(false);
  });
});

describe("JobCard Y3 W2 shared ✅ (non-symmetric: manual stays local, W2 shares)", () => {
  // The shared inserted-store is module-level; reset it so each case starts clean.
  beforeEach(() => {
    resetProvisionalReservation();
  });

  it("flips to ✅ when W2 marks THIS job inserted (markJobInserted)", async () => {
    const { bridge } = stubBridge();
    renderCard({ job: completedJob(), nativeBridge: bridge });
    // Starts idle (🎞 Insert).
    expect(screen.getByRole("button", { name: /^insert$/i })).toHaveTextContent("🎞️");

    // W2 (AppShell) publishes this job's insert -> the card flips to ✅.
    act(() => markJobInserted("job-abcdef12", "C:/w2-out.mp4"));
    const inserted = await screen.findByRole("button", { name: /^inserted$/i });
    expect(inserted).toHaveTextContent("✅");
  });

  it("a W2 filePath prime lets a ✅→🎞 re-insert reuse the file (no re-download)", async () => {
    const { bridge, request } = stubBridge();
    renderCard({ job: completedJob(), nativeBridge: bridge });

    // W2 marks it inserted WITH the already-downloaded path (no bridge call here).
    act(() => markJobInserted("job-abcdef12", "C:/w2-out.mp4"));
    const inserted = await screen.findByRole("button", { name: /^inserted$/i });
    // No download happened from the mark itself.
    expect(request.mock.calls.filter((c) => c[0] === "backend.downloadVideo")).toHaveLength(0);

    // ✅ -> idle reset (clears the shared flag).
    fireEvent.click(inserted);
    const idle = screen.getByRole("button", { name: /^insert$/i });

    // Re-insert (🎞): the W2-primed remembered path is reused, so backend
    // .downloadVideo is NEVER called — only the replace-insert runs.
    fireEvent.click(idle);
    await screen.findByRole("button", { name: /^inserted$/i });
    expect(request.mock.calls.filter((c) => c[0] === "backend.downloadVideo")).toHaveLength(0);
    expect(request.mock.calls.filter((c) => c[0] === "timeline.insertMediaForJob")).toHaveLength(1);
  });

  it("does NOT overwrite an in-progress manual insert (⏳ downloading) when W2 marks the same job", async () => {
    // A bridge whose download never resolves, so the manual insert parks at ⏳.
    const request = vi.fn((method: string) => {
      if (method === "backend.downloadVideo") return new Promise(() => {}); // hangs forever
      if (method === "timeline.insertMediaForJob")
        return Promise.resolve({ ok: true, mode: "replaced", layer: 1, frame: 0 });
      return Promise.resolve({});
    });
    const bridge = { request, requestWithFiles: vi.fn(), on: vi.fn(() => () => {}) } as unknown as NativeBridge;

    renderCard({ job: completedJob(), nativeBridge: bridge });
    const insertBtn = screen.getByRole("button", { name: /^insert$/i });

    // Manual 🎞: download hangs -> the card is stuck at ⏳ (downloading), disabled.
    fireEvent.click(insertBtn);
    await waitFor(() => expect(insertBtn).toBeDisabled());
    expect(insertBtn).toHaveTextContent("⏳");

    // A same-job W2 mark arrives mid-flight: the guard must NOT stomp the ⏳ into
    // ✅ — the in-progress manual insert owns the state until it settles.
    act(() => markJobInserted("job-abcdef12", "C:/w2-out.mp4"));
    expect(screen.queryByRole("button", { name: /^inserted$/i })).toBeNull();
    expect(insertBtn).toHaveTextContent("⏳");
  });
});
