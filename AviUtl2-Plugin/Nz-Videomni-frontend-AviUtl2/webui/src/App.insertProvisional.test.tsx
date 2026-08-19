import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, render, screen, within } from "@testing-library/react";
import { createMockBridge } from "./bridge/mockBridge";
import { TIMELINE_MENU_INVOKED_EVENT } from "./bridge";
import type { ResultOf } from "./bridge";
import type { ApiClient } from "./api/client";
import type { JobResponse, JobStatus } from "./api/types";
import { AppShell } from "./shell/AppShell";
import { resetProvisionalReservation } from "./timeline/provisionalReservation";

// W2 (⬇ insertProvisionalResult): right-clicking a Nz-Videomni provisional object
// runs the SAME replace-insert the panel's 🎞 button does when the backing job
// is completed. `downloadAndInsertVideo` is mocked so these tests assert the
// routing/branching (a–e) without any real download/insert.
vi.mock("./jobs/downloadAndInsert", () => ({
  downloadAndInsertVideo: vi.fn(),
}));
import { downloadAndInsertVideo } from "./jobs/downloadAndInsert";
const mockDownloadAndInsert = vi.mocked(downloadAndInsertVideo);

/** A full `JobResponse` with sensible defaults; override only what a test needs. */
function makeJob(job_id: string, status: JobStatus, overrides: Partial<JobResponse> = {}): JobResponse {
  return {
    job_id,
    status,
    progress: status === "completed" ? 100 : null,
    current_step: null,
    total_steps: null,
    stage: null,
    clip: null,
    clip_count: null,
    created_at: "2026-07-22T00:00:00Z",
    started_at: null,
    completed_at: status === "completed" ? "2026-07-22T00:01:00Z" : null,
    error: status === "failed" ? "boom" : null,
    is_v2v: false,
    joined: false,
    request: {},
    result: null,
    ...overrides,
  };
}

/** A minimal `ApiClient` whose `GET /jobs` poll returns the seeded ledger. Only
 * `listJobs` is exercised here (the poll); the rest are inert stubs. */
function fakeApiClient(jobs: JobResponse[]): ApiClient {
  return {
    getStatus: vi.fn(),
    getConfig: vi.fn(),
    generate: vi.fn(),
    generateChain: vi.fn(),
    getJob: vi.fn(),
    listJobs: vi.fn(async () => jobs),
    deleteJob: vi.fn(),
    joinJob: vi.fn(),
    getLoras: vi.fn(),
    reloadLoras: vi.fn(),
    getModels: vi.fn(),
    loadPipeline: vi.fn(),
    unloadPipeline: vi.fn(),
  } as unknown as ApiClient;
}

/** A menu-invoked selection whose single object carries `objectName`
 * (the reservation naming native uses: `NzVideomni#<jobId>`). */
function provisionalSelection(objectName: string | null): ResultOf<"timeline.getSelection"> {
  return {
    hasRange: false,
    rangeStart: 0,
    rangeEnd: 0,
    selected: [
      {
        layer: 3,
        frameStart: 100,
        frameEnd: 200,
        effectName: "テキスト",
        filePath: null,
        objectName,
        textContent: null,
        mediaWidth: 0,
        mediaHeight: 0,
        mediaDurationSec: 0,
      },
    ],
    cursorFrame: 0,
    cursorLayer: 0,
    rate: 30,
    scale: 1,
    sampleRate: 44100,
  };
}

/** The single visible mode panel (both Create and Chain mount a JobLedger, so
 * queries into the ledger must be scoped to the active tabpanel). */
const panel = () => screen.getByRole("tabpanel");

/** Wait until the (visible-panel) ledger heading reflects the seeded count, so
 * the mirrored `jobsRef` handleRoute reads is guaranteed populated before we
 * emit the menu event. (Not keyed off the Generate button: a seeded *running*
 * job flips that button to its "busy" label, so it is not a stable ready
 * signal here.) */
async function waitForJobCount(count: number) {
  await within(panel()).findByText(new RegExp(`jobs \\(${count}\\)`, "i"), undefined, { timeout: 5_000 });
}

async function renderShell(jobs: JobResponse[]) {
  const bridge = createMockBridge({ delayMs: 0 });
  render(<AppShell nativeBridge={bridge} apiClient={fakeApiClient(jobs)} />);
  // Confirms the shell mounted (menu router subscribed) AND the ledger poll
  // populated the mirrored jobsRef before any emit.
  await waitForJobCount(jobs.length);
  return bridge;
}

function emit(bridge: ReturnType<typeof createMockBridge>, selection: ResultOf<"timeline.getSelection">) {
  act(() => {
    bridge.emit(TIMELINE_MENU_INVOKED_EVENT, { action: "insertProvisionalResult", selection });
  });
}

describe("App / W2 ⬇ insert this generated result now (insertProvisionalResult)", () => {
  beforeEach(() => {
    window.localStorage.clear();
    resetProvisionalReservation();
    mockDownloadAndInsert.mockReset();
    mockDownloadAndInsert.mockResolvedValue({ layer: 0, frame: 0, filePath: "C:/out.mp4" });
  });
  afterEach(() => {
    window.localStorage.clear();
  });

  it(
    "(a) a non-Nz-Videomni object shows the 'right-click a reservation object' toast and inserts nothing",
    async () => {
      const bridge = await renderShell([makeJob("job-done", "completed")]);
      await waitForJobCount(1);

      emit(bridge, provisionalSelection("randomClip"));

      await screen.findByText(/right-click a nz-videomni reservation object/i, undefined, { timeout: 5_000 });
      expect(mockDownloadAndInsert).not.toHaveBeenCalled();
    },
    15_000,
  );

  it(
    "(b) a reservation whose job is not in the ledger shows the 'no matching job' toast and inserts nothing",
    async () => {
      // Empty ledger: an unbound `pending-…` reservation has no job to insert.
      const bridge = await renderShell([]);
      // 0 jobs -> the (b) branch is deterministic regardless of poll timing.

      emit(bridge, provisionalSelection("NzVideomni#pending-xyz"));

      await screen.findByText(/no job matching this reservation object/i, undefined, { timeout: 5_000 });
      expect(mockDownloadAndInsert).not.toHaveBeenCalled();
    },
    15_000,
  );

  it(
    "(c) a completed reservation runs the replace-insert (no plainInsertAt) and shows the success toast",
    async () => {
      const bridge = await renderShell([makeJob("job-done", "completed")]);
      await waitForJobCount(1);

      emit(bridge, provisionalSelection("NzVideomni#job-done"));

      await screen.findByText(/inserted the generated result/i, undefined, { timeout: 5_000 });
      // Same 🎞 replace-insert: `downloadAndInsertVideo(bridge, jobId)` with NO
      // options object (no plainInsertAt), so the provisional marker is replaced.
      expect(mockDownloadAndInsert).toHaveBeenCalledTimes(1);
      expect(mockDownloadAndInsert).toHaveBeenCalledWith(bridge, "job-done");
    },
    15_000,
  );

  it(
    "(c) Y3: flips the target JobCard to ✅ (inserted) after the W2 insert (shared store)",
    async () => {
      const bridge = await renderShell([makeJob("job-done", "completed")]);
      await waitForJobCount(1);

      // Before: the completed job's card shows the idle 🎞 Insert action, not ✅.
      expect(screen.getAllByRole("button", { name: /^insert$/i }).length).toBeGreaterThan(0);
      expect(screen.queryByRole("button", { name: /^inserted$/i })).toBeNull();

      emit(bridge, provisionalSelection("NzVideomni#job-done"));

      await screen.findByText(/inserted the generated result/i, undefined, { timeout: 5_000 });
      // W2 published the insert to the shared store -> the job's card(s) flip to
      // ✅ (this is what a JobCard's LOCAL insertState could never reach on its
      // own). Manual per-card 🎞 stays local; only W2 shares (Y3 非対称案).
      const inserted = await screen.findAllByRole("button", { name: /^inserted$/i });
      expect(inserted.length).toBeGreaterThan(0);
      expect(inserted[0]).toHaveTextContent("✅");
    },
    15_000,
  );

  it(
    "(c) shows the failure toast when the insert throws",
    async () => {
      mockDownloadAndInsert.mockRejectedValueOnce(new Error("INSERT_FAILED"));
      const bridge = await renderShell([makeJob("job-done", "completed")]);
      await waitForJobCount(1);

      emit(bridge, provisionalSelection("NzVideomni#job-done"));

      await screen.findByText(/could not insert the generated result/i, undefined, { timeout: 5_000 });
      expect(mockDownloadAndInsert).toHaveBeenCalledWith(bridge, "job-done");
    },
    15_000,
  );

  it(
    "(d) a still-generating job shows the 'still generating' toast and inserts nothing",
    async () => {
      const bridge = await renderShell([makeJob("job-run", "running")]);
      await waitForJobCount(1);

      emit(bridge, provisionalSelection("NzVideomni#job-run"));

      await screen.findByText(/still generating/i, undefined, { timeout: 5_000 });
      expect(mockDownloadAndInsert).not.toHaveBeenCalled();
    },
    15_000,
  );

  it(
    "(e) a failed job shows the 'did not finish successfully' toast and inserts nothing",
    async () => {
      const bridge = await renderShell([makeJob("job-fail", "failed")]);
      await waitForJobCount(1);

      emit(bridge, provisionalSelection("NzVideomni#job-fail"));

      await screen.findByText(/did not finish successfully/i, undefined, { timeout: 5_000 });
      expect(mockDownloadAndInsert).not.toHaveBeenCalled();
    },
    15_000,
  );

  it(
    "(e) a cancelled job also shows the failure toast and inserts nothing",
    async () => {
      const bridge = await renderShell([makeJob("job-cancel", "cancelled")]);
      await waitForJobCount(1);

      emit(bridge, provisionalSelection("NzVideomni#job-cancel"));

      await screen.findByText(/did not finish successfully/i, undefined, { timeout: 5_000 });
      expect(mockDownloadAndInsert).not.toHaveBeenCalled();
    },
    15_000,
  );
});
