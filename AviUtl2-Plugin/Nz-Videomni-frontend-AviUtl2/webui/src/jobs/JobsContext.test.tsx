import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ReactNode } from "react";
import { createApiClient } from "../api/client";
import { createMockBridge } from "../bridge/mockBridge";
import type { GenerateRequest } from "../api/types";
import { LanguageProvider } from "../i18n/LanguageContext";
import { ToastProvider, useToasts } from "../shell/ToastContext";
import {
  bindToJob,
  getReservationPhase,
  reserveAtCursor,
  resetProvisionalReservation,
} from "../timeline/provisionalReservation";
import { JobsProvider, useJobsContext } from "./JobsContext";

const REQUEST: GenerateRequest = {
  prompt: "a cat riding a skateboard",
  width: 384,
  height: 256,
  num_frames: 17,
  frame_rate: 24,
  seed: 1,
};

function setup(runningPollCount: number) {
  const mockBridge = createMockBridge({ delayMs: 0, runningPollCount });
  const apiClient = createApiClient(mockBridge);
  const wrapper = ({ children }: { children: ReactNode }) => (
    <LanguageProvider>
      <ToastProvider>
        <JobsProvider apiClient={apiClient} nativeBridge={mockBridge} intervalMs={20}>
          {children}
        </JobsProvider>
      </ToastProvider>
    </LanguageProvider>
  );
  return { mockBridge, apiClient, wrapper };
}

// The provisional reservation seat is module-level state; keep every test from a
// clean idle seat (only the I12 tests below touch it, but resetting is cheap).
beforeEach(() => {
  resetProvisionalReservation();
});

describe("JobsContext", () => {
  it("polls GET /jobs and reflects serverBusy while a job is queued/running, then completes", async () => {
    const { apiClient, wrapper } = setup(2);
    await apiClient.generate(REQUEST);

    const { result } = renderHook(() => useJobsContext(), { wrapper });

    await waitFor(() => expect(result.current.jobs.length).toBe(1));
    expect(result.current.serverBusy).toBe(true);
    // The job half only — no `POST /pipeline/load` was tracked.
    expect(result.current.pipelineLoading).toBe(false);

    await waitFor(() => expect(result.current.serverBusy).toBe(false), { timeout: 8_000 });
    expect(result.current.jobs[0]?.status).toBe("completed");
  }, 10_000);

  it("trackPipelineLoad raises serverBusy/pipelineLoading for exactly the promise's flight", async () => {
    const { wrapper } = setup(1);
    const { result } = renderHook(() => useJobsContext(), { wrapper });

    await waitFor(() => expect(result.current.serverBusy).toBe(false));

    let resolveLoad!: (value: string) => void;
    const load = new Promise<string>((resolve) => {
      resolveLoad = resolve;
    });

    let tracked!: Promise<string>;
    act(() => {
      tracked = result.current.trackPipelineLoad(load);
    });
    expect(result.current.pipelineLoading).toBe(true);
    expect(result.current.serverBusy).toBe(true);

    await act(async () => {
      resolveLoad("loaded");
      // The value is handed straight back to the caller.
      expect(await tracked).toBe("loaded");
    });
    expect(result.current.pipelineLoading).toBe(false);
    expect(result.current.serverBusy).toBe(false);
  }, 10_000);

  it("trackPipelineLoad lowers the flag on rejection and re-raises the error to the caller", async () => {
    const { wrapper } = setup(1);
    const { result } = renderHook(() => useJobsContext(), { wrapper });

    let rejectLoad!: (reason: Error) => void;
    const load = new Promise<never>((_resolve, reject) => {
      rejectLoad = reject;
    });

    let tracked!: Promise<never>;
    act(() => {
      tracked = result.current.trackPipelineLoad(load);
    });
    expect(result.current.pipelineLoading).toBe(true);

    await act(async () => {
      rejectLoad(new Error("boom"));
      await expect(tracked).rejects.toThrow("boom");
    });
    expect(result.current.pipelineLoading).toBe(false);
    expect(result.current.serverBusy).toBe(false);
  }, 10_000);

  it("pushes a success toast when a job it observes transitions to completed", async () => {
    const { apiClient, wrapper } = setup(1);
    await apiClient.generate(REQUEST);

    const { result } = renderHook(() => ({ jobs: useJobsContext(), toasts: useToasts() }), { wrapper });

    await waitFor(() => expect(result.current.jobs.serverBusy).toBe(false), { timeout: 8_000 });
    await waitFor(() => expect(result.current.toasts.toasts.length).toBe(1));
    expect(result.current.toasts.toasts[0]?.kind).toBe("success");
    expect(result.current.toasts.toasts[0]?.jobId).toBe(result.current.jobs.jobs[0]?.job_id);
  }, 10_000);

  it("pushes an error toast for a forced-failure job", async () => {
    const { apiClient, wrapper } = setup(1);
    await apiClient.generate({ ...REQUEST, prompt: "__MOCK_FAIL__ trigger a failure" });

    const { result } = renderHook(() => useToasts(), { wrapper });

    await waitFor(() => expect(result.current.toasts.length).toBe(1), { timeout: 8_000 });
    expect(result.current.toasts[0]?.kind).toBe("error");
  }, 10_000);

  it("cancelJob marks the job as cancelling until it settles into cancelled, then clears", async () => {
    const { apiClient, wrapper } = setup(3);
    const accepted = await apiClient.generate(REQUEST);

    const { result } = renderHook(() => useJobsContext(), { wrapper });
    await waitFor(() => expect(result.current.jobs.length).toBe(1));

    await act(async () => {
      await result.current.cancelJob(accepted.job_id);
    });
    expect(result.current.cancellingIds.has(accepted.job_id)).toBe(true);

    await waitFor(() => expect(result.current.cancellingIds.has(accepted.job_id)).toBe(false), { timeout: 8_000 });
    await waitFor(() => expect(result.current.jobs.find((j) => j.job_id === accepted.job_id)?.status).toBe("cancelled"));
  }, 10_000);

  it("I12: a completed job it tracks rewrites the provisional to the ✅ done text and frees the seat", async () => {
    const { mockBridge, apiClient, wrapper } = setup(1);
    const accepted = await apiClient.generate(REQUEST);

    // A ✨-style reservation is waiting and bound to THIS job (§5-3 用途a).
    await reserveAtCursor(mockBridge, {
      cursorLayer: 3,
      cursorFrame: 100,
      numFrames: 17,
      genFps: 24,
      displayText: "(t2v)",
    });
    await bindToJob(mockBridge, { jobId: accepted.job_id, numFrames: 17, genFps: 24, displayText: "head" });
    expect(getReservationPhase()).toBe("generating");

    const spy = vi.spyOn(mockBridge, "request");
    renderHook(() => useJobsContext(), { wrapper });

    // Once the poll observes the job settle, the ✅ terminal text is pushed to the
    // tracked placeholder (§5-5 段階4) and the single seat is freed (§5-6).
    await waitFor(
      () => {
        expect(spy).toHaveBeenCalledWith(
          "timeline.updateProvisionalText",
          expect.objectContaining({ jobId: accepted.job_id, layer: 3, frame: 100, text: expect.stringContaining("Done") }),
        );
      },
      { timeout: 8_000 },
    );
    await waitFor(() => expect(getReservationPhase()).toBe("idle"));
  }, 10_000);

  it("I12: a failed job it tracks rewrites the provisional to the ❌ failed text", async () => {
    const { mockBridge, apiClient, wrapper } = setup(1);
    const accepted = await apiClient.generate({ ...REQUEST, prompt: "__MOCK_FAIL__ trigger a failure" });

    await reserveAtCursor(mockBridge, {
      cursorLayer: 3,
      cursorFrame: 100,
      numFrames: 17,
      genFps: 24,
      displayText: "(t2v)",
    });
    await bindToJob(mockBridge, { jobId: accepted.job_id, numFrames: 17, genFps: 24, displayText: "head" });

    const spy = vi.spyOn(mockBridge, "request");
    renderHook(() => useJobsContext(), { wrapper });

    await waitFor(
      () => {
        expect(spy).toHaveBeenCalledWith(
          "timeline.updateProvisionalText",
          expect.objectContaining({ jobId: accepted.job_id, text: expect.stringContaining("Failed") }),
        );
      },
      { timeout: 8_000 },
    );
    await waitFor(() => expect(getReservationPhase()).toBe("idle"));
  }, 10_000);

  it("I12: a settling job it does NOT track leaves the reservation untouched (no text RPC)", async () => {
    const { mockBridge, apiClient, wrapper } = setup(1);
    await apiClient.generate(REQUEST);

    // A reservation is bound to a DIFFERENT job than the one that will settle.
    await reserveAtCursor(mockBridge, {
      cursorLayer: 3,
      cursorFrame: 100,
      numFrames: 17,
      genFps: 24,
      displayText: "(t2v)",
    });
    await bindToJob(mockBridge, { jobId: "unrelated-job", numFrames: 17, genFps: 24, displayText: "head" });

    const spy = vi.spyOn(mockBridge, "request");
    const { result } = renderHook(() => useJobsContext(), { wrapper });

    await waitFor(() => expect(result.current.jobs[0]?.status).toBe("completed"), { timeout: 8_000 });
    // The unrelated settle never wrote to (or freed) our reservation.
    expect(spy).not.toHaveBeenCalledWith("timeline.updateProvisionalText", expect.anything());
    expect(getReservationPhase()).toBe("generating");
  }, 10_000);

  it("deleteJob removes a terminal job from the list", async () => {
    const { apiClient, wrapper } = setup(1);
    const accepted = await apiClient.generate(REQUEST);

    const { result } = renderHook(() => useJobsContext(), { wrapper });
    await waitFor(() => expect(result.current.jobs[0]?.status).toBe("completed"), { timeout: 8_000 });

    await act(async () => {
      await result.current.deleteJob(accepted.job_id);
    });

    await waitFor(() => expect(result.current.jobs.length).toBe(0));
    expect(result.current.deletingIds.has(accepted.job_id)).toBe(false);
  }, 10_000);
});
