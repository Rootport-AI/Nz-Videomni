import { act, renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { createApiClient } from "../api/client";
import { createMockBridge } from "../bridge/mockBridge";
import { useDangerZone } from "./useDangerZone";

const GEN_FIELDS = { width: 512, height: 320, num_frames: 49, frame_rate: 24 } as const;

describe("useDangerZone", () => {
  it("unloadPipeline() succeeds against an idle mock bridge", async () => {
    const apiClient = createApiClient(createMockBridge({ delayMs: 0 }));
    const { result } = renderHook(() => useDangerZone({ apiClient }));

    expect(result.current.unload.status).toBe("idle");
    act(() => {
      result.current.unloadPipeline();
    });
    expect(result.current.unload.status).toBe("loading");

    await waitFor(() => expect(result.current.unload.status).toBe("done"));
  });

  it("unloadPipeline() surfaces a dedicated busy state for 409 JOB_BUSY", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    const apiClient = createApiClient(bridge);
    // Put the mock bridge's single job slot in flight so the
    // pipeline-unload fixture's active-job check trips, mirroring
    // `useModels.test.ts`'s analogous `loadPipeline()` busy-state test
    // (lines 64-79) against the sibling `/pipeline/load` endpoint.
    await apiClient.generate({ prompt: "busy check", ...GEN_FIELDS, seed: 1 });

    const { result } = renderHook(() => useDangerZone({ apiClient }));

    act(() => {
      result.current.unloadPipeline();
    });

    await waitFor(() => expect(result.current.unload.status).toBe("busy"));
  });

  it("purgeTerminalJobs() deletes only terminal jobs, skips a failing delete, and counts successes", async () => {
    // runningPollCount: 0 means any job reports "completed" after its second
    // poll (deriveJobFields: pollCount<=1 -> "queued", else terminal) — used
    // here purely to get two jobs to a terminal status quickly and
    // deterministically, without depending on the running-phase progress
    // math.
    const bridge = createMockBridge({ delayMs: 0, runningPollCount: 0 });
    const apiClient = createApiClient(bridge);

    const jobA = await apiClient.generate({ prompt: "job A", ...GEN_FIELDS, seed: 1 });
    await apiClient.getJob(jobA.job_id);
    await apiClient.getJob(jobA.job_id);
    expect((await apiClient.getJob(jobA.job_id)).status).toBe("completed");

    // The mock bridge only allows one non-terminal ("active") job at a time;
    // jobA is terminal by now so this doesn't trip JOB_BUSY.
    const jobB = await apiClient.generate({ prompt: "job B", ...GEN_FIELDS, seed: 2 });
    await apiClient.getJob(jobB.job_id);
    await apiClient.getJob(jobB.job_id);
    expect((await apiClient.getJob(jobB.job_id)).status).toBe("completed");

    // jobC is left unpolled (pollCount 0) and never made active again after
    // jobB, so it stays "queued" — a non-terminal control that purge must
    // leave untouched. (Not generated, since only one job may be
    // non-terminal at a time and jobB itself must not be re-activated.)

    const realDeleteJob = apiClient.deleteJob.bind(apiClient);
    const deleteSpy = vi
      .spyOn(apiClient, "deleteJob")
      .mockImplementation((jobId: string) =>
        jobId === jobB.job_id
          ? Promise.reject(new Error("simulated delete race (e.g. JOB_NOT_FOUND)"))
          : realDeleteJob(jobId),
      );

    const { result } = renderHook(() => useDangerZone({ apiClient }));
    act(() => {
      result.current.purgeTerminalJobs();
    });
    expect(result.current.purge.status).toBe("loading");

    await waitFor(() => expect(result.current.purge.status).toBe("done"));
    if (result.current.purge.status !== "done") throw new Error("unreachable");

    // Both terminal jobs (A, B) were attempted; B's delete rejected and was
    // silently skipped, so only A's success is counted.
    expect(deleteSpy).toHaveBeenCalledWith(jobA.job_id);
    expect(deleteSpy).toHaveBeenCalledWith(jobB.job_id);
    expect(result.current.purge.deleted).toBe(1);
    expect(result.current.purge.attempted).toBe(2);

    deleteSpy.mockRestore();
    const jobsAfter = await apiClient.listJobs();
    // A was actually deleted; B survives because the spy intercepted its
    // delete before it reached the real bridge.
    expect(jobsAfter.find((j) => j.job_id === jobA.job_id)).toBeUndefined();
    expect(jobsAfter.find((j) => j.job_id === jobB.job_id)).toBeDefined();
  });

  it("purgeTerminalJobs() leaves a non-terminal (queued/running) job untouched", async () => {
    const bridge = createMockBridge({ delayMs: 0, runningPollCount: 0 });
    const apiClient = createApiClient(bridge);

    const queued = await apiClient.generate({ prompt: "still queued", ...GEN_FIELDS, seed: 1 });
    // Never polled: pollCount stays 0, so it derives as "queued" (non-terminal).

    const { result } = renderHook(() => useDangerZone({ apiClient }));
    act(() => {
      result.current.purgeTerminalJobs();
    });

    await waitFor(() => expect(result.current.purge.status).toBe("done"));
    if (result.current.purge.status !== "done") throw new Error("unreachable");
    expect(result.current.purge.deleted).toBe(0);

    const jobsAfter = await apiClient.listJobs();
    expect(jobsAfter.find((j) => j.job_id === queued.job_id)).toBeDefined();
  });

  it("purgeTerminalJobs() reports zero deletions when there is nothing terminal to purge", async () => {
    const apiClient = createApiClient(createMockBridge({ delayMs: 0 }));
    const { result } = renderHook(() => useDangerZone({ apiClient }));

    act(() => {
      result.current.purgeTerminalJobs();
    });

    await waitFor(() => expect(result.current.purge.status).toBe("done"));
    if (result.current.purge.status !== "done") throw new Error("unreachable");
    expect(result.current.purge.deleted).toBe(0);
    expect(result.current.purge.attempted).toBe(0);
  });

  it("purgeTerminalJobs() reports attempted>0 with deleted===0 when every delete fails, distinct from 'nothing to purge'", async () => {
    // Regression for the "purge found jobs but every delete failed" case
    // being misreported as "no finished jobs to delete" (attempted===0):
    // here attempted must reflect the terminal jobs found even though none
    // of the deletes actually succeeded.
    const bridge = createMockBridge({ delayMs: 0, runningPollCount: 0 });
    const apiClient = createApiClient(bridge);

    const jobA = await apiClient.generate({ prompt: "job A", ...GEN_FIELDS, seed: 1 });
    await apiClient.getJob(jobA.job_id);
    await apiClient.getJob(jobA.job_id);
    expect((await apiClient.getJob(jobA.job_id)).status).toBe("completed");

    const jobB = await apiClient.generate({ prompt: "job B", ...GEN_FIELDS, seed: 2 });
    await apiClient.getJob(jobB.job_id);
    await apiClient.getJob(jobB.job_id);
    expect((await apiClient.getJob(jobB.job_id)).status).toBe("completed");

    const deleteSpy = vi
      .spyOn(apiClient, "deleteJob")
      .mockImplementation(() => Promise.reject(new Error("simulated total delete failure")));

    const { result } = renderHook(() => useDangerZone({ apiClient }));
    act(() => {
      result.current.purgeTerminalJobs();
    });

    await waitFor(() => expect(result.current.purge.status).toBe("done"));
    if (result.current.purge.status !== "done") throw new Error("unreachable");

    expect(deleteSpy).toHaveBeenCalledWith(jobA.job_id);
    expect(deleteSpy).toHaveBeenCalledWith(jobB.job_id);
    expect(result.current.purge.attempted).toBe(2);
    expect(result.current.purge.deleted).toBe(0);

    deleteSpy.mockRestore();
  });
});
