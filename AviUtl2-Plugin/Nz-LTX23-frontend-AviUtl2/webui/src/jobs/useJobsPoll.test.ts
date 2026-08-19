import { renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { createApiClient } from "../api/client";
import { createMockBridge } from "../bridge/mockBridge";
import type { GenerateRequest } from "../api/types";
import { useJobsPoll } from "./useJobsPoll";

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
  return { mockBridge, apiClient };
}

describe("useJobsPoll", () => {
  it("starts empty and picks up jobs created before the hook mounted", async () => {
    const { apiClient } = setup(2);
    await apiClient.generate(REQUEST);

    const { result } = renderHook(() => useJobsPoll({ apiClient, intervalMs: 20 }));

    await waitFor(() => expect(result.current.jobs.length).toBe(1));
    expect(result.current.jobs[0]?.status).not.toBe("cancelled");
  });

  it("hasActiveJob is true while queued/running and flips false once the job settles", async () => {
    const { apiClient } = setup(2);
    await apiClient.generate(REQUEST);

    const { result } = renderHook(() => useJobsPoll({ apiClient, intervalMs: 20 }));

    await waitFor(() => expect(result.current.jobs.length).toBe(1));
    expect(result.current.hasActiveJob).toBe(true);

    await waitFor(() => expect(result.current.hasActiveJob).toBe(false), { timeout: 5_000 });
    expect(result.current.jobs[0]?.status).toBe("completed");
  }, 8_000);

  it(
    "fires onJobSettled exactly once when a job transitions to a terminal state, never for the initial baseline poll",
    async () => {
      const { apiClient } = setup(2);
      const onJobSettled = vi.fn();
      await apiClient.generate(REQUEST);

      renderHook(() => useJobsPoll({ apiClient, intervalMs: 20, onJobSettled }));

      await waitFor(() => expect(onJobSettled).toHaveBeenCalledTimes(1), { timeout: 5_000 });
      expect(onJobSettled.mock.calls[0]?.[0].status).toBe("completed");

      // Give it a few more poll ticks: must not fire again for the same job.
      await new Promise((resolve) => setTimeout(resolve, 200));
      expect(onJobSettled).toHaveBeenCalledTimes(1);
    },
    8_000,
  );

  it("does not fire onJobSettled for a job that was already terminal on the very first poll", async () => {
    const { apiClient } = setup(1);
    const accepted = await apiClient.generate(REQUEST);
    // Drive it to completion via direct getJob calls before the hook ever mounts.
    for (let i = 0; i < 5; i += 1) {
      await apiClient.getJob(accepted.job_id);
    }

    const onJobSettled = vi.fn();
    const { result } = renderHook(() => useJobsPoll({ apiClient, intervalMs: 20, onJobSettled }));

    await waitFor(() => expect(result.current.jobs.length).toBe(1));
    expect(result.current.jobs[0]?.status).toBe("completed");

    await new Promise((resolve) => setTimeout(resolve, 100));
    expect(onJobSettled).not.toHaveBeenCalled();
  });

  it("refresh() forces an immediate poll", async () => {
    const { apiClient } = setup(50);
    const { result } = renderHook(() => useJobsPoll({ apiClient, intervalMs: 60_000 })); // effectively never auto-polls again

    await waitFor(() => expect(result.current.jobs.length).toBe(0));

    await apiClient.generate(REQUEST);
    result.current.refresh();

    await waitFor(() => expect(result.current.jobs.length).toBe(1));
  });

  it("surfaces a transport error without throwing", async () => {
    const mockBridge = createMockBridge({ delayMs: 0, backendUnreachable: true });
    const apiClient = createApiClient(mockBridge);

    const { result } = renderHook(() => useJobsPoll({ apiClient, intervalMs: 20 }));

    await waitFor(() => expect(result.current.error).not.toBeNull());
    expect(result.current.jobs).toEqual([]);
  });
});
