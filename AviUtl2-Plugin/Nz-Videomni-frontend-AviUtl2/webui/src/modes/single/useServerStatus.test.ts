import { renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { createMockBridge } from "../../bridge/mockBridge";
import { createApiClient } from "../../api/client";
import type { ApiClient } from "../../api/client";
import type { NativeBridge } from "../../bridge";
import type { StatusResponse } from "../../api/types";
import { useServerStatus } from "./useServerStatus";

describe("useServerStatus", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it("reports 'bridge-unavailable' when ping fails (native bridge itself unreachable)", async () => {
    const mockBridge = createMockBridge({ delayMs: 0, failPing: true });
    const apiClient = createApiClient(mockBridge);
    const { result } = renderHook(() => useServerStatus(60_000, { nativeBridge: mockBridge, apiClient }));

    await waitFor(() => {
      expect(result.current.state.kind).toBe("bridge-unavailable");
    });
  });

  it("reports 'offline' when the backend is unreachable but the bridge itself is fine", async () => {
    const mockBridge = createMockBridge({ delayMs: 0, backendUnreachable: true });
    const apiClient = createApiClient(mockBridge);
    const { result } = renderHook(() => useServerStatus(60_000, { nativeBridge: mockBridge, apiClient }));

    await waitFor(() => {
      expect(result.current.state.kind).toBe("offline");
    });
  });

  it("reports 'online' when the bridge and backend are both reachable with no active job", async () => {
    const mockBridge = createMockBridge({ delayMs: 0 });
    const apiClient = createApiClient(mockBridge);
    const { result } = renderHook(() => useServerStatus(60_000, { nativeBridge: mockBridge, apiClient }));

    await waitFor(() => {
      expect(result.current.state.kind).toBe("online");
    });
  });

  it("reports 'busy' when a job currently occupies the single-job queue", async () => {
    const mockBridge = createMockBridge({ delayMs: 0, runningPollCount: 50 });
    const apiClient = createApiClient(mockBridge);
    await apiClient.generate({
      prompt: "occupy the queue",
      width: 384,
      height: 256,
      num_frames: 17,
      frame_rate: 24,
      seed: 1,
    });

    const { result } = renderHook(() => useServerStatus(60_000, { nativeBridge: mockBridge, apiClient }));

    await waitFor(() => {
      expect(result.current.state.kind).toBe("busy");
    });
  });

  it("reports 'loading-models' while the server rebuilds its worker", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    const base = await createApiClient(bridge).getStatus();
    const apiClient: ApiClient = {
      ...createApiClient(bridge),
      getStatus: async () => ({ ...base, state: "loading" }) as StatusResponse,
    };
    const { result } = renderHook(() =>
      useServerStatus(60_000, { nativeBridge: bridge as NativeBridge, apiClient }),
    );

    await waitFor(() => {
      expect(result.current.state.kind).toBe("loading-models");
    });
  });

  it("localLoading reports 'loading-models' immediately, with no poll involved", async () => {
    // Replaces the old "the default period catches an 8-second load" test: the
    // WebUI's own load is no longer something the poll has to catch. The issuer
    // hands its `POST /pipeline/load` to `JobsContext.trackPipelineLoad`, and
    // this flag reports both edges at 0ms — which is why the default period
    // went back up to 10s.
    const bridge = createMockBridge({ delayMs: 0 });
    const real = createApiClient(bridge);
    const getStatus = vi.fn(() => real.getStatus());
    const apiClient: ApiClient = { ...real, getStatus };

    // 60s period: every `getStatus` call counted below is the mount check or
    // the falling edge, never the interval.
    const { result, rerender } = renderHook(
      ({ localLoading }: { localLoading: boolean }) =>
        useServerStatus(60_000, { nativeBridge: bridge as NativeBridge, apiClient, localLoading }),
      { initialProps: { localLoading: false } },
    );

    await waitFor(() => expect(result.current.state.kind).toBe("online"));
    const callsBefore = getStatus.mock.calls.length;

    rerender({ localLoading: true });
    // Synchronously, on the very render that raised the flag — no await.
    expect(result.current.state.kind).toBe("loading-models");
    expect(getStatus).toHaveBeenCalledTimes(callsBefore);

    rerender({ localLoading: false });
    // The falling edge fetches exactly once, so the badge clears against a
    // fresh body instead of waiting out the rest of the period.
    await waitFor(() => expect(result.current.state.kind).toBe("online"));
    expect(getStatus).toHaveBeenCalledTimes(callsBefore + 1);
  });

  it("retry() re-runs the check on demand", async () => {
    const mockBridge = createMockBridge({ delayMs: 0, backendUnreachable: true });
    const apiClient = createApiClient(mockBridge);
    const { result } = renderHook(() => useServerStatus(60_000, { nativeBridge: mockBridge, apiClient }));

    await waitFor(() => {
      expect(result.current.state.kind).toBe("offline");
    });

    result.current.retry();

    await waitFor(() => {
      expect(result.current.state.kind).toBe("offline");
    });
  });
});
