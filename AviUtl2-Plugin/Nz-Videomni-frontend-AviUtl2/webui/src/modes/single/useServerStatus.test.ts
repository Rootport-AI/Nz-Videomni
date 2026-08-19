import { renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { createMockBridge } from "../../bridge/mockBridge";
import { createApiClient } from "../../api/client";
import { useServerStatus } from "./useServerStatus";

describe("useServerStatus", () => {
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
