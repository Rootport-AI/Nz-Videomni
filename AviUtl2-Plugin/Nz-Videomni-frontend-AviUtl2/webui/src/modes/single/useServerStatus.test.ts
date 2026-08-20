import { act, renderHook, waitFor } from "@testing-library/react";
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

  it("the DEFAULT poll period catches an 8-second model load", async () => {
    // The real regression (2026-08-20): a checkpoint swap on a warm machine
    // takes ~8s (Docs/VERIFICATION_LOG.md §68.5), and the old 10s period could
    // put the whole rebuild between two polls — the badge never appeared. The
    // load window below (1s..9s after mount) is placed to be missed by a 10s
    // poll on purpose, so this test fails if the default goes back up.
    const bridge = createMockBridge({ delayMs: 0 });
    const base = await createApiClient(bridge).getStatus();
    const LOAD_START_MS = 1_000;
    const LOAD_END_MS = 9_000;

    vi.useFakeTimers();
    const mountedAt = Date.now();
    const apiClient: ApiClient = {
      ...createApiClient(bridge),
      getStatus: async () => {
        const elapsed = Date.now() - mountedAt;
        const loading = elapsed >= LOAD_START_MS && elapsed < LOAD_END_MS;
        return { ...base, state: loading ? "loading" : "ready" } as StatusResponse;
      },
    };

    // No `intervalMs` argument: this exercises the SHIPPED default.
    const { result } = renderHook(() =>
      useServerStatus(undefined, { nativeBridge: bridge as NativeBridge, apiClient }),
    );

    let sawLoadingBadge = false;
    for (let elapsed = 0; elapsed < LOAD_END_MS; elapsed += 250) {
      await act(async () => {
        await vi.advanceTimersByTimeAsync(250);
      });
      if (result.current.state.kind === "loading-models") sawLoadingBadge = true;
    }

    expect(sawLoadingBadge).toBe(true);
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
