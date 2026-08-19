import { act, renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { createApiClient } from "../api/client";
import { createMockBridge } from "../bridge/mockBridge";
import { useModels } from "./useModels";

describe("useModels", () => {
  it("fetches GET /models on mount, all-default selection, entries per category", async () => {
    const apiClient = createApiClient(createMockBridge({ delayMs: 0 }));
    const { result } = renderHook(() => useModels({ apiClient }));

    expect(result.current.list.status).toBe("loading");
    await waitFor(() => expect(result.current.list.status).toBe("ready"));
    if (result.current.list.status !== "ready") throw new Error("unreachable");

    expect(result.current.list.models.categories.transformer.entries.map((e) => e.name)).toEqual(
      expect.arrayContaining(["default", "Sulphur-2-base-distil-Q4_K_M"]),
    );
    expect(result.current.selection).toEqual({
      transformer: "default",
      text_encoder: "default",
      video_vae: "default",
      audio: "default",
    });
  });

  it("refresh() re-fetches so a newly detected entry (or active mark) shows up", async () => {
    const apiClient = createApiClient(createMockBridge({ delayMs: 0 }));
    const { result } = renderHook(() => useModels({ apiClient }));
    await waitFor(() => expect(result.current.list.status).toBe("ready"));

    await act(async () => {
      await result.current.refresh();
    });

    expect(result.current.list.status).toBe("ready");
    if (result.current.list.status !== "ready") throw new Error("unreachable");
    expect(result.current.list.models.categories.audio.entries.length).toBeGreaterThan(0);
  });

  it("loadPipeline() success updates the selected category's active name (via refresh)", async () => {
    const apiClient = createApiClient(createMockBridge({ delayMs: 0 }));
    const { result } = renderHook(() => useModels({ apiClient }));
    await waitFor(() => expect(result.current.list.status).toBe("ready"));

    act(() => {
      result.current.setSelection("transformer", "Sulphur-2-base-distil-Q4_K_M");
    });
    act(() => {
      result.current.loadPipeline();
    });
    expect(result.current.load.status).toBe("loading");

    await waitFor(() => expect(result.current.load.status).toBe("done"));
    if (result.current.load.status !== "done") throw new Error("unreachable");
    expect(result.current.load.models.transformer).toBe("Sulphur-2-base-distil-Q4_K_M");

    await waitFor(() => {
      if (result.current.list.status !== "ready") throw new Error("not ready");
      expect(result.current.list.models.categories.transformer.active).toBe("Sulphur-2-base-distil-Q4_K_M");
    });
  });

  it("loadPipeline() surfaces a dedicated busy state for 409 JOB_BUSY", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    const apiClient = createApiClient(bridge);
    // Put the mock bridge's single job slot in flight so the pipeline-load
    // fixture's active-job check trips, mirroring the real
    // `context.job_store.has_active()` guard (api/pipeline.py:54-55).
    await apiClient.generate({ prompt: "busy check", width: 512, height: 320, num_frames: 49, frame_rate: 24, seed: 1 });

    const { result } = renderHook(() => useModels({ apiClient }));
    await waitFor(() => expect(result.current.list.status).toBe("ready"));

    act(() => {
      result.current.loadPipeline();
    });

    await waitFor(() => expect(result.current.load.status).toBe("busy"));
  });

  it("loadPipeline() surfaces MODEL_FILE_MISSING for a registered-but-missing entry", async () => {
    const apiClient = createApiClient(createMockBridge({ delayMs: 0 }));
    const { result } = renderHook(() => useModels({ apiClient }));
    await waitFor(() => expect(result.current.list.status).toBe("ready"));

    act(() => {
      result.current.setSelection("text_encoder", "gemma-3-12b-it-Q8_0");
    });
    act(() => {
      result.current.loadPipeline();
    });

    await waitFor(() => expect(result.current.load.status).toBe("error"));
    if (result.current.load.status !== "error") throw new Error("unreachable");
    expect(result.current.load.code).toBe("MODEL_FILE_MISSING");
  });

  it("loadPipeline() sends timeoutMs=600000 to the bridge (model swap can take minutes)", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    const requestSpy = vi.spyOn(bridge, "request");
    const apiClient = createApiClient(bridge);

    const { result } = renderHook(() => useModels({ apiClient }));
    await waitFor(() => expect(result.current.list.status).toBe("ready"));

    act(() => {
      result.current.loadPipeline();
    });
    await waitFor(() => expect(result.current.load.status).toBe("done"));

    const loadCall = requestSpy.mock.calls.find(
      ([method, params]) =>
        method === "backend.request" && (params as { path?: string }).path === "/api/v1/pipeline/load",
    );
    expect(loadCall).toBeDefined();
    expect((loadCall?.[1] as { timeoutMs?: number }).timeoutMs).toBe(600_000);
  });
});
