import { renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { createApiClient } from "../../api/client";
import { createMockBridge } from "../../bridge/mockBridge";
import { useLoras } from "./useLoras";

describe("useLoras", () => {
  it("starts loading, then resolves with the mock's style/control LoRAs", async () => {
    const apiClient = createApiClient(createMockBridge({ delayMs: 0 }));
    const { result } = renderHook(() => useLoras({ apiClient }));

    expect(result.current.status).toBe("loading");

    await waitFor(() => expect(result.current.status).toBe("ready"));
    if (result.current.status !== "ready") throw new Error("unreachable");
    expect(result.current.loras.map((l) => l.name).sort()).toEqual(
      ["LTX-2.3-Henshin", "Pixar_Toon", "canny-control", "pose-control", "depth-control", "deblur", "in-outpainting"].sort(),
    );
  });

  it("reports an error state when the backend is unreachable", async () => {
    const apiClient = createApiClient(createMockBridge({ delayMs: 0, backendUnreachable: true }));
    const { result } = renderHook(() => useLoras({ apiClient }));

    await waitFor(() => expect(result.current.status).toBe("error"));
  });

  it("keeps the server's `lora_dir` as `loraDir` (台帳 §1-104)", async () => {
    const apiClient = createApiClient(createMockBridge({ delayMs: 0 }));
    const { result } = renderHook(() => useLoras({ apiClient }));

    await waitFor(() => expect(result.current.status).toBe("ready"));
    if (result.current.status !== "ready") throw new Error("unreachable");
    expect(result.current.loraDir).toBe("models/LTX23/StyleLoRA");
  });

  it("leaves `loraDir` absent when an older server sends no `lora_dir`", async () => {
    const base = createApiClient(createMockBridge({ delayMs: 0 }));
    const apiClient = { ...base, getLoras: async () => ({ loras: (await base.getLoras()).loras }) };
    const { result } = renderHook(() => useLoras({ apiClient }));

    await waitFor(() => expect(result.current.status).toBe("ready"));
    if (result.current.status !== "ready") throw new Error("unreachable");
    expect(result.current.loraDir).toBeUndefined();
  });

  it("refetches whenever `refreshKey` changes, and only then (台帳 §1-104)", async () => {
    const base = createApiClient(createMockBridge({ delayMs: 0 }));
    const getLoras = vi.fn(() => base.getLoras());
    const apiClient = { ...base, getLoras };
    const { result, rerender } = renderHook(({ key }: { key: string }) => useLoras({ apiClient, refreshKey: key }), {
      initialProps: { key: "LTX23" },
    });

    await waitFor(() => expect(result.current.status).toBe("ready"));
    expect(getLoras).toHaveBeenCalledTimes(1);

    rerender({ key: "LTX23" });
    expect(getLoras).toHaveBeenCalledTimes(1);

    rerender({ key: "MiniMaxH3" });
    await waitFor(() => expect(getLoras).toHaveBeenCalledTimes(2));
  });

  it("refresh() re-fetches the list", async () => {
    const apiClient = createApiClient(createMockBridge({ delayMs: 0 }));
    const { result } = renderHook(() => useLoras({ apiClient }));

    await waitFor(() => expect(result.current.status).toBe("ready"));
    await result.current.refresh();
    expect(result.current.status).toBe("ready");
  });
});
