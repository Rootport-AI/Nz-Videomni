import { renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it } from "vitest";
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

  it("refresh() re-fetches the list", async () => {
    const apiClient = createApiClient(createMockBridge({ delayMs: 0 }));
    const { result } = renderHook(() => useLoras({ apiClient }));

    await waitFor(() => expect(result.current.status).toBe("ready"));
    await result.current.refresh();
    expect(result.current.status).toBe("ready");
  });
});
