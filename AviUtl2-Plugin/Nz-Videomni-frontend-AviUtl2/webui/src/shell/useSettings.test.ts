import { act, renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { createMockBridge } from "../bridge/mockBridge";
import type { NativeBridge } from "../bridge";
import { useSettings } from "./useSettings";

describe("useSettings", () => {
  it("loads the current baseUrl via settings.get on mount", async () => {
    const bridge = createMockBridge({ delayMs: 0, baseUrl: "http://127.0.0.1:18620" });
    const { result } = renderHook(() => useSettings(bridge));

    await waitFor(() => {
      expect(result.current.load).toEqual({ status: "ready", baseUrl: "http://127.0.0.1:18620" });
    });
    expect(result.current.baseUrlInput).toBe("http://127.0.0.1:18620");
  });

  it("submit() saves a new baseUrl via settings.set and refreshes load/baseUrlInput", async () => {
    const bridge = createMockBridge({ delayMs: 0, baseUrl: "http://127.0.0.1:18620" });
    const { result } = renderHook(() => useSettings(bridge));

    await waitFor(() => expect(result.current.load.status).toBe("ready"));

    act(() => {
      result.current.setBaseUrlInput("http://192.168.1.20:18620");
    });

    await act(async () => {
      await result.current.submit();
    });

    expect(result.current.save).toEqual({ status: "idle" });
    expect(result.current.load).toEqual({ status: "ready", baseUrl: "http://192.168.1.20:18620" });

    await expect(bridge.request("settings.get", {})).resolves.toEqual({ baseUrl: "http://192.168.1.20:18620" });
  });

  it("submit() with an invalid URL surfaces a BAD_REQUEST save error and leaves load untouched", async () => {
    const bridge = createMockBridge({ delayMs: 0, baseUrl: "http://127.0.0.1:18620" });
    const { result } = renderHook(() => useSettings(bridge));

    await waitFor(() => expect(result.current.load.status).toBe("ready"));

    act(() => {
      result.current.setBaseUrlInput("not a url");
    });

    await act(async () => {
      await expect(result.current.submit()).rejects.toBeTruthy();
    });

    expect(result.current.save).toMatchObject({ status: "error", code: "BAD_REQUEST" });
    expect(result.current.load).toEqual({ status: "ready", baseUrl: "http://127.0.0.1:18620" });
  });

  it("reports a load error when settings.get itself fails", async () => {
    const bridge = {
      request: () => Promise.reject(new Error("transport down")),
      on: () => () => {},
    } as unknown as NativeBridge;
    const { result } = renderHook(() => useSettings(bridge));

    await waitFor(() => {
      expect(result.current.load).toMatchObject({ status: "error" });
    });
  });
});
