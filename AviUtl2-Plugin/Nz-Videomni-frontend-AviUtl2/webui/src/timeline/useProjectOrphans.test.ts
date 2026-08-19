import { act, renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { createMockBridge } from "../bridge/mockBridge";
import { TIMELINE_PROJECT_LOADED_EVENT } from "../bridge";
import { useProjectOrphans } from "./useProjectOrphans";

const ORPHANS = [
  { jobId: "gone-1", layer: 3, frame: 100 },
  { jobId: "gone-2", layer: 4, frame: 200 },
];

describe("useProjectOrphans", () => {
  it("starts empty and un-scanned, with no scan on mount by default", async () => {
    const scan = vi.fn(async () => ({ orphans: ORPHANS }));
    const bridge = createMockBridge({ delayMs: 0 });
    const { result } = renderHook(() => useProjectOrphans({ nativeBridge: bridge, scan }));

    expect(result.current.orphans).toEqual([]);
    expect(result.current.scanned).toBe(false);
    // Give any stray effect a tick; scan must still not have run.
    await act(async () => {
      await Promise.resolve();
    });
    expect(scan).not.toHaveBeenCalled();
  });

  it("scans on a projectLoaded event and holds the orphan list", async () => {
    const scan = vi.fn(async () => ({ orphans: ORPHANS }));
    const bridge = createMockBridge({ delayMs: 0 });
    const { result } = renderHook(() => useProjectOrphans({ nativeBridge: bridge, scan }));

    act(() => {
      bridge.emit(TIMELINE_PROJECT_LOADED_EVENT, {});
    });

    await waitFor(() => {
      expect(result.current.orphans).toHaveLength(2);
    });
    expect(scan).toHaveBeenCalledTimes(1);
    expect(result.current.scanned).toBe(true);
    expect(result.current.error).toBeNull();
    expect(result.current.orphans[0]?.jobId).toBe("gone-1");
  });

  it("reports no orphans (scanned=true, empty list) when the scan finds none", async () => {
    const scan = vi.fn(async () => ({ orphans: [] }));
    const bridge = createMockBridge({ delayMs: 0 });
    const { result } = renderHook(() => useProjectOrphans({ nativeBridge: bridge, scan }));

    act(() => {
      bridge.emit(TIMELINE_PROJECT_LOADED_EVENT, {});
    });

    await waitFor(() => {
      expect(result.current.scanned).toBe(true);
    });
    expect(result.current.orphans).toEqual([]);
  });

  it("surfaces a scan failure as error and an empty list", async () => {
    const scan = vi.fn(async () => {
      throw new Error("scan boom");
    });
    const bridge = createMockBridge({ delayMs: 0 });
    const { result } = renderHook(() => useProjectOrphans({ nativeBridge: bridge, scan }));

    act(() => {
      bridge.emit(TIMELINE_PROJECT_LOADED_EVENT, {});
    });

    await waitFor(() => {
      expect(result.current.error).toBe("scan boom");
    });
    expect(result.current.orphans).toEqual([]);
    expect(result.current.scanned).toBe(true);
  });

  it("refresh() runs an on-demand scan and returns the list", async () => {
    const scan = vi.fn(async () => ({ orphans: ORPHANS }));
    const bridge = createMockBridge({ delayMs: 0 });
    const { result } = renderHook(() => useProjectOrphans({ nativeBridge: bridge, scan }));

    let returned: Awaited<ReturnType<typeof result.current.refresh>> = [];
    await act(async () => {
      returned = await result.current.refresh();
    });
    expect(returned).toHaveLength(2);
    expect(result.current.orphans).toHaveLength(2);
  });

  it("clear() empties the orphan list after the user acts on it", async () => {
    const scan = vi.fn(async () => ({ orphans: ORPHANS }));
    const bridge = createMockBridge({ delayMs: 0 });
    const { result } = renderHook(() => useProjectOrphans({ nativeBridge: bridge, scan }));

    await act(async () => {
      await result.current.refresh();
    });
    expect(result.current.orphans).toHaveLength(2);

    act(() => {
      result.current.clear();
    });
    expect(result.current.orphans).toEqual([]);
  });

  it("scanOnMount runs one scan without an event", async () => {
    const scan = vi.fn(async () => ({ orphans: ORPHANS }));
    const bridge = createMockBridge({ delayMs: 0 });
    const { result } = renderHook(() =>
      useProjectOrphans({ nativeBridge: bridge, scan, scanOnMount: true }),
    );

    await waitFor(() => {
      expect(result.current.orphans).toHaveLength(2);
    });
    expect(scan).toHaveBeenCalledTimes(1);
  });

  it("defaults to a real timeline.scanProvisionals bridge call when scan is omitted", async () => {
    const bridge = createMockBridge({ delayMs: 0, provisionalOrphans: ORPHANS });
    const spy = vi.spyOn(bridge, "request");
    const { result } = renderHook(() => useProjectOrphans({ nativeBridge: bridge }));

    act(() => {
      bridge.emit(TIMELINE_PROJECT_LOADED_EVENT, {});
    });

    await waitFor(() => {
      expect(result.current.orphans).toHaveLength(2);
    });
    expect(spy).toHaveBeenCalledWith("timeline.scanProvisionals", {});
  });
});
