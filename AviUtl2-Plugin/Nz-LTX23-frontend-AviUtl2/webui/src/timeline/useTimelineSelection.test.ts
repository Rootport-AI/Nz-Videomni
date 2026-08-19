import { act, renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { createMockBridge } from "../bridge/mockBridge";
import { TIMELINE_MENU_INVOKED_EVENT } from "../bridge";
import type { TimelineMenuInvokedData } from "../bridge";
import { useTimelineSelection } from "./useTimelineSelection";

describe("useTimelineSelection", () => {
  it("pulls the initial selection on mount", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    const { result } = renderHook(() => useTimelineSelection({ nativeBridge: bridge }));

    await waitFor(() => {
      expect(result.current.selection).not.toBeNull();
    });
    expect(result.current.selection?.selected).toHaveLength(1);
    expect(result.current.error).toBeNull();
  });

  it("surfaces an error when getSelection rejects", async () => {
    const bridge = createMockBridge({ delayMs: 0, failGetSelection: true });
    const { result } = renderHook(() => useTimelineSelection({ nativeBridge: bridge }));

    await waitFor(() => {
      expect(result.current.error).not.toBeNull();
    });
    expect(result.current.selection).toBeNull();
  });

  it("refresh() re-queries and returns the fresh snapshot", async () => {
    const bridge = createMockBridge({ delayMs: 0, selection: { cursorFrame: 42 } });
    const { result } = renderHook(() => useTimelineSelection({ nativeBridge: bridge }));

    await waitFor(() => {
      expect(result.current.selection).not.toBeNull();
    });

    let returned: Awaited<ReturnType<typeof result.current.refresh>> = null;
    await act(async () => {
      returned = await result.current.refresh();
    });
    expect(returned).not.toBeNull();
    expect(result.current.selection?.cursorFrame).toBe(42);
  });

  it("updates selection and lastMenuAction when a menuInvoked event is emitted", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    const { result } = renderHook(() => useTimelineSelection({ nativeBridge: bridge }));

    await waitFor(() => {
      expect(result.current.selection).not.toBeNull();
    });

    const pushed: TimelineMenuInvokedData = {
      action: "cutout",
      selection: {
        hasRange: true,
        rangeStart: 5,
        rangeEnd: 65,
        selected: [
          { layer: 4, frameStart: 5, frameEnd: 65, effectName: "動画ファイル", filePath: null, objectName: "pushed", textContent: null, mediaWidth: 0, mediaHeight: 0, mediaDurationSec: 0 },
        ],
        cursorFrame: 65,
        cursorLayer: 4,
        rate: 30,
        scale: 1,
        sampleRate: 44100,
      },
    };

    act(() => {
      bridge.emit(TIMELINE_MENU_INVOKED_EVENT, pushed);
    });

    expect(result.current.lastMenuAction).toBe("cutout");
    expect(result.current.selection?.rangeStart).toBe(5);
    expect(result.current.selection?.selected[0]?.objectName).toBe("pushed");
  });
});
