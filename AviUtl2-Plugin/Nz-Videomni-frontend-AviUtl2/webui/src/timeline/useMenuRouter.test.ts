import { act, renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { createMockBridge } from "../bridge/mockBridge";
import { TIMELINE_MENU_INVOKED_EVENT } from "../bridge";
import type { TimelineMenuInvokedData } from "../bridge";
import { useMenuRouter } from "./useMenuRouter";
import type { TimelineSelection } from "./menuSelection";

function selection(overrides: Partial<TimelineSelection> = {}): TimelineSelection {
  return {
    hasRange: true,
    rangeStart: 0,
    rangeEnd: 60,
    selected: [
      { layer: 1, frameStart: 0, frameEnd: 60, effectName: "動画ファイル", filePath: "C:\\v\\a.mp4", objectName: "a", textContent: null, mediaWidth: 1920, mediaHeight: 1080, mediaDurationSec: 0 },
    ],
    cursorFrame: 30,
    cursorLayer: 1,
    rate: 30,
    scale: 1,
    sampleRate: 44100,
    ...overrides,
  };
}

function menuEvent(action: string, sel = selection()): TimelineMenuInvokedData {
  return { action, selection: sel };
}

describe("useMenuRouter", () => {
  it("routes a known action and exposes the command + fires onRoute", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    const onRoute = vi.fn();
    const { result } = renderHook(() => useMenuRouter({ nativeBridge: bridge, onRoute }));

    act(() => {
      bridge.emit(TIMELINE_MENU_INVOKED_EVENT, menuEvent("textToVideoHere"));
    });

    await waitFor(() => {
      expect(result.current.command).not.toBeNull();
    });
    expect(result.current.lastAction).toBe("textToVideoHere");
    expect(result.current.command?.route.targetMode).toBe("single");
    expect(result.current.command?.route.intent).toBe("text-to-video");
    expect(result.current.command?.refreshed).toBe(false);
    expect(onRoute).toHaveBeenCalledTimes(1);
    expect(onRoute.mock.calls[0]?.[0].route.action).toBe("textToVideoHere");
  });

  it("records lastAction but does not route (nor fire onRoute) for an unknown action", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    const onRoute = vi.fn();
    const onUnknownAction = vi.fn();
    const { result } = renderHook(() =>
      useMenuRouter({ nativeBridge: bridge, onRoute, onUnknownAction }),
    );

    act(() => {
      bridge.emit(TIMELINE_MENU_INVOKED_EVENT, menuEvent("bogusThing"));
    });

    await waitFor(() => {
      expect(result.current.lastAction).toBe("bogusThing");
    });
    expect(result.current.command).toBeNull();
    expect(onRoute).not.toHaveBeenCalled();
    expect(onUnknownAction).toHaveBeenCalledWith("bogusThing", expect.objectContaining({ action: "bogusThing" }));
  });

  it("completes missing file paths via getSelection for a needsSelection action", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    const spy = vi.spyOn(bridge, "request");
    const { result } = renderHook(() => useMenuRouter({ nativeBridge: bridge }));

    // extendVideo needsSelection; the pushed snapshot has a null filePath.
    const staleSel = selection({
      selected: [
        { layer: 1, frameStart: 0, frameEnd: 60, effectName: "動画ファイル", filePath: null, objectName: "stale", textContent: null, mediaWidth: 0, mediaHeight: 0, mediaDurationSec: 0 },
      ],
    });

    act(() => {
      bridge.emit(TIMELINE_MENU_INVOKED_EVENT, menuEvent("extendVideo", staleSel));
    });

    await waitFor(() => {
      expect(result.current.command?.refreshed).toBe(true);
    });
    expect(spy).toHaveBeenCalledWith("timeline.getSelection", {});
    expect(result.current.command?.selection.selected[0]?.filePath).not.toBeNull();
    expect(result.current.command?.route.targetMode).toBe("chained");
  });

  it("does not re-query getSelection for a layer command (needsSelection=false)", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    const spy = vi.spyOn(bridge, "request");
    const { result } = renderHook(() => useMenuRouter({ nativeBridge: bridge }));

    // Even with a null filePath, a layer command must not trigger a re-query.
    const sel = selection({
      selected: [
        { layer: 1, frameStart: 0, frameEnd: 60, effectName: "動画ファイル", filePath: null, objectName: "x", textContent: null, mediaWidth: 0, mediaHeight: 0, mediaDurationSec: 0 },
      ],
    });

    act(() => {
      bridge.emit(TIMELINE_MENU_INVOKED_EVENT, menuEvent("imageFromCurrentFrame", sel));
    });

    await waitFor(() => {
      expect(result.current.command).not.toBeNull();
    });
    expect(result.current.command?.refreshed).toBe(false);
    expect(spy).not.toHaveBeenCalledWith("timeline.getSelection", {});
  });

  it("surfaces a getSelection failure as error and leaves command unchanged", async () => {
    const bridge = createMockBridge({ delayMs: 0, failGetSelection: true });
    const { result } = renderHook(() => useMenuRouter({ nativeBridge: bridge }));

    const staleSel = selection({
      selected: [
        { layer: 1, frameStart: 0, frameEnd: 60, effectName: "動画ファイル", filePath: null, objectName: "stale", textContent: null, mediaWidth: 0, mediaHeight: 0, mediaDurationSec: 0 },
      ],
    });

    act(() => {
      bridge.emit(TIMELINE_MENU_INVOKED_EVENT, menuEvent("referenceVideo", staleSel));
    });

    await waitFor(() => {
      expect(result.current.error).not.toBeNull();
    });
    expect(result.current.command).toBeNull();
  });

  it("clear() drops the current command", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    const { result } = renderHook(() => useMenuRouter({ nativeBridge: bridge }));

    act(() => {
      bridge.emit(TIMELINE_MENU_INVOKED_EVENT, menuEvent("textToVideoHere"));
    });
    await waitFor(() => {
      expect(result.current.command).not.toBeNull();
    });

    act(() => {
      result.current.clear();
    });
    expect(result.current.command).toBeNull();
  });
});
