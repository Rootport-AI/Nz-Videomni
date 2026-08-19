import { act, renderHook, waitFor } from "@testing-library/react";
import type { DragEvent } from "react";
import { describe, expect, it, vi } from "vitest";
import type { NativeBridge } from "../bridge";
import { createMockBridge } from "../bridge/mockBridge";
import { useFileDrop } from "./useFileDrop";

/** Minimal stand-in for a React `DragEvent` — `useFileDrop` only ever reads
 * `preventDefault()` and `dataTransfer.files`, so a full jsdom `DragEvent`
 * (which doesn't implement a real `DataTransfer` anyway) isn't needed. */
function makeEvent(files: File[]): DragEvent<HTMLElement> {
  return {
    preventDefault: vi.fn(),
    dataTransfer: { files },
  } as unknown as DragEvent<HTMLElement>;
}

describe("useFileDrop", () => {
  it("dragenter/dragover set isDragOver; dragleave clears it", () => {
    const mockBridge = createMockBridge({ delayMs: 0 });
    const onFile = vi.fn();
    const { result } = renderHook(() => useFileDrop({ accept: ["mp4"], onFile, nativeBridge: mockBridge }));

    act(() => result.current.handlers.onDragEnter(makeEvent([])));
    expect(result.current.isDragOver).toBe(true);

    act(() => result.current.handlers.onDragLeave(makeEvent([])));
    expect(result.current.isDragOver).toBe(false);
  });

  it("uses only the first dropped file when several are dropped at once", async () => {
    const mockBridge = createMockBridge({ delayMs: 0 });
    const onFile = vi.fn();
    const { result } = renderHook(() => useFileDrop({ accept: ["mp4"], onFile, nativeBridge: mockBridge }));

    const first = new File([], "one.mp4");
    const second = new File([], "two.mp4");
    act(() => result.current.handlers.onDrop(makeEvent([first, second])));

    await waitFor(() => expect(onFile).toHaveBeenCalledTimes(1));
    expect(onFile).toHaveBeenCalledWith(expect.stringContaining("one.mp4"), "one.mp4");
  });

  it("clears isDragOver on drop and calls onFile for an accepted extension", async () => {
    const mockBridge = createMockBridge({ delayMs: 0 });
    const onFile = vi.fn();
    const { result } = renderHook(() => useFileDrop({ accept: ["mp4", "mov"], onFile, nativeBridge: mockBridge }));

    act(() => result.current.handlers.onDragEnter(makeEvent([])));
    expect(result.current.isDragOver).toBe(true);

    act(() => result.current.handlers.onDrop(makeEvent([new File([], "clip.mov")])));
    expect(result.current.isDragOver).toBe(false);

    await waitFor(() => expect(onFile).toHaveBeenCalledWith(expect.stringContaining("clip.mov"), "clip.mov"));
    expect(result.current.error).toBeNull();
  });

  it("sets error 'unsupported' for an extension not in accept, without calling onFile", async () => {
    const mockBridge = createMockBridge({ delayMs: 0 });
    const onFile = vi.fn();
    const { result } = renderHook(() => useFileDrop({ accept: ["mp4"], onFile, nativeBridge: mockBridge }));

    act(() => result.current.handlers.onDrop(makeEvent([new File([], "notes.txt")])));

    await waitFor(() => expect(result.current.error).toBe("unsupported"));
    expect(onFile).not.toHaveBeenCalled();
  });

  it("sets error 'unsupported' for a file with no extension at all", async () => {
    const mockBridge = createMockBridge({ delayMs: 0 });
    const onFile = vi.fn();
    const { result } = renderHook(() => useFileDrop({ accept: ["mp4"], onFile, nativeBridge: mockBridge }));

    act(() => result.current.handlers.onDrop(makeEvent([new File([], "no_extension")])));

    await waitFor(() => expect(result.current.error).toBe("unsupported"));
    expect(onFile).not.toHaveBeenCalled();
  });

  it("sets error 'resolveFailed' when ui.resolveDroppedFiles rejects", async () => {
    const mockBridge = createMockBridge({ delayMs: 0, failResolveDroppedFiles: true });
    const onFile = vi.fn();
    const { result } = renderHook(() => useFileDrop({ accept: ["mp4"], onFile, nativeBridge: mockBridge }));

    act(() => result.current.handlers.onDrop(makeEvent([new File([], "clip.mp4")])));

    await waitFor(() => expect(result.current.error).toBe("resolveFailed"));
    expect(onFile).not.toHaveBeenCalled();
  });

  it("sets error 'resolveFailed' when resolution yields no usable file (empty files[])", async () => {
    const mockBridge = createMockBridge({ delayMs: 0 });
    const onFile = vi.fn();
    const { result } = renderHook(() => useFileDrop({ accept: ["mp4"], onFile, nativeBridge: mockBridge }));

    // A File-like object with no `.name` resolves to an empty files[] in the
    // mock (mirrors a drop native genuinely couldn't resolve to a path).
    act(() => result.current.handlers.onDrop(makeEvent([{} as File])));

    await waitFor(() => expect(result.current.error).toBe("resolveFailed"));
    expect(onFile).not.toHaveBeenCalled();
  });

  it("disabled ignores dragenter/dragover/drop entirely", async () => {
    const mockBridge = createMockBridge({ delayMs: 0 });
    const onFile = vi.fn();
    const { result } = renderHook(() => useFileDrop({ accept: ["mp4"], onFile, disabled: true, nativeBridge: mockBridge }));

    act(() => result.current.handlers.onDragEnter(makeEvent([])));
    expect(result.current.isDragOver).toBe(false);

    act(() => result.current.handlers.onDrop(makeEvent([new File([], "clip.mp4")])));
    // Give any (incorrectly fired) async resolution a chance to land before
    // asserting nothing happened.
    await new Promise((r) => setTimeout(r, 10));
    expect(onFile).not.toHaveBeenCalled();
    expect(result.current.error).toBeNull();
  });

  // MINOR fix (post-review): a naive boolean (rather than a counter) flickers
  // the highlight off/on whenever the pointer crosses from the zone onto one
  // of its own children (e.g. the choose/change/clear buttons `DropZone`
  // wraps), because the child's dragenter fires and the zone's dragleave
  // fires for the same pointer movement. Real browsers fire the child's
  // dragenter before the parent's dragleave, so the sequence below (two
  // enters, then two leaves) mirrors that ordering.
  it("nested enter/leave pairs (crossing a child element) don't flicker isDragOver off", () => {
    const mockBridge = createMockBridge({ delayMs: 0 });
    const onFile = vi.fn();
    const { result } = renderHook(() => useFileDrop({ accept: ["mp4"], onFile, nativeBridge: mockBridge }));

    act(() => result.current.handlers.onDragEnter(makeEvent([]))); // enters the outer zone
    expect(result.current.isDragOver).toBe(true);

    act(() => result.current.handlers.onDragEnter(makeEvent([]))); // enters a child element
    expect(result.current.isDragOver).toBe(true);

    act(() => result.current.handlers.onDragLeave(makeEvent([]))); // leaves the outer zone's own boundary
    expect(result.current.isDragOver).toBe(true); // still true: counter is 1, not 0

    act(() => result.current.handlers.onDragLeave(makeEvent([]))); // leaves the child (and the whole zone)
    expect(result.current.isDragOver).toBe(false);
  });

  it("drop resets the enter/leave counter, so a stray leave afterward is a no-op", () => {
    const mockBridge = createMockBridge({ delayMs: 0 });
    const onFile = vi.fn();
    const { result } = renderHook(() => useFileDrop({ accept: ["mp4"], onFile, nativeBridge: mockBridge }));

    act(() => result.current.handlers.onDragEnter(makeEvent([])));
    act(() => result.current.handlers.onDragEnter(makeEvent([])));
    act(() => result.current.handlers.onDrop(makeEvent([new File([], "clip.mp4")])));
    expect(result.current.isDragOver).toBe(false);

    // A leftover dragleave (e.g. from a child element) after the drop must
    // not push the counter negative and corrupt the next drag's count.
    act(() => result.current.handlers.onDragLeave(makeEvent([])));
    act(() => result.current.handlers.onDragEnter(makeEvent([])));
    expect(result.current.isDragOver).toBe(true);
  });

  // Bug fix (owner real-device report, 2026-07-18): `clear()` on the
  // caller's own state (e.g. GenerationForm's "Remove audio") is entirely
  // independent of this hook's `error` — without `resetErrorSignal`, a
  // rejected drop's "unsupported" message would survive a Clear click since
  // `error` otherwise only resets at the START of the next drop.
  it("resetErrorSignal changing clears a set error", async () => {
    const mockBridge = createMockBridge({ delayMs: 0 });
    const onFile = vi.fn();
    const { result, rerender } = renderHook(
      ({ resetErrorSignal }: { resetErrorSignal: number }) =>
        useFileDrop({ accept: ["mp4"], onFile, nativeBridge: mockBridge, resetErrorSignal }),
      { initialProps: { resetErrorSignal: 0 } },
    );

    act(() => result.current.handlers.onDrop(makeEvent([new File([], "notes.txt")])));
    await waitFor(() => expect(result.current.error).toBe("unsupported"));

    rerender({ resetErrorSignal: 1 });
    expect(result.current.error).toBeNull();
  });

  it("resetErrorSignal identity change with no prior error is a harmless no-op", () => {
    const mockBridge = createMockBridge({ delayMs: 0 });
    const onFile = vi.fn();
    const { result, rerender } = renderHook(
      ({ resetErrorSignal }: { resetErrorSignal: number }) =>
        useFileDrop({ accept: ["mp4"], onFile, nativeBridge: mockBridge, resetErrorSignal }),
      { initialProps: { resetErrorSignal: 0 } },
    );

    expect(result.current.error).toBeNull();
    rerender({ resetErrorSignal: 1 });
    expect(result.current.error).toBeNull();
  });

  // MINOR fix (post-review): a slow ui.resolveDroppedFiles round trip must
  // never call back into a form that has since unmounted (e.g. the user
  // switched screens mid-resolve).
  it("does not call onFile or set an error after unmount while resolution is still pending", async () => {
    let resolveRequest!: (value: { files: Array<{ filePath: string; fileName: string }> }) => void;
    const pending = new Promise<{ files: Array<{ filePath: string; fileName: string }> }>((resolve) => {
      resolveRequest = resolve;
    });
    const onFile = vi.fn();
    const nativeBridge = {
      request: () => Promise.reject(new Error("unused in this test")),
      requestWithFiles: () => pending,
      on: () => () => {},
    } as unknown as NativeBridge;

    const { result, unmount } = renderHook(() => useFileDrop({ accept: ["mp4"], onFile, nativeBridge }));

    act(() => result.current.handlers.onDrop(makeEvent([new File([], "clip.mp4")])));

    unmount();
    resolveRequest({ files: [{ filePath: "C:\\x\\clip.mp4", fileName: "clip.mp4" }] });
    await new Promise((r) => setTimeout(r, 10));

    expect(onFile).not.toHaveBeenCalled();
  });
});
