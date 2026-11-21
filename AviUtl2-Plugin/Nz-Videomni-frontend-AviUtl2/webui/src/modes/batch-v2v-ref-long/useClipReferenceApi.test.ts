import { act, renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { NativeBridge } from "../../bridge";
import { useClipReferenceApi } from "./useClipReferenceApi";
import { RULES, emptyRefs, makeForm, withCard } from "./fakeForm.testSupport";
import type { ResolvedClip } from "./types";

function bridgeWith(request: (method: string, params: { filePath: string }) => Promise<unknown>) {
  const spy = vi.fn(request);
  return { bridge: { request: spy } as unknown as NativeBridge, spy };
}

function clipWith(...cards: [kind: "picture" | "video" | "audio", slot: number, path: string][]): ResolvedClip {
  let refs = emptyRefs();
  for (const [kind, slot, path] of cards) refs = withCard(refs, kind, slot, path, kind === "picture" ? null : 5);
  return { promptMode: "add", prompt: "", refs };
}

describe("useClipReferenceApi", () => {
  it("attach / pick / clear go to the form's clip operations; nothing is uploaded", async () => {
    const { bridge, spy } = bridgeWith(async () => ({ dataUrl: null }));
    const form = makeForm();
    const { result } = renderHook(() => useClipReferenceApi(form, 2, 1, bridge));

    await act(async () => {
      await result.current.attach("video", 2, "C:\\v.mp4", "v.mp4");
      await result.current.pick("audio", 1);
      result.current.clear("picture", 3);
    });
    expect(form.attachClipFile).toHaveBeenCalledWith(2, 1, "video", 2, "C:\\v.mp4", "v.mp4");
    expect(form.pickClipFile).toHaveBeenCalledWith(2, 1, "audio", 1);
    expect(form.clearClipSlot).toHaveBeenCalledWith(2, 1, "picture", 3);
    expect(spy).not.toHaveBeenCalledWith("backend.uploadFile", expect.anything());
    expect(result.current.attachToFirstEmpty("picture", "C:\\x.png", "x.png")).toBeNull();
  });

  it("state, summary and rules come from the row's clip", () => {
    const { bridge } = bridgeWith(async () => ({ dataUrl: null }));
    const form = makeForm({ clips: { "0:0": clipWith(["picture", 1, "C:\\a.png"], ["video", 1, "C:\\b.mp4"]) } });
    const { result } = renderHook(() => useClipReferenceApi(form, 0, 0, bridge));
    expect(result.current.rules).toBe(RULES);
    expect(result.current.state.enabled).toBe(true);
    expect(result.current.state.slots.picture[0]!.filePath).toBe("C:\\a.png");
    expect(result.current.summary.filled.total).toBe(2);
  });

  it("fetches the thumbnails of the ready picture/video cards on mount, never the audio ones", async () => {
    const { bridge, spy } = bridgeWith(async (_method, params) => ({ dataUrl: `data:${params.filePath}` }));
    const form = makeForm({
      clips: { "0:0": clipWith(["picture", 1, "C:\\a.png"], ["video", 2, "C:\\b.mp4"], ["audio", 1, "C:\\c.wav"]) },
    });
    const { result } = renderHook(() => useClipReferenceApi(form, 0, 0, bridge));

    await waitFor(() => expect(result.current.state.slots.video[1]!.thumbnailDataUrl).toBe("data:C:\\b.mp4"));
    expect(result.current.state.slots.picture[0]!.thumbnailDataUrl).toBe("data:C:\\a.png");
    expect(spy).toHaveBeenCalledWith("ui.makeThumbnail", { filePath: "C:\\a.png" });
    expect(spy).toHaveBeenCalledWith("ui.makeThumbnail", { filePath: "C:\\b.mp4" });
    expect(spy).not.toHaveBeenCalledWith("ui.makeThumbnail", { filePath: "C:\\c.wav" });
    expect(spy).toHaveBeenCalledTimes(2);
    expect(result.current.state.slots.audio[0]!.thumbnailDataUrl).toBeNull();
  });

  it("asks once per file, and again only for a card that is new", async () => {
    const { bridge, spy } = bridgeWith(async (_method, params) => ({ dataUrl: `data:${params.filePath}` }));
    let form = makeForm({ clips: { "0:0": clipWith(["picture", 1, "C:\\a.png"]) } });
    const { result, rerender } = renderHook(() => useClipReferenceApi(form, 0, 0, bridge));
    await waitFor(() => expect(result.current.state.slots.picture[0]!.thumbnailDataUrl).toBe("data:C:\\a.png"));
    rerender();
    expect(spy).toHaveBeenCalledTimes(1);

    form = makeForm({ clips: { "0:0": clipWith(["picture", 1, "C:\\a.png"], ["picture", 2, "C:\\d.png"]) } });
    rerender();
    await waitFor(() => expect(result.current.state.slots.picture[1]!.thumbnailDataUrl).toBe("data:C:\\d.png"));
    expect(spy).toHaveBeenCalledTimes(2);
  });
});
