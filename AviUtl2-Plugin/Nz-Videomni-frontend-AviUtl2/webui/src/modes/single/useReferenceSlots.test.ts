import { act, renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { BridgeMethod, NativeBridge, ParamsOf, ResultOf } from "../../bridge";
import { createMockBridge, createMockFs } from "../../bridge/mockBridge";
import type { MockBridgeOptions } from "../../bridge/mockBridge";
import { useSourceUpload } from "../chained/useSourceUpload";
import type { ReferenceRules } from "./referenceRules";
import { useReferenceA2vSync, useReferenceSlots } from "./useReferenceSlots";

// §1-107 (UI first): the Reference accordion's state hook and its A2V sync.

const RULES: ReferenceRules = {
  maxImages: 9,
  maxVideos: 3,
  maxAudios: 3,
  maxTotal: 12,
  videoTotalSec: 15,
  audioTotalSec: 15,
  clipMinSec: 2,
};

const DIR = "C:\\m";

/** A mock whose virtual folder gives each clip its own length. */
function bridgeWithClips(extra: MockBridgeOptions = {}) {
  return createMockBridge({
    delayMs: 0,
    fs: createMockFs({
      folders: {
        [DIR]: [
          { name: "a.mp4", sizeBytes: 1, mtimeMs: 0, durationSec: 7 },
          { name: "b.wav", sizeBytes: 1, mtimeMs: 0, durationSec: 4 },
        ],
      },
    }),
    ...extra,
  });
}

/** Wraps a mock so selected methods answer from `overrides` (or are held). */
function wrap(
  base: NativeBridge,
  overrides: Partial<Record<BridgeMethod, (params: unknown) => Promise<unknown>>>,
): NativeBridge {
  return {
    ...base,
    async request<M extends BridgeMethod>(method: M, params: ParamsOf<M>): Promise<ResultOf<M>> {
      const override = overrides[method];
      if (override) return (await override(params)) as ResultOf<M>;
      return base.request(method, params);
    },
  };
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((r) => {
    resolve = r;
  });
  return { promise, resolve };
}

describe("useReferenceSlots — attach", () => {
  it("a picture: upload as image + thumbnail, no duration probe", async () => {
    const bridge = bridgeWithClips();
    const spy = vi.spyOn(bridge, "request");
    const { result } = renderHook(() => useReferenceSlots(RULES, { nativeBridge: bridge }));

    await act(async () => {
      await result.current.attach("picture", 2, `${DIR}\\p.png`, "p.png");
    });

    const slot = result.current.state.slots.picture[1]!;
    expect(slot.status).toBe("ready");
    expect(slot.id).toMatch(/^mock-image-/);
    expect(slot.thumbnailDataUrl).toMatch(/^data:/);
    expect(slot.durationSec).toBeNull();
    expect(slot.fileName).toBe("p.png");
    expect(spy).toHaveBeenCalledWith("backend.uploadFile", { kind: "image", filePath: `${DIR}\\p.png` });
    expect(spy).toHaveBeenCalledWith("ui.makeThumbnail", { filePath: `${DIR}\\p.png` });
    expect(spy).not.toHaveBeenCalledWith("fs.probeMediaInfo", expect.anything());
  });

  it("a video: upload + thumbnail + probeMediaInfo run together", async () => {
    const bridge = bridgeWithClips();
    const spy = vi.spyOn(bridge, "request");
    const { result } = renderHook(() => useReferenceSlots(RULES, { nativeBridge: bridge }));

    await act(async () => {
      await result.current.attach("video", 1, `${DIR}\\a.mp4`, "a.mp4");
    });

    const slot = result.current.state.slots.video[0]!;
    expect(slot.status).toBe("ready");
    expect(slot.id).toMatch(/^mock-video-/);
    expect(slot.durationSec).toBe(7);
    expect(slot.thumbnailDataUrl).not.toBeNull();
    expect(slot.trimFailed).toBe(false);
    const methods = spy.mock.calls.map(([m]) => m);
    expect(methods).toEqual(expect.arrayContaining(["backend.uploadFile", "ui.makeThumbnail", "fs.probeMediaInfo"]));
  });

  it("the three calls are issued before any of them settles", async () => {
    const base = bridgeWithClips();
    const gate = deferred<void>();
    const started: string[] = [];
    const held = (method: BridgeMethod) => async (params: unknown) => {
      started.push(method);
      await gate.promise;
      return base.request(method, params as never);
    };
    const bridge = wrap(base, {
      "backend.uploadFile": held("backend.uploadFile"),
      "ui.makeThumbnail": held("ui.makeThumbnail"),
      "fs.probeMediaInfo": held("fs.probeMediaInfo"),
    });
    const { result } = renderHook(() => useReferenceSlots(RULES, { nativeBridge: bridge }));

    let pending!: Promise<void>;
    act(() => {
      pending = result.current.attach("video", 1, `${DIR}\\a.mp4`, "a.mp4");
    });
    await waitFor(() => expect(started).toHaveLength(3));
    expect(result.current.state.slots.video[0]!.status).toBe("uploading");

    await act(async () => {
      gate.resolve();
      await pending;
    });
    expect(result.current.state.slots.video[0]!.status).toBe("ready");
  });

  it("a video trim that the server did not apply sets trimFailed", async () => {
    const bridge = bridgeWithClips();
    const { result } = renderHook(() => useReferenceSlots(RULES, { nativeBridge: bridge }));

    await act(async () => {
      // The mock reports `trimmed: true` only when BOTH trim keys arrive.
      await result.current.attach("video", 1, `${DIR}\\a.mp4`, "a.mp4", { query: { trim_start_sec: "1.000" } });
      await result.current.attach("video", 2, `${DIR}\\a.mp4`, "a.mp4", {
        query: { trim_start_sec: "1.000", trim_duration_sec: "3.000" },
      });
    });

    expect(result.current.state.slots.video[0]!.trimFailed).toBe(true);
    expect(result.current.state.slots.video[1]!.trimFailed).toBe(false);
  });

  it("opts.durationSec is used instead of probing", async () => {
    const bridge = bridgeWithClips();
    const spy = vi.spyOn(bridge, "request");
    const { result } = renderHook(() => useReferenceSlots(RULES, { nativeBridge: bridge }));

    await act(async () => {
      await result.current.attach("video", 1, `${DIR}\\a.mp4`, "a.mp4", { durationSec: 3.5 });
    });

    expect(result.current.state.slots.video[0]!.durationSec).toBe(3.5);
    expect(spy).not.toHaveBeenCalledWith("fs.probeMediaInfo", expect.anything());
  });

  it("an audio: probeMediaInfo's length, no thumbnail", async () => {
    const bridge = bridgeWithClips();
    const spy = vi.spyOn(bridge, "request");
    const { result } = renderHook(() => useReferenceSlots(RULES, { nativeBridge: bridge }));

    await act(async () => {
      await result.current.attach("audio", 1, `${DIR}\\b.wav`, "b.wav");
    });

    const slot = result.current.state.slots.audio[0]!;
    expect(slot.status).toBe("ready");
    expect(slot.id).toMatch(/^mock-audio-/);
    expect(slot.durationSec).toBe(4);
    expect(slot.thumbnailDataUrl).toBeNull();
    expect(spy).not.toHaveBeenCalledWith("ui.makeThumbnail", expect.anything());
  });

  it("an audio that probeMediaInfo cannot measure falls back to the wav header", async () => {
    const base = createMockBridge({ delayMs: 0 });
    const bridge = wrap(base, {
      "fs.probeMediaInfo": async () => ({ durationSec: 0, width: 0, height: 0 }),
      "fs.probeAudioDuration": async (params) =>
        (params as { filePath: string }).filePath.endsWith(".wav")
          ? { durationSec: 6, isWav: true }
          : { durationSec: 0, isWav: false },
    });
    const { result } = renderHook(() => useReferenceSlots(RULES, { nativeBridge: bridge }));

    await act(async () => {
      await result.current.attach("audio", 1, `${DIR}\\x.wav`, "x.wav");
      await result.current.attach("audio", 2, `${DIR}\\x.mp3`, "x.mp3");
      await result.current.attach("video", 1, `${DIR}\\x.mp4`, "x.mp4");
    });

    expect(result.current.state.slots.audio[0]!.durationSec).toBe(6);
    // Neither measure answered: the length is unknown.
    expect(result.current.state.slots.audio[1]!.durationSec).toBeNull();
    expect(result.current.state.slots.video[0]!.durationSec).toBeNull();
  });

  it("an upload failure leaves the slot in 'error' with the bridge's code", async () => {
    const bridge = bridgeWithClips({ failUploadFile: "BACKEND_UNREACHABLE" });
    const { result } = renderHook(() => useReferenceSlots(RULES, { nativeBridge: bridge }));

    await act(async () => {
      await result.current.attach("video", 1, `${DIR}\\a.mp4`, "a.mp4");
    });

    const slot = result.current.state.slots.video[0]!;
    expect(slot.status).toBe("error");
    expect(slot.errorCode).toBe("BACKEND_UNREACHABLE");
  });

  it("an HTTP error body leaves the slot in 'error' with its code", async () => {
    const base = createMockBridge({ delayMs: 0 });
    const bridge = wrap(base, {
      "backend.uploadFile": async () => ({ status: 413, body: { error: { code: "FILE_TOO_LARGE" } } }),
    });
    const { result } = renderHook(() => useReferenceSlots(RULES, { nativeBridge: bridge }));

    await act(async () => {
      await result.current.attach("picture", 1, `${DIR}\\p.png`, "p.png");
    });

    expect(result.current.state.slots.picture[0]!.errorCode).toBe("FILE_TOO_LARGE");
  });

  it("a thumbnail failure is silent: the slot is ready without one", async () => {
    const bridge = bridgeWithClips({ failMakeThumbnail: "THUMBNAIL_FAILED" });
    const { result } = renderHook(() => useReferenceSlots(RULES, { nativeBridge: bridge }));

    await act(async () => {
      await result.current.attach("video", 1, `${DIR}\\a.mp4`, "a.mp4");
    });

    const slot = result.current.state.slots.video[0]!;
    expect(slot.status).toBe("ready");
    expect(slot.thumbnailDataUrl).toBeNull();
  });

  it("generation: a clear mid-upload discards the late result", async () => {
    const bridge = bridgeWithClips({ holdUploads: true });
    const { result } = renderHook(() => useReferenceSlots(RULES, { nativeBridge: bridge }));

    let pending!: Promise<void>;
    act(() => {
      pending = result.current.attach("video", 1, `${DIR}\\a.mp4`, "a.mp4");
    });
    expect(result.current.state.slots.video[0]!.status).toBe("uploading");

    act(() => result.current.clear("video", 1));
    await act(async () => {
      bridge.releaseUploads();
      await pending;
    });

    expect(result.current.state.slots.video[0]!.status).toBe("empty");
    expect(result.current.state.slots.video[0]!.id).toBeNull();
  });

  it("generation: a superseded attach's late result never lands on the newer upload", async () => {
    const base = bridgeWithClips();
    const first = deferred<void>();
    const second = deferred<void>();
    const bridge = wrap(base, {
      "backend.uploadFile": async (params) => {
        const p = params as ParamsOf<"backend.uploadFile">;
        await (p.filePath.endsWith("first.mp4") ? first.promise : second.promise);
        return { status: 200, body: { video_id: p.filePath.endsWith("first.mp4") ? "late-first" : "second" } };
      },
    });
    const { result } = renderHook(() => useReferenceSlots(RULES, { nativeBridge: bridge }));

    let pendingFirst!: Promise<void>;
    let pendingSecond!: Promise<void>;
    act(() => {
      pendingFirst = result.current.attach("video", 1, `${DIR}\\first.mp4`, "first.mp4");
    });
    act(() => {
      pendingSecond = result.current.attach("video", 1, `${DIR}\\a.mp4`, "a.mp4");
    });
    // The first settles while the second is still uploading: dropped by the
    // generation number (the reducer alone would accept it on an uploading slot).
    await act(async () => {
      first.resolve();
      await pendingFirst;
    });
    expect(result.current.state.slots.video[0]!.status).toBe("uploading");
    expect(result.current.state.slots.video[0]!.fileName).toBe("a.mp4");

    await act(async () => {
      second.resolve();
      await pendingSecond;
    });
    expect(result.current.state.slots.video[0]!.id).toBe("second");
  });

  it("addedAt follows input order across kinds; a re-attach gets a new number", async () => {
    const bridge = bridgeWithClips();
    const { result } = renderHook(() => useReferenceSlots(RULES, { nativeBridge: bridge }));

    await act(async () => {
      await result.current.attach("picture", 3, `${DIR}\\p.png`, "p.png");
      await result.current.attach("video", 1, `${DIR}\\a.mp4`, "a.mp4");
      await result.current.attach("picture", 1, `${DIR}\\q.png`, "q.png");
    });
    const at = (k: "picture" | "video", i: number) => result.current.state.slots[k][i]!.addedAt!;
    expect(at("picture", 2)).toBeLessThan(at("video", 0));
    expect(at("video", 0)).toBeLessThan(at("picture", 0));

    await act(async () => {
      await result.current.attach("picture", 3, `${DIR}\\p.png`, "p.png");
    });
    expect(at("picture", 2)).toBeGreaterThan(at("picture", 0));
  });
});

describe("useReferenceSlots — pick / attachToFirstEmpty / clear", () => {
  it("pick: a CANCELLED dialog does nothing", async () => {
    const bridge = createMockBridge({ delayMs: 0, failPickFile: "CANCELLED" });
    const spy = vi.spyOn(bridge, "request");
    const { result } = renderHook(() => useReferenceSlots(RULES, { nativeBridge: bridge }));
    const before = result.current.state;

    await act(async () => {
      await result.current.pick("picture", 1);
    });

    expect(result.current.state).toBe(before);
    expect(spy).not.toHaveBeenCalledWith("backend.uploadFile", expect.anything());
  });

  it("pick: the picked file is attached (kind picture asks for an image)", async () => {
    const bridge = createMockBridge({ delayMs: 0, pickFileName: "pic.png" });
    const spy = vi.spyOn(bridge, "request");
    const { result } = renderHook(() => useReferenceSlots(RULES, { nativeBridge: bridge }));

    await act(async () => {
      await result.current.pick("picture", 4);
    });

    expect(spy).toHaveBeenCalledWith("ui.pickFile", { kind: "image" });
    const slot = result.current.state.slots.picture[3]!;
    expect(slot.status).toBe("ready");
    expect(slot.fileName).toBe("pic.png");
  });

  it("attachToFirstEmpty fills the lowest free slots, then returns null when full", async () => {
    const bridge = bridgeWithClips();
    const spy = vi.spyOn(bridge, "request");
    const { result } = renderHook(() => useReferenceSlots(RULES, { nativeBridge: bridge }));

    const slots: Array<number | null> = [];
    await act(async () => {
      // Back to back, before any re-render: each call still gets its own slot.
      slots.push(result.current.attachToFirstEmpty("video", `${DIR}\\a.mp4`, "a.mp4"));
      slots.push(result.current.attachToFirstEmpty("video", `${DIR}\\a.mp4`, "a.mp4"));
      slots.push(result.current.attachToFirstEmpty("video", `${DIR}\\a.mp4`, "a.mp4"));
    });
    expect(slots).toEqual([1, 2, 3]);
    await waitFor(() => expect(result.current.state.slots.video.every((s) => s.status === "ready")).toBe(true));

    const uploadsBefore = spy.mock.calls.filter(([m]) => m === "backend.uploadFile").length;
    let full: number | null = 0;
    act(() => {
      full = result.current.attachToFirstEmpty("video", `${DIR}\\a.mp4`, "a.mp4");
    });
    expect(full).toBeNull();
    expect(spy.mock.calls.filter(([m]) => m === "backend.uploadFile").length).toBe(uploadsBefore);

    // A cleared slot is free again (numbers are not packed).
    act(() => result.current.clear("video", 2));
    await act(async () => {
      slots.push(result.current.attachToFirstEmpty("video", `${DIR}\\a.mp4`, "a.mp4"));
    });
    expect(slots[3]).toBe(2);
  });

  it("clear of the A2V slot unchecks it; setEnabled / setA2vSlot pass through", async () => {
    const bridge = bridgeWithClips();
    const { result } = renderHook(() => useReferenceSlots(RULES, { nativeBridge: bridge }));

    await act(async () => {
      await result.current.attach("audio", 2, `${DIR}\\b.wav`, "b.wav");
    });
    act(() => {
      result.current.setEnabled(true);
      result.current.setA2vSlot(2);
    });
    expect(result.current.state.enabled).toBe(true);
    expect(result.current.state.a2vSlot).toBe(2);
    expect(result.current.summary.slotUse.audio[1]!.isA2v).toBe(true);

    act(() => result.current.clear("audio", 2));
    expect(result.current.state.a2vSlot).toBeNull();
    expect(result.current.state.slots.audio[1]!.status).toBe("empty");
  });
});

describe("useReferenceSlots — resize", () => {
  it("rebuilds the rows when the rule counts change, keeping slots that still fit", async () => {
    const bridge = bridgeWithClips();
    const { result, rerender } = renderHook(({ rules }) => useReferenceSlots(rules, { nativeBridge: bridge }), {
      initialProps: { rules: RULES },
    });

    await act(async () => {
      await result.current.attach("picture", 1, `${DIR}\\p.png`, "p.png");
      await result.current.attach("picture", 5, `${DIR}\\q.png`, "q.png");
    });

    const smaller = { ...RULES, maxImages: 2, maxVideos: 1 };
    rerender({ rules: smaller });
    expect(result.current.state.slots.picture).toHaveLength(2);
    expect(result.current.state.slots.video).toHaveLength(1);
    expect(result.current.state.slots.picture[0]!.fileName).toBe("p.png");
    expect(result.current.rules).toBe(smaller);

    rerender({ rules: RULES });
    expect(result.current.state.slots.picture).toHaveLength(9);
    expect(result.current.state.slots.picture[0]!.fileName).toBe("p.png");
    expect(result.current.state.slots.picture[4]!.status).toBe("empty");
  });

  it("a new rules object with the same counts does not resize", () => {
    const bridge = bridgeWithClips();
    const { result, rerender } = renderHook(({ rules }) => useReferenceSlots(rules, { nativeBridge: bridge }), {
      initialProps: { rules: RULES },
    });
    const before = result.current.state;
    rerender({ rules: { ...RULES, videoTotalSec: 20 } });
    expect(result.current.state).toBe(before);
    expect(result.current.rules.videoTotalSec).toBe(20);
  });
});

describe("useReferenceA2vSync", () => {
  function setup(initialAvailable: boolean) {
    const bridge = bridgeWithClips();
    const spy = vi.spyOn(bridge, "request");
    const hook = renderHook(
      ({ available }) => {
        const reference = useReferenceSlots(RULES, { nativeBridge: bridge });
        const sourceAudio = useSourceUpload("audio", { nativeBridge: bridge });
        useReferenceA2vSync(reference, sourceAudio, available);
        return { reference, sourceAudio };
      },
      { initialProps: { available: initialAvailable } },
    );
    return { ...hook, spy };
  }

  async function checkA2v(result: ReturnType<typeof setup>["result"]) {
    await act(async () => {
      await result.current.reference.attach("audio", 2, `${DIR}\\b.wav`, "b.wav");
    });
    act(() => {
      result.current.reference.setEnabled(true);
      result.current.reference.setA2vSlot(2);
    });
  }

  it("adopts the effective A2V slot into sourceAudio, without a second upload", async () => {
    const { result, spy } = setup(true);
    await checkA2v(result);

    const slot = result.current.reference.state.slots.audio[1]!;
    await waitFor(() => expect(result.current.sourceAudio.state.id).toBe(slot.id));
    expect(result.current.sourceAudio.state).toMatchObject({
      status: "ready",
      fileName: "b.wav",
      filePath: `${DIR}\\b.wav`,
    });
    expect(spy.mock.calls.filter(([m]) => m === "backend.uploadFile")).toHaveLength(1);
  });

  it("Enable off (or unchecking) clears sourceAudio", async () => {
    const { result } = setup(true);
    await checkA2v(result);
    await waitFor(() => expect(result.current.sourceAudio.state.status).toBe("ready"));

    act(() => result.current.reference.setEnabled(false));
    await waitFor(() => expect(result.current.sourceAudio.state.status).toBe("idle"));

    act(() => result.current.reference.setEnabled(true));
    await waitFor(() => expect(result.current.sourceAudio.state.status).toBe("ready"));
    act(() => result.current.reference.setA2vSlot(null));
    await waitFor(() => expect(result.current.sourceAudio.state.status).toBe("idle"));
  });

  it("while the Reference is shown, an audio from another path is cleared", async () => {
    const { result, rerender } = setup(false);
    await act(async () => {
      await result.current.sourceAudio.uploadPath(`${DIR}\\b.wav`, "b.wav");
    });
    expect(result.current.sourceAudio.state.status).toBe("ready");

    rerender({ available: true });
    await waitFor(() => expect(result.current.sourceAudio.state.status).toBe("idle"));
  });

  it("when the Reference goes away, only the adopted id is cleared", async () => {
    const { result, rerender } = setup(true);
    await checkA2v(result);
    await waitFor(() => expect(result.current.sourceAudio.state.status).toBe("ready"));

    rerender({ available: false });
    await waitFor(() => expect(result.current.sourceAudio.state.status).toBe("idle"));

    // Without the Reference, an audio the user attaches is left alone.
    await act(async () => {
      await result.current.sourceAudio.uploadPath(`${DIR}\\b.wav`, "b.wav");
    });
    rerender({ available: false });
    expect(result.current.sourceAudio.state.status).toBe("ready");
  });

  it("does not touch a user audio when the Reference was never shown", async () => {
    const { result, rerender } = setup(false);
    await act(async () => {
      await result.current.sourceAudio.uploadPath(`${DIR}\\b.wav`, "b.wav");
    });
    rerender({ available: false });
    expect(result.current.sourceAudio.state.status).toBe("ready");
  });
});
