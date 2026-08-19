import { act, renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { BridgeError } from "../../bridge";
import type { BridgeMethod, NativeBridge, ParamsOf, ResultOf } from "../../bridge";
import { createMockBridge } from "../../bridge/mockBridge";
import { FALLBACK_APP_CONFIG } from "./defaultConfig";
import type { KeyframeItem } from "./keyframeUtils";
import { useKeyframes, type UseKeyframesResult } from "./useKeyframes";

/** Wraps a fresh mock bridge so `backend.uploadFile` fails (with `code`) only
 * for calls whose `filePath` matches `failForPath` — every other call (and
 * every other method) behaves exactly like the plain mock. Used by the
 * `replaceFile` failure test below, which needs an item that already has a
 * real `imageId` from a first *successful* upload before a second upload (the
 * replace) is made to fail — `MockBridgeOptions.failUploadFile` alone can't
 * express that (it's all-or-nothing for the whole bridge instance). */
function bridgeThatFailsUploadFor(failForPath: string, code = "BACKEND_UNREACHABLE"): NativeBridge {
  const base = createMockBridge({ delayMs: 0 });
  return {
    ...base,
    async request<M extends BridgeMethod>(method: M, params: ParamsOf<M>): Promise<ResultOf<M>> {
      if (method === "backend.uploadFile" && (params as { filePath?: string }).filePath === failForPath) {
        throw new BridgeError(code, "Test: forced backend.uploadFile failure for a specific path");
      }
      return base.request(method, params);
    },
  };
}

/** Wraps a fresh mock bridge so `timeline.captureFrame` rejects with `code`
 * unconditionally. `MockBridgeOptions.failCaptureFrame` only types
 * `"NO_EDIT_HANDLE" | "CAPTURE_FAILED"` (the codes the native capture
 * pipeline itself can report) — it has no way to express a `CANCELLED`
 * capture (e.g. a "select region" step the user backs out of), so
 * `captureIntoCard`'s CANCELLED test below needs this thin override instead,
 * mirroring `bridgeThatFailsUploadFor` above. */
function bridgeThatCancelsCaptureFrame(): NativeBridge {
  const base = createMockBridge({ delayMs: 0 });
  return {
    ...base,
    async request<M extends BridgeMethod>(method: M, params: ParamsOf<M>): Promise<ResultOf<M>> {
      if (method === "timeline.captureFrame") {
        throw new BridgeError("CANCELLED", "Test: forced timeline.captureFrame CANCELLED");
      }
      return base.request(method, params);
    },
  };
}

// Real timers throughout, mirroring useGeneration.test.ts: the mock bridge's
// simulated delay is 0ms so `waitFor`'s default timeout is more than enough.

/** `result.current.items[0]` is `KeyframeItem | undefined` under
 * `noUncheckedIndexedAccess` — every test here asserts the item exists first
 * (via `toHaveLength`/`waitFor`), so this just gives that a non-null type. */
function firstItem(keyframes: UseKeyframesResult): KeyframeItem {
  const item = keyframes.items[0];
  if (!item) throw new Error("expected at least one keyframe item");
  return item;
}

describe("useKeyframes", () => {
  it("starts with no items and add enabled", () => {
    const mockBridge = createMockBridge({ delayMs: 0 });
    const { result } = renderHook(() => useKeyframes(FALLBACK_APP_CONFIG, { nativeBridge: mockBridge }));

    expect(result.current.items).toEqual([]);
    expect(result.current.canAdd).toBe(true);
    expect(result.current.addDisabledReason).toBeNull();
    expect(result.current.isUploading).toBe(false);
    expect(result.current.conditioningImages).toEqual([]);
  });

  it(
    "addFromCapture: captureFrame -> uploadFile -> makeThumbnail settles the card into 'ready' with an image_id",
    async () => {
      const mockBridge = createMockBridge({ delayMs: 0 });
      const { result } = renderHook(() => useKeyframes(FALLBACK_APP_CONFIG, { nativeBridge: mockBridge }));

      act(() => {
        void result.current.addFromCapture();
      });

      // The card appears immediately, in the uploading state, before the
      // capture round trip even settles.
      await waitFor(() => expect(result.current.items).toHaveLength(1));
      expect(firstItem(result.current).status).toBe("uploading");
      expect(result.current.isUploading).toBe(true);

      await waitFor(() => expect(firstItem(result.current).status).toBe("ready"));
      const item = firstItem(result.current);
      expect(item.imageId).toMatch(/^mock-image-/);
      expect(item.fileName).toMatch(/\.png$/);
      expect(item.thumbnailDataUrl).toMatch(/^data:image\/png;base64,/);
      expect(result.current.isUploading).toBe(false);
      expect(result.current.conditioningImages).toEqual([
        { image_id: item.imageId, frame_idx: 0, strength: 0.8 },
      ]);
    },
  );

  it("addFromCapture: leaves the card in 'error' with the bridge's code when captureFrame fails", async () => {
    const mockBridge = createMockBridge({ delayMs: 0, failCaptureFrame: "NO_EDIT_HANDLE" });
    const { result } = renderHook(() => useKeyframes(FALLBACK_APP_CONFIG, { nativeBridge: mockBridge }));

    act(() => {
      void result.current.addFromCapture();
    });

    await waitFor(() => expect(result.current.items).toHaveLength(1));
    await waitFor(() => expect(firstItem(result.current).status).toBe("error"));
    expect(firstItem(result.current).errorCode).toBe("NO_EDIT_HANDLE");
    expect(result.current.conditioningImages).toEqual([]);
  });

  it("addFromFile: pickFile -> uploadFile settles the card into 'ready'", async () => {
    const mockBridge = createMockBridge({ delayMs: 0, pickFileName: "my-keyframe.png" });
    const { result } = renderHook(() => useKeyframes(FALLBACK_APP_CONFIG, { nativeBridge: mockBridge }));

    act(() => {
      void result.current.addFromFile();
    });

    await waitFor(() => expect(result.current.items).toHaveLength(1));
    await waitFor(() => expect(firstItem(result.current).status).toBe("ready"));
    expect(firstItem(result.current).fileName).toBe("my-keyframe.png");
    expect(firstItem(result.current).imageId).toMatch(/^mock-image-/);
  });

  it("addFromFile: a CANCELLED pickFile silently removes the pending card (no error state)", async () => {
    const mockBridge = createMockBridge({ delayMs: 0, failPickFile: "CANCELLED" });
    const { result } = renderHook(() => useKeyframes(FALLBACK_APP_CONFIG, { nativeBridge: mockBridge }));

    await act(async () => {
      await result.current.addFromFile();
    });

    expect(result.current.items).toEqual([]);
  });

  it("addFromFile: a DIALOG_FAILED pickFile leaves the card in 'error'", async () => {
    const mockBridge = createMockBridge({ delayMs: 0, failPickFile: "DIALOG_FAILED" });
    const { result } = renderHook(() => useKeyframes(FALLBACK_APP_CONFIG, { nativeBridge: mockBridge }));

    await act(async () => {
      await result.current.addFromFile();
    });

    expect(result.current.items).toHaveLength(1);
    expect(firstItem(result.current).status).toBe("error");
    expect(firstItem(result.current).errorCode).toBe("DIALOG_FAILED");
  });

  it("addFromCapture: an uploadFile transport failure leaves the card in 'error', thumbnail still best-effort applied", async () => {
    const mockBridge = createMockBridge({ delayMs: 0, failUploadFile: "BACKEND_UNREACHABLE" });
    const { result } = renderHook(() => useKeyframes(FALLBACK_APP_CONFIG, { nativeBridge: mockBridge }));

    act(() => {
      void result.current.addFromCapture();
    });

    await waitFor(() => expect(result.current.items).toHaveLength(1));
    await waitFor(() => expect(firstItem(result.current).status).toBe("error"));
    expect(firstItem(result.current).errorCode).toBe("BACKEND_UNREACHABLE");
    // The thumbnail call isn't forced to fail in this test, so it should
    // still have been applied even though the upload failed.
    expect(firstItem(result.current).thumbnailDataUrl).toMatch(/^data:image\/png;base64,/);
  });

  it("a thumbnail failure leaves thumbnailDataUrl null but doesn't block the card from becoming 'ready'", async () => {
    const mockBridge = createMockBridge({ delayMs: 0, failMakeThumbnail: "THUMBNAIL_FAILED" });
    const { result } = renderHook(() => useKeyframes(FALLBACK_APP_CONFIG, { nativeBridge: mockBridge }));

    act(() => {
      void result.current.addFromCapture();
    });

    await waitFor(() => expect(result.current.items).toHaveLength(1));
    await waitFor(() => expect(firstItem(result.current).status).toBe("ready"));
    expect(firstItem(result.current).thumbnailDataUrl).toBeNull();
    expect(firstItem(result.current).imageId).not.toBeNull();
  });

  it("enforces max_conditioning_images: add becomes disabled once the limit is reached", async () => {
    const mockBridge = createMockBridge({ delayMs: 0 });
    const config = { ...FALLBACK_APP_CONFIG, limits: { ...FALLBACK_APP_CONFIG.limits, max_conditioning_images: 2 } };
    const { result } = renderHook(() => useKeyframes(config, { nativeBridge: mockBridge }));

    await act(async () => {
      await result.current.addFromCapture();
    });
    await act(async () => {
      await result.current.addFromCapture();
    });

    expect(result.current.items).toHaveLength(2);
    expect(result.current.canAdd).toBe(false);
    expect(result.current.addDisabledReason).toContain("2");

    // A third attempt is a no-op: addFromCapture bails out before touching
    // the bridge or the item list once the cap is reached.
    await act(async () => {
      await result.current.addFromCapture();
    });
    expect(result.current.items).toHaveLength(2);
  });

  it("remove() drops an item by id", async () => {
    const mockBridge = createMockBridge({ delayMs: 0 });
    const { result } = renderHook(() => useKeyframes(FALLBACK_APP_CONFIG, { nativeBridge: mockBridge }));

    await act(async () => {
      await result.current.addFromCapture();
    });
    const id = firstItem(result.current).id;

    act(() => {
      result.current.remove(id);
    });
    expect(result.current.items).toEqual([]);
  });

  it("setFrameIdx clamps to a non-negative integer and setStrength clamps to [0,1]", async () => {
    const mockBridge = createMockBridge({ delayMs: 0 });
    const { result } = renderHook(() => useKeyframes(FALLBACK_APP_CONFIG, { nativeBridge: mockBridge }));

    await act(async () => {
      await result.current.addFromCapture();
    });
    const id = firstItem(result.current).id;

    act(() => result.current.setFrameIdx(id, -5));
    expect(firstItem(result.current).frameIdx).toBe(0);

    act(() => result.current.setFrameIdx(id, 17.9));
    expect(firstItem(result.current).frameIdx).toBe(17);

    act(() => result.current.setStrength(id, 5));
    expect(firstItem(result.current).strength).toBe(1);

    act(() => result.current.setStrength(id, -1));
    expect(firstItem(result.current).strength).toBe(0);
  });

  it("items are exposed sorted by ascending frame_idx regardless of insertion order", async () => {
    const mockBridge = createMockBridge({ delayMs: 0 });
    const { result } = renderHook(() => useKeyframes(FALLBACK_APP_CONFIG, { nativeBridge: mockBridge }));

    await act(async () => {
      await result.current.addFromCapture();
    });
    await act(async () => {
      await result.current.addFromCapture();
    });
    const [firstId, secondId] = result.current.items.map((i) => i.id);
    if (!firstId || !secondId) throw new Error("expected two keyframe items");

    act(() => result.current.setFrameIdx(firstId, 40));
    act(() => result.current.setFrameIdx(secondId, 5));

    expect(result.current.items.map((i) => i.id)).toEqual([secondId, firstId]);
  });

  // --- Keyframe timeline rework (2026-07-18): optional initialFrameIdx ------

  it("addFromPath: an explicit initialFrameIdx seeds the new card's frameIdx", async () => {
    const mockBridge = createMockBridge({ delayMs: 0 });
    const { result } = renderHook(() => useKeyframes(FALLBACK_APP_CONFIG, { nativeBridge: mockBridge }));

    await act(async () => {
      await result.current.addFromPath("C:\\in\\a.png", "a.png", 17);
    });

    expect(firstItem(result.current).frameIdx).toBe(17);
  });

  it("addFromCapture: an explicit initialFrameIdx seeds the new card's frameIdx, defaulting to 0 when omitted", async () => {
    const mockBridge = createMockBridge({ delayMs: 0 });
    const { result } = renderHook(() => useKeyframes(FALLBACK_APP_CONFIG, { nativeBridge: mockBridge }));

    act(() => {
      void result.current.addFromCapture(25);
    });
    await waitFor(() => expect(result.current.items).toHaveLength(1));
    expect(firstItem(result.current).frameIdx).toBe(25);

    // The existing (no-arg) call path is unaffected: still defaults to 0 —
    // this is the Chain-non-regression guarantee the task brief calls out.
    await act(async () => {
      await result.current.addFromCapture();
    });
    const second = result.current.items.find((i) => i.frameIdx !== 25);
    expect(second?.frameIdx).toBe(0);
  });

  // --- Keyframe timeline rework (2026-07-18): addEmpty ----------------------

  it("addEmpty: adds an image-less 'empty' card at the given frameIdx, excluded from conditioningImages", () => {
    const mockBridge = createMockBridge({ delayMs: 0 });
    const { result } = renderHook(() => useKeyframes(FALLBACK_APP_CONFIG, { nativeBridge: mockBridge }));

    act(() => result.current.addEmpty(9));

    expect(result.current.items).toHaveLength(1);
    const item = firstItem(result.current);
    expect(item.status).toBe("empty");
    expect(item.frameIdx).toBe(9);
    expect(item.filePath).toBe("");
    expect(item.fileName).toBe("");
    expect(item.imageId).toBeNull();
    expect(item.thumbnailDataUrl).toBeNull();
    expect(item.errorCode).toBeNull();
    // "empty" is neither "ready" nor "error" — buildConditioningImages only
    // ever includes "ready" items, so this is excluded automatically.
    expect(result.current.conditioningImages).toEqual([]);
  });

  it("addEmpty: a no-op once max_conditioning_images is reached, mirroring the other add* methods", () => {
    const mockBridge = createMockBridge({ delayMs: 0 });
    const config = { ...FALLBACK_APP_CONFIG, limits: { ...FALLBACK_APP_CONFIG.limits, max_conditioning_images: 1 } };
    const { result } = renderHook(() => useKeyframes(config, { nativeBridge: mockBridge }));

    act(() => result.current.addEmpty(0));
    expect(result.current.items).toHaveLength(1);

    act(() => result.current.addEmpty(9));
    expect(result.current.items).toHaveLength(1);
  });

  // --- Keyframe timeline rework (2026-07-18): replaceFile -------------------

  it("replaceFile: success keeps frameIdx/strength, resets to 'uploading' then settles 'ready' with a fresh imageId", async () => {
    const mockBridge = createMockBridge({ delayMs: 0 });
    const { result } = renderHook(() => useKeyframes(FALLBACK_APP_CONFIG, { nativeBridge: mockBridge }));

    act(() => result.current.addEmpty(33));
    act(() => result.current.setStrength(firstItem(result.current).id, 0.42));
    const id = firstItem(result.current).id;

    const replacePromise = act(async () => {
      await result.current.replaceFile(id, "C:\\in\\new.png", "new.png");
    });
    await replacePromise;

    const item = firstItem(result.current);
    expect(item.status).toBe("ready");
    expect(item.imageId).toMatch(/^mock-image-/);
    expect(item.fileName).toBe("new.png");
    expect(item.frameIdx).toBe(33);
    expect(item.strength).toBe(0.42);
  });

  // --- Keyframe timeline rework (2026-07-18): pickReplaceFile ---------------

  it("pickReplaceFile: pickFile -> replaceFile swaps the image onto the existing card, keeping frameIdx", async () => {
    const mockBridge = createMockBridge({ delayMs: 0, pickFileName: "swapped.png" });
    const { result } = renderHook(() => useKeyframes(FALLBACK_APP_CONFIG, { nativeBridge: mockBridge }));

    act(() => result.current.addEmpty(9));
    const id = firstItem(result.current).id;
    expect(firstItem(result.current).status).toBe("empty");

    await act(async () => {
      await result.current.pickReplaceFile(id);
    });

    const item = firstItem(result.current);
    // Same single card (no new one added), now filled in from the picked file.
    expect(result.current.items).toHaveLength(1);
    expect(item.id).toBe(id);
    expect(item.status).toBe("ready");
    expect(item.fileName).toBe("swapped.png");
    expect(item.imageId).toMatch(/^mock-image-/);
    expect(item.frameIdx).toBe(9);
  });

  it("pickReplaceFile: a CANCELLED dialog leaves the card exactly as it was", async () => {
    const mockBridge = createMockBridge({ delayMs: 0, failPickFile: "CANCELLED" });
    const { result } = renderHook(() => useKeyframes(FALLBACK_APP_CONFIG, { nativeBridge: mockBridge }));

    act(() => result.current.addEmpty(17));
    const id = firstItem(result.current).id;

    await act(async () => {
      await result.current.pickReplaceFile(id);
    });

    const item = firstItem(result.current);
    expect(result.current.items).toHaveLength(1);
    expect(item.status).toBe("empty");
    expect(item.fileName).toBe("");
    expect(item.imageId).toBeNull();
    expect(item.frameIdx).toBe(17);
  });

  it("replaceFile: a failed replace leaves 'error' with no stale imageId left over from the previous file", async () => {
    const mockBridge = bridgeThatFailsUploadFor("C:\\in\\bad.png", "BACKEND_UNREACHABLE");
    const { result } = renderHook(() => useKeyframes(FALLBACK_APP_CONFIG, { nativeBridge: mockBridge }));

    // First, a genuinely successful add so the card starts with a real imageId.
    await act(async () => {
      await result.current.addFromPath("C:\\in\\good.png", "good.png", 5);
    });
    const id = firstItem(result.current).id;
    const originalImageId = firstItem(result.current).imageId;
    expect(originalImageId).toMatch(/^mock-image-/);

    // Then replace it with a file this bridge is rigged to fail on.
    await act(async () => {
      await result.current.replaceFile(id, "C:\\in\\bad.png", "bad.png");
    });

    const item = firstItem(result.current);
    expect(item.status).toBe("error");
    expect(item.errorCode).toBe("BACKEND_UNREACHABLE");
    expect(item.imageId).toBeNull();
    expect(item.imageId).not.toBe(originalImageId);
    // frameIdx survives the replace even though it failed.
    expect(item.frameIdx).toBe(5);
  });

  // --- captureIntoCard -------------------------------------------------------

  it("captureIntoCard: captureFrame -> replaceFile swaps the image onto the existing card, keeping frameIdx/strength", async () => {
    const mockBridge = createMockBridge({ delayMs: 0 });
    const { result } = renderHook(() => useKeyframes(FALLBACK_APP_CONFIG, { nativeBridge: mockBridge }));

    act(() => result.current.addEmpty(12));
    const id = firstItem(result.current).id;
    act(() => result.current.setStrength(id, 0.3));
    expect(firstItem(result.current).status).toBe("empty");

    await act(async () => {
      await result.current.captureIntoCard(id);
    });

    const item = firstItem(result.current);
    // Same single card (no new one added), now filled in from the capture.
    expect(result.current.items).toHaveLength(1);
    expect(item.id).toBe(id);
    expect(item.status).toBe("ready");
    expect(item.fileName).toMatch(/\.png$/);
    expect(item.imageId).toMatch(/^mock-image-/);
    expect(item.frameIdx).toBe(12);
    expect(item.strength).toBe(0.3);
  });

  it("captureIntoCard: flips the card to 'uploading' synchronously, before the captureFrame round trip settles", async () => {
    const mockBridge = createMockBridge({ delayMs: 0 });
    const { result } = renderHook(() => useKeyframes(FALLBACK_APP_CONFIG, { nativeBridge: mockBridge }));

    act(() => result.current.addEmpty(0));
    const id = firstItem(result.current).id;

    act(() => {
      void result.current.captureIntoCard(id);
    });

    // The uploading flip (and isCapturing) happen before the capture round
    // trip even starts, so both are observable immediately.
    expect(firstItem(result.current).status).toBe("uploading");
    expect(result.current.isCapturing).toBe(true);

    await waitFor(() => expect(firstItem(result.current).status).toBe("ready"));
    expect(result.current.isCapturing).toBe(false);
  });

  it("captureIntoCard: a CANCELLED capture leaves the card exactly as it was", async () => {
    const mockBridge = bridgeThatCancelsCaptureFrame();
    const { result } = renderHook(() => useKeyframes(FALLBACK_APP_CONFIG, { nativeBridge: mockBridge }));

    act(() => result.current.addEmpty(17));
    const id = firstItem(result.current).id;

    await act(async () => {
      await result.current.captureIntoCard(id);
    });

    const item = firstItem(result.current);
    expect(result.current.items).toHaveLength(1);
    // Restored to the pre-call status ("empty"), not left "uploading"/"error".
    expect(item.status).toBe("empty");
    expect(item.fileName).toBe("");
    expect(item.imageId).toBeNull();
    expect(item.frameIdx).toBe(17);
    expect(result.current.isCapturing).toBe(false);
  });

  it("captureIntoCard: a CANCELLED capture on an already-'ready' card restores 'ready' (not 'empty' or 'error')", async () => {
    // First captureFrame call succeeds (so the card starts "ready"); every
    // subsequent captureFrame call on the same bridge is forced CANCELLED —
    // isolating the "re-capture into an already-filled card" case from the
    // "empty card" case the earlier CANCELLED test covers.
    const base = createMockBridge({ delayMs: 0 });
    let captureCalls = 0;
    const mockBridge: NativeBridge = {
      ...base,
      async request<M extends BridgeMethod>(method: M, params: ParamsOf<M>): Promise<ResultOf<M>> {
        if (method === "timeline.captureFrame") {
          captureCalls += 1;
          if (captureCalls > 1) {
            throw new BridgeError("CANCELLED", "Test: forced timeline.captureFrame CANCELLED on 2nd+ call");
          }
        }
        return base.request(method, params);
      },
    };
    const { result } = renderHook(() => useKeyframes(FALLBACK_APP_CONFIG, { nativeBridge: mockBridge }));

    await act(async () => {
      await result.current.addFromCapture(4);
    });
    const id = firstItem(result.current).id;
    expect(firstItem(result.current).status).toBe("ready");
    const originalImageId = firstItem(result.current).imageId;

    await act(async () => {
      await result.current.captureIntoCard(id);
    });

    const item = firstItem(result.current);
    expect(item.status).toBe("ready");
    expect(item.imageId).toBe(originalImageId);
    expect(item.frameIdx).toBe(4);
  });

  it("captureIntoCard: a non-CANCELLED capture failure leaves the card 'error' with the bridge's code", async () => {
    const mockBridge = createMockBridge({ delayMs: 0, failCaptureFrame: "CAPTURE_FAILED" });
    const { result } = renderHook(() => useKeyframes(FALLBACK_APP_CONFIG, { nativeBridge: mockBridge }));

    act(() => result.current.addEmpty(0));
    const id = firstItem(result.current).id;

    await act(async () => {
      await result.current.captureIntoCard(id);
    });

    const item = firstItem(result.current);
    expect(item.status).toBe("error");
    expect(item.errorCode).toBe("CAPTURE_FAILED");
    expect(result.current.isCapturing).toBe(false);
  });

  it("captureIntoCard: ignores a card that is already 'uploading' (no-op, no re-entrant capture)", async () => {
    const mockBridge = createMockBridge({ delayMs: 0, holdUploads: true });
    const { result } = renderHook(() => useKeyframes(FALLBACK_APP_CONFIG, { nativeBridge: mockBridge }));

    act(() => {
      void result.current.addFromCapture(0);
    });
    await waitFor(() => expect(result.current.items).toHaveLength(1));
    const id = firstItem(result.current).id;
    expect(firstItem(result.current).status).toBe("uploading");

    // A second call while the first is still in flight (held open via
    // holdUploads) must be a complete no-op: no second captureFrame/upload
    // round trip, card untouched.
    await act(async () => {
      await result.current.captureIntoCard(id);
    });
    expect(result.current.items).toHaveLength(1);
    expect(firstItem(result.current).status).toBe("uploading");

    mockBridge.releaseUploads();
    await waitFor(() => expect(firstItem(result.current).status).toBe("ready"));
  });
});
