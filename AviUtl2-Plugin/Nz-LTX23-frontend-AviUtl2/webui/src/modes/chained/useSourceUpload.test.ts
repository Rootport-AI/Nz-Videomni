import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { createMockBridge } from "../../bridge/mockBridge";
import type { BridgeMethod, NativeBridge, ParamsOf, ResultOf } from "../../bridge";
import { useSourceUpload } from "./useSourceUpload";

describe("useSourceUpload", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it("starts idle", () => {
    const mockBridge = createMockBridge({ delayMs: 0 });
    const { result } = renderHook(() => useSourceUpload("video", { nativeBridge: mockBridge }));

    expect(result.current.state).toEqual({ status: "idle", fileName: null, filePath: null, id: null, errorCode: null, trimFailed: false, frames: null, fps: null });
  });

  it("pick (video): pickFile -> uploadFile settles into 'ready' with a video_id", async () => {
    const mockBridge = createMockBridge({ delayMs: 0, pickFileName: "source.mp4" });
    const { result } = renderHook(() => useSourceUpload("video", { nativeBridge: mockBridge }));

    await act(async () => {
      await result.current.pick();
    });

    expect(result.current.state.status).toBe("ready");
    expect(result.current.state.fileName).toBe("source.mp4");
    expect(result.current.state.id).toMatch(/^mock-video-/);
  });

  it("exposes an 'uploading' status while the upload is in flight", async () => {
    vi.useFakeTimers();
    const mockBridge = createMockBridge({ delayMs: 50, pickFileName: "source.mp4" });
    const { result } = renderHook(() => useSourceUpload("video", { nativeBridge: mockBridge }));

    let pickPromise: Promise<void>;
    act(() => {
      pickPromise = result.current.pick();
    });

    // First bridge call (ui.pickFile) resolves after ~50ms; the hook then
    // sets "uploading" before making the second call.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(51);
    });
    expect(result.current.state.status).toBe("uploading");
    expect(result.current.state.fileName).toBe("source.mp4");

    await act(async () => {
      await vi.advanceTimersByTimeAsync(51);
      await pickPromise;
    });
    expect(result.current.state.status).toBe("ready");
  });

  it("pick (audio): pickFile -> uploadFile settles into 'ready' with an audio_id", async () => {
    const mockBridge = createMockBridge({ delayMs: 0, pickFileName: "source.wav" });
    const { result } = renderHook(() => useSourceUpload("audio", { nativeBridge: mockBridge }));

    await act(async () => {
      await result.current.pick();
    });

    expect(result.current.state.status).toBe("ready");
    expect(result.current.state.id).toMatch(/^mock-audio-/);
    expect(result.current.state.fileName).toBe("source.wav");
  });

  it("a CANCELLED pick leaves the previous state untouched", async () => {
    const mockBridge = createMockBridge({ delayMs: 0, failPickFile: "CANCELLED" });
    const { result } = renderHook(() => useSourceUpload("video", { nativeBridge: mockBridge }));

    await act(async () => {
      await result.current.pick();
    });

    expect(result.current.state).toEqual({ status: "idle", fileName: null, filePath: null, id: null, errorCode: null, trimFailed: false, frames: null, fps: null });
  });

  it("a DIALOG_FAILED pick surfaces an error", async () => {
    const mockBridge = createMockBridge({ delayMs: 0, failPickFile: "DIALOG_FAILED" });
    const { result } = renderHook(() => useSourceUpload("video", { nativeBridge: mockBridge }));

    await act(async () => {
      await result.current.pick();
    });

    expect(result.current.state.status).toBe("error");
    expect(result.current.state.errorCode).toBe("DIALOG_FAILED");
  });

  it("an uploadFile transport failure leaves the card in 'error' with the bridge's code", async () => {
    const mockBridge = createMockBridge({ delayMs: 0, failUploadFile: "BACKEND_UNREACHABLE" });
    const { result } = renderHook(() => useSourceUpload("video", { nativeBridge: mockBridge }));

    await act(async () => {
      await result.current.pick();
    });

    expect(result.current.state.status).toBe("error");
    expect(result.current.state.errorCode).toBe("BACKEND_UNREACHABLE");
  });

  it("clear() resets back to idle", async () => {
    const mockBridge = createMockBridge({ delayMs: 0, pickFileName: "source.mp4" });
    const { result } = renderHook(() => useSourceUpload("video", { nativeBridge: mockBridge }));

    await act(async () => {
      await result.current.pick();
    });
    expect(result.current.state.status).toBe("ready");

    act(() => result.current.clear());
    expect(result.current.state).toEqual({ status: "idle", fileName: null, filePath: null, id: null, errorCode: null, trimFailed: false, frames: null, fps: null });
  });

  describe("W5 initial (carried-over ready state)", () => {
    it("starts in a ready state seeded from deps.initial, with no upload call", () => {
      const mockBridge = createMockBridge({ delayMs: 0 });
      const spy = vi.spyOn(mockBridge, "request");
      const { result } = renderHook(() =>
        useSourceUpload("video", {
          nativeBridge: mockBridge,
          initial: { id: "carried-video-1", fileName: "ref.mp4", filePath: "C:\\v\\ref.mp4" },
        }),
      );

      expect(result.current.state).toEqual({
        status: "ready",
        fileName: "ref.mp4",
        filePath: "C:\\v\\ref.mp4",
        id: "carried-video-1",
        errorCode: null,
        // §1-6: a carried-over slot was never re-uploaded, so no trim was asked
        // for it (the field is part of every state, always false here).
        trimFailed: false,
        // 素材（末尾）v2: no response body was ever seen for a carried slot,
        // so nothing was measured either.
        frames: null,
        fps: null,
      });
      // No re-upload: the carried id is already valid server-side.
      expect(spy).not.toHaveBeenCalledWith("backend.uploadFile", expect.anything());
    });

    it("clear() from a carried initial resets back to idle", () => {
      const mockBridge = createMockBridge({ delayMs: 0 });
      const { result } = renderHook(() =>
        useSourceUpload("audio", {
          nativeBridge: mockBridge,
          initial: { id: "carried-audio-1", fileName: "voice.wav", filePath: "C:\\a\\voice.wav" },
        }),
      );
      expect(result.current.state.status).toBe("ready");

      act(() => result.current.clear());
      expect(result.current.state).toEqual({ status: "idle", fileName: null, filePath: null, id: null, errorCode: null, trimFailed: false, frames: null, fps: null });
    });

    it("a null initial behaves exactly like the default idle start", () => {
      const mockBridge = createMockBridge({ delayMs: 0 });
      const { result } = renderHook(() => useSourceUpload("video", { nativeBridge: mockBridge, initial: null }));
      expect(result.current.state).toEqual({ status: "idle", fileName: null, filePath: null, id: null, errorCode: null, trimFailed: false, frames: null, fps: null });
    });
  });

  describe("uploadPath (already-resolved file path, skipping ui.pickFile)", () => {
    it("uploads a given path directly, settling into 'ready'", async () => {
      const mockBridge = createMockBridge({ delayMs: 0 });
      const { result } = renderHook(() => useSourceUpload("video", { nativeBridge: mockBridge }));

      await act(async () => {
        await result.current.uploadPath("C:\\Users\\mock\\Videos\\clip.mp4", "clip.mp4");
      });

      expect(result.current.state.status).toBe("ready");
      expect(result.current.state.fileName).toBe("clip.mp4");
      expect(result.current.state.id).toMatch(/^mock-video-/);
    });

    it("a transport failure leaves the state in 'error' with the bridge's code", async () => {
      const mockBridge = createMockBridge({ delayMs: 0, failUploadFile: "BACKEND_UNREACHABLE" });
      const { result } = renderHook(() => useSourceUpload("video", { nativeBridge: mockBridge }));

      await act(async () => {
        await result.current.uploadPath("C:\\Users\\mock\\Videos\\clip.mp4", "clip.mp4");
      });

      expect(result.current.state.status).toBe("error");
      expect(result.current.state.errorCode).toBe("BACKEND_UNREACHABLE");
    });

    it("generation token: a clear() mid-upload discards the stale result once it settles (no resurrection to 'ready')", async () => {
      const mockBridge = createMockBridge({ delayMs: 0, holdUploads: true });
      const { result } = renderHook(() => useSourceUpload("video", { nativeBridge: mockBridge }));

      act(() => {
        void result.current.uploadPath("C:\\Users\\mock\\Videos\\clip.mp4", "clip.mp4");
      });
      await waitFor(() => expect(result.current.state.status).toBe("uploading"));

      act(() => result.current.clear());
      expect(result.current.state.status).toBe("idle");

      act(() => {
        mockBridge.releaseUploads();
      });
      // Give the now-released upload a real tick to settle; it must stay idle.
      await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 0));
      });
      expect(result.current.state.status).toBe("idle");
    });

    it("generation token: a second uploadPath() call supersedes the first, whose late result is discarded", async () => {
      const mockBridge = createMockBridge({ delayMs: 0, holdUploads: true });
      const { result } = renderHook(() => useSourceUpload("video", { nativeBridge: mockBridge }));

      act(() => {
        void result.current.uploadPath("C:\\Users\\mock\\Videos\\first.mp4", "first.mp4");
      });
      await waitFor(() => expect(result.current.state.status).toBe("uploading"));

      // Re-picking before the first upload settles supersedes it in place —
      // `clear()` isn't involved here, just a second `uploadPath()` call.
      act(() => {
        void result.current.uploadPath("C:\\Users\\mock\\Videos\\second.mp4", "second.mp4");
      });
      expect(result.current.state.fileName).toBe("second.mp4");

      act(() => {
        mockBridge.releaseUploads();
      });
      await waitFor(() => expect(result.current.state.status).toBe("ready"));
      // The surviving state is the SECOND call's, not a stale first result.
      expect(result.current.state.fileName).toBe("second.mp4");
    });
  });
  // §1-6（元動画の範囲トリム, 契約v10）: `query` 未指定時にリクエストへキーが
  // 生えないことが「無トリム時のバイト等価」のFE側の要。
  describe("uploadPath query / trimFailed (§1-6, contract v10)", () => {
    it("omitting `query` sends EXACTLY the pre-v10 params (no key, not even undefined)", async () => {
      const mockBridge = createMockBridge({ delayMs: 0 });
      const spy = vi.spyOn(mockBridge, "request");
      const { result } = renderHook(() => useSourceUpload("video", { nativeBridge: mockBridge }));

      await act(async () => {
        await result.current.uploadPath("C:\\v\\clip.mp4", "clip.mp4");
      });

      expect(spy).toHaveBeenCalledWith("backend.uploadFile", { kind: "video", filePath: "C:\\v\\clip.mp4" });
      expect(result.current.state.trimFailed).toBe(false);
    });

    it("passes a supplied `query` through and clears trimFailed when the server confirms the trim", async () => {
      const mockBridge = createMockBridge({ delayMs: 0 });
      const spy = vi.spyOn(mockBridge, "request");
      const { result } = renderHook(() => useSourceUpload("video", { nativeBridge: mockBridge }));

      await act(async () => {
        await result.current.uploadPath("C:\\v\\clip.mp4", "clip.mp4", {
          trim_start_sec: "1.500",
          trim_duration_sec: "4.000",
        });
      });

      expect(spy).toHaveBeenCalledWith("backend.uploadFile", {
        kind: "video",
        filePath: "C:\\v\\clip.mp4",
        query: { trim_start_sec: "1.500", trim_duration_sec: "4.000" },
      });
      expect(result.current.state.status).toBe("ready");
      expect(result.current.state.trimFailed).toBe(false);
    });

    it("sets trimFailed when a trim was requested but the response does not report `trimmed: true`", async () => {
      // Stands in for an OLD native build that drops the unknown `query` key: the
      // upload succeeds, but against the FULL file.
      const inner = createMockBridge({ delayMs: 0 });
      const legacy: NativeBridge = {
        async request<M extends BridgeMethod>(method: M, params: ParamsOf<M>): Promise<ResultOf<M>> {
          if (method === "backend.uploadFile") {
            const { query: _dropped, ...rest } = params as ParamsOf<"backend.uploadFile">;
            return inner.request(method, rest as ParamsOf<M>);
          }
          return inner.request(method, params);
        },
        requestWithFiles: (m, p, f) => inner.requestWithFiles(m, p, f),
        on: (e, h) => inner.on(e, h),
      };
      const { result } = renderHook(() => useSourceUpload("video", { nativeBridge: legacy }));

      await act(async () => {
        await result.current.uploadPath("C:\\v\\clip.mp4", "clip.mp4", {
          trim_start_sec: "1.500",
          trim_duration_sec: "4.000",
        });
      });

      // Still `ready` with a usable id — the flag, not the status, is what the
      // Chain form turns into a Generate block.
      expect(result.current.state.status).toBe("ready");
      expect(result.current.state.id).toMatch(/^mock-video-/);
      expect(result.current.state.trimFailed).toBe(true);
    });

    it("clear() drops the trimFailed flag", async () => {
      const inner = createMockBridge({ delayMs: 0 });
      const legacy: NativeBridge = {
        async request<M extends BridgeMethod>(method: M, params: ParamsOf<M>): Promise<ResultOf<M>> {
          if (method === "backend.uploadFile") {
            const { query: _dropped, ...rest } = params as ParamsOf<"backend.uploadFile">;
            return inner.request(method, rest as ParamsOf<M>);
          }
          return inner.request(method, params);
        },
        requestWithFiles: (m, p, f) => inner.requestWithFiles(m, p, f),
        on: (e, h) => inner.on(e, h),
      };
      const { result } = renderHook(() => useSourceUpload("video", { nativeBridge: legacy }));

      await act(async () => {
        await result.current.uploadPath("C:\\v\\clip.mp4", "clip.mp4", {
          trim_start_sec: "0.000",
          trim_duration_sec: "1.000",
        });
      });
      expect(result.current.state.trimFailed).toBe(true);

      act(() => result.current.clear());
      expect(result.current.state.trimFailed).toBe(false);
    });

    // §1-15: `trimFailed` is keyed off the TRIM keys, not off "a query was
    // passed at all" — the chain reference video uploads with a `max_frames`
    // cap for which `trimmed: false` is the normal, correct answer.
    it("a NON-trim query (max_frames) never arms trimFailed, even with `trimmed: false`", async () => {
      const mockBridge = createMockBridge({ delayMs: 0 });
      const spy = vi.spyOn(mockBridge, "request");
      const { result } = renderHook(() => useSourceUpload("video", { nativeBridge: mockBridge }));

      await act(async () => {
        await result.current.uploadPath("C:\\v\\long.mp4", "long.mp4", { max_frames: "11544" });
      });

      // The query IS sent (the cap has to reach the server)...
      expect(spy).toHaveBeenCalledWith("backend.uploadFile", {
        kind: "video",
        filePath: "C:\\v\\long.mp4",
        query: { max_frames: "11544" },
      });
      // ...and the mock answers `trimmed: false` for it (no trim keys), which
      // must NOT read as "the trim we asked for did not happen".
      expect(result.current.state.status).toBe("ready");
      expect(result.current.state.trimFailed).toBe(false);
    });

    it("still arms trimFailed when the trim keys ride ALONGSIDE a non-trim key", async () => {
      const inner = createMockBridge({ delayMs: 0 });
      const legacy: NativeBridge = {
        async request<M extends BridgeMethod>(method: M, params: ParamsOf<M>): Promise<ResultOf<M>> {
          if (method === "backend.uploadFile") {
            const { query: _dropped, ...rest } = params as ParamsOf<"backend.uploadFile">;
            return inner.request(method, rest as ParamsOf<M>);
          }
          return inner.request(method, params);
        },
        requestWithFiles: (m, p, f) => inner.requestWithFiles(m, p, f),
        on: (e, h) => inner.on(e, h),
      };
      const { result } = renderHook(() => useSourceUpload("video", { nativeBridge: legacy }));

      await act(async () => {
        await result.current.uploadPath("C:\\v\\clip.mp4", "clip.mp4", {
          max_frames: "11544",
          trim_start_sec: "1.500",
          trim_duration_sec: "4.000",
        });
      });

      expect(result.current.state.trimFailed).toBe(true);
    });
  });
});
