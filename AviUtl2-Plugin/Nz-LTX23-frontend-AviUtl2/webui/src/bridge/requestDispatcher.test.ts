import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { RequestDispatcher } from "./requestDispatcher";
import { BridgeError, type BridgeRequest } from "./types";

describe("RequestDispatcher", () => {
  let sent: BridgeRequest[];
  let dispatcher: RequestDispatcher;

  beforeEach(() => {
    sent = [];
    dispatcher = new RequestDispatcher({
      send: (request) => sent.push(request),
      timeoutMs: 1_000,
    });
  });

  afterEach(() => {
    // Clears any timers left over from tests that didn't resolve their
    // requests, and rejects (already-caught) leftover promises.
    dispatcher.dispose();
    vi.useRealTimers();
  });

  it("assigns incrementing ids starting at 1, per dispatcher instance", () => {
    void dispatcher.request("ping", {}).catch(() => {});
    void dispatcher.request("ping", {}).catch(() => {});
    void dispatcher.request("getEditInfo", {}).catch(() => {});

    expect(sent.map((r) => r.id)).toEqual([1, 2, 3]);
    expect(sent[0]?.method).toBe("ping");
    expect(sent[2]?.method).toBe("getEditInfo");
  });

  it("resolves the pending request matching the response id, even out of order", async () => {
    const pingPromise = dispatcher.request("ping", {});
    const editInfoPromise = dispatcher.request("getEditInfo", {});

    // Respond to id 2 before id 1 to prove matching is id-based, not FIFO.
    dispatcher.handleIncoming({
      id: 2,
      ok: true,
      result: {
        width: 1920,
        height: 1080,
        rate: 30,
        scale: 1,
        sampleRate: 44100,
        frame: 120,
      },
    });
    dispatcher.handleIncoming({
      id: 1,
      ok: true,
      result: { pong: true, pluginVersion: "0.1.0-M1" },
    });

    await expect(pingPromise).resolves.toEqual({
      pong: true,
      pluginVersion: "0.1.0-M1",
    });
    await expect(editInfoPromise).resolves.toMatchObject({
      width: 1920,
      height: 1080,
    });
  });

  it("rejects with a BridgeError carrying the native error code and message", async () => {
    const promise = dispatcher.request("getEditInfo", {});

    dispatcher.handleIncoming({
      id: 1,
      ok: false,
      error: { code: "NO_EDIT_HANDLE", message: "no active edit session" },
    });

    await expect(promise).rejects.toBeInstanceOf(BridgeError);
    await expect(promise).rejects.toMatchObject({
      code: "NO_EDIT_HANDLE",
      message: "no active edit session",
    });
  });

  it("times out a request that never receives a response", async () => {
    vi.useFakeTimers();

    const promise = dispatcher.request("ping", {});
    const assertion = expect(promise).rejects.toMatchObject({ code: "TIMEOUT" });

    await vi.advanceTimersByTimeAsync(1_000);
    await assertion;
  });

  it("does not resolve a late response for a request that already timed out", async () => {
    vi.useFakeTimers();

    const promise = dispatcher.request("ping", {});
    const assertion = expect(promise).rejects.toMatchObject({ code: "TIMEOUT" });
    await vi.advanceTimersByTimeAsync(1_000);
    await assertion;

    // A response arriving after the timeout should be ignored, not throw.
    expect(() =>
      dispatcher.handleIncoming({
        id: 1,
        ok: true,
        result: { pong: true, pluginVersion: "late" },
      }),
    ).not.toThrow();
  });

  it("silently ignores a response for an unknown id", () => {
    expect(() =>
      dispatcher.handleIncoming({
        id: 999,
        ok: true,
        result: { pong: true, pluginVersion: "x" },
      }),
    ).not.toThrow();
  });

  it("dispatches a native event only to subscribers of that event name", () => {
    const progressHandler = vi.fn();
    const otherHandler = vi.fn();
    dispatcher.on("job.progress", progressHandler);
    dispatcher.on("other.event", otherHandler);

    dispatcher.handleIncoming({ event: "job.progress", data: { percent: 42 } });

    expect(progressHandler).toHaveBeenCalledWith({ percent: 42 });
    expect(otherHandler).not.toHaveBeenCalled();
  });

  it("stops notifying a handler once it unsubscribes", () => {
    const handler = vi.fn();
    const unsubscribe = dispatcher.on("job.progress", handler);
    unsubscribe();

    dispatcher.handleIncoming({ event: "job.progress", data: {} });

    expect(handler).not.toHaveBeenCalled();
  });

  it("ignores incoming messages that are neither a response nor an event", () => {
    expect(() => dispatcher.handleIncoming({ foo: "bar" })).not.toThrow();
    expect(() => dispatcher.handleIncoming(null)).not.toThrow();
    expect(() => dispatcher.handleIncoming("not an object")).not.toThrow();
    expect(() => dispatcher.handleIncoming(undefined)).not.toThrow();
  });

  it("honors a per-call timeoutMs in params, overriding the dispatcher's default local wait", async () => {
    vi.useFakeTimers();

    // Default (1_000ms, from beforeEach) would already have fired by 1_000ms;
    // prove the per-call override (2_000ms) is what's actually driving this
    // particular request's local timeout.
    let settled = false;
    const promise = dispatcher
      .request("backend.request", {
        method: "GET",
        path: "/api/v1/jobs/some-id/join",
        timeoutMs: 2_000,
      })
      .catch((err) => {
        settled = true;
        throw err;
      });
    const assertion = expect(promise).rejects.toMatchObject({ code: "TIMEOUT" });

    await vi.advanceTimersByTimeAsync(1_500);
    expect(settled).toBe(false); // the 1_000ms default would have fired by now

    await vi.advanceTimersByTimeAsync(600);
    await assertion;
  });

  it("falls back to the dispatcher's default timeout when a request has no timeoutMs", async () => {
    vi.useFakeTimers();

    const promise = dispatcher.request("ping", {});
    const assertion = expect(promise).rejects.toMatchObject({ code: "TIMEOUT" });

    await vi.advanceTimersByTimeAsync(1_000);
    await assertion;
  });

  it("rejects all still-pending requests with a DISPOSED error on dispose()", async () => {
    const promise = dispatcher.request("ping", {});
    dispatcher.dispose();

    await expect(promise).rejects.toBeInstanceOf(BridgeError);
    await expect(promise).rejects.toMatchObject({ code: "DISPOSED" });
  });

  describe("ui.pickFile / ui.pickFolder (no local timeout ceiling)", () => {
    it("does not time out ui.pickFile even after far longer than the default wait", async () => {
      vi.useFakeTimers();

      const promise = dispatcher.request("ui.pickFile", { kind: "image" });
      let settled = false;
      void promise.finally(() => {
        settled = true;
      });

      // Way past the 1_000ms default configured in beforeEach.
      await vi.advanceTimersByTimeAsync(60_000);
      expect(settled).toBe(false);

      // Still resolves normally once the (slow) user response arrives.
      dispatcher.handleIncoming({
        id: 1,
        ok: true,
        result: { filePath: "C:\\clip.mp4", fileName: "clip.mp4" },
      });
      await expect(promise).resolves.toMatchObject({ fileName: "clip.mp4" });
    });

    it("does not time out ui.pickFolder even after far longer than the default wait", async () => {
      vi.useFakeTimers();

      const promise = dispatcher.request("ui.pickFolder", {});
      let settled = false;
      void promise.finally(() => {
        settled = true;
      });

      await vi.advanceTimersByTimeAsync(60_000);
      expect(settled).toBe(false);

      dispatcher.handleIncoming({
        id: 1,
        ok: true,
        result: { folderPath: "C:\\out" },
      });
      await expect(promise).resolves.toMatchObject({ folderPath: "C:\\out" });
    });

    it("still rejects a pending ui.pickFile with DISPOSED when the dispatcher is disposed", async () => {
      const promise = dispatcher.request("ui.pickFile", { kind: "image" });
      dispatcher.dispose();

      await expect(promise).rejects.toBeInstanceOf(BridgeError);
      await expect(promise).rejects.toMatchObject({ code: "DISPOSED" });
    });
  });

  it("still times out an ordinary method (e.g. ping) after the default wait, unaffected by the no-timeout allowlist", async () => {
    vi.useFakeTimers();

    const promise = dispatcher.request("ping", {});
    const assertion = expect(promise).rejects.toMatchObject({ code: "TIMEOUT" });

    await vi.advanceTimersByTimeAsync(1_000);
    await assertion;
  });
});
