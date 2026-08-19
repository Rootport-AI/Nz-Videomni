import { describe, expect, it, vi } from "vitest";
import type { NativeBridge } from "../bridge";
import { createMockBridge } from "../bridge/mockBridge";
import { downloadAndInsertVideo } from "./downloadAndInsert";
import {
  getReservationPhase,
  reconcileFromTimeline,
  resetProvisionalReservation,
} from "../timeline/provisionalReservation";

/** A bridge stub recording every `request(method, params)` call. `downloadVideo`
 * returns a canned path/size, `insertMediaForJob` a canned replaced position,
 * and `insertMedia` a canned layer/frame (joined path). */
function stubBridge() {
  const request = vi.fn((method: string, _params?: unknown) => {
    if (method === "backend.downloadVideo") return Promise.resolve({ filePath: "C:/dl/out.mp4", sizeBytes: 123 });
    if (method === "timeline.insertMediaForJob")
      return Promise.resolve({ ok: true, mode: "replaced", layer: 2, frame: 5, usedFallback: false });
    if (method === "timeline.insertMedia") return Promise.resolve({ inserted: true, layer: 2, frame: 5 });
    return Promise.resolve({});
  });
  const bridge = { request, requestWithFiles: vi.fn(), on: vi.fn(() => () => {}) } as unknown as NativeBridge;
  return { bridge, request };
}

/** Params the stub recorded for a given method (its first matching call). */
function paramsFor(request: ReturnType<typeof stubBridge>["request"], method: string): Record<string, unknown> {
  const call = request.mock.calls.find((c) => c[0] === method);
  return (call?.[1] ?? {}) as Record<string, unknown>;
}

describe("downloadAndInsertVideo", () => {
  it("plain path passes reuseIfPresent:true to backend.downloadVideo and inserts via insertMediaForJob (one call)", async () => {
    resetProvisionalReservation();
    const { bridge, request } = stubBridge();
    const result = await downloadAndInsertVideo(bridge, "job-1");

    expect(paramsFor(request, "backend.downloadVideo")).toEqual({ jobId: "job-1", reuseIfPresent: true });
    // The per-clip insert is the single atomic replace-insert RPC, not the old
    // insertMedia + deleteProvisionalByJob pair.
    expect(paramsFor(request, "timeline.insertMediaForJob")).toEqual({ jobId: "job-1", filePath: "C:/dl/out.mp4" });
    expect(request.mock.calls.filter((c) => c[0] === "timeline.insertMediaForJob")).toHaveLength(1);
    expect(request.mock.calls.some((c) => c[0] === "timeline.insertMedia")).toBe(false);
    expect(request.mock.calls.some((c) => c[0] === "timeline.deleteProvisionalByJob")).toBe(false);
    expect(result).toEqual({ layer: 2, frame: 5, filePath: "C:/dl/out.mp4" });
  });

  it("joined path does NOT pass reuseIfPresent AND inserts via the plain insertMedia (never insertMediaForJob, Med 6)", async () => {
    resetProvisionalReservation();
    const { bridge, request } = stubBridge();
    await downloadAndInsertVideo(bridge, "job-2", undefined, { joined: true });

    const params = paramsFor(request, "backend.downloadVideo");
    expect(params).toEqual({ jobId: "job-2", joined: true });
    expect(params).not.toHaveProperty("reuseIfPresent");
    // The joined clip must never replace a per-clip marker: plain insertMedia,
    // no insertMediaForJob, no marker delete.
    expect(paramsFor(request, "timeline.insertMedia")).toEqual({ filePath: "C:/dl/out.mp4" });
    expect(request.mock.calls.some((c) => c[0] === "timeline.insertMediaForJob")).toBe(false);
    expect(request.mock.calls.some((c) => c[0] === "timeline.deleteProvisionalByJob")).toBe(false);
  });

  it("plainInsertAt (W3) inserts via a positioned plain insertMedia and never the replace-RPC (provisional survives)", async () => {
    resetProvisionalReservation();
    const { bridge, request } = stubBridge();
    const result = await downloadAndInsertVideo(bridge, "job-w3", undefined, {
      plainInsertAt: { layer: 4, frame: 360 },
    });

    // Still downloads the plain per-clip output (reuseIfPresent), then drops it
    // at the exact right-click layer/frame via the plain insertMedia.
    expect(paramsFor(request, "backend.downloadVideo")).toEqual({ jobId: "job-w3", reuseIfPresent: true });
    expect(paramsFor(request, "timeline.insertMedia")).toEqual({ filePath: "C:/dl/out.mp4", layer: 4, frame: 360 });
    // The W3 ⬇ is additive: it must NOT go through the per-clip replace-RPC, so a
    // provisional object sitting at the cursor is left intact.
    expect(request.mock.calls.some((c) => c[0] === "timeline.insertMediaForJob")).toBe(false);
    expect(request.mock.calls.some((c) => c[0] === "timeline.deleteProvisionalByJob")).toBe(false);
    expect(result).toEqual({ layer: 2, frame: 5, filePath: "C:/dl/out.mp4" });
  });

  it("knownFilePath skips backend.downloadVideo entirely and inserts the given path via insertMediaForJob", async () => {
    resetProvisionalReservation();
    const { bridge, request } = stubBridge();
    const onPhase = vi.fn();
    const result = await downloadAndInsertVideo(bridge, "job-3", onPhase, { knownFilePath: "C:/kept/prev.mp4" });

    expect(request.mock.calls.filter((c) => c[0] === "backend.downloadVideo")).toHaveLength(0);
    expect(paramsFor(request, "timeline.insertMediaForJob")).toEqual({ jobId: "job-3", filePath: "C:/kept/prev.mp4" });
    expect(result).toEqual({ layer: 2, frame: 5, filePath: "C:/kept/prev.mp4" });
    // No "downloading" phase when the download is skipped — straight to inserting.
    expect(onPhase).not.toHaveBeenCalledWith("downloading");
    expect(onPhase).toHaveBeenCalledWith("inserting");
  });

  it("releases the reservation seat on a successful insert (adversarial-review Med 7)", async () => {
    // Simulate the post-reload state: the seat was restored to `generating` from
    // a surviving job-bound marker (reconcileFromTimeline). A successful 🎞 insert
    // must free that seat so no false ⏳生成中 busy-guard lingers.
    resetProvisionalReservation();
    const seatBridge = {
      request: vi.fn((method: string) => {
        if (method === "timeline.scanProvisionals")
          return Promise.resolve({ orphans: [{ jobId: "job-seat", layer: 0, frame: 0 }] });
        if (method === "backend.downloadVideo") return Promise.resolve({ filePath: "C:/dl/out.mp4", sizeBytes: 1 });
        if (method === "timeline.insertMediaForJob")
          return Promise.resolve({ ok: true, mode: "replaced", layer: 2, frame: 5, usedFallback: false });
        return Promise.resolve({});
      }),
      requestWithFiles: vi.fn(),
      on: vi.fn(() => () => {}),
    } as unknown as NativeBridge;

    await reconcileFromTimeline(seatBridge);
    expect(getReservationPhase()).toBe("generating");

    await downloadAndInsertVideo(seatBridge, "job-seat");
    expect(getReservationPhase()).toBe("idle");
  });

  it("does NOT release the seat when the insert fails (error propagates)", async () => {
    resetProvisionalReservation();
    const request = vi.fn((method: string) => {
      if (method === "backend.downloadVideo") return Promise.resolve({ filePath: "C:/dl/out.mp4", sizeBytes: 1 });
      if (method === "timeline.insertMediaForJob") return Promise.reject(new Error("NO_EDIT_HANDLE"));
      return Promise.resolve({});
    });
    const bridge = { request, requestWithFiles: vi.fn(), on: vi.fn(() => () => {}) } as unknown as NativeBridge;

    await expect(downloadAndInsertVideo(bridge, "job-6")).rejects.toThrow("NO_EDIT_HANDLE");
  });
});

describe("downloadAndInsertVideo against the mock bridge (replace vs. insert mode)", () => {
  it("returns mode:\"replaced\" (marker present) then mode:\"inserted\" on a re-insert after the marker is gone", async () => {
    resetProvisionalReservation();
    const bridge = createMockBridge({ delayMs: 0 });

    // Seed a completed job + its provisional marker so the first insert replaces.
    const gen = await bridge.request("backend.request", {
      method: "POST",
      path: "/api/v1/generate",
      body: { prompt: "hi", width: 512, height: 320, num_frames: 49 },
    });
    const jobId = (gen.body as { job_id: string }).job_id;
    for (let i = 0; i < 12; i += 1) {
      const r = await bridge.request("backend.request", { method: "GET", path: `/api/v1/jobs/${jobId}` });
      if ((r.body as { status: string }).status === "completed") break;
    }
    await bridge.request("timeline.insertProvisional", {
      jobId,
      displayText: "done",
      numFrames: 49,
      genFps: 24,
      placement: "C",
      cursorLayer: 0,
      cursorFrame: 0,
    });

    const spy = vi.spyOn(bridge, "request");
    // First 🎞: marker present -> replaced, and the marker is consumed.
    await downloadAndInsertVideo(bridge, jobId);
    const first = spy.mock.calls.findIndex((c) => c[0] === "timeline.insertMediaForJob");
    expect(await spy.mock.results[first]!.value).toMatchObject({ mode: "replaced" });
    const scan = await bridge.request("timeline.scanProvisionals", {});
    expect(scan.orphans.map((o) => o.jobId)).not.toContain(jobId);

    // Re-insert 🎞 (knownFilePath, marker already gone) -> inserted.
    spy.mockClear();
    await downloadAndInsertVideo(bridge, jobId, undefined, { knownFilePath: "C:/kept/prev.mp4" });
    const second = spy.mock.calls.findIndex((c) => c[0] === "timeline.insertMediaForJob");
    expect(await spy.mock.results[second]!.value).toMatchObject({ mode: "inserted" });
  });
});
