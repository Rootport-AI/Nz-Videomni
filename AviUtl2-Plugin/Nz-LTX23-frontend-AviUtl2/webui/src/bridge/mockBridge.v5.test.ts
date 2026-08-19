import { describe, expect, it, vi } from "vitest";
import { createMockBridge } from "./mockBridge";
import { TIMELINE_MENU_INVOKED_EVENT } from "./types";

describe("createMockBridge — contract v5 (timeline generation-AI bridge)", () => {
  describe("timeline.getSelection", () => {
    it("resolves the default selection snapshot", async () => {
      const bridge = createMockBridge({ delayMs: 0 });
      const result = await bridge.request("timeline.getSelection", {});
      expect(result.hasRange).toBe(true);
      expect(result.selected).toHaveLength(1);
      expect(result.selected[0]).toMatchObject({ layer: 1, frameStart: 0, frameEnd: 120 });
      // Project fields mirror getEditInfo so frame<->time math lines up.
      expect(result.rate).toBe(30);
      expect(result.sampleRate).toBe(44100);
    });

    it("merges a selection override over the default", async () => {
      const bridge = createMockBridge({
        delayMs: 0,
        selection: { hasRange: false, selected: [] },
      });
      const result = await bridge.request("timeline.getSelection", {});
      expect(result.hasRange).toBe(false);
      expect(result.selected).toEqual([]);
      expect(result.cursorLayer).toBe(1); // untouched default
    });

    it("rejects with NO_EDIT_HANDLE when failGetSelection is set", async () => {
      const bridge = createMockBridge({ delayMs: 0, failGetSelection: true });
      await expect(bridge.request("timeline.getSelection", {})).rejects.toMatchObject({
        code: "NO_EDIT_HANDLE",
      });
    });
  });

  describe("timeline.cutoutRange", () => {
    it("resolves a mp4 filePath echoing frameCount and withAudio", async () => {
      const bridge = createMockBridge({ delayMs: 0 });
      const result = await bridge.request("timeline.cutoutRange", {
        layer: 2,
        frameStart: 10,
        frameCount: 60,
        withAudio: true,
        audioMode: "mix",
        soloKeepLayers: [],
      });
      expect(result.filePath).toMatch(/\.mp4$/);
      expect(result.frameCount).toBe(60);
      expect(result.hasAudio).toBe(true);
      expect(result.width).toBe(1920);
      expect(result.height).toBe(1080);
    });

    it("reports hasAudio=false when withAudio is false", async () => {
      const bridge = createMockBridge({ delayMs: 0 });
      const result = await bridge.request("timeline.cutoutRange", {
        layer: 1,
        frameStart: 0,
        frameCount: 30,
        withAudio: false,
        audioMode: "solo",
        soloKeepLayers: [1],
      });
      expect(result.hasAudio).toBe(false);
    });

    it("mints a fresh filePath per call", async () => {
      const bridge = createMockBridge({ delayMs: 0 });
      const params = { layer: 1, frameStart: 0, frameCount: 30, withAudio: true, audioMode: "mix" as const, soloKeepLayers: [] };
      const a = await bridge.request("timeline.cutoutRange", params);
      const b = await bridge.request("timeline.cutoutRange", params);
      expect(a.filePath).not.toBe(b.filePath);
    });

    it("rejects with the configured code when failCutoutRange is set", async () => {
      const bridge = createMockBridge({ delayMs: 0, failCutoutRange: "CUTOUT_FAILED" });
      await expect(
        bridge.request("timeline.cutoutRange", {
          layer: 1,
          frameStart: 0,
          frameCount: 30,
          withAudio: true,
          audioMode: "mix",
          soloKeepLayers: [],
        }),
      ).rejects.toMatchObject({ code: "CUTOUT_FAILED" });
    });
  });

  describe("timeline.extractAudio", () => {
    it("resolves a wav filePath with a duration derived from frameCount/rate", async () => {
      const bridge = createMockBridge({ delayMs: 0 });
      const result = await bridge.request("timeline.extractAudio", {
        layer: 1,
        frameStart: 0,
        frameCount: 60,
        audioMode: "mix",
        soloKeepLayers: [],
      });
      expect(result.filePath).toMatch(/\.wav$/);
      // 60 frames at rate 30 / scale 1 == 2 seconds.
      expect(result.durationSec).toBeCloseTo(2, 3);
      expect(result.sampleRate).toBe(44100);
      expect(result.hasAudioStream).toBe(true);
    });

    it("rejects with the configured code when failExtractAudio is set", async () => {
      const bridge = createMockBridge({ delayMs: 0, failExtractAudio: "EXTRACT_FAILED" });
      await expect(
        bridge.request("timeline.extractAudio", {
          layer: 1,
          frameStart: 0,
          frameCount: 30,
          audioMode: "mix",
          soloKeepLayers: [],
        }),
      ).rejects.toMatchObject({ code: "EXTRACT_FAILED" });
    });
  });

  describe("provisional placeholder lifecycle", () => {
    // rate 30 / scale 1 -> project fps 30; 49 frames at genFps 30 stays 49
    // frames long. Placement "C" resolves to the cursor (layer 3, frame 100).
    const reserve = {
      jobId: "job-1",
      displayText: "生成中…",
      numFrames: 49,
      genFps: 30,
      placement: "C" as const,
      cursorLayer: 3,
      cursorFrame: 100,
    };

    it("insertProvisional reserves a placeholder and returns its coordinates + objectName", async () => {
      const bridge = createMockBridge({ delayMs: 0 });
      const result = await bridge.request("timeline.insertProvisional", reserve);
      expect(result.inserted).toBe(true);
      expect(result.layer).toBe(3);
      expect(result.frame).toBe(100);
      expect(result.objectName).toContain("job-1");
    });

    it("resolveProvisional reports mode 'replaced' when the placeholder is still present", async () => {
      const bridge = createMockBridge({ delayMs: 0 });
      await bridge.request("timeline.insertProvisional", reserve);
      const result = await bridge.request("timeline.resolveProvisional", {
        jobId: "job-1",
        videoFilePath: "C:\\out\\job-1.mp4",
        reservedLayer: 3,
        reservedFrame: 100,
        lengthFrames: 49,
      });
      expect(result.mode).toBe("replaced");
      expect(result.layer).toBe(3);
      expect(result.frame).toBe(100);
    });

    it("resolveProvisional reports 'insertedReserved' when no placeholder exists (e.g. user deleted it)", async () => {
      const bridge = createMockBridge({ delayMs: 0 });
      const result = await bridge.request("timeline.resolveProvisional", {
        jobId: "never-reserved",
        videoFilePath: "C:\\out\\x.mp4",
        reservedLayer: 5,
        reservedFrame: 200,
        lengthFrames: 17,
      });
      expect(result.mode).toBe("insertedReserved");
    });

    it("updateProvisionalText reports updated=true for a known placeholder, false otherwise", async () => {
      const bridge = createMockBridge({ delayMs: 0 });
      await bridge.request("timeline.insertProvisional", reserve);
      const known = await bridge.request("timeline.updateProvisionalText", {
        jobId: "job-1",
        layer: 3,
        frame: 100,
        text: "50%",
      });
      expect(known.updated).toBe(true);

      const unknown = await bridge.request("timeline.updateProvisionalText", {
        jobId: "ghost",
        layer: 0,
        frame: 0,
        text: "x",
      });
      expect(unknown.updated).toBe(false);
    });

    it("scanProvisionals returns the configured orphans (empty by default)", async () => {
      const empty = createMockBridge({ delayMs: 0 });
      await expect(empty.request("timeline.scanProvisionals", {})).resolves.toEqual({ orphans: [] });

      const withOrphans = createMockBridge({
        delayMs: 0,
        provisionalOrphans: [{ jobId: "dead-1", layer: 2, frame: 40 }],
      });
      const result = await withOrphans.request("timeline.scanProvisionals", {});
      expect(result.orphans).toEqual([{ jobId: "dead-1", layer: 2, frame: 40 }]);
    });

    it("rejects with PROVISIONAL_FAILED when failProvisional is set", async () => {
      const bridge = createMockBridge({ delayMs: 0, failProvisional: true });
      await expect(bridge.request("timeline.insertProvisional", reserve)).rejects.toMatchObject({
        code: "PROVISIONAL_FAILED",
      });
    });
  });

  // I3: staged migration (numFrames+genFps length, placement position) and the
  // new updateProvisionalReservation RPC.
  describe("provisional reservation lifecycle (I3)", () => {
    it("insertProvisional accepts the new numFrames+genFps + placement shape", async () => {
      const bridge = createMockBridge({ delayMs: 0 });
      const result = await bridge.request("timeline.insertProvisional", {
        jobId: "pending-abc",
        displayText: "⏳ Reserved",
        // rate 30 / scale 1 -> project fps 30; a 49-frame clip at genFps 30
        // stays 49 frames long.
        numFrames: 49,
        genFps: 30,
        // Placement (A): after the material (materialFrameEnd is inclusive).
        placement: "A",
        materialLayer: 2,
        materialFrameStart: 10,
        materialFrameEnd: 60,
      });
      expect(result.inserted).toBe(true);
      expect(result.layer).toBe(2); // materialLayer
      expect(result.frame).toBe(61); // materialFrameEnd + 1
      expect(result.placedLayer).toBe(2);
      expect(result.placedFrame).toBe(61);
      expect(result.usedFallback).toBe(false);
    });

    it("insertProvisional placement B resolves to layer_max+1 at the material start", async () => {
      const bridge = createMockBridge({ delayMs: 0 });
      const result = await bridge.request("timeline.insertProvisional", {
        jobId: "pending-b",
        displayText: "⏳ Reserved",
        numFrames: 49,
        genFps: 30,
        placement: "B",
        materialLayer: 2,
        materialFrameStart: 40,
        materialFrameEnd: 88,
      });
      expect(result.layer).toBe(11); // MOCK_LAYER_MAX (10) + 1
      expect(result.frame).toBe(40); // material start
    });

    it("updateProvisionalReservation deletes the old placeholder and creates the new one", async () => {
      const bridge = createMockBridge({ delayMs: 0 });
      // Seed an existing reservation.
      await bridge.request("timeline.insertProvisional", {
        jobId: "pending-1",
        displayText: "⏳ Reserved",
        numFrames: 49,
        genFps: 30,
        placement: "C",
        cursorLayer: 3,
        cursorFrame: 100,
      });
      const result = await bridge.request("timeline.updateProvisionalReservation", {
        oldJobId: "pending-1",
        newJobId: "job-42",
        displayText: "⏳ Generating",
        numFrames: 49,
        genFps: 30,
        placement: "C",
        cursorLayer: 4,
        cursorFrame: 200,
      });
      expect(result.ok).toBe(true);
      expect(result.deletedOld).toBe(true);
      expect(result.placedLayer).toBe(4);
      expect(result.placedFrame).toBe(200);
      expect(result.usedFallback).toBe(false);

      // The old placeholder is gone; the new one is now tracked.
      const oldResolve = await bridge.request("timeline.updateProvisionalText", {
        jobId: "pending-1",
        layer: 3,
        frame: 100,
        text: "x",
      });
      expect(oldResolve.updated).toBe(false);
      const newResolve = await bridge.request("timeline.updateProvisionalText", {
        jobId: "job-42",
        layer: 4,
        frame: 200,
        text: "50%",
      });
      expect(newResolve.updated).toBe(true);
    });

    it("updateProvisionalReservation reports deletedOld=false for an unknown/empty oldJobId (create-only)", async () => {
      const bridge = createMockBridge({ delayMs: 0 });
      const result = await bridge.request("timeline.updateProvisionalReservation", {
        oldJobId: "",
        newJobId: "job-new",
        displayText: "⏳ Reserved",
        numFrames: 49,
        genFps: 30,
        placement: "C",
        cursorLayer: 1,
        cursorFrame: 5,
      });
      expect(result.deletedOld).toBe(false);
      expect(result.ok).toBe(true);
    });

    it("updateProvisionalReservation rejects with PROVISIONAL_FAILED when failProvisional is set", async () => {
      const bridge = createMockBridge({ delayMs: 0, failProvisional: true });
      await expect(
        bridge.request("timeline.updateProvisionalReservation", {
          oldJobId: "",
          newJobId: "job-new",
          displayText: "x",
          numFrames: 49,
          genFps: 30,
          placement: "C",
          cursorLayer: 1,
          cursorFrame: 5,
        }),
      ).rejects.toMatchObject({ code: "PROVISIONAL_FAILED" });
    });
  });

  describe("timeline.menuInvoked event (on/emit)", () => {
    it("delivers an emitted event to subscribers and stops after unsubscribe", async () => {
      const bridge = createMockBridge({ delayMs: 0 });
      const handler = vi.fn();
      const unsubscribe = bridge.on(TIMELINE_MENU_INVOKED_EVENT, handler);

      const selection = await bridge.request("timeline.getSelection", {});
      bridge.emit(TIMELINE_MENU_INVOKED_EVENT, { action: "cutout", selection });
      expect(handler).toHaveBeenCalledTimes(1);
      expect(handler).toHaveBeenCalledWith({ action: "cutout", selection });

      unsubscribe();
      bridge.emit(TIMELINE_MENU_INVOKED_EVENT, { action: "cutout", selection });
      expect(handler).toHaveBeenCalledTimes(1);
    });
  });
});
