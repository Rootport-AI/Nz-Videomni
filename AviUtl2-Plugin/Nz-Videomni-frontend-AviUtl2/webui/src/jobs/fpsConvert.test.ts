import { describe, expect, it } from "vitest";
import type { JobResponse } from "../api/types";
import { editInfoFps, formatFps, framesToSeconds, joinedInsertFrame, jobFrameRate, secondsToFrames } from "./fpsConvert";

/** Minimal `JobResponse` stub carrying only the echoed `request` these tests
 * read — everything else is irrelevant to `jobFrameRate`. */
function jobWithRequest(request: Record<string, unknown>): JobResponse {
  return { request } as unknown as JobResponse;
}

describe("fpsConvert", () => {
  describe("framesToSeconds", () => {
    it("converts frames to seconds at the given fps", () => {
      expect(framesToSeconds(120, 24)).toBe(5.0);
      expect(framesToSeconds(240, 24)).toBe(10.0);
      expect(framesToSeconds(120, 30)).toBe(4.0);
      expect(framesToSeconds(240, 30)).toBe(8.0);
    });

    it("returns 0 for 0 frames (boundary)", () => {
      expect(framesToSeconds(0, 24)).toBe(0);
    });
  });

  describe("secondsToFrames", () => {
    it("rounds to the nearest whole frame", () => {
      expect(secondsToFrames(5, 24)).toBe(120);
      expect(secondsToFrames(0, 24)).toBe(0);
      // 1.02s @ 24fps = 24.48 -> 24; 1.03s -> 24.72 -> 25 (round, not truncate).
      expect(secondsToFrames(1.02, 24)).toBe(24);
      expect(secondsToFrames(1.03, 24)).toBe(25);
    });
  });

  describe("editInfoFps", () => {
    it("resolves rate/scale to fps", () => {
      expect(editInfoFps(30, 1)).toBe(30);
      expect(editInfoFps(24000, 1001)).toBeCloseTo(23.976, 3);
    });

    it("returns null for non-positive rate or scale (boundary)", () => {
      expect(editInfoFps(0, 1)).toBeNull();
      expect(editInfoFps(30, 0)).toBeNull();
      expect(editInfoFps(-30, 1)).toBeNull();
      expect(editInfoFps(30, -1)).toBeNull();
    });
  });

  describe("jobFrameRate", () => {
    it("reads a positive frame_rate from the echoed request", () => {
      expect(jobFrameRate(jobWithRequest({ frame_rate: 30 }))).toBe(30);
    });

    it("falls back to 24 when frame_rate is missing or invalid (boundary)", () => {
      expect(jobFrameRate(jobWithRequest({}))).toBe(24);
      expect(jobFrameRate(jobWithRequest({ frame_rate: 0 }))).toBe(24);
      expect(jobFrameRate(jobWithRequest({ frame_rate: -1 }))).toBe(24);
      expect(jobFrameRate(jobWithRequest({ frame_rate: "30" }))).toBe(24);
    });
  });

  describe("formatFps", () => {
    it("drops trailing zeros and rounds to at most two decimals", () => {
      expect(formatFps(30)).toBe("30");
      expect(formatFps(29.97)).toBe("29.97");
      expect(formatFps(23.976)).toBe("23.98");
    });
  });

  describe("joinedInsertFrame (JOIN_FEATURE_RESEARCH.md §4.5)", () => {
    // A source spanning frames [100, 220] inclusive at 24fps: real length is
    // (220 - 100 + 1) / 24 = 121/24 ≈ 5.0417s. frameEnd is INCLUSIVE, so the
    // source tail-next frame is 221.
    const base = { frameStart: 100, frameEnd: 220, projectFps: 24 };

    it("keeps only the trimmed tail: head = tail-next − round(keptSeconds × fps)", () => {
      // trimmed 3.0s -> kept ≈ 2.0417s -> round(2.0417*24)=49 -> 221 - 49 = 172.
      expect(joinedInsertFrame({ ...base, trimmedSourceSeconds: 3.0 })).toBe(172);
    });

    it("trimmed 0 keeps the whole source: head lands back at frameStart (min position)", () => {
      // kept = full length -> round(121/24 * 24) = 121 -> 221 - 121 = 100 = frameStart.
      // This is the earliest the head can land; the `Math.max(frameStart, …)` clamp
      // only ever guards this boundary against a rounding overshoot.
      expect(joinedInsertFrame({ ...base, trimmedSourceSeconds: 0 })).toBe(100);
    });

    it("a trim exceeding the source keeps nothing: head lands at the tail-next frame", () => {
      // kept = max(0, 5.04 - 999) = 0 -> 221 - round(0) = 221 (the whole joined
      // clip is continuation, starting right after the original source's tail).
      expect(joinedInsertFrame({ ...base, trimmedSourceSeconds: 999 })).toBe(221);
    });

    it("places the head near the tail for a large trim within range", () => {
      // trimmed 5.0s -> kept ≈ 0.0417s -> round(0.0417*24)=1 -> 221 - 1 = 220.
      expect(joinedInsertFrame({ ...base, trimmedSourceSeconds: 5.0 })).toBe(220);
    });

    it("uses the project fps for the frame<->seconds conversion (30fps)", () => {
      // [0, 149] inclusive at 30fps: length 150/30 = 5.0s, tail-next = 150.
      // trimmed 2.0s -> kept 3.0s -> round(3*30)=90 -> 150 - 90 = 60.
      expect(joinedInsertFrame({ frameStart: 0, frameEnd: 149, projectFps: 30, trimmedSourceSeconds: 2.0 })).toBe(60);
    });
  });
});
