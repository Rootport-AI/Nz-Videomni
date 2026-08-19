import { beforeEach, describe, expect, it, vi } from "vitest";
import { createMockBridge } from "../bridge/mockBridge";
import {
  bindToJob,
  clearJobInserted,
  getInsertedFilePath,
  getPublishedFormValues,
  getReservationPhase,
  getReservationState,
  isJobInserted,
  markJobInserted,
  publishFormValues,
  reconcileFromTimeline,
  releaseIfSettled,
  reserveAtCursor,
  reservePlacement,
  resetProvisionalReservation,
  rollbackReservedPlacement,
  settleProvisionalText,
  subscribeInserted,
} from "./provisionalReservation";
import {
  getSourceLocation,
  recordSourceLocation,
  resetSourceLocationMap,
} from "./sourceLocationMap";
import type { MockBridgeOptions, NativeBridge } from "../bridge";

// The single reservation seat is module-level state (one seat per app, spec
// §5-6), so every test starts from a clean idle seat.
beforeEach(() => {
  resetProvisionalReservation();
});

const CURSOR = { cursorLayer: 3, cursorFrame: 100, numFrames: 49, genFps: 30, displayText: "(t2v)" };

describe("reserveAtCursor", () => {
  it("inserts a fresh provisional at the cursor (placement C) and enters `reserved`", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    const spy = vi.spyOn(bridge, "request");

    const result = await reserveAtCursor(bridge, CURSOR);

    // One insert call, no reservation-move call.
    expect(spy).toHaveBeenCalledTimes(1);
    const [method, params] = spy.mock.calls[0]!;
    expect(method).toBe("timeline.insertProvisional");
    expect(params).toMatchObject({
      jobId: result.pendingId,
      placement: "C",
      numFrames: 49,
      genFps: 30,
      cursorLayer: 3,
      cursorFrame: 100,
      displayText: "(t2v)",
    });

    // The mock resolves placement C to the cursor position.
    expect(result.moved).toBe(false);
    expect(result.placedLayer).toBe(3);
    expect(result.placedFrame).toBe(100);
    expect(result.usedFallback).toBe(false);
    expect(result.pendingId).toMatch(/^pending-/);

    expect(getReservationPhase()).toBe("reserved");
    const state = getReservationState();
    expect(state.pendingId).toBe(result.pendingId);
    expect(state.jobId).toBeNull();
    expect(state.placedLayer).toBe(3);
    expect(state.placedFrame).toBe(100);
  });

  it("moves the existing (unbound) reservation instead of inserting a second one", async () => {
    const bridge = createMockBridge({ delayMs: 0 });

    const first = await reserveAtCursor(bridge, CURSOR);
    const spy = vi.spyOn(bridge, "request");

    const second = await reserveAtCursor(bridge, {
      ...CURSOR,
      cursorLayer: 5,
      cursorFrame: 250,
    });

    // The second click MOVES via updateProvisionalReservation (not a 2nd insert).
    expect(spy).toHaveBeenCalledTimes(1);
    const [method, params] = spy.mock.calls[0]!;
    expect(method).toBe("timeline.updateProvisionalReservation");
    expect(params).toMatchObject({
      oldJobId: first.pendingId,
      newJobId: second.pendingId,
      placement: "C",
      cursorLayer: 5,
      cursorFrame: 250,
    });
    expect(first.pendingId).not.toBe(second.pendingId);

    expect(second.moved).toBe(true);
    expect(second.placedLayer).toBe(5);
    expect(second.placedFrame).toBe(250);

    // Seat is still a single reservation, now with the new id/position.
    expect(getReservationPhase()).toBe("reserved");
    expect(getReservationState().pendingId).toBe(second.pendingId);
  });
});

describe("reservePlacement (§5-9 systems A/B/C/D)", () => {
  const LEN = { numFrames: 49, genFps: 30 };
  const MATERIAL = { layer: 2, frameStart: 10, frameEnd: 130 };

  it("placement A inserts after the material (materialFrameEnd+1) with the material range", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    const spy = vi.spyOn(bridge, "request");

    const result = await reservePlacement(bridge, {
      placement: "A",
      material: MATERIAL,
      ...LEN,
      displayText: "(gen)",
    });

    const [method, params] = spy.mock.calls[0]!;
    expect(method).toBe("timeline.insertProvisional");
    expect(params).toMatchObject({
      placement: "A",
      materialLayer: 2,
      materialFrameStart: 10,
      materialFrameEnd: 130,
    });
    // No cursor fields leak onto a material placement.
    expect(params).not.toHaveProperty("cursorLayer");
    // Mock resolves A to {materialLayer, materialFrameEnd+1}.
    expect(result.placedLayer).toBe(2);
    expect(result.placedFrame).toBe(131);
    expect(result.moved).toBe(false);
    expect(getReservationPhase()).toBe("reserved");
  });

  it("placement B overlays head-aligned on layer_max+1 with the material start", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    const spy = vi.spyOn(bridge, "request");

    const result = await reservePlacement(bridge, {
      placement: "B",
      material: MATERIAL,
      ...LEN,
      displayText: "(gen)",
    });

    const [, params] = spy.mock.calls[0]!;
    expect(params).toMatchObject({ placement: "B", materialFrameStart: 10 });
    // Mock's MOCK_LAYER_MAX is 10 -> layer_max+1 = 11; frame = material start.
    expect(result.placedLayer).toBe(11);
    expect(result.placedFrame).toBe(10);
  });

  // W0 (2026-08-09) 系統D: Retake head-aligns the provisional with the user's
  // SELECTED FRAME RANGE. Native has no "D" (its `ParsePlacementFields` rejects
  // anything but A/B/C), and D's formula is identical to B's, so the wire must
  // carry "B" — the difference lives entirely in what the CALLER puts in
  // `material` (the range, not the object's own span).
  it("placement D maps onto native's B on the wire", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    const spy = vi.spyOn(bridge, "request");

    const result = await reservePlacement(bridge, {
      placement: "D",
      // The RANGE (frames 48..96), deliberately different from MATERIAL's own
      // 10..130, so a regression that used the object's span would fail here.
      material: { layer: 2, frameStart: 48, frameEnd: 96 },
      ...LEN,
      displayText: "(retake)",
    });

    const [method, params] = spy.mock.calls[0]!;
    expect(method).toBe("timeline.insertProvisional");
    // "D" must NEVER reach native.
    expect(params).toMatchObject({ placement: "B", materialFrameStart: 48, materialFrameEnd: 96 });
    expect(params).not.toMatchObject({ placement: "D" });
    expect(params).not.toHaveProperty("cursorLayer");
    // Resolves exactly like B: layer_max+1 (mock's MOCK_LAYER_MAX is 10), and
    // the frame is the RANGE start.
    expect(result.placedLayer).toBe(11);
    expect(result.placedFrame).toBe(48);
    expect(getReservationPhase()).toBe("reserved");
  });

  it("placement D maps onto B on the MOVE path too (updateProvisionalReservation)", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    // Seat first taken by a cursor reservation, then moved by a Retake click.
    await reservePlacement(bridge, { placement: "C", cursor: { layer: 3, frame: 100 }, ...LEN, displayText: "(t2v)" });
    const spy = vi.spyOn(bridge, "request");

    const moved = await reservePlacement(bridge, {
      placement: "D",
      material: { layer: 2, frameStart: 48, frameEnd: 96 },
      ...LEN,
      displayText: "(retake)",
    });

    const [method, params] = spy.mock.calls[0]!;
    expect(method).toBe("timeline.updateProvisionalReservation");
    expect(params).toMatchObject({ placement: "B", materialFrameStart: 48 });
    expect(moved.moved).toBe(true);
    expect(moved.placedFrame).toBe(48);
  });

  // 素材（末尾）v2 系統E: the same "WebUI-only系統, mapped onto B, caller supplies
  // the shifted frame" discipline as D — verified separately because the two
  // carry DIFFERENT numbers in `material.frameStart` (D: the selected range's
  // start; E: `tailAlign.tailAlignedStartFrame`'s already-shifted value).
  it("placement E maps onto native's B and passes the SHIFTED frame through untouched", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    const spy = vi.spyOn(bridge, "request");

    // 末尾合わせ: the material starts at 500, the caller already subtracted the
    // newly-generated part and hands over 260.
    const result = await reservePlacement(bridge, {
      placement: "E",
      material: { layer: 2, frameStart: 260, frameEnd: 620 },
      ...LEN,
      displayText: "(end with this)",
    });

    const [method, params] = spy.mock.calls[0]!;
    expect(method).toBe("timeline.insertProvisional");
    // "E" must NEVER reach native — `bridge/types.ts` types the wire field as
    // "A" | "B" | "C", so this is also a compile-time guarantee.
    expect(params).toMatchObject({ placement: "B", materialFrameStart: 260, materialFrameEnd: 620 });
    expect(params).not.toMatchObject({ placement: "E" });
    expect(params).not.toHaveProperty("cursorLayer");
    // Resolves exactly like B: layer_max+1 (mock's MOCK_LAYER_MAX is 10), at the
    // frame the CALLER computed — the shift is never re-derived down here.
    expect(result.placedLayer).toBe(11);
    expect(result.placedFrame).toBe(260);
    expect(getReservationPhase()).toBe("reserved");
  });

  it("placement E maps onto B on the MOVE path too (the Generate-time 打ち直し)", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    // The right-click's own reservation, placed with the ESTIMATED band…
    await reservePlacement(bridge, {
      placement: "E",
      material: { layer: 2, frameStart: 300, frameEnd: 620 },
      ...LEN,
      displayText: "(end with this)",
    });
    const spy = vi.spyOn(bridge, "request");

    // …then re-placed at Generate with the confirmed one.
    const moved = await reservePlacement(bridge, {
      placement: "E",
      material: { layer: 2, frameStart: 260, frameEnd: 620 },
      ...LEN,
      displayText: "(end with this)",
    });

    const [method, params] = spy.mock.calls[0]!;
    expect(method).toBe("timeline.updateProvisionalReservation");
    expect(params).toMatchObject({ placement: "B", materialFrameStart: 260 });
    expect(moved.moved).toBe(true);
    expect(moved.placedFrame).toBe(260);
  });

  it("placement C reserves at the cursor", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    const spy = vi.spyOn(bridge, "request");

    const result = await reservePlacement(bridge, {
      placement: "C",
      cursor: { layer: 3, frame: 100 },
      ...LEN,
      displayText: "(t2v)",
    });

    const [, params] = spy.mock.calls[0]!;
    expect(params).toMatchObject({ placement: "C", cursorLayer: 3, cursorFrame: 100 });
    expect(params).not.toHaveProperty("materialLayer");
    expect(result.placedLayer).toBe(3);
    expect(result.placedFrame).toBe(100);
  });

  it("moves the single seat across placement systems (reserved -> updateProvisionalReservation)", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    // First reserve at the cursor (C), then re-click a material item (A).
    const first = await reservePlacement(bridge, {
      placement: "C",
      cursor: { layer: 3, frame: 100 },
      ...LEN,
      displayText: "(t2v)",
    });
    const spy = vi.spyOn(bridge, "request");

    const second = await reservePlacement(bridge, {
      placement: "A",
      material: MATERIAL,
      ...LEN,
      displayText: "(gen)",
    });

    const [method, params] = spy.mock.calls[0]!;
    expect(method).toBe("timeline.updateProvisionalReservation");
    expect(params).toMatchObject({
      oldJobId: first.pendingId,
      newJobId: second.pendingId,
      placement: "A",
      materialFrameEnd: 130,
    });
    expect(second.moved).toBe(true);
    // Still a single seat.
    expect(getReservationState().pendingId).toBe(second.pendingId);
  });

  it("carries the v2v source-location memo onto the new pending id when the seat moves", async () => {
    // I6 Join: #1 extend-video records the SOURCE object's timeline location
    // keyed by the reservation's pending id. A move mints a NEW pending id, so
    // the memo must follow the seat — otherwise it is orphaned under the retired
    // id and the joined 🎞 insert falls back to "position unknown".
    resetSourceLocationMap();
    const bridge = createMockBridge({ delayMs: 0 });
    const first = await reservePlacement(bridge, {
      placement: "A",
      material: MATERIAL,
      ...LEN,
      displayText: "(v2v)",
    });
    const memo = {
      layer: 2,
      frameStart: 10,
      frameEnd: 130,
      filePath: "C:\\src.mp4",
      rate: 30,
      scale: 1,
    };
    recordSourceLocation(first.pendingId, memo);

    const second = await reservePlacement(bridge, {
      placement: "C",
      cursor: { layer: 3, frame: 100 },
      ...LEN,
      displayText: "(v2v moved)",
    });

    expect(second.pendingId).not.toBe(first.pendingId);
    expect(getSourceLocation(second.pendingId)).toEqual(memo);
    expect(getSourceLocation(first.pendingId)).toBeNull();
  });

  it("a move with no recorded memo leaves the source-location map untouched", async () => {
    // The vast majority of moves (✨/📷/Retake) never recorded anything; the
    // carry-over must stay a no-op there rather than inventing an entry.
    resetSourceLocationMap();
    const bridge = createMockBridge({ delayMs: 0 });
    const first = await reservePlacement(bridge, {
      placement: "C",
      cursor: { layer: 3, frame: 100 },
      ...LEN,
      displayText: "(t2v)",
    });
    const second = await reservePlacement(bridge, {
      placement: "A",
      material: MATERIAL,
      ...LEN,
      displayText: "(gen)",
    });

    expect(second.moved).toBe(true);
    expect(getSourceLocation(first.pendingId)).toBeNull();
    expect(getSourceLocation(second.pendingId)).toBeNull();
  });
});

describe("rollbackReservedPlacement (X2(a) 送信失敗時のロールバック)", () => {
  it("deletes the placeholder and frees the seat when merely reserved", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    const reservation = await reserveAtCursor(bridge, CURSOR);
    expect(getReservationPhase()).toBe("reserved");
    const spy = vi.spyOn(bridge, "request");

    const rolled = await rollbackReservedPlacement(bridge);

    expect(rolled).toBe(true);
    // The still-unbound placeholder is deleted by its pending id.
    expect(spy).toHaveBeenCalledTimes(1);
    const [method, params] = spy.mock.calls[0]!;
    expect(method).toBe("timeline.deleteProvisionalByJob");
    expect(params).toMatchObject({ jobId: reservation.pendingId });
    // Seat returns to idle.
    expect(getReservationPhase()).toBe("idle");
    expect(getReservationState().pendingId).toBeNull();
  });

  it("is a no-op (returns false, no RPC) when a real job is bound (generating)", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    await reserveAtCursor(bridge, CURSOR);
    await bindToJob(bridge, { jobId: "job-live", numFrames: 49, genFps: 30, displayText: "x" });
    expect(getReservationPhase()).toBe("generating");
    const spy = vi.spyOn(bridge, "request");

    const rolled = await rollbackReservedPlacement(bridge);

    expect(rolled).toBe(false);
    expect(spy).not.toHaveBeenCalled();
    // A bound (generating) real job is never touched.
    expect(getReservationPhase()).toBe("generating");
    expect(getReservationState().jobId).toBe("job-live");
  });

  it("is a no-op (returns false, no RPC) when idle", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    const spy = vi.spyOn(bridge, "request");

    const rolled = await rollbackReservedPlacement(bridge);

    expect(rolled).toBe(false);
    expect(spy).not.toHaveBeenCalled();
    expect(getReservationPhase()).toBe("idle");
  });

  it("still frees the seat when the delete RPC rejects (swallowed)", async () => {
    // A failed placeholder cleanup must not strand the seat "reserved" — the
    // next reconcile reclaims a lingering pending- orphan harmlessly. Reserve on
    // a healthy bridge first, then make JUST the delete reject.
    const bridge = createMockBridge({ delayMs: 0 });
    await reserveAtCursor(bridge, CURSOR);
    const original = bridge.request.bind(bridge);
    vi.spyOn(bridge, "request").mockImplementation((method, params) => {
      if (method === "timeline.deleteProvisionalByJob") {
        return Promise.reject(new Error("boom")) as ReturnType<typeof original>;
      }
      return original(method, params);
    });

    const rolled = await rollbackReservedPlacement(bridge);

    expect(rolled).toBe(true);
    expect(getReservationPhase()).toBe("idle");
  });
});

describe("reconcileFromTimeline (§2 reload re-sync)", () => {
  it("restores `reserved` from a surviving pending- placeholder", async () => {
    const bridge = createMockBridge({
      delayMs: 0,
      provisionalOrphans: [{ jobId: "pending-abc", layer: 4, frame: 200 }],
    });

    const phase = await reconcileFromTimeline(bridge);

    expect(phase).toBe("reserved");
    const state = getReservationState();
    expect(state.pendingId).toBe("pending-abc");
    expect(state.jobId).toBeNull();
    expect(state.placedLayer).toBe(4);
    expect(state.placedFrame).toBe(200);
  });

  it("restores `generating` from a surviving job-bound placeholder", async () => {
    const bridge = createMockBridge({
      delayMs: 0,
      provisionalOrphans: [{ jobId: "job-7", layer: 2, frame: 50 }],
    });

    const phase = await reconcileFromTimeline(bridge);

    expect(phase).toBe("generating");
    const state = getReservationState();
    expect(state.jobId).toBe("job-7");
    expect(state.pendingId).toBeNull();
    // A later poll seeing that job settled frees the seat (harmless convergence).
    expect(releaseIfSettled("job-7")).toBe(true);
    expect(getReservationPhase()).toBe("idle");
  });

  it("prefers a job-bound marker over a pending- one when both survive", async () => {
    const bridge = createMockBridge({
      delayMs: 0,
      provisionalOrphans: [
        { jobId: "pending-xyz", layer: 1, frame: 10 },
        { jobId: "job-9", layer: 2, frame: 20 },
      ],
    });

    const phase = await reconcileFromTimeline(bridge);

    expect(phase).toBe("generating");
    expect(getReservationState().jobId).toBe("job-9");
  });

  it("resolves to `idle` when no placeholders survive", async () => {
    const bridge = createMockBridge({ delayMs: 0 }); // no orphans

    const phase = await reconcileFromTimeline(bridge);

    expect(phase).toBe("idle");
    expect(getReservationPhase()).toBe("idle");
  });

  it("re-running the reconcile overwrites the seat from the current timeline (reserved -> idle -> generating)", async () => {
    // §6-b: the on-demand / projectLoaded re-sync calls this repeatedly, so a
    // later run must fully OVERWRITE the seat from whatever the timeline now
    // holds — not merge with, or defer to, the earlier state.
    const opts: MockBridgeOptions = {
      delayMs: 0,
      provisionalOrphans: [{ jobId: "pending-abc", layer: 4, frame: 200 }],
    };
    const bridge = createMockBridge(opts);

    // First run sees the surviving pending- placeholder -> reserved.
    expect(await reconcileFromTimeline(bridge)).toBe("reserved");
    expect(getReservationState().pendingId).toBe("pending-abc");

    // The placeholder is gone on the next load; a re-run overwrites to idle.
    opts.provisionalOrphans = [];
    expect(await reconcileFromTimeline(bridge)).toBe("idle");
    expect(getReservationPhase()).toBe("idle");
    expect(getReservationState().pendingId).toBeNull();

    // A further re-run can flip it the other way (idle -> generating).
    opts.provisionalOrphans = [{ jobId: "job-live", layer: 1, frame: 10 }];
    expect(await reconcileFromTimeline(bridge)).toBe("generating");
    expect(getReservationState().jobId).toBe("job-live");
  });

  describe("X2(b) settledJobIds (台帳認識)", () => {
    it("resolves to idle when the ONLY job-bound tag belongs to a settled job", async () => {
      // The permanent-block scenario: a 422 left a `NzVideomni#<jobId>` tag whose
      // job the ledger reports terminal. Excluding it means nothing holds the
      // seat generating — it converges to idle instead of resurrecting.
      const bridge = createMockBridge({
        delayMs: 0,
        provisionalOrphans: [{ jobId: "job-done", layer: 2, frame: 50 }],
      });

      const phase = await reconcileFromTimeline(bridge, new Set(["job-done"]));

      expect(phase).toBe("idle");
      expect(getReservationPhase()).toBe("idle");
    });

    it("restores reserved from a pending- tag when the job-bound tag is settled (not masked)", async () => {
      // A settled job-bound tag must NOT mask a genuine pending- reservation:
      // the reserved seat wins once the terminal tag is excluded.
      const bridge = createMockBridge({
        delayMs: 0,
        provisionalOrphans: [
          { jobId: "job-done", layer: 2, frame: 20 },
          { jobId: "pending-xyz", layer: 1, frame: 10 },
        ],
      });

      const phase = await reconcileFromTimeline(bridge, new Set(["job-done"]));

      expect(phase).toBe("reserved");
      expect(getReservationState().pendingId).toBe("pending-xyz");
    });

    it("still restores generating for a NON-terminal job-bound tag (running job protected)", async () => {
      // A job-bound tag whose job is still running is NOT in the settled set, so
      // it holds the seat generating exactly as before.
      const bridge = createMockBridge({
        delayMs: 0,
        provisionalOrphans: [{ jobId: "job-running", layer: 3, frame: 30 }],
      });

      const phase = await reconcileFromTimeline(bridge, new Set(["some-other-done-job"]));

      expect(phase).toBe("generating");
      expect(getReservationState().jobId).toBe("job-running");
    });

    it("omitting settledJobIds keeps the original behavior (job-bound wins)", async () => {
      // Existing call sites pass no set — a job-bound tag is adopted as before.
      const bridge = createMockBridge({
        delayMs: 0,
        provisionalOrphans: [{ jobId: "job-legacy", layer: 2, frame: 50 }],
      });

      const phase = await reconcileFromTimeline(bridge);

      expect(phase).toBe("generating");
      expect(getReservationState().jobId).toBe("job-legacy");
    });
  });

  it("resolves to `idle` (never throws) when the scan RPC fails", async () => {
    const failing: NativeBridge = {
      request: () => Promise.reject(new Error("boom")),
      requestWithFiles: () => Promise.reject(new Error("boom")),
      on: () => () => {},
    };

    const phase = await reconcileFromTimeline(failing);

    expect(phase).toBe("idle");
    expect(getReservationPhase()).toBe("idle");
  });
});

describe("bindToJob", () => {
  it("binds a waiting reservation to the real job, keeping the reserved position", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    const reservation = await reserveAtCursor(bridge, CURSOR);
    const spy = vi.spyOn(bridge, "request");

    const bound = await bindToJob(bridge, {
      jobId: "job-42",
      numFrames: 49,
      genFps: 30,
      displayText: "⏳ hello",
    });

    expect(bound).toBe(true);
    expect(spy).toHaveBeenCalledTimes(1);
    const [method, params] = spy.mock.calls[0]!;
    expect(method).toBe("timeline.updateProvisionalReservation");
    expect(params).toMatchObject({
      oldJobId: reservation.pendingId,
      newJobId: "job-42",
      placement: "C",
      // Same position it was reserved at — only the id/length are confirmed.
      cursorLayer: 3,
      cursorFrame: 100,
    });

    expect(getReservationPhase()).toBe("generating");
    const state = getReservationState();
    expect(state.jobId).toBe("job-42");
    expect(state.pendingId).toBe(reservation.pendingId);
  });

  it("re-stamps the reservation with the Generate-time DURATION when it differs from the reserved length", async () => {
    // Owner requirement 2026-07-21 #2: a provisional placed at the seed DURATION
    // (A) must follow the user's post-placement DURATION edit (B) once they press
    // Generate — `bindToJob` forwards the LIVE form's numFrames to
    // updateProvisionalReservation, so native re-derives the ribbon length. This
    // pins that the confirmed length wins over the reserved one.
    const bridge = createMockBridge({ delayMs: 0 });
    await reserveAtCursor(bridge, { ...CURSOR, numFrames: 49 }); // reserved length A = 49
    const spy = vi.spyOn(bridge, "request");

    const bound = await bindToJob(bridge, {
      jobId: "job-restamp",
      numFrames: 233, // confirmed length B ≠ A
      genFps: 24,
      displayText: "head",
    });

    expect(bound).toBe(true);
    const [method, params] = spy.mock.calls[0]!;
    expect(method).toBe("timeline.updateProvisionalReservation");
    // The re-stamp carries the Generate-time B (233), not the reserved A (49).
    expect(params).toMatchObject({ newJobId: "job-restamp", numFrames: 233, genFps: 24 });
  });

  it("is a no-op (no bridge call, returns false) when no reservation is waiting", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    const spy = vi.spyOn(bridge, "request");

    const bound = await bindToJob(bridge, { jobId: "job-1", numFrames: 49, genFps: 30, displayText: "x" });

    expect(bound).toBe(false);
    expect(spy).not.toHaveBeenCalled();
    expect(getReservationPhase()).toBe("idle");
  });

  it("does not re-bind once already generating (guards a non-✨ Generate)", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    await reserveAtCursor(bridge, CURSOR);
    await bindToJob(bridge, { jobId: "job-A", numFrames: 49, genFps: 30, displayText: "a" });

    const spy = vi.spyOn(bridge, "request");
    const bound = await bindToJob(bridge, { jobId: "job-B", numFrames: 49, genFps: 30, displayText: "b" });

    expect(bound).toBe(false);
    expect(spy).not.toHaveBeenCalled();
    expect(getReservationState().jobId).toBe("job-A");
  });
});

describe("stage-1 textPrefix (§5-5 段階1)", () => {
  it("forwards textPrefix on a fresh insert", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    const spy = vi.spyOn(bridge, "request");

    await reservePlacement(bridge, {
      placement: "C",
      cursor: { layer: 3, frame: 100 },
      numFrames: 49,
      genFps: 30,
      displayText: "(この位置に生成されます)",
      textPrefix: "予約:",
    });

    const [method, params] = spy.mock.calls[0]!;
    expect(method).toBe("timeline.insertProvisional");
    expect(params).toMatchObject({ textPrefix: "予約:", displayText: "(この位置に生成されます)" });
  });

  it("forwards textPrefix on a reservation MOVE too", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    await reserveAtCursor(bridge, { ...CURSOR, textPrefix: "予約:" });
    const spy = vi.spyOn(bridge, "request");

    await reserveAtCursor(bridge, { ...CURSOR, cursorLayer: 5, cursorFrame: 250, textPrefix: "予約:" });

    const [method, params] = spy.mock.calls[0]!;
    expect(method).toBe("timeline.updateProvisionalReservation");
    expect(params).toMatchObject({ textPrefix: "予約:" });
  });

  it("omits textPrefix entirely when the caller passes none (native default fallback)", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    const spy = vi.spyOn(bridge, "request");

    // With no textPrefix supplied, bindToJob leaves native on its default prefix.
    await reserveAtCursor(bridge, CURSOR);
    spy.mockClear();
    await bindToJob(bridge, { jobId: "job-2", numFrames: 49, genFps: 30, displayText: "a prompt head" });

    const [, params] = spy.mock.calls[0]!;
    expect(params).not.toHaveProperty("textPrefix");
    expect(params).toMatchObject({ displayText: "a prompt head" });
  });

  it("forwards the stage-2 textPrefix through bindToJob (§5-5 段階2)", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    await reserveAtCursor(bridge, CURSOR);
    const spy = vi.spyOn(bridge, "request");

    // Production (Create/Chain) passes "Generating: " so native's ⏳生成中 default
    // is never used and the burned text reads "Generating: 〔head〕…".
    await bindToJob(bridge, {
      jobId: "job-2",
      numFrames: 49,
      genFps: 30,
      displayText: "a prompt head",
      textPrefix: "Generating: ",
    });

    const [method, params] = spy.mock.calls[0]!;
    expect(method).toBe("timeline.updateProvisionalReservation");
    expect(params).toMatchObject({ textPrefix: "Generating: ", displayText: "a prompt head" });
  });
});

describe("settleProvisionalText (§5-5 段階3/4)", () => {
  it("rewrites the tracked placeholder to the terminal text (with the job marker) and frees the seat", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    await reserveAtCursor(bridge, CURSOR);
    await bindToJob(bridge, { jobId: "job-done", numFrames: 49, genFps: 30, displayText: "head" });
    const spy = vi.spyOn(bridge, "request");

    const acted = await settleProvisionalText(bridge, "job-done", "Done");

    expect(acted).toBe(true);
    expect(spy).toHaveBeenCalledTimes(1);
    const [method, params] = spy.mock.calls[0]!;
    expect(method).toBe("timeline.updateProvisionalText");
    // Position is the one captured at bind time (the cursor it was reserved at);
    // the [#job] marker is re-appended so the placeholder stays re-discoverable.
    expect(params).toMatchObject({
      jobId: "job-done",
      layer: 3,
      frame: 100,
      text: "Done [#job-done]",
    });
    // The seat is freed even though the text update is a separate async call.
    expect(getReservationPhase()).toBe("idle");
  });

  it("is a no-op (no RPC, returns false) for an unrelated settling job", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    await reserveAtCursor(bridge, CURSOR);
    await bindToJob(bridge, { jobId: "job-X", numFrames: 49, genFps: 30, displayText: "head" });
    const spy = vi.spyOn(bridge, "request");

    const acted = await settleProvisionalText(bridge, "some-other-job", "Failed: boom");

    expect(acted).toBe(false);
    expect(spy).not.toHaveBeenCalled();
    // The tracked seat is untouched.
    expect(getReservationPhase()).toBe("generating");
    expect(getReservationState().jobId).toBe("job-X");
  });

  it("is a no-op while merely reserved (no job bound yet)", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    const reservation = await reserveAtCursor(bridge, CURSOR);
    const spy = vi.spyOn(bridge, "request");

    // The pending id is not a job id, so no settling job can match it.
    const acted = await settleProvisionalText(bridge, reservation.pendingId, "Done");

    expect(acted).toBe(false);
    expect(spy).not.toHaveBeenCalled();
    expect(getReservationPhase()).toBe("reserved");
  });

  it("still frees the seat when the text-update RPC rejects (swallowed)", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    await reserveAtCursor(bridge, CURSOR);
    await bindToJob(bridge, { jobId: "job-fail", numFrames: 49, genFps: 30, displayText: "head" });

    // Make JUST the terminal text update reject; the seat must still free.
    const original = bridge.request.bind(bridge);
    vi.spyOn(bridge, "request").mockImplementation((method, params) => {
      if (method === "timeline.updateProvisionalText") {
        return Promise.reject(new Error("boom")) as ReturnType<typeof original>;
      }
      return original(method, params);
    });

    const acted = await settleProvisionalText(bridge, "job-fail", "Done");

    expect(acted).toBe(true);
    expect(getReservationPhase()).toBe("idle");
  });
});

describe("releaseIfSettled", () => {
  it("frees the seat only for the tracked job id", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    await reserveAtCursor(bridge, CURSOR);
    await bindToJob(bridge, { jobId: "job-99", numFrames: 49, genFps: 30, displayText: "x" });
    expect(getReservationPhase()).toBe("generating");

    // A settling *unrelated* job must not release the seat.
    expect(releaseIfSettled("some-other-job")).toBe(false);
    expect(getReservationPhase()).toBe("generating");

    // The tracked job settling frees the seat back to idle.
    expect(releaseIfSettled("job-99")).toBe(true);
    expect(getReservationPhase()).toBe("idle");
    expect(getReservationState().jobId).toBeNull();
    expect(getReservationState().pendingId).toBeNull();
  });

  it("is a no-op while merely `reserved` (nothing bound to a job yet)", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    const reservation = await reserveAtCursor(bridge, CURSOR);

    // The pending id is not a job id — a settled job can never match it.
    expect(releaseIfSettled(reservation.pendingId)).toBe(false);
    expect(getReservationPhase()).toBe("reserved");
  });
});

describe("publishFormValues / getPublishedFormValues (D3)", () => {
  it("returns null before Create publishes anything (fresh module state)", () => {
    expect(getPublishedFormValues()).toBeNull();
  });

  it("round-trips the last-published length/fps + width/height (W8)", () => {
    publishFormValues({ numFrames: 121, frameRate: 30, width: 512, height: 320 });
    expect(getPublishedFormValues()).toEqual({ numFrames: 121, frameRate: 30, width: 512, height: 320 });

    // A later publish overwrites (the live form's newest values win).
    publishFormValues({ numFrames: 49, frameRate: 24, width: 1280, height: 768 });
    expect(getPublishedFormValues()).toEqual({ numFrames: 49, frameRate: 24, width: 1280, height: 768 });
  });

  it("is cleared back to null by resetProvisionalReservation", () => {
    publishFormValues({ numFrames: 89, frameRate: 25, width: 960, height: 576 });
    expect(getPublishedFormValues()).not.toBeNull();
    resetProvisionalReservation();
    expect(getPublishedFormValues()).toBeNull();
  });
});

describe("phase transitions", () => {
  it("idle -> reserved -> generating -> idle", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    expect(getReservationPhase()).toBe("idle");

    await reserveAtCursor(bridge, CURSOR);
    expect(getReservationPhase()).toBe("reserved");

    await bindToJob(bridge, { jobId: "job-1", numFrames: 49, genFps: 30, displayText: "x" });
    expect(getReservationPhase()).toBe("generating");

    releaseIfSettled("job-1");
    expect(getReservationPhase()).toBe("idle");
  });
});

// --- Y3: W2 shared "inserted" store -----------------------------------------

describe("Y3 inserted-jobs shared store (W2)", () => {
  it("markJobInserted round-trips isJobInserted + getInsertedFilePath", () => {
    expect(isJobInserted("job-1")).toBe(false);
    expect(getInsertedFilePath("job-1")).toBeNull();

    markJobInserted("job-1", "C:/out.mp4");
    expect(isJobInserted("job-1")).toBe(true);
    expect(getInsertedFilePath("job-1")).toBe("C:/out.mp4");
  });

  it("markJobInserted defaults filePath to null when omitted", () => {
    markJobInserted("job-2");
    expect(isJobInserted("job-2")).toBe(true);
    expect(getInsertedFilePath("job-2")).toBeNull();
  });

  it("notifies subscribers on mark, and stops after unsubscribe", () => {
    const listener = vi.fn();
    const unsubscribe = subscribeInserted(listener);

    markJobInserted("job-1", "C:/a.mp4");
    expect(listener).toHaveBeenCalledTimes(1);

    unsubscribe();
    markJobInserted("job-3", "C:/b.mp4");
    expect(listener).toHaveBeenCalledTimes(1);
  });

  it("clearJobInserted removes the flag and notifies only when something was removed", () => {
    const listener = vi.fn();
    subscribeInserted(listener);

    markJobInserted("job-1", "C:/a.mp4");
    expect(listener).toHaveBeenCalledTimes(1);

    clearJobInserted("job-1");
    expect(isJobInserted("job-1")).toBe(false);
    expect(listener).toHaveBeenCalledTimes(2);

    // Clearing an absent job is a no-op (no extra notify).
    clearJobInserted("job-1");
    expect(listener).toHaveBeenCalledTimes(2);
  });

  it("resetProvisionalReservation clears the store and notifies subscribers", () => {
    const listener = vi.fn();
    subscribeInserted(listener);

    markJobInserted("job-1", "C:/a.mp4");
    listener.mockClear();

    resetProvisionalReservation();
    expect(isJobInserted("job-1")).toBe(false);
    expect(getInsertedFilePath("job-1")).toBeNull();
    expect(listener).toHaveBeenCalled();
  });
});
