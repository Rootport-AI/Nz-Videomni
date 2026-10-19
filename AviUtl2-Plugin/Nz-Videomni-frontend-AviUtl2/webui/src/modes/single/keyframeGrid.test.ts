import { describe, expect, it } from "vitest";
import type { KeyframeItem } from "./keyframeUtils";
import {
  canAddKeyframe,
  findOverflowFrameIdxs,
  maxPlaceablePosition,
  nearestGridPosition,
  nextAddPosition,
  placeableSlotCount,
  positionToRatio,
  ratioToFrame,
  resolveLanding,
  stepPin,
  validPositions,
} from "./keyframeGrid";

// The LTX keyframe grid (`constraints.keyframes`, Docs/API_REFERENCE.md §5.1).
const MULTIPLE = 8;
const OFFSET = 1;

function makeItem(overrides: Partial<KeyframeItem> = {}): KeyframeItem {
  return {
    id: "kf-1",
    filePath: "C:\\Users\\mock\\frame_1.png",
    fileName: "frame_1.png",
    imageId: "img-1",
    thumbnailDataUrl: null,
    frameIdx: 0,
    strength: 0.8,
    status: "ready",
    errorCode: null,
    ...overrides,
  };
}

describe("maxPlaceablePosition", () => {
  it("floors an off-grid numFrames down before computing the ceiling", () => {
    expect(maxPlaceablePosition(257, MULTIPLE, OFFSET)).toBe(249);
    expect(maxPlaceablePosition(250, MULTIPLE, OFFSET)).toBe(241);
  });

  it("can go negative when numFrames is too small for a non-zero slot", () => {
    expect(maxPlaceablePosition(9, MULTIPLE, OFFSET)).toBe(1);
    expect(maxPlaceablePosition(1, MULTIPLE, OFFSET)).toBeLessThan(OFFSET);
  });
});

describe("validPositions", () => {
  it("returns [0, 1] for a numFrames that only fits one grid step", () => {
    expect(validPositions(9, MULTIPLE, OFFSET)).toEqual([0, 1]);
  });

  it("returns the full grid for a canonical 8n+1 numFrames", () => {
    const positions = validPositions(257, MULTIPLE, OFFSET);
    expect(positions.length).toBe(33);
    expect(positions[0]).toBe(0);
    expect(positions[1]).toBe(1);
    expect(positions[positions.length - 1]).toBe(249);
  });

  it("floors an off-grid numFrames to the grid before listing positions", () => {
    const positions = validPositions(250, MULTIPLE, OFFSET);
    expect(positions[positions.length - 1]).toBe(241);
  });

  it("returns just [0] when numFrames is too small for any non-zero slot", () => {
    expect(validPositions(1, MULTIPLE, OFFSET)).toEqual([0]);
    expect(validPositions(0, MULTIPLE, OFFSET)).toEqual([0]);
  });
});

describe("nearestGridPosition", () => {
  it("returns 0 for frames at or near the left edge", () => {
    expect(nearestGridPosition(0, 257, MULTIPLE, OFFSET)).toBe(0);
    expect(nearestGridPosition(0.4, 257, MULTIPLE, OFFSET)).toBe(0);
  });

  it("breaks an equidistant 0/offset tie toward 0", () => {
    // 0.5 is exactly 0.5 away from both 0 and 1 (offset).
    expect(nearestGridPosition(0.5, 257, MULTIPLE, OFFSET)).toBe(0);
  });

  it("snaps to the nearest grid point", () => {
    expect(nearestGridPosition(12, 257, MULTIPLE, OFFSET)).toBe(9);
  });

  it("breaks equidistant ties toward the smaller position", () => {
    // 13 is exactly 4 away from both 9 and 17.
    expect(nearestGridPosition(13, 257, MULTIPLE, OFFSET)).toBe(9);
  });

  it("clamps to the nearest end of the range for out-of-range frames", () => {
    // -50 is closer to 0 than to 1 now that 0 is a normal candidate.
    expect(nearestGridPosition(-50, 257, MULTIPLE, OFFSET)).toBe(0);
    expect(nearestGridPosition(1000, 257, MULTIPLE, OFFSET)).toBe(249);
  });
});

describe("resolveLanding", () => {
  const base = { numFrames: 257, multiple: MULTIPLE, offset: OFFSET };

  it("lands directly on desired when it's free", () => {
    expect(resolveLanding({ ...base, desired: 17, occupied: new Set(), origin: 1 })).toBe(17);
  });

  it("jumps over a single occupied position forward", () => {
    expect(resolveLanding({ ...base, desired: 17, occupied: new Set([17]), origin: 1 })).toBe(25);
  });

  it("jumps over a two-position occupied run forward", () => {
    expect(resolveLanding({ ...base, desired: 9, occupied: new Set([9, 17]), origin: 1 })).toBe(25);
  });

  it("jumps over a three-position occupied run forward", () => {
    expect(resolveLanding({ ...base, desired: 9, occupied: new Set([9, 17, 25]), origin: 1 })).toBe(33);
  });

  it("falls back to scanning backward when nothing is free forward", () => {
    // 241 and 249 (the last grid position) are both taken, so the forward
    // scan from 241 has nowhere to go; it should walk back to 233 instead.
    expect(resolveLanding({ ...base, desired: 241, occupied: new Set([241, 249]), origin: 1 })).toBe(233);
  });

  it("falls back to origin when the entire grid is occupied", () => {
    const occupied = new Set(validPositions(257, MULTIPLE, OFFSET));
    expect(resolveLanding({ ...base, desired: 17, occupied, origin: 99 })).toBe(99);
  });

  it("treats position 0 as an ordinary grid slot during collision resolution", () => {
    expect(resolveLanding({ ...base, desired: 0, occupied: new Set([0]), origin: 1 })).toBe(1);
  });
});

describe("stepPin", () => {
  const args = (numFrames = 257) => [numFrames, MULTIPLE, OFFSET] as const;

  it("moves to the adjacent grid position in either direction", () => {
    expect(stepPin(9, 1, new Set(), ...args())).toBe(17);
    expect(stepPin(17, -1, new Set(), ...args())).toBe(9);
  });

  it("jumps over an occupied run in the step direction", () => {
    expect(stepPin(9, 1, new Set([17, 25]), ...args())).toBe(33);
  });

  it("does not wrap and stays put at the end of the grid", () => {
    expect(stepPin(249, 1, new Set(), ...args())).toBe(249);
    expect(stepPin(0, -1, new Set(), ...args())).toBe(0);
  });

  it("steps from position 1 down to 0 when 0 is free", () => {
    expect(stepPin(1, -1, new Set(), ...args())).toBe(0);
  });
});

describe("nextAddPosition", () => {
  it("returns 0 when there are no pins yet", () => {
    expect(nextAddPosition(new Set(), 257, MULTIPLE, OFFSET)).toBe(0);
  });

  it("places the new pin at the rounded midpoint to the end", () => {
    expect(nextAddPosition(new Set([0]), 257, MULTIPLE, OFFSET)).toBe(129);
  });

  it("skips forward to the first free slot when the midpoint doesn't clear lastPos", () => {
    expect(nextAddPosition(new Set([0, 241]), 257, MULTIPLE, OFFSET)).toBe(249);
  });

  it("returns null when there's no free slot after the last pin", () => {
    expect(nextAddPosition(new Set([0, 249]), 257, MULTIPLE, OFFSET)).toBeNull();
  });

  it("never inserts before the last pin, returning null instead", () => {
    expect(nextAddPosition(new Set([0, 1]), 9, MULTIPLE, OFFSET)).toBeNull();
  });
});

describe("canAddKeyframe", () => {
  it("mirrors nextAddPosition's null-ness as a boolean", () => {
    expect(canAddKeyframe(new Set(), 257, MULTIPLE, OFFSET)).toBe(true);
    expect(canAddKeyframe(new Set([0, 249]), 257, MULTIPLE, OFFSET)).toBe(false);
  });
});

describe("findOverflowFrameIdxs", () => {
  it("uses maxPlaceablePosition, not numFrames-1, as the cutoff", () => {
    // For numFrames=257, numFrames-1 is 256 but maxPlaceable is 249: a pin
    // at 250-256 is within [0, numFrames-1] yet off the placeable grid.
    const items = [makeItem({ id: "a", frameIdx: 249 }), makeItem({ id: "b", frameIdx: 250 }), makeItem({ id: "c", frameIdx: 256 })];
    expect(findOverflowFrameIdxs(items, 257, MULTIPLE, OFFSET).map((i) => i.id)).toEqual(["b", "c"]);
  });

  it("keeps the boundary frameIdx (== maxPlaceable) as non-overflow", () => {
    const items = [makeItem({ frameIdx: 249 })];
    expect(findOverflowFrameIdxs(items, 257, MULTIPLE, OFFSET)).toEqual([]);
  });

  it("never flags frameIdx 0, even when maxPlaceable is negative", () => {
    const items = [makeItem({ frameIdx: 0 })];
    expect(findOverflowFrameIdxs(items, 1, MULTIPLE, OFFSET)).toEqual([]);
  });
});

describe("positionToRatio / ratioToFrame", () => {
  it("round-trips frame -> ratio -> frame", () => {
    expect(positionToRatio(0, 257)).toBe(0);
    expect(positionToRatio(256, 257)).toBe(1);
    expect(positionToRatio(128, 257)).toBeCloseTo(0.5, 5);
    expect(ratioToFrame(0.5, 257)).toBeCloseTo(128, 5);
  });

  it("clamps out-of-range ratios and frames", () => {
    expect(positionToRatio(-10, 257)).toBe(0);
    expect(positionToRatio(1000, 257)).toBe(1);
    expect(ratioToFrame(-1, 257)).toBe(0);
    expect(ratioToFrame(2, 257)).toBe(256);
  });

  it("guards against numFrames <= 1", () => {
    expect(positionToRatio(0, 1)).toBe(0);
    expect(positionToRatio(0, 0)).toBe(0);
  });
});

describe("placeableSlotCount", () => {
  it("counts every valid position including 0", () => {
    expect(placeableSlotCount(9, MULTIPLE, OFFSET)).toBe(2);
    expect(placeableSlotCount(257, MULTIPLE, OFFSET)).toBe(33);
  });
});

// 台帳 §1-97: MiniMax H3's keyframe grid is step 1 / offset 0 — any frame in
// [0, F-1]. The 8/1 expectations above are unchanged.
describe("MiniMax H3's 1/0 keyframe grid (any frame)", () => {
  const STEP = 1;
  const ORIGIN = 0;
  const F = 22; // H3's shortest length (17n+5)

  it("lists every frame 0..F-1 once (0 is not doubled)", () => {
    const positions = validPositions(F, STEP, ORIGIN);
    expect(positions).toEqual(Array.from({ length: F }, (_, i) => i));
    expect(new Set(positions).size).toBe(positions.length);
    expect(placeableSlotCount(F, STEP, ORIGIN)).toBe(F);
    expect(validPositions(1, STEP, ORIGIN)).toEqual([0]);
  });

  it("places up to the last frame", () => {
    expect(maxPlaceablePosition(F, STEP, ORIGIN)).toBe(F - 1);
    expect(maxPlaceablePosition(124, STEP, ORIGIN)).toBe(123);
  });

  it("nextAddPosition starts at 0, then lands at the midpoint, and runs out at the last frame", () => {
    expect(nextAddPosition(new Set(), F, STEP, ORIGIN)).toBe(0);
    // midpoint of [0, 21] = 10.5, tie toward the smaller position
    expect(nextAddPosition(new Set([0]), F, STEP, ORIGIN)).toBe(10);
    expect(nextAddPosition(new Set([0, 10]), F, STEP, ORIGIN)).toBe(15);
    expect(nextAddPosition(new Set([0, F - 1]), F, STEP, ORIGIN)).toBeNull();
  });

  it("stepPin moves one frame and jumps an occupied run", () => {
    expect(stepPin(5, 1, new Set(), F, STEP, ORIGIN)).toBe(6);
    expect(stepPin(5, 1, new Set([6, 7]), F, STEP, ORIGIN)).toBe(8);
    expect(stepPin(1, -1, new Set(), F, STEP, ORIGIN)).toBe(0);
    expect(stepPin(0, -1, new Set(), F, STEP, ORIGIN)).toBe(0);
    expect(stepPin(F - 1, 1, new Set(), F, STEP, ORIGIN)).toBe(F - 1);
  });

  it("resolveLanding takes the frame itself, or the next free one either way", () => {
    const base = { numFrames: F, multiple: STEP, offset: ORIGIN, origin: 3 };
    expect(resolveLanding({ ...base, desired: 7, occupied: new Set() })).toBe(7);
    expect(resolveLanding({ ...base, desired: 7, occupied: new Set([7, 8]) })).toBe(9);
    expect(resolveLanding({ ...base, desired: F - 1, occupied: new Set([F - 1]) })).toBe(F - 2);
  });

  it("findOverflowFrameIdxs flags only frames past F-1", () => {
    const items = [
      makeItem({ id: "a", frameIdx: 0 }),
      makeItem({ id: "b", frameIdx: F - 1 }),
      makeItem({ id: "c", frameIdx: F }),
    ];
    expect(findOverflowFrameIdxs(items, F, STEP, ORIGIN).map((i) => i.id)).toEqual(["c"]);
    expect(findOverflowFrameIdxs(items, F - 1, STEP, ORIGIN).map((i) => i.id)).toEqual(["b", "c"]);
  });
});
