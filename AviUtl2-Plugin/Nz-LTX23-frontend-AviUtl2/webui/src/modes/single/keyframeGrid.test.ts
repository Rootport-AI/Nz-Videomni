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

// The real config: server default conditioning_frame_idx_multiple/grid_offset
// (Docs/API_REFERENCE.md §5.1).
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
