import { describe, expect, it } from "vitest";
import {
  buildConditioningImages,
  clampStrength,
  fileNameFromPath,
  isFrameIdxOutOfRange,
  snapFrameIdx,
  sortByFrameIdx,
  type KeyframeItem,
} from "./keyframeUtils";

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

describe("fileNameFromPath", () => {
  it("returns the last component after a backslash", () => {
    expect(fileNameFromPath("C:\\Users\\mock\\Pictures\\frame_1.png")).toBe("frame_1.png");
  });

  it("returns the last component after a forward slash", () => {
    expect(fileNameFromPath("/home/mock/frame_1.png")).toBe("frame_1.png");
  });

  it("returns the input unchanged when there is no separator", () => {
    expect(fileNameFromPath("frame_1.png")).toBe("frame_1.png");
  });
});

describe("snapFrameIdx", () => {
  // Docs/API_REFERENCE.md §5.1: frame_idx==0 stays 0; >0 snaps to
  // (f-1)//multiple*multiple + gridOffset.
  it("keeps 0 as the leading frame", () => {
    expect(snapFrameIdx(0, 8, 1)).toBe(0);
  });

  it("treats negative input as 0", () => {
    expect(snapFrameIdx(-5, 8, 1)).toBe(0);
  });

  it("snaps values already on the grid to themselves", () => {
    expect(snapFrameIdx(1, 8, 1)).toBe(1);
    expect(snapFrameIdx(9, 8, 1)).toBe(9);
    expect(snapFrameIdx(17, 8, 1)).toBe(17);
  });

  it("floors values between grid points down to the previous grid point", () => {
    expect(snapFrameIdx(5, 8, 1)).toBe(1);
    expect(snapFrameIdx(8, 8, 1)).toBe(1);
    expect(snapFrameIdx(16, 8, 1)).toBe(9);
    expect(snapFrameIdx(24, 8, 1)).toBe(17);
  });

  it("works with a different multiple/offset", () => {
    // Grid points for multiple=4, offset=2 are 2, 6, 10, 14, ...
    expect(snapFrameIdx(10, 4, 2)).toBe(10);
    expect(snapFrameIdx(11, 4, 2)).toBe(10);
    expect(snapFrameIdx(12, 4, 2)).toBe(10);
    expect(snapFrameIdx(13, 4, 2)).toBe(14);
    expect(snapFrameIdx(14, 4, 2)).toBe(14);
  });
});

describe("isFrameIdxOutOfRange", () => {
  it("is false for values within [0, numFrames-1]", () => {
    expect(isFrameIdxOutOfRange(0, 49)).toBe(false);
    expect(isFrameIdxOutOfRange(48, 49)).toBe(false);
  });

  it("is true for values at or beyond numFrames", () => {
    expect(isFrameIdxOutOfRange(49, 49)).toBe(true);
    expect(isFrameIdxOutOfRange(100, 49)).toBe(true);
  });

  it("is true for negative values", () => {
    expect(isFrameIdxOutOfRange(-1, 49)).toBe(true);
  });
});

describe("clampStrength", () => {
  it("clamps into [0, 1]", () => {
    expect(clampStrength(-0.5)).toBe(0);
    expect(clampStrength(1.5)).toBe(1);
    expect(clampStrength(0.42)).toBe(0.42);
  });
});

describe("sortByFrameIdx", () => {
  it("returns a new array sorted by ascending frame_idx without mutating the input", () => {
    const items = [makeItem({ id: "a", frameIdx: 17 }), makeItem({ id: "b", frameIdx: 0 }), makeItem({ id: "c", frameIdx: 9 })];
    const sorted = sortByFrameIdx(items);
    expect(sorted.map((i) => i.id)).toEqual(["b", "c", "a"]);
    expect(items.map((i) => i.id)).toEqual(["a", "b", "c"]); // original untouched
  });
});

describe("buildConditioningImages", () => {
  it("returns [] when there are no items", () => {
    expect(buildConditioningImages([])).toEqual([]);
  });

  it("includes only ready items with a resolved imageId", () => {
    const items = [
      makeItem({ id: "ready-1", imageId: "img-1", frameIdx: 0, strength: 0.8, status: "ready" }),
      makeItem({ id: "uploading-1", imageId: null, status: "uploading" }),
      makeItem({ id: "error-1", imageId: null, status: "error", errorCode: "FILE_NOT_FOUND" }),
      makeItem({ id: "ready-2", imageId: "img-2", frameIdx: 17, strength: 0.5, status: "ready" }),
    ];

    expect(buildConditioningImages(items)).toEqual([
      { image_id: "img-1", frame_idx: 0, strength: 0.8 },
      { image_id: "img-2", frame_idx: 17, strength: 0.5 },
    ]);
  });

  it("omits crf and any client-only fields", () => {
    const payload = buildConditioningImages([makeItem()])[0];
    if (!payload) throw new Error("expected one conditioning image");
    expect(Object.keys(payload).sort()).toEqual(["frame_idx", "image_id", "strength"]);
  });
});
