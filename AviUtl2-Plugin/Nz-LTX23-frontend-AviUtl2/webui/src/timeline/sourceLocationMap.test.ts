import { beforeEach, describe, expect, it } from "vitest";
import {
  getSourceLocation,
  recordSourceLocation,
  rekeySourceLocation,
  releaseSourceLocation,
  resetSourceLocationMap,
} from "./sourceLocationMap";
import type { SourceLocation } from "./sourceLocationMap";

function loc(overrides: Partial<SourceLocation> = {}): SourceLocation {
  return { layer: 2, frameStart: 100, frameEnd: 220, filePath: "C:\\src.mp4", rate: 30, scale: 1, ...overrides };
}

describe("sourceLocationMap", () => {
  beforeEach(() => {
    resetSourceLocationMap();
  });

  it("record -> rekey -> get -> release round-trips through the pending and job keys", () => {
    recordSourceLocation("pending-abc", loc());
    // Before rekey the entry lives under the pending id, not the job id.
    expect(getSourceLocation("pending-abc")).toEqual(loc());
    expect(getSourceLocation("job-1")).toBeNull();

    rekeySourceLocation("pending-abc", "job-1");
    // After rekey it moved to the job id and the pending id is gone.
    expect(getSourceLocation("pending-abc")).toBeNull();
    expect(getSourceLocation("job-1")).toEqual(loc());

    releaseSourceLocation("job-1");
    expect(getSourceLocation("job-1")).toBeNull();
  });

  it("getSourceLocation returns null for an unknown job id", () => {
    expect(getSourceLocation("never-recorded")).toBeNull();
  });

  it("rekeySourceLocation is a no-op when the pending id has no entry", () => {
    rekeySourceLocation("pending-missing", "job-2");
    expect(getSourceLocation("job-2")).toBeNull();
    expect(getSourceLocation("pending-missing")).toBeNull();
  });

  it("recording the same pending id twice overwrites the earlier location", () => {
    recordSourceLocation("pending-x", loc({ layer: 1, frameEnd: 200 }));
    recordSourceLocation("pending-x", loc({ layer: 5, frameEnd: 999 }));
    expect(getSourceLocation("pending-x")).toEqual(loc({ layer: 5, frameEnd: 999 }));
  });

  it("resetSourceLocationMap clears every entry (test isolation)", () => {
    recordSourceLocation("pending-a", loc());
    recordSourceLocation("job-b", loc());
    resetSourceLocationMap();
    expect(getSourceLocation("pending-a")).toBeNull();
    expect(getSourceLocation("job-b")).toBeNull();
  });

  it("releaseSourceLocation is a no-op for an absent key", () => {
    expect(() => releaseSourceLocation("absent")).not.toThrow();
  });
});
