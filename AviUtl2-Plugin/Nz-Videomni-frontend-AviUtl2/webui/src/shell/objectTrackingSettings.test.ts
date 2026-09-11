import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  OBJECT_TRACKING_DEFAULTS,
  OBJECT_TRACKING_STORAGE_KEY,
  readStoredObjectTracking,
  resolveTrackingStatus,
  writeStoredObjectTracking,
} from "./objectTrackingSettings";

// §3-54 物体追尾 (2026-09-11). The one thing worth proving about this module is
// that NOTHING out of range can reach `timeline.trackObject`: native rejects a
// bad value with `BAD_REQUEST`, and the user would meet that refusal at
// right-click time, several steps away from the stored value that caused it.
// So every field gets a "junk in, default out" case, and the per-field
// fallback is proven to leave its neighbours alone.

function seed(value: unknown): void {
  window.localStorage.setItem(OBJECT_TRACKING_STORAGE_KEY, JSON.stringify(value));
}

describe("readStoredObjectTracking", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });
  afterEach(() => {
    window.localStorage.clear();
    vi.restoreAllMocks();
  });

  it("returns the defaults when nothing is stored", () => {
    expect(readStoredObjectTracking()).toEqual(OBJECT_TRACKING_DEFAULTS);
  });

  it("returns a COPY of the defaults, not the frozen object itself", () => {
    const read = readStoredObjectTracking();
    expect(read).not.toBe(OBJECT_TRACKING_DEFAULTS);
    // A mutation of the returned object must not reach the shared constant.
    read.searchFactor = 2;
    expect(OBJECT_TRACKING_DEFAULTS.searchFactor).toBe(4);
  });

  it("round-trips a fully-populated object", () => {
    const value = {
      searchFactor: 5.5,
      lostScoreThreshold: 0.6,
      lostBehavior: "continue" as const,
      smoothing: 0.75,
      followSize: false,
      keyframeStride: 4,
    };
    writeStoredObjectTracking(value);
    expect(readStoredObjectTracking()).toEqual(value);
  });

  it("falls back to the defaults on malformed JSON", () => {
    window.localStorage.setItem(OBJECT_TRACKING_STORAGE_KEY, "{not json");
    expect(readStoredObjectTracking()).toEqual(OBJECT_TRACKING_DEFAULTS);
  });

  it("falls back to the defaults on a non-object payload", () => {
    seed("sdpa");
    expect(readStoredObjectTracking()).toEqual(OBJECT_TRACKING_DEFAULTS);
    seed(42);
    expect(readStoredObjectTracking()).toEqual(OBJECT_TRACKING_DEFAULTS);
    seed(null);
    expect(readStoredObjectTracking()).toEqual(OBJECT_TRACKING_DEFAULTS);
    seed([1, 2, 3]);
    expect(readStoredObjectTracking()).toEqual(OBJECT_TRACKING_DEFAULTS);
  });

  it("falls back to the defaults when localStorage itself throws", () => {
    vi.spyOn(window.localStorage.__proto__ as Storage, "getItem").mockImplementation(() => {
      throw new Error("storage disabled");
    });
    expect(readStoredObjectTracking()).toEqual(OBJECT_TRACKING_DEFAULTS);
  });

  // --- per-field fallback --------------------------------------------------

  it("keeps the good fields when ONE is missing (a blob from an older build)", () => {
    seed({ searchFactor: 5, lostBehavior: "continue", followSize: false });
    expect(readStoredObjectTracking()).toEqual({
      searchFactor: 5,
      lostScoreThreshold: OBJECT_TRACKING_DEFAULTS.lostScoreThreshold,
      lostBehavior: "continue",
      smoothing: OBJECT_TRACKING_DEFAULTS.smoothing,
      followSize: false,
      keyframeStride: OBJECT_TRACKING_DEFAULTS.keyframeStride,
    });
  });

  it("keeps the good fields when ONE is the wrong type", () => {
    seed({
      searchFactor: "wide",
      lostScoreThreshold: 0.9,
      lostBehavior: 7,
      smoothing: null,
      followSize: "yes",
      keyframeStride: 3,
    });
    expect(readStoredObjectTracking()).toEqual({
      searchFactor: OBJECT_TRACKING_DEFAULTS.searchFactor,
      lostScoreThreshold: 0.9,
      lostBehavior: OBJECT_TRACKING_DEFAULTS.lostBehavior,
      smoothing: OBJECT_TRACKING_DEFAULTS.smoothing,
      followSize: OBJECT_TRACKING_DEFAULTS.followSize,
      keyframeStride: 3,
    });
  });

  // --- range checks --------------------------------------------------------

  it.each([
    ["searchFactor", 1.9],
    ["searchFactor", 6.1],
    ["lostScoreThreshold", -0.01],
    ["lostScoreThreshold", 1.01],
    ["smoothing", -1],
    ["smoothing", 1.5],
    ["keyframeStride", 0],
    ["keyframeStride", 11],
  ])("replaces an out-of-range %s (%s) with its default", (field, value) => {
    seed({ [field]: value });
    const read = readStoredObjectTracking() as unknown as Record<string, unknown>;
    expect(read[field]).toBe((OBJECT_TRACKING_DEFAULTS as unknown as Record<string, unknown>)[field]);
  });

  it.each([
    ["searchFactor", 2],
    ["searchFactor", 6],
    ["lostScoreThreshold", 0],
    ["lostScoreThreshold", 1],
    ["smoothing", 0],
    ["smoothing", 1],
    ["keyframeStride", 1],
    ["keyframeStride", 10],
  ])("accepts the boundary value %s = %s", (field, value) => {
    seed({ [field]: value });
    const read = readStoredObjectTracking() as unknown as Record<string, unknown>;
    expect(read[field]).toBe(value);
  });

  it("rejects a NaN / Infinity number", () => {
    // JSON has no NaN literal, so this is what a hand-written blob looks like.
    window.localStorage.setItem(
      OBJECT_TRACKING_STORAGE_KEY,
      '{"searchFactor": null, "smoothing": 1e999}',
    );
    const read = readStoredObjectTracking();
    expect(read.searchFactor).toBe(OBJECT_TRACKING_DEFAULTS.searchFactor);
    expect(read.smoothing).toBe(OBJECT_TRACKING_DEFAULTS.smoothing);
  });

  // The one field with a rule of its own: native rejects a fractional stride
  // outright rather than rounding it, so a stored `1.5` must not travel.
  it("replaces a FRACTIONAL keyframeStride with the default, even in range", () => {
    seed({ keyframeStride: 1.5 });
    expect(readStoredObjectTracking().keyframeStride).toBe(
      OBJECT_TRACKING_DEFAULTS.keyframeStride,
    );
  });

  it("accepts only the two lostBehavior literals", () => {
    seed({ lostBehavior: "hold" });
    expect(readStoredObjectTracking().lostBehavior).toBe("hold");
    seed({ lostBehavior: "continue" });
    expect(readStoredObjectTracking().lostBehavior).toBe("continue");
    seed({ lostBehavior: "freeze" });
    expect(readStoredObjectTracking().lostBehavior).toBe(OBJECT_TRACKING_DEFAULTS.lostBehavior);
  });
});

describe("writeStoredObjectTracking", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });
  afterEach(() => {
    window.localStorage.clear();
    vi.restoreAllMocks();
  });

  it("writes JSON under the section-named key", () => {
    writeStoredObjectTracking({ ...OBJECT_TRACKING_DEFAULTS, searchFactor: 3.2 });
    const raw = window.localStorage.getItem(OBJECT_TRACKING_STORAGE_KEY);
    expect(raw).not.toBeNull();
    expect(JSON.parse(raw ?? "{}")).toMatchObject({ searchFactor: 3.2 });
  });

  it("swallows a localStorage failure instead of throwing", () => {
    vi.spyOn(window.localStorage.__proto__ as Storage, "setItem").mockImplementation(() => {
      throw new Error("quota exceeded");
    });
    expect(() => writeStoredObjectTracking({ ...OBJECT_TRACKING_DEFAULTS })).not.toThrow();
  });
});

describe("resolveTrackingStatus", () => {
  it("keeps the last observed tracking block while no status body is in hand", () => {
    // Threaded exactly as `AppShell` threads it: every answer becomes the next
    // call's `lastSeen`, which is what its ref holds between renders. The
    // sequence below is one real session — the first poll has not landed yet,
    // then it does, then the user switches base model and the status parks at
    // `loading-models` with no body of its own (`status: null`).
    let last = resolveTrackingStatus(undefined, undefined);
    expect(last?.available ?? false).toBe(false);
    last = resolveTrackingStatus({ tracking: { available: true } }, last);
    last = resolveTrackingStatus(null, last);
    last = resolveTrackingStatus(null, last);
    // Still true minutes later: greying the panel out here would tell a user
    // whose tracking works to go and run `install-UETrack.bat`.
    expect(last?.available).toBe(true);
    // A body that HAS arrived always wins, including one carrying no block at
    // all (a server too old to report it) — that is an observation, not a gap.
    last = resolveTrackingStatus({ tracking: { available: false, reason: "worker failed" } }, last);
    expect(last).toEqual({ available: false, reason: "worker failed" });
    expect(resolveTrackingStatus({}, last)).toBeUndefined();
  });
});
