import { afterEach, describe, expect, it, vi } from "vitest";
import {
  REPEAT_MAX,
  REPEAT_MIN,
  __resetRepeatCountForTests,
  clampRepeatCount,
  decideAfterSettle,
  getRepeatCountText,
  isTerminal,
  setRepeatCountText,
  shouldNotifyChange,
  submissionFingerprint,
  subscribeRepeatCountText,
} from "./repeatRun";

afterEach(() => {
  __resetRepeatCountForTests();
});

describe("clampRepeatCount", () => {
  it("falls back to 1 for empty / non-numeric text", () => {
    expect(clampRepeatCount("")).toBe(1);
    expect(clampRepeatCount("   ")).toBe(1);
    expect(clampRepeatCount("abc")).toBe(1);
    expect(clampRepeatCount(Number.NaN)).toBe(1);
  });

  it("clamps to the 1–99 range", () => {
    expect(REPEAT_MIN).toBe(1);
    expect(REPEAT_MAX).toBe(99);
    expect(clampRepeatCount("0")).toBe(1);
    expect(clampRepeatCount("-5")).toBe(1);
    expect(clampRepeatCount("100")).toBe(99);
    expect(clampRepeatCount(1000)).toBe(99);
  });

  it("truncates fractions and passes valid integers through", () => {
    expect(clampRepeatCount("3.9")).toBe(3);
    expect(clampRepeatCount(2.2)).toBe(2);
    expect(clampRepeatCount("1")).toBe(1);
    expect(clampRepeatCount("42")).toBe(42);
    expect(clampRepeatCount("99")).toBe(99);
  });
});

describe("isTerminal", () => {
  it("is true only for completed / failed / cancelled", () => {
    expect(isTerminal("completed")).toBe(true);
    expect(isTerminal("failed")).toBe(true);
    expect(isTerminal("cancelled")).toBe(true);
    expect(isTerminal("queued")).toBe(false);
    expect(isTerminal("running")).toBe(false);
  });
});

describe("decideAfterSettle", () => {
  it("sends only when completed, runs remain, and the panel can submit", () => {
    expect(decideAfterSettle("completed", 2, true)).toBe("send");
    expect(decideAfterSettle("completed", 1, true)).toBe("send");
  });

  it("stops on failed / cancelled", () => {
    expect(decideAfterSettle("failed", 2, true)).toBe("stop");
    expect(decideAfterSettle("cancelled", 2, true)).toBe("stop");
  });

  it("stops when nothing remains", () => {
    expect(decideAfterSettle("completed", 0, true)).toBe("stop");
  });

  it("stops when the panel cannot submit", () => {
    expect(decideAfterSettle("completed", 2, false)).toBe("stop");
  });
});

describe("submissionFingerprint", () => {
  it("is equal for equal content and differs for different content", () => {
    expect(submissionFingerprint({ prompt: "a", seed: -1 })).toBe(submissionFingerprint({ prompt: "a", seed: -1 }));
    expect(submissionFingerprint({ prompt: "a" })).not.toBe(submissionFingerprint({ prompt: "b" }));
  });
});

describe("shouldNotifyChange", () => {
  const base = { active: true, notified: false, baseline: "A", current: "B" };

  it("is true while active, not yet notified, and the content differs", () => {
    expect(shouldNotifyChange(base)).toBe(true);
  });

  it("is false when not active", () => {
    expect(shouldNotifyChange({ ...base, active: false })).toBe(false);
  });

  it("is false once already notified", () => {
    expect(shouldNotifyChange({ ...base, notified: true })).toBe(false);
  });

  it("is false when the content is unchanged", () => {
    expect(shouldNotifyChange({ ...base, current: "A" })).toBe(false);
  });

  it("is false when either fingerprint is missing", () => {
    expect(shouldNotifyChange({ ...base, baseline: null })).toBe(false);
    expect(shouldNotifyChange({ ...base, current: null })).toBe(false);
  });
});

describe("repeat count text (module variable)", () => {
  it("defaults to '1' for both screens", () => {
    expect(getRepeatCountText("single")).toBe("1");
    expect(getRepeatCountText("chained")).toBe("1");
  });

  it("keeps a separate value per screen", () => {
    setRepeatCountText("single", "5");
    expect(getRepeatCountText("single")).toBe("5");
    expect(getRepeatCountText("chained")).toBe("1");
  });

  it("notifies subscribers on change only, and stops after unsubscribe", () => {
    const listener = vi.fn();
    const unsubscribe = subscribeRepeatCountText(listener);
    setRepeatCountText("chained", "3");
    expect(listener).toHaveBeenCalledTimes(1);
    setRepeatCountText("chained", "3");
    expect(listener).toHaveBeenCalledTimes(1);
    unsubscribe();
    setRepeatCountText("chained", "4");
    expect(listener).toHaveBeenCalledTimes(1);
  });

  it("__resetRepeatCountForTests restores the defaults", () => {
    setRepeatCountText("single", "7");
    setRepeatCountText("chained", "");
    __resetRepeatCountForTests();
    expect(getRepeatCountText("single")).toBe("1");
    expect(getRepeatCountText("chained")).toBe("1");
  });
});
