import { describe, expect, it } from "vitest";
import {
  appendReferenceTag,
  createInitialState,
  effectiveA2vSlot,
  firstEmptySlot,
  referenceReducer,
  referenceTag,
  summarizeReference,
} from "./referenceRules";
import type { ReferenceAction, ReferenceKind, ReferenceRules, ReferenceState } from "./referenceRules";

// §1-107 (UI first): the Reference accordion's reducer and rules.

const RULES: ReferenceRules = {
  maxImages: 9,
  maxVideos: 3,
  maxAudios: 3,
  maxTotal: 12,
  videoTotalSec: 15,
  audioTotalSec: 15,
  clipMinSec: 2,
};

function run(state: ReferenceState, ...actions: ReferenceAction[]): ReferenceState {
  return actions.reduce(referenceReducer, state);
}

/** attachStart + attachDone: a ready slot. */
function put(
  state: ReferenceState,
  kind: ReferenceKind,
  slot: number,
  durationSec: number | null = kind === "picture" ? null : 5,
): ReferenceState {
  return run(
    state,
    { type: "attachStart", kind, slot, filePath: `C:\\m\\${kind}${slot}`, fileName: `${kind}${slot}` },
    { type: "attachDone", kind, slot, id: `${kind}-${slot}`, durationSec, thumbnailDataUrl: null, trimFailed: false },
  );
}

function enabled(rules: ReferenceRules = RULES): ReferenceState {
  return referenceReducer(createInitialState(rules), { type: "setEnabled", enabled: true });
}

describe("referenceRules — state", () => {
  it("builds the slot arrays from the rules, Enable off, no A2V", () => {
    const s = createInitialState(RULES);
    expect(s.enabled).toBe(false);
    expect(s.a2vSlot).toBeNull();
    expect(s.slots.picture).toHaveLength(9);
    expect(s.slots.video).toHaveLength(3);
    expect(s.slots.audio).toHaveLength(3);
    expect(s.slots.picture.map((x) => x.slot)).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9]);
    expect(createInitialState({ ...RULES, maxImages: 2, maxVideos: 1, maxAudios: 4 }).slots.audio).toHaveLength(4);
  });

  it("an attach goes uploading → ready (or error); a late result on a cleared slot is dropped", () => {
    let s = run(createInitialState(RULES), {
      type: "attachStart",
      kind: "video",
      slot: 2,
      filePath: "C:\\a.mp4",
      fileName: "a.mp4",
    });
    expect(s.slots.video[1]).toMatchObject({ status: "uploading", fileName: "a.mp4", addedAt: 1 });
    s = run(s, { type: "attachFail", kind: "video", slot: 2, errorCode: "UPLOAD_FAILED" });
    expect(s.slots.video[1]).toMatchObject({ status: "error", errorCode: "UPLOAD_FAILED" });
    s = run(
      s,
      { type: "clear", kind: "video", slot: 2 },
      { type: "attachDone", kind: "video", slot: 2, id: "v", durationSec: 3, thumbnailDataUrl: null, trimFailed: false },
    );
    expect(s.slots.video[1]?.status).toBe("empty");
  });

  it("clearing the checked audio slot unchecks A2V", () => {
    let s = run(put(enabled(), "audio", 2), { type: "setA2vSlot", slot: 2 });
    expect(s.a2vSlot).toBe(2);
    s = run(s, { type: "clear", kind: "audio", slot: 2 });
    expect(s.a2vSlot).toBeNull();
    expect(s.slots.audio[1]?.status).toBe("empty");
  });

  it("resize keeps the existing slots that still fit and adds/drops the rest", () => {
    let s = put(put(put(enabled(), "picture", 1), "picture", 3), "audio", 3);
    s = run(s, { type: "setA2vSlot", slot: 3 });
    const smaller = run(s, { type: "resize", rules: { ...RULES, maxImages: 2, maxAudios: 2 } });
    expect(smaller.slots.picture).toHaveLength(2);
    expect(smaller.slots.picture[0]).toMatchObject({ status: "ready", id: "picture-1" });
    expect(smaller.slots.audio).toHaveLength(2);
    expect(smaller.a2vSlot).toBeNull();
    expect(smaller.enabled).toBe(true);

    const larger = run(smaller, { type: "resize", rules: { ...RULES, maxImages: 12 } });
    expect(larger.slots.picture).toHaveLength(12);
    expect(larger.slots.picture[0]?.id).toBe("picture-1");
    expect(larger.slots.picture[11]).toMatchObject({ slot: 12, status: "empty" });
  });
});

describe("referenceRules — summarizeReference", () => {
  it("10s / 6s / 5s videos: full length, the head 5s (truncated), left out — the raw total is over", () => {
    const s = put(put(put(enabled(), "video", 1, 10), "video", 2, 6), "video", 3, 5);
    const sum = summarizeReference(s, RULES);
    const [v1, v2, v3] = sum.slotUse.video;
    expect(v1).toMatchObject({ used: true, ignored: null, usedSec: 10, truncated: false });
    expect(v2).toMatchObject({ used: true, ignored: null, usedSec: 5, truncated: true, tooShort: false });
    expect(v3).toMatchObject({ used: false, ignored: "overDuration", usedSec: null });
    expect(sum.seconds.video).toBe(21);
    expect(sum.over.videoSec).toBe(true);
    expect(sum.filled).toEqual({ total: 3, picture: 0, video: 3, audio: 0 });
  });

  it("cuts by slot NUMBER, not by input order", () => {
    // Slot 3 entered first, slot 1 last: slot 1 still comes first in seconds.
    const s = put(put(put(enabled(), "audio", 3, 10), "audio", 2, 6), "audio", 1, 10);
    const [a1, a2, a3] = summarizeReference(s, RULES).slotUse.audio;
    expect(a1).toMatchObject({ used: true, usedSec: 10 });
    expect(a2).toMatchObject({ used: true, usedSec: 5, truncated: true });
    expect(a3).toMatchObject({ ignored: "overDuration" });
  });

  it("13 filled: the latest input is left out; re-attaching an older one swaps them", () => {
    let s = enabled();
    for (let i = 1; i <= 9; i += 1) s = put(s, "picture", i);
    for (let i = 1; i <= 3; i += 1) s = put(s, "video", i, 3);
    s = put(s, "audio", 1, 3);
    let sum = summarizeReference(s, RULES);
    expect(sum.filled.total).toBe(13);
    expect(sum.over.total).toBe(true);
    expect(sum.slotUse.audio[0]?.ignored).toBe("overTotal");
    expect(sum.slotUse.picture.every((u) => u.used)).toBe(true);

    // 🔁 on Picture 1: it gets the newest input number, so it is the 13th now.
    s = put(s, "picture", 1);
    sum = summarizeReference(s, RULES);
    expect(sum.slotUse.picture[0]?.ignored).toBe("overTotal");
    expect(sum.slotUse.audio[0]).toMatchObject({ used: true, ignored: null });

    // Clearing any slot brings the total back to 12: nothing is left out.
    s = run(s, { type: "clear", kind: "video", slot: 2 });
    sum = summarizeReference(s, RULES);
    expect(sum.filled.total).toBe(12);
    expect(sum.over.total).toBe(false);
    expect(sum.slotUse.picture[0]).toMatchObject({ used: true, ignored: null });
  });

  it("an overTotal slot does not use up the seconds", () => {
    let s = enabled();
    for (let i = 1; i <= 9; i += 1) s = put(s, "picture", i);
    s = put(s, "video", 2, 3);
    s = put(s, "video", 3, 3);
    s = put(s, "audio", 1, 3);
    s = put(s, "video", 1, 14); // 13th by input order
    const use = summarizeReference(s, RULES).slotUse.video;
    expect(use[0]?.ignored).toBe("overTotal");
    expect(use[1]).toMatchObject({ used: true, usedSec: 3, truncated: false });
    expect(use[2]).toMatchObject({ used: true, usedSec: 3, truncated: false });
  });

  it("Enable off: every filled slot is left out, the counters still count", () => {
    const s = run(put(put(enabled(), "picture", 1), "video", 1, 1), { type: "setEnabled", enabled: false });
    const sum = summarizeReference(s, RULES);
    expect(sum.slotUse.picture[0]).toMatchObject({ used: false, ignored: "disabled" });
    // The clip's own warning still shows on a left-out card.
    expect(sum.slotUse.video[0]).toMatchObject({ used: false, ignored: "disabled", tooShort: true });
    expect(sum.slotUse.picture[1]).toMatchObject({ used: false, ignored: null });
    expect(sum.filled.total).toBe(2);
    expect(sum.seconds.video).toBe(1);
  });

  it("A2V is exclusive: the checked audio is used, every other filled slot is left out", () => {
    let s = put(put(put(enabled(), "picture", 1), "video", 1, 5), "audio", 2, 8);
    s = put(s, "audio", 1, 4);
    s = run(s, { type: "setA2vSlot", slot: 2 });
    expect(effectiveA2vSlot(s)).toBe(2);
    const sum = summarizeReference(s, RULES);
    expect(sum.slotUse.audio[1]).toMatchObject({ used: true, ignored: null, isA2v: true, usedSec: null });
    expect(sum.slotUse.audio[0]?.ignored).toBe("a2vExclusive");
    expect(sum.slotUse.picture[0]?.ignored).toBe("a2vExclusive");
    expect(sum.slotUse.video[0]?.ignored).toBe("a2vExclusive");
  });

  it("the A2V slot is outside n/12, n/3 and n/15s (a 20s wav does not turn the counter red)", () => {
    let s = put(put(enabled(), "audio", 1, 20), "audio", 3, 4);
    s = run(s, { type: "setA2vSlot", slot: 1 });
    const sum = summarizeReference(s, RULES);
    expect(sum.filled).toEqual({ total: 1, picture: 0, video: 0, audio: 1 });
    expect(sum.seconds.audio).toBe(4);
    expect(sum.over.audioSec).toBe(false);
    expect(sum.slotUse.audio[0]).toMatchObject({ isA2v: true, truncated: false, tooShort: false });
  });

  it("Enable off: the checked A2V slot is an ordinary (left-out) card and counts", () => {
    let s = put(enabled(), "audio", 1, 20);
    s = run(s, { type: "setA2vSlot", slot: 1 }, { type: "setEnabled", enabled: false });
    expect(effectiveA2vSlot(s)).toBeNull();
    const sum = summarizeReference(s, RULES);
    expect(sum.filled.audio).toBe(1);
    expect(sum.seconds.audio).toBe(20);
    expect(sum.over.audioSec).toBe(true);
    expect(sum.slotUse.audio[0]).toMatchObject({ isA2v: false, ignored: "disabled" });
  });

  it("A2V on a slot still uploading is not in effect yet", () => {
    let s = run(enabled(), {
      type: "attachStart",
      kind: "audio",
      slot: 1,
      filePath: "C:\\a.wav",
      fileName: "a.wav",
    });
    s = run(s, { type: "setA2vSlot", slot: 1 });
    expect(effectiveA2vSlot(s)).toBeNull();
  });

  it("under 2 seconds: warned, still used — also when a truncation makes it short", () => {
    const s = put(put(enabled(), "video", 1, 1.5), "video", 2, 14);
    const [v1, v2] = summarizeReference(s, RULES).slotUse.video;
    expect(v1).toMatchObject({ used: true, ignored: null, tooShort: true });
    expect(v2).toMatchObject({ used: true, truncated: true, usedSec: 13.5, tooShort: false });

    const s2 = put(put(enabled(), "audio", 1, 14), "audio", 2, 6);
    expect(summarizeReference(s2, RULES).slotUse.audio[1]).toMatchObject({
      used: true,
      truncated: true,
      usedSec: 1,
      tooShort: true,
    });
  });

  it("left-out cards still warn: under 2 seconds on the raw length, unknown length; no truncation", () => {
    // Enable off (the default): the warnings show from the first clip.
    let s = put(put(createInitialState(RULES), "video", 1, 1.5), "audio", 1, null);
    let sum = summarizeReference(s, RULES);
    expect(sum.slotUse.video[0]).toMatchObject({ ignored: "disabled", tooShort: true, truncated: false });
    expect(sum.slotUse.audio[0]).toMatchObject({ ignored: "disabled", durationUnknown: true });

    // Past 12 by input order: the 13th (a 1s video) is left out and still warned.
    s = enabled();
    for (let i = 1; i <= 9; i += 1) s = put(s, "picture", i);
    s = put(put(put(s, "video", 2, 3), "video", 3, 3), "audio", 1, 3);
    s = put(s, "video", 1, 1);
    sum = summarizeReference(s, RULES);
    expect(sum.slotUse.video[0]).toMatchObject({ ignored: "overTotal", tooShort: true, truncated: false });

    // A left-out card is judged on its raw length, not the 0s left in the budget.
    s = put(put(enabled(), "video", 1, 15), "video", 2, 5);
    expect(summarizeReference(s, RULES).slotUse.video[1]).toMatchObject({
      ignored: "overDuration",
      tooShort: false,
    });
  });

  it("fills 15s exactly despite float sums (10.2 + 4.8)", () => {
    const s = put(put(put(enabled(), "video", 1, 10.2), "video", 2, 4.8), "video", 3, 3);
    const [v1, v2, v3] = summarizeReference(s, RULES).slotUse.video;
    expect(v1?.truncated).toBe(false);
    expect(v2).toMatchObject({ used: true, truncated: false });
    expect(v3?.ignored).toBe("overDuration");
  });

  it("an unknown length: warned, not greyed, not in the seconds", () => {
    const s = put(put(enabled(), "video", 1, null), "video", 2, 15);
    const sum = summarizeReference(s, RULES);
    expect(sum.slotUse.video[0]).toMatchObject({ used: true, ignored: null, durationUnknown: true, usedSec: null });
    expect(sum.slotUse.video[1]).toMatchObject({ used: true, usedSec: 15, truncated: false });
    expect(sum.seconds.video).toBe(15);
    // Still uploading: no warning yet.
    const uploading = run(enabled(), {
      type: "attachStart",
      kind: "audio",
      slot: 1,
      filePath: "C:\\a.mp3",
      fileName: "a.mp3",
    });
    expect(summarizeReference(uploading, RULES).slotUse.audio[0]).toMatchObject({
      used: true,
      durationUnknown: false,
    });
  });

  it("counts uploading slots as filled, error slots not", () => {
    let s = run(enabled(), { type: "attachStart", kind: "picture", slot: 1, filePath: "C:\\p", fileName: "p" });
    s = run(
      s,
      { type: "attachStart", kind: "picture", slot: 2, filePath: "C:\\q", fileName: "q" },
      { type: "attachFail", kind: "picture", slot: 2, errorCode: "X" },
    );
    expect(summarizeReference(s, RULES).filled.picture).toBe(1);
  });
});

describe("referenceRules — helpers", () => {
  it("referenceTag", () => {
    expect(referenceTag("picture", 1)).toBe("<Picture 1>");
    expect(referenceTag("video", 3)).toBe("<Video 3>");
    expect(referenceTag("audio", 2)).toBe("<Audio 2>");
  });

  it("appendReferenceTag: empty, trailing whitespace, ordinary, duplicate", () => {
    expect(appendReferenceTag("", "<Picture 1>")).toBe("<Picture 1>");
    expect(appendReferenceTag("   \n", "<Picture 1>")).toBe("<Picture 1>");
    expect(appendReferenceTag("a cat  \n", "<Video 1>")).toBe("a cat <Video 1>");
    expect(appendReferenceTag("a cat", "<Audio 2>")).toBe("a cat <Audio 2>");
    expect(appendReferenceTag("a <Audio 2>", "<Audio 2>")).toBe("a <Audio 2> <Audio 2>");
  });

  it("firstEmptySlot: the lowest empty-or-error slot, null when all are filled", () => {
    let s = put(put(enabled(), "video", 1), "video", 3);
    expect(firstEmptySlot(s.slots.video)).toBe(2);
    s = put(s, "video", 2);
    expect(firstEmptySlot(s.slots.video)).toBeNull();
    s = run(
      s,
      { type: "attachStart", kind: "video", slot: 3, filePath: "C:\\v", fileName: "v" },
      { type: "attachFail", kind: "video", slot: 3, errorCode: "X" },
    );
    expect(firstEmptySlot(s.slots.video)).toBe(3);
    // An uploading slot is not free.
    s = run(s, { type: "attachStart", kind: "video", slot: 3, filePath: "C:\\v", fileName: "v" });
    expect(firstEmptySlot(s.slots.video)).toBeNull();
  });
});
