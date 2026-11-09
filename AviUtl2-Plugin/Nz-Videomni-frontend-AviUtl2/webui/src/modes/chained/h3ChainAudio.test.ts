import { describe, expect, it } from "vitest";
import {
  clipA2vTooShort,
  clipSourceAudioBody,
  h3ChainA2vReasons,
  pickDistributionSlot,
  planH3AudioDistribution,
  type H3AudioDistributionInput,
  type H3ChainA2vReasonClip,
  type H3ClipSlotsView,
} from "./h3ChainAudio";
import {
  createInitialState,
  referenceReducer,
  summarizeReference,
  type ReferenceRules,
  type ReferenceState,
} from "../single/referenceRules";

// §1-120 (plan §7.3): the distribution plan's boundary table at 24 fps on
// MiniMax H3's grid (17n+5, [22, 481]) with the 22-frame seam.

const BASE: Omit<H3AudioDistributionInput, "clipFrames" | "intact" | "audioSec"> = {
  fps: 24,
  grid: { step: 17, offset: 5, min: 22, max: 481 },
  overlap: 22,
  laterMin: 39,
  addedMaxSec: 10,
  maxClips: 24,
  minClips: 2,
};

function plan(clipFrames: number[], audioFrames: number, intact?: boolean[]) {
  return planH3AudioDistribution({
    ...BASE,
    clipFrames,
    intact: intact ?? clipFrames.map(() => true),
    audioSec: audioFrames / 24,
  });
}

describe("planH3AudioDistribution (plan §7.3 boundary table)", () => {
  it("[226,226] and A=430 already fits", () => {
    const p = plan([226, 226], 430);
    expect(p).toMatchObject({ frames: [226, 226], added: 0, outcome: "alreadyFits", unusedFrames: 0 });
  });

  it("[226,226] and A=431 is left as it is with 1 frame unused", () => {
    const p = plan([226, 226], 431);
    expect(p).toMatchObject({ frames: [226, 226], added: 0, outcome: "alreadyFits", unusedFrames: 1 });
  });

  it("30 s (A=720) appends 226 and then the largest grid value within the rest", () => {
    const p = plan([226, 226], 720);
    expect(p.frames).toEqual([226, 226, 226, 107]);
    expect(p).toMatchObject({ added: 2, outcome: "extended", unusedFrames: 1, needsConfirm: false });
    expect(p.shrunk).toEqual([]);
    expect(p.removed).toEqual([]);
  });

  it("[243 x 5] and A=804 removes clip 5 and shortens clip 4 to 141", () => {
    const untouched = plan([243, 243, 243, 243, 243], 804);
    expect(untouched.frames).toEqual([243, 243, 243, 141]);
    expect(untouched.removed).toEqual([4]);
    expect(untouched.shrunk).toEqual([{ index: 3, from: 243, to: 141 }]);
    expect(untouched.outcome).toBe("shrunk");
    // Cards left at their defaults: no confirmation.
    expect(untouched.needsConfirm).toBe(false);
    // Clip 4 or clip 5 touched: confirm.
    expect(plan([243, 243, 243, 243, 243], 804, [true, true, true, false, true]).needsConfirm).toBe(true);
    expect(plan([243, 243, 243, 243, 243], 804, [true, true, true, true, false]).needsConfirm).toBe(true);
    // A touched clip that is neither shortened nor removed: no confirmation.
    expect(plan([243, 243, 243, 243, 243], 804, [false, true, true, true, true]).needsConfirm).toBe(false);
  });

  it("[226,226] and A=250 shortens clip 2 to 39", () => {
    const p = plan([226, 226], 250);
    expect(p.frames).toEqual([226, 39]);
    expect(p.shrunk).toEqual([{ index: 1, from: 226, to: 39 }]);
    expect(p.removed).toEqual([]);
  });

  it("[226,226] and A=240 sets clip 2 to 39 and shortens clip 1 to 209", () => {
    const p = plan([226, 226], 240);
    expect(p.frames).toEqual([209, 39]);
    expect(p.shrunk).toEqual([
      { index: 0, from: 226, to: 209 },
      { index: 1, from: 226, to: 39 },
    ]);
  });

  it("3 s (A=72) gives [39, 39]", () => {
    expect(plan([226, 226], 72).frames).toEqual([39, 39]);
  });

  it("1.5 s (A=36) is too short for a chain and changes nothing", () => {
    const p = plan([226, 226], 36);
    expect(p).toMatchObject({ frames: [226, 226], outcome: "tooShortForChain", needsConfirm: false });
  });

  it("300 s stops at 24 clips with about 95 s unused", () => {
    const p = plan([226, 226], 7200);
    expect(p.frames).toHaveLength(24);
    expect(p.frames.slice(2).every((f) => f === 226)).toBe(true);
    expect(p.outcome).toBe("cappedAtMaxClips");
    expect(p.unusedFrames).toBe(7200 - (430 + 22 * 204));
    expect(p.unusedFrames / 24).toBeCloseTo(95.1, 1);
  });

  it("reads A as floor(seconds x fps)", () => {
    // 17.9166... s = 430 frames exactly; a hair under it is 429.
    expect(planH3AudioDistribution({ ...BASE, clipFrames: [226, 226], intact: [true, true], audioSec: 430 / 24 }).outcome).toBe(
      "alreadyFits",
    );
    expect(
      planH3AudioDistribution({ ...BASE, clipFrames: [226, 226], intact: [true, true], audioSec: 429.5 / 24 }).outcome,
    ).toBe("shrunk");
  });
});

// --- slots --------------------------------------------------------------------

const RULES: ReferenceRules = {
  maxImages: 9,
  maxVideos: 3,
  maxAudios: 3,
  maxTotal: 12,
  videoTotalSec: 15,
  audioTotalSec: 15,
  clipMinSec: 2,
};

function stateWith(apply: (s: ReferenceState) => ReferenceState): H3ClipSlotsView {
  const base = referenceReducer(createInitialState(RULES), { type: "setEnabled", enabled: true });
  const state = apply(base);
  return { summary: summarizeReference(state, RULES), slots: state.slots };
}

function adopt(state: ReferenceState, slot: number, id: string, durationSec: number | null = 30): ReferenceState {
  return referenceReducer(state, {
    type: "adopt",
    kind: "audio",
    slot,
    id,
    fileName: `${id}.wav`,
    filePath: `C:/${id}.wav`,
    durationSec,
  });
}

describe("pickDistributionSlot", () => {
  it("prefers the slot that already holds the long audio", () => {
    const entry = stateWith((s) => adopt(adopt(s, 1, "hand"), 2, "long"));
    expect(pickDistributionSlot(entry, "long")).toBe(2);
  });

  it("then the first free slot", () => {
    const entry = stateWith((s) => adopt(s, 1, "hand"));
    expect(pickDistributionSlot(entry, "long")).toBe(2);
    expect(pickDistributionSlot(stateWith((s) => s), "long")).toBe(1);
  });

  it("then the declared A2V slot (its hand-put audio is replaced)", () => {
    const entry = stateWith((s) =>
      referenceReducer(adopt(adopt(adopt(s, 1, "a"), 2, "b"), 3, "c"), { type: "setA2vSlot", slot: 2 }),
    );
    expect(pickDistributionSlot(entry, "long")).toBe(2);
  });

  it("else nothing (no card is overwritten)", () => {
    const entry = stateWith((s) => adopt(adopt(adopt(s, 1, "a"), 2, "b"), 3, "c"));
    expect(pickDistributionSlot(entry, "long")).toBeNull();
  });
});

describe("clipSourceAudioBody", () => {
  it("sends the long audio with offset_frames = S_i", () => {
    const entry = stateWith((s) => referenceReducer(adopt(s, 1, "long"), { type: "setA2vSlot", slot: 1 }));
    expect(clipSourceAudioBody(entry, { longAudioId: "long", start: 204, strength: 0.5 })).toEqual({
      audio_id: "long",
      strength: 0.5,
      offset_frames: 204,
    });
  });

  it("sends a hand-put audio with offset_frames 0", () => {
    const entry = stateWith((s) => referenceReducer(adopt(s, 2, "hand"), { type: "setA2vSlot", slot: 2 }));
    expect(clipSourceAudioBody(entry, { longAudioId: "long", start: 204, strength: 0.8 })).toEqual({
      audio_id: "hand",
      strength: 0.8,
      offset_frames: 0,
    });
    expect(clipSourceAudioBody(entry, { longAudioId: null, start: 204, strength: 0.8 })?.offset_frames).toBe(0);
  });

  it("sends nothing without a declared, ready A2V card", () => {
    expect(clipSourceAudioBody(stateWith((s) => adopt(s, 1, "long")), { longAudioId: "long", start: 0, strength: 0.5 })).toBeNull();
    const empty = stateWith((s) => referenceReducer(s, { type: "setA2vSlot", slot: 1 }));
    expect(clipSourceAudioBody(empty, { longAudioId: "long", start: 0, strength: 0.5 })).toBeNull();
  });
});

// --- Generate reasons (plan §7.8) -----------------------------------------------

function clip(partial: Partial<H3ChainA2vReasonClip> = {}): H3ChainA2vReasonClip {
  return { numFrames: 226, start: 0, a2v: null, holdsLongAudio: false, ...partial };
}
const LONG = { status: "ready" as const, id: "long", durationSec: 30 };

function reasons(partial: Partial<Parameters<typeof h3ChainA2vReasons>[0]>) {
  return h3ChainA2vReasons({
    clips: [],
    referencesOn: true,
    longAudioReadyId: "long",
    audioFrames: 720,
    hasSourceVideo: false,
    fps: 24,
    ...partial,
  });
}

describe("clipA2vTooShort", () => {
  const base = { start: 204, numFrames: 226, longAudioId: "long", longAudioFrames: 430, fps: 24 };
  it("long: the clip ends past the long audio (S_i + F_i > A)", () => {
    expect(clipA2vTooShort({ ...base, card: LONG })).toBeNull();
    expect(clipA2vTooShort({ ...base, numFrames: 227, card: LONG })).toBe("long");
    expect(clipA2vTooShort({ ...base, numFrames: 227, longAudioFrames: null, card: LONG })).toBeNull();
  });
  it("hand: an audio put in by hand that does not cover the clip", () => {
    expect(clipA2vTooShort({ ...base, card: { status: "ready", id: "hand", durationSec: 5 } })).toBe("hand");
    expect(clipA2vTooShort({ ...base, numFrames: 120, card: { status: "ready", id: "hand", durationSec: 5 } })).toBeNull();
    expect(clipA2vTooShort({ ...base, card: { status: "ready", id: "hand", durationSec: null } })).toBeNull();
  });
  it("nothing for a card that is not ready, or no card", () => {
    expect(clipA2vTooShort({ ...base, card: { status: "uploading", id: null, durationSec: null } })).toBeNull();
    expect(clipA2vTooShort({ ...base, card: null })).toBeNull();
  });
});

describe("h3ChainA2vReasons", () => {
  it("is empty for a distributed chain that fits", () => {
    const r = reasons({
      clips: [clip({ a2v: LONG, holdsLongAudio: true }), clip({ start: 204, a2v: LONG, holdsLongAudio: true })],
    });
    expect(r.codes).toEqual([]);
  });

  it("chainA2vReferencesOff: References off with any declared A2V (hidden cards stop nothing else)", () => {
    const r = reasons({
      referencesOn: false,
      clips: [
        clip({ a2v: { status: "empty", id: null, durationSec: null }, holdsLongAudio: true }),
        clip({ a2v: { status: "ready", id: "hand", durationSec: 1 } }),
      ],
    });
    expect(r.codes).toEqual(["chainA2vReferencesOff"]);
  });

  it("chainA2vNotDistributed: a ready long audio no clip holds", () => {
    expect(reasons({ clips: [clip(), clip()] }).codes).toEqual(["chainA2vNotDistributed"]);
    expect(reasons({ clips: [clip(), clip()], longAudioReadyId: null }).codes).toEqual([]);
  });

  it("chainA2vAudioTooShort: a distributed clip ending past the audio (per clip, not the whole output)", () => {
    const r = reasons({
      audioFrames: 430,
      clips: [clip({ a2v: LONG, holdsLongAudio: true }), clip({ start: 204, numFrames: 243, a2v: LONG, holdsLongAudio: true })],
    });
    expect(r.codes).toEqual(["chainA2vAudioTooShort"]);
    // A clip with no audio past the end does not stop Generate (decision 13).
    const tail = reasons({
      audioFrames: 430,
      clips: [clip({ a2v: LONG, holdsLongAudio: true }), clip({ start: 204, a2v: LONG, holdsLongAudio: true }), clip({ start: 408 })],
    });
    expect(tail.codes).toEqual([]);
  });

  it("clipA2vAudioTooShort / clipA2vAudioMissing carry 1-based clip numbers", () => {
    const r = reasons({
      longAudioReadyId: null,
      clips: [
        clip({ a2v: { status: "ready", id: "hand", durationSec: 5 } }),
        clip({ a2v: { status: "empty", id: null, durationSec: null } }),
        clip({ a2v: { status: "error", id: null, durationSec: null } }),
        clip({ numFrames: 90, a2v: { status: "ready", id: "hand2", durationSec: 5 } }),
      ],
    });
    expect(r.codes).toEqual(["clipA2vAudioTooShort", "clipA2vAudioMissing"]);
    expect(r.tooShortClips).toEqual([1]);
    expect(r.missingClips).toEqual([2, 3]);
  });

  it("clipA2vConflictsWithSourceVideo: a source video with any declared A2V", () => {
    const r = reasons({ hasSourceVideo: true, clips: [clip({ a2v: LONG, holdsLongAudio: true })] });
    expect(r.codes).toEqual(["clipA2vConflictsWithSourceVideo"]);
  });
});
