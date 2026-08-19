import { describe, expect, it, vi } from "vitest";
import { createMockBridge } from "../bridge/mockBridge";
import {
  EFFECT_NAME_AUDIO,
  EFFECT_NAME_IMAGE,
  EFFECT_NAME_TEXT,
  EFFECT_NAME_VIDEO,
  END_SOURCE_MIN_DURATION_SEC,
  GEN_FPS,
  ICLORA_MIN_DURATION_SEC,
  V2V_MIN_DURATION_SEC,
  classifySelectionKind,
  formatMenuGuardNote,
  guardMenuSelection,
  lengthFloorForAction,
  lengthMarginForAction,
  requiredKindsOf,
  resolveMenuSelection,
  selectionHasMissingFilePath,
} from "./menuSelection";
import type { MenuGuardNote, SelectionItem, TimelineSelection } from "./menuSelection";
import type { TrimSelectionItem } from "./sourceTrim";
import { routeMenuAction } from "./menuRouting";
import type { MaterialKind, MenuRoute } from "./menuRouting";
import { en, ja } from "../i18n/strings";

function makeSelection(overrides: Partial<TimelineSelection> = {}): TimelineSelection {
  return {
    hasRange: true,
    rangeStart: 0,
    rangeEnd: 60,
    selected: [
      { layer: 1, frameStart: 0, frameEnd: 60, effectName: "動画ファイル", filePath: "C:\\v\\a.mp4", objectName: "a", textContent: null, mediaWidth: 1920, mediaHeight: 1080, mediaDurationSec: 0 },
    ],
    cursorFrame: 30,
    cursorLayer: 1,
    rate: 30,
    scale: 1,
    sampleRate: 44100,
    ...overrides,
  };
}

describe("selectionHasMissingFilePath", () => {
  it("is false when every selected object has a filePath", () => {
    expect(selectionHasMissingFilePath(makeSelection())).toBe(false);
  });

  it("is true when any selected object has a null filePath", () => {
    const sel = makeSelection({
      selected: [
        { layer: 1, frameStart: 0, frameEnd: 60, effectName: "動画ファイル", filePath: null, objectName: "a", textContent: null, mediaWidth: 0, mediaHeight: 0, mediaDurationSec: 0 },
      ],
    });
    expect(selectionHasMissingFilePath(sel)).toBe(true);
  });

  it("is true when only one of several objects is missing its filePath", () => {
    const sel = makeSelection({
      selected: [
        { layer: 1, frameStart: 0, frameEnd: 60, effectName: "動画ファイル", filePath: "C:\\v\\a.mp4", objectName: "a", textContent: null, mediaWidth: 1920, mediaHeight: 1080, mediaDurationSec: 0 },
        { layer: 2, frameStart: 0, frameEnd: 60, effectName: "動画ファイル", filePath: null, objectName: "b", textContent: null, mediaWidth: 0, mediaHeight: 0, mediaDurationSec: 0 },
      ],
    });
    expect(selectionHasMissingFilePath(sel)).toBe(true);
  });

  it("is false for an empty selection (nothing to complete)", () => {
    expect(selectionHasMissingFilePath(makeSelection({ selected: [] }))).toBe(false);
  });
});

describe("resolveMenuSelection", () => {
  it("does not re-query when needsSelection is false", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    const spy = vi.spyOn(bridge, "request");
    const snapshot = makeSelection({
      selected: [
        { layer: 1, frameStart: 0, frameEnd: 60, effectName: "動画ファイル", filePath: null, objectName: "a", textContent: null, mediaWidth: 0, mediaHeight: 0, mediaDurationSec: 0 },
      ],
    });

    const result = await resolveMenuSelection(bridge, { snapshot, needsSelection: false });

    expect(result.refreshed).toBe(false);
    expect(result.selection).toBe(snapshot);
    expect(spy).not.toHaveBeenCalled();
  });

  it("does not re-query when file paths are already present", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    const spy = vi.spyOn(bridge, "request");
    const snapshot = makeSelection();

    const result = await resolveMenuSelection(bridge, { snapshot, needsSelection: true });

    expect(result.refreshed).toBe(false);
    expect(result.selection).toBe(snapshot);
    expect(spy).not.toHaveBeenCalled();
  });

  it("re-queries getSelection when a filePath is missing and needsSelection is true", async () => {
    // The mock's default getSelection resolves a snapshot WITH a filePath.
    const bridge = createMockBridge({ delayMs: 0 });
    const spy = vi.spyOn(bridge, "request");
    const snapshot = makeSelection({
      selected: [
        { layer: 1, frameStart: 0, frameEnd: 60, effectName: "動画ファイル", filePath: null, objectName: "stale", textContent: null, mediaWidth: 0, mediaHeight: 0, mediaDurationSec: 0 },
      ],
    });

    const result = await resolveMenuSelection(bridge, { snapshot, needsSelection: true });

    expect(spy).toHaveBeenCalledWith("timeline.getSelection", {});
    expect(result.refreshed).toBe(true);
    // The returned selection is the authoritative one, with the filePath filled.
    expect(result.selection).not.toBe(snapshot);
    expect(result.selection.selected[0]?.filePath).not.toBeNull();
  });

  it("propagates a bridge transport failure from the re-query", async () => {
    const bridge = createMockBridge({ delayMs: 0, failGetSelection: true });
    const snapshot = makeSelection({
      selected: [
        { layer: 1, frameStart: 0, frameEnd: 60, effectName: "動画ファイル", filePath: null, objectName: "a", textContent: null, mediaWidth: 0, mediaHeight: 0, mediaDurationSec: 0 },
      ],
    });

    await expect(
      resolveMenuSelection(bridge, { snapshot, needsSelection: true }),
    ).rejects.toThrow();
  });
});

// --- I6 guards (§4-2 / §4-3 / §4-5) ---------------------------------------

const EFFECT_BY_KIND: Record<MaterialKind, string> = {
  video: EFFECT_NAME_VIDEO,
  image: EFFECT_NAME_IMAGE,
  audio: EFFECT_NAME_AUDIO,
  text: EFFECT_NAME_TEXT,
};

/** A single selected object with a given effectName and duration; other fields
 * are inert defaults. */
function makeItem(effectName: string, mediaDurationSec = 0): SelectionItem {
  return {
    layer: 1, frameStart: 0, frameEnd: 60, effectName,
    filePath: "C:\\m\\x", objectName: "x", textContent: null,
    mediaWidth: 1920, mediaHeight: 1080, mediaDurationSec,
  };
}

/** Non-null resolved route (all eight object actions + two layer actions
 * exist in the table). */
function route(action: string): MenuRoute {
  const r = routeMenuAction(action);
  if (r === null) throw new Error(`no route for ${action}`);
  return r;
}

const OBJECT_ACTIONS = [
  "extendVideo", "referenceVideo", "videoAudioToVideo", "imageToVideo",
  "addImageKeyframe", "imageToClipChain", "audioToVideo", "appendText",
  // 台帳§1-16 長尺A2V (2026-08-10). `audioToLongA2v` is the SECOND member of the
  // audio-required set, so the shared matrix below now exercises the near-
  // neighbor rescue on both of them (see NEAR_NEIGHBOR_NOTE).
  "videoAudioToLongA2v", "audioToLongA2v",
  // W0 Edit-系 (2026-08-09). `makeSelection` defaults to `hasRange: true`, so
  // the (e) range guard never fires in the shared kind/multi-select matrices
  // below — exactly right: those exercise (a)-(c), and (e) has its own describe.
  "outpaintVideo", "retakeRange",
] as const;
const ALL_KINDS: MaterialKind[] = ["video", "image", "audio", "text"];

/** §4-3 近傍アクション: audio-required action -> the note a mis-selected VIDEO
 * gets instead of the bare `typeMismatch`. Mirrors the production table
 * (`VIDEO_FOR_AUDIO_ACTION_NOTE`), kept here as an independent expectation so a
 * silent change to the pairing fails. */
const NEAR_NEIGHBOR_NOTE: Record<string, string> = {
  audioToVideo: "useVideoAudioInstead",
  audioToLongA2v: "useVideoAudioLongInstead",
};

describe("classifySelectionKind", () => {
  it("maps each native effectName to its kind", () => {
    expect(classifySelectionKind(makeItem(EFFECT_NAME_VIDEO))).toBe("video");
    expect(classifySelectionKind(makeItem(EFFECT_NAME_IMAGE))).toBe("image");
    expect(classifySelectionKind(makeItem(EFFECT_NAME_AUDIO))).toBe("audio");
    expect(classifySelectionKind(makeItem(EFFECT_NAME_TEXT))).toBe("text");
  });

  it("returns 'unknown' for an empty or unrecognized effectName", () => {
    expect(classifySelectionKind(makeItem(""))).toBe("unknown");
    expect(classifySelectionKind(makeItem("図形"))).toBe("unknown");
    expect(classifySelectionKind(makeItem("動画ファイル "))).toBe("unknown"); // trailing space
  });
});

describe("guardMenuSelection — multiple selection (§4-2)", () => {
  it("blocks any object command when two or more objects are selected", () => {
    const sel = makeSelection({
      selected: [makeItem(EFFECT_NAME_VIDEO, 10), makeItem(EFFECT_NAME_VIDEO, 10)],
    });
    for (const action of OBJECT_ACTIONS) {
      const r = guardMenuSelection(route(action), sel);
      expect(r.ok).toBe(false);
      if (!r.ok) expect(r.note.noteKind).toBe("multipleSelection");
    }
  });

  it("does not fire multipleSelection for the layer commands (exempt)", () => {
    const sel = makeSelection({ selected: [makeItem(EFFECT_NAME_VIDEO), makeItem(EFFECT_NAME_IMAGE)] });
    expect(guardMenuSelection(route("textToVideoHere"), sel).ok).toBe(true);
    expect(guardMenuSelection(route("imageFromCurrentFrame"), sel).ok).toBe(true);
  });
});

describe("guardMenuSelection — undeterminable kind (§4-3)", () => {
  it("returns unsupportedType for an unrecognized effectName", () => {
    const sel = makeSelection({ selected: [makeItem("")] });
    const r = guardMenuSelection(route("imageToVideo"), sel);
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.note.noteKind).toBe("unsupportedType");
  });

  it("returns unsupportedType for an object command with an empty selection", () => {
    const sel = makeSelection({ selected: [] });
    const r = guardMenuSelection(route("imageToVideo"), sel);
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.note.noteKind).toBe("unsupportedType");
  });
});

describe("guardMenuSelection — required-kind matrix (§4-3)", () => {
  // Table-driven: every object action × every kind. A long (10s) media duration
  // keeps the #1/#2 length guard out of the way so this isolates the type check.
  for (const action of OBJECT_ACTIONS) {
    const required = route(action).requiredKind;
    for (const kind of ALL_KINDS) {
      it(`${action} + ${kind}`, () => {
        const sel = makeSelection({ selected: [makeItem(EFFECT_BY_KIND[kind], 10)] });
        const r = guardMenuSelection(route(action), sel);
        if (kind === required) {
          expect(r.ok).toBe(true);
        } else if (kind === "video" && NEAR_NEIGHBOR_NOTE[action]) {
          // The near-neighbor special cases (§4-3): a video selected for an
          // audio-required action steers to that action's own video sibling.
          expect(r.ok).toBe(false);
          if (!r.ok) expect(r.note.noteKind).toBe(NEAR_NEIGHBOR_NOTE[action]);
        } else {
          expect(r.ok).toBe(false);
          if (!r.ok && r.note.noteKind === "typeMismatch") {
            expect(r.note.selected).toBe(kind);
            expect(r.note.required).toBe(required);
          } else {
            throw new Error(`expected typeMismatch for ${action}+${kind}`);
          }
        }
      });
    }
  }
});

describe("guardMenuSelection — near-neighbor special case (§4-3)", () => {
  it("steers a video selected for #7 audioToVideo to the video-audio a2v", () => {
    const sel = makeSelection({ selected: [makeItem(EFFECT_NAME_VIDEO, 10)] });
    const r = guardMenuSelection(route("audioToVideo"), sel);
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.note.noteKind).toBe("useVideoAudioInstead");
  });

  it("is the ONLY near-neighbor: a video for #3 videoAudioToVideo just passes", () => {
    const sel = makeSelection({ selected: [makeItem(EFFECT_NAME_VIDEO, 10)] });
    expect(guardMenuSelection(route("videoAudioToVideo"), sel).ok).toBe(true);
  });

  // 台帳§1-16 長尺A2V: the rescue is a two-row table now, not a single literal.
  it("steers a video selected for 🎵 audioToLongA2v to the LONG video sibling (not #3)", () => {
    const sel = makeSelection({ selected: [makeItem(EFFECT_NAME_VIDEO, 10)] });
    const r = guardMenuSelection(route("audioToLongA2v"), sel);
    expect(r.ok).toBe(false);
    // Deliberately asserts the LONG variant: steering to the single-shot #3
    // would send the user to a different feature.
    if (!r.ok) expect(r.note.noteKind).toBe("useVideoAudioLongInstead");
  });

  it("🎬 videoAudioToLongA2v with an AUDIO object is a plain typeMismatch (no reverse rescue)", () => {
    const sel = makeSelection({ selected: [makeItem(EFFECT_NAME_AUDIO, 10)] });
    const r = guardMenuSelection(route("videoAudioToLongA2v"), sel);
    expect(r.ok).toBe(false);
    if (!r.ok && r.note.noteKind === "typeMismatch") {
      expect(r.note.selected).toBe("audio");
      expect(r.note.required).toBe("video");
    } else {
      throw new Error("expected typeMismatch");
    }
  });

  it("🎬 videoAudioToLongA2v passes with a video and has no length floor", () => {
    // The long-a2v items have no §4-5 floor (only #1/#2 do), so even a 2-frame
    // ribbon passes the guard — the audio panel's own gates take over later.
    const sel = makeSelection({ selected: [makeItem(EFFECT_NAME_VIDEO, 0.05)] });
    expect(guardMenuSelection(route("videoAudioToLongA2v"), sel).ok).toBe(true);
    expect(lengthFloorForAction("videoAudioToLongA2v")).toBeNull();
    expect(lengthFloorForAction("audioToLongA2v")).toBeNull();
  });
});

describe("guardMenuSelection — length guard (§4-5)", () => {
  // #1 extendVideo floor = V2V_MIN_DURATION_SEC (73f/24fps ≈ 3.04s);
  // #2 referenceVideo floor = ICLORA_MIN_DURATION_SEC (25f/24fps ≈ 1.04s).
  //
  // 2026-08-01 (§1-6 拡張): BOTH are now measured the same way — the
  // `decideSourceTrim` window when a trim fires, the full `mediaDurationSec`
  // otherwise — because #2's upload gained the same ribbon-range trim #1 has.
  // The #2 cases below were rewritten for that (each one says how); the earlier
  // W4 rule measured #2's timeline SPAN unconditionally.
  // A video object with an inclusive `frames`-long span at 24fps -> span =
  // frames/24 s.
  function videoSpan(frames: number, rate = 24, scale = 1): TimelineSelection {
    return makeSelection({
      selected: [{ ...makeItem(EFFECT_NAME_VIDEO), frameStart: 0, frameEnd: frames - 1 }],
      rate,
      scale,
    });
  }

  /** §1-6: a 60-second video object carrying the contract-v10 playback fields,
   * so `decideSourceTrim` actually reaches a `trim: true` — the ribbon plays
   * from 5.0s to the end of the file. */
  function trimmableVideoItem(overrides: Partial<TrimSelectionItem>): SelectionItem {
    const item: TrimSelectionItem = {
      ...makeItem(EFFECT_NAME_VIDEO, 60),
      hasPlaybackRange: true,
      playbackStartSec: 5,
      playbackEndSec: 60,
      ...overrides,
    };
    return item;
  }

  // --- #1 extendVideo: measured on the FULL source duration -----------------
  it("extendVideo: passes at exactly the floor (full mediaDurationSec)", () => {
    const sel = makeSelection({ selected: [makeItem(EFFECT_NAME_VIDEO, V2V_MIN_DURATION_SEC)] });
    expect(guardMenuSelection(route("extendVideo"), sel).ok).toBe(true);
  });

  it("extendVideo: blocks just under the floor", () => {
    const sel = makeSelection({ selected: [makeItem(EFFECT_NAME_VIDEO, V2V_MIN_DURATION_SEC - 0.01)] });
    const r = guardMenuSelection(route("extendVideo"), sel);
    expect(r.ok).toBe(false);
    if (!r.ok && r.note.noteKind === "videoTooShort") {
      expect(r.note.requiredSeconds).toBe(V2V_MIN_DURATION_SEC);
    } else {
      throw new Error("expected videoTooShort");
    }
  });

  // §1-6: #1 now measures "the material that will actually be uploaded", i.e.
  // `decideSourceTrim`'s window when a trim applies. Today's bridge reports no
  // playback position, so every case ABOVE (and every real right-click) keeps
  // measuring the full `mediaDurationSec` — these two pin the trimmed branch,
  // using synthesized playback fields.
  it("extendVideo: with a live trim decision, measures the TRIMMED window, not the full file", () => {
    // A 60s file whose ribbon uses only 1s (30 frames @30fps) starting at 5s.
    // The full file would sail past the 3.04s floor; the trim window must not.
    const sel = makeSelection({
      selected: [trimmableVideoItem({ frameStart: 0, frameEnd: 29 })],
      rate: 30,
      scale: 1,
    });
    const r = guardMenuSelection(route("extendVideo"), sel);
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.note.noteKind).toBe("videoTooShort");
  });

  it("extendVideo: a trimmed window that is long enough still passes", () => {
    // 150 frames @30fps = 5.0s of a 60s file -> above the 3.04s floor.
    const sel = makeSelection({
      selected: [trimmableVideoItem({ frameStart: 0, frameEnd: 149 })],
      rate: 30,
      scale: 1,
    });
    expect(guardMenuSelection(route("extendVideo"), sel).ok).toBe(true);
  });

  it("extendVideo: passes when duration is unknown (mediaDurationSec === 0)", () => {
    // 0 means "duration unknown" (a failed get_media_info lookup), not
    // "0 seconds" — per §4-5's crash-avoidance intent we must not assert
    // "too short" without a measurement, so it passes and the server arbitrates.
    const sel = makeSelection({ selected: [makeItem(EFFECT_NAME_VIDEO, 0)] });
    expect(guardMenuSelection(route("extendVideo"), sel).ok).toBe(true);
  });

  // --- #2 referenceVideo: SAME rule as #1 since the 2026-08-01 §1-6 拡張 ------
  // (the trim window when one fires, the full file when it does not).

  /** A #2 selection whose ribbon really is trimmable: `frames` frames @24fps out
   * of a 60s file, played from 5.0s to the end. `decideSourceTrim` reaches
   * `trim: true` and its window is the ribbon's own span (the played window,
   * 55s, is far longer, so `min` picks the span). */
  function referenceTrimmable(frames: number): TimelineSelection {
    return makeSelection({
      selected: [trimmableVideoItem({ frameStart: 0, frameEnd: frames - 1 })],
      rate: 24,
      scale: 1,
    });
  }

  it("referenceVideo: passes at exactly the floor (25-frame trim window @24fps = ICLORA floor)", () => {
    // 挙動更新 (§1-6 拡張): the measured quantity is now the TRIM WINDOW rather
    // than the raw span, so the fixture carries the playback fields that make a
    // trim fire. The window equals the span here (25/24 s ===
    // ICLORA_MIN_DURATION_SEC), so the boundary itself is unchanged.
    const sel = referenceTrimmable(25);
    expect(spanSeconds(sel)).toBeCloseTo(ICLORA_MIN_DURATION_SEC, 6);
    expect(guardMenuSelection(route("referenceVideo"), sel).ok).toBe(true);
  });

  it("referenceVideo: blocks a trim window just under the floor (24 frames @24fps < floor)", () => {
    // 挙動更新: previously this was `videoSpan(24)` — a fixture with
    // `mediaDurationSec: 0` and no playback fields, which under the unified rule
    // measures nothing at all and passes (see the "unknown duration" case
    // below). It now carries the playback fields so a real 1.0s trim window is
    // what falls under the 25/24 ≈ 1.04s floor.
    const sel = referenceTrimmable(24);
    const r = guardMenuSelection(route("referenceVideo"), sel);
    expect(r.ok).toBe(false);
    if (!r.ok && r.note.noteKind === "videoTooShort") {
      expect(r.note.requiredSeconds).toBe(ICLORA_MIN_DURATION_SEC);
    } else {
      throw new Error("expected videoTooShort");
    }
  });

  it("referenceVideo: a too-short trim window blocks even on a very long file", () => {
    // 挙動更新: the former "uses the SPAN, not the full file" case, restated. A
    // generous full-file `mediaDurationSec` must not save a too-short window —
    // still true, but only because a trim actually fires. 1 frame @24fps out of
    // a 999s file, played from 5.0s.
    const sel = makeSelection({
      selected: [
        trimmableVideoItem({ mediaDurationSec: 999, playbackEndSec: 999, frameStart: 0, frameEnd: 0 }),
      ],
      rate: 24,
      scale: 1,
    });
    const r = guardMenuSelection(route("referenceVideo"), sel);
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.note.noteKind).toBe("videoTooShort");
  });

  it("referenceVideo: falls back to the FULL file when no trim applies", () => {
    // 挙動更新 — this is the case the unification deliberately flips. A short
    // ribbon over a long file WITHOUT the playback fields (`hasPlaybackRange`
    // unset: an older plugin, or an object native could not read) uploads the
    // whole file, so the material really is 999s and the guard must pass.
    // Under W4's span-only rule this blocked, which would now false-reject a
    // perfectly usable full-file reference.
    const sel = makeSelection({
      selected: [{ ...makeItem(EFFECT_NAME_VIDEO, 999), frameStart: 0, frameEnd: 23 }],
      rate: 24,
      scale: 1,
    });
    expect(guardMenuSelection(route("referenceVideo"), sel).ok).toBe(true);
  });

  it("referenceVideo: blocks a short FULL file when no trim applies", () => {
    // The other half of the fallback: no trim, and the whole file is itself
    // under the floor -> block, exactly like #1 does on `mediaDurationSec`.
    const sel = makeSelection({
      selected: [{ ...makeItem(EFFECT_NAME_VIDEO, 1.0), frameStart: 0, frameEnd: 23 }],
      rate: 24,
      scale: 1,
    });
    const r = guardMenuSelection(route("referenceVideo"), sel);
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.note.noteKind).toBe("videoTooShort");
  });

  it("referenceVideo: passes when the duration is unknown (mediaDurationSec === 0)", () => {
    // 挙動更新: the unknown-measurement escape hatch is now #1's — a failed
    // `get_media_info` (0) rather than an unresolvable span. Pass and let the
    // server arbitrate.
    const sel = makeSelection({
      selected: [{ ...makeItem(EFFECT_NAME_VIDEO, 0), frameStart: 0, frameEnd: 5 }],
      rate: 24,
      scale: 1,
    });
    expect(guardMenuSelection(route("referenceVideo"), sel).ok).toBe(true);
  });

  it("does not apply a length guard to other video actions (#3 videoAudioToVideo)", () => {
    // A very short span for #3 is fine — only #1/#2 have a length floor.
    const sel = videoSpan(2);
    expect(guardMenuSelection(route("videoAudioToVideo"), sel).ok).toBe(true);
  });
});

/** The inclusive timeline span (seconds) of a selection's first object — mirrors
 * `prefillSeed.spanDurationSec`, kept local so this test file stays focused on
 * the guard's own span-vs-floor comparison. */
function spanSeconds(sel: TimelineSelection): number {
  const item = sel.selected[0]!;
  return (item.frameEnd - item.frameStart + 1) / (sel.rate / sel.scale);
}

describe("formatMenuGuardNote", () => {
  const notes: MenuGuardNote[] = [
    { noteKind: "multipleSelection" },
    { noteKind: "unsupportedType" },
    { noteKind: "typeMismatch", selected: "audio", required: "video" },
    // 素材（末尾）: the multi-kind variant.
    { noteKind: "typeMismatchAny", selected: "audio", required: ["video", "image"] },
    { noteKind: "useVideoAudioInstead" },
    // 台帳§1-16: the long-form near-neighbor variant.
    { noteKind: "useVideoAudioLongInstead" },
    { noteKind: "videoTooShort", requiredSeconds: V2V_MIN_DURATION_SEC },
    // W0 範囲系. `rangeTooShort` is reserved by W0 and raised by W3, so this
    // format test is currently its ONLY caller — deliberate, so the copy and the
    // union member are proven before W3 wires the fps-dependent judgement.
    { noteKind: "rangeNotSelected" },
    { noteKind: "rangeTooShort", requiredFrames: 25 },
  ];

  it("renders a non-empty message for every note kind, in both languages", () => {
    for (const note of notes) {
      expect(formatMenuGuardNote(note, en.notes).length).toBeGreaterThan(0);
      expect(formatMenuGuardNote(note, ja.notes).length).toBeGreaterThan(0);
    }
  });

  it("fills the mismatch template with localized kind display names (§4-3)", () => {
    const note: MenuGuardNote = { noteKind: "typeMismatch", selected: "audio", required: "video" };
    // Spec example (en): "The selected object is a audio. This action requires a video."
    expect(formatMenuGuardNote(note, en.notes)).toBe(
      "The selected object is a audio. This action requires a video.",
    );
    // ja resolves 音声 / 動画.
    expect(formatMenuGuardNote(note, ja.notes)).toContain("音声");
    expect(formatMenuGuardNote(note, ja.notes)).toContain("動画");
  });

  it("shows the rounded-up required seconds in the too-short note", () => {
    const note: MenuGuardNote = { noteKind: "videoTooShort", requiredSeconds: V2V_MIN_DURATION_SEC };
    // 73/24 ≈ 3.04 -> ceil 4.
    expect(formatMenuGuardNote(note, en.notes)).toContain("4");
    expect(formatMenuGuardNote(note, ja.notes)).toContain("4");
  });

  it("enumerates every accepted kind in the multi-kind mismatch template (素材（末尾）)", () => {
    const note: MenuGuardNote = { noteKind: "typeMismatchAny", selected: "audio", required: ["video", "image"] };
    expect(formatMenuGuardNote(note, en.notes)).toBe(
      "The selected object is a audio. This action requires a video or image.",
    );
    expect(formatMenuGuardNote(note, ja.notes)).toBe(
      "選択されているのは音声です。この操作には動画か画像が必要です。",
    );
  });

  it("shows the required frame count in the range-too-short note (W0/W3)", () => {
    const note: MenuGuardNote = { noteKind: "rangeTooShort", requiredFrames: 25 };
    expect(formatMenuGuardNote(note, en.notes)).toContain("25");
    expect(formatMenuGuardNote(note, ja.notes)).toContain("25");
  });
});

// ---------------------------------------------------------------------------
// W0 (2026-08-09) §4 範囲系ガード — `retakeRange` only.
// ---------------------------------------------------------------------------

describe("guardMenuSelection: range guard (W0)", () => {
  it("refuses retakeRange when no frame range is selected", () => {
    const result = guardMenuSelection(route("retakeRange"), makeSelection({ hasRange: false }));
    expect(result).toEqual({ ok: false, note: { noteKind: "rangeNotSelected" } });
  });

  it("passes retakeRange when a frame range IS selected", () => {
    const result = guardMenuSelection(
      route("retakeRange"),
      makeSelection({ hasRange: true, rangeStart: 24, rangeEnd: 96 }),
    );
    expect(result).toEqual({ ok: true });
  });

  it("does NOT apply the range guard to any other action", () => {
    // Only retakeRange treats the range as its input; outpaintVideo (the other
    // Edit-系 item) and every legacy object item must be unaffected by it.
    const noRange = makeSelection({ hasRange: false });
    expect(guardMenuSelection(route("outpaintVideo"), noRange)).toEqual({ ok: true });
    expect(guardMenuSelection(route("referenceVideo"), noRange)).toEqual({ ok: true });
  });

  it("reports the type mismatch before the missing range (guard order)", () => {
    // A non-video with no range selected fails on kind first — the (c) check
    // runs before (e), so the user is told the more fundamental problem.
    const sel = makeSelection({
      hasRange: false,
      selected: [makeItem(EFFECT_NAME_AUDIO)],
    });
    expect(guardMenuSelection(route("retakeRange"), sel)).toEqual({
      ok: false,
      note: { noteKind: "typeMismatch", selected: "audio", required: "video" },
    });
  });
});

describe("guardMenuSelection — (e2) 範囲が短すぎる（§1-17 Retake、W3 が起こす）", () => {
  // 判定は「生成 fps の上限 60 でも足りない」= 73/60 ≈ 1.2167 秒未満のときだけ。
  // @30fps ならプロジェクト 37 フレーム以上あれば通る（ceil(1.2167 × 30) = 37）。
  it("どの生成 fps でも足りない範囲だけを弾き、必要フレーム数を伝える", () => {
    const sel = makeSelection({ rangeStart: 0, rangeEnd: 34 }); // 35 フレーム = 1.167 秒
    expect(guardMenuSelection(route("retakeRange"), sel)).toEqual({
      ok: false,
      note: { noteKind: "rangeTooShort", requiredFrames: 37 },
    });
  });

  it("境界のすぐ内側（37 フレーム @30fps）は通す", () => {
    const sel = makeSelection({ rangeStart: 0, rangeEnd: 36 }); // 37 フレーム = 1.2333 秒
    expect(guardMenuSelection(route("retakeRange"), sel)).toEqual({ ok: true });
  });

  it("必要フレーム数はプロジェクトの fps で変わる", () => {
    // @60fps: ceil(1.2167 × 60) = 73 プロジェクトフレーム。
    const sel = makeSelection({ rangeStart: 0, rangeEnd: 9, rate: 60, scale: 1 });
    expect(guardMenuSelection(route("retakeRange"), sel)).toEqual({
      ok: false,
      note: { noteKind: "rangeTooShort", requiredFrames: 73 },
    });
  });

  it("rate/scale が解けないときは弾かない（測れないことを理由にブロックしない）", () => {
    const sel = makeSelection({ rangeStart: 0, rangeEnd: 1, rate: 0, scale: 1 });
    expect(guardMenuSelection(route("retakeRange"), sel)).toEqual({ ok: true });
  });

  it("範囲未選択のほうが先に出る（rangeNotSelected が rangeTooShort に優先）", () => {
    const sel = makeSelection({ hasRange: false, rangeStart: 0, rangeEnd: 0 });
    expect(guardMenuSelection(route("retakeRange"), sel)).toEqual({
      ok: false,
      note: { noteKind: "rangeNotSelected" },
    });
  });

  it("他の項目は短い範囲でも影響を受けない", () => {
    const shortRange = makeSelection({ rangeStart: 0, rangeEnd: 1 });
    expect(guardMenuSelection(route("outpaintVideo"), shortRange)).toEqual({ ok: true });
    expect(guardMenuSelection(route("referenceVideo"), shortRange)).toEqual({ ok: true });
  });
});

// ---------------------------------------------------------------------------
// 素材（末尾）(2026-08-15) — the first action that accepts a SET of kinds
// (`requiredKind: ["video", "image"]`), and the first whose §4-5 length floor
// must apply to only one of them.
// ---------------------------------------------------------------------------

describe("requiredKindsOf", () => {
  it("normalizes null to the empty set (the layer-menu exemption)", () => {
    expect(requiredKindsOf(null)).toEqual([]);
  });

  it("wraps a single kind into a one-element set", () => {
    expect(requiredKindsOf("video")).toEqual(["video"]);
    expect(requiredKindsOf("text")).toEqual(["text"]);
  });

  it("passes an array through unchanged", () => {
    expect(requiredKindsOf(["video", "image"])).toEqual(["video", "image"]);
  });
});

describe("guardMenuSelection — endWithThis (素材（末尾）)", () => {
  /** A long-enough video so the (d) length floor never masks a kind result. */
  const longVideo = () => makeSelection({ selected: [makeItem(EFFECT_NAME_VIDEO, 10)] });

  it("accepts a video (first member of the required set)", () => {
    expect(guardMenuSelection(route("endWithThis"), longVideo())).toEqual({ ok: true });
  });

  it("accepts an image (second member of the required set)", () => {
    const sel = makeSelection({ selected: [makeItem(EFFECT_NAME_IMAGE)] });
    expect(guardMenuSelection(route("endWithThis"), sel)).toEqual({ ok: true });
  });

  it("refuses audio with the multi-kind mismatch note naming BOTH accepted kinds", () => {
    const sel = makeSelection({ selected: [makeItem(EFFECT_NAME_AUDIO, 10)] });
    expect(guardMenuSelection(route("endWithThis"), sel)).toEqual({
      ok: false,
      note: { noteKind: "typeMismatchAny", selected: "audio", required: ["video", "image"] },
    });
  });

  it("refuses text the same way", () => {
    const sel = makeSelection({ selected: [makeItem(EFFECT_NAME_TEXT)] });
    const r = guardMenuSelection(route("endWithThis"), sel);
    expect(r.ok).toBe(false);
    if (!r.ok && r.note.noteKind === "typeMismatchAny") {
      expect(r.note.selected).toBe("text");
      expect(r.note.required).toEqual(["video", "image"]);
    } else {
      throw new Error("expected typeMismatchAny");
    }
  });

  it("refuses an unrecognized effectName as unsupportedType (kind check order unchanged)", () => {
    const sel = makeSelection({ selected: [makeItem("図形")] });
    const r = guardMenuSelection(route("endWithThis"), sel);
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.note.noteKind).toBe("unsupportedType");
  });

  it("refuses a multi-object selection before anything else (§4-2)", () => {
    const sel = makeSelection({
      selected: [makeItem(EFFECT_NAME_VIDEO, 10), makeItem(EFFECT_NAME_IMAGE)],
    });
    expect(guardMenuSelection(route("endWithThis"), sel)).toEqual({
      ok: false,
      note: { noteKind: "multipleSelection" },
    });
  });

  // --- §4-5 length floor: VIDEO only ---------------------------------------

  it("窓内モード: has a 9-frame floor — no longer v2's 25 nor V2V's 73", () => {
    expect(lengthFloorForAction("endWithThis")).toBe(END_SOURCE_MIN_DURATION_SEC);
    expect(END_SOURCE_MIN_DURATION_SEC).toBeCloseTo(9 / GEN_FPS, 12);
    // v1's floor was 73 frames (the band was a slider prefilled at 72) and v2's
    // was an owner-picked 25 (a material-derived band needed length to be worth
    // deriving). The anchor is now the fixed 8 whatever the material is, so the
    // only floor left is the technical one — pinned here so a regression to
    // either old number is loud.
    expect(END_SOURCE_MIN_DURATION_SEC).toBeLessThan(V2V_MIN_DURATION_SEC);
    expect(END_SOURCE_MIN_DURATION_SEC).toBeLessThan(ICLORA_MIN_DURATION_SEC);
  });

  it("blocks a video clearly under the floor", () => {
    const sel = makeSelection({
      // Well below even the one-frame measurement margin (see
      // END_SOURCE_DURATION_MARGIN_SEC).
      selected: [makeItem(EFFECT_NAME_VIDEO, END_SOURCE_MIN_DURATION_SEC - 0.2)],
    });
    const r = guardMenuSelection(route("endWithThis"), sel);
    expect(r.ok).toBe(false);
    if (!r.ok && r.note.noteKind === "videoTooShort") {
      // The NOTE quotes the honest floor, not floor-minus-margin.
      expect(r.note.requiredSeconds).toBe(END_SOURCE_MIN_DURATION_SEC);
    } else {
      throw new Error("expected videoTooShort");
    }
  });

  it("passes a video at exactly the floor", () => {
    const sel = makeSelection({
      selected: [makeItem(EFFECT_NAME_VIDEO, END_SOURCE_MIN_DURATION_SEC)],
    });
    expect(guardMenuSelection(route("endWithThis"), sel)).toEqual({ ok: true });
  });

  it("allows one frame of container-measurement slack below the floor", () => {
    // A 9-frame material whose container under-reports by a hair must not be
    // refused destructively at the right click — the form (which measures the
    // SERVER's frame count) is the real arbiter.
    const sel = makeSelection({
      selected: [makeItem(EFFECT_NAME_VIDEO, END_SOURCE_MIN_DURATION_SEC - 0.001)],
    });
    expect(guardMenuSelection(route("endWithThis"), sel)).toEqual({ ok: true });
    expect(lengthMarginForAction("endWithThis")).toBeCloseTo(1 / GEN_FPS, 12);
  });

  it("the slack is scoped to endWithThis — every other action keeps a margin of 0", () => {
    for (const action of ["extendVideo", "referenceVideo", "referenceVideoChain", "imageToVideo"]) {
      expect(lengthMarginForAction(action)).toBe(0);
    }
  });

  it("does NOT apply the floor to an image — a still with a 0 duration passes", () => {
    // The whole point of scoping (d) to `kind === "video"`: a still has no
    // duration to measure, so an image must never be called "too short".
    const zeroDuration = makeSelection({ selected: [makeItem(EFFECT_NAME_IMAGE, 0)] });
    expect(guardMenuSelection(route("endWithThis"), zeroDuration)).toEqual({ ok: true });
    // ...and not even a positive-but-tiny duration (some probes report one for
    // an image) may trip it.
    const tinyDuration = makeSelection({ selected: [makeItem(EFFECT_NAME_IMAGE, 0.04)] });
    expect(guardMenuSelection(route("endWithThis"), tinyDuration)).toEqual({ ok: true });
  });
});
