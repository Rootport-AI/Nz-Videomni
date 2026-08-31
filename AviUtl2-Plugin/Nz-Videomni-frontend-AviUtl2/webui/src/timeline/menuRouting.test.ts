import { describe, expect, it } from "vitest";
import {
  MENU_ROUTING_TABLE,
  knownMenuActions,
  routeMenuAction,
  targetModeForIntent,
} from "./menuRouting";
import type { MenuPlacement, MenuRequiredKind, MenuTargetMode } from "./menuRouting";

/** The full contract from the right-click redesign spec
 * (Docs/RIGHTCLICK_REDESIGN_SPEC.md §3-4 object menu / §3-6 layer menu,
 * mapping §7-1, kind/placement §4 / §5-9). Each row asserts the exact routing
 * decision AND the I6 metadata, so a change to the table (or a drift from
 * `plugin.cpp`'s identifiers) fails loudly. */
const EXPECTED: Array<{
  action: string;
  targetMode: MenuTargetMode;
  intent: string;
  needsSelection: boolean;
  requiredKind: MenuRequiredKind;
  placement: MenuPlacement;
}> = [
  // Object-menu commands (#1-#8 + the §1-16 long-a2v pair), in spec order. All
  // needsSelection.
  // Metadata per §3-4/§5-9: #1(video,A) #2(video,B) #3(video,B) #4(image,A)
  // #5(image,branch=null) #6(image,A) #7(audio,B) #8(text,B).
  { action: "extendVideo", targetMode: "chained", intent: "extend-video", needsSelection: true, requiredKind: "video", placement: "A" },
  { action: "referenceVideo", targetMode: "single", intent: "reference-video", needsSelection: true, requiredKind: "video", placement: "B" },
  // 台帳§1-15 W4 (2026-08-11): the Chain-targeted IC-LoRA sibling of #2. Same
  // requiredKind / placement as the item it mirrors; only the target mode
  // (chained) and the intent differ — exactly like the §1-16 pair below.
  { action: "referenceVideoChain", targetMode: "chained", intent: "reference-video-chain", needsSelection: true, requiredKind: "video", placement: "B" },
  { action: "videoAudioToVideo", targetMode: "single", intent: "video-audio-to-video", needsSelection: true, requiredKind: "video", placement: "B" },
  // 台帳§1-16 長尺A2V (2026-08-10): the two Chain-targeted a2v siblings. Same
  // requiredKind / placement as the #3/#7 items they mirror; only the target
  // mode (chained) and the intent differ.
  { action: "videoAudioToLongA2v", targetMode: "chained", intent: "video-audio-to-long-a2v", needsSelection: true, requiredKind: "video", placement: "B" },
  { action: "audioToLongA2v", targetMode: "chained", intent: "audio-to-long-a2v", needsSelection: true, requiredKind: "audio", placement: "B" },
  { action: "imageToVideo", targetMode: "single", intent: "image-to-video", needsSelection: true, requiredKind: "image", placement: "A" },
  { action: "addImageKeyframe", targetMode: "single", intent: "add-image-keyframe", needsSelection: true, requiredKind: "image", placement: null },
  { action: "imageToClipChain", targetMode: "chained", intent: "image-to-clip-chain", needsSelection: true, requiredKind: "image", placement: "A" },
  // 素材（末尾）(2026-08-15): the mirror of #1 `extendVideo`, and the FIRST entry
  // whose requiredKind is a SET — it accepts a video or an image. The row-level
  // assertion below uses `toEqual`, so the array compares structurally.
  { action: "endWithThis", targetMode: "chained", intent: "end-with-this", needsSelection: true, requiredKind: ["video", "image"], placement: "E" },
  { action: "audioToVideo", targetMode: "single", intent: "audio-to-video", needsSelection: true, requiredKind: "audio", placement: "B" },
  { action: "appendText", targetMode: "single", intent: "append-text", needsSelection: true, requiredKind: "text", placement: "B" },
  // W2 ⬇ object quick-insert: position/type-check exempt (requiredKind null),
  // no placement (early-return replace-insert, no provisional reserved).
  { action: "insertProvisionalResult", targetMode: "single", intent: "insert-provisional-result", needsSelection: false, requiredKind: null, placement: null },
  // W0 Edit-系 (2026-08-09). Both require a video selection and land on the
  // Edit tab. outpaintVideo places nothing (placement null, MenuPlacement doc
  // meaning 3 — W2 owns the output geometry); retakeRange uses系統 D (選択範囲
  // と頭を揃える), which `placementParams` maps onto native's B.
  { action: "outpaintVideo", targetMode: "edit", intent: "outpaint", needsSelection: true, requiredKind: "video", placement: null },
  { action: "retakeRange", targetMode: "edit", intent: "retake", needsSelection: true, requiredKind: "video", placement: "D" },
  // Layer-menu commands (#9-#12). needsSelection false (position-based),
  // requiredKind null (exempt from the type check). #9/#10/#12 placement C
  // (cursor); #11 branches on the live keyframe count at append time (null).
  { action: "textToVideoHere", targetMode: "single", intent: "text-to-video", needsSelection: false, requiredKind: null, placement: "C" },
  { action: "imageFromCurrentFrame", targetMode: "single", intent: "image-from-frame", needsSelection: false, requiredKind: null, placement: "C" },
  { action: "addCurrentFrameAsKeyframe", targetMode: "single", intent: "add-current-frame-keyframe", needsSelection: false, requiredKind: null, placement: null },
  { action: "currentFrameToClipChain", targetMode: "chained", intent: "current-frame-to-clip-chain", needsSelection: false, requiredKind: null, placement: "C" },
  // W3 ⬇ layer quick-insert here: position-based, no provisional (plain insert
  // at the cursor), so requiredKind and placement are both null.
  { action: "insertLatestResultHere", targetMode: "single", intent: "insert-latest-result", needsSelection: false, requiredKind: null, placement: null },
];

describe("routeMenuAction", () => {
  it.each(EXPECTED)(
    "routes $action -> $targetMode / $intent (needsSelection=$needsSelection, kind=$requiredKind, placement=$placement)",
    ({ action, targetMode, intent, needsSelection, requiredKind, placement }) => {
      const route = routeMenuAction(action);
      expect(route).not.toBeNull();
      expect(route).toEqual({ action, targetMode, intent, needsSelection, requiredKind, placement });
    },
  );

  it("returns null for an unknown / not-yet-supported action", () => {
    expect(routeMenuAction("totallyBogusAction")).toBeNull();
    expect(routeMenuAction("")).toBeNull();
    // Retired actions from the骨組み table must no longer resolve.
    expect(routeMenuAction("regenerate")).toBeNull();
    expect(routeMenuAction("replace")).toBeNull();
    expect(routeMenuAction("sendToChain")).toBeNull();
    expect(routeMenuAction("fillGap")).toBeNull();
    expect(routeMenuAction("generateFromSelection")).toBeNull();
    expect(routeMenuAction("generateAtCursor")).toBeNull();
  });

  it("covers every action in the table and nothing extra", () => {
    // Guards against silently adding a table entry without a matching
    // assertion above (and vice-versa).
    expect(knownMenuActions().sort()).toEqual(EXPECTED.map((e) => e.action).sort());
  });

  it("exposes exactly the twenty redesign actions (15 object + 5 layer)", () => {
    // 素材（末尾）(2026-08-15) added the 15th object item, `endWithThis`.
    expect(knownMenuActions()).toHaveLength(20);
  });
});

describe("MENU_ROUTING_TABLE invariants", () => {
  it("only ever targets one of the known modes", () => {
    const allowed: MenuTargetMode[] = ["single", "chained", "edit", "inventory"];
    for (const info of Object.values(MENU_ROUTING_TABLE)) {
      expect(allowed).toContain(info.targetMode);
    }
  });

  it("no redesign action targets library (regenerate/replace retired)", () => {
    for (const info of Object.values(MENU_ROUTING_TABLE)) {
      expect(info.targetMode).not.toBe("inventory");
    }
  });

  it("uses a unique, non-empty intent per action", () => {
    const intents = Object.values(MENU_ROUTING_TABLE).map((i) => i.intent);
    for (const intent of intents) expect(intent.length).toBeGreaterThan(0);
    expect(new Set(intents).size).toBe(intents.length);
  });

  it("marks the four layer commands position-based and the ten object commands selection-based", () => {
    // Layer/cursor commands generate at a position, so must not require a
    // selected object.
    expect(MENU_ROUTING_TABLE.textToVideoHere?.needsSelection).toBe(false);
    expect(MENU_ROUTING_TABLE.imageFromCurrentFrame?.needsSelection).toBe(false);
    expect(MENU_ROUTING_TABLE.addCurrentFrameAsKeyframe?.needsSelection).toBe(false);
    expect(MENU_ROUTING_TABLE.currentFrameToClipChain?.needsSelection).toBe(false);
    // Every object-menu command operates on the selection.
    for (const action of [
      "extendVideo",
      "referenceVideo",
      "referenceVideoChain",
      "videoAudioToVideo",
      "imageToVideo",
      "addImageKeyframe",
      "imageToClipChain",
      "audioToVideo",
      "appendText",
      // W0 Edit-系.
      "outpaintVideo",
      "retakeRange",
      // 素材（末尾）: also an object command (it acts on the selected material).
      "endWithThis",
    ]) {
      expect(MENU_ROUTING_TABLE[action]?.needsSelection).toBe(true);
    }
  });

  it("keeps the Chain-routed items (extendVideo, imageToClipChain, currentFrameToClipChain) on chain and A2V on create", () => {
    expect(MENU_ROUTING_TABLE.extendVideo?.targetMode).toBe("chained");
    expect(MENU_ROUTING_TABLE.imageToClipChain?.targetMode).toBe("chained");
    // W5: the current-frame clip-chain layer item also lands on Chain.
    expect(MENU_ROUTING_TABLE.currentFrameToClipChain?.targetMode).toBe("chained");
    // audioToVideo was corrected from chain -> create in the redesign (§7-1).
    expect(MENU_ROUTING_TABLE.audioToVideo?.targetMode).toBe("single");
    expect(MENU_ROUTING_TABLE.videoAudioToVideo?.targetMode).toBe("single");
  });

  it("gives every object-menu command a concrete requiredKind and the layer pair null (§4/§3-6)", () => {
    // The eight object commands must name a required kind (drives §4-3).
    const objectKinds: Record<string, MenuRequiredKind> = {
      extendVideo: "video",
      referenceVideo: "video",
      referenceVideoChain: "video",
      videoAudioToVideo: "video",
      imageToVideo: "image",
      addImageKeyframe: "image",
      imageToClipChain: "image",
      audioToVideo: "audio",
      appendText: "text",
      // W0 Edit-系: both operate on a video object.
      outpaintVideo: "video",
      retakeRange: "video",
    };
    for (const [action, kind] of Object.entries(objectKinds)) {
      expect(MENU_ROUTING_TABLE[action]?.requiredKind).toBe(kind);
    }
    // 素材（末尾）(2026-08-15): the one entry whose requiredKind is a SET. It MUST
    // be asserted with `toEqual` — `toBe` compares by reference and would fail
    // on a structurally identical array.
    expect(MENU_ROUTING_TABLE.endWithThis?.requiredKind).toEqual(["video", "image"]);
    // ...and it is a real check, not the layer-menu exemption: an array means
    // "one of these", null means "no check at all".
    expect(MENU_ROUTING_TABLE.endWithThis?.requiredKind).not.toBeNull();
    // The four layer commands are exempt from the type check.
    expect(MENU_ROUTING_TABLE.textToVideoHere?.requiredKind).toBeNull();
    expect(MENU_ROUTING_TABLE.imageFromCurrentFrame?.requiredKind).toBeNull();
    expect(MENU_ROUTING_TABLE.addCurrentFrameAsKeyframe?.requiredKind).toBeNull();
    expect(MENU_ROUTING_TABLE.currentFrameToClipChain?.requiredKind).toBeNull();
  });

  it("assigns each action its §5-9 placement系統 (A/B/C/D/E, with #5 branch=null)", () => {
    const placements: Record<string, MenuPlacement> = {
      extendVideo: "A",
      referenceVideo: "B",
      referenceVideoChain: "B",
      videoAudioToVideo: "B",
      imageToVideo: "A",
      addImageKeyframe: null, // #5 branches on live keyframe count at append time
      imageToClipChain: "A",
      audioToVideo: "B",
      appendText: "B",
      textToVideoHere: "C",
      imageFromCurrentFrame: "C",
      addCurrentFrameAsKeyframe: null, // #11 branches on live keyframe count at append time
      currentFrameToClipChain: "C",
      outpaintVideo: null, // W0: reaches Step 8 but reserves nothing (W2 decides)
      retakeRange: "D", // W0: head-aligned with the SELECTED RANGE (mapped to B)
      // 素材（末尾）v2: 末尾合わせ系統 E。native には無いので
      // `provisionalReservation.placementParams` が B へ写像し、呼び出し側が
      // シフト済みの開始フレームを入れる（系統D と同じ規律）。
      endWithThis: "E",
    };
    for (const [action, placement] of Object.entries(placements)) {
      expect(MENU_ROUTING_TABLE[action]?.placement).toBe(placement);
    }
  });

  it("only ever uses the six A/B/C/D/E/null placement values", () => {
    const allowed: MenuPlacement[] = ["A", "B", "C", "D", "E", null];
    for (const info of Object.values(MENU_ROUTING_TABLE)) {
      expect(allowed).toContain(info.placement);
    }
  });
});

// 2026-08-31: the reverse direction, for the code downstream of routing that
// only ever sees the `intent` string (`timeline/prefillSeed.ts` decides from it
// whether the SINGLE screen's comfort ceiling applies to a DURATION seed).
describe("targetModeForIntent", () => {
  it("is the exact inverse of the table for every row", () => {
    for (const info of Object.values(MENU_ROUTING_TABLE)) {
      expect(targetModeForIntent(info.intent)).toBe(info.targetMode);
    }
  });

  it("is well-defined: no intent string appears on two rows", () => {
    const intents = Object.values(MENU_ROUTING_TABLE).map((info) => info.intent);
    expect(new Set(intents).size).toBe(intents.length);
  });

  it("returns null for an intent the table has no row for", () => {
    expect(targetModeForIntent("not-a-real-intent")).toBeNull();
    expect(targetModeForIntent("")).toBeNull();
    // An ACTION name is not an intent — passing one in must not accidentally
    // resolve (the two vocabularies are deliberately separate).
    expect(targetModeForIntent("referenceVideo")).toBeNull();
  });

  it("keeps the Single-系 DURATION flows on single, and retake/end-with-this off it", () => {
    // The four intents that actually carry a DURATION policy...
    expect(targetModeForIntent("reference-video")).toBe("single"); // #2
    expect(targetModeForIntent("video-audio-to-video")).toBe("single"); // #3
    expect(targetModeForIntent("image-to-video")).toBe("single"); // #4
    expect(targetModeForIntent("audio-to-video")).toBe("single"); // #7
    // ...and the two that must NOT pick up Create's per-clip comfort ceiling.
    expect(targetModeForIntent("retake")).toBe("edit");
    expect(targetModeForIntent("end-with-this")).toBe("chained");
  });
});
