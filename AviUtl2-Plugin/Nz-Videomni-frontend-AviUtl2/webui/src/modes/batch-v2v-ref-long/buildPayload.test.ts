import { describe, expect, it } from "vitest";
import type { GenerateChainRequest } from "../../api/types";
import type { UseChainFormResult } from "../chained/useChainForm";
import { createInitialState, emptySlot } from "../single/referenceRules";
import type { ReferenceKind, ReferenceRules, ReferenceState } from "../single/referenceRules";
import { buildRowRequest, buildV2vRefLongTemplate, rowUploadTargets, uploadKey, uploadKindOf } from "./buildPayload";
import type { RowRequestSettings } from "./buildPayload";
import type { RowPlan, V2vRefLongChainSource } from "./types";

// Compile-time: the Chain form's result satisfies what the batch reads.
type ChainFormSatisfiesSource = UseChainFormResult extends V2vRefLongChainSource ? true : never;
const chainFormSatisfiesSource: ChainFormSatisfiesSource = true;
void chainFormSatisfiesSource;

const RULES: ReferenceRules = {
  maxImages: 9,
  maxVideos: 3,
  maxAudios: 3,
  maxTotal: 12,
  videoTotalSec: 15,
  audioTotalSec: 15,
  clipMinSec: 1,
};

const SETTINGS: RowRequestSettings = { mainPromptMode: "add", contextFrames: 22, rules: RULES, role: "ref2va" };

function base(): GenerateChainRequest {
  return {
    prompt: "shared",
    width: 768,
    height: 512,
    frame_rate: 24,
    seed: 7,
    clips: [
      { prompt: "c1", num_frames: 49, conditioning_images: [{ image_id: "start", frame_idx: 0, strength: 1 }] },
      { num_frames: 49 },
    ],
    source_video: { video_id: "screen-src", context_frames: 22 },
    source_audio: { audio_id: "long" },
    reference_video_id: "ic",
    conditioning_attention_strength: 0.5,
    reference_video_strength: 1,
    end_source: { video_id: "end", context_frames: 8, strength: 1 },
    loras: [{ name: "style", strength: 0.8 }],
  };
}

function refsWith(cards: { kind: ReferenceKind; slot: number; path: string; durationSec?: number | null }[]): ReferenceState {
  const state = { ...createInitialState(RULES), enabled: true };
  const slots = { picture: [...state.slots.picture], video: [...state.slots.video], audio: [...state.slots.audio] };
  cards.forEach((c, i) => {
    slots[c.kind][c.slot - 1] = {
      ...emptySlot(c.kind, c.slot),
      status: "ready",
      fileName: c.path.split("\\").pop() ?? c.path,
      filePath: c.path,
      durationSec: c.durationSec ?? null,
      addedAt: i + 1,
    };
  });
  return { ...state, slots };
}

function plan(patch: Partial<RowPlan> = {}): RowPlan {
  return {
    index: 0,
    sourcePath: "D:\\v\\a.mp4",
    mainPrompt: "",
    clips: [
      { promptMode: "add", prompt: "", refs: refsWith([{ kind: "picture", slot: 1, path: "D:\\p\\1.png" }]) },
      { promptMode: "add", prompt: "", refs: createInitialState(RULES) },
    ],
    outFileName: "a.mp4",
    ...patch,
  };
}

function idsFor(p: RowPlan): Map<string, string> {
  return new Map(rowUploadTargets(p, RULES).map((t, i) => [t.key, `id${i}`]));
}

describe("buildV2vRefLongTemplate", () => {
  it("drops the screen's material and keeps the settings", () => {
    const template = buildV2vRefLongTemplate(base());
    expect(template).not.toHaveProperty("source_video");
    expect(template).not.toHaveProperty("source_audio");
    expect(template).not.toHaveProperty("reference_video_id");
    expect(template).not.toHaveProperty("conditioning_attention_strength");
    expect(template).not.toHaveProperty("reference_video_strength");
    expect(template).not.toHaveProperty("end_source");
    expect(template.clips[0]).not.toHaveProperty("conditioning_images");
    expect(template.loras).toEqual([{ name: "style", strength: 0.8 }]);
    expect(template.seed).toBe(7);
  });
});

describe("upload keys", () => {
  it("map the card kind to the upload kind", () => {
    expect(uploadKindOf("picture")).toBe("image");
    expect(uploadKindOf("video")).toBe("video");
    expect(uploadKindOf("audio")).toBe("audio");
  });

  it("carry the kind: the same file as Video and as Audio is two uploads", () => {
    expect(uploadKey("video", "D:\\x.mp4")).not.toBe(uploadKey("audio", "D:\\x.mp4"));
    const p = plan({
      sourcePath: "D:\\x.mp4",
      clips: [
        {
          promptMode: "add",
          prompt: "",
          refs: refsWith([
            { kind: "video", slot: 1, path: "D:\\x.mp4" },
            { kind: "audio", slot: 1, path: "D:\\x.mp4" },
          ]),
        },
        { promptMode: "add", prompt: "", refs: refsWith([{ kind: "video", slot: 1, path: "D:\\x.mp4" }]) },
      ],
    });
    expect(rowUploadTargets(p, RULES).map((t) => t.key)).toEqual([uploadKey("video", "D:\\x.mp4"), uploadKey("audio", "D:\\x.mp4")]);
  });
});

describe("buildRowRequest", () => {
  it("composes the main prompt onto the shared one", () => {
    const template = buildV2vRefLongTemplate(base());
    const p = plan({ mainPrompt: "row" });
    expect(buildRowRequest(template, p, idsFor(p), SETTINGS).prompt).toBe("shared row");
    expect(buildRowRequest(template, p, idsFor(p), { ...SETTINGS, mainPromptMode: "replace" }).prompt).toBe("row");
    expect(buildRowRequest(template, plan(), idsFor(plan()), SETTINGS).prompt).toBe("shared");
  });

  it("composes each clip prompt onto the Chain card's, and leaves the key out when empty", () => {
    const template = buildV2vRefLongTemplate(base());
    const p = plan({
      clips: [
        { promptMode: "replace", prompt: "mine", refs: createInitialState(RULES) },
        { promptMode: "add", prompt: "", refs: createInitialState(RULES) },
      ],
    });
    const req = buildRowRequest(template, p, idsFor(p), SETTINGS);
    expect(req.clips[0]?.prompt).toBe("mine");
    expect(req.clips[1]).not.toHaveProperty("prompt");

    const added = plan({
      clips: [
        { promptMode: "add", prompt: "more", refs: createInitialState(RULES) },
        { promptMode: "add", prompt: "two", refs: createInitialState(RULES) },
      ],
    });
    const req2 = buildRowRequest(template, added, idsFor(added), SETTINGS);
    expect(req2.clips[0]?.prompt).toBe("c1 more");
    expect(req2.clips[1]?.prompt).toBe("two");
    expect(req2.clips[0]?.num_frames).toBe(49);
  });

  it("fills references from the cards with their upload ids (max_sec on a truncated card)", () => {
    const template = buildV2vRefLongTemplate(base());
    const p = plan({
      clips: [
        {
          promptMode: "add",
          prompt: "",
          refs: refsWith([
            { kind: "picture", slot: 1, path: "D:\\p\\1.png" },
            { kind: "video", slot: 1, path: "D:\\v1.mp4", durationSec: 10 },
            { kind: "video", slot: 2, path: "D:\\v2.mp4", durationSec: 8 },
          ]),
        },
        { promptMode: "add", prompt: "", refs: createInitialState(RULES) },
      ],
    });
    const ids = new Map([
      [uploadKey("video", "D:\\v\\a.mp4"), "src"],
      [uploadKey("image", "D:\\p\\1.png"), "img1"],
      [uploadKey("video", "D:\\v1.mp4"), "vid1"],
      [uploadKey("video", "D:\\v2.mp4"), "vid2"],
    ]);
    const req = buildRowRequest(template, p, ids, SETTINGS);
    expect(req.clips[0]?.references).toEqual([
      { kind: "picture", id: "img1", slot: 1 },
      { kind: "video", id: "vid1", slot: 1 },
      { kind: "video", id: "vid2", slot: 2, max_sec: 5 },
    ]);
    expect(req.clips[1]).not.toHaveProperty("references");
  });

  it("sends no references on FL2VA", () => {
    const template = buildV2vRefLongTemplate(base());
    const p = plan();
    const req = buildRowRequest(template, p, idsFor(p), { ...SETTINGS, role: "fl2va" });
    expect(req.clips[0]).not.toHaveProperty("references");
  });

  it("adds source_video with context_frames 22 when the row has an opening video", () => {
    const template = buildV2vRefLongTemplate(base());
    const p = plan();
    const ids = idsFor(p);
    const req = buildRowRequest(template, p, ids, SETTINGS);
    expect(req.source_video).toEqual({ video_id: ids.get(uploadKey("video", "D:\\v\\a.mp4")), context_frames: 22 });
    const none = plan({ sourcePath: null });
    expect(buildRowRequest(template, none, idsFor(none), SETTINGS)).not.toHaveProperty("source_video");
  });

  it("greyed cards are not uploaded and need no upload id", () => {
    const template = buildV2vRefLongTemplate(base());
    const p = plan({
      clips: [
        {
          promptMode: "add",
          prompt: "",
          refs: refsWith([
            { kind: "video", slot: 1, path: "D:\\long.mp4", durationSec: 16 },
            { kind: "video", slot: 2, path: "D:\\grey.mp4", durationSec: 3 },
          ]),
        },
        { promptMode: "add", prompt: "", refs: createInitialState(RULES) },
      ],
    });
    const targets = rowUploadTargets(p, RULES).map((t) => t.key);
    expect(targets).toEqual([uploadKey("video", "D:\\v\\a.mp4"), uploadKey("video", "D:\\long.mp4")]);
    const req = buildRowRequest(template, p, idsFor(p), SETTINGS);
    expect(req.clips[0]?.references).toEqual([{ kind: "video", id: "id1", slot: 1, max_sec: 15 }]);
  });

  it("drops the template clips' own references and source_audio", () => {
    const withClipMaterial = buildV2vRefLongTemplate(base());
    const template: GenerateChainRequest = {
      ...withClipMaterial,
      clips: withClipMaterial.clips.map((clip) => ({
        ...clip,
        references: [{ kind: "picture", id: "screen", slot: 1 }],
        source_audio: { audio_id: "screen-a", strength: 0.5, offset_frames: 0 },
      })),
    };
    const p = plan({ sourcePath: null, clips: [] });
    const req = buildRowRequest(template, p, idsFor(p), SETTINGS);
    for (const clip of req.clips) {
      expect(clip).not.toHaveProperty("references");
      expect(clip).not.toHaveProperty("source_audio");
    }
  });

  it("throws when an upload id is missing", () => {
    const template = buildV2vRefLongTemplate(base());
    expect(() => buildRowRequest(template, plan(), new Map(), SETTINGS)).toThrow();
  });

  it("never mutates the template", () => {
    const template = buildV2vRefLongTemplate(base());
    const before = JSON.stringify(template);
    const p = plan({ mainPrompt: "x" });
    buildRowRequest(template, p, idsFor(p), SETTINGS);
    buildRowRequest(template, p, idsFor(p), SETTINGS);
    expect(JSON.stringify(template)).toBe(before);
  });
});
