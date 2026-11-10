import { describe, expect, it } from "vitest";
import type { BaseModelBlock, ModelsResponse } from "../api/types";
import { createMockBridge, createMockFs } from "./mockBridge";
import type { MockBridgeOptions } from "./mockBridge";

// MiniMax H3 第 2 弾 (2026-10-17): the fixture server's third base model —
// its `constraints`, its feature names, its own install knob, and the
// `POST /generate` checks it judges against the ACTIVE base model's rules.

type Bridge = ReturnType<typeof createMockBridge>;

async function listModels(bridge: Bridge): Promise<BaseModelBlock[]> {
  const res = await bridge.request("backend.request", { method: "GET", path: "/api/v1/models" });
  return ((res.body as ModelsResponse).base_models ?? []) as BaseModelBlock[];
}

async function load(bridge: Bridge, id: string) {
  return bridge.request("backend.request", {
    method: "POST",
    path: "/api/v1/pipeline/load",
    body: { base_model: id },
  });
}

function generate(bridge: Bridge, body: Record<string, unknown>) {
  return bridge.request("backend.request", { method: "POST", path: "/api/v1/generate", body });
}

const BODY = { prompt: "a cat", width: 1280, height: 704, num_frames: 124, frame_rate: 24, seed: 1 };

describe("mockBridge — MiniMax H3", () => {
  it("publishes MiniMax H3 with its constraints and feature names; LTX gets the LTX rules", async () => {
    const blocks = await listModels(createMockBridge({ delayMs: 0 }));
    const h3 = blocks.find((b) => b.id === "MiniMaxH3");
    expect(h3).toMatchObject({ display_name: "MiniMax H3", engine_family: "h3" });
    expect(h3?.constraints).toMatchObject({
      num_frames: { step: 17, offset: 5, min: 22, max: 481 },
      dimension_multiple: 32,
      frame_rate: { min: 24, max: 24 },
      keyframes: { step: 1, offset: 0 },
    });
    expect(h3?.constraints?.sampling?.num_inference_steps.default).toBe(20);
    expect(h3?.unsupported_features).toEqual(
      expect.arrayContaining(["reference_video", "nag", "keyframe_strength"]),
    );
    // §1-123: H3 runs a Retake now — `retake` left, and `constraints.retake`
    // carries the middle-audio Strength slider. LTX publishes no `retake` block.
    expect(h3?.unsupported_features).not.toContain("retake");
    expect(h3?.constraints?.retake).toEqual({ audio_strength: { min: 0, max: 1, step: 0.05, default: 0.5 } });
    // §1-125: H3 runs an Outpainting now — `outpaint` left, and
    // `constraints.outpaint` carries the source-audio Strength and mask-blur
    // sliders. LTX publishes no `outpaint` block.
    expect(h3?.unsupported_features).not.toContain("outpaint");
    expect(h3?.constraints?.outpaint).toEqual({
      audio_strength: { min: 0, max: 1, step: 0.05, default: 0.5 },
      blend_dilation_stage1: { min: 0, max: 15, step: 1, default: 1 },
    });
    // §1-106: H3 chains now — `chain` left, and the chain material it cannot
    // take is published instead; `constraints.chain` carries the fixed seam.
    // 台帳 §1-121: V2V and the opening image opened, so `v2v` and
    // `chain_start_image` left as well.
    expect(h3?.unsupported_features).not.toContain("chain");
    expect(h3?.unsupported_features).toEqual(expect.arrayContaining(["end_source", "a2v"]));
    expect(h3?.unsupported_features).not.toContain("v2v");
    expect(h3?.unsupported_features).not.toContain("chain_start_image");
    // §1-120: the per-clip A2V of the chain (its Strength slider).
    expect(h3?.constraints?.chain).toEqual({
      overlap_frames: 22,
      source_audio: { strength: { min: 0, max: 1, step: 0.05, default: 0.5 } },
    });
    expect(h3?.unsupported_features).not.toContain("multi_keyframes");
    const ltx = blocks.find((b) => b.id === "LTX23");
    expect(ltx?.constraints).toMatchObject({
      num_frames: { step: 8, offset: 1 },
      sampling: null,
      keyframes: { step: 8, offset: 1 },
    });
    expect(ltx?.constraints?.chain).toBeUndefined();
    expect(ltx?.constraints?.retake).toBeUndefined();
    expect(ltx?.constraints?.outpaint).toBeUndefined();
    expect(ltx?.unsupported_features).not.toContain("chain_start_image");
    // 台帳 §1-101: VRAM 16GB mode is H3's own — H3 does NOT publish the name,
    // both LTX engines do (which hides the Settings row there).
    expect(h3?.unsupported_features).not.toContain("vram16_mode");
    expect(ltx?.unsupported_features).toContain("vram16_mode");
    expect(blocks.find((b) => b.id === "LTX25")?.unsupported_features).toContain("vram16_mode");
    // 台帳 §1-100: the fused GGUF kernel is the other way round — H3 (no GGUF
    // weights) publishes the name, neither LTX engine does.
    expect(h3?.unsupported_features).toContain("fused_gguf_dequant_kernel");
    expect(ltx?.unsupported_features).not.toContain("fused_gguf_dequant_kernel");
    expect(blocks.find((b) => b.id === "LTX25")?.unsupported_features).not.toContain("fused_gguf_dequant_kernel");
  });

  it("follows its OWN install knob, not LTX 2.5's", async () => {
    const options: MockBridgeOptions = { delayMs: 0, ltx25Install: "full", h3Install: "none" };
    const blocks = await listModels(createMockBridge(options));
    expect(blocks.find((b) => b.id === "LTX25")).toMatchObject({ installed: true });
    expect(blocks.find((b) => b.id === "MiniMaxH3")).toMatchObject({ present: false, installed: false });
  });

  it("judges POST /generate by the ACTIVE base model's rules", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    // On LTX 2.3 a 17n+5 length and a 32-only width are refused, as before.
    expect((await generate(bridge, BODY)).status).toBe(422);

    expect((await load(bridge, "MiniMaxH3")).status).toBe(200);
    // On MiniMax H3: 8n+1 and a non-24 fps are refused…
    expect((await generate(bridge, { ...BODY, num_frames: 121 })).status).toBe(422);
    expect((await generate(bridge, { ...BODY, frame_rate: 30 })).status).toBe(422);
    expect((await generate(bridge, { ...BODY, num_inference_steps: 99 })).status).toBe(422);
    // …and its own grid, size multiple and sampling values are accepted.
    const ok = await generate(bridge, { ...BODY, num_inference_steps: 20, shift_video: 12, shift_audio: 3 });
    expect(ok.status).toBe(202);
  });

  // §1-106: H3 chains — a plain chain is accepted, and only the
  // material it declares unsupported is refused, field by field.
  function chain(bridge: Bridge, extra: Record<string, unknown> = {}, clips: unknown[] = [{ num_frames: 124 }, { num_frames: 124 }]) {
    return bridge.request("backend.request", {
      method: "POST",
      path: "/api/v1/generate/chain",
      body: { prompt: "a cat", width: 1280, height: 704, frame_rate: 24, seed: 1, clips, ...extra },
    });
  }

  it("accepts a text-only POST /generate/chain while MiniMax H3 is loaded", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    expect((await load(bridge, "MiniMaxH3")).status).toBe(200);
    const res = await chain(bridge, { num_inference_steps: 20, shift_video: 12, shift_audio: 3 });
    expect(res.status).toBe(202);
  });

  it("refuses the chain material MiniMax H3 declares (an image strength, end source, A2V)", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    expect((await load(bridge, "MiniMaxH3")).status).toBe(200);
    const cases: Array<[string, Awaited<ReturnType<typeof chain>>]> = [
      [
        "keyframe_strength",
        await chain(bridge, {}, [
          { num_frames: 124, conditioning_images: [{ image_id: "img", frame_idx: 0, strength: 1 }] },
          { num_frames: 124 },
        ]),
      ],
      ["end_source", await chain(bridge, { end_source: { image_id: "img", context_frames: 8, strength: 1 } })],
      ["a2v", await chain(bridge, { source_audio: { audio_id: "aud" } })],
    ];
    for (const [feature, res] of cases) {
      expect(res.status, feature).toBe(422);
      const error = (res.body as { error: { code: string; message: string } }).error;
      expect(error.code, feature).toBe("FEATURE_UNSUPPORTED");
      expect(error.message, feature).toContain(`'${feature}'`);
    }
  });

  // 台帳 §1-121: V2V (with the fixed 22-frame context, on one clip) and an
  // opening image with no strength are accepted on H3.
  it("accepts a V2V chain and a strength-less opening image while MiniMax H3 is loaded", async () => {
    const v2v = createMockBridge({ delayMs: 0 });
    expect((await load(v2v, "MiniMaxH3")).status).toBe(200);
    const res = await chain(v2v, { source_video: { video_id: "vid", context_frames: 22 } }, [{ num_frames: 39 }]);
    expect(res.status).toBe(202);

    const i2v = createMockBridge({ delayMs: 0 });
    expect((await load(i2v, "MiniMaxH3")).status).toBe(200);
    const res2 = await chain(i2v, {}, [
      { num_frames: 124, conditioning_images: [{ image_id: "img", frame_idx: 0 }] },
      { num_frames: 124 },
    ]);
    expect(res2.status).toBe(202);
  });

  // §1-123: a one-clip Retake with `audio_strength` (and no seam / Stage-2 /
  // `regenerate_audio`) is accepted on H3, as an LTX Retake is.
  it("accepts a Retake chain with `audio_strength` while MiniMax H3 is loaded", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    expect((await load(bridge, "MiniMaxH3")).status).toBe(200);
    const res = await chain(
      bridge,
      {
        num_inference_steps: 20,
        shift_video: 12,
        shift_audio: 3,
        retake: { video_id: "vid", window_start_sec: 1.5, audio_strength: 0.5 },
      },
      [{ num_frames: 124 }],
    );
    expect(res.status).toBe(202);
  });

  // §1-125: the H3 Outpainting body (a 32-multiple canvas, 17n+5 frames, 24
  // fps, `outpaint.source_video_id` and no `reference_video_id` / control LoRA)
  // is accepted on `/generate`.
  it("accepts an Outpainting with `outpaint.source_video_id` while MiniMax H3 is loaded", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    expect((await load(bridge, "MiniMaxH3")).status).toBe(200);
    const res = await generate(bridge, {
      prompt: "wider",
      width: 1344,
      height: 768,
      num_frames: 124,
      frame_rate: 24,
      seed: -1,
      num_inference_steps: 20,
      shift_video: 12,
      shift_audio: 3,
      outpaint: {
        pad_left: 12,
        pad_right: 20,
        pad_top: 0,
        pad_bottom: 48,
        source_video_id: "vid",
        audio_strength: 0.5,
        blend_dilation_stage1: 1,
      },
    });
    expect(res.status).toBe(202);
  });

  it("does not refuse an opening image with a strength on LTX 2.3 (the predicate is H3's declared list)", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    const res = await chain(bridge, {}, [
      { num_frames: 121, conditioning_images: [{ image_id: "img", frame_idx: 0, strength: 1 }] },
      { num_frames: 121 },
    ]);
    expect(res.status).toBe(202);
  });

  // §1-127: `nag` on H3 closes the NAG METHOD only — a VSF negative prompt
  // is accepted on `/generate` and `/generate/chain` alike, and the NAG
  // method (named, or left to the server's default) is refused as `nag`.
  describe("§1-127 negative prompt on MiniMax H3: VSF only", () => {
    const NEG = { nag_enabled: true, negative_prompt: "blurry" };
    const H3_SAMPLING = { num_inference_steps: 20, shift_video: 12, shift_audio: 3 };

    async function h3Bridge(): Promise<Bridge> {
      const bridge = createMockBridge({ delayMs: 0 });
      expect((await load(bridge, "MiniMaxH3")).status).toBe(200);
      return bridge;
    }

    function expectNagRefusal(res: Awaited<ReturnType<typeof generate>>) {
      expect(res.status).toBe(422);
      const error = (res.body as { error: { code: string; message: string } }).error;
      expect(error.code).toBe("FEATURE_UNSUPPORTED");
      expect(error.message).toContain("'nag'");
    }

    it.each([
      ["omitted", {}],
      ['"nag"', { neg_method: "nag" }],
    ] as const)("POST /generate refuses the NAG method (%s) as 'nag'", async (_label, method) => {
      expectNagRefusal(await generate(await h3Bridge(), { ...BODY, ...H3_SAMPLING, ...NEG, ...method }));
    });

    it("POST /generate accepts the VSF method", async () => {
      const res = await generate(await h3Bridge(), { ...BODY, ...H3_SAMPLING, ...NEG, neg_method: "vsf", vsf_scale: 3 });
      expect(res.status).toBe(202);
    });

    it.each([
      ["omitted", {}],
      ['"nag"', { neg_method: "nag" }],
    ] as const)("POST /generate/chain refuses the NAG method (%s) as 'nag'", async (_label, method) => {
      expectNagRefusal(await chain(await h3Bridge(), { ...H3_SAMPLING, ...NEG, ...method }));
    });

    it("POST /generate/chain accepts the VSF method", async () => {
      const res = await chain(await h3Bridge(), { ...H3_SAMPLING, ...NEG, neg_method: "vsf", vsf_scale: 3 });
      expect(res.status).toBe(202);
    });

    it("LTX 2.3 keeps accepting the NAG method on both routes", async () => {
      const single = await generate(createMockBridge({ delayMs: 0 }), {
        ...BODY,
        num_frames: 121,
        ...NEG,
        neg_method: "nag",
      });
      expect(single.status).toBe(202);
      const chained = await chain(createMockBridge({ delayMs: 0 }), { ...NEG, neg_method: "nag" }, [
        { num_frames: 121 },
        { num_frames: 121 },
      ]);
      expect(chained.status).toBe(202);
    });
  });

  // §1-107: the Ref2VA reference limits are H3's own.
  it("publishes `reference` for MiniMax H3 only (LTX publishes none)", async () => {
    const blocks = await listModels(createMockBridge({ delayMs: 0 }));
    expect(blocks.find((b) => b.id === "MiniMaxH3")?.constraints?.reference).toEqual({
      max_images: 9,
      max_videos: 3,
      max_audios: 3,
      max_total: 12,
      video_total_sec: 15,
      audio_total_sec: 15,
      clip_min_sec: 2,
    });
    expect(blocks.find((b) => b.id === "LTX23")?.constraints?.reference).toBeUndefined();
    expect(blocks.find((b) => b.id === "LTX25")?.constraints?.reference).toBeUndefined();
  });

  it("fs.probeMediaInfo returns a MockFs entry's own durationSec ahead of the global knob", async () => {
    const fs = createMockFs({
      folders: {
        "C:\\clips": [
          { name: "a.mp4", sizeBytes: 1, mtimeMs: 0, durationSec: 6 },
          { name: "b.mp4", sizeBytes: 1, mtimeMs: 0 },
        ],
      },
    });
    const bridge = createMockBridge({ delayMs: 0, fs, probeMediaInfoDurationSec: 10 });
    expect((await bridge.request("fs.probeMediaInfo", { filePath: "C:\\clips\\a.mp4" })).durationSec).toBe(6);
    expect((await bridge.request("fs.probeMediaInfo", { filePath: "C:\\clips\\b.mp4" })).durationSec).toBe(10);
    expect((await bridge.request("fs.probeMediaInfo", { filePath: "C:\\elsewhere\\c.mp4" })).durationSec).toBe(10);
  });

  // §1-107: `references` on POST /generate — LTX refuses it, H3 checks the shape.
  it("refuses `references` on LTX with 422 FEATURE_UNSUPPORTED; H3 checks the shape and stores it", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    const references = [
      { kind: "picture", id: "mock-image-1", slot: 1 },
      { kind: "video", id: "mock-video-1", slot: 2, max_sec: 7.5 },
    ];
    for (const id of ["LTX23", "LTX25"]) {
      if (id !== "LTX23") expect((await load(bridge, id)).status).toBe(200);
      const res = await generate(bridge, { ...BODY, num_frames: 121, references });
      expect(res.status).toBe(422);
      const error = (res.body as { error: { code: string; message: string } }).error;
      expect(error.code).toBe("FEATURE_UNSUPPORTED");
      expect(error.message).toContain("'references'");
      expect(blocksFeature(await listModels(bridge), id)).toContain("references");
    }
    expect(blocksFeature(await listModels(bridge), "MiniMaxH3")).not.toContain("references");

    expect((await load(bridge, "MiniMaxH3")).status).toBe(200);
    expect((await generate(bridge, { ...BODY, references: "nope" })).status).toBe(422);
    expect((await generate(bridge, { ...BODY, references: [{ kind: "image", id: "x", slot: 1 }] })).status).toBe(422);
    expect((await generate(bridge, { ...BODY, references: [{ kind: "picture", id: "", slot: 1 }] })).status).toBe(422);
    expect((await generate(bridge, { ...BODY, references: [{ kind: "picture", id: "x", slot: 0 }] })).status).toBe(422);
    expect((await generate(bridge, { ...BODY, references })).status).toBe(202);
  });

  // §1-109: a clip's own `references` on POST /generate/chain.
  it("refuses a clip's `references` on LTX with 422 FEATURE_UNSUPPORTED; H3 checks each clip's shape", async () => {
    const references = [{ kind: "picture", id: "mock-image-1", slot: 1 }];
    for (const id of ["LTX23", "LTX25"]) {
      // A fresh fixture per engine: the accepted chain below leaves a job running.
      const bridge = createMockBridge({ delayMs: 0 });
      if (id !== "LTX23") expect((await load(bridge, id)).status).toBe(200);
      const res = await chain(bridge, {}, [{ num_frames: 121 }, { num_frames: 121, references }]);
      expect(res.status, id).toBe(422);
      const error = (res.body as { error: { code: string; message: string } }).error;
      expect(error.code, id).toBe("FEATURE_UNSUPPORTED");
      expect(error.message, id).toContain("'references'");
      // An empty list is not a request.
      expect((await chain(bridge, {}, [{ num_frames: 121, references: [] }, { num_frames: 121 }])).status, id).toBe(
        202,
      );
    }

    const bridge = createMockBridge({ delayMs: 0 });
    expect((await load(bridge, "MiniMaxH3")).status).toBe(200);
    const bad = await chain(bridge, {}, [{ num_frames: 124 }, { num_frames: 124, references: [{ kind: "image", id: "x", slot: 1 }] }]);
    expect(bad.status).toBe(422);
    expect((await chain(bridge, {}, [{ num_frames: 124, references: "nope" }, { num_frames: 124 }])).status).toBe(422);
    // An empty list passes the shape check on H3 too (a fresh fixture: the
    // accepted chain below leaves a job running).
    const fresh = createMockBridge({ delayMs: 0 });
    expect((await load(fresh, "MiniMaxH3")).status).toBe(200);
    expect((await chain(fresh, {}, [{ num_frames: 124, references: [] }, { num_frames: 124 }])).status).toBe(202);
    expect((await chain(bridge, {}, [{ num_frames: 124 }, { num_frames: 124, references }])).status).toBe(202);
  });

  // §1-120: a clip's own `source_audio` on POST /generate/chain.
  it("refuses a clip's `source_audio` on LTX with 422 FEATURE_UNSUPPORTED; H3 checks its shape and the V2V clash", async () => {
    const sourceAudio = { audio_id: "mock-audio-1", strength: 0.5, offset_frames: 102 };
    for (const id of ["LTX23", "LTX25"]) {
      const bridge = createMockBridge({ delayMs: 0 });
      if (id !== "LTX23") expect((await load(bridge, id)).status).toBe(200);
      const res = await chain(bridge, {}, [{ num_frames: 121 }, { num_frames: 121, source_audio: sourceAudio }]);
      expect(res.status, id).toBe(422);
      const error = (res.body as { error: { code: string; message: string } }).error;
      expect(error.code, id).toBe("FEATURE_UNSUPPORTED");
      expect(error.message, id).toContain("'source_audio'");
    }

    const bridge = createMockBridge({ delayMs: 0 });
    expect((await load(bridge, "MiniMaxH3")).status).toBe(200);
    const bad: unknown[] = [
      { audio_id: "", strength: 0.5, offset_frames: 0 },
      { audio_id: "a", strength: 1.5, offset_frames: 0 },
      { audio_id: "a", strength: 0.5, offset_frames: -1 },
      { audio_id: "a", strength: 0.5, offset_frames: 1.5 },
      "nope",
    ];
    for (const spec of bad) {
      const res = await chain(bridge, {}, [{ num_frames: 124, source_audio: spec }, { num_frames: 124 }]);
      expect(res.status, JSON.stringify(spec)).toBe(422);
      expect((res.body as { error: { code: string } }).error.code).toBe("VALIDATION_ERROR");
    }
    // Exclusive with a V2V source.
    const v2v = await chain(bridge, { source_video: { video_id: "vid", context_frames: 22 } }, [
      { num_frames: 124, source_audio: sourceAudio },
    ]);
    expect(v2v.status).toBe(422);
    // The chain-wide `source_audio` stays refused (`a2v`).
    const top = await chain(bridge, { source_audio: { audio_id: "aud" } });
    expect(top.status).toBe(422);
    expect((top.body as { error: { message: string } }).error.message).toContain("'a2v'");
    // The defaults may be left out; the request is stored as sent.
    const ok = await chain(bridge, {}, [
      { num_frames: 124, source_audio: { audio_id: "mock-audio-1" } },
      { num_frames: 124, source_audio: sourceAudio },
    ]);
    expect(ok.status).toBe(202);
  });

  it("publishes `transformer_role` for the active H3 from the selected transformer's name (null otherwise)", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    const roleOf = async (id: string) => (await listModels(bridge)).find((b) => b.id === id)?.transformer_role;
    expect(await roleOf("LTX23")).toBeNull();
    expect((await load(bridge, "MiniMaxH3")).status).toBe(200);
    // The fixture's H3 transformer is called "default": no role.
    expect(await roleOf("MiniMaxH3")).toBeNull();
    expect(await roleOf("LTX23")).toBeNull();
  });
});

function blocksFeature(blocks: BaseModelBlock[], id: string): string[] {
  return blocks.find((b) => b.id === id)?.unsupported_features ?? [];
}
