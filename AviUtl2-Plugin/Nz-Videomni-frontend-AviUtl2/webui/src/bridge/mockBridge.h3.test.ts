import { describe, expect, it } from "vitest";
import type { BaseModelBlock, ModelsResponse } from "../api/types";
import { createMockBridge } from "./mockBridge";
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
      expect.arrayContaining(["chain", "retake", "reference_video", "nag", "keyframe_strength"]),
    );
    expect(h3?.unsupported_features).not.toContain("multi_keyframes");
    const ltx = blocks.find((b) => b.id === "LTX23");
    expect(ltx?.constraints).toMatchObject({
      num_frames: { step: 8, offset: 1 },
      sampling: null,
      keyframes: { step: 8, offset: 1 },
    });
    // 台帳 §1-101: VRAM 16GB mode is H3's own — H3 does NOT publish the name,
    // both LTX engines do (which hides the Settings row there).
    expect(h3?.unsupported_features).not.toContain("vram16_mode");
    expect(ltx?.unsupported_features).toContain("vram16_mode");
    expect(blocks.find((b) => b.id === "LTX25")?.unsupported_features).toContain("vram16_mode");
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

  it("refuses a whole POST /generate/chain while MiniMax H3 (which declares `chain`) is loaded", async () => {
    const bridge = createMockBridge({ delayMs: 0 });
    expect((await load(bridge, "MiniMaxH3")).status).toBe(200);
    const res = await bridge.request("backend.request", {
      method: "POST",
      path: "/api/v1/generate/chain",
      body: { prompt: "a cat", width: 1280, height: 704, frame_rate: 24, seed: 1, clips: [{ num_frames: 124 }] },
    });
    expect(res.status).toBe(422);
    const error = (res.body as { error: { code: string; message: string } }).error;
    expect(error.code).toBe("FEATURE_UNSUPPORTED");
    expect(error.message).toContain("'chain'");
  });
});
