import { describe, expect, it } from "vitest";
import type { BaseModelBlock, ModelsResponse } from "../api/types";
import { createMockBridge } from "./mockBridge";

// §1-111: the fixture's `prompt_token_limit` and `POST /utils/prompt-tokens`.

type Bridge = ReturnType<typeof createMockBridge>;

function count(bridge: Bridge, body: Record<string, unknown>) {
  return bridge.request("backend.request", { method: "POST", path: "/api/v1/utils/prompt-tokens", body });
}

describe("mock bridge prompt tokens", () => {
  it("publishes 1024 for the LTX families and null for H3", async () => {
    const bridge = createMockBridge();
    const res = await bridge.request("backend.request", { method: "GET", path: "/api/v1/models" });
    const blocks = (res.body as ModelsResponse).base_models as BaseModelBlock[];
    const limits = Object.fromEntries(blocks.map((b) => [b.id, b.constraints?.prompt_token_limit]));
    expect(limits).toEqual({ LTX23: 1024, LTX25: 1024, MiniMaxH3: null });
  });

  it("counts words plus a start token, composes clips, and nulls unsent fields", async () => {
    const bridge = createMockBridge();
    const res = await count(bridge, { prompt: "a cat sits", clips: ["", "  on a mat "] });
    expect(res.status).toBe(200);
    expect(res.body).toEqual({ prompt: 4, negative_prompt: null, clips: [4, 7] });
    const neg = await count(bridge, { prompt: "", negative_prompt: "blurry" });
    expect(neg.body).toEqual({ prompt: 1, negative_prompt: 2, clips: null });
  });
});
