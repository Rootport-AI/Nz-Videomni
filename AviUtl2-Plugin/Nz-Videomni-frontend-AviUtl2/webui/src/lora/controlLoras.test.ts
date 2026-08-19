import { describe, expect, it } from "vitest";
import type { AppConfig, LoraEntry } from "../api/types";
import type { LorasState } from "../modes/inventory/useLoras";
import { combineLoras, OUTPAINT_LORA_NAME, resolveControlLoraNames, selectableControlLoraNames } from "./controlLoras";

const IC_LORAS_MODEL: AppConfig["model"] = {
  ic_loras: {
    "canny-control": { path: "mock/canny.safetensors", preprocess: "canny" },
    "pose-control": { path: "mock/pose.safetensors", preprocess: "dwpose" },
  },
};

/** Only `model` is ever read by this module — the rest of `AppConfig` is
 * irrelevant to these tests, so this stub only needs to satisfy the type,
 * not represent a fully-populated config. */
function configWith(model: AppConfig["model"]): AppConfig {
  return { model } as AppConfig;
}

const LORAS_READY: LorasState = {
  status: "ready",
  loras: [
    { name: "Pixar_Toon", kind: "style", has_thumbnail: true, exists: true, source: "scan" },
    { name: "canny-control", kind: "control", has_thumbnail: true, exists: true, source: "config" },
    { name: "pose-control", kind: "control", has_thumbnail: false, exists: true, source: "config" },
  ] satisfies LoraEntry[],
};

describe("resolveControlLoraNames", () => {
  it("uses GET /loras' kind===\"control\" entries when the loras state is ready, excluding style entries", () => {
    const names = resolveControlLoraNames(configWith(IC_LORAS_MODEL), LORAS_READY);
    expect(names).toEqual(new Set(["canny-control", "pose-control"]));
  });

  it("falls back to config.model.ic_loras' keys while loading", () => {
    const names = resolveControlLoraNames(configWith(IC_LORAS_MODEL), { status: "loading" });
    expect(names).toEqual(new Set(["canny-control", "pose-control"]));
  });

  it("falls back to config.model.ic_loras' keys on a GET /loras error", () => {
    const names = resolveControlLoraNames(configWith(IC_LORAS_MODEL), { status: "error", message: "network" });
    expect(names).toEqual(new Set(["canny-control", "pose-control"]));
  });

  it("returns an empty set when both the loras state and config.model are absent", () => {
    const names = resolveControlLoraNames(configWith(undefined), { status: "loading" });
    expect(names).toEqual(new Set());
  });

  it("returns an empty set when ready but GET /loras has no control entries", () => {
    const names = resolveControlLoraNames(configWith(undefined), {
      status: "ready",
      loras: [{ name: "Pixar_Toon", kind: "style", has_thumbnail: true, exists: true, source: "scan" }],
    });
    expect(names).toEqual(new Set());
  });
});

describe("selectableControlLoraNames", () => {
  it("removes in-outpainting from an otherwise-untouched set", () => {
    const names = new Set(["canny-control", OUTPAINT_LORA_NAME, "pose-control"]);
    expect(selectableControlLoraNames(names)).toEqual(new Set(["canny-control", "pose-control"]));
  });

  it("leaves a set with no hidden names untouched", () => {
    const names = new Set(["canny-control", "pose-control"]);
    expect(selectableControlLoraNames(names)).toEqual(names);
  });

  it("returns an empty set when the only entry is in-outpainting", () => {
    expect(selectableControlLoraNames(new Set([OUTPAINT_LORA_NAME]))).toEqual(new Set());
  });

  it("returns an empty set unchanged", () => {
    expect(selectableControlLoraNames(new Set())).toEqual(new Set());
  });
});

describe("combineLoras (Gradio-faithful merge, mirrors _combine_generate_loras)", () => {
  it("returns the prompt loras unchanged when no control LoRA is selected", () => {
    const promptLoras = [{ name: "Pixar_Toon", strength: 1.0 }];
    expect(combineLoras(null, promptLoras)).toEqual(promptLoras);
  });

  it("puts the control LoRA first, followed by the prompt's own tags", () => {
    const result = combineLoras({ name: "canny-control", strength: 0.8 }, [{ name: "Pixar_Toon", strength: 1.0 }]);
    expect(result).toEqual([
      { name: "canny-control", strength: 0.8 },
      { name: "Pixar_Toon", strength: 1.0 },
    ]);
  });

  it("returns just the control LoRA when the prompt has no style tags", () => {
    const result = combineLoras({ name: "canny-control", strength: 0.8 }, []);
    expect(result).toEqual([{ name: "canny-control", strength: 0.8 }]);
  });

  it("dedupes by name, keeping the FIRST occurrence's position but the LAST occurrence's strength", () => {
    // The control LoRA and a prompt tag happen to share a name — position 0
    // (the control slot) is kept, but the prompt tag's strength wins.
    const result = combineLoras({ name: "Shared", strength: 0.3 }, [
      { name: "Other", strength: 1.0 },
      { name: "Shared", strength: 0.9 },
    ]);
    expect(result).toEqual([
      { name: "Shared", strength: 0.9 },
      { name: "Other", strength: 1.0 },
    ]);
  });

  it("dedupes duplicate prompt-only tags the same way (first position, last strength)", () => {
    const result = combineLoras(null, [
      { name: "A", strength: 1.0 },
      { name: "B", strength: 0.5 },
      { name: "A", strength: 1.5 },
    ]);
    expect(result).toEqual([
      { name: "A", strength: 1.5 },
      { name: "B", strength: 0.5 },
    ]);
  });

  it("never mutates the input promptLoras array", () => {
    const promptLoras = [{ name: "Pixar_Toon", strength: 1.0 }];
    const snapshot = JSON.parse(JSON.stringify(promptLoras));
    combineLoras({ name: "canny-control", strength: 0.8 }, promptLoras);
    expect(promptLoras).toEqual(snapshot);
  });

  it("passes audio_strength through untouched — no field projection strips it", () => {
    const promptLoras = [{ name: "X", strength: 0.8, audio_strength: 0.3 }];
    expect(combineLoras(null, promptLoras)).toEqual([{ name: "X", strength: 0.8, audio_strength: 0.3 }]);
  });

  it("keeps audio_strength on the LAST occurrence's strength when deduping by name", () => {
    const result = combineLoras(null, [
      { name: "A", strength: 1.0, audio_strength: 0.5 },
      { name: "A", strength: 1.5 },
    ]);
    expect(result).toEqual([{ name: "A", strength: 1.5 }]);
    expect(result[0]).not.toHaveProperty("audio_strength");
  });
});
