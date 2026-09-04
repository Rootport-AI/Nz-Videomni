import { act, renderHook, waitFor } from "@testing-library/react";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";
import type { AppConfig, GenerateChainRequest } from "../../api/types";
import { createApiClient } from "../../api/client";
import { createMockBridge, createMockFs } from "../../bridge/mockBridge";
import type { BridgeMethod, NativeBridge, ParamsOf, ResultOf } from "../../bridge";
import type { ControlLoraSelection } from "../../lora/controlLoras";
import type { NagSettings } from "../../shell/nagSettings";
import { ACCELERATION_DEFAULTS } from "../../shell/accelerationSettings";
import type { AccelerationSettings } from "../../shell/accelerationSettings";
import { suggestFramesForAudio } from "../chained/chainUtils";
import { FALLBACK_APP_CONFIG } from "./defaultConfig";
import { estimateGenerationSeconds } from "./estimateUtils";
import { resolveSpillFreeFrames } from "./spillUtils";
import { useGenerationForm } from "./useGenerationForm";
import type { UseGenerationFormDeps } from "./useGenerationForm";

/** IC-LoRA UI redesign (第5波): `controlLora`/`setControlLora` are owned by
 * the CALLER (`AppShell` in production — see `useGenerationForm`'s own doc
 * comment for why), not by the hook itself. This harness reproduces that
 * shape with a plain `useState` so tests can drive `result.current.setControlLora`
 * the same way `AppShell`'s own state would react. */
function renderFormWithControlLora(
  prompt: string,
  config: AppConfig = FALLBACK_APP_CONFIG,
  deps: Omit<UseGenerationFormDeps, "controlLora" | "setControlLora"> = {},
) {
  return renderHook(() => {
    const [controlLora, setControlLora] = useState<ControlLoraSelection | null>(null);
    return useGenerationForm(config, prompt, { ...deps, controlLora, setControlLora });
  });
}

/** Renders the Create form backed by a mock bridge whose virtual filesystem
 * knows the wav duration `ui.pickFile` (mocked to resolve `picked-audio-1.wav`
 * under `C:\Users\mock\Pictures`, per `mockBridge.ts`'s `handlePickFile`) will
 * resolve to — so `sourceAudio.pick()` exercises the full pick -> probe ->
 * auto-adjust chain. Mirrors `modes/chained/useChainForm.test.ts`'s `setupA2V`. */
function setupA2V(durationSec: number, prompt = "a music video", deps: UseGenerationFormDeps = {}) {
  const mockFs = createMockFs({
    folders: {
      "C:\\Users\\mock\\Pictures": [{ name: "picked-audio-1.wav", sizeBytes: 4_096, mtimeMs: 0, durationSec }],
    },
  });
  const mockBridge = createMockBridge({ delayMs: 0, fs: mockFs });
  const { result } = renderHook(() =>
    useGenerationForm(FALLBACK_APP_CONFIG, prompt, { ...deps, nativeBridge: mockBridge }),
  );
  return { result, mockBridge };
}

describe("useGenerationForm", () => {
  it("seeds values from config.generation_defaults, already snapped to valid multiples", () => {
    const { result } = renderHook(() => useGenerationForm(FALLBACK_APP_CONFIG, ""));

    expect(result.current.values).toMatchObject({
      width: 1280,
      height: 768,
      numFrames: 361,
      frameRate: 24,
      seed: -1,
    });
    expect(result.current.promptError).not.toBeNull(); // empty prompt
    expect(result.current.isValid).toBe(false);
  });

  // W8: a right-click prefill can seed the initial DURATION (num_frames) from
  // the DURATION policy engine — the fourth `initial` arg's `numFrames`.
  describe("initial.numFrames (W8 right-click DURATION seed)", () => {
    it("seeds the initial DURATION from initial.numFrames instead of the config default (raise)", () => {
      // 512x320's comfort ceiling (481) is ABOVE the config default (361) — the
      // seed raises DURATION, not just lowers it.
      const { result } = renderHook(() => useGenerationForm(FALLBACK_APP_CONFIG, "", {}, { numFrames: 481 }));
      expect(result.current.values.numFrames).toBe(481);
    });

    it("seeds a LOWER initial DURATION when the policy resolved a smaller value", () => {
      const { result } = renderHook(() => useGenerationForm(FALLBACK_APP_CONFIG, "", {}, { numFrames: 153 }));
      expect(result.current.values.numFrames).toBe(153);
    });

    it("re-snaps the seed onto the 8n+1 grid and clamps it into range", () => {
      // 999999 clamps to max (481); an off-grid 100 snaps to the nearest 8n+1 (97).
      const { result: high } = renderHook(() => useGenerationForm(FALLBACK_APP_CONFIG, "", {}, { numFrames: 999_999 }));
      expect(high.current.values.numFrames).toBe(FALLBACK_APP_CONFIG.limits.max_num_frames);
      const { result: offGrid } = renderHook(() => useGenerationForm(FALLBACK_APP_CONFIG, "", {}, { numFrames: 100 }));
      expect(offGrid.current.values.numFrames).toBe(97);
    });

    it("keeps the config default when initial.numFrames is omitted (normal tab-opened form)", () => {
      const { result } = renderHook(() => useGenerationForm(FALLBACK_APP_CONFIG, "", {}, {}));
      expect(result.current.values.numFrames).toBe(361);
    });
  });

  // 台帳§3-71/§3-72: the fps field only ever holds a whole frame rate in [1, 60].
  // The rounding rules themselves are `paramUtils.test.ts`'s; these pin that
  // Create's setter and lazy init are wired to them.
  describe("frame rate (whole frame rates only)", () => {
    it("setFrameRate rounds a non-integer rate to the nearest integer", () => {
      const { result } = renderHook(() => useGenerationForm(FALLBACK_APP_CONFIG, "x"));
      act(() => result.current.setFrameRate(29.97));
      expect(result.current.values.frameRate).toBe(30);
      act(() => result.current.setFrameRate(23.976));
      expect(result.current.values.frameRate).toBe(24);
    });

    it("setFrameRate still clamps to [1, 60]", () => {
      const { result } = renderHook(() => useGenerationForm(FALLBACK_APP_CONFIG, "x"));
      act(() => result.current.setFrameRate(120));
      expect(result.current.values.frameRate).toBe(60);
      act(() => result.current.setFrameRate(-5));
      expect(result.current.values.frameRate).toBe(1);
    });

    it('setFrameRate lands an emptied box (Number("") === 0) and NaN on 1fps, exactly as before', () => {
      const { result } = renderHook(() => useGenerationForm(FALLBACK_APP_CONFIG, "x"));
      act(() => result.current.setFrameRate(0));
      expect(result.current.values.frameRate).toBe(1);
      act(() => result.current.setFrameRate(Number.NaN));
      expect(result.current.values.frameRate).toBe(1);
    });

    it("snaps a right-click fps seed as well as the config default", () => {
      const { result: seeded } = renderHook(() =>
        useGenerationForm(FALLBACK_APP_CONFIG, "x", {}, { frameRate: 29.97 }),
      );
      expect(seeded.current.values.frameRate).toBe(30);

      const ntscConfig: AppConfig = {
        ...FALLBACK_APP_CONFIG,
        generation_defaults: { ...FALLBACK_APP_CONFIG.generation_defaults, frame_rate: 23.976 },
      };
      const { result: fromConfig } = renderHook(() => useGenerationForm(ntscConfig, "x"));
      expect(fromConfig.current.values.frameRate).toBe(24);
    });

    it("falls back to FRAME_RATE_FALLBACK (24) rather than leaking an unusable config value", () => {
      // `?? config…` would put the 0 straight back into the field — the whole
      // reason the initializer chains to the constant instead.
      const brokenConfig: AppConfig = {
        ...FALLBACK_APP_CONFIG,
        generation_defaults: { ...FALLBACK_APP_CONFIG.generation_defaults, frame_rate: 0 },
      };
      const { result } = renderHook(() => useGenerationForm(brokenConfig, "x"));
      expect(result.current.values.frameRate).toBe(24);
    });
  });

  it("seeds cropOutput from config.generation_defaults.crop_output (real server standard_720p defaults)", () => {
    const config: AppConfig = {
      ...FALLBACK_APP_CONFIG,
      generation_defaults: {
        ...FALLBACK_APP_CONFIG.generation_defaults,
        width: 1280,
        height: 768,
        num_frames: 257,
        crop_output: { width: 1280, height: 720 },
      },
    };
    const { result } = renderHook(() => useGenerationForm(config, "x"));

    expect(result.current.cropOutput).toEqual({ width: 1280, height: 720 });
  });

  it("isValid/promptError follow the externally-provided prompt argument (owned by the shared PromptBar)", () => {
    const { result, rerender } = renderHook(({ prompt }) => useGenerationForm(FALLBACK_APP_CONFIG, prompt), {
      initialProps: { prompt: "" },
    });
    expect(result.current.isValid).toBe(false);
    expect(result.current.promptError).not.toBeNull();

    rerender({ prompt: "a cat riding a skateboard" });
    expect(result.current.isValid).toBe(true);
    expect(result.current.promptError).toBeNull();

    const request = result.current.toGenerateRequest();
    expect(request).toEqual({
      prompt: "a cat riding a skateboard",
      width: 1280,
      height: 768,
      // generation_defaults.crop_output is null by default (2026-08-18: crop
      // OFF by default; a preset selection still carries its own crop_output),
      // so the field is absent from the request entirely.
      num_frames: 361,
      frame_rate: 24,
      seed: -1,
    });
    // negative_prompt is absent only when nag is not supplied (2026-07-28 NAG
    // contract); the other three remain permanently banned (see NAG-enabled
    // coverage further below for the ON-time shape).
    expect(request).not.toHaveProperty("negative_prompt");
    expect(request).not.toHaveProperty("guidance_scale");
    expect(request).not.toHaveProperty("num_inference_steps");
    expect(request).not.toHaveProperty("pipeline");
  });

  // NAG (2026-07-28): `deps.nag` threads into both `validityReasons` and
  // `toGenerateRequest` via the shared `nagRequestFields`/`isNagNegativeEmpty`
  // contract (`shell/nagSettings.ts`).
  describe("NAG (deps.nag)", () => {
    const ENABLED_NAG: NagSettings = {
      text: "blurry, low quality, distorted, watermark, text",
      enabled: true,
      scale: 11.0,
      tau: 2.5,
      alpha: 0.25,
      method: "nag",
      vsfScale: 1.5,
    };

    it("enabled with a non-empty body: the request carries exactly the 7 additive fields with their values", () => {
      const { result } = renderHook(() => useGenerationForm(FALLBACK_APP_CONFIG, "a cat riding a skateboard", { nag: ENABLED_NAG }));

      const request = result.current.toGenerateRequest();
      expect(request.negative_prompt).toBe(ENABLED_NAG.text);
      expect(request.nag_enabled).toBe(true);
      expect(request.nag_scale).toBe(11.0);
      expect(request.nag_tau).toBe(2.5);
      expect(request.nag_alpha).toBe(0.25);
      expect(request.neg_method).toBe("nag");
      expect(request.vsf_scale).toBe(1.5);
      expect(result.current.validityReasons).not.toContain("nagNegativeEmpty");
      expect(result.current.isValid).toBe(true);
    });

    it("enabled with method 'vsf': the request carries neg_method/vsf_scale reflecting VSF", () => {
      const { result } = renderHook(() =>
        useGenerationForm(FALLBACK_APP_CONFIG, "a cat riding a skateboard", {
          nag: { ...ENABLED_NAG, method: "vsf", vsfScale: 4 },
        }),
      );

      const request = result.current.toGenerateRequest();
      expect(request.neg_method).toBe("vsf");
      expect(request.vsf_scale).toBe(4);
    });

    it("enabled with an empty body: validityReasons carries nagNegativeEmpty and isValid is false", () => {
      const { result } = renderHook(() =>
        useGenerationForm(FALLBACK_APP_CONFIG, "a cat riding a skateboard", { nag: { ...ENABLED_NAG, text: "" } }),
      );
      expect(result.current.validityReasons).toContain("nagNegativeEmpty");
      expect(result.current.isValid).toBe(false);
    });

    it("enabled with a whitespace-only body: same block as an empty body", () => {
      const { result } = renderHook(() =>
        useGenerationForm(FALLBACK_APP_CONFIG, "a cat riding a skateboard", { nag: { ...ENABLED_NAG, text: "   " } }),
      );
      expect(result.current.validityReasons).toContain("nagNegativeEmpty");
      expect(result.current.isValid).toBe(false);
    });

    it("disabled: no nagNegativeEmpty reason even with an empty body, and no additive fields on the request", () => {
      const { result } = renderHook(() =>
        useGenerationForm(FALLBACK_APP_CONFIG, "a cat riding a skateboard", { nag: { ...ENABLED_NAG, enabled: false, text: "" } }),
      );
      expect(result.current.validityReasons).not.toContain("nagNegativeEmpty");
      expect(result.current.isValid).toBe(true);

      const request = result.current.toGenerateRequest();
      expect(request).not.toHaveProperty("negative_prompt");
      expect(request).not.toHaveProperty("nag_enabled");
      expect(request).not.toHaveProperty("nag_scale");
      expect(request).not.toHaveProperty("nag_tau");
      expect(request).not.toHaveProperty("nag_alpha");
      expect(request).not.toHaveProperty("neg_method");
      expect(request).not.toHaveProperty("vsf_scale");
    });
  });

  // Acceleration (2026-07-31): `deps.acceleration` threads into
  // `toGenerateRequest`/`buildA2vRequest` through the single
  // `accelerationRequestFields` contract (`shell/accelerationSettings.ts`).
  describe("Acceleration (deps.acceleration)", () => {
    const SAGE: AccelerationSettings = { ...ACCELERATION_DEFAULTS, attentionBackend: "sage" };

    it("default: the request is byte-identical to one built with no acceleration dep at all", () => {
      const withoutDep = renderHook(() => useGenerationForm(FALLBACK_APP_CONFIG, "a cat riding a skateboard"));
      const withDefaults = renderHook(() =>
        useGenerationForm(FALLBACK_APP_CONFIG, "a cat riding a skateboard", { acceleration: ACCELERATION_DEFAULTS }),
      );

      const baseline = withoutDep.result.current.toGenerateRequest();
      const withDefault = withDefaults.result.current.toGenerateRequest();
      // Full JSON equality, key ORDER included — the byte-identical contract.
      expect(JSON.stringify(withDefault)).toBe(JSON.stringify(baseline));
      expect(withDefault).not.toHaveProperty("attention_backend");
      expect(withDefault).not.toHaveProperty("fused_gguf_dequant_kernel");
      expect(withDefault).not.toHaveProperty("vae_mode");
    });

    it("sage: exactly ONE key more than the default request", () => {
      const baseline = renderHook(() =>
        useGenerationForm(FALLBACK_APP_CONFIG, "a cat riding a skateboard"),
      ).result.current.toGenerateRequest();
      const sage = renderHook(() =>
        useGenerationForm(FALLBACK_APP_CONFIG, "a cat riding a skateboard", { acceleration: SAGE }),
      ).result.current.toGenerateRequest();

      expect(sage.attention_backend).toBe("sage");
      expect(Object.keys(sage)).toEqual([...Object.keys(baseline), "attention_backend"]);
      expect(sage).toEqual({ ...baseline, attention_backend: "sage" });
    });

    it("block-swap prefetch off: exactly ONE key more than the default request", () => {
      // S4 (2026-08-01): ACCELERATION_DEFAULTS.blockSwapPrefetch is now true
      // (server default), so it's turning it OFF that now differs from the
      // default request.
      const baseline = renderHook(() =>
        useGenerationForm(FALLBACK_APP_CONFIG, "a cat riding a skateboard"),
      ).result.current.toGenerateRequest();
      const prefetch = renderHook(() =>
        useGenerationForm(FALLBACK_APP_CONFIG, "a cat riding a skateboard", {
          acceleration: { ...ACCELERATION_DEFAULTS, blockSwapPrefetch: false },
        }),
      ).result.current.toGenerateRequest();

      expect(prefetch.block_swap_prefetch).toBe(false);
      expect(Object.keys(prefetch)).toEqual([...Object.keys(baseline), "block_swap_prefetch"]);
      expect(prefetch).toEqual({ ...baseline, block_swap_prefetch: false });
    });

    it("keep-resident on: exactly ONE key more than the default request", () => {
      // §48 (2026-08-02): the server default is FALSE here — the opposite
      // direction from `block_swap_prefetch` above — so it's turning it ON
      // that differs from the default request.
      const baseline = renderHook(() =>
        useGenerationForm(FALLBACK_APP_CONFIG, "a cat riding a skateboard"),
      ).result.current.toGenerateRequest();
      const keepResident = renderHook(() =>
        useGenerationForm(FALLBACK_APP_CONFIG, "a cat riding a skateboard", {
          acceleration: { ...ACCELERATION_DEFAULTS, keepResident: true },
        }),
      ).result.current.toGenerateRequest();

      expect(keepResident.keep_resident).toBe(true);
      expect(Object.keys(keepResident)).toEqual([...Object.keys(baseline), "keep_resident"]);
      expect(keepResident).toEqual({ ...baseline, keep_resident: true });
    });

    it("fused GGUF dequant kernel off: exactly ONE key more than the default request (§1-11)", () => {
      // §51 (2026-08-04): the server default flipped to TRUE, so turning it
      // OFF is what diverges — and unlike keep-resident there is no gate of
      // any kind.
      const baseline = renderHook(() =>
        useGenerationForm(FALLBACK_APP_CONFIG, "a cat riding a skateboard"),
      ).result.current.toGenerateRequest();
      const fused = renderHook(() =>
        useGenerationForm(FALLBACK_APP_CONFIG, "a cat riding a skateboard", {
          acceleration: { ...ACCELERATION_DEFAULTS, fusedGgufDequantKernel: false },
        }),
      ).result.current.toGenerateRequest();

      expect(fused.fused_gguf_dequant_kernel).toBe(false);
      expect(Object.keys(fused)).toEqual([...Object.keys(baseline), "fused_gguf_dequant_kernel"]);
      expect(fused).toEqual({ ...baseline, fused_gguf_dequant_kernel: false });
    });

    it("PrunaVAED: exactly ONE key more than the default request (§52)", () => {
      // §52 (2026-08-05): the server default is `"default"` — the same
      // direction as keep-resident — so picking PrunaVAED is what diverges.
      // The API VALUE stays `"prune_vaed"` even though the display name is
      // PrunaVAED.
      const baseline = renderHook(() =>
        useGenerationForm(FALLBACK_APP_CONFIG, "a cat riding a skateboard"),
      ).result.current.toGenerateRequest();
      const pruned = renderHook(() =>
        useGenerationForm(FALLBACK_APP_CONFIG, "a cat riding a skateboard", {
          acceleration: { ...ACCELERATION_DEFAULTS, vaeMode: "prune_vaed" },
        }),
      ).result.current.toGenerateRequest();

      expect(pruned.vae_mode).toBe("prune_vaed");
      expect(Object.keys(pruned)).toEqual([...Object.keys(baseline), "vae_mode"]);
      expect(pruned).toEqual({ ...baseline, vae_mode: "prune_vaed" });
      expect(pruned).not.toHaveProperty("attention_backend");
    });

    it("A2V: the same rule applies to buildA2vRequest (absent by default, one key under sage)", async () => {
      const plain = setupA2V(2.0);
      act(() => {
        void plain.result.current.sourceAudio.pick();
      });
      await waitFor(() => expect(plain.result.current.isA2v).toBe(true));
      const plainPayload = plain.result.current.buildA2vRequest([]);
      expect(plainPayload).not.toHaveProperty("attention_backend");

      const sage = setupA2V(2.0, "a music video", { acceleration: SAGE });
      act(() => {
        void sage.result.current.sourceAudio.pick();
      });
      await waitFor(() => expect(sage.result.current.isA2v).toBe(true));
      const sagePayload = sage.result.current.buildA2vRequest([]);
      expect(sagePayload.attention_backend).toBe("sage");
      // `source_audio.audio_id` differs per run, so compare key sets rather
      // than whole payloads.
      expect(Object.keys(sagePayload)).toEqual([...Object.keys(plainPayload), "attention_backend"]);
    });
  });

  it("toGenerateRequest strips <lora:...> tags from the prompt and moves them into loras[]", () => {
    const { result } = renderHook(() => useGenerationForm(FALLBACK_APP_CONFIG, "a cat <lora:Pixar_Toon:1.5> riding a skateboard"));

    const request = result.current.toGenerateRequest();
    expect(request.prompt).toBe("a cat riding a skateboard");
    expect(request.loras).toEqual([{ name: "Pixar_Toon", strength: 1.5 }]);
  });

  it("toGenerateRequest omits loras entirely (no key at all) when the prompt has no tags", () => {
    const { result } = renderHook(() => useGenerationForm(FALLBACK_APP_CONFIG, "a cat riding a skateboard"));

    const request = result.current.toGenerateRequest();
    expect(request).not.toHaveProperty("loras");
  });

  it("setWidth/setHeight snap to the nearest multiple of 64 and clamp to limits when snap is on (stepper/slider)", () => {
    const { result } = renderHook(() => useGenerationForm(FALLBACK_APP_CONFIG, "x"));

    act(() => result.current.setWidth(500, true));
    expect(result.current.values.width).toBe(512);

    act(() => result.current.setWidth(999_999, true));
    expect(result.current.values.width).toBe(FALLBACK_APP_CONFIG.limits.max_width);

    act(() => result.current.setHeight(10, true));
    expect(result.current.values.height).toBe(128);
  });

  it("setWidth/setHeight default to snap=true when the flag is omitted (backward compatible)", () => {
    const { result } = renderHook(() => useGenerationForm(FALLBACK_APP_CONFIG, "x"));

    act(() => result.current.setWidth(500));
    expect(result.current.values.width).toBe(512);

    act(() => result.current.setHeight(10));
    expect(result.current.values.height).toBe(128);
  });

  it("setWidth/setHeight pass free manual entry straight through (snap=false: no rounding, no clamp)", () => {
    const { result } = renderHook(() => useGenerationForm(FALLBACK_APP_CONFIG, "x"));

    act(() => result.current.setWidth(500, false));
    expect(result.current.values.width).toBe(500);

    // Off-grid + out-of-range values are NOT clamped when typed by hand.
    act(() => result.current.setHeight(999_999, false));
    expect(result.current.values.height).toBe(999_999);

    // Non-finite input is rejected, keeping the prior value.
    act(() => result.current.setWidth(Number.NaN, false));
    expect(result.current.values.width).toBe(500);
  });

  it("isValid gates on the width/height grid: an off-grid manual entry blocks submit until re-snapped", () => {
    const { result } = renderHook(() => useGenerationForm(FALLBACK_APP_CONFIG, "a cat"));

    // Clear the default crop so this test isolates the width/height grid gate:
    // otherwise shrinking width below the default crop_output width would block
    // submit for an unrelated (crop) reason.
    act(() => result.current.setCropOutput(null));

    // A valid prompt + on-grid seed (1280x768) is submittable.
    expect(result.current.isValid).toBe(true);

    // Typing an off-grid (non-64-multiple) width blocks submit.
    act(() => result.current.setWidth(500, false));
    expect(result.current.values.width).toBe(500);
    expect(result.current.isValid).toBe(false);

    // Snapping back onto the 64 grid re-enables submit.
    act(() => result.current.setWidth(500, true));
    expect(result.current.values.width).toBe(512);
    expect(result.current.isValid).toBe(true);
  });

  it("setNumFrames snaps to the nearest 8n+1 and clamps to limits when snap is on (default/stepper/slider)", () => {
    const { result } = renderHook(() => useGenerationForm(FALLBACK_APP_CONFIG, "x"));

    act(() => result.current.setNumFrames(100));
    expect(result.current.values.numFrames).toBe(97);

    act(() => result.current.setNumFrames(1));
    expect(result.current.values.numFrames).toBe(9);

    act(() => result.current.setNumFrames(10_000));
    expect(result.current.values.numFrames).toBe(FALLBACK_APP_CONFIG.limits.max_num_frames);

    act(() => result.current.setNumFrames(100, true));
    expect(result.current.values.numFrames).toBe(97);
  });

  it("setNumFrames passes free manual entry straight through (snap=false: no rounding, no clamp)", () => {
    const { result } = renderHook(() => useGenerationForm(FALLBACK_APP_CONFIG, "x"));

    act(() => result.current.setNumFrames(50, false));
    expect(result.current.values.numFrames).toBe(50);

    // Out-of-range values are NOT clamped when typed by hand.
    act(() => result.current.setNumFrames(999_999, false));
    expect(result.current.values.numFrames).toBe(999_999);

    // Non-finite input is rejected, keeping the prior value.
    act(() => result.current.setNumFrames(Number.NaN, false));
    expect(result.current.values.numFrames).toBe(999_999);
  });

  it("isValid gates on the num_frames 8n+1 grid: an off-grid manual entry blocks submit until re-snapped", () => {
    const { result } = renderHook(() => useGenerationForm(FALLBACK_APP_CONFIG, "a cat"));

    expect(result.current.isValid).toBe(true);

    act(() => result.current.setNumFrames(50, false)); // not 8n+1
    expect(result.current.values.numFrames).toBe(50);
    expect(result.current.isValid).toBe(false);

    act(() => result.current.setNumFrames(50, true)); // stepper/slider snaps back onto the grid
    expect(result.current.values.numFrames).toBe(49);
    expect(result.current.isValid).toBe(true);
  });

  describe("applyPreset", () => {
    it("loads width/height/num_frames from the named preset", () => {
      const { result } = renderHook(() => useGenerationForm(FALLBACK_APP_CONFIG, "x"));

      act(() => result.current.applyPreset("standard_720p"));

      expect(result.current.values).toMatchObject({ width: 1280, height: 768, numFrames: 361 });
    });

    it("reflects the preset's own crop_output (N1, Gradio-faithful)", () => {
      const { result } = renderHook(() => useGenerationForm(FALLBACK_APP_CONFIG, "x"));

      // standard_720p: { width: 1280, height: 768, crop_output: { width: 1280, height: 720 } }
      act(() => result.current.applyPreset("standard_720p"));
      expect(result.current.cropOutput).toEqual({ width: 1280, height: 720 });
    });

    it("turns crop OFF (null) when the preset defines none, clearing a previously-set crop", () => {
      const { result } = renderHook(() => useGenerationForm(FALLBACK_APP_CONFIG, "x"));

      act(() => result.current.applyPreset("standard_720p")); // has a crop_output
      expect(result.current.cropOutput).not.toBeNull();

      act(() => result.current.applyPreset("minimal")); // crop_output: null
      expect(result.current.cropOutput).toBeNull();
    });

    it("clamps the preset's crop_output against the ROUNDED post-apply generation size, not the preset's raw width/height (IC-LoRA 128 grid shrinks it)", async () => {
      // Pin the config maxima to 1920x1088 (not the real-server 4096, which is
      // itself a clean 128-multiple) so the IC-LoRA floor is observable: 1088
      // is not a multiple of 128, so it floors to 1024 and the demonstration
      // below (crop clamped against the ROUNDED size) is exercised.
      const config: AppConfig = {
        ...FALLBACK_APP_CONFIG,
        limits: { ...FALLBACK_APP_CONFIG.limits, max_width: 1920, max_height: 1088 },
      };
      // Y1: IC-LoRA mode is now driven by an ATTACHED reference video, not a
      // mount-frozen `isICLora` intent flag. A prompt STYLE tag supplies the
      // loRA N3 requires alongside the reference video, so `isValid` can reach
      // true once the video is ready (a control-LoRA-only inject would keep
      // `controlLoraNeedsReferenceVideo` false only after the video attaches too,
      // but the STYLE tag is the simpler both-conditions route).
      const mockBridge = createMockBridge({ delayMs: 0 });
      const { result } = renderHook(() =>
        useGenerationForm(config, "a cat <lora:Pixar_Toon:1.0>", { nativeBridge: mockBridge }),
      );

      await act(async () => {
        await result.current.referenceVideo.pick();
      });
      await waitFor(() => expect(result.current.referenceVideo.state.status).toBe("ready"));
      expect(result.current.isICLora).toBe(true);

      // FHD_1080p: { width: 1920, height: 1088, crop_output: { width: 1920, height: 1080 } }.
      // In IC-LoRA mode (128 grid), 1088 rounds/clamps down to 1024 (floored
      // max), so a crop clamped against the RAW 1088 (1080 <= 1088, no-op)
      // would leave cropOutput.height (1080) bigger than the actual applied
      // generation height (1024) — permanently invalid. It must instead be
      // clamped against 1024.
      act(() => result.current.applyPreset("FHD_1080p"));

      expect(result.current.values.height).toBe(1024);
      expect(result.current.cropOutput).toEqual({ width: 1920, height: 1024 });
      // The self-clamped crop never blocks Generate on its own.
      expect(result.current.isValid).toBe(true);
    });
  });

  it("exposes the spill-free frame threshold and warns once numFrames exceeds it", () => {
    const { result } = renderHook(() => useGenerationForm(FALLBACK_APP_CONFIG, "x"));

    // Default 1280x768 resolves exactly to the 1280x768 spill_free_frames entry.
    expect(result.current.spillThresholdFrames).toBe(
      resolveSpillFreeFrames(FALLBACK_APP_CONFIG.limits.spill_free_frames, 1280, 768),
    );
    expect(result.current.spillThresholdFrames).toBe(273);
    // Preset num_frames (361, raised 2026-08-19 to the SMART comfort ceiling —
    // see `comfortTable.comfortFramesForBudget`) now itself sits ABOVE the
    // coarse spill_free_frames threshold (273, re-measured 2026-08-31); the two
    // are no longer expected to coincide the way they did before the raise.
    expect(result.current.isOverSpillThreshold).toBe(true);

    act(() => result.current.setNumFrames(273)); // exactly at threshold: not over
    expect(result.current.isOverSpillThreshold).toBe(false);

    act(() => result.current.setNumFrames(300));
    expect(result.current.isOverSpillThreshold).toBe(true);
  });

  describe("smart comfort marker (2026-08-18, served as a table since 2026-08-31)", () => {
    // All five Acceleration toggles at the calibrated all-on configuration —
    // exactly what the served `comfort_budgets.ltx` row's `requires` demands;
    // see `accelerationSettings.effectiveAccelerationFields` for why every one
    // of these must hold and how sage's 3-valued availability plays in.
    const FULL_ACCELERATION: AccelerationSettings = {
      attentionBackend: "sage",
      blockSwapPrefetch: true,
      keepResident: true,
      fusedGgufDequantKernel: true,
      vaeMode: "prune_vaed",
      // 台帳 §3-114: at its server default on purpose — no served row's
      // `requires` map names `keep_resident_embeddings`, so it takes no part in
      // the match, and the calibration behind these rows is the five above.
      keepResidentEmbeddings: false,
    };

    /** LTX 2.3's engine family — the id whose only served row is the all-on
     * one above (its DEFAULT configuration deliberately has no row). */
    const LTX = "ltx";

    it("switches to the smart per-resolution ceiling when all five toggles are on, and follows resolution changes", () => {
      const { result } = renderHook(() =>
        useGenerationForm(FALLBACK_APP_CONFIG, "x", {
          acceleration: FULL_ACCELERATION,
          sageAvailable: true,
          engineFamily: LTX,
        }),
      );

      // Default 1280x768.
      expect(result.current.isComfortMarkerSmart).toBe(true);
      expect(result.current.spillThresholdFrames).toBe(361);

      act(() => {
        result.current.setWidth(1920);
        result.current.setHeight(1088);
      });
      expect(result.current.spillThresholdFrames).toBe(169);

      act(() => {
        result.current.setWidth(2560);
        result.current.setHeight(1472);
      });
      expect(result.current.spillThresholdFrames).toBe(89);
    });

    it("falls back to the coarse spill_free_frames marker the instant even one toggle is off", () => {
      const oneOff: AccelerationSettings = { ...FULL_ACCELERATION, vaeMode: "default" };
      const { result } = renderHook(() =>
        useGenerationForm(FALLBACK_APP_CONFIG, "x", {
          acceleration: oneOff,
          sageAvailable: true,
          engineFamily: LTX,
        }),
      );
      expect(result.current.isComfortMarkerSmart).toBe(false);
      // Default 1280x768 resolves the fallback map's exact entry.
      expect(result.current.spillThresholdFrames).toBe(273);
    });

    it("uses the legacy table for LTX 2.3's DEFAULT (all-off) configuration — intentionally no smart row", () => {
      // 2026-08-31: LTX 2.3's plain VAE decoder makes the comfort boundary
      // non-monotone in token count (the boundaries line up with the decoder's
      // chunk-count steps), so no token line describes it and the re-measured
      // `spill_free_frames` points are the truth. A `requires: {}` row for
      // `ltx` would be a regression, not an improvement.
      const { result } = renderHook(() =>
        useGenerationForm(FALLBACK_APP_CONFIG, "x", {
          acceleration: ACCELERATION_DEFAULTS,
          sageAvailable: true,
          engineFamily: LTX,
        }),
      );
      expect(result.current.isComfortMarkerSmart).toBe(false);
      expect(result.current.spillThresholdFrames).toBe(273);
    });

    it("is smart on LTX 2.5 even with every toggle off — its served row is unconditional", () => {
      const { result } = renderHook(() =>
        useGenerationForm(FALLBACK_APP_CONFIG, "x", {
          acceleration: ACCELERATION_DEFAULTS,
          sageAvailable: false,
          engineFamily: "ltx25",
        }),
      );
      expect(result.current.isComfortMarkerSmart).toBe(true);
      expect(result.current.spillThresholdFrames).toBe(361);
    });

    it("treats an explicit sageAvailable=false as sage not really running (falls back), and null as unknown (stays smart)", () => {
      const { result: withFalse } = renderHook(() =>
        useGenerationForm(FALLBACK_APP_CONFIG, "x", {
          acceleration: FULL_ACCELERATION,
          sageAvailable: false,
          engineFamily: LTX,
        }),
      );
      expect(withFalse.current.isComfortMarkerSmart).toBe(false);
      expect(withFalse.current.spillThresholdFrames).toBe(273);

      const { result: withNull } = renderHook(() =>
        useGenerationForm(FALLBACK_APP_CONFIG, "x", {
          acceleration: FULL_ACCELERATION,
          sageAvailable: null,
          engineFamily: LTX,
        }),
      );
      expect(withNull.current.isComfortMarkerSmart).toBe(true);
      expect(withNull.current.spillThresholdFrames).toBe(361);
    });

    it("falls to the compatibility shim on a backend that publishes no comfort_budgets table at all", () => {
      // Old backend, new WebUI: no table, so `resolveComfortRow` applies its
      // shim — the pre-2026-08-31 rule verbatim (smart ONLY while all five
      // toggles are on, budget from the legacy scalar key), and the engine
      // family is not consulted at all.
      const { comfort_budgets: _dropTable, ...limitsWithoutTable } = FALLBACK_APP_CONFIG.limits;
      const config: AppConfig = { ...FALLBACK_APP_CONFIG, limits: limitsWithoutTable };

      const { result: allOn } = renderHook(() =>
        useGenerationForm(config, "x", { acceleration: FULL_ACCELERATION, sageAvailable: true, engineFamily: LTX }),
      );
      expect(allOn.current.isComfortMarkerSmart).toBe(true);
      expect(allOn.current.spillThresholdFrames).toBe(361);

      const oneOff: AccelerationSettings = { ...FULL_ACCELERATION, vaeMode: "default" };
      const { result: notAllOn } = renderHook(() =>
        useGenerationForm(config, "x", { acceleration: oneOff, sageAvailable: true, engineFamily: LTX }),
      );
      expect(notAllOn.current.isComfortMarkerSmart).toBe(false);
      expect(notAllOn.current.spillThresholdFrames).toBe(273);

      // ...and with the legacy scalar key gone too, the shim mirrors 44,880.
      const { single_comfort_token_budget: _dropKey, ...bareLimits } = limitsWithoutTable;
      const bareConfig: AppConfig = { ...FALLBACK_APP_CONFIG, limits: bareLimits };
      const { result: bare } = renderHook(() =>
        useGenerationForm(bareConfig, "x", {
          acceleration: FULL_ACCELERATION,
          sageAvailable: true,
          engineFamily: LTX,
        }),
      );
      expect(bare.current.isComfortMarkerSmart).toBe(true);
      expect(bare.current.spillThresholdFrames).toBe(361);
    });

    it("clamps the smart marker to maxNumFrames at a low resolution (matches the fallback value at that point)", () => {
      const { result } = renderHook(() =>
        useGenerationForm(FALLBACK_APP_CONFIG, "x", {
          acceleration: FULL_ACCELERATION,
          sageAvailable: true,
          engineFamily: LTX,
        }),
      );
      act(() => {
        result.current.setWidth(512);
        result.current.setHeight(320);
      });
      // Owner target (plan §5 目視⑩): the marker stays at 481 regardless of
      // full-acceleration state at this resolution — only the warning COPY
      // differs (`GenerationForm`'s ternary), not the value.
      expect(result.current.isComfortMarkerSmart).toBe(true);
      expect(result.current.spillThresholdFrames).toBe(481);
    });

    it("moves isOverSpillThreshold's boundary to the smart 361/369 line instead of the fallback 273", () => {
      const { result } = renderHook(() =>
        useGenerationForm(FALLBACK_APP_CONFIG, "x", {
          acceleration: FULL_ACCELERATION,
          sageAvailable: true,
          engineFamily: LTX,
        }),
      );
      act(() => result.current.setNumFrames(361));
      expect(result.current.isOverSpillThreshold).toBe(false);
      act(() => result.current.setNumFrames(369));
      expect(result.current.isOverSpillThreshold).toBe(true);
    });

    it("regression: reads the matched row's SINGLE budget, not its chain budget", () => {
      // Misreading the CHAIN budget (40,000) here would compute 321 at
      // 1280x768 instead of 361 — a mistake TypeScript's structural typing
      // cannot catch (both are plain `number` fields on the same row), so this
      // has to be an explicit runtime assertion.
      const config: AppConfig = {
        ...FALLBACK_APP_CONFIG,
        limits: {
          ...FALLBACK_APP_CONFIG.limits,
          single_comfort_token_budget: 44880,
          chain_comfort_token_budget: 40000,
        },
      };
      const { result } = renderHook(() =>
        useGenerationForm(config, "x", {
          acceleration: FULL_ACCELERATION,
          sageAvailable: true,
          engineFamily: LTX,
        }),
      );
      expect(result.current.spillThresholdFrames).toBe(361);
      expect(result.current.spillThresholdFrames).not.toBe(321);
    });
  });

  it("exposes a duration estimate consistent with estimateGenerationSeconds for the current values", () => {
    const { result } = renderHook(() => useGenerationForm(FALLBACK_APP_CONFIG, "x"));

    const expected = estimateGenerationSeconds(
      result.current.values.width,
      result.current.values.height,
      result.current.values.numFrames,
    );
    expect(result.current.estimateSeconds).toBeCloseTo(expected, 5);
    expect(result.current.estimateLabel).toMatch(/^Est\. ~/);
  });

  it("getSizeFromAviUtl2 rounds the edit's resolution UP to the nearest 64 (never down)", async () => {
    // 1920x1080 edit: width is already a multiple of 64, height (1080) is not
    // and must round up to 1088, not down to 1024.
    const mockBridge = createMockBridge({ delayMs: 0, editInfo: { width: 1920, height: 1080 } });
    const { result } = renderHook(() => useGenerationForm(FALLBACK_APP_CONFIG, "x", { nativeBridge: mockBridge }));

    act(() => {
      void result.current.getSizeFromAviUtl2();
    });

    await waitFor(() => {
      expect(result.current.gettingSize).toBe(false);
    });

    expect(result.current.values.width).toBe(1920);
    expect(result.current.values.height).toBe(1088);
    expect(result.current.getSizeError).toBeNull();
  });

  it("getSizeFromAviUtl2 surfaces an error when there is no active edit session", async () => {
    const mockBridge = createMockBridge({ delayMs: 0, failEditInfo: true });
    const { result } = renderHook(() => useGenerationForm(FALLBACK_APP_CONFIG, "x", { nativeBridge: mockBridge }));

    act(() => {
      void result.current.getSizeFromAviUtl2();
    });

    await waitFor(() => {
      expect(result.current.gettingSize).toBe(false);
    });

    expect(result.current.getSizeError).not.toBeNull();
  });

  it("seeds width/height from the initial override (right-click prefill)", () => {
    const { result } = renderHook(() =>
      useGenerationForm(FALLBACK_APP_CONFIG, "x", {}, { width: 832, height: 448 }),
    );
    expect(result.current.values).toMatchObject({ width: 832, height: 448 });
    expect(result.current.isICLora).toBe(false);
  });

  it("in IC-LoRA mode snaps width/height to multiples of 128 and floors the max onto that grid (1088 -> 1024)", () => {
    // Pin the config maxima to 1920x1088 so the 128-grid floor is observable
    // (the real-server 4096 is itself a clean 128-multiple, floors to a no-op).
    const config: AppConfig = {
      ...FALLBACK_APP_CONFIG,
      limits: { ...FALLBACK_APP_CONFIG.limits, max_width: 1920, max_height: 1088 },
    };
    // Y1: IC-LoRA mode is now dynamic — a selected control LoRA (which mandates a
    // reference video) puts the form on the 128 grid without needing an actual
    // upload here (this test doesn't assert isValid). `renderFormWithControlLora`
    // is REQUIRED: a bare `renderHook`'s `setControlLora` is a no-op (AppShell
    // owns the selection state), so the injection would silently do nothing.
    const { result } = renderFormWithControlLora("x", config);
    act(() => result.current.setControlLora({ name: "canny-control", strength: 1.0 }));

    expect(result.current.isICLora).toBe(true);
    // Exposed max_height (1088) is not a clean multiple of 128 -> floored to 1024.
    expect(result.current.limits.maxHeight).toBe(1024);

    // Requesting the full 1088 must clamp to 1024, not round up past the ceiling.
    act(() => result.current.setHeight(1088, true));
    expect(result.current.values.height).toBe(1024);

    // Arbitrary values snap to the nearest 128 (snap path).
    act(() => result.current.setWidth(1000, true));
    expect(result.current.values.width).toBe(1024);
    expect(result.current.values.width % 128).toBe(0);

    act(() => result.current.setHeight(200, true));
    expect(result.current.values.height).toBe(256);
    expect(result.current.values.height % 128).toBe(0);
  });

  it("in IC-LoRA mode, free manual entry is passed through and gated by the 128 grid in isValid", async () => {
    // Y1: IC-LoRA mode now comes from an attached reference video (+ a loRA for
    // N3), not a mount `isICLora` flag. A prompt STYLE tag supplies the loRA so
    // `isValid` can reach true once the reference video is ready.
    const mockBridge = createMockBridge({ delayMs: 0 });
    const { result } = renderHook(() =>
      useGenerationForm(FALLBACK_APP_CONFIG, "a cat <lora:Pixar_Toon:1.0>", { nativeBridge: mockBridge }),
    );

    await act(async () => {
      await result.current.referenceVideo.pick();
    });
    await waitFor(() => expect(result.current.referenceVideo.state.status).toBe("ready"));

    // Clear the default crop so this test isolates the 128-grid gate (shrinking
    // width below the default crop_output width would otherwise block submit
    // for an unrelated crop reason).
    act(() => result.current.setCropOutput(null));

    expect(result.current.isICLora).toBe(true);
    // Seeded on-grid (128 multiples) -> submittable.
    expect(result.current.isValid).toBe(true);

    // An off-128-grid value is passed through unsnapped and blocks submit.
    act(() => result.current.setWidth(500, false));
    expect(result.current.values.width).toBe(500);
    expect(result.current.isValid).toBe(false);

    // Snapping onto the 128 grid re-enables submit (500 -> 512).
    act(() => result.current.setWidth(500, true));
    expect(result.current.values.width).toBe(512);
    expect(result.current.values.width % 128).toBe(0);
    expect(result.current.isValid).toBe(true);
  });

  it("toGenerateRequest omits reference_video_id until a reference video is uploaded, then includes it (IC-LoRA)", async () => {
    const mockBridge = createMockBridge({ delayMs: 0 });
    const { result } = renderHook(() =>
      // A LoRA tag is required alongside the reference video (N3's loras
      // gate) — see the two tests below for the gate itself.
      useGenerationForm(FALLBACK_APP_CONFIG, "a cat <lora:Pixar_Toon:1.0>", { nativeBridge: mockBridge }, { isICLora: true }),
    );

    // Nothing uploaded yet -> the field is absent entirely.
    expect(result.current.toGenerateRequest()).not.toHaveProperty("reference_video_id");

    await act(async () => {
      await result.current.referenceVideo.pick();
    });

    await waitFor(() => {
      expect(result.current.referenceVideo.state.status).toBe("ready");
    });

    const request = result.current.toGenerateRequest();
    expect(request.reference_video_id).toBe(result.current.referenceVideo.state.id);
    expect(request.reference_video_id).toBeTruthy();
  });

  describe("N3: reference-video loras gate", () => {
    it("omits reference_video_id (and the strength fields) once the reference video is ready but the prompt has no loras — the server 422s otherwise", async () => {
      const mockBridge = createMockBridge({ delayMs: 0 });
      const { result } = renderHook(() =>
        useGenerationForm(FALLBACK_APP_CONFIG, "a cat", { nativeBridge: mockBridge }, { isICLora: true }),
      );

      await act(async () => {
        await result.current.referenceVideo.pick();
      });
      await waitFor(() => expect(result.current.referenceVideo.state.status).toBe("ready"));

      expect(result.current.referenceVideoNeedsLoras).toBe(true);
      expect(result.current.isReferenceValid).toBe(false);
      expect(result.current.isValid).toBe(false);

      const request = result.current.toGenerateRequest();
      expect(request).not.toHaveProperty("reference_video_id");
      expect(request).not.toHaveProperty("conditioning_attention_strength");
      expect(request).not.toHaveProperty("reference_video_strength");
    });

    it("conditioningAttentionStrength/referenceVideoStrength default to null and are sent only once set, alongside a ready + lora'd reference video", async () => {
      const mockBridge = createMockBridge({ delayMs: 0 });
      const { result } = renderHook(() =>
        useGenerationForm(FALLBACK_APP_CONFIG, "a cat <lora:Pixar_Toon:1.0>", { nativeBridge: mockBridge }, { isICLora: true }),
      );
      expect(result.current.conditioningAttentionStrength).toBeNull();
      expect(result.current.referenceVideoStrength).toBeNull();

      await act(async () => {
        await result.current.referenceVideo.pick();
      });
      await waitFor(() => expect(result.current.referenceVideo.state.status).toBe("ready"));

      expect(result.current.toGenerateRequest()).not.toHaveProperty("conditioning_attention_strength");

      act(() => result.current.setConditioningAttentionStrength(0.7));
      act(() => result.current.setReferenceVideoStrength(0));

      const request = result.current.toGenerateRequest();
      expect(request.conditioning_attention_strength).toBe(0.7);
      expect(request.reference_video_strength).toBe(0); // 0 is a valid, distinct value
    });
  });

  describe("IC-LoRA UI redesign (第5波): state-owned control LoRA <-> reference-video gate", () => {
    it("controlLora defaults to null and setControlLoraStrength is a no-op while it is", () => {
      const { result } = renderHook(() => useGenerationForm(FALLBACK_APP_CONFIG, "a cat"));
      expect(result.current.controlLora).toBeNull();
      expect(() => act(() => result.current.setControlLoraStrength(0.5))).not.toThrow();
      expect(result.current.controlLora).toBeNull();
    });

    it("controlLoraNames falls back to config.model.ic_loras' keys when not injected", () => {
      const configWithIcLoras = {
        ...FALLBACK_APP_CONFIG,
        model: { ic_loras: { "canny-control": { path: "x", preprocess: "canny" as const } } },
      };
      const { result } = renderHook(() => useGenerationForm(configWithIcLoras, "a cat"));
      expect(result.current.controlLoraNames).toEqual(new Set(["canny-control"]));
    });

    it("a selected control LoRA (state, not a prompt tag) requires a reference video before it's valid", () => {
      const { result } = renderFormWithControlLora("a cat");

      act(() => result.current.setControlLora({ name: "canny-control", strength: 1.0 }));

      expect(result.current.controlLoraNeedsReferenceVideo).toBe(true);
      expect(result.current.isReferenceValid).toBe(false);
      expect(result.current.isValid).toBe(false);
      // Selecting it never writes a prompt tag — the prompt argument stays
      // whatever the caller passed in, untouched by this hook.
      expect(result.current.toGenerateRequest().prompt).toBe("a cat");
    });

    it("is valid once the reference video is attached for a selected control LoRA — no style tag needed", async () => {
      const mockBridge = createMockBridge({ delayMs: 0 });
      const { result } = renderFormWithControlLora("a cat", FALLBACK_APP_CONFIG, { nativeBridge: mockBridge });

      act(() => result.current.setControlLora({ name: "canny-control", strength: 1.0 }));

      await act(async () => {
        await result.current.referenceVideo.pick();
      });
      await waitFor(() => expect(result.current.referenceVideo.state.status).toBe("ready"));

      // The control selection alone satisfies N3's "reference video needs at
      // least one loRA" rule — no separate style tag is required.
      expect(result.current.referenceVideoNeedsLoras).toBe(false);
      expect(result.current.controlLoraNeedsReferenceVideo).toBe(false);
      expect(result.current.isReferenceValid).toBe(true);
      expect(result.current.isValid).toBe(true);
    });

    it("toGenerateRequest merges the control selection FIRST, then the prompt's style tags", async () => {
      const mockBridge = createMockBridge({ delayMs: 0 });
      const { result } = renderFormWithControlLora("a cat <lora:Pixar_Toon:1.0>", FALLBACK_APP_CONFIG, {
        nativeBridge: mockBridge,
      });

      act(() => result.current.setControlLora({ name: "canny-control", strength: 0.8 }));
      await act(async () => {
        await result.current.referenceVideo.pick();
      });
      await waitFor(() => expect(result.current.referenceVideo.state.status).toBe("ready"));

      const request = result.current.toGenerateRequest();
      expect(request.loras).toEqual([
        { name: "canny-control", strength: 0.8 },
        { name: "Pixar_Toon", strength: 1.0 },
      ]);
      expect(request.prompt).toBe("a cat"); // the style tag is still stripped from the body
      expect(request.reference_video_id).toBe(result.current.referenceVideo.state.id);
    });

    it("setControlLoraStrength clamps to [LORA_STRENGTH_MIN, LORA_STRENGTH_MAX]", () => {
      const { result } = renderFormWithControlLora("a cat");
      act(() => result.current.setControlLora({ name: "canny-control", strength: 1.0 }));

      act(() => result.current.setControlLoraStrength(10));
      expect(result.current.controlLora?.strength).toBe(2.0);

      act(() => result.current.setControlLoraStrength(-5));
      expect(result.current.controlLora?.strength).toBe(0.05);
    });
  });

  describe("A2V (Group3 item11)", () => {
    it("attaching a wav flips isA2v and auto-adjusts numFrames to the audio length", async () => {
      const { result } = setupA2V(2.0);
      expect(result.current.isA2v).toBe(false);
      expect(result.current.audioFramesAdjustedEvent).toBeNull();

      act(() => {
        void result.current.sourceAudio.pick();
      });
      await waitFor(() => expect(result.current.sourceAudio.state.status).toBe("ready"));
      expect(result.current.isA2v).toBe(true);

      const expected = suggestFramesForAudio(2.0, result.current.values.frameRate);
      await waitFor(() => expect(result.current.values.numFrames).toBe(expected));
      expect(result.current.audioFramesAdjustedEvent).toEqual({ id: 1, frames: expected, durationSec: 2.0 });
      expect(result.current.audioPrecheck?.ok).toBe(true);
    });

    // W8: a long wav's auto-adjust suggestion is capped at the current
    // resolution's comfort ceiling — the SAME `spillThresholdFrames` the
    // comfort marker draws (2026-08-31), not just the config max.
    it("caps the wav auto-adjust at the resolution's comfort ceiling (W8)", async () => {
      // 30s @ 24fps -> suggestFramesForAudio floors to 713, which even after the
      // max clamp (481) is well ABOVE 1280x768's comfort ceiling (273). W8 clamps
      // the suggestion down to that ceiling.
      const raw = suggestFramesForAudio(30, 24);
      expect(raw).toBeGreaterThan(273);
      const { result } = setupA2V(30);
      act(() => {
        void result.current.sourceAudio.pick();
      });
      await waitFor(() => expect(result.current.sourceAudio.state.status).toBe("ready"));

      // Default resolution 1280x768 -> spill_free_frames ceiling 273.
      await waitFor(() => expect(result.current.values.numFrames).toBe(273));
      expect(result.current.audioFramesAdjustedEvent?.frames).toBe(273);
    });

    // 2026-08-31: the clamp reads the marker's own value, so the two can never
    // disagree. These pin BOTH branches of the marker against the SAME hook
    // result rather than against a hard-coded number.
    it("clamps the wav auto-adjust to exactly spillThresholdFrames, on the legacy branch and the smart one", async () => {
      // Legacy branch: LTX 2.3's default configuration has no served row.
      const { result: legacy } = setupA2V(30, "a music video", { engineFamily: "ltx" });
      act(() => {
        void legacy.current.sourceAudio.pick();
      });
      await waitFor(() => expect(legacy.current.sourceAudio.state.status).toBe("ready"));
      await waitFor(() => expect(legacy.current.values.numFrames).toBe(legacy.current.spillThresholdFrames));
      expect(legacy.current.isComfortMarkerSmart).toBe(false);
      expect(legacy.current.values.numFrames).toBe(273);

      // Smart branch: LTX 2.5's served row is unconditional, so the same wav
      // auto-adjusts to the smart 361 instead — matching what the marker shows.
      const { result: smart } = setupA2V(30, "a music video", { engineFamily: "ltx25" });
      act(() => {
        void smart.current.sourceAudio.pick();
      });
      await waitFor(() => expect(smart.current.sourceAudio.state.status).toBe("ready"));
      await waitFor(() => expect(smart.current.values.numFrames).toBe(smart.current.spillThresholdFrames));
      expect(smart.current.isComfortMarkerSmart).toBe(true);
      expect(smart.current.values.numFrames).toBe(361);
    });

    // W4 (#7 trim追従): an audio-to-video right-click prefill uploads the WHOLE
    // audio file, so the wav auto-adjust is additionally capped at the span
    // (trimmed ribbon) DURATION the prefill seeded (`a2vSeedMaxFrames`) — but
    // only while the attached audio is still the prefill-attached one. Swapping
    // the audio lifts the cap and restores the ordinary full-wav follow.
    it("caps the #7 prefill wav auto-adjust at a2vSeedMaxFrames, and lifts the cap once the audio is swapped", async () => {
      const SPAN_CAP = 121;
      const mockFs = createMockFs({
        folders: {
          // The prefill-attached audio: a long (30s) file whose raw suggestion is
          // far above the span cap (and even above the 273 spill ceiling).
          "C:\\a": [{ name: "voice.wav", sizeBytes: 4_096, mtimeMs: 0, durationSec: 30 }],
          // The user's replacement pick (8s), well under the spill ceiling.
          "C:\\Users\\mock\\Pictures": [{ name: "picked-audio-1.wav", sizeBytes: 4_096, mtimeMs: 0, durationSec: 8 }],
        },
      });
      const mockBridge = createMockBridge({ delayMs: 0, fs: mockFs });
      const { result } = renderHook(() =>
        useGenerationForm(FALLBACK_APP_CONFIG, "a music video", { nativeBridge: mockBridge }, { a2vSeedMaxFrames: SPAN_CAP }),
      );

      // Attach the prefill audio (the #7 auto-load path). Its raw 30s suggestion
      // is held down to the span cap, not the full-file length.
      await act(async () => {
        await result.current.attachSourceAudioByPath("C:\\a\\voice.wav", "voice.wav");
      });
      await waitFor(() => expect(result.current.sourceAudio.state.status).toBe("ready"));
      await waitFor(() => expect(result.current.values.numFrames).toBe(SPAN_CAP));
      expect(result.current.audioFramesAdjustedEvent?.frames).toBe(SPAN_CAP);

      // Swap the audio via a normal pick -> a DIFFERENT path: the cap lifts and
      // the ordinary full-wav follow resumes (the 8s suggestion is above the cap).
      const uncapped = suggestFramesForAudio(8, result.current.values.frameRate);
      expect(uncapped).toBeGreaterThan(SPAN_CAP);
      act(() => {
        void result.current.sourceAudio.pick();
      });
      await waitFor(() => expect(result.current.values.numFrames).toBe(uncapped));
    });

    it("buildA2vRequest assembles a distilled single-clip A2V chain payload", async () => {
      const { result } = setupA2V(2.0);
      act(() => {
        void result.current.sourceAudio.pick();
      });
      await waitFor(() => expect(result.current.isA2v).toBe(true));

      const payload = result.current.buildA2vRequest([]);
      expect(payload.pipeline).toBe("distilled");
      expect(payload.num_inference_steps).toBe(8);
      expect(payload.guidance_scale).toBe(1.0);
      expect(payload.overlap_frames).toBe(3);
      expect(payload.overlap_strength).toBe(0.5);
      expect(payload.chunked_upsample).toBe(true);
      // §1-19 (2026-08-11): Single's A2V chain always requests the full-length
      // stage-2 window (1 clip = 1 tile, no seams).
      expect(payload.stage2_window).toBe("full_length");
      expect(payload.clips).toHaveLength(1);
      expect(payload.source_audio.audio_id).toBe(result.current.sourceAudio.state.id);
      expect(payload.source_audio.audio_id).toBeTruthy();
      // No reference-video / adapter fields on a plain A2V request.
      expect(payload).not.toHaveProperty("reference_video_id");
    });

    it("buildA2vRequest strips <lora:...> tags into loras[] and carries keyframes on the clip", async () => {
      const { result } = setupA2V(2.0, "a cat <lora:Pixar_Toon:1.5> dancing");
      act(() => {
        void result.current.sourceAudio.pick();
      });
      await waitFor(() => expect(result.current.isA2v).toBe(true));

      const conditioning = [{ image_id: "img-1", frame_idx: 0, strength: 1.0 }];
      const payload = result.current.buildA2vRequest(conditioning);
      expect(payload.prompt).toBe("a cat dancing");
      expect(payload.loras).toEqual([{ name: "Pixar_Toon", strength: 1.5 }]);
      expect(payload.clips[0].conditioning_images).toEqual(conditioning);
    });

    it("source audio and the IC-LoRA reference video are combinable (both isA2v and isICLora hold), carrying the reference fields when a loRA is present", async () => {
      const mockBridge = createMockBridge({ delayMs: 0 });
      const { result } = renderHook(() =>
        // A prompt STYLE tag supplies the loRA N3 requires alongside the
        // reference video, so `mergedLoras` is non-empty.
        useGenerationForm(FALLBACK_APP_CONFIG, "a cat <lora:Pixar_Toon:1.0>", { nativeBridge: mockBridge }),
      );

      // A reference video puts the form in IC-LoRA mode.
      await act(async () => {
        await result.current.referenceVideo.pick();
      });
      await waitFor(() => expect(result.current.referenceVideo.state.status).toBe("ready"));
      expect(result.current.isICLora).toBe(true);
      expect(result.current.isA2v).toBe(false);

      // Attaching audio now ALSO turns on A2V — the two coexist.
      act(() => {
        void result.current.sourceAudio.pick();
      });
      await waitFor(() => expect(result.current.sourceAudio.state.status).toBe("ready"));
      expect(result.current.isA2v).toBe(true);
      expect(result.current.isICLora).toBe(true);

      // The audio-driven chain payload carries the reference-video fields.
      const payload = result.current.buildA2vRequest([]);
      expect(payload.reference_video_id).toBe(result.current.referenceVideo.state.id);
      expect(payload.reference_video_id).toBeTruthy();
      expect(payload.loras).toEqual([{ name: "Pixar_Toon", strength: 1.0 }]);
    });

    it("a lora-less reference video + audio omits reference_video_id from the A2V payload (N3 gate)", async () => {
      const mockBridge = createMockBridge({ delayMs: 0 });
      const { result } = renderHook(() => useGenerationForm(FALLBACK_APP_CONFIG, "a cat", { nativeBridge: mockBridge }));

      await act(async () => {
        await result.current.referenceVideo.pick();
      });
      await waitFor(() => expect(result.current.referenceVideo.state.status).toBe("ready"));

      act(() => {
        void result.current.sourceAudio.pick();
      });
      await waitFor(() => expect(result.current.sourceAudio.state.status).toBe("ready"));
      expect(result.current.isA2v).toBe(true);
      expect(result.current.isICLora).toBe(true);

      // No loRA in the merged set -> N3 gate keeps the reference fields off the
      // audio-driven payload (the server 422s `reference_video_id` without a loRA).
      const payload = result.current.buildA2vRequest([]);
      expect(payload).not.toHaveProperty("reference_video_id");
    });

    // I7 通し(end-to-end): the exact theme-1 combo the workorder singles out —
    // source_audio + a reference video + a CONTROL LoRA only (no <lora:...>
    // STYLE tag in the prompt). Builds the A2V chain payload the way the Create
    // form does, then dispatches it through the mock bridge's
    // `handleGenerateChain`, polls to completed, and asserts the echoed
    // `GET /jobs` request is the whitelisted per-clip shape (source_audio /
    // source_video / clips / overlap_* stripped, loras []/reference_video_id
    // null per `to_clip_request`, is_v2v false because it's audio- not
    // video-driven). This is the "test-green / device-invisible" guard applied
    // to the combo path: it proves the request the server actually accepts and
    // echoes matches the contract, end to end.
    it("E2E: a control-LoRA-only A2V+reference combo posts a 1-clip chain that completes with a whitelisted echoed request", async () => {
      const mockBridge = createMockBridge({ delayMs: 0 });
      const apiClient = createApiClient(mockBridge);
      // Prompt carries NO STYLE tag — the only loRA is the state-owned control
      // selection, so N3's "reference video needs a loRA" gate is satisfied by
      // the control LoRA alone.
      const { result } = renderFormWithControlLora("a cat", FALLBACK_APP_CONFIG, { nativeBridge: mockBridge });

      act(() => result.current.setControlLora({ name: "canny-control", strength: 1.0 }));

      await act(async () => {
        await result.current.referenceVideo.pick();
      });
      await waitFor(() => expect(result.current.referenceVideo.state.status).toBe("ready"));

      act(() => {
        void result.current.sourceAudio.pick();
      });
      await waitFor(() => expect(result.current.sourceAudio.state.status).toBe("ready"));
      expect(result.current.isA2v).toBe(true);
      expect(result.current.isICLora).toBe(true);

      // The merged set is control-only (no STYLE tag was written into the prompt).
      const payload = result.current.buildA2vRequest([]);
      expect(payload.loras).toEqual([{ name: "canny-control", strength: 1.0 }]);
      expect(payload.reference_video_id).toBe(result.current.referenceVideo.state.id);
      expect(payload.source_audio.audio_id).toBe(result.current.sourceAudio.state.id);
      expect(payload.source_audio.audio_id).toBeTruthy();
      expect(payload.clips).toHaveLength(1);
      // §1-19 (2026-08-11): full-length stage-2 window rides the combo path too.
      expect(payload.stage2_window).toBe("full_length");

      // Dispatch it exactly like the Create form does (A2V rides the chain
      // endpoint). The payload is a superset of GenerateChainRequest (it freezes
      // pipeline/steps/guidance) — the bridge posts the body verbatim.
      const accepted = await apiClient.generateChain(payload as unknown as GenerateChainRequest);
      expect(accepted.status).toBe("queued");
      expect(accepted.num_clips).toBe(1);

      let job = await apiClient.getJob(accepted.job_id);
      for (let i = 0; i < 20 && job.status !== "completed"; i += 1) {
        job = await apiClient.getJob(accepted.job_id);
      }
      expect(job.status).toBe("completed");

      // The echoed request is the whitelisted per-clip GenerateRequest — the
      // chain-only + adapter fields are gone, matching a real `to_clip_request`.
      const req = job.request;
      expect(req.source_audio).toBeUndefined();
      expect(req.source_video).toBeUndefined();
      expect(req.clips).toBeUndefined();
      expect(req.overlap_frames).toBeUndefined();
      expect(req.overlap_strength).toBeUndefined();
      expect(req.loras).toEqual([]);
      expect(req.reference_video_id).toBeNull();
      expect(req.prompt).toBe("a cat");
      expect(req.num_frames).toBe(payload.clips[0].num_frames);
      // Audio-driven, not video-driven: the Join gate must stay off.
      expect(job.is_v2v).toBe(false);
    });

    it("a too-short audio (numFrames grown past what it supports) fails audioPrecheck and blocks isValid", async () => {
      const { result } = setupA2V(2.0);
      act(() => {
        void result.current.sourceAudio.pick();
      });
      await waitFor(() => expect(result.current.isA2v).toBe(true));
      await waitFor(() => expect(result.current.audioPrecheck?.ok).toBe(true));
      expect(result.current.isValid).toBe(true);

      act(() => result.current.setNumFrames(FALLBACK_APP_CONFIG.limits.max_num_frames));
      expect(result.current.audioPrecheck?.ok).toBe(false);
      expect(result.current.audioPrecheck?.requiredSeconds).toBeGreaterThan(2.0);
      expect(result.current.isValid).toBe(false);
    });

    // Contract v7 (drag-and-drop): `GenerationForm`'s SourceAudioSection
    // DropZone calls `form.attachSourceAudioByPath` (never
    // `sourceAudio.uploadPath` directly). This asserts the wav-duration
    // auto-adjust probe fires for a drop exactly as it does for a pick —
    // since §3-36 both paths feed the probe the same
    // `sourceAudio.state.filePath`, so this is the regression guard for that
    // unification as much as for the drop path itself.
    it("attachSourceAudioByPath (drag-and-drop) auto-adjusts numFrames just like pick() does", async () => {
      const mockFs = createMockFs({
        folders: {
          "C:\\Users\\mock\\Downloads": [{ name: "dropped.wav", sizeBytes: 4_096, mtimeMs: 0, durationSec: 2.0 }],
        },
      });
      const mockBridge = createMockBridge({ delayMs: 0, fs: mockFs });
      const { result } = renderHook(() =>
        useGenerationForm(FALLBACK_APP_CONFIG, "a music video", { nativeBridge: mockBridge }),
      );
      expect(result.current.isA2v).toBe(false);

      await act(async () => {
        await result.current.attachSourceAudioByPath("C:\\Users\\mock\\Downloads\\dropped.wav", "dropped.wav");
      });

      await waitFor(() => expect(result.current.sourceAudio.state.status).toBe("ready"));
      expect(result.current.isA2v).toBe(true);

      const expected = suggestFramesForAudio(2.0, result.current.values.frameRate);
      await waitFor(() => expect(result.current.values.numFrames).toBe(expected));
      expect(result.current.audioFramesAdjustedEvent).toEqual({ id: 1, frames: expected, durationSec: 2.0 });
    });
  });

  // Contract v7 (drag-and-drop): GenerationForm's ReferenceVideoSection
  // DropZone calls `form.referenceVideo.uploadPath` directly (no new
  // function needed — see useGenerationForm.ts's comment on the resnap
  // effect). This confirms that path alone, without ever going through
  // `pick()`/`ui.pickFile`, still drives the 128-grid re-snap.
  it("referenceVideo.uploadPath (drag-and-drop path) alone still triggers the IC-LoRA 128-grid re-snap", async () => {
    const mockBridge = createMockBridge({ delayMs: 0 });
    // Pin a generation_defaults height that is NOT a multiple of 128 (the real
    // default 768 happens to be one), so the 128-grid re-snap is observable
    // once IC-LoRA activates.
    const config: AppConfig = {
      ...FALLBACK_APP_CONFIG,
      generation_defaults: {
        ...FALLBACK_APP_CONFIG.generation_defaults,
        width: 512,
        height: 320,
        crop_output: null,
      },
    };
    const { result } = renderHook(() => useGenerationForm(config, "a cat", { nativeBridge: mockBridge }));

    expect(result.current.isICLora).toBe(false);
    // Default seed (config.generation_defaults): height 320 is NOT a multiple
    // of 128, so a real re-snap is observable once IC-LoRA activates.
    expect(result.current.values.height % 128).not.toBe(0);

    await act(async () => {
      await result.current.referenceVideo.uploadPath("C:\\Users\\mock\\Downloads\\ref.mp4", "ref.mp4");
    });

    await waitFor(() => expect(result.current.referenceVideo.state.status).toBe("ready"));
    expect(result.current.isICLora).toBe(true);
    expect(result.current.values.width % 128).toBe(0);
    expect(result.current.values.height % 128).toBe(0);
  });

  // Y1 (参照動画を外すとGenerate復帰不能になるバグの根治・案A): IC-LoRA mode is now
  // determined dynamically off the live reference video / control LoRA, not a
  // mount-frozen intent flag — so removing the reference video that a routed #2
  // auto-loaded relaxes the form back out of IC-LoRA mode and restores Generate.
  describe("Y1: dynamic IC-LoRA mode (removing the reference video restores Generate)", () => {
    it("a routed #2 (isICLora seed) drops out of IC-LoRA mode when the reference video is cleared, un-blocking isValid", async () => {
      const mockBridge = createMockBridge({ delayMs: 0 });
      // Routed #2: mounts with the isICLora seed (128-grid width/height) and a
      // valid prompt but no loRA.
      const { result } = renderHook(() =>
        useGenerationForm(FALLBACK_APP_CONFIG, "a cat", { nativeBridge: mockBridge }, { isICLora: true }),
      );
      // Clear the default crop so the isValid assertions isolate the IC-LoRA gate.
      act(() => result.current.setCropOutput(null));

      // Attach the reference video (the #2 auto-load). Now genuinely in IC-LoRA
      // mode; the lora-less prompt makes referenceNeedsLoras block Generate.
      await act(async () => {
        await result.current.referenceVideo.pick();
      });
      await waitFor(() => expect(result.current.referenceVideo.state.status).toBe("ready"));
      expect(result.current.isICLora).toBe(true);
      expect(result.current.validityReasons).toContain("referenceNeedsLoras");
      expect(result.current.isValid).toBe(false);

      // Remove the reference video (🔁/❌, id -> null). Pre-Y1 the frozen intent
      // kept isICLora true forever, leaving referenceNeedsLoras + Generate stuck.
      // Now the mode relaxes and Generate is restored; the 128-seeded dimensions
      // stay valid on the relaxed 64 grid (no dimensionsOffGrid).
      act(() => result.current.referenceVideo.clear());
      await waitFor(() => expect(result.current.referenceVideo.state.status).not.toBe("ready"));
      expect(result.current.isICLora).toBe(false);
      expect(result.current.validityReasons).not.toContain("referenceNeedsLoras");
      expect(result.current.validityReasons).not.toContain("dimensionsOffGrid");
      expect(result.current.isValid).toBe(true);
    });
  });

  // W5 (反対スロット保持): the opposite source slot carried across a #2/#3/#7
  // remount is seeded into the form's one-shot `initial` (no re-upload).
  describe("W5 carry-over (initial.carryReferenceVideo / initial.carryAudio)", () => {
    // A config whose default height (320) is NOT a multiple of 128, so the
    // 回帰対策1 128-grid seed is observable.
    const off128Config: AppConfig = {
      ...FALLBACK_APP_CONFIG,
      generation_defaults: { ...FALLBACK_APP_CONFIG.generation_defaults, width: 512, height: 320, crop_output: null },
    };

    it("carrying a reference video (#3/#7) mounts it ready, forces the 128 grid, and seeds both strengths (回帰対策1)", () => {
      const mockBridge = createMockBridge({ delayMs: 0 });
      const spy = vi.spyOn(mockBridge, "request");
      const { result } = renderHook(() =>
        useGenerationForm(
          off128Config,
          "a cat",
          { nativeBridge: mockBridge },
          {
            carryReferenceVideo: { id: "carried-ref-1", fileName: "ref.mp4", filePath: "C:\\v\\ref.mp4" },
            carryConditioningAttentionStrength: 0.5,
            carryReferenceVideoStrength: 0.8,
          },
        ),
      );

      // Mounted ready with the carried id — no re-upload.
      expect(result.current.referenceVideo.state.status).toBe("ready");
      expect(result.current.referenceVideo.state.id).toBe("carried-ref-1");
      expect(spy).not.toHaveBeenCalledWith("backend.uploadFile", expect.anything());
      // 回帰対策1: IC-LoRA active from mount, width/height seeded on the 128 grid,
      // so `dimensionsOffGrid` never blocks Generate.
      expect(result.current.isICLora).toBe(true);
      expect(result.current.values.width % 128).toBe(0);
      expect(result.current.values.height % 128).toBe(0);
      expect(result.current.validityReasons).not.toContain("dimensionsOffGrid");
      // Strengths survived the carry.
      expect(result.current.conditioningAttentionStrength).toBe(0.5);
      expect(result.current.referenceVideoStrength).toBe(0.8);
    });

    it("a carried reference video with un-opted (null) strengths starts them at null", () => {
      const mockBridge = createMockBridge({ delayMs: 0 });
      const { result } = renderHook(() =>
        useGenerationForm(
          FALLBACK_APP_CONFIG,
          "a cat",
          { nativeBridge: mockBridge },
          {
            carryReferenceVideo: { id: "carried-ref-1", fileName: "ref.mp4", filePath: "C:\\v\\ref.mp4" },
            carryConditioningAttentionStrength: null,
            carryReferenceVideoStrength: null,
          },
        ),
      );
      expect(result.current.isICLora).toBe(true);
      expect(result.current.conditioningAttentionStrength).toBeNull();
      expect(result.current.referenceVideoStrength).toBeNull();
    });

    it("carrying a source audio (#2) mounts it ready and fires the mount-time wav auto-adjust (回帰対策2)", async () => {
      const mockFs = createMockFs({
        folders: {
          "C:\\a": [{ name: "voice.wav", sizeBytes: 4_096, mtimeMs: 0, durationSec: 2.0 }],
        },
      });
      const mockBridge = createMockBridge({ delayMs: 0, fs: mockFs });
      const spy = vi.spyOn(mockBridge, "request");
      const { result } = renderHook(() =>
        useGenerationForm(
          FALLBACK_APP_CONFIG,
          "a music video",
          { nativeBridge: mockBridge },
          { carryAudio: { id: "carried-audio-1", fileName: "voice.wav", filePath: "C:\\a\\voice.wav" } },
        ),
      );

      // Mounted ready with the carried id (A2V active) — no re-upload.
      expect(result.current.sourceAudio.state.status).toBe("ready");
      expect(result.current.sourceAudio.state.id).toBe("carried-audio-1");
      expect(result.current.isA2v).toBe(true);
      expect(spy).not.toHaveBeenCalledWith("backend.uploadFile", expect.anything());

      // 回帰対策2: the mount-time probe fires off the carried filePath and the
      // DURATION auto-adjusts to the audio (A2V DURATION priority preserved).
      const expected = suggestFramesForAudio(2.0, result.current.values.frameRate);
      await waitFor(() => expect(result.current.values.numFrames).toBe(expected));
      expect(result.current.audioFramesAdjustedEvent?.frames).toBe(expected);
    });
  });
});

// W7: `validityReasons` is the structured breakdown behind `isValid`. Each case
// breaks exactly one gate and asserts the matching code appears; the invariant
// `isValid === (validityReasons.length === 0)` is checked throughout.
describe("useGenerationForm — validityReasons (W7)", () => {
  it("is empty (and isValid true) when every gate passes", () => {
    const { result } = renderHook(() => useGenerationForm(FALLBACK_APP_CONFIG, "a cat riding a skateboard"));
    expect(result.current.validityReasons).toEqual([]);
    expect(result.current.isValid).toBe(true);
  });

  it("reports promptEmpty for a blank prompt", () => {
    const { result } = renderHook(() => useGenerationForm(FALLBACK_APP_CONFIG, ""));
    expect(result.current.validityReasons).toContain("promptEmpty");
    expect(result.current.isValid).toBe(result.current.validityReasons.length === 0);
    expect(result.current.isValid).toBe(false);
  });

  it("reports dimensionsOffGrid for an off-grid width, and clears it once re-snapped", () => {
    const { result } = renderHook(() => useGenerationForm(FALLBACK_APP_CONFIG, "a cat"));
    act(() => result.current.setCropOutput(null)); // isolate the dimension gate
    expect(result.current.validityReasons).toEqual([]);

    act(() => result.current.setWidth(500, false));
    expect(result.current.validityReasons).toContain("dimensionsOffGrid");
    expect(result.current.isValid).toBe(false);

    act(() => result.current.setWidth(500, true));
    expect(result.current.validityReasons).not.toContain("dimensionsOffGrid");
    expect(result.current.isValid).toBe(true);
  });

  it("reports numFramesOffGrid for an off-grid frame count", () => {
    const { result } = renderHook(() => useGenerationForm(FALLBACK_APP_CONFIG, "a cat"));
    act(() => result.current.setNumFrames(50, false)); // not 8n+1
    expect(result.current.validityReasons).toContain("numFramesOffGrid");
    expect(result.current.isValid).toBe(false);
  });

  it("reports cropInvalid when a crop is left larger than the current width/height", () => {
    const { result } = renderHook(() => useGenerationForm(FALLBACK_APP_CONFIG, "a cat"));
    // A full-size crop at 1280x768, then shrink width onto the grid to 640 —
    // the crop (1280) now exceeds it while width itself stays on-grid.
    act(() => result.current.setCropOutput({ width: 1280, height: 768 }));
    act(() => result.current.setWidth(640, true));
    expect(result.current.validityReasons).toContain("cropInvalid");
    expect(result.current.validityReasons).not.toContain("dimensionsOffGrid");
    expect(result.current.isValid).toBe(false);
  });

  it("reports controlNeedsReference when a control LoRA is selected with no reference video", () => {
    const { result } = renderFormWithControlLora("a cat");
    act(() => result.current.setControlLora({ name: "canny-control", strength: 1.0 }));
    expect(result.current.validityReasons).toContain("controlNeedsReference");
    expect(result.current.isValid).toBe(false);
  });

  it("reports referenceNeedsLoras when a reference video is ready but no LoRA is present", async () => {
    const mockBridge = createMockBridge({ delayMs: 0 });
    const { result } = renderHook(() =>
      useGenerationForm(FALLBACK_APP_CONFIG, "a cat", { nativeBridge: mockBridge }, { isICLora: true }),
    );
    await act(async () => {
      await result.current.referenceVideo.pick();
    });
    await waitFor(() => expect(result.current.referenceVideo.state.status).toBe("ready"));
    expect(result.current.validityReasons).toContain("referenceNeedsLoras");
    expect(result.current.validityReasons).not.toContain("controlNeedsReference");
    expect(result.current.isValid).toBe(false);
  });

  it("reports audioTooShort once numFrames outgrows the attached wav", async () => {
    const { result } = setupA2V(2.0);
    act(() => {
      void result.current.sourceAudio.pick();
    });
    await waitFor(() => expect(result.current.isA2v).toBe(true));
    await waitFor(() => expect(result.current.audioPrecheck?.ok).toBe(true));
    expect(result.current.validityReasons).not.toContain("audioTooShort");

    act(() => result.current.setNumFrames(FALLBACK_APP_CONFIG.limits.max_num_frames));
    expect(result.current.validityReasons).toContain("audioTooShort");
    expect(result.current.isValid).toBe(false);
  });

  // §1-6 拡張 (2026-08-01): #2 IC-LoRA の参照動画もリボン範囲で切り出すように
  // なったので、切り出しを頼んだのに `trimmed: true` が返らなかった場合は
  // Generate を止める。Chain 側 `sourceTrimFailed` と同型。
  describe("referenceTrimFailed (§1-6 拡張)", () => {
    it("reports referenceTrimFailed when a requested reference trim is not confirmed", async () => {
      const inner = createMockBridge({ delayMs: 0 });
      const { result } = renderHook(() =>
        useGenerationForm(FALLBACK_APP_CONFIG, "a cat", { nativeBridge: dropsTrimQuery(inner) }, { isICLora: true }),
      );

      await act(async () => {
        await result.current.referenceVideo.uploadPath("C:\\v\\ref.mp4", "ref.mp4", {
          trim_start_sec: "1.000",
          trim_duration_sec: "5.000",
        });
      });
      await waitFor(() => expect(result.current.referenceVideo.state.status).toBe("ready"));

      expect(result.current.referenceVideo.state.trimFailed).toBe(true);
      expect(result.current.validityReasons).toContain("referenceTrimFailed");
      expect(result.current.isValid).toBe(false);
    });

    it("does not fire when the trim IS confirmed", async () => {
      const mockBridge = createMockBridge({ delayMs: 0 });
      const { result } = renderHook(() =>
        useGenerationForm(FALLBACK_APP_CONFIG, "a cat", { nativeBridge: mockBridge }, { isICLora: true }),
      );

      await act(async () => {
        await result.current.referenceVideo.uploadPath("C:\\v\\ref.mp4", "ref.mp4", {
          trim_start_sec: "1.000",
          trim_duration_sec: "5.000",
        });
      });
      await waitFor(() => expect(result.current.referenceVideo.state.status).toBe("ready"));

      expect(result.current.referenceVideo.state.trimFailed).toBe(false);
      expect(result.current.validityReasons).not.toContain("referenceTrimFailed");
    });

    it("does not fire when no trim was requested at all (manual 📁 pick / untrimmed route)", async () => {
      const inner = createMockBridge({ delayMs: 0 });
      const { result } = renderHook(() =>
        useGenerationForm(FALLBACK_APP_CONFIG, "a cat", { nativeBridge: dropsTrimQuery(inner) }, { isICLora: true }),
      );

      await act(async () => {
        await result.current.referenceVideo.uploadPath("C:\\v\\ref.mp4", "ref.mp4");
      });
      await waitFor(() => expect(result.current.referenceVideo.state.status).toBe("ready"));

      expect(result.current.referenceVideo.state.trimFailed).toBe(false);
      expect(result.current.validityReasons).not.toContain("referenceTrimFailed");
    });
  });

  it("keeps isValid === (validityReasons.length === 0) with multiple gates broken at once", () => {
    // Blank prompt + off-grid frames: two independent reasons, still un-submittable.
    const { result } = renderHook(() => useGenerationForm(FALLBACK_APP_CONFIG, ""));
    act(() => result.current.setNumFrames(50, false));
    expect(result.current.validityReasons).toEqual(expect.arrayContaining(["promptEmpty", "numFramesOffGrid"]));
    expect(result.current.isValid).toBe(result.current.validityReasons.length === 0);
    expect(result.current.isValid).toBe(false);
  });
});

/**
 * §1-6 拡張: a stand-in for an OLD native build (or any backend that cannot cut)
 * — it strips contract v10's `query` before the call reaches the mock, so the
 * response comes back with `trimmed: false` even though a trim was asked for.
 * That is exactly the situation `referenceTrimFailed` exists to catch. Mirrors
 * `modes/chained/useChainForm.test.ts`'s helper of the same name.
 */
function dropsTrimQuery(inner: NativeBridge): NativeBridge {
  return {
    ...inner,
    request<M extends BridgeMethod>(method: M, params: ParamsOf<M>): Promise<ResultOf<M>> {
      if (method === "backend.uploadFile") {
        const { query: _dropped, ...rest } = params as ParamsOf<"backend.uploadFile">;
        return inner.request(method, rest as ParamsOf<M>);
      }
      return inner.request(method, params);
    },
  };
}
