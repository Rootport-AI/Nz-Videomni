import { act, renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { AppConfig } from "../../api/types";
import { createMockBridge } from "../../bridge/mockBridge";
import type { MockBridgeOptions } from "../../bridge/mockBridge";
import { BridgeError } from "../../bridge";
import type { BridgeMethod, NativeBridge, ParamsOf, ResultOf } from "../../bridge";
import type { NagSettings } from "../../shell/nagSettings";
import { ACCELERATION_DEFAULTS } from "../../shell/accelerationSettings";
import type { AccelerationSettings } from "../../shell/accelerationSettings";
import { en, ja } from "../../i18n/strings";
import { FALLBACK_APP_CONFIG } from "../single/defaultConfig";
import { DEFAULT_OVERLAP_FRAMES } from "./chainUtils";
import { buildChainReasonMessages, CHAIN_AUDIO_REASON_CODES } from "./generateReasonMessages";
import { useChainForm } from "./useChainForm";

function setup(
  prompt = "a cat riding a skateboard",
  delayMs = 0,
  controlLoraNames?: ReadonlySet<string>,
  nag?: NagSettings,
  acceleration?: AccelerationSettings,
) {
  const mockBridge = createMockBridge({ delayMs });
  const { result } = renderHook(() =>
    useChainForm(FALLBACK_APP_CONFIG, prompt, { nativeBridge: mockBridge, controlLoraNames, nag, acceleration }),
  );
  return { result, mockBridge };
}

describe("useChainForm", () => {
  it("starts in scratch mode with 2 default clips (the no-source minimum)", () => {
    const { result } = setup();
    expect(result.current.hasSourceVideo).toBe(false);
    expect(result.current.mode).toBe("scratch");
    expect(result.current.clips).toHaveLength(2);
    expect(result.current.minClips).toBe(2);
    expect(result.current.isClipCountValid).toBe(true);
  });

  // W8: a right-click prefill (comfortCeiling policy) can seed clip 0's initial
  // DURATION via the fourth `initial` arg's `numFrames` — only clip 0, never the
  // from-scratch padding clip.
  describe("initial.numFrames (W8 right-click DURATION seed)", () => {
    it("seeds ONLY clip 0's DURATION from initial.numFrames; the padding clip keeps the config default", () => {
      const mockBridge = createMockBridge({ delayMs: 0 });
      // 481 (a comfort ceiling) differs from the config default (361).
      const { result } = renderHook(() => useChainForm(FALLBACK_APP_CONFIG, "x", { nativeBridge: mockBridge }, { numFrames: 481 }));
      expect(result.current.clips).toHaveLength(2);
      expect(result.current.clips[0]?.numFrames).toBe(481);
      // The second (padding) clip is untouched — the config default (361).
      expect(result.current.clips[1]?.numFrames).toBe(361);
    });

    it("keeps clip 0 at the config default when initial.numFrames is omitted", () => {
      const mockBridge = createMockBridge({ delayMs: 0 });
      const { result } = renderHook(() => useChainForm(FALLBACK_APP_CONFIG, "x", { nativeBridge: mockBridge }, {}));
      expect(result.current.clips[0]?.numFrames).toBe(361);
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
    const mockBridge = createMockBridge({ delayMs: 0 });
    const { result } = renderHook(() => useChainForm(config, "a cat riding a skateboard", { nativeBridge: mockBridge }));

    expect(result.current.cropOutput).toEqual({ width: 1280, height: 720 });
  });

  it("addClip/removeClip respect the [min,max] bounds (max 24)", () => {
    const { result } = setup();
    act(() => {
      for (let i = 0; i < 30; i += 1) result.current.addClip();
    });
    expect(result.current.clips).toHaveLength(24);
    expect(result.current.canAddClip).toBe(false);

    act(() => {
      const id = result.current.clips[0]?.id;
      if (id) result.current.removeClip(id);
    });
    expect(result.current.clips).toHaveLength(23);
  });

  it("removeClip never drops below 1 clip card, even below the scratch minimum", () => {
    const { result } = setup();
    expect(result.current.clips).toHaveLength(2);
    act(() => {
      const id = result.current.clips[0]?.id;
      if (id) result.current.removeClip(id);
    });
    expect(result.current.clips).toHaveLength(1); // below the scratch min of 2, but allowed structurally
    expect(result.current.isClipCountValid).toBe(false); // ...and flagged invalid

    act(() => {
      const id = result.current.clips[0]?.id;
      if (id) result.current.removeClip(id);
    });
    expect(result.current.clips).toHaveLength(1); // floor: never below 1
  });

  it("stays exactly at the 11544 ceiling (24 clips x 481 frames) and remains valid there", () => {
    // Per-clip num_frames is capped at 481 and clip count at 24, so 24*481=
    // 11544 — the documented ceiling itself — is the largest total actually
    // reachable through the form; there's no way to *exceed* it while
    // respecting the other per-field constraints.
    const { result } = setup();
    act(() => {
      for (let i = 0; i < 22; i += 1) result.current.addClip();
    });
    expect(result.current.clips).toHaveLength(24);

    act(() => {
      for (const clip of result.current.clips) {
        result.current.setClipNumFrames(clip.id, 481);
      }
    });

    expect(result.current.totalFrames).toBe(11544);
    expect(result.current.isTotalFramesValid).toBe(true);
  });

  it("buildRequest strips <lora:...> tags from the prompt and sends them as loras", () => {
    const { result } = setup("a cat <lora:Pixar_Toon:1.0> riding a skateboard");
    const request = result.current.buildRequest();
    expect(request.prompt).toBe("a cat riding a skateboard");
    expect(request.loras).toEqual([{ name: "Pixar_Toon", strength: 1.0 }]);
  });

  it("buildRequest omits loras entirely when the prompt has no tags", () => {
    const { result } = setup("a cat riding a skateboard");
    const request = result.current.buildRequest();
    expect(request).not.toHaveProperty("loras");
  });

  // §1-15 参照動画: the old "control LoRA tags aren't supported in Chain" hard
  // block is GONE. A control tag is now legal exactly as it is on Create — it
  // just requires a reference video, and the gate that says so is judged on the
  // MERGED loras' NAMES (so a hand-typed tag counts, not only a panel choice).
  describe("§1-15: control LoRA without a reference video", () => {
    it("does not gate at all when controlLoraNames is omitted (no name can be a control LoRA)", () => {
      const { result } = setup("a cat <lora:canny-control:1.0> riding a skateboard");
      expect(result.current.controlLoraNeedsReference).toBe(false);
      expect(result.current.isValid).toBe(true);
    });

    it("reports controlLoraNeedsReference for a HAND-TYPED control tag with no reference attached", () => {
      const { result } = setup("a cat <lora:canny-control:1.0> riding a skateboard", 0, new Set(["canny-control"]));
      expect(result.current.controlLoraNeedsReference).toBe(true);
      expect(result.current.validityReasons).toContain("controlLoraNeedsReference");
      expect(result.current.isValid).toBe(false);
    });

    it("reports controlLoraNeedsReference for a PANEL selection with no reference attached", () => {
      const { result } = setup("a cat riding a skateboard", 0, new Set(["canny-control"]));
      act(() => result.current.setControlLora({ name: "canny-control", strength: 1.0 }));
      expect(result.current.controlLoraNeedsReference).toBe(true);
      expect(result.current.isValid).toBe(false);
    });

    it("a STYLE tag whose name isn't in controlLoraNames doesn't trip the gate", () => {
      const { result } = setup("a cat <lora:Pixar_Toon:1.0> riding a skateboard", 0, new Set(["canny-control"]));
      expect(result.current.controlLoraNeedsReference).toBe(false);
      expect(result.current.isValid).toBe(true);
    });

    it("removing the offending tag (mirroring the chip's x button) clears the gate and re-enables Generate", () => {
      const { result, rerender } = renderHook(
        ({ prompt }) =>
          useChainForm(FALLBACK_APP_CONFIG, prompt, {
            nativeBridge: createMockBridge({ delayMs: 0 }),
            controlLoraNames: new Set(["canny-control"]),
          }),
        { initialProps: { prompt: "a cat <lora:canny-control:1.0> riding a skateboard" } },
      );
      expect(result.current.controlLoraNeedsReference).toBe(true);
      expect(result.current.isValid).toBe(false);

      rerender({ prompt: "a cat riding a skateboard" });

      expect(result.current.controlLoraNeedsReference).toBe(false);
      expect(result.current.isValid).toBe(true);
    });

    it("attaching a reference video clears it, and buildRequest carries both the loRA and the reference", async () => {
      const { result } = setup("a cat <lora:canny-control:1.0> riding a skateboard", 0, new Set(["canny-control"]));
      await act(async () => {
        await result.current.attachReferenceByPath("C:\\ref\\control.mp4", "control.mp4");
      });
      await waitFor(() => expect(result.current.referenceVideo.state.status).toBe("ready"));

      expect(result.current.controlLoraNeedsReference).toBe(false);
      expect(result.current.validityReasons).toEqual([]);
      const request = result.current.buildRequest();
      expect(request.loras).toEqual([{ name: "canny-control", strength: 1.0 }]);
      expect(request.reference_video_id).toBe(result.current.referenceVideo.state.id);
    });

    it("the panel selection leads the merged loras[] and dedupes against a same-named tag", () => {
      const { result } = setup("a cat <lora:Pixar_Toon:0.8> riding", 0, new Set(["canny-control"]));
      act(() => result.current.setControlLora({ name: "canny-control", strength: 0.6 }));
      const request = result.current.buildRequest();
      expect(request.loras).toEqual([
        { name: "canny-control", strength: 0.6 },
        { name: "Pixar_Toon", strength: 0.8 },
      ]);
    });

    it("setControlLoraStrength clamps, and is a no-op while nothing is selected", () => {
      const { result } = setup("a cat riding", 0, new Set(["canny-control"]));
      act(() => result.current.setControlLoraStrength(0.5));
      expect(result.current.controlLora).toBeNull();

      act(() => result.current.setControlLora({ name: "canny-control", strength: 1.0 }));
      act(() => result.current.setControlLoraStrength(99));
      expect(result.current.controlLora?.strength).toBeLessThanOrEqual(2);
      expect(result.current.controlLora?.name).toBe("canny-control");
    });
  });

  it("buildRequest's clips carry per-clip prompt overrides and only clip 0 gets the start frame (scratch mode)", async () => {
    const { result } = setup();
    act(() => {
      const secondId = result.current.clips[1]?.id;
      if (secondId) result.current.setClipPrompt(secondId, "a dog joins in");
    });

    act(() => {
      void result.current.startFrame.addFromCapture();
    });
    await waitFor(() => expect(result.current.startFrame.items).toHaveLength(1));
    await waitFor(() => expect(result.current.startFrame.items[0]?.status).toBe("ready"));

    const request = result.current.buildRequest();
    expect(request.clips[0]?.conditioning_images).toHaveLength(1);
    expect(request.clips[1]).not.toHaveProperty("conditioning_images");
    expect(request.clips[1]?.prompt).toBe("a dog joins in");
    expect(request.clips[0]).not.toHaveProperty("prompt"); // left blank -> omitted
  });

  it("start frame is capped at a single image (maxItemsOverride: 1)", async () => {
    const { result } = setup();
    expect(result.current.startFrame.maxItems).toBe(1);
    act(() => {
      void result.current.startFrame.addFromCapture();
    });
    await waitFor(() => expect(result.current.startFrame.items).toHaveLength(1));
    expect(result.current.startFrame.canAdd).toBe(false);
  });

  it("attaching a source video switches to v2v mode: relaxes the floor to 1 and bumps clip 0 so context_frames stays valid", async () => {
    // Pin a small generation_defaults.num_frames (49) so the default clip
    // length is shorter than the default context_frames (73) and the "bump
    // clip 0 to keep context valid" path is observable — the real default
    // (257) already exceeds 73, so no bump would fire.
    const config: AppConfig = {
      ...FALLBACK_APP_CONFIG,
      generation_defaults: { ...FALLBACK_APP_CONFIG.generation_defaults, num_frames: 49 },
    };
    const mockBridge = createMockBridge({ delayMs: 0 });
    const { result } = renderHook(() => useChainForm(config, "a cat riding a skateboard", { nativeBridge: mockBridge }));
    expect(result.current.mode).toBe("scratch");
    expect(result.current.clips[0]?.numFrames).toBe(49); // config default

    act(() => {
      void result.current.sourceVideo.pick();
    });
    await waitFor(() => expect(result.current.sourceVideo.state.status).toBe("ready"));

    expect(result.current.hasSourceVideo).toBe(true);
    expect(result.current.mode).toBe("v2v");
    expect(result.current.minClips).toBe(1);
    // Default context_frames (73) would exceed the default clip length (49);
    // the ready transition bumps clip 0 up (73+8 -> 81) to keep it valid.
    expect(result.current.clips[0]?.numFrames).toBe(81);
    expect(result.current.contextFrames).toBe(73);
    expect(result.current.isContextFramesValid).toBe(true);
    expect(result.current.isValid).toBe(true);

    const request = result.current.buildRequest();
    expect(request.source_video).toEqual({
      video_id: result.current.sourceVideo.state.id,
      context_frames: result.current.contextFrames,
    });
  });

  it("clearing the source video returns to scratch mode and pads clips back up to 2", async () => {
    const { result } = setup();
    act(() => {
      void result.current.sourceVideo.pick();
    });
    await waitFor(() => expect(result.current.sourceVideo.state.status).toBe("ready"));
    expect(result.current.mode).toBe("v2v");

    // Trim to a single clip (valid in v2v)...
    act(() => {
      const id = result.current.clips[1]?.id;
      if (id) result.current.removeClip(id);
    });
    expect(result.current.clips).toHaveLength(1);

    // ...then clear the source: back to scratch, padded up to the 2-clip floor.
    act(() => result.current.sourceVideo.clear());
    expect(result.current.hasSourceVideo).toBe(false);
    expect(result.current.mode).toBe("scratch");
    expect(result.current.clips).toHaveLength(2);
    expect(result.current.minClips).toBe(2);
  });

  it("the start frame is never injected into a V2V request (source supplies clip 0's frames)", async () => {
    const { result } = setup();
    // Add a start frame while still in scratch mode...
    act(() => {
      void result.current.startFrame.addFromCapture();
    });
    await waitFor(() => expect(result.current.startFrame.items[0]?.status).toBe("ready"));

    // ...then attach a source video (switches to v2v).
    act(() => {
      void result.current.sourceVideo.pick();
    });
    await waitFor(() => expect(result.current.sourceVideo.state.status).toBe("ready"));
    expect(result.current.mode).toBe("v2v");

    const request = result.current.buildRequest();
    expect(request.clips[0]).not.toHaveProperty("conditioning_images");
  });

  it("stays in v2v mode without re-padding clips while the source upload is still in flight", async () => {
    // `holdUploads` makes `backend.uploadFile` block until we explicitly
    // release it, instead of racing a real setTimeout (`delayMs`) against
    // `waitFor`'s polling — so "uploading" is a stable state to observe here,
    // not a fleeting window that only sometimes gets caught.
    const mockBridge = createMockBridge({ delayMs: 0, holdUploads: true });
    const { result } = renderHook(() =>
      useChainForm(FALLBACK_APP_CONFIG, "a cat riding a skateboard", { nativeBridge: mockBridge }),
    );
    expect(result.current.clips).toHaveLength(2);

    act(() => {
      void result.current.sourceVideo.pick();
    });
    // Catch the mid-upload window: mode already flips to v2v (attach intent),
    // but the clip list must NOT wobble/pad during it. The upload gate is
    // still held here, so this state can't have advanced past "uploading".
    await waitFor(() => expect(result.current.sourceVideo.state.status).toBe("uploading"));
    expect(result.current.hasSourceVideo).toBe(true);
    expect(result.current.mode).toBe("v2v");
    const clipsWhileUploading = result.current.clips.length;
    expect(clipsWhileUploading).toBe(2); // unchanged: no re-pad while attach is pending

    act(() => {
      mockBridge.releaseUploads();
    });
    await waitFor(() => expect(result.current.sourceVideo.state.status).toBe("ready"));
    expect(result.current.mode).toBe("v2v");
    expect(result.current.clips.length).toBe(clipsWhileUploading);
  });

  it("a failed source upload keeps v2v mode but blocks Generate (isValid=false)", async () => {
    const mockBridge = createMockBridge({ delayMs: 0, failUploadFile: "BACKEND_UNREACHABLE" });
    const { result } = renderHook(() =>
      useChainForm(FALLBACK_APP_CONFIG, "a cat riding a skateboard", { nativeBridge: mockBridge }),
    );

    act(() => {
      void result.current.sourceVideo.pick();
    });
    await waitFor(() => expect(result.current.sourceVideo.state.status).toBe("error"));

    // Attach intent (error) still reads as v2v so the screen doesn't flip back
    // and spawn ghost clips, but Generate is gated shut.
    expect(result.current.hasSourceVideo).toBe(true);
    expect(result.current.mode).toBe("v2v");
    expect(result.current.isValid).toBe(false);
  });

  it("V2V context_frames must be less than clips[0].num_frames", async () => {
    const { result } = setup();
    act(() => {
      void result.current.sourceVideo.pick();
    });
    await waitFor(() => expect(result.current.sourceVideo.state.status).toBe("ready"));

    const firstClipId = result.current.clips[0]?.id;
    if (!firstClipId) throw new Error("expected a clip");
    act(() => result.current.setClipNumFrames(firstClipId, 9)); // shorter than default context_frames (73)

    expect(result.current.isContextFramesValid).toBe(false);
    expect(result.current.isValid).toBe(false);
  });

  it("setWidth/setHeight snap to multiples of 64 by default and on explicit snap=true, mirroring Create's rules", () => {
    const { result } = setup();
    act(() => result.current.setWidth(500)); // default snap=true
    expect(result.current.common.width).toBe(512);
    act(() => result.current.setHeight(10, true));
    expect(result.current.common.height).toBe(128);
  });

  it("setWidth/setHeight with snap=false pass the raw value through untouched (Gradio-faithful manual typing)", () => {
    const { result } = setup();
    act(() => result.current.setWidth(500, false));
    expect(result.current.common.width).toBe(500);
    act(() => result.current.setWidth(500, true));
    expect(result.current.common.width).toBe(512);

    act(() => result.current.setHeight(700, false));
    expect(result.current.common.height).toBe(700);
  });

  it("isValid requires width/height to be on the 64 grid: off-grid manual entry blocks submit, snapping restores it", () => {
    const { result } = setup();
    // Clear the default crop so this test isolates the width/height grid gate:
    // otherwise shrinking width below the default crop_output width would block
    // submit for an unrelated (crop) reason.
    act(() => result.current.setCropOutput(null));
    expect(result.current.isValid).toBe(true); // starts on-grid

    act(() => result.current.setWidth(500, false)); // manual, off-grid
    expect(result.current.common.width).toBe(500);
    expect(result.current.isValid).toBe(false);

    act(() => result.current.setWidth(512, false)); // manual, but already on-grid
    expect(result.current.isValid).toBe(true);

    act(() => result.current.setWidth(500, true)); // stepper/slider snaps back onto the grid
    expect(result.current.common.width).toBe(512);
    expect(result.current.isValid).toBe(true);
  });

  it("overlapFrames/overlapStrength clamp to their documented ranges", () => {
    const { result } = setup();
    act(() => result.current.setOverlapFrames(20));
    expect(result.current.overlapFrames).toBe(8);
    act(() => result.current.setOverlapFrames(0));
    expect(result.current.overlapFrames).toBe(1);
    act(() => result.current.setOverlapStrength(5));
    expect(result.current.overlapStrength).toBe(1);
    act(() => result.current.setOverlapStrength(-1));
    expect(result.current.overlapStrength).toBe(0);
  });

  it("chunkedUpsample defaults to true and toggling it flips buildRequest's chunked_upsample", () => {
    const { result } = setup();
    expect(result.current.chunkedUpsample).toBe(true);
    expect(result.current.buildRequest().chunked_upsample).toBe(true);

    act(() => result.current.setChunkedUpsample(false));
    expect(result.current.chunkedUpsample).toBe(false);
    expect(result.current.buildRequest().chunked_upsample).toBe(false);
  });

  it("outputFrames/outputSeconds recompute when overlapFrames or a clip's duration changes", () => {
    const { result } = setup();
    const before = result.current.outputFrames;
    expect(result.current.outputSeconds).toBeCloseTo(before / result.current.common.frameRate, 5);

    act(() => result.current.setOverlapFrames(8));
    expect(result.current.outputFrames).not.toBe(before);
    expect(result.current.outputSeconds).toBeCloseTo(result.current.outputFrames / result.current.common.frameRate, 5);

    const beforeDurationChange = result.current.outputFrames;
    act(() => {
      const firstId = result.current.clips[0]?.id;
      if (firstId) result.current.setClipNumFrames(firstId, 481);
    });
    expect(result.current.outputFrames).not.toBe(beforeDurationChange);
  });

  it("setClipNumFrames snaps to the nearest 8n+1 by default and on explicit snap=true", () => {
    const { result } = setup();
    const firstClipId = result.current.clips[0]?.id;
    if (!firstClipId) throw new Error("expected a clip");

    act(() => result.current.setClipNumFrames(firstClipId, 100));
    expect(result.current.clips[0]?.numFrames).toBe(97);

    act(() => result.current.setClipNumFrames(firstClipId, 100, true));
    expect(result.current.clips[0]?.numFrames).toBe(97);
  });

  it("setClipNumFrames with snap=false passes the raw value through untouched (Gradio-faithful manual typing)", () => {
    const { result } = setup();
    const firstClipId = result.current.clips[0]?.id;
    if (!firstClipId) throw new Error("expected a clip");

    act(() => result.current.setClipNumFrames(firstClipId, 50, false));
    expect(result.current.clips[0]?.numFrames).toBe(50);

    // Non-finite input is rejected, keeping the prior value.
    act(() => result.current.setClipNumFrames(firstClipId, Number.NaN, false));
    expect(result.current.clips[0]?.numFrames).toBe(50);
  });

  it("isValid requires EVERY clip's numFrames to be on the 8n+1 grid, not just the first", () => {
    const { result } = setup();
    expect(result.current.clips).toHaveLength(2);
    expect(result.current.isValid).toBe(true); // starts on-grid

    const secondClipId = result.current.clips[1]?.id;
    if (!secondClipId) throw new Error("expected a second clip");

    act(() => result.current.setClipNumFrames(secondClipId, 50, false)); // manual, off-grid
    expect(result.current.clips[1]?.numFrames).toBe(50);
    expect(result.current.isValid).toBe(false);

    act(() => result.current.setClipNumFrames(secondClipId, 50, true)); // stepper/slider snaps back
    expect(result.current.clips[1]?.numFrames).toBe(49);
    expect(result.current.isValid).toBe(true);
  });

  describe("applyPreset", () => {
    it("sets common width/height and uniformly sets every clip's numFrames to the recommendation", () => {
      const { result } = setup();
      expect(result.current.clips).toHaveLength(2); // scratch-mode default

      act(() => result.current.applyPreset("standard_720p"));

      expect(result.current.common.width).toBe(1280);
      expect(result.current.common.height).toBe(768);
      // standard_720p (1280x768, 257f) has a spill_free_frames["1280x768"]=257
      // entry in FALLBACK_APP_CONFIG, so the recommendation equals it exactly.
      expect(result.current.clips.every((clip) => clip.numFrames === 257)).toBe(true);
    });

    it("reflects the preset's own crop_output (N1, Gradio-faithful)", () => {
      const { result } = setup();
      // standard_720p: { width: 1280, height: 768, crop_output: { width: 1280, height: 720 } }
      act(() => result.current.applyPreset("standard_720p"));
      expect(result.current.cropOutput).toEqual({ width: 1280, height: 720 });
    });

    it("turns crop OFF (null) when the preset defines none, clearing a previously-set crop", () => {
      const { result } = setup();
      act(() => result.current.applyPreset("standard_720p")); // has a crop_output
      expect(result.current.cropOutput).not.toBeNull();

      act(() => result.current.applyPreset("minimal")); // crop_output: null
      expect(result.current.cropOutput).toBeNull();
    });

    it("clamps the preset's crop_output against the ROUNDED post-apply common width/height, not the preset's raw values", () => {
      // Pin the config maxima to 1920x1088 (below WQHD_1440p's raw resolution)
      // so the clamp is exercised — the real-server 4096 maxima would leave
      // WQHD's 2560x1472 unclamped and there'd be nothing to demonstrate.
      const config: AppConfig = {
        ...FALLBACK_APP_CONFIG,
        limits: { ...FALLBACK_APP_CONFIG.limits, max_width: 1920, max_height: 1088 },
      };
      const mockBridge = createMockBridge({ delayMs: 0 });
      const { result } = renderHook(() => useChainForm(config, "a cat riding a skateboard", { nativeBridge: mockBridge }));
      // WQHD_1440p: { width: 2560, height: 1472, crop_output: { width: 2560, height: 1440 } };
      // max_width/max_height (1920/1088) clamp common down well below the
      // preset's raw resolution, so the crop must clamp against 1920/1088,
      // not 2560/1440 (which would leave the stored crop bigger than the
      // actual applied width/height).
      act(() => result.current.applyPreset("WQHD_1440p"));

      expect(result.current.common.width).toBe(1920);
      expect(result.current.common.height).toBe(1088);
      expect(result.current.cropOutput).toEqual({ width: 1920, height: 1088 });
    });

    it("falls back to the preset's own num_frames when its resolution has no spill_free_frames entry", () => {
      const { result } = setup();
      // smoke_test (384x256, 17f) has no spill_free_frames key (unlike minimal's
      // 512x320, which now carries a 481 entry), so the recommendation falls
      // back to the preset's own num_frames (17).
      act(() => result.current.applyPreset("smoke_test"));
      expect(result.current.common.width).toBe(384);
      expect(result.current.common.height).toBe(256);
      expect(result.current.clips.every((clip) => clip.numFrames === 17)).toBe(true);
    });

    it("clamps width/height into the configured limits (WQHD_1440p's 2560 exceeds max_width=1920)", () => {
      // Pin the config maxima to 1920x1088 so WQHD_1440p's raw 2560x1472 is
      // clamped (the real-server 4096 maxima would let it through unclamped).
      const config: AppConfig = {
        ...FALLBACK_APP_CONFIG,
        limits: { ...FALLBACK_APP_CONFIG.limits, max_width: 1920, max_height: 1088 },
      };
      const mockBridge = createMockBridge({ delayMs: 0 });
      const { result } = renderHook(() => useChainForm(config, "a cat riding a skateboard", { nativeBridge: mockBridge }));
      act(() => result.current.applyPreset("WQHD_1440p"));
      expect(result.current.common.width).toBe(1920); // clamped at max_width
      expect(result.current.common.height).toBe(1088); // clamped at max_height
    });

    it("is a no-op for an unknown preset name", () => {
      const { result } = setup();
      const before = { ...result.current.common };
      const beforeFrames = result.current.clips.map((c) => c.numFrames);
      act(() => result.current.applyPreset("does-not-exist"));
      expect(result.current.common).toEqual(before);
      expect(result.current.clips.map((c) => c.numFrames)).toEqual(beforeFrames);
    });
  });

  describe("pickSource / clearSource (unified source-input picker)", () => {
    it("starts with no source attached", () => {
      const { result } = setup();
      expect(result.current.activeSourceKind).toBeNull();
      expect(result.current.sourceError).toBeNull();
      expect(result.current.isPickingSource).toBe(false);
    });

    it("picking an image: adds the start frame, stays in scratch mode, and leaves sourceVideo idle", async () => {
      const mockBridge = createMockBridge({ delayMs: 0, pickFileName: "photo.png" });
      const { result } = renderHook(() =>
        useChainForm(FALLBACK_APP_CONFIG, "a cat riding a skateboard", { nativeBridge: mockBridge }),
      );

      act(() => {
        void result.current.pickSource();
      });
      await waitFor(() => expect(result.current.startFrame.items).toHaveLength(1));
      await waitFor(() => expect(result.current.startFrame.items[0]?.status).toBe("ready"));

      expect(result.current.mode).toBe("scratch");
      expect(result.current.hasSourceVideo).toBe(false);
      expect(result.current.sourceVideo.state.status).toBe("idle");
      expect(result.current.activeSourceKind).toBe("image");
    });

    it("picking a video: uploads the source video, switches to v2v, and never touches the start frame", async () => {
      const mockBridge = createMockBridge({ delayMs: 0, pickFileName: "clip.mp4" });
      const { result } = renderHook(() =>
        useChainForm(FALLBACK_APP_CONFIG, "a cat riding a skateboard", { nativeBridge: mockBridge }),
      );

      act(() => {
        void result.current.pickSource();
      });
      await waitFor(() => expect(result.current.sourceVideo.state.status).toBe("ready"));

      expect(result.current.mode).toBe("v2v");
      expect(result.current.startFrame.items).toHaveLength(0);
      expect(result.current.activeSourceKind).toBe("video");
    });

    it("picking an image after a video replaces it (single-slot exclusivity: image wins, video drops)", async () => {
      const opts: MockBridgeOptions = { delayMs: 0, pickFileName: "clip.mp4" };
      const mockBridge = createMockBridge(opts);
      const { result } = renderHook(() =>
        useChainForm(FALLBACK_APP_CONFIG, "a cat riding a skateboard", { nativeBridge: mockBridge }),
      );

      act(() => {
        void result.current.pickSource();
      });
      await waitFor(() => expect(result.current.sourceVideo.state.status).toBe("ready"));
      expect(result.current.activeSourceKind).toBe("video");

      opts.pickFileName = "photo.png";
      act(() => {
        void result.current.pickSource();
      });
      await waitFor(() => expect(result.current.startFrame.items).toHaveLength(1));
      await waitFor(() => expect(result.current.startFrame.items[0]?.status).toBe("ready"));

      expect(result.current.activeSourceKind).toBe("image");
      expect(result.current.mode).toBe("scratch");
      expect(result.current.sourceVideo.state.status).toBe("idle");
    });

    it("picking a video after an image replaces it (single-slot exclusivity: video wins, image drops)", async () => {
      const opts: MockBridgeOptions = { delayMs: 0, pickFileName: "photo.png" };
      const mockBridge = createMockBridge(opts);
      const { result } = renderHook(() =>
        useChainForm(FALLBACK_APP_CONFIG, "a cat riding a skateboard", { nativeBridge: mockBridge }),
      );

      act(() => {
        void result.current.pickSource();
      });
      await waitFor(() => expect(result.current.startFrame.items).toHaveLength(1));
      await waitFor(() => expect(result.current.startFrame.items[0]?.status).toBe("ready"));
      expect(result.current.activeSourceKind).toBe("image");

      opts.pickFileName = "clip.mp4";
      act(() => {
        void result.current.pickSource();
      });
      await waitFor(() => expect(result.current.sourceVideo.state.status).toBe("ready"));

      expect(result.current.activeSourceKind).toBe("video");
      expect(result.current.mode).toBe("v2v");
      expect(result.current.startFrame.items).toHaveLength(0);
    });

    it("clearSource detaches both slots, returns to scratch, and re-pads clips back to 2", async () => {
      const mockBridge = createMockBridge({ delayMs: 0, pickFileName: "clip.mp4" });
      const { result } = renderHook(() =>
        useChainForm(FALLBACK_APP_CONFIG, "a cat riding a skateboard", { nativeBridge: mockBridge }),
      );

      act(() => {
        void result.current.pickSource();
      });
      await waitFor(() => expect(result.current.sourceVideo.state.status).toBe("ready"));
      expect(result.current.mode).toBe("v2v");

      // Trim to a single clip (valid in v2v), same as the plain
      // `sourceVideo.clear()` coverage above.
      act(() => {
        const id = result.current.clips[1]?.id;
        if (id) result.current.removeClip(id);
      });
      expect(result.current.clips).toHaveLength(1);

      act(() => result.current.clearSource());

      expect(result.current.activeSourceKind).toBeNull();
      expect(result.current.hasSourceVideo).toBe(false);
      expect(result.current.mode).toBe("scratch");
      expect(result.current.clips).toHaveLength(2);
      expect(result.current.sourceVideo.state.status).toBe("idle");
      expect(result.current.startFrame.items).toHaveLength(0);
    });

    it("clearSource resets a prior sourceError even with nothing attached", async () => {
      const mockBridge = createMockBridge({ delayMs: 0, pickFileName: "not-a-media-file.txt" });
      const { result } = renderHook(() =>
        useChainForm(FALLBACK_APP_CONFIG, "a cat riding a skateboard", { nativeBridge: mockBridge }),
      );

      // An unsupported extension sets sourceError without touching either slot.
      act(() => {
        void result.current.pickSource();
      });
      await waitFor(() => expect(result.current.sourceError).not.toBeNull());
      expect(result.current.activeSourceKind).toBeNull();

      act(() => result.current.clearSource());
      expect(result.current.sourceError).toBeNull();
    });

    it("an unrecognised extension sets sourceError and leaves clips/mode untouched", async () => {
      const mockBridge = createMockBridge({ delayMs: 0, pickFileName: "notes.txt" });
      const { result } = renderHook(() =>
        useChainForm(FALLBACK_APP_CONFIG, "a cat riding a skateboard", { nativeBridge: mockBridge }),
      );
      const clipsBefore = result.current.clips;

      act(() => {
        void result.current.pickSource();
      });
      await waitFor(() => expect(result.current.sourceError).toBe("UNSUPPORTED_FILE_TYPE"));

      expect(result.current.activeSourceKind).toBeNull();
      expect(result.current.mode).toBe("scratch");
      expect(result.current.clips).toBe(clipsBefore);
      expect(result.current.startFrame.items).toHaveLength(0);
      expect(result.current.sourceVideo.state.status).toBe("idle");
    });

    it("sourceError clears on the next successful pick after an unsupported one", async () => {
      const opts: MockBridgeOptions = { delayMs: 0, pickFileName: "notes.txt" };
      const mockBridge = createMockBridge(opts);
      const { result } = renderHook(() =>
        useChainForm(FALLBACK_APP_CONFIG, "a cat riding a skateboard", { nativeBridge: mockBridge }),
      );

      act(() => {
        void result.current.pickSource();
      });
      await waitFor(() => expect(result.current.sourceError).toBe("UNSUPPORTED_FILE_TYPE"));

      opts.pickFileName = "photo.png";
      act(() => {
        void result.current.pickSource();
      });
      await waitFor(() => expect(result.current.startFrame.items).toHaveLength(1));
      await waitFor(() => expect(result.current.startFrame.items[0]?.status).toBe("ready"));

      expect(result.current.sourceError).toBeNull();
    });

    it("a CANCELLED pick leaves everything untouched, including a prior sourceError", async () => {
      const opts: MockBridgeOptions = { delayMs: 0, failPickFile: "CANCELLED" };
      const mockBridge = createMockBridge(opts);
      const { result } = renderHook(() =>
        useChainForm(FALLBACK_APP_CONFIG, "a cat riding a skateboard", { nativeBridge: mockBridge }),
      );

      act(() => {
        void result.current.pickSource();
      });
      await waitFor(() => expect(result.current.isPickingSource).toBe(false));

      expect(result.current.activeSourceKind).toBeNull();
      expect(result.current.sourceError).toBeNull();
      expect(result.current.mode).toBe("scratch");
    });

    it("a non-CANCELLED pick dialog failure surfaces the bridge's error code as sourceError", async () => {
      const mockBridge = createMockBridge({ delayMs: 0, failPickFile: "DIALOG_FAILED" });
      const { result } = renderHook(() =>
        useChainForm(FALLBACK_APP_CONFIG, "a cat riding a skateboard", { nativeBridge: mockBridge }),
      );

      act(() => {
        void result.current.pickSource();
      });
      await waitFor(() => expect(result.current.sourceError).toBe("DIALOG_FAILED"));
      expect(result.current.activeSourceKind).toBeNull();
    });

    it("a video pick still in flight when clearSource fires doesn't resurrect as ready (generation-token race)", async () => {
      // `holdUploads` blocks `backend.uploadFile` until `releaseUploads()` is
      // called, giving a stable window to fire `clearSource()` mid-upload —
      // exactly the "video upload racing a clear" scenario the generation
      // counter in `useSourceUpload.uploadPath` exists to close.
      const mockBridge = createMockBridge({ delayMs: 0, pickFileName: "clip.mp4", holdUploads: true });
      const { result } = renderHook(() =>
        useChainForm(FALLBACK_APP_CONFIG, "a cat riding a skateboard", { nativeBridge: mockBridge }),
      );

      act(() => {
        void result.current.pickSource();
      });
      await waitFor(() => expect(result.current.sourceVideo.state.status).toBe("uploading"));

      act(() => result.current.clearSource());
      expect(result.current.sourceVideo.state.status).toBe("idle");

      act(() => {
        mockBridge.releaseUploads();
      });

      // Give the (now-released) held upload's promise a real tick to settle;
      // it must NOT resurrect the (now cleared) source video back into
      // "ready" — the generation-token mismatch inside `uploadPath` should
      // discard it silently instead.
      await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 0));
      });
      expect(result.current.sourceVideo.state.status).toBe("idle");
      expect(result.current.activeSourceKind).toBeNull();
    });
  });

  // Contract v7 (drag-and-drop): `SourceInputPanel`'s DropZone calls
  // `attachSourceByPath` directly (skipping `ui.pickFile`'s native dialog).
  // It shares `pickSource`'s exact routing/exclusivity/sourceError logic via
  // `attachRoutedSource`, so these mirror the `pickSource` tests above one
  // for one, just entering through the drag-and-drop seam instead.
  describe("attachSourceByPath (drag-and-drop unified source input)", () => {
    it("an image path: adds the start frame and stays in scratch mode", async () => {
      const { result } = setup();

      await act(async () => {
        await result.current.attachSourceByPath("C:\\Users\\mock\\Downloads\\photo.png", "photo.png");
      });
      await waitFor(() => expect(result.current.startFrame.items[0]?.status).toBe("ready"));

      expect(result.current.mode).toBe("scratch");
      expect(result.current.activeSourceKind).toBe("image");
      expect(result.current.sourceVideo.state.status).toBe("idle");
    });

    it("a video path: uploads the source video and switches to v2v", async () => {
      const { result } = setup();

      await act(async () => {
        await result.current.attachSourceByPath("C:\\Users\\mock\\Downloads\\clip.mp4", "clip.mp4");
      });
      await waitFor(() => expect(result.current.sourceVideo.state.status).toBe("ready"));

      expect(result.current.mode).toBe("v2v");
      expect(result.current.activeSourceKind).toBe("video");
      expect(result.current.startFrame.items).toHaveLength(0);
    });

    it("an unrecognised extension sets sourceError and leaves clips/mode untouched", async () => {
      const { result } = setup();
      const clipsBefore = result.current.clips;

      await act(async () => {
        await result.current.attachSourceByPath("C:\\Users\\mock\\Downloads\\notes.txt", "notes.txt");
      });

      expect(result.current.sourceError).toBe("UNSUPPORTED_FILE_TYPE");
      expect(result.current.activeSourceKind).toBeNull();
      expect(result.current.mode).toBe("scratch");
      expect(result.current.clips).toBe(clipsBefore);
    });

    it("dropping a video after an image replaces it (single-slot exclusivity, same as pickSource)", async () => {
      const { result } = setup();

      await act(async () => {
        await result.current.attachSourceByPath("C:\\Users\\mock\\Downloads\\photo.png", "photo.png");
      });
      await waitFor(() => expect(result.current.startFrame.items[0]?.status).toBe("ready"));

      await act(async () => {
        await result.current.attachSourceByPath("C:\\Users\\mock\\Downloads\\clip.mp4", "clip.mp4");
      });
      await waitFor(() => expect(result.current.sourceVideo.state.status).toBe("ready"));

      expect(result.current.activeSourceKind).toBe("video");
      expect(result.current.mode).toBe("v2v");
      expect(result.current.startFrame.items).toHaveLength(0);
    });
  });

  // Owner request 2026-08-10: the SOURCE card shows the attached material's
  // pixel size, measured by ONE `fs.probeMediaInfo` per ready attachment —
  // covering both source modes and every way the size can be unknown.
  describe("sourceMediaSize (SOURCE card resolution readout)", () => {
    function setupProbe(options: { width?: number; height?: number; rejects?: boolean } = {}) {
      const calls = { media: 0 };
      const inner = createMockBridge({
        delayMs: 0,
        probeMediaInfoWidth: options.width ?? 0,
        probeMediaInfoHeight: options.height ?? 0,
      });
      const nativeBridge = withMediaInfoProbe(inner, { rejects: options.rejects === true }, calls);
      const { result } = renderHook(() => useChainForm(FALLBACK_APP_CONFIG, "x", { nativeBridge }));
      return { result, calls };
    }

    it("is null before anything is attached", () => {
      const { result, calls } = setupProbe({ width: 1280, height: 768 });
      expect(result.current.sourceMediaSize).toBeNull();
      expect(calls.media).toBe(0);
    });

    it("probes the source VIDEO once it is ready, exactly once", async () => {
      const { result, calls } = setupProbe({ width: 1280, height: 768 });

      await act(async () => {
        await result.current.attachSourceByPath("C:\\Users\\mock\\Downloads\\clip.mp4", "clip.mp4");
      });
      await waitFor(() => expect(result.current.sourceMediaSize).toEqual({ width: 1280, height: 768 }));
      expect(calls.media).toBe(1);

      // A re-render (any unrelated state change) must not re-probe the same
      // attachment — the guard ref is what makes this "once per material".
      act(() => result.current.setContextFrames(41));
      await waitFor(() => expect(result.current.contextFrames).toBe(41));
      expect(calls.media).toBe(1);
    });

    it("probes the start IMAGE once it is ready", async () => {
      const { result, calls } = setupProbe({ width: 1920, height: 1080 });

      await act(async () => {
        await result.current.attachSourceByPath("C:\\Users\\mock\\Downloads\\photo.png", "photo.png");
      });
      await waitFor(() => expect(result.current.sourceMediaSize).toEqual({ width: 1920, height: 1080 }));
      expect(calls.media).toBe(1);
    });

    it("resets to null when the source is cleared", async () => {
      const { result } = setupProbe({ width: 1280, height: 768 });

      await act(async () => {
        await result.current.attachSourceByPath("C:\\Users\\mock\\Downloads\\clip.mp4", "clip.mp4");
      });
      await waitFor(() => expect(result.current.sourceMediaSize).toEqual({ width: 1280, height: 768 }));

      act(() => result.current.clearSource());
      await waitFor(() => expect(result.current.sourceMediaSize).toBeNull());
    });

    it("re-probes after swapping the material (image -> video)", async () => {
      const { result, calls } = setupProbe({ width: 1280, height: 768 });

      await act(async () => {
        await result.current.attachSourceByPath("C:\\Users\\mock\\Downloads\\photo.png", "photo.png");
      });
      await waitFor(() => expect(result.current.sourceMediaSize).toEqual({ width: 1280, height: 768 }));

      await act(async () => {
        await result.current.attachSourceByPath("C:\\Users\\mock\\Downloads\\clip.mp4", "clip.mp4");
      });
      await waitFor(() => expect(result.current.activeSourceKind).toBe("video"));
      await waitFor(() => expect(result.current.sourceMediaSize).toEqual({ width: 1280, height: 768 }));
      expect(calls.media).toBe(2);
    });

    it("stays null when the probe reports an unknown (0) dimension", async () => {
      const { result, calls } = setupProbe({ width: 1280, height: 0 });

      await act(async () => {
        await result.current.attachSourceByPath("C:\\Users\\mock\\Downloads\\clip.mp4", "clip.mp4");
      });
      await waitFor(() => expect(result.current.sourceVideo.state.status).toBe("ready"));
      await waitFor(() => expect(calls.media).toBe(1));
      expect(result.current.sourceMediaSize).toBeNull();
    });

    it("survives a REJECTING fs.probeMediaInfo (native's BAD_REQUEST) with no size", async () => {
      const { result, calls } = setupProbe({ width: 1280, height: 768, rejects: true });

      await act(async () => {
        await result.current.attachSourceByPath("C:\\Users\\mock\\Downloads\\clip.mp4", "clip.mp4");
      });
      await waitFor(() => expect(result.current.sourceVideo.state.status).toBe("ready"));
      await waitFor(() => expect(calls.media).toBe(1));
      expect(result.current.sourceMediaSize).toBeNull();
    });
  });

  // I8 §3-4 (※): `isDirty` drives AppShell's Chain-discard confirm gate.
  describe("isDirty (Chain-discard confirm gate)", () => {
    it("is false for a pristine, freshly-mounted form", () => {
      const { result } = setup();
      expect(result.current.isDirty).toBe(false);
    });

    it("becomes true once a source (image start frame) is attached", async () => {
      const { result } = setup();
      expect(result.current.isDirty).toBe(false);
      await act(async () => {
        await result.current.attachSourceByPath("C:\\d\\photo.png", "photo.png");
      });
      expect(result.current.isDirty).toBe(true);
    });

    it("becomes true once a source video is attached", async () => {
      const { result } = setup();
      await act(async () => {
        await result.current.attachSourceByPath("C:\\d\\clip.mp4", "clip.mp4");
      });
      expect(result.current.isDirty).toBe(true);
    });

    // §1-16 長尺A2V (Step F3): an attached track is the same kind of losable work
    // as an attached source — and the ONE audio state that must flag dirty on its
    // own, since a chain whose clip list the auto-fit did not touch (a track too
    // short to fit) otherwise looks pristine.
    it("becomes true once an audio track is attached, on attach INTENT alone", async () => {
      const bridge = withAudioProbes(createMockBridge({ delayMs: 0 }), {}, { wav: 0, media: 0 });
      const { result } = renderHook(() =>
        useChainForm(FALLBACK_APP_CONFIG, "a cat riding a skateboard", { nativeBridge: bridge }),
      );
      expect(result.current.isDirty).toBe(false);

      // No probe answers anything, so nothing length-derived happens and the clip
      // list is untouched: the attachment itself is the only thing that changed.
      await act(async () => {
        await result.current.attachAudioByPath("C:\\audio\\track.mp3", "track.mp3");
      });

      expect(result.current.clips.map((clip) => clip.numFrames)).toEqual([361, 361]);
      expect(result.current.isDirty).toBe(true);
    });

    it("becomes true after editing a clip prompt", () => {
      const { result } = setup();
      act(() => {
        const id = result.current.clips[0]?.id;
        if (id) result.current.setClipPrompt(id, "a dog appears");
      });
      expect(result.current.isDirty).toBe(true);
    });

    it("becomes true after editing a clip's duration", () => {
      const { result } = setup();
      act(() => {
        const id = result.current.clips[0]?.id;
        if (id) result.current.setClipNumFrames(id, 481);
      });
      expect(result.current.isDirty).toBe(true);
    });

    it("becomes true after adding a clip (count diverges from the default 2)", () => {
      const { result } = setup();
      act(() => result.current.addClip());
      expect(result.current.isDirty).toBe(true);
    });

    it("becomes true after editing the common width", () => {
      const { result } = setup();
      act(() => result.current.setWidth(1024));
      expect(result.current.isDirty).toBe(true);
    });

    it("becomes true after editing the seam-blend overlap", () => {
      const { result } = setup();
      act(() => result.current.setOverlapFrames(8));
      expect(result.current.isDirty).toBe(true);
    });

    it("is dirty when seeded from a right-click resolution prefill (treated as in-progress work)", () => {
      const mockBridge = createMockBridge({ delayMs: 0 });
      const { result } = renderHook(() =>
        useChainForm(FALLBACK_APP_CONFIG, "a cat riding a skateboard", { nativeBridge: mockBridge }, { width: 1024, height: 768 }),
      );
      // A prefilled width/height differs from the config-default baseline, so a
      // second right-click still prompts before overwriting it.
      expect(result.current.isDirty).toBe(true);
    });

    it("a fresh mount (remount-equivalent) is clean again", () => {
      // Dirty one instance...
      const first = setup();
      act(() => first.result.current.setWidth(1024));
      expect(first.result.current.isDirty).toBe(true);
      // ...a brand-new mount (what a remount produces) starts clean.
      const second = setup();
      expect(second.result.current.isDirty).toBe(false);
    });
  });

  // W7: `validityReasons` is the structured breakdown behind `isValid`. Each
  // case breaks one gate and asserts the code; the invariant `isValid ===
  // (validityReasons.length === 0)` holds throughout. The three overlapping
  // source gates are normalized to a SINGLE source code (R3).
  const SOURCE_CODES = ["sourceUploading", "sourceNotReady"];

  describe("validityReasons (W7)", () => {
    it("is empty (and isValid true) for a pristine scratch form with a valid prompt", () => {
      const { result } = setup();
      expect(result.current.validityReasons).toEqual([]);
      expect(result.current.isValid).toBe(true);
    });

    it("reports promptEmpty for a blank prompt", () => {
      const { result } = setup("");
      expect(result.current.validityReasons).toContain("promptEmpty");
      expect(result.current.isValid).toBe(result.current.validityReasons.length === 0);
      expect(result.current.isValid).toBe(false);
    });

    it("reports dimensionsOffGrid for an off-grid width", () => {
      const { result } = setup();
      act(() => result.current.setCropOutput(null)); // isolate the dimension gate
      act(() => result.current.setWidth(500, false));
      expect(result.current.validityReasons).toContain("dimensionsOffGrid");
      expect(result.current.isValid).toBe(false);
    });

    it("reports clipFramesOffGrid for an off-grid clip frame count", () => {
      const { result } = setup();
      const id = result.current.clips[0]?.id;
      if (!id) throw new Error("expected a clip");
      act(() => result.current.setClipNumFrames(id, 50, false)); // not 8n+1
      expect(result.current.validityReasons).toContain("clipFramesOffGrid");
      expect(result.current.isValid).toBe(false);
    });

    it("reports minClips when the clip list drops below the from-scratch minimum", () => {
      const { result } = setup();
      const id = result.current.clips[1]?.id;
      if (!id) throw new Error("expected two clips");
      act(() => result.current.removeClip(id)); // 2 -> 1, below the no-source floor of 2
      expect(result.current.validityReasons).toContain("minClips");
      expect(result.current.isValid).toBe(false);
    });

    it("reports totalFramesExceeded when the summed clip frames pass the cap", () => {
      const { result } = setup();
      const id = result.current.clips[0]?.id;
      if (!id) throw new Error("expected a clip");
      // 24 clips x 481 = the cap exactly, so overshooting on-grid is impossible;
      // a hand-typed over-cap value trips totalFramesExceeded (alongside the
      // off-grid gate — both being present is expected here).
      act(() => result.current.setClipNumFrames(id, 100_000, false));
      expect(result.current.validityReasons).toContain("totalFramesExceeded");
      expect(result.current.isValid).toBe(false);
    });

    it("reports controlLoraNeedsReference for a hand-typed control-LoRA tag with no reference", () => {
      const { result } = setup("a cat <lora:canny-control:1.0> riding", 0, new Set(["canny-control"]));
      expect(result.current.validityReasons).toContain("controlLoraNeedsReference");
      expect(result.current.isValid).toBe(false);
    });

    it("reports cropInvalid when a crop is left larger than the current width/height", () => {
      const { result } = setup();
      act(() => result.current.setCropOutput({ width: 1280, height: 768 }));
      act(() => result.current.setWidth(640, true)); // crop (1280) now exceeds width, which stays on-grid
      expect(result.current.validityReasons).toContain("cropInvalid");
      expect(result.current.validityReasons).not.toContain("dimensionsOffGrid");
      expect(result.current.isValid).toBe(false);
    });

    // NAG (2026-07-28): `deps.nag` propagates into both `validityReasons` and
    // `buildRequest` via the shared `nagRequestFields`/`isNagNegativeEmpty`
    // contract (`shell/nagSettings.ts`).
    describe("nag", () => {
      const ENABLED_NAG: NagSettings = {
        text: "blurry, low quality, distorted, watermark, text",
        enabled: true,
        scale: 11.0,
        tau: 2.5,
        alpha: 0.25,
        method: "nag",
        vsfScale: 1.5,
      };

      it("propagates into buildRequest: the 7 additive fields appear only while enabled", () => {
        const { result: off } = setup(undefined, 0, undefined, undefined);
        expect(off.current.buildRequest()).not.toHaveProperty("negative_prompt");

        const { result: on } = setup(undefined, 0, undefined, ENABLED_NAG);
        const request = on.current.buildRequest();
        expect(request.negative_prompt).toBe(ENABLED_NAG.text);
        expect(request.nag_enabled).toBe(true);
        expect(request.nag_scale).toBe(11.0);
        expect(request.nag_tau).toBe(2.5);
        expect(request.nag_alpha).toBe(0.25);
        expect(request.neg_method).toBe("nag");
        expect(request.vsf_scale).toBe(1.5);
      });

      it("propagates method 'vsf' into buildRequest's neg_method/vsf_scale", () => {
        const { result } = setup(undefined, 0, undefined, { ...ENABLED_NAG, method: "vsf", vsfScale: 2.5 });
        const request = result.current.buildRequest();
        expect(request.neg_method).toBe("vsf");
        expect(request.vsf_scale).toBe(2.5);
      });

      it("reports nagNegativeEmpty (and blocks isValid) when enabled with a blank body", () => {
        const { result } = setup(undefined, 0, undefined, { ...ENABLED_NAG, text: "   " });
        expect(result.current.validityReasons).toContain("nagNegativeEmpty");
        expect(result.current.isValid).toBe(false);
      });

      it("does not report nagNegativeEmpty while disabled, even with a blank body", () => {
        const { result } = setup(undefined, 0, undefined, { ...ENABLED_NAG, enabled: false, text: "" });
        expect(result.current.validityReasons).not.toContain("nagNegativeEmpty");
      });
    });

    // Acceleration (2026-07-31): `deps.acceleration` reaches `buildRequest`
    // through the single `accelerationRequestFields` contract.
    describe("acceleration", () => {
      it("omitted / all-defaults: buildRequest is byte-identical (key order included)", () => {
        const baseline = setup().result.current.buildRequest();
        const withDefaults = setup(undefined, 0, undefined, undefined, ACCELERATION_DEFAULTS).result.current.buildRequest();
        expect(JSON.stringify(withDefaults)).toBe(JSON.stringify(baseline));
        expect(withDefaults).not.toHaveProperty("attention_backend");
      });

      it("sage: exactly one key more", () => {
        const baseline = setup().result.current.buildRequest();
        const sage = setup(undefined, 0, undefined, undefined, {
          ...ACCELERATION_DEFAULTS,
          attentionBackend: "sage",
        }).result.current.buildRequest();
        expect(sage.attention_backend).toBe("sage");
        // Exactly ONE key more. Compared as a SET, not a sequence: the key
        // lands inside `buildChainRequest`'s object literal (right after the
        // NAG block), so the conditional fields assigned afterwards —
        // `crop_output` here — still trail it.
        expect(Object.keys(sage).sort()).toEqual([...Object.keys(baseline), "attention_backend"].sort());
      });

      it("block-swap prefetch off: exactly one key more", () => {
        // S4: default is now true, so OFF is what diverges.
        const baseline = setup().result.current.buildRequest();
        const prefetch = setup(undefined, 0, undefined, undefined, {
          ...ACCELERATION_DEFAULTS,
          blockSwapPrefetch: false,
        }).result.current.buildRequest();
        expect(prefetch.block_swap_prefetch).toBe(false);
        expect(Object.keys(prefetch).sort()).toEqual([...Object.keys(baseline), "block_swap_prefetch"].sort());
      });

      it("keep-resident on: exactly one key more", () => {
        // §48: server default is false, so ON is what diverges.
        const baseline = setup().result.current.buildRequest();
        const keepResident = setup(undefined, 0, undefined, undefined, {
          ...ACCELERATION_DEFAULTS,
          keepResident: true,
        }).result.current.buildRequest();
        expect(keepResident.keep_resident).toBe(true);
        expect(Object.keys(keepResident).sort()).toEqual([...Object.keys(baseline), "keep_resident"].sort());
      });

      it("fused GGUF dequant kernel off: exactly one key more (§1-11)", () => {
        // §51 (2026-08-04): the server default flipped to true, so OFF is what
        // diverges — and unlike keep-resident above there is no gate, so this
        // holds unconditionally.
        const baseline = setup().result.current.buildRequest();
        const fused = setup(undefined, 0, undefined, undefined, {
          ...ACCELERATION_DEFAULTS,
          fusedGgufDequantKernel: false,
        }).result.current.buildRequest();
        expect(fused.fused_gguf_dequant_kernel).toBe(false);
        expect(Object.keys(fused).sort()).toEqual([...Object.keys(baseline), "fused_gguf_dequant_kernel"].sort());
      });

      it("PrunaVAED: exactly one key more (§52)", () => {
        // §52 (2026-08-05): the server default is "default", so picking
        // PrunaVAED is what diverges — the same direction as keep-resident,
        // and like the fused kernel there is no gate of any kind.
        const baseline = setup().result.current.buildRequest();
        const request = setup(undefined, 0, undefined, undefined, {
          ...ACCELERATION_DEFAULTS,
          vaeMode: "prune_vaed",
        }).result.current.buildRequest();
        expect(request.vae_mode).toBe("prune_vaed");
        expect(Object.keys(request).sort()).toEqual([...Object.keys(baseline), "vae_mode"].sort());
        expect(request).not.toHaveProperty("attention_backend");
      });

      it("leaves vae_mode off entirely at the server default", () => {
        const request = setup(undefined, 0, undefined, undefined, ACCELERATION_DEFAULTS).result.current.buildRequest();
        expect(request).not.toHaveProperty("vae_mode");
        expect(request).not.toHaveProperty("attention_backend");
      });
    });

    it("normalizes an in-flight source upload to the single sourceUploading reason (R3)", async () => {
      const mockBridge = createMockBridge({ delayMs: 0, holdUploads: true });
      const { result } = renderHook(() =>
        useChainForm(FALLBACK_APP_CONFIG, "a cat riding a skateboard", { nativeBridge: mockBridge }),
      );
      act(() => {
        void result.current.sourceVideo.pick();
      });
      await waitFor(() => expect(result.current.sourceVideo.state.status).toBe("uploading"));

      const sourceReasons = result.current.validityReasons.filter((r) => SOURCE_CODES.includes(r));
      // Exactly ONE source reason even though all three source gates are false.
      expect(sourceReasons).toEqual(["sourceUploading"]);
      act(() => mockBridge.releaseUploads());
    });

    it("reports sourceNotReady for a failed source upload", async () => {
      const mockBridge = createMockBridge({ delayMs: 0, failUploadFile: "BACKEND_UNREACHABLE" });
      const { result } = renderHook(() =>
        useChainForm(FALLBACK_APP_CONFIG, "a cat riding a skateboard", { nativeBridge: mockBridge }),
      );
      act(() => {
        void result.current.sourceVideo.pick();
      });
      await waitFor(() => expect(result.current.sourceVideo.state.status).toBe("error"));

      const sourceReasons = result.current.validityReasons.filter((r) => SOURCE_CODES.includes(r));
      expect(sourceReasons).toEqual(["sourceNotReady"]);
      expect(result.current.isValid).toBe(false);
    });

    it("reports sourceNotReady when a ready V2V source has invalid context frames", async () => {
      const { result } = setup();
      act(() => {
        void result.current.sourceVideo.pick();
      });
      await waitFor(() => expect(result.current.sourceVideo.state.status).toBe("ready"));
      const id = result.current.clips[0]?.id;
      if (!id) throw new Error("expected a clip");
      act(() => result.current.setClipNumFrames(id, 9)); // shorter than default context_frames

      const sourceReasons = result.current.validityReasons.filter((r) => SOURCE_CODES.includes(r));
      expect(sourceReasons).toEqual(["sourceNotReady"]);
      expect(result.current.isValid).toBe(false);
    });
  });

  // X3: client-side gate for the backend's per-clip seam-blend minimum
  // (`num_frames >= 8*overlap_frames + 1`, HTTP 422 otherwise). DEFAULT_OVERLAP_FRAMES
  // is 3, so the pristine minimum is n = 25; the default clips (257f) clear it.
  describe("clipTooShortForOverlap (X3 short-clip SEAM BLEND gate)", () => {
    it("minFramesForOverlap tracks 8*overlapFrames+1 and updates with overlapFrames", () => {
      const { result } = setup();
      expect(result.current.overlapFrames).toBe(3);
      expect(result.current.minFramesForOverlap).toBe(25); // 8*3+1

      act(() => result.current.setOverlapFrames(1));
      expect(result.current.minFramesForOverlap).toBe(9); // 8*1+1

      act(() => result.current.setOverlapFrames(8));
      expect(result.current.minFramesForOverlap).toBe(65); // 8*8+1
    });

    it("flags a clip shorter than n (overlap=3 → n=25, a 9f clip) with clipTooShortForOverlap", () => {
      const { result } = setup();
      const id = result.current.clips[0]?.id;
      if (!id) throw new Error("expected a clip");
      // 9 is on the 8n+1 grid, so this trips ONLY the overlap-minimum gate, not
      // clipFramesOffGrid.
      act(() => result.current.setClipNumFrames(id, 9));
      expect(result.current.clips[0]?.numFrames).toBe(9);
      expect(result.current.validityReasons).toContain("clipTooShortForOverlap");
      expect(result.current.validityReasons).not.toContain("clipFramesOffGrid");
      expect(result.current.isValid).toBe(false);
    });

    it("passes a 9f clip when overlap=1 (n=9): the 9f clip meets the lowered minimum", () => {
      const { result } = setup();
      act(() => result.current.setOverlapFrames(1));
      // Set BOTH clips to 9 so the whole list clears the n=9 minimum.
      act(() => {
        for (const clip of result.current.clips) result.current.setClipNumFrames(clip.id, 9);
      });
      expect(result.current.clips.every((c) => c.numFrames === 9)).toBe(true);
      expect(result.current.validityReasons).not.toContain("clipTooShortForOverlap");
      expect(result.current.isValid).toBe(true);
    });

    it("treats the boundary num_frames = 8*overlap+1 (25f at overlap=3) as valid (server condition is strict <)", () => {
      const { result } = setup();
      // Set every clip exactly to the n=25 boundary.
      act(() => {
        for (const clip of result.current.clips) result.current.setClipNumFrames(clip.id, 25);
      });
      expect(result.current.clips.every((c) => c.numFrames === 25)).toBe(true);
      expect(result.current.validityReasons).not.toContain("clipTooShortForOverlap");
      expect(result.current.isValid).toBe(true);
    });

    it("re-trips when overlapFrames is raised so an on-grid clip becomes too short", () => {
      const { result } = setup();
      // Every clip at 25f: fine for overlap=3 (n=25)...
      act(() => {
        for (const clip of result.current.clips) result.current.setClipNumFrames(clip.id, 25);
      });
      expect(result.current.validityReasons).not.toContain("clipTooShortForOverlap");
      // ...raising overlap to 4 (n=33) makes the same 25f clips too short.
      act(() => result.current.setOverlapFrames(4));
      expect(result.current.minFramesForOverlap).toBe(33);
      expect(result.current.validityReasons).toContain("clipTooShortForOverlap");
      expect(result.current.isValid).toBe(false);
    });
  });

  // X4: the #1 right-click extend-video prefill opens with a SINGLE clip (V2V's
  // 1-clip floor), avoiding a stray 2nd clip card — the source is still idle at
  // mount, so without this the scratch 2-clip minimum would pad it.
  describe("forceSingleClip (X4 right-click v2v single clip)", () => {
    it("opens with a single clip when forceSingleClip is set", () => {
      const mockBridge = createMockBridge({ delayMs: 0 });
      const { result } = renderHook(() =>
        useChainForm(FALLBACK_APP_CONFIG, "a cat riding a skateboard", { nativeBridge: mockBridge }, { forceSingleClip: true }),
      );
      expect(result.current.clips).toHaveLength(1);
    });

    it("opens with the scratch 2-clip minimum when forceSingleClip is omitted", () => {
      const mockBridge = createMockBridge({ delayMs: 0 });
      const { result } = renderHook(() =>
        useChainForm(FALLBACK_APP_CONFIG, "a cat riding a skateboard", { nativeBridge: mockBridge }, {}),
      );
      expect(result.current.clips).toHaveLength(2);
    });

    it("does not disturb the manual-attach path: a plain mount still pads to 2 and attaching a source later relaxes to 1", async () => {
      const { result } = setup();
      expect(result.current.clips).toHaveLength(2); // manual/tab-opened default unchanged
      act(() => {
        void result.current.sourceVideo.pick();
      });
      await waitFor(() => expect(result.current.sourceVideo.state.status).toBe("ready"));
      expect(result.current.mode).toBe("v2v");
      expect(result.current.minClips).toBe(1);
    });
  });

  // §1-6 (V2Vリボン範囲トリム): 実際に上がる素材の尺で `context_frames` を賄えるか
  // というゲートB。判定は1フレームの安全マージン付きで、境界ちょうどは弾く側に倒す。
  //
  // Gate B: the material that will actually be uploaded must be able to supply
  // `contextFrames`. Only the right-click #1 path passes an `AttachedSourceInfo`,
  // so only it can arm this gate — a manual pick / drop stays unmeasured and
  // therefore ungated. FALLBACK_APP_CONFIG: fps 24, context_frames 73, so the
  // gate fires while `floor(d*24) - 1 < 73`, i.e. `floor(d*24) <= 73`.
  describe("sourceVideoTooShortForContext (§1-6 ゲートB)", () => {
    /** Attaches a video whose upload-time length is exactly `durationSec`, via
     * the trimmed branch (the only one with a length known up front). */
    async function attachTrimmed(result: { current: ReturnType<typeof useChainForm> }, durationSec: number) {
      await act(async () => {
        await result.current.attachSourceByPath("C:\\v\\clip.mp4", "clip.mp4", {
          trim: { trim: true, startSec: 0, durationSec },
          knownDurationSec: 600,
        });
      });
      await waitFor(() => expect(result.current.sourceVideo.state.status).toBe("ready"));
    }

    it("blocks Generate when the trimmed range is clearly too short", async () => {
      const { result } = setup();
      await attachTrimmed(result, 2.0); // floor(48) - 1 = 47 < 73
      expect(result.current.sourceDurationSec).toBe(2.0);
      expect(result.current.validityReasons).toContain("sourceVideoTooShortForContext");
      expect(result.current.isValid).toBe(false);
    });

    it("STILL blocks at the exact fit (floor(d*fps) === contextFrames) — the boundary is conservative", async () => {
      const { result } = setup();
      await attachTrimmed(result, 73 / 24); // floor(73) - 1 = 72 < 73
      expect(result.current.validityReasons).toContain("sourceVideoTooShortForContext");
    });

    it("passes with one frame of headroom over the carry-over", async () => {
      const { result } = setup();
      await attachTrimmed(result, 74 / 24); // floor(74) - 1 = 73, not < 73
      expect(result.current.validityReasons).not.toContain("sourceVideoTooShortForContext");
    });

    it("re-fires when contextFrames is raised afterwards", async () => {
      const { result } = setup();
      await attachTrimmed(result, 5.0); // floor(120) - 1 = 119 >= 73 -> fine at the default
      expect(result.current.validityReasons).not.toContain("sourceVideoTooShortForContext");

      act(() => result.current.setContextFrames(145)); // the config maximum
      expect(result.current.validityReasons).toContain("sourceVideoTooShortForContext");
      expect(result.current.isValid).toBe(false);
    });

    it("uses the FULL probed duration when the attach reports no trim", async () => {
      const { result } = setup();
      await act(async () => {
        await result.current.attachSourceByPath("C:\\v\\clip.mp4", "clip.mp4", {
          trim: { trim: false, reason: "unknownPlaybackPosition" },
          knownDurationSec: 1.0,
        });
      });
      await waitFor(() => expect(result.current.sourceVideo.state.status).toBe("ready"));
      expect(result.current.sourceDurationSec).toBe(1.0);
      expect(result.current.validityReasons).toContain("sourceVideoTooShortForContext");
    });

    it("never fires for a manually picked / dropped source (no measurement at all)", async () => {
      const { result } = setup();
      await act(async () => {
        // The contract-v7 drag-and-drop / unified-picker shape: no third argument.
        await result.current.attachSourceByPath("C:\\v\\clip.mp4", "clip.mp4");
      });
      await waitFor(() => expect(result.current.sourceVideo.state.status).toBe("ready"));
      expect(result.current.sourceDurationSec).toBeNull();
      expect(result.current.validityReasons).not.toContain("sourceVideoTooShortForContext");
    });

    it("clearSource drops the measurement again", async () => {
      const { result } = setup();
      await attachTrimmed(result, 2.0);
      expect(result.current.sourceDurationSec).toBe(2.0);
      act(() => result.current.clearSource());
      expect(result.current.sourceDurationSec).toBeNull();
      expect(result.current.validityReasons).not.toContain("sourceVideoTooShortForContext");
    });
  });

  // §1-6: 切り出しを頼んだのに `trimmed: true` が返らなかった場合。古いプラグイン
  // （クエリを捨てる）や ffmpeg 不在のバックエンドがこれに当たる。
  describe("sourceTrimFailed (§1-6)", () => {
    it("blocks Generate when a requested trim is not confirmed by the response", async () => {
      const inner = createMockBridge({ delayMs: 0 });
      const { result } = renderHook(() =>
        useChainForm(FALLBACK_APP_CONFIG, "a cat riding a skateboard", { nativeBridge: dropsTrimQuery(inner) }),
      );

      await act(async () => {
        await result.current.attachSourceByPath("C:\\v\\clip.mp4", "clip.mp4", {
          trim: { trim: true, startSec: 1, durationSec: 5 },
          knownDurationSec: 600,
        });
      });
      await waitFor(() => expect(result.current.sourceVideo.state.status).toBe("ready"));

      expect(result.current.sourceVideo.state.trimFailed).toBe(true);
      expect(result.current.validityReasons).toContain("sourceTrimFailed");
      expect(result.current.isValid).toBe(false);
    });

    it("does not fire when the trim IS confirmed", async () => {
      const { result } = setup();
      await act(async () => {
        await result.current.attachSourceByPath("C:\\v\\clip.mp4", "clip.mp4", {
          trim: { trim: true, startSec: 1, durationSec: 5 },
          knownDurationSec: 600,
        });
      });
      await waitFor(() => expect(result.current.sourceVideo.state.status).toBe("ready"));
      expect(result.current.sourceVideo.state.trimFailed).toBe(false);
      expect(result.current.validityReasons).not.toContain("sourceTrimFailed");
    });

    it("does not fire when no trim was requested at all", async () => {
      const inner = createMockBridge({ delayMs: 0 });
      const { result } = renderHook(() =>
        useChainForm(FALLBACK_APP_CONFIG, "a cat riding a skateboard", { nativeBridge: dropsTrimQuery(inner) }),
      );
      await act(async () => {
        await result.current.attachSourceByPath("C:\\v\\clip.mp4", "clip.mp4");
      });
      await waitFor(() => expect(result.current.sourceVideo.state.status).toBe("ready"));
      expect(result.current.sourceVideo.state.trimFailed).toBe(false);
      expect(result.current.validityReasons).not.toContain("sourceTrimFailed");
    });
  });

  // ── §1-16 長尺A2V (Step F2) ───────────────────────────────────────────────
  // FALLBACK_APP_CONFIG: 24fps, 257-frame clips, kv=3. The geometry constants
  // every expectation below leans on (all pinned in `chainUtils.test.ts` /
  // `audioFit.test.ts` too):
  //   audioLatentsRequired([257,257], 24, 3) === 518  -> 20.72s of audio
  //   audioLatentsRequired([257, 25], 24, 3) === 276  -> 11.04s (2-clip floor)
  //   maxChainAudioLatents(24, 3)            === 11618 (24x481)
  //   maxChainAudioLatents(60, 3)            ===  4647
  //   audioLatentsAvailable(sec)             === round(sec * 25)

  /** Renders the hook against a mock bridge whose two audio probes answer
   * `probes`. Returns the probe call counters as well. */
  function setupAudio(probes: AudioProbeFixture = {}, options: MockBridgeOptions = {}) {
    const calls = { wav: 0, media: 0 };
    const bridge = withAudioProbes(createMockBridge({ delayMs: 0, ...options }), probes, calls);
    const { result } = renderHook(() =>
      useChainForm(FALLBACK_APP_CONFIG, "a cat riding a skateboard", { nativeBridge: bridge }),
    );
    return { result, calls };
  }

  type ChainForm = ReturnType<typeof useChainForm>;

  /** Attaches an audio file the drag-and-drop / right-click way. With
   * `knownDurationSec` this is the right-click route (an exact length the caller
   * already measured); without it, the probes decide. */
  async function attachAudio(
    result: { current: ChainForm },
    opts: { knownDurationSec?: number; fileName?: string } = {},
  ) {
    const fileName = opts.fileName ?? "track.wav";
    await act(async () => {
      await result.current.attachAudioByPath(
        `C:\\audio\\${fileName}`,
        fileName,
        opts.knownDurationSec === undefined ? undefined : { knownDurationSec: opts.knownDurationSec },
      );
    });
  }

  describe("§1-16: the clip `intact` flag", () => {
    it("every card is born intact — the initial pair, and one added by hand", () => {
      const { result } = setup();
      expect(result.current.clips.map((clip) => clip.intact)).toEqual([true, true]);
      act(() => result.current.addClip());
      expect(result.current.clips[2]?.intact).toBe(true);
    });

    it("a REAL duration change consumes the flag, irreversibly", () => {
      const { result } = setup();
      const id = result.current.clips[0]?.id;
      if (!id) throw new Error("expected a clip");

      act(() => result.current.setClipNumFrames(id, 121));
      expect(result.current.clips[0]?.intact).toBe(false);

      // Putting the ORIGINAL length back does not un-touch the card.
      act(() => result.current.setClipNumFrames(id, 361));
      expect(result.current.clips[0]?.numFrames).toBe(361);
      expect(result.current.clips[0]?.intact).toBe(false);
    });

    it("re-setting the SAME (post-snap) duration leaves the card intact", () => {
      const { result } = setup();
      const id = result.current.clips[0]?.id;
      if (!id) throw new Error("expected a clip");

      act(() => result.current.setClipNumFrames(id, 361));
      expect(result.current.clips[0]?.intact).toBe(true);
      // 362 snaps back to 361 — the APPLIED value is unchanged, so is the flag.
      act(() => result.current.setClipNumFrames(id, 362));
      expect(result.current.clips[0]?.numFrames).toBe(361);
      expect(result.current.clips[0]?.intact).toBe(true);
    });

    it("a prompt edit consumes the flag; re-setting the same text does not", () => {
      const { result } = setup();
      const id = result.current.clips[1]?.id;
      if (!id) throw new Error("expected a second clip");

      act(() => result.current.setClipPrompt(id, ""));
      expect(result.current.clips[1]?.intact).toBe(true); // no actual change

      act(() => result.current.setClipPrompt(id, "a sunset"));
      expect(result.current.clips[1]?.intact).toBe(false);
      expect(result.current.clips[0]?.intact).toBe(true); // the other card is untouched
    });

    it("applyPreset does NOT consume it (judgement point 7: presets are not user edits)", () => {
      const { result } = setup();
      act(() => result.current.applyPreset("smoke_test"));
      expect(result.current.clips.every((clip) => clip.numFrames === 17)).toBe(true);
      expect(result.current.clips.map((clip) => clip.intact)).toEqual([true, true]);
    });

    it("applyPreset leaves an ALREADY-touched card touched", () => {
      const { result } = setup();
      const id = result.current.clips[0]?.id;
      if (!id) throw new Error("expected a clip");
      act(() => result.current.setClipNumFrames(id, 121));
      act(() => result.current.applyPreset("smoke_test"));
      expect(result.current.clips.map((clip) => clip.intact)).toEqual([false, true]);
    });

    it("removeClip leaves the surviving cards' flags alone", () => {
      const { result } = setup();
      act(() => result.current.addClip());
      const touched = result.current.clips[1]?.id;
      const doomed = result.current.clips[0]?.id;
      if (!touched || !doomed) throw new Error("expected three clips");
      act(() => result.current.setClipPrompt(touched, "a sunset"));

      act(() => result.current.removeClip(doomed));
      expect(result.current.clips).toHaveLength(2);
      expect(result.current.clips[0]?.intact).toBe(false); // still the touched one
      expect(result.current.clips[1]?.intact).toBe(true);
    });
  });

  describe("§1-16: audio length resolution (three tiers)", () => {
    it("adopts a caller-supplied knownDurationSec and never probes at all", async () => {
      const { result, calls } = setupAudio({ wav: { durationSec: 99, isWav: true } });
      await attachAudio(result, { knownDurationSec: 21 });
      await waitFor(() => expect(result.current.audioDurationSec).toBe(21));
      expect(result.current.audioProbeFailed).toBe(false);
      expect(calls).toEqual({ wav: 0, media: 0 });
    });

    it("reads the wav header for a .wav (fs.probeAudioDuration only)", async () => {
      const { result, calls } = setupAudio({ wav: { durationSec: 21, isWav: true }, mediaInfoDurationSec: 99 });
      await attachAudio(result);
      await waitFor(() => expect(result.current.audioDurationSec).toBe(21));
      expect(calls.wav).toBe(1);
      expect(calls.media).toBe(0); // the wav answer settled it
    });

    it("falls back to fs.probeMediaInfo when the file is not a wav", async () => {
      const { result, calls } = setupAudio({ mediaInfoDurationSec: 21 }, {});
      await attachAudio(result, { fileName: "track.mp3" });
      await waitFor(() => expect(result.current.audioDurationSec).toBe(21));
      expect(calls).toEqual({ wav: 1, media: 1 });
      expect(result.current.audioProbeFailed).toBe(false);
    });

    it("flags audioProbeFailed when BOTH probes come up empty (server becomes the authority)", async () => {
      const { result } = setupAudio({});
      await attachAudio(result, { fileName: "track.ogg" });
      await waitFor(() => expect(result.current.audioProbeFailed).toBe(true));
      expect(result.current.audioDurationSec).toBeNull();
      // Nothing length-derived may fire without a measurement.
      expect(result.current.validityReasons).not.toContain("audioTooShort");
      expect(result.current.validityReasons).not.toContain("audioTooShortForChain");
      expect(result.current.validityReasons).not.toContain("audioTooLong");
      // ...and the auto-fit is a no-op.
      act(() => result.current.adjustClipsForAudio());
      expect(result.current.audioFitEvent).toBeNull();
      expect(result.current.clips).toHaveLength(2);
    });

    it("pickAudio attaches what the native dialog returned, and probes it", async () => {
      const { result } = setupAudio({ wav: { durationSec: 21, isWav: true } }, { pickFileName: "bgm.wav" });
      await act(async () => {
        await result.current.pickAudio();
      });
      await waitFor(() => expect(result.current.audioDurationSec).toBe(21));
      expect(result.current.sourceAudio.state.fileName).toBe("bgm.wav");
      expect(result.current.audioError).toBeNull();
    });

    it("pickAudio leaves the slot untouched on CANCELLED, and reports any other dialog failure", async () => {
      const cancelled = setupAudio({}, { failPickFile: "CANCELLED" });
      await act(async () => {
        await cancelled.result.current.pickAudio();
      });
      expect(cancelled.result.current.hasSourceAudio).toBe(false);
      expect(cancelled.result.current.audioError).toBeNull();

      const failed = setupAudio({}, { failPickFile: "DIALOG_FAILED" });
      await act(async () => {
        await failed.result.current.pickAudio();
      });
      expect(failed.result.current.hasSourceAudio).toBe(false);
      expect(failed.result.current.audioError).toBe("DIALOG_FAILED");
    });

    it("survives a REJECTING fs.probeMediaInfo (native's BAD_REQUEST) as a plain probe failure", async () => {
      const { result } = setupAudio({ mediaInfoRejects: true });
      await attachAudio(result, { fileName: "track.m4a" });
      await waitFor(() => expect(result.current.audioProbeFailed).toBe(true));
      expect(result.current.audioDurationSec).toBeNull();
    });
  });

  describe("§1-16: attach-time hard error (owner spec 8)", () => {
    it("detaches a track no chain could ever consume, and reports audioAttachError", async () => {
      // 600s -> 15000 audio latents, far past maxChainAudioLatents(24,3)=11618.
      const { result } = setupAudio();
      await attachAudio(result, { knownDurationSec: 600 });
      await waitFor(() => expect(result.current.audioAttachError).toBe("tooLong"));
      expect(result.current.sourceAudio.state.status).toBe("idle");
      expect(result.current.hasSourceAudio).toBe(false);
      expect(result.current.audioDurationSec).toBeNull();
      // Nothing was ever laid out for it.
      expect(result.current.clips).toHaveLength(2);
      expect(result.current.audioFitEvent).toBeNull();
    });

    it("the next attach clears the error", async () => {
      const { result } = setupAudio();
      await attachAudio(result, { knownDurationSec: 600 });
      await waitFor(() => expect(result.current.audioAttachError).toBe("tooLong"));

      await attachAudio(result, { knownDurationSec: 21, fileName: "short.wav" });
      await waitFor(() => expect(result.current.audioDurationSec).toBe(21));
      expect(result.current.audioAttachError).toBeNull();
    });
  });

  describe("§1-16: the once-per-track auto-fit", () => {
    it("fires exactly once for an attached track, and not again on unrelated re-renders", async () => {
      const { result } = setupAudio();
      await attachAudio(result, { knownDurationSec: 40 });
      await waitFor(() => expect(result.current.audioFitEvent?.id).toBe(1));
      expect(result.current.audioFitEvent?.outcome).toBe("adjusted");
      expect(result.current.clips).toHaveLength(4);

      // A re-render with the SAME track must not re-run it.
      act(() => result.current.setAddedClipFrames(129));
      act(() => result.current.setFrameRate(30));
      expect(result.current.audioFitEvent?.id).toBe(1);
      expect(result.current.clips).toHaveLength(4);
    });

    it("fires again — exactly once — for a REPLACEMENT track", async () => {
      const { result } = setupAudio();
      await attachAudio(result, { knownDurationSec: 40 });
      await waitFor(() => expect(result.current.audioFitEvent?.id).toBe(1));
      const firstId = result.current.sourceAudio.state.id;

      await attachAudio(result, { knownDurationSec: 21, fileName: "other.wav" });
      await waitFor(() => expect(result.current.audioFitEvent?.id).toBe(2));
      expect(result.current.sourceAudio.state.id).not.toBe(firstId);
      expect(result.current.audioFitEvent?.durationSec).toBe(21);
    });
  });

  describe("§1-16: applying the auto-fit plan", () => {
    it("keeps a touched card's length, position and prompt; new cards are born intact", async () => {
      const { result } = setupAudio();
      const secondId = result.current.clips[1]?.id;
      const firstId = result.current.clips[0]?.id;
      if (!secondId || !firstId) throw new Error("expected two clips");
      act(() => result.current.setClipNumFrames(secondId, 121));
      act(() => result.current.setClipPrompt(secondId, "a sunset"));

      await attachAudio(result, { knownDurationSec: 40 });
      await waitFor(() => expect(result.current.audioFitEvent?.outcome).toBe("adjusted"));

      // The pinned card stays at index 1, at 121 frames, with its prompt.
      expect(result.current.clips).toHaveLength(5);
      expect(result.current.clips[1]?.id).toBe(secondId);
      expect(result.current.clips[1]?.numFrames).toBe(121);
      expect(result.current.clips[1]?.prompt).toBe("a sunset");
      expect(result.current.clips[1]?.intact).toBe(false);
      // The resized intact card kept its identity (not recreated) and its flag.
      expect(result.current.clips[0]?.id).toBe(firstId);
      expect(result.current.clips[0]?.intact).toBe(true);
      // ...and the appended cards are intact too.
      expect(result.current.clips.slice(2).every((clip) => clip.intact === true)).toBe(true);
    });

    it("leaves the clip list untouched when nothing fits (cannotFit)", async () => {
      const { result } = setupAudio();
      const ids = result.current.clips.map((clip) => clip.id);
      await attachAudio(result, { knownDurationSec: 5 }); // well below the 2-clip floor
      await waitFor(() => expect(result.current.audioFitEvent?.outcome).toBe("cannotFit"));
      expect(result.current.clips.map((clip) => clip.id)).toEqual(ids);
      expect(result.current.clips.map((clip) => clip.numFrames)).toEqual([361, 361]);
    });

    it("reports cappedAtMaxClips with the leftover tail at the 24-card ceiling", async () => {
      const { result } = setupAudio();
      await attachAudio(result, { knownDurationSec: 400 });
      await waitFor(() => expect(result.current.audioFitEvent?.outcome).toBe("cappedAtMaxClips"));
      expect(result.current.clips).toHaveLength(24);
      expect(result.current.audioAtMaxClips).toBe(true);
      // available 10000 - required 6018 = 3982 latents = 159.28s unused.
      expect(result.current.audioSurplusSec).toBeCloseTo(159.28, 5);
      // Advisory only — never a Generate gate.
      expect(result.current.validityReasons).not.toContain("audioTooShort");
      expect(result.current.validityReasons).not.toContain("audioTooLong");
    });
  });

  describe("§1-16: audio validity reasons", () => {
    it("audioConflictsWithSourceVideo while both slots are filled — and clears with the video", async () => {
      const { result } = setupAudio();
      await attachAudio(result, { knownDurationSec: 21 });
      await waitFor(() => expect(result.current.audioDurationSec).toBe(21));
      expect(result.current.validityReasons).not.toContain("audioConflictsWithSourceVideo");

      await act(async () => {
        await result.current.attachSourceByPath("C:\\v\\clip.mp4", "clip.mp4");
      });
      await waitFor(() => expect(result.current.sourceVideo.state.status).toBe("ready"));
      expect(result.current.validityReasons).toContain("audioConflictsWithSourceVideo");
      expect(result.current.isValid).toBe(false);

      act(() => result.current.clearSource());
      expect(result.current.validityReasons).not.toContain("audioConflictsWithSourceVideo");
    });

    it("audioUploading while the upload is still in flight", async () => {
      const { result } = setupAudio({ wav: { durationSec: 21, isWav: true } }, { delayMs: 20 });
      let pending: Promise<void> | undefined;
      act(() => {
        pending = result.current.attachAudioByPath("C:\\audio\\track.wav", "track.wav");
      });
      expect(result.current.sourceAudio.state.status).toBe("uploading");
      expect(result.current.validityReasons).toContain("audioUploading");
      expect(result.current.isValid).toBe(false);

      await act(async () => {
        await pending;
      });
      await waitFor(() => expect(result.current.audioDurationSec).toBe(21));
      expect(result.current.validityReasons).not.toContain("audioUploading");
    });

    it("audioNotReady when the upload failed", async () => {
      const { result } = setupAudio({}, { failUploadFile: "BACKEND_UNREACHABLE" });
      await attachAudio(result);
      await waitFor(() => expect(result.current.sourceAudio.state.status).toBe("error"));
      expect(result.current.validityReasons).toContain("audioNotReady");
      expect(result.current.isValid).toBe(false);
    });

    it("audioTooShortForChain (and NOT audioTooShort) for a track below the 2-clip floor", async () => {
      const { result } = setupAudio();
      await attachAudio(result, { knownDurationSec: 5 });
      await waitFor(() => expect(result.current.audioDurationSec).toBe(5));
      expect(result.current.validityReasons).toContain("audioTooShortForChain");
      expect(result.current.validityReasons).not.toContain("audioTooShort");
      expect(result.current.isValid).toBe(false);
    });

    it("audioTooShort (and NOT audioTooShortForChain) once the clip list outgrows the track", async () => {
      const { result } = setupAudio();
      // 15s auto-fits to [257, 113] — a fit, so no gate yet.
      await attachAudio(result, { knownDurationSec: 15 });
      await waitFor(() => expect(result.current.clips[1]?.numFrames).toBe(113));
      expect(result.current.validityReasons).not.toContain("audioTooShort");

      // The user then stretches that card back out by hand.
      const secondId = result.current.clips[1]?.id;
      if (!secondId) throw new Error("expected a second clip");
      act(() => result.current.setClipNumFrames(secondId, 257));
      expect(result.current.validityReasons).toContain("audioTooShort");
      expect(result.current.validityReasons).not.toContain("audioTooShortForChain");
      expect(result.current.isValid).toBe(false);
    });

    it("audioTooLong defensively, after the frame rate is raised past a borderline attach", async () => {
      // 200s = 5000 latents: fine at 24fps (ceiling 11618), too long at 60 (4647).
      const { result } = setupAudio();
      await attachAudio(result, { knownDurationSec: 200 });
      await waitFor(() => expect(result.current.audioDurationSec).toBe(200));
      expect(result.current.validityReasons).not.toContain("audioTooLong");
      // The slot is NOT detached retroactively — only the attach-time check does that.
      act(() => result.current.setFrameRate(60));
      expect(result.current.sourceAudio.state.status).toBe("ready");
      expect(result.current.validityReasons).toContain("audioTooLong");
      expect(result.current.validityReasons).not.toContain("audioTooShort");
      expect(result.current.audioAttachError).toBeNull();
    });

    it("chainLayoutInvalid at 29.97fps — and never without audio attached", async () => {
      // [257,257] @ 29.97 is `audioReassemblyMismatch` server-side; at 24 it is fine.
      const bare = setup();
      act(() => bare.result.current.setFrameRate(29.97));
      expect(bare.result.current.validityReasons).not.toContain("chainLayoutInvalid");

      const { result } = setupAudio();
      await attachAudio(result, { knownDurationSec: 21 });
      // The default clips (361,361, raised 2026-08-19) no longer already fit 21s
      // of audio, so the resolver now shrinks them down to [257,257] instead of
      // leaving them untouched — `adjusted`, not `alreadyFits`.
      await waitFor(() => expect(result.current.audioFitEvent?.outcome).toBe("adjusted"));
      expect(result.current.clips.map((clip) => clip.numFrames)).toEqual([257, 257]);
      expect(result.current.validityReasons).not.toContain("chainLayoutInvalid");

      act(() => result.current.setFrameRate(29.97));
      expect(result.current.validityReasons).toContain("chainLayoutInvalid");
      expect(result.current.isValid).toBe(false);
    });
  });

  describe("§1-16: buildRequest source_audio wiring", () => {
    it("sends source_audio once the track is ready", async () => {
      const { result } = setupAudio();
      await attachAudio(result, { knownDurationSec: 21 });
      await waitFor(() => expect(result.current.sourceAudio.state.id).not.toBeNull());
      const request = result.current.buildRequest();
      expect(request.source_audio).toEqual({ audio_id: result.current.sourceAudio.state.id });
      expect(request).not.toHaveProperty("source_video");
    });

    it("omits it entirely when no audio is attached", () => {
      const { result } = setup();
      expect(result.current.buildRequest()).not.toHaveProperty("source_audio");
    });

    it("drops it defensively while a source VIDEO is also attached", async () => {
      const { result } = setupAudio();
      await attachAudio(result, { knownDurationSec: 21 });
      await waitFor(() => expect(result.current.sourceAudio.state.id).not.toBeNull());
      await act(async () => {
        await result.current.attachSourceByPath("C:\\v\\clip.mp4", "clip.mp4");
      });
      await waitFor(() => expect(result.current.sourceVideo.state.status).toBe("ready"));

      const request = result.current.buildRequest();
      expect(request).not.toHaveProperty("source_audio");
      expect(request.source_video).toBeDefined();
    });
  });

  describe("§1-16: the 追加クリップの長さ slider", () => {
    it("opens at 257 and snaps onto the 8n+1 grid", () => {
      const { result } = setup();
      expect(result.current.addedClipFrames).toBe(257);
      act(() => result.current.setAddedClipFrames(130));
      expect(result.current.addedClipFrames).toBe(129);
      act(() => result.current.setAddedClipFrames(9999));
      expect(result.current.addedClipFrames).toBe(481); // MAX_CLIP_NUM_FRAMES
      act(() => result.current.setAddedClipFrames(0));
      expect(result.current.addedClipFrames).toBe(9); // MIN_CLIP_NUM_FRAMES
    });

    it("is re-seeded by applyPreset to THAT preset's recommendation (judgement point 7)", () => {
      const { result } = setup();
      act(() => result.current.setAddedClipFrames(129));
      expect(result.current.addedClipFrames).toBe(129);

      act(() => result.current.applyPreset("smoke_test")); // 384x256, 17f, no spill entry
      expect(result.current.addedClipFrames).toBe(17);

      act(() => result.current.applyPreset("minimal")); // 512x320 -> spill_free_frames 481
      expect(result.current.addedClipFrames).toBe(481);
    });

    it("drives the length the auto-fit gives the cards it may resize", async () => {
      const { result } = setupAudio();
      act(() => result.current.setAddedClipFrames(129));
      await attachAudio(result, { knownDurationSec: 21 });
      await waitFor(() => expect(result.current.audioFitEvent).not.toBeNull());
      // Every card but the remainder-absorbing one sits at the slider value.
      expect(result.current.clips[0]?.numFrames).toBe(129);
    });
  });

  describe("§1-16: every audio reason code is spelled out to the user", () => {
    it("buildChainReasonMessages has a non-empty line for each of them, in both languages", () => {
      const audioCodes = [
        "audioConflictsWithSourceVideo",
        "audioUploading",
        "audioNotReady",
        "audioTooLong",
        "audioTooShortForChain",
        "audioTooShort",
        "chainLayoutInvalid",
      ];
      // The batch's suppression list must stay in step with the codes themselves
      // — that is the whole point of exporting it.
      expect([...CHAIN_AUDIO_REASON_CODES].sort()).toEqual([...audioCodes].sort());
      for (const dictionary of [en, ja]) {
        const messages = buildChainReasonMessages(dictionary, { minFramesForOverlap: 25, minClips: 2 });
        for (const code of audioCodes) {
          expect(messages[code], code).toBeTruthy();
        }
      }
    });
  });

  describe("§1-16: advisory readouts and clearAudio", () => {
    it("audioSegmentWindowsForClips gives one window per clip, ending at a_total", async () => {
      const { result } = setupAudio();
      expect(result.current.audioSegmentWindowsForClips).toBeNull();

      await attachAudio(result, { knownDurationSec: 21 });
      await waitFor(() => expect(result.current.audioSegmentWindowsForClips).not.toBeNull());
      const windows = result.current.audioSegmentWindowsForClips;
      if (!windows) throw new Error("expected windows");
      expect(windows).toHaveLength(2);
      expect(windows[0]?.startSec).toBe(0);
      // a_total = 518 latents / 25 = 20.72s.
      expect(windows[1]?.endSec).toBeCloseTo(20.72, 5);
    });

    it("clearAudio empties the slot and every derived value, but keeps the clip list", async () => {
      const { result } = setupAudio();
      await attachAudio(result, { knownDurationSec: 40 });
      await waitFor(() => expect(result.current.clips).toHaveLength(4));

      act(() => result.current.clearAudio());
      expect(result.current.hasSourceAudio).toBe(false);
      expect(result.current.sourceAudio.state.status).toBe("idle");
      expect(result.current.audioDurationSec).toBeNull();
      expect(result.current.audioProbeFailed).toBe(false);
      expect(result.current.audioAttachError).toBeNull();
      expect(result.current.audioSegmentWindowsForClips).toBeNull();
      expect(result.current.audioSurplusSec).toBeNull();
      expect(result.current.audioAtMaxClips).toBe(false);
      expect(result.current.validityReasons).not.toContain("audioTooShort");
      // The chain the auto-fit produced is now simply the user's chain.
      expect(result.current.clips).toHaveLength(4);
      expect(result.current.buildRequest()).not.toHaveProperty("source_audio");
    });
  });

  // ── §1-15 参照動画: ONE long reference video for the whole chain ────────────
  describe("§1-15 reference video slot", () => {
    const CONTROL_NAMES = new Set(["canny-control"]);
    const DEPTH_NAMES = new Set(["depth-control"]);

    function setupReference(
      options: {
        prompt?: string;
        uploads?: { query?: Record<string, string> | undefined }[];
        probeWidth?: number;
        probeHeight?: number;
        depthLoraNames?: ReadonlySet<string>;
      } = {},
    ) {
      const uploads = options.uploads ?? [];
      const inner = createMockBridge({
        delayMs: 0,
        probeMediaInfoWidth: options.probeWidth ?? 0,
        probeMediaInfoHeight: options.probeHeight ?? 0,
      });
      const nativeBridge = recordsUploadQuery(inner, uploads);
      const { result } = renderHook(() =>
        useChainForm(FALLBACK_APP_CONFIG, options.prompt ?? "a cat riding a skateboard", {
          nativeBridge,
          controlLoraNames: CONTROL_NAMES,
          ...(options.depthLoraNames ? { depthLoraNames: options.depthLoraNames } : {}),
        }),
      );
      return { result, uploads };
    }

    async function attachReference(result: { current: { attachReferenceByPath: (a: string, b: string) => Promise<void>; referenceVideo: { state: { status: string } } } }) {
      await act(async () => {
        await result.current.attachReferenceByPath("C:\\ref\\long.mp4", "long.mp4");
      });
      await waitFor(() => expect(result.current.referenceVideo.state.status).toBe("ready"));
    }

    it("starts empty and sends no reference fields at all", () => {
      const { result } = setupReference();
      expect(result.current.hasReferenceVideo).toBe(false);
      expect(result.current.isReferenceActive).toBe(false);
      const request = result.current.buildRequest();
      expect(request).not.toHaveProperty("reference_video_id");
      expect(request).not.toHaveProperty("conditioning_attention_strength");
      expect(request).not.toHaveProperty("reference_video_strength");
    });

    it("uploads with the max_frames cap (11544, as a string) and never arms trimFailed", async () => {
      const { result, uploads } = setupReference();
      await attachReference(result);

      expect(uploads).toHaveLength(1);
      expect(uploads[0]?.query).toEqual({ max_frames: "11544" });
      // The mock answers `trimmed: false` for a query with no trim keys — which
      // is the CORRECT answer for a file that was already short enough, so the
      // §1-6 trim gate must stay silent.
      expect(result.current.referenceVideo.state.trimFailed).toBe(false);
      expect(result.current.validityReasons).not.toContain("sourceTrimFailed");
    });

    // ── §1-15 W4 (2026-08-11): the right-click `reference-video-chain` route ──
    // It is the only caller that passes a trim decision, so these cover the two
    // things that only exist for it: the merged+clamped query it sends, and the
    // Generate gate that fires when the cut is not confirmed.
    describe("ribbon-range trim (§1-15 W4)", () => {
      it("merges the trim window with the max_frames cap instead of replacing it", async () => {
        const { result, uploads } = setupReference();
        await act(async () => {
          await result.current.attachReferenceByPath("C:\\ref\\long.mp4", "long.mp4", {
            trim: true,
            startSec: 12.5,
            durationSec: 5,
          });
        });
        await waitFor(() => expect(result.current.referenceVideo.state.status).toBe("ready"));

        // Both key sets ride on the SAME request — the cap is not dropped just
        // because a window is present (the server ignores it in that case, but
        // sending it keeps every reference upload's shape identical).
        expect(uploads).toHaveLength(1);
        expect(uploads[0]?.query).toEqual({
          max_frames: "11544",
          trim_start_sec: "12.500",
          trim_duration_sec: "5.000",
        });
      });

      it("clamps trim_duration_sec to MAX_CHAIN_TOTAL_FRAMES / generation fps", async () => {
        const { result, uploads } = setupReference();
        // A 10-minute ribbon at the config's 24fps. The cap the (now-ignored)
        // `max_frames` used to enforce is 11544 frames = 481.000s there, so the
        // window must arrive clamped to that rather than at the full 600s.
        await act(async () => {
          await result.current.attachReferenceByPath("C:\\ref\\long.mp4", "long.mp4", {
            trim: true,
            startSec: 0,
            durationSec: 600,
          });
        });
        await waitFor(() => expect(result.current.referenceVideo.state.status).toBe("ready"));

        expect(uploads[0]?.query?.trim_duration_sec).toBe("481.000");
        expect(uploads[0]?.query?.trim_start_sec).toBe("0.000");
      });

      it("sends no trim keys for a no-trim decision (byte-identical to a manual attach)", async () => {
        const { result, uploads } = setupReference();
        await act(async () => {
          await result.current.attachReferenceByPath("C:\\ref\\long.mp4", "long.mp4", {
            trim: false,
            reason: "noItem",
          });
        });
        await waitFor(() => expect(result.current.referenceVideo.state.status).toBe("ready"));
        expect(uploads[0]?.query).toEqual({ max_frames: "11544" });
      });

      it("blocks Generate with referenceTrimFailed when the cut is not confirmed", async () => {
        const inner = createMockBridge({ delayMs: 0 });
        const { result } = renderHook(() =>
          useChainForm(FALLBACK_APP_CONFIG, "a cat <lora:Pixar_Toon:1.0> riding", {
            nativeBridge: dropsTrimQuery(inner),
          }),
        );
        await act(async () => {
          await result.current.attachReferenceByPath("C:\\ref\\long.mp4", "long.mp4", {
            trim: true,
            startSec: 1,
            durationSec: 5,
          });
        });
        await waitFor(() => expect(result.current.referenceVideo.state.status).toBe("ready"));

        expect(result.current.referenceVideo.state.trimFailed).toBe(true);
        expect(result.current.validityReasons).toContain("referenceTrimFailed");
        expect(result.current.isValid).toBe(false);
      });

      it("stays silent when the cut IS confirmed, and when no trim was asked for", async () => {
        const { result } = setupReference({ prompt: "a cat <lora:Pixar_Toon:1.0> riding" });
        await act(async () => {
          await result.current.attachReferenceByPath("C:\\ref\\long.mp4", "long.mp4", {
            trim: true,
            startSec: 1,
            durationSec: 5,
          });
        });
        await waitFor(() => expect(result.current.referenceVideo.state.status).toBe("ready"));
        expect(result.current.referenceVideo.state.trimFailed).toBe(false);
        expect(result.current.validityReasons).not.toContain("referenceTrimFailed");

        // The manual 📁 / drag-and-drop route sends only `max_frames`, for which
        // `trimmed: false` is the correct answer — it must never arm the gate.
        const { result: manual } = setupReference({ prompt: "a cat <lora:Pixar_Toon:1.0> riding" });
        await attachReference(manual);
        expect(manual.current.validityReasons).not.toContain("referenceTrimFailed");
      });
    });

    it("reports referenceNeedsLoras until a loRA exists, then goes valid", async () => {
      const { result } = setupReference();
      await attachReference(result);
      expect(result.current.referenceNeedsLoras).toBe(true);
      expect(result.current.validityReasons).toContain("referenceNeedsLoras");
      // ...and the request drops the reference rather than sending a body the
      // server would 422.
      expect(result.current.buildRequest()).not.toHaveProperty("reference_video_id");

      act(() => result.current.setControlLora({ name: "canny-control", strength: 1.0 }));
      expect(result.current.referenceNeedsLoras).toBe(false);
      expect(result.current.validityReasons).toEqual([]);
    });

    it("reports referenceConflictsWithSourceVideo with a V2V source attached, and buildRequest drops the reference", async () => {
      const { result } = setupReference({ prompt: "a cat <lora:Pixar_Toon:1.0> riding" });
      await attachReference(result);
      await act(async () => {
        await result.current.attachSourceByPath("C:\\v\\clip.mp4", "clip.mp4");
      });
      await waitFor(() => expect(result.current.sourceVideo.state.status).toBe("ready"));

      expect(result.current.validityReasons).toContain("referenceConflictsWithSourceVideo");
      const request = result.current.buildRequest();
      expect(request).toHaveProperty("source_video");
      expect(request).not.toHaveProperty("reference_video_id");
    });

    it("reports referenceUploading while the upload is in flight and referenceNotReady when it fails", async () => {
      const uploading = createMockBridge({ delayMs: 0, holdUploads: true });
      const { result } = renderHook(() =>
        useChainForm(FALLBACK_APP_CONFIG, "a cat <lora:Pixar_Toon:1.0> riding", { nativeBridge: uploading }),
      );
      act(() => {
        void result.current.attachReferenceByPath("C:\\ref\\long.mp4", "long.mp4");
      });
      await waitFor(() => expect(result.current.referenceVideo.state.status).toBe("uploading"));
      expect(result.current.validityReasons).toContain("referenceUploading");

      const failing = createMockBridge({ delayMs: 0, failUploadFile: "BACKEND_UNREACHABLE" });
      const { result: failed } = renderHook(() =>
        useChainForm(FALLBACK_APP_CONFIG, "a cat <lora:Pixar_Toon:1.0> riding", { nativeBridge: failing }),
      );
      await act(async () => {
        await failed.current.attachReferenceByPath("C:\\ref\\long.mp4", "long.mp4");
      });
      await waitFor(() => expect(failed.current.referenceVideo.state.status).toBe("error"));
      expect(failed.current.validityReasons).toContain("referenceNotReady");
    });

    it("depthChainUnsupported fires for a depth adapter regardless of clip count (owner decision 2026-08-11: FE is deliberately stricter than the server, which allows depth on a single-clip chain)", async () => {
      const { result } = setupReference({
        prompt: "a cat <lora:depth-control:1.0> riding",
        depthLoraNames: DEPTH_NAMES,
      });
      await attachReference(result);
      expect(result.current.clips.length).toBeGreaterThan(1);
      expect(result.current.validityReasons).toContain("depthChainUnsupported");

      // Down to a single clip: still blocked. The server would accept depth
      // on this exact shape, but the FE does not carve out that exception —
      // this state is already unreachable via the normal UI anyway (minClips
      // / referenceConflictsWithSourceVideo / controlLoraNeedsReference all
      // block a bare single-clip-plus-depth chain before this gate matters).
      act(() => {
        const id = result.current.clips[1]?.id;
        if (id) result.current.removeClip(id);
      });
      expect(result.current.validityReasons).toContain("depthChainUnsupported");
    });

    it("does not fire depthChainUnsupported when the server published no preprocess (deps omitted)", async () => {
      const { result } = setupReference({ prompt: "a cat <lora:depth-control:1.0> riding" });
      await attachReference(result);
      expect(result.current.validityReasons).not.toContain("depthChainUnsupported");
    });

    describe("the 128 resolution grid", () => {
      it("snaps the current size onto 128 the moment the reference is ready, and steps by 128 afterwards", async () => {
        const { result } = setupReference({ prompt: "a cat <lora:Pixar_Toon:1.0> riding" });
        // A 64-grid size that is NOT on the 128 grid.
        act(() => result.current.setWidth(1216));
        act(() => result.current.setHeight(704));
        expect(result.current.common.width).toBe(1216);
        expect(result.current.sizeMultiple).toBe(64);

        await attachReference(result);

        expect(result.current.sizeMultiple).toBe(128);
        expect(result.current.common.width % 128).toBe(0);
        expect(result.current.common.height % 128).toBe(0);
        expect(result.current.validityReasons).not.toContain("referenceDimensionsOffGrid");

        // A slider drag now lands on the 128 grid instead of falling back to 64.
        act(() => result.current.setWidth(1216));
        expect(result.current.common.width % 128).toBe(0);
      });

      it("reports referenceDimensionsOffGrid (not dimensionsOffGrid) for a free-typed off-grid size", async () => {
        const { result } = setupReference({ prompt: "a cat <lora:Pixar_Toon:1.0> riding" });
        await attachReference(result);
        act(() => result.current.setWidth(1216, false)); // on the 64 grid, off the 128 one
        expect(result.current.validityReasons).toContain("referenceDimensionsOffGrid");
        expect(result.current.validityReasons).not.toContain("dimensionsOffGrid");
      });

      it("returns to the 64 grid once the reference is cleared", async () => {
        const { result } = setupReference({ prompt: "a cat <lora:Pixar_Toon:1.0> riding" });
        await attachReference(result);
        expect(result.current.sizeMultiple).toBe(128);

        act(() => result.current.clearReference());
        expect(result.current.hasReferenceVideo).toBe(false);
        expect(result.current.sizeMultiple).toBe(64);
        act(() => result.current.setWidth(1216));
        expect(result.current.common.width).toBe(1216);
        expect(result.current.validityReasons).not.toContain("referenceDimensionsOffGrid");
      });

      it("a control-LoRA selection alone already forces the 128 grid (it mandates a reference)", () => {
        const { result } = setupReference();
        act(() => result.current.setControlLora({ name: "canny-control", strength: 1.0 }));
        expect(result.current.sizeMultiple).toBe(128);
        expect(result.current.isReferenceActive).toBe(true);
      });
    });

    it("probes the reference's pixel size exactly once, and drops it on clear", async () => {
      const { result } = setupReference({ probeWidth: 1280, probeHeight: 768 });
      expect(result.current.referenceMediaSize).toBeNull();
      await attachReference(result);
      await waitFor(() => expect(result.current.referenceMediaSize).toEqual({ width: 1280, height: 768 }));

      act(() => result.current.clearReference());
      await waitFor(() => expect(result.current.referenceMediaSize).toBeNull());
    });

    it("carries both strengths only alongside a sent reference, and keeps 0 distinct from unset", async () => {
      const { result } = setupReference({ prompt: "a cat <lora:Pixar_Toon:1.0> riding" });
      act(() => result.current.setConditioningAttentionStrength(0));
      act(() => result.current.setReferenceVideoStrength(0.5));
      // No reference attached yet -> no reference fields at all.
      expect(result.current.buildRequest()).not.toHaveProperty("conditioning_attention_strength");

      await attachReference(result);
      const request = result.current.buildRequest();
      expect(request.conditioning_attention_strength).toBe(0);
      expect(request.reference_video_strength).toBe(0.5);

      act(() => result.current.setConditioningAttentionStrength(null));
      expect(result.current.buildRequest()).not.toHaveProperty("conditioning_attention_strength");
    });

    it("clamps both strengths into [0,1]", () => {
      const { result } = setupReference();
      act(() => result.current.setConditioningAttentionStrength(5));
      act(() => result.current.setReferenceVideoStrength(-1));
      expect(result.current.conditioningAttentionStrength).toBe(1);
      expect(result.current.referenceVideoStrength).toBe(0);
    });

    it("isDirty flags an attached reference (attach INTENT) and a control-LoRA selection", async () => {
      const { result } = setupReference();
      expect(result.current.isDirty).toBe(false);
      act(() => result.current.setControlLora({ name: "canny-control", strength: 1.0 }));
      expect(result.current.isDirty).toBe(true);
      act(() => result.current.setControlLora(null));
      expect(result.current.isDirty).toBe(false);

      await attachReference(result);
      expect(result.current.isDirty).toBe(true);
    });

    it("pickReference ignores a CANCELLED dialog and records any other failure", async () => {
      const cancelled = createMockBridge({ delayMs: 0, failPickFile: "CANCELLED" });
      const { result } = renderHook(() => useChainForm(FALLBACK_APP_CONFIG, "x", { nativeBridge: cancelled }));
      await act(async () => {
        await result.current.pickReference();
      });
      expect(result.current.hasReferenceVideo).toBe(false);
      expect(result.current.referenceError).toBeNull();

      const failing = createMockBridge({ delayMs: 0, failPickFile: "DIALOG_FAILED" });
      const { result: errored } = renderHook(() => useChainForm(FALLBACK_APP_CONFIG, "x", { nativeBridge: failing }));
      await act(async () => {
        await errored.current.pickReference();
      });
      expect(errored.current.referenceError).toBe("DIALOG_FAILED");
    });
  });

  // ── 素材（末尾）: the end slot ─────────────────────────────────────────────
  // One user-facing slot backed by two sub-slots (image/video), routed by
  // extension exactly like the START slot — plus the two things that are
  // unique to this end under 窓内モード (2026-08-17): the anchor is the constant
  // 8 whatever the material is, and the output does NOT get longer. 逆順Chained
  // (2026-08-18, second stage) lifted the third — the chain no longer has to be
  // a single clip — so most of this block covers BOTH clip counts; the tests
  // that are specifically about the 2+-clip `reverse` mode say so in their name.
  describe("素材（末尾）end source (窓内モード + 逆順Chained)", () => {
    /** A config whose default clip length (49) is short — kept from v1 so the
     * "the clip list is NOT touched any more" assertions have something that
     * WOULD have moved under v1's push-up (49 -> 105). It is also comfortably
     * inside the 169-frame quality ceiling, which keeps the warning out of the
     * way of tests that are not about it. */
    const SHORT_CLIP_CONFIG: AppConfig = {
      ...FALLBACK_APP_CONFIG,
      generation_defaults: { ...FALLBACK_APP_CONFIG.generation_defaults, num_frames: 49 },
    };

    /** A config whose default clip length is pinned at 257 — the value
     * `FALLBACK_APP_CONFIG.generation_defaults.num_frames` held before the
     * 2026-08-19 preset raise (to 361, the SMART comfort ceiling — see
     * `spillUtils.singleComfortFrames`). Several worked examples below (497,
     * 514, 424, 705 output/total frames; the 30fps/kv=1 overlap-budget
     * geometry) are calibrated specifically against 257f clips and are
     * otherwise unrelated to the live preset value, so they pin it here
     * instead of drifting with `FALLBACK_APP_CONFIG`. */
    const STANDARD_CLIP_CONFIG: AppConfig = {
      ...FALLBACK_APP_CONFIG,
      generation_defaults: { ...FALLBACK_APP_CONFIG.generation_defaults, num_frames: 257 },
    };

    function setupEnd(config: AppConfig = FALLBACK_APP_CONFIG, options: MockBridgeOptions = {}) {
      const mockBridge = createMockBridge({ delayMs: 0, ...options });
      const { result } = renderHook(() =>
        useChainForm(config, "a cat riding a skateboard", { nativeBridge: mockBridge }),
      );
      return { result, mockBridge };
    }

    async function attachEndVideo(
      result: { current: ReturnType<typeof useChainForm> },
      fileName = "tail.mp4",
    ) {
      await act(async () => {
        await result.current.attachEndSourceByPath(`C:\\v\\${fileName}`, fileName);
      });
      await waitFor(() => expect(result.current.endSourceStatus).toBe("ready"));
    }

    async function attachEndImage(
      result: { current: ReturnType<typeof useChainForm> },
      fileName = "tail.png",
    ) {
      await act(async () => {
        await result.current.attachEndSourceByPath(`C:\\i\\${fileName}`, fileName);
      });
      await waitFor(() => expect(result.current.endSourceStatus).toBe("ready"));
    }

    /** Drops the chain to the single clip 窓内モード requires, and sets that
     * clip's length. Most assertions below are about a chain the user has
     * already made legal, so this is the shared "arrange" step. */
    function makeSingleClip(result: { current: ReturnType<typeof useChainForm> }, numFrames = 169) {
      act(() => {
        const id = result.current.clips[1]?.id;
        if (id) result.current.removeClip(id);
      });
      act(() => {
        const id = result.current.clips[0]?.id;
        if (id) result.current.setClipNumFrames(id, numFrames);
      });
    }

    it("opens empty, with NO anchor at all, changing nothing about the request", () => {
      const { result } = setupEnd();
      expect(result.current.hasEndSource).toBe(false);
      expect(result.current.endSourceKind).toBeNull();
      expect(result.current.endSourceStatus).toBe("idle");
      // The anchor only exists once material does.
      expect(result.current.endContextFrames).toBeNull();
      expect(result.current.endSourceLengthIssue).toBeNull();
      expect(result.current.endSourceQualityLimitFrames).toBeNull();
      expect(result.current.outputTailFrames).toBe(0);
      expect(result.current.buildRequest()).not.toHaveProperty("end_source");
      expect(result.current.isDirty).toBe(false);
    });

    describe("routing by extension", () => {
      it("a video goes to the video half and takes the fixed 8-frame anchor", async () => {
        // The mock measures 300 frames @ 24fps — far past the 9-frame floor, and
        // the length changes nothing about the anchor.
        const { result } = setupEnd();
        await attachEndVideo(result);
        expect(result.current.endSourceKind).toBe("video");
        expect(result.current.endSourceVideo.state.fileName).toBe("tail.mp4");
        expect(result.current.endContextFrames).toBe(8);
      });

      it("an image goes to the image half and takes the SAME 8", async () => {
        const { result } = setupEnd();
        await attachEndImage(result);
        expect(result.current.endSourceKind).toBe("image");
        expect(result.current.endSourceImage.items).toHaveLength(1);
        expect(result.current.endContextFrames).toBe(8);
      });

      it("switching image -> video keeps the anchor at 8 and empties the image half", async () => {
        const { result } = setupEnd();
        await attachEndImage(result);
        expect(result.current.endContextFrames).toBe(8);

        await attachEndVideo(result);
        expect(result.current.endSourceKind).toBe("video");
        expect(result.current.endContextFrames).toBe(8);
        // Replaced, not accumulated: the image half is empty again.
        expect(result.current.endSourceImage.items).toHaveLength(0);
      });

      it("switching video -> image drops the video", async () => {
        const { result } = setupEnd();
        await attachEndVideo(result);
        await attachEndImage(result);
        expect(result.current.endSourceKind).toBe("image");
        expect(result.current.endSourceVideo.state.status).toBe("idle");
        expect(result.current.endContextFrames).toBe(8);
      });

      it("re-attaching a video REPLACES the previous one (single slot)", async () => {
        const { result } = setupEnd();
        await attachEndVideo(result, "first.mp4");
        await attachEndVideo(result, "second.mp4");
        expect(result.current.endSourceVideo.state.fileName).toBe("second.mp4");
        expect(result.current.endSourceImage.items).toHaveLength(0);
      });

      it("an unrecognised extension touches neither half and reports UNSUPPORTED_FILE_TYPE", async () => {
        const { result } = setupEnd();
        await act(async () => {
          await result.current.attachEndSourceByPath("C:\\x\\notes.txt", "notes.txt");
        });
        expect(result.current.endSourceError).toBe("UNSUPPORTED_FILE_TYPE");
        expect(result.current.hasEndSource).toBe(false);
      });

      it("clearEndSource empties whichever half held the material, and the anchor with it", async () => {
        const { result } = setupEnd();
        await attachEndVideo(result);
        act(() => result.current.clearEndSource());
        expect(result.current.hasEndSource).toBe(false);
        expect(result.current.endSourceKind).toBeNull();
        expect(result.current.endContextFrames).toBeNull();

        await attachEndImage(result);
        act(() => result.current.clearEndSource());
        expect(result.current.hasEndSource).toBe(false);
        expect(result.current.endSourceImage.items).toHaveLength(0);
      });

      it("pickEndSource ignores a CANCELLED dialog and records any other failure", async () => {
        const cancelled = createMockBridge({ delayMs: 0, failPickFile: "CANCELLED" });
        const { result } = renderHook(() => useChainForm(FALLBACK_APP_CONFIG, "x", { nativeBridge: cancelled }));
        await act(async () => {
          await result.current.pickEndSource();
        });
        expect(result.current.hasEndSource).toBe(false);
        expect(result.current.endSourceError).toBeNull();

        const failing = createMockBridge({ delayMs: 0, failPickFile: "DIALOG_FAILED" });
        const { result: errored } = renderHook(() => useChainForm(FALLBACK_APP_CONFIG, "x", { nativeBridge: failing }));
        await act(async () => {
          await errored.current.pickEndSource();
        });
        expect(errored.current.endSourceError).toBe("DIALOG_FAILED");
      });
    });

    // The upload cap. Load-bearing twice over — it bounds what lands in
    // `uploads/`, AND it is what makes the server measure the file at all.
    describe("the upload query", () => {
      it("always carries max_frames=685", async () => {
        const { result, mockBridge } = setupEnd();
        const spy = vi.spyOn(mockBridge, "request");
        await attachEndVideo(result);
        expect(spy).toHaveBeenCalledWith("backend.uploadFile", {
          kind: "video",
          filePath: "C:\\v\\tail.mp4",
          query: { max_frames: "685" },
        });
      });

      it("merges a ribbon trim with the cap and clamps the window to 685 / gen fps", async () => {
        const { result, mockBridge } = setupEnd();
        const spy = vi.spyOn(mockBridge, "request");
        await act(async () => {
          await result.current.attachEndSourceByPath("C:\\v\\tail.mp4", "tail.mp4", {
            // 600s at 24fps is 14400 frames — far past the cap, which the server
            // would otherwise ignore because a trim window is present.
            trim: { trim: true, startSec: 2, durationSec: 600 },
            knownDurationSec: 900,
          });
        });
        await waitFor(() => expect(result.current.endSourceStatus).toBe("ready"));
        expect(spy).toHaveBeenCalledWith("backend.uploadFile", {
          kind: "video",
          filePath: "C:\\v\\tail.mp4",
          query: {
            max_frames: "685",
            trim_start_sec: "2.000",
            // 685 / 24 = 28.5416666…, serialized at millisecond resolution.
            trim_duration_sec: (685 / 24).toFixed(3),
          },
        });
      });

      it("leaves a SHORT trim window alone (the clamp is a ceiling, not a rewrite)", async () => {
        const { result, mockBridge } = setupEnd();
        const spy = vi.spyOn(mockBridge, "request");
        await act(async () => {
          await result.current.attachEndSourceByPath("C:\\v\\tail.mp4", "tail.mp4", {
            trim: { trim: true, startSec: 1, durationSec: 5 },
            knownDurationSec: 900,
          });
        });
        await waitFor(() => expect(result.current.endSourceStatus).toBe("ready"));
        expect(spy).toHaveBeenCalledWith("backend.uploadFile", {
          kind: "video",
          filePath: "C:\\v\\tail.mp4",
          query: { max_frames: "685", trim_start_sec: "1.000", trim_duration_sec: "5.000" },
        });
      });
    });

    // 窓内モード: the anchor is a CONSTANT. All the measurement still does is
    // decide whether the material can supply it at all.
    describe("the anchor and the 9-frame floor", () => {
      it("is 8 for a video of any length — a long one is not treated differently", async () => {
        const long = setupEnd(FALLBACK_APP_CONFIG, { uploadVideoFrameCount: 1000, uploadVideoFps: 24 });
        await attachEndVideo(long.result);
        expect(long.result.current.endContextFrames).toBe(8);

        const short = setupEnd(FALLBACK_APP_CONFIG, { uploadVideoFrameCount: 73, uploadVideoFps: 24 });
        await attachEndVideo(short.result);
        expect(short.result.current.endContextFrames).toBe(8);
        expect(short.result.current.endSourceLengthIssue).toBeNull();
      });

      it("accepts the 9-frame floor exactly", async () => {
        const { result } = setupEnd(FALLBACK_APP_CONFIG, { uploadVideoFrameCount: 9, uploadVideoFps: 24 });
        await attachEndVideo(result);
        expect(result.current.endContextFrames).toBe(8);
        expect(result.current.endSourceLengthIssue).toBeNull();
        expect(result.current.validityReasons).not.toContain("endSourceTooShort");
      });

      it("blocks Generate one frame below the floor", async () => {
        const { result } = setupEnd(FALLBACK_APP_CONFIG, { uploadVideoFrameCount: 8, uploadVideoFps: 24 });
        await attachEndVideo(result);
        expect(result.current.endContextFrames).toBeNull();
        expect(result.current.endSourceLengthIssue).toBe("tooShort");
        expect(result.current.validityReasons).toContain("endSourceTooShort");
        expect(result.current.isValid).toBe(false);
        // …and the request drops the slot rather than sending an anchor the
        // material cannot supply.
        expect(result.current.buildRequest()).not.toHaveProperty("end_source");
      });

      it("converts a mismatched source fps CONSERVATIVELY (floor − 1)", async () => {
        // 12 frames @ 30fps, generating at 24 -> floor(12*24/30) - 1 = 8, one
        // below the floor even though the raw count is above it.
        const { result } = setupEnd(FALLBACK_APP_CONFIG, { uploadVideoFrameCount: 12, uploadVideoFps: 30 });
        await attachEndVideo(result);
        expect(result.current.endSourceLengthIssue).toBe("tooShort");

        // Matching the rates makes the conversion the identity, and 12 clears 9.
        act(() => result.current.setFrameRate(30));
        expect(result.current.endSourceLengthIssue).toBeNull();
        expect(result.current.endContextFrames).toBe(8);
      });

      it("falls back to the attach's own measured duration when the server measured nothing", async () => {
        const { result } = setupEnd(FALLBACK_APP_CONFIG, { uploadVideoFrameCount: null, uploadVideoFps: null });
        await act(async () => {
          await result.current.attachEndSourceByPath("C:\\v\\tail.mp4", "tail.mp4", {
            trim: { trim: false, reason: "spanCoversWholeMedia" },
            // 5s at 24fps -> floor(120) - 1 = 119, comfortably past the floor.
            knownDurationSec: 5,
          });
        });
        await waitFor(() => expect(result.current.endSourceStatus).toBe("ready"));
        expect(result.current.endContextFrames).toBe(8);
        expect(result.current.validityReasons).not.toContain("endSourceLengthUnknown");
      });

      it("blocks with endSourceLengthUnknown when NOTHING could be measured", async () => {
        const { result } = setupEnd(FALLBACK_APP_CONFIG, { uploadVideoFrameCount: null, uploadVideoFps: null });
        await attachEndVideo(result); // a manual drop: no duration either
        expect(result.current.endContextFrames).toBeNull();
        expect(result.current.endSourceLengthIssue).toBe("unknown");
        expect(result.current.validityReasons).toContain("endSourceLengthUnknown");
        expect(result.current.isValid).toBe(false);
      });

      it("falls back to the duration when the frame count arrives WITHOUT its fps", async () => {
        // A count with no rate is not a length — see `resolveEndSourceContext`.
        const { result } = setupEnd(FALLBACK_APP_CONFIG, { uploadVideoFrameCount: 300, uploadVideoFps: null });
        await act(async () => {
          await result.current.attachEndSourceByPath("C:\\v\\tail.mp4", "tail.mp4", {
            trim: { trim: false, reason: "spanCoversWholeMedia" },
            knownDurationSec: 5,
          });
        });
        await waitFor(() => expect(result.current.endSourceStatus).toBe("ready"));
        expect(result.current.endContextFrames).toBe(8);
      });

      it("never applies the floor to an IMAGE — a still supplies the anchor whatever it is", async () => {
        const { result } = setupEnd(FALLBACK_APP_CONFIG, { uploadVideoFrameCount: null, uploadVideoFps: null });
        await attachEndImage(result);
        expect(result.current.endSourceLengthIssue).toBeNull();
        expect(result.current.endContextFrames).toBe(8);
      });

      it("stays silent about every length rule while nothing is attached", () => {
        const { result } = setupEnd();
        expect(result.current.validityReasons).not.toContain("endSourceTooShort");
        expect(result.current.validityReasons).not.toContain("endSourceLengthUnknown");
        expect(result.current.validityReasons).not.toContain("endSourceNeedsOverlap");
        expect(result.current.isValid).toBe(true);
      });
    });

    // 逆順Chained (2026-08-18, second stage): `endSourceNeedsOverlap` is now
    // SINGLE-CLIP ONLY — the server exempts `reverse` mode (2+ clips) from the
    // `kv >= 2` floor entirely (`endSourceAudioOverlapBudget` below is what
    // replaces it there).
    describe("endSourceNeedsOverlap (kv=1, single clip)", () => {
      it("blocks at seam-blend width 1 and clears at 2", async () => {
        const { result } = setupEnd();
        await attachEndVideo(result);
        makeSingleClip(result);
        expect(result.current.validityReasons).not.toContain("endSourceNeedsOverlap");

        act(() => result.current.setOverlapFrames(1));
        expect(result.current.validityReasons).toContain("endSourceNeedsOverlap");
        expect(result.current.isValid).toBe(false);

        act(() => result.current.setOverlapFrames(2));
        expect(result.current.validityReasons).not.toContain("endSourceNeedsOverlap");
      });

      it("never fires without an end source (kv=1 is a perfectly ordinary chain)", () => {
        const { result } = setupEnd();
        act(() => result.current.setOverlapFrames(1));
        expect(result.current.validityReasons).not.toContain("endSourceNeedsOverlap");
        expect(result.current.isValid).toBe(true);
      });

      it("never fires on a 2+-clip chain even at kv=1 — the server's own reverse-mode exemption", async () => {
        const { result } = setupEnd();
        await attachEndVideo(result); // stays at the default 2 clips
        // 2.6(e)'s transition effect already nudged this to 1 on attach.
        expect(result.current.overlapFrames).toBe(1);
        expect(result.current.validityReasons).not.toContain("endSourceNeedsOverlap");
        expect(result.current.isValid).toBe(true);
      });
    });

    // 逆順Chained (2026-08-18, second stage): the former single-clip ceiling
    // (`endSourceSingleClipOnly`) is GONE — 2+ clips is the `reverse` mode the
    // server now accepts. `canAddClip` no longer closes for an end source at
    // any clip count, and a multi-clip chain simply validates.
    describe("multi-clip end source (reverse mode, 2.6(a))", () => {
      it("does not close the add-clip button while material is attached", async () => {
        const { result } = setupEnd(SHORT_CLIP_CONFIG);
        expect(result.current.canAddClip).toBe(true);
        expect(result.current.maxClips).toBe(24);

        await attachEndVideo(result);
        expect(result.current.canAddClip).toBe(true);
        expect(result.current.clips).toHaveLength(2);

        act(() => result.current.addClip());
        expect(result.current.clips).toHaveLength(3);

        act(() => result.current.clearEndSource());
        expect(result.current.canAddClip).toBe(true);
      });

      it("a default 2-clip [257,257] chain at 24fps validates end-to-end", async () => {
        const { result } = setupEnd();
        await attachEndVideo(result);
        expect(result.current.clips).toHaveLength(2);
        expect(result.current.isValid).toBe(true);
        const request = result.current.buildRequest();
        expect(request.end_source).toBeTruthy();
        expect(request.clips).toHaveLength(2);
      });
    });

    // のりしろ既定値1 (2.6(e)): the SYMMETRIC one-shot nudge on the "素材（末尾）
    // x 2+ clips" transition, and its undo.
    describe("the overlapFrames transition effect", () => {
      it("nudges to 1 on entering 素材（末尾）x 2+ clips, and restores the default on the clear-edge", async () => {
        const { result } = setupEnd();
        expect(result.current.overlapFrames).toBe(DEFAULT_OVERLAP_FRAMES);

        await attachEndVideo(result); // stays at the default 2 clips
        expect(result.current.overlapFrames).toBe(1);

        act(() => result.current.clearEndSource());
        expect(result.current.overlapFrames).toBe(DEFAULT_OVERLAP_FRAMES);
      });

      it("also restores on the OTHER exit edge — dropping back to 1 clip", async () => {
        const { result } = setupEnd();
        await attachEndVideo(result);
        expect(result.current.overlapFrames).toBe(1);

        makeSingleClip(result);
        expect(result.current.overlapFrames).toBe(DEFAULT_OVERLAP_FRAMES);
      });

      it("does not fight a value the user dials in mid-state", async () => {
        const { result } = setupEnd();
        await attachEndVideo(result);
        expect(result.current.overlapFrames).toBe(1);

        act(() => result.current.setOverlapFrames(5));
        expect(result.current.overlapFrames).toBe(5);

        // A re-render that does not cross the reverse-mode boundary must not
        // push the value back down.
        act(() => result.current.setClipPrompt(result.current.clips[0]!.id, "x"));
        expect(result.current.overlapFrames).toBe(5);
      });

      it("never fires when material is attached to a chain that is already single-clip", async () => {
        const { result } = setupEnd(SHORT_CLIP_CONFIG);
        makeSingleClip(result); // reduce to 1 clip BEFORE attaching
        await attachEndVideo(result);
        expect(result.current.overlapFrames).toBe(DEFAULT_OVERLAP_FRAMES);
      });
    });

    // 逆順Chained (2026-08-18, second stage), 2.6(f): the audio-overlap-budget
    // mirror (`chainUtils.endSourceAudioOverlapOk`) — the gate that replaces
    // `endSourceNeedsOverlap` once there are 2+ clips.
    describe("endSourceAudioOverlapBudget (reverse mode's own audio-overlap mirror)", () => {
      it("blocks the plan's own headline failure — 30fps x 257f x 2 clips at kv=1 — and clears at kv=3", async () => {
        const { result } = setupEnd(STANDARD_CLIP_CONFIG);
        act(() => result.current.setFrameRate(30));
        await attachEndVideo(result);
        // 2.6(e)'s transition effect already nudged this to 1 on attach — the
        // exact failing width for this geometry (chainUtils.test.ts's own
        // worked example).
        expect(result.current.overlapFrames).toBe(1);
        expect(result.current.validityReasons).toContain("endSourceAudioOverlapBudget");
        expect(result.current.isValid).toBe(false);

        act(() => result.current.setOverlapFrames(3));
        expect(result.current.validityReasons).not.toContain("endSourceAudioOverlapBudget");
        expect(result.current.isValid).toBe(true);
      });

      it("never fires on a single-clip chain — that geometry has no join at all", async () => {
        const { result } = setupEnd();
        act(() => result.current.setFrameRate(30));
        await attachEndVideo(result);
        makeSingleClip(result);
        expect(result.current.validityReasons).not.toContain("endSourceAudioOverlapBudget");
      });

      it("never fires without an end source", () => {
        const { result } = setupEnd();
        act(() => result.current.setFrameRate(30));
        expect(result.current.validityReasons).not.toContain("endSourceAudioOverlapBudget");
      });
    });

    // 逆順Chained (2026-08-18, second stage), 2.6(f): the last-clip free-latent
    // mirror (`chainUtils.endSourceLastClipHasFreeLatents`) — reuses the
    // EXISTING `clipTooShortForOverlap` reason code (plan's own instruction:
    // don't grow the surface for a second, closely-related failure).
    describe("the last-clip free-latent mirror (reuses clipTooShortForOverlap)", () => {
      it("blocks exactly at the boundary and clears one latent frame above it", async () => {
        const { result } = setupEnd(SHORT_CLIP_CONFIG); // [49, 49]
        await attachEndVideo(result);
        act(() => result.current.setOverlapFrames(3));
        // v_latent(25) = 4; kv(3) + nEndV(1) = 4 -> exactly fills the last
        // clip, nothing left to carry backwards. clipsMeetOverlapMinimum
        // itself stays satisfied (25 >= 8*3+1 = 25), isolating this check.
        act(() => {
          const id = result.current.clips[1]?.id;
          if (id) result.current.setClipNumFrames(id, 25);
        });
        expect(result.current.clips.map((c) => c.numFrames)).toEqual([49, 25]);
        expect(result.current.validityReasons).toContain("clipTooShortForOverlap");
        expect(result.current.isValid).toBe(false);

        // v_latent(33) = 5; 4 < 5 -> one free latent remains.
        act(() => {
          const id = result.current.clips[1]?.id;
          if (id) result.current.setClipNumFrames(id, 33);
        });
        expect(result.current.validityReasons).not.toContain("clipTooShortForOverlap");
        expect(result.current.isValid).toBe(true);
      });

      it("never fires on a single-clip chain (窓内モード never appends to the last clip's head)", async () => {
        const { result } = setupEnd(SHORT_CLIP_CONFIG);
        await attachEndVideo(result);
        makeSingleClip(result, 25); // well below the boundary above, on 1 clip
        expect(result.current.validityReasons).not.toContain("clipTooShortForOverlap");
      });

      it("never fires while the material can't supply the anchor (no end_source would be sent)", async () => {
        const { result } = setupEnd(SHORT_CLIP_CONFIG, { uploadVideoFrameCount: 8, uploadVideoFps: 24 });
        await attachEndVideo(result);
        act(() => result.current.setOverlapFrames(3));
        act(() => {
          const id = result.current.clips[1]?.id;
          if (id) result.current.setClipNumFrames(id, 25);
        });
        expect(result.current.endContextFrames).toBeNull();
        expect(result.current.validityReasons).not.toContain("clipTooShortForOverlap");
      });
    });

    // 窓内モード品質警告: a clip longer than ONE stage-2 window puts the frozen
    // tail and the frames blending into it in different tiles. Advisory only.
    describe("the clip-length quality warning", () => {
      it("fires at 170 frames on the standard window and clears at 169", async () => {
        const { result } = setupEnd();
        await attachEndVideo(result);
        makeSingleClip(result, 177); // the next 8n+1 step above 169
        expect(result.current.endSourceQualityLimitFrames).toBe(169);
        // Advisory, never a block.
        expect(result.current.isValid).toBe(true);

        act(() => {
          const id = result.current.clips[0]?.id;
          if (id) result.current.setClipNumFrames(id, 169);
        });
        expect(result.current.endSourceQualityLimitFrames).toBeNull();
      });

      it("follows the stage-2 window down to 145 on high_resolution", async () => {
        const { result } = setupEnd();
        await attachEndVideo(result);
        makeSingleClip(result, 169);
        expect(result.current.endSourceQualityLimitFrames).toBeNull();

        act(() => result.current.setStage2Window("high_resolution"));
        expect(result.current.endSourceQualityLimitFrames).toBe(145);

        act(() => {
          const id = result.current.clips[0]?.id;
          if (id) result.current.setClipNumFrames(id, 145);
        });
        expect(result.current.endSourceQualityLimitFrames).toBeNull();
      });

      it("stays silent with no end source, however long the clips are", () => {
        const { result } = setupEnd(STANDARD_CLIP_CONFIG);
        // The default chain is [257, 257] — both well past 169.
        expect(result.current.clips.map((c) => c.numFrames)).toEqual([257, 257]);
        expect(result.current.endSourceQualityLimitFrames).toBeNull();
      });

      // 逆順Chained (2026-08-18, second stage): the ceiling is single-clip
      // only — §62's reconnaissance study behind it never ran a multi-clip
      // reverse chain, so it stays silent rather than mis-warning.
      it("stays silent on a 2+-clip end-source chain, however long the clips are", async () => {
        const { result } = setupEnd();
        await attachEndVideo(result); // stays at the default [257, 257]
        expect(result.current.clips).toHaveLength(2);
        expect(result.current.endSourceQualityLimitFrames).toBeNull();
      });
    });

    // §1-22 素材（末尾）複数クリップ品質警告: the exact complement of the
    // single-clip ceiling above — fires on 2+ clips regardless of length,
    // silent on exactly 1 clip. Advisory only, mirroring the ceiling's own
    // "never a validityReason" treatment.
    describe("the multi-clip quality warning", () => {
      it("fires once a 2nd clip exists alongside an attached end source", async () => {
        const { result } = setupEnd();
        await attachEndVideo(result); // default chain is [257, 257] — 2 clips
        expect(result.current.clips).toHaveLength(2);
        expect(result.current.endSourceMultiClipQualityWarning).toBe(true);
        // Advisory, never a block.
        expect(result.current.isValid).toBe(true);
      });

      it("stays silent on a single-clip 窓内モード chain", async () => {
        const { result } = setupEnd();
        await attachEndVideo(result);
        makeSingleClip(result, 169);
        expect(result.current.endSourceMultiClipQualityWarning).toBe(false);
      });

      it("stays silent with no end source, however many clips exist", () => {
        const { result } = setupEnd();
        expect(result.current.clips).toHaveLength(2);
        expect(result.current.endSourceMultiClipQualityWarning).toBe(false);
      });
    });

    // The v1 push-up is gone: the anchor no longer competes with the clips.
    describe("the clip list is left alone", () => {
      it("does not touch the clip lengths on attach", async () => {
        const { result } = setupEnd(SHORT_CLIP_CONFIG);
        expect(result.current.clips.map((c) => c.numFrames)).toEqual([49, 49]);

        await attachEndVideo(result);
        expect(result.current.endContextFrames).toBe(8);
        // v1 would have pushed the last clip to 105 here.
        expect(result.current.clips.map((c) => c.numFrames)).toEqual([49, 49]);
      });

      it("a single SHORT clip plus material is simply valid", async () => {
        const { result } = setupEnd(SHORT_CLIP_CONFIG);
        await attachEndVideo(result);
        act(() => {
          const id = result.current.clips[1]?.id;
          if (id) result.current.removeClip(id);
        });
        // v1 rejected exactly this with `endContextFramesTooLongForClip`.
        expect(result.current.validityReasons).toEqual([]);
        expect(result.current.isValid).toBe(true);
      });

      it("re-fits back to the 2-clip floor when the end source is removed", async () => {
        const { result } = setupEnd(SHORT_CLIP_CONFIG);
        await attachEndVideo(result);
        // The floor is relaxed to 1 while material is attached, so the user may
        // drop the second card.
        expect(result.current.minClips).toBe(1);
        act(() => {
          const id = result.current.clips[0]?.id;
          if (id) result.current.removeClip(id);
        });
        expect(result.current.clips).toHaveLength(1);

        act(() => result.current.clearEndSource());
        expect(result.current.minClips).toBe(2);
        expect(result.current.clips).toHaveLength(2);
      });

      it("removing the end source from a V2V chain does NOT pad a stray clip back in", async () => {
        const { result } = setupEnd(SHORT_CLIP_CONFIG);
        act(() => {
          void result.current.sourceVideo.pick();
        });
        await waitFor(() => expect(result.current.sourceVideo.state.status).toBe("ready"));
        await attachEndVideo(result);
        act(() => {
          const id = result.current.clips[0]?.id;
          if (id) result.current.removeClip(id);
        });
        expect(result.current.clips).toHaveLength(1);

        act(() => result.current.clearEndSource());
        // The source video still holds the floor at 1 — no re-padding.
        expect(result.current.minClips).toBe(1);
        expect(result.current.clips).toHaveLength(1);
      });

      it("removing the SOURCE VIDEO does not pad a stray clip back into an end-source chain", async () => {
        // The mirror of the case above, and the one that would silently pad a
        // stray clip back in if the source slot's own re-fit used the
        // hard-coded from-scratch floor instead of `hasEndSource`'s own.
        const { result } = setupEnd(SHORT_CLIP_CONFIG);
        act(() => {
          void result.current.sourceVideo.pick();
        });
        await waitFor(() => expect(result.current.sourceVideo.state.status).toBe("ready"));
        await attachEndVideo(result);
        act(() => {
          const id = result.current.clips[0]?.id;
          if (id) result.current.removeClip(id);
        });
        expect(result.current.clips).toHaveLength(1);

        act(() => result.current.clearSource());
        expect(result.current.clips).toHaveLength(1);
      });
    });

    describe("the clip-count floor", () => {
      it("a single clip becomes legal with an end source attached (owner decision E)", async () => {
        const { result } = setupEnd();
        expect(result.current.minClips).toBe(2);
        await attachEndVideo(result);
        expect(result.current.minClips).toBe(1);

        act(() => {
          const id = result.current.clips[0]?.id;
          if (id) result.current.removeClip(id);
        });
        expect(result.current.clips).toHaveLength(1);
        expect(result.current.isClipCountValid).toBe(true);
        expect(result.current.validityReasons).not.toContain("minClips");
      });
    });

    // 出力長はクリップ尺そのもの — the single most visible 窓内モード spec change.
    describe("the output length", () => {
      it("does NOT grow when material is attached; the tail is a SLICE of it", async () => {
        const { result } = setupEnd(STANDARD_CLIP_CONFIG);
        // [257, 257] at kv=3 -> 497 output frames.
        expect(result.current.outputFrames).toBe(497);
        expect(result.current.outputTailFrames).toBe(0);

        await attachEndVideo(result);
        // 逆順Chained (2026-08-18): attaching an end source on a 2+-clip chain
        // also nudges the seam-blend width to 1 (2.6(e), separately tested) —
        // pin it back to isolate "does material attachment alone grow the
        // output" from that unrelated, already-covered nudge.
        act(() => result.current.setOverlapFrames(DEFAULT_OVERLAP_FRAMES));

        expect(result.current.outputFrames).toBe(497);
        expect(result.current.outputTailFrames).toBe(8);
        expect(result.current.outputSeconds).toBeCloseTo(497 / 24, 6);
        expect(result.current.outputTailSeconds).toBeCloseTo(8 / 24, 6);
        // The tail is INSIDE the output, so the breakdown never exceeds it.
        expect(result.current.outputTailFrames).toBeLessThan(result.current.outputFrames);
      });

      it("subtracts the V2V head exactly as it always did", async () => {
        const { result } = setupEnd(STANDARD_CLIP_CONFIG);
        act(() => {
          void result.current.sourceVideo.pick();
        });
        await waitFor(() => expect(result.current.sourceVideo.state.status).toBe("ready"));
        const v2vOutput = result.current.outputFrames;
        // V2V replaces its leading `contextFrames` with the source footage.
        expect(v2vOutput).toBe(497 - result.current.contextFrames);

        await attachEndVideo(result);
        // Attaching end material changes nothing about that arithmetic — pin
        // the 2.6(e) seam-blend nudge back out first (see the test above).
        act(() => result.current.setOverlapFrames(DEFAULT_OVERLAP_FRAMES));
        expect(result.current.outputFrames).toBe(v2vOutput);
        expect(result.current.outputTailFrames).toBe(8);
      });

      it("leaves totalFrames (the CLIP sum) unchanged, so isTotalFramesValid is unchanged", async () => {
        const { result } = setupEnd(STANDARD_CLIP_CONFIG);
        const before = result.current.totalFrames;
        await attachEndVideo(result);
        expect(result.current.totalFrames).toBe(before);
        expect(result.current.totalFrames).toBe(514);
        expect(result.current.isTotalFramesValid).toBe(true);
      });

      it("does not change the time estimate — the anchor is not extra work", async () => {
        const { result } = setupEnd();
        const before = result.current.estimateSeconds;
        await attachEndVideo(result);
        expect(result.current.estimateSeconds).toBe(before);
      });

      it("reports no tail while the material cannot supply the anchor", async () => {
        const { result } = setupEnd(STANDARD_CLIP_CONFIG, { uploadVideoFrameCount: 8, uploadVideoFps: 24 });
        await attachEndVideo(result);
        // The 2.6(e) nudge still fires (it keys on ATTACH intent, not on the
        // material actually being usable) — pin it back for the same reason
        // as the tests above.
        act(() => result.current.setOverlapFrames(DEFAULT_OVERLAP_FRAMES));
        expect(result.current.outputTailFrames).toBe(0);
        expect(result.current.outputFrames).toBe(497);
      });
    });

    describe("exclusivity", () => {
      it("blocks (and drops) an end source attached alongside an audio track", async () => {
        const { result } = setupEnd();
        await attachEndVideo(result);
        await act(async () => {
          await result.current.attachAudioByPath("C:\\a\\track.wav", "track.wav", { knownDurationSec: 30 });
        });
        await waitFor(() => expect(result.current.sourceAudio.state.status).toBe("ready"));

        expect(result.current.validityReasons).toContain("endSourceConflictsWithAudio");
        expect(result.current.isValid).toBe(false);
        // Defensive second line: the request drops the END source, never the audio.
        const request = result.current.buildRequest();
        expect(request).not.toHaveProperty("end_source");
        expect(request.source_audio).toBeTruthy();
      });

      it("blocks (and drops) an end source attached alongside a reference video", async () => {
        const { result } = setupEnd();
        await attachEndVideo(result);
        await act(async () => {
          await result.current.attachReferenceByPath("C:\\r\\ref.mp4", "ref.mp4");
        });
        await waitFor(() => expect(result.current.referenceVideo.state.status).toBe("ready"));

        expect(result.current.validityReasons).toContain("endSourceConflictsWithReference");
        expect(result.current.buildRequest()).not.toHaveProperty("end_source");
      });

      it("is NOT exclusive with a V2V source video on a SINGLE clip (start + end is the interpolation case)", async () => {
        const { result } = setupEnd();
        act(() => {
          void result.current.sourceVideo.pick();
        });
        await waitFor(() => expect(result.current.sourceVideo.state.status).toBe("ready"));
        await attachEndVideo(result);
        makeSingleClip(result);

        expect(result.current.validityReasons).not.toContain("endSourceConflictsWithAudio");
        expect(result.current.validityReasons).not.toContain("endSourceConflictsWithReference");
        expect(result.current.validityReasons).not.toContain("endSourceWithSourceVideoMultiClip");
        expect(result.current.isValid).toBe(true);
        const request = result.current.buildRequest();
        expect(request.source_video).toBeTruthy();
        expect(request.end_source).toBeTruthy();
      });

      // 逆順Chained (2026-08-18, second stage): mirrors `chain_math.py:1097-1105`
      // — a V2V source video and an end source on 2+ clips is a combination
      // nothing has ever run (clip 0 would be frozen at both ends), refused
      // outright rather than accepted untested.
      it("blocks (and drops) an end source attached alongside a V2V source video once the chain grows past ONE clip", async () => {
        const { result } = setupEnd();
        act(() => {
          void result.current.sourceVideo.pick();
        });
        await waitFor(() => expect(result.current.sourceVideo.state.status).toBe("ready"));
        await attachEndVideo(result);
        // The default chain is already 2 clips — nothing more to do.
        expect(result.current.clips).toHaveLength(2);

        expect(result.current.validityReasons).toContain("endSourceWithSourceVideoMultiClip");
        expect(result.current.isValid).toBe(false);
        const request = result.current.buildRequest();
        expect(request).not.toHaveProperty("end_source");
        expect(request.source_video).toBeTruthy();

        // Dropping to 1 clip clears it — the same interpolation case as the
        // test above.
        makeSingleClip(result);
        expect(result.current.validityReasons).not.toContain("endSourceWithSourceVideoMultiClip");
      });

      it("never fires without BOTH slots filled", () => {
        const { result } = setupEnd();
        expect(result.current.validityReasons).not.toContain("endSourceWithSourceVideoMultiClip");
      });
    });

    describe("the upload's own failure states", () => {
      it("blocks on a trim that was requested but not applied", async () => {
        const inner = createMockBridge({ delayMs: 0 });
        const { result } = renderHook(() =>
          useChainForm(FALLBACK_APP_CONFIG, "a cat riding a skateboard", { nativeBridge: dropsTrimQuery(inner) }),
        );
        await act(async () => {
          await result.current.attachEndSourceByPath("C:\\v\\tail.mp4", "tail.mp4", {
            trim: { trim: true, startSec: 1, durationSec: 30 },
            knownDurationSec: 600,
          });
        });
        await waitFor(() => expect(result.current.endSourceStatus).toBe("ready"));
        expect(result.current.validityReasons).toContain("endSourceTrimFailed");
        expect(result.current.isValid).toBe(false);
      });

      it("reports a failed upload", async () => {
        const failing = createMockBridge({ delayMs: 0, failUploadFile: "BACKEND_UNREACHABLE" });
        const { result } = renderHook(() =>
          useChainForm(FALLBACK_APP_CONFIG, "a cat riding a skateboard", { nativeBridge: failing }),
        );
        await act(async () => {
          await result.current.attachEndSourceByPath("C:\\v\\tail.mp4", "tail.mp4");
        });
        await waitFor(() => expect(result.current.endSourceStatus).toBe("error"));
        expect(result.current.validityReasons).toContain("endSourceNotReady");
        expect(result.current.isValid).toBe(false);
      });

      it("does not send the key while the upload is still in flight", async () => {
        const held = createMockBridge({ delayMs: 0, holdUploads: true });
        const { result } = renderHook(() =>
          useChainForm(FALLBACK_APP_CONFIG, "a cat riding a skateboard", { nativeBridge: held }),
        );
        act(() => {
          void result.current.attachEndSourceByPath("C:\\v\\tail.mp4", "tail.mp4");
        });
        await waitFor(() => expect(result.current.endSourceStatus).toBe("uploading"));
        expect(result.current.buildRequest()).not.toHaveProperty("end_source");
        expect(result.current.validityReasons).toContain("endSourceUploading");

        act(() => held.releaseUploads());
        await waitFor(() => expect(result.current.endSourceStatus).toBe("ready"));
      });
    });

    describe("buildRequest", () => {
      it("sends video_id with the EXPLICIT 8-frame anchor and the default 1.0 strength", async () => {
        const { result } = setupEnd(FALLBACK_APP_CONFIG, { uploadVideoFrameCount: 73, uploadVideoFps: 24 });
        await attachEndVideo(result);

        const request = result.current.buildRequest();
        expect(request.end_source).toEqual({
          video_id: result.current.endSourceVideo.state.id,
          context_frames: 8,
          strength: 1.0,
        });
      });

      it("sends image_id with the same 8 for an image end source", async () => {
        const { result } = setupEnd();
        await attachEndImage(result);

        const request = result.current.buildRequest();
        expect(request.end_source).toEqual({
          image_id: result.current.endSourceImage.items[0]?.imageId,
          context_frames: 8,
          strength: 1.0,
        });
      });

      // バッチ1 (2026-08-18): `setEndSourceStrength` rides straight into the
      // request the same turn — no separate "commit" step, matching
      // `overlapStrength`'s own wiring.
      it("follows setEndSourceStrength into the request", async () => {
        const { result } = setupEnd();
        await attachEndVideo(result);

        act(() => result.current.setEndSourceStrength(0.5));
        expect(result.current.buildRequest().end_source?.strength).toBe(0.5);
      });
    });

    // バッチ1 (2026-08-18): the anchor's fixed-strength slider state, exposed
    // and clamped independently of whether material is attached yet — same
    // shape as `overlapFrames`/`overlapStrength`'s own test above.
    it("endSourceStrength defaults to 1.0 and clamps to [0, 1]", () => {
      const { result } = setup();
      expect(result.current.endSourceStrength).toBe(1.0);

      act(() => result.current.setEndSourceStrength(5));
      expect(result.current.endSourceStrength).toBe(1);
      act(() => result.current.setEndSourceStrength(-1));
      expect(result.current.endSourceStrength).toBe(0);
      act(() => result.current.setEndSourceStrength(0.5));
      expect(result.current.endSourceStrength).toBe(0.5);
    });

    it("isDirty flags an attached end source (attach INTENT)", async () => {
      const { result } = setupEnd();
      expect(result.current.isDirty).toBe(false);
      await attachEndVideo(result);
      expect(result.current.isDirty).toBe(true);
    });

    it("every end-source reason code has a message, in both languages", () => {
      const codes = [
        "endSourceUploading",
        "endSourceNotReady",
        "endSourceTrimFailed",
        "endSourceConflictsWithAudio",
        "endSourceConflictsWithReference",
        "endSourceWithSourceVideoMultiClip",
        "endSourceTooShort",
        "endSourceLengthUnknown",
        "endSourceNeedsOverlap",
        "endSourceAudioOverlapBudget",
      ];
      for (const strings of [en, ja]) {
        const messages = buildChainReasonMessages(strings, { minFramesForOverlap: 25, minClips: 2 });
        for (const code of codes) {
          expect(messages[code], `${code} (${strings === en ? "en" : "ja"})`).toBeTruthy();
        }
      }
      // The v1 geometry lines are gone for good — a stale code would silently
      // render as no line at all in the Generate-reasons note.
      const messages = buildChainReasonMessages(en, { minFramesForOverlap: 25, minClips: 2 });
      expect(messages.endContextFramesInvalid).toBeUndefined();
      expect(messages.endContextFramesTooLongForWindow).toBeUndefined();
      expect(messages.endContextFramesTooLongForClip).toBeUndefined();
      expect(messages.endSourceTooShortForContext).toBeUndefined();
    });

    // §4-4: Batch i2v-long must never send end material.
    it("exposes hasEndSource for the batch's own block gate", async () => {
      const { result } = setupEnd();
      expect(result.current.hasEndSource).toBe(false);
      await attachEndVideo(result);
      expect(result.current.hasEndSource).toBe(true);
    });
  });
});

/**
 * A stand-in for an OLD native build (or any backend that cannot cut): it strips
 * contract v10's `query` before the call reaches the mock, so the response comes
 * back with `trimmed: false` even though a trim was asked for. That is exactly
 * the situation `sourceTrimFailed` exists to catch.
 */
/** What the two audio probes should answer for a §1-16 test. `wav` is
 * `fs.probeAudioDuration`'s result (default: "not a wav"), `mediaInfoDurationSec`
 * is `fs.probeMediaInfo`'s (default: 0 = unknown), and `mediaInfoRejects` makes
 * that second probe REJECT the way native really does for a bad path — the case
 * the hook must survive without an unhandled rejection. */
interface AudioProbeFixture {
  wav?: { durationSec: number; isWav: boolean };
  mediaInfoDurationSec?: number;
  mediaInfoRejects?: boolean;
}

/** Wraps a mock bridge with deterministic audio-probe answers (the real mock's
 * `fs.probeAudioDuration` is keyed off its in-memory filesystem, which these
 * tests don't populate) and counts how many times each probe was called — the
 * three-tier resolution order is only observable through those counts. Same
 * pass-through shape as {@link dropsTrimQuery}. */
function withAudioProbes(
  inner: NativeBridge,
  probes: AudioProbeFixture,
  calls: { wav: number; media: number },
): NativeBridge {
  return {
    ...inner,
    request<M extends BridgeMethod>(method: M, params: ParamsOf<M>): Promise<ResultOf<M>> {
      if (method === "fs.probeAudioDuration") {
        calls.wav += 1;
        return Promise.resolve((probes.wav ?? { durationSec: 0, isWav: false }) as ResultOf<M>);
      }
      if (method === "fs.probeMediaInfo") {
        calls.media += 1;
        if (probes.mediaInfoRejects) {
          return Promise.reject(new BridgeError("BAD_REQUEST", "Mock: unreadable path"));
        }
        return Promise.resolve({
          durationSec: probes.mediaInfoDurationSec ?? 0,
          width: 0,
          height: 0,
        } as ResultOf<M>);
      }
      return inner.request(method, params);
    },
  };
}

/** Counts `fs.probeMediaInfo` calls (the SOURCE card's resolution readout is
 * specified as ONE probe per ready material, which is only observable as a
 * count) and, with `rejects`, makes that probe fail the way native really does
 * for an unreadable path. Everything else passes through to `inner`, whose own
 * `probeMediaInfoWidth`/`Height` options supply the successful answer. Same
 * pass-through shape as {@link withAudioProbes}. */
function withMediaInfoProbe(
  inner: NativeBridge,
  behavior: { rejects: boolean },
  calls: { media: number },
): NativeBridge {
  return {
    ...inner,
    request<M extends BridgeMethod>(method: M, params: ParamsOf<M>): Promise<ResultOf<M>> {
      if (method === "fs.probeMediaInfo") {
        calls.media += 1;
        if (behavior.rejects) return Promise.reject(new BridgeError("BAD_REQUEST", "Mock: unreadable path"));
      }
      return inner.request(method, params);
    },
  };
}

/** §1-15: records the `query` every `backend.uploadFile` call carried, so the
 * reference slot's `max_frames` cap is assertable (it is invisible in the
 * resulting state — the mock answers the same body either way). */
function recordsUploadQuery(
  inner: NativeBridge,
  uploads: { query?: Record<string, string> | undefined }[],
): NativeBridge {
  return {
    ...inner,
    request<M extends BridgeMethod>(method: M, params: ParamsOf<M>): Promise<ResultOf<M>> {
      if (method === "backend.uploadFile") {
        uploads.push({ query: (params as ParamsOf<"backend.uploadFile">).query });
      }
      return inner.request(method, params);
    },
  };
}

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
