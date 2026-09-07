import { describe, expect, it } from "vitest";
import type { LoraSpec } from "../../api/types";
import type { NagSettings } from "../../shell/nagSettings";
import { ACCELERATION_DEFAULTS } from "../../shell/accelerationSettings";
import {
  audioLatentsAvailable,
  audioLatentsRequired,
  audioLengthPrecheck,
  audioSegmentWindows,
  AUDIO_LATENTS_PER_SEC,
  buildChainRequest,
  chainLayoutError,
  clampCropOutput,
  clampEndSourceStrength,
  clampOverlapFrames,
  clampOverlapStrength,
  computeOutputFrames,
  computeTotalFrames,
  CROP_OUTPUT_MIN,
  endSourceAudioOverlapOk,
  endSourceLastClipHasFreeLatents,
  isClipCountValid,
  isClipIntact,
  isContextFramesValid,
  isCropOutputValid,
  isTotalFramesValid,
  MAX_CHAIN_TOTAL_FRAMES,
  maxChainAudioLatents,
  MIN_CLIPS_NO_SOURCE,
  minClipFramesForKv,
  minClipsForChain,
  pxFromVLatent,
  pyRound,
  rawFramesForAudio,
  recommendedClipFrames,
  snapClipNumFrames,
  snapContextFrames,
  suggestFramesForAudio,
  vLatentFrames,
  vTailLatents,
} from "./chainUtils";
import type { ChainLayoutError } from "./chainUtils";
import { STAGE2_WINDOW_PRESETS } from "../../shell/tokenBudget";

describe("chainUtils", () => {
  describe("minClipsForChain", () => {
    it("requires at least 2 clips with no source video (from-scratch concat)", () => {
      expect(minClipsForChain(false)).toBe(MIN_CLIPS_NO_SOURCE);
    });

    it("allows a single clip once a source video is attached (V2V)", () => {
      expect(minClipsForChain(true)).toBe(1);
    });
  });

  describe("isClipCountValid", () => {
    it("rejects a single clip with no source video (needs >=2), up to 24", () => {
      expect(isClipCountValid(false, 1)).toBe(false);
      expect(isClipCountValid(false, 2)).toBe(true);
      expect(isClipCountValid(false, 24)).toBe(true);
      expect(isClipCountValid(false, 25)).toBe(false);
    });

    it("accepts a single clip once a source video is attached (V2V)", () => {
      expect(isClipCountValid(true, 1)).toBe(true);
      expect(isClipCountValid(true, 0)).toBe(false);
      expect(isClipCountValid(true, 24)).toBe(true);
      expect(isClipCountValid(true, 25)).toBe(false);
    });
  });

  describe("computeTotalFrames / isTotalFramesValid", () => {
    it("sums num_frames across clips", () => {
      expect(computeTotalFrames([{ numFrames: 49 }, { numFrames: 97 }, { numFrames: 17 }])).toBe(163);
      expect(computeTotalFrames([])).toBe(0);
    });

    it("is valid up to and including the 11544 ceiling, invalid beyond it", () => {
      expect(isTotalFramesValid(MAX_CHAIN_TOTAL_FRAMES)).toBe(true);
      expect(isTotalFramesValid(MAX_CHAIN_TOTAL_FRAMES + 1)).toBe(false);
      expect(isTotalFramesValid(0)).toBe(false);
      expect(isTotalFramesValid(-1)).toBe(false);
    });
  });

  describe("isContextFramesValid", () => {
    const min = 25;
    const max = 145;

    it("requires 8n+1", () => {
      expect(isContextFramesValid(73, min, max, 481)).toBe(true);
      expect(isContextFramesValid(74, min, max, 481)).toBe(false);
    });

    it("requires within [min, max]", () => {
      expect(isContextFramesValid(min, min, max, 481)).toBe(true);
      expect(isContextFramesValid(max, min, max, 481)).toBe(true);
      expect(isContextFramesValid(17, min, max, 481)).toBe(false); // below min, even though 8n+1
      expect(isContextFramesValid(153, min, max, 481)).toBe(false); // above max
    });

    it("rejects when context_frames >= clips[0].num_frames", () => {
      expect(isContextFramesValid(73, min, max, 73)).toBe(false);
      expect(isContextFramesValid(73, min, max, 65)).toBe(false);
      expect(isContextFramesValid(73, min, max, 81)).toBe(true);
    });
  });

  describe("snapClipNumFrames / snapContextFrames", () => {
    it("snaps to the nearest 8n+1 within bounds", () => {
      expect(snapClipNumFrames(100)).toBe(97);
      expect(snapClipNumFrames(1)).toBe(9);
      expect(snapClipNumFrames(10_000)).toBe(481);
      expect(snapClipNumFrames(100, 50)).toBeLessThanOrEqual(50);
    });

    it("snapContextFrames stays within the given [min, max]", () => {
      expect(snapContextFrames(73, 25, 145)).toBe(73);
      expect(snapContextFrames(1, 25, 145)).toBe(25);
      expect(snapContextFrames(1000, 25, 145)).toBe(145);
    });
  });

  describe("clampCropOutput (N1: no 32-multiple grid, Gradio-faithful)", () => {
    it("passes an arbitrary in-range integer through unchanged (e.g. 1080, previously wrongly rounded to 1088)", () => {
      expect(clampCropOutput({ width: 1920, height: 1080 }, 1920, 1088)).toEqual({ width: 1920, height: 1080 });
    });

    it("rounds a non-integer to the nearest integer without snapping to any grid", () => {
      expect(clampCropOutput({ width: 1000.4, height: 1000.6 }, 1920, 1088)).toEqual({ width: 1000, height: 1001 });
    });

    it("clamps below CROP_OUTPUT_MIN (32) up to the floor", () => {
      expect(clampCropOutput({ width: 10, height: 31 }, 1920, 1088)).toEqual({ width: CROP_OUTPUT_MIN, height: CROP_OUTPUT_MIN });
    });

    it("clamps above the current generation width/height down to the ceiling", () => {
      expect(clampCropOutput({ width: 5000, height: 5000 }, 1920, 1088)).toEqual({ width: 1920, height: 1088 });
    });
  });

  describe("isCropOutputValid", () => {
    it("accepts null (crop disabled)", () => {
      expect(isCropOutputValid(null, 1920, 1088)).toBe(true);
    });

    it("accepts an arbitrary in-range integer with no grid constraint (1920x1080 within a 1920x1088 generation)", () => {
      expect(isCropOutputValid({ width: 1920, height: 1080 }, 1920, 1088)).toBe(true);
    });

    it("rejects a crop larger than the current generation width/height", () => {
      expect(isCropOutputValid({ width: 1921, height: 1080 }, 1920, 1088)).toBe(false);
      expect(isCropOutputValid({ width: 1920, height: 1089 }, 1920, 1088)).toBe(false);
    });

    it("rejects below CROP_OUTPUT_MIN (32)", () => {
      expect(isCropOutputValid({ width: 31, height: 100 }, 1920, 1088)).toBe(false);
    });

    it("rejects a non-integer value", () => {
      expect(isCropOutputValid({ width: 100.5, height: 100 }, 1920, 1088)).toBe(false);
    });

    it("accepts the exact min/max boundaries", () => {
      expect(isCropOutputValid({ width: CROP_OUTPUT_MIN, height: CROP_OUTPUT_MIN }, 1920, 1088)).toBe(true);
      expect(isCropOutputValid({ width: 1920, height: 1088 }, 1920, 1088)).toBe(true);
    });
  });

  describe("clampOverlapFrames / clampOverlapStrength", () => {
    it("clamps overlap_frames to [1, 8]", () => {
      expect(clampOverlapFrames(0)).toBe(1);
      expect(clampOverlapFrames(3)).toBe(3);
      expect(clampOverlapFrames(20)).toBe(8);
    });

    it("clamps overlap_strength to [0, 1]", () => {
      expect(clampOverlapStrength(-0.5)).toBe(0);
      expect(clampOverlapStrength(0.5)).toBe(0.5);
      expect(clampOverlapStrength(2)).toBe(1);
    });
  });

  describe("clampEndSourceStrength", () => {
    it("clamps to [0, 1], same range as overlap_strength", () => {
      expect(clampEndSourceStrength(-0.5)).toBe(0);
      expect(clampEndSourceStrength(0.5)).toBe(0.5);
      expect(clampEndSourceStrength(2)).toBe(1);
    });
  });

  describe("buildChainRequest", () => {
    const base = {
      prompt: "a cat riding a skateboard",
      width: 512,
      height: 320,
      frameRate: 24,
      seed: -1,
      overlapFrames: 3,
      overlapStrength: 0.5,
      chunkedUpsample: true,
    };

    // §1-14/§3-57 stage-2 window. Unlike `chunked_upsample` (always explicit),
    // this follows the same "omit at the default" rule as every other optional
    // field here, so a chain that never touches the control sends a request
    // byte-identical to before the field existed.
    describe("stage2_window", () => {
      const clips = [
        { id: "c0", prompt: "", numFrames: 49 },
        { id: "c1", prompt: "", numFrames: 49 },
      ];

      it("omits the key entirely when the window is left at its default", () => {
        expect(buildChainRequest({ ...base, clips })).not.toHaveProperty("stage2_window");
        expect(
          buildChainRequest({ ...base, clips, stage2Window: "standard" }),
        ).not.toHaveProperty("stage2_window");
      });

      it("sends the key only when the user opts into the shorter step", () => {
        expect(buildChainRequest({ ...base, clips, stage2Window: "high_resolution" }).stage2_window).toBe(
          "high_resolution",
        );
      });

      it("changes nothing else about the request", () => {
        const withDefault = buildChainRequest({ ...base, clips, stage2Window: "standard" });
        const optedIn = buildChainRequest({ ...base, clips, stage2Window: "high_resolution" });
        const { stage2_window: _omitted, ...rest } = optedIn;
        expect(rest).toEqual(withDefault);
      });
    });

    it("builds a Clips request with per-clip prompt overrides and clip-0-only conditioning images", () => {
      const request = buildChainRequest({
        ...base,
        clips: [
          {
            id: "c0",
            prompt: "",
            numFrames: 49,
            conditioningImages: [{ image_id: "img-1", frame_idx: 0, strength: 0.8 }],
          },
          { id: "c1", prompt: "a dog joins in", numFrames: 33 },
        ],
      });

      expect(request.clips).toEqual([
        { num_frames: 49, conditioning_images: [{ image_id: "img-1", frame_idx: 0, strength: 0.8 }] },
        { num_frames: 33, prompt: "a dog joins in" },
      ]);
      expect(request.source_video).toBeUndefined();
      expect(request.source_audio).toBeUndefined();
    });

    it("never attaches conditioning_images to any clip but index 0, even if the input carries them", () => {
      const request = buildChainRequest({
        ...base,
        clips: [
          { id: "c0", prompt: "", numFrames: 49 },
          { id: "c1", prompt: "", numFrames: 33, conditioningImages: [{ image_id: "img-2", frame_idx: 0, strength: 0.8 }] },
        ],
      });

      expect(request.clips[1]).toEqual({ num_frames: 33 });
    });

    it("omits a blank clip prompt entirely rather than sending an empty string", () => {
      const request = buildChainRequest({ ...base, clips: [{ id: "c0", prompt: "   ", numFrames: 49 }] });
      expect(request.clips[0]).not.toHaveProperty("prompt");
    });

    it("attaches source_video for V2V", () => {
      const request = buildChainRequest({
        ...base,
        clips: [{ id: "c0", prompt: "", numFrames: 49 }],
        sourceVideo: { videoId: "vid-1", contextFrames: 73 },
      });
      expect(request.source_video).toEqual({ video_id: "vid-1", context_frames: 73 });
      expect(request.source_audio).toBeUndefined();
    });

    // §1-16 long-form A2V: one audio track for the whole chain.
    describe("source_audio", () => {
      const clips = [
        { id: "c0", prompt: "", numFrames: 257 },
        { id: "c1", prompt: "", numFrames: 257 },
      ];

      it("attaches source_audio as { audio_id } when given", () => {
        const request = buildChainRequest({ ...base, clips, sourceAudio: { audioId: "aud-1" } });
        expect(request.source_audio).toEqual({ audio_id: "aud-1" });
      });

      it("omits the key entirely when unset or explicitly null", () => {
        expect(buildChainRequest({ ...base, clips })).not.toHaveProperty("source_audio");
        expect(buildChainRequest({ ...base, clips, sourceAudio: null })).not.toHaveProperty("source_audio");
      });

      it("changes nothing else about the request", () => {
        const baseline = buildChainRequest({ ...base, clips });
        const withAudio = buildChainRequest({ ...base, clips, sourceAudio: { audioId: "aud-1" } });
        const { source_audio: _omitted, ...rest } = withAudio;
        expect(rest).toEqual(baseline);
      });

      it("carries multiple clips through unchanged (the 1-clip A2V guard is gone)", () => {
        const request = buildChainRequest({
          ...base,
          clips: [...clips, { id: "c2", prompt: "", numFrames: 121 }],
          sourceAudio: { audioId: "aud-1" },
        });
        expect(request.clips).toHaveLength(3);
        expect(request.source_audio).toEqual({ audio_id: "aud-1" });
      });
    });

    // 素材（末尾）: exactly ONE of `video_id`/`image_id`, decided by `kind`.
    describe("end_source", () => {
      const clips = [
        { id: "c0", prompt: "", numFrames: 257 },
        { id: "c1", prompt: "", numFrames: 257 },
      ];

      it("sends video_id (and no image_id) for a video end source", () => {
        const request = buildChainRequest({
          ...base,
          clips,
          endSource: { kind: "video", id: "vid-9", contextFrames: 72, strength: 1.0 },
        });
        expect(request.end_source).toEqual({ video_id: "vid-9", context_frames: 72, strength: 1.0 });
        expect(request.end_source).not.toHaveProperty("image_id");
      });

      it("sends image_id (and no video_id) for an image end source", () => {
        const request = buildChainRequest({
          ...base,
          clips,
          endSource: { kind: "image", id: "img-9", contextFrames: 8, strength: 1.0 },
        });
        expect(request.end_source).toEqual({ image_id: "img-9", context_frames: 8, strength: 1.0 });
        expect(request.end_source).not.toHaveProperty("video_id");
      });

      it("omits the key entirely when unset or explicitly null", () => {
        expect(buildChainRequest({ ...base, clips })).not.toHaveProperty("end_source");
        expect(buildChainRequest({ ...base, clips, endSource: null })).not.toHaveProperty("end_source");
      });

      it("changes nothing else about the request (additive contract)", () => {
        const baseline = buildChainRequest({ ...base, clips });
        const withEnd = buildChainRequest({
          ...base,
          clips,
          endSource: { kind: "video", id: "vid-9", contextFrames: 72, strength: 1.0 },
        });
        const { end_source: _omitted, ...rest } = withEnd;
        expect(rest).toEqual(baseline);
      });

      it("rides alongside source_video (start + end is the interpolation case)", () => {
        const request = buildChainRequest({
          ...base,
          clips,
          sourceVideo: { videoId: "start-1", contextFrames: 73 },
          endSource: { kind: "video", id: "end-1", contextFrames: 72, strength: 1.0 },
        });
        expect(request.source_video).toEqual({ video_id: "start-1", context_frames: 73 });
        expect(request.end_source).toEqual({ video_id: "end-1", context_frames: 72, strength: 1.0 });
      });

      // バッチ1 (2026-08-18): `strength` rides alongside `end_source` — see
      // `EndSourceInput.strength`'s own doc comment for why it is always
      // explicit, unlike `stage2Window`'s "omit at the default" rule.
      describe("strength", () => {
        it("sends strength even at its 1.0 default — same 'always explicit' rule as context_frames", () => {
          const request = buildChainRequest({
            ...base,
            clips,
            endSource: { kind: "video", id: "vid-9", contextFrames: 72, strength: 1.0 },
          });
          expect(request.end_source?.strength).toBe(1.0);
        });

        it("sends a softened strength unchanged", () => {
          const request = buildChainRequest({
            ...base,
            clips,
            endSource: { kind: "video", id: "vid-9", contextFrames: 72, strength: 0.5 },
          });
          expect(request.end_source?.strength).toBe(0.5);
        });
      });
    });

    it("always sends chunked_upsample explicitly, mirroring whatever the caller passed", () => {
      const on = buildChainRequest({ ...base, chunkedUpsample: true, clips: [{ id: "c0", prompt: "", numFrames: 49 }] });
      expect(on.chunked_upsample).toBe(true);

      const off = buildChainRequest({ ...base, chunkedUpsample: false, clips: [{ id: "c0", prompt: "", numFrames: 49 }] });
      expect(off.chunked_upsample).toBe(false);
    });

    it("omits loras entirely when unset or empty, includes it when non-empty", () => {
      const noLoras = buildChainRequest({ ...base, clips: [{ id: "c0", prompt: "", numFrames: 49 }] });
      expect(noLoras).not.toHaveProperty("loras");

      const emptyLoras = buildChainRequest({ ...base, clips: [{ id: "c0", prompt: "", numFrames: 49 }], loras: [] });
      expect(emptyLoras).not.toHaveProperty("loras");

      const loras: LoraSpec[] = [{ name: "my-style", strength: 0.8 }];
      const withLoras = buildChainRequest({ ...base, clips: [{ id: "c0", prompt: "", numFrames: 49 }], loras });
      expect(withLoras.loras).toEqual(loras);
    });

    it("carries a lora's audio_strength through to the request body untouched (no field projection)", () => {
      const loras: LoraSpec[] = [{ name: "X", strength: 0.8, audio_strength: 0.3 }];
      const request = buildChainRequest({ ...base, clips: [{ id: "c0", prompt: "", numFrames: 49 }], loras });
      expect(request.loras).toEqual([{ name: "X", strength: 0.8, audio_strength: 0.3 }]);
    });

    it("omits reference_video_id/conditioning_attention_strength/reference_video_strength entirely when unset", () => {
      const request = buildChainRequest({ ...base, clips: [{ id: "c0", prompt: "", numFrames: 49 }] });
      expect(request).not.toHaveProperty("reference_video_id");
      expect(request).not.toHaveProperty("conditioning_attention_strength");
      expect(request).not.toHaveProperty("reference_video_strength");
    });

    it("attaches reference_video_id and the two strength fields when present", () => {
      const loras: LoraSpec[] = [{ name: "control-lora", strength: 1.0 }];
      const request = buildChainRequest({
        ...base,
        clips: [{ id: "c0", prompt: "", numFrames: 49 }],
        loras,
        referenceVideoId: "ref-vid-1",
        conditioningAttentionStrength: 0.7,
        referenceVideoStrength: 0.3,
      });
      expect(request.reference_video_id).toBe("ref-vid-1");
      expect(request.conditioning_attention_strength).toBe(0.7);
      expect(request.reference_video_strength).toBe(0.3);
    });

    it("sends 0.0 for the strength fields rather than treating it as unset (falsy != null)", () => {
      const loras: LoraSpec[] = [{ name: "control-lora", strength: 1.0 }];
      const request = buildChainRequest({
        ...base,
        clips: [{ id: "c0", prompt: "", numFrames: 49 }],
        loras,
        referenceVideoId: "ref-vid-1",
        conditioningAttentionStrength: 0.0,
        referenceVideoStrength: 0.0,
      });
      expect(request.conditioning_attention_strength).toBe(0.0);
      expect(request.reference_video_strength).toBe(0.0);
      expect(request).toHaveProperty("conditioning_attention_strength");
      expect(request).toHaveProperty("reference_video_strength");
    });

    it("treats explicit null the same as omitted for all three reference fields", () => {
      const request = buildChainRequest({
        ...base,
        clips: [{ id: "c0", prompt: "", numFrames: 49 }],
        referenceVideoId: null,
        conditioningAttentionStrength: null,
        referenceVideoStrength: null,
      });
      expect(request).not.toHaveProperty("reference_video_id");
      expect(request).not.toHaveProperty("conditioning_attention_strength");
      expect(request).not.toHaveProperty("reference_video_strength");
    });

    // NAG (2026-07-28): the additive `nagRequestFields(params.nag)` spread.
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

      it("without nag: none of the 7 additive keys are present", () => {
        const request = buildChainRequest({ ...base, clips: [{ id: "c0", prompt: "", numFrames: 49 }] });
        expect(request).not.toHaveProperty("negative_prompt");
        expect(request).not.toHaveProperty("nag_enabled");
        expect(request).not.toHaveProperty("nag_scale");
        expect(request).not.toHaveProperty("nag_tau");
        expect(request).not.toHaveProperty("nag_alpha");
        expect(request).not.toHaveProperty("neg_method");
        expect(request).not.toHaveProperty("vsf_scale");
      });

      it("with enabled nag: all 7 fields are present with their values", () => {
        const request = buildChainRequest({ ...base, clips: [{ id: "c0", prompt: "", numFrames: 49 }], nag: ENABLED_NAG });
        expect(request.negative_prompt).toBe(ENABLED_NAG.text);
        expect(request.nag_enabled).toBe(true);
        expect(request.nag_scale).toBe(11.0);
        expect(request.nag_tau).toBe(2.5);
        expect(request.nag_alpha).toBe(0.25);
        expect(request.neg_method).toBe("nag");
        expect(request.vsf_scale).toBe(1.5);
      });

      it("with enabled nag and method 'vsf': neg_method/vsf_scale reflect VSF", () => {
        const request = buildChainRequest({
          ...base,
          clips: [{ id: "c0", prompt: "", numFrames: 49 }],
          nag: { ...ENABLED_NAG, method: "vsf", vsfScale: 3 },
        });
        expect(request.neg_method).toBe("vsf");
        expect(request.vsf_scale).toBe(3);
      });

      it("with disabled nag: same as omitted (soft-disable keeps the fields off the wire)", () => {
        const request = buildChainRequest({
          ...base,
          clips: [{ id: "c0", prompt: "", numFrames: 49 }],
          nag: { ...ENABLED_NAG, enabled: false },
        });
        expect(request).not.toHaveProperty("negative_prompt");
        expect(request).not.toHaveProperty("nag_enabled");
      });
    });

    // Acceleration (2026-07-31): the additive
    // `accelerationRequestFields(params.acceleration)` spread.
    describe("acceleration", () => {
      const clips = [{ id: "c0", prompt: "", numFrames: 49 }];

      it("omitted / all-defaults: the request is byte-identical (key order included)", () => {
        const baseline = buildChainRequest({ ...base, clips });
        const withDefaults = buildChainRequest({ ...base, clips, acceleration: ACCELERATION_DEFAULTS });
        expect(JSON.stringify(withDefaults)).toBe(JSON.stringify(baseline));
        expect(withDefaults).not.toHaveProperty("attention_backend");
      });

      it("sage: exactly one key more than the default request", () => {
        const baseline = buildChainRequest({ ...base, clips });
        const sage = buildChainRequest({
          ...base,
          clips,
          acceleration: { ...ACCELERATION_DEFAULTS, attentionBackend: "sage" },
        });
        expect(sage.attention_backend).toBe("sage");
        expect(Object.keys(sage)).toEqual([...Object.keys(baseline), "attention_backend"]);
      });

      it("block-swap prefetch off: exactly one key more than the default request", () => {
        // S4: default is now true, so OFF is what diverges.
        const baseline = buildChainRequest({ ...base, clips });
        const prefetch = buildChainRequest({
          ...base,
          clips,
          acceleration: { ...ACCELERATION_DEFAULTS, blockSwapPrefetch: false },
        });
        expect(prefetch.block_swap_prefetch).toBe(false);
        expect(Object.keys(prefetch)).toEqual([...Object.keys(baseline), "block_swap_prefetch"]);
      });

      it("keep-resident on: exactly one key more than the default request", () => {
        // §48: server default is false, so ON is what diverges.
        const baseline = buildChainRequest({ ...base, clips });
        const keepResident = buildChainRequest({
          ...base,
          clips,
          acceleration: { ...ACCELERATION_DEFAULTS, keepResident: true },
        });
        expect(keepResident.keep_resident).toBe(true);
        expect(Object.keys(keepResident)).toEqual([...Object.keys(baseline), "keep_resident"]);
      });

      it("fused GGUF dequant kernel off: exactly one key more than the default request (§1-11)", () => {
        // §51 (2026-08-04): server default flipped to true, so OFF is what
        // diverges.
        const baseline = buildChainRequest({ ...base, clips });
        const fused = buildChainRequest({
          ...base,
          clips,
          acceleration: { ...ACCELERATION_DEFAULTS, fusedGgufDequantKernel: false },
        });
        expect(fused.fused_gguf_dequant_kernel).toBe(false);
        expect(Object.keys(fused)).toEqual([...Object.keys(baseline), "fused_gguf_dequant_kernel"]);
      });

      it("PrunaVAED: exactly one key more than the default request (§52)", () => {
        // §52 (2026-08-05): server default is "default", so PrunaVAED is what
        // diverges. The API VALUE stays `"prune_vaed"` even though the display
        // name is PrunaVAED.
        const baseline = buildChainRequest({ ...base, clips });
        const request = buildChainRequest({
          ...base,
          clips,
          acceleration: { ...ACCELERATION_DEFAULTS, vaeMode: "prune_vaed" },
        });
        expect(request.vae_mode).toBe("prune_vaed");
        expect(Object.keys(request)).toEqual([...Object.keys(baseline), "vae_mode"]);
        expect(request).not.toHaveProperty("attention_backend");
      });

      it("leaves vae_mode off entirely at the server default", () => {
        const request = buildChainRequest({ ...base, clips, acceleration: ACCELERATION_DEFAULTS });
        expect(request).not.toHaveProperty("vae_mode");
        expect(request).not.toHaveProperty("attention_backend");
      });
    });
  });

  describe("vLatentFrames / pxFromVLatent", () => {
    it("matches chain_math.py's (P-1)//8+1 / (N-1)*8+1", () => {
      expect(vLatentFrames(1)).toBe(1);
      expect(vLatentFrames(9)).toBe(2);
      expect(vLatentFrames(121)).toBe(16);
      expect(vLatentFrames(481)).toBe(61);
      expect(pxFromVLatent(1)).toBe(1);
      expect(pxFromVLatent(16)).toBe(121);
      expect(pxFromVLatent(61)).toBe(481);
    });

    it("round-trips for any pixel-frame value already on the 8n+1 grid", () => {
      for (const px of [1, 9, 49, 121, 225, 481]) {
        expect(pxFromVLatent(vLatentFrames(px))).toBe(px);
      }
    });
  });

  describe("computeOutputFrames", () => {
    it("matches the PHASE3_CLIP_CONCAT_STATUS.md worked example: [121,121] overlap=3 -> 225", () => {
      expect(computeOutputFrames([121, 121], 3, null)).toBe(225);
    });

    it("subtracts V2V context_frames from the same layout: 225 - 73 -> 152", () => {
      expect(computeOutputFrames([121, 121], 3, 73)).toBe(152);
    });

    it("a single clip passes through unchanged (no seam to fuse)", () => {
      expect(computeOutputFrames([49], 3, null)).toBe(49);
      expect(computeOutputFrames([121], 0, null)).toBe(121);
    });

    it("handles a full 24-clip chain", () => {
      const clips = Array<number>(24).fill(481);
      // v_latent(481) = 61 each; f_total = 24*61 - 23*3 = 1464 - 69 = 1395;
      // total_px = (1395-1)*8+1 = 11153.
      expect(computeOutputFrames(clips, 3, null)).toBe(11153);
    });

    it("overlap=0 is a no-op on the seam subtraction", () => {
      // v_latent(49) = 7 each; f_total = 14 - 0 = 14; total_px = 13*8+1 = 105.
      expect(computeOutputFrames([49, 49], 0, null)).toBe(105);
    });

    it("clamps to 0 instead of going negative when overlap swallows every clip", () => {
      // v_latent(9) = 2 each; f_total = 4 - 1*8 = -4 -> clamped to 0.
      expect(computeOutputFrames([9, 9], 8, null)).toBe(0);
    });

    it("clamps to 0 instead of going negative when context_frames exceeds total_px", () => {
      expect(computeOutputFrames([121], 3, 200)).toBe(0);
    });

    it("returns 0 for an empty clip list", () => {
      expect(computeOutputFrames([], 3, null)).toBe(0);
    });
  });

  describe("recommendedClipFrames", () => {
    // `spillFreeFrames` mirrors `webui/src/bridge/mockBridge.ts`'s
    // `MOCK_CONFIG_BODY.limits.spill_free_frames` (same as `defaultConfig.ts`'s
    // `FALLBACK_APP_CONFIG`). The `presetNumFrames` arguments below (257/153)
    // are arbitrary placeholders distinct from today's real `generation_presets`
    // values — they exist only to prove the function prefers the
    // `spill_free_frames` threshold over whatever preset value it is handed,
    // not to mirror any current default.
    const spillFreeFrames = { "1280x768": 273, "1920x1088": 161, "2560x1472": 81 };

    it("prefers the spill_free_frames threshold when the resolution has one", () => {
      expect(recommendedClipFrames(1280, 768, 257, spillFreeFrames)).toBe(273);
      expect(recommendedClipFrames(1920, 1088, 153, spillFreeFrames)).toBe(161);
    });

    it("falls back to the preset's own num_frames when the resolution has no entry", () => {
      expect(recommendedClipFrames(512, 320, 49, spillFreeFrames)).toBe(49);
      expect(recommendedClipFrames(512, 320, 49, undefined)).toBe(49);
    });

    it("snaps the result onto the 8n+1 grid (defensive — real config values are already valid)", () => {
      expect(recommendedClipFrames(999, 999, 100, undefined)).toBe(97); // matches snapClipNumFrames(100) above
      expect(recommendedClipFrames(1280, 768, 999_999, { "1280x768": 999_999 })).toBe(481); // clamped at the ceiling
    });
  });

  describe("audioLatentsRequired", () => {
    it("AUDIO_LATENTS_PER_SEC matches chain_math.py's constant (16000/160/4)", () => {
      expect(AUDIO_LATENTS_PER_SEC).toBe(25.0);
    });

    it("matches chain_math.py's audio_latents_required for a single clip (hand-verified)", () => {
      // vLatentFrames(49)=7 (>kv=3) -> fTotal=7 -> totalPx=49 -> a_total=round(49/24*25)=51.
      expect(audioLatentsRequired([49], 24, 3)).toBe(51);
      // vLatentFrames(25)=4 -> fTotal=4 -> totalPx=25 -> a_total=round(25/24*25)=26.
      expect(audioLatentsRequired([25], 24, 3)).toBe(26);
    });

    it("matches the multi-clip join subtraction (cross-checked against computeOutputFrames' own [121,121] worked example)", () => {
      // Same [121,121] overlap=3 layout as the computeOutputFrames test above:
      // f_total=29 -> total_px=225 (matches that test's video-frame result) ->
      // a_total=round(225/24*25)=234.
      expect(audioLatentsRequired([121, 121], 24, 3)).toBe(234);
    });

    it("returns null when kv >= a clip's video-latent frames (chain_math.py:289-293's ValueError guard)", () => {
      // vLatentFrames(17)=3, kv=3 -> 3>=3 -> geometrically invalid.
      expect(audioLatentsRequired([17], 24, 3)).toBeNull();
      // vLatentFrames(9)=2, kv=3 -> 3>=2 -> invalid.
      expect(audioLatentsRequired([9], 24, 3)).toBeNull();
    });

    it("defaults kv to DEFAULT_OVERLAP_FRAMES (3), matching chain_math.py's own default", () => {
      expect(audioLatentsRequired([49], 24)).toBe(audioLatentsRequired([49], 24, 3));
    });
  });

  describe("rawFramesForAudio / suggestFramesForAudio", () => {
    it("rawFramesForAudio computes the pre-shrink 8n+1 value, clamped only at the 9 floor", () => {
      expect(rawFramesForAudio(2.0, 24)).toBe(41); // floor(2*24)=48 -> (48-1)//8*8+1=41
      expect(rawFramesForAudio(0.1, 24)).toBe(9); // floor(0.1*24)=2 -> (2-1)//8*8+1=1 -> clamped to 9
      expect(rawFramesForAudio(100, 24)).toBeGreaterThan(481); // NOT clamped at 481, unlike suggestFramesForAudio
    });

    it("suggestFramesForAudio matches handlers.suggest_frames_for_audio's formula for representative durations", () => {
      expect(suggestFramesForAudio(2.0, 24)).toBe(41);
      expect(suggestFramesForAudio(10.0, 24)).toBe(rawFramesForAudio(10.0, 24));
    });

    it("clamps to the documented [9, 481] range at both ends", () => {
      expect(suggestFramesForAudio(0.1, 24)).toBe(9); // far too short -> floor
      expect(suggestFramesForAudio(100, 24)).toBe(481); // far too long -> ceiling
    });

    it("falls back to 24fps when fps is falsy (0/NaN), matching handlers._resolve_fps", () => {
      expect(suggestFramesForAudio(2.0, 0)).toBe(suggestFramesForAudio(2.0, 24));
      expect(rawFramesForAudio(2.0, Number.NaN)).toBe(rawFramesForAudio(2.0, 24));
    });

    it("every suggested value stays on the 8n+1 grid", () => {
      for (const dur of [0.1, 0.5, 1.0, 2.3, 5.7, 12.0, 40.0, 100.0]) {
        const nf = suggestFramesForAudio(dur, 24);
        expect((nf - 1) % 8).toBe(0);
      }
    });
  });

  describe("audioLengthPrecheck", () => {
    it("reports insufficient audio with the exact seconds required (mirrors a2v_msg_too_short's need_s)", () => {
      // audioLatentsRequired([49],24,3)=51 -> requiredSeconds=51/25=2.04; 1.0s
      // of audio (available=round(25)=25) falls well short of that.
      const result = audioLengthPrecheck(1.0, 49, 24);
      expect(result.ok).toBe(false);
      expect(result.requiredSeconds).toBeCloseTo(2.04, 5);
    });

    it("reports sufficient audio once the duration crosses the exact requirement", () => {
      expect(audioLengthPrecheck(2.04, 49, 24).ok).toBe(true); // available=round(2.04*25)=51=required
      expect(audioLengthPrecheck(10.0, 49, 24).ok).toBe(true);
    });

    it("is a no-op (ok:true) when the requirement can't be determined, matching the audioLatentsRequired(null) case", () => {
      // numFrames=17 -> vLatentFrames=3 <= kv(3) -> audioLatentsRequired returns null.
      const result = audioLengthPrecheck(0.1, 17, 24);
      expect(result.ok).toBe(true);
      expect(result.requiredSeconds).toBe(0);
    });
  });

  // ── §1-16 long-form A2V geometry ──────────────────────────────────────────
  // Every expected value below was produced by RUNNING the backend
  // (`Nz-Videomni/.venv/Scripts/python.exe` importing `chain_math`), not
  // by re-deriving the formulas here — the backend is the single source of
  // truth and these tests exist to catch the mirror drifting from it.

  describe("pyRound (Python's round(), banker's rounding)", () => {
    it("rounds exact halves to EVEN, where Math.round rounds up", () => {
      // Paired with the backend's own `round()`:
      //   round(0.5)=0  round(1.5)=2  round(2.5)=2  round(12.5)=12
      //   round(13.5)=14  round(24.5)=24
      expect(pyRound(0.5)).toBe(0);
      expect(pyRound(1.5)).toBe(2);
      expect(pyRound(2.5)).toBe(2);
      expect(pyRound(12.5)).toBe(12);
      expect(pyRound(13.5)).toBe(14);
      expect(pyRound(24.5)).toBe(24);
      // The whole point: Math.round disagrees on every one of the "down" cases.
      expect(Math.round(12.5)).toBe(13);
    });

    it("rounds negative halves to EVEN too (round(-0.5)=0, round(-1.5)=-2, round(-2.5)=-2)", () => {
      expect(pyRound(-0.5)).toBe(0);
      expect(pyRound(-1.5)).toBe(-2);
      expect(pyRound(-2.5)).toBe(-2);
    });

    it("is plain nearest-integer rounding everywhere else", () => {
      expect(pyRound(2.4)).toBe(2);
      expect(pyRound(2.6)).toBe(3);
      expect(pyRound(-2.4)).toBe(-2);
      expect(pyRound(-2.6)).toBe(-3);
      expect(pyRound(7)).toBe(7);
      // 0.49999999999999994 is the largest double below 0.5 — it must NOT be
      // treated as a half (Python agrees: round(0.49999999999999994) == 0).
      expect(pyRound(0.49999999999999994)).toBe(0);
    });

    it("changes a REAL audio-latent count: 49 frames @ 50fps is 24.5 -> 24, not 25", () => {
      // chain_math.a_frames_for_px(49, 50.0) == 24 (verified against the
      // backend venv). With Math.round the mirror would have said 25 and every
      // 50fps A2V length check would have been off by one latent frame.
      expect(audioLatentsRequired([49], 50, 3)).toBe(24);
      expect(audioLatentsRequired([25], 50, 3)).toBe(12);
    });
  });

  describe("audioLatentsAvailable", () => {
    it("mirrors preflight_source_audio's `available = round(duration * 25)`", () => {
      expect(audioLatentsAvailable(8)).toBe(200);
      expect(audioLatentsAvailable(2.04)).toBe(51);
      expect(audioLatentsAvailable(0)).toBe(0);
    });

    it("uses banker's rounding on the exact half, like the server does", () => {
      // 0.02s * 25 = 0.5 -> 0 (Python round), not 1.
      expect(audioLatentsAvailable(0.02)).toBe(0);
      // 0.06s * 25 = 1.5 -> 2.
      expect(audioLatentsAvailable(0.06)).toBe(2);
    });
  });

  describe("minClipFramesForKv", () => {
    it("is max(9, 8*kv+1) — the floor compute_chain_layout's kv guard implies", () => {
      expect(minClipFramesForKv(1)).toBe(9);
      expect(minClipFramesForKv(3)).toBe(25);
      expect(minClipFramesForKv(8)).toBe(65);
    });

    it("produces exactly the shortest clip audioLatentsRequired accepts", () => {
      for (const kv of [1, 2, 3, 5, 8]) {
        const floor = minClipFramesForKv(kv);
        expect(audioLatentsRequired([floor], 24, kv)).not.toBeNull();
        expect(audioLatentsRequired([floor - 8], 24, kv)).toBeNull();
      }
    });
  });

  describe("isClipIntact", () => {
    it("treats an absent flag as intact (out-of-the-box cards are untouched)", () => {
      expect(isClipIntact({})).toBe(true);
      expect(isClipIntact({ intact: true })).toBe(true);
      expect(isClipIntact({ intact: false })).toBe(false);
    });
  });

  describe("maxChainAudioLatents", () => {
    it("is the 24x481 chain's requirement (backend-verified)", () => {
      // chain_math.audio_latents_required([481]*24, fps, kv=kv):
      expect(maxChainAudioLatents(24, 3)).toBe(11618);
      expect(maxChainAudioLatents(25, 3)).toBe(11153);
      expect(maxChainAudioLatents(30, 3)).toBe(9294);
      expect(maxChainAudioLatents(24, 1)).toBe(12001);
      expect(maxChainAudioLatents(24, 8)).toBe(10659);
    });

    it("is exactly what audioLatentsRequired says for that chain (no duplicate arithmetic)", () => {
      expect(maxChainAudioLatents(24, 3)).toBe(audioLatentsRequired(Array<number>(24).fill(481), 24, 3));
    });

    it("defaults kv to 3 and falls back to 24fps like every other A2V helper", () => {
      expect(maxChainAudioLatents(24)).toBe(maxChainAudioLatents(24, 3));
      expect(maxChainAudioLatents(0, 3)).toBe(maxChainAudioLatents(24, 3));
    });
  });

  describe("audioSegmentWindows", () => {
    const pairs = (windows: ReturnType<typeof audioSegmentWindows>) =>
      windows === null ? null : windows.map((w) => [w.startLatent, w.lenLatent]);

    it("a single clip is one window covering the whole track [(0, a_total)]", () => {
      // chain_math: seg_audio=[268], a_total=268.
      expect(pairs(audioSegmentWindows([257], 24, 3))).toEqual([[0, 268]]);
      expect(pairs(audioSegmentWindows([49], 24, 3))).toEqual([[0, 51]]);
    });

    it("two clips overlap by exactly ka: [257,257] @24fps kv=3 -> (0,268),(250,268)", () => {
      // Backend: seg_audio=[268,268], a_total=518, ka_list=[18] -> the second
      // window starts at 268-18=250 and ends at 250+268=518=a_total.
      const windows = audioSegmentWindows([257, 257], 24, 3);
      expect(pairs(windows)).toEqual([
        [0, 268],
        [250, 268],
      ]);
      const [first, second] = windows!;
      expect(first!.startLatent).toBe(0);
      expect(second!.startLatent + second!.lenLatent).toBe(518);
      // Overlap between consecutive windows == ka_list[0] == 18.
      expect(first!.lenLatent - second!.startLatent).toBe(18);
    });

    it("three clips distribute the remainder across joins (ka_list=[18,18])", () => {
      expect(pairs(audioSegmentWindows([257, 257, 257], 24, 3))).toEqual([
        [0, 268],
        [250, 268],
        [500, 268],
      ]);
    });

    it("matches the backend for uneven clips and non-default fps/kv", () => {
      // [257,129,65] @25fps kv=3: seg_audio=[257,129,65], a_total=417,
      // ka_list=[17,17].
      expect(pairs(audioSegmentWindows([257, 129, 65], 25, 3))).toEqual([
        [0, 257],
        [240, 129],
        [352, 65],
      ]);
      // [49,49,49] @24fps kv=1: a_total=151, ka_list=[1,1].
      expect(pairs(audioSegmentWindows([49, 49, 49], 24, 1))).toEqual([
        [0, 51],
        [50, 51],
        [100, 51],
      ]);
      // [121,121] @24fps kv=3: a_total=234, ka_list=[18].
      expect(pairs(audioSegmentWindows([121, 121], 24, 3))).toEqual([
        [0, 126],
        [108, 126],
      ]);
      // [257,257] @30fps kv=3: a_total=414, ka_list=[14].
      expect(pairs(audioSegmentWindows([257, 257], 30, 3))).toEqual([
        [0, 214],
        [200, 214],
      ]);
    });

    it("the last window always ends exactly at a_total (== audioLatentsRequired)", () => {
      for (const [clips, fps, kv] of [
        [[257, 257], 24, 3],
        [[257, 129, 65], 25, 3],
        [[49, 49, 49], 24, 1],
        [[121, 121], 24, 3],
      ] as const) {
        const windows = audioSegmentWindows([...clips], fps, kv)!;
        const last = windows[windows.length - 1]!;
        expect(last.startLatent + last.lenLatent).toBe(audioLatentsRequired([...clips], fps, kv));
      }
    });

    it("exposes the same span in seconds (latents / 25)", () => {
      const [first, second] = audioSegmentWindows([257, 257], 24, 3)!;
      expect(first!.startSec).toBeCloseTo(0, 10);
      expect(first!.endSec).toBeCloseTo(268 / 25, 10);
      expect(second!.startSec).toBeCloseTo(250 / 25, 10);
      expect(second!.endSec).toBeCloseTo(518 / 25, 10);
    });

    it("returns null (never throws) on the geometries the backend raises for", () => {
      expect(audioSegmentWindows([], 24, 3)).toBeNull();
      expect(audioSegmentWindows([17], 24, 3)).toBeNull(); // kv >= v_latent(17)=3
      // Degenerate audio overlap: [9,9] @24fps kv=1 -> sum_ka=0 < joins=1.
      expect(audioSegmentWindows([9, 9], 24, 1)).toBeNull();
    });
  });

  describe("chainLayoutError", () => {
    // Non-integer NTSC rates, spelled the way the form computes them.
    const FPS_23_976 = 24000 / 1001;
    const FPS_24 = 24;
    const FPS_25 = 25;
    const FPS_29_97 = 30000 / 1001;
    const FPS_30 = 30;
    const FPS_50 = 50;
    const FPS_60 = 60;

    // Ground truth: each row was produced by calling
    // `chain_math.compute_chain_layout(clips, fps, kv=kv, v_tile=…, v_adv=…)`
    // in the backend venv and recording whether/how it raised. A 6,000-row
    // randomised superset of this table (12 frame rates x kv 1-8 x both
    // windows x 1-24 clips) was checked against this mirror during development
    // and matched on every row; the rows kept here are the interesting ones.
    const CASES: Array<{
      clips: number[];
      fps: number;
      kv: number;
      window: "standard" | "high_resolution";
      expected: ChainLayoutError | null;
    }> = [
      { clips: [257, 257], fps: FPS_24, kv: 3, window: "standard", expected: null },
      { clips: [257, 257], fps: FPS_29_97, kv: 3, window: "standard", expected: "audioReassemblyMismatch" },
      { clips: [257, 257], fps: FPS_29_97, kv: 3, window: "high_resolution", expected: "audioReassemblyMismatch" },
      { clips: [257, 257, 257], fps: FPS_29_97, kv: 3, window: "standard", expected: "audioReassemblyMismatch" },
      { clips: [257, 257], fps: FPS_23_976, kv: 3, window: "standard", expected: null },
      { clips: [121, 121], fps: FPS_23_976, kv: 3, window: "standard", expected: "audioReassemblyMismatch" },
      { clips: [481], fps: FPS_23_976, kv: 3, window: "standard", expected: "audioReassemblyMismatch" },
      { clips: [481], fps: FPS_24, kv: 3, window: "standard", expected: null },
      { clips: [257, 257], fps: FPS_25, kv: 3, window: "standard", expected: null },
      { clips: [257, 257], fps: FPS_30, kv: 3, window: "standard", expected: null },
      { clips: [257, 257], fps: FPS_50, kv: 3, window: "standard", expected: null },
      { clips: [257, 257], fps: FPS_60, kv: 3, window: "standard", expected: null },
      { clips: [257, 257], fps: FPS_50, kv: 1, window: "standard", expected: "degenerateAudioOverlap" },
      { clips: [153, 153], fps: FPS_50, kv: 8, window: "standard", expected: "audioReassemblyMismatch" },
      { clips: [9, 9], fps: FPS_24, kv: 1, window: "standard", expected: "degenerateAudioOverlap" },
      { clips: [17], fps: FPS_24, kv: 3, window: "standard", expected: "kvTooLarge" },
      { clips: [25, 25], fps: FPS_24, kv: 3, window: "standard", expected: null },
      { clips: [201, 201], fps: FPS_24, kv: 1, window: "standard", expected: "degenerateAudioOverlap" },
      { clips: [201, 201], fps: FPS_29_97, kv: 1, window: "standard", expected: "audioReassemblyMismatch" },
      { clips: Array<number>(24).fill(481), fps: FPS_24, kv: 3, window: "standard", expected: null },
      { clips: Array<number>(24).fill(481), fps: FPS_30, kv: 8, window: "standard", expected: "audioReassemblyMismatch" },
      { clips: Array<number>(24).fill(481), fps: FPS_29_97, kv: 3, window: "standard", expected: "audioReassemblyMismatch" },
      { clips: [257, 129, 65], fps: FPS_25, kv: 3, window: "standard", expected: null },
      { clips: [169, 169], fps: FPS_23_976, kv: 3, window: "standard", expected: "audioReassemblyMismatch" },
      { clips: [169, 169], fps: FPS_24, kv: 3, window: "high_resolution", expected: null },
      { clips: [97, 97], fps: FPS_23_976, kv: 3, window: "high_resolution", expected: "audioReassemblyMismatch" },
      { clips: [65, 65, 65, 65], fps: FPS_30, kv: 1, window: "standard", expected: "degenerateAudioOverlap" },
      { clips: [33, 33, 33], fps: FPS_24, kv: 1, window: "high_resolution", expected: "degenerateAudioOverlap" },
    ];

    it.each(CASES)(
      "$clips @ $fps fps, kv=$kv, $window -> $expected",
      ({ clips, fps, kv, window, expected }) => {
        expect(chainLayoutError(clips, fps, kv, STAGE2_WINDOW_PRESETS[window])).toBe(expected);
      },
    );

    it("29.97fps is the headline case: the SAME clip list is fine at 24fps and a 422 at 29.97", () => {
      const standard = STAGE2_WINDOW_PRESETS.standard;
      expect(chainLayoutError([257, 257], 24, 3, standard)).toBeNull();
      // Backend message: "audio reassembly 414 != a_total 415".
      expect(chainLayoutError([257, 257], 30000 / 1001, 3, standard)).toBe("audioReassemblyMismatch");
    });

    it("rejects an empty clip list rather than throwing", () => {
      expect(chainLayoutError([], 24, 3, STAGE2_WINDOW_PRESETS.standard)).toBe("noClips");
    });

    it("agrees with audioLatentsRequired's own null on the kv guard", () => {
      expect(chainLayoutError([17], 24, 3, STAGE2_WINDOW_PRESETS.standard)).toBe("kvTooLarge");
      expect(audioLatentsRequired([17], 24, 3)).toBeNull();
    });

    it("falls back to 24fps when fps is falsy, like every other A2V helper", () => {
      const standard = STAGE2_WINDOW_PRESETS.standard;
      expect(chainLayoutError([257, 257], 0, 3, standard)).toBe(chainLayoutError([257, 257], 24, 3, standard));
    });
  });

  // ── 素材（末尾）end source ──────────────────────────────
  // v2 (2026-08-15) deleted the whole v1 block that stood here —
  // `snapEndContextFrames` / `isEndContextFramesValid` /
  // `endContextRequiredLastClipFrames` / `endContextFitsChain`, together with
  // the ground-truth fit table. Every one of those tests asserted a REJECTION
  // the server no longer performs (the band is its own internal segment now, so
  // it never competes with the last clip or the last stage-2 tile), so keeping
  // them — even inverted — would pin behaviour that has no counterpart on the
  // backend. The arithmetic that replaced them is tested in
  // `timeline/tailAlign.test.ts`, and its effect on the form in
  // `useChainForm.test.ts`'s end-source block.

  describe("computeOutputFrames still knows nothing about an end source", () => {
    it("has no end-source parameter, and needs none — the band lives INSIDE the clip it belongs to", () => {
      // 窓内モード (2026-08-17) / 逆順Chained's reverse mode (2026-08-18) both
      // freeze the band onto the anchor CLIP's own tail rather than appending a
      // segment, so this function's arithmetic is byte-identical whether or not
      // `end_source` is attached — for ANY clip count, not just one.
      // `useChainForm.outputFrames` therefore adds nothing on top of this call.
      // v_latent(257) = 33 each; f_total = 66 - 3 = 63; total_px = 62*8+1 = 497.
      expect(computeOutputFrames([257, 257], 3, null)).toBe(497);
      expect(computeOutputFrames.length).toBe(3);
    });

    it("stays the same 497 for a 3-clip reverse-mode chain too (kv doesn't change the identity)", () => {
      // v_latent(257)=33 x3; f_total = 99 - 2*3 = 93; total_px = 92*8+1 = 737.
      expect(computeOutputFrames([257, 257, 257], 3, null)).toBe(737);
      // Same clips at the reverse-mode default kv=1: f_total = 99-2 = 97;
      // total_px = 96*8+1 = 769. Neither call knows an end source is attached.
      expect(computeOutputFrames([257, 257, 257], 1, null)).toBe(769);
    });
  });

  // ── 素材（末尾）逆順Chained (2026-08-18, second stage): the two acceptance
  // checks the server ONLY runs in `reverse` mode (2+ clips) — mirrors of
  // `chain_math.py`'s own guards, added for `useChainForm`'s `endSourceNeedsOverlap`
  // clips.length===1 exemption (2.6(f)).
  describe("vTailLatents", () => {
    it("is plain floor division by 8, NOT vLatentFrames' +1", () => {
      expect(vTailLatents(8)).toBe(1);
      expect(vTailLatents(16)).toBe(2);
      expect(vTailLatents(24)).toBe(3);
    });

    it("floors a non-multiple of 8 down", () => {
      expect(vTailLatents(15)).toBe(1);
      expect(vTailLatents(7)).toBe(0);
      expect(vTailLatents(0)).toBe(0);
    });
  });

  describe("endSourceLastClipHasFreeLatents (mirrors chain_math.py:1234: kv + n_end_v >= clip_latent[-1] rejects)", () => {
    it("blocks exactly at the boundary and passes one latent frame above it", () => {
      // v_latent(25) = (25-1)/8 + 1 = 4. kv(3) + nEndV(1) = 4 -> NOT strictly
      // less than 4, so this clip has nothing left to generate.
      expect(endSourceLastClipHasFreeLatents(25, 3, 1)).toBe(false);
      // v_latent(33) = (33-1)/8 + 1 = 5. 4 < 5 -> one free latent remains.
      expect(endSourceLastClipHasFreeLatents(33, 3, 1)).toBe(true);
    });

    it("is the §2.2 worked example (169f, kv=1, 8f anchor -> nEndV=1)", () => {
      // v_latent(169) = 22. kv(1) + nEndV(1) = 2 < 22.
      expect(endSourceLastClipHasFreeLatents(169, 1, vTailLatents(8))).toBe(true);
    });
  });

  describe("endSourceAudioOverlapOk (mirrors chain_math.py:1278-1280's sum_ka >= n_join guard, freed of chainLayoutError's audioReady precondition)", () => {
    it("is the plan's own headline failure: 30fps x 257f x 2 clips at kv=1", () => {
      // segLatent=[33,33]; segAudio=[214,214] (round(257/30*25)=214.1667->214);
      // fTotal=66-1=65; totalPx=513; aTotal=round(513/30*25)=427.5->428 (even);
      // sumKa=214+214-428=0 < nJoin=1 -> degenerate.
      expect(endSourceAudioOverlapOk([257, 257], 30, 1)).toBe(false);
    });

    it("the SAME clips clear the budget at kv=3", () => {
      // fTotal=66-3=63; totalPx=497; aTotal=round(497/30*25)=414.1667->414;
      // sumKa=214+214-414=14 >= nJoin=1.
      expect(endSourceAudioOverlapOk([257, 257], 30, 3)).toBe(true);
    });

    it("is true (not this gate's job) for a single clip — nothing to join", () => {
      expect(endSourceAudioOverlapOk([257], 30, 1)).toBe(true);
      expect(endSourceAudioOverlapOk([], 30, 1)).toBe(true);
    });

    it("is true (not this gate's job) when kv is too large for a clip — clipTooShortForOverlap's own failure", () => {
      // v_latent(9) = 2; kv=8 >= 2, the OTHER rejection this function stays
      // silent about.
      expect(endSourceAudioOverlapOk([9, 9], 30, 8)).toBe(true);
    });
  });
});
