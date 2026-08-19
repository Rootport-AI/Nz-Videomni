import { describe, expect, it } from "vitest";
import type { ConditioningImage, LoraSpec } from "../../api/types";
import type { NagSettings } from "../../shell/nagSettings";
import { ACCELERATION_DEFAULTS } from "../../shell/accelerationSettings";
import { buildA2vChainPayload, composeRowPrompt, resolveConditioningImages, type BuildA2vChainPayloadParams } from "./buildA2vChainPayload";
import { IMAGE_SHARED } from "./manifestMerge";

describe("composeRowPrompt", () => {
  it("行プロンプトが空(空白のみ含む)なら常に共通プロンプトのみ（modeを問わない）", () => {
    expect(composeRowPrompt("common", "", "add")).toBe("common");
    expect(composeRowPrompt("common", "   ", "replace")).toBe("common");
  });

  it("add: 共通プロンプトの末尾に行プロンプトを追記する", () => {
    expect(composeRowPrompt("common", "row", "add")).toBe("common row");
    expect(composeRowPrompt("", "row", "add")).toBe("row");
  });

  it("replace: 行プロンプトのみを使う（共通プロンプトを丸ごと置換）", () => {
    expect(composeRowPrompt("common", "row", "replace")).toBe("row");
  });
});

describe("resolveConditioningImages", () => {
  const shared: ConditioningImage[] = [
    { image_id: "img-shared-1", frame_idx: 0, strength: 1.0 },
    { image_id: "img-shared-2", frame_idx: 40, strength: 0.7 },
  ];

  it("image=Shared: 共通キーフレームをそのまま使う", () => {
    const result = resolveConditioningImages({ image: IMAGE_SHARED, sharedImages: shared });
    expect(result).toBe(shared);
  });

  it("個別画像: frame_idx=0, strength=1.0の単一エントリになる", () => {
    const result = resolveConditioningImages({ image: "row.png", sharedImages: shared, rowImageId: "img-row-1" });
    expect(result).toEqual([{ image_id: "img-row-1", frame_idx: 0, strength: 1.0 }]);
  });

  it("個別画像だがimage_id未解決なら空配列", () => {
    const result = resolveConditioningImages({ image: "row.png", sharedImages: shared });
    expect(result).toEqual([]);
  });

  it("image=空文字は空配列（Shared正規化はparseManifest側の責務でここでは行わない）", () => {
    const result = resolveConditioningImages({ image: "", sharedImages: shared, rowImageId: "img-row-1" });
    expect(result).toEqual([]);
  });
});

describe("buildA2vChainPayload", () => {
  const base: BuildA2vChainPayloadParams = {
    audioId: "audio-1",
    numFrames: 121,
    prompt: "a cat talking",
    width: 768,
    height: 512,
    frameRate: 24,
    seed: 42,
    chunkedUpsample: true,
  };

  it("最小構成: 固定フィールド一式・crop_outputはnullでも常に出力される", () => {
    const payload = buildA2vChainPayload(base);
    expect(payload).toEqual({
      prompt: "a cat talking",
      negative_prompt: "",
      width: 768,
      height: 512,
      crop_output: null,
      frame_rate: 24,
      num_inference_steps: 8,
      guidance_scale: 1.0,
      seed: 42,
      pipeline: "distilled",
      overlap_frames: 3,
      overlap_strength: 0.5,
      clips: [{ num_frames: 121 }],
      source_audio: { audio_id: "audio-1" },
      chunked_upsample: true,
      stage2_window: "full_length",
    });
  });

  it("conditioning_images/loras/adapter系は非指定なら省略される", () => {
    const payload = buildA2vChainPayload(base);
    expect(payload).not.toHaveProperty("loras");
    expect(payload).not.toHaveProperty("reference_video_id");
    expect(payload).not.toHaveProperty("conditioning_attention_strength");
    expect(payload).not.toHaveProperty("reference_video_strength");
    expect(payload.clips[0]).not.toHaveProperty("conditioning_images");
  });

  it("conditioning_imagesが非空ならclipに載る", () => {
    const images: ConditioningImage[] = [{ image_id: "img-1", frame_idx: 0, strength: 1.0 }];
    const payload = buildA2vChainPayload({ ...base, conditioningImages: images });
    expect(payload.clips[0].conditioning_images).toEqual(images);
  });

  it("conditioning_imagesが空配列なら省略される", () => {
    const payload = buildA2vChainPayload({ ...base, conditioningImages: [] });
    expect(payload.clips[0]).not.toHaveProperty("conditioning_images");
  });

  it("lorasが非空ならlorasキーが付く", () => {
    const loras: LoraSpec[] = [{ name: "style-a", strength: 0.8 }];
    const payload = buildA2vChainPayload({ ...base, loras });
    expect(payload.loras).toEqual(loras);
  });

  it("lorasのaudio_strengthはそのままリクエストボディに載る（射影なし）", () => {
    const loras: LoraSpec[] = [{ name: "X", strength: 0.8, audio_strength: 0.3 }];
    const payload = buildA2vChainPayload({ ...base, loras });
    expect(payload.loras).toEqual([{ name: "X", strength: 0.8, audio_strength: 0.3 }]);
  });

  it("useAdapter=trueならreference_video_idが付く。強度は1.0未満のときだけ付く", () => {
    const p1 = buildA2vChainPayload({ ...base, useAdapter: true, referenceVideoId: "vid-1" });
    expect(p1.reference_video_id).toBe("vid-1");
    expect(p1).not.toHaveProperty("conditioning_attention_strength");
    expect(p1).not.toHaveProperty("reference_video_strength");

    const p2 = buildA2vChainPayload({
      ...base,
      useAdapter: true,
      referenceVideoId: "vid-1",
      controlAdherence: 0.6,
      referenceStrength: 0.9,
    });
    expect(p2.conditioning_attention_strength).toBe(0.6);
    expect(p2.reference_video_strength).toBe(0.9);

    const p3 = buildA2vChainPayload({
      ...base,
      useAdapter: true,
      referenceVideoId: "vid-1",
      controlAdherence: 1.0,
      referenceStrength: 1.0,
    });
    expect(p3).not.toHaveProperty("conditioning_attention_strength");
    expect(p3).not.toHaveProperty("reference_video_strength");
  });

  it("useAdapter=falseならreference_video_id等は一切付かない（referenceVideoIdを渡していても）", () => {
    const payload = buildA2vChainPayload({ ...base, useAdapter: false, referenceVideoId: "vid-1" });
    expect(payload).not.toHaveProperty("reference_video_id");
  });

  it("negative_promptはnull/undefinedで空文字になる", () => {
    expect(buildA2vChainPayload({ ...base, negativePrompt: null }).negative_prompt).toBe("");
    expect(buildA2vChainPayload(base).negative_prompt).toBe("");
    expect(buildA2vChainPayload({ ...base, negativePrompt: "bad quality" }).negative_prompt).toBe("bad quality");
  });

  it("crop_outputを渡すとそのまま出力に反映される", () => {
    const payload = buildA2vChainPayload({ ...base, cropOutput: { width: 704, height: 480 } });
    expect(payload.crop_output).toEqual({ width: 704, height: 480 });
  });

  it("chunked_upsampleは常に明示的に送られる（true/false問わず省略しない）", () => {
    expect(buildA2vChainPayload({ ...base, chunkedUpsample: false }).chunked_upsample).toBe(false);
    expect(buildA2vChainPayload({ ...base, chunkedUpsample: true }).chunked_upsample).toBe(true);
  });

  // §1-19 (2026-08-11): stage2_window is a fixed literal, not a param — there
  // is no `BuildA2vChainPayloadParams` field that could change it, so this
  // pins the value/position rather than trying (and failing) to override it.
  it("stage2_windowは常に'full_length'で固定（chunkedUpsampleやwidth/height等を変えても不変）", () => {
    expect(buildA2vChainPayload(base).stage2_window).toBe("full_length");
    expect(buildA2vChainPayload({ ...base, chunkedUpsample: false }).stage2_window).toBe("full_length");
    expect(buildA2vChainPayload({ ...base, width: 1152, height: 1536 }).stage2_window).toBe("full_length");
  });

  it("stage2_windowはキー順でchunked_upsampleの直後・acceleration系より前に来る", () => {
    const payload = buildA2vChainPayload(base);
    const keys = Object.keys(payload);
    const idx = keys.indexOf("chunked_upsample");
    expect(keys[idx + 1]).toBe("stage2_window");
  });

  it("composeRowPrompt/resolveConditioningImagesと組み合わせた一気通貫例（Add合成・Shared画像）", () => {
    const prompt = composeRowPrompt("cinematic style", "waving hello", "add");
    const conditioning = resolveConditioningImages({
      image: IMAGE_SHARED,
      sharedImages: [{ image_id: "kf-1", frame_idx: 0, strength: 1.0 }],
    });
    const payload = buildA2vChainPayload({ ...base, prompt, conditioningImages: conditioning });
    expect(payload.prompt).toBe("cinematic style waving hello");
    expect(payload.clips[0].conditioning_images).toEqual([{ image_id: "kf-1", frame_idx: 0, strength: 1.0 }]);
  });

  it("composeRowPrompt/resolveConditioningImagesと組み合わせた一気通貫例（Replace合成・個別画像）", () => {
    const prompt = composeRowPrompt("cinematic style", "waving hello", "replace");
    const conditioning = resolveConditioningImages({
      image: "row01.png",
      sharedImages: [{ image_id: "kf-1", frame_idx: 0, strength: 1.0 }],
      rowImageId: "img-row01",
    });
    const payload = buildA2vChainPayload({ ...base, prompt, conditioningImages: conditioning });
    expect(payload.prompt).toBe("waving hello");
    expect(payload.clips[0].conditioning_images).toEqual([{ image_id: "img-row01", frame_idx: 0, strength: 1.0 }]);
  });

  // NAG (2026-07-28): the additive `nagRequestFields(params.nag)` spread,
  // placed immediately after `negative_prompt` in the payload literal.
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

    it("without nag: JSON.stringify is byte-identical to a payload literally built without any NAG fields (key order + absence)", () => {
      const payload = buildA2vChainPayload(base);
      const expected = {
        prompt: "a cat talking",
        negative_prompt: "",
        width: 768,
        height: 512,
        crop_output: null,
        frame_rate: 24,
        num_inference_steps: 8,
        guidance_scale: 1.0,
        seed: 42,
        pipeline: "distilled",
        overlap_frames: 3,
        overlap_strength: 0.5,
        clips: [{ num_frames: 121 }],
        source_audio: { audio_id: "audio-1" },
        chunked_upsample: true,
        stage2_window: "full_length",
      };
      expect(JSON.stringify(payload)).toBe(JSON.stringify(expected));
    });

    it("with disabled nag: also byte-identical (soft-disable is the same as omitted)", () => {
      const payload = buildA2vChainPayload({ ...base, nag: { ...ENABLED_NAG, enabled: false } });
      const withoutNag = buildA2vChainPayload(base);
      expect(JSON.stringify(payload)).toBe(JSON.stringify(withoutNag));
    });

    it("with enabled nag: negative_prompt is overridden by nag.text, and the 6 nag_*/neg_method/vsf_scale keys appear right after it", () => {
      const payload = buildA2vChainPayload({ ...base, negativePrompt: "old negative", nag: ENABLED_NAG });
      expect(payload.negative_prompt).toBe(ENABLED_NAG.text);
      expect(payload.nag_enabled).toBe(true);
      expect(payload.nag_scale).toBe(11.0);
      expect(payload.nag_tau).toBe(2.5);
      expect(payload.nag_alpha).toBe(0.25);
      expect(payload.neg_method).toBe("nag");
      expect(payload.vsf_scale).toBe(1.5);

      // Key-order proof: negative_prompt, then the 6 additive keys, in that order.
      const keys = Object.keys(payload);
      const idx = keys.indexOf("negative_prompt");
      expect(keys.slice(idx, idx + 7)).toEqual([
        "negative_prompt",
        "nag_enabled",
        "nag_scale",
        "nag_tau",
        "nag_alpha",
        "neg_method",
        "vsf_scale",
      ]);
    });

    it("with enabled nag and method 'vsf': neg_method/vsf_scale reflect VSF", () => {
      const payload = buildA2vChainPayload({ ...base, nag: { ...ENABLED_NAG, method: "vsf", vsfScale: 5 } });
      expect(payload.neg_method).toBe("vsf");
      expect(payload.vsf_scale).toBe(5);
    });
  });

  // Acceleration (2026-07-31): the additive
  // `accelerationRequestFields(params.acceleration)` spread, LAST in the
  // payload literal.
  describe("acceleration", () => {
    it("omitted / all-defaults: JSON.stringify is byte-identical to the pre-feature payload", () => {
      const baseline = buildA2vChainPayload(base);
      const withDefaults = buildA2vChainPayload({ ...base, acceleration: ACCELERATION_DEFAULTS });
      expect(JSON.stringify(withDefaults)).toBe(JSON.stringify(baseline));
      expect(withDefaults).not.toHaveProperty("attention_backend");
    });

    it("sage: exactly one key more, appended at the end", () => {
      const baseline = buildA2vChainPayload(base);
      const sage = buildA2vChainPayload({
        ...base,
        acceleration: { ...ACCELERATION_DEFAULTS, attentionBackend: "sage" },
      });
      expect(sage.attention_backend).toBe("sage");
      expect(Object.keys(sage)).toEqual([...Object.keys(baseline), "attention_backend"]);
    });

    it("block-swap prefetch off: exactly one key more, appended at the end", () => {
      // S4: default is now true, so OFF is what diverges.
      const baseline = buildA2vChainPayload(base);
      const prefetch = buildA2vChainPayload({
        ...base,
        acceleration: { ...ACCELERATION_DEFAULTS, blockSwapPrefetch: false },
      });
      expect(prefetch.block_swap_prefetch).toBe(false);
      expect(Object.keys(prefetch)).toEqual([...Object.keys(baseline), "block_swap_prefetch"]);
    });

    it("keep-resident on: exactly one key more, appended at the end", () => {
      // §48: server default is false, so ON is what diverges.
      const baseline = buildA2vChainPayload(base);
      const keepResident = buildA2vChainPayload({
        ...base,
        acceleration: { ...ACCELERATION_DEFAULTS, keepResident: true },
      });
      expect(keepResident.keep_resident).toBe(true);
      expect(Object.keys(keepResident)).toEqual([...Object.keys(baseline), "keep_resident"]);
    });

    it("fused GGUF dequant kernel off: exactly one key more, appended at the end (§1-11)", () => {
      // §51 (2026-08-04): server default flipped to true, so OFF is what
      // diverges.
      const baseline = buildA2vChainPayload(base);
      const fused = buildA2vChainPayload({
        ...base,
        acceleration: { ...ACCELERATION_DEFAULTS, fusedGgufDequantKernel: false },
      });
      expect(fused.fused_gguf_dequant_kernel).toBe(false);
      expect(Object.keys(fused)).toEqual([...Object.keys(baseline), "fused_gguf_dequant_kernel"]);
    });

    it("PrunaVAED: exactly one key more, appended at the end (§52)", () => {
      // §52 (2026-08-05): server default is "default", so PrunaVAED is what
      // diverges. The API VALUE stays `"prune_vaed"` even though the display
      // name is PrunaVAED.
      const baseline = buildA2vChainPayload(base);
      const pruned = buildA2vChainPayload({
        ...base,
        acceleration: { ...ACCELERATION_DEFAULTS, vaeMode: "prune_vaed" },
      });
      expect(pruned.vae_mode).toBe("prune_vaed");
      expect(Object.keys(pruned)).toEqual([...Object.keys(baseline), "vae_mode"]);
      // Still additive: nothing else rides along.
      expect(pruned).not.toHaveProperty("attention_backend");
    });

    it("leaves vae_mode off entirely at the server default", () => {
      const payload = buildA2vChainPayload({ ...base, acceleration: ACCELERATION_DEFAULTS });
      expect(payload).not.toHaveProperty("vae_mode");
      expect(payload).not.toHaveProperty("attention_backend");
    });

    it("stage2_window stays 'full_length' and stays right after chunked_upsample even with acceleration fields appended", () => {
      const payload = buildA2vChainPayload({
        ...base,
        acceleration: { ...ACCELERATION_DEFAULTS, attentionBackend: "sage", keepResident: true },
      });
      expect(payload.stage2_window).toBe("full_length");
      const keys = Object.keys(payload);
      const idx = keys.indexOf("chunked_upsample");
      expect(keys[idx + 1]).toBe("stage2_window");
    });
  });
});
