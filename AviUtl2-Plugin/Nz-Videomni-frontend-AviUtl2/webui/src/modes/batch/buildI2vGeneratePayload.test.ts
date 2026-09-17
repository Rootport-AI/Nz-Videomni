import { act, renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { ConditioningImage, CropOutput, LoraSpec } from "../../api/types";
import { parseLoraPrompt } from "../../lora/loraTags";
import { ACCELERATION_DEFAULTS } from "../../shell/accelerationSettings";
import type { AccelerationSettings } from "../../shell/accelerationSettings";
import type { NagSettings } from "../../shell/nagSettings";
import { FALLBACK_APP_CONFIG } from "../single/defaultConfig";
import { useGenerationForm } from "../single/useGenerationForm";
import { buildI2vGeneratePayload } from "./buildI2vGeneratePayload";
import type { BuildI2vGeneratePayloadParams } from "./buildI2vGeneratePayload";

// Deliberately NOT `FALLBACK_APP_CONFIG.generation_defaults`: the byte-equality
// block below drives `useGenerationForm` to exactly these values, so a future
// change to the Create defaults can neither break the comparison nor make it
// pass for the wrong reason. 768/512 are on the 64 grid and 121 is 8n+1, so the
// hook's initializer stores them verbatim.
const WIDTH = 768;
const HEIGHT = 512;
const NUM_FRAMES = 121;
const FRAME_RATE = 30;
const SEED = 42;

const BASE: BuildI2vGeneratePayloadParams = {
  prompt: "a cat riding a skateboard",
  width: WIDTH,
  height: HEIGHT,
  cropOutput: null,
  numFrames: NUM_FRAMES,
  frameRate: FRAME_RATE,
  seed: SEED,
};

const ENABLED_NAG: NagSettings = {
  text: "blurry, low quality, distorted, watermark, text",
  enabled: true,
  scale: 11.0,
  tau: 2.5,
  alpha: 0.25,
  method: "nag",
  vsfScale: 1.5,
};

const SAGE: AccelerationSettings = { ...ACCELERATION_DEFAULTS, attentionBackend: "sage" };

/** §3-153: WIDTH×HEIGHT (768×512) に対して妥当なクロップ。`setCropOutput` は
 * 与えられた値をそのまま保持する（件D 自由入力化 2026-09-16）ので、この定数が
 * 範囲内であること自体が fixture の番人。 */
const CROP: CropOutput = { width: 640, height: 384 };

const KEYFRAME: ConditioningImage = { image_id: "img-1", frame_idx: 0, strength: 0.8 };
const LORAS: LoraSpec[] = [{ name: "Pixar_Toon", strength: 1.5 }];

describe("buildI2vGeneratePayload", () => {
  it("最小構成: 6欄をこの順で送り、チェーン専用欄は送らない。クロップOFFならcrop_outputも送らない", () => {
    const payload = buildI2vGeneratePayload(BASE);
    expect(Object.keys(payload)).toEqual(["prompt", "width", "height", "num_frames", "frame_rate", "seed"]);
    expect(payload).toEqual({
      prompt: "a cat riding a skateboard",
      width: WIDTH,
      height: HEIGHT,
      num_frames: NUM_FRAMES,
      frame_rate: FRAME_RATE,
      seed: SEED,
    });
    // §3-153 (2026-09-16): `crop_output` は「決して送らない欄」ではなくなった
    // ので、恒久的な禁止キー一覧から外して条件つきの独立断言にする。ここは
    // BASE がクロップOFF（`cropOutput: null`）だから無い、という意味。
    expect(payload).not.toHaveProperty("crop_output");
    for (const banned of [
      "clips",
      "source_audio",
      "stage2_window",
      "chunked_upsample",
      "pipeline",
      "num_inference_steps",
      "guidance_scale",
      "overlap_frames",
      "overlap_strength",
    ]) {
      expect(payload).not.toHaveProperty(banned);
    }
  });

  it("クロップON: crop_outputがheightの直後に1欄だけ増える（toGenerateRequestと同じ位置）", () => {
    const payload = buildI2vGeneratePayload({ ...BASE, cropOutput: CROP });
    expect(Object.keys(payload)).toEqual([
      "prompt",
      "width",
      "height",
      "crop_output",
      "num_frames",
      "frame_rate",
      "seed",
    ]);
    expect(payload.crop_output).toEqual(CROP);
  });

  it("クロップOFF(null)は1欄も足さず、キー順も変わらない", () => {
    const baseline = buildI2vGeneratePayload(BASE);
    const off = buildI2vGeneratePayload({ ...BASE, cropOutput: null });
    expect(JSON.stringify(off)).toBe(JSON.stringify(baseline));
    expect(off).not.toHaveProperty("crop_output");
  });

  it("seedは切り捨てない（toGenerateRequestと同じ。a2vビルダーのMath.truncとは違う）", () => {
    expect(buildI2vGeneratePayload({ ...BASE, seed: 12.7 }).seed).toBe(12.7);
  });

  it("NAG off（未指定・無効の両方）は1欄も足さず、キー順も変わらない", () => {
    const baseline = buildI2vGeneratePayload(BASE);
    const disabled = buildI2vGeneratePayload({ ...BASE, nag: { ...ENABLED_NAG, enabled: false } });
    expect(JSON.stringify(disabled)).toBe(JSON.stringify(baseline));
    expect(disabled).not.toHaveProperty("negative_prompt");
    expect(disabled).not.toHaveProperty("nag_enabled");
  });

  it("NAG on: 加算7欄がseedの直後に並ぶ", () => {
    const payload = buildI2vGeneratePayload({ ...BASE, nag: ENABLED_NAG });
    expect(Object.keys(payload)).toEqual([
      "prompt",
      "width",
      "height",
      "num_frames",
      "frame_rate",
      "seed",
      "negative_prompt",
      "nag_enabled",
      "nag_scale",
      "nag_tau",
      "nag_alpha",
      "neg_method",
      "vsf_scale",
    ]);
    expect(payload.negative_prompt).toBe(ENABLED_NAG.text);
  });

  it("acceleration既定は1欄も足さず、非既定(sage)はちょうど1欄増える", () => {
    const baseline = buildI2vGeneratePayload(BASE);
    const withDefaults = buildI2vGeneratePayload({ ...BASE, acceleration: ACCELERATION_DEFAULTS });
    expect(JSON.stringify(withDefaults)).toBe(JSON.stringify(baseline));

    const sage = buildI2vGeneratePayload({ ...BASE, acceleration: SAGE });
    expect(sage.attention_backend).toBe("sage");
    expect(Object.keys(sage)).toEqual([...Object.keys(baseline), "attention_backend"]);
  });

  it("loras: 空配列/未指定ならキーごと省略、1件以上なら末尾寄りに1欄", () => {
    expect(buildI2vGeneratePayload(BASE)).not.toHaveProperty("loras");
    expect(buildI2vGeneratePayload({ ...BASE, loras: [] })).not.toHaveProperty("loras");
    const withLoras = buildI2vGeneratePayload({ ...BASE, loras: LORAS });
    expect(withLoras.loras).toEqual(LORAS);
    expect(Object.keys(withLoras)).toEqual([...Object.keys(buildI2vGeneratePayload(BASE)), "loras"]);
  });

  it("conditioning_images: 空配列/未指定ならキーごと省略、1件以上なら最後のキー", () => {
    expect(buildI2vGeneratePayload(BASE)).not.toHaveProperty("conditioning_images");
    expect(buildI2vGeneratePayload({ ...BASE, conditioningImages: [] })).not.toHaveProperty("conditioning_images");
    const withImage = buildI2vGeneratePayload({
      ...BASE,
      acceleration: SAGE,
      loras: LORAS,
      conditioningImages: [KEYFRAME],
    });
    expect(withImage.conditioning_images).toEqual([KEYFRAME]);
    expect(Object.keys(withImage).at(-1)).toBe("conditioning_images");
  });

  // D11: 共通欄はCreate画面の単発i2vとバイト等価でなければならない。参照は
  // `useGenerationForm.toGenerateRequest()` ＋ `SingleScreen`のconditioning付与
  // （参照動画無し・制御LoRA無しが前提。§3-153 でクロップは等価対象に入った）。
  describe("Single（toGenerateRequest）とのバイト等価", () => {
    function singleRequest(
      prompt: string,
      deps: { nag?: NagSettings; acceleration?: AccelerationSettings } = {},
      conditioningImages: ConditioningImage[] = [],
      cropOutput: CropOutput | null = null,
    ): string {
      // width/height/numFrames/frameRate are seeded through
      // `GenerationFormInitial`; `seed` has no initial field, so it is driven
      // through the hook's own `setSeed`.
      const { result } = renderHook(() =>
        useGenerationForm(FALLBACK_APP_CONFIG, prompt, deps, {
          width: WIDTH,
          height: HEIGHT,
          numFrames: NUM_FRAMES,
          frameRate: FRAME_RATE,
        }),
      );
      act(() => {
        result.current.setSeed(SEED);
      });
      if (cropOutput !== null) {
        act(() => {
          result.current.setCropOutput(cropOutput);
        });
        // `setCropOutput` stores the value verbatim (free entry, 2026-09-16),
        // so the comparison is only meaningful once the hook really holds it —
        // this line is what guards the fixture against drifting out of range.
        expect(result.current.cropOutput).toEqual(cropOutput);
      }
      // The comparison is only meaningful if the hook really holds the values
      // this test compares against.
      expect(result.current.values).toMatchObject({
        width: WIDTH,
        height: HEIGHT,
        numFrames: NUM_FRAMES,
        frameRate: FRAME_RATE,
        seed: SEED,
      });
      const request = result.current.toGenerateRequest();
      return JSON.stringify(
        conditioningImages.length > 0 ? { ...request, conditioning_images: conditioningImages } : request,
      );
    }

    /** `batchRunner.processRow`と同じ前処理: 行を合成した文字列から`<lora:>`
     * タグを切り出し、残りの本文だけをビルダーへ渡す（ここでは行が空なので、
     * 合成結果＝共通プロンプトそのもの）。 */
    function batchRequest(
      prompt: string,
      extra: Partial<BuildI2vGeneratePayloadParams> = {},
      conditioningImages: ConditioningImage[] = [],
    ): string {
      const { strippedPrompt, loras } = parseLoraPrompt(prompt);
      return JSON.stringify(
        buildI2vGeneratePayload({
          ...BASE,
          prompt: strippedPrompt,
          ...(loras.length > 0 ? { loras } : {}),
          ...extra,
          ...(conditioningImages.length > 0 ? { conditioningImages } : {}),
        }),
      );
    }

    it("素のt2v要求（Createの既定値そのまま）", () => {
      const prompt = "a cat riding a skateboard";
      expect(batchRequest(prompt)).toBe(singleRequest(prompt));
    });

    it("conditioning_images付き（i2v）", () => {
      const prompt = "a cat riding a skateboard";
      expect(batchRequest(prompt, {}, [KEYFRAME])).toBe(singleRequest(prompt, {}, [KEYFRAME]));
    });

    it("NAG on", () => {
      const prompt = "a cat riding a skateboard";
      expect(batchRequest(prompt, { nag: ENABLED_NAG })).toBe(singleRequest(prompt, { nag: ENABLED_NAG }));
    });

    it("acceleration非既定(sage)", () => {
      const prompt = "a cat riding a skateboard";
      expect(batchRequest(prompt, { acceleration: SAGE })).toBe(singleRequest(prompt, { acceleration: SAGE }));
    });

    it("<lora:>タグ入りプロンプト（本文の切り出しとloras欄の位置）", () => {
      const prompt = "a cat <lora:Pixar_Toon:1.5> riding a skateboard";
      expect(batchRequest(prompt)).toBe(singleRequest(prompt));
    });

    it("出力クロップON（§3-153: Createの継承値がSingleと同じ位置・同じ値で載る）", () => {
      const prompt = "a cat riding a skateboard";
      expect(batchRequest(prompt, { cropOutput: CROP })).toBe(singleRequest(prompt, {}, [], CROP));
    });

    it("全部乗せ（NAG on・sage・LoRA・conditioning_images）", () => {
      const prompt = "a cat <lora:Pixar_Toon:1.5> riding a skateboard";
      const deps = { nag: ENABLED_NAG, acceleration: SAGE };
      expect(batchRequest(prompt, deps, [KEYFRAME])).toBe(singleRequest(prompt, deps, [KEYFRAME]));
    });
  });
});
