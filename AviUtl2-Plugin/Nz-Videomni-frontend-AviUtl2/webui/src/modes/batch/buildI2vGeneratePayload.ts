/**
 * Batch i2v — assembling the `POST /api/v1/generate` body for a single row
 * (D11, 2026-09-15). An i2v row is one image, one clip, no audio, so it goes
 * to the plain generate endpoint rather than `/generate/chain`: a chain with
 * no `source_audio`/`source_video`/`reference_video_id`/`retake`/`end_source`
 * needs at least 2 clips (`api/models.py`), and the `stage2_window:
 * "full_length"` every `buildA2vChainPayload` body carries requires a
 * `source_audio`. Both of those are 422s, so the a2v builder cannot be reused
 * for an audio-free row.
 *
 * A parallel builder rather than Create's own `toGenerateRequest()`: that one
 * is a `useCallback` closing over the LIVE Create form (the reference video,
 * the control LoRA), which would break the batch's "frozen at `start()`,
 * immune to later UI edits" contract, drag in fields an i2v row must not
 * send, and be called from `runtime.ts` — outside React — after the Create
 * screen may already have unmounted. The common fields are byte-equivalent to
 * `toGenerateRequest()` all the same (same key order, same omission rules),
 * which `buildI2vGeneratePayload.test.ts` pins directly against the hook.
 *
 * §3-153 出力クロップの継承 (2026-09-16, D7 撤回): `crop_output` IS sent now.
 * Reading the live form is what was forbidden — taking an already-frozen
 * value as an argument is not, so the crop arrives via `cropOutput` and lands
 * in the same position `toGenerateRequest` puts it (right after `height`).
 * OFF (`null`) omits the key entirely, which is exactly what Create's own
 * `/generate` body does, so the byte-equivalence contract is unaffected.
 *
 * `seed` is passed straight through, NOT `Math.trunc`-ed — matching
 * `useGenerationForm.ts`'s `toGenerateRequest` (whose `setSeed` already
 * truncates), and deliberately unlike `buildA2vChainPayload`, which truncates
 * because the Gradio function it mirrors does.
 */
import type { ConditioningImage, CropOutput, GenerateRequest, LoraSpec } from "../../api/types";
import { nagRequestFields } from "../../shell/nagSettings";
import type { NagSettings } from "../../shell/nagSettings";
import { accelerationRequestFields } from "../../shell/accelerationSettings";
import type { AccelerationSettings } from "../../shell/accelerationSettings";

export interface BuildI2vGeneratePayloadParams {
  /** Already composed (see `composeRowPrompt` in `./buildA2vChainPayload`) —
   * this function does no prompt composition of its own. */
  prompt: string;
  width: number;
  height: number;
  /** §3-153 出力クロップの継承 (2026-09-16): the Create screen's crop setting,
   * already frozen by `useBatchForm`'s `start()` — required, so a caller
   * cannot quietly drop it. `null` (OFF) omits `crop_output` entirely. */
  cropOutput: CropOutput | null;
  /** The row's frame count (the Create form's DURATION, frozen at `start()`).
   * Top level, since there are no `clips` here. */
  numFrames: number;
  frameRate: number;
  seed: number;
  /** The shared Negative Prompt accordion's settings — adds the additive set
   * of 7 only while enabled, exactly as in `toGenerateRequest`. */
  nag?: NagSettings;
  /** The Settings panel's shared acceleration choice — adds only the fields
   * the user has moved off their server defaults. */
  acceleration?: AccelerationSettings;
  loras?: LoraSpec[];
  /** Already resolved (see `resolveConditioningImages` in
   * `./buildA2vChainPayload`) — omitted or empty omits the key entirely, which
   * is what makes a row with neither its own image nor a Shared keyframe a
   * plain T2V request. */
  conditioningImages?: ConditioningImage[];
}

/**
 * Assembles one i2v row's `POST /api/v1/generate` body. Key order mirrors
 * `useGenerationForm.ts`'s `toGenerateRequest` exactly, with
 * `conditioning_images` appended last the way `SingleScreen.tsx` appends it.
 *
 * `crop_output` rides along only while the (already-frozen) crop is ON, in
 * `toGenerateRequest`'s own position — right after `height`.
 *
 * Deliberately absent: `source_audio`, `stage2_window`, `pipeline`,
 * `num_inference_steps`, `guidance_scale`, `overlap_frames`,
 * `overlap_strength`, `chunked_upsample` and `clips`.
 */
export function buildI2vGeneratePayload(params: BuildI2vGeneratePayloadParams): GenerateRequest {
  return {
    prompt: params.prompt,
    width: params.width,
    height: params.height,
    ...(params.cropOutput ? { crop_output: params.cropOutput } : {}),
    num_frames: params.numFrames,
    frame_rate: params.frameRate,
    seed: params.seed,
    ...nagRequestFields(params.nag),
    ...accelerationRequestFields(params.acceleration),
    ...(params.loras && params.loras.length > 0 ? { loras: params.loras } : {}),
    ...(params.conditioningImages && params.conditioningImages.length > 0
      ? { conditioning_images: params.conditioningImages }
      : {}),
  };
}
