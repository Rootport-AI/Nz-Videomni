import type { SamplingConstraints } from "../../api/types";
import { clamp } from "./paramUtils";

/**
 * The three sampling sliders' values and rules — step count, Shift (video),
 * Shift (audio) — shared by Create (`useGenerationForm`) and Chained
 * (`useChainForm`, §1-106). Moved here from `useGenerationForm.ts` (behaviour unchanged) so
 * both forms seed, clamp and send them by the same rules; the ranges and
 * defaults are the server's (`constraints.sampling`).
 */

/** The sampling sliders' current values. */
export interface SamplingValues {
  numInferenceSteps: number;
  shiftVideo: number;
  shiftAudio: number;
}

/** The server defaults of a `constraints.sampling` block, or `null` for an
 * engine without one. */
export function samplingDefaults(sampling: SamplingConstraints | null): SamplingValues | null {
  if (!sampling) return null;
  return {
    numInferenceSteps: sampling.num_inference_steps.default,
    shiftVideo: sampling.shift_video.default,
    shiftAudio: sampling.shift_audio.default,
  };
}

/** One sampling slider's value clamped into the served range, or `null` when
 * there is nothing to set (no `constraints.sampling`, or a non-finite input).
 * The step count is a whole number on the wire. */
export function clampSamplingValue(
  sampling: SamplingConstraints | null,
  key: keyof SamplingValues,
  raw: number,
): number | null {
  if (!sampling || !Number.isFinite(raw)) return null;
  const range =
    key === "numInferenceSteps"
      ? sampling.num_inference_steps
      : key === "shiftVideo"
        ? sampling.shift_video
        : sampling.shift_audio;
  const clamped = clamp(raw, range.min, range.max);
  return key === "numInferenceSteps" ? Math.round(clamped) : clamped;
}

/** The request keys for the sampling values: all three while `values` is set
 * (the loaded base model publishes `constraints.sampling` — one rule, the
 * server would write its defaults back anyway), none otherwise. */
export function samplingRequestFields(
  values: SamplingValues | null,
): { num_inference_steps?: number; shift_video?: number; shift_audio?: number } {
  if (!values) return {};
  return {
    num_inference_steps: values.numInferenceSteps,
    shift_video: values.shiftVideo,
    shift_audio: values.shiftAudio,
  };
}
