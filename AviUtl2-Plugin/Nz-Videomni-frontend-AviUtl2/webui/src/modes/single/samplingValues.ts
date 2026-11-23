import type { SamplingConstraints } from "../../api/types";
import type { AccelerationSettings } from "../../shell/accelerationSettings";
import type { GenerationConstraints } from "../../shell/generationConstraints";
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

/** The sampling values the Settings panel's pi-H3 choice fixes (ruling 107),
 * read off the server's `constraints.pi_h3` (flattened as
 * `GenerationConstraints.piH3`), or `null` when nothing is fixed: pi-H3 off,
 * or the loaded base model publishes no pi-H3 values (LTX, fallback). The
 * sliders show this value and are locked while it is non-null. The SAME
 * condition that decides "locked" decides "not sent" in
 * {@link samplingRequestFields} — the choice other than off. */
export function piH3FixedSampling(
  piH3: AccelerationSettings["piH3"],
  table: GenerationConstraints["piH3"],
): SamplingValues | null {
  if (piH3 === "off" || !table) return null;
  return table[piH3] ?? null;
}

/** The request keys for the sampling values: all three while `values` is set
 * (the loaded base model publishes `constraints.sampling` — one rule, the
 * server would write its defaults back anyway), none otherwise.
 *
 * Ruling 107: none either while the Settings panel's pi-H3 choice is not off —
 * the server fills the three with the choice's values then (and refuses an
 * explicit different one). `acceleration` is a REQUIRED argument (the value
 * may be `undefined`, read as everything at its default, like
 * `accelerationRequestFields`) so a submission path that forgets it fails
 * `typecheck`. */
export function samplingRequestFields(
  values: SamplingValues | null,
  acceleration: AccelerationSettings | undefined,
): { num_inference_steps?: number; shift_video?: number; shift_audio?: number } {
  if (!values) return {};
  if (acceleration !== undefined && acceleration.piH3 !== "off") return {};
  return {
    num_inference_steps: values.numInferenceSteps,
    shift_video: values.shiftVideo,
    shift_audio: values.shiftAudio,
  };
}
