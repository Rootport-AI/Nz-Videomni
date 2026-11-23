import { useEffect, useRef } from "react";
import type { PiH3Mode } from "../../shell/accelerationSettings";

/**
 * Ruling 107 ② (2026-11-23): "off にしたら既定値に戻す". Calls `reset` whenever
 * the Settings panel's pi-H3 choice differs from the one last seen — never on
 * the first render (the form's own initializer already seeded its values).
 *
 * Shared by the five forms that own sampling sliders (Create, Chained,
 * Retake, Outpainting, Inpainting), each calling it the same way with a
 * `reset` that puts ONLY the step count and the two shifts back to the served
 * defaults. Deliberately separate from the forms' "the base model's rules
 * changed" effects, which also reset other values (the audio Strength etc.).
 *
 * `reset` need not be stable: the ref guard keeps a re-created function from
 * calling it again while the choice stays put.
 */
export function usePiH3SamplingReset(piH3: PiH3Mode, reset: () => void): void {
  const appliedRef = useRef(piH3);
  useEffect(() => {
    if (appliedRef.current === piH3) return;
    appliedRef.current = piH3;
    reset();
  }, [piH3, reset]);
}
