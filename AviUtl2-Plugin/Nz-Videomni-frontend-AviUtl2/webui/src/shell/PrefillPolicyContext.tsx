import { createContext, useCallback, useContext, useMemo, useState } from "react";
import type { ReactNode } from "react";

/**
 * W1 (Settings size/fps split): how a right-click prefill decides the
 * generation width/height and, separately, the fps. The two axes are chosen
 * independently now — a `sizePolicy` for width/height and an `fpsPolicy` for the
 * frame rate — each taking one of the same three values:
 *  - `"material"`: the selected material's real resolution (falling back to the
 *    project defaults) / the selected object's own probed framerate (falling back
 *    to the project's rate/scale).
 *  - `"project"`: the AviUtl2 project's own resolution/fps, read via
 *    `timeline.getEditInfo` once on the prefilling remount.
 *  - `"defaults"` (label "Dev"): always the backend
 *    `config.generation_defaults` (width/height / frame_rate), ignoring the
 *    material's real size/rate.
 *
 * §3-13 (2026-09-01): BOTH axes offer all three choices. The fps axis's
 * `"material"` was retired by X1 while native had no way to read a material's
 * real fps; contract v11 gives it one (`mediaFps`, probed via Media Foundation)
 * and `timeline/prefillSeed.ts` snaps that raw rate to an integer, so the choice
 * is live again and the X1 coercions that forced a stored/incoming `"material"`
 * back to `"project"` are gone — this context now stores and restores all three
 * values on either axis.
 *
 * This choice affects ONLY the right-click prefill's initial value decision —
 * ordinary panel edits, presets and the "Get size from AviUtl2" button are
 * unchanged.
 */
export type PrefillResolutionPolicy = "defaults" | "project" | "material";

/** `localStorage` keys the active size/fps prefill policies are persisted
 * under (W1: split from the former single `nzvideomni.prefillResolutionPolicy`,
 * which is deliberately no longer read). Mirrors `shell/ThemeContext.tsx`'s
 * `THEME_STORAGE_KEY` naming (owner decision: localStorage-only, no
 * native/settings.json involvement). */
export const PREFILL_SIZE_POLICY_STORAGE_KEY = "nzvideomni.prefillSizePolicy";
export const PREFILL_FPS_POLICY_STORAGE_KEY = "nzvideomni.prefillFpsPolicy";

export const DEFAULT_PREFILL_SIZE_POLICY: PrefillResolutionPolicy = "material";
export const DEFAULT_PREFILL_FPS_POLICY: PrefillResolutionPolicy = "project";

function isPolicy(value: string | null): value is PrefillResolutionPolicy {
  return value === "defaults" || value === "project" || value === "material";
}

/** Reads a persisted prefill-policy choice for the given key, falling back to
 * `fallback` on an unrecognized/absent value or a `localStorage` failure.
 * Wrapped in a `try` since `localStorage` can throw in some restricted
 * embeddings (e.g. a WebView2 host with storage disabled) — falling back to the
 * default is preferable to a crash. Mirrors `ThemeContext`'s `readStoredTheme`. */
function readStoredPolicy(key: string, fallback: PrefillResolutionPolicy): PrefillResolutionPolicy {
  try {
    const stored = window.localStorage.getItem(key);
    if (isPolicy(stored)) return stored;
  } catch {
    // Ignore — localStorage unavailable.
  }
  return fallback;
}

export function readStoredSizePolicy(): PrefillResolutionPolicy {
  return readStoredPolicy(PREFILL_SIZE_POLICY_STORAGE_KEY, DEFAULT_PREFILL_SIZE_POLICY);
}

export function readStoredFpsPolicy(): PrefillResolutionPolicy {
  return readStoredPolicy(PREFILL_FPS_POLICY_STORAGE_KEY, DEFAULT_PREFILL_FPS_POLICY);
}

function writeStoredPolicy(key: string, policy: PrefillResolutionPolicy): void {
  try {
    window.localStorage.setItem(key, policy);
  } catch {
    // Ignore — localStorage unavailable (e.g. private mode).
  }
}

export interface PrefillPolicyContextValue {
  /** W1: the size (width/height) prefill policy. */
  sizePolicy: PrefillResolutionPolicy;
  setSizePolicy: (policy: PrefillResolutionPolicy) => void;
  /** W1: the fps (frame rate) prefill policy. */
  fpsPolicy: PrefillResolutionPolicy;
  setFpsPolicy: (policy: PrefillResolutionPolicy) => void;
}

const PrefillPolicyContext = createContext<PrefillPolicyContextValue | null>(null);

/** Wraps the app (see `shell/AppShell.tsx`) so the Create/Chain screens and the
 * settings panel can read/change the right-click prefill size/fps policies
 * without prop drilling. Structurally identical to `ThemeContext`'s
 * `ThemeProvider`. */
export function PrefillPolicyProvider({ children }: { children: ReactNode }) {
  const [sizePolicy, setSizePolicyState] = useState<PrefillResolutionPolicy>(() => readStoredSizePolicy());
  const [fpsPolicy, setFpsPolicyState] = useState<PrefillResolutionPolicy>(() => readStoredFpsPolicy());

  const setSizePolicy = useCallback((next: PrefillResolutionPolicy) => {
    setSizePolicyState(next);
    writeStoredPolicy(PREFILL_SIZE_POLICY_STORAGE_KEY, next);
  }, []);
  const setFpsPolicy = useCallback((next: PrefillResolutionPolicy) => {
    setFpsPolicyState(next);
    writeStoredPolicy(PREFILL_FPS_POLICY_STORAGE_KEY, next);
  }, []);

  const value = useMemo<PrefillPolicyContextValue>(
    () => ({ sizePolicy, setSizePolicy, fpsPolicy, setFpsPolicy }),
    [sizePolicy, setSizePolicy, fpsPolicy, setFpsPolicy],
  );

  return <PrefillPolicyContext.Provider value={value}>{children}</PrefillPolicyContext.Provider>;
}

export function usePrefillPolicy(): PrefillPolicyContextValue {
  const ctx = useContext(PrefillPolicyContext);
  if (!ctx) throw new Error("usePrefillPolicy must be used within a PrefillPolicyProvider");
  return ctx;
}
