import { createContext, useCallback, useContext, useMemo, useState } from "react";
import type { ReactNode } from "react";

/**
 * W1 (Settings size/fps split): how a right-click prefill decides the
 * generation width/height and, separately, the fps. The two axes are chosen
 * independently now — a `sizePolicy` for width/height and an `fpsPolicy` for the
 * frame rate — each taking one of the same three values:
 *  - `"material"`: the selected material's real resolution (falling back to the
 *    project defaults) / the selection's project rate/scale (falling back to the
 *    backend default).
 *  - `"project"`: the AviUtl2 project's own resolution/fps, read via
 *    `timeline.getEditInfo` once on the prefilling remount.
 *  - `"defaults"` (label "Dev"): always the backend
 *    `config.generation_defaults` (width/height / frame_rate), ignoring the
 *    material's real size/rate.
 *
 * X1: the FPS axis no longer offers `"material"` — the SDK cannot read a
 * material's real fps (deferred to a future update), so on the fps axis only
 * `"defaults"`/`"project"` are selectable and a persisted/incoming `"material"`
 * is coerced to the default `"project"` (see `readStoredFpsPolicy`/`setFpsPolicy`
 * below). The SIZE axis keeps all three choices.
 *
 * This choice affects ONLY the right-click prefill's initial value decision —
 * ordinary panel edits, presets and the "Get size from AviUtl2" button are
 * unchanged.
 */
export type PrefillResolutionPolicy = "defaults" | "project" | "material";

/** `localStorage` keys the active size/fps prefill policies are persisted
 * under (W1: split from the former single `nzltx23.prefillResolutionPolicy`,
 * which is deliberately no longer read). Mirrors `shell/ThemeContext.tsx`'s
 * `THEME_STORAGE_KEY` naming (owner decision: localStorage-only, no
 * native/settings.json involvement). */
export const PREFILL_SIZE_POLICY_STORAGE_KEY = "nzltx23.prefillSizePolicy";
export const PREFILL_FPS_POLICY_STORAGE_KEY = "nzltx23.prefillFpsPolicy";

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
  const stored = readStoredPolicy(PREFILL_FPS_POLICY_STORAGE_KEY, DEFAULT_PREFILL_FPS_POLICY);
  // X1: the fps axis no longer offers "material" (SDK can't read a material's
  // real fps — deferred to a future update), so a persisted "material" (e.g. a
  // value stored before X1) is treated as invalid and falls back to the default.
  return stored === "material" ? DEFAULT_PREFILL_FPS_POLICY : stored;
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
    // X1 (二重防御): "material" is not a valid fps choice — the Settings UI
    // disables that button, but coerce here too so no stray call can persist it.
    const coerced = next === "material" ? DEFAULT_PREFILL_FPS_POLICY : next;
    setFpsPolicyState(coerced);
    writeStoredPolicy(PREFILL_FPS_POLICY_STORAGE_KEY, coerced);
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
