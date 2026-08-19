/**
 * NAG (Normalized Attention Guidance) settings — the CFG-free way to make a
 * negative prompt actually affect generation (2026-07-28, backend commit
 * 2ae497b). Shared by Create/Chain/Batch via a single `NagSettings` object
 * (D1: `AppShell` useState + props, no Context — mirrors `prompt`/
 * `controlLora`; see `shell/useNagSettings.ts`). This module is the pure,
 * UI-free layer: the type, the constants, the localStorage read/write for the
 * negative text, and the additive request-field contract (D2).
 */

/** One clip/chain/batch's NAG configuration. `text` is the negative prompt
 * body (persisted, see {@link readStoredNagText}); `enabled` gates whether
 * the non-CFG negative (NAG or VSF, see `method`) is sent at all —
 * soft-disable per the owner's decision (§UX 5): unchecking keeps
 * `text`/`scale`/`tau`/`alpha`/`vsfScale` intact, it just stops them from
 * reaching the request. `method` picks which of the two engines the backend
 * should use (2026-07-29, backend §41); `scale`/`tau`/`alpha`/`vsfScale` are
 * all session-local (never persisted) per the same decision. */
export interface NagSettings {
  text: string;
  enabled: boolean;
  scale: number;
  tau: number;
  alpha: number;
  method: "nag" | "vsf";
  vsfScale: number;
}

/** Seeded into a fresh `text` (and used as the textarea `placeholder`) —
 * LTX's own recommended negative prompt (blurry/low-quality output plus the
 * watermark/subtitle-burn-in artifact LTX-2.3 is known to produce). Not an
 * i18n string (D6): it's an English prompt fragment sent to the model, not
 * UI copy the user reads. */
export const DEFAULT_NEGATIVE_PROMPT = "blurry, low quality, distorted, watermark, text";

/** `nag_scale`/`nag_tau`/`nag_alpha` defaults, ranges and slider `step`s.
 * These mirror the backend's `api/models.py` `Field(...)` defaults as of
 * commit 2ae497b (11.0 / 2.5 / 0.25) — if the backend's defaults ever change,
 * update both sides together. NAG's own guidance is to leave tau/alpha fixed
 * and only tune scale (hence scale gets top-level UI placement while tau/
 * alpha live in an "Advanced" sub-accordion, see `shell/NagAccordion.tsx`). */
export const NAG_SCALE_DEFAULT = 11.0;
export const NAG_SCALE_MIN = 1;
export const NAG_SCALE_MAX = 20;
export const NAG_SCALE_STEP = 0.5;

export const NAG_TAU_DEFAULT = 2.5;
export const NAG_TAU_MIN = 1;
export const NAG_TAU_MAX = 10;
export const NAG_TAU_STEP = 0.1;

export const NAG_ALPHA_DEFAULT = 0.25;
export const NAG_ALPHA_MIN = 0;
export const NAG_ALPHA_MAX = 1;
export const NAG_ALPHA_STEP = 0.05;

/** Which non-CFG negative-prompt engine the backend should run (2026-07-29,
 * backend §41: VSF/"Value Sign Flip" joins NAG as a second method). Mirrors
 * the backend's `api/models.py` `Field(...)` default ("nag") as of that
 * commit — if the backend's default ever changes, update both sides
 * together. */
export const NEG_METHOD_DEFAULT: NagSettings["method"] = "nag";

/** `vsf_scale` default, range and slider `step`. Mirrors the backend's
 * `api/models.py` `Field(...)` defaults as of backend §41 (1.5 / 0-10) —
 * if the backend's defaults ever change, update both sides together. */
export const VSF_SCALE_DEFAULT = 1.5;
export const VSF_SCALE_MIN = 0;
export const VSF_SCALE_MAX = 10;
export const VSF_SCALE_STEP = 0.1;

/** `localStorage` key the negative-prompt TEXT alone is persisted under (only
 * `text` persists — `enabled`/`scale`/`tau`/`alpha`/`method`/`vsfScale` are
 * session-local, owner decision §UX 5). Mirrors `shell/ThemeContext.tsx`'s
 * `THEME_STORAGE_KEY` naming. */
export const NAG_TEXT_STORAGE_KEY = "nzltx23.nagNegativeText";

/** The frozen "NAG is off" sentinel every form hook defaults `nag` to when
 * its caller omits it (D3, mirrors `controlLora`'s own optional-dep pattern
 * in `useGenerationForm.ts`). `Object.freeze`d (unlike the readonly-typed
 * `EMPTY_CONTROL_LORA_NAMES` in `useChainForm.ts`) since this is a full
 * settings object or is passed straight into `nagRequestFields` across many
 * calls — freezing catches any accidental in-place mutation immediately
 * rather than as a hard-to-trace shared-reference bug. */
export const NAG_OFF: Readonly<NagSettings> = Object.freeze({
  text: DEFAULT_NEGATIVE_PROMPT,
  enabled: false,
  scale: NAG_SCALE_DEFAULT,
  tau: NAG_TAU_DEFAULT,
  alpha: NAG_ALPHA_DEFAULT,
  method: NEG_METHOD_DEFAULT,
  vsfScale: VSF_SCALE_DEFAULT,
});

/** Reads the persisted negative-prompt text. Wrapped in a `try` since
 * `localStorage` can throw in some restricted embeddings (e.g. a WebView2
 * host with storage disabled) — falling back to the default is preferable to
 * a crash. Mirrors `ThemeContext.tsx`'s `readStoredTheme`. Falls back to
 * {@link DEFAULT_NEGATIVE_PROMPT} when the stored value is missing OR
 * whitespace-only — an empty string can end up persisted (the textarea has
 * no non-empty guard while `enabled` is false) and without this fallback
 * there would be no way back to the default text short of clearing storage
 * by hand. */
export function readStoredNagText(): string {
  try {
    const stored = window.localStorage.getItem(NAG_TEXT_STORAGE_KEY);
    if (stored !== null && stored.trim() !== "") return stored;
  } catch {
    // Ignore — localStorage unavailable.
  }
  return DEFAULT_NEGATIVE_PROMPT;
}

/** Persists the negative-prompt text. Wrapped in a `try` since `localStorage`
 * can throw in some restricted embeddings (e.g. private mode) — a failed
 * write is silently swallowed, matching `ThemeContext.tsx`'s
 * `writeStoredTheme`. */
export function writeStoredNagText(text: string): void {
  try {
    window.localStorage.setItem(NAG_TEXT_STORAGE_KEY, text);
  } catch {
    // Ignore — localStorage unavailable (e.g. private mode).
  }
}

/** True when NAG is enabled but the negative-prompt body is empty/
 * whitespace-only — the one client-side validity gate this feature adds
 * (backend also 422s this combination). `false` whenever `enabled` is
 * false: a blank body is harmless while NAG isn't going to be sent at all. */
export function isNagNegativeEmpty(nag: NagSettings): boolean {
  return nag.enabled && nag.text.trim() === "";
}

/** The additive NAG fields as sent on a `/generate` or `/generate/chain`
 * request. All properties optional so the result can be spread directly into
 * `A2vChainPayload`'s literal (which already declares a required
 * `negative_prompt: string` for its own reasons) without a duplicate-key type
 * error. */
export interface NagRequestFields {
  negative_prompt?: string;
  nag_enabled?: boolean;
  nag_scale?: number;
  nag_tau?: number;
  nag_alpha?: number;
  neg_method?: NagSettings["method"];
  vsf_scale?: number;
}

/** Builds the fields to spread into a generate/chain request body (D2: the
 * single place the additive-contract "omit entirely when off" rule lives).
 * `nag` absent or `enabled: false` (including the {@link NAG_OFF} sentinel)
 * returns `{}` — every caller's request stays byte-identical to the
 * pre-NAG contract. `enabled: true` returns exactly the 7 keys below,
 * REGARDLESS of `method` — the backend reads `neg_method` to pick which
 * engine `nag_*`/`vsf_scale` apply to, the same "send everything, engine
 * reads what it needs" shape the backend's own Gradio UI uses (backend
 * §41). */
export function nagRequestFields(nag: NagSettings | undefined): NagRequestFields {
  if (!nag || !nag.enabled) return {};
  return {
    negative_prompt: nag.text,
    nag_enabled: true,
    nag_scale: nag.scale,
    nag_tau: nag.tau,
    nag_alpha: nag.alpha,
    neg_method: nag.method,
    vsf_scale: nag.vsfScale,
  };
}
