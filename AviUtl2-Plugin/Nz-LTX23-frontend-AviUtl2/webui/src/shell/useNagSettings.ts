import { useCallback, useState } from "react";
import {
  NAG_ALPHA_DEFAULT,
  NAG_SCALE_DEFAULT,
  NAG_TAU_DEFAULT,
  NEG_METHOD_DEFAULT,
  VSF_SCALE_DEFAULT,
  readStoredNagText,
  writeStoredNagText,
} from "./nagSettings";
import type { NagSettings } from "./nagSettings";

export interface UseNagSettingsResult {
  nag: NagSettings;
  setEnabled: (value: boolean) => void;
  /** Write-through: every keystroke is mirrored to `localStorage` immediately
   * (see {@link writeStoredNagText}) — `text` is the one field of
   * {@link NagSettings} that persists (owner decision §UX 5), and there is no
   * separate "commit" gesture (blur/submit) for a shared, always-mounted
   * textarea to hook a delayed write to. */
  setText: (value: string) => void;
  setScale: (value: number) => void;
  setTau: (value: number) => void;
  setAlpha: (value: number) => void;
  setMethod: (value: NagSettings["method"]) => void;
  setVsfScale: (value: number) => void;
  /** Restores `scale`/`tau`/`alpha`/`vsfScale` to their defaults ONLY —
   * `method`/`text`/`enabled` are left untouched (owner decision §UX 3: the
   * 🔄 button never touches the negative-prompt body, never flips the
   * checkbox, and never changes which method is selected — it resets
   * numeric parameters back to their defaults, not the choice of engine). */
  resetParams: () => void;
}

/**
 * NAG (Normalized Attention Guidance) settings, owned by `AppShell` and called
 * exactly ONCE (D1) so Create/Chain/Batch all read the identical
 * `NagSettings` object — mirrors how `prompt`/`controlLora` are themselves a
 * single `AppShell`-level `useState` passed down as props, rather than each
 * mode screen owning its own copy (a per-screen copy would silently reset on
 * every right-click remount's `key` bump, the same failure mode `controlLora`
 * was pulled up here to avoid).
 *
 * One `useState<NagSettings>` with a lazy initializer, plus plain functional
 * updates for every setter (so a setter never captures a stale `nag` — the
 * same shape `AppShell`'s own `setControlLora`/`setPrompt` already use):
 *  - `text` restores from `localStorage` via {@link readStoredNagText} — the
 *    only field of `NagSettings` that survives a reload.
 *  - `enabled` ALWAYS starts `false`. This is deliberate and NOT an oversight:
 *    owner decision §UX 5 ("チェックは毎セッションOFF開始") means a fresh
 *    mount never resumes a previous session's checked state, even though the
 *    negative-prompt TEXT itself does. There is no `enabled`-persistence path
 *    anywhere in this module on purpose.
 *  - `scale`/`tau`/`alpha`/`method`/`vsfScale` seed at their defaults every
 *    mount — session-local, never persisted (owner decision §UX 5; `method`
 *    joins this list 2026-07-29 alongside VSF for the same reason).
 */
export function useNagSettings(): UseNagSettingsResult {
  const [nag, setNag] = useState<NagSettings>(() => ({
    text: readStoredNagText(),
    enabled: false,
    scale: NAG_SCALE_DEFAULT,
    tau: NAG_TAU_DEFAULT,
    alpha: NAG_ALPHA_DEFAULT,
    method: NEG_METHOD_DEFAULT,
    vsfScale: VSF_SCALE_DEFAULT,
  }));

  const setEnabled = useCallback((value: boolean) => {
    setNag((prev) => ({ ...prev, enabled: value }));
  }, []);

  const setText = useCallback((value: string) => {
    writeStoredNagText(value);
    setNag((prev) => ({ ...prev, text: value }));
  }, []);

  const setScale = useCallback((value: number) => {
    setNag((prev) => ({ ...prev, scale: value }));
  }, []);

  const setTau = useCallback((value: number) => {
    setNag((prev) => ({ ...prev, tau: value }));
  }, []);

  const setAlpha = useCallback((value: number) => {
    setNag((prev) => ({ ...prev, alpha: value }));
  }, []);

  const setMethod = useCallback((value: NagSettings["method"]) => {
    setNag((prev) => ({ ...prev, method: value }));
  }, []);

  const setVsfScale = useCallback((value: number) => {
    setNag((prev) => ({ ...prev, vsfScale: value }));
  }, []);

  const resetParams = useCallback(() => {
    setNag((prev) => ({
      ...prev,
      scale: NAG_SCALE_DEFAULT,
      tau: NAG_TAU_DEFAULT,
      alpha: NAG_ALPHA_DEFAULT,
      vsfScale: VSF_SCALE_DEFAULT,
    }));
  }, []);

  return { nag, setEnabled, setText, setScale, setTau, setAlpha, setMethod, setVsfScale, resetParams };
}
