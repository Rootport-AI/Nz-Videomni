import type { JobStatus } from "../../api/types";

/** §1-80 Repeat count: pure helpers + the per-screen field value shared by
 * `useRepeatRun` (Single / Chained Generate bar). Everything here is free of
 * React so it can be unit-tested directly. */

export const REPEAT_MIN = 1;
export const REPEAT_MAX = 99;

/** Field text → total run count. Anything that is not a number (empty, NaN)
 * falls back to 1; fractions are truncated; the result is clamped to 1–99. */
export function clampRepeatCount(raw: string | number): number {
  const n = typeof raw === "number" ? raw : Number(raw.trim() === "" ? Number.NaN : raw);
  if (!Number.isFinite(n)) return REPEAT_MIN;
  return Math.min(REPEAT_MAX, Math.max(REPEAT_MIN, Math.trunc(n)));
}

/** A job status after which the job will never change again. Kept local
 * (`jobs/useJobsPoll.ts` has the same rule but does not export it). */
export function isTerminal(status: JobStatus): boolean {
  return status === "completed" || status === "failed" || status === "cancelled";
}

/** What to do once the job we were waiting on has settled: send the next run
 * only when it completed, runs remain, and the panel can submit right now;
 * otherwise stop the whole repeat. */
export function decideAfterSettle(status: JobStatus, remaining: number, canSubmit: boolean): "send" | "stop" {
  return status === "completed" && remaining > 0 && canSubmit ? "send" : "stop";
}

/** Identity of "what would be sent" — compared between the last submission
 * and the next one to tell whether the generation settings changed. */
export function submissionFingerprint(sub: unknown): string {
  return JSON.stringify(sub);
}

/** The "settings changed" toast fires once per gap between two submissions,
 * only while a repeat is active, and only when the next submission would
 * differ from the last one. */
export function shouldNotifyChange({
  active,
  notified,
  baseline,
  current,
}: {
  active: boolean;
  notified: boolean;
  baseline: string | null;
  current: string | null;
}): boolean {
  return active && !notified && baseline !== null && current !== null && current !== baseline;
}

// ---- The field text, kept per screen for the session (not in localStorage):
// it survives a screen remount (right-click `key` rebuild) and resets on reload.

export type RepeatScreen = "single" | "chained";

const DEFAULT_COUNT_TEXT = "1";
const countTexts: Record<RepeatScreen, string> = { single: DEFAULT_COUNT_TEXT, chained: DEFAULT_COUNT_TEXT };
const listeners = new Set<() => void>();

export function getRepeatCountText(screen: RepeatScreen): string {
  return countTexts[screen];
}

export function setRepeatCountText(screen: RepeatScreen, text: string): void {
  if (countTexts[screen] === text) return;
  countTexts[screen] = text;
  for (const listener of listeners) listener();
}

export function subscribeRepeatCountText(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

/** Tests only: put both screens back to the default "1". */
export function __resetRepeatCountForTests(): void {
  countTexts.single = DEFAULT_COUNT_TEXT;
  countTexts.chained = DEFAULT_COUNT_TEXT;
  for (const listener of listeners) listener();
}
