import { useCallback, useEffect, useRef, useState } from "react";
import { BridgeError } from "../../bridge";
import { apiClient as defaultApiClient, BackendApiError } from "../../api/client";
import type { ApiClient } from "../../api/client";
import type { GenerateChainRequest, GenerateRequest } from "../../api/types";

/** Submit-only lifecycle (U-R1): the backend runs a single job at a time with
 * no queue, and the job list itself is now owned by the job ledger
 * (`jobs/JobLedger.tsx`, fed by `jobs/JobsContext.tsx`'s `GET /jobs` poll), so
 * this hook no longer polls, tracks completion, or inserts. It just fires
 * `POST /generate` (or `/generate/chain`), and on a `202` hands the new
 * `job_id` back to the caller (which refreshes the ledger + highlights it).
 *
 * `error` covers both transport failures and the `409 JOB_BUSY`/`409
 * PIPELINE_LOADING` a stale
 * `serverBusy` can still let through (Docs/API_REFERENCE.md §7 "同時1ジョブ",
 * reservation/auto-retry removed 2026-07-17): `describeError` surfaces the
 * backend code/message just below the Generate button, and re-clicking retries. */
export type SubmitState =
  | { phase: "idle" }
  | { phase: "submitting" }
  | { phase: "error"; code: string; message: string };

function describeError(error: unknown): { code: string; message: string } {
  if (error instanceof BackendApiError) return { code: error.code, message: error.message };
  if (error instanceof BridgeError) return { code: error.code, message: error.message };
  if (error instanceof Error) return { code: "UNKNOWN", message: error.message };
  return { code: "UNKNOWN", message: String(error) };
}

export interface UseGenerationSubmitDeps {
  apiClient?: ApiClient;
  /** Called with the accepted `job_id` after a successful `202`. The submit
   * stays in `submitting` until this resolves (U-R1 MN-6: don't re-enable the
   * button before the job appears in the ledger), so pass the ledger
   * `refresh()` here — its resolution is what closes the double-click window. */
  onSubmitted?: (jobId: string) => void | Promise<void>;
  /** X2(a): called when the submit rejects synchronously (a 422/busy that never
   * reached `onSubmitted`), so a right-click reservation seat + its
   * `NzVideomni#pending-…` placeholder can be rolled back rather than left stranded
   * (the permanent "予約不可" block the owner hit). Fired fire-and-forget from the
   * `begin` catch BEFORE the `mountedRef` guard — it cleans up module-level
   * reservation state, so it must run even if the screen unmounted mid-submit
   * (a tab switch while submitting) so the seat can never linger. A plain
   * panel-origin Generate (no reservation waiting) makes this a no-op in the
   * caller. */
  onFailed?: () => void | Promise<void>;
}

export interface UseGenerationSubmitResult {
  submitState: SubmitState;
  submit: (request: GenerateRequest) => void;
  /** Same lifecycle as `submit`, but through `POST /generate/chain`
   * (Docs/API_REFERENCE.md §3.13) — Chain mode and Create's A2V fast path. */
  submitChain: (request: GenerateChainRequest) => void;
  reset: () => void;
}

/** `deps` lets tests bind a purpose-configured mock bridge/client instead of
 * the app-wide singleton; production call sites pass `onSubmitted` (and omit
 * `apiClient`). */
export function useGenerationSubmit(deps: UseGenerationSubmitDeps = {}): UseGenerationSubmitResult {
  const client = deps.apiClient ?? defaultApiClient;
  const [submitState, setSubmitState] = useState<SubmitState>({ phase: "idle" });

  const mountedRef = useRef(true);
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  // Latest `onSubmitted` in a ref so `submit`/`submitChain` keep stable
  // identities even as the caller's closure (which captures a fresh
  // `refresh`/highlight setter each render) changes.
  const onSubmittedRef = useRef(deps.onSubmitted);
  useEffect(() => {
    onSubmittedRef.current = deps.onSubmitted;
  }, [deps.onSubmitted]);

  // Latest `onFailed` in a ref (same reason as `onSubmittedRef`), so `begin`
  // keeps a stable identity while the caller's closure changes each render.
  const onFailedRef = useRef(deps.onFailed);
  useEffect(() => {
    onFailedRef.current = deps.onFailed;
  }, [deps.onFailed]);

  const begin = useCallback((fire: () => Promise<{ job_id: string }>) => {
    setSubmitState({ phase: "submitting" });
    void (async () => {
      let jobId: string;
      try {
        const accepted = await fire();
        jobId = accepted.job_id;
      } catch (err) {
        // X2(a): roll back a stranded right-click reservation FIRST — before the
        // mountedRef guard, since this cleans up module-level reservation state
        // (not React state), so it must run even when the screen unmounted
        // mid-submit. Fire-and-forget: never awaited, never blocks the error
        // surfacing below.
        void onFailedRef.current?.();
        if (!mountedRef.current) return;
        const { code, message } = describeError(err);
        setSubmitState({ phase: "error", code, message });
        return;
      }
      // MN-6: hold "submitting" across the caller's `onSubmitted` (the ledger
      // refresh) so the button doesn't re-enable before the job is visible.
      // A rejection here (e.g. the refresh's own GET failing) is swallowed
      // rather than left as an unhandled rejection: the job was already
      // accepted by the backend, so the job rail's own poll picks it up on
      // its next tick regardless of whether this particular refresh worked.
      try {
        await onSubmittedRef.current?.(jobId);
      } catch {
        // refresh失敗はポーリングが回収する
      } finally {
        if (mountedRef.current) setSubmitState({ phase: "idle" });
      }
    })();
  }, []);

  const submit = useCallback(
    (request: GenerateRequest) => begin(() => client.generate(request)),
    [begin, client],
  );

  const submitChain = useCallback(
    (request: GenerateChainRequest) => begin(() => client.generateChain(request)),
    [begin, client],
  );

  const reset = useCallback(() => setSubmitState({ phase: "idle" }), []);

  return { submitState, submit, submitChain, reset };
}
