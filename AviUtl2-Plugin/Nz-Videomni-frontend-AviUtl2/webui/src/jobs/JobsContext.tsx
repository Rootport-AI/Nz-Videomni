import { createContext, useCallback, useContext, useMemo, useState } from "react";
import type { ReactNode } from "react";
import { apiClient as defaultApiClient } from "../api/client";
import type { ApiClient } from "../api/client";
import type { JobResponse } from "../api/types";
import { bridge as defaultBridge } from "../bridge";
import type { NativeBridge } from "../bridge";
import { useStrings } from "../i18n/LanguageContext";
import { useToasts } from "../shell/ToastContext";
import { settleProvisionalText } from "../timeline/provisionalReservation";
import { useJobsPoll } from "./useJobsPoll";

export interface JobsContextValue {
  jobs: JobResponse[];
  hasActiveJob: boolean;
  /** job_ids for which `DELETE /jobs/{id}` has been requested but the job
   * hasn't settled into a terminal state yet — cancel is best-effort/non-
   * immediate (Docs/API_REFERENCE.md §3.19), so the UI must keep showing
   * "Cancelling…" the whole time. */
  cancellingIds: ReadonlySet<string>;
  deletingIds: ReadonlySet<string>;
  refresh: () => Promise<void>;
  cancelJob: (jobId: string) => Promise<void>;
  deleteJob: (jobId: string) => Promise<void>;
}

const JobsContext = createContext<JobsContextValue | null>(null);

export interface JobsProviderProps {
  children: ReactNode;
  apiClient?: ApiClient;
  /** Test/integration seam: the bridge used to push the tracked provisional's
   * ✅/❌ terminal text on settle (I12 §5-5). Production omits it — the app-wide
   * singleton is used, matching every other Screen's `nativeBridge` default. */
  nativeBridge?: NativeBridge | undefined;
  /** Test-only override for the job-list poll cadence (production always
   * uses `useJobsPoll`'s 2s default). */
  intervalMs?: number;
}

/** Owns the M3 job rail's data: the `GET /jobs` poll (`useJobsPoll`), the
 * per-job "cancelling"/"deleting" in-flight flags for `DELETE /jobs/{id}`,
 * and pushing a toast (via `ToastContext`) whenever *any* job — not just
 * one the current session submitted — settles into a terminal state. */
export function JobsProvider({ children, apiClient, nativeBridge, intervalMs }: JobsProviderProps) {
  const strings = useStrings();
  const client = apiClient ?? defaultApiClient;
  const bridge = nativeBridge ?? defaultBridge;
  const { push } = useToasts();
  const [cancellingIds, setCancellingIds] = useState<Set<string>>(new Set());
  const [deletingIds, setDeletingIds] = useState<Set<string>>(new Set());

  const clearCancelling = useCallback((jobId: string) => {
    setCancellingIds((prev) => {
      if (!prev.has(jobId)) return prev;
      const next = new Set(prev);
      next.delete(jobId);
      return next;
    });
  }, []);

  const handleSettled = useCallback(
    (job: JobResponse) => {
      clearCancelling(job.job_id);
      // I12: rewrite the tracked provisional to its ✅/❌ terminal text (spec
      // §5-5 stage 3/4) and free the single reservation seat (§5-6). No-op unless
      // this is the tracked (bound) job, so an unrelated settling job never
      // rewrites or clears the seat; a failed text update is swallowed inside
      // `settleProvisionalText` so it can never break settle handling. Fire-and-
      // forget: the seat release happens synchronously inside, before the RPC.
      const provisionalText =
        job.status === "completed"
          ? strings.provisional.done
          : job.status === "cancelled"
            ? strings.provisional.failed(strings.provisional.cancelled)
            : // "failed" (or any other terminal status): a human-readable reason
              // from the job's error message when present, else the generic copy.
              strings.provisional.failed(job.error?.trim() || strings.provisional.failedGeneric);
      void settleProvisionalText(bridge, job.job_id, provisionalText);
      if (job.status === "completed") {
        push({ kind: "success", message: strings.jobs.toastCompleted(job.job_id), jobId: job.job_id });
      } else if (job.status === "failed") {
        push({ kind: "error", message: strings.jobs.toastFailed(job.job_id), jobId: job.job_id });
      } else if (job.status === "cancelled") {
        push({ kind: "error", message: strings.jobs.toastCancelled(job.job_id), jobId: job.job_id });
      }
    },
    [bridge, clearCancelling, push, strings],
  );

  const { jobs, hasActiveJob, refresh } = useJobsPoll({
    apiClient: client,
    onJobSettled: handleSettled,
    ...(intervalMs !== undefined ? { intervalMs } : {}),
  });

  const cancelJob = useCallback(
    async (jobId: string) => {
      setCancellingIds((prev) => new Set(prev).add(jobId));
      try {
        await client.deleteJob(jobId);
      } finally {
        await refresh();
      }
    },
    [client, refresh],
  );

  const deleteJob = useCallback(
    async (jobId: string) => {
      setDeletingIds((prev) => new Set(prev).add(jobId));
      try {
        await client.deleteJob(jobId);
      } finally {
        await refresh();
        setDeletingIds((prev) => {
          const next = new Set(prev);
          next.delete(jobId);
          return next;
        });
      }
    },
    [client, refresh],
  );

  const value = useMemo<JobsContextValue>(
    () => ({ jobs, hasActiveJob, cancellingIds, deletingIds, refresh, cancelJob, deleteJob }),
    [jobs, hasActiveJob, cancellingIds, deletingIds, refresh, cancelJob, deleteJob],
  );

  return <JobsContext.Provider value={value}>{children}</JobsContext.Provider>;
}

export function useJobsContext(): JobsContextValue {
  const ctx = useContext(JobsContext);
  if (!ctx) throw new Error("useJobsContext must be used within a JobsProvider");
  return ctx;
}
