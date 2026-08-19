import { useCallback, useEffect, useRef, useState } from "react";
import { apiClient as defaultApiClient } from "../api/client";
import type { ApiClient } from "../api/client";
import type { JobResponse, JobStatus } from "../api/types";

/** Docs/API_REFERENCE.md §11 step 6 / task brief: the job rail polls `GET
 * /jobs` every 2s. `modes/single/useGeneration.ts` no longer runs its own
 * poll (U-R1: it's submit-only — see that file's header comment), so this is
 * now the job list's sole source of truth. */
const DEFAULT_INTERVAL_MS = 2_000;

function isTerminal(status: JobStatus): boolean {
  return status === "completed" || status === "failed" || status === "cancelled";
}

/** True iff any job in `jobs` is `queued` or `running` — the app-wide "the
 * server is busy" predicate behind every Generate button's busy state.
 *
 * A plain exported function (not just the hook's `hasActiveJob` field) because
 * `shell/AppShell.tsx`'s right-click router cannot read the context: its
 * `handleRoute` is a `useCallback` that deliberately does NOT depend on `jobs`
 * (a 2s poll would re-create it constantly), so it reads the ledger through a
 * ref and needs the very same predicate applied by hand. One definition means
 * the button and the right-click can never disagree about what "busy" is. */
export function hasActiveJob(jobs: readonly JobResponse[]): boolean {
  return jobs.some((job) => job.status === "queued" || job.status === "running");
}

export interface UseJobsPollDeps {
  apiClient?: ApiClient;
  intervalMs?: number;
  /** Fired once per job the first time it's *observed* to settle into a
   * terminal state during this hook's lifetime (i.e. it was non-terminal,
   * or simply unseen, on a previous poll, and it's now terminal). Never
   * fires for the very first poll after mount — that establishes the
   * baseline "already-terminal" history instead of retroactively toasting
   * for old jobs from a previous session. */
  onJobSettled?: (job: JobResponse) => void;
}

export interface UseJobsPollResult {
  jobs: JobResponse[];
  /** True iff any job in the list is `queued` or `running`. */
  hasActiveJob: boolean;
  error: string | null;
  /** Forces an immediate poll (in addition to the regular interval).
   * Resolves once that poll has updated `jobs`, so callers that need the
   * freshest list right after a mutating call (cancel/delete) can await it. */
  refresh: () => Promise<void>;
}

/** Polls `GET /jobs` (Docs/API_REFERENCE.md §3.14) on an interval, exposing
 * the full list plus a settle-detection callback for toast notifications. */
export function useJobsPoll(deps: UseJobsPollDeps = {}): UseJobsPollResult {
  const client = deps.apiClient ?? defaultApiClient;
  const intervalMs = deps.intervalMs ?? DEFAULT_INTERVAL_MS;
  const onJobSettled = deps.onJobSettled;

  const [jobs, setJobs] = useState<JobResponse[]>([]);
  const [error, setError] = useState<string | null>(null);
  const mountedRef = useRef(true);
  // `null` until the first poll resolves, so that poll can seed history
  // without firing any onJobSettled callbacks.
  const lastStatusRef = useRef<Map<string, JobStatus> | null>(null);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const poll = useCallback(async () => {
    try {
      const list = await client.listJobs();
      if (!mountedRef.current) return;
      setError(null);

      const previous = lastStatusRef.current;
      const nextStatus = new Map<string, JobStatus>();
      for (const job of list) {
        nextStatus.set(job.job_id, job.status);
        if (previous !== null) {
          const prevStatus = previous.get(job.job_id);
          const wasNonTerminal = prevStatus === undefined || !isTerminal(prevStatus);
          if (wasNonTerminal && isTerminal(job.status) && prevStatus !== job.status) {
            onJobSettled?.(job);
          }
        }
      }
      lastStatusRef.current = nextStatus;
      setJobs(list);
    } catch (err) {
      if (!mountedRef.current) return;
      setError(err instanceof Error ? err.message : String(err));
    }
  }, [client, onJobSettled]);

  useEffect(() => {
    void poll();
    const timer = setInterval(() => void poll(), intervalMs);
    return () => clearInterval(timer);
  }, [poll, intervalMs]);

  return { jobs, hasActiveJob: hasActiveJob(jobs), error, refresh: poll };
}
