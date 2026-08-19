import { useCallback, useEffect, useRef, useState } from "react";
import { apiClient as defaultApiClient, BackendApiError } from "../api/client";
import type { ApiClient } from "../api/client";
import type { JobStatus } from "../api/types";

/** Terminal `JobStatus` values eligible for the purge action — the other two
 * (`queued`/`running`) are never touched by purge, mirroring `deleteJob`'s
 * own "in-flight jobs get a cancel request, not a delete" split
 * server-side. */
const TERMINAL_STATUSES: readonly JobStatus[] = ["completed", "failed", "cancelled"];

export type PipelineUnloadState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "done" }
  /** 409 JOB_BUSY — a generation job is running, so the engine cannot be
   * torn down right now (mirrors `useModels`'s `ModelsLoadState` "busy"
   * branch for the sibling `/pipeline/load` action). */
  | { status: "busy" }
  | { status: "error"; code: string; message: string };

export type JobsPurgeState =
  | { status: "idle" }
  | { status: "loading" }
  /** `attempted` is how many terminal jobs the sweep found (from `GET
   * /jobs`); `deleted` is how many of those `DELETE /jobs/{id}` calls
   * actually succeeded. The two can diverge — `attempted > deleted` means
   * some deletes failed, which callers must distinguish from "there was
   * nothing to purge" (`attempted === 0`). */
  | { status: "done"; deleted: number; attempted: number }
  | { status: "error"; message: string };

export interface UseDangerZoneDeps {
  apiClient?: ApiClient;
}

export interface UseDangerZoneResult {
  unload: PipelineUnloadState;
  /** `POST /pipeline/unload`. Fire-and-forget: watch `unload` for the
   * outcome (same convention as `useModels.loadPipeline`). */
  unloadPipeline: () => void;
  purge: JobsPurgeState;
  /** `GET /jobs` then a sequential `DELETE /jobs/{id}` per terminal
   * (completed/failed/cancelled) job. Each delete is caught individually so
   * a lost race (e.g. the job already vanished, `JOB_NOT_FOUND`) is skipped
   * rather than aborting the rest of the sweep — `purge.deleted` only counts
   * calls that did not throw. Fire-and-forget: watch `purge` for the
   * outcome. */
  purgeTerminalJobs: () => void;
}

/** Backs the Settings panel's "danger zone" section (N4): an explicit engine
 * unload (`POST /pipeline/unload`) and a bulk purge of every terminal job
 * (`GET /jobs` + `DELETE /jobs/{id}` per completed/failed/cancelled entry).
 * Two fully independent action/state pairs — mirrors `useModels`'s
 * list/load split, just without the shared list fetch. `apiClient` defaults
 * to the app-wide singleton but accepts an override so tests (and
 * `SettingsPanel`'s `nativeBridge` test hook) can bind a purpose-configured
 * mock bridge instead. */
export function useDangerZone(deps: UseDangerZoneDeps = {}): UseDangerZoneResult {
  const client = deps.apiClient ?? defaultApiClient;
  const [unload, setUnload] = useState<PipelineUnloadState>({ status: "idle" });
  const [purge, setPurge] = useState<JobsPurgeState>({ status: "idle" });
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const unloadPipeline = useCallback(() => {
    setUnload({ status: "loading" });
    void (async () => {
      try {
        await client.unloadPipeline();
        if (!mountedRef.current) return;
        setUnload({ status: "done" });
      } catch (err) {
        if (!mountedRef.current) return;
        if (err instanceof BackendApiError && err.code === "JOB_BUSY") {
          setUnload({ status: "busy" });
          return;
        }
        const { code, message } =
          err instanceof BackendApiError
            ? { code: err.code, message: err.message }
            : { code: "UNKNOWN", message: err instanceof Error ? err.message : String(err) };
        setUnload({ status: "error", code, message });
      }
    })();
  }, [client]);

  const purgeTerminalJobs = useCallback(() => {
    setPurge({ status: "loading" });
    void (async () => {
      let jobs;
      try {
        jobs = await client.listJobs();
      } catch (err) {
        if (!mountedRef.current) return;
        setPurge({ status: "error", message: err instanceof Error ? err.message : String(err) });
        return;
      }

      const terminal = jobs.filter((job) => TERMINAL_STATUSES.includes(job.status));
      let deleted = 0;
      for (const job of terminal) {
        try {
          await client.deleteJob(job.job_id);
          deleted += 1;
        } catch {
          // Best-effort sweep: a delete that lost a race (e.g. the job was
          // already removed) is skipped, not fatal to the rest of the purge.
        }
      }

      if (!mountedRef.current) return;
      setPurge({ status: "done", deleted, attempted: terminal.length });
    })();
  }, [client]);

  return { unload, unloadPipeline, purge, purgeTerminalJobs };
}
