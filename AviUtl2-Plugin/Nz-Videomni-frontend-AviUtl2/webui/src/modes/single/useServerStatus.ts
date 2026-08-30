import { useCallback, useEffect, useRef, useState } from "react";
import { bridge as defaultBridge } from "../../bridge";
import type { NativeBridge } from "../../bridge";
import { apiClient as defaultApiClient, BackendApiError } from "../../api/client";
import type { ApiClient } from "../../api/client";
import type { StatusResponse } from "../../api/types";

export type ServerStatusState =
  /** The native bridge itself is unreachable (WebView2 channel down or
   * missing) — distinct from the backend HTTP server being offline. Mirrors
   * the M1 "Native bridge not available" display. */
  | { kind: "bridge-unavailable" }
  | { kind: "checking" }
  /** `backend.request` failed at the transport level (BACKEND_UNREACHABLE /
   * BACKEND_TIMEOUT): the backend process is not running or not responding. */
  | { kind: "offline"; message: string }
  | { kind: "online"; status: StatusResponse }
  /** Reachable, and the server is rebuilding its worker — `GET /status`'s
   * `state === "loading"` (multi-engine groundwork §3-97 P6;
   * `Docs/MULTI_ENGINE_DESIGN.md` §6.5).
   *
   * This is checked BEFORE `busy` below, and the order is the whole point.
   * A load takes minutes and `pipeline_loaded` stays `false` throughout, so
   * without this the badge sat on plain "online" the entire time — which
   * reads as "ready, go ahead and generate" and earns the user a failure.
   * Loading is also the stronger claim of the two when both could apply:
   * a queue count is stale bookkeeping next to a worker that is being torn
   * down and rebuilt.
   *
   * `status` is `null` when this WebUI's OWN `POST /pipeline/load` raised the
   * flag (`UseServerStatusDeps.localLoading`): there is no `/status` body
   * behind that claim, only the request in flight. Nothing reads `.status` on
   * this variant — the two readers (`AppShell`, `SettingsPanel`) narrow to
   * `online`/`busy` first. */
  | { kind: "loading-models"; status: StatusResponse | null }
  /** Reachable, but a job is currently occupying the single-job queue. */
  | { kind: "busy"; status: StatusResponse }
  /** Reachable bridge and backend, but an unexpected error code came back. */
  | { kind: "error"; message: string };

/** Poll period.
 *
 * A load this WebUI starts itself no longer needs to be caught by the poll at
 * all: the issuer hands its `POST /pipeline/load` to `JobsContext`'s
 * `trackPipelineLoad`, and `localLoading` reports both edges at 0ms. Chasing
 * short loads with a short period — the reason this was once 2.5s — solved a
 * problem that no longer exists.
 *
 * What is left for the poll is a safety net for the things this WebUI cannot
 * see for itself: (1) a load started from another entry point (the Gradio UI,
 * the MCP server), and (2) the backend process appearing or disappearing.
 * Neither is urgent to the millisecond, so 10s.
 *
 * Accepted by design: an EXTERNAL load shorter than one period can begin and
 * end between two polls and never show a badge. That window belongs to a
 * client that is not this one, which is showing its own progress. */
const DEFAULT_INTERVAL_MS = 10_000;

/** The `loading-models` state raised by our own in-flight `POST
 * /pipeline/load` rather than by a `GET /status` body. Module-level so the
 * returned object is referentially stable across renders. */
const LOCAL_LOADING: ServerStatusState = { kind: "loading-models", status: null };

/** Polls server connectivity on mount and every `intervalMs` (default 10s),
 * per Docs/API_REFERENCE.md §11 step 1. Two independent failure axes are
 * distinguished: the native bridge being unreachable (checked via `ping`)
 * vs. the backend HTTP server being unreachable (checked via `GET /status`,
 * surfaced as `BACKEND_UNREACHABLE`/`BACKEND_TIMEOUT`). */
export interface UseServerStatusDeps {
  nativeBridge?: NativeBridge;
  apiClient?: ApiClient;
  /** This WebUI's own `POST /pipeline/load` is in flight
   * (`JobsContext.pipelineLoading`). While true the hook reports
   * `loading-models` outright, without waiting for a poll to confirm it.
   *
   * Accepted degradation: `offline`/`error` are hidden for that window too. A
   * dead backend rejects the POST, which lowers the flag and lets the next
   * fetch surface `offline`; a hung one is bounded by `loadPipeline`'s own
   * 600s timeout — the same window the dropdown was already disabled for. */
  localLoading?: boolean;
}

/** `deps` lets tests bind a purpose-configured mock bridge instead of the
 * app-wide singleton; production call sites can omit it entirely. */
export function useServerStatus(intervalMs = DEFAULT_INTERVAL_MS, deps: UseServerStatusDeps = {}): {
  state: ServerStatusState;
  retry: () => void;
} {
  const localLoading = deps.localLoading ?? false;
  const nativeBridge = deps.nativeBridge ?? defaultBridge;
  const client = deps.apiClient ?? defaultApiClient;
  const [state, setState] = useState<ServerStatusState>({ kind: "checking" });
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const check = useCallback(async () => {
    try {
      await nativeBridge.request("ping", {});
    } catch {
      if (mountedRef.current) setState({ kind: "bridge-unavailable" });
      return;
    }

    try {
      const status = await client.getStatus();
      if (!mountedRef.current) return;
      if (status.state === "loading") {
        setState({ kind: "loading-models", status });
      } else if (status.queue.running > 0) {
        setState({ kind: "busy", status });
      } else {
        setState({ kind: "online", status });
      }
    } catch (err) {
      if (!mountedRef.current) return;
      if (err instanceof BackendApiError && (err.code === "BACKEND_UNREACHABLE" || err.code === "BACKEND_TIMEOUT")) {
        setState({ kind: "offline", message: err.message });
      } else {
        const message = err instanceof Error ? err.message : String(err);
        setState({ kind: "error", message });
      }
    }
  }, [nativeBridge, client]);

  useEffect(() => {
    void check();
    const timer = setInterval(() => {
      void check();
    }, intervalMs);
    return () => clearInterval(timer);
  }, [check, intervalMs]);

  // Falling edge (true -> false): our load just finished, so fetch once
  // immediately instead of leaving a stale body on screen for up to a full
  // period. The ref seeds from the current value, so neither the first mount
  // nor StrictMode's double-invoke fires it.
  const prevLocalLoadingRef = useRef(localLoading);
  useEffect(() => {
    if (prevLocalLoadingRef.current && !localLoading) void check();
    prevLocalLoadingRef.current = localLoading;
  }, [localLoading, check]);

  const retry = useCallback(() => void check(), [check]);

  return { state: localLoading ? LOCAL_LOADING : state, retry };
}
