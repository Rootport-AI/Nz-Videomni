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
   * down and rebuilt. */
  | { kind: "loading-models"; status: StatusResponse }
  /** Reachable, but a job is currently occupying the single-job queue. */
  | { kind: "busy"; status: StatusResponse }
  /** Reachable bridge and backend, but an unexpected error code came back. */
  | { kind: "error"; message: string };

/** Poll period. 2.5s, NOT the 10s this used to be (2026-08-20).
 *
 * The badge is the only thing that tells the user the engine is being rebuilt,
 * and a rebuild is not always long: swapping one GGUF checkpoint on a warm
 * machine takes about 8 seconds (`Docs/VERIFICATION_LOG.md` §68.5 measured
 * exactly that for default -> Sulphur). At 10s a whole rebuild could start and
 * finish between two polls, so `loading-models` was never rendered — which is
 * what the owner saw. 2.5s puts at least three polls inside an 8s load.
 *
 * A single-step interval rather than a fast/slow state machine, deliberately:
 * a two-speed poller can only speed up AFTER it has seen `state === "loading"`
 * once, which is precisely the observation the short load denies it. `GET
 * /status` is a cheap read (`api/status.py`: in-memory fields plus one
 * `torch.cuda.mem_get_info`; ~2ms round trip on the real machine, per the
 * plugin log) against localhost, and the job ledger already polls every 2s
 * next to it, so this changes no order of magnitude. */
const DEFAULT_INTERVAL_MS = 2_500;

/** Polls server connectivity on mount and every `intervalMs` (default 2.5s),
 * per Docs/API_REFERENCE.md §11 step 1. Two independent failure axes are
 * distinguished: the native bridge being unreachable (checked via `ping`)
 * vs. the backend HTTP server being unreachable (checked via `GET /status`,
 * surfaced as `BACKEND_UNREACHABLE`/`BACKEND_TIMEOUT`). */
export interface UseServerStatusDeps {
  nativeBridge?: NativeBridge;
  apiClient?: ApiClient;
}

/** `deps` lets tests bind a purpose-configured mock bridge instead of the
 * app-wide singleton; production call sites can omit it entirely. */
export function useServerStatus(intervalMs = DEFAULT_INTERVAL_MS, deps: UseServerStatusDeps = {}): {
  state: ServerStatusState;
  retry: () => void;
} {
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

  return { state, retry: () => void check() };
}
