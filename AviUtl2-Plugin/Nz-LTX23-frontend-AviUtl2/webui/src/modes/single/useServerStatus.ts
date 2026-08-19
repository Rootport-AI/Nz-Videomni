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
  /** Reachable, but a job is currently occupying the single-job queue. */
  | { kind: "busy"; status: StatusResponse }
  /** Reachable bridge and backend, but an unexpected error code came back. */
  | { kind: "error"; message: string };

const DEFAULT_INTERVAL_MS = 10_000;

/** Polls server connectivity on mount and every `intervalMs` (default 10s),
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
      if (status.queue.running > 0) {
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
