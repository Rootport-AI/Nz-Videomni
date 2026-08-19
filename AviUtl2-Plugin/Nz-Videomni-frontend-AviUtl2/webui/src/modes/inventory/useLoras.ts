import { useCallback, useEffect, useRef, useState } from "react";
import { apiClient as defaultApiClient } from "../../api/client";
import type { ApiClient } from "../../api/client";
import type { LoraEntry } from "../../api/types";

export type LorasState =
  | { status: "loading" }
  | { status: "ready"; loras: LoraEntry[] }
  | { status: "error"; message: string };

export interface UseLorasDeps {
  apiClient?: ApiClient;
}

/** Fetches `GET /loras` on mount (Docs/API_REFERENCE.md §3.6) for the
 * Library screen's Style/Control browser. Mirrors `useConfig`'s
 * loading/ready/error shape; unlike `useConfig` there's no built-in fallback
 * data (an empty LoRA list simply means "no LoRAs installed", which the
 * screen already renders as its own empty state) — a fetch failure instead
 * of a wrong empty list.
 *
 * `refresh` is exposed separately from the initial fetch so the "Reload"
 * button (`POST /loras/reload` -> `refresh()`) reuses the exact same
 * request path as the mount-time load. */
export function useLoras(deps: UseLorasDeps = {}): LorasState & { refresh: () => Promise<void> } {
  const client = deps.apiClient ?? defaultApiClient;
  const [state, setState] = useState<LorasState>({ status: "loading" });
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const refresh = useCallback(async () => {
    try {
      const { loras } = await client.getLoras();
      if (!mountedRef.current) return;
      setState({ status: "ready", loras });
    } catch (err) {
      if (!mountedRef.current) return;
      setState({ status: "error", message: err instanceof Error ? err.message : String(err) });
    }
  }, [client]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  return { ...state, refresh };
}
