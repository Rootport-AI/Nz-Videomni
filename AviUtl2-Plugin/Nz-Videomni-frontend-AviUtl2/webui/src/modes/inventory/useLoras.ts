import { useCallback, useEffect, useRef, useState } from "react";
import { apiClient as defaultApiClient } from "../../api/client";
import type { ApiClient } from "../../api/client";
import type { LoraEntry } from "../../api/types";

export type LorasState =
  | { status: "loading" }
  | { status: "ready"; loras: LoraEntry[]; loraDir?: string }
  | { status: "error"; message: string };

export interface UseLorasDeps {
  apiClient?: ApiClient;
  /** 台帳 §1-104: the list follows the active base model on the server (H3
   * and LTX scan different directories), so the caller passes the active
   * base-model id here and every change refetches. The hook never reads the
   * value — it is only a trigger. The first key is `""` until `GET /models`
   * lands, so start-up fetches twice by design (`""` is not skipped: an older
   * server never sets an active base model). */
  refreshKey?: string | undefined;
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
 * request path as the mount-time load.
 *
 * 台帳 §1-104: also refetches whenever `refreshKey` (the active base-model
 * id) changes, and keeps the response's `lora_dir` as `loraDir`. */
export function useLoras(deps: UseLorasDeps = {}): LorasState & { refresh: () => Promise<void> } {
  const client = deps.apiClient ?? defaultApiClient;
  const refreshKey = deps.refreshKey;
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
      const { loras, lora_dir } = await client.getLoras();
      if (!mountedRef.current) return;
      setState(lora_dir !== undefined ? { status: "ready", loras, loraDir: lora_dir } : { status: "ready", loras });
    } catch (err) {
      if (!mountedRef.current) return;
      setState({ status: "error", message: err instanceof Error ? err.message : String(err) });
    }
  }, [client]);

  useEffect(() => {
    void refresh();
    // `refreshKey` is a trigger only (see `UseLorasDeps.refreshKey`).
  }, [refresh, refreshKey]);

  return { ...state, refresh };
}
