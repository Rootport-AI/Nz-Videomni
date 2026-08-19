import { useEffect, useRef, useState } from "react";
import { getConfig } from "../../api/client";
import type { AppConfig } from "../../api/types";
import { FALLBACK_APP_CONFIG } from "./defaultConfig";

export type ConfigState =
  | { status: "loading" }
  | { status: "ready"; config: AppConfig; usingFallback: boolean };

/** Fetches `/config` once on mount (Docs/API_REFERENCE.md §3.2 — "フロントは
 * ここから制約値・プリセット・上限を動的に取得すべき"). Falls back to a
 * built-in default set (`defaultConfig.ts`) if the request fails for any
 * reason, so the form is still usable; the caller should surface
 * `usingFallback` as a warning. */
export function useConfig(): ConfigState & { retry: () => void } {
  const [state, setState] = useState<ConfigState>({ status: "loading" });
  const mountedRef = useRef(true);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  useEffect(() => {
    setState({ status: "loading" });
    getConfig()
      .then((config) => {
        if (!mountedRef.current) return;
        setState({ status: "ready", config, usingFallback: false });
      })
      .catch(() => {
        if (!mountedRef.current) return;
        setState({ status: "ready", config: FALLBACK_APP_CONFIG, usingFallback: true });
      });
  }, [attempt]);

  return { ...state, retry: () => setAttempt((n) => n + 1) };
}
