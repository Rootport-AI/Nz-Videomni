import { useCallback, useEffect, useRef, useState } from "react";
import { bridge as defaultBridge, TIMELINE_PROJECT_LOADED_EVENT } from "../bridge";
import type { NativeBridge, ResultOf } from "../bridge";

/** One orphaned provisional placeholder, as reported by
 * `timeline.scanProvisionals` (a placeholder whose backing job the WebUI no
 * longer knows about). */
export type ProvisionalOrphan = ResultOf<"timeline.scanProvisionals">["orphans"][number];

/** Injectable side effect: performs the `timeline.scanProvisionals` call.
 * Defaults to a real bridge call; tests substitute a plain mock so the hook's
 * wiring can be verified without a bridge. */
export type ScanProvisionals = () => Promise<ResultOf<"timeline.scanProvisionals">>;

export interface UseProjectOrphansDeps {
  /** Bridge used to subscribe to `timeline.projectLoaded` (and, by default, to
   * run the scan). Production call sites omit it. */
  nativeBridge?: NativeBridge;
  /** Overrides how the scan is performed. When omitted, a `nativeBridge`-bound
   * `timeline.scanProvisionals` request is used. */
  scan?: ScanProvisionals;
  /** When true, run one scan on mount in addition to reacting to
   * `projectLoaded` (the plugin fires `projectLoaded` at init, so this is off
   * by default to avoid a redundant scan; enable it for a UI that mounts after
   * the initial project is already open). */
  scanOnMount?: boolean;
}

export interface UseProjectOrphansResult {
  /** Orphaned provisional placeholders from the most recent scan. Empty until
   * the first `projectLoaded` (or `refresh()`), and reset to `[]` at the start
   * of each scan. */
  orphans: ProvisionalOrphan[];
  /** True while a scan is in flight. */
  scanning: boolean;
  /** Message of the last failed scan, else null. */
  error: string | null;
  /** True once at least one scan has completed (so the UI can distinguish
   * "no orphans found" from "not scanned yet"). */
  scanned: boolean;
  /** Run `timeline.scanProvisionals` on demand; resolves with the orphan list
   * (or `[]` on failure). Exposed so a "再スキャン" control, or a consumer that
   * wants to re-check, can trigger it without a `projectLoaded` event. */
  refresh: () => Promise<ProvisionalOrphan[]>;
  /** Clears the current orphan list — call after the user has acted on the
   * suggestion (re-generated or dismissed) so the prompt goes away. */
  clear: () => void;
}

/**
 * Wires the native `timeline.projectLoaded` trigger to the orphan-detection
 * scan (contract v5, task 1). On each `projectLoaded` event the hook calls
 * `timeline.scanProvisionals` and holds the resulting orphan list as state, so
 * a consumer can show a "再生成しますか？" ("re-generate these?") prompt for
 * placeholders left behind by a prior session.
 *
 * The scan side effect is injectable (`deps.scan`) so the wiring — subscribe,
 * scan-on-event, state management — is unit-testable with a plain mock and no
 * real bridge. The visual prompt itself is left to the consumer; this hook only
 * owns the trigger->scan->state plumbing.
 */
export function useProjectOrphans(deps: UseProjectOrphansDeps = {}): UseProjectOrphansResult {
  const nativeBridge = deps.nativeBridge ?? defaultBridge;
  const scan: ScanProvisionals =
    deps.scan ?? (() => nativeBridge.request("timeline.scanProvisionals", {}));
  const scanOnMount = deps.scanOnMount ?? false;

  const [orphans, setOrphans] = useState<ProvisionalOrphan[]>([]);
  const [scanning, setScanning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [scanned, setScanned] = useState(false);
  const mountedRef = useRef(true);

  // Keep the latest `scan` callback in a ref so the `projectLoaded`
  // subscription effect need not re-subscribe when the caller passes a fresh
  // inline function each render.
  const scanRef = useRef(scan);
  scanRef.current = scan;

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const refresh = useCallback(async (): Promise<ProvisionalOrphan[]> => {
    setScanning(true);
    setOrphans([]);
    try {
      const result = await scanRef.current();
      if (mountedRef.current) {
        setOrphans(result.orphans);
        setError(null);
        setScanned(true);
      }
      return result.orphans;
    } catch (err) {
      if (mountedRef.current) {
        setError(err instanceof Error ? err.message : String(err));
        setScanned(true);
      }
      return [];
    } finally {
      if (mountedRef.current) setScanning(false);
    }
  }, []);

  const clear = useCallback(() => {
    setOrphans([]);
  }, []);

  // Push: subscribe to the native project-load trigger and scan on each fire.
  useEffect(() => {
    const unsubscribe = nativeBridge.on(TIMELINE_PROJECT_LOADED_EVENT, () => {
      void refresh();
    });
    return unsubscribe;
  }, [nativeBridge, refresh]);

  // Optional initial scan (see `scanOnMount`).
  useEffect(() => {
    if (scanOnMount) void refresh();
  }, [scanOnMount, refresh]);

  return { orphans, scanning, error, scanned, refresh, clear };
}
