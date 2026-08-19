import { useCallback, useEffect, useRef, useState } from "react";
import { bridge as defaultBridge, TIMELINE_MENU_INVOKED_EVENT } from "../bridge";
import type { NativeBridge, ResultOf, TimelineMenuInvokedData } from "../bridge";

/** The `timeline.getSelection` snapshot (also the shape carried by the
 * `timeline.menuInvoked` event's `selection`). */
export type TimelineSelection = ResultOf<"timeline.getSelection">;

export interface UseTimelineSelectionDeps {
  nativeBridge?: NativeBridge;
}

export interface UseTimelineSelectionResult {
  /** The latest selection snapshot, or null before the first successful load. */
  selection: TimelineSelection | null;
  /** True while a `refresh()` (or the initial load) is in flight. */
  loading: boolean;
  /** Message of the last failed `getSelection`, else null. */
  error: string | null;
  /** The `action` of the most recent `timeline.menuInvoked` event, else null.
   * Lets a consumer react to a specific menu command (e.g. "cutout"/"i2v"). */
  lastMenuAction: string | null;
  /** Re-query `timeline.getSelection` on demand; resolves with the fresh
   * snapshot (or null on failure). */
  refresh: () => Promise<TimelineSelection | null>;
}

/**
 * Reads and tracks the current AviUtl2 timeline selection (contract v5).
 *
 * Two update paths:
 *  - Pull: `refresh()` (and an initial load on mount) call
 *    `timeline.getSelection`.
 *  - Push: a `timeline.menuInvoked` event carries a fresh selection snapshot
 *    captured at the moment the user invoked a plugin timeline-menu action, so
 *    state updates without a round trip. `lastMenuAction` exposes which action
 *    fired.
 *
 * `deps.nativeBridge` lets tests bind a purpose-built mock bridge (which can
 * `emit` the menuInvoked event); production call sites omit it.
 */
export function useTimelineSelection(deps: UseTimelineSelectionDeps = {}): UseTimelineSelectionResult {
  const nativeBridge = deps.nativeBridge ?? defaultBridge;
  const [selection, setSelection] = useState<TimelineSelection | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastMenuAction, setLastMenuAction] = useState<string | null>(null);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const refresh = useCallback(async (): Promise<TimelineSelection | null> => {
    setLoading(true);
    try {
      const next = await nativeBridge.request("timeline.getSelection", {});
      if (mountedRef.current) {
        setSelection(next);
        setError(null);
      }
      return next;
    } catch (err) {
      if (mountedRef.current) {
        setError(err instanceof Error ? err.message : String(err));
      }
      return null;
    } finally {
      if (mountedRef.current) setLoading(false);
    }
  }, [nativeBridge]);

  // Initial pull on mount.
  useEffect(() => {
    void refresh();
  }, [refresh]);

  // Push: subscribe to the native timeline-menu event.
  useEffect(() => {
    const unsubscribe = nativeBridge.on(TIMELINE_MENU_INVOKED_EVENT, (data) => {
      if (!mountedRef.current) return;
      const payload = data as TimelineMenuInvokedData;
      setLastMenuAction(payload.action);
      setSelection(payload.selection);
      setError(null);
    });
    return unsubscribe;
  }, [nativeBridge]);

  return { selection, loading, error, lastMenuAction, refresh };
}
