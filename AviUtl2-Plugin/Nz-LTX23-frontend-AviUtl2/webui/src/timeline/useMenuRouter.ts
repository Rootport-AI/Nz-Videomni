import { useCallback, useEffect, useRef, useState } from "react";
import { bridge as defaultBridge, TIMELINE_MENU_INVOKED_EVENT } from "../bridge";
import type { NativeBridge, TimelineMenuInvokedData } from "../bridge";
import { routeMenuAction } from "./menuRouting";
import type { MenuRoute } from "./menuRouting";
import { resolveMenuSelection } from "./menuSelection";
import type { TimelineSelection } from "./menuSelection";

/** A fully-resolved menu command, ready for a mode to consume: the routing
 * decision plus the (file-path-completed) selection to act on. */
export interface RoutedMenuCommand {
  /** The routing decision (`targetMode`/`intent`/`needsSelection`). */
  route: MenuRoute;
  /** The selection to act on. When `route.needsSelection` and the pushed
   * snapshot had missing file paths, this is the authoritative re-queried
   * snapshot; otherwise it's the snapshot carried by the event. */
  selection: TimelineSelection;
  /** True when `timeline.getSelection` was re-queried to complete file paths. */
  refreshed: boolean;
}

export interface UseMenuRouterDeps {
  /** Bridge used to subscribe to `menuInvoked` and (when needed) re-query the
   * selection. Production call sites omit it. */
  nativeBridge?: NativeBridge | undefined;
  /** Called after a menu action is routed and its selection resolved. This is
   * the seam a host wires to its mode/store to actually open the target mode
   * and prefill it — kept as an out-parameter so this hook never reaches into
   * `modes/*` itself (their existing behavior is unchanged). */
  onRoute?: (command: RoutedMenuCommand) => void;
  /** Called when a `menuInvoked` action has no routing-table entry (unknown /
   * not-yet-supported native identifier). Optional; defaults to a no-op. */
  onUnknownAction?: (action: string, data: TimelineMenuInvokedData) => void;
}

export interface UseMenuRouterResult {
  /** The most recently routed command (route + resolved selection), or null
   * before the first routable `menuInvoked`. */
  command: RoutedMenuCommand | null;
  /** The action of the last `menuInvoked` event, whether or not it routed. */
  lastAction: string | null;
  /** Message of the last selection-resolution failure, else null. */
  error: string | null;
  /** Clears `command` (e.g. after the consuming mode has picked it up). */
  clear: () => void;
}

/**
 * Subscribes to `timeline.menuInvoked` and turns each event into a routed
 * command (contract v5, task 2 + task 3): it maps the native action to a
 * `{ targetMode, intent, needsSelection }` route (`menuRouting.ts`), then runs
 * the two-stage file-path completion (`menuSelection.ts`) so the exposed
 * selection has authoritative file paths before any cutout/generation.
 *
 * The hook only manages state and fires `onRoute`; it never opens a mode or
 * mutates a `modes/*` form itself, so existing mode behavior is untouched. A
 * host binds `onRoute` to its own mode switching / prefill.
 */
export function useMenuRouter(deps: UseMenuRouterDeps = {}): UseMenuRouterResult {
  const nativeBridge = deps.nativeBridge ?? defaultBridge;
  const [command, setCommand] = useState<RoutedMenuCommand | null>(null);
  const [lastAction, setLastAction] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const mountedRef = useRef(true);

  // Keep the latest callbacks in refs so the subscription effect need not
  // re-subscribe (and drop events) when a caller passes fresh inline handlers.
  const onRouteRef = useRef(deps.onRoute);
  onRouteRef.current = deps.onRoute;
  const onUnknownRef = useRef(deps.onUnknownAction);
  onUnknownRef.current = deps.onUnknownAction;

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const clear = useCallback(() => {
    setCommand(null);
  }, []);

  useEffect(() => {
    const unsubscribe = nativeBridge.on(TIMELINE_MENU_INVOKED_EVENT, (raw) => {
      const data = raw as TimelineMenuInvokedData;
      if (mountedRef.current) setLastAction(data.action);

      const route = routeMenuAction(data.action);
      if (!route) {
        onUnknownRef.current?.(data.action, data);
        return;
      }

      // Two-stage file-path completion, then publish the routed command.
      void (async () => {
        try {
          const { selection, refreshed } = await resolveMenuSelection(nativeBridge, {
            snapshot: data.selection,
            needsSelection: route.needsSelection,
          });
          const routed: RoutedMenuCommand = { route, selection, refreshed };
          if (mountedRef.current) {
            setCommand(routed);
            setError(null);
          }
          onRouteRef.current?.(routed);
        } catch (err) {
          if (mountedRef.current) {
            setError(err instanceof Error ? err.message : String(err));
          }
        }
      })();
    });
    return unsubscribe;
  }, [nativeBridge]);

  return { command, lastAction, error, clear };
}
