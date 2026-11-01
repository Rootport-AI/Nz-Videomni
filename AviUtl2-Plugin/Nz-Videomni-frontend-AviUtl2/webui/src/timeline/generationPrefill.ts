import type { AppMode } from "../shell/AppShell";
import type { TimelineSelection } from "./menuSelection";

/**
 * The one-shot prefill payload a right-click menu route hands to the mode it
 * opens (task: "ルーティング→プリフィル") — Chained and Edit only; the Single
 * screen receives its material items live (`createLiveCommands.ts`'s
 * `prefillMaterial`, 台帳 §1-114) and is never remounted. `AppShell` builds it in its
 * `useMenuRouter` `onRoute` seam from the resolved `RoutedMenuCommand`, then
 * passes it to *only* the target mode's screen as `initialIntent`, alongside a
 * monotonic remount key — the receiving screen consumes it exactly once, in a
 * `useState(() => ...)` lazy initializer (never via a prop-following effect).
 *
 * Deliberately small and transport-agnostic: `intent` is the WebUI sub-flow
 * identifier from `menuRouting.ts` (e.g. `"extend-video"`)
 * a screen branches on; `selection` is the file-path-completed timeline
 * selection the mode derives its initial resolution / source material from.
 */
export interface GenerationPrefill {
  /** The routed sub-flow (`MenuRoute.intent`), e.g. `"extend-video"` for a
   * Chained continuation or `"outpaint"` for Edit. */
  intent: string;
  /** The mode this prefill targets (`MenuRoute.targetMode`). Only the screen
   * for this mode ever receives the payload. */
  targetMode: AppMode;
  /** The resolved (file-path-completed) selection to derive initial params
   * from — same shape as `timeline.getSelection`. */
  selection: TimelineSelection;
}
