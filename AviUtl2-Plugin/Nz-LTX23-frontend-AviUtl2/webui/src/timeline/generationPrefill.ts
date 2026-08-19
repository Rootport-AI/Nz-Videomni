import type { AppMode } from "../shell/AppShell";
import type { TimelineSelection } from "./menuSelection";

/**
 * The one-shot prefill payload a right-click menu route hands to the mode it
 * opens (task: "ルーティング→プリフィル"). `AppShell` builds it in its
 * `useMenuRouter` `onRoute` seam from the resolved `RoutedMenuCommand`, then
 * passes it to *only* the target mode's screen as `initialIntent`, alongside a
 * monotonic remount key — the receiving screen consumes it exactly once, in a
 * `useState(() => ...)` lazy initializer (never via a prop-following effect).
 *
 * Deliberately small and transport-agnostic: `intent` is the WebUI sub-flow
 * identifier from `menuRouting.ts` (e.g. `"reference-video"`, `"extend-video"`)
 * a screen branches on; `selection` is the file-path-completed timeline
 * selection the mode derives its initial resolution / source material from.
 */
export interface GenerationPrefill {
  /** The routed sub-flow (`MenuRoute.intent`), e.g. `"reference-video"` for
   * IC-LoRA or `"image-to-video"` for I2V. */
  intent: string;
  /** The mode this prefill targets (`MenuRoute.targetMode`). Only the screen
   * for this mode ever receives the payload. */
  targetMode: AppMode;
  /** The resolved (file-path-completed) selection to derive initial params
   * from — same shape as `timeline.getSelection`. */
  selection: TimelineSelection;
  /** W5 (反対スロット保持): the OPPOSITE source slot carried across the Create
   * remount a material-item prefill triggers. The material intents #2/#3/#7
   * reset the whole Create form (owner-approved), but the slot they do NOT
   * replace should survive: #2 (`reference-video`) carries the existing
   * `sourceAudio`; #3 (`video-audio-to-video`) / #7 (`audio-to-video`) carry the
   * existing `referenceVideo` plus its two adapter strengths. The uploaded ids
   * are still valid server-side, so the material is re-attached with no
   * re-upload. `AppShell.handleRoute` reads `getPublishedSourceSlots()` at route
   * time and fills only the relevant, ready (non-null) side; absent = nothing to
   * carry (identical to the pre-W5 behaviour). */
  carryOver?: {
    referenceVideo?: {
      id: string;
      fileName: string;
      filePath: string;
      conditioningAttentionStrength: number | null;
      referenceVideoStrength: number | null;
    } | null;
    sourceAudio?: {
      id: string;
      fileName: string;
      filePath: string;
    } | null;
  };
}
