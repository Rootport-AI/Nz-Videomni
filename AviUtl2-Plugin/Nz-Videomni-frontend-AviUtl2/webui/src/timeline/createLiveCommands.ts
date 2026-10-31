/**
 * A minimal live-command channel from `AppShell.handleRoute` to the
 * always-mounted Create screen (RIGHTCLICK_REDESIGN_SPEC.md §3-6 #10, §7-2).
 *
 * The Chained / Edit right-click routes deliver their intent through a REMOUNT
 * (a fresh `pendingIntent` + a bumped per-mode remount token), which the target
 * screen consumes once in a mount effect. Create is never remounted by a
 * right-click: the keep-panel-state items ✨ (#9) and 📷 (#10, D3 decision)
 * and, since 台帳 §1-114 (2026-10-31), the material items #2/#4/#7
 * (`prefillMaterial`) all reach the live screen through this channel, WITHOUT
 * wiping the form state the user already has.
 *
 * This is the smallest shared mechanism for that: a module-level subscription
 * store, mirroring the module-level shared-reference pattern
 * `provisionalReservation.ts` already uses (one seat per app). `AppShell`
 * `dispatchCreateCommand`s; the mounted `SingleScreen` `subscribeCreateCommands`
 * once and reacts. I11 added #5's keyframe-append variant (`addImageKeyframe`)
 * alongside #10's original `captureFrameToKeyframe` — the discriminated `type`
 * field keeps subscribers that only care about one variant untouched by the
 * other.
 *
 * Side-effect-free and React-free so it can be unit-tested in isolation.
 */

import type { GenerationPrefill } from "./generationPrefill";

/** A command delivered to the mounted Create screen. Discriminated on `type`:
 *  - `captureFrameToKeyframe` (#10): grab the current timeline frame into a new
 *    keyframe card at frame 0 (the OPENING keyframe; D3 keep-panel-state).
 *  - `addCaptureAsKeyframe` (#11, W5): grab the current timeline frame into a
 *    new keyframe card at the END of the keyframe list (the tail grid slot
 *    `nextAddPosition` picks) rather than frame 0. This is a DIFFERENT command
 *    from `captureFrameToKeyframe` (#10): #10 always lands the capture at frame
 *    0 as the opening keyframe, whereas this one appends it as a trailing
 *    keyframe. Payload-free — the subscriber computes the tail slot itself.
 *  - `addImageKeyframe` (#5): append the right-clicked image to the live
 *    keyframe panel WITHOUT remounting Create (§3-4 #5). `filePath` is the
 *    resolved source path (or `null` when native could not resolve it, so the
 *    subscriber surfaces the §6-2 missing-material note); `fileName` is its
 *    display name for the receipt note.
 *  - `setDuration` (W8, #9/#10): live-set the Create form's DURATION
 *    (`num_frames`) WITHOUT touching any other form value — the DURATION engine's
 *    keep-panel-state write for ✨ text-to-video / 📷 image-from-frame, whose
 *    comfort-ceiling target `AppShell` resolves from the published width/height.
 *    The subscriber calls `form.setNumFrames(numFrames)` and nothing else.
 *  - `addReferenceVideo` / `addReferenceAudio` (§1-107, #2/#7 on a base model
 *    whose constraints carry `reference` — MiniMax H3): put the right-clicked
 *    video / audio into the FIRST FREE card of the Reference accordion's
 *    Videos / Audios row WITHOUT remounting Create (the prompt and the other
 *    cards survive). `filePath` is `null` when native could not resolve the
 *    object's file (→ the §6-2 missing-material note). The video variant
 *    carries the ribbon's trim window (`trimQuery`) and the trimmed length
 *    (`durationSec`), both omitted when no trim applies. Enable is untouched.
 *  - `prefillMaterial` (台帳 §1-114, #2/#4/#7 on every base model except the
 *    H3 #2/#7 above): re-seed width/height/fps/DURATION per the Settings
 *    policies and attach the selected material (#2 reference video, #4 frame-0
 *    keyframe — replacing an existing frame-0 card, #7 the object's solo-rendered
 *    sound) WITHOUT remounting Create, so the prompt and every other slot
 *    survive. `selection` is the file-path-completed routed selection. */
export type CreateLiveCommand =
  | { type: "captureFrameToKeyframe" }
  | { type: "addCaptureAsKeyframe" }
  | { type: "addImageKeyframe"; filePath: string | null; fileName: string }
  | { type: "setDuration"; numFrames: number }
  | {
      type: "addReferenceVideo";
      filePath: string | null;
      fileName: string;
      trimQuery?: Record<string, string>;
      durationSec?: number;
    }
  | { type: "addReferenceAudio"; filePath: string | null; fileName: string }
  | {
      type: "prefillMaterial";
      intent: PrefillMaterialIntent;
      selection: GenerationPrefill["selection"];
    };

/** The three Single-bound material intents `prefillMaterial` carries. */
export type PrefillMaterialIntent = "reference-video" | "image-to-video" | "audio-to-video";

type CreateCommandListener = (command: CreateLiveCommand) => void;

const listeners = new Set<CreateCommandListener>();

/** Deliver a command to every current subscriber (synchronously). No-op when
 * nothing is subscribed — e.g. a dispatch that somehow races Create's mount;
 * in production Create is always mounted, so a listener is present. */
export function dispatchCreateCommand(command: CreateLiveCommand): void {
  for (const listener of listeners) {
    listener(command);
  }
}

/** Subscribe to live Create commands; returns an unsubscribe function (the
 * usual effect-cleanup shape). */
export function subscribeCreateCommands(listener: CreateCommandListener): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

/** Test-only: drop every subscriber so each test starts from a clean channel. */
export function resetCreateLiveCommands(): void {
  listeners.clear();
}
