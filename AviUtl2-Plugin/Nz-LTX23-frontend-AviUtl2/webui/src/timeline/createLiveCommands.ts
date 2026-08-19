/**
 * A minimal live-command channel from `AppShell.handleRoute` to the
 * always-mounted Create screen (RIGHTCLICK_REDESIGN_SPEC.md §3-6 #10, §7-2).
 *
 * Most right-click routes deliver their intent through a REMOUNT (a fresh
 * `pendingIntent` + a bumped per-mode remount token), which the target screen
 * consumes once in a mount effect. But the two keep-panel-state items — ✨
 * (#9) and 📷 (#10, D3 decision) — must NOT remount Create, so there is no
 * remount for a `pendingIntent` to ride. #9 needs nothing delivered to the
 * mounted form (it only reserves a provisional). #10 DOES: it must tell the
 * live Create screen to grab the current timeline frame into KEYFRAMES,
 * WITHOUT wiping the form state the user already has.
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
 *    The subscriber calls `form.setNumFrames(numFrames)` and nothing else. */
export type CreateLiveCommand =
  | { type: "captureFrameToKeyframe" }
  | { type: "addCaptureAsKeyframe" }
  | { type: "addImageKeyframe"; filePath: string | null; fileName: string }
  | { type: "setDuration"; numFrames: number };

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
