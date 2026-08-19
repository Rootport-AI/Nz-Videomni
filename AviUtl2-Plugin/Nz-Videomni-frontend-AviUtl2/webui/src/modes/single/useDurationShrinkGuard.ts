/**
 * Coordinates the "shrinking DURATION (num_frames) would strand a keyframe
 * off the placeable grid" confirmation flow for the Create screen. Two paths
 * feed into it: the DURATION field itself (`onDurationChange`/
 * `onDurationCommit`) and applying a generation preset (`onApplyPreset`),
 * since presets carry their own `num_frames` and can shrink it just like a
 * manual edit. Both funnel through the same overflow check
 * (`findOverflowFrameIdxs`, keyframeGrid.ts) and the same `pendingShrink`
 * confirmation state, so the modal (owned by the caller — this hook only
 * supplies state and callbacks, never JSX) has a single shape to render
 * regardless of which path triggered it.
 *
 * Dependency-injected (`DurationShrinkGuardDeps`) rather than importing
 * `useGenerationForm`/`useKeyframes` directly, so this hook's own logic is
 * unit-testable with plain `vi.fn()` mocks instead of the full Create form.
 */
import { useCallback, useRef, useState } from "react";
import { findOverflowFrameIdxs } from "./keyframeGrid";
import type { KeyframeItem } from "./keyframeUtils";

export interface DurationShrinkGuardDeps {
  /** The form's current `num_frames`; changes live as the user drags/types. */
  numFrames: number;
  /** `useGenerationForm`'s `setNumFrames` — applies the value immediately
   * (live), independent of whether a confirmation ends up being needed. */
  setNumFrames: (value: number, snap?: boolean) => void;
  /** `useGenerationForm`'s `applyPreset` — applies a preset's full value set
   * (width/height/num_frames/crop) immediately. */
  applyPreset: (name: string) => void;
  /** Resolves a preset name to the `num_frames` it would apply, already run
   * through `snapNumFrames` by the caller (i.e. the exact value
   * `applyPreset(name)` would end up setting) — this hook never derives or
   * re-snaps it itself. Returns `null` for an unknown preset name, in which
   * case `onApplyPreset` is a no-op. */
  getPresetNumFrames: (name: string) => number | null;
  /** All keyframes, regardless of upload status — overflow is a purely
   * position-based (`frameIdx`) check, not a readiness one. */
  items: readonly KeyframeItem[];
  removeKeyframe: (id: string) => void;
  /** `config.limits.conditioning_frame_idx_multiple`. */
  multiple: number;
  /** `config.limits.conditioning_keyframe_grid_offset`. */
  offset: number;
}

export interface PendingShrink {
  kind: "duration" | "preset";
  /** Set iff `kind === "preset"`. */
  presetName?: string;
  pendingNumFrames: number;
  overflowIds: string[];
}

export interface UseDurationShrinkGuardResult {
  /** Live-applies `value` via `deps.setNumFrames` on every call (the DURATION
   * field keeps working exactly as before), while also capturing the
   * pre-edit `numFrames` the first time it's called in an edit session so a
   * later `cancelShrink` has something to restore. */
  onDurationChange: (value: number, snap: boolean) => void;
  /** Call when the DURATION edit settles (blur/pointerup/Enter). Raises
   * `pendingShrink` for the caller's modal to render ONLY when this commit
   * actually shrank the value within an edit session (an `onDurationChange`
   * ran first AND `value` is less than what `numFrames` was before that
   * session) AND doing so strands a keyframe off the grid. A commit with no
   * prior `onDurationChange` (e.g. focus→blur with no edit) or one that
   * didn't shrink the value never raises the modal, even if an out-of-range
   * pin already happens to exist — that pin isn't this commit's doing, so
   * misattributing it here would show a "shrinking to N…" modal for a commit
   * that didn't shrink anything; Generate's own out-of-range guard/reason
   * hint is what's responsible for a pre-existing stray pin. Either way
   * `value` is NOT re-applied here since `onDurationChange` already applied
   * it live. */
  onDurationCommit: (value: number, snap: boolean) => void;
  /** Call when the user picks a generation preset. Applies it immediately if
   * it doesn't shrink the placeable grid past any existing keyframe;
   * otherwise raises `pendingShrink` and withholds the preset (unlike the
   * duration path, nothing is applied live here yet). */
  onApplyPreset: (name: string) => void;
  /** Non-null while a shrink confirmation is awaiting the user's decision. */
  pendingShrink: PendingShrink | null;
  /** Confirms the pending shrink: for `"duration"`, removes the stranded
   * keyframes (the duration itself is already live-applied); for
   * `"preset"`, applies the preset and then removes the stranded keyframes. */
  confirmShrink: () => void;
  /** Cancels the pending shrink: for `"duration"`, restores `numFrames` to
   * its pre-edit value; for `"preset"`, does nothing (it was never applied).
   * Either way clears `pendingShrink` and the pre-edit snapshot. */
  cancelShrink: () => void;
}

/**
 * See module docstring. Pure state/coordination — renders nothing; the
 * confirmation modal itself and its wiring into the DURATION field/preset
 * picker are owned by the caller.
 */
export function useDurationShrinkGuard(deps: DurationShrinkGuardDeps): UseDurationShrinkGuardResult {
  const { numFrames, setNumFrames, applyPreset, getPresetNumFrames, items, removeKeyframe, multiple, offset } = deps;

  const [pendingShrink, setPendingShrink] = useState<PendingShrink | null>(null);

  // The form's `numFrames` immediately before the current duration-edit
  // session began, or null when no session is in flight. Captured lazily —
  // only on the FIRST `onDurationChange` call of a session — so a burst of
  // live drag/type updates never overwrites it with an intermediate value;
  // if it did, `cancelShrink` would restore to the wrong (mid-drag) value
  // instead of what was on screen before the user started editing.
  const preEditRef = useRef<number | null>(null);

  const onDurationChange = useCallback(
    (value: number, snap: boolean) => {
      if (preEditRef.current === null) preEditRef.current = numFrames;
      setNumFrames(value, snap);
    },
    [numFrames, setNumFrames],
  );

  const onDurationCommit = useCallback(
    (value: number, snap: boolean) => {
      void snap; // Overflow is position-based only; the snap flag doesn't affect it.

      // m2 fix: only a commit that actually shrank the value THIS edit
      // session is eligible to raise the modal. Without this gate, a
      // focus->blur with no change (or any commit while a keyframe is
      // already stranded off-grid for unrelated reasons, e.g. a missed
      // arrow-key nudge) would misreport itself as "shrinking to N frames
      // will remove..." even though nothing shrank. `preEditRef.current ===
      // null` covers "no onDurationChange ran before this commit" (no
      // session at all); `value >= preEditRef.current` covers "a session ran
      // but the value didn't end up smaller than it started".
      const shrunkThisSession = preEditRef.current !== null && value < preEditRef.current;
      if (!shrunkThisSession) {
        // Not a shrink (or no session): the edit is final as-is (already
        // live-applied by onDurationChange, if any). Dropping the snapshot
        // here — rather than only in confirmShrink/cancelShrink — is what
        // makes this a complete no-op from the guard's point of view.
        preEditRef.current = null;
        return;
      }

      const overflow = findOverflowFrameIdxs(items, value, multiple, offset);
      if (overflow.length === 0) {
        preEditRef.current = null;
        return;
      }
      // Recomputing and overwriting on every call (rather than bailing when
      // pendingShrink is already set) keeps duplicate commits — e.g. a
      // pointerup immediately followed by a blur on the same field — idempotent.
      setPendingShrink({ kind: "duration", pendingNumFrames: value, overflowIds: overflow.map((item) => item.id) });
    },
    [items, multiple, offset],
  );

  const onApplyPreset = useCallback(
    (name: string) => {
      const target = getPresetNumFrames(name);
      if (target === null) return;
      const overflow = findOverflowFrameIdxs(items, target, multiple, offset);
      if (overflow.length === 0) {
        applyPreset(name);
        return;
      }
      setPendingShrink({
        kind: "preset",
        presetName: name,
        pendingNumFrames: target,
        overflowIds: overflow.map((item) => item.id),
      });
    },
    [getPresetNumFrames, items, multiple, offset, applyPreset],
  );

  const confirmShrink = useCallback(() => {
    if (!pendingShrink) return;
    if (pendingShrink.kind === "preset") {
      // Never applied at onApplyPreset time (unlike the duration path), so it
      // has to happen here, before removing the now-stranded keyframes.
      if (pendingShrink.presetName !== undefined) applyPreset(pendingShrink.presetName);
    }
    // kind === "duration": `pendingNumFrames` was already live-applied by
    // onDurationChange's setNumFrames pass-through, so re-calling
    // setNumFrames here would be redundant — only the now-out-of-range
    // keyframes still need removing.
    for (const id of pendingShrink.overflowIds) removeKeyframe(id);
    setPendingShrink(null);
    preEditRef.current = null;
  }, [pendingShrink, applyPreset, removeKeyframe]);

  const cancelShrink = useCallback(() => {
    if (pendingShrink?.kind === "duration" && preEditRef.current !== null) {
      setNumFrames(preEditRef.current, false);
    }
    // kind === "preset": nothing was ever applied, so there's nothing to undo.
    setPendingShrink(null);
    preEditRef.current = null;
  }, [pendingShrink, setNumFrames]);

  return { onDurationChange, onDurationCommit, onApplyPreset, pendingShrink, confirmShrink, cancelShrink };
}
