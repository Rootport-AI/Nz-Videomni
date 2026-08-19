import { useStrings } from "../../i18n/LanguageContext";
import { nextAddPosition, placeableSlotCount } from "./keyframeGrid";
import { KeyframeCard } from "./KeyframeCard";
import { KeyframeTimeline } from "./KeyframeTimeline";
import type { UseKeyframesResult } from "./useKeyframes";

export interface KeyframesPanelProps {
  keyframes: UseKeyframesResult;
  /** Current form `num_frames`, bounding the placeable grid and every card's
   * FRAME input. */
  numFrames: number;
  /** Current form `frame_rate`, used only for the timeline/card "Xs" labels. */
  frameRate: number;
  /** `config.limits.conditioning_frame_idx_multiple` (default 8). */
  frameIdxMultiple: number;
  /** `config.limits.conditioning_keyframe_grid_offset` (default 1). */
  gridOffset: number;
  /** Disabled while a generation is in flight/reserved, mirroring the rest
   * of the form's fields. */
  disabled: boolean;
}

/**
 * The keyframe timeline rework's (2026-07-18) I2V keyframe panel: a
 * {@link KeyframeTimeline} bar (draggable/keyboard-steppable pins) above a
 * column of {@link KeyframeCard}s (thumbnail + FRAME/STRENGTH editor +
 * per-card capture/replace/remove), and a single full-width "Add keyframe"
 * action that lands a new empty card at `nextAddPosition` on the server grid
 * (image capture/pick now live on the card itself via `onCaptureInto`/
 * `onPickFileFor`, not in this add row). The panel's effective cap is the
 * tighter of the server's `max_conditioning_images` and how many grid
 * positions the current `num_frames` can even hold (`placeableSlotCount`), so
 * the "N / max" counter and the add button agree on when the timeline is
 * genuinely full. Lives inside `GenerationForm`; `SingleScreen` owns the
 * duration-shrink guard and the out-of-range Generate blocker that surround
 * it.
 */
export function KeyframesPanel({ keyframes, numFrames, frameRate, frameIdxMultiple, gridOffset, disabled }: KeyframesPanelProps) {
  const strings = useStrings();
  const t = strings.single.keyframes;

  const items = keyframes.items;
  const occupied = new Set(items.map((item) => item.frameIdx));
  // The counter/cap clamps against BOTH the server's own image cap and the
  // number of grid positions the current duration can hold — a short duration
  // (e.g. numFrames=9 -> only {0, 1}) tightens "max" below the server cap.
  const effectiveMax = Math.min(keyframes.maxItems, placeableSlotCount(numFrames, frameIdxMultiple, gridOffset));
  const nextPos = nextAddPosition(occupied, numFrames, frameIdxMultiple, gridOffset);
  const atCapacity = items.length >= effectiveMax;
  // `nextPos === null` means there's no free grid slot after the last pin even
  // though the count cap isn't reached yet (a distinct, more actionable state
  // than "full" — move/remove a pin rather than nothing you can do).
  const noRoom = nextPos === null;
  const addDisabled = disabled || atCapacity || noRoom;

  return (
    <div className="field keyframes-panel">
      <span className="field-label keyframes-label-row">
        {t.label}
        <span className="field-hint keyframes-count">{t.count(items.length, effectiveMax)}</span>
      </span>

      <KeyframeTimeline
        items={items}
        numFrames={numFrames}
        frameRate={frameRate}
        multiple={frameIdxMultiple}
        offset={gridOffset}
        disabled={disabled}
        onCommitFrameIdx={keyframes.setFrameIdx}
      />

      {items.length > 0 && (
        <ul className="keyframe-list">
          {items.map((item) => (
            <KeyframeCard
              key={item.id}
              item={item}
              numFrames={numFrames}
              frameRate={frameRate}
              multiple={frameIdxMultiple}
              offset={gridOffset}
              occupiedOthers={new Set(items.filter((other) => other.id !== item.id).map((other) => other.frameIdx))}
              disabled={disabled}
              onCommitFrameIdx={keyframes.setFrameIdx}
              onStrengthChange={keyframes.setStrength}
              onRemove={keyframes.remove}
              onReplaceFile={keyframes.replaceFile}
              onPickFileFor={keyframes.pickReplaceFile}
              onCaptureInto={keyframes.captureIntoCard}
              isCapturing={keyframes.isCapturing}
              isPicking={keyframes.isPicking}
            />
          ))}
        </ul>
      )}

      <div className="keyframes-add-row">
        <button
          type="button"
          className="keyframes-add-button"
          disabled={addDisabled}
          onClick={() => {
            if (nextPos !== null) keyframes.addEmpty(nextPos);
          }}
        >
          {t.addKeyframeButton(items.length, effectiveMax)}
        </button>
      </div>
      {atCapacity ? (
        <span className="field-hint">{t.maxReached(effectiveMax)}</span>
      ) : (
        noRoom && <span className="field-hint">{t.addNoRoomHint}</span>
      )}
    </div>
  );
}
