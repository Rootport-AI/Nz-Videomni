import { useStrings } from "../../i18n/LanguageContext";
import { DurationField } from "../single/CommonGenerationFields";
import { formatDurationHint } from "../single/paramUtils";
import { MAX_CLIP_NUM_FRAMES, MIN_CLIP_NUM_FRAMES } from "./chainUtils";
import type { AudioSegmentWindow, ChainClipInput } from "./chainUtils";

export interface ClipCardProps {
  clip: ChainClipInput;
  index: number;
  frameRate: number;
  disabled: boolean;
  canRemove: boolean;
  /** §1-16 長尺A2V: this clip's slice of the attached audio track
   * (`useChainForm.audioSegmentWindowsForClips[index]`), or `null`/omitted when
   * no audio is attached — which is also the only condition on the 🎵 badge
   * below. The window is computed by the caller, not here: it depends on the
   * WHOLE clip list, which one card cannot see. */
  audioWindow?: AudioSegmentWindow | null;
  onPromptChange: (value: string) => void;
  /** See `useChainForm.setClipNumFrames`'s `snap` semantics — `DurationField`
   * forwards its slider (snap=true) vs. typed (isStepEvent-derived) calls
   * straight through. */
  onNumFramesChange: (value: number, snap: boolean) => void;
  onRemove: () => void;
}

/** One clip's card in Chain mode's clip list: an optional per-clip prompt
 * override (blank = "use the shared prompt", task brief §1) and its
 * `num_frames` (8n+1, reusing `formatDurationHint` verbatim from Create's
 * `paramUtils` — the hint text is identical in meaning). Clip 0's keyframe
 * panel is rendered by the caller (`ChainClipsForm`), not here, since it's
 * the one piece of clip-card UI that only exists for index 0. */
export function ClipCard({
  clip,
  index,
  frameRate,
  disabled,
  canRemove,
  audioWindow,
  onPromptChange,
  onNumFramesChange,
  onRemove,
}: ClipCardProps) {
  const strings = useStrings();
  return (
    <li className="clip-card">
      <div className="clip-card-head">
        <span className="field-label">{strings.chained.clipLabel(index)}</span>
        {/* §1-16 長尺A2V: which part of the attached track this clip covers.
            Between the label and ❌ so the header reads "Clip 3 — 🎵 4.20-8.40s
            — ❌"; absent entirely when there is no audio. */}
        {audioWindow && (
          <span className="clip-card-audio-window" title={strings.chained.clipAudioWindow.tooltip}>
            {strings.chained.clipAudioWindow.label(audioWindow.startSec.toFixed(2), audioWindow.endSec.toFixed(2))}
          </span>
        )}
        {canRemove && (
          <button
            type="button"
            className="icon-action-button clip-remove"
            disabled={disabled}
            title={strings.chained.removeClipButton}
            aria-label={strings.chained.removeClipButton}
            onClick={onRemove}
          >
            <span aria-hidden="true">❌</span>
          </button>
        )}
      </div>

      <label className="field">
        <span className="field-label">{strings.chained.clipPromptLabel}</span>
        <textarea
          className="clip-prompt-input"
          rows={2}
          value={clip.prompt}
          placeholder={strings.chained.clipPromptPlaceholder}
          disabled={disabled}
          onChange={(e) => onPromptChange(e.target.value)}
        />
      </label>

      <DurationField
        label={strings.chained.clipNumFramesLabel}
        value={clip.numFrames}
        min={MIN_CLIP_NUM_FRAMES}
        max={MAX_CLIP_NUM_FRAMES}
        disabled={disabled}
        onChange={onNumFramesChange}
        hint={formatDurationHint(clip.numFrames, frameRate)}
      />
    </li>
  );
}
