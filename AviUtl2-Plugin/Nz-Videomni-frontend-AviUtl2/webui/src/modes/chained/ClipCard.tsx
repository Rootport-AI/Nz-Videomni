import { useStrings } from "../../i18n/LanguageContext";
import { TokenCounter, isOverTokenLimit } from "../../shell/TokenCounter";
import { DurationField } from "../single/CommonGenerationFields";
import { formatDurationHint, LTX_FRAME_GRID } from "../single/paramUtils";
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
  /** §1-111: token count of this clip's composed prompt (shared + own, as the
   * server composes it) and the engine's limit — `null`/omitted count shows no
   * counter. */
  tokenCount?: number | null | undefined;
  tokenLimit?: number | null | undefined;
  /** §1-106: the duration slider's grid step and bounds
   * (`useChainForm.clipFrameLimits`). Omitted = LTX's 8n+1 in
   * `[MIN_CLIP_NUM_FRAMES, MAX_CLIP_NUM_FRAMES]`. */
  frameLimits?: { min: number; max: number; step: number } | undefined;
  /** Ruling 84-6: show the one-line audio note under the duration
   * (`useChainForm.clipAudioDriftFlags[index]`). Omitted = hidden. */
  showAudioDriftNote?: boolean | undefined;
}

/** One clip's card in Chain mode's clip list: an optional per-clip prompt
 * that the server appends after the shared prompt for this clip's Stage-1
 * pass (blank = the shared prompt alone; Stage-2 always uses the shared
 * prompt alone) and its
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
  tokenCount,
  tokenLimit,
  frameLimits = { min: MIN_CLIP_NUM_FRAMES, max: MAX_CLIP_NUM_FRAMES, step: LTX_FRAME_GRID.step },
  showAudioDriftNote = false,
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
        <span className="prompt-field-head">
          <span className="field-label">{strings.chained.clipPromptLabel}</span>
          <TokenCounter count={tokenCount} limit={tokenLimit} />
        </span>
        <textarea
          className={`clip-prompt-input${isOverTokenLimit(tokenCount, tokenLimit) ? " input-warning" : ""}`}
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
        min={frameLimits.min}
        max={frameLimits.max}
        step={frameLimits.step}
        disabled={disabled}
        onChange={onNumFramesChange}
        hint={formatDurationHint(clip.numFrames, frameRate)}
      />
      {showAudioDriftNote && <p className="clip-note">{strings.chained.audioDriftNote}</p>}
    </li>
  );
}
