import { useStrings } from "../i18n/LanguageContext";
import {
  clampLoraStrength,
  LORA_AUDIO_STRENGTH_DEFAULT,
  LORA_AUDIO_STRENGTH_MIN,
  LORA_STRENGTH_STEP,
  parseLoraPrompt,
  removeLoraTag,
  setLoraTagAudioStrength,
  setLoraTagStrength,
} from "./loraTags";
import type { ParsedLoraTag } from "./loraTags";
import "./lora.css";

export interface LoraChipsProps {
  /** The shared prompt (`shell/PromptBar.tsx`) — the single source of truth
   * this row renders a view over. */
  prompt: string;
  onChange: (prompt: string) => void;
  /** IC-LoRA UI redesign (第5波): names to hide from the chip row without
   * touching the prompt string itself — `AppShell` passes the control-LoRA
   * name set here (only while `mode === "single"`, right up until its
   * auto-migration effect strips the matching tag out of the prompt on its
   * own) so a hand-typed control tag doesn't flash a chip for the one
   * render before the effect removes it. Purely a display filter: a hidden
   * tag's `-`/`+`/`×` never render, and it's still counted by
   * `parseLoraPrompt` for everything else (submission, migration). */
  hiddenNames?: ReadonlySet<string> | undefined;
}

/** Renders the shared prompt's `<lora:name:strength>` tags as chips right
 * under the prompt textarea, in both Create and Chain mode. Every button here
 * computes a new prompt string and hands it to `onChange`; there is no
 * separate chip state to keep in sync —
 * the next render re-derives the chip list from the (now-updated) prompt
 * (task brief §1: "プロンプト手編集とチップは双方向同期"). Renders nothing
 * when the prompt has no valid, un-hidden LoRA tags. */
export function LoraChips({ prompt, onChange, hiddenNames }: LoraChipsProps) {
  const strings = useStrings();
  const { tags } = parseLoraPrompt(prompt);
  const chips = tags.filter((tag) => tag.valid && !hiddenNames?.has(tag.name));
  if (chips.length === 0) return null;

  return (
    <ul className="lora-chip-row" aria-label={strings.common.loraTagsAriaLabel}>
      {chips.map((tag) => (
        <LoraChip
          key={`${tag.name}@${tag.index}`}
          tag={tag}
          onAdjust={(delta) => onChange(setLoraTagStrength(prompt, tag, clampLoraStrength(tag.strength + delta)))}
          onRemove={() => onChange(removeLoraTag(prompt, tag))}
          onToggleAudio={() =>
            onChange(
              setLoraTagAudioStrength(
                prompt,
                tag,
                tag.audio_strength === LORA_AUDIO_STRENGTH_MIN ? LORA_AUDIO_STRENGTH_DEFAULT : LORA_AUDIO_STRENGTH_MIN,
              ),
            )
          }
        />
      ))}
    </ul>
  );
}

interface LoraChipProps {
  tag: ParsedLoraTag;
  onAdjust: (delta: number) => void;
  onRemove: () => void;
  /** Style LoRA音声強度制御 (2026-08-02): the mute toggle's click handler —
   * see `LoraChip`'s own doc comment for the toggle rule. */
  onToggleAudio: () => void;
}

function LoraChip({ tag, onAdjust, onRemove, onToggleAudio }: LoraChipProps) {
  const strings = useStrings();
  // Style LoRA音声強度制御 (2026-08-02): "muted" is specifically
  // `audio_strength === 0` (the LOWEST valid value, not falsy-check —
  // `LORA_AUDIO_STRENGTH_MIN` is 0.0, so this IS a `=== 0` check written
  // through the constant for clarity). `undefined` (2-arg tag, audio follows
  // video) counts as "audible" for the toggle's starting state, exactly like
  // any positive audio strength — only an explicit `:0` third argument reads
  // as muted.
  const audioStrength = tag.audio_strength;
  const muted = audioStrength === LORA_AUDIO_STRENGTH_MIN;
  // Chip reorder (owner real-device review, 2026-08-02): the audio badge is
  // ALWAYS shown now — not just for a 3-argument tag — so users can learn
  // where the number lives by toggling mute and watching it move to 0.00
  // and back. When `audio_strength` is `undefined` (2-arg tag, audio follows
  // video), the badge's effective value is the video-side `strength`.
  const audioDisplay = audioStrength ?? tag.strength;
  return (
    <li className="lora-chip" data-lora-name={tag.name}>
      <span className="lora-chip-name" title={tag.name}>
        {tag.name}
      </span>
      <span className="lora-chip-video-icon" aria-hidden="true">
        🎥
      </span>
      <span className="lora-chip-strength">{tag.strength.toFixed(2)}</span>
      <button
        type="button"
        className="lora-chip-btn"
        aria-label={strings.common.decreaseStrength(tag.name)}
        onClick={() => onAdjust(-LORA_STRENGTH_STEP)}
      >
        −
      </button>
      <button
        type="button"
        className="lora-chip-btn"
        aria-label={strings.common.increaseStrength(tag.name)}
        onClick={() => onAdjust(LORA_STRENGTH_STEP)}
      >
        +
      </button>
      <button
        type="button"
        className={muted ? "lora-chip-audio-toggle lora-chip-audio-toggle-muted" : "lora-chip-audio-toggle"}
        aria-pressed={muted}
        aria-label={muted ? strings.common.unmuteLoraAudio(tag.name) : strings.common.muteLoraAudio(tag.name)}
        onClick={onToggleAudio}
      >
        {muted ? "🔇" : "🔊"}
      </button>
      <span className={muted ? "lora-chip-audio lora-chip-audio-muted" : "lora-chip-audio"}>
        {audioDisplay.toFixed(2)}
      </span>
      <button type="button" className="lora-chip-remove" aria-label={strings.common.removeLora(tag.name)} onClick={onRemove}>
        ×
      </button>
    </li>
  );
}
