import { useStrings } from "../i18n/LanguageContext";
import { LoraChips } from "../lora/LoraChips";
import { isOverTokenLimit, TokenCounter } from "./TokenCounter";

export interface PromptBarProps {
  value: string;
  onChange: (value: string) => void;
  /** IC-LoRA UI redesign (第5波): forwarded straight to `LoraChips` — see its
   * own doc comment. `AppShell` passes the control-LoRA name set here, only
   * while `mode === "single"`. */
  hiddenNames?: ReadonlySet<string> | undefined;
  /** 台帳 §1-104: forwarded straight to `LoraChips` (hides each chip's audio
   * side). */
  audioStrengthHidden?: boolean | undefined;
  /** IC-LoRA UI redesign (第5波): IME composition boundary, forwarded to the
   * `<textarea>` so `AppShell`'s control-LoRA auto-migration effect can defer
   * while a Japanese (or other IME) composition is in progress — reading the
   * partially-composed text as a "typed tag" mid-conversion would migrate on
   * an intermediate, not-yet-final string. */
  onCompositionStart?: (() => void) | undefined;
  onCompositionEnd?: (() => void) | undefined;
  /** §1-111: token count of the shared prompt (`<lora:>` stripped) and the
   * engine's limit; see `TokenCounter`. Omitted/`null` count = no counter. */
  tokenCount?: number | null | undefined;
  tokenLimit?: number | null | undefined;
}

/** The prompt input, shared across Create and Chain (Mock/AVIUTL2_DESIGN_BRIEF.md
 * §11: "プロンプト欄はタブの外・上部に一本化"). Lives above the mode tabs'
 * body — owned by `AppShell`, not by either mode screen — so switching
 * modes never loses what was typed.
 * M5 adds `LoraChips` directly under the field: a chip per `<lora:...>` tag
 * found in `value`, kept in sync with the prompt string bidirectionally (the
 * prompt is the single source of truth — see `lora/loraTags.ts`).
 *
 * 2026-07-30 owner feedback: the Chain-mode note slot ("style LoRA tags also
 * apply in Chain mode", §3-3) was removed together with its `promptBar.chainNote`
 * string — with it gone this component no longer depends on the current mode at
 * all, so the `mode` prop is gone too. */
export function PromptBar({ value, onChange, hiddenNames, audioStrengthHidden, onCompositionStart, onCompositionEnd, tokenCount, tokenLimit }: PromptBarProps) {
  const strings = useStrings();

  return (
    <div className="prompt-bar">
      <label className="field prompt-bar-field">
        <span className="prompt-field-head">
          <span className="field-label">{strings.promptBar.label}</span>
          <TokenCounter count={tokenCount} limit={tokenLimit} />
        </span>
        <textarea
          className={`prompt-input${isOverTokenLimit(tokenCount, tokenLimit) ? " input-warning" : ""}`}
          value={value}
          placeholder={strings.promptBar.placeholder}
          rows={3}
          onChange={(e) => onChange(e.target.value)}
          onCompositionStart={onCompositionStart}
          onCompositionEnd={onCompositionEnd}
        />
      </label>
      <LoraChips prompt={value} onChange={onChange} hiddenNames={hiddenNames} audioStrengthHidden={audioStrengthHidden} />
    </div>
  );
}
