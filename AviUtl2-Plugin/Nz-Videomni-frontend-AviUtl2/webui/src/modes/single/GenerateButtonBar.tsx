import type { ReactNode } from "react";
import "../../styles/formAccordion.css";

export interface GenerateButtonBarProps {
  label: string;
  disabled: boolean;
  onGenerate: () => void;
  /** The always-visible duration/estimate readout next to Generate (M3 task
   * brief §5) — e.g. `useGenerationForm`'s `estimateLabel`. Omitted entirely
   * (no empty `<span>`) when the caller has nothing to show. */
  hint?: ReactNode;
}

/** Shared Generate button + estimate-hint bar: the top section of the
 * `.generation-column` panel (`styles/formAccordion.css`, imported here
 * since that panel is what this component exists to head), above
 * `GenerationPanel`'s job/preview area. Split out into its own component so
 * both Create and Chain can compose that panel without duplicating the
 * markup.
 *
 * Stacked vertically (`.generate-bar` in `styles/formAccordion.css`) rather
 * than the horizontal `.generate-row` this used to reuse: the right column
 * this bar lives in can be squeezed to ~280px by the job rail
 * (`AppShell.css`'s `.app-body`), and a horizontal row with a
 * `white-space: nowrap` hint next to a flexed button overflowed its card at
 * that width. Stacking removes the width contention — the button is always
 * full width, and the hint wraps freely underneath it. */
export function GenerateButtonBar({ label, disabled, onGenerate, hint }: GenerateButtonBarProps) {
  return (
    <div className="generate-bar">
      <button type="button" className="primary-button generate-button" disabled={disabled} onClick={onGenerate}>
        {label}
      </button>
      {hint !== undefined && <span className="field-hint generate-bar-hint">{hint}</span>}
    </div>
  );
}
