import { DEFAULT_REFERENCE_STRENGTH, MAX_REFERENCE_STRENGTH, MIN_REFERENCE_STRENGTH } from "../chained/chainUtils";

export interface ReferenceStrengthFieldProps {
  label: string;
  value: number | null;
  onChange: (value: number | null) => void;
  disabled: boolean;
}

/** One opt-in `[0,1]` strength override: a checkbox that flips `value`
 * between `null` ("not set" — the field is omitted from the request) and a
 * seeded {@link DEFAULT_REFERENCE_STRENGTH}, plus a range input shown only
 * while enabled. `0.0` is a valid, distinct value from "unset", so the
 * range's presence — not its numeric value — is what `onChange(null)` vs.
 * `onChange(number)` tracks.
 *
 * Shared by Chain's `ReferenceVideoSection` (`modes/chained/ChainedScreen.tsx`,
 * extracted from there) and Create's `ReferenceVideoSection`
 * (`modes/single/GenerationForm.tsx`, N3) — both drive the same
 * `conditioning_attention_strength`/`reference_video_strength` request
 * fields off a reference-video CONTROL IC-LoRA. */
export function ReferenceStrengthField({ label, value, onChange, disabled }: ReferenceStrengthFieldProps) {
  return (
    <div className="field">
      <label className="field field-inline">
        <input
          type="checkbox"
          checked={value !== null}
          disabled={disabled}
          onChange={(e) => onChange(e.target.checked ? DEFAULT_REFERENCE_STRENGTH : null)}
        />
        <span className="field-label">{label}</span>
      </label>
      {value !== null && (
        <label className="field">
          <input
            type="range"
            min={MIN_REFERENCE_STRENGTH}
            max={MAX_REFERENCE_STRENGTH}
            step={0.05}
            value={value}
            disabled={disabled}
            onChange={(e) => onChange(Number(e.target.value))}
          />
          <span className="field-hint">{value.toFixed(2)}</span>
        </label>
      )}
    </div>
  );
}
