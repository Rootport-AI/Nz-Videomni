import { useStrings } from "../../i18n/LanguageContext";
import { REPEAT_MAX, REPEAT_MIN } from "./repeatRun";

export interface RepeatCountFieldProps {
  value: string;
  onChange: (text: string) => void;
  onBlur: () => void;
  disabled: boolean;
}

/** §1-80 Repeat count: the one-line `Repeat count: [n]` field under the
 * Single / Chained Generate button (`GenerateButtonBar`'s `belowButton`). */
export function RepeatCountField({ value, onChange, onBlur, disabled }: RepeatCountFieldProps) {
  const strings = useStrings();
  return (
    <label className="field field-inline repeat-count-field">
      <span className="field-label">{strings.repeatRun.countLabel}</span>
      <input
        type="number"
        min={REPEAT_MIN}
        max={REPEAT_MAX}
        step={1}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value)}
        onBlur={onBlur}
      />
    </label>
  );
}
