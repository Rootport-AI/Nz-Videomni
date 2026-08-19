export interface FloatSliderFieldProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  disabled: boolean;
  onChange: (value: number) => void;
}

/**
 * A single float-valued range+number pair (D5): unlike `CommonGenerationFields
 * .tsx`'s `SizeFields`/`DurationField` (whose integer range+number snap onto
 * an app-specific grid via `isStepEvent`, branching on which control fired),
 * this is ONE unconditional path for both inputs — parse, ignore a
 * non-finite result (e.g. a momentarily-empty number field), then clamp into
 * `[min, max]`. Used four times by `shell/NagAccordion.tsx` (`nag_scale`/
 * `vsf_scale`/`nag_tau`/`nag_alpha`) — `nag_scale` and `vsf_scale` render
 * conditionally on `nag.method`, so at most three of the four are ever on
 * screen at once; pulled into its own file the same way
 * `ReferenceStrengthField.tsx` was, rather than folded into the already
 * 400+-line `CommonGenerationFields.tsx`.
 */
export function FloatSliderField({ label, value, min, max, step, disabled, onChange }: FloatSliderFieldProps) {
  const handleChange = (raw: string) => {
    const v = Number(raw);
    if (!Number.isFinite(v)) return;
    onChange(Math.min(max, Math.max(min, v)));
  };

  return (
    <label className="field">
      <span className="field-label">{label}</span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => handleChange(e.target.value)}
      />
      <input
        type="number"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => handleChange(e.target.value)}
      />
    </label>
  );
}
