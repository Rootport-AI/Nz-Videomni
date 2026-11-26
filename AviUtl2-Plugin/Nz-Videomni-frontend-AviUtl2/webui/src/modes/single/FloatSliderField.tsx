import type { CSSProperties } from "react";

/** One guide line drawn above the range input (see {@link FloatSliderFieldProps.markers}). */
export interface FloatSliderMarker {
  value: number;
  /** The tooltip on the line. */
  title: string;
}

export interface FloatSliderFieldProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  disabled: boolean;
  onChange: (value: number) => void;
  /** Lay the label, range and number out on ONE row
   * (`.field-inline`) instead of stacked. Omitted = stacked, as NAG's four
   * sliders have always been. */
  inline?: boolean;
  /** A tooltip on the whole row (the `<label>`), e.g. why a disabled slider
   * cannot be moved — the same `title` shape as Create's fixed fps row.
   * Omitted = no tooltip. */
  title?: string | undefined;
  /** Guide lines over the range input, drawn exactly like the width/height
   * sliders' comfort marker (`CommonGenerationFields.tsx`'s `SizeSlider`: the
   * same `size-marker size-marker-comfort` classes, so the same
   * `--marker-comfort` colour). A line is drawn on `min <= value <= max`.
   * Omitted (or empty) = no wrapper and no line, so every existing caller's DOM
   * stays exactly what it was. */
  markers?: readonly FloatSliderMarker[] | undefined;
}

/**
 * A single float-valued range+number pair (D5): unlike `CommonGenerationFields
 * .tsx`'s `SizeFields`/`DurationField` (whose integer range+number snap onto
 * an app-specific grid via `isStepEvent`, branching on which control fired),
 * this is ONE unconditional path for both inputs — parse, ignore a
 * non-finite result (e.g. a momentarily-empty number field), then clamp into
 * `[min, max]`. Used by `shell/NagAccordion.tsx` (`nag_scale`/`vsf_scale`/
 * `nag_tau`/`nag_alpha` — `nag_scale` and `vsf_scale` render conditionally
 * on `nag.method`) and by `CommonGenerationFields.tsx`'s `SamplingFields`
 * (in its `inline` form); pulled into its own file the same way
 * `ReferenceStrengthField.tsx` was, rather than folded into the already
 * 400+-line `CommonGenerationFields.tsx`.
 */
export function FloatSliderField({
  label,
  value,
  min,
  max,
  step,
  disabled,
  onChange,
  inline = false,
  title,
  markers,
}: FloatSliderFieldProps) {
  const handleChange = (raw: string) => {
    const v = Number(raw);
    if (!Number.isFinite(v)) return;
    onChange(Math.min(max, Math.max(min, v)));
  };

  const range = (
    <input
      type="range"
      min={min}
      max={max}
      step={step}
      value={value}
      disabled={disabled}
      onChange={(e) => handleChange(e.target.value)}
    />
  );
  // `SizeSlider`'s rule: drawn on the closed interval, nothing on a degenerate range.
  const span = max - min;
  const drawn = (markers ?? []).filter((marker) => span > 0 && marker.value >= min && marker.value <= max);

  return (
    <label className={inline ? "field field-inline" : "field"} title={title}>
      <span className="field-label">{label}</span>
      {drawn.length > 0 ? (
        <span className="size-slider-wrap size-slider-wrap-marked">
          {range}
          {drawn.map((marker) => (
            <span
              key={marker.value}
              className="size-marker size-marker-comfort"
              data-size-marker="comfort"
              data-marker-value={marker.value}
              title={marker.title}
              aria-hidden="true"
              // The same unitless 0..1 custom property `SizeSlider` injects.
              style={{ "--marker-pos": (marker.value - min) / span } as CSSProperties}
            />
          ))}
        </span>
      ) : (
        range
      )}
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
