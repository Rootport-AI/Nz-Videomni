import { useStrings } from "../i18n/LanguageContext";
import type { BaseModelOption } from "./useBaseModels";

export interface BaseModelSelectProps {
  /** `useBaseModels().options` — server-declared order, never re-sorted. */
  options: BaseModelOption[];
  /** `useBaseModels().current` — the in-flight target while a switch runs. */
  value: string;
  /** `serverBusy`: a pick while the server is occupied would only earn a 409. */
  disabled: boolean;
  onChange: (id: string) => void;
  className: string;
}

/**
 * The base-model dropdown (§3-97 P7), extracted from `AppShell`'s header so
 * the Settings panel can show the same control (§3-166, 2026-09-24). It holds
 * no state: both places render it from the ONE `useBaseModels()` call in
 * `AppShell` and route a pick through the same `handleBaseModelChange`, so the
 * two can never disagree. The header passes `app-title-select`, Settings
 * passes `field-select`.
 */
export function BaseModelSelect({ options, value, disabled, onChange, className }: BaseModelSelectProps) {
  const strings = useStrings();
  return (
    <select
      className={className}
      aria-label={strings.toolVersion.ariaLabel}
      value={value}
      disabled={disabled}
      onChange={(e) => onChange(e.target.value)}
    >
      {options.length === 0 ? (
        <option value="">{strings.toolVersion.unknown}</option>
      ) : (
        options.map((option) => (
          <option key={option.id} value={option.id}>
            {option.installed
              ? option.displayName
              : option.present
                ? strings.toolVersion.optionPartial(option.displayName)
                : strings.toolVersion.optionNotInstalled(option.displayName)}
          </option>
        ))
      )}
    </select>
  );
}
