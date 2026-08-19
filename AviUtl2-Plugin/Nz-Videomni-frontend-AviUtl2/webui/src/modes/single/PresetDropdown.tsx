import { useState } from "react";
import type { AppConfig, GenerationPreset } from "../../api/types";

export interface PresetDropdownProps {
  config: AppConfig;
  onSelect: (name: string) => void;
  disabled: boolean;
  label: string;
  placeholder: string;
  /** Headings for the orientation `<optgroup>`s (U-R2). Sourced from
   * `strings.single.presets.orientation` by both call sites — this
   * component stays string-agnostic (doesn't call `useStrings` itself),
   * matching the existing "copy comes in via props" contract. */
  orientationLabels: {
    landscape: string;
    square: string;
    portrait: string;
  };
}

/** 0 = landscape (w>h), 1 = square (w===h), 2 = portrait (w<h). The primary
 * sort/grouping key for `sortPresets` and the dropdown's `<optgroup>`s. */
function orientationRank(preset: GenerationPreset): 0 | 1 | 2 {
  if (preset.width > preset.height) return 0;
  if (preset.width === preset.height) return 1;
  return 2;
}

function presetOptionLabel(name: string, preset: GenerationPreset): string {
  return `${name} (${preset.width}x${preset.height}, ${preset.num_frames}f)`;
}

/** Orders preset entries landscape → square → portrait, then by area
 * (ascending) within an orientation, then by name (so ties are resolved
 * deterministically regardless of the input order/object-key order). */
export function sortPresets(entries: [string, GenerationPreset][]): [string, GenerationPreset][] {
  return [...entries].sort(([nameA, presetA], [nameB, presetB]) => {
    const rankDiff = orientationRank(presetA) - orientationRank(presetB);
    if (rankDiff !== 0) return rankDiff;
    const areaDiff = presetA.width * presetA.height - presetB.width * presetB.height;
    if (areaDiff !== 0) return areaDiff;
    return nameA.localeCompare(nameB);
  });
}

/** Generic "apply a generation preset" `<select>`, shared by Create/Chain —
 * sourced from `config.generation_presets` (the same server-side records
 * both screens already read). Originally generalized `modes/chained/
 * ChainedScreen.tsx`'s `ChainPresetField`, a Chain-only preset dropdown with
 * the same markup/option format; that component was deleted in the
 * 2026-07-16/17 UI redesign once Chain switched to using this shared
 * component directly, so the comparisons below are to how that now-removed
 * component used to behave, not to anything currently in the tree. Its
 * persistence behavior was deliberately different from this one:
 * `ChainPresetField` used to reset the `<select>` back to the placeholder
 * after every choice (a one-shot bulk edit, mirroring `gradio_ui/
 * presets.py`'s `apply_chain_preset` — a `Dropdown.change` handler that
 * isn't a value the form "stays on"). This component instead keeps the
 * chosen option visible (Gradio's own dropdown widget persists its selected
 * value across changes) — the `<select>` is controlled by local state here
 * rather than snapping back to `""`.
 *
 * Renders nothing when `config.generation_presets` is empty (fallback config
 * not yet loaded, or a genuinely empty server list) rather than showing a
 * useless empty dropdown — matching what `ChainPresetField` used to do.
 *
 * Re-applying the SAME already-selected preset a second time (e.g. after
 * hand-editing width/height away from it) has no dedicated control — go via
 * a different preset and back (A→B→A). Each hop is a genuine `<select>`
 * value change, so both fire the `onChange` → `onSelect` auto-apply below;
 * since `applyPreset` unconditionally overwrites its governed fields (width/
 * height/numFrames/cropOutput, or on Chain all clips' numFrames) on every
 * call, landing back on A fully restores A's fields — B's intermediate
 * values are completely overwritten. This detour requires 2+ presets to
 * exist (the current config ships 6). An "Apply" button covering this case
 * was added 2026-07-17 and removed again 2026-07-18 per owner decision. */
export function PresetDropdown({
  config,
  onSelect,
  disabled,
  label,
  placeholder,
  orientationLabels,
}: PresetDropdownProps) {
  const [selected, setSelected] = useState("");
  const sortedEntries = sortPresets(Object.entries(config.generation_presets));
  if (sortedEntries.length === 0) return null;

  // Only group into <optgroup>s when 2+ orientations are actually present —
  // a single-orientation preset list (the common case) stays a flat list,
  // same as before U-R2.
  const presentRanks = new Set(sortedEntries.map(([, preset]) => orientationRank(preset)));
  const useGroups = presentRanks.size >= 2;

  return (
    <label className="field">
      <span className="field-label">{label}</span>
      <select
        className="field-select"
        disabled={disabled}
        value={selected}
        onChange={(e) => {
          const name = e.target.value;
          if (!name) return;
          setSelected(name);
          onSelect(name);
        }}
      >
        <option value="">{placeholder}</option>
        {useGroups
          ? ([0, 1, 2] as const).map((rank) => {
              const group = sortedEntries.filter(([, preset]) => orientationRank(preset) === rank);
              if (group.length === 0) return null;
              const groupLabel =
                rank === 0 ? orientationLabels.landscape : rank === 1 ? orientationLabels.square : orientationLabels.portrait;
              return (
                <optgroup key={rank} label={groupLabel}>
                  {group.map(([name, preset]) => (
                    <option key={name} value={name}>
                      {presetOptionLabel(name, preset)}
                    </option>
                  ))}
                </optgroup>
              );
            })
          : sortedEntries.map(([name, preset]) => (
              <option key={name} value={name}>
                {presetOptionLabel(name, preset)}
              </option>
            ))}
      </select>
    </label>
  );
}
