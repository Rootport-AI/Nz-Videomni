import { useStrings } from "../i18n/LanguageContext";
import type { AppMode } from "./AppShell";

export interface ModeTabsProps {
  mode: AppMode;
  onChange: (mode: AppMode) => void;
  /** §3-98 P5: modes the LOADED base model's engine cannot run
   * (`shell/useBaseModels.ts`'s `disabledModes`). Rendered with the SAME
   * disabled treatment the Toolbox mock has had since Phase 6 — one greyed
   * tab, not two visually different kinds of unavailable — because from the
   * user's side the distinction ("not built yet" vs "not on this base model")
   * is carried by the tooltip, not by a second style. Omitted ⇒ none. */
  disabledModes?: readonly AppMode[];
}

/** モックタブの id。2026-08-09 に Edit が実タブへ昇格したので、残るモックは
 * **Toolbox の1つだけ**である。`AppMode` には意図的に含めない — パネルも
 * remountTokens も MenuTargetMode も持たない「見た目だけの枠」であり、AppMode に
 * 混ぜると AppShell の Record<AppMode, number> や JSX の hidden 分岐が実体のない
 * モードを持つことになる（将来 disabled を外す日まで型で塞ぐ）。
 * このモジュール内に閉じており、外部へは export しない。 */
type MockTabId = "toolbox";

/** disabled を判別子にした union。無効タブ側だけが MockTabId を許すので、
 * onChange(tab.id) を呼ぶ有効タブ側では tab.id が AppMode に自動で絞られる。 */
type ModeTabSpec =
  | { id: AppMode; label: string; disabled?: false }
  | { id: MockTabId; label: string; disabled: true };

/** The mode switch (task brief §1 / Mock/AVIUTL2_DESIGN_BRIEF.md §11:
 * "5タブ→3モード＋常設ジョブレーンに再編"). Phase 6 added two disabled mock
 * tabs (Toolbox / Edit) that were visual-only placeholders. 2026-08-09: **Edit
 * is now a real tab** — it has an `AppMode` value, a mounted panel
 * (`modes/edit/EditScreen.tsx`) and its own sub-tab row — so **Toolbox is the
 * only remaining mock**. The display order is unchanged.
 *
 * §3-98 P5 added a SECOND reason a tab can be greyed: the loaded base model's
 * engine cannot run it (`disabledModes`). This component does not know what a
 * base model is — it is handed a list of mode ids and renders them disabled.
 * Deciding that list is `shell/featureScope.ts`'s `disabledModesFor`. */
export function ModeTabs({ mode, onChange, disabledModes = [] }: ModeTabsProps) {
  const strings = useStrings();
  const tabs: ModeTabSpec[] = [
    { id: "toolbox", label: strings.modes.toolbox, disabled: true },
    { id: "single", label: strings.modes.single },
    { id: "chained", label: strings.modes.chained },
    { id: "edit", label: strings.modes.edit },
    { id: "inventory", label: strings.modes.inventory },
  ];
  // `Set<string>` rather than `Set<AppMode>` so the mock tab's id can be asked
  // about too without a cast; it is never in the set, and the `tab.disabled ||`
  // below is what narrows the union for the enabled branch.
  const unsupported = new Set<string>(disabledModes);

  return (
    <nav className="mode-tabs" role="tablist" aria-label={strings.common.modeAriaLabel}>
      {tabs.map((tab) =>
        tab.disabled || unsupported.has(tab.id) ? (
          <button
            key={tab.id}
            type="button"
            role="tab"
            aria-selected={false}
            disabled
            className="mode-tab"
            // Only the base-model case gets a reason: the Toolbox mock is
            // "not built yet", which its own absence of a panel already says.
            {...(unsupported.has(tab.id) ? { title: strings.modes.unsupportedByBaseModel } : {})}
          >
            {tab.label}
          </button>
        ) : (
          <button
            key={tab.id}
            type="button"
            role="tab"
            aria-selected={mode === tab.id}
            className={`mode-tab${mode === tab.id ? " mode-tab--active" : ""}`}
            onClick={() => onChange(tab.id)}
          >
            {tab.label}
          </button>
        ),
      )}
    </nav>
  );
}
