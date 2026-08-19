import { useStrings } from "../i18n/LanguageContext";
import type { AppMode } from "./AppShell";

export interface ModeTabsProps {
  mode: AppMode;
  onChange: (mode: AppMode) => void;
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
 * only remaining mock**. The display order is unchanged. */
export function ModeTabs({ mode, onChange }: ModeTabsProps) {
  const strings = useStrings();
  const tabs: ModeTabSpec[] = [
    { id: "toolbox", label: strings.modes.toolbox, disabled: true },
    { id: "single", label: strings.modes.single },
    { id: "chained", label: strings.modes.chained },
    { id: "edit", label: strings.modes.edit },
    { id: "inventory", label: strings.modes.inventory },
  ];

  return (
    <nav className="mode-tabs" role="tablist" aria-label={strings.common.modeAriaLabel}>
      {tabs.map((tab) =>
        tab.disabled ? (
          <button key={tab.id} type="button" role="tab" aria-selected={false} disabled className="mode-tab">
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
