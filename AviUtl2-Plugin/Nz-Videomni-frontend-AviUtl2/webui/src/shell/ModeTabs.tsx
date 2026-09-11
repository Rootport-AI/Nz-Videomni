import { useStrings } from "../i18n/LanguageContext";
import type { AppMode } from "./AppShell";

export interface ModeTabsProps {
  mode: AppMode;
  onChange: (mode: AppMode) => void;
  /** §3-98 P5: modes the LOADED base model's engine cannot run
   * (`shell/useBaseModels.ts`'s `disabledModes`). Since 2026-09-11 this is the
   * ONLY reason a tab is greyed — the Toolbox mock, whose "not built yet"
   * greying shared this treatment from Phase 6 onwards, became a real tab that
   * day. Omitted ⇒ none. */
  disabledModes?: readonly AppMode[];
}

/** 1 タブ 1 行。2026-09-11（§3-54 物体追尾）に Toolbox が実タブへ昇格し、
 * **モックタブは 1 つも残っていない** —— それまではここに `MockTabId`（"toolbox"）と
 * `disabled` を判別子にした union があり、「パネルも remountTokens も
 * MenuTargetMode も持たない見た目だけの枠」を型で塞いでいた。Toolbox がその 3 つ
 * すべてを持った時点で塞ぐ対象が消えたので、union ごと畳んである。
 *
 * タブが灰色になる理由は、いまは 1 つだけ —— 読み込み中のベースモデルのエンジンが
 * そのタブを実行できないとき（`disabledModes`）。 */
type ModeTabSpec = { id: AppMode; label: string };

/** The mode switch (task brief §1 / Mock/AVIUTL2_DESIGN_BRIEF.md §11:
 * "5タブ→3モード＋常設ジョブレーンに再編"). Phase 6 added two disabled mock
 * tabs (Toolbox / Edit) that were visual-only placeholders. Edit became a real
 * tab on 2026-08-09 and **Toolbox on 2026-09-11** (§3-54 物体追尾), each with
 * an `AppMode` value and a mounted panel of its own — so **no mock tabs are
 * left**. The display order has never changed.
 *
 * That leaves `disabledModes` (§3-98 P5) as the ONE reason a tab is greyed: the
 * loaded base model's engine cannot run it. This component does not know what a
 * base model is — it is handed a list of mode ids and renders them disabled.
 * Deciding that list is `shell/featureScope.ts`'s `disabledModesFor`. */
export function ModeTabs({ mode, onChange, disabledModes = [] }: ModeTabsProps) {
  const strings = useStrings();
  const tabs: ModeTabSpec[] = [
    { id: "toolbox", label: strings.modes.toolbox },
    { id: "single", label: strings.modes.single },
    { id: "chained", label: strings.modes.chained },
    { id: "edit", label: strings.modes.edit },
    { id: "inventory", label: strings.modes.inventory },
  ];
  const unsupported = new Set<AppMode>(disabledModes);

  return (
    <nav className="mode-tabs" role="tablist" aria-label={strings.common.modeAriaLabel}>
      {tabs.map((tab) =>
        unsupported.has(tab.id) ? (
          <button
            key={tab.id}
            type="button"
            role="tab"
            aria-selected={false}
            disabled
            className="mode-tab"
            // Unconditional now: reaching this branch IS the base-model case,
            // since the mock tab that used to share it is gone.
            title={strings.modes.unsupportedByBaseModel}
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
