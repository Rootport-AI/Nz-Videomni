import { useStrings } from "../../i18n/LanguageContext";

/** The Toolbox tab's sub-modes (§3-164, 2026-09-24). Exported because
 * `ToolboxScreen` owns the selection state. */
export type ToolboxSubMode = "tracking" | "mp4info";

export interface ToolboxSubTabsProps {
  mode: ToolboxSubMode;
  onChange: (mode: ToolboxSubMode) => void;
}

/** Toolbox タブ配下のサブタブ列（2026-09-24 新設・§3-164）。並びは
 * Tracking / mp4 info。`modes/edit/EditSubTabs.tsx` と同じ型で、灰色になる
 * サブタブは無い（どちらの道具もベースモデルに依存しない）。
 *
 * パネル側に `role="tabpanel"` は **付けない**。アプリの既存テスト群
 * （`App.trackRoute.test.tsx` など）が `getByRole("tabpanel")` の単数取得で
 * 「いま見えている画面」を掴む前提で書かれているため（`EditSubTabs` と同じ
 * 理由）。aria の紐付けは `role="tab"` と `aria-selected` までに留める。 */
export function ToolboxSubTabs({ mode, onChange }: ToolboxSubTabsProps) {
  const strings = useStrings();
  const tabs: ReadonlyArray<{ id: ToolboxSubMode; label: string }> = [
    { id: "tracking", label: strings.toolbox.subTabs.tracking },
    { id: "mp4info", label: strings.toolbox.subTabs.mp4info },
  ];

  return (
    <nav className="toolbox-subtabs" role="tablist" aria-label={strings.toolbox.subTabs.ariaLabel}>
      {tabs.map((tab) => (
        <button
          key={tab.id}
          type="button"
          role="tab"
          aria-selected={mode === tab.id}
          className={`toolbox-subtab${mode === tab.id ? " toolbox-subtab--active" : ""}`}
          onClick={() => onChange(tab.id)}
        >
          {tab.label}
        </button>
      ))}
    </nav>
  );
}
