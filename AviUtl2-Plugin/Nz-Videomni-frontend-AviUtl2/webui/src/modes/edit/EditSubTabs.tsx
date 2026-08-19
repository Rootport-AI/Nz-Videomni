import { useStrings } from "../../i18n/LanguageContext";

/** The Edit tab's own sub-modes — the ones that actually have a panel behind
 * them. Exported because `EditScreen` owns the selection state. */
export type EditSubMode = "retake" | "outpainting";

/** モックサブタブ（Inpainting）の id。`EditSubMode` には意図的に含めない —
 * `shell/ModeTabs.tsx` の `MockTabId` とまったく同じ理由で、パネルを持たない
 * 「見た目だけの枠」を実体のあるモードと同じ型に混ぜると、`EditScreen` の
 * hidden 分岐が存在しないパネルを持つことになる（将来 disabled を外す日まで
 * 型で塞ぐ）。このモジュール内に閉じており、外部へは export しない。 */
type MockSubTabId = "inpainting";

/** disabled を判別子にした union（`ModeTabs.tsx` の `ModeTabSpec` と同型）。
 * 無効タブ側だけが `MockSubTabId` を許すので、`onChange(tab.id)` を呼ぶ有効
 * タブ側では `tab.id` が `EditSubMode` に自動で絞られる。 */
type EditSubTabSpec =
  | { id: EditSubMode; label: string; disabled?: false }
  | { id: MockSubTabId; label: string; disabled: true };

export interface EditSubTabsProps {
  mode: EditSubMode;
  onChange: (mode: EditSubMode) => void;
}

/** Edit タブ配下のサブタブ列（2026-08-09 新設）。並びは Retake / Outpainting /
 * Inpainting で、Inpainting だけが `disabled` のモック——`<button disabled>` を
 * 描くだけで **onClick ハンドラ自体を付与しない**（`ModeTabs.tsx` の作法）。
 *
 * パネル側に `role="tabpanel"` は **付けない**。アプリの既存テスト群が
 * `getByRole("tabpanel")` の単数取得で「いま見えている画面」を掴む前提で
 * 書かれており、Edit 表示中に tabpanel が2つ見えるとその前提が壊れるため。
 * aria の紐付けも既存のメインタブと同水準（`role="tab"` と `aria-selected`
 * まで）に留め、`aria-controls` 等は付けない。 */
export function EditSubTabs({ mode, onChange }: EditSubTabsProps) {
  const strings = useStrings();
  const tabs: EditSubTabSpec[] = [
    { id: "retake", label: strings.edit.subTabs.retake },
    { id: "outpainting", label: strings.edit.subTabs.outpainting },
    { id: "inpainting", label: strings.edit.subTabs.inpainting, disabled: true },
  ];

  return (
    <nav className="edit-subtabs" role="tablist" aria-label={strings.edit.subTabsAriaLabel}>
      {tabs.map((tab) =>
        tab.disabled ? (
          <button key={tab.id} type="button" role="tab" aria-selected={false} disabled className="edit-subtab">
            {tab.label}
          </button>
        ) : (
          <button
            key={tab.id}
            type="button"
            role="tab"
            aria-selected={mode === tab.id}
            className={`edit-subtab${mode === tab.id ? " edit-subtab--active" : ""}`}
            onClick={() => onChange(tab.id)}
          >
            {tab.label}
          </button>
        ),
      )}
    </nav>
  );
}
