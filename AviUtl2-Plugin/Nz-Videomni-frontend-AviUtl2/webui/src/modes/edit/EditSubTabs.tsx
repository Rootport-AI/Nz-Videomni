import { useStrings } from "../../i18n/LanguageContext";
import type { EditSubTabsDisabled } from "../../shell/featureScope";

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
 *
 * 有効タブ側は `disabled?: boolean` —— 当初は `disabled?: false`（＝無効に
 * できるのはモック id だけ）だったが、ベースモデルのフィーチャ範囲で
 * **実体のあるサブタブも灰色になりうる**ようになったので型を広げた
 * （`subTabsDisabled`）。`disabled: true` は依然リテラル型なので、
 * `tab.disabled` が falsy な枝ではモック側の腕が消え、`onChange(tab.id)` を
 * 呼ぶ側の `tab.id` は `EditSubMode` に自動で絞られる —— 型で守られている
 * ものは変わっていない。 */
type EditSubTabSpec =
  | { id: EditSubMode; label: string; disabled?: boolean }
  | { id: MockSubTabId; label: string; disabled: true };

export interface EditSubTabsProps {
  mode: EditSubMode;
  onChange: (mode: EditSubMode) => void;
  /** §3-98 P5 / §3-102: 実体のあるサブタブのうち、**読み込み中のベースモデルの
   * エンジンが実行できない**もの（`shell/featureScope.ts` の
   * `editSubTabsDisabledFor`）。`ModeTabs` の `disabledModes` と同じ考え方で、
   * この部品はベースモデルを一切知らない —— 渡された真偽値を描くだけ。
   * 省略時はどちらも有効（＝この機能が存在しなかった頃と一字も変わらない）。 */
  disabled?: EditSubTabsDisabled;
}

/** Edit タブ配下のサブタブ列（2026-08-09 新設）。並びは Retake / Outpainting /
 * Inpainting で、Inpainting は**常に** `disabled` のモック。無効タブは
 * `<button disabled>` を描くだけで **onClick ハンドラ自体を付与しない**
 * （`ModeTabs.tsx` の作法）。
 *
 * 灰色になる理由は 2 つあるが、**見た目は 1 種類**に留める（`ModeTabs` と同じ
 * 判断）——「まだ作っていない」のか「このベースモデルでは動かない」のかは
 * ツールチップが伝えることで、2 つ目のスタイルを増やす話ではない。理由の
 * ツールチップが付くのはベースモデル起因の側だけで、モックには付かない。
 *
 * パネル側に `role="tabpanel"` は **付けない**。アプリの既存テスト群が
 * `getByRole("tabpanel")` の単数取得で「いま見えている画面」を掴む前提で
 * 書かれており、Edit 表示中に tabpanel が2つ見えるとその前提が壊れるため。
 * aria の紐付けも既存のメインタブと同水準（`role="tab"` と `aria-selected`
 * まで）に留め、`aria-controls` 等は付けない。 */
export function EditSubTabs({ mode, onChange, disabled }: EditSubTabsProps) {
  const strings = useStrings();
  const unavailable = strings.edit.unavailableOnBaseModel;
  // ベースモデル起因で灰色になったサブタブ → その理由文。モック（Inpainting）は
  // ここに載らないので、下の `reasons[tab.id]` が undefined になり、ツールチップ
  // 属性そのものが付かない。
  const reasons: Partial<Record<string, string>> = {
    ...(disabled?.retake ? { retake: unavailable.retake } : {}),
    ...(disabled?.outpainting ? { outpainting: unavailable.outpainting } : {}),
  };
  const tabs: EditSubTabSpec[] = [
    { id: "retake", label: strings.edit.subTabs.retake, disabled: disabled?.retake ?? false },
    {
      id: "outpainting",
      label: strings.edit.subTabs.outpainting,
      disabled: disabled?.outpainting ?? false,
    },
    { id: "inpainting", label: strings.edit.subTabs.inpainting, disabled: true },
  ];

  return (
    <nav className="edit-subtabs" role="tablist" aria-label={strings.edit.subTabsAriaLabel}>
      {tabs.map((tab) =>
        tab.disabled ? (
          <button
            key={tab.id}
            type="button"
            role="tab"
            aria-selected={false}
            disabled
            className="edit-subtab"
            {...(reasons[tab.id] ? { title: reasons[tab.id] } : {})}
          >
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
