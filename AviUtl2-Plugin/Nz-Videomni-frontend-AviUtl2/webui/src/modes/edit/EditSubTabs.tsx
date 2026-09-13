import { useStrings } from "../../i18n/LanguageContext";
import type { EditSubTabsDisabled } from "../../shell/featureScope";

/** The Edit tab's own sub-modes — the ones that actually have a panel behind
 * them. Exported because `EditScreen` owns the selection state.
 *
 * `"inpainting"` joined on 2026-09-14 (台帳 §3-55): it was the mock id this
 * union deliberately excluded until then, and the exclusion ended the moment a
 * panel appeared behind it — exactly the promotion `shell/ModeTabs.tsx` made
 * for Toolbox on 2026-09-11 (and this file made for Edit itself on
 * 2026-08-09). With it, `MockSubTabId` and the two-armed `EditSubTabSpec`
 * union are gone: there is no mock left under Edit to keep them for. */
export type EditSubMode = "retake" | "outpainting" | "inpainting";

/** One sub-tab's spec. A plain interface again (it was a disabled-discriminated
 * union while a mock id existed): every id here has a panel, and `disabled`
 * now means one thing only — the LOADED base model's engine cannot run it. */
interface EditSubTabSpec {
  id: EditSubMode;
  label: string;
  disabled: boolean;
}

export interface EditSubTabsProps {
  mode: EditSubMode;
  onChange: (mode: EditSubMode) => void;
  /** §3-98 P5 / §3-102: サブタブのうち、**読み込み中のベースモデルの
   * エンジンが実行できない**もの（`shell/featureScope.ts` の
   * `editSubTabsDisabledFor`）。`ModeTabs` の `disabledModes` と同じ考え方で、
   * この部品はベースモデルを一切知らない —— 渡された真偽値を描くだけ。
   * 省略時は3つとも有効（＝この機能が存在しなかった頃と一字も変わらない）。 */
  disabled?: EditSubTabsDisabled;
}

/** Edit タブ配下のサブタブ列（2026-08-09 新設）。並びは Retake / Outpainting /
 * Inpainting。無効タブは `<button disabled>` を描くだけで **onClick ハンドラ
 * 自体を付与しない**（`ModeTabs.tsx` の作法）。
 *
 * 2026-09-14（台帳 §3-55）に Inpainting が実体化したので、**灰色になる理由は
 * 1 つだけ**になった ——「このベースモデルでは動かない」。それまであった
 * 「まだ作っていない（モック）」の側は消え、灰色のサブタブには必ず理由の
 * ツールチップが付く。
 *
 * パネル側に `role="tabpanel"` は **付けない**。アプリの既存テスト群が
 * `getByRole("tabpanel")` の単数取得で「いま見えている画面」を掴む前提で
 * 書かれており、Edit 表示中に tabpanel が2つ見えるとその前提が壊れるため。
 * aria の紐付けも既存のメインタブと同水準（`role="tab"` と `aria-selected`
 * まで）に留め、`aria-controls` 等は付けない。 */
export function EditSubTabs({ mode, onChange, disabled }: EditSubTabsProps) {
  const strings = useStrings();
  const unavailable = strings.edit.unavailableOnBaseModel;
  // 灰色になったサブタブ → その理由文。灰色でないタブはここに載らないので、下の
  // `reasons[tab.id]` が undefined になり、ツールチップ属性そのものが付かない
  // （§3-55 以前は、理由の無い灰色＝Inpainting のモックがこの経路を通っていた）。
  const reasons: Partial<Record<string, string>> = {
    ...(disabled?.retake ? { retake: unavailable.retake } : {}),
    ...(disabled?.outpainting ? { outpainting: unavailable.outpainting } : {}),
    ...(disabled?.inpainting ? { inpainting: unavailable.inpainting } : {}),
  };
  const tabs: EditSubTabSpec[] = [
    { id: "retake", label: strings.edit.subTabs.retake, disabled: disabled?.retake ?? false },
    {
      id: "outpainting",
      label: strings.edit.subTabs.outpainting,
      disabled: disabled?.outpainting ?? false,
    },
    {
      id: "inpainting",
      label: strings.edit.subTabs.inpainting,
      disabled: disabled?.inpainting ?? false,
    },
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
