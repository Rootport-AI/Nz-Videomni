/**
 * 画角拡張（Outpainting）の快適トークン予算を、サーバー配信の
 * `limits.comfort_budgets[系統].outpaint_budget` から1つ決める解決器（§3-135）。
 *
 * **`null` ＝ 線が無い ＝ 警告を出さない**。未較正の系統に「仮の数字」を当てて
 * 分かったふりの表示をしない、というオーナー裁定（2026-09-05）。以前フロントが
 * 持っていた据え置き値 40,000 は概念ごと廃止した。
 *
 * このファイルが `shell/comfortTable.ts` と別モジュールなのは、Chained 専用の
 * `shell/tokenBudget.ts` が分かれているのと同じ**軸の分離**である。数値の正本は
 * バックエンドの `Docs/COMFORT_LIMIT_TABLE.md` §9。
 */
import type { AppLimits } from "../api/types";

/**
 * 系統名から画角拡張の予算を引く。線が無ければ `null`（＝呼び手は警告を出さない）。
 *
 * `rows` / `requires` は**一切見ない**固定線である（裁定 J1・2026-09-05）。
 * `comfortTable.ts` の `resolveComfortRow` が行う加速構成との照合とは無関係で、
 * 単発（Create）・連結（Chained）の予算とも**別軸**——同じトークン式を使うが
 * ワークロードが違う（`Docs/COMFORT_LIMIT_TABLE.md` §9.4）。一方を他方で
 * 置き換えないこと。
 *
 * 空文字を `undefined` と同じ「まだ分からない」として扱うのは、
 * `activeEngineFamily` が未確定を `""` で表すため（`resolveComfortRow` も同じ
 * 2値を同じ意味で見る）。
 *
 * 素の object index で足りる: `"constructor"` のような原型の鍵を系統名として
 * 渡しても、`?.outpaint_budget` が `undefined` になり下の型ガードで `null` へ
 * 落ちる（先例 = `comfortTable.ts` の `table[engineFamily]` 照合）。
 */
export function resolveOutpaintComfortBudget(limits: AppLimits, engineFamily: string | undefined): number | null {
  if (engineFamily === undefined || engineFamily === "") return null;
  const published = limits.comfort_budgets?.[engineFamily]?.outpaint_budget;
  return typeof published === "number" && Number.isFinite(published) && published > 0 ? published : null;
}
