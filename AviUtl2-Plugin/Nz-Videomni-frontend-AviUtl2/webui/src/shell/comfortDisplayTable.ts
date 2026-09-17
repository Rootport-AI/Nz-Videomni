/**
 * Settings の「快適上限の目安」表の**形だけ**を持つ宣言モジュール——行＝解像度、
 * 列＝出典。数字はここに書かない。各セルは {@link resolveComfortCell} が
 * サーバー配信の `AppConfig.limits` から1つずつ引く。
 *
 * なぜ形だけか: レガシー表 `limits.spill_free_frames` の実体は git 追跡外の
 * `config.yaml` にあり、数値の正本（バックエンドの `Docs/COMFORT_LIMIT_TABLE.md`）
 * が「この表の値をここ以外へ書き写さないこと」と明記している。写しを作らずに
 * 表を出す方法は、配信値を描き続けることだけである。
 *
 * 唯一の例外が `static` 列＝**較正済みだが製品が配っていない**実測点で、配信値が
 * 存在しない以上ここに置くほかない。出典は定数のコメントに残す。
 *
 * 換算式はここには無い: `budget` 列は `shell/comfortTable.ts` の
 * `comfortFramesForBudget` をそのまま呼ぶ。Create の快適上限マーカーと**同じ関数**
 * ＝同じ線であることが、この表が目安として意味を持つ根拠である。
 */

import type { AppLimits } from "../api/types";
import type { Strings } from "../i18n/strings";
import { MIN_NUM_FRAMES } from "../modes/single/defaultConfig";
import type { AccelerationSettings } from "./accelerationSettings";
import { comfortFramesForBudget, resolveComfortRow } from "./comfortTable";

/** `strings.settings` のうち**素のテキスト**の鍵。列見出しが差し込みの無い
 * 1行なので、テンプレート関数の鍵をうっかり並べられないよう値の型で絞る。 */
type ComfortTextKey = {
  [K in keyof Strings["settings"]]: Strings["settings"][K] extends string ? K : never;
}[keyof Strings["settings"]];

/** 1列ぶんの数字がどこから来るか。
 *
 *  - `legacy` ＝ レガシー表 `limits.spill_free_frames` の実測値そのまま。
 *  - `budget` ＝ その系統の「全on」条件に一致する配信行（{@link resolveComfortRow}
 *    が選ぶ）のトークン予算から換算した線。
 *  - `static` ＝ 較正済みだが配信されていない実測点（{@link LTX25_Q6_FRAMES}）。
 *
 * どれも「その解像度の答えが無い」を `null`（表示は「—」）で返す。無い理由は列に
 * よって違う（測っていない／その系統に線が無い）が、読み手に見せる区別ではない。 */
export type ComfortColumnSource =
  | { readonly kind: "legacy" }
  | { readonly kind: "budget"; readonly engineFamily: string }
  | { readonly kind: "static"; readonly frames: Readonly<Record<string, number>> };

export interface ComfortDisplayColumn {
  /** React の `key` と、テストが列を名指しするための識別子。画面には出ない。 */
  readonly id: string;
  /** 列見出しの `strings.settings` の鍵。 */
  readonly labelKey: ComfortTextKey;
  readonly source: ComfortColumnSource;
}

export interface ComfortDisplayTable {
  readonly id: string;
  /** 行の解像度。鍵は `spill_free_frames` と同じ綴り（`"1280x768"`）——`legacy`
   * 列がこの文字列でそのまま引けることが、行の綴りを1つに保つ理由。 */
  readonly rows: readonly string[];
  readonly columns: readonly ComfortDisplayColumn[];
}

/** 「2.5 Q6」列の実測点。出典はバックエンドの `Docs/COMFORT_LIMIT_TABLE.md` §10。
 *
 * この表で唯一ハードコードしてよい数字である——サーバーは transformer の量子化を
 * 知らないので、この3点を配る経路が無い。載っていない解像度は測っていない＝「—」。 */
const LTX25_Q6_FRAMES: Readonly<Record<string, number>> = {
  "1280x768": 361,
  "1920x1088": 161,
  "896x1152": 313,
};

/** LTX 系の表。行は縦横比の違う6サイズで、`spill_free_frames` に無い
 * `896x1152` も含む（`legacy` 列だけが「—」になる、という**列ごとに答えが違う**
 * ことがそのまま見える並び）。 */
const LTX_TABLE: ComfortDisplayTable = {
  id: "LTX",
  // rows は表自体の形の一部——配信される `spill_free_frames` のキーであっても
  // ここに無ければ表示されない。
  rows: ["512x320", "960x576", "896x1152", "1280x768", "1920x1088", "2560x1472"],
  columns: [
    { id: "ltx-default", labelKey: "comfortColumnLtxDefault", source: { kind: "legacy" } },
    { id: "ltx-all-on", labelKey: "comfortColumnLtxAllOn", source: { kind: "budget", engineFamily: "ltx" } },
    { id: "ltx25", labelKey: "comfortColumnLtx25", source: { kind: "budget", engineFamily: "ltx25" } },
    { id: "ltx25-q6", labelKey: "comfortColumnLtx25Q6", source: { kind: "static", frames: LTX25_Q6_FRAMES } },
  ],
};

export const COMFORT_DISPLAY_TABLES: Readonly<Record<string, ComfortDisplayTable>> = {
  [LTX_TABLE.id]: LTX_TABLE,
};

/** 系統→表。LTX 2.3 と LTX 2.5 は**同じ1枚**を見る（列で並べて比べるための表
 * なので、載っているモデルがどれか読み手に分かればよい）。
 *
 * 将来の系統は「表を1枚足し、この対応に1行足す」で済む。対応に無い系統は
 * {@link comfortDisplayTableFor} が `null` を返し、節ごと描かれない。 */
export const COMFORT_TABLE_BY_ENGINE_FAMILY: Readonly<Record<string, string>> = {
  ltx: LTX_TABLE.id,
  ltx25: LTX_TABLE.id,
};

/**
 * アクティブな系統の表、無ければ `null`（＝節を描かない）。
 *
 * `""` を弾くのは `activeEngineFamily` が「まだ `GET /models` が返っていない」と
 * オフラインの両方を `""` で表すため（`shell/outpaintBudget.ts` と同じ2値の扱い）。
 * 未確定のあいだ、どの系統のものとも言えない表を出すよりは出さないほうがよい。
 */
export function comfortDisplayTableFor(engineFamily: string): ComfortDisplayTable | null {
  if (engineFamily === "") return null;
  const id = COMFORT_TABLE_BY_ENGINE_FAMILY[engineFamily];
  if (id === undefined) return null;
  return COMFORT_DISPLAY_TABLES[id] ?? null;
}

/** 「全on」列の条件——5つの高速化トグルすべてが on の構成。`ltx` はその全on行に、
 * `ltx25` は無条件行に一致する。選定は必ず {@link resolveComfortRow} を通す
 * （`api/types.ts` が生の `rows` を自前で走査することを禁じており、正規化は
 * そこに1箇所だけある）。`keepResidentEmbeddings` はどの行の `requires` にも
 * 現れない鍵なので一致判定に関与せず、サーバ既定のままでよい。 */
const ALL_ON_ACCELERATION: AccelerationSettings = {
  attentionBackend: "sage",
  blockSwapPrefetch: true,
  keepResident: true,
  fusedGgufDequantKernel: true,
  vaeMode: "prune_vaed",
  keepResidentEmbeddings: false,
};

/**
 * 1セットの数字。`null` は「この列にこの解像度の答えは無い」＝画面では「—」。
 *
 * `budget` 列の下限・上限は Create の快適上限マーカーと同じものを渡す
 * （`MIN_NUM_FRAMES` と配信の `limits.max_num_frames`——`modes/single/
 * useGenerationForm.ts` の `limits.minNumFrames`/`maxNumFrames` の出どころ）。
 * 係数は {@link resolveComfortRow} が正規化した `spatialFactor`/`temporalFactor`
 * を使う。
 *
 * 行は必ず `"<w>x<h>"` なので分解は1回で済ませ、3分岐が同じ幅・高さを見る。
 */
export function resolveComfortCell(
  limits: AppLimits,
  resolutionKey: string,
  column: ComfortDisplayColumn,
): number | null {
  const [widthText, heightText] = resolutionKey.split("x");
  const width = Number(widthText);
  const height = Number(heightText);

  switch (column.source.kind) {
    case "legacy":
      return limits.spill_free_frames[resolutionKey] ?? null;
    case "budget": {
      const resolved = resolveComfortRow(limits, column.source.engineFamily, ALL_ON_ACCELERATION, true);
      if (!resolved) return null;
      return comfortFramesForBudget(
        width,
        height,
        resolved.singleBudget,
        MIN_NUM_FRAMES,
        limits.max_num_frames,
        resolved.spatialFactor,
        resolved.temporalFactor,
      );
    }
    case "static":
      return column.source.frames[resolutionKey] ?? null;
  }
}
