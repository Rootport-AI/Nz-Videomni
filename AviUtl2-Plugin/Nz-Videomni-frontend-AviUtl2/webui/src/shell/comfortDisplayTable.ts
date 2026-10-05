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
 * 第 9 弾（台帳 §1-31・2026-10-06）で例外が無くなった: 以前は配信されていない
 * 実測点・線（2.5 Q6・2.5 fp8・2.3 fp8 既定）をここに直書きしていたが、配信行が
 * 重みの種別（`requires.weight_class`）を持つようになったので、どの列も配信行
 * から引く。行が無い列は「—」。
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
 *  - `legacy` ＝ レガシー表 `limits.spill_free_frames` の実測値そのまま
 *    （2.3 の既定構成。1本の線で表せないので配信行を持たない）。
 *  - `budget` ＝ その系統・その重みの種別の「全on」条件に一致する配信行
 *    （{@link resolveComfortRow} が選ぶ）のトークン予算から換算した線。
 *
 * どちらも「その解像度の答えが無い」を `null`（表示は「—」）で返す。無い理由は列に
 * よって違う（測っていない／その系統・種別に行が無い）が、読み手に見せる区別ではない。 */
export type ComfortColumnSource =
  | { readonly kind: "legacy" }
  | { readonly kind: "budget"; readonly engineFamily: string; readonly weightClass: string };

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

/** LTX 系の表。行は縦横比の違う6サイズで、`spill_free_frames` に無い
 * `896x1152` も含む（`legacy` 列だけが「—」になる、という**列ごとに答えが違う**
 * ことがそのまま見える並び）。列は 2.3 の既定構成（`legacy`）と、系統 2 つ ×
 * 重みの種別 3 つ（4bit／8bit／Q6_K）の `budget` 列。 */
const LTX_TABLE: ComfortDisplayTable = {
  id: "LTX",
  // rows は表自体の形の一部——配信される `spill_free_frames` のキーであっても
  // ここに無ければ表示されない。
  rows: ["512x320", "960x576", "896x1152", "1280x768", "1920x1088", "2560x1472"],
  columns: [
    { id: "ltx-default", labelKey: "comfortColumnLtxDefault", source: { kind: "legacy" } },
    {
      id: "ltx-4bit",
      labelKey: "comfortColumnLtx4bit",
      source: { kind: "budget", engineFamily: "ltx", weightClass: "4bit" },
    },
    {
      id: "ltx-8bit",
      labelKey: "comfortColumnLtx8bit",
      source: { kind: "budget", engineFamily: "ltx", weightClass: "8bit" },
    },
    {
      id: "ltx-q6k",
      labelKey: "comfortColumnLtxQ6k",
      source: { kind: "budget", engineFamily: "ltx", weightClass: "q6k" },
    },
    {
      id: "ltx25-4bit",
      labelKey: "comfortColumnLtx25_4bit",
      source: { kind: "budget", engineFamily: "ltx25", weightClass: "4bit" },
    },
    {
      id: "ltx25-8bit",
      labelKey: "comfortColumnLtx25_8bit",
      source: { kind: "budget", engineFamily: "ltx25", weightClass: "8bit" },
    },
    {
      id: "ltx25-q6k",
      labelKey: "comfortColumnLtx25Q6k",
      source: { kind: "budget", engineFamily: "ltx25", weightClass: "q6k" },
    },
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

/** 「全on」列の条件——5つの高速化トグルすべてが on の構成。列の系統・種別の
 * 配信行が全on の条件つき（2.3 の 4bit／Q6_K）でも無条件（2.3 の 8bit・2.5 の
 * 3 行）でも、この構成なら一致する。選定は必ず {@link resolveComfortRow} を通す
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
  // Output setting sharing the store (§3-164); the comfort matcher ignores it.
  embedMp4Metadata: true,
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
 * 行は必ず `"<w>x<h>"` なので分解は1回で済ませる。
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
      const resolved = resolveComfortRow(
        limits,
        column.source.engineFamily,
        ALL_ON_ACCELERATION,
        true,
        column.source.weightClass,
      );
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
  }
}
