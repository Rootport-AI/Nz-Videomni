/**
 * `timeline.getSelection` の **時間範囲**（`rangeStart`/`rangeEnd`）の開閉区間
 * 規約を引き受ける、依存ゼロの葉モジュール。
 *
 * ## なぜ独立したファイルなのか
 *
 * この規約は §1-17 Retake の 2 箇所から必要になる:
 *  - `timeline/retakeWindow.ts`（窓の写像）
 *  - `timeline/prefillSeed.ts`（右クリック時の仮オブジェクト尺のシード）
 *
 * ところが `retakeWindow.ts` は `sourceTrim.ts` を、`sourceTrim.ts` は
 * `prefillSeed.ts` を実行時 import しているので、`prefillSeed.ts` から
 * `retakeWindow.ts` を import すると**循環 import** になる。かといって
 * `+ 1` を両方に書き写すと、実機で規約が確定したときに直す場所が 2 つになり、
 * 「1 箇所で直せる構造」という要件そのものが壊れる。
 *
 * そこで規約だけを import を一切持たない葉へ出し、両方がここを見る。
 * `retakeWindow.ts` は後方互換のためこの 2 つを再 export しているので、
 * 呼び出し側は今までどおり `retakeWindow` から取ってもよい。
 */

/** タイムラインの時間範囲。`timeline.getSelection` の `rangeStart`/`rangeEnd`
 * と同じ形（プロジェクトフレーム）。パネル側で編集した値もこの形で渡す。 */
export interface TimelineFrameRange {
  rangeStart: number;
  rangeEnd: number;
}

/** 半開区間 `[startFrame, endFrameEx)` に正規化した時間範囲。 */
export interface HalfOpenFrameRange {
  startFrame: number;
  /** 終端（**含まない**）。 */
  endFrameEx: number;
}

/**
 * 選択範囲を半開区間へ正規化する。**この関数が閉区間仮説の唯一の隔離点**。
 *
 * ⚠️ 仮説（未確定・F0）: `timeline.getSelection` の `rangeStart`/`rangeEnd` は
 * **閉区間**（`rangeEnd` を含む）とみなしている。根拠は同じスナップショット内の
 * `frameStart`/`frameEnd` の既存慣行 —— `prefillSeed.spanDurationSec` の
 * `frameEnd - frameStart + 1` と native の `FindObjectByJob`（`next = frameEnd + 1`）
 * —— だけで、範囲側の規約は **TS/native とも未文書**（native の bridge.cpp は
 * SDK の `select_range_start`/`select_range_end` を無変換で流すだけ）。
 *
 * 実機でオーナーに「既知の長さの範囲を選択 → 右クリック」を 1 回してもらい、
 * ログの値から閉/半開を確定させる予定。**確定後に直すのはこの関数の本体
 * 1 行（`rangeEnd + 1`）だけ**で済むようにしてあり、
 * `retakeWindow.test.ts` の "closed-interval hypothesis" テストが pin している。
 * 半開だと判明した場合は `+ 1` を落とし、同テストの期待値を書き換える。
 */
export function normalizeSelectionRange(range: TimelineFrameRange): HalfOpenFrameRange {
  return { startFrame: range.rangeStart, endFrameEx: range.rangeEnd + 1 };
}

/**
 * 選択範囲の長さ（プロジェクトフレーム数）。空/逆転/非有限は `0`。
 * {@link normalizeSelectionRange} 越しに数えるので、開閉区間の仮説が変わっても
 * ここは自動で追随する。
 */
export function selectionRangeFrames(range: TimelineFrameRange): number {
  const { startFrame, endFrameEx } = normalizeSelectionRange(range);
  if (!Number.isFinite(startFrame) || !Number.isFinite(endFrameEx)) return 0;
  return Math.max(0, endFrameEx - startFrame);
}
