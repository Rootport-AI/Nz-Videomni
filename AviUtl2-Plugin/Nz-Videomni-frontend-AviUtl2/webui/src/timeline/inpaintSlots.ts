import { useSyncExternalStore } from "react";
import type { SelectionItem, TimelineSelection } from "./menuSelection";

/**
 * Inpainting の2つの入力（部分フィルタ／対象動画）を持つ、モジュール単一の
 * 保管庫（台帳 §3-55、実装計画 §7.2）。
 *
 * ## なぜ保管庫が要るのか
 *
 * Inpainting は**右クリックを2回**受けて成立する唯一の機能である（オーナー裁定
 * D6: 部分フィルタ用と動画用で項目を分ける）。1回目の右クリックで届いた値を、
 * 2回目の右クリックが消してはならない。
 *
 * そこで「値はモジュール側に置き、画面は読むだけ」にする。前例は
 * `provisionalReservation.ts` の publish ミラー群（`publishFormValues` /
 * `publishSourceSlots` など）で、あちらと違うのはリスナーを持つこと ——
 * 画面は右クリックの到着で**再描画されなければならない**ので、
 * `useSyncExternalStore` が読める形にしてある（`jobs/JobCard.tsx` の
 * `subscribeInserted` と同じ作り）。
 *
 * ## 再マウントは**しない**（敵対的レビュー M2/M3）
 *
 * 他タブの右クリックは `remountTokens` を進めて行き先の画面を作り直すが、
 * **Inpainting の2本だけはそれをしない**。この機能は右クリックを2回受けるので、
 * 再マウントすると2回目が1回目のフォーム状態（フレーム数・シード・アップロード
 * の進み具合・マスクの進捗）を毎回捨ててしまう。保管庫があるのに画面を作り直す
 * のは、保管庫が守れない状態まで道連れにする行為でしかない。
 *
 * 代わりに {@link InpaintSlots.subTabRequest} を置いた。両方の publish で増える
 * 単調なカウンタで、`EditScreen` はこれが増えたときだけサブタブを Inpainting へ
 * 切り替える —— タブは常時マウントなので、切り替えるだけで画面は出る。
 */

/** 右クリックで届いた部分フィルタのスナップショット。
 *
 * ファイルパスもサムネイルも持たない —— 部分フィルタは素材ではなく、画面に
 * 出すのはレイヤーとフレーム範囲と中間点の枚数だけ（オーナー裁定 D7）。 */
export interface InpaintPartialFilterSlot {
  /** 部分フィルタのレイヤー（0 始まり。表示は 1 始まりへ直す）。 */
  layer: number;
  /** 部分フィルタの開始フレーム（絶対）。 */
  frameStart: number;
  /** 部分フィルタの終了フレーム（絶対・閉区間）。 */
  frameEnd: number;
  /** プロジェクトの fps 分子／分母。生成 fps の出どころで、
   * `getEditInfo` を待たずに窓を描けるようにここへ持つ。 */
  rate: number;
  scale: number;
  /**
   * 中間点の枚数（設計正本 `Docs/INPAINTING_DESIGN.md` §3.1 がカードに出すと
   * 決めた 3 つ目の値）。
   *
   * 出どころは `timeline.getSelection` の `sectionCount`（
   * `get_object_section_num` ＝ **中間点で区切られた区間の数**）で、中間点の
   * 枚数はそれより 1 つ少ない。native が答えられないときは `1` を返す契約なので、
   * その場合は 0 枚になる —— 追尾していない素の部分フィルタと同じ表示で、
   * 嘘にはならない。
   */
  midpoints: number;
}

/** `sectionCount`（区間の数）→ 中間点の枚数。純関数で、変換の知識をここ 1 箇所に
 * 閉じ込める。欠けている・壊れている値は 0 枚へ落とす。 */
export function midpointsFromSectionCount(sectionCount: number | undefined): number {
  if (typeof sectionCount !== "number" || !Number.isFinite(sectionCount)) return 0;
  return Math.max(0, Math.round(sectionCount) - 1);
}

/** 右クリックで届いた対象動画のスナップショットと、そのアップロード結果。 */
export interface InpaintTargetSlot {
  /** 選択されていた動画オブジェクト（`selection.selected[0]`）。トリム判定
   * （`decideSourceTrim`）と素材寸の読み出しに使う。 */
  item: SelectionItem;
  /** その時の選択スナップショット全体（rate/scale と範囲）。 */
  selection: TimelineSelection;
  /** `POST /upload/video` の `video_id`。アップロードが終わるまで `null`。 */
  videoId: string | null;
  /** §1-6: トリムを頼んだのに適用されなかった。`true` なら生成を止める
   * （別の場所を描き替えてしまうため）。 */
  trimFailed: boolean;
}

export interface InpaintSlots {
  partialFilter: InpaintPartialFilterSlot | null;
  target: InpaintTargetSlot | null;
  /**
   * **部分フィルタスロットの改訂番号**。`publishInpaintPartialFilter` と
   * `clearInpaintSlots` でだけ増える。
   *
   * 対象動画の差し替えでは増えない——これが load-bearing で、フレーム数欄の
   * 再シードはこの番号の変化で駆動される（`useInpaintForm`）。全ての変更で
   * 増やすと、対象動画を差し替えただけでユーザーが打ち直したフレーム数が
   * 既定値へ戻ってしまう。
   */
  version: number;
  /** **対象動画スロットの改訂番号**。中身が本当に別のオブジェクトへ変わった
   * ときだけ増える（{@link inpaintTargetKey} の一部）。 */
  targetVersion: number;
  /**
   * サブタブを Inpainting へ寄せてほしい、という要求の通し番号。**両方**の
   * publish で増える。`EditScreen` は増分を見てサブタブを切り替えるだけで、
   * 画面を作り直さない（このファイル冒頭の M2/M3 の注記）。
   */
  subTabRequest: number;
  /**
   * どの対象について**アップロードを始めたか**（{@link inpaintTargetKey} の値）。
   *
   * 画面側の ref ではなく保管庫に置くのが要点: 画面が作り直されても、別の
   * インスタンスが同じファイルをもう一度上げ直さない。
   */
  uploadStartedFor: string | null;
  /**
   * Inpainting が動いている最中か（マスク描画・マスク送信・生成要求の送信）。
   * `useInpaintForm` が書き、`AppShell` が読む —— 途中で右クリックが来たら
   * 断るため（敵対的レビュー m1）。
   */
  busy: boolean;
}

const EMPTY_SLOTS: InpaintSlots = {
  partialFilter: null,
  target: null,
  version: 0,
  targetVersion: 0,
  subTabRequest: 0,
  uploadStartedFor: null,
  busy: false,
};

let slots: InpaintSlots = EMPTY_SLOTS;
const listeners = new Set<() => void>();

function commit(next: InpaintSlots): void {
  slots = next;
  // 購読中のリスナーが購読解除しても walk が乱れないようコピーしてから回す
  // （`mockBridge` の `emitEvent` と同じ作法）。
  for (const listener of [...listeners]) listener();
}

/** いまの保管庫。`useSyncExternalStore` の `getSnapshot` そのもの。 */
export function getInpaintSlots(): InpaintSlots {
  return slots;
}

/** 変更の購読（`useSyncExternalStore` 用）。戻り値は購読解除。 */
export function subscribeInpaintSlots(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

/**
 * アップロードの一意鍵。**同じオブジェクトを指す限り同じ文字列**になるので、
 * 「この対象については既に上げ始めた」を保管庫 1 箇所で判定できる。
 *
 * `targetVersion` を混ぜてあるのは、❌ で捨てたあとに同じ動画をもう一度
 * 右クリックした場合に**上げ直す**ため（捨てた時点で `videoId` も鍵も消える）。
 */
export function inpaintTargetKey(target: InpaintTargetSlot | null, targetVersion: number): string | null {
  if (!target) return null;
  const { layer, frameStart, filePath } = target.item;
  return `${layer}:${frameStart}:${filePath ?? ""}:${targetVersion}`;
}

/** 2 つの選択オブジェクトが「同じ対象」か。レイヤー・頭フレーム・ファイルパスの
 * 3 つで見る —— 窓も予約席もこの 3 つから導かれるので、どれかが違えば別の対象。 */
function sameTargetIdentity(a: SelectionItem, b: SelectionItem): boolean {
  return a.layer === b.layer && a.frameStart === b.frameStart && a.filePath === b.filePath;
}

/**
 * 部分フィルタの右クリック（`inpaintPartialFilter`）を受け取る。
 *
 * 対象動画スロットには**触らない** —— 順序不問（実機ゲート G2）は、片方の
 * publish がもう片方を消さないことで成り立つ。
 */
export function publishInpaintPartialFilter(partialFilter: InpaintPartialFilterSlot): void {
  commit({
    ...slots,
    partialFilter,
    version: slots.version + 1,
    subTabRequest: slots.subTabRequest + 1,
  });
}

/**
 * 対象動画の右クリック（`inpaintVideo`）を受け取る。
 *
 * **同じオブジェクトをもう一度右クリックしただけなら、上げ直さない。**
 * `videoId` も `targetVersion` もそのままで、新しい選択スナップショットだけを
 * 差し替える（飛行中のアップロードを捨てて掛け直す理由が無い）。
 *
 * 本当に別の対象へ変わったときは `videoId` を `null` へ戻し `targetVersion` を
 * 進める —— 前の素材の ID を持ち越すと**別のファイル**を描き替えることになる。
 * `version`（部分フィルタ側）は増やさない: フレーム数欄は部分フィルタが決める
 * もので、対象動画の差し替えで既定へ戻ってはならない。
 */
export function publishInpaintTarget(target: { item: SelectionItem; selection: TimelineSelection }): void {
  const current = slots.target;
  if (current && sameTargetIdentity(current.item, target.item)) {
    commit({
      ...slots,
      target: { ...current, selection: target.selection },
      subTabRequest: slots.subTabRequest + 1,
    });
    return;
  }
  commit({
    ...slots,
    target: { item: target.item, selection: target.selection, videoId: null, trimFailed: false },
    targetVersion: slots.targetVersion + 1,
    subTabRequest: slots.subTabRequest + 1,
  });
}

/** 「この鍵の対象についてアップロードを始めた」を記録する。同じ値なら何もしない
 * （余計なスナップショット更新でループを作らないため）。 */
export function markInpaintUploadStarted(key: string): void {
  if (slots.uploadStartedFor === key) return;
  commit({ ...slots, uploadStartedFor: key });
}

/**
 * アップロードの結果を書き戻す。対象スロットが無ければ何もしない
 * （❌ で捨てたあとに飛行中のアップロードが着地した場合）。
 *
 * `trimFailed` は**上書き**ではなく OR を取る: 一度立った「トリムできなかった」
 * は、その素材を使い続ける限り消えない事実だから。
 */
export function setInpaintTargetUpload(videoId: string | null, trimFailed: boolean): void {
  const target = slots.target;
  if (!target) return;
  const nextTrimFailed = target.trimFailed || trimFailed;
  if (target.videoId === videoId && target.trimFailed === nextTrimFailed) return;
  commit({
    ...slots,
    target: { ...target, videoId, trimFailed: nextTrimFailed },
  });
}

/** Inpainting が動いている最中かを書く（`useInpaintForm` だけが呼ぶ）。
 * 変化が無ければ何もしない。 */
export function setInpaintBusy(busy: boolean): void {
  if (slots.busy === busy) return;
  commit({ ...slots, busy });
}

/** ❌: 両方のスロットを捨てる。`version` は増やす —— 次に来る部分フィルタが
 * 「新しい部分フィルタ」として再シードされるように。アップロードの鍵も捨てるので、
 * 同じ動画をもう一度右クリックすれば改めて上がる。 */
export function clearInpaintSlots(): void {
  commit({
    ...slots,
    partialFilter: null,
    target: null,
    version: slots.version + 1,
    uploadStartedFor: null,
  });
}

/**
 * テスト用のリセット。**購読は解除しない**（敵対的レビュー m2）——
 * `resetProvisionalReservation` と同じで、状態を初期値へ戻して購読者へ知らせる
 * だけ。リスナーを捨てると、マウント済みの `useSyncExternalStore` が二度と
 * 更新を受け取れない購読者になり、リセット後のテストが静かに壊れる。
 */
export function resetInpaintSlots(): void {
  commit(EMPTY_SLOTS);
}

/** React から保管庫を読む。 */
export function useInpaintSlots(): InpaintSlots {
  return useSyncExternalStore(subscribeInpaintSlots, getInpaintSlots, getInpaintSlots);
}
