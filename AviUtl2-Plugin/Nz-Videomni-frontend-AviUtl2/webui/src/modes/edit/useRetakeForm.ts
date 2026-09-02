import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { GenerateChainRequest } from "../../api/types";
import type { AppConfig } from "../../api/types";
import { bridge as defaultBridge } from "../../bridge";
import type { NativeBridge } from "../../bridge";
import { parseLoraPrompt } from "../../lora/loraTags";
import { deriveGenerationParams } from "../../timeline/deriveGenerationParams";
import type { GenerationPrefill } from "../../timeline/generationPrefill";
import type { TimelineSelection } from "../../timeline/menuSelection";
import {
  RETAKE_WINDOW_MAX_PX,
  RETAKE_WINDOW_MIN_PX,
  mapRangeToMaterialSec,
  resolveRetakeWindow,
  retakeMaxWindowPx,
} from "../../timeline/retakeWindow";
import type { RetakeRangeResult, RetakeWindowResult } from "../../timeline/retakeWindow";
import {
  getReservationState,
  getRetakeReservationPendingId,
  publishRetakeReservation,
  rollbackReservedPlacement,
} from "../../timeline/provisionalReservation";
import { decideSourceTrim, trimQuery } from "../../timeline/sourceTrim";
import { STAGE2_WINDOW_DEFAULT, STAGE2_WINDOW_PRESETS } from "../../shell/tokenBudget";
import type { Stage2Window } from "../../shell/tokenBudget";
import { FALLBACK_APP_CONFIG, MIN_HEIGHT, MIN_WIDTH } from "../single/defaultConfig";
import { fileNameFromPath } from "../single/keyframeUtils";
import { isDimensionOnGrid, roundToMultiple } from "../single/paramUtils";
// クロスモード import。`useSourceUpload` は `modes/chained/` にあるが動かさない
// —— `useOutpaintForm` がすでに同じ理由（chained/ は別ワークストリームの占有領域）で
// ここから直接 import しており、本ファイルはその先例に倣っているだけ。
import { useSourceUpload } from "../chained/useSourceUpload";
import type { UseSourceUploadResult } from "../chained/useSourceUpload";

/**
 * 糊代（のりしろ）の既定値。**サーバ既定の写しで、送信はしない**（リクエストに
 * `head_px`/`tail_px` を載せないのは実装計画 §2 の裁定 —— サーバ側の較正値に
 * 自動追随させるため）。ここに置いてあるのは RangeBand の縞と hint 1 行を
 * 描くためだけで、値そのものはスパイクの実測（`outputs/retake_spike/`、
 * VERIFICATION_LOG §55）で決まった 25/24。非対称なのは音声 VAE の因果性由来。
 *
 * サーバがこの既定を変えたらここも直す必要がある（表示だけが古くなる）。
 * それでも API に載せないのは、「UI から動かせないつまみ」を送ると、送った値が
 * 正であるかのように見えてしまうため。
 */
export const RETAKE_GLUE_HEAD_FRAMES = 25;
export const RETAKE_GLUE_TAIL_FRAMES = 24;

/** Chain の継ぎ目パラメータ。Retake は 1 クリップなので継ぎ目は実際には使われ
 * ないが、`GenerateChainRequest` の必須フィールドなので既定値を送る（契約どおり）。 */
const RETAKE_OVERLAP_FRAMES = 3;
const RETAKE_OVERLAP_STRENGTH = 0.5;

/** Generate が押せない理由コード。`GenerateReasonsNote` が
 * `editReasonMessages.buildRetakeReasonMessages` の表と突き合わせて 1 行ずつ出す。 */
export type RetakeReasonCode =
  | "sourceMissing"
  | "sourceUploading"
  | "sourceUploadFailed"
  | "sourceTrimFailed"
  | "rangeUnusable"
  | "outOfMaterial"
  /** 幅か高さが 64 の格子から外れている（数値欄に手打ちした値）。Create/Chain と
   * 同じ理由コード名なので、文言も `single.generateReasons` のものを共有する。 */
  | "dimensionsOffGrid";

/** 右クリック時に撮ったスナップショットのうち、パネルが使う分。 */
export interface RetakeSnapshot {
  selection: TimelineSelection;
  /** 選択オブジェクト（`selection.selected[0]`）。ガードが単一選択を保証済み。 */
  layer: number;
  frameStart: number;
  filePath: string | null;
  fileName: string;
  projectFps: number;
  /** 素材の実寸。**0 は「不明」**（`bridge/types.ts` の `mediaWidth` 参照）で、
   * `null` は来ない。素材カードの解像度表示と、生成解像度の初期値の出どころ。 */
  mediaWidth: number;
  mediaHeight: number;
}

/** F6 が Generate 押下時に予約を打ち直すための、確定した窓の置き場所。 */
export interface RetakePlacement {
  layer: number;
  /** 窓の頭（プロジェクトフレーム）。 */
  frameStart: number;
  /** 窓の尻尾（プロジェクトフレーム、閉区間）。 */
  frameEnd: number;
  /** 窓の長さ（生成フレーム）。`reservePlacement` はこれと `genFps` から
   * リボン長を出す。 */
  numFrames: number;
  genFps: number;
}

export interface UseRetakeFormDeps {
  /** 共有プロンプト（`AppShell` の `PromptBar`）。Retake はプロンプト欄を
   * 持たない —— Create/Chain/Outpainting と同じ 1 本のバーを読むだけ。 */
  prompt?: string | undefined;
  /** `GET /config`。省略時は組み込みの既定（`FALLBACK_APP_CONFIG`）。窓長の
   * 上下限と、既定の解像度/fps/seed をここから読む。 */
  config?: AppConfig | undefined;
  /** テスト/結合の継ぎ目。省略時はアプリ全体の singleton bridge。 */
  nativeBridge?: NativeBridge | undefined;
  /** 右クリックの一回限りのペイロード。**マウント時にちょうど 1 回**だけ
   * 消費する（`AppShell` が `remountTokens.edit` を進めるので、新しい右クリックは
   * 新しいマウントとして届く）。 */
  initialIntent?: GenerationPrefill | undefined;
}

export interface UseRetakeFormResult {
  /** 右クリック由来のスナップショットがあるか。`false` = 手動でタブを開いた
   * だけの状態（または❌/🔁で捨てたあと）で、パネルは案内文だけを出す。 */
  snapshot: RetakeSnapshot | null;
  source: UseSourceUploadResult;

  /**
   * 🔁（素材だけ捨てた）待機中か。`snapshot === null` かつこれが `true` の
   * ときだけ「素材待ち」状態 —— 設定欄は出したまま、次の右クリックを待つ。
   * ❌（片付け）と手動でタブを開いただけの状態では `false`（＝案内文だけ）。
   */
  awaitingSource: boolean;
  /** ❌: このサブタブを右クリック前の空状態へ戻す。素材・区間・設定すべてを
   * 捨て、⏳予約リボンも（自分が置いたものなら）片付ける。 */
  clearAll: () => void;
  /** 🔁: 動画と区間だけを捨てて「素材待ち」へ。設定は表示されたまま残り、
   * 次の Retake 右クリック **1 回きり** に持ち越される（幅・高さは持ち越さず、
   * 新しい素材の実寸に追随する）。 */
  resetSource: () => void;

  /** 範囲→素材秒の写像（右クリック時に 1 回だけ作る。以後タイムラインを
   * 追いかけることはなく、変わるのは❌/🔁で捨てるときだけ）。 */
  mapping: RetakeRangeResult | null;
  /** 確定した窓（8n+1・[下限,上限]・素材端でクランプ済み）。毎描画導出で、
   * 状態としては持たない —— `useOutpaintForm` の「上限は保存しない」規律と同じ。 */
  window: RetakeWindowResult | null;

  /** RangeBand の帯の全幅（生成フレーム）＝素材のうち実際に再生されている長さ。 */
  materialFrames: number;
  /** 同じ長さを秒で。**ファイル全長ではなく、このリボンが実際に再生している
   * 区間の長さ**（`mapping.materialDurationSec − materialStartSec`）——
   * 素材カードの「◯秒」はこれ。ファイル全長を出すと、頭出しでトリムされた
   * 素材で RangeBand の帯と数字が食い違う。 */
  materialSeconds: number;
  /** RangeBand 座標での窓の頭（0 = 使える素材の先頭）。 */
  windowStartFrame: number;
  /** 窓の長さ（生成フレーム）。 */
  windowFrames: number;
  /** RangeBand の `onChange` をそのまま繋ぐ。8n+1 スナップとクランプは
   * `resolveRetakeWindow` が引き受けるので、ここは要求値を保存するだけ。 */
  setWindow: (next: { startFrame: number; frames: number }) => void;

  /** 窓長の下限・上限。下限は config 由来（読めなければ F1 の定数）。上限は
   * {@link stage2Window} 連動 —— 潜在19フレームを選ぶと 145 へ縮む。 */
  minWindowFrames: number;
  maxWindowFrames: number;
  glueHeadFrames: number;
  glueTailFrames: number;

  frameRate: number;
  setFrameRate: (value: number) => void;
  seed: number;
  setSeed: (value: number) => void;
  regenerateAudio: boolean;
  setRegenerateAudio: (value: boolean) => void;

  /** Stage-2（アップスケール工程）のクリップ長。Chain と同じ 2 択で、既定は
   * `standard`。既定のときは**リクエストに載せない**（サーバ既定に追随）。 */
  stage2Window: Stage2Window;
  setStage2Window: (value: Stage2Window) => void;

  /**
   * 生成される解像度。初期値は素材の実寸ベース（{@link RetakeSnapshot.mediaWidth}
   * を `deriveGenerationParams` に通したもの）で、**ユーザーが触るまでその導出値に
   * 追随する**。触ったあとはその値で固定される。
   */
  width: number;
  height: number;
  /** `SizeFields` の `snap` 規約は Create/Chain と同一: `true` = 64 の倍数へ
   * 丸めて上下限へクランプ、`false` = 手打ちをそのまま通す（格子外れは
   * `dimensionsOffGrid` が送信時に止める）。 */
  setWidth: (value: number, snap?: boolean) => void;
  setHeight: (value: number, snap?: boolean) => void;
  /** `SizeFields` のスライダー範囲（config の `limits` 由来）。 */
  limits: { minWidth: number; maxWidth: number; minHeight: number; maxHeight: number };

  /**
   * RangeBand 座標の窓（頭フレーム・長さ）を、**プロジェクトのフレーム番号**
   * （0 始まり・閉区間）へ写す。{@link placement} と同じ写像を、ドラッグ中の
   * プレビュー値に対しても呼べるように関数として出したもの——読み出し行が
   * ドラッグ中に固まらないために要る（RangeBand は `pointerup` まで
   * `onChange` を上げない）。写像の知識はこのフックの中の 1 本のまま。
   */
  toProjectRange: (startFrameOnGenFps: number, frames: number) => { frameStart: number; frameEnd: number } | null;

  /** 右クリック時とタイムラインがずれているか。`checkStale()` を呼ぶまで
   * `false`。**注意文を出すだけで、`isValid` には一切影響しない**。 */
  stale: boolean;
  /** Generate 押下時に 1 回だけ呼ぶ。`timeline.getSelection` を 1 回叩いて
   * スナップショットと比べ、違えば {@link stale} を立てる。失敗は無視 ——
   * 比較できないことを理由に生成を止めるほどの話ではない。 */
  checkStale: () => Promise<void>;

  validityReasons: RetakeReasonCode[];
  isValid: boolean;

  /** 確定した窓の置き場所（Generate 時の予約打ち直し用）。 */
  placement: RetakePlacement | null;
  /** `POST /generate/chain` の body。`isValid` のときだけ意味がある。 */
  buildRequest: () => GenerateChainRequest;
}

/**
 * §1-17 Retake パネルのフォーム全体。手本は `useOutpaintForm`。
 *
 * ## 3 つの構造的な決めごと
 *
 * 1. **スナップショットは 1 回だけ読み、あとはユーザーが捨てられるだけ**。
 *    右クリックした瞬間の選択（オブジェクトと範囲）を lazy initializer で 1 回
 *    だけ読み、以後タイムラインが動いても追いかけない。パネルのレンジ指定が正で
 *    あり、ずれは `checkStale()` が注意文で伝えるだけ（生成は止めない）。
 *    書き換わる経路は {@link UseRetakeFormResult.clearAll}（❌）と
 *    {@link UseRetakeFormResult.resetSource}（🔁）の 2 つだけで、どちらも
 *    「捨てる」方向にしか動かない —— 別の素材が横から入ってくることはない。
 * 2. **窓は保存しない、要求値を保存する**。状態が持つのは「ユーザーが要求した
 *    開始秒と長さ」だけで、実際の窓は毎描画 `resolveRetakeWindow` で導出する。
 *    8n+1 の格子も [下限,上限] のクランプも素材端の詰めも、判定は 1 箇所
 *    （`timeline/retakeWindow.ts`）にしかない。RangeBand が格子を知らないのも
 *    同じ理由で、あちらは表示と操作だけを担当する controlled な子。
 * 3. **素材はこのパネルの中では差し替えられない**。アップロードは右クリック由来の
 *    1 回きりで、ファイル選択ボタンも drag&drop も無い。窓は「そのオブジェクトの
 *    その場所」に対して測った量なので、素材だけ入れ替えると意味が変わってしまう。
 *    別の場所を撮り直したくなったら 🔁（{@link UseRetakeFormResult.resetSource}）
 *    で素材と区間を捨て、タイムラインでもう一度右クリックする —— 設定だけが
 *    そのまま持ち越される導線で、「素材の差し替え」ではない。
 */
export function useRetakeForm(deps: UseRetakeFormDeps = {}): UseRetakeFormResult {
  const { nativeBridge, initialIntent } = deps;
  const config = deps.config ?? FALLBACK_APP_CONFIG;
  const prompt = deps.prompt ?? "";

  // --- スナップショット（1 回だけ） ---------------------------------------
  const [snapshot, setSnapshot] = useState<RetakeSnapshot | null>(() => buildSnapshot(initialIntent));
  const [mapping, setMapping] = useState<RetakeRangeResult | null>(() => {
    const selection = snapshotSelection(initialIntent);
    if (!selection) return null;
    // `selection` はそのまま `TimelineFrameRange`（`rangeStart`/`rangeEnd`）でもある。
    return mapRangeToMaterialSec(selection.selected[0], selection, selection);
  });

  // --- ユーザーが要求した窓（保存するのはこれだけ） ------------------------
  const [requested, setRequested] = useState<{ startSec: number; durationSec: number } | null>(() =>
    mapping?.ok ? { startSec: mapping.startSec, durationSec: mapping.durationSec } : null,
  );

  /**
   * 🔁 の持ち越し（このファイル末尾のモジュールスコープ）の**消費側**。
   * lazy initializer から呼ぶ純関数で、`peek` は読むだけ（捨てるのは下の
   * one-shot effect）なので、StrictMode の初期化二度撃ちでも結果が変わらない。
   * 判定は `snapshotSelection` —— **Retake の右クリックで来たマウントだけ**が
   * 受け取る（Outpainting の右クリックや手動タブ切り替えでは持ち越さない）。
   */
  const carried = (): RetakeCarryOver | null =>
    snapshotSelection(initialIntent) ? peekRetakeCarryOver() : null;

  const [frameRate, setFrameRate] = useState(() => carried()?.frameRate ?? config.generation_defaults.frame_rate);
  const [seed, setSeed] = useState(() => carried()?.seed ?? config.generation_defaults.seed);
  // 既定は「映像と音声」（実装計画 §1 の確定仕様）。
  const [regenerateAudio, setRegenerateAudio] = useState(() => carried()?.regenerateAudio ?? true);
  const [stale, setStale] = useState(false);
  const [stage2Window, setStage2Window] = useState<Stage2Window>(
    () => carried()?.stage2Window ?? STAGE2_WINDOW_DEFAULT,
  );
  /** 🔁 で素材だけ捨てた待機状態か（`snapshot === null` との組で 3 状態になる）。 */
  const [awaitingSource, setAwaitingSource] = useState(false);
  // 解像度は「ユーザーが触ったか」だけを持つ（`null` = 触っていない）。
  // **導出値を useState の初期値に入れてはいけない**: このフックは `GET /config`
  // より先にマウントされうるので、そのとき凍るのはフォールバック上限（4096）で
  // クランプされた値になる。4K 素材で 422 を食う経路がこれ。
  const [widthOverride, setWidthOverride] = useState<number | null>(null);
  const [heightOverride, setHeightOverride] = useState<number | null>(null);

  const source = useSourceUpload("video", nativeBridge ? { nativeBridge } : {});
  const { uploadPath } = source;

  // --- 一回限りの自動読み込み ---------------------------------------------
  // `useOutpaintForm` / `ChainedScreen` の #1 と同じ形: トリム判定は同じ
  // (item, selection) の組から行い、`trimQuery` は no-trim のとき `undefined` を
  // 返すので、トリムしない素材のアップロードは §1-6 以前と 1 バイトも変わらない。
  const autoLoadRef = useRef(false);
  useEffect(() => {
    if (autoLoadRef.current) return;
    autoLoadRef.current = true;
    // `snapshotSelection` 越しなので、Retake 以外の右クリックでは何もしない
    // （常時マウントの隣のパネル用ファイルを二重にアップロードしない）。
    const selection = snapshotSelection(initialIntent);
    const item = selection?.selected[0];
    const filePath = item?.filePath;
    if (!selection || !filePath) return;
    const decision = decideSourceTrim(item, selection);
    void uploadPath(filePath, fileNameFromPath(filePath), trimQuery(decision));
    // eslint-disable-next-line react-hooks/exhaustive-deps -- one-shot mount consume
  }, []);

  // 🔁 の持ち越しは**一度きり**（オーナー確定②）。上の lazy initializer 群が
  // 読み終わったので、ここで捨てる。判定式は `carried()` と**同じ**
  // `snapshotSelection(initialIntent)` にしてある —— 例えば `filePath` の有無で
  // ずらすと、「読んだのに捨てない」「捨てたのに読んでいない」の組み合わせが
  // 生まれて、次の普通の右クリックに古い設定が漏れる。
  // 二度撃ち（StrictMode）で 2 回走っても `clear` は冪等なので ref は要らない。
  useEffect(() => {
    if (snapshotSelection(initialIntent)) clearRetakeCarryOver();
    // eslint-disable-next-line react-hooks/exhaustive-deps -- one-shot mount consume
  }, []);

  // --- 窓の導出 -----------------------------------------------------------
  const minWindowFrames = resolveWindowLimit(config.limits.retake_window_min_frames, RETAKE_WINDOW_MIN_PX);
  // 上限は Stage-2 のクリップ長に連動する。Retake の窓は stage-2 の**1タイル**で
  // 精錬されるので、タイルに収まる最大画素フレーム数（`8·vTile − 7`）が窓長の
  // 上限そのもの。config が公開している `retake_window_max_frames` は
  // `standard`（潜在22フレーム = 169）の値なので、短い方のプリセットを選んだ
  // ときは `min` を取って 145 まで下げる。backend の
  // `chain_math.retake_max_window_px(v_tile)` のミラー。
  const maxWindowFrames = Math.min(
    resolveWindowLimit(config.limits.retake_window_max_frames, RETAKE_WINDOW_MAX_PX),
    retakeMaxWindowPx(STAGE2_WINDOW_PRESETS[stage2Window].vTile),
  );

  const window = useMemo<RetakeWindowResult | null>(() => {
    if (!mapping?.ok || !requested) return null;
    return resolveRetakeWindow({
      startSec: requested.startSec,
      durationSec: requested.durationSec,
      genFps: frameRate,
      materialStartSec: mapping.materialStartSec,
      materialDurationSec: mapping.materialDurationSec,
      minFrames: minWindowFrames,
      maxFrames: maxWindowFrames,
    });
  }, [mapping, requested, frameRate, minWindowFrames, maxWindowFrames]);

  // RangeBand の座標系: 0 = 使える素材の先頭（= `materialStartSec`）。ファイルの
  // 先頭ではない —— 頭がトリムされたリボンでは、画面に出ていない前半を帯に
  // 描いても操作できないだけで嘘になる。
  const materialStartSec = mapping?.ok ? mapping.materialStartSec : 0;
  const materialSeconds = mapping?.ok ? Math.max(0, mapping.materialDurationSec - materialStartSec) : 0;
  const materialFrames = frameRate > 0 ? Math.round(materialSeconds * frameRate) : 0;
  const windowStartFrame =
    window?.ok && frameRate > 0 ? Math.max(0, Math.round((window.startSec - materialStartSec) * frameRate)) : 0;
  const windowFrames = window?.ok ? window.frames : minWindowFrames;

  const setWindow = useCallback(
    (next: { startFrame: number; frames: number }) => {
      if (!(frameRate > 0)) return;
      setRequested({
        startSec: materialStartSec + next.startFrame / frameRate,
        durationSec: next.frames / frameRate,
      });
    },
    [materialStartSec, frameRate],
  );

  // --- 解像度（素材の実寸を初期値に、ユーザーが動かせる） ------------------
  // 既存の継ぎ目 `deriveGenerationParams` をそのまま使う: 素材の実寸を、生成が
  // 受け付ける倍数・上下限へ落とすのはこの関数の仕事であって、ここで割り算を
  // 書き直す場所ではない。素材寸が不明（0）なら設定の既定へ落ちる。
  //
  // 2026-08-10（オーナー目視 ④）: これは**初期値**であって固定値ではない。
  // 実効値は `widthOverride ?? derived.width` —— ユーザーが幅・高さを触るまでは
  // 導出値に追随し（`GET /config` が遅れて届いても正しい上限で計算し直される）、
  // 触った瞬間からその値になる。
  const limits = useMemo(
    () => ({
      minWidth: MIN_WIDTH,
      maxWidth: config.limits.max_width,
      minHeight: MIN_HEIGHT,
      maxHeight: config.limits.max_height,
    }),
    [config.limits.max_width, config.limits.max_height],
  );
  const derived = useMemo(
    () =>
      deriveGenerationParams({
        mediaWidth: snapshot?.mediaWidth ?? null,
        mediaHeight: snapshot?.mediaHeight ?? null,
        projectWidth: config.generation_defaults.width,
        projectHeight: config.generation_defaults.height,
        isICLora: false,
        limits,
      }),
    [snapshot?.mediaWidth, snapshot?.mediaHeight, config.generation_defaults, limits],
  );
  const width = widthOverride ?? derived.width;
  const height = heightOverride ?? derived.height;

  // `snap` の意味は Create/Chain の `setWidth`/`setHeight` と一字一句同じ:
  // スライダー（と数値欄のステッパー）は丸めてクランプ、手打ちは素通し。
  // 素通しを許すのは Gradio 追従で、格子外れは送信直前に `dimensionsOffGrid` が
  // 止める（黙って直さない）。
  const setWidth = useCallback(
    (raw: number, snap = true) => {
      if (snap) return setWidthOverride(roundToMultiple(raw, 64, limits.minWidth, limits.maxWidth));
      if (!Number.isFinite(raw)) return;
      setWidthOverride(raw);
    },
    [limits.minWidth, limits.maxWidth],
  );
  const setHeight = useCallback(
    (raw: number, snap = true) => {
      if (snap) return setHeightOverride(roundToMultiple(raw, 64, limits.minHeight, limits.maxHeight));
      if (!Number.isFinite(raw)) return;
      setHeightOverride(raw);
    },
    [limits.minHeight, limits.maxHeight],
  );

  // --- ❌片付け / 🔁別の場所を撮り直す ------------------------------------
  /**
   * ❌ と 🔁 の共通部分: 素材と区間だけを捨てる。
   *
   * 予約席（⏳リボン）の片付けは**条件付き**。席はアプリに 1 つしか無く持ち主を
   * 持たないので、無条件に `rollbackReservedPlacement` を呼ぶと
   * 「Retake の右クリック → Create 側で別の右クリック（席がそちらへ移動）→
   * Edit に戻って❌」で他タブの⏳リボンを消してしまう。系統D の予約時に publish
   * した `pending-…` id と、いまの席の id が**一致するときだけ**片付ける
   * （席が移動すると新しい id が振られるので、一致しなくなる）。
   */
  const dropSource = useCallback(() => {
    setSnapshot(null);
    setMapping(null);
    setRequested(null);
    setStale(false);
    source.clear();
    const mine = getRetakeReservationPendingId();
    if (mine !== null && getReservationState().pendingId === mine) {
      // 失敗は握り潰される（`rollbackReservedPlacement` の中で完結）。席の解放は
      // 同期的に済むので、await せずに投げっぱなしで良い。
      void rollbackReservedPlacement(nativeBridge ?? defaultBridge);
    }
    // 自分の席はもう無い（片付けたか、他所へ移った）。
    publishRetakeReservation(null);
  }, [source, nativeBridge]);

  /** ❌: 右クリック前の空状態へ。設定も既定へ戻し、🔁 の持ち越しも捨てる。 */
  const clearAll = useCallback(() => {
    dropSource();
    setAwaitingSource(false);
    setFrameRate(config.generation_defaults.frame_rate);
    setSeed(config.generation_defaults.seed);
    setRegenerateAudio(true);
    setStage2Window(STAGE2_WINDOW_DEFAULT);
    setWidthOverride(null);
    setHeightOverride(null);
    clearRetakeCarryOver();
  }, [dropSource, config.generation_defaults.frame_rate, config.generation_defaults.seed]);

  /**
   * 🔁: 動画と区間だけを捨てて「素材待ち」へ。
   *
   * 幅・高さは**いま出ている実効値を override へ格上げ**してから捨てる。素材が
   * 消えると `derived` が config の既定へ落ちるので、そうしないと待機中の表示が
   * 勝手に別の数字へ跳ぶ（表示の凍結）。この override は次の右クリック＝次の
   * マウントで自然に消えるので、新しい素材の実寸には素直に追随する
   * （＝持ち越し内容にも幅・高さは入れない・オーナー確定①）。
   */
  const resetSource = useCallback(() => {
    setWidthOverride(width);
    setHeightOverride(height);
    dropSource();
    setAwaitingSource(true);
    stashRetakeCarryOver({ frameRate, seed, regenerateAudio, stage2Window });
  }, [dropSource, width, height, frameRate, seed, regenerateAudio, stage2Window]);

  // --- checkStale ---------------------------------------------------------
  const checkStale = useCallback(async () => {
    if (!snapshot) return;
    try {
      const fresh = await (nativeBridge ?? defaultBridge).request("timeline.getSelection", {});
      setStale(!sameRetakeSelection(fresh, snapshot.selection));
    } catch {
      // 比較できなかっただけ。注意文を出す根拠が無いので、現状のままにする。
    }
  }, [snapshot, nativeBridge]);

  // --- ゲート -------------------------------------------------------------
  const validityReasons: RetakeReasonCode[] = [];
  if (source.state.status === "uploading") validityReasons.push("sourceUploading");
  else if (source.state.status === "error") validityReasons.push("sourceUploadFailed");
  else if (!source.state.id) validityReasons.push("sourceMissing");
  // 必須ゲート: 「一部だけ使っているリボンなのに切り出せなかった」を通すと、
  // アップロード後のファイル頭が素材頭と一致しないまま `window_start_sec` を
  // 送ることになり、**別の場所を撮り直す**。
  if (source.state.trimFailed) validityReasons.push("sourceTrimFailed");
  if (!mapping?.ok) validityReasons.push("rangeUnusable");
  else if (!window?.ok) validityReasons.push("outOfMaterial");
  // 送信直前の格子ゲート（Create/Chain と同じ流儀）。数値欄への手打ちは
  // 素通しで届くので、外れた値は黙って直さずここで止める。
  if (
    !isDimensionOnGrid(width, 64, limits.minWidth, limits.maxWidth) ||
    !isDimensionOnGrid(height, 64, limits.minHeight, limits.maxHeight)
  ) {
    validityReasons.push("dimensionsOffGrid");
  }
  const isValid = snapshot !== null && validityReasons.length === 0;

  // --- 予約の打ち直し先 ----------------------------------------------------
  // RangeBand 座標（素材頭からの生成フレーム）→ プロジェクトのフレーム番号。
  // 写像の逆（`mapRangeToMaterialSec` の
  // `sourceSec = 起点 + (frame - frameStart)/projectFps`）をそのまま解いた形で、
  // `startSec - materialStartSec` はちょうど `startFrameOnGenFps / genFps`。
  // `placement`（確定値）と読み出し行（ドラッグ中のプレビュー値）の**両方**が
  // ここを通るので、2 つの数字がずれることはない。
  const toProjectRange = useCallback(
    (startFrameOnGenFps: number, frames: number): { frameStart: number; frameEnd: number } | null => {
      if (!snapshot) return null;
      const projectFps = snapshot.projectFps;
      if (!(projectFps > 0) || !(frameRate > 0)) return null;
      const frameStart = snapshot.frameStart + Math.round((startFrameOnGenFps / frameRate) * projectFps);
      const projectFrames = Math.max(1, Math.round((frames / frameRate) * projectFps));
      return { frameStart, frameEnd: frameStart + projectFrames - 1 };
    },
    [snapshot, frameRate],
  );

  const placement = useMemo<RetakePlacement | null>(() => {
    if (!snapshot || !mapping?.ok || !window?.ok) return null;
    // 読み出し行が使うのと**同じ入力**（RangeBand 座標の窓頭）で写す。
    const range = toProjectRange(windowStartFrame, window.frames);
    if (!range) return null;
    return {
      layer: snapshot.layer,
      frameStart: range.frameStart,
      frameEnd: range.frameEnd,
      numFrames: window.frames,
      genFps: frameRate,
    };
  }, [snapshot, mapping, window, frameRate, windowStartFrame, toProjectRange]);

  const buildRequest = useCallback((): GenerateChainRequest => {
    const windowStartSec = window?.ok && mapping?.ok ? window.startSec - mapping.trimOffsetSec : 0;
    // プロンプトの `<lora:…>` タグは Create/Chain とまったく同じ扱い（§3-62）:
    // 同じパーサで指示文（`strippedPrompt`）と `loras[]` に分け、両方を送る。
    // タグを外すだけで `loras` を捨てていた旧契約は、Retake も 1 クリップの
    // チェーンジョブでバックエンドが素通しで受けるため、意味を失った。
    //
    // `combineLoras` は**通さない**: それは Create/Chain が持つ制御系（IC-LoRA）
    // 選択パネルと混ぜるための関数で、Retake にそのパネルは無い。制御系の名前を
    // 手打ちされた場合もクライアントでは止めず、バックエンドの既存 422
    // （`LORA_REQUIRES_REFERENCE`）に委ねる —— ここにだけ別のゲートを増やすと、
    // 「どのタブで何が弾かれるか」の規則が増える。
    const { strippedPrompt, loras } = parseLoraPrompt(prompt);
    return {
      prompt: strippedPrompt.trim(),
      // タグが 1 つも無ければキーごと省く（Chain の `buildRequest` と同じ作法）。
      ...(loras.length > 0 ? { loras } : {}),
      width,
      height,
      frame_rate: frameRate,
      seed,
      overlap_frames: RETAKE_OVERLAP_FRAMES,
      overlap_strength: RETAKE_OVERLAP_STRENGTH,
      // 窓長の単一ソース。サーバはこの 1 本の `num_frames` を窓長として読む。
      clips: [{ num_frames: window?.ok ? window.frames : minWindowFrames }],
      // Stage-2 のクリップ長。**既定（standard）のときはキーごと省く** ——
      // サーバ側の既定に追随させるため。潜在19フレームを選んだときだけ載せる。
      ...(stage2Window === STAGE2_WINDOW_DEFAULT ? {} : { stage2_window: stage2Window }),
      // `chunked_upsample` は**意図的に載せない**。`api/types.ts` の同フィールドは
      // 「WebUI は常に明示送信すること」と書いており Chain 系の他のビルダーは全て
      // そうしているが、Retake の窓は上限 169 フレーム＝ stage-2 のタイル1枚に
      // 収まる長さで、一括アップサンプルで足りる（チャンク化して得るものが無い）。
      // ※ Retake も stage-2 での精錬そのものは通る（`upscaled_v` 経由。
      //   `chain_pipeline.py`）。「stage-2 を使わない」ではない。
      retake: {
        video_id: source.state.id ?? "",
        // **アップロード後ファイルの時間軸**へ直す。トリムしていれば
        // `trimOffsetSec` はその切り出し開始秒、していなければ 0。
        window_start_sec: Math.max(0, windowStartSec),
        regenerate_audio: regenerateAudio,
      },
    };
  }, [
    prompt,
    width,
    height,
    frameRate,
    seed,
    window,
    mapping,
    minWindowFrames,
    stage2Window,
    source.state.id,
    regenerateAudio,
  ]);

  return {
    snapshot,
    source,
    awaitingSource,
    clearAll,
    resetSource,
    mapping,
    window,
    materialFrames,
    materialSeconds,
    windowStartFrame,
    windowFrames,
    setWindow,
    minWindowFrames,
    maxWindowFrames,
    glueHeadFrames: RETAKE_GLUE_HEAD_FRAMES,
    glueTailFrames: RETAKE_GLUE_TAIL_FRAMES,
    frameRate,
    setFrameRate,
    seed,
    setSeed,
    regenerateAudio,
    setRegenerateAudio,
    stage2Window,
    setStage2Window,
    width,
    height,
    setWidth,
    setHeight,
    limits,
    toProjectRange,
    stale,
    checkStale,
    validityReasons,
    isValid,
    placement,
    buildRequest,
  };
}

/**
 * 右クリックのペイロードから、パネルが使う分だけを抜き出す。**`intent` が
 * `"retake"` のときだけ**成立し、範囲が無い/オブジェクトが無い/プロジェクトの
 * fps が解けないペイロードは「スナップショット無し」= 案内状態になる。
 *
 * `intent` を見るのが load-bearing: `EditScreen` は両サブパネルを常時マウント
 * するので、Outpainting の右クリック（同じ `initialIntent` が届く）でここが
 * 成立してしまうと、**見てもいないパネルが同じファイルをもう一度アップロード
 * する**。サブタブの選択と同じ判定基準（`initialIntent?.intent`）に揃えてある。
 */
/** Retake の右クリックのときだけ選択スナップショットを返す。{@link buildSnapshot}
 * と自動読み込みの両方がここを通るので、「どの右クリックに反応するか」の判定は
 * 1 箇所にしかない。 */
function snapshotSelection(initialIntent: GenerationPrefill | undefined): TimelineSelection | undefined {
  return initialIntent?.intent === "retake" ? initialIntent.selection : undefined;
}

function buildSnapshot(initialIntent: GenerationPrefill | undefined): RetakeSnapshot | null {
  const selection = snapshotSelection(initialIntent);
  const item = selection?.selected[0];
  if (!selection || !item || !selection.hasRange) return null;
  if (!(selection.rate > 0) || !(selection.scale > 0)) return null;
  return {
    selection,
    layer: item.layer,
    frameStart: item.frameStart,
    filePath: item.filePath,
    fileName: item.filePath ? fileNameFromPath(item.filePath) : "",
    projectFps: selection.rate / selection.scale,
    // 0 = 不明。ここで既定へ落とさず**そのまま**持つ: 素材カードは 0 のときに
    // 解像度の表示自体を省く（「0×0」と出すより黙っている方が正しい）。
    mediaWidth: item.mediaWidth,
    mediaHeight: item.mediaHeight,
  };
}

/**
 * 右クリック時のスナップショットと、いま取り直した選択が「同じ撮り直し」を
 * 指しているか。純関数。
 *
 * 比べるのは**撮り直す場所を決めている値だけ** —— 範囲の有無と両端、そして
 * 選択オブジェクトの素性（レイヤー・両端フレーム・ファイルパス）。カーソル位置や
 * 素材の寸法まで比べると、無関係な操作で注意文が出て狼少年になる。
 */
export function sameRetakeSelection(fresh: TimelineSelection, snapshot: TimelineSelection): boolean {
  if (fresh.hasRange !== snapshot.hasRange) return false;
  if (fresh.rangeStart !== snapshot.rangeStart || fresh.rangeEnd !== snapshot.rangeEnd) return false;
  const a = fresh.selected[0];
  const b = snapshot.selected[0];
  if (!a || !b) return a === b;
  return (
    a.layer === b.layer &&
    a.frameStart === b.frameStart &&
    a.frameEnd === b.frameEnd &&
    a.filePath === b.filePath
  );
}

// --- 🔁 の持ち越し（モジュールスコープ・一度きり） ---------------------------
//
// 🔁 は素材と区間だけを捨てるが、その次の右クリックは**新しいマウント**として
// 届く（`AppShell` が `remountTokens.edit` を進める）ので、素の React 状態では
// 設定がすべて既定へ戻ってしまう。そこで 🔁 が押された瞬間の設定をここへ置き、
// 次の Retake マウントの lazy initializer が拾う。
//
// 新しいファイルを作らないのは、これが「このフックの 2 つのマウントの間だけ
// 生きる値」であり、他のどこからも読まれないため（`provisionalReservation.ts`
// の publish ミラー群と同じ、リスナー無しのモジュール単一値）。
//
// **一度きり**（オーナー確定②）: 消費側のマウントが one-shot effect で捨てるので、
// 🔁 の直後の 1 回だけに効く。普通の右クリックは従来どおり全部が既定から始まる。

/** 🔁 が次のマウントへ渡す設定。**幅・高さは入れない** —— 新しい素材の実寸に
 * 追随させるため（オーナー確定①）。 */
interface RetakeCarryOver {
  frameRate: number;
  seed: number;
  regenerateAudio: boolean;
  stage2Window: Stage2Window;
}

let pendingCarryOver: RetakeCarryOver | null = null;

function stashRetakeCarryOver(values: RetakeCarryOver): void {
  pendingCarryOver = values;
}

/** 読むだけ（捨てない）。lazy initializer から呼ぶので純関数でなければならない。 */
function peekRetakeCarryOver(): RetakeCarryOver | null {
  return pendingCarryOver;
}

/** 持ち越しを捨てる。消費側の one-shot effect と ❌ が呼ぶ。テストの
 * `afterEach` からも呼べるよう export してある（モジュール状態なので、
 * 捨て忘れると次のテストへ漏れる）。 */
export function clearRetakeCarryOver(): void {
  pendingCarryOver = null;
}

/** config の窓長上下限を読む。サーバが古くて欠けている・壊れている場合だけ
 * F1 の定数へ落ちる（型上は必須だが、実行時に来ないことはありうる）。 */
function resolveWindowLimit(value: number | undefined, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) && value > 0 ? Math.round(value) : fallback;
}
