import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { AppConfig, GenerateRequest } from "../../api/types";
import { BridgeError, TIMELINE_MASK_PROGRESS_EVENT, bridge as defaultBridge } from "../../bridge";
import type { NativeBridge, TimelineMaskProgressData } from "../../bridge";
import { OUTPAINT_LORA_NAME } from "../../lora/controlLoras";
import { parseLoraPrompt } from "../../lora/loraTags";
import { accelerationRequestFields } from "../../shell/accelerationSettings";
import type { AccelerationSettings } from "../../shell/accelerationSettings";
import { comfortFramesForBudget } from "../../shell/comfortTable";
import { resolveOutpaintComfortBudget } from "../../shell/outpaintBudget";
import {
  clearInpaintSlots,
  inpaintTargetKey,
  markInpaintUploadStarted,
  setInpaintBusy,
  setInpaintTargetUpload,
  useInpaintSlots,
} from "../../timeline/inpaintSlots";
import type { InpaintSlots } from "../../timeline/inpaintSlots";
import { mapRangeToMaterialSec } from "../../timeline/retakeWindow";
import type { RetakeRangeResult } from "../../timeline/retakeWindow";
import { decideSourceTrim, trimQuery } from "../../timeline/sourceTrim";
import { FALLBACK_APP_CONFIG } from "../single/defaultConfig";
import { fileNameFromPath } from "../single/keyframeUtils";
import { FRAME_RATE_FALLBACK, snapFrameRate, snapNumFrames } from "../single/paramUtils";
// クロスモード import。`useSourceUpload` は `modes/chained/` にあるが動かさない
// —— `useOutpaintForm` / `useRetakeForm` が同じ理由（chained/ は別ワークストリーム
// の占有領域）でここから直接 import しており、本ファイルはその先例に倣っている。
import { useSourceUpload } from "../chained/useSourceUpload";
import {
  BLEND_DILATION_MAX,
  BLEND_DILATION_MIN,
  DEFAULT_BLEND_DILATION_STAGE1,
  DEFAULT_BLEND_DILATION_STAGE2,
  comfortTokenEstimate,
  isOverComfortBudget,
} from "./outpaintGeometry";
import {
  INPAINT_FRAMES_MAX,
  INPAINT_FRAMES_MIN,
  defaultInpaintFrames,
  placeInpaintWindow,
  roundUpTo128,
} from "./inpaintWindow";
import type { InpaintWindow } from "./inpaintWindow";

/** 画角拡張と同じ強度で制御系 LoRA を差す（公式ワークフローの 1.0）。 */
const INPAINT_LORA_STRENGTH = 1.0;

/** Generate が押せない理由コード。`GenerateReasonsNote` が
 * `editReasonMessages.buildInpaintReasonMessages` の表と突き合わせて 1 行ずつ出す。 */
export type InpaintReasonCode =
  /** 部分フィルタの右クリックがまだ来ていない。 */
  | "partialFilterMissing"
  /** 対象動画の右クリックがまだ来ていない。 */
  | "targetMissing"
  | "sourceUploading"
  | "sourceUploadFailed"
  /** §1-6: トリムを頼んだのに適用されなかった（別の場所を描き替えてしまう）。 */
  | "sourceTrimFailed"
  /** プロジェクトか対象動画の解像度が読めない。 */
  | "mediaInfoUnknown"
  /** オーナー裁定 D5: プロジェクト解像度 ≠ 対象動画の解像度。 */
  | "resolutionMismatch"
  /** 窓を対象動画の中に**置けない**（D3: 収まらなければ生成不可）。
   * 「窓が部分フィルタより短い」の注意文とは別物 —— あちらはブロックしない。 */
  | "windowNotCovered"
  /** マスクの描画・送信が飛行中。 */
  | "maskRendering";

/** マスク作成の進み具合。`idle` 以外のときは Generate を押せない。 */
export type InpaintMaskPhase = "idle" | "rendering" | "uploading";

/** Generate 押下時に予約を打ち直すための、確定した窓の置き場所。 */
export interface InpaintPlacement {
  /** 対象動画のレイヤー。 */
  layer: number;
  frameStart: number;
  /** 閉区間。 */
  frameEnd: number;
  numFrames: number;
  genFps: number;
}

export interface UseInpaintFormDeps {
  /** 共有プロンプト（`AppShell` の `PromptBar`）。Inpainting はプロンプト欄を
   * 持たない —— 他タブと同じ 1 本のバーを読むだけ。**空欄でも生成できる**
   * （物体を消すだけなら書くことが無い）。 */
  prompt?: string | undefined;
  /** `GET /config`。快適上限の線（`limits.comfort_budgets`）と seed の既定だけを
   * 読む。省略時は組み込みの既定。 */
  config?: AppConfig | undefined;
  /** テスト/結合の継ぎ目。省略時はアプリ全体の singleton bridge。 */
  nativeBridge?: NativeBridge | undefined;
  /** 読み込み中のベースモデルの系統。快適上限マーカーの線を配信テーブルから
   * 引くためだけに使う（`shell/outpaintBudget.ts`）。線が無ければ警告も出ない。 */
  engineFamily?: string | undefined;
  /** Settings の共有加速設定。Create/Chain/Retake と同じ「呼び手が状態を持つ」形。 */
  acceleration?: AccelerationSettings | undefined;
  /**
   * 生成要求を送っている最中か（`EditScreen` の `useGenerationSubmit`）。
   *
   * このフックが受け取るのは、保管庫の `busy` を**1 箇所から**書くため
   * （敵対的レビュー m1）。マスクの描画・送信はこのフックが持ち、生成要求の
   * 送信は画面が持つので、両方を知っているのはここしかない。
   */
  submitting?: boolean | undefined;
  /**
   * マスクの描画・送信が失敗したときに、そのコードを 1 回だけ知らせる（F3）。
   *
   * 文言への写像はしない —— 受け取った `EditScreen` がトースト
   * （`shell/ToastContext`）へ流す。**コールバックにしてある理由**は、失敗が
   * `await` の向こう側で起きるため: 画面が `renderAndUploadMask()` の戻り値を
   * 待ってからフックの `maskErrorCode` を読むと、その時点のクロージャは古い
   * 描画のもので、まだ空文字のことがある。押し出す側（ここ）から知らせれば、
   * その取り違えが構造的に起きない。
   */
  onMaskError?: ((code: string) => void) | undefined;
}

export interface UseInpaintFormResult {
  /** 保管庫の中身そのもの（部分フィルタ／対象動画）。パネルはこれを描く。 */
  slots: InpaintSlots;
  /** 対象動画のアップロード状態。保管庫の `videoId` が入っていれば、この
   * マウントがアップロードしていなくても `ready`（再マウント後の状態）。 */
  uploadStatus: "idle" | "uploading" | "ready" | "error";
  /** 対象動画の `video_id`。`null` = まだ使えない。 */
  targetVideoId: string | null;
  /** 対象動画のファイル名（表示用）。 */
  targetFileName: string;
  /** §1-6 のトリム失敗（保管庫といまのアップロードの OR）。 */
  trimFailed: boolean;

  /** プロジェクトの解像度（＝部分フィルタの解像度）。`getEditInfo` 前は 0。 */
  projectWidth: number;
  projectHeight: number;
  /** 生成 fps（プロジェクトの rate/scale を整数へスナップしたもの）。 */
  frameRate: number;

  /** 送信する解像度＝素材の実寸を 128 の倍数へ切り上げたキャンバス。 */
  canvasWidth: number;
  canvasHeight: number;
  /** キャンバス寸が測れているか（＝対象動画が入っているか）。なじみ幅の実寸を
   * 出すかどうかの判断に使う。 */
  canvasKnown: boolean;

  numFrames: number;
  /** `snap` の意味は Create/Chain の同名 setter と同じ: スライダー・ステッパーは
   * 8n+1 へ丸める、手打ちは素通し。ただし**確定時**（blur/Enter/離し）に
   * {@link commitNumFrames} が必ず格子へ乗せる。 */
  setNumFrames: (value: number, snap?: boolean) => void;
  /** 値が確定したときに呼ぶ。8n+1 の格子と [9,481] のクランプはここで効く。 */
  commitNumFrames: (value: number) => void;
  minFrames: number;
  maxFrames: number;

  seed: number;
  setSeed: (value: number) => void;

  /**
   * マスク周囲の「のりしろ」＝ブレンドの膨張段数（Stage-1 / Stage-2）。既定は
   * 5 / 2 で、setter は整数へ丸めて 0〜15 へクランプする（サーバーの
   * `ge=0, le=15` と同じ）。シードと同じくフックの状態なので、右クリックを
   * またいでも保持され、❌ で既定へ戻る。
   */
  blendStage1: number;
  blendStage2: number;
  setBlendStage1: (value: number) => void;
  setBlendStage2: (value: number) => void;

  /** 確定した窓（毎描画導出。置けなければ `null`）。 */
  window: InpaintWindow | null;
  /** 窓 → 素材秒の写像。 */
  mapping: RetakeRangeResult | null;

  /** 快適上限（助言のみ・生成は止めない）。線が無い系統では常に `false`。 */
  isOverComfortBudget: boolean;
  comfortTokens: number;
  /**
   * 快適上限の目盛りを打つフレーム数（`DurationField` の
   * `spillThresholdFrames`）。線が無い・キャンバス寸が不明なら `null` ＝
   * 目盛りを描かない（敵対的レビュー M1）。
   */
  comfortFrames: number | null;

  /** マスク作成の進み具合と、その進捗表示。 */
  maskPhase: InpaintMaskPhase;
  maskProgress: { index: number; total: number } | null;
  /** 直近のマスク失敗のエラーコード（`""` ＝ 失敗していない）。文言への写像は
   * `EditScreen` の仕事（`useObjectTracking` と同じ分担）。F3 以降、パネルは
   * これを描かない —— 案内はトーストで出る。 */
  maskErrorCode: string;

  validityReasons: InpaintReasonCode[];
  isValid: boolean;

  /** 確定した窓の置き場所（Generate 時の予約用）。 */
  placement: InpaintPlacement | null;

  /**
   * マスクを描いてアップロードする。成功なら `mask_video_id`、失敗なら `null`
   * （失敗の理由は {@link maskErrorCode} に入る）。
   *
   * 購読を**先に**張ってから RPC を投げる —— 進捗イベントは promise が
   * pending の間に流れてくるので、あとから張ると最初の数フレームを取り落とす。
   */
  renderAndUploadMask: () => Promise<string | null>;

  /** `POST /generate` の body。`isValid` かつマスクが取れたときだけ意味がある。 */
  buildRequest: (maskVideoId: string) => GenerateRequest;

  /** ❌: 両方のスロットと設定を捨てる。 */
  clearAll: () => void;
}

/**
 * 台帳 §3-55 Inpainting のフォーム全体。手本は `useRetakeForm`。
 *
 * ## `useRetakeForm` との 3 つの違い
 *
 * 1. **`initialIntent` を読まない**。入力は右クリック**2 回**（部分フィルタと
 *    対象動画）で届くので、1 回きりのペイロードには収まらない。値は
 *    `timeline/inpaintSlots.ts` の保管庫にあり、このフックは読むだけ ——
 *    だから Edit の再マウント（`remountTokens.edit`）を挟んでも、2 回目の
 *    右クリックが 1 回目を消さない。
 * 2. **幅・高さのつまみが無い**（オーナー裁定 D5）。出力解像度は対象動画の
 *    実寸で決まり、送るのはそれを 128 の倍数へ切り上げたキャンバスだけ。
 *    プロジェクト解像度と一致しない素材は**生成させない**（伸縮もしない）。
 * 3. **マスクは Generate 押下時に作る**（D8）。独立したボタンは置かず、
 *    描画→アップロード→送信を一続きで行う。マスク動画は利用者に見せない（D7）。
 */
export function useInpaintForm(deps: UseInpaintFormDeps = {}): UseInpaintFormResult {
  const { nativeBridge, engineFamily, acceleration } = deps;
  const submitting = deps.submitting ?? false;
  const config = deps.config ?? FALLBACK_APP_CONFIG;
  const prompt = deps.prompt ?? "";
  const bridge = nativeBridge ?? defaultBridge;

  const slots = useInpaintSlots();
  const partialFilter = slots.partialFilter;
  const target = slots.target;

  // --- プロジェクトの解像度と fps（マウント時 1 回） -----------------------
  // `config.generation_defaults` は**使わない**: 比べる相手は「いま開いている
  // プロジェクトの解像度」であって、サーバーの生成既定ではない（D5）。
  const [editInfo, setEditInfo] = useState<{ width: number; height: number; rate: number; scale: number } | null>(
    null,
  );
  useEffect(() => {
    let alive = true;
    void (async () => {
      try {
        const info = await bridge.request("getEditInfo", {});
        if (!alive) return;
        setEditInfo({ width: info.width, height: info.height, rate: info.rate, scale: info.scale });
      } catch {
        // 読めなければ `mediaInfoUnknown` で止まる。推測で埋めない。
      }
    })();
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- one-shot mount read
  }, []);

  const [seed, setSeed] = useState(config.generation_defaults.seed);

  // --- マスク周囲の「のりしろ」（ブレンドの膨張段数） -----------------------
  // 既定 5 / 2 は公式ワークフロー（node 5266 / 5226）と同じ値。丸めとクランプの
  // 式は `useOutpaintForm.setBlendDilation` と同じもの ——「受け付ける値」の
  // 定義が 2 か所に割れないよう、定数も関数の形もそちらへ揃えてある。
  const [blendStage1, setBlendStage1State] = useState(DEFAULT_BLEND_DILATION_STAGE1);
  const [blendStage2, setBlendStage2State] = useState(DEFAULT_BLEND_DILATION_STAGE2);
  const setBlendStage1 = useCallback((value: number) => {
    if (!Number.isFinite(value)) return;
    setBlendStage1State(Math.min(BLEND_DILATION_MAX, Math.max(BLEND_DILATION_MIN, Math.round(value))));
  }, []);
  const setBlendStage2 = useCallback((value: number) => {
    if (!Number.isFinite(value)) return;
    setBlendStage2State(Math.min(BLEND_DILATION_MAX, Math.max(BLEND_DILATION_MIN, Math.round(value))));
  }, []);

  // --- 対象動画のアップロード（対象 1 つにつき 1 回） ------------------------
  const source = useSourceUpload("video", nativeBridge ? { nativeBridge } : {});
  const { uploadPath } = source;
  /** この対象の一意鍵。`uploadStartedFor` と突き合わせる唯一の値。 */
  const targetKey = inpaintTargetKey(target, slots.targetVersion);
  const uploadStartedFor = slots.uploadStartedFor;
  useEffect(() => {
    if (!target || targetKey === null) return;
    // もう上がっている、または誰か（＝以前のこのフック）が既に始めている。
    // 「始めた」の記録は**保管庫**にあるので、画面が作り直されても二重に
    // 上げない（敵対的レビュー M3）。
    if (target.videoId !== null) return;
    if (uploadStartedFor === targetKey) return;
    const filePath = target.item.filePath;
    // 裏のファイルが無い対象は上げようがない。`uploadStatus` が `error` を
    // 答えるので「読み込み中」のまま止まることはない（敵対的レビュー m3）。
    if (!filePath) return;
    markInpaintUploadStarted(targetKey);
    const decision = decideSourceTrim(target.item, target.selection);
    void uploadPath(filePath, fileNameFromPath(filePath), trimQuery(decision));
  }, [target, targetKey, uploadStartedFor, uploadPath]);

  // アップロードが決着したら保管庫へ書き戻す（次のマウントが読めるように）。
  const sourceId = source.state.id;
  const sourceTrimFailed = source.state.trimFailed;
  const sourceStatus = source.state.status;
  useEffect(() => {
    if (sourceStatus !== "ready" || !sourceId) return;
    setInpaintTargetUpload(sourceId, sourceTrimFailed);
  }, [sourceStatus, sourceId, sourceTrimFailed]);

  const targetVideoId = target?.videoId ?? sourceId ?? null;
  const trimFailed = (target?.trimFailed ?? false) || sourceTrimFailed;
  const uploadStatus: UseInpaintFormResult["uploadStatus"] = !target
    ? "idle"
    : targetVideoId
      ? "ready"
      : // 裏のファイルが無い対象は、待っても永遠に ready にならない。「読み込み
        // 中」で固まるより、失敗として言い切る（敵対的レビュー m3）。
        !target.item.filePath || sourceStatus === "error"
        ? "error"
        : "uploading";

  // --- フレーム数（部分フィルタが来るたびに再シード） ----------------------
  // lazy initializer が保管庫を直読みするので、右クリック直後の**最初の描画**
  // から正しい値が出る（effect の 1 テンポ遅れで既定値がちらつかない）。
  const [numFrames, setNumFramesState] = useState(() =>
    slots.partialFilter
      ? defaultInpaintFrames(slots.partialFilter.frameStart, slots.partialFilter.frameEnd)
      : FALLBACK_APP_CONFIG.generation_defaults.num_frames,
  );
  // 再シードの判定は**部分フィルタスロットの改訂番号**だけ（`inpaintSlots.ts`
  // の `version` の doc 参照）。対象動画を差し替えただけで打ち直した値が既定へ
  // 戻る、という事故をここで塞いでいる。
  const seededVersionRef = useRef(slots.version);
  useEffect(() => {
    if (slots.version === seededVersionRef.current) return;
    seededVersionRef.current = slots.version;
    if (!slots.partialFilter) return;
    setNumFramesState(defaultInpaintFrames(slots.partialFilter.frameStart, slots.partialFilter.frameEnd));
  }, [slots.version, slots.partialFilter]);

  const setNumFrames = useCallback((raw: number, snap = true) => {
    if (!Number.isFinite(raw)) return;
    setNumFramesState(snap ? snapNumFrames(raw, INPAINT_FRAMES_MIN, INPAINT_FRAMES_MAX) : raw);
  }, []);
  /**
   * 値が確定した瞬間に 8n+1 の格子へ乗せる。
   *
   * Create/Chain は格子外れを**理由コードで止める**（黙って直さない）が、
   * こちらには格子外れのコードが無い（実装計画 §7.5-10 の理由 9 種に入って
   * いない）。ならば「確定時に必ず正しい値になる」方を選ぶ —— 手打ち中の
   * 途中の数字は素通しのままなので、入力の邪魔にはならない。
   */
  const commitNumFrames = useCallback((raw: number) => {
    if (!Number.isFinite(raw)) return;
    setNumFramesState(snapNumFrames(raw, INPAINT_FRAMES_MIN, INPAINT_FRAMES_MAX));
  }, []);

  // --- 窓（毎描画導出・状態として持たない） --------------------------------
  const placedWindow = useMemo<InpaintWindow | null>(() => {
    if (!partialFilter || !target) return null;
    return placeInpaintWindow({
      filterStart: partialFilter.frameStart,
      filterEnd: partialFilter.frameEnd,
      ribbonStart: target.item.frameStart,
      ribbonEnd: target.item.frameEnd,
      frames: numFrames,
    });
  }, [partialFilter, target, numFrames]);

  const mapping = useMemo<RetakeRangeResult | null>(() => {
    if (!target || !placedWindow) return null;
    return mapRangeToMaterialSec(target.item, target.selection, {
      rangeStart: placedWindow.windowStart,
      // `rangeEnd` は閉区間（`timeline/selectionRange.ts`）なので、窓の尻尾を
      // そのまま渡す。
      rangeEnd: placedWindow.windowEnd,
    });
  }, [target, placedWindow]);

  // --- 解像度と fps --------------------------------------------------------
  const projectWidth = editInfo?.width ?? 0;
  const projectHeight = editInfo?.height ?? 0;
  const targetWidth = target?.item.mediaWidth ?? 0;
  const targetHeight = target?.item.mediaHeight ?? 0;
  const canvasWidth = roundUpTo128(targetWidth);
  const canvasHeight = roundUpTo128(targetHeight);

  // 生成 fps はプロジェクトの rate/scale。部分フィルタのスナップショットが
  // 先に届くので普通はそちらから取れ、まだ無ければ `getEditInfo` の値を使う。
  // §3-71/§3-72: 整数へのスナップは `paramUtils.snapFrameRate` が唯一の正本。
  const frameRate = useMemo(() => {
    const rate = partialFilter?.rate ?? editInfo?.rate ?? 0;
    const scale = partialFilter?.scale ?? editInfo?.scale ?? 0;
    if (!(rate > 0) || !(scale > 0)) return FRAME_RATE_FALLBACK;
    return snapFrameRate(rate / scale) ?? FRAME_RATE_FALLBACK;
  }, [partialFilter?.rate, partialFilter?.scale, editInfo?.rate, editInfo?.scale]);

  // --- 快適上限（助言のみ） ------------------------------------------------
  // 線は画角拡張と同じ配信値（`outpaint_budget`）。評価するのは**キャンバス寸**
  // ——実際に生成が回る大きさがそれだから。
  const comfortBudget = resolveOutpaintComfortBudget(config.limits, engineFamily);
  const comfortTokens = comfortTokenEstimate(canvasWidth, canvasHeight, numFrames);
  const canvasKnown = canvasWidth > 0 && canvasHeight > 0;
  // 判定式は書き直さず、画角拡張と**同じ関数**を呼ぶ（敵対的レビュー m5）——
  // 「超えたか」の定義が 2 箇所に分かれると、片方だけ直る事故になる。
  const overComfort =
    canvasKnown && comfortBudget !== null && isOverComfortBudget(canvasWidth, canvasHeight, numFrames, comfortBudget);
  // スライダーに打つ目盛り。式の逆関数は `shell/comfortTable.ts` が正本で、
  // 32（潜在1マス＝32px角）と 8（潜在1枚＝8フレーム）は LTX の VAE 幾何。
  const comfortFrames =
    canvasKnown && comfortBudget !== null
      ? comfortFramesForBudget(canvasWidth, canvasHeight, comfortBudget, INPAINT_FRAMES_MIN, INPAINT_FRAMES_MAX, 32, 8)
      : null;

  // --- マスク作成 ----------------------------------------------------------
  const [maskPhase, setMaskPhase] = useState<InpaintMaskPhase>("idle");
  const [maskProgress, setMaskProgress] = useState<{ index: number; total: number } | null>(null);
  const [maskErrorCode, setMaskErrorCode] = useState("");

  // 最新のコールバックを ref で持つ（`useGenerationSubmit` と同じ作法）。呼び手が
  // 毎描画あたらしい関数を渡しても `renderAndUploadMask` の同一性が揺れない。
  const onMaskErrorRef = useRef(deps.onMaskError);
  onMaskErrorRef.current = deps.onMaskError;
  /** 失敗を 1 か所から書く: 状態（パネルは読まないが通しテストが見る）と、
   * 画面のトースト（F3）。片方だけ書き忘れる余地を残さない。 */
  const failMask = useCallback((code: string) => {
    setMaskErrorCode(code);
    onMaskErrorRef.current?.(code);
  }, []);

  const renderAndUploadMask = useCallback(async (): Promise<string | null> => {
    if (!partialFilter || !placedWindow) return null;
    setMaskErrorCode("");
    setMaskProgress(null);
    setMaskPhase("rendering");
    // 購読が先（実装計画 §7.5-10）。RPC を投げてから張ると、最初の数フレームの
    // 進捗を取り落とす —— 本番では pending の間に流れてくるのが常態。
    const unsubscribe = bridge.on(TIMELINE_MASK_PROGRESS_EVENT, (data) => {
      const push = data as TimelineMaskProgressData | null;
      if (!push || typeof push.index !== "number") return;
      setMaskProgress({ index: push.index, total: push.total });
    });
    try {
      const rendered = await bridge.request("timeline.renderMaskVideo", {
        layer: partialFilter.layer,
        frameStart: partialFilter.frameStart,
        frameEnd: partialFilter.frameEnd,
        windowStart: placedWindow.windowStart,
        windowEnd: placedWindow.windowEnd,
      });
      setMaskPhase("uploading");
      // **トリムは付けない**: マスクは窓そのものを描いたものなので、切り出す
      // 範囲が無い（対象動画の方だけがリボンのトリムを持つ）。
      const { status, body } = await bridge.request("backend.uploadFile", {
        kind: "video",
        filePath: rendered.filePath,
      });
      const id =
        status >= 200 && status < 300 && body && typeof (body as Record<string, unknown>).video_id === "string"
          ? ((body as Record<string, unknown>).video_id as string)
          : null;
      if (!id) {
        failMask("MASK_UPLOAD_FAILED");
        return null;
      }
      return id;
    } catch (err) {
      // 3 つの native コード（`MASK_SEED_INVALID`/`MASK_BUSY`/`MASK_FAILED`）は
      // 画面がそれぞれの文言へ写してトーストに出す。ここは運ぶだけ。
      failMask(err instanceof BridgeError ? String(err.code) : "MASK_FAILED");
      return null;
    } finally {
      unsubscribe();
      setMaskPhase("idle");
      setMaskProgress(null);
    }
  }, [bridge, failMask, partialFilter, placedWindow]);

  // 保管庫へ「動いている最中」を知らせる。`AppShell` はこれを読んで、途中に
  // 飛び込んできた右クリックを断る（敵対的レビュー m1）。書き手はこの 1 本だけ。
  const busy = maskPhase !== "idle" || submitting;
  useEffect(() => {
    setInpaintBusy(busy);
  }, [busy]);
  // アンマウント時に立てっぱなしにしない（画面が消えたら動いてもいない）。
  useEffect(() => () => setInpaintBusy(false), []);

  // --- ゲート --------------------------------------------------------------
  const validityReasons: InpaintReasonCode[] = [];
  if (!partialFilter) validityReasons.push("partialFilterMissing");
  if (!target) validityReasons.push("targetMissing");
  else {
    if (uploadStatus === "uploading") validityReasons.push("sourceUploading");
    else if (uploadStatus === "error") validityReasons.push("sourceUploadFailed");
    if (trimFailed) validityReasons.push("sourceTrimFailed");
    // 解像度の突き合わせは**部分フィルタも届いてから**（F2・実機ゲート G3）。
    // 部分フィルタが無い間は「まず右クリックしてください」だけが要るのであって、
    // そこに解像度の話を重ねると、まだ何もしていない人に 2 行の説教が出る。
    if (partialFilter) {
      if (projectWidth <= 0 || projectHeight <= 0 || targetWidth <= 0 || targetHeight <= 0) {
        validityReasons.push("mediaInfoUnknown");
      } else if (projectWidth !== targetWidth || projectHeight !== targetHeight) {
        // D5: サーバーも同じ検査をして 422 を返す。伸縮は**どちらの側でも**しない。
        validityReasons.push("resolutionMismatch");
      }
    }
  }
  if (partialFilter && target && (!placedWindow || !mapping?.ok || mapping.frameCount !== numFrames)) {
    // 窓がリボンに収まらない／素材秒へ写せない／写した結果が短くなった
    // （＝静止フレームの尻尾に掛かっている）。どれも「その長さでは描き替え
    // られない」なので 1 つの理由にまとめる。
    validityReasons.push("windowNotCovered");
  }
  if (maskPhase !== "idle") validityReasons.push("maskRendering");
  const isValid = validityReasons.length === 0;

  // --- 予約の置き場所 ------------------------------------------------------
  const placement = useMemo<InpaintPlacement | null>(() => {
    if (!target || !placedWindow) return null;
    return {
      layer: target.item.layer,
      frameStart: placedWindow.windowStart,
      frameEnd: placedWindow.windowEnd,
      numFrames: placedWindow.frames,
      genFps: frameRate,
    };
  }, [target, placedWindow, frameRate]);

  const buildRequest = useCallback(
    (maskVideoId: string): GenerateRequest => {
      // **アップロード後ファイルの時間軸**へ直す（`useRetakeForm` と同じ式）。
      // トリムしていれば `trimOffsetSec` はその切り出し開始秒、していなければ 0。
      const windowStartSec = mapping?.ok ? mapping.startSec - mapping.trimOffsetSec : 0;
      // プロンプトの `<lora:…>` タグは他タブとまったく同じ扱い: 同じパーサで
      // 指示文と `loras[]` に分け、両方を送る。空欄は正当な使い方なので、
      // ここでは何も足さない。
      const { strippedPrompt, loras } = parseLoraPrompt(prompt);
      return {
        prompt: strippedPrompt.trim(),
        // 送るのは**キャンバス**（素材寸を 128 の倍数へ切り上げたもの）。余白は
        // サーバーが素材の実寸から導出する（正本＝素材ファイルの実寸 1 つ）。
        width: canvasWidth,
        height: canvasHeight,
        num_frames: numFrames,
        frame_rate: frameRate,
        seed,
        reference_video_id: targetVideoId,
        // 制御系 LoRA は画角拡張と同じものを固定で差す（サーバーが要求する）。
        // 手打ちのタグ由来 LoRA はその後ろに並べる。
        loras: [{ name: OUTPAINT_LORA_NAME, strength: INPAINT_LORA_STRENGTH }, ...loras],
        inpaint: {
          mask_video_id: maskVideoId,
          window_start_sec: Math.max(0, windowStartSec),
          // のりしろは**常に**載せる（既定のままでも）。画面の数字と送った値が
          // 食い違う経路を作らないため（台帳 §2-10 ①）。
          blend_dilation_stage1: blendStage1,
          blend_dilation_stage2: blendStage2,
        },
        // 既定から動かしていなければキーごと出ない（リクエストは従来と同形）。
        ...accelerationRequestFields(acceleration),
      };
    },
    [
      mapping,
      prompt,
      canvasWidth,
      canvasHeight,
      numFrames,
      frameRate,
      seed,
      blendStage1,
      blendStage2,
      targetVideoId,
      acceleration,
    ],
  );

  const clearAll = useCallback(() => {
    clearInpaintSlots();
    source.clear();
    setSeed(config.generation_defaults.seed);
    setBlendStage1State(DEFAULT_BLEND_DILATION_STAGE1);
    setBlendStage2State(DEFAULT_BLEND_DILATION_STAGE2);
    setMaskErrorCode("");
    setMaskProgress(null);
    setMaskPhase("idle");
    // 予約席は**触らない**: Inpainting は Generate 押下時にしか席を取らない
    // （右クリックの `placement` は `null`）ので、❌ の時点で片付けるものが無い。
  }, [source, config.generation_defaults.seed]);

  return {
    slots,
    uploadStatus,
    targetVideoId,
    targetFileName: source.state.fileName ?? (target?.item.filePath ? fileNameFromPath(target.item.filePath) : ""),
    trimFailed,
    projectWidth,
    projectHeight,
    frameRate,
    canvasWidth,
    canvasHeight,
    canvasKnown,
    numFrames,
    setNumFrames,
    commitNumFrames,
    minFrames: INPAINT_FRAMES_MIN,
    maxFrames: INPAINT_FRAMES_MAX,
    seed,
    setSeed,
    blendStage1,
    blendStage2,
    setBlendStage1,
    setBlendStage2,
    window: placedWindow,
    mapping,
    isOverComfortBudget: overComfort,
    comfortTokens,
    comfortFrames,
    maskPhase,
    maskProgress,
    maskErrorCode,
    validityReasons,
    isValid,
    placement,
    renderAndUploadMask,
    buildRequest,
    clearAll,
  };
}
