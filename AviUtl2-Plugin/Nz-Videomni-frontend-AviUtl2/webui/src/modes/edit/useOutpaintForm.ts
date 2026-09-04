import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { apiClient as defaultApiClient, createApiClient } from "../../api/client";
import type { ApiClient } from "../../api/client";
import type { GenerateRequest } from "../../api/types";
import { bridge as defaultBridge } from "../../bridge";
import type { NativeBridge } from "../../bridge";
import { OUTPAINT_LORA_NAME } from "../../lora/controlLoras";
import type { GenerationPrefill } from "../../timeline/generationPrefill";
import { decideSourceTrim, trimQuery } from "../../timeline/sourceTrim";
import { useLoras } from "../inventory/useLoras";
import { FALLBACK_APP_CONFIG, MIN_NUM_FRAMES } from "../single/defaultConfig";
import { fileNameFromPath } from "../single/keyframeUtils";
import { FRAME_RATE_FALLBACK, FRAME_RATE_MIN, snapFrameRate, snapNumFrames } from "../single/paramUtils";
// クロスモード import。`useSourceUpload` は `modes/chained/` にあるが、ファイル
// 移動はしない —— 他ワークストリームが chained/ を占有しているため、ここで動かす
// と衝突する。Create（`modes/single/useGenerationForm.ts`）も同じ理由でこのフック
// を chained/ から直接 import しており、本ファイルはその先例に倣っているだけ。
import { useSourceUpload } from "../chained/useSourceUpload";
import type { UseSourceUploadResult } from "../chained/useSourceUpload";
import {
  BLEND_DILATION_MAX,
  BLEND_DILATION_MIN,
  canvasSize,
  centerPads,
  clampPad,
  comfortTokenEstimate,
  DEFAULT_BLEND_DILATION_STAGE1,
  isOverComfortBudget,
  maxNumFrames,
  outpaintReasons,
  resolveOutpaintComfortBudget,
  stage2FromStage1,
  ZERO_PADS,
} from "./outpaintGeometry";
import type { Pads } from "./outpaintGeometry";

// `OUTPAINT_LORA_NAME` (the control LoRA this panel pins) now lives in
// `lora/controlLoras.ts` — imported above, not redeclared here — so that
// module's `UI_HIDDEN_CONTROL_LORA_NAMES` can reference the same literal
// without a value-import into `lora/` (see that module's doc comment).

/** The strength the pinned adapter is applied at. 1.0 is the value the official
 * ComfyUI sample workflow uses for the In-Outpainting IC-LoRA (設計方針書 §3-2). */
export const OUTPAINT_LORA_STRENGTH = 1.0;

/** Accepted drag-and-drop extensions — mirrors native's `ui.pickFile({kind:
 * "video"})` filter, the same list Create's reference-video card uses. */
export const OUTPAINT_VIDEO_EXTENSIONS = ["mp4", "mov", "webm", "mkv"];

/** The source video's measured geometry. All three are `0` when unknown —
 * `fs.probeMediaInfo` never rejects for an unreadable/unsupported file, it
 * reports zeros (bridge contract v9), and the panel turns that into the
 * `mediaInfoUnknown` block rather than guessing. */
export interface OutpaintMediaInfo {
  durationSec: number;
  width: number;
  height: number;
}

const UNKNOWN_MEDIA_INFO: OutpaintMediaInfo = { durationSec: 0, width: 0, height: 0 };

export type PadSide = "left" | "right" | "top" | "bottom";

export interface UseOutpaintFormDeps {
  /** The SHARED prompt (`AppShell`'s `PromptBar`, threaded through
   * `EditScreen` -> `OutpaintingPanel`). The panel has no prompt field of its
   * own — Outpainting is one more consumer of the same bar Create/Chain read,
   * so there is exactly one prompt in the app. Omitted = empty, which the
   * `promptEmpty` gate then blocks on. */
  prompt?: string | undefined;
  /** Test/integration seam. Production omits it — the app-wide singleton bridge
   * is used, and the API client is the app-wide singleton too. Supplying one
   * binds BOTH (upload, media probe, `GET /loras` and `POST /generate`) to it,
   * so a test drives the whole panel through a single mock. */
  nativeBridge?: NativeBridge | undefined;
  /** Direct API-client override, for a test that wants to stub `getLoras`
   * without building a whole bridge. Takes precedence over `nativeBridge`. */
  apiClient?: ApiClient | undefined;
  /** The one-shot right-click payload (`AppShell` -> `EditScreen` ->
   * `OutpaintingPanel`). Consumed EXACTLY ONCE on mount to auto-load the
   * material; `AppShell` bumps `remountTokens.edit` per route, so a fresh
   * right-click always arrives as a fresh mount. */
  initialIntent?: GenerationPrefill | undefined;
  /** §3-134 (2026-09-04): the LOADED base model's engine family
   * (`BaseModelBlock.engine_family` — `"ltx"`, `"ltx25"`, …), owned by
   * `shell/AppShell.tsx` via `useBaseModels().activeEngineFamily`. It picks the
   * 快適上限 WARNING's token budget out of
   * `outpaintGeometry.OUTPAINT_COMFORT_TOKEN_BUDGETS` — the same "caller owns
   * the state" shape Create/Chain use for their own comfort markers.
   *
   * Omitted (every pre-existing unit test), `undefined` or `""` (before `GET
   * /models` lands, or offline) all mean "engine unknown" and take the
   * `COMFORT_TOKEN_BUDGET` fallback, i.e. exactly the pre-2026-09-04
   * behaviour. */
  engineFamily?: string | undefined;
}

export interface UseOutpaintFormResult {
  /** The source-video upload slot (`kind: "video"`). */
  source: UseSourceUploadResult;
  /** `fs.probeMediaInfo` output for the attached file — the kept region's size
   * and the ceiling for `num_frames`. Zeros = unknown. */
  mediaInfo: OutpaintMediaInfo;
  /** Uploads an already-resolved local path (drag-and-drop). No trim query: a
   * dropped file has no timeline ribbon behind it to measure. */
  attachByPath: (filePath: string, fileName: string) => void;

  /** The CURRENT pads, in absolute pixels. 0 起点・1px 刻み（2026-08-11）。 */
  pads: Pads;
  setPad: (side: PadSide, value: number) => void;

  /** 「センタリング」（設計方針書 §5-D、2026-08-12）。入のあいだは片辺を動かすと
   * 対辺に同じ値が入り、元動画が中央に残る。既定は切。
   *
   * PURELY AN INPUT AID: this flag never reaches the wire — `buildRequest` does
   * not read it and no request field corresponds to it. What it changes is only
   * how {@link setPad} writes into `pads`, and the pads themselves are what get
   * sent. 素材差し替え（`applyMediaInfo` の 0 リセット）でも切には戻さない ——
   * `ZERO_PADS` は対称なので不整合が生じない。 */
  centering: boolean;
  /** 切→入のときだけ各軸を等分し（`centerPads`、端数切捨て）、入→切では値に
   * 触らない。 */
  setCentering: (value: boolean) => void;

  /** The extended canvas. NOT guaranteed to sit on the 128 grid — pads are
   * never auto-corrected, so an off-grid canvas is a BLOCK REASON
   * (`canvasWidthOffGrid` / `canvasHeightOffGrid`) instead. `{0,0}` while the
   * source size is unknown. */
  canvas: { width: number; height: number };

  /** マスクブラーの膨張段数（stage 1）。`0..15`。UI はこの段数を実寸ピクセルへ
   * 直して見せるだけで、状態としては段数のまま持つ —— キャンバスが変われば
   * 同じ段数でも実寸は変わるので、ピクセルを保存すると嘘になる。 */
  blendDilation: number;
  setBlendDilation: (value: number) => void;

  /** The `num_frames` that will actually be sent: the stored value capped by
   * {@link numFramesCeiling}. DERIVED every render, never stored (§4-4). */
  numFrames: number;
  /** The dynamic ceiling from the source video's length + the current frame
   * rate. Re-derived every render for the same reason. */
  numFramesCeiling: number;
  setNumFrames: (value: number, snap: boolean) => void;
  frameRate: number;
  setFrameRate: (value: number) => void;
  seed: number;
  setSeed: (value: number) => void;

  /** Is `in-outpainting` installed? `false` also while `GET /loras` is failing
   * — an unknown adapter list is not a reason to let a doomed request through. */
  hasOutpaintLora: boolean;
  /** One code per failing gate; empty exactly when Generate is pressable. */
  validityReasons: string[];
  isValid: boolean;
  /** 快適上限: a WARNING, never part of `isValid`. The budget it is measured
   * against comes from {@link UseOutpaintFormDeps.engineFamily}. */
  isOverComfortBudget: boolean;
  /** The rough token count the warning quotes — `outpaintGeometry`'s
   * `comfortTokenEstimate` of the EXTENDED canvas. */
  comfortTokens: number;

  /** The `POST /generate` body. Only meaningful while `isValid`. */
  buildRequest: () => GenerateRequest;
}

/**
 * Owns the Outpainting panel's whole form: the source-video upload slot, the
 * measured media info behind it, the four pad sliders, the mask-blur dilation,
 * the duration / frame-rate / seed trio, and the gate list. The prompt is NOT
 * among them — it is the shared `PromptBar`'s, passed in.
 *
 * ## The two structural rules this hook exists to enforce
 *
 * 1. **Pads are absolute pixel amounts, never snapped.** Each side starts at 0
 *    and moves one pixel at a time; the only correction applied is
 *    {@link clampPad}'s range clamp (`[0, MAX_PAD]`). A canvas that is not
 *    a multiple of 128 is therefore perfectly reachable — and is stopped by the
 *    `canvasWidthOffGrid` / `canvasHeightOffGrid` gate rather than by rewriting
 *    the user's number (オーナー決定 2026-08-11). 第2弾でこの分担を徹底した:
 *    スライダーの上限は定数 `PAD_SLIDER_MAX`（220。マウスで扱える入力補助の
 *    可動域）、値そのものの上限は `MAX_PAD`（4096。API のパッド上限）で、どちらも
 *    元動画の寸法にも対辺の値にも依存しない。キャンバスの制約（128 の格子・4096
 *    の上限）は**すべて理由コード**が担保する。パッドは測定された素材の寸法が
 *    CHANGES したときだけ 0 に戻る（`applyMediaInfo`）。
 * 2. **The `num_frames` ceiling is never stored** (設計方針書 §4-4, the same
 *    single-source lesson as prefillSeed): it is derived from the CURRENT source
 *    duration and frame rate on every render, so detaching or replacing the
 *    source can't leave a stale clamp behind.
 */
export function useOutpaintForm(deps: UseOutpaintFormDeps = {}): UseOutpaintFormResult {
  const { nativeBridge, initialIntent, engineFamily } = deps;
  const apiClient = useMemo<ApiClient>(
    () => deps.apiClient ?? (nativeBridge ? createApiClient(nativeBridge) : defaultApiClient),
    [deps.apiClient, nativeBridge],
  );

  // The shared `PromptBar`'s text, owned by `AppShell`. Never mirrored into
  // local state: a second copy is the classic way for the two to drift.
  const prompt = deps.prompt ?? "";
  // 上下左右に描き足す量そのもの（絶対値・1px 刻み）。See rule 1 in the doc comment.
  const [pads, setPads] = useState<Pads>(ZERO_PADS);
  // 「センタリング」（§5-D）。入力の書き込み方だけを変えるスイッチで、リクエストの
  // 中身にはならない。
  const [centering, setCenteringState] = useState(false);
  const [blendDilation, setBlendDilationState] = useState(DEFAULT_BLEND_DILATION_STAGE1);
  const [numFramesInput, setNumFramesInput] = useState(FALLBACK_APP_CONFIG.generation_defaults.num_frames);
  // 台帳§3-71/§3-72: 生成 fps は必ず整数（`paramUtils.snapFrameRate` が正本）。
  // 初期値も通す —— 既定を非整数へ変えたときに、このパネルだけ素通しになるのを
  // 防ぐため。フォールバックは定数（`?? FALLBACK_APP_CONFIG…` と繋ぐと、スナップ
  // が弾いたその値がそのまま漏れる）。なお、このパネルが実 config ではなく
  // `FALLBACK_APP_CONFIG` を見ている件は本改修のスコープ外（別件）。
  const [frameRate, setFrameRateState] = useState(
    () => snapFrameRate(FALLBACK_APP_CONFIG.generation_defaults.frame_rate) ?? FRAME_RATE_FALLBACK,
  );
  /** fps 欄の setter。空欄（`Number("")` = 0）・非有限・範囲外は他パネルと同じ
   * 意味論（0 以下 → 1、60 超 → 60、小数 → 四捨五入）。 */
  const setFrameRate = useCallback((raw: number) => setFrameRateState(snapFrameRate(raw) ?? FRAME_RATE_MIN), []);
  const [seed, setSeed] = useState(FALLBACK_APP_CONFIG.generation_defaults.seed);

  const source = useSourceUpload("video", nativeBridge ? { nativeBridge } : {});
  const { uploadPath } = source;

  const attachByPath = useCallback(
    (filePath: string, fileName: string) => {
      void uploadPath(filePath, fileName);
    },
    [uploadPath],
  );

  // --- one-shot right-click auto-load -------------------------------------
  // Mirrors `ChainedScreen`'s #1 extend-video path exactly: the trim decision is
  // made from the SAME (item, selection) pair (`decideSourceTrim`), and
  // `trimQuery` returns `undefined` for every no-trim decision, so a
  // non-trimmable object produces a byte-identical plain upload.
  const autoLoadRef = useRef(false);
  useEffect(() => {
    if (autoLoadRef.current) return;
    autoLoadRef.current = true;
    // `intent` を見るのが load-bearing: `EditScreen` は両サブパネルを常時マウントするので、
    // Retake の右クリック（同じ `initialIntent` が届く）でここが素通しだと、見てもいない
    // パネルが同じファイルをもう一度アップロードしてしまう。判定基準はサブタブ選択
    // （`EditScreen.tsx` の `initialIntent?.intent === "outpaint"`）と揃え、双子は
    // `useRetakeForm.ts` の `snapshotSelection`。（台帳 §3-63）
    const selection = initialIntent?.intent === "outpaint" ? initialIntent.selection : undefined;
    const item = selection?.selected[0];
    const filePath = item?.filePath;
    if (!selection || !filePath) return;
    const decision = decideSourceTrim(item, selection);
    void uploadPath(filePath, fileNameFromPath(filePath), trimQuery(decision));
    // eslint-disable-next-line react-hooks/exhaustive-deps -- one-shot mount consume
  }, []);

  // --- media probe --------------------------------------------------------
  // Same shape as Create's reference-video probe (`useGenerationForm.ts`): keyed
  // off the upload id turning non-null, guarded against re-probing the same id,
  // with a cancelled/settled teardown that re-claims an interrupted probe.
  // `.catch()` is mandatory — `fs.probeMediaInfo` DOES reject on a BAD_REQUEST
  // (empty path), and an unhandled rejection fails the test run.
  const [mediaInfo, setMediaInfo] = useState<OutpaintMediaInfo>(UNKNOWN_MEDIA_INFO);

  /** Stores the measured info AND resets the pads when the source's KNOWN size
   * changes — the one place a pad is allowed to move on its own.
   *
   * 狭い規則である理由: リセットするのは「前に判明していた寸法」と「今判明した
   * 寸法」が違うときだけで、unknown への遷移では何もしない。同じファイルを添付し
   * 直したときや probe が一時的に失敗したときに、利用者が 1px ずつ入れた値を消さ
   * ないため。寸法不明のあいだスライダーは `known` ゲート（`OutpaintingPanel`）で
   * そもそも触れないので、これだけで「差し替えた素材に対して大きすぎるパッドが
   * 残る」穴は閉じる。
   *
   * `useCallback(..., [])` は必須 —— setter は安定なので依存は要らず、毎描画で
   * 別物になると下の probe effect の cleanup が `probedIdRef` を null に戻して
   * 無限に probe し続ける。 */
  const lastKnownSizeRef = useRef<string | null>(null);
  const applyMediaInfo = useCallback((info: OutpaintMediaInfo) => {
    setMediaInfo(info);
    if (info.width <= 0 || info.height <= 0) return;
    const size = `${info.width}x${info.height}`;
    if (lastKnownSizeRef.current !== null && lastKnownSizeRef.current !== size) setPads(ZERO_PADS);
    lastKnownSizeRef.current = size;
  }, []);

  const probedIdRef = useRef<string | null>(null);
  const sourceStatus = source.state.status;
  const sourceId = source.state.id;
  const sourceFilePath = source.state.filePath;
  useEffect(() => {
    if (sourceStatus !== "ready") {
      probedIdRef.current = null;
      applyMediaInfo(UNKNOWN_MEDIA_INFO);
      return;
    }
    if (!sourceId || probedIdRef.current === sourceId) return;
    probedIdRef.current = sourceId;
    if (!sourceFilePath) return;

    let cancelled = false;
    let settled = false;
    void (nativeBridge ?? defaultBridge)
      .request("fs.probeMediaInfo", { filePath: sourceFilePath })
      .then((info) => {
        settled = true;
        if (cancelled) return;
        applyMediaInfo({ durationSec: info.durationSec, width: info.width, height: info.height });
      })
      .catch(() => {
        settled = true;
        if (cancelled) return;
        applyMediaInfo(UNKNOWN_MEDIA_INFO);
      });
    return () => {
      cancelled = true;
      if (!settled) probedIdRef.current = null;
    };
  }, [sourceStatus, sourceId, sourceFilePath, nativeBridge, applyMediaInfo]);

  // --- pads ---------------------------------------------------------------
  const known = mediaInfo.width > 0 && mediaInfo.height > 0;

  const setPad = useCallback(
    (side: PadSide, value: number) => {
      // 上限は `clampPad` が持つ定数 `MAX_PAD` ひとつだけ（第2弾で対辺連動を廃止した
      // ため、ここで上限を計算して渡す必要が無くなった）。スライダーからも数値
      // ボックスからも同じここを通るので、pad が不正になり得る場所は 1 か所もない。
      //
      // センタリング中は対辺にも同じ値を入れる（§5-D）。**1 回の updater で 2 値を
      // 同時に書く**のが肝で、useEffect で相互に監視する形にはしない —— 非対称な
      // 中間状態が一瞬たりとも存在しないため、循環更新の余地が無い。
      const next = clampPad(value);
      setPads((prev) =>
        !centering
          ? { ...prev, [side]: next }
          : side === "top" || side === "bottom"
            ? { ...prev, top: next, bottom: next }
            : { ...prev, left: next, right: next },
      );
    },
    // `centering` は必須の依存 —— 落とすと入にしても古いクロージャが残り、連動
    // しなくなる（同フックの `setNumFrames` が同じ形の先例）。
    [centering],
  );

  /** 切→入の瞬間だけ各軸を等分する（§5-D）。入→切では値に触らない ——
   * 「揃えるのを手伝う」機能であって「揃った値を取り消す」機能ではない。
   *
   * `setPads` は updater 形式なので現在の pads を読まず、依存は空でよい。2 つの
   * setState が続くが、React 19 の自動バッチで 1 回の再描画にまとまる。 */
  const setCentering = useCallback((next: boolean) => {
    setCenteringState(next);
    if (next) setPads(centerPads);
  }, []);

  const canvas = useMemo(
    () => (known ? canvasSize(mediaInfo.width, mediaInfo.height, pads) : { width: 0, height: 0 }),
    [known, mediaInfo.width, mediaInfo.height, pads],
  );

  // --- マスクブラー --------------------------------------------------------
  const setBlendDilation = useCallback((value: number) => {
    if (!Number.isFinite(value)) return;
    setBlendDilationState(Math.min(BLEND_DILATION_MAX, Math.max(BLEND_DILATION_MIN, Math.round(value))));
  }, []);

  // --- duration -----------------------------------------------------------
  // Derived, never stored (§4-4). `maxNumFrames` floors at 9 for an unknown
  // duration, but an unknown duration is already a `mediaInfoUnknown` block, so
  // the panel is never generating off that floor.
  const numFramesCeiling = Math.min(
    FALLBACK_APP_CONFIG.limits.max_num_frames,
    mediaInfo.durationSec > 0 ? maxNumFrames(mediaInfo.durationSec, frameRate) : FALLBACK_APP_CONFIG.limits.max_num_frames,
  );
  const numFrames = Math.min(numFramesInput, numFramesCeiling);

  const setNumFrames = useCallback(
    (value: number, snap: boolean) => {
      // Free typing passes through untouched (mirroring Create's field), so the
      // user can type "104" without the first keystroke rewriting the box; the
      // 8n+1 grid is a submit gate (`numFramesOffGrid`), not a live correction.
      setNumFramesInput(snap ? snapNumFrames(value, MIN_NUM_FRAMES, numFramesCeiling) : value);
    },
    [numFramesCeiling],
  );

  // --- the pinned control LoRA -------------------------------------------
  const lorasState = useLoras({ apiClient });
  const hasOutpaintLora =
    lorasState.status === "ready" &&
    lorasState.loras.some((lora) => lora.name === OUTPAINT_LORA_NAME && lora.kind === "control");

  // --- gates --------------------------------------------------------------
  const validityReasons = outpaintReasons({
    prompt,
    sourceStatus,
    sourceVideoId: sourceId,
    sourceTrimFailed: source.state.trimFailed,
    sourceWidth: mediaInfo.width,
    sourceHeight: mediaInfo.height,
    sourceDurationSec: mediaInfo.durationSec,
    pads,
    numFrames,
    maxNumFrames: numFramesCeiling,
    hasOutpaintLora,
  });
  const isValid = validityReasons.length === 0;

  // 警告の閾値と、警告文が引用する概算トークン数。式は `comfortTokenEstimate`
  // ただ1つを通す —— ここで同じ式を書き直すと、切り捨ての有無のような細部が
  // 表示と判定で食い違う。予算はエンジン系統から引き（系統不明なら従来の
  // 40,000 へ）、超過は WARNING だけで `isValid` には一切関与しない。
  const comfortBudget = resolveOutpaintComfortBudget(engineFamily);
  const comfortTokens = comfortTokenEstimate(canvas.width, canvas.height, numFrames);
  const overComfort = known && isOverComfortBudget(canvas.width, canvas.height, numFrames, comfortBudget);

  const buildRequest = useCallback((): GenerateRequest => {
    // 既定（5 / 2）のときは両キーごと省略する —— 送っても意味は同じだが、
    // 省略しておけばこのフィールドを知らない古いサーバーにも通る（後方互換）。
    const blend =
      blendDilation === DEFAULT_BLEND_DILATION_STAGE1
        ? {}
        : { blend_dilation_stage1: blendDilation, blend_dilation_stage2: stage2FromStage1(blendDilation) };
    return {
      prompt: prompt.trim(),
      // The EXTENDED canvas — the server cross-checks
      // `width - pad_left - pad_right` against the reference video's own width.
      width: canvas.width,
      height: canvas.height,
      num_frames: numFrames,
      frame_rate: frameRate,
      seed,
      reference_video_id: sourceId,
      // Pinned, never user-selectable: the server requires exactly one
      // control-kind LoRA alongside `outpaint`.
      loras: [{ name: OUTPAINT_LORA_NAME, strength: OUTPAINT_LORA_STRENGTH }],
      outpaint: {
        pad_left: pads.left,
        pad_right: pads.right,
        pad_top: pads.top,
        pad_bottom: pads.bottom,
        ...blend,
        freeze_source_audio: true,
      },
    };
  }, [prompt, canvas.width, canvas.height, numFrames, frameRate, seed, sourceId, pads, blendDilation]);

  return {
    source,
    mediaInfo,
    attachByPath,
    pads,
    setPad,
    centering,
    setCentering,
    canvas,
    blendDilation,
    setBlendDilation,
    numFrames,
    numFramesCeiling,
    setNumFrames,
    frameRate,
    setFrameRate,
    seed,
    setSeed,
    hasOutpaintLora,
    validityReasons,
    isValid,
    isOverComfortBudget: overComfort,
    comfortTokens,
    buildRequest,
  };
}
