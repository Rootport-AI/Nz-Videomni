import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { apiClient as defaultApiClient, createApiClient } from "../../api/client";
import type { ApiClient } from "../../api/client";
import { EditSubTabs } from "./EditSubTabs";
import type { EditSubMode } from "./EditSubTabs";
import type { EditSubTabsDisabled } from "../../shell/featureScope";
import { InpaintingPanel } from "./InpaintingPanel";
import { OutpaintingPanel } from "./OutpaintingPanel";
import { RetakePanel } from "./RetakePanel";
import { bridge as defaultBridge } from "../../bridge";
import type { NativeBridge } from "../../bridge";
import type { AccelerationSettings } from "../../shell/accelerationSettings";
import { useToasts } from "../../shell/ToastContext";
import { useStrings } from "../../i18n/LanguageContext";
import { JobLedger } from "../../jobs/JobLedger";
import { useJobsContext } from "../../jobs/JobsContext";
import type { GenerationPrefill } from "../../timeline/generationPrefill";
import { bindToJob, reservePlacement, rollbackReservedPlacement } from "../../timeline/provisionalReservation";
import { useInpaintSlots } from "../../timeline/inpaintSlots";
import { GenerateButtonBar } from "../single/GenerateButtonBar";
import { GenerateReasonsNote } from "../single/GenerateReasonsNote";
import { useConfig } from "../single/useConfig";
import { FALLBACK_APP_CONFIG } from "../single/defaultConfig";
import { useGenerationSubmit } from "../single/useGeneration";
import {
  buildInpaintReasonMessages,
  buildOutpaintReasonMessages,
  buildRetakeReasonMessages,
} from "./editReasonMessages";
import { OUTPAINT_LORA_NAME } from "../../lora/controlLoras";
import { useInpaintForm } from "./useInpaintForm";
import { useOutpaintForm } from "./useOutpaintForm";
import { useRetakeForm } from "./useRetakeForm";
import "./EditScreen.css";

export interface EditScreenProps {
  /** The routed right-click payload, delivered by `AppShell` alongside a bumped
   * `remountTokens.edit` (so this screen remounts and the lazy initializers
   * below re-read it). `undefined` for a manual tab switch. */
  initialIntent?: GenerationPrefill | undefined;
  /** The SHARED prompt (`AppShell`'s `PromptBar`, above the mode tabs). Edit's
   * panels read it exactly as Create/Chain do — there is one prompt in the app,
   * and no screen keeps a copy of its own. */
  prompt?: string | undefined;
  /** The three props the job ledger in the generation column needs, all
   * threaded from `AppShell` in the same shape Create/Chain take them. */
  baseUrl?: string | null | undefined;
  highlightedJobId?: string | null | undefined;
  nativeBridge?: NativeBridge | undefined;
  /** Reports a freshly submitted `job_id` back up to `AppShell`, which makes it
   * the highlighted job — so the card appears outlined in the ledger beside the
   * form. */
  onJobSubmitted?: ((jobId: string) => void) | undefined;
  /** §3-98 P5 / §3-102: サブタブのうち、**読み込み中のベースモデルの
   * エンジンが実行できない**もの（`shell/featureScope.ts` の
   * `editSubTabsDisabledFor`）。`AppShell` が確定済みの真偽値として渡すので、
   * この画面もパネルもフィーチャ名を一切知らない（`ChainedScreen` が
   * `chainPanels` を受け取るのと同じ作法）。省略時はどちらも有効。 */
  subTabsDisabled?: EditSubTabsDisabled | undefined;
  /** §3-134 (2026-09-04): the LOADED base model's engine family
   * (`useBaseModels().activeEngineFamily`), passed straight through to
   * `useOutpaintForm`, where it picks the 快適上限 warning's token budget out of
   * the SERVED table (`shell/outpaintBudget.ts`, §3-135). Create/Chain take the
   * very same prop for their own comfort markers. Omitted/`""` ⇒ "engine
   * unknown" ⇒ **no line, hence no warning**, so every direct-render test that
   * predates it keeps compiling. */
  engineFamily?: string | undefined;
  /** §1-27 (2026-09-05): Settings' shared acceleration choice, owned by
   * `AppShell` — same "caller owns the state" shape Create/Chain take it in.
   * Threaded to both `useOutpaintForm` and `useRetakeForm`. */
  acceleration?: AccelerationSettings | undefined;
  /** §3-165: whether the server reports SageAttention as installed — with
   * {@link engineFamily} and {@link acceleration} it picks the served comfort
   * row whose CHAIN budget sizes the Retake stage-2 window labels (the same
   * three inputs `ChainedScreen` passes to `useChainForm`). Omitted ⇒ `null`. */
  sageAvailable?: boolean | null | undefined;
  /** §3-165: the LOADED base model's display name (`/models`
   * `base_models[].display_name`), printed in the Retake stage-2 window labels.
   * Omitted ⇒ `""`. */
  engineLabel?: string | undefined;
}

/** The Edit mode screen (2026-08-09). Until this day the Edit tab was a
 * disabled mock in `shell/ModeTabs.tsx`; it is now a real `AppMode` with panels
 * of its own, so the Outpainting work (`Docs/PENDING_TASKS_CLOSED.md` §3-70,
 * filed as §1-13 at the time; design notes `Docs/OUTPAINTING_DESIGN_NOTES.md`
 * §5-8) has somewhere to land.
 *
 * ## Layout (owner feedback batch 2, 2026-08-09)
 *
 * The screen is the SAME two-column arrangement Create and Chain use, built out
 * of the very same classes so there is one layout to reason about (and one set
 * of breakpoints): `.single-layout` (`modes/single/SingleScreen.css`) holds a
 * form column on the left and `.generation-column`
 * (`styles/formAccordion.css`) — Generate button, submit-error row, "why can't
 * I press Generate?" note, then the job ledger — on the right. Below
 * `AppShell.css`'s 680px `mode-body` container query both collapse into one
 * column with the generation panel lifted above the form, exactly as they do on
 * Create/Chain; none of that behaviour is re-implemented here.
 *
 * Before this the panels were a single full-width stack with the ledger at the
 * very bottom, where it was effectively invisible.
 *
 * ## Why the Outpainting form state lives HERE and not in the panel
 *
 * The Generate button belongs in the generation column (the Create/Chain
 * placement rule), and its `disabled`/`onGenerate` need the form — so the
 * screen owns `useOutpaintForm` + `useGenerationSubmit` and `OutpaintingPanel`
 * takes the result as a prop. That is precisely how `SingleScreen` owns
 * `useGenerationForm` and hands it to `GenerationForm`, and how `ChainedScreen`
 * owns `useChainForm`: the screen owns the state, the form component draws it.
 * When Retake grows a real body it will own its hook here the same way.
 *
 * The sub-tab selection is plain `useState` — deliberately NOT lifted to
 * `AppShell`. W0 (共通スパイン) made Edit a right-click destination, so a route
 * CAN pre-select a sub-tab; it does so through the one-shot `initialIntent` +
 * remount arrangement every other screen already uses (a lazy `useState`
 * initializer, never a prop-following effect), which needs no shell-level
 * state. The selection survives a main-tab switch anyway, since `AppShell`
 * keeps every mode screen mounted and merely hides the inactive ones.
 *
 * Both sub-panels stay mounted and are hidden with the plain `hidden` attribute
 * (the UA's `[hidden] { display: none }`), mirroring `AppShell`'s own
 * always-mounted arrangement so a panel's form state survives a sub-tab switch.
 * They carry NO `role="tabpanel"` on purpose — see `EditSubTabs`'s doc comment.
 *
 * The job ledger is the shared `jobs/JobLedger`, the very same component Create
 * and Chain mount, fed by the same app-wide `JobsProvider` poll. It sits
 * OUTSIDE the sub-tab switch, so it is there whichever sub-tab is showing;
 * only the Generate group above it belongs to Outpainting and is dropped on the
 * Retake sub-tab (there is nothing to generate there yet). */
/** サブタブの並び順（画面の並びと同じ）。{@link fallbackSubMode} が
 * 「最初に生きているタブ」を選ぶときの唯一の順序表。 */
const SUB_TAB_ORDER: readonly EditSubMode[] = ["retake", "outpainting", "inpainting"];

/**
 * 行き先のサブタブが灰色だったときの逃げ先 —— **並び順で最初に生きているもの**。
 *
 * §3-98 P5 / §3-102 で 2 つだったころは三項 1 本で済んでいたが、§3-55 で
 * Inpainting が実体化して 3 つになったので、「もう片方」では答えが決まらない。
 * ここを 1 本の純関数にしておけば、4 つ目が来ても規則は増えない。
 *
 * 全部が灰色ならこの関数へは到達しない: 3 つ**すべて**が閉じたときだけ
 * `disabledModesFor` が Edit タブごと落とすので、この画面はマウントされて
 * いない（`shell/featureScope.ts` の `CONTAINER_TARGETS`）。それでも最後の
 * 保険として、見つからなければ要求されたタブをそのまま返す。
 */
function fallbackSubMode(disabled: EditSubTabsDisabled, requested: EditSubMode): EditSubMode {
  return SUB_TAB_ORDER.find((id) => !disabled[id]) ?? requested;
}

export function EditScreen({
  initialIntent,
  prompt,
  baseUrl = null,
  highlightedJobId = null,
  nativeBridge,
  onJobSubmitted,
  subTabsDisabled = { retake: false, outpainting: false, inpainting: false },
  engineFamily,
  acceleration,
  sageAvailable,
  engineLabel,
}: EditScreenProps = {}) {
  const strings = useStrings();
  // Consumed EXACTLY ONCE, in the lazy initializer: `AppShell` bumps
  // `remountTokens.edit` on every Edit-系 route, so a fresh right-click arrives
  // as a fresh mount and re-runs this. Only `"outpaint"` steers away from the
  // default; `"retake"` and every other/absent intent land on Retake, which is
  // also the screen's plain default.
  //
  // §3-98 P5 / §3-102 のフォールバック: 行き先のサブタブが**そのベースモデルで
  // 灰色**なら、もう片方へ回す —— 灰色のタブを選択状態にすると、押せないタブの
  // 下に生成群が出てしまう（`subMode` が生成群の出し分けそのものだから）。
  //
  // 両方が無効な場合はここへ到達しない: 2つのサブタブが**両方**閉じたときだけ
  // `disabledModesFor` が Edit タブごと落とすので、この画面はそもそも
  // マウントされていない（`shell/featureScope.ts` の `CONTAINER_TARGETS`）。
  // なので下の2本の三項は「片方は必ず有効」を前提にしてよい。
  const [subMode, setSubMode] = useState<EditSubMode>(() => {
    if (initialIntent?.intent === "outpaint") {
      return subTabsDisabled.outpainting ? fallbackSubMode(subTabsDisabled, "outpainting") : "outpainting";
    }
    // 台帳 §3-55: Inpainting は右クリック**2 種**（部分フィルタ／対象動画）が
    // 同じサブタブへ来る。どちらの intent も行き先は同じなので、1 本の条件で
    // まとめてある。
    if (initialIntent?.intent === "inpaint-mask" || initialIntent?.intent === "inpaint-target") {
      return subTabsDisabled.inpainting ? fallbackSubMode(subTabsDisabled, "inpainting") : "inpainting";
    }
    return subTabsDisabled.retake ? fallbackSubMode(subTabsDisabled, "retake") : "retake";
  });

  // One client for everything the Outpainting flow does (`GET /loras` inside
  // the form hook, `POST /generate` here), so a single injected bridge drives
  // the whole screen in a test.
  const client = useMemo<ApiClient>(
    () => (nativeBridge ? createApiClient(nativeBridge) : defaultApiClient),
    [nativeBridge],
  );

  // `GET /config`。Retake の窓長の上下限（`limits.retake_window_*`）と、
  // Outpainting の快適予算（`limits.comfort_budgets[系統].outpaint_budget`、
  // §3-135）をここから取る。読めない間・失敗時は組み込みの既定へ落ちるので、
  // どちらのパネルも常に描ける。
  //
  // 両フックより前に呼ぶ（§3-135）: フックの呼び出し順はレンダ間で安定である
  // 必要があり、ここから下の並びには条件分岐も早期 return も無い。
  const configState = useConfig();
  const config = configState.status === "ready" ? configState.config : FALLBACK_APP_CONFIG;

  const outpaintForm = useOutpaintForm({
    prompt,
    apiClient: client,
    nativeBridge,
    initialIntent,
    config,
    engineFamily,
    acceleration,
  });
  // Wrapped rather than passed straight through: `onSubmitted` is an OPTIONAL
  // property and `exactOptionalPropertyTypes` forbids handing it an explicit
  // `undefined`. The hook keeps the latest callback in a ref, so a fresh
  // closure each render costs nothing.
  const { submitState, submit } = useGenerationSubmit({
    apiClient: client,
    onSubmitted: (jobId: string) => onJobSubmitted?.(jobId),
  });
  const submitting = submitState.phase === "submitting";

  const outpaintReasonMessages = useMemo(
    () => buildOutpaintReasonMessages(strings, { loraName: OUTPAINT_LORA_NAME }),
    [strings],
  );
  const t = strings.edit.outpainting;

  // ── §1-17 Retake ────────────────────────────────────────────────────────
  // 窓長の上下限（`limits.retake_window_*`）は上で読んだ `config` から取る。
  const retakeForm = useRetakeForm({
    prompt,
    config,
    nativeBridge,
    initialIntent,
    acceleration,
    engineFamily,
    sageAvailable,
    engineLabel,
  });
  const retakeReasonMessages = useMemo(() => buildRetakeReasonMessages(strings), [strings]);
  const tr = strings.edit.retake;

  // `bindToJob` に渡す確定値は、コールバックが作られたあとに変わりうる（窓を
  // 動かす・fps を変える）ので、毎描画更新する ref 越しに読む。ChainedScreen /
  // SingleScreen と同じ作法。
  const retakeBindRef = useRef({ numFrames: 0, genFps: 0, displayText: "", textPrefix: "" });
  retakeBindRef.current = {
    numFrames: retakeForm.placement?.numFrames ?? 0,
    genFps: retakeForm.placement?.genFps ?? 0,
    displayText: prompt ?? "",
    textPrefix: strings.provisional.generatingPrefix,
  };

  // 必須その1: 生成が受理されたら、右クリックで置いた ⏳生成予約 をこのジョブへ
  // 渡す（§5-3 用途a）。予約席が空いていれば `bindToJob` は何もしないので、
  // 予約を伴わない経路でも安全。バインドの失敗で送信を失敗させない。
  const onRetakeSubmitted = useCallback(
    async (jobId: string) => {
      onJobSubmitted?.(jobId);
      try {
        await bindToJob(nativeBridge ?? defaultBridge, { jobId, ...retakeBindRef.current });
      } catch {
        // 席はそのまま残る。ユーザーはタイムライン側からやり直せる。
      }
    },
    [onJobSubmitted, nativeBridge],
  );

  // 必須その2: 同期的な送信失敗（422 / busy）は `onSubmitted` に届かないので、
  // 未バインドの席と仮オブジェクトを idle へ戻す。放置すると以後の生成が
  // 永久にブロックされる（X2(a) と同じ穴）。
  const onRetakeFailed = useCallback(async () => {
    await rollbackReservedPlacement(nativeBridge ?? defaultBridge);
  }, [nativeBridge]);

  // Outpainting とは**別の**送信インスタンス。Outpainting のルートは
  // `placement: null`（席を取らない）ので、bind/rollback を共有インスタンスへ
  // 足すと、席を取っていない流れにまで予約の後始末が走ることになる。
  const { submitState: retakeSubmitState, submitChain } = useGenerationSubmit({
    apiClient: client,
    onSubmitted: onRetakeSubmitted,
    onFailed: onRetakeFailed,
  });
  const retakeSubmitting = retakeSubmitState.phase === "submitting";
  // サーバが塞がっている（queued/running のジョブがある、または自分が出した
  // モデル読み込みが飛行中）間は Retake と Outpainting の Generate を押させない
  // （オーナー目視 2026-08-10 ①）。Create/Chain と同じ作法で、**ボタンだけ**を
  // 止め（フォームは触れるまま）、ラベルは既存の `single.busyButton` を流用する
  // （同じ状態に 2 つ目の文言を作らない）。409 はサーバ側の最終防衛として従来
  // どおり残る。
  const { serverBusy } = useJobsContext();

  /**
   * Generate 押下時の順序（実装計画 §1 の「可動窓と配置の整合」）:
   *  1. `checkStale()` —— タイムラインが動いていたら注意文を出す（**止めない**）
   *  2. 確定した窓の開始フレームと長さで**予約を打ち直す** —— 右クリック時の
   *     予約は「選択範囲の長さ」で置いてあるが、実際に生成されるのは 8n+1 に
   *     丸めてクランプした窓なので、ここで実物へ揃える。`reservePlacement` は
   *     席が予約済みなら移動（§5-6 用途b）になる
   *  3. 送信
   * 予約の打ち直しに失敗しても送信は続ける —— リボンの長さが少しずれるだけで、
   * 生成そのものは正しいため。
   */
  // ── 台帳 §3-55 Inpainting ────────────────────────────────────────────────
  // 右クリック由来の値は `timeline/inpaintSlots.ts` の保管庫にあるので、この
  // フックは `initialIntent` を受け取らない（2 回の右クリックを 1 回きりの
  // ペイロードでは運べない）。
  //
  // 送信インスタンスを**フックより先に**作るのは、`submitting` をフックへ
  // 渡すため（敵対的レビュー m1）—— 保管庫の `busy` を書くのは
  // `useInpaintForm` の 1 本だけにしたいので、マスクの進み具合と送信の
  // 進み具合が合流する場所をそこに揃える。下の 3 つは `inpaintForm` を
  // 参照しない（`inpaintBindRef` は ref 越しに読む）ので、この順序で作れる。
  const ti = strings.edit.inpainting;
  /**
   * マスクの失敗はトーストで出す（F3・実機ゲート G9）。
   *
   * パネルの中の 1 行ではなく**アプリ共通のトースト**にしたのは、Generate を
   * 押した人の目がそのとき生成列（右）にあるため —— 左のパネルの下の方に出た
   * 注意文は、実機で素通りされた。文言は既存のまま動かさない。
   *
   * `MASK_SEED_INVALID`（部分フィルタが消えていた）と `MASK_BUSY`（追尾と
   * 同じスロットが埋まっている）は、やり直せば済む話なので `warning`。
   * 残り（`MASK_UPLOAD_FAILED`／`MASK_FAILED`、および知らないコード）は
   * `error` で、文言は総称の 1 行へ落ちる（黙って消さない）。
   */
  const toasts = useToasts();
  const onInpaintMaskError = useCallback(
    (code: string) => {
      if (code === "MASK_SEED_INVALID") {
        toasts.push({ kind: "warning", message: ti.seedGone });
      } else if (code === "MASK_BUSY") {
        toasts.push({ kind: "warning", message: ti.maskBusy });
      } else {
        toasts.push({ kind: "error", message: ti.maskFailed });
      }
    },
    [toasts, ti],
  );

  const inpaintBindRef = useRef({ numFrames: 0, genFps: 0, displayText: "", textPrefix: "" });
  const onInpaintSubmitted = useCallback(
    async (jobId: string) => {
      onJobSubmitted?.(jobId);
      try {
        await bindToJob(nativeBridge ?? defaultBridge, { jobId, ...inpaintBindRef.current });
      } catch {
        // 席はそのまま残る。ユーザーはタイムライン側からやり直せる。
      }
    },
    [onJobSubmitted, nativeBridge],
  );
  const onInpaintFailed = useCallback(async () => {
    await rollbackReservedPlacement(nativeBridge ?? defaultBridge);
  }, [nativeBridge]);
  // Retake と同じ 2 本立て（`bindToJob` / `rollbackReservedPlacement`）。
  // Inpainting も席を取る流れなので、席を取らない Outpainting の送信
  // インスタンスとは分ける。
  const { submitState: inpaintSubmitState, submit: submitInpaint } = useGenerationSubmit({
    apiClient: client,
    onSubmitted: onInpaintSubmitted,
    onFailed: onInpaintFailed,
  });
  const inpaintSubmitting = inpaintSubmitState.phase === "submitting";

  const inpaintForm = useInpaintForm({
    prompt,
    config,
    nativeBridge,
    engineFamily,
    acceleration,
    submitting: inpaintSubmitting,
    onMaskError: onInpaintMaskError,
  });
  inpaintBindRef.current = {
    numFrames: inpaintForm.placement?.numFrames ?? 0,
    genFps: inpaintForm.placement?.genFps ?? 0,
    displayText: prompt ?? "",
    textPrefix: strings.provisional.generatingPrefix,
  };
  const inpaintReasonMessages = useMemo(
    () =>
      buildInpaintReasonMessages(strings, {
        filterWidth: inpaintForm.projectWidth,
        filterHeight: inpaintForm.projectHeight,
        targetWidth: inpaintForm.slots.target?.item.mediaWidth ?? 0,
        targetHeight: inpaintForm.slots.target?.item.mediaHeight ?? 0,
      }),
    [
      strings,
      inpaintForm.projectWidth,
      inpaintForm.projectHeight,
      inpaintForm.slots.target?.item.mediaWidth,
      inpaintForm.slots.target?.item.mediaHeight,
    ],
  );

  /**
   * 右クリックが来たらサブタブを Inpainting へ寄せる（敵対的レビュー M2/M3）。
   *
   * **画面は作り直さない。** タブは常時マウントなので、`AppShell` は保管庫へ
   * publish して `setMode("edit")` するだけで、`remountTokens.edit` を進めない
   * ——進めるとフォームの状態（フレーム数・シード・アップロードの進み具合・
   * マスクの進捗）が右クリックのたびに消え、保管庫を置いた意味が半分無くなる。
   *
   * 代わりに保管庫の `subTabRequest` が増えたことを見る。**増分だけ**に反応する
   * ので、他の変更（アップロードの完了など）ではタブは動かない。灰色のときは
   * 何もしない: 押せないタブへ寄せると、押し戻せないタブの下に生成群が出る。
   */
  const inpaintSubTabRequest = useInpaintSlots().subTabRequest;
  const seenSubTabRequestRef = useRef(inpaintSubTabRequest);
  useEffect(() => {
    if (inpaintSubTabRequest === seenSubTabRequestRef.current) return;
    seenSubTabRequestRef.current = inpaintSubTabRequest;
    if (subTabsDisabled.inpainting) return;
    setSubMode("inpainting");
  }, [inpaintSubTabRequest, subTabsDisabled.inpainting]);

  /**
   * Generate 押下時の順序（実装計画 §7.5-12）:
   *  1. `renderAndUploadMask()` —— マスクを描いて上げる。**失敗したらここで
   *     止める**（マスクが無ければ描き替える場所が決まらない）。理由は
   *     パネルの注意文が言う
   *  2. 予約を打つ（系統 D・対象レイヤー × 窓）。**失敗しても続行** ——
   *     リボンが出ないだけで、生成そのものは正しい
   *  3. 送信
   *
   * 順序が load-bearing: マスクを先に作るので、`MASK_SEED_INVALID`（部分
   * フィルタが消えていた）のときに席も仮オブジェクトも残らない。
   */
  const handleInpaintGenerate = useCallback(async () => {
    const maskVideoId = await inpaintForm.renderAndUploadMask();
    if (!maskVideoId) return;
    const placement = inpaintForm.placement;
    if (placement) {
      try {
        await reservePlacement(nativeBridge ?? defaultBridge, {
          placement: "D",
          material: {
            layer: placement.layer,
            frameStart: placement.frameStart,
            frameEnd: placement.frameEnd,
          },
          numFrames: placement.numFrames,
          genFps: placement.genFps,
          textPrefix: strings.provisional.reservedPrefix,
          displayText: strings.provisional.reservedBody,
        });
      } catch {
        // 打てなくても送信は続ける（上の doc 参照）。
      }
    }
    submitInpaint(inpaintForm.buildRequest(maskVideoId));
  }, [inpaintForm, nativeBridge, strings, submitInpaint]);

  const handleRetakeGenerate = useCallback(async () => {
    await retakeForm.checkStale();
    const placement = retakeForm.placement;
    if (placement) {
      try {
        await reservePlacement(nativeBridge ?? defaultBridge, {
          placement: "D",
          material: {
            layer: placement.layer,
            frameStart: placement.frameStart,
            frameEnd: placement.frameEnd,
          },
          numFrames: placement.numFrames,
          genFps: placement.genFps,
          textPrefix: strings.provisional.reservedPrefix,
          displayText: strings.provisional.reservedBody,
        });
      } catch {
        // 打ち直せなくても送信は続ける（上の doc 参照）。
      }
    }
    submitChain(retakeForm.buildRequest());
  }, [retakeForm, nativeBridge, strings, submitChain]);

  return (
    <div className="edit-screen">
      <EditSubTabs mode={subMode} onChange={setSubMode} disabled={subTabsDisabled} />
      <div className="single-layout">
        <div className="edit-form-column">
          <div className="edit-subpanel" hidden={subMode !== "retake"}>
            <RetakePanel form={retakeForm} disabled={retakeSubmitting} />
          </div>
          <div className="edit-subpanel" hidden={subMode !== "outpainting"}>
            <OutpaintingPanel form={outpaintForm} disabled={submitting} nativeBridge={nativeBridge} />
          </div>
          <div className="edit-subpanel" hidden={subMode !== "inpainting"}>
            <InpaintingPanel form={inpaintForm} disabled={inpaintSubmitting} />
          </div>
        </div>
        <div className="generation-column">
          {/* Retake は右クリック由来のスナップショットがあるときだけ生成群を出す
              —— 手動でタブを開いただけの案内状態には、押せるものが何も無い。 */}
          {subMode === "retake" && retakeForm.snapshot !== null && (
            <>
              <GenerateButtonBar
                label={
                  retakeSubmitting
                    ? tr.generatingButton
                    : serverBusy
                      ? strings.single.busyButton
                      : tr.generateButton
                }
                disabled={retakeSubmitting || serverBusy || !retakeForm.isValid}
                onGenerate={() => void handleRetakeGenerate()}
              />
              {retakeSubmitState.phase === "error" && (
                <div className="card card-error">
                  <p className="error-code">{retakeSubmitState.code}</p>
                  <p>{retakeSubmitState.message}</p>
                </div>
              )}
              <GenerateReasonsNote reasons={retakeForm.validityReasons} messages={retakeReasonMessages} />
            </>
          )}
          {subMode === "outpainting" && (
            <>
              <GenerateButtonBar
                label={
                  submitting ? t.generatingButton : serverBusy ? strings.single.busyButton : t.generateButton
                }
                disabled={submitting || serverBusy || !outpaintForm.isValid}
                onGenerate={() => submit(outpaintForm.buildRequest())}
              />
              {submitState.phase === "error" && (
                <div className="card card-error">
                  <p className="error-code">{submitState.code}</p>
                  <p>{submitState.message}</p>
                </div>
              )}
              <GenerateReasonsNote reasons={outpaintForm.validityReasons} messages={outpaintReasonMessages} />
            </>
          )}
          {/* Inpainting は Retake と違って**常に**生成群を出す —— 右クリックが
              まだ 1 つも来ていない状態そのものが理由コード
              （`partialFilterMissing`/`targetMissing`）で説明されるので、
              「何をすれば押せるのか」がボタンの真下に出る。 */}
          {subMode === "inpainting" && (
            <>
              <GenerateButtonBar
                label={
                  inpaintSubmitting
                    ? ti.generatingButton
                    : serverBusy
                      ? strings.single.busyButton
                      : ti.generateButton
                }
                disabled={inpaintSubmitting || serverBusy || !inpaintForm.isValid}
                onGenerate={() => void handleInpaintGenerate()}
              />
              {inpaintSubmitState.phase === "error" && (
                <div className="card card-error">
                  <p className="error-code">{inpaintSubmitState.code}</p>
                  <p>{inpaintSubmitState.message}</p>
                </div>
              )}
              <GenerateReasonsNote reasons={inpaintForm.validityReasons} messages={inpaintReasonMessages} />
            </>
          )}
          <JobLedger baseUrl={baseUrl} highlightedJobId={highlightedJobId} nativeBridge={nativeBridge} />
        </div>
      </div>
    </div>
  );
}
