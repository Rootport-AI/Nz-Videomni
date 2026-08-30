import { useCallback, useMemo, useRef, useState } from "react";
import { apiClient as defaultApiClient, createApiClient } from "../../api/client";
import type { ApiClient } from "../../api/client";
import { EditSubTabs } from "./EditSubTabs";
import type { EditSubMode } from "./EditSubTabs";
import type { EditSubTabsDisabled } from "../../shell/useBaseModels";
import { OutpaintingPanel } from "./OutpaintingPanel";
import { RetakePanel } from "./RetakePanel";
import { bridge as defaultBridge } from "../../bridge";
import type { NativeBridge } from "../../bridge";
import { useStrings } from "../../i18n/LanguageContext";
import { JobLedger } from "../../jobs/JobLedger";
import { useJobsContext } from "../../jobs/JobsContext";
import type { GenerationPrefill } from "../../timeline/generationPrefill";
import { bindToJob, reservePlacement, rollbackReservedPlacement } from "../../timeline/provisionalReservation";
import { GenerateButtonBar } from "../single/GenerateButtonBar";
import { GenerateReasonsNote } from "../single/GenerateReasonsNote";
import { useConfig } from "../single/useConfig";
import { FALLBACK_APP_CONFIG } from "../single/defaultConfig";
import { useGenerationSubmit } from "../single/useGeneration";
import { buildOutpaintReasonMessages, buildRetakeReasonMessages } from "./editReasonMessages";
import { OUTPAINT_LORA_NAME } from "../../lora/controlLoras";
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
   * エンジンが実行できない**もの（`shell/useBaseModels.ts` の
   * `editSubTabsDisabledFor`）。`AppShell` が確定済みの真偽値として渡すので、
   * この画面もパネルもフィーチャ名を一切知らない（`ChainedScreen` が
   * `chainPanels` を受け取るのと同じ作法）。省略時はどちらも有効。 */
  subTabsDisabled?: EditSubTabsDisabled | undefined;
}

/** The Edit mode screen (2026-08-09). Until this day the Edit tab was a
 * disabled mock in `shell/ModeTabs.tsx`; it is now a real `AppMode` with panels
 * of its own, so the Outpainting work (`Docs/PENDING_TASKS.md` §1-13, design
 * notes `Docs/OUTPAINTING_DESIGN_NOTES.md` §5-8) has somewhere to land.
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
export function EditScreen({
  initialIntent,
  prompt,
  baseUrl = null,
  highlightedJobId = null,
  nativeBridge,
  onJobSubmitted,
  subTabsDisabled = { retake: false, outpainting: false },
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
  // 両方が無効な場合はここへ到達しない: `disabledModesFor` の
  // `needsAnyOf: ["retake", "outpaint"]` が Edit タブごと落とすので、この画面は
  // そもそもマウントされていない（`useBaseModels.ts` の `MODE_REQUIREMENTS`）。
  // なので下の2本の三項は「片方は必ず有効」を前提にしてよい。
  const [subMode, setSubMode] = useState<EditSubMode>(() => {
    if (initialIntent?.intent === "outpaint") {
      return subTabsDisabled.outpainting ? "retake" : "outpainting";
    }
    return subTabsDisabled.retake ? "outpainting" : "retake";
  });

  // One client for everything the Outpainting flow does (`GET /loras` inside
  // the form hook, `POST /generate` here), so a single injected bridge drives
  // the whole screen in a test.
  const client = useMemo<ApiClient>(
    () => (nativeBridge ? createApiClient(nativeBridge) : defaultApiClient),
    [nativeBridge],
  );

  const outpaintForm = useOutpaintForm({
    prompt,
    apiClient: client,
    nativeBridge,
    initialIntent,
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
  // 窓長の上下限（`limits.retake_window_*`）は `GET /config` から取る。読めない
  // 間・失敗時は組み込みの既定へ落ちるので、パネルは常に描ける。
  const configState = useConfig();
  const config = configState.status === "ready" ? configState.config : FALLBACK_APP_CONFIG;
  const retakeForm = useRetakeForm({ prompt, config, nativeBridge, initialIntent });
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
          <JobLedger baseUrl={baseUrl} highlightedJobId={highlightedJobId} nativeBridge={nativeBridge} />
        </div>
      </div>
    </div>
  );
}
