import { useCallback, useEffect, useLayoutEffect, useMemo, useRef } from "react";
import type { AppConfig } from "../../api/types";
import { bridge as defaultBridge } from "../../bridge";
import type { NativeBridge } from "../../bridge";
import { useStrings } from "../../i18n/LanguageContext";
import { useJobsContext } from "../../jobs/JobsContext";
import { JobLedger } from "../../jobs/JobLedger";
import { useShowNote } from "../../shell/NoteArea";
import { useToasts } from "../../shell/ToastContext";
import type { NagSettings } from "../../shell/nagSettings";
import type { AccelerationSettings } from "../../shell/accelerationSettings";
import { usePrefillPolicy } from "../../shell/PrefillPolicyContext";
import { isChainStage1OverBudget } from "../../shell/tokenBudget";
import type { Stage2Window } from "../../shell/tokenBudget";
import { computeTargetNumFrames } from "../../timeline/deriveDuration";
import { resolvePrefillSeed } from "../../timeline/prefillSeed";
import type { GenerationPrefill } from "../../timeline/generationPrefill";
import {
  bindToJob,
  getReservationState,
  publishChainDirty,
  reservePlacement,
  rollbackReservedPlacement,
} from "../../timeline/provisionalReservation";
import { rekeySourceLocation } from "../../timeline/sourceLocationMap";
import { decideSourceTrim } from "../../timeline/sourceTrim";
import type { SourceTrimDecision } from "../../timeline/sourceTrim";
import { tailAlignedStartFrame } from "../../timeline/tailAlign";
import { fileNameFromPath } from "../single/keyframeUtils";
import { CropOutputField, FrameRateSeedFields, ReservedFields, SizeFields } from "../single/CommonGenerationFields";
import { MIN_NUM_FRAMES } from "../single/defaultConfig";
import { GenerateButtonBar } from "../single/GenerateButtonBar";
import { GenerateReasonsNote } from "../single/GenerateReasonsNote";
import { PresetDropdown } from "../single/PresetDropdown";
import { useGenerationSubmit } from "../single/useGeneration";
import { useConfig } from "../single/useConfig";
import { BatchI2vLongSection } from "../batch-i2v-long/BatchI2vLongSection";
import { ChainAudioPanel } from "./ChainAudioPanel";
import { ChainEndSourcePanel } from "./ChainEndSourcePanel";
import { ChainReferencePanel } from "./ChainReferencePanel";
import { ClipCard } from "./ClipCard";
import { buildChainReasonMessages } from "./generateReasonMessages";
import {
  MAX_CHAIN_TOTAL_FRAMES,
  MAX_OVERLAP_FRAMES,
  MAX_OVERLAP_STRENGTH,
  MIN_OVERLAP_FRAMES,
  MIN_OVERLAP_STRENGTH,
} from "./chainUtils";
import { SourceInputPanel } from "./SourceInputPanel";
import { useChainForm } from "./useChainForm";
import type { UseChainFormResult } from "./useChainForm";
import "./ChainedScreen.css";

export interface ChainedScreenProps {
  /** Owned by `AppShell`'s shared `PromptBar`, above the mode tabs — same
   * convention as `SingleScreen`. */
  prompt: string;
  baseUrl: string | null;
  /** One-shot right-click prefill. Seeds the common width/height from the routed
   * selection and (I8, SPEC §3-4) auto-loads the material: #1 `extend-video` →
   * the source video slot, #6 `image-to-clip-chain` → clip 0's opening
   * keyframe. Consumed once on the remount that carries it. */
  initialIntent?: GenerationPrefill | undefined;
  /** Test/integration seam threaded from `AppShell`; production omits it. */
  nativeBridge?: NativeBridge | undefined;
  /** The currently highlighted job (last submitted / toast-clicked). */
  highlightedJobId: string | null;
  /** U-R1: report a freshly submitted `job_id` up to `AppShell`. */
  onJobSubmitted: (jobId: string) => void;
  /** Control-LoRA names (`lora/controlLoras.ts`'s `resolveControlLoraNames`,
   * computed by `AppShell`) — §1-15: Chain now HAS a control-LoRA selection of
   * its own (`useChainForm.controlLora`), so this feeds both that dropdown and
   * the `controlLoraNeedsReference` gate. */
  controlLoraNames: ReadonlySet<string>;
  /** §1-15: the depth-preprocess control adapters (`resolveDepthLoraNames`,
   * computed by `AppShell`), for the `depthChainUnsupported` gate. Optional so
   * direct-render tests that predate it keep compiling — omitted means the gate
   * never fires, which is also what an older server (no `preprocess` field on
   * `GET /loras`) produces. */
  depthLoraNames?: ReadonlySet<string> | undefined;
  /** §1-15, plan F5/F6: control-LoRA name -> `reference_downscale_factor`
   * (`resolveReferenceDownscaleFactors`, computed by `AppShell`) — feeds the
   * stage-1 comfort-budget warning banner's `refScale`. Optional for the same
   * reason as `depthLoraNames`: a direct-render test that omits it, or a
   * server whose `GET /loras` doesn't publish the factor, simply falls back
   * to `2` (the union-control adapters' own factor) for every name. */
  referenceDownscaleFactors?: ReadonlyMap<string, number> | undefined;
  /** NAG (2026-07-28): the shared Negative Prompt accordion's settings,
   * owned by `AppShell` (single `NagSettings` object, D1). Optional so
   * pre-existing direct-render tests that omit it keep compiling —
   * `useChainForm` defaults an omitted `nag` to the frozen `NAG_OFF`
   * sentinel. */
  nag?: NagSettings | undefined;
  /** Acceleration (2026-07-31): the Settings panel's shared
   * attention-backend choice, owned by `AppShell` (single
   * `AccelerationSettings` object, D1). Optional for the same reason as
   * `nag` above — `useChainForm` defaults it to the frozen
   * `ACCELERATION_DEFAULTS` sentinel, which sends nothing. */
  acceleration?: AccelerationSettings | undefined;
}

/** The Chain screen: `POST /generate/chain` with a single, auto-detected mode
 * (attach a source video → V2V continuation, otherwise a from-scratch clip
 * concatenation — no user-selectable submode). U-R1: submission goes through
 * the slim `useGenerationSubmit` (`submitChain`) Create also uses, and the
 * right column stacks the Generate button, a submit-error row, and the shared
 * `JobLedger`; `SizeFields`/`FrameRateSeedFields` still supply the common
 * width/height/frame_rate/seed section. */
export function ChainedScreen({
  prompt,
  baseUrl,
  initialIntent,
  nativeBridge,
  highlightedJobId,
  onJobSubmitted,
  controlLoraNames,
  depthLoraNames,
  referenceDownscaleFactors,
  nag,
  acceleration,
}: ChainedScreenProps) {
  const strings = useStrings();
  const configState = useConfig();

  if (configState.status === "loading") {
    return <p className="hint">{strings.single.loadingConfig}</p>;
  }

  return (
    <ChainedScreenBody
      config={configState.config}
      usingFallback={configState.usingFallback}
      prompt={prompt}
      baseUrl={baseUrl}
      initialIntent={initialIntent}
      nativeBridge={nativeBridge}
      highlightedJobId={highlightedJobId}
      onJobSubmitted={onJobSubmitted}
      controlLoraNames={controlLoraNames}
      depthLoraNames={depthLoraNames}
      referenceDownscaleFactors={referenceDownscaleFactors}
      nag={nag}
      acceleration={acceleration}
    />
  );
}

interface ChainedScreenBodyProps {
  config: AppConfig;
  usingFallback: boolean;
  prompt: string;
  baseUrl: string | null;
  initialIntent?: GenerationPrefill | undefined;
  nativeBridge?: NativeBridge | undefined;
  highlightedJobId: string | null;
  onJobSubmitted: (jobId: string) => void;
  controlLoraNames: ReadonlySet<string>;
  depthLoraNames?: ReadonlySet<string> | undefined;
  referenceDownscaleFactors?: ReadonlyMap<string, number> | undefined;
  nag?: NagSettings | undefined;
  acceleration?: AccelerationSettings | undefined;
}

function ChainedScreenBody({
  config,
  usingFallback,
  prompt,
  baseUrl,
  initialIntent,
  nativeBridge,
  highlightedJobId,
  onJobSubmitted,
  controlLoraNames,
  depthLoraNames,
  referenceDownscaleFactors,
  nag,
  acceleration,
}: ChainedScreenBodyProps) {
  const strings = useStrings();
  const jobsCtx = useJobsContext();

  // I12 (§5-3 用途a): `bindToJob` needs the confirmed length/fps + display text,
  // but `form`/`prompt` are read below this callback, so `onSubmitted` reads the
  // latest values through a ref refreshed on every render (mirrors SingleScreen).
  const bindInfoRef = useRef({ numFrames: 0, genFps: 0, displayText: "", textPrefix: "" });

  // U-R1: refresh the ledger + highlight the new job on a successful submit;
  // the submit hook holds `submitting` until this resolves (MN-6).
  const onSubmitted = useCallback(
    async (jobId: string) => {
      await jobsCtx.refresh();
      onJobSubmitted(jobId);
      // I12 §5-3 用途a: hand a waiting #1/#6 reservation off to this real job.
      // `bindToJob` is a no-op unless a reservation is actually waiting (phase
      // "reserved"), so a plain Chain Generate (not via #1/#6) never touches the
      // seat. Runs only after the backend accepts the job, so a busy/422 sync
      // rejection leaves the reservation at stage 1 (spec design intent) — same
      // as SingleScreen.onSubmitted.
      const bridge = nativeBridge ?? defaultBridge;
      // I6 Join: re-key the v2v source location (recorded under the reservation's
      // pending id at right-click) to this real job id, at the same hand-off
      // moment `bindToJob` re-stamps the placeholder. Read the pending id BEFORE
      // the bind (which keeps it in state but sets the jobId); `rekeySourceLocation`
      // is a no-op when nothing was recorded (a plain manual v2v / #6 chain), so
      // this is safe for every Chain Generate. (JOIN_FEATURE_RESEARCH.md §4.5.)
      const reservedPendingId = getReservationState().pendingId;
      try {
        await bindToJob(bridge, { jobId, ...bindInfoRef.current });
      } catch {
        // A bind failure must never fail the submit; the reservation just stays
        // where it is and the user can retry from the timeline.
      }
      if (reservedPendingId !== null) rekeySourceLocation(reservedPendingId, jobId);
    },
    [jobsCtx, onJobSubmitted, nativeBridge],
  );
  // X2(a): a synchronous submit failure (422/busy) never reaches `onSubmitted`,
  // so a #1/#6/#12 right-click reservation would stay `reserved` forever and
  // block every later generation. Roll the still-unbound seat + its placeholder
  // back to idle. `rollbackReservedPlacement` is a no-op unless a reservation is
  // actually waiting; it never touches a bound (`generating`) real job. The
  // `sourceLocationMap` pending-key residue is harmless (never re-keyed to a
  // real job, ignored by everything) so it is deliberately left uncleaned.
  const onFailed = useCallback(async () => {
    const bridge = nativeBridge ?? defaultBridge;
    await rollbackReservedPlacement(bridge);
  }, [nativeBridge]);
  const { submitState, submitChain } = useGenerationSubmit({ onSubmitted, onFailed });
  // Derive the initial common resolution from the routed selection. Computed
  // once — the body remounts via `key` on a new intent — and handed to the
  // form's lazy init. (`resolvePrefillSeed` decides the grid itself: §1-15 W4's
  // `reference-video-chain` seeds on 128, every other Chain intent on 64.)
  // W7: the right-click prefill resolution/fps policy (Settings). Only affects
  // the initial-value decision below — ordinary edits/presets are unchanged.
  const { sizePolicy, fpsPolicy } = usePrefillPolicy();
  const initialCommon = useMemo(() => {
    if (!initialIntent) return {};
    // W8: the shared純関数 `resolvePrefillSeed` decides the common width/height +
    // fps + clip-0 DURATION from the SAME right-click inputs `AppShell`'s
    // reservation length uses, so a placed provisional's ribbon matches the
    // remounted Chain form (owner requirement 2026-07-21 #1). Every Chain prefill
    // (#1 extend-video / #6 image-to-clip-chain / #12 current-frame-to-clip-chain
    // / 台帳§1-16's two long-a2v items) is the `comfortCeiling` DURATION policy —
    // no Chain intent is materialClampedToCeiling, including the long-a2v pair,
    // whose track length re-lays the WHOLE clip list through the audio auto-fit
    // instead of clamping clip 0 — so the seed's `numFrames` is that
    // resolution's comfort ceiling. `defaults` ignores the material's real
    // size/rate; `material`/`project` seed from it (`project` is overwritten
    // below off getEditInfo). A `null`/undefined seed value leaves the common
    // field at the config default.
    const seed = resolvePrefillSeed({
      intent: initialIntent.intent,
      selection: initialIntent.selection,
      config,
      sizePolicy,
      fpsPolicy,
    });
    return {
      width: seed.derived.width,
      height: seed.derived.height,
      ...(seed.frameRate != null ? { frameRate: seed.frameRate } : {}),
      ...(seed.numFrames != null ? { numFrames: seed.numFrames } : {}),
      // X4: #1 extend-video attaches a source video (V2V, 1-clip floor), but the
      // source is still idle at mount, so open the clip list at a single clip
      // instead of the scratch 2-clip minimum — otherwise a stray 2nd clip card
      // lingers even after the source turns ready. #6/#12 scratch chains keep the
      // 2-clip default.
      //
      // 素材（末尾）: `end-with-this` joins it for the same structural reason —
      // the end slot ALSO relaxes the floor to 1, and is also still idle at
      // mount, so it would otherwise pad a stray 2nd card too. 逆順Chained
      // (2026-08-18, second stage) removed this route's OWN reason (窓内モード's
      // now-gone single-clip ceiling) — a right-click reservation still starts
      // at the seat's own single clip either way, and the user is free to
      // `addClip` afterward now that a multi-clip end-source chain validates.
      ...(initialIntent.intent === "extend-video" || initialIntent.intent === "end-with-this"
        ? { forceSingleClip: true }
        : {}),
    };
  }, [initialIntent, config, sizePolicy, fpsPolicy]);

  const form = useChainForm(
    config,
    prompt,
    { nativeBridge, controlLoraNames, depthLoraNames, nag, acceleration },
    initialCommon,
  );

  // §1-15, plan F5: the stage-1 comfort-budget warning banner's inputs — a
  // reference active AND the LONGEST clip (the one whose stage-1 segment has
  // the most video-latent frames) would exceed `isChainStage1OverBudget` at
  // the current resolution. `referenceScale` looks up the SELECTED control
  // LoRA's `reference_downscale_factor` off the map `AppShell` resolved from
  // `GET /loras`; falls back to `2` (the union-control adapters' own factor)
  // whenever nothing is selected yet or the map has no entry for it — never
  // blocks the banner on missing data, just estimates conservatively.
  const maxClipFrames = form.clips.reduce((max, clip) => Math.max(max, clip.numFrames), 0);
  const referenceScale = form.controlLora ? (referenceDownscaleFactors?.get(form.controlLora.name) ?? 2) : 2;
  const stage1ReferenceOverBudget =
    form.isReferenceActive &&
    isChainStage1OverBudget(form.common.width, form.common.height, maxClipFrames, referenceScale);

  const showNote = useShowNote();
  const toasts = useToasts();

  // §1-16 長尺A2V: turn `adjustClipsForAudio`'s published outcome into a toast.
  // `useChainForm` never touches the toast system itself (it stays usable from a
  // plain `renderHook`) — it just bumps `audioFitEvent.id` once per run, and this
  // effect reports it. Keyed on `id` through a ref so StrictMode's double-invoked
  // effect shows the message once, exactly as `SingleScreen` does for Create's own
  // audio auto-adjust. Every outcome is reported, including the two that changed
  // nothing: an ↔️ press that silently did nothing is indistinguishable from a
  // dead button.
  const lastToastedAudioFitId = useRef<number | null>(null);
  useEffect(() => {
    const event = form.audioFitEvent;
    if (!event || lastToastedAudioFitId.current === event.id) return;
    lastToastedAudioFitId.current = event.id;
    const fit = strings.chained.sourceAudio.fitToast;
    switch (event.outcome) {
      case "adjusted":
      // The clip list WAS re-laid out; it just could not reach the end of the
      // track within 24 clips. The leftover tail is the panel's own note
      // (`atMaxClipsNote`), so the toast stays the plain success line.
      case "cappedAtMaxClips":
        toasts.push({ kind: "success", message: fit.adjusted(event.durationSec) });
        break;
      case "alreadyFits":
        toasts.push({ kind: "success", message: fit.alreadyFits });
        break;
      case "cannotFit":
        toasts.push({ kind: "warning", message: fit.cannotFit });
        break;
      case "noFlexibleClips":
        toasts.push({ kind: "warning", message: fit.noFlexibleClips });
        break;
    }
  }, [form.audioFitEvent, strings, toasts]);

  // W7 `project` policy: read the AviUtl2 project's own resolution/fps via
  // `getEditInfo` ONCE on this prefilling remount and overwrite the common
  // width/height/frameRate. A consumed ref limits it to a single run; a
  // getEditInfo failure leaves the material-derived seed in place (③ fallback).
  // Only runs for an actual prefill under the `project` policy.
  const projectOverwriteConsumedRef = useRef(false);
  useEffect(() => {
    if (projectOverwriteConsumedRef.current) return;
    // W1: independent axes — the overwrite runs whenever EITHER policy is
    // `project`, and each axis is applied only when its own policy is `project`.
    if (!initialIntent || (sizePolicy !== "project" && fpsPolicy !== "project")) return;
    projectOverwriteConsumedRef.current = true;
    void (async () => {
      const bridge = nativeBridge ?? defaultBridge;
      try {
        const editInfo = await bridge.request("getEditInfo", {});
        if (sizePolicy === "project") {
          form.setWidth(editInfo.width);
          form.setHeight(editInfo.height);
        }
        if (fpsPolicy === "project" && editInfo.scale > 0 && editInfo.rate > 0) {
          form.setFrameRate(editInfo.rate / editInfo.scale);
        }
        // W1/W8 (② project policy): every Chain intent is `comfortCeiling`, whose
        // DURATION depends on width/height ONLY (fps never enters the ceiling
        // calc), so clip 0's DURATION is recomputed only when the SIZE axis is
        // `project` — an fps-only project overwrite leaves the width/height seed
        // (and therefore the ceiling) untouched. Leaves the seed in place when
        // the ceiling can't be resolved (null).
        if (sizePolicy === "project") {
          const target = computeTargetNumFrames({
            policy: "comfortCeiling",
            width: editInfo.width,
            height: editInfo.height,
            spillFreeFrames: config.limits.spill_free_frames,
            minNumFrames: MIN_NUM_FRAMES,
            maxNumFrames: config.limits.max_num_frames,
          });
          const clip0 = form.clips[0];
          if (target !== null && clip0) form.setClipNumFrames(clip0.id, target);
        }
      } catch {
        // Fall back to the material-derived seed already in the form (③ behavior).
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps -- one-shot mount consume
  }, []);

  // I12: keep the bind-time snapshot current for `onSubmitted` above. The chain's
  // output length + common fps size the placeholder; the prompt head is the
  // stage-2 display text, prefixed with "Generating: " (spec §5-5 段階2 — passed
  // explicitly so native's default ⏳生成中 prefix is never used).
  bindInfoRef.current = {
    numFrames: form.outputFrames,
    genFps: form.common.frameRate,
    displayText: prompt.trim().slice(0, 20) || "(gen)",
    textPrefix: strings.provisional.generatingPrefix,
  };

  // I8 §3-4 (※): publish the live dirty state so `AppShell.handleRoute` can
  // decide whether a #1/#6 remount needs the "破棄して読み込みますか？" confirm.
  // Chain is always mounted (all tabs at once), so this keeps the shared flag
  // current no matter which tab is visible.
  useEffect(() => {
    publishChainDirty(form.isDirty);
  }, [form.isDirty]);

  // I8 §3-4 / §6: auto-load the routed material ONCE on this remount. #1
  // `extend-video` loads the selected video into the source_video slot (V2V
  // continuation); #6 `image-to-clip-chain` sets the selected image as clip 0's
  // opening keyframe; §1-15 W4 `reference-video-chain` loads it into the
  // reference slot. A lazy useState initializer can't drive the async upload,
  // so this runs in a mount effect guarded by a consumed ref (StrictMode-safe;
  // the whole body remounts via `key` on a new intent, so a fresh ref = a fresh
  // one-shot per prefill). `autoLoadPendingRef` lets the status-watch effect
  // below report only THIS load's outcome (the §6-2 missing-material note),
  // never a later user-initiated source error.
  const autoLoadConsumedRef = useRef(false);
  const autoLoadPendingRef = useRef(false);
  /** 素材（末尾）v2 系統E: where the right-clicked END material sits on the
   * timeline, captured at auto-load time so `handleGenerate` can RE-PLACE the
   * provisional at the confirmed 末尾合わせ position (`timeline/tailAlign.ts`).
   *
   * A LOCAL ref rather than the shared `sourceLocationMap`, deliberately: the
   * re-place issues a NEW `pending-…` id (the seat's 用途b "move" path always
   * does), which orphans anything keyed by the old one — verified — and the
   * shared map is also Join's, so riding on it risks a cross-feature accident.
   * The anchor never has to survive a remount: a remount means a new right-click,
   * which brings its own anchor.
   *
   * `null` for every other route, INCLUDING a manual tab-opened Chain — which is
   * exactly what makes `handleGenerate` skip the re-place there instead of
   * inserting a placeholder nobody asked for. */
  const endSourceAnchorRef = useRef<{
    layer: number;
    frameStart: number;
    frameEnd: number;
    rate: number;
    scale: number;
  } | null>(null);
  useEffect(() => {
    if (autoLoadConsumedRef.current) return;
    const intent = initialIntent?.intent;
    if (
      intent !== "extend-video" &&
      intent !== "image-to-clip-chain" &&
      intent !== "current-frame-to-clip-chain" &&
      intent !== "video-audio-to-long-a2v" &&
      intent !== "audio-to-long-a2v" &&
      intent !== "reference-video-chain" &&
      // 素材（末尾）: 🎬🖼️「これで終わる動画を作る」. Forgetting this line is a
      // SILENT no-op (the branch below simply never runs), so it is listed here
      // together with its branch and its `autoLoadStatus` arm — the three always
      // change as a set.
      intent !== "end-with-this"
    ) {
      return;
    }
    autoLoadConsumedRef.current = true;

    // 台帳§1-16 長尺A2V: the two right-click audio items. Both cut the SELECTED
    // OBJECT'S RIBBON RANGE to a temp wav through `timeline.extractAudio` and
    // hand it to Chain's audio slot, which then auto-fits the clip list to the
    // measured length (exactly once — `useChainForm` keys that on the upload id).
    // They differ only in what is recorded:
    //  - 🎬 `video-audio-to-long-a2v` uses the MIX path, the same semantics as
    //    Create's #3 `videoAudioToVideo`: the sound as it actually plays on the
    //    timeline over that range.
    //  - 🎵 `audio-to-long-a2v` uses SOLO, keeping only the selected object's
    //    own layer. This is deliberately NOT #7's "upload the backing file
    //    whole": going through extraction is what makes a TRIMMED audio object
    //    contribute exactly the seconds its ribbon shows.
    // Kept as their own async routine (like SingleScreen's #3) because of the
    // "extracting…" note and the native-specific failure branches.
    if (intent === "video-audio-to-long-a2v" || intent === "audio-to-long-a2v") {
      const item = initialIntent?.selection?.selected[0];
      if (!item) {
        showNote("warning", strings.notes.sourceFileMissing);
        return;
      }
      const isMix = intent === "video-audio-to-long-a2v";
      void (async () => {
        // frameEnd is INCLUSIVE (native's `FindObjectByJob` walks with
        // `next = frameEnd + 1`), so the ribbon spans [frameStart, frameEnd].
        const frameCount = item.frameEnd - item.frameStart + 1;
        showNote("info", isMix ? strings.notes.extractingAudio : strings.notes.extractingObjectAudio);
        const bridge = nativeBridge ?? defaultBridge;
        try {
          const extracted = await bridge.request("timeline.extractAudio", {
            layer: item.layer,
            frameStart: item.frameStart,
            frameCount,
            ...(isMix
              ? { audioMode: "mix" as const, soloKeepLayers: [] }
              : { audioMode: "solo" as const, soloKeepLayers: [item.layer] }),
          });
          showNote("info", isMix ? strings.notes.loadedVideoAudio : strings.notes.loadedObjectAudio);
          autoLoadPendingRef.current = true;
          // `durationSec` is native's EXACT length for the wav it just wrote, so
          // it is handed over as `knownDurationSec` — tier 1 of the hook's
          // three-tier length resolution, which then skips every `fs.probe*`.
          void form.attachAudioByPath(extracted.filePath, fileNameFromPath(extracted.filePath), {
            knownDurationSec: extracted.durationSec,
          });
        } catch (err) {
          // Same three failure branches as SingleScreen's #3: native returns one
          // `EXTRACT_FAILED` code for everything and only the message tells the
          // silent range / render-timeout / setup cases apart
          // (`native/src/bridge.cpp`'s `ExtractAudioWorker`).
          const message = err instanceof Error ? err.message : "";
          if (/silent range|no audio stream/i.test(message)) {
            showNote("warning", strings.notes.audioExtractSilent);
          } else if (/render failed/i.test(message)) {
            showNote("warning", strings.notes.audioExtractTimeout);
          } else {
            showNote("warning", strings.notes.audioExtractFailed);
          }
        }
      })();
      return;
    }

    // W5 #12 (current-frame-to-clip-chain): unlike #1/#6, this layer item has NO
    // selected object. It CAPTURES the current timeline frame and loads that PNG
    // as clip 0's opening keyframe (the same start-frame slot #6 uses). The
    // capture is async, so it runs as its own routine (mirroring SingleScreen's
    // videoAudioToVideo shape); the fresh remount guarantees sourceVideo is idle,
    // so the form stays in "scratch" mode and buildRequest injects startFrame.
    if (intent === "current-frame-to-clip-chain") {
      void (async () => {
        const bridge = nativeBridge ?? defaultBridge;
        try {
          const captured = await bridge.request("timeline.captureFrame", {});
          const fileName = fileNameFromPath(captured.filePath);
          // §6 receipt at capture success — the visible start-frame fill is the
          // primary feedback; this reinforces it. The status-watch below
          // downgrades to the missing-material note if the upload then fails.
          showNote("info", strings.notes.capturedFrameToKeyframe);
          autoLoadPendingRef.current = true;
          void form.startFrame.addFromPath(captured.filePath, fileName);
        } catch {
          // The captureFrame round trip itself failed — nothing to load.
          showNote("warning", strings.notes.captureFrameFailed);
        }
      })();
      return;
    }

    const selection = initialIntent?.selection;
    const item = selection?.selected[0];
    const filePath = item?.filePath ?? null;
    if (!filePath) {
      // §6-2: the path couldn't even be resolved (object without a backing file).
      showNote("warning", strings.notes.sourceFileMissing);
      return;
    }
    const fileName = fileNameFromPath(filePath);
    // §6 receipt note, shown at load START — the visible source-slot fill is the
    // primary feedback; this reinforces it.
    showNote("info", strings.notes.loadedFromRightClick(fileName));
    autoLoadPendingRef.current = true;
    if (intent === "extend-video") {
      // #1: video path → source_video (routes by extension via attachSourceByPath).
      // §1-6: this is the ONE attach route with a timeline selection to measure,
      // so it is the only one that passes an `AttachedSourceInfo` — the trim
      // decision (which adds the upload's trim query when it fires) plus the
      // probed full-file duration gate B falls back to when it does not. Every
      // other route omits it and behaves exactly as before §1-6.
      const decision: SourceTrimDecision = selection
        ? decideSourceTrim(item, selection)
        : { trim: false, reason: "noItem" };
      void form.attachSourceByPath(filePath, fileName, {
        trim: decision,
        knownDurationSec: item && item.mediaDurationSec > 0 ? item.mediaDurationSec : null,
      });
    } else if (intent === "reference-video-chain") {
      // §1-15 W4 (2026-08-11): video path → the CHAIN's reference slot, the
      // long-form twin of Create's #2. The ribbon range is decided exactly the
      // way #1 above (and Create's #2) decides it — one `decideSourceTrim` call
      // on the same (item, selection) pair — and handed to the hook, which
      // merges it with the slot's standing `max_frames` cap and clamps the
      // window before sending. No `knownDurationSec`: the reference slot has no
      // `context_frames` length gate to feed, so there is nothing to measure.
      const decision: SourceTrimDecision = selection
        ? decideSourceTrim(item, selection)
        : { trim: false, reason: "noItem" };
      void form.attachReferenceByPath(filePath, fileName, decision);
    } else if (intent === "end-with-this") {
      // 素材（末尾）: image OR video path → the END slot, which routes by
      // extension itself (`attachEndSourceByPath` -> `routeSourceByExtension`),
      // exactly like #1's start-slot twin above. Placed BEFORE the trailing
      // `else` on purpose: that arm is the image → START-frame fallback, so an
      // image dropped through this intent would otherwise land in the wrong slot.
      //
      // The ribbon range is honoured the same way #1 and `reference-video-chain`
      // honour it — one `decideSourceTrim` on the same (item, selection) pair —
      // and `knownDurationSec` is handed over as the FALLBACK the 9-frame length
      // gate reads when the server reports no measurement of its own (see
      // `useChainForm`'s `resolveEndSourceContext`). Both are meaningless for an
      // IMAGE and are simply ignored by the image branch.
      const decision: SourceTrimDecision = selection
        ? decideSourceTrim(item, selection)
        : { trim: false, reason: "noItem" };
      // v2 系統E: remember WHERE the material is, before the async attach — the
      // Generate-time re-place needs the object's own frame range and the
      // project's frame rate, and `initialIntent` is a one-shot this effect is
      // the last reader of.
      if (item && selection) {
        endSourceAnchorRef.current = {
          layer: item.layer,
          frameStart: item.frameStart,
          frameEnd: item.frameEnd,
          rate: selection.rate,
          scale: selection.scale,
        };
      }
      void form.attachEndSourceByPath(filePath, fileName, {
        trim: decision,
        knownDurationSec: item && item.mediaDurationSec > 0 ? item.mediaDurationSec : null,
      });
    } else {
      // #6: image → clip 0's start frame (useKeyframes maxItems 1). The fresh
      // remount guarantees sourceVideo is idle, so the form stays in "scratch"
      // mode and buildRequest injects startFrame onto clip 0 — startFrame is
      // scratch-only (a V2V source would supply clip 0 instead). See useChainForm.
      void form.startFrame.addFromPath(filePath, fileName);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- one-shot mount consume
  }, []);

  // §6-2: replace the receipt with the missing-material note if the auto-load
  // upload failed. Gated on `autoLoadPendingRef` so it fires only for the
  // auto-load's own terminal state, then stops watching (a later manual re-pick
  // that errors won't re-trigger it). #6 image-to-clip-chain and #12
  // current-frame-to-clip-chain both feed the start-frame slot, so its status is
  // the terminal signal for either; only #1 extend-video watches sourceVideo.
  // 台帳§1-16: the two long-a2v intents feed the AUDIO slot instead, so they
  // watch that one — the extraction itself already reported its own failures
  // above, and this covers the upload of the extracted wav failing afterwards.
  // §1-15 W4: `reference-video-chain` feeds the REFERENCE slot, so it watches
  // that one — same one-shot receipt/failure contract as every branch above.
  const autoLoadIntent = initialIntent?.intent;
  const autoLoadStatus =
    autoLoadIntent === "extend-video"
      ? form.sourceVideo.state.status
      : autoLoadIntent === "video-audio-to-long-a2v" || autoLoadIntent === "audio-to-long-a2v"
        ? form.sourceAudio.state.status
        : autoLoadIntent === "reference-video-chain"
          ? form.referenceVideo.state.status
          : // 素材（末尾）: the END slot's own unified status covers both halves
            // (image or video), so one arm serves both material kinds.
            autoLoadIntent === "end-with-this"
            ? form.endSourceStatus
            : form.startFrame.items[0]?.status;
  useEffect(() => {
    if (!autoLoadPendingRef.current) return;
    if (autoLoadStatus === "error") {
      showNote("warning", strings.notes.sourceFileMissing);
      autoLoadPendingRef.current = false;
    } else if (autoLoadStatus === "ready") {
      autoLoadPendingRef.current = false;
    }
  }, [autoLoadStatus, showNote, strings]);

  // MJ-1: the form is only frozen mid-submit; a running job blocks the
  // Generate button alone (`hasActiveJob`), leaving fields editable.
  const submitting = submitState.phase === "submitting";
  const hasActiveJob = jobsCtx.hasActiveJob;
  const disabled = submitting;

  /**
   * 素材（末尾）系統E: Generate 押下時の順序（EditScreen の Retake と同じ作法）:
   *  1. 右クリック由来のアンカーがあり、席が予約済みで、素材（末尾）が実際に
   *     使える状態なら、**確定した出力長で予約を打ち直す** —— 右クリック時の
   *     予約はシード値（快適上限を169fに丸めた長さ）で置いてあるので、ユーザーが
   *     クリップ尺をいじっていた場合にここで実物へ揃える
   *  2. 送信
   *
   * 打ち直しに失敗しても送信は続ける（リボンの位置が少しずれるだけで、生成その
   * ものは正しい）。手動でタブを開いた Generate はアンカーが無いので丸ごと
   * スキップされる —— そこで打ち直すと、誰も頼んでいない仮オブジェクトを
   * タイムラインに挿してしまう。
   *
   * `inFlightRef` は async 化に伴う連打ガード。`submitChain` 自身は同期的に
   * `submitting` を立てるが、それはこの関数の await のあとなので、その隙間に
   * 2 回目のクリックが入ると予約を二重に打ち直すことになる。
   *
   * 逆順Chained（2026-08-18・第2段階）: `clips.length` を一切見ていないのは
   * 意図どおり — `form.outputFrames`/`form.outputTailFrames` は複数クリップ
   * でも同じ意味（帯は常に最終クリップの内側）のままなので、この打ち直しは
   * クリップ数によらず成立する。
   */
  const inFlightRef = useRef(false);
  const handleGenerate = useCallback(async () => {
    if (inFlightRef.current) return;
    inFlightRef.current = true;
    try {
      const anchor = endSourceAnchorRef.current;
      const frozenTail = form.outputTailFrames;
      if (anchor && form.hasEndSource && frozenTail > 0 && getReservationState().phase === "reserved") {
        try {
          await reservePlacement(nativeBridge ?? defaultBridge, {
            placement: "E",
            material: {
              layer: anchor.layer,
              // 系統E の本体: 厳密整列（素材1フレーム目=出力の第(L−錨)フレーム、
              // 2026-08-17実測確認）。素材に重なる末尾（凍結分）に加え、因果VAEの
              // プライマ1フレーム分（素材の2〜9フレーム目が錨に重なるため）も
              // 除いた長さだけ、素材の手前へずらす。`placementParams` が
              // "E" → "B" に写像するので、native にはこのシフト済みの値が
              // `materialFrameStart` として渡る。
              frameStart: tailAlignedStartFrame({
                materialFrameStart: anchor.frameStart,
                leadPixelFrames: form.outputFrames - (frozenTail + 1),
                genFps: form.common.frameRate,
                projectRate: anchor.rate,
                projectScale: anchor.scale,
              }),
              frameEnd: anchor.frameEnd,
            },
            numFrames: form.outputFrames,
            genFps: form.common.frameRate,
            textPrefix: strings.provisional.reservedPrefix,
            displayText: strings.provisional.reservedBody,
          });
        } catch {
          // 打ち直せなくても送信は続ける（上の doc 参照）。
        }
      }
      submitChain(form.buildRequest());
    } finally {
      inFlightRef.current = false;
    }
  }, [form, submitChain, nativeBridge, strings]);

  const generateLabel = submitting
    ? strings.chained.generatingButton
    : hasActiveJob
      ? strings.chained.busyButton
      : strings.chained.generateButton;

  // W7: the imperative "why is Generate disabled?" note under the button. Chain
  // has no screen-level gates beyond the hook's, so this is just the hook's
  // `validityReasons` mapped to copy. The map itself now lives in
  // `generateReasonMessages.ts` (§1-7) so Batch i2v-long can expand the very
  // same `validityReasons` with the very same wording; the content is
  // unchanged. `submitting`/`hasActiveJob` are not reasons (the label already
  // changes).
  const generateReasonMessages = buildChainReasonMessages(strings, {
    minFramesForOverlap: form.minFramesForOverlap,
    minClips: form.minClips,
    stage2MaxContextFrames: form.stage2MaxContextFrames,
  });

  // W3 レイアウト再構成 (2026-08-11): the reference-video (§1-15) and A2V
  // (§1-16) cards now live inside collapsed `<details className="form-
  // accordion">` wrappers below (reference first, audio second — owner
  // order), mirroring Single's own IC-LoRA/A2V accordions
  // (`modes/single/GenerationForm.tsx`). A right-click prefill that lands
  // material inside one of them should open it on mount instead of hiding
  // where the material went — same `useLayoutEffect` rationale as Single's:
  // this write must land in the SAME synchronous commit as the rest of the
  // mount, ahead of the auto-load `useEffect` above (a passive effect), or a
  // transient note from that effect can get coalesced away under React's
  // batching (see `GenerationForm.tsx`'s identical comment for the
  // regression this fixes — do NOT downgrade this to a plain `useEffect`).
  //
  // Both flags now have a real firing path, and both are handled by the
  // auto-load effect above: `autoOpenAudio` by 台帳§1-16's two long-a2v
  // right-click intents, `autoOpenReference` by §1-15 W4's
  // `reference-video-chain` (2026-08-11).
  const audioAccordionRef = useRef<HTMLDetailsElement>(null);
  const referenceAccordionRef = useRef<HTMLDetailsElement>(null);
  const endSourceAccordionRef = useRef<HTMLDetailsElement>(null);
  const autoOpenAudio =
    initialIntent?.intent === "video-audio-to-long-a2v" || initialIntent?.intent === "audio-to-long-a2v";
  const autoOpenReference = initialIntent?.intent === "reference-video-chain";
  // 素材（末尾）: 🎬🖼️「これで終わる動画を作る」 lands material in the end
  // accordion, so it opens on mount rather than hiding where the material went.
  const autoOpenEndSource = initialIntent?.intent === "end-with-this";
  useLayoutEffect(() => {
    if (autoOpenEndSource && endSourceAccordionRef.current) {
      endSourceAccordionRef.current.open = true;
    }
    if (autoOpenReference && referenceAccordionRef.current) {
      referenceAccordionRef.current.open = true;
    }
    if (autoOpenAudio && audioAccordionRef.current) {
      audioAccordionRef.current.open = true;
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- mount-only: this body remounts fresh per intent (key change on ChainedScreen), never re-renders with a new autoOpen* value on the same instance
  }, []);

  return (
    <>
    <div className="single-layout">
      <div className="generation-form chained-form">
        {usingFallback && <p className="warning-banner">{strings.chained.configFallbackWarning}</p>}

        <div className="mode-badge-row">
          <span className={`mode-badge${form.mode === "v2v" ? " mode-badge-i2v" : ""}`}>
            {form.mode === "v2v" ? strings.chained.modeBadge.v2v : strings.chained.modeBadge.scratch}
          </span>
        </div>

        {/* W3 レイアウト再構成 (2026-08-11, moved up from just above "Stage-2窓
            セレクト" below): both are advisories surfaced as early as possible,
            ahead of every field they're about, rather than buried after the
            clip list. §1-14: never a Generate gate — the budget marks "this
            will get heavy", not "this is invalid".

            2026-08-12 オーナー決定により、旧仕様（PENDING_TASKS_CLOSED.md §3-68
            由来の「潜在19フレームに切り替えれば解消する場合＝standardのときだけ
            表示」）を反転して BOTH の窓で出すようにした: 潜在19フレームでも予算
            超過は普通に起こり、そのとき黙っているのは「重くなる」という事実を
            隠すことになる。代わりに文面を2つに割り、事実（遅くなる）は常に、
            誘導（19にすれば軽くなるかも）は効き目のある standard のときだけ、
            空白1つで連結して見せる。 */}
        {form.chainWindowOverBudget && (
          <p className="warning-banner">
            {strings.chained.stage2Window.overBudgetWarning}
            {form.stage2Window === "standard" && ` ${strings.chained.stage2Window.overBudgetShorterWindowHint}`}
          </p>
        )}

        {/* §1-15, plan F5: the reference video's OWN comfort-budget advisory —
            amber/non-blocking like the stage-2 one right above, and likewise
            never added to `validityReasons` (Generate stays enabled). */}
        {stage1ReferenceOverBudget && (
          <p className="warning-banner warning-banner-mild">{strings.chained.referenceVideo.stage1OverBudgetWarning}</p>
        )}

        <PresetDropdown
          config={config}
          onSelect={form.applyPreset}
          disabled={disabled}
          label={strings.chained.presets.label}
          placeholder={strings.single.presets.placeholder}
          orientationLabels={strings.single.presets.orientation}
        />

        <SizeFields
          width={form.common.width}
          height={form.common.height}
          minWidth={form.limits.minWidth}
          maxWidth={form.limits.maxWidth}
          minHeight={form.limits.minHeight}
          maxHeight={form.limits.maxHeight}
          disabled={disabled}
          onWidthChange={form.setWidth}
          onHeightChange={form.setHeight}
          gettingSize={form.gettingSize}
          getSizeError={form.getSizeError}
          onGetSize={() => void form.getSizeFromAviUtl2()}
          budgetMarkers={form.chainWindowMarkers}
        />

        <CropOutputField
          value={form.cropOutput}
          onChange={form.setCropOutput}
          maxWidth={form.common.width}
          maxHeight={form.common.height}
          disabled={disabled}
        />

        <FrameRateSeedFields
          frameRate={form.common.frameRate}
          seed={form.common.seed}
          disabled={disabled}
          onFrameRateChange={form.setFrameRate}
          onSeedChange={form.setSeed}
        />

        <ReservedFields />

        <SourceInputPanel
          form={form}
          disabled={disabled}
          nativeBridge={nativeBridge}
          mediaSize={form.sourceMediaSize}
        />

        <ClipsSection form={form} disabled={disabled} />

        {/* 素材（末尾）: directly after the clip list (plan §2-H) — the START
            material is above the list, the END material below it, so the two
            read in the order the output plays. Same accordion treatment as the
            two cards below (`showHeading={false}`; the heading lives in the
            `<summary>`), and the same mount-time auto-open ref for the
            right-click intent that lands material inside it. */}
        <details className="form-accordion" ref={endSourceAccordionRef}>
          <summary className="form-accordion-summary">{strings.chained.endSource.heading}</summary>
          <div className="form-accordion-body">
            <ChainEndSourcePanel form={form} disabled={disabled} nativeBridge={nativeBridge} showHeading={false} />
          </div>
        </details>

        {/* W3 レイアウト再構成 (2026-08-11): §1-15 参照動画 then §1-16 長尺A2V,
            each collapsed inside its own accordion (owner order: reference
            first, audio second — swapped from the previous audio-first
            layout). `showHeading={false}` on the panel: the heading now lives
            in the `<summary>` instead. `ref` feeds the mount-time auto-open
            effect above (right-click prefills that land material inside one
            of these should open it, not hide it). */}
        <details className="form-accordion" ref={referenceAccordionRef}>
          <summary className="form-accordion-summary">{strings.chained.referenceVideo.heading}</summary>
          <div className="form-accordion-body">
            <ChainReferencePanel form={form} disabled={disabled} nativeBridge={nativeBridge} showHeading={false} />
          </div>
        </details>

        <details className="form-accordion" ref={audioAccordionRef}>
          <summary className="form-accordion-summary">{strings.chained.sourceAudio.heading}</summary>
          <div className="form-accordion-body">
            <ChainAudioPanel form={form} disabled={disabled} nativeBridge={nativeBridge} showHeading={false} />
          </div>
        </details>

        <hr className="chained-section-divider" />

        <div className="field">
          <span className="field-label">{strings.chained.overlapHeading}</span>
          <div className="field-row">
            <label className="field">
              <span className="field-label">{strings.chained.overlapFramesLabel}</span>
              <input
                type="range"
                min={MIN_OVERLAP_FRAMES}
                max={MAX_OVERLAP_FRAMES}
                step={1}
                value={form.overlapFrames}
                disabled={disabled}
                onChange={(e) => form.setOverlapFrames(Number(e.target.value))}
              />
              <span className="field-hint">{form.overlapFrames} frames</span>
            </label>
            <label className="field">
              <span className="field-label">{strings.chained.overlapStrengthLabel}</span>
              <input
                type="range"
                min={MIN_OVERLAP_STRENGTH}
                max={MAX_OVERLAP_STRENGTH}
                step={0.05}
                value={form.overlapStrength}
                disabled={disabled}
                onChange={(e) => form.setOverlapStrength(Number(e.target.value))}
              />
              <span className="field-hint">{form.overlapStrength.toFixed(2)}</span>
            </label>
          </div>
          {/* 逆順Chained (2026-08-18, second stage), 2.6(e): the seam-blend
              width is nudged to 1 the moment 素材（末尾）meets a 2nd clip (and
              restored on the way back out) — this line is the ONLY signal that
              happened, so the user knows the slider above is still theirs to
              move. */}
          {form.hasEndSource && form.clips.length > 1 && (
            <p className="field-hint">{strings.chained.endSource.reverseOverlapHint}</p>
          )}
          <label className="field field-inline">
            <input
              type="checkbox"
              checked={form.chunkedUpsample}
              disabled={disabled}
              onChange={(e) => form.setChunkedUpsample(e.target.checked)}
            />
            <span className="field-label">{strings.chained.chunkedUpsampleLabel}</span>
          </label>
          <p className="field-hint">{strings.chained.chunkedUpsampleHint}</p>
        </div>

        {/* §1-14/§3-57 stage-2 window. Copy rework (owner decision, 2026-08-09):
            presented as the LATENT-FRAME LENGTH of the clip stage-2 cuts the
            draft into (`vTile`), with fixed approximate seconds baked into the
            copy rather than derived live from the form's frame rate — see
            `strings.chained.stage2Window`'s own JSDoc in `i18n/strings.ts` for
            why. The hint is two paragraphs (`\n\n`-separated in the dictionary)
            rendered as two stacked `.field-hint`s so the break survives — that
            class doesn't set `white-space`, so a lone `\n` would collapse. */}
        <label className="field">
          <span className="field-label">{strings.chained.stage2Window.label}</span>
          <select
            value={form.stage2Window}
            disabled={disabled}
            onChange={(e) => form.setStage2Window(e.target.value as Stage2Window)}
          >
            <option value="standard">{strings.chained.stage2Window.standardOption}</option>
            <option value="high_resolution">{strings.chained.stage2Window.highResolutionOption}</option>
          </select>
          {strings.chained.stage2Window.hint.split("\n\n").map((paragraph, i) => (
            <p className="field-hint" key={i}>
              {paragraph}
            </p>
          ))}
        </label>

        {!form.isClipCountValid && (
          <p className="warning-banner">{strings.chained.minClipsError(form.minClips)}</p>
        )}
        {!form.isTotalFramesValid && (
          <p className="warning-banner">{strings.chained.totalFramesExceeded(MAX_CHAIN_TOTAL_FRAMES)}</p>
        )}
        {form.activeSourceKind === "video" && form.sourceVideo.state.status === "ready" && !form.isContextFramesValid && (
          <p className="warning-banner">{strings.chained.sourceVideo.contextFramesError}</p>
        )}
      </div>

      <div className="generation-column">
        <GenerateButtonBar
          label={generateLabel}
          disabled={submitting || hasActiveJob || !form.isValid}
          onGenerate={() => void handleGenerate()}
          hint={`${strings.chained.totalFramesLabel(form.totalFrames, MAX_CHAIN_TOTAL_FRAMES)} — ${form.estimateLabel}`}
        />
        <GenerateReasonsNote reasons={form.validityReasons} messages={generateReasonMessages} />
        {submitState.phase === "error" && (
          <div className="card card-error">
            <p className="error-code">{submitState.code}</p>
            <p>{submitState.message}</p>
          </div>
        )}
        <JobLedger baseUrl={baseUrl} highlightedJobId={highlightedJobId} nativeBridge={nativeBridge} />
      </div>
    </div>
    {/* §1-7 バッチi2v-long: Chainの設定を流用して画像1枚ごとに長尺chainを直列生成 */}
    <BatchI2vLongSection
      config={config}
      chain={form}
      hasActiveJob={hasActiveJob}
      nativeBridge={nativeBridge}
    />
    </>
  );
}

function ClipsSection({ form, disabled }: { form: UseChainFormResult; disabled: boolean }) {
  const strings = useStrings();
  return (
    <div className="field">
      <span className="field-label">{strings.chained.clipsHeading}</span>
      <ul className="clip-list">
        {/* §1-16 長尺A2V: `audioWindow` is the per-clip 🎵 badge's data. The
            windows are computed for the WHOLE list at once (each depends on
            every clip's length), so they are handed out here by index; `null`
            — no audio attached, or a geometry the mirror can't resolve — hides
            the badge. */}
        {form.clips.map((clip, index) => (
          <ClipCard
            key={clip.id}
            clip={clip}
            index={index}
            frameRate={form.common.frameRate}
            disabled={disabled}
            canRemove={form.clips.length > form.minClips}
            audioWindow={form.audioSegmentWindowsForClips?.[index] ?? null}
            onPromptChange={(value) => form.setClipPrompt(clip.id, value)}
            onNumFramesChange={(value, snap) => form.setClipNumFrames(clip.id, value, snap)}
            onRemove={() => form.removeClip(clip.id)}
          />
        ))}
      </ul>

      <button type="button" className="secondary-button" disabled={disabled || !form.canAddClip} onClick={form.addClip}>
        {strings.chained.addClipButton}
      </button>
      {/* 逆順Chained (2026-08-18, second stage): an end source no longer caps
          the button at 1 clip — 2+ clips is the `reverse` mode the server now
          accepts (窓内モード, 2026-08-17, was single-clip only) — so this is
          back to the ONE plain reason every other Chain mode shares. */}
      {!form.canAddClip && <span className="field-hint"> {strings.chained.maxClipsReached(form.maxClips)}</span>}

      {/* W3 レイアウト再構成 (2026-08-11): moved here from the bottom of the
          form (below "クリップを追加" + the maxClipsReached span, NOT between
          them — an inline span sitting between a button and this paragraph
          would break the button/span's inline flow). The old accompanying
          `outputFramesNote` sentence is dropped (2026-08-11 owner feedback) —
          this predicted-output line now stands alone. */}
      {/* 素材（末尾）窓内モード: the TOTAL is the same either way (the anchor is
          frozen inside the last clip, not appended after it) — what the
          breakdown adds is WHICH part of that total is the material
          ("…・うち末尾0.3秒（8フレーム）は素材"). Without material attached it is
          the plain label, byte-identical to before this feature existed. */}
      <p className="field-hint">
        {form.outputTailFrames > 0
          ? strings.chained.outputFramesWithTailLabel(
              form.outputFrames,
              form.outputSeconds,
              form.outputTailFrames,
              form.outputTailSeconds,
            )
          : strings.chained.outputFramesLabel(form.outputFrames, form.outputSeconds)}
      </p>
    </div>
  );
}
