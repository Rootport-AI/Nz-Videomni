import { useCallback, useEffect, useMemo, useRef } from "react";
import type { AppConfig } from "../../api/types";
import { bridge as defaultBridge } from "../../bridge";
import type { NativeBridge } from "../../bridge";
import { useStrings } from "../../i18n/LanguageContext";
import { useJobsContext } from "../../jobs/JobsContext";
import { JobLedger } from "../../jobs/JobLedger";
import type { ControlLoraSelection } from "../../lora/controlLoras";
import type { NagSettings } from "../../shell/nagSettings";
import type { AccelerationSettings } from "../../shell/accelerationSettings";
import { useToasts } from "../../shell/ToastContext";
import { usePrefillPolicy } from "../../shell/PrefillPolicyContext";
import { BatchSection } from "../batch/BatchSection";
import { computeTargetNumFrames } from "../../timeline/deriveDuration";
import { DURATION_POLICY_BY_INTENT, materialDurationForIntent, resolvePrefillSeed } from "../../timeline/prefillSeed";
import type { GenerationPrefill } from "../../timeline/generationPrefill";
import { decideSourceTrim, trimQuery } from "../../timeline/sourceTrim";
import type { SourceTrimDecision } from "../../timeline/sourceTrim";
import { bindToJob, publishFormValues, publishKeyframeCount, publishSourceSlots, rollbackReservedPlacement } from "../../timeline/provisionalReservation";
import { subscribeCreateCommands } from "../../timeline/createLiveCommands";
import { useShowNote } from "../../shell/NoteArea";
import { fileNameFromPath } from "./keyframeUtils";
import { MIN_NUM_FRAMES } from "./defaultConfig";
import { findOverflowFrameIdxs, nextAddPosition, placeableSlotCount } from "./keyframeGrid";
import { KeyframeShrinkModal } from "./KeyframeShrinkModal";
import { GenerateButtonBar } from "./GenerateButtonBar";
import { GenerateReasonsNote } from "./GenerateReasonsNote";
import { GenerationForm } from "./GenerationForm";
import { snapNumFrames } from "./paramUtils";
import { useConfig } from "./useConfig";
import { useDurationShrinkGuard } from "./useDurationShrinkGuard";
import { useGenerationSubmit } from "./useGeneration";
import { useGenerationForm } from "./useGenerationForm";
import { useKeyframes } from "./useKeyframes";
import "./SingleScreen.css";

export interface SingleScreenProps {
  /** Owned by `AppShell`'s shared `PromptBar`, above the mode tabs. */
  prompt: string;
  baseUrl: string | null;
  /** One-shot right-click prefill (task: "ルーティング→プリフィル"). Present
   * only when this screen was opened by a routed menu action; consumed once in
   * the form's lazy init. `intent === "reference-video"` selects IC-LoRA. */
  initialIntent?: GenerationPrefill | undefined;
  /** Test/integration seam threaded from `AppShell`; production omits it. */
  nativeBridge?: NativeBridge | undefined;
  /** The currently highlighted job (last submitted / toast-clicked), threaded
   * to the ledger below. */
  highlightedJobId: string | null;
  /** U-R1: report a freshly submitted `job_id` up to `AppShell`, which sets it
   * as the highlighted job (`setHighlightedJobId`). */
  onJobSubmitted: (jobId: string) => void;
  /** IC-LoRA UI redesign (第5波): the panel's selection state, owned by
   * `AppShell` (not this screen) — see its doc comment. Threaded straight
   * into `useGenerationForm`. */
  controlLora: ControlLoraSelection | null;
  setControlLora: (value: ControlLoraSelection | null) => void;
  controlLoraNames: ReadonlySet<string>;
  /** NAG (2026-07-28): the shared Negative Prompt accordion's settings,
   * owned by `AppShell` (single `NagSettings` object, D1) — same
   * "caller owns the state" shape as `controlLora` above. Optional (rather
   * than required) so pre-existing direct-render tests that don't pass it
   * keep compiling; every hook this flows into already defaults an omitted
   * `nag` to the frozen `NAG_OFF` sentinel. */
  nag?: NagSettings | undefined;
  /** Acceleration (2026-07-31): the Settings panel's shared attention-backend
   * choice, owned by `AppShell` (single `AccelerationSettings` object, D1) —
   * same "caller owns the state" shape as `nag` above. Optional so
   * pre-existing direct-render tests that omit it keep compiling;
   * `useGenerationForm` defaults it to the frozen `ACCELERATION_DEFAULTS`
   * sentinel, which sends nothing at all. */
  acceleration?: AccelerationSettings | undefined;
  /** Smart comfort marker (2026-08-18): whether the server reports
   * SageAttention as installed (`shell/accelerationSettings.sageAvailability`,
   * off the same `/status` poll `AppShell` already threads
   * `blockSwapPrefetchAvailability` from). `undefined`/omitted defaults to
   * `null` ("unknown") in `useGenerationForm`, same optional-dep shape as
   * `nag`/`acceleration` above — every pre-existing direct-render test that
   * doesn't pass it keeps compiling. */
  sageAvailable?: boolean | null | undefined;
  /** §3-98 P5: Batch A2V submits `POST /generate/chain`, which the loaded base
   * model's engine may not support (LTX 2.5 v1 refuses the whole chain family
   * with 422 `FEATURE_UNSUPPORTED`). `true` greys the panel's controls the same
   * way its own runner state does — no new mechanism, just another reason.
   * Omitted ⇒ available, so every direct-render test keeps compiling. */
  batchUnavailable?: boolean | undefined;
}

/** The "Create" screen: T2V generation form (minus the prompt, which lives in
 * the shared `PromptBar`). U-R1: submission goes through the slim
 * `useGenerationSubmit` (`POST /generate` / `/generate/chain`, no polling), and
 * the right-hand `.generation-column` stacks the Generate button, a submit-error
 * row, and the full job ledger (`JobLedger`) — there is nothing to the right of
 * that panel. */
export function SingleScreen({
  prompt,
  baseUrl,
  initialIntent,
  nativeBridge,
  highlightedJobId,
  onJobSubmitted,
  controlLora,
  setControlLora,
  controlLoraNames,
  nag,
  acceleration,
  sageAvailable,
  batchUnavailable,
}: SingleScreenProps) {
  const strings = useStrings();
  const configState = useConfig();

  if (configState.status === "loading") {
    return <p className="hint">{strings.single.loadingConfig}</p>;
  }

  return (
    <SingleScreenBody
      config={configState.config}
      usingFallback={configState.usingFallback}
      prompt={prompt}
      baseUrl={baseUrl}
      initialIntent={initialIntent}
      nativeBridge={nativeBridge}
      highlightedJobId={highlightedJobId}
      onJobSubmitted={onJobSubmitted}
      controlLora={controlLora}
      setControlLora={setControlLora}
      controlLoraNames={controlLoraNames}
      nag={nag}
      acceleration={acceleration}
      sageAvailable={sageAvailable}
      batchUnavailable={batchUnavailable}
    />
  );
}

interface SingleScreenBodyProps {
  config: AppConfig;
  usingFallback: boolean;
  prompt: string;
  baseUrl: string | null;
  initialIntent?: GenerationPrefill | undefined;
  nativeBridge?: NativeBridge | undefined;
  highlightedJobId: string | null;
  onJobSubmitted: (jobId: string) => void;
  controlLora: ControlLoraSelection | null;
  setControlLora: (value: ControlLoraSelection | null) => void;
  controlLoraNames: ReadonlySet<string>;
  nag?: NagSettings | undefined;
  acceleration?: AccelerationSettings | undefined;
  sageAvailable?: boolean | null | undefined;
  batchUnavailable?: boolean | undefined;
}

function SingleScreenBody({
  config,
  usingFallback,
  prompt,
  baseUrl,
  initialIntent,
  nativeBridge,
  highlightedJobId,
  onJobSubmitted,
  controlLora,
  setControlLora,
  controlLoraNames,
  nag,
  acceleration,
  sageAvailable,
  batchUnavailable = false,
}: SingleScreenBodyProps) {
  const strings = useStrings();
  const toasts = useToasts();
  const jobsCtx = useJobsContext();
  const showNote = useShowNote();

  // I4 (✨ minimal binding): `bindToJob` needs the confirmed length/fps and a
  // display text, but `form`/`prompt` are declared below this callback, so we
  // read the latest values through a ref updated on every render (see below).
  const bindInfoRef = useRef({ numFrames: 0, genFps: 0, displayText: "", textPrefix: "" });

  // U-R1: on a successful submit, refresh the ledger (so the job is visible
  // before the button re-enables — MN-6) and highlight it. The submit hook
  // holds `submitting` until this resolves.
  const onSubmitted = useCallback(
    async (jobId: string) => {
      await jobsCtx.refresh();
      onJobSubmitted(jobId);
      // I4: hand the reserved provisional (if any) off to this real job
      // (RIGHTCLICK_REDESIGN_SPEC.md §5-3 用途a). `bindToJob` is a no-op unless
      // a ✨ reservation is actually waiting (phase "reserved"), so a plain
      // panel-origin Generate never touches it. `onSubmitted` runs only after
      // the backend accepts the job — a 422/busy sync failure never gets here,
      // so the reservation correctly stays "reserved" (spec design intent).
      const bridge = nativeBridge ?? defaultBridge;
      try {
        await bindToJob(bridge, { jobId, ...bindInfoRef.current });
      } catch {
        // A bind failure must never fail the submit; the reservation just
        // stays where it is and the user can retry from the timeline.
      }
    },
    [jobsCtx, onJobSubmitted, nativeBridge],
  );
  // X2(a): a synchronous submit failure (422/busy) never reaches `onSubmitted`,
  // so a ✨/📷/#4/#5 right-click reservation would stay `reserved` forever and
  // block every later generation. Roll the still-unbound seat + its placeholder
  // back to idle. `rollbackReservedPlacement` is a no-op unless a reservation is
  // actually waiting (phase "reserved"), so a plain panel-origin Generate leaves
  // the seat untouched; it never touches a bound (`generating`) real job.
  const onFailed = useCallback(async () => {
    const bridge = nativeBridge ?? defaultBridge;
    await rollbackReservedPlacement(bridge);
  }, [nativeBridge]);
  const { submitState, submit, submitChain } = useGenerationSubmit({ onSubmitted, onFailed });

  const isICLora = initialIntent?.intent === "reference-video";
  // Right-click redesign W2 (accordion auto-expand): mirrors the intent
  // branch the I9/I10 auto-load effect below already switches on
  // (`menuRouting.ts`'s `reference-video` / `video-audio-to-video` /
  // `audio-to-video`) — the material lands in the IC-LoRA or A2V accordion,
  // so that accordion should start open instead of hiding where it went.
  // Passed straight through to `GenerationForm`, which applies them once on
  // mount (this whole body remounts via `key` on a new intent, so a fresh
  // mount = a fresh one-shot open, same as the auto-load effect).
  // X5 (併用プリフィルでアコーディオンを閉じない): also open an accordion when the
  // OPPOSITE slot was carried across this remount (`carryOver`), not only when
  // this intent targets it. #3/#7 carry the existing reference video (keep
  // IC-LoRA open), #2 carries the existing audio (keep A2V open) — otherwise the
  // carried material survives but its accordion collapses, hiding it. `!= null`
  // coerces the optional-chain result to a plain boolean for GenerationForm's
  // boolean props. `carryOver` is declared below, so reference it inline here.
  const autoOpenReference =
    initialIntent?.intent === "reference-video" || initialIntent?.carryOver?.referenceVideo != null;
  const autoOpenA2v =
    initialIntent?.intent === "video-audio-to-video" ||
    initialIntent?.intent === "audio-to-video" ||
    initialIntent?.carryOver?.sourceAudio != null;
  // Derive the whole right-click prefill seed (width/height + fps + DURATION)
  // in ONE call to the shared純関数 `resolvePrefillSeed`, so the form seed and
  // `AppShell`'s provisional-reservation length come from the identical
  // arithmetic on the identical inputs — a placed provisional's ribbon can't
  // diverge from what the remounted form shows (owner requirement 2026-07-21
  // #1). Computed once (this whole body remounts via `key` on a new intent) and
  // handed to the form's lazy init. `derived` is returned whole so `notes`/
  // `width`/`height` references below are unchanged; the seed also decides the
  // #2 IC-LoRA reference / #3/#7 A2V material-clamped DURATION and the #4/#6/#12
  // comfort-ceiling DURATION (A2V's is then superseded by the wav auto-adjust —
  // see the A2V auto-load below). `deriveGenerationParams` remains the frozen
  // Phase-1 width/height contract inside `resolvePrefillSeed`.
  // W7: the right-click prefill resolution/fps policy (Settings). Only affects
  // the initial-value decision here — ordinary edits/presets are unchanged; the
  // `project` policy overwrites width/height/fps AND recomputes DURATION off
  // `getEditInfo` in the mount effect below.
  const { sizePolicy, fpsPolicy } = usePrefillPolicy();
  const seed = useMemo(
    () =>
      initialIntent
        ? resolvePrefillSeed({
            intent: initialIntent.intent,
            selection: initialIntent.selection,
            config,
            sizePolicy,
            fpsPolicy,
          })
        : null,
    [initialIntent, config, sizePolicy, fpsPolicy],
  );
  const derived = seed?.derived ?? null;
  const prefillFrameRate = seed?.frameRate;
  const prefillNumFrames = seed?.numFrames;

  // W5 (反対スロット保持): the opposite source slot carried across this remount —
  // #2 carries the existing audio, #3/#7 carry the existing reference video +
  // strengths (AppShell filled the relevant side). Threaded into the form's
  // one-shot initial so the carried slot mounts ready (no re-upload).
  const carryOver = initialIntent?.carryOver;
  const form = useGenerationForm(
    config,
    prompt,
    { nativeBridge, controlLora, setControlLora, controlLoraNames, nag, acceleration, sageAvailable },
    {
      ...(derived
        ? {
            width: derived.width,
            height: derived.height,
            isICLora,
            ...(prefillFrameRate != null ? { frameRate: prefillFrameRate } : {}),
            ...(prefillNumFrames != null ? { numFrames: prefillNumFrames } : {}),
            // W4 (#7 trim追従): audio-to-video uploads the whole audio file, so the
            // form's wav auto-adjust would otherwise re-seed DURATION off the FULL
            // file length, overriding the span (trimmed ribbon) seed. Cap the wav
            // suggestion at the span-derived DURATION so #7 follows the ribbon; the
            // cap lifts once the user swaps the audio (see useGenerationForm).
            ...(initialIntent?.intent === "audio-to-video" && prefillNumFrames != null
              ? { a2vSeedMaxFrames: prefillNumFrames }
              : {}),
          }
        : {}),
      // W5: #3/#7 carry the existing reference video + its two strengths. This
      // also forces the 128 grid from mount (回帰対策1, in useGenerationForm).
      ...(carryOver?.referenceVideo
        ? {
            carryReferenceVideo: {
              id: carryOver.referenceVideo.id,
              fileName: carryOver.referenceVideo.fileName,
              filePath: carryOver.referenceVideo.filePath,
            },
            carryConditioningAttentionStrength: carryOver.referenceVideo.conditioningAttentionStrength,
            carryReferenceVideoStrength: carryOver.referenceVideo.referenceVideoStrength,
          }
        : {}),
      // W5: #2 carries the existing audio; NOT paired with `a2vSeedMaxFrames`
      // (a carried audio is not a #7 prefill material, so it follows the full
      // wav length — 回帰対策2 fires the mount-time probe in useGenerationForm).
      ...(carryOver?.sourceAudio ? { carryAudio: carryOver.sourceAudio } : {}),
    },
  );

  // W7 `project` policy: read the AviUtl2 project's own resolution/fps via
  // `getEditInfo` ONCE on this prefilling remount and overwrite the form's
  // width/height/frameRate. A consumed ref limits it to a single run before the
  // user touches anything (the "form initializes once, never re-applied later"
  // invariant is preserved for every other path — this is a deliberate,
  // prefill-only, mount-time overwrite). getEditInfo failure leaves the
  // material-derived seed in place (③ fallback). Only runs for an actual
  // prefill under the `project` policy; every other case is a no-op.
  const projectOverwriteConsumedRef = useRef(false);
  useEffect(() => {
    if (projectOverwriteConsumedRef.current) return;
    // W1: the two axes are independent, so the overwrite runs whenever EITHER
    // the size or the fps policy is `project`. Each axis is then applied only
    // when its own policy is `project`; the other axis keeps the material seed.
    if (!initialIntent || (sizePolicy !== "project" && fpsPolicy !== "project")) return;
    projectOverwriteConsumedRef.current = true;
    void (async () => {
      const bridge = nativeBridge ?? defaultBridge;
      try {
        const editInfo = await bridge.request("getEditInfo", {});
        const projectGenFps =
          editInfo.scale > 0 && editInfo.rate > 0 ? editInfo.rate / editInfo.scale : config.generation_defaults.frame_rate;
        // Apply only the axis (or axes) whose policy is `project`.
        if (sizePolicy === "project") {
          form.setWidth(editInfo.width);
          form.setHeight(editInfo.height);
        }
        if (fpsPolicy === "project" && editInfo.scale > 0 && editInfo.rate > 0) {
          form.setFrameRate(projectGenFps);
        }
        // W1/W8 (② project policy): keep DURATION consistent with the just-applied
        // axes. The DURATION target is recomputed off the EFFECTIVE resolution +
        // fps: the `project` axis takes the getEditInfo value, while a non-project
        // axis reads back the material seed already in the form (form.values here
        // is the mount-time seed, since this effect runs once). This is the only
        // way the crossing cases (size=project × fps=material, and the reverse)
        // land correctly (adversarial review R4). The material duration input
        // still goes through `materialDurationForIntent` (M-1) so the span
        // semantics match the initial seed. Leaves the seed in place when the
        // policy can't resolve a value (null) or the intent has no DURATION policy.
        const effectiveWidth = sizePolicy === "project" ? editInfo.width : form.values.width;
        const effectiveHeight = sizePolicy === "project" ? editInfo.height : form.values.height;
        const effectiveGenFps = fpsPolicy === "project" ? projectGenFps : form.values.frameRate;
        const policy = DURATION_POLICY_BY_INTENT[initialIntent.intent];
        if (policy) {
          const materialDurationSec = materialDurationForIntent(initialIntent.intent, initialIntent.selection);
          const target = computeTargetNumFrames({
            policy,
            width: effectiveWidth,
            height: effectiveHeight,
            spillFreeFrames: config.limits.spill_free_frames,
            minNumFrames: MIN_NUM_FRAMES,
            maxNumFrames: config.limits.max_num_frames,
            ...(materialDurationSec !== undefined ? { materialDurationSec } : {}),
            genFps: effectiveGenFps,
          });
          if (target !== null) form.setNumFrames(target);
        }
      } catch {
        // Fall back to the material-derived seed already in the form (③ behavior).
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps -- one-shot mount consume
  }, []);
  const keyframes = useKeyframes(config, { nativeBridge });

  // I9 §3-4 (#2/#4) / §6: auto-load the routed material ONCE on this remount,
  // mirroring ChainedScreen's I8 pattern. #2 `reference-video` uploads the
  // selected video into the IC-LoRA reference slot; #4 `image-to-video` adds
  // the selected image as KEYFRAMES' 1st card (frame 0) — the fresh remount
  // guarantees KEYFRAMES is empty, so it always lands as the 1st keyframe and
  // the mode badge becomes I2V. A lazy useState initializer can't drive the
  // async upload, so this runs in a mount effect guarded by a consumed ref
  // (StrictMode-safe; the body remounts via `key` on a new intent, so a fresh
  // ref = a fresh one-shot per prefill). `autoLoadPendingRef` lets the
  // status-watch effect below report only THIS load's outcome (the §6-2
  // missing-material note), never a later user-initiated error. (#10
  // image-from-frame is NOT here — it keeps the panel state and arrives via the
  // live-command channel below, not a remount/intent.)
  const autoLoadConsumedRef = useRef(false);
  const autoLoadPendingRef = useRef(false);
  useEffect(() => {
    if (autoLoadConsumedRef.current) return;
    const intent = initialIntent?.intent;
    if (
      intent !== "reference-video" &&
      intent !== "image-to-video" &&
      intent !== "audio-to-video" &&
      intent !== "video-audio-to-video"
    ) {
      return;
    }
    autoLoadConsumedRef.current = true;
    // §1-6 拡張: the routed selection is also the trim measurement input for #2
    // below, so it is bound once here (`selection`) rather than re-derived —
    // `item` is its first object, exactly as before.
    const selection = initialIntent?.selection;
    const item = selection?.selected[0];

    // I10 §3-4 #3 (videoAudioToVideo): unlike every other material item, this
    // one does NOT load the selected file directly. It extracts the audio *as it
    // actually plays on the timeline* (mix path) over the object's real frame
    // range, then feeds that wav into the SAME source-audio slot the rest of the
    // A2V flow uses. Kept as its own async routine because it opens with a
    // distinct "extracting…" note and has native-specific failure branches.
    if (intent === "video-audio-to-video") {
      if (!item) {
        showNote("warning", strings.notes.sourceFileMissing);
        return;
      }
      // Extraction can take several seconds; the one-shot `autoLoadConsumedRef`
      // above already prevents a second run (StrictMode / re-render).
      void (async () => {
        // frameEnd is INCLUSIVE: native's `FindObjectByJob` walks objects with
        // `next = frameEnd + 1` (native/src/bridge.cpp), so the object spans
        // [frameStart, frameEnd] and its real length is frameEnd - frameStart + 1.
        // (If a 1-frame boundary drift ever surfaces on real hardware, this `+1`
        // is the first thing to re-check against the object's true frame span.)
        const frameCount = item.frameEnd - item.frameStart + 1;
        // §3-4 #3: extracting note + mix-path explanation, up front.
        showNote("info", strings.notes.extractingAudio);
        const bridge = nativeBridge ?? defaultBridge;
        try {
          // §7-2 design decision: call `timeline.extractAudio` DIRECTLY rather
          // than `timeline/cutoutAndUpload.ts`'s `extractAudioAndUpload` — that
          // helper *also* uploads to the backend itself, which would
          // double-upload against the source-audio slot (`useSourceUpload`) we
          // hand the wav to below. Uploading is funnelled through the single
          // source-audio channel; `cutoutAndUpload.ts` is intentionally left
          // untouched.
          const extracted = await bridge.request("timeline.extractAudio", {
            layer: item.layer,
            frameStart: item.frameStart,
            frameCount,
            // mix path (§3-4 #3): capture the sound as it actually plays; solo
            // separation is a future task, so no layers are kept solo.
            audioMode: "mix",
            soloKeepLayers: [],
          });
          // Success: the wav goes into the source-audio slot (wav output means
          // the form's Frames auto-adjust fires — §3-4 #3). `autoLoadPendingRef`
          // lets the status-watch below downgrade to the missing-material note
          // if the (temp-file) upload itself fails.
          showNote("info", strings.notes.loadedVideoAudio);
          autoLoadPendingRef.current = true;
          void form.attachSourceAudioByPath(extracted.filePath, fileNameFromPath(extracted.filePath));
        } catch (err) {
          // §3-4 #3 failure branches. Native returns a single `EXTRACT_FAILED`
          // code for every failure and only the message differentiates them
          // (native/src/bridge.cpp `ExtractAudioWorker`): "…silent range" for a
          // muted/disabled range; "audio render failed at frame N" for a
          // per-frame render failure whose dominant cause is the worker's render
          // timeout; and setup failures ("rendering_scene_audio is not
          // available" / "failed to write wav file") otherwise. Native does not
          // hand us finer-grained codes, so we classify on those message
          // substrings and fall through to the generic note for anything we
          // cannot place.
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

    const filePath = item?.filePath ?? null;
    if (!filePath) {
      // §6-2: the path couldn't even be resolved (object without a backing file).
      showNote("warning", strings.notes.sourceFileMissing);
      return;
    }
    const fileName = fileNameFromPath(filePath);
    // §6 receipt note, shown at load START — the visible reference slot / new
    // keyframe card / filled audio slot is the primary feedback; this reinforces it.
    showNote("info", strings.notes.loadedFromRightClick(fileName));
    autoLoadPendingRef.current = true;
    if (intent === "reference-video") {
      // #2: video path → IC-LoRA reference slot. `deriveGenerationParams`
      // already seeded 128-rounded width/height (isICLora), and the form
      // re-snaps to 128 again once the upload id turns non-null.
      //
      // §1-6 拡張 (2026-08-01): the reference video is trimmed to the ribbon's
      // own range, exactly like #1 extend-video's source. Before this, a ribbon
      // showing frames 31..120 of a 321-frame file uploaded the WHOLE file, and
      // the pipeline (which reads `frame_cap` frames from the START) conditioned
      // on frames 1..90 — visibly the wrong footage. `decideSourceTrim` is the
      // SAME conservative decision #1 uses (`modes/chained/ChainedScreen.tsx`), and
      // `trimQuery` returns `undefined` for every no-trim decision, so an
      // untrimmed upload's request params stay byte-identical to before —
      // machine-checked by `SingleScreen.prefill.test.tsx`'s strict assertion.
      // The manual 📁 pick path (`referenceVideo.pick()`) has no timeline
      // object to measure and is deliberately left untrimmed.
      const decision: SourceTrimDecision = selection
        ? decideSourceTrim(item, selection)
        : { trim: false, reason: "noItem" };
      void form.referenceVideo.uploadPath(filePath, fileName, trimQuery(decision));
    } else if (intent === "audio-to-video") {
      // #7 (§3-4 #7): audio object → source-audio slot, enabling A2V once the
      // upload is ready. `attachSourceAudioByPath` primes `lastAudioFilePathRef`
      // so a wav triggers the form's Frames auto-adjust (fs.probeAudioDuration);
      // a non-wav is accepted without adjustment (existing behavior).
      void form.attachSourceAudioByPath(filePath, fileName);
    } else {
      // #4: image → KEYFRAMES 1st card (frame 0), flipping the mode badge to I2V.
      void keyframes.addFromPath(filePath, fileName, 0);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- one-shot mount consume
  }, []);

  // §6-2: replace the receipt with the missing-material note if the auto-load
  // upload failed. Gated on `autoLoadPendingRef` so it fires only for the
  // auto-load's own terminal state, then stops watching (a later manual
  // re-pick/replace that errors won't re-trigger it).
  const autoLoadIntent = initialIntent?.intent;
  const autoLoadStatus =
    autoLoadIntent === "reference-video"
      ? form.referenceVideo.state.status
      : // #7 audio-to-video / #3 video-audio-to-video (after extraction) both feed
        // the source-audio slot, so its upload status is the terminal signal to
        // watch for the §6-2 missing-material downgrade.
        autoLoadIntent === "audio-to-video" || autoLoadIntent === "video-audio-to-video"
        ? form.sourceAudio.state.status
        : keyframes.items[0]?.status;
  useEffect(() => {
    if (!autoLoadPendingRef.current) return;
    if (autoLoadStatus === "error") {
      showNote("warning", strings.notes.sourceFileMissing);
      autoLoadPendingRef.current = false;
    } else if (autoLoadStatus === "ready") {
      autoLoadPendingRef.current = false;
    }
  }, [autoLoadStatus, showNote, strings]);

  // I9 §3-6 (#10): 📷 imageFromCurrentFrame keeps the panel state (D3: no
  // remount), so AppShell can't hand it a `pendingIntent`. It instead dispatches
  // a live command to this always-mounted screen. On receipt we capture the
  // current timeline frame into KEYFRAMES at frame 0 — the SAME semantics as
  // the KEYFRAMES panel's own new-card 📸 button — with one collision fix
  // (W6, owner-confirmed spec):
  //
  //   • a frame-0 card already exists → REPLACE its image in place via
  //     `keyframes.captureIntoCard(existing.id)` (card count unchanged,
  //     frameIdx/strength preserved). This is the fix: the old code always
  //     called `addFromCapture(0)`, which APPENDED a second frame-0 card on
  //     top of the existing one whenever one was already there — a visible
  //     0-position collision that nothing auto-resolved.
  //   • no frame-0 card, under the cap → append a new one via
  //     `addFromCapture(0)`, exactly as before.
  //   • no frame-0 card, at the cap → `addFromCapture` would no-op, so the
  //     cap is checked up front and reported as a failure inline. The cap
  //     check only applies to this append path — the replace path above never
  //     grows the item count, so it proceeds even when the panel is full.
  //
  // Both `addFromCapture` and `captureIntoCard` swallow their own errors into
  // the target card's `"error"` status (neither throws), so the outcome can't
  // be read right after the await — the error setState hasn't committed yet.
  // Instead the handler records what to watch for in `capturePendingRef`
  // (the pre-capture card ids for the append path, or the target card's own
  // id for the replace path) and a status-watch effect (below) reports the
  // receipt/failure note once the relevant card reaches a terminal status
  // (`ready`/`error`) — the same watch pattern the #2/#4 auto-load uses. Refs
  // keep the subscription itself stable (subscribe once on mount) while
  // reading the latest keyframes/showNote/strings at dispatch time.
  const keyframesRef = useRef(keyframes);
  keyframesRef.current = keyframes;
  const showNoteRef = useRef(showNote);
  showNoteRef.current = showNote;
  const stringsRef = useRef(strings);
  stringsRef.current = strings;
  // W8 (#9/#10 setDuration): the live-command subscription is created once, so
  // reach the current `form.setNumFrames` through a ref (mirrors keyframesRef)
  // rather than closing over the mount-time closure's stale setter.
  const setNumFramesRef = useRef(form.setNumFrames);
  setNumFramesRef.current = form.setNumFrames;
  // I11 §3-4 #5: the "Add keyframe" button's landing/cap semantics
  // (keyframeGrid) depend on the live DURATION and the server grid params. The
  // subscription below is created once, so read these through a ref updated on
  // every render (mirroring keyframesRef) rather than closing over stale values.
  const keyframeAddCtxRef = useRef({ numFrames: 0, multiple: 0, offset: 0 });
  keyframeAddCtxRef.current = {
    numFrames: form.values.numFrames,
    multiple: config.limits.conditioning_frame_idx_multiple,
    offset: config.limits.conditioning_keyframe_grid_offset,
  };
  // W6: the append path (new frame-0 card, or #11's tail card) is watched by
  // `beforeIds` diff (the card that wasn't there before); the replace path
  // (an existing frame-0 card whose image gets swapped via `captureIntoCard`)
  // never changes the id set, so it's watched by `cardId` — the target
  // card's own status transition — instead.
  const capturePendingRef = useRef<{ beforeIds: Set<string> } | { cardId: string } | null>(null);
  // I11 §3-4 #5: the #5 append's own pending card, watched below for its
  // receipt/missing-material note (same pattern as capturePendingRef for #10).
  const appendPendingRef = useRef<{ beforeIds: Set<string>; fileName: string } | null>(null);
  useEffect(() => {
    return subscribeCreateCommands((command) => {
      if (command.type === "captureFrameToKeyframe") {
        const kf = keyframesRef.current;
        const existingAtZero = kf.items.find((item) => item.frameIdx === 0);
        if (existingAtZero) {
          // W6 bugfix: a frame-0 card is already there — replace its image in
          // place instead of appending a second frame-0 card on top of it (the
          // collision the old always-append code produced). Card count is
          // unchanged, so the cap below doesn't apply to this path.
          capturePendingRef.current = { cardId: existingAtZero.id };
          void kf.captureIntoCard(existingAtZero.id);
          return;
        }
        if (!kf.canAdd) {
          // At the keyframe cap: addFromCapture would no-op (no card added), so
          // report the failure here — nothing was captured.
          showNoteRef.current("warning", stringsRef.current.notes.captureFrameFailed);
          return;
        }
        capturePendingRef.current = { beforeIds: new Set(kf.items.map((item) => item.id)) };
        void kf.addFromCapture(0);
        return;
      }
      if (command.type === "addCaptureAsKeyframe") {
        // W5 #11: the capture twin of #10, but the captured frame lands at the
        // TAIL grid slot (the "Add keyframe" button's landing) instead of frame
        // 0. Same landing + cap semantics as #5's `addImageKeyframe` branch
        // below (nextAddPosition / effectiveMax), just fed by a capture rather
        // than a resolved file path. Reuses `capturePendingRef` so the #10
        // capture status-watch reports its receipt/failure note (capture-based:
        // the fileName is only known once the round trip completes).
        const kf = keyframesRef.current;
        const { numFrames, multiple, offset } = keyframeAddCtxRef.current;
        const occupied = new Set(kf.items.map((item) => item.frameIdx));
        const effectiveMax = Math.min(kf.maxItems, placeableSlotCount(numFrames, multiple, offset));
        const nextPos = nextAddPosition(occupied, numFrames, multiple, offset);
        if (kf.items.length >= effectiveMax || nextPos === null) {
          showNoteRef.current("warning", stringsRef.current.notes.keyframeLimitReached);
          return;
        }
        capturePendingRef.current = { beforeIds: new Set(kf.items.map((item) => item.id)) };
        void kf.addFromCapture(nextPos);
        return;
      }
      if (command.type === "addImageKeyframe") {
        // I11 §3-4 #5: append the right-clicked image to KEYFRAMES WITHOUT
        // remounting Create (the form/existing cards survive). AppShell already
        // decided the provisional (system A / none) from the pre-append count;
        // here we only add the card.
        const kf = keyframesRef.current;
        // §6-2: native couldn't resolve the object's backing file — surface the
        // shared missing-material note, mirroring #4's `!filePath` branch.
        if (!command.filePath) {
          showNoteRef.current("warning", stringsRef.current.notes.sourceFileMissing);
          return;
        }
        // Match the "Add keyframe" button's landing + cap semantics exactly
        // (KeyframesPanel): empty → frame 0; existing cards → the tail slot
        // `nextAddPosition` picks; blocked when the item cap is hit OR there is
        // no free grid slot left (atCapacity || noRoom → the button is disabled).
        const { numFrames, multiple, offset } = keyframeAddCtxRef.current;
        const occupied = new Set(kf.items.map((item) => item.frameIdx));
        const effectiveMax = Math.min(kf.maxItems, placeableSlotCount(numFrames, multiple, offset));
        const nextPos = nextAddPosition(occupied, numFrames, multiple, offset);
        if (kf.items.length >= effectiveMax || nextPos === null) {
          showNoteRef.current("warning", stringsRef.current.notes.keyframeLimitReached);
          return;
        }
        appendPendingRef.current = {
          beforeIds: new Set(kf.items.map((item) => item.id)),
          fileName: command.fileName,
        };
        void kf.addFromPath(command.filePath, command.fileName, nextPos);
        return;
      }
      if (command.type === "setDuration") {
        // W8 (#9/#10): live-set DURATION only — no other form value is touched
        // (prompt / width / height / keyframes all survive). `setNumFrames`
        // re-snaps onto the 8n+1 grid and clamps into range, so an already-valid
        // ceiling passes through unchanged.
        setNumFramesRef.current(command.numFrames);
        return;
      }
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps -- subscribe once; refs carry live values
  }, []);

  // I11 §3-4 #5: report the append's outcome once the newly-added card settles
  // (receipt on "ready", §6-2 missing-material on "error") — the #10 watch's
  // twin for the live keyframe append.
  useEffect(() => {
    const pending = appendPendingRef.current;
    if (!pending) return;
    const added = keyframes.items.find((item) => !pending.beforeIds.has(item.id));
    if (!added) return; // card not appended yet
    if (added.status === "ready") {
      appendPendingRef.current = null;
      showNote("info", strings.notes.appendedImageToKeyframe(pending.fileName));
    } else if (added.status === "error") {
      appendPendingRef.current = null;
      showNote("warning", strings.notes.sourceFileMissing);
    }
    // "uploading"/"empty": keep waiting for the terminal status.
  }, [keyframes.items, showNote, strings]);

  // Report the #10 capture's outcome once the relevant card settles — either
  // the newly-appended card (`beforeIds` diff) or, for a W6 replace, the
  // existing frame-0 card itself (`cardId`, watched for its own status
  // transition since no id is added/removed in that path).
  useEffect(() => {
    const pending = capturePendingRef.current;
    if (!pending) return;
    const target =
      "cardId" in pending
        ? keyframes.items.find((item) => item.id === pending.cardId)
        : keyframes.items.find((item) => !pending.beforeIds.has(item.id));
    if (!target) return; // card not appended (or replace target gone) yet
    if (target.status === "ready") {
      capturePendingRef.current = null;
      showNote("info", strings.notes.capturedFrameToKeyframe);
    } else if (target.status === "error") {
      capturePendingRef.current = null;
      showNote("warning", strings.notes.captureFrameFailed);
    }
    // "uploading"/"empty": keep waiting for the terminal status.
  }, [keyframes.items, showNote, strings]);

  // I4: keep the bind-time snapshot current for `onSubmitted` above. `frameRate`
  // is the generation fps native uses to convert the placeholder length; the
  // prompt head is the confirmed stage-2 display text, prefixed with
  // "Generating: " (spec §5-5 段階2 — passed explicitly so native's default
  // ⏳生成中 prefix is never used).
  bindInfoRef.current = {
    numFrames: form.values.numFrames,
    genFps: form.values.frameRate,
    displayText: prompt.trim().slice(0, 20) || "(t2v)",
    textPrefix: strings.provisional.generatingPrefix,
  };

  // D3: publish the live DURATION/FPS so `AppShell`'s ✨ reservation — which no
  // longer remounts Create — sizes its provisional placeholder to the current
  // form rather than the config default. Cheap effect keyed only on the two
  // scalars, so it never adds a render.
  useEffect(() => {
    publishFormValues({
      numFrames: form.values.numFrames,
      frameRate: form.values.frameRate,
      // W8: width/height are the resolution AppShell resolves the #9/#10
      // comfort-ceiling DURATION from before dispatching `setDuration`.
      width: form.values.width,
      height: form.values.height,
    });
  }, [form.values.numFrames, form.values.frameRate, form.values.width, form.values.height]);

  // I11 §3-4 #5: publish the live keyframe count so `AppShell.handleRoute` can
  // branch #5 (addImageKeyframe) at append time — placement A only when this
  // append becomes the 1st keyframe (count 0), else no provisional (§5-9).
  // Keyed only on the length, so an unrelated card edit never re-publishes.
  useEffect(() => {
    publishKeyframeCount(keyframes.items.length);
  }, [keyframes.items.length]);

  // W5 (反対スロット保持): publish the two source slots so `AppShell.handleRoute`
  // can carry the OPPOSITE slot across a #2/#3/#7 remount. Only a `ready` slot is
  // published (id/path valid, re-attachable with no re-upload); uploading/idle/
  // error publish `null` so a half-finished slot is never carried (safe side).
  // The reference video also carries its two adapter strengths.
  const refVideoState = form.referenceVideo.state;
  const sourceAudioState = form.sourceAudio.state;
  const { conditioningAttentionStrength, referenceVideoStrength } = form;
  useEffect(() => {
    publishSourceSlots({
      referenceVideo:
        refVideoState.status === "ready" && refVideoState.id && refVideoState.filePath
          ? {
              id: refVideoState.id,
              fileName: refVideoState.fileName ?? "",
              filePath: refVideoState.filePath,
              conditioningAttentionStrength,
              referenceVideoStrength,
            }
          : null,
      sourceAudio:
        sourceAudioState.status === "ready" && sourceAudioState.id && sourceAudioState.filePath
          ? {
              id: sourceAudioState.id,
              fileName: sourceAudioState.fileName ?? "",
              filePath: sourceAudioState.filePath,
            }
          : null,
    });
  }, [refVideoState, sourceAudioState, conditioningAttentionStrength, referenceVideoStrength]);

  const frameIdxMultiple = config.limits.conditioning_frame_idx_multiple;
  const gridOffset = config.limits.conditioning_keyframe_grid_offset;

  // Keyframe timeline rework (2026-07-18): a DURATION shrink (or a preset that
  // carries a shorter num_frames) can strand a keyframe past the placeable
  // grid. The guard live-applies the duration as before but raises a
  // confirmation modal before deleting the now-out-of-range keyframes.
  const shrinkGuard = useDurationShrinkGuard({
    numFrames: form.values.numFrames,
    setNumFrames: form.setNumFrames,
    applyPreset: form.applyPreset,
    getPresetNumFrames: (name) => {
      const preset = config.generation_presets[name];
      // Mirror exactly what `applyPreset(name)` would set: the preset's
      // num_frames run through the same snap/clamp the form uses.
      return preset ? snapNumFrames(preset.num_frames, MIN_NUM_FRAMES, config.limits.max_num_frames) : null;
    },
    items: keyframes.items,
    removeKeyframe: keyframes.remove,
    multiple: frameIdxMultiple,
    offset: gridOffset,
  });

  // Final防壁 for out-of-range keyframes: the shrink guard covers deliberate
  // duration/preset edits, but a range-key nudge on the timeline, a click
  // before a field's blur commits, or an async race can still leave a pin
  // past the grid. Checked against ALL items (any status), so an "empty" or
  // still-"uploading" stranded pin blocks Generate just the same.
  const overflowKeyframes = findOverflowFrameIdxs(keyframes.items, form.values.numFrames, frameIdxMultiple, gridOffset);
  const hasOverflowKeyframes = overflowKeyframes.length > 0;

  // MJ-1: the form is only frozen while a submit is in flight; a running job
  // (`hasActiveJob`) leaves the form editable and blocks the Generate button
  // alone.
  const submitting = submitState.phase === "submitting";
  const hasActiveJob = jobsCtx.hasActiveJob;

  const handleGenerate = () => {
    // Group3 item11: an attached source audio switches submission to
    // `POST /generate/chain` (the distilled A2V fast path), carrying Create's
    // I2V keyframes on the single clip. Otherwise this is the usual T2V/I2V
    // `/generate` call.
    if (form.isA2v) {
      submitChain(form.buildA2vRequest(keyframes.conditioningImages));
      return;
    }
    const request = form.toGenerateRequest();
    submit(
      keyframes.conditioningImages.length > 0
        ? { ...request, conditioning_images: keyframes.conditioningImages }
        : request,
    );
  };

  // Group3 item11: `useGenerationForm` never touches the toast system (keeps it
  // usable from a plain `renderHook`) — it just bumps
  // `audioFramesAdjustedEvent.id` once per wav attach it could auto-adjust
  // `numFrames` from. This effect turns that into a visible toast, keyed on
  // `id` (ported from `ChainedScreen`).
  const lastToastedAudioEventId = useRef<number | null>(null);
  useEffect(() => {
    const event = form.audioFramesAdjustedEvent;
    if (!event || lastToastedAudioEventId.current === event.id) return;
    lastToastedAudioEventId.current = event.id;
    toasts.push({ kind: "success", message: strings.single.sourceAudio.framesAdjustedToast(event.frames, event.durationSec) });
  }, [form.audioFramesAdjustedEvent, strings, toasts]);

  // The Generate button's own disabled/label logic (`.generation-column`,
  // above the ledger). `hasActiveJob` blocks only this button (label
  // `busyButton`); everything else is a genuine "can't submit yet" guard.
  const generateDisabled =
    submitting ||
    hasActiveJob ||
    !form.isValid ||
    keyframes.isUploading ||
    // A source-audio upload still in flight must block submit — until it
    // resolves, `isA2v` is false and the request would wrongly go out as a
    // plain T2V/I2V.
    form.sourceAudio.state.status === "uploading" ||
    // Y1: a reference-video upload still in flight must block submit too. Since
    // 案A made `isICLora` fully dynamic off the live id, a routed #2 whose
    // reference video is still uploading has `isICLora === false` (id null), so
    // the `isICLora && !ready` gate below no longer covers this window — without
    // this the form would submit a reference-less plain T2V. Mirrors the
    // source-audio uploading gate directly above.
    form.referenceVideo.state.status === "uploading" ||
    // IC-LoRA requires a reference video before it can submit
    // (`reference_video_id` is mandatory for that flow).
    (form.isICLora && form.referenceVideo.state.status !== "ready") ||
    // Keyframe timeline rework: an out-of-range keyframe blocks submit until
    // the user resolves it (moves/removes the pin or lengthens the duration).
    hasOverflowKeyframes;
  const generateLabel = submitting
    ? strings.single.generatingButton
    : hasActiveJob
      ? strings.single.busyButton
      : strings.single.generateButton;

  // W7: the imperative "why is Generate disabled?" note shown right under the
  // button. Composes the hook's own `validityReasons` with the four screen-level
  // gates that also feed `generateDisabled` above (each condition here is the
  // SAME one used there): keyframe/audio uploads in flight, an IC-LoRA route
  // whose reference video isn't ready yet, and an out-of-range keyframe.
  // `submitting`/`hasActiveJob` are deliberately NOT reasons — the button's own
  // label already changes for those. `overflowKeyframes` reuses the keyframe
  // out-of-range copy (its old standalone banner below is removed, folded here).
  const generateReasonCodes: string[] = [
    ...form.validityReasons,
    ...(keyframes.isUploading ? ["keyframeUploading"] : []),
    ...(form.sourceAudio.state.status === "uploading" ? ["audioUploading"] : []),
    // Y1: reference-video upload in flight (see the matching `generateDisabled`
    // gate) — the note tells the user to wait rather than leaving Generate
    // disabled with no reason line.
    ...(form.referenceVideo.state.status === "uploading" ? ["referenceUploading"] : []),
    ...(form.isICLora && form.referenceVideo.state.status !== "ready" ? ["referenceNotReady"] : []),
    ...(hasOverflowKeyframes ? ["overflowKeyframes"] : []),
  ];
  const generateReasonMessages: Record<string, string> = {
    promptEmpty: strings.single.generateReasons.promptEmpty,
    dimensionsOffGrid: strings.single.generateReasons.dimensionsOffGrid,
    numFramesOffGrid: strings.single.generateReasons.numFramesOffGrid,
    audioTooShort: strings.single.generateReasons.audioTooShort,
    referenceNeedsLoras: strings.single.generateReasons.referenceNeedsLoras,
    // Both share one line — `GenerateReasonsNote` de-dups the duplicate (R2).
    controlNeedsReference: strings.single.generateReasons.attachReferenceVideo,
    referenceNotReady: strings.single.generateReasons.attachReferenceVideo,
    cropInvalid: strings.single.generateReasons.cropInvalid,
    keyframeUploading: strings.single.generateReasons.keyframeUploading,
    audioUploading: strings.single.generateReasons.audioUploading,
    referenceUploading: strings.single.generateReasons.referenceUploading,
    overflowKeyframes: strings.single.keyframes.outOfRangeGenerateHint,
    // §1-6 拡張 (2026-08-01): a reference-video range cut was requested but the
    // response did not confirm it (old plugin / no ffmpeg), so the stored upload
    // is the WHOLE file — generating would condition on the wrong footage. The
    // hook's own `validityReasons` pushes this code; the line mirrors Chain's
    // `chain.generateReasons.sourceTrimFailed`.
    referenceTrimFailed: strings.single.generateReasons.referenceTrimFailed,
    // NAG (2026-07-28): "non-CFG Negative" enabled with an empty body — the
    // hook's own `validityReasons` already pushes this code (mirrors the
    // backend's 422 for the same combination).
    nagNegativeEmpty: strings.single.generateReasons.nagNegativeEmpty,
  };

  // U4: the batch panel no longer owns width/height/frameRate/seed — it reads
  // them from the Create form (`BatchGenerationValues`). Memoized so the
  // batch hook's callbacks don't rebuild on every unrelated Create re-render.
  const batchGenerationValues = useMemo(
    () => ({
      width: form.values.width,
      height: form.values.height,
      frameRate: form.values.frameRate,
      seed: form.values.seed,
      numFrames: form.values.numFrames,
    }),
    [
      form.values.width,
      form.values.height,
      form.values.frameRate,
      form.values.seed,
      form.values.numFrames,
    ],
  );

  return (
    <>
      <div className="single-layout">
        <GenerationForm
          form={form}
          keyframes={keyframes}
          frameIdxMultiple={frameIdxMultiple}
          gridOffset={gridOffset}
          onDurationChange={shrinkGuard.onDurationChange}
          onDurationCommit={shrinkGuard.onDurationCommit}
          onApplyPreset={shrinkGuard.onApplyPreset}
          disabled={submitting}
          configFallbackWarning={usingFallback}
          resolutionNotes={derived?.notes}
          config={config}
          nativeBridge={nativeBridge}
          autoOpenReference={autoOpenReference}
          autoOpenA2v={autoOpenA2v}
        />
        <div className="generation-column">
          <GenerateButtonBar
            label={generateLabel}
            disabled={generateDisabled}
            onGenerate={handleGenerate}
            hint={form.estimateLabel}
          />
          {/* W7: the out-of-range keyframe hint is now one of these reasons — the
              old standalone `hasOverflowKeyframes` banner was folded in here. */}
          <GenerateReasonsNote reasons={generateReasonCodes} messages={generateReasonMessages} />
          {submitState.phase === "error" && (
            <div className="card card-error">
              <p className="error-code">{submitState.code}</p>
              <p>{submitState.message}</p>
            </div>
          )}
          <JobLedger baseUrl={baseUrl} highlightedJobId={highlightedJobId} nativeBridge={nativeBridge} />
        </div>
      </div>
      {/* webui-B (batch A2V): a collapsed-by-default section below the main
          single-generation layout — see BatchSection.tsx's doc comment for
          why it reuses this screen's `config`/`prompt` rather than owning
          its own config fetch or a separate prompt box. U4: the batch's
          width/height/frameRate/seed now come from this form's values, and
          `icLoraActive` drives the batch's 128-grid warning. */}
      <BatchSection
        config={config}
        prompt={prompt}
        generationValues={batchGenerationValues}
        icLoraActive={form.isICLora}
        keyframes={keyframes}
        nativeBridge={nativeBridge}
        nag={nag}
        acceleration={acceleration}
        /* §1-7 相互ロック 第2段 (2026-07-31): the same job-slot gate the Chain
           screen already hands to Batch i2v-long. */
        hasActiveJob={hasActiveJob}
        /* §3-98 P5: the loaded base model's engine cannot chain, so this
           panel's every job would come back 422. */
        unavailable={batchUnavailable}
      />
      {shrinkGuard.pendingShrink && (
        <KeyframeShrinkModal
          pendingNumFrames={shrinkGuard.pendingShrink.pendingNumFrames}
          overflowCount={shrinkGuard.pendingShrink.overflowIds.length}
          onApply={shrinkGuard.confirmShrink}
          onCancel={shrinkGuard.cancelShrink}
        />
      )}
    </>
  );
}
