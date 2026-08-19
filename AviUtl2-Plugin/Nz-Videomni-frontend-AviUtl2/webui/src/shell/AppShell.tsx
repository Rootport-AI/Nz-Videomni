import { useCallback, useEffect, useRef, useState } from "react";
import { bridge as defaultBridge, TIMELINE_PROJECT_LOADED_EVENT } from "../bridge";
import type { NativeBridge } from "../bridge";
import {
  getChainDirty,
  getPublishedFormValues,
  getPublishedKeyframeCount,
  getPublishedSourceSlots,
  getReservationPhase,
  markJobInserted,
  publishRetakeReservation,
  reconcileFromTimeline,
  reservePlacement,
} from "../timeline/provisionalReservation";
import { recordSourceLocation } from "../timeline/sourceLocationMap";
import { formatMenuGuardNote, guardMenuSelection } from "../timeline/menuSelection";
import { END_SOURCE_CONTEXT_FRAMES, tailAlignedStartFrame } from "../timeline/tailAlign";
import { computeTargetNumFrames } from "../timeline/deriveDuration";
import { resolvePrefillSeed } from "../timeline/prefillSeed";
import { dispatchCreateCommand } from "../timeline/createLiveCommands";
import { fileNameFromPath } from "../modes/single/keyframeUtils";
import { LanguageProvider, useStrings } from "../i18n/LanguageContext";
import { JobsProvider, useJobsContext } from "../jobs/JobsContext";
import { hasActiveJob } from "../jobs/useJobsPoll";
import { downloadAndInsertVideo } from "../jobs/downloadAndInsert";
import type { ApiClient } from "../api/client";
import type { JobResponse } from "../api/types";
import type { ControlLoraSelection } from "../lora/controlLoras";
import { parseLoraPrompt, stripLoraTagsByName } from "../lora/loraTags";
import { ChainedScreen } from "../modes/chained/ChainedScreen";
import { EditScreen } from "../modes/edit/EditScreen";
import { SingleScreen } from "../modes/single/SingleScreen";
import { FALLBACK_APP_CONFIG, MIN_NUM_FRAMES } from "../modes/single/defaultConfig";
import { StatusHeader } from "../modes/single/StatusHeader";
import { useBaseUrl } from "../modes/single/useBaseUrl";
import { useConfig } from "../modes/single/useConfig";
import { useServerStatus } from "../modes/single/useServerStatus";
import { InventoryScreen } from "../modes/inventory/InventoryScreen";
import { useLoras } from "../modes/inventory/useLoras";
import type { GenerationPrefill } from "../timeline/generationPrefill";
import { useMenuRouter } from "../timeline/useMenuRouter";
import type { RoutedMenuCommand } from "../timeline/useMenuRouter";
import { ConfirmDialog } from "./ConfirmDialog";
import { ModeTabs } from "./ModeTabs";
import { NagAccordion } from "./NagAccordion";
import { NoteArea, ShowNoteProvider } from "./NoteArea";
import type { AppNote, ShowNote } from "./NoteArea";
import { PromptBar } from "./PromptBar";
import { SettingsPanel } from "./SettingsPanel";
import { PrefillPolicyProvider, usePrefillPolicy } from "./PrefillPolicyContext";
import { ThemeProvider } from "./ThemeContext";
import { ToastProvider, useToasts } from "./ToastContext";
import { Toasts } from "./Toasts";
import { blockSwapPrefetchAvailability, sageAvailability } from "./accelerationSettings";
import { useAccelerationSettings } from "./useAccelerationSettings";
import { useControlLoraNames, useDepthLoraNames, useReferenceDownscaleFactors } from "./useControlLoraNames";
import { useNagSettings } from "./useNagSettings";
import "./AppShell.css";

export type AppMode = "single" | "chained" | "edit" | "inventory";

/** W3 (⬇ "insert the latest generation result here"): pick the most recently
 * completed job from the ledger. Sorts `completed_at` DESCENDING (latest first);
 * a `null` `completed_at` is treated as the OLDEST (sorted to the tail), and
 * ties (equal timestamps, or two nulls) break deterministically on `job_id` so
 * the "latest" choice is stable across re-polls. Pure. */
export function latestCompletedJob(jobs: readonly JobResponse[]): JobResponse | null {
  const completed = jobs.filter((job) => job.status === "completed");
  if (completed.length === 0) return null;
  const sorted = [...completed].sort((a, b) => {
    const ca = a.completed_at;
    const cb = b.completed_at;
    if (ca !== cb) {
      // A null timestamp is the oldest -> pushed to the descending tail.
      if (ca === null) return 1;
      if (cb === null) return -1;
      return ca < cb ? 1 : -1; // ISO 8601 strings sort chronologically.
    }
    // Deterministic tiebreak on the (unique) job id.
    return a.job_id < b.job_id ? 1 : a.job_id > b.job_id ? -1 : 0;
  });
  return sorted[0] ?? null;
}

/** X2(b): the ledger's terminal (completed/failed/cancelled) job ids, handed to
 * `reconcileFromTimeline` so a leftover `NzVideomni#<jobId>` tag from an
 * already-settled job can't hold the reservation seat `generating` (the
 * permanent post-422 block). An empty ledger (startup) yields an empty set —
 * the safe side: nothing is masked, so a real running job is still protected.
 * Pure. */
export function terminalJobIds(jobs: readonly JobResponse[]): ReadonlySet<string> {
  const ids = new Set<string>();
  for (const job of jobs) {
    if (job.status === "completed" || job.status === "failed" || job.status === "cancelled") {
      ids.add(job.job_id);
    }
  }
  return ids;
}

export interface AppShellProps {
  /** Test/integration seam: inject a mock `NativeBridge` so the whole shell
   * (menu router + every deps-injectable Screen) talks to it instead of the
   * app-wide singleton. Production (`App.tsx`) omits it — the singleton bridge
   * is used exactly as before. */
  nativeBridge?: NativeBridge | undefined;
  /** Test/integration seam: inject a mock `ApiClient` for the job-ledger poll
   * (`JobsProvider`), which otherwise binds to the app-wide singleton client
   * (its `GET /jobs` does NOT flow through the injected `nativeBridge`). Lets a
   * test seed a deterministic job list — e.g. the W2 ⬇ quick-insert, which
   * resolves the right-clicked reservation object's job out of that ledger.
   * Production (`App.tsx`) omits it — the singleton client is used as before. */
  apiClient?: ApiClient | undefined;
}

/** The app shell (task brief §1/§2/§6): header (title, server status, mode
 * tabs) -> shared prompt bar -> single-column mode body. U-R1: the always-on
 * right-hand job rail is gone — the job ledger now lives inside Create/Chain's
 * generation panel, so `.app-body` is one column. Wraps everything in the
 * theme/language/toast/jobs providers so any mode screen and the ledger can
 * share the same theme/language selection, job-list poll and toast queue —
 * `ThemeProvider` is
 * outermost (N8: more fundamental than language, and neither depends on the
 * other), then `LanguageProvider`, since `JobsProvider` itself renders
 * translated toast copy (`jobs/JobsContext.tsx`). Nested here (rather than in
 * `App.tsx`, which just renders `<AppShell/>`) so a direct `<AppShell/>`
 * render — as `App.prefill.test.tsx` does — still gets both providers. */
export function AppShell({ nativeBridge, apiClient }: AppShellProps = {}) {
  return (
    <ThemeProvider>
      <LanguageProvider>
        <PrefillPolicyProvider>
          <ToastProvider>
            <JobsProvider nativeBridge={nativeBridge} {...(apiClient ? { apiClient } : {})}>
              <AppShellBody nativeBridge={nativeBridge} />
            </JobsProvider>
          </ToastProvider>
        </PrefillPolicyProvider>
      </LanguageProvider>
    </ThemeProvider>
  );
}

function AppShellBody({ nativeBridge }: AppShellProps) {
  const strings = useStrings();
  // W2/W3 (⬇ quick-insert): the job ledger, mirrored into a ref so the stable
  // `handleRoute` callback can read the latest list without taking `jobs` as a
  // dependency (which would re-create the callback on every 2s poll). Updated
  // every render. `AppShellBody` renders inside `JobsProvider` (see `AppShell`),
  // so `useJobsContext` is in scope here.
  const { jobs } = useJobsContext();
  const jobsRef = useRef<JobResponse[]>(jobs);
  jobsRef.current = jobs;
  const { state: serverStatus, retry } = useServerStatus();
  const baseUrl = useBaseUrl();
  const [mode, setMode] = useState<AppMode>("single");
  // W8 (owner requirement 2026-07-21 #1): the right-click prefill resolution/fps
  // policy (Settings), read here so `handleRoute`'s material-item reservation
  // length comes from the SAME `resolvePrefillSeed` inputs the remounted
  // Create/Chain form seeds from — the placed provisional's ribbon then matches
  // the form's own initial DURATION/fps rather than the config default.
  const { sizePolicy, fpsPolicy } = usePrefillPolicy();
  const [prompt, setPrompt] = useState("");
  const [highlightedJobId, setHighlightedJobId] = useState<string | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  // Tool-version dropdown (2026-08-19, mock): replaces the old static
  // "Nz-Videomni" header title. "LTX 2.5" is not a real, backed model yet —
  // selecting it silently snaps back to "LTX 2.3" (no toast, no persistence;
  // see the `onChange` below and `strings.toolVersion`'s own doc comment).
  const [toolVersion, setToolVersion] = useState<"LTX 2.3" | "LTX 2.5">("LTX 2.3");

  // Shared note area (RIGHTCLICK_REDESIGN_SPEC.md §6): a single persistent slot
  // above the operation panel. One note at a time — `showNote` REPLACES whatever
  // is shown (§6-1). Owned here so the single `NoteArea` renders in its fixed
  // tab-bar-level slot; `showNote` is shared down to the screens via
  // `ShowNoteProvider` for I6/I7 (mismatch/receipt notes).
  const [note, setNote] = useState<AppNote | null>(null);
  const showNote = useCallback<ShowNote>((kind, message) => setNote({ kind, message }), []);
  const dismissNote = useCallback(() => setNote(null), []);
  // Right-click routing -> prefill. `pendingIntent` is the payload the target
  // mode consumes once.
  //
  // Tab-state persistence (2026-07-18): all three mode screens are now mounted
  // at once (`.app-mode-body` below) and merely hidden when inactive, so a tab
  // switch no longer unmounts the form/preview/preset state. That makes the
  // Screen `key` the ONLY remaining remount lever, and it must be driven by
  // something that changes *only* when we actually want a remount.
  //
  // `remountTokens` is a per-mode monotonic counter, bumped in `handleRoute`
  // when a right-click routes an intent AT that mode — EXCEPT the two
  // keep-panel-state intents ✨ text-to-video / 📷 image-from-frame (D3 owner
  // decision), which deliberately skip the bump so Create is not remounted. The
  // Screen `key` is `${mode}-${remountTokens[mode]}`, so:
  //  - a fresh intent (even to the mode we're already on) bumps the counter ->
  //    the target Screen remounts and re-applies the new prefill (the frozen
  //    right-click behaviour), and
  //  - a manual tab switch, which nulls `pendingIntent`, leaves every counter
  //    untouched -> no Screen remounts, so edited form/preview state survives.
  //
  // The earlier design keyed off a `pendingIntent`-derived token, which under
  // always-mounted screens flipped the key back to a "base" value the moment
  // `pendingIntent` expired — silently remounting (and wiping) the very screen
  // the user was editing (adversarial review M-1). The counter fixes that: it
  // never decreases, so nothing an expiring `pendingIntent` does can change it.
  const [pendingIntent, setPendingIntent] = useState<GenerationPrefill | null>(null);
  const [remountTokens, setRemountTokens] = useState<Record<AppMode, number>>({
    single: 0,
    chained: 0,
    // 2026-08-09 W0（共通スパイン）: `MenuTargetMode` に "edit" が入り、Edit も
    // 右クリックの転送先になった（outpaintVideo / retakeRange）。この枠は Step 6
    // の `setRemountTokens(prev => ({...prev, [target]: prev[target] + 1}))` が
    // target === "edit" のときそのまま加算するので、専用の分岐は無い。
    edit: 0,
    inventory: 0,
  });

  // IC-LoRA UI redesign (2026-07-17, 第5波): the control-LoRA panel selection
  // is owned HERE, not by Create's own form state, for the same reason the
  // prompt itself lives here — it must survive a tab switch away from Create
  // and back (M2 of the plan's adversarial review: a form-local `controlLora`
  // would silently reset on every Create remount, e.g. a right-click prefill's
  // `key` bump). `useLoras`/`useConfig` are each also called independently by
  // Create/Chain/Library (no shared cache) — this is one more call in that
  // same pre-existing pattern, needed here only to resolve `controlLoraNames`
  // for the auto-migration effect below.
  const [controlLora, setControlLora] = useState<ControlLoraSelection | null>(null);
  const configState = useConfig();
  const config = configState.status === "ready" ? configState.config : FALLBACK_APP_CONFIG;
  const lorasState = useLoras();
  // M-1 (post-implementation adversarial review, 2026-07-17): stability of
  // this Set is load-bearing (it's a dependency of the auto-migration effect
  // below) — extracted into its own hook so that stability is independently
  // unit-tested (`useControlLoraNames.test.ts`) rather than only implied by
  // this call site. See that hook's own doc comment for why a naive
  // `useMemo(..., [config, lorasState])` would be unstable.
  const controlLoraNames = useControlLoraNames(config, lorasState);
  // §1-15 (chain reference video): the depth-preprocess subset of those
  // adapters, memoized against the same stable pieces. Chain-only — it feeds
  // `useChainForm`'s `depthChainUnsupported` gate, which pre-empts the server's
  // 422 for a depth adapter on a multi-clip chain. Empty (gate off) on any
  // server whose `GET /loras` does not publish `preprocess` yet.
  const depthLoraNames = useDepthLoraNames(lorasState);
  // §1-15, plan F5/F6: control-LoRA name -> `reference_downscale_factor`, off
  // the same stable pieces — feeds `ChainedScreen`'s stage-1 comfort-budget
  // warning banner (`shell/tokenBudget.ts`'s `chainStage1Tokens`' `refScale`).
  const referenceDownscaleFactors = useReferenceDownscaleFactors(lorasState);

  // NAG (2026-07-28): the shared "Negative Prompt" accordion's settings live
  // HERE for the identical reason `prompt`/`controlLora` do — a single
  // `AppShell`-level object survives a tab switch and a right-click remount's
  // `key` bump, so Create/Chain/Batch all keep reading the same state instead
  // of each mode screen silently resetting its own copy.
  const nagControls = useNagSettings();

  // Acceleration (2026-07-31, backend §43): same single-owner arrangement as
  // `nagControls` right above — one call, one object, handed to the Settings
  // panel (the only editor) and to Create/Chain/Batch (readers, via each
  // screen's own props). Called HERE and nowhere else so a right-click remount
  // or a tab switch can never leave two screens submitting with different
  // attention backends.
  // §1-10 (2026-08-03): the hook also needs the block-swap-prefetch capability
  // flag off the 10s `/status` poll above, so keep-resident can be folded down
  // to its effective value for readers. Three-valued (see
  // `blockSwapPrefetchAvailability`) — `null` never disables anything.
  // Smart comfort marker (2026-08-18): `statusBody` is pulled out so
  // `sageAvailability` can read the SAME `/status` body below, one poll
  // feeding both capability flags — see the `SingleScreen` prop for why the
  // marker needs sage's own availability, not just prefetch's.
  const statusBody = serverStatus.kind === "online" || serverStatus.kind === "busy" ? serverStatus.status : null;
  const accelerationControls = useAccelerationSettings(blockSwapPrefetchAvailability(statusBody));

  const toasts = useToasts();
  // IME composition guard for the auto-migration effect below: a control-LoRA
  // tag typed mid-composition (e.g. Japanese IME) must not be migrated out
  // from under the still-uncommitted input. A ref (not state) because it must
  // never itself trigger a re-render/re-run — only gates what the `prompt`
  // effect below does once IT re-runs for some other reason.
  const isComposingRef = useRef(false);
  const handlePromptCompositionStart = useCallback(() => {
    isComposingRef.current = true;
  }, []);
  const handlePromptCompositionEnd = useCallback(() => {
    isComposingRef.current = false;
  }, []);

  // Auto-migration (owner decision #2): a hand-typed control-LoRA tag in the
  // prompt moves into the `controlLora` panel state instead of staying a
  // prompt tag. `mode === "single"` only — Chain has no panel to migrate
  // into, so a control tag typed there is left for `useChainForm`'s own gate
  // (Step 5) to warn about and block on instead. Idempotent: once the
  // matching tag(s) are stripped, `tags`/`matched` are empty on the next run,
  // so this is safe under StrictMode's double-invoke and safe to depend on
  // `prompt` directly (no separate "already migrated" flag needed). The last
  // matched tag wins (mirrors a user re-typing/correcting their choice); any
  // earlier ones are discarded along with it, which is what the "multiple"
  // toast variant calls out.
  //
  // Known, accepted residual gaps (post-implementation adversarial review,
  // 2026-07-17 — neither is worth the added complexity to close):
  // (a) a SECOND control tag that finishes composing in the snapshot-to-commit
  //     window of this same effect run is discarded silently (no "multiple"
  //     toast for it) rather than folded in — a one-render-frame race that's
  //     practically unreachable outside of scripted/automated input.
  // (b) under StrictMode's double-invoke, a control tag already present in the
  //     prompt at MOUNT time (not typed afterward) would toast twice — not
  //     reachable today since nothing that seeds an initial prompt (the
  //     right-click prefill flow) ever writes a lora tag into it.
  useEffect(() => {
    if (mode !== "single") return;
    if (isComposingRef.current) return;
    if (controlLoraNames.size === 0) return;
    const { tags } = parseLoraPrompt(prompt);
    const matched = tags.filter((tag) => tag.valid && controlLoraNames.has(tag.name));
    if (matched.length === 0) return;
    const chosen = matched[matched.length - 1]!;
    setControlLora({ name: chosen.name, strength: chosen.strength });
    // Functional setter (M1 of the plan's adversarial review): rewrites
    // whatever the LATEST prompt is at commit time, not the `prompt` this
    // effect closed over — otherwise a keystroke landing between this
    // effect's schedule and its commit would be silently discarded.
    setPrompt((prev) => stripLoraTagsByName(prev, controlLoraNames));
    toasts.push({
      kind: "success",
      message:
        matched.length > 1
          ? strings.single.referenceVideo.controlLoraMigratedMultiple(chosen.name)
          : strings.single.referenceVideo.controlLoraMigratedToast(chosen.name),
    });
  }, [prompt, mode, controlLoraNames, toasts, strings]);

  // Toast click: highlight the job in the ledger. Create, Chain AND Edit each
  // mount one (Edit gained its own on 2026-08-09), so the only mode left with
  // no ledger to highlight in is Inventory — from there, switch to Create
  // first; otherwise the highlight would target a ledger that isn't rendered
  // and the toast click would look like it did nothing.
  const handleSelectJob = useCallback((jobId: string) => {
    setMode((current) => (current === "inventory" ? "single" : current));
    setHighlightedJobId(jobId);
  }, []);

  // I8 §3-4 (※): the Chain-discard confirm gate. `handleRoute` is async, so it
  // awaits a boolean from this modal for a #1/#6 route that would remount (and
  // discard) a dirty Chain form. The Promise's `resolve` is parked in a ref
  // (not state — a function-valued state confuses the setter's updater/value
  // overload) while a plain boolean flag drives whether the dialog renders.
  const chainDiscardResolveRef = useRef<((ok: boolean) => void) | null>(null);
  const [chainDiscardOpen, setChainDiscardOpen] = useState(false);
  const confirmChainDiscard = useCallback(
    () =>
      new Promise<boolean>((resolve) => {
        chainDiscardResolveRef.current = resolve;
        setChainDiscardOpen(true);
      }),
    [],
  );
  const answerChainDiscard = useCallback((ok: boolean) => {
    setChainDiscardOpen(false);
    const resolve = chainDiscardResolveRef.current;
    chainDiscardResolveRef.current = null;
    resolve?.(ok);
  }, []);

  // Unified right-click handler (I7/I11, RIGHTCLICK_REDESIGN_SPEC.md §4/§5-6/§5-9).
  // A single fixed order runs for EVERY routed command:
  //   1. selection guard (§4)  2. on-demand reconcile (§6-b)  3. #8 appendText
  //   (early-return prompt append + placement B, §3-4 #8)  4. #5 addImageKeyframe
  //   (live keyframe append, conditional placement A on the append-time count,
  //   §3-4 #5)  4b. #11 addCurrentFrameAsKeyframe (W5: the capture twin of #5 —
  //   live keyframe append, conditional placement C at the cursor)  5.
  //   reservation-seat busy-guard (§5-6)  6. prefill delivery (§3 remount for
  //   material items + #12, keep-state for ✨/📷/#11)  7. reservation length
  //   source (§5-1/D3)  8. provisional placement/move (§5-9).
  const handleRoute = useCallback(async (command: RoutedMenuCommand) => {
    const bridge = nativeBridge ?? defaultBridge;
    const route = command.route;
    const action = route.action;
    const target = route.targetMode;
    const intent = route.intent;

    // ── Step 1: selection guard (§4). Runs FIRST for every command: multiple-
    // selection -> undeterminable kind -> required-kind mismatch (+ #7 near-
    // neighbor) -> #1/#2 length guard. On any failure, show the guidance note
    // and do NOTHING else — no tab switch, no remount, no reservation, no
    // prefill (§4 "破壊的な動作をしない"). Layer-menu items (#9/#10,
    // requiredKind null) always pass the guard.
    const guard = guardMenuSelection(route, command.selection);
    if (!guard.ok) {
      showNote("warning", formatMenuGuardNote(guard.note, strings.notes));
      return;
    }

    // ── W2 (⬇ insertProvisionalResult): the right-clicked object is a Nz-Videomni
    // provisional reservation. When its backing job is already completed, run
    // the SAME replace-insert the panel's 🎞 button does (`downloadAndInsertVideo`
    // with no `plainInsertAt` → `insertForJob`, which replaces the marker in
    // place). This is an early-return channel: no mode switch, no reservation, no
    // provisional placement — feedback is a toast (not a note), matching the job
    // ledger's own settle toasts. The job is resolved out of the mirrored ledger
    // by the object's `NzVideomni#<jobId>` name (the `menuInvoked` snapshot already
    // carries `objectName`, so `needsSelection` is false and nothing is re-queried).
    if (action === "insertProvisionalResult") {
      const objectName = command.selection.selected[0]?.objectName ?? null;
      const match = objectName ? /^NzVideomni#(.+)$/.exec(objectName) : null;
      if (match === null) {
        // (a) not a Nz-Videomni reservation object.
        toasts.push({ kind: "warning", message: strings.menuInsert.notProvisional });
        return;
      }
      const jobId = match[1]!;
      const job = jobsRef.current.find((j) => j.job_id === jobId);
      if (job === undefined) {
        // (b) the reservation's job is not (or no longer) in the ledger — e.g. a
        // never-generated `pending-…` reservation, or a job cleared server-side.
        toasts.push({ kind: "warning", message: strings.menuInsert.jobNotFound });
        return;
      }
      if (job.status === "completed") {
        // (c) done — insert (replace) exactly like the 🎞 button.
        try {
          const result = await downloadAndInsertVideo(bridge, jobId);
          // Y3 (W2 shared ✅, 非対称案): publish this job's insert so its
          // JobCard(s) flip 🎞→✅ in sync — W2's insert happens HERE in AppShell,
          // not in a card, so without this the ledger/History cards for the
          // just-inserted job would stay 🎞. Carry `filePath` so a ✅→🎞
          // re-insert reuses the download (avoids "Could not open destination
          // file"). ONLY W2 publishes — the manual per-card 🎞 stays
          // instance-local (unchanged). Marked ONLY on success (never in catch).
          markJobInserted(jobId, result.filePath);
          toasts.push({ kind: "success", message: strings.menuInsert.inserted });
        } catch {
          toasts.push({ kind: "warning", message: strings.menuInsert.insertFailed });
        }
      } else if (job.status === "queued" || job.status === "running") {
        // (d) still generating.
        toasts.push({ kind: "warning", message: strings.menuInsert.notReady });
      } else {
        // (e) failed / cancelled — no result to insert.
        toasts.push({ kind: "warning", message: strings.menuInsert.jobFailed });
      }
      return;
    }

    // ── W3 (⬇ insertLatestResultHere): a layer-menu quick-insert. Drop the most
    // recently completed generation result at the right-click position as a
    // PLAIN insert (`downloadAndInsertVideo`'s `plainInsertAt`), which bypasses
    // the replace-RPC so any provisional at the cursor is left untouched. Also an
    // early-return channel (no mode switch, no reservation): toast feedback only.
    if (action === "insertLatestResultHere") {
      const latest = latestCompletedJob(jobsRef.current);
      if (latest === null) {
        toasts.push({ kind: "warning", message: strings.menuInsert.noCompleted });
        return;
      }
      const { cursorLayer, cursorFrame } = command.selection;
      // Guard against an unresolved cursor (native reports -1 for an unknown
      // layer/frame). Inserting position-less could drop the clip somewhere
      // unexpected, so abort with a note instead of mis-placing it.
      if (cursorLayer < 0 || cursorFrame < 0) {
        toasts.push({ kind: "warning", message: strings.menuInsert.noCursor });
        return;
      }
      try {
        await downloadAndInsertVideo(bridge, latest.job_id, undefined, {
          plainInsertAt: { layer: cursorLayer, frame: cursorFrame },
        });
        toasts.push({ kind: "success", message: strings.menuInsert.insertedLatest });
      } catch {
        toasts.push({ kind: "warning", message: strings.menuInsert.insertLatestFailed });
      }
      return;
    }

    // ── Step 2: on-demand re-sync (§6-b insurance). Rebuild the seat from the
    // LIVE timeline BEFORE any busy-guard so the decision is based on the
    // timeline 実体 rather than stale in-memory state — now ahead of the #5/#8
    // channels too, since both consult the reservation seat (§5-6). Closes the
    // reload gap even when the `projectLoaded` push (subscribed below) never
    // fired. Swallow any rejection so a reconcile hiccup never blocks the
    // right-click; `reconcileFromTimeline` resolves an RPC failure to `idle`
    // internally, so on failure we simply fall through.
    try {
      // X2(b): pass the ledger's terminal jobs so a stale `NzVideomni#<jobId>` tag
      // from an already-settled job doesn't get resurrected to `generating` (the
      // permanent post-422 block) — this reconcile runs BEFORE the busy-guard.
      await reconcileFromTimeline(bridge, terminalJobIds(jobsRef.current));
    } catch {
      // never block the right-click on a reconcile hiccup
    }

    // Shared provisional reservation for every generation-origin item (§5-9):
    // `reservePlacement` inserts a fresh placeholder when the seat is idle, or
    // MOVES the existing unbound reservation (§5-6 用途b). A normal insert/move
    // shows NO note — the provisional appearing on the timeline is the feedback
    // (§6 思想); only the layer_max+1 fallback gets one, layer shown +1 (§6-3).
    // Fire-and-forget so it never blocks the tab switch / prefill already done.
    const placeProvisional = (opts: {
      // W0: `"D"` (Retake, 選択範囲と頭を揃える) rides the same helper —
      // `reservePlacement`/`placementParams` maps it onto native's `"B"`.
      // 素材（末尾）v2: so does `"E"` (末尾合わせ), by the same mapping.
      placement: "A" | "B" | "C" | "D" | "E";
      material?: { layer: number; frameStart: number; frameEnd: number } | undefined;
      cursor?: { layer: number; frame: number } | undefined;
      numFrames: number;
      genFps: number;
      /** I6 Join: called with the fresh reservation's `pending-…` id once the
       * placeholder is on the timeline, so a v2v continuation can record its
       * source object's location keyed by that id (`sourceLocationMap`). */
      onReserved?: ((pendingId: string) => void) | undefined;
    }) => {
      void (async () => {
        try {
          // Stage 1 (§5-5 段階1): a fresh reservation / move always reads
          // "AI video will be placed here…" — fixed ASCII copy, NOT the prompt
          // (emoji-free so it never tofus in the AviUtl2 default font). The prompt
          // head only appears at stage 2 (Generate-time bind).
          const result = await reservePlacement(bridge, {
            placement: opts.placement,
            material: opts.material,
            cursor: opts.cursor,
            numFrames: opts.numFrames,
            genFps: opts.genFps,
            textPrefix: strings.provisional.reservedPrefix,
            displayText: strings.provisional.reservedBody,
          });
          opts.onReserved?.(result.pendingId);
          if (result.usedFallback) {
            showNote("warning", strings.notes.insertedOnFrontmostLayer(result.placedLayer + 1));
          }
        } catch {
          // A reservation-insert failure note is out of scope here; swallow so
          // it never breaks the mode switch / prefill already applied.
        }
      })();
    };

    // The keep-panel-state length source (§5-1/D3): #8/#5/✨/📷 do not remount
    // Create, so their placeholder length must match the LIVE form the user will
    // generate from — the published DURATION/fps, falling back to config defaults
    // before Create ever publishes.
    const publishedForKeepState = getPublishedFormValues();
    const keepStateNumFrames = publishedForKeepState?.numFrames ?? config.generation_defaults.num_frames;
    const keepStateGenFps = publishedForKeepState?.frameRate ?? config.generation_defaults.frame_rate;

    // W8 (#9 ✨ / #10 📷): the two keep-panel-state cursor items live-SET the
    // Create form's DURATION to the resolution's comfort ceiling — resolved ONCE
    // here from the live form's published width/height (these items never remount
    // Create, so there is no initial-seed path). The same value is used for both
    // the `setDuration` live command AND the reservation length below, so the
    // placeholder and the form can never diverge. `null` — nothing published yet,
    // or an unresolvable ceiling — means neither the form nor the reservation is
    // rewritten: both stay at the current published DURATION.
    const cursorItemCeilingNumFrames =
      (intent === "text-to-video" || intent === "image-from-frame") && publishedForKeepState
        ? computeTargetNumFrames({
            policy: "comfortCeiling",
            width: publishedForKeepState.width,
            height: publishedForKeepState.height,
            spillFreeFrames: config.limits.spill_free_frames,
            minNumFrames: MIN_NUM_FRAMES,
            maxNumFrames: config.limits.max_num_frames,
          })
        : null;

    // ── Step 3: #8 appendText (§3-4 #8 / §5-9 system B). Early-return channel:
    // append the text object's body to the shared prompt, then reserve a
    // provisional head-aligned with the text object. No tab switch, no remount.
    // Order is the strict §5-6 reading: busy-guard FIRST (a ⏳生成中 seat refuses
    // even the append), then empty-body guard (no append, no reservation), then
    // append + receipt + placement.
    if (action === "appendText") {
      if (getReservationPhase() === "generating") {
        showNote("warning", strings.notes.reservationBusy);
        return;
      }
      const text = command.selection.selected[0]?.textContent ?? "";
      if (text.length === 0) {
        // §3-4 #8 空本文: nothing to append, and no provisional either.
        showNote("warning", strings.notes.promptTextEmpty);
        return;
      }
      // §3-4 #8 区切り: a single newline between the existing prompt and the
      // appended text; no leading newline when the prompt was empty. No length cap.
      setPrompt((prev) => (prev ? `${prev}\n${text}` : text));
      // §3-4 #8 受領ノート: shown because the change is otherwise easy to miss.
      // The ellipsis is added ONLY when the body ran past the 20-char head.
      showNote("info", strings.notes.appendedToPrompt(text.slice(0, 20), text.length > 20));
      // §5-9 system B: head-aligned with the text object on layer_max+1.
      const item = command.selection.selected[0]!;
      placeProvisional({
        placement: "B",
        material: { layer: item.layer, frameStart: item.frameStart, frameEnd: item.frameEnd },
        numFrames: keepStateNumFrames,
        genFps: keepStateGenFps,
      });
      return;
    }

    // ── Step 4: #5 addImageKeyframe (§3-4 #5 branch / §5-9). Live-append channel
    // (no remount): the branch turns on the LIVE keyframe count Create publishes
    // (`getPublishedKeyframeCount`), captured BEFORE the append the live command
    // triggers — this append 直前 count is exactly what the spec keys on.
    //  - count 0 (becomes the 1st keyframe) = a new generation origin: busy-guard,
    //    then live-append + switch + placement A off the image (§5-9 A).
    //  - count > 0 (2枚目以降) = a plain append to an existing plan: the seat is
    //    never touched (no guard, no placement) — just live-append + switch.
    if (action === "addImageKeyframe") {
      const item = command.selection.selected[0]!;
      const filePath = item.filePath;
      const fileName = filePath ? fileNameFromPath(filePath) : "";
      const becomesFirstKeyframe = getPublishedKeyframeCount() === 0;
      if (becomesFirstKeyframe) {
        if (getReservationPhase() === "generating") {
          showNote("warning", strings.notes.reservationBusy);
          return;
        }
        dispatchCreateCommand({ type: "addImageKeyframe", filePath, fileName });
        setMode("single");
        placeProvisional({
          placement: "A",
          material: { layer: item.layer, frameStart: item.frameStart, frameEnd: item.frameEnd },
          numFrames: keepStateNumFrames,
          genFps: keepStateGenFps,
        });
      } else {
        dispatchCreateCommand({ type: "addImageKeyframe", filePath, fileName });
        setMode("single");
      }
      return;
    }

    // ── Step 4b: #11 addCurrentFrameAsKeyframe (W5 §5-9). The capture twin of #5
    // (addImageKeyframe): a live keyframe append that never remounts Create, but
    // the source is the CURRENT timeline frame (captured on the Create side via
    // the `addCaptureAsKeyframe` live command) rather than a right-clicked image.
    // Same count-branched placement as #5, but system C (the layer cursor) since
    // this is a layer-menu item with no selected object:
    //  - count 0 (becomes the 1st keyframe) = a new generation origin: busy-guard,
    //    then live-append + switch + placement C at the cursor.
    //  - count > 0 (2枚目以降) = a plain append to an existing plan: the seat is
    //    never touched (no guard, no placement) — just live-append + switch.
    if (action === "addCurrentFrameAsKeyframe") {
      const becomesFirstKeyframe = getPublishedKeyframeCount() === 0;
      if (becomesFirstKeyframe) {
        if (getReservationPhase() === "generating") {
          showNote("warning", strings.notes.reservationBusy);
          return;
        }
        dispatchCreateCommand({ type: "addCaptureAsKeyframe" });
        setMode("single");
        // System C: reserve at the right-click cursor. numFrames/genFps come from
        // the LIVE published form (this keeps the panel state, no remount), same
        // source #9/#10's placement-C reservation uses.
        placeProvisional({
          placement: "C",
          cursor: { layer: command.selection.cursorLayer, frame: command.selection.cursorFrame },
          numFrames: keepStateNumFrames,
          genFps: keepStateGenFps,
        });
      } else {
        dispatchCreateCommand({ type: "addCaptureAsKeyframe" });
        setMode("single");
      }
      return;
    }

    // ── Step 4c: §1-17 Retake の server-busy ガード（オーナー目視 2026-08-10 ①）。
    // `retakeRange` **だけ**に効かせる。Step 5 の共通ガード（予約席）に足すと、
    // 素材系/カーソル系の 9 起点まで一緒に塞がってしまう —— あちらが見ているのは
    // 「席が ⏳生成中 で埋まっているか」であって、「サーバが忙しいか」ではない。
    //
    // 判定は `jobsRef`（毎描画更新）越し。`handleRoute` は `useCallback` で
    // `jobs` を依存に取っていない（2 秒ポーリングのたびに作り直さないため）ので、
    // context を直読みすると初回の `false` に凍りつく。述語 `hasActiveJob` は
    // `jobs/useJobsPoll.ts` から借りたもので、Retake パネルの Generate ボタンが
    // 使っているのと同じ 1 本。
    if (action === "retakeRange" && hasActiveJob(jobsRef.current)) {
      showNote("warning", strings.notes.reservationBusy);
      return;
    }

    // Everything past here is a generation-origin item with placement A/B/C
    // (#1/#2/#3/#4/#6/#7 material, #9/#10/#12 cursor).
    //
    // ── Step 5: reservation-seat busy-guard (§5-6). A ⏳生成中 (a job actually in
    // flight) holds the single seat, so a new generation reservation is refused:
    // note only — no prefill, no reservation, no tab switch.
    if (getReservationPhase() === "generating") {
      showNote("warning", strings.notes.reservationBusy);
      return;
    }

    // ── Step 5.5: Chain-discard confirm (§3-4 ※). #1/#6/#12 land on Chain and
    // remount its form, discarding any in-progress chain build. Prompt ONLY when
    // the live Chain form is actually dirty (a pristine/default Chain loads
    // uninterrupted). Cancel aborts EVERYTHING that follows — no prefill, no
    // reservation, no tab switch — so this sits before Step 6. Only #1/#6/#12
    // target Chain, so `target === "chained"` is the exact gate.
    if (target === "chained" && getChainDirty()) {
      const confirmed = await confirmChainDiscard();
      if (!confirmed) return;
    }

    // ── Step 6: prefill delivery. Material items (#1/#2/#3/#4/#6/#7) remount the
    // target Create/Chain form and re-apply the fresh selection (pendingIntent +
    // per-mode remount token, so the other always-mounted screens keep state).
    // The two keep-panel-state cursor items ✨/📷 (D3) skip both so their edited
    // form survives; they only switch mode and reserve. W5 #11
    // add-current-frame-keyframe is likewise a no-remount item (it lands via its
    // own early-return block above, mirroring #5, before this line is reached —
    // this keeps the keep-panel-state set an accurate description of every
    // no-remount intent). #12 current-frame-to-clip-chain is NOT here: it is a
    // Chain-remount item (like #6).
    const keepPanelState =
      intent === "text-to-video" ||
      intent === "image-from-frame" ||
      intent === "add-current-frame-keyframe";
    if (!keepPanelState) {
      // W5 (反対スロット保持): a #2/#3/#7 remount resets the whole Create form, but
      // the OPPOSITE source slot survives — #2 (`reference-video`) keeps the
      // existing audio; #3/#7 (`video-audio-to-video`/`audio-to-video`) keep the
      // existing reference video + strengths. Read the live-published slots
      // synchronously here (they still reflect the currently-mounted Create, the
      // one about to remount) and load only the relevant, ready side onto the
      // pending intent's `carryOver`. Every other intent — and an unpublished/
      // not-ready slot — carries nothing (identical to the pre-W5 behaviour).
      const slots = getPublishedSourceSlots();
      let carryOver: GenerationPrefill["carryOver"];
      if (intent === "reference-video") {
        if (slots.sourceAudio) carryOver = { sourceAudio: slots.sourceAudio };
      } else if (intent === "video-audio-to-video" || intent === "audio-to-video") {
        if (slots.referenceVideo) carryOver = { referenceVideo: slots.referenceVideo };
      }
      setPendingIntent({
        intent,
        targetMode: target,
        selection: command.selection,
        ...(carryOver ? { carryOver } : {}),
      });
      setRemountTokens((prev) => ({ ...prev, [target]: prev[target] + 1 }));
    }
    setMode(target);

    // I9 §3-6 #10: 📷 imageFromCurrentFrame is keep-panel-state (no remount), so
    // there is no `pendingIntent` to carry its "grab the current frame" work.
    // Deliver it over the live-command channel to the always-mounted Create
    // screen instead, which captures the frame into KEYFRAMES without wiping the
    // form. Sits AFTER the busy-guard/chain-discard gates (a refused route
    // returns above and never reaches here) and BEFORE the placement-C
    // reservation below (which still runs as system C for #10). ✨ #9
    // (text-to-video) shares `keepPanelState` but delivers nothing to the form.
    if (intent === "image-from-frame") {
      dispatchCreateCommand({ type: "captureFrameToKeyframe" });
    }

    // W8 (#9/#10): live-set the Create form's DURATION to the resolution's
    // comfort ceiling (computed above from the published width/height). Only
    // fired when it actually resolved — a null ceiling leaves the form's current
    // DURATION alone. This dispatch reaches the always-mounted Create screen's
    // `setDuration` subscriber (form value only; no remount).
    if (cursorItemCeilingNumFrames !== null) {
      dispatchCreateCommand({ type: "setDuration", numFrames: cursorItemCeilingNumFrames });
    }

    // ── Step 7: reservation length source. Two families:
    //  - ✨/📷 keep their panel state, so their placeholder length matches the
    //    LIVE form the user will generate from — the published DURATION/fps
    //    (`keepStateNumFrames`/`keepStateGenFps`), or the resolution's comfort
    //    ceiling live-set above for #9/#10 (`cursorItemCeilingNumFrames`),
    //    falling back to the config defaults before Create ever publishes (D3).
    //  - Material items (#1/#2/#3/#4/#6/#7/#12) REMOUNT the form and seed its
    //    initial DURATION/fps from the shared純関数 `resolvePrefillSeed`. Owner
    //    requirement 2026-07-21 #1: place the provisional at that SAME derived
    //    length from the outset (not the config default), so its ribbon matches
    //    the remounted form the instant it appears. Both call `resolvePrefillSeed`
    //    with the identical right-click inputs, so they cannot diverge here.
    //
    // Two residual, transient divergences remain — both are accepted (not bugs)
    // and both converge at Generate via `bindToJob`'s
    // `timeline.updateProvisionalReservation` re-stamp (§5-3 用途b / owner
    // requirement #2): (a) the `project` policy's async `getEditInfo` overwrite
    // recomputes the form's DURATION a beat after placement, and (b) A2V (#3/#7)
    // supersedes the material-clamped seed once the wav is actually measured.
    //
    // §1-17 Retake (`retakeRange`, 系統"D") is a THIRD, deliberately different
    // case — and it is not a residual divergence at all. Its seed is the user's
    // SELECTED RANGE converted to generation frames (`resolvePrefillSeed`'s new
    // `selectedRangeLength` policy), which is the honest length to show the
    // moment the range is picked, but it is NOT the length that will be
    // generated: the pipeline can only produce an 8n+1 window inside [73,169]
    // (`timeline/retakeWindow.ts`). Rather than let the two drift, the Retake
    // panel RE-PLACES the reservation when Generate is pressed —
    // `reservePlacement` again, with the confirmed window's start frame and
    // length, which the seat's 用途b "move" path handles — and only then
    // submits. So this one genuinely converges, by打ち直し rather than by the
    // `bindToJob` re-stamp above.
    let numFrames: number;
    let genFps: number;
    if (keepPanelState) {
      numFrames = cursorItemCeilingNumFrames ?? keepStateNumFrames;
      genFps = keepStateGenFps;
    } else {
      const seed = resolvePrefillSeed({ intent, selection: command.selection, config, sizePolicy, fpsPolicy });
      numFrames = seed.numFrames ?? config.generation_defaults.num_frames;
      genFps = seed.frameRate ?? config.generation_defaults.frame_rate;
    }

    // ── Step 8: place (or move) the provisional per the route's placement系統
    // (§5-9). Material A/B derive the position from the selected object's frame
    // range; cursor C from the edit cursor. Shares the `placeProvisional` helper
    // with #5/#8 (a fresh insert when the seat is idle, a MOVE when reserved —
    // §5-6 用途b).
    const placement = route.placement;
    if (
      placement === "A" ||
      placement === "B" ||
      placement === "C" ||
      placement === "D" ||
      placement === "E"
    ) {
      if (placement === "C") {
        placeProvisional({
          placement,
          cursor: { layer: command.selection.cursorLayer, frame: command.selection.cursorFrame },
          numFrames,
          genFps,
        });
      } else if (placement === "D") {
        // W0 系統D (retakeRange): the position is derived from the user's
        // SELECTED FRAME RANGE, not from the object's own span — a retake's
        // output covers the range, so its head belongs at `rangeStart`. The
        // arithmetic is native's `"B"` (`layer_max+1` + `materialFrameStart`),
        // so this passes the RANGE through the same `material` slot and
        // `placementParams` maps `"D"` → `"B"` on the way out. The layer is the
        // selected object's (honest bookkeeping; `"B"` ignores it and always
        // lands on `layer_max+1`). `guardMenuSelection` already refused a
        // `hasRange: false` selection, so the range here is real.
        const item = command.selection.selected[0]!;
        placeProvisional({
          placement,
          material: {
            layer: item.layer,
            frameStart: command.selection.rangeStart,
            frameEnd: command.selection.rangeEnd,
          },
          numFrames,
          genFps,
          // Retake パネルの❌（片付け）が「自分が置いた席かどうか」を見分ける
          // ための publish。これが無いと、Create 側の右クリックで席が移動した
          // あとの❌が、他タブの⏳リボンを消してしまう
          // （`provisionalReservation.ts` の同 publish の doc を参照）。
          onReserved: (pendingId) => publishRetakeReservation(pendingId),
        });
      } else if (placement === "E") {
        // 素材（末尾）系統E (末尾合わせ): the result ENDS with the selected
        // material, so its frozen tail has to land on the material's head — i.e.
        // the provisional starts `出力長 − 凍結分` frames EARLIER than the object.
        // Native has no such placement (see `menuRouting.ts`'s `MenuPlacement`),
        // so the shift is computed here and `placementParams` maps `"E"` onto
        // `"B"`, exactly the way 系統D already does.
        //
        // 窓内モード (2026-08-17): NEITHER length is an estimate any more. The
        // frozen tail is the constant `END_SOURCE_CONTEXT_FRAMES` whatever the
        // material turns out to be, and the output length is the seeded clip
        // length itself (`forceSingleClip`), so the seat lands where the finished
        // job lands. `ChainedScreen.handleGenerate` still re-places before
        // submitting, but now only to follow an edit the user made to the clip.
        //
        // 厳密整列（素材1フレーム目=出力の第(L−錨)フレーム、2026-08-17実測確認）:
        // the server cuts `context_frames + 1` frames off the material's head
        // (the causal VAE's keyframe primer), so the output's last 8 frames
        // correspond to the material's 2nd–9th frame, and the material's OWN
        // 1st frame lands one frame further back — at output frame `L − 錨`
        // (1-indexed). `leadPixelFrames` below is therefore `numFrames −
        // (END_SOURCE_CONTEXT_FRAMES + 1)`, not just `numFrames −
        // END_SOURCE_CONTEXT_FRAMES`: the `+1` is the primer frame, which is
        // material frames 2–9 (not 1–8) that get frozen onto the anchor.
        //
        // Deliberately NO `sourceLocationMap` entry: the Generate-time re-place
        // issues a new pending id, so a map entry keyed by this one would be
        // orphaned — the screen keeps its own local anchor instead.
        //
        // 逆順Chained (2026-08-18, second stage): unaffected by clip count. The
        // right-click reservation always starts the form at ONE clip
        // (`ChainedScreen`'s `forceSingleClip`), so `numFrames` here is still
        // exactly the seat's own single-clip length whether or not the user
        // goes on to add clips — `ChainedScreen.handleGenerate`'s re-place is
        // what catches up to a later multi-clip edit, not this initial seat.
        const item = command.selection.selected[0]!;
        placeProvisional({
          placement,
          material: {
            layer: item.layer,
            frameStart: tailAlignedStartFrame({
              materialFrameStart: item.frameStart,
              leadPixelFrames: numFrames - (END_SOURCE_CONTEXT_FRAMES + 1),
              genFps,
              projectRate: command.selection.rate,
              projectScale: command.selection.scale,
            }),
            frameEnd: item.frameEnd,
          },
          // 出力長 = クリップ尺そのもの。窓内モードでは素材の分だけ伸びたりしない。
          numFrames,
          genFps,
        });
      } else {
        // Guard (step 1) already enforced single-selection of the right kind, so
        // selected[0] exists for material placements.
        const item = command.selection.selected[0]!;
        placeProvisional({
          placement,
          material: { layer: item.layer, frameStart: item.frameStart, frameEnd: item.frameEnd },
          numFrames,
          genFps,
          // I6 Join: only the v2v continuation (#1 extend-video) needs its source
          // object's timeline location remembered — for the joined clip's
          // "source tail − trim" insert position (JOIN_FEATURE_RESEARCH.md §4.5).
          // Keyed by the reservation's pending id; re-keyed to the real job id at
          // Generate (ChainedScreen.onSubmitted).
          onReserved:
            intent === "extend-video"
              ? (pendingId) =>
                  recordSourceLocation(pendingId, {
                    layer: item.layer,
                    frameStart: item.frameStart,
                    frameEnd: item.frameEnd,
                    filePath: item.filePath,
                    rate: command.selection.rate,
                    scale: command.selection.scale,
                  })
              : undefined,
        });
      }
    }
  }, [nativeBridge, prompt, config, sizePolicy, fpsPolicy, showNote, strings, confirmChainDiscard, toasts]);

  // Reload/startup re-sync (§2): rebuild the single reservation seat from any
  // NzVideomni#… placeholders that survived a project reload, so the busy-guard and
  // the "予約席は1つ" move behave correctly after a restart. Runs once on mount;
  // `reconcileFromTimeline` swallows any RPC failure to `idle` internally, so a
  // host without the method (or a transport error) never blocks startup. A
  // job-bound marker restored as `generating` converges back to idle via the
  // JobsContext poll -> `releaseIfSettled` path (see that function's doc).
  const reconciledRef = useRef(false);
  useEffect(() => {
    if (reconciledRef.current) return;
    reconciledRef.current = true;
    // X2(b): terminal jobs from the ledger (empty on first paint = safe side).
    void reconcileFromTimeline(nativeBridge ?? defaultBridge, terminalJobIds(jobsRef.current));
  }, [nativeBridge]);

  // §6-b (projectLoaded push): the plugin fires `timeline.projectLoaded` after a
  // project is opened/reopened (plugin.cpp's OnProjectLoad). Re-run the reconcile
  // on every fire so a provisional that survived in a reopened .aup2 rebuilds the
  // single seat — the mount-once effect above cannot catch a load that happens
  // LATER (nor a second/third open). Kept separate from that effect (which stays
  // as a harmless first-paint attempt): this one only re-subscribes when the
  // bridge identity changes, mirroring `useMenuRouter`/`useProjectOrphans`'
  // subscribe->unsubscribe pattern.
  useEffect(() => {
    const b = nativeBridge ?? defaultBridge;
    const unsubscribe = b.on(TIMELINE_PROJECT_LOADED_EVENT, () => {
      // X2(b): terminal jobs from the ledger so a reopened .aup2's stale settled
      // tag can't hold the seat `generating`.
      void reconcileFromTimeline(b, terminalJobIds(jobsRef.current));
    });
    return unsubscribe;
  }, [nativeBridge]);

  useMenuRouter({ onRoute: handleRoute, nativeBridge });

  // A manual tab switch is a deliberate "start fresh" gesture: drop any pending
  // prefill so returning to a mode by hand shows its plain default form rather
  // than silently re-applying a stale right-click selection.
  const handleModeChange = useCallback((next: AppMode) => {
    setPendingIntent(null);
    setMode(next);
  }, []);

  const singleIntent = pendingIntent?.targetMode === "single" ? pendingIntent : undefined;
  const chainedIntent = pendingIntent?.targetMode === "chained" ? pendingIntent : undefined;
  // W0 (2026-08-09): same one-shot hand-off as the two above — Edit is now a
  // right-click destination (`outpaintVideo`/`retakeRange`).
  const editIntent = pendingIntent?.targetMode === "edit" ? pendingIntent : undefined;

  return (
    <div className="app-shell">
      <header className="app-header">
        {/* Tool-version dropdown (2026-08-19, mock — owner-directed): occupies
            the old static "Nz-Videomni" title's spot. Selecting "LTX 2.5" has no
            real effect yet; the onChange below immediately reverts it to
            "LTX 2.3", silently (no toast/warning). */}
        <select
          className="app-title-select"
          aria-label={strings.toolVersion.ariaLabel}
          value={toolVersion}
          onChange={() => setToolVersion("LTX 2.3")}
        >
          <option value="LTX 2.3">{strings.toolVersion.ltx23}</option>
          <option value="LTX 2.5">{strings.toolVersion.ltx25}</option>
        </select>
        <StatusHeader state={serverStatus} onRetry={retry} />
        <ModeTabs mode={mode} onChange={handleModeChange} />
        <button
          type="button"
          className="icon-button settings-gear"
          aria-label={strings.settings.title}
          title={strings.settings.title}
          onClick={() => setSettingsOpen(true)}
        >
          ⚙
        </button>
      </header>

      <PromptBar
        value={prompt}
        onChange={setPrompt}
        // Only suppress chips on Create — the panel these names would move
        // INTO only renders there; hiding them anywhere else would just make
        // a stray control tag invisible with no other way to see/remove it.
        hiddenNames={mode === "single" ? controlLoraNames : undefined}
        onCompositionStart={handlePromptCompositionStart}
        onCompositionEnd={handlePromptCompositionEnd}
      />

      {/* NAG (2026-07-28): one shared accordion, outside every mode tab —
          Create/Chain/Batch all read `nagControls.nag` (owner decision §UX 1). */}
      <NagAccordion nag={nagControls.nag} controls={nagControls} />

      {/* Shared note area (§6): tab-bar-level slot, visible on all three
          screens, above the operation panel. Renders nothing until a note is
          shown. */}
      <NoteArea note={note} onDismiss={dismissNote} />

      <div className="app-body">
        <ShowNoteProvider showNote={showNote}>
        <main className="app-mode-body">
          {/* Every mode screen stays mounted; only the active one is
              visible. The wrapper is a bare <div> (no display styling) so the
              UA's `[hidden] { display: none }` does the hiding — form values,
              expanded previews, preset selection and the like therefore
              survive a tab switch instead of being torn down with the screen.
              `role="tabpanel"` also lets tests grab the single visible panel
              (`getByRole` skips hidden subtrees) to disambiguate the now
              double-mounted UI. See `remountTokens` above for how right-click
              prefills still force a targeted remount via `key`. */}
          <div role="tabpanel" hidden={mode !== "single"}>
            <SingleScreen
              key={`single-${remountTokens.single}`}
              prompt={prompt}
              baseUrl={baseUrl}
              initialIntent={singleIntent}
              nativeBridge={nativeBridge}
              highlightedJobId={highlightedJobId}
              onJobSubmitted={setHighlightedJobId}
              controlLora={controlLora}
              setControlLora={setControlLora}
              controlLoraNames={controlLoraNames}
              nag={nagControls.nag}
              acceleration={accelerationControls.acceleration}
              sageAvailable={sageAvailability(statusBody)}
            />
          </div>
          <div role="tabpanel" hidden={mode !== "chained"}>
            <ChainedScreen
              key={`chained-${remountTokens.chained}`}
              prompt={prompt}
              baseUrl={baseUrl}
              initialIntent={chainedIntent}
              nativeBridge={nativeBridge}
              highlightedJobId={highlightedJobId}
              onJobSubmitted={setHighlightedJobId}
              controlLoraNames={controlLoraNames}
              depthLoraNames={depthLoraNames}
              referenceDownscaleFactors={referenceDownscaleFactors}
              nag={nagControls.nag}
              acceleration={accelerationControls.acceleration}
            />
          </div>
          {/* Edit (2026-08-09): promoted from a disabled mock tab to a real
              mode, and since W0 (共通スパイン) a right-click destination too —
              `initialIntent` + `remountTokens.edit` are the identical one-shot
              prefill arrangement Create/Chain use, so an Edit-系 right-click
              remounts the screen and re-applies the fresh selection while a
              manual tab switch leaves it (and its state) alone. Outpainting is
              a real panel now (Retake is still a placeholder), so the screen
              takes the same shared prompt + ledger props Create/Chain do — it
              mounts its own `JobLedger` below the panels. */}
          <div role="tabpanel" hidden={mode !== "edit"}>
            <EditScreen
              key={`edit-${remountTokens.edit}`}
              initialIntent={editIntent}
              prompt={prompt}
              baseUrl={baseUrl}
              nativeBridge={nativeBridge}
              highlightedJobId={highlightedJobId}
              onJobSubmitted={setHighlightedJobId}
            />
          </div>
          <div role="tabpanel" hidden={mode !== "inventory"}>
            <InventoryScreen
              key={`inventory-${remountTokens.inventory}`}
              prompt={prompt}
              onPromptChange={setPrompt}
              baseUrl={baseUrl}
              nativeBridge={nativeBridge}
            />
          </div>
        </main>
        </ShowNoteProvider>
      </div>

      <Toasts onSelectJob={handleSelectJob} />

      {settingsOpen && (
        <SettingsPanel
          onClose={() => setSettingsOpen(false)}
          onSaved={retry}
          acceleration={accelerationControls.acceleration}
          onAttentionBackendChange={accelerationControls.setAttentionBackend}
          onBlockSwapPrefetchChange={accelerationControls.setBlockSwapPrefetch}
          onKeepResidentChange={accelerationControls.setKeepResident}
          onFusedGgufDequantKernelChange={accelerationControls.setFusedGgufDequantKernel}
          onVaeModeChange={accelerationControls.setVaeMode}
          // The panel reads sage availability straight off this existing
          // 10s /status poll — no capability fetch of its own.
          serverStatus={serverStatus}
        />
      )}

      {/* I8 §3-4 (※): Chain-discard confirm. Rendered while `handleRoute` awaits
          the user's answer; OK continues the #1/#6 load, Cancel aborts it. */}
      {chainDiscardOpen && (
        <ConfirmDialog
          title={strings.dialogs.chainDiscard.title}
          body={strings.dialogs.chainDiscard.body}
          confirmLabel={strings.dialogs.chainDiscard.confirmButton}
          cancelLabel={strings.dialogs.chainDiscard.cancelButton}
          onConfirm={() => answerChainDiscard(true)}
          onCancel={() => answerChainDiscard(false)}
        />
      )}
    </div>
  );
}
