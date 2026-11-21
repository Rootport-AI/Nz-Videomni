/**
 * Batch v2v-ref-long: what the panel reads and calls. Every value lives in
 * `v2vRefLongStore` (and the run in `runtime.ts`), so the panel may unmount at
 * any time — a scan or a run in flight keeps going and lands in the store, and
 * a remount reads it back. The derived values are the first-stage pure
 * functions, memoised.
 */
import { useCallback, useMemo, useSyncExternalStore } from "react";
import type { ApiClient } from "../../api/client";
import type { AppConfig, TransformerRole } from "../../api/types";
import { bridge as defaultBridge, BridgeError } from "../../bridge";
import type { NativeBridge } from "../../bridge";
import { useStrings } from "../../i18n/LanguageContext";
import { RUN_LOCK_OWNER_BATCH_V2V_REF_LONG, getRunLockOwner, subscribeRunLock } from "../../shell/runLock";
import { useLoras } from "../inventory/useLoras";
import { appendReferenceTag } from "../single/referenceRules";
import type { ReferenceKind, ReferenceRules } from "../single/referenceRules";
import { PICK_KIND, measureDuration } from "../single/useReferenceSlots";
import { buildV2vRefLongTemplate } from "./buildPayload";
import {
  emptyFieldsOf,
  filesOfField,
  filledFolderCount,
  filledPath,
  pickBaseField,
} from "./folderRules";
import { buildRowPlan, effectiveClip, rowBlockReasons, rowIgnoredClips, rowSourceFile } from "./rowModel";
import type { RowEnv } from "./rowModel";
import { runV2vRefLong, stopV2vRefLong } from "./runtime";
import { scanFolders } from "./scan";
import { chainSettingReasons, startBlockReasons } from "./startGuards";
import { v2vRefLongStore } from "./store";
import type { PromptMode, UseV2vRefLongFormResult, V2vRefLongChainSource, V2vRowView } from "./types";

export interface UseV2vRefLongFormDeps {
  config: AppConfig;
  chain: V2vRefLongChainSource;
  rules: ReferenceRules;
  role: TransformerRole | null;
  serverBusy: boolean;
  nativeBridge?: NativeBridge | undefined;
  /** The loaded base model's id (`useLoras`'s `refreshKey`). */
  loraRefreshKey?: string | undefined;
  /** Test-only seam for `useLoras`. */
  apiClient?: ApiClient | undefined;
  /** Test-only pass-throughs to the runner. */
  pollIntervalMs?: number | undefined;
  jobBusyBackoffMs?: number | undefined;
}

const BYTES_PER_MB = 1024 * 1024;

/** Per card, the newest `attachClipFile`/`clearClipSlot`; an older measure that
 * resolves later is dropped. Module-level so it outlives the hook. */
const cardGenerations = new Map<string, number>();

function bumpCardGeneration(row: number, clip: number, kind: ReferenceKind, slot: number): { key: string; gen: number } {
  const key = `${row}:${clip}:${kind}:${slot}`;
  const gen = (cardGenerations.get(key) ?? 0) + 1;
  cardGenerations.set(key, gen);
  return { key, gen };
}

function errorMessage(err: unknown): string {
  return err instanceof BridgeError ? `${err.code}: ${err.message}` : err instanceof Error ? err.message : String(err);
}

export function useV2vRefLongForm(deps: UseV2vRefLongFormDeps): UseV2vRefLongFormResult {
  const { config, chain, rules, role, serverBusy } = deps;
  const nativeBridge = deps.nativeBridge ?? defaultBridge;
  const strings = useStrings();
  const state = useSyncExternalStore(v2vRefLongStore.subscribe, v2vRefLongStore.getSnapshot, v2vRefLongStore.getSnapshot);
  const dispatch = v2vRefLongStore.dispatch;
  const { scan, inputs, run } = state;

  // `chain.buildRequest` changes identity exactly when the request would (same
  // reasoning as `batch-i2v-long/useBatchI2vLongForm.ts`).
  const chainBuildRequest = chain.buildRequest;
  const template = useMemo(() => buildV2vRefLongTemplate(chainBuildRequest()), [chainBuildRequest]);

  const clipCount = run.state !== "idle" && run.clipCount !== null ? run.clipCount : chain.clips.length;
  const inputsLocked = run.state !== "idle" || state.scanning;

  const { upload } = config;
  const clip0Frames = chain.clips[0]?.numFrames ?? 0;
  const minFramesWithSource = chain.minFramesForOverlap;
  const frameRate = chain.common.frameRate;
  const contextFrames = chain.fixedOverlapFrames;
  const env = useMemo<RowEnv>(
    () => ({
      clipCount,
      clip0Frames,
      minFramesWithSource,
      // The panel is only drawn on H3, where this is set; `null` blocks Start below.
      contextFrames: contextFrames ?? 0,
      frameRate,
      maxBytes: {
        image: upload.max_image_size_mb * BYTES_PER_MB,
        video: upload.max_video_size_mb * BYTES_PER_MB,
        audio: upload.max_audio_size_mb * BYTES_PER_MB,
      },
      fillFollowing: inputs.fillFollowing,
    }),
    [clipCount, clip0Frames, minFramesWithSource, contextFrames, frameRate, upload, inputs.fillFollowing],
  );

  // --- derived ---------------------------------------------------------------

  const baseField = useMemo(() => (scan ? pickBaseField(scan) : null), [scan]);

  const rowViews = useMemo<V2vRowView[]>(() => {
    if (!scan) return [];
    const baseFiles = baseField ? filesOfField(scan, baseField) : [];
    return state.rows.map((row) => ({
      row,
      baseFileName: baseFiles[row.index]?.name ?? null,
      hasSource: rowSourceFile(scan, row.index) !== null,
      blockReasons: rowBlockReasons(row, scan, env),
      ignoredClips: rowIgnoredClips(row, scan, env.clipCount, env.fillFollowing),
    }));
  }, [scan, baseField, state.rows, env]);

  const emptyFields = useMemo(() => (scan ? emptyFieldsOf(scan) : []), [scan]);
  const allFieldsEmpty = scan !== null && baseField === null;
  const unusedFrameStart = inputs.frames.length > clipCount ? clipCount : null;
  const folderCount = useMemo(() => filledFolderCount(inputs.frames), [inputs.frames]);

  const lorasState = useLoras(
    deps.apiClient ? { apiClient: deps.apiClient, refreshKey: deps.loraRefreshKey } : { refreshKey: deps.loraRefreshKey },
  );
  const readyLoras = lorasState.status === "ready" ? lorasState.loras : null;
  // Only judged once the list loaded: a failed `GET /loras` must not block.
  const hasUnknownLoraTag = useMemo(() => {
    if (readyLoras === null) return false;
    const known = new Set(readyLoras.map((l) => l.name));
    return (template.loras ?? []).some((l) => !known.has(l.name));
  }, [readyLoras, template]);

  const runLockOwner = useSyncExternalStore(subscribeRunLock, getRunLockOwner, getRunLockOwner);
  const lockedByOther = runLockOwner !== null && runLockOwner !== RUN_LOCK_OWNER_BATCH_V2V_REF_LONG;

  // `chain` itself is a fresh object per Chain-screen render; depend on the members read.
  const chainClips = chain.clips;
  const minFramesForOverlap = chain.minFramesForOverlap;
  const hasSourceAudio = chain.hasSourceAudio;
  const guard = useMemo(
    () =>
      startBlockReasons({
        state,
        chain: { clips: chainClips, minFramesForOverlap, hasSourceAudio },
        env,
        commonPrompt: template.prompt,
        role,
        hasUnknownLoraTag,
        serverBusy,
        lockedByOther,
      }),
    [state, chainClips, minFramesForOverlap, hasSourceAudio, env, template, role, hasUnknownLoraTag, serverBusy, lockedByOther],
  );
  const chainValidityReasons = chain.validityReasons;
  const chainBlockReasons = useMemo(
    () => chainSettingReasons({ validityReasons: chainValidityReasons }),
    [chainValidityReasons],
  );

  const summary = useMemo(
    () => ({ done: state.rows.filter((r) => r.stat === "Done").length, total: state.rows.length }),
    [state.rows],
  );
  const currentRow = useMemo(() => state.rows.find((r) => r.stat === "Generating") ?? null, [state.rows]);

  const canStart =
    run.state === "idle" &&
    !state.scanning &&
    contextFrames !== null &&
    guard.reasons.length === 0 &&
    chainBlockReasons.length === 0;
  const anyFolder = filledPath(inputs.videoDir) !== "" || folderCount > 0;
  const canScan = !inputsLocked && anyFolder;

  const effectiveClipOf = useCallback(
    (rowIndex: number, clip: number) => {
      const row = state.rows.find((r) => r.index === rowIndex);
      if (!scan || !row) return null;
      return effectiveClip(row, scan, clip, inputs.fillFollowing);
    },
    [scan, state.rows, inputs.fillFollowing],
  );

  // --- inputs ----------------------------------------------------------------

  const pickFolder = useCallback(
    async (title: string | undefined): Promise<string | null> => {
      try {
        const { folderPath } = await nativeBridge.request("ui.pickFolder", title ? { title } : {});
        return folderPath;
      } catch (err) {
        // A failed dialog is shown in the scan-error line (as Batch i2v-long does).
        if (!(err instanceof BridgeError && err.code === "CANCELLED")) {
          dispatch({ type: "scanFailed", message: errorMessage(err) });
        }
        return null;
      }
    },
    [nativeBridge, dispatch],
  );

  const setVideoDir = useCallback((path: string) => dispatch({ type: "setVideoDir", path }), [dispatch]);
  const pickVideoDir = useCallback(
    async (title?: string) => {
      const path = await pickFolder(title);
      if (path !== null) dispatch({ type: "setVideoDir", path });
    },
    [pickFolder, dispatch],
  );
  const setJoined = useCallback((joined: boolean) => dispatch({ type: "setJoined", joined }), [dispatch]);
  const addField = useCallback(
    (frame: number, kind: ReferenceKind) => dispatch({ type: "addField", frame, kind, rules }),
    [dispatch, rules],
  );
  const setFieldPath = useCallback(
    (frame: number, fieldId: number, path: string) => dispatch({ type: "setFieldPath", frame, fieldId, path }),
    [dispatch],
  );
  const pickFieldDir = useCallback(
    async (frame: number, fieldId: number, title?: string) => {
      const path = await pickFolder(title);
      if (path !== null) dispatch({ type: "setFieldPath", frame, fieldId, path });
    },
    [pickFolder, dispatch],
  );
  const removeField = useCallback(
    (frame: number, fieldId: number) => dispatch({ type: "removeField", frame, fieldId }),
    [dispatch],
  );
  const addFrame = useCallback(() => dispatch({ type: "addFrame" }), [dispatch]);
  const removeLastFrame = useCallback(() => dispatch({ type: "removeLastFrame" }), [dispatch]);
  const setFillFollowing = useCallback((value: boolean) => dispatch({ type: "setFillFollowing", value }), [dispatch]);
  const setOutDir = useCallback((path: string) => dispatch({ type: "setOutDir", path }), [dispatch]);
  const pickOutDir = useCallback(
    async (title?: string) => {
      const path = await pickFolder(title);
      if (path !== null) dispatch({ type: "setOutDir", path });
    },
    [pickFolder, dispatch],
  );

  const runScan = useCallback(async () => {
    // Read the store, not this render: the call may come from a stale closure.
    const current = v2vRefLongStore.getSnapshot();
    if (current.scanning || current.run.state !== "idle") return;
    dispatch({ type: "scanStarted" });
    try {
      const result = await scanFolders(
        nativeBridge,
        { videoDir: current.inputs.videoDir, frames: current.inputs.frames },
        upload,
        rules,
        (done, total) => dispatch({ type: "scanProgress", done, total }),
      );
      dispatch({ type: "scanSucceeded", result });
    } catch (err) {
      dispatch({ type: "scanFailed", message: errorMessage(err) });
    }
  }, [dispatch, nativeBridge, upload, rules]);

  // --- rows ------------------------------------------------------------------

  const setMainPromptMode = useCallback((mode: PromptMode) => dispatch({ type: "setMainPromptMode", mode }), [dispatch]);
  const setRowMainPrompt = useCallback(
    (row: number, prompt: string) => dispatch({ type: "setRowMainPrompt", row, prompt }),
    [dispatch],
  );
  const chainPrompt = template.prompt;
  const copyChainPromptToRow = useCallback(
    (row: number) => dispatch({ type: "setRowMainPrompt", row, prompt: chainPrompt }),
    [dispatch, chainPrompt],
  );
  const setClipPrompt = useCallback(
    (row: number, clip: number, prompt: string) => dispatch({ type: "setClipPrompt", row, clip, prompt }),
    [dispatch],
  );
  const setClipPromptMode = useCallback(
    (row: number, clip: number, mode: PromptMode) => dispatch({ type: "setClipPromptMode", row, clip, mode }),
    [dispatch],
  );

  const attachClipFile = useCallback(
    async (row: number, clip: number, kind: ReferenceKind, slot: number, filePath: string, fileName: string) => {
      const { key, gen } = bumpCardGeneration(row, clip, kind, slot);
      const scanAtCall = v2vRefLongStore.getSnapshot().scan;
      const durationSec = await measureDuration(nativeBridge, kind, filePath, undefined);
      if (cardGenerations.get(key) !== gen) return;
      // Dropped when the rows were rebuilt or locked meanwhile.
      const now = v2vRefLongStore.getSnapshot();
      if (now.scanning || now.run.state !== "idle" || now.scan !== scanAtCall) return;
      dispatch({ type: "setClipSlot", row, clip, kind, slot, file: { fileName, filePath, durationSec } });
    },
    [nativeBridge, dispatch],
  );
  const pickClipFile = useCallback(
    async (row: number, clip: number, kind: ReferenceKind, slot: number) => {
      let picked: { filePath: string; fileName: string };
      try {
        picked = await nativeBridge.request("ui.pickFile", { kind: PICK_KIND[kind] });
      } catch {
        // Cancelled or failed: nothing happens (as `useReferenceSlots`'s `pick`).
        return;
      }
      await attachClipFile(row, clip, kind, slot, picked.filePath, picked.fileName);
    },
    [nativeBridge, attachClipFile],
  );
  const clearClipSlot = useCallback(
    (row: number, clip: number, kind: ReferenceKind, slot: number) => {
      bumpCardGeneration(row, clip, kind, slot);
      dispatch({ type: "setClipSlot", row, clip, kind, slot, file: null });
    },
    [dispatch],
  );
  const appendClipTag = useCallback(
    (row: number, clip: number, tag: string) => {
      const target = v2vRefLongStore.getSnapshot().rows.find((r) => r.index === row);
      if (!target) return;
      const prompt = target.clipEdits[clip]?.prompt ?? "";
      dispatch({ type: "setClipPrompt", row, clip, prompt: appendReferenceTag(prompt, tag) });
    },
    [dispatch],
  );
  const resetRow = useCallback((row: number) => dispatch({ type: "resetRow", row }), [dispatch]);

  // --- run -------------------------------------------------------------------

  const joinFailedText = strings.batchV2vRefLong.joinFailed;
  const { pollIntervalMs, jobBusyBackoffMs } = deps;
  const start = useCallback(() => {
    if (!canStart || !scan || contextFrames === null) return;
    // Read the store too, like `scan()`: the call may come from a stale closure.
    const current = v2vRefLongStore.getSnapshot();
    if (current.scanning || current.run.state !== "idle") return;
    const rowsByIndex = new Map(state.rows.map((r) => [r.index, r]));
    const plans = guard.runnableRows.flatMap((index) => {
      const row = rowsByIndex.get(index);
      return row ? [buildRowPlan(row, scan, clipCount, inputs.fillFollowing)] : [];
    });
    runV2vRefLong({
      bridge: nativeBridge,
      template,
      settings: {
        mainPromptMode: state.mainPromptMode,
        contextFrames,
        // The rules the rows' cards were built and judged with.
        rules: scan.rules,
        role,
        joined: inputs.joined,
        joinFailedText,
      },
      plans,
      outDir: inputs.outDir.trim(),
      clipCount,
      ...(pollIntervalMs !== undefined ? { pollIntervalMs } : {}),
      ...(jobBusyBackoffMs !== undefined ? { jobBusyBackoffMs } : {}),
    });
  }, [canStart, scan, contextFrames, state.rows, state.mainPromptMode, guard.runnableRows, clipCount, inputs.fillFollowing, inputs.joined, inputs.outDir, nativeBridge, template, role, joinFailedText, pollIntervalMs, jobBusyBackoffMs]);

  const stop = useCallback(() => stopV2vRefLong(), []);

  return {
    state,
    rules,
    rowViews,
    baseField,
    clipCount,
    emptyFields,
    allFieldsEmpty,
    unusedFrameStart,
    filledFolderCount: folderCount,
    blockReasons: guard.reasons,
    chainBlockReasons,
    promptEmptyRows: guard.promptEmptyRows,
    minFramesForOverlap,
    fixedOverlapFrames: chain.fixedOverlapFrames,
    seedFixed: template.seed >= 0,
    summary,
    currentRow,
    canStart,
    canScan,
    inputsLocked,
    effectiveClipOf,
    setVideoDir,
    pickVideoDir,
    setJoined,
    addField,
    setFieldPath,
    pickFieldDir,
    removeField,
    addFrame,
    removeLastFrame,
    setFillFollowing,
    setOutDir,
    pickOutDir,
    scan: runScan,
    setMainPromptMode,
    setRowMainPrompt,
    copyChainPromptToRow,
    setClipPrompt,
    setClipPromptMode,
    attachClipFile,
    pickClipFile,
    clearClipSlot,
    appendClipTag,
    resetRow,
    start,
    stop,
  };
}
