/**
 * バッチi2v-long（§1-7）のフォーム状態。フォルダ選択・スキャン・行ごとの
 * プロンプト・開始ガード・実行制御をすべて持つ。
 *
 * Batch i2v-long's whole browser-side form state: the image/output folders,
 * the folder scan -> in-memory row list, each row's own prompt (+ the one
 * add/replace mode radio shared by every row), the start guards, and the
 * run/stop controls (through {@link useBatchI2vLongRunner}, which is backed by
 * `runtime.ts`'s singleton).
 *
 * 2026-07-30 owner feedback: the batch-wide prompt textarea this panel used to
 * own is GONE. A prompt applied to every image was indistinguishable from just
 * editing the shared prompt above, so the text moved onto the rows —
 * `I2vLongRow.prompt`, one per image, combined with the shared prompt at submit
 * time. This mirrors Batch A2V exactly (one mode radio for the batch, one text
 * input per row, plus a 📝 button that copies the shared prompt into a row).
 *
 * The design's backbone is the TEMPLATE SNAPSHOT (see `buildI2vLongPayload.ts`):
 * every generation parameter comes from the Chain form's own `buildRequest()`,
 * so this hook owns no size/fps/seed/clip fields at all. {@link templatePreview}
 * is built once per Chain-form change and is the SINGLE value both the guards
 * and `start()` read — the panel can never block on one body while submitting a
 * different one. Its `prompt` is the SHARED prompt only; the per-row composition
 * happens inside the runner, and the guards below re-run the same
 * `composeBatchPrompt` over every runnable row to judge exactly what will be
 * sent.
 *
 * Folder handling deliberately mirrors Batch A2V's `useBatchForm`
 * (`modes/batch/useBatchForm.ts`): 📁 picker + hand-typed path committed on
 * blur, an `fs.listFiles` existence probe for the INPUT folder only (the output
 * folder may legitimately not exist yet), and an auto-derived output folder that
 * stops following once the user picks their own. That file is not imported —
 * A2V is outside this sprint's edit scope and the two panels are expected to
 * diverge — but the behaviour is intentionally identical.
 */
import { useCallback, useEffect, useMemo, useState, useSyncExternalStore } from "react";
import type { ApiClient } from "../../api/client";
import type { AppConfig, GenerateChainRequest } from "../../api/types";
import { bridge as defaultBridge, BridgeError } from "../../bridge";
import type { NativeBridge } from "../../bridge";
import { RUN_LOCK_OWNER_BATCH_I2V_LONG, getRunLockOwner, subscribeRunLock } from "../../shell/runLock";
import {
  CHAIN_AUDIO_REASON_CODES,
  CHAIN_END_SOURCE_REASON_CODES,
  CHAIN_REFERENCE_REASON_CODES,
  CHAIN_SOURCE_REASON_CODES,
} from "../chained/generateReasonMessages";
import { useLoras } from "../inventory/useLoras";
import { buildI2vLongTemplate, composeBatchPrompt } from "./buildI2vLongPayload";
import type { PromptMode } from "./buildI2vLongPayload";
import type { ChainSnapshotSource } from "./chainSnapshot";
import { deriveI2vLongOutDir, scanImagesToRows } from "./imageRows";
import type { I2vLongRow, ScannedImageFile } from "./imageRows";
import { useBatchI2vLongRunner } from "./useBatchI2vLongRunner";
import type { BatchI2vLongRunnerState } from "./batchI2vLongRunner";

/** Server-side prompt length cap (`api/models.py`'s `prompt: str = Field(...,
 * max_length=2000)`). The Chain screen's own `PromptBar` already enforces it
 * with a `maxLength`, but each ROW adds its own text on top of that value, so
 * the COMBINED prompt has to be re-checked here — a row's prompt input is a new
 * input outside `PromptBar`'s protection. */
export const MAX_PROMPT_LENGTH = 2000;

/** A source-less chain needs at least 2 clips or the server 422s
 * (`validate_chain_constraints`, `api/models.py`). */
const MIN_CHAIN_CLIPS = 2;

/** Row states a `start()` would (re)process — the same set
 * `batchI2vLongRunner.ts`'s `UNFINISHED_STATS` uses, kept in sync by both
 * files' tests rather than by an import (the runner's copy is private). */
const UNFINISHED_STATS: ReadonlySet<I2vLongRow["stat"]> = new Set(["Waiting", "Failed", "Generating"]);

/** Rows 🔁 can put back into the queue. `Generating`/`Waiting` are already
 * queued, so the button is inert for them. */
const ROWS_ELIGIBLE_FOR_RESET: ReadonlySet<I2vLongRow["stat"]> = new Set(["Done", "Failed"]);

/**
 * Every reason Start is blocked, as a stable code. Rendered ONE LINE PER CODE
 * (`strings.batchI2vLong.blockReasons`) — an opaque "cannot start" would leave
 * an unattended overnight run undiagnosable.
 *
 * The Chain form's OWN failures are not in this union: they are surfaced
 * verbatim as {@link UseBatchI2vLongFormResult.chainBlockReasons}
 * (`ChainValidityReason` codes), so the batch never re-judges — and never
 * drifts from — the Chain screen's own gates.
 */
export type I2vLongBlockReason =
  | "imgDirMissing"
  | "outDirMissing"
  | "noRows"
  | "noRunnableRows"
  | "sourceVideoAttached"
  /** §1-16 長尺A2V: an audio track is attached to the Chain form. The batch
   * strips `source_audio` from every row, so this is a "you attached something
   * that will be ignored" block rather than a technical impossibility. */
  | "sourceAudioAttached"
  /** §1-15 参照動画: a reference video is attached to the Chain form. Same
   * "you attached something that will be ignored" block as the audio line
   * above — the batch strips `reference_video_id` and both strengths from
   * every row. */
  | "referenceVideoAttached"
  /** 素材（末尾）v2: end material is attached to the Chain form. Same
   * "you attached something that will be ignored" block as the three slots
   * above — the batch strips `end_source` from every row, so an attached
   * material would silently do nothing AND every delivered file would be
   * shorter than the Chain screen's own 予想出力 said. */
  | "endSourceAttached"
  | "clipsTooFew"
  | "promptEmpty"
  | "promptTooLong"
  | "unknownLoraTag"
  | "jobActive"
  /** The shared run lock (`shell/runLock.ts`) is currently held by the OTHER
   * batch — Batch A2V on the Create screen. Without this line the Start button
   * would simply do nothing: `runtime.run()` fails to acquire the lock, records
   * a `{started:false}` result and never leaves `idle`, which from the panel is
   * indistinguishable from a dead button. */
  | "lockedByOther";

/** Non-blocking advisories. Each is a "this will happen, and you probably did
 * not mean it" note rather than a gate. */
export interface I2vLongNotes {
  /** `seed >= 0`: every image gets the SAME seed. */
  seedFixed: boolean;
  /** Clip 0 has a prompt of its own. The strongest of these notes: each row's
   * image lands on clip 0, so a clip-0 override means the shared/row prompt
   * never reaches the part of the video the image actually controls — even in
   * `replace` mode. */
  clip0PromptOverride: boolean;
  /** Some clip AFTER clip 0 has a prompt of its own (案A: the shared/row
   * prompt never rewrites per-clip prompts). */
  otherClipPromptOverride: boolean;
  /** The Chain form has a start-frame image on clip 0; each row's own image
   * replaces it. */
  startFrameIgnored: boolean;
}

export interface I2vLongSummary {
  total: number;
  waiting: number;
  generating: number;
  done: number;
  failed: number;
}

export interface UseBatchI2vLongFormDeps {
  nativeBridge?: NativeBridge;
  /** Seam for {@link useLoras}, which supplies the known-LoRA-name set behind
   * the `unknownLoraTag` guard. Production omits it (the shared default client,
   * exactly like every other `useLoras()` call site). */
  apiClient?: ApiClient;
  /** Test-only pass-throughs to `BatchI2vLongRunner.start()` (the same pair
   * `useBatchI2vLongRunner`/`runtime.ts` already forward). Production omits
   * both; they exist so a test can run a whole batch to completion without
   * waiting on the 1s production poll interval. */
  pollIntervalMs?: number;
  jobBusyBackoffMs?: number;
}

export interface UseBatchI2vLongFormResult {
  // --- Folders ---
  imgDir: string | null;
  outDir: string | null;
  /** True while `outDir` is still the auto-derived `{imgDir's own name}
   * _i2vlong_out` sibling — flips to false the moment the user picks or types
   * their own, and never flips back (same rule as Batch A2V's `outDirIsAuto`). */
  outDirIsAuto: boolean;
  pickImgDir: (title?: string) => Promise<void>;
  pickOutDir: (title?: string) => Promise<void>;
  /** Commits a hand-typed folder path — wire to the text input's `onBlur`, NOT
   * `onChange` (an empty string clears the folder). A no-op when the trimmed
   * value already matches, so a bare focus/blur can never wipe scanned rows.
   * `setImgDir` probes existence with `fs.listFiles` (the bridge has no
   * `fs.exists`) and reports a failure through `scanError`; `setOutDir` does
   * not probe — the output folder is created at run time. */
  setImgDir: (value: string) => Promise<void>;
  setOutDir: (value: string) => Promise<void>;

  // --- Row prompts ---
  /** How EVERY row's own prompt combines with the Chain form's shared prompt.
   * One mode for the whole batch, exactly like Batch A2V's
   * `UseBatchFormResult.values.promptMode`. */
  promptMode: PromptMode;
  setPromptMode: (value: PromptMode) => void;
  /** Overwrites row `index`'s `prompt` cell verbatim (never `<lora:>`-tag
   * parsed) — wire straight to a controlled `<input>`'s `onChange`, like A2V's
   * `setRowPromptLocal`. In-memory only; there is nothing to persist to. */
  setRowPromptLocal: (index: number, value: string) => void;
  /** 📝: overwrites row `index`'s prompt with the Chain form's shared prompt
   * (`templatePreview.prompt`, already `<lora:>`-stripped) — the counterpart of
   * A2V's `copyCommonPromptToRow`, with the source prompt supplied from here
   * rather than by the caller (this panel already holds it). */
  copyChainPromptToRow: (index: number) => void;

  /** The row-neutral `POST /generate/chain` body every row is derived from, for
   * the CURRENT Chain form. Its `prompt` is the SHARED prompt only — a row's
   * own text is composed on top of it at submit time. Both the guards below and
   * `start()` read exactly this object, so what is judged is what is sent. */
  templatePreview: GenerateChainRequest;

  // --- Rows / scan ---
  rows: I2vLongRow[];
  isScanning: boolean;
  scanError: string | null;
  canScan: boolean;
  scan: () => Promise<void>;
  /** 🔁: puts a `Done`/`Failed` row back to `Waiting` (in memory — this batch
   * has no manifest file). A no-op for any other state. */
  resetRowToWaiting: (queue: number) => void;

  // --- Guards ---
  /** This panel's own block reasons, in a stable display order. */
  blockReasons: I2vLongBlockReason[];
  /** The Chain form's own `validityReasons`, passed through untouched (minus
   * the source-video codes while `sourceVideoAttached` is already saying it
   * better). Empty while the Chain form is valid. */
  chainBlockReasons: string[];
  /** `queue` numbers of the runnable rows whose COMPOSED prompt is empty —
   * the rows behind a `promptEmpty` reason. Empty otherwise. Surfaced so the
   * panel can name them instead of showing a generic line. */
  promptEmptyQueues: number[];
  /** `queue` numbers of the runnable rows whose COMPOSED prompt is over
   * {@link MAX_PROMPT_LENGTH} — the rows behind a `promptTooLong` reason. */
  promptTooLongQueues: number[];
  notes: I2vLongNotes;
  canStart: boolean;

  // --- Run ---
  runnerState: BatchI2vLongRunnerState;
  summary: I2vLongSummary;
  /** The row currently `Generating`, if any. */
  currentRow: I2vLongRow | null;
  start: () => void;
  /** Graceful stop: no further rows are submitted, but the image generating
   * right now finishes and is still saved (no DELETE is sent). */
  stop: () => void;
}

export function useBatchI2vLongForm(
  config: AppConfig,
  chain: ChainSnapshotSource,
  serverBusy: boolean,
  deps: UseBatchI2vLongFormDeps = {},
): UseBatchI2vLongFormResult {
  const nativeBridge = deps.nativeBridge ?? defaultBridge;

  // The runner is a module-level singleton (`runtime.ts`), so a `key`-remount
  // of the Chain screen must not look like "nothing is running". Seeding the
  // folders/rows from its snapshot with LAZY initializers re-attaches the whole
  // panel — folders included — on the remount's very first render.
  const runner = useBatchI2vLongRunner({ nativeBridge });
  const [imgDir, setImgDirState] = useState<string | null>(() => runner.imgDir);
  const [outDir, setOutDirState] = useState<string | null>(() => runner.outDir);
  const [outDirIsAuto, setOutDirIsAuto] = useState<boolean>(() => runner.outDir === null);
  const [rows, setRows] = useState<I2vLongRow[]>(() => runner.rows);

  const [promptMode, setPromptMode] = useState<PromptMode>("add");
  const [isScanning, setIsScanning] = useState(false);
  const [scanError, setScanError] = useState<string | null>(null);

  // Live row updates while a batch runs. `runtime.ts` replaces its snapshot
  // wholesale on every change, so this effect fires exactly once per row
  // transition. The empty-array guard keeps a fresh (never-run) runtime from
  // clobbering rows that were just scanned.
  const runnerRows = runner.rows;
  useEffect(() => {
    if (runnerRows.length === 0) return;
    setRows(runnerRows);
  }, [runnerRows]);

  const pickFolder = useCallback(
    async (title: string | undefined): Promise<string | null> => {
      try {
        const { folderPath } = await nativeBridge.request("ui.pickFolder", title ? { title } : {});
        return folderPath;
      } catch (err) {
        if (err instanceof BridgeError && err.code === "CANCELLED") return null;
        throw err;
      }
    },
    [nativeBridge],
  );

  /** Existence probe for a hand-typed INPUT folder (the bridge has no
   * `fs.exists`, so a zero-cost `fs.listFiles` stands in). An empty listing is
   * a valid empty folder, not an error. */
  const probeFolder = useCallback(
    async (folderPath: string) => {
      try {
        await nativeBridge.request("fs.listFiles", { folderPath });
      } catch (err) {
        setScanError(err instanceof BridgeError ? `${err.code}: ${err.message}` : String(err));
      }
    },
    [nativeBridge],
  );

  const pickImgDir = useCallback(
    async (title?: string) => {
      try {
        const folderPath = await pickFolder(title);
        if (!folderPath) return;
        setImgDirState(folderPath);
        setRows([]);
        setScanError(null);
        if (outDirIsAuto) setOutDirState(deriveI2vLongOutDir(folderPath));
      } catch (err) {
        setScanError(err instanceof BridgeError ? `${err.code}: ${err.message}` : String(err));
      }
    },
    [pickFolder, outDirIsAuto],
  );

  const pickOutDir = useCallback(
    async (title?: string) => {
      try {
        const folderPath = await pickFolder(title);
        if (!folderPath) return;
        setOutDirState(folderPath);
        setOutDirIsAuto(false);
      } catch (err) {
        setScanError(err instanceof BridgeError ? `${err.code}: ${err.message}` : String(err));
      }
    },
    [pickFolder],
  );

  const setImgDir = useCallback(
    async (value: string) => {
      const trimmed = value.trim();
      const next = trimmed === "" ? null : trimmed;
      if (next === imgDir) return;
      setImgDirState(next);
      setRows([]);
      setScanError(null);
      if (outDirIsAuto) setOutDirState(next ? deriveI2vLongOutDir(next) : null);
      if (next) await probeFolder(next);
    },
    [imgDir, outDirIsAuto, probeFolder],
  );

  const setOutDir = useCallback(
    async (value: string) => {
      const trimmed = value.trim();
      const next = trimmed === "" ? null : trimmed;
      if (next === outDir) return;
      setOutDirState(next);
      setOutDirIsAuto(false);
    },
    [outDir],
  );

  // --- The template snapshot -------------------------------------------------
  //
  // `chain.buildRequest` is a `useCallback` whose identity changes exactly when
  // one of the Chain form's request-relevant values does, which is what makes it
  // a correct (and cheap) memo dependency. `startFrameIgnored` is derived from
  // the SAME `base` the template was built from — building it here avoids a
  // second `buildRequest()` call and keeps the note in lockstep with the body
  // that will actually be sent.
  //
  // oxlint's `exhaustive-deps` wants `chain` here instead of `chain.buildRequest`
  // — deliberately not done: `chain` is a fresh object on EVERY Chain-screen
  // render, so depending on it would rebuild the template (and every guard that
  // reads it) on every keystroke anywhere on the screen. `buildRequest` is the
  // only member this memo actually calls, and it is a `useCallback` whose
  // identity already changes exactly when the request would. Same trade-off, and
  // the same suppressed warning, as `shell/useControlLoraNames.ts`.
  const chainBuildRequest = chain.buildRequest;
  const built = useMemo(() => {
    const base = chainBuildRequest();
    return {
      template: buildI2vLongTemplate({ base }),
      startFrameIgnored: (base.clips[0]?.conditioning_images?.length ?? 0) > 0,
    };
  }, [chainBuildRequest]);
  const templatePreview = built.template;

  // --- Scan ------------------------------------------------------------------

  const allowedExtensions = config.upload.allowed_image_extensions;
  const maxImageBytes = config.upload.max_image_size_mb * 1024 * 1024;

  const scan = useCallback(async () => {
    if (!imgDir) return;
    setIsScanning(true);
    setScanError(null);
    try {
      const listing = await nativeBridge.request("fs.listFiles", {
        folderPath: imgDir,
        extensions: [...allowedExtensions],
      });
      const files: ScannedImageFile[] = listing.files;
      // A re-scan always starts every row from scratch (stateless batch). Any
      // already-written output is protected by the runner's `noClobber`.
      setRows(scanImagesToRows(files, allowedExtensions, maxImageBytes));
    } catch (err) {
      setScanError(err instanceof BridgeError ? `${err.code}: ${err.message}` : String(err));
    } finally {
      setIsScanning(false);
    }
  }, [imgDir, nativeBridge, allowedExtensions, maxImageBytes]);

  // Row prompt editing (2026-07-30). Both are in-memory only and both use a
  // functional `setRows` update so rapid keystrokes can never race each other —
  // the same shape as Batch A2V's `setRowPromptLocal`/`copyCommonPromptToRow`.
  // Neither is ever routed through `parseLoraPrompt`: a `<lora:>` tag typed into
  // a row stays plain text (LoRAs come from the shared prompt only).
  const setRowPromptLocal = useCallback((index: number, value: string) => {
    setRows((prev) => prev.map((r, i) => (i === index ? { ...r, prompt: value } : r)));
  }, []);

  const chainPrompt = built.template.prompt;
  const copyChainPromptToRow = useCallback(
    (index: number) => {
      setRows((prev) => prev.map((r, i) => (i === index ? { ...r, prompt: chainPrompt } : r)));
    },
    [chainPrompt],
  );

  const resetRowToWaiting = useCallback((queue: number) => {
    setRows((prev) => {
      const target = prev.find((r) => r.queue === queue);
      if (!target || !ROWS_ELIGIBLE_FOR_RESET.has(target.stat)) return prev;
      return prev.map((r) => (r.queue === queue ? { ...r, stat: "Waiting" as const, error: "" } : r));
    });
  }, []);

  // --- Guards ----------------------------------------------------------------

  const runnableRows = useMemo(() => rows.filter((r) => UNFINISHED_STATS.has(r.stat)), [rows]);

  // The prompt each runnable row would actually be submitted with — the SAME
  // `composeBatchPrompt` the runner calls, over the same inputs, so the two can
  // never disagree about what is being judged. Both prompt guards below are
  // derived from this one pass (2026-07-30 row-prompt redesign: an empty or
  // over-long prompt is now a per-ROW property, not a property of the panel).
  const promptIssues = useMemo(() => {
    const empty: number[] = [];
    const tooLong: number[] = [];
    for (const row of runnableRows) {
      const composed = composeBatchPrompt(templatePreview.prompt, row.prompt, promptMode);
      if (composed.trim().length === 0) empty.push(row.queue);
      if (composed.length > MAX_PROMPT_LENGTH) tooLong.push(row.queue);
    }
    return { empty, tooLong };
  }, [runnableRows, templatePreview, promptMode]);

  // The known-LoRA-name set behind `unknownLoraTag`. A `<lora:typo>` in the
  // shared prompt would 404 EVERY row of an overnight run, so it is worth
  // catching up front — but only when the list actually loaded: while it is
  // loading (or failed), the guard passes, because blocking a whole batch on a
  // transient `GET /loras` failure would be far worse than the 404 it prevents.
  const lorasState = useLoras(deps.apiClient ? { apiClient: deps.apiClient } : {});
  const readyLoras = lorasState.status === "ready" ? lorasState.loras : null;
  const knownLoraNames = useMemo(
    () => (readyLoras === null ? null : new Set(readyLoras.map((l) => l.name))),
    [readyLoras],
  );
  const unknownLoraNames = useMemo(() => {
    if (knownLoraNames === null) return [];
    return (templatePreview.loras ?? []).map((l) => l.name).filter((name) => !knownLoraNames.has(name));
  }, [knownLoraNames, templatePreview]);

  const sourceVideoAttached = chain.mode === "v2v";
  // §1-16 長尺A2V: the exact audio counterpart. `=== true` rather than a truthy
  // read because the field is optional on `ChainSnapshotSource` (see its doc).
  const sourceAudioAttached = chain.hasSourceAudio === true;
  // §1-15 参照動画: the reference-video counterpart, same optional-field read.
  const referenceVideoAttached = chain.hasReferenceVideo === true;
  // 素材（末尾）v2: the end-slot counterpart, same optional-field read.
  const endSourceAttached = chain.hasEndSource === true;

  // The shared run lock's current holder. `useSyncExternalStore` (rather than a
  // subscribe effect) so the very first render already sees a lock Batch A2V
  // took before this panel mounted — the snapshot is a primitive that changes
  // only when the lock does. Our OWN hold is not a block: while this batch runs
  // the Start button is already disabled by `runner.state !== "idle"`, and
  // reporting `lockedByOther` for ourselves would be simply wrong.
  const runLockOwner = useSyncExternalStore(subscribeRunLock, getRunLockOwner, getRunLockOwner);
  const lockedByOther = runLockOwner !== null && runLockOwner !== RUN_LOCK_OWNER_BATCH_I2V_LONG;

  const blockReasons = useMemo<I2vLongBlockReason[]>(() => {
    const reasons: I2vLongBlockReason[] = [];
    if (imgDir === null) reasons.push("imgDirMissing");
    if (outDir === null) reasons.push("outDirMissing");
    if (rows.length === 0) reasons.push("noRows");
    else if (runnableRows.length === 0) reasons.push("noRunnableRows");
    if (sourceVideoAttached) reasons.push("sourceVideoAttached");
    if (sourceAudioAttached) reasons.push("sourceAudioAttached");
    if (referenceVideoAttached) reasons.push("referenceVideoAttached");
    if (endSourceAttached) reasons.push("endSourceAttached");
    if (templatePreview.clips.length < MIN_CHAIN_CLIPS) reasons.push("clipsTooFew");
    // A row whose composed prompt is empty would 422 (or generate noise) on its
    // own; before a scan there are no rows to judge, so the shared prompt alone
    // decides — that keeps "write a prompt" visible from the very first render
    // instead of only appearing after the folder is scanned.
    if (rows.length === 0 ? templatePreview.prompt.trim().length === 0 : promptIssues.empty.length > 0) {
      reasons.push("promptEmpty");
    }
    if (rows.length === 0 ? templatePreview.prompt.length > MAX_PROMPT_LENGTH : promptIssues.tooLong.length > 0) {
      reasons.push("promptTooLong");
    }
    if (unknownLoraNames.length > 0) reasons.push("unknownLoraTag");
    if (serverBusy) reasons.push("jobActive");
    if (lockedByOther) reasons.push("lockedByOther");
    return reasons;
  }, [imgDir, outDir, rows.length, runnableRows.length, sourceVideoAttached, sourceAudioAttached, referenceVideoAttached, endSourceAttached, templatePreview, promptIssues, unknownLoraNames, serverBusy, lockedByOther]);

  // The Chain form's own gates, expanded verbatim rather than collapsed into a
  // single "chain settings are invalid" line. The source-video codes are
  // dropped while `sourceVideoAttached` is showing, since that line already
  // says the same thing far more precisely.
  const chainBlockReasons = useMemo<string[]>(() => {
    if (chain.isValid) return [];
    let reasons = [...chain.validityReasons];
    if (sourceVideoAttached) reasons = reasons.filter((code) => !CHAIN_SOURCE_REASON_CODES.includes(code));
    // §1-16 長尺A2V: same rule, same reason — `sourceAudioAttached` already says
    // "remove the audio", so expanding the Chain form's audio gates underneath
    // it would ask the user to fix a track the batch is not going to send.
    if (sourceAudioAttached) reasons = reasons.filter((code) => !CHAIN_AUDIO_REASON_CODES.includes(code));
    // §1-15 参照動画: same rule once more — `referenceVideoAttached` already says
    // "remove the reference video", so the Chain form's reference gates would
    // only ask the user to fix material the batch is not going to send. (The
    // 128-grid line is deliberately NOT in that set — see its doc comment.)
    if (referenceVideoAttached) reasons = reasons.filter((code) => !CHAIN_REFERENCE_REASON_CODES.includes(code));
    // 素材（末尾）v2: same rule a fourth time — `endSourceAttached` already
    // says "remove the end material", so the Chain form's own end-slot gates
    // underneath it would only ask the user to fix something the batch drops.
    if (endSourceAttached) reasons = reasons.filter((code) => !CHAIN_END_SOURCE_REASON_CODES.includes(code));
    return reasons;
  }, [chain.isValid, chain.validityReasons, sourceVideoAttached, sourceAudioAttached, referenceVideoAttached, endSourceAttached]);

  const notes = useMemo<I2vLongNotes>(
    () => ({
      seedFixed: templatePreview.seed >= 0,
      clip0PromptOverride: (chain.clips[0]?.prompt ?? "").trim() !== "",
      otherClipPromptOverride: chain.clips.slice(1).some((clip) => clip.prompt.trim() !== ""),
      startFrameIgnored: built.startFrameIgnored,
    }),
    [templatePreview, chain.clips, built.startFrameIgnored],
  );

  const canStart =
    runner.state === "idle" && !isScanning && blockReasons.length === 0 && chainBlockReasons.length === 0;

  // --- Run -------------------------------------------------------------------

  const { pollIntervalMs, jobBusyBackoffMs } = deps;
  const start = useCallback(() => {
    if (!canStart || imgDir === null || outDir === null) return;
    // Snapshot by value: `templatePreview` is a fresh plain object from
    // `buildChainRequest`, and the runner holds THIS reference for the whole
    // run — later Chain-screen edits rebuild the memo into a NEW object and
    // cannot reach the batch in flight.
    runner.run(
      {
        imgDir,
        outDir,
        template: templatePreview,
        // Frozen with the template: flipping the radio mid-run must not change
        // how the rows still queued behind it are composed.
        settings: { promptMode },
        rows,
        ...(pollIntervalMs !== undefined ? { pollIntervalMs } : {}),
        ...(jobBusyBackoffMs !== undefined ? { jobBusyBackoffMs } : {}),
      },
      setRows,
    );
  }, [canStart, imgDir, outDir, templatePreview, promptMode, rows, runner, pollIntervalMs, jobBusyBackoffMs]);

  const summary = useMemo<I2vLongSummary>(() => {
    const counts: I2vLongSummary = { total: rows.length, waiting: 0, generating: 0, done: 0, failed: 0 };
    for (const r of rows) {
      if (r.stat === "Waiting") counts.waiting += 1;
      else if (r.stat === "Generating") counts.generating += 1;
      else if (r.stat === "Done") counts.done += 1;
      else if (r.stat === "Failed") counts.failed += 1;
    }
    return counts;
  }, [rows]);

  const currentRow = useMemo(() => rows.find((r) => r.stat === "Generating") ?? null, [rows]);

  return {
    imgDir,
    outDir,
    outDirIsAuto,
    pickImgDir,
    pickOutDir,
    setImgDir,
    setOutDir,
    promptMode,
    setPromptMode,
    setRowPromptLocal,
    copyChainPromptToRow,
    templatePreview,
    rows,
    isScanning,
    scanError,
    canScan: imgDir !== null && !isScanning && runner.state === "idle",
    scan,
    resetRowToWaiting,
    blockReasons,
    chainBlockReasons,
    promptEmptyQueues: promptIssues.empty,
    promptTooLongQueues: promptIssues.tooLong,
    notes,
    canStart,
    runnerState: runner.state,
    summary,
    currentRow,
    start,
    stop: runner.stop,
  };
}
