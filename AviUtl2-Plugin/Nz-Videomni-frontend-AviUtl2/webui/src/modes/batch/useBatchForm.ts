import { useCallback, useEffect, useMemo, useState, useSyncExternalStore } from "react";
import type { AppConfig } from "../../api/types";
import { bridge as defaultBridge, BridgeError } from "../../bridge";
import type { NativeBridge } from "../../bridge";
import { parseLoraPrompt } from "../../lora/loraTags";
import { isNagNegativeEmpty, NAG_OFF } from "../../shell/nagSettings";
import {
  ACCELERATION_DEFAULTS,
  ATTENTION_BACKEND_DEFAULT,
  BLOCK_SWAP_PREFETCH_SERVER_DEFAULT,
  FUSED_GGUF_DEQUANT_KERNEL_SERVER_DEFAULT,
  KEEP_RESIDENT_EMBEDDINGS_SERVER_DEFAULT,
  KEEP_RESIDENT_SERVER_DEFAULT,
  VAE_MODE_DEFAULT,
} from "../../shell/accelerationSettings";
import { RUN_LOCK_OWNER_BATCH_A2V, getRunLockOwner, subscribeRunLock } from "../../shell/runLock";
import type { NagSettings } from "../../shell/nagSettings";
import type { AccelerationSettings } from "../../shell/accelerationSettings";
import { rawFramesForAudio, suggestFramesForAudio } from "../chained/chainUtils";
import { MIN_HEIGHT, MIN_WIDTH } from "../single/defaultConfig";
import { isDimensionOnGrid } from "../single/paramUtils";
import type { UseKeyframesResult } from "../single/useKeyframes";
import type { BatchRunnerSettings, BatchRunnerState } from "./batchRunner";
import { IMAGE_SHARED, rejudgeRows, scanToRows } from "./manifestMerge";
import type { BatchRow, ScannedFile } from "./manifestMerge";
import type { RunBatchParams } from "./useBatchRunner";
import { useBatchRunner } from "./useBatchRunner";

// `manifestMerge.ts` doesn't export a `PromptMode` type of its own (that lives
// in `buildA2vChainPayload.ts`, as `BatchRunnerSettings.promptMode`'s type) —
// re-declared locally to avoid a needless extra import for one type alias.
export type PromptMode = "add" | "replace";

export interface BatchFormValues {
  width: number;
  height: number;
  frameRate: number;
  seed: number;
  chunkedUpsample: boolean;
  promptMode: PromptMode;
}

/** Narrower slice of {@link BatchFormValues} shared with Create's own
 * generation-values shape (webui-U1: a common type later units can use to
 * align Batch's minimal width/height/frameRate/seed/numFrames fields with
 * Create's equivalent, without pulling in Batch's chunkedUpsample/promptMode
 * extras). `numFrames` is the Create-form DURATION value, injected as the
 * per-row Skip cap (`scanToRows`/`rejudgeRows`'s `maxFrames`) so a batch never
 * submits a clip longer than the current DURATION setting. */
export interface BatchGenerationValues {
  width: number;
  height: number;
  frameRate: number;
  seed: number;
  numFrames: number;
}

export interface BatchFormLimits {
  minWidth: number;
  maxWidth: number;
  minHeight: number;
  maxHeight: number;
}

export interface BatchSummary {
  total: number;
  waiting: number;
  generating: number;
  done: number;
  failed: number;
  skip: number;
}

export interface UseBatchFormDeps {
  nativeBridge?: NativeBridge;
  /** NAG (2026-07-28): the shared Negative Prompt accordion's settings,
   * owned by `shell/AppShell.tsx` (single `NagSettings` object, D1) — Batch
   * has no NAG UI of its own; it silently inherits whatever Create's
   * accordion currently holds, exactly like it already does for
   * width/height/frameRate/seed via `generationValues`. Defaults to the
   * frozen `NAG_OFF` sentinel when omitted (every pre-existing unit test). */
  nag?: NagSettings | undefined;
  /** Acceleration (2026-07-31): the Settings panel's shared
   * attention-backend choice, owned by `shell/AppShell.tsx` — Batch
   * inherits it silently, exactly like `nag` above. Defaults to the frozen
   * `ACCELERATION_DEFAULTS` sentinel when omitted. */
  acceleration?: AccelerationSettings | undefined;
  /** §1-7 相互ロック 第2段（2026-07-31 オーナー実機報告）: `JobsContext.serverBusy`
   * — the backend is occupied, either by a generation (a single Generate, or
   * ANOTHER batch's row) holding its single job slot or by a model load this
   * WebUI started, so this batch may not start.
   *
   * Batch i2v-long has had this gate since day one (`jobActive`); Batch A2V
   * relied on the shared run lock alone, which is not enough: the lock is
   * browser-volatile module state, so it is absent after a reload (while the
   * job it was protecting keeps running server-side), after a mid-run remount
   * of the Create screen, and for any job started outside the two batch panels.
   * In every one of those cases Start stayed enabled and pressing it either
   * did nothing at all (the lock refused it silently) or 409-spammed the
   * backend. Defaults to false so every pre-existing unit test is unaffected. */
  serverBusy?: boolean;
}

export interface UseBatchFormResult {
  // Folders (spec §1/§7.1).
  wavDir: string | null;
  imgDir: string | null;
  outDir: string | null;
  /** True while `outDir` is still the auto-derived
   * `{wavDir's own name}_a2v_out` sibling (spec §7.1) — flips to false the
   * moment the user explicitly picks their own output folder, and never
   * flips back until the audio folder itself changes again. */
  outDirIsAuto: boolean;
  pickWavDir: (title?: string) => Promise<void>;
  pickImgDir: (title?: string) => Promise<void>;
  pickOutDir: (title?: string) => Promise<void>;
  /** U4 (spec §7.1 hand-typed paths): commits a manually-typed folder path
   * (wire to the text input's `onBlur`, NOT `onChange` — an empty string
   * clears the folder). Same side effects as the matching `pick*` variant:
   * `setWavDir` clears rows/scanError and (while auto) re-derives `outDir`;
   * `setImgDir` drops the stale image file names; `setOutDir` flips
   * `outDirIsAuto` off. On confirm each probes the path with `fs.listFiles`
   * (bridge has no `fs.exists`) and surfaces a not-found scan error. A no-op
   * when the typed value already matches the current folder (so a focus/blur
   * without an edit never clobbers scanned rows). `outDir` is not probed — it
   * may legitimately not exist yet (created at run time). */
  setWavDir: (value: string) => Promise<void>;
  setImgDir: (value: string) => Promise<void>;
  setOutDir: (value: string) => Promise<void>;

  // Generation settings. width/height/frameRate/seed are NO LONGER owned here
  // (U4): they come from the Create form via the `generationValues` param and
  // are surfaced back through `values` for `start()`/`scan()`. Only
  // `chunkedUpsample`/`promptMode` remain batch-owned.
  values: BatchFormValues;
  limits: BatchFormLimits;
  setChunkedUpsample: (value: boolean) => void;
  setPromptMode: (value: PromptMode) => void;
  /** U4 guard 1: false when the shared width/height aren't valid multiples of
   * 64 within bounds — a plain Create-form off-grid entry would 422 every row.
   * Blocks `canStart`; surfaced so `BatchSection` can show the specific
   * reason. */
  resolutionValid: boolean;
  /** U4 guard 2 (informational only now): an information hint that the Create
   * form's FPS or DURATION drifted from the values in effect at the last
   * `scan()` (both were baked into each row's frame count / over-cap Skip).
   * NO LONGER blocks `canStart` — `start()` re-judges every runnable row at
   * the current FPS and DURATION cap via `rejudgeRows`, so any drift is fixed
   * automatically at run time. Field name kept as `fpsMismatch` for API
   * stability; its semantics now cover DURATION too. Surfaced purely so
   * `BatchSection` can show an informational hint that frames will be
   * recomputed on start. */
  fpsMismatch: boolean;

  /** Batch A2V Shared spec (2026-07-18): the SHARED `useKeyframes` instance the
   * Create screen owns (its KEYFRAMES panel), passed in — Batch has no keyframe
   * panel of its own. A `Shared`-image row's `conditioning_images` is derived
   * from this at `start()` time (see `start` below): its first (lowest-
   * `frame_idx`) ready image, forced to `frame_idx=0`. `keyframes.isUploading`
   * blocks `canStart` below. Re-exported here as a pass-through so
   * consumers/tests can read the same instance off the form result. */
  keyframes: UseKeyframesResult;
  /** True when at least one runnable row's `image` column is the `Shared`
   * sentinel but the Create-owned `keyframes` panel has no ready image yet.
   * Batch A2V Shared spec (2026-07-18): the gate is now simply "at least one
   * ready image exists" (any slider position) — the leading image is force-
   * pinned to frame 0 at run time, so a `frame_idx===0` entry is no longer
   * required. Blocks `canStart` below; surfaced standalone so `BatchSection`
   * can render the specific warning rather than a generic "can't start". */
  sharedKeyframeMissing: boolean;
  /** NAG (2026-07-28): true when the (silently-inherited) NAG accordion is
   * enabled with a blank negative-prompt body — the same
   * `isNagNegativeEmpty` gate Create/Chain apply, folded into `canStart`
   * below. Surfaced standalone so `BatchSection` can render the specific
   * warning banner (D7: Batch uses a plain warning banner, not
   * `GenerateReasonsNote`). */
  nagInvalid: boolean;

  // Rows / scan (spec §6).
  rows: BatchRow[];
  isScanning: boolean;
  scanError: string | null;
  canScan: boolean;
  scan: () => Promise<void>;
  /** "Waitingに戻す" row action (spec §4's manual regenerate rule): only
   * `Done`/`Failed`/`Skip` rows are eligible; `output`/`error` are left
   * untouched. A no-op for a `Waiting`/`Generating` row. In-memory only
   * (stateless batch, owner decision 2026-07-18). */
  resetRowToWaiting: (queue: number) => void;
  /** N5 "A1: プロンプト直接編集" — overwrites row `index`'s `prompt` cell with
   * `value` verbatim (never `<lora:>`-tag-parsed) in the in-memory row list.
   * Wired directly to the prompt `<input>`'s `onChange` (a controlled input) —
   * there is no CSV to persist to (stateless batch, owner decision
   * 2026-07-18). */
  setRowPromptLocal: (index: number, value: string) => void;
  /** N5 "A2: 共通プロンプト流し込み" — overwrites row `index`'s `prompt`
   * cell with `commonPrompt` (the Create screen's shared `PromptBar` value)
   * verbatim (a discrete button press). */
  copyCommonPromptToRow: (index: number, commonPrompt: string) => void;
  /** N5 "A3: 行別画像割当" — the per-row `<select>`'s option list:
   * `IMAGE_SHARED` first, then the image folder's file names as of the last
   * `scan()`, sorted by name. Just `[IMAGE_SHARED]` while `imgDir` is unset. */
  imageOptions: string[];
  /** N5 "A3: 行別画像割当" — assigns row `index`'s `image` column (an empty
   * string normalizes to `IMAGE_SHARED`) in memory, unless the resolved value
   * already matches the row's current `image` (no-op). */
  updateRowImage: (index: number, imageName: string) => void;

  // Run controls.
  runnerState: BatchRunnerState;
  summary: BatchSummary;
  /** The row currently `Generating`, if any — for the "current row" progress
   * readout (task brief: "進捗表示（現在行 queue…）"). */
  currentRow: BatchRow | null;
  canStart: boolean;
  /** §1-7 相互ロック: true while the OTHER batch feature (Batch i2v-long, on
   * the Chain screen) holds the shared run lock — `shell/runLock.ts`. Blocks
   * `canStart` and is surfaced standalone so `BatchSection` can explain WHY
   * Start is disabled rather than just greying it out. False while this panel
   * holds the lock itself (that case is already covered by `runnerState`). */
  lockedByOther: boolean;
  /** §1-7 相互ロック 第2段: true while the backend is occupied — a job in its
   * single slot, or a model load (`UseBatchFormDeps.serverBusy`) — the same
   * gate Batch i2v-long's `jobActive` block reason applies. The name is the
   * block-reason code, which is shared with the i18n key and left alone.
   * Blocks `canStart` and is surfaced
   * standalone so `BatchSection` can explain the disabled button. Unlike
   * `lockedByOther` this survives a page reload, because it is derived from
   * the SERVER's job list rather than from browser-local lock state. */
  jobActive: boolean;
  start: () => void;
  stop: () => void;
}

/** Joins a folder path with a file name using a backslash, unless `dir`
 * already ends with a path separator — same convention as
 * `batchRunner.ts`'s own (unexported) `joinPath`; duplicated here rather
 * than imported since neither file exports it and both are tiny/pure. */
function joinPath(dir: string, name: string): string {
  const sep = dir.endsWith("\\") || dir.endsWith("/") ? "" : "\\";
  return `${dir}${sep}${name}`;
}

/** Splits a folder path into its parent folder and its own last-segment
 * name, tolerating a trailing separator. `parent === ""` means `path` had no
 * separator at all (a bare name, or a drive root like `C:`). */
function splitDir(path: string): { parent: string; base: string } {
  const trimmed = path.replace(/[\\/]+$/, "");
  const idx = Math.max(trimmed.lastIndexOf("\\"), trimmed.lastIndexOf("/"));
  if (idx === -1) return { parent: "", base: trimmed };
  return { parent: trimmed.slice(0, idx), base: trimmed.slice(idx + 1) };
}

/** Auto-derives the batch output folder (spec §7.1's "既定（自動）" mode):
 * a sibling of `wavDir` named `{wavDir's own folder name}_a2v_out`. */
function deriveOutDir(wavDir: string): string {
  const { parent, base } = splitDir(wavDir);
  const name = `${base}_a2v_out`;
  return parent ? joinPath(parent, name) : name;
}

const ROWS_ELIGIBLE_FOR_RESET: ReadonlySet<BatchRow["stat"]> = new Set(["Done", "Failed", "Skip"]);

/** Image file extensions considered valid choices for a row's `image` column
 * (N5 "A3: 行別画像割当") — passed as `fs.listFiles`'s `extensions` filter
 * when populating the per-row image `<select>`'s option list at scan time. */
const IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"];

/**
 * Owns the batch A2V panel's whole browser-side state: folder selection
 * (spec §1/§7.1), the folder-scan -> in-memory row list (spec §6; stateless —
 * no CSV persistence, owner decision 2026-07-18), the batch's own minimal
 * generation settings, and the run/stop controls (via
 * {@link useBatchRunner}). Mirrors `gradio_ui/ui.py`'s batch accordion
 * wiring, adapted to this app's hook-per-screen-section convention (see
 * `useChainForm`/`useGenerationForm` for the sibling patterns this follows).
 *
 * Deliberate scope-narrowing vs. the Gradio reference (see this sprint's
 * report for the full list): the common prompt is the app's single shared
 * `PromptBar` value (`prompt` param, exactly like Gradio's batch closure
 * reusing the Generate tab's own `prompt` component) rather than a
 * batch-local textbox.
 *
 * U4: width/height/frame_rate/seed are NO LONGER owned here — they are the
 * Create form's own values, passed in as `generationValues` and surfaced back
 * through `values` (merged with the batch-owned `chunkedUpsample`/`promptMode`)
 * for `start()`/`scan()`. This mirrors Gradio, where the batch closure reuses
 * the Generate tab's resolution/fps/seed components directly rather than
 * duplicating them. Because those values are shared, three guards protect the
 * overnight run from Create-side state the batch can't itself constrain:
 * `resolutionValid` (the free-typed Create width/height must land on the 64
 * grid), `fpsMismatch` (the fps baked into scanned rows must still match), and
 * (in `BatchSection`) an IC-LoRA-active warning.
 *
 * Batch A2V Shared spec (2026-07-18): the Generate-tab's common i2v keyframe
 * ("Shared" image rows) is resolved from the SHARED `useKeyframes` instance
 * the Create screen owns — passed in as `keyframes` rather than owned here, so
 * Batch has no keyframe panel of its own (the Gradio reference has none
 * either). At `start()` time a `Shared` row takes only that panel's FIRST
 * (lowest-`frame_idx`) ready image, forces it to `frame_idx=0` (the leading
 * frame) while preserving its `strength`, and ignores every later keyframe.
 * That length-0-or-1 array is snapshotted into `sharedConditioningImages` and
 * threaded through to `batchRunner.ts`'s `resolveConditioning`. See
 * `keyframes`/`sharedKeyframeMissing`'s doc comments on
 * {@link UseBatchFormResult} above.
 */
export function useBatchForm(
  config: AppConfig,
  prompt: string,
  generationValues: BatchGenerationValues,
  keyframes: UseKeyframesResult,
  deps: UseBatchFormDeps = {},
): UseBatchFormResult {
  const nativeBridge = deps.nativeBridge ?? defaultBridge;
  // NAG (2026-07-28): mirrors `nativeBridge`'s optional-dep pattern — the
  // caller (`AppShell`) owns the actual `NagSettings` state; Batch just
  // reads it (silent inheritance, no accordion of its own).
  const nag = deps.nag ?? NAG_OFF;
  // Acceleration (2026-07-31): same optional-dep shape as `nag`.
  const acceleration = deps.acceleration ?? ACCELERATION_DEFAULTS;
  // §1-7 相互ロック 第2段: the server-side "is the backend occupied right now?"
  // gate (see `UseBatchFormDeps.serverBusy`). Same default-off shape as `nag`.
  const serverBusy = deps.serverBusy ?? false;

  // バックエンドの `Docs/PENDING_TASKS_CLOSED.md` §3-47-02（2026-09-02）: ランナーはモジュールレベルのシングルトン
  // (`runtime.ts`) なので、走行中にCreate画面が`key`リマウントされても走行は
  // 生き続ける。このフックは購読するだけ——だから`useBatchRunner`は他の
  // `useState`より前で呼び、下の遅延初期化子がその場でスナップショットを
  // 読めるようにしている。
  const batchRunner = useBatchRunner({ nativeBridge });

  // 復元するのは「走行中のときだけ」（設計判断1）。非走行時のリマウントは従来
  // どおり白紙から始める——A2Vはステートレスバッチ（行はメモリのみ・CSVなし、
  // オーナー裁定2026-07-18）で、走り終わった後の入力を勝手に復活させるのは
  // その設計に反する。走行中は行編集がUI無効（`BatchSection`の`disabled`）
  // なので、凍結した選択肢やfps表示と実際の走行が食い違うこともない。
  //
  // Restore ONLY while a run is in flight: a remount during a run must
  // re-attach to it (folders, rows, and the scan-derived state the table
  // renders from), while a remount with nothing running keeps the pre-existing
  // blank-slate behavior. Lazy initializers, so this happens on the remount's
  // very first render — no effect, no flash of an empty panel.
  const restoring = batchRunner.state !== "idle";
  const [wavDir, setWavDirState] = useState<string | null>(() => (restoring ? batchRunner.wavDir : null));
  const [imgDir, setImgDirState] = useState<string | null>(() => (restoring ? batchRunner.imgDir : null));
  const [outDir, setOutDirState] = useState<string | null>(() => (restoring ? batchRunner.outDir : null));
  // 復元しない（設計判断3）: `outDirIsAuto`は「利用者が出力先を自分で選んだか」
  // という意思であって、走行入力から導出できるものではない。i2v-long側の
  // `runner.outDir === null`という初期化子をそのまま写すと、一度走行した後は
  // 自動導出が恒久的に死に、音声フォルダを変えても出力先が前回のままになる
  // （＝別バッチの成果物が混ざる）。
  const [outDirIsAuto, setOutDirIsAuto] = useState(true);

  const limits: BatchFormLimits = useMemo(
    () => ({
      minWidth: MIN_WIDTH,
      maxWidth: config.limits.max_width,
      minHeight: MIN_HEIGHT,
      maxHeight: config.limits.max_height,
    }),
    [config.limits.max_width, config.limits.max_height],
  );

  // Only `chunkedUpsample`/`promptMode` are batch-owned now; width/height/
  // frameRate/seed live on `generationValues` and are merged into `values`
  // (below) so `BatchSection`/`start()`/`scan()` still read a single object.
  const [own, setOwn] = useState<{ chunkedUpsample: boolean; promptMode: PromptMode }>(() => ({
    chunkedUpsample: true,
    promptMode: "add",
  }));

  const values: BatchFormValues = useMemo(
    () => ({
      width: generationValues.width,
      height: generationValues.height,
      frameRate: generationValues.frameRate,
      seed: generationValues.seed,
      chunkedUpsample: own.chunkedUpsample,
      promptMode: own.promptMode,
    }),
    [generationValues, own],
  );

  const setChunkedUpsample = useCallback((value: boolean) => setOwn((prev) => ({ ...prev, chunkedUpsample: value })), []);
  const setPromptMode = useCallback((value: PromptMode) => setOwn((prev) => ({ ...prev, promptMode: value })), []);

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

  const [rows, setRows] = useState<BatchRow[]>(() => (restoring ? batchRunner.rows : []));
  const [scanError, setScanError] = useState<string | null>(null);
  // N5 A3: the image folder's file names, fetched once per `scan()` call and
  // sorted here (contract: "native's result order is unspecified — callers
  // must sort `files` themselves"). `imageOptions` below derives the actual
  // per-row `<select>` option list, always leading with `IMAGE_SHARED` and
  // collapsing to just that sentinel whenever `imgDir` is unset — so clearing
  // the image folder narrows the choices immediately even before the next scan.
  const [imageFileNames, setImageFileNames] = useState<string[]>(() => (restoring ? batchRunner.imageFileNames : []));
  // U4 guard 2: the FPS in effect at the last successful `scan()` — baked into
  // each row's frame count. Compared against the live Create-form FPS below;
  // reset to null whenever the row set is invalidated (folder change).
  const [scannedFps, setScannedFps] = useState<number | null>(() => (restoring ? batchRunner.scannedFps : null));
  // Paired with `scannedFps` (same lifecycle): the DURATION (`numFrames`) Skip
  // cap in effect at the last successful `scan()` — baked into each row's
  // over-cap Skip judgment. Compared against the live Create-form DURATION
  // below to widen `fpsMismatch` to also cover a DURATION change; reset to
  // null on the same folder-change invalidations as `scannedFps`.
  const [scannedMaxFrames, setScannedMaxFrames] = useState<number | null>(() =>
    restoring ? batchRunner.scannedMaxFrames : null,
  );

  // バックエンドの `Docs/PENDING_TASKS_CLOSED.md` §3-47-02: 走行中に「再接続した」マウントへ行の更新を流し続けるための経路。
  // `runtime.ts`は変化のたびにスナップショットを丸ごと差し替えるので、この効果は
  // 行の遷移1回につきちょうど1回発火する。
  //
  // 走行へ再接続したマウント（＝`restoring`だったマウント）だけが対象なのが要点。
  // 自分で`start()`したマウントは`setRows`を直接シンクとして渡しているのでこの
  // 効果を必要とせず、逆にここを無条件にすると「走り終わった後に白紙で始まった
  // マウント」が前回の走行の行だけを拾ってしまう——フォルダ欄は空なのに前回の
  // 行だけ表に残る、という設計判断1に反する半端な状態になる。
  // 判定はマウント時に凍結する（走行が終わって`state`がidleへ落ちた瞬間に効果を
  // 切ると、最後の行更新を取りこぼしうるため）。
  const [attachedToRun] = useState(restoring);
  const runnerRows = batchRunner.rows;
  useEffect(() => {
    if (!attachedToRun) return;
    if (runnerRows.length === 0) return;
    setRows(runnerRows);
  }, [attachedToRun, runnerRows]);

  const pickWavDir = useCallback(
    async (title?: string) => {
      try {
        const folderPath = await pickFolder(title);
        if (!folderPath) return;
        setWavDirState(folderPath);
        setRows([]);
        setScanError(null);
        setScannedFps(null);
        setScannedMaxFrames(null);
        if (outDirIsAuto) setOutDirState(deriveOutDir(folderPath));
      } catch (err) {
        setScanError(err instanceof BridgeError ? `${err.code}: ${err.message}` : String(err));
      }
    },
    [pickFolder, outDirIsAuto],
  );

  const pickImgDir = useCallback(
    async (title?: string) => {
      try {
        const folderPath = await pickFolder(title);
        if (folderPath) {
          setImgDirState(folderPath);
          // M1 remediation: drop the previous folder's file names immediately
          // — `imageOptions` below collapses to just `[IMAGE_SHARED]` until
          // the next `scan()` repopulates `imageFileNames` for the NEW
          // folder, so a row's `<select>` can never keep offering (and a
          // user can never commit) a file name that only existed in the
          // folder that was just replaced.
          setImageFileNames([]);
        }
      } catch (err) {
        setScanError(err instanceof BridgeError ? `${err.code}: ${err.message}` : String(err));
      }
    },
    [pickFolder],
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

  // U4: onBlur existence probe for a hand-typed input folder path (the bridge
  // has no `fs.exists`, so a zero-cost `fs.listFiles` stands in). A thrown
  // error (e.g. FOLDER_NOT_FOUND on native) surfaces as a scan error; an empty
  // listing is a valid empty folder, not an error.
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

  // U4: hand-typed folder setters (onBlur-confirmed). Each mirrors its `pick*`
  // sibling's side effects, then probes existence. A no-op when the trimmed
  // value already equals the current folder, so a bare focus/blur can't wipe
  // scanned rows.
  const setWavDir = useCallback(
    async (value: string) => {
      const trimmed = value.trim();
      const next = trimmed === "" ? null : trimmed;
      if (next === wavDir) return;
      setWavDirState(next);
      setRows([]);
      setScanError(null);
      setScannedFps(null);
      setScannedMaxFrames(null);
      if (outDirIsAuto) setOutDirState(next ? deriveOutDir(next) : null);
      if (next) await probeFolder(next);
    },
    [wavDir, outDirIsAuto, probeFolder],
  );

  const setImgDir = useCallback(
    async (value: string) => {
      const trimmed = value.trim();
      const next = trimmed === "" ? null : trimmed;
      if (next === imgDir) return;
      setImgDirState(next);
      setImageFileNames([]);
      if (next) await probeFolder(next);
    },
    [imgDir, probeFolder],
  );

  // The output folder is NOT probed — it may legitimately not exist yet
  // (`batchRunner` creates it at run time). Typing a path just flips
  // `outDirIsAuto` off, exactly like `pickOutDir`.
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

  const [isScanning, setIsScanning] = useState(false);

  const scan = useCallback(async () => {
    if (!wavDir) return;
    setIsScanning(true);
    setScanError(null);
    try {
      const listing = await nativeBridge.request("fs.listFiles", { folderPath: wavDir, withAudioDuration: true });
      const scannedFiles: ScannedFile[] = listing.files;
      const fps = generationValues.frameRate;
      const maxFrames = generationValues.numFrames;
      const fresh = scanToRows(scannedFiles, {
        fps,
        framesFor: suggestFramesForAudio,
        rawFramesFor: rawFramesForAudio,
        maxFrames,
      });

      if (imgDir) {
        const imgListing = await nativeBridge.request("fs.listFiles", {
          folderPath: imgDir,
          extensions: IMAGE_EXTENSIONS,
        });
        setImageFileNames(imgListing.files.map((f) => f.name).sort());
      } else {
        setImageFileNames([]);
      }

      // Stateless batch (owner decision 2026-07-18): a scan simply replaces
      // the in-memory row list with a fresh listing — there is no manifest CSV
      // to read back or merge against, so a re-scan always starts every row
      // from scratch. (Existing output files are never overwritten: the runner
      // downloads with `noClobber`, so a re-run writes a numbered sibling.)
      setRows(fresh);
      // U4 guard 2: remember the FPS and DURATION cap baked into these rows'
      // frame counts / over-cap Skips so a later Create-side FPS or DURATION
      // change can be detected before `start()`.
      setScannedFps(fps);
      setScannedMaxFrames(maxFrames);
    } catch (err) {
      setScanError(err instanceof BridgeError ? `${err.code}: ${err.message}` : String(err));
    } finally {
      setIsScanning(false);
    }
    // M1 (adversarial review): `generationValues.numFrames` MUST be in the dep
    // list — without it `scan` would close over a stale DURATION cap and keep
    // scanning against the old `maxFrames` after the Create form changed it.
  }, [wavDir, imgDir, generationValues.frameRate, generationValues.numFrames, nativeBridge]);

  const resetRowToWaiting = useCallback(
    (queue: number) => {
      const target = rows.find((r) => r.queue === queue);
      if (!target || !ROWS_ELIGIBLE_FOR_RESET.has(target.stat)) return;
      const next = rows.map((r) => (r.queue === queue ? { ...r, stat: "Waiting" as const } : r));
      setRows(next);
    },
    [rows],
  );

  // N5 A1: raw in-place edit of a single row's `prompt` cell, called on EVERY
  // keystroke (the prompt `<input>`'s `onChange`, a controlled input).
  // Deliberately NEVER routed through `parseLoraPrompt` — spec §8 excludes a
  // row's own `prompt` column from `<lora:>` tag parsing, so the string is
  // stored verbatim. Updates the in-memory `rows` (and so the on-screen value)
  // immediately; there is nothing else to persist to (stateless batch, owner
  // decision 2026-07-18). Uses a functional `setRows` update (rather than
  // closing over `rows`) so rapid keystrokes can never race each other.
  const setRowPromptLocal = useCallback((index: number, value: string) => {
    setRows((prev) => prev.map((r, i) => (i === index ? { ...r, prompt: value } : r)));
  }, []);

  // N5 A2: overwrites row `index`'s prompt with the Create screen's shared
  // `commonPrompt` verbatim (also never `parseLoraPrompt`-parsed) — a discrete
  // button press. In-memory only.
  const copyCommonPromptToRow = useCallback(
    (index: number, commonPrompt: string) => {
      setRows((prev) => prev.map((r, i) => (i === index ? { ...r, prompt: commonPrompt } : r)));
    },
    [],
  );

  // N5 A3: assigns row `index`'s `image` column from its `<select>`. An
  // empty string (a blank/committed-nothing selection) normalizes to the
  // `IMAGE_SHARED` sentinel. No-op (skips `setRows` entirely) when the
  // resolved value already equals the row's current `image`.
  const updateRowImage = useCallback(
    (index: number, imageName: string) => {
      const normalized = imageName === "" ? IMAGE_SHARED : imageName;
      const target = rows[index];
      if (!target || target.image === normalized) return;
      setRows((prev) => prev.map((r, i) => (i === index ? { ...r, image: normalized } : r)));
    },
    [rows],
  );

  // N5 A3: the per-row `<select>`'s option list — `IMAGE_SHARED` always
  // first, followed by the image folder's file names (sorted at scan time).
  // Collapses to just the sentinel whenever no image folder is set, even if
  // a previous scan's `imageFileNames` is still around (spec: "imgDir が未
  // 設定なら選択肢は Shared のみ").
  const imageOptions = useMemo<string[]>(
    () => (imgDir ? [IMAGE_SHARED, ...imageFileNames] : [IMAGE_SHARED]),
    [imgDir, imageFileNames],
  );

  // --- §1-7 相互ロック（shell/runLock.ts） ---------------------------------
  //
  // Batch A2V and Batch i2v-long may never run at the same time (both drive the
  // backend's single job slot). The lock is owner-token based BECAUSE both
  // panels are permanently mounted: a plain "release when my runner is idle"
  // effect would fire on this panel's very first mount and free a lock the
  // OTHER panel is holding.
  //
  // 取得も返却もこのフックではなく`runtime.ts`（モジュールレベル・シングルトン）
  // の責務: 取得は`start()`の前、返却は実行Promiseの`.then`。Reactのeffectでは
  // ないので、走行中にこのフックがアンマウントされてもロックは必ず戻る。ここに
  // 残っているのは「相手が握っているか」を読むための購読だけ。
  //
  // 2026-09-02（バックエンドの `Docs/PENDING_TASKS_CLOSED.md` §3-47-02）: ランナー実体も`runtime.ts`へ移したので、走行中の
  // リマウントで走行が孤児化することもなくなった（リマウント後のパネルはその
  // まま走行へ再接続し、Stopも行の進捗も生きている）。
  const lockOwner = useSyncExternalStore(subscribeRunLock, getRunLockOwner, getRunLockOwner);
  const lockedByOther = lockOwner !== null && lockOwner !== RUN_LOCK_OWNER_BATCH_A2V;

  const start = useCallback(() => {
    if (!wavDir || !outDir) return;
    // M-1 start-time re-judgment (mirrors `batch.py`'s `_plan_rejudgement`):
    // recompute every runnable row's frame count / 481-frame Skip at the
    // CURRENT Create-form fps, so a row scanned at one fps but started at
    // another is corrected in place instead of silently 422-ing the whole run.
    // This is the whole reason the stale-fps `fpsMismatch` no longer blocks
    // `canStart` (see below). `setRows(rejudged)` runs UNCONDITIONALLY — even
    // when nothing is runnable, so a row that just fell back to `Skip` (e.g.
    // over-cap at the higher fps) is immediately visible in the table — and
    // the run itself MUST be handed `rejudged`, never the closure's stale
    // `rows` (passing `rows` would re-introduce the original over-cap bug the
    // re-judgment exists to fix).
    const rejudged = rejudgeRows(rows, {
      fps: generationValues.frameRate,
      framesFor: suggestFramesForAudio,
      rawFramesFor: rawFramesForAudio,
      maxFrames: generationValues.numFrames,
    });
    setRows(rejudged);
    // §1-7 相互ロック 第2段: never submit into a busy job slot, even if this is
    // reached with a stale closure or by a direct call — the run lock cannot
    // catch a job that no batch panel started (or one that outlived a reload).
    if (serverBusy) return;
    const { strippedPrompt, loras } = parseLoraPrompt(prompt);
    const settings: BatchRunnerSettings = {
      promptCommon: strippedPrompt,
      promptMode: own.promptMode,
      width: generationValues.width,
      height: generationValues.height,
      frameRate: generationValues.frameRate,
      seed: generationValues.seed,
      chunkedUpsample: own.chunkedUpsample,
      ...(loras.length > 0 ? { loras } : {}),
      // NAG (2026-07-28): silently inherits whatever the Create-owned
      // accordion currently holds — only threaded through while enabled, so
      // an OFF NAG state stays a no-op for every row's payload.
      ...(nag.enabled ? { nag } : {}),
      // Acceleration (2026-07-31; block-swap prefetch added 2026-08-01,
      // keep-resident 2026-08-02, fused GGUF dequant kernel 2026-08-04,
      // PrunaVAED 2026-08-05, keep-resident-embeddings 2026-09-03): only
      // threaded through when it would
      // actually change the payload, so an all-defaults choice stays a no-op
      // for every row (`accelerationRequestFields` would return `{}` anyway —
      // this keeps the settings object itself minimal too). Checked
      // field-by-field: any ONE field moved off its own server default is
      // enough to thread the whole object through. EVERY new real field must
      // be added to this OR-list too — omitting it makes the toggle work
      // everywhere except Batch A2V.
      ...(acceleration.attentionBackend !== ATTENTION_BACKEND_DEFAULT ||
      acceleration.blockSwapPrefetch !== BLOCK_SWAP_PREFETCH_SERVER_DEFAULT ||
      acceleration.keepResident !== KEEP_RESIDENT_SERVER_DEFAULT ||
      acceleration.fusedGgufDequantKernel !== FUSED_GGUF_DEQUANT_KERNEL_SERVER_DEFAULT ||
      acceleration.vaeMode !== VAE_MODE_DEFAULT ||
      acceleration.keepResidentEmbeddings !== KEEP_RESIDENT_EMBEDDINGS_SERVER_DEFAULT
        ? { acceleration }
        : {}),
    };
    // Batch A2V Shared spec (2026-07-18): a `Shared` row uses ONLY the first
    // (sort-order-first == lowest `frame_idx`) ready image from the Create-
    // owned KEYFRAMES panel, force-pinned to `frame_idx=0` (the leading frame)
    // regardless of where its slider sits — the card's own `strength` is kept.
    // Any later ready keyframes are ignored entirely (no error/warning). The
    // result is a length-0 (no ready image) or length-1 array;
    // `resolveConditioning` downstream is unchanged — it just receives a
    // pre-narrowed input. `conditioningImages` is built from the panel's
    // unsorted items, so sort a copy to pick the true first card.
    const [firstReady] = [...keyframes.conditioningImages].sort((a, b) => a.frame_idx - b.frame_idx);
    const sharedConditioningImages = firstReady ? [{ ...firstReady, frame_idx: 0 }] : [];
    const params: RunBatchParams = {
      wavDir,
      outDir,
      settings,
      rows: rejudged,
      sharedConditioningImages,
      // backend `Docs/PENDING_TASKS_CLOSED.md` §3-47-02: frozen alongside the rows so a mid-run remount restores the
      // table's own derived state (image `<select>` options, fps/DURATION
      // drift hint) instead of showing a half-blank panel over a live run.
      imageFileNames,
      scannedFps,
      scannedMaxFrames,
      ...(imgDir ? { imgDir } : {}),
    };
    // §1-7 相互ロック: the shared lock is taken inside `runtime.run()` (before
    // the runner's `start()`) and handed back from the run promise, so a
    // remount in between cannot strand it. A refusal (`{started:false}`,
    // including "the other batch holds the lock") is visible through the
    // runtime's `lastStartResult`; nothing is submitted in that case.
    //
    // `setRows` is passed as an EXTRA sink on top of the runtime snapshot: the
    // runner flushes row 1's `Waiting -> Generating` synchronously inside
    // `start()`, and this keeps that visible on the same tick as the click
    // (the snapshot-driven effect above is what survives a remount).
    batchRunner.run(params, setRows);
    // NAG (2026-07-28) adversarial-review note: `nag` MUST be in this
    // dependency list — omitting it would let `start()` close over a STALE
    // NAG state and silently run a whole overnight batch against an old
    // enabled/method/scale/tau/alpha/vsfScale/text combination (an
    // intermittent-bug source flagged in the plan's own adversarial review,
    // same class of bug as the pre-existing `generationValues.numFrames` note
    // above `scan`).
    // `serverBusy` is in the list for the same reason `nag` is: without it
    // this callback would close over a STALE job-slot state and could start a
    // whole batch into a slot that became busy since the last render.
  // `acceleration` is in this dependency list for the same reason `nag` is:
  // without it an overnight batch could run against a stale attention choice.
  }, [
    wavDir,
    outDir,
    imgDir,
    prompt,
    own,
    generationValues,
    rows,
    keyframes.conditioningImages,
    batchRunner,
    nag,
    acceleration,
    serverBusy,
    // backend `Docs/PENDING_TASKS_CLOSED.md` §3-47-02: the scan-derived state frozen into the run's snapshot — same
    // stale-closure reasoning as `nag`/`serverBusy` above.
    imageFileNames,
    scannedFps,
    scannedMaxFrames,
  ]);

  const summary: BatchSummary = useMemo(() => {
    const counts: BatchSummary = { total: rows.length, waiting: 0, generating: 0, done: 0, failed: 0, skip: 0 };
    for (const r of rows) {
      if (r.stat === "Waiting") counts.waiting += 1;
      else if (r.stat === "Generating") counts.generating += 1;
      else if (r.stat === "Done") counts.done += 1;
      else if (r.stat === "Failed") counts.failed += 1;
      else if (r.stat === "Skip") counts.skip += 1;
    }
    return counts;
  }, [rows]);

  const currentRow = useMemo(() => rows.find((r) => r.stat === "Generating") ?? null, [rows]);

  // The rows a `start()` would actually (re)process — the SAME `stat` set
  // `batchRunner.ts`'s `UNFINISHED_STATS` uses. Both the "there's something to
  // run" `canStart` gate AND N7's shared-keyframe requirement key off this one
  // list so they can never drift apart (see `sharedKeyframeMissing` below).
  const runnableRows = useMemo(
    () => rows.filter((r) => r.stat === "Waiting" || r.stat === "Failed" || r.stat === "Generating"),
    [rows],
  );

  // Batch A2V Shared spec (2026-07-18): a `Shared`-image row needs at least one
  // READY image in the Create-owned KEYFRAMES panel before a run can start —
  // its slider position no longer matters (the leading image is force-pinned to
  // frame 0 at run time in `start()`), so the gate relaxed from "a
  // `frame_idx===0` entry exists" to simply "a ready image exists". Restricted
  // to `runnableRows` (the rows about to be processed), matching Gradio's
  // `batch.py` `_validate()`, which only inspects its `targets` (unfinished
  // rows): a leftover `Done` `Shared` row must NOT force a shared keyframe when
  // every still-to-run row supplies its own image.
  const hasSharedRow = runnableRows.some((r) => r.image === IMAGE_SHARED);
  const sharedKeyframeMissing = hasSharedRow && keyframes.conditioningImages.length === 0;

  // NAG (2026-07-28): the same enabled+blank-body gate Create/Chain apply,
  // against the (silently-inherited) shared accordion state.
  const nagInvalid = isNagNegativeEmpty(nag);

  // U4 guard 1: the shared width/height (owned by the Create form) must land on
  // the 64 grid — Create passes free-typed values through unsnapped, and an
  // off-grid dimension would 422 every row of an unattended overnight run.
  const resolutionValid =
    isDimensionOnGrid(generationValues.width, 64, limits.minWidth, limits.maxWidth) &&
    isDimensionOnGrid(generationValues.height, 64, limits.minHeight, limits.maxHeight);

  // U4 guard 2 (now informational only): true when the fps OR the DURATION cap
  // baked into each scanned row's frame count / over-cap Skip differs from the
  // Create form's current values. NOTE: this NO LONGER blocks `canStart`. It
  // was originally a start-block stand-in for a missing start-time re-judgment
  // — a stale-fps run would have sent every row's old frame count. Now
  // `start()` re-judges every runnable row at the current fps AND DURATION cap
  // (see `rejudgeRows` there), so the mismatch is corrected automatically at
  // run time and the flag survives purely as a display hint ("frames will be
  // recomputed on start"), not a gate. Field name kept as `fpsMismatch` for
  // API stability even though its semantics now also cover DURATION.
  const fpsMismatch =
    (scannedFps !== null && scannedFps !== generationValues.frameRate) ||
    (scannedMaxFrames !== null && scannedMaxFrames !== generationValues.numFrames);

  const canStart =
    wavDir !== null &&
    outDir !== null &&
    !lockedByOther &&
    // §1-7 相互ロック 第2段: the backend is occupied (a job, or a model load).
    // Kept separate from `lockedByOther` on purpose — this one is
    // server-derived, so it still holds after a reload (or for a job no batch
    // panel started).
    !serverBusy &&
    batchRunner.state === "idle" &&
    !keyframes.isUploading &&
    !sharedKeyframeMissing &&
    !nagInvalid &&
    resolutionValid &&
    runnableRows.length > 0;

  return {
    wavDir,
    imgDir,
    outDir,
    outDirIsAuto,
    pickWavDir,
    pickImgDir,
    pickOutDir,
    setWavDir,
    setImgDir,
    setOutDir,
    values,
    limits,
    setChunkedUpsample,
    setPromptMode,
    resolutionValid,
    fpsMismatch,
    keyframes,
    sharedKeyframeMissing,
    nagInvalid,
    rows,
    isScanning,
    scanError,
    canScan: wavDir !== null && !isScanning && batchRunner.state === "idle",
    scan,
    resetRowToWaiting,
    setRowPromptLocal,
    copyCommonPromptToRow,
    imageOptions,
    updateRowImage,
    runnerState: batchRunner.state,
    summary,
    currentRow,
    canStart,
    lockedByOther,
    jobActive: serverBusy,
    start,
    stop: batchRunner.stop,
  };
}
