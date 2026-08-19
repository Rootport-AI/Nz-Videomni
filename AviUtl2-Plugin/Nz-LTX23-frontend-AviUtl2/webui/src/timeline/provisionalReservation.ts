import type { NativeBridge, ParamsOf } from "../bridge";
import { rekeySourceLocation } from "./sourceLocationMap";

/**
 * The single "reservation seat" tracker for the right-click generation flow
 * (RIGHTCLICK_REDESIGN_SPEC.md §5-1 / §5-3 / §5-6).
 *
 * I4 scope: this is the *minimal* skeleton that makes ✨ (#9 `textToVideoHere`)
 * work end-to-end and opens the owner's G1 gate. It tracks a single provisional
 * placeholder through three phases — `idle` -> `reserved` (a `pending-…`
 * placeholder is on the timeline but not yet a real job) -> `generating` (the
 * placeholder was handed off to a real backend `jobId`). The seat is freed back
 * to `idle` when that job settles.
 *
 * Deliberately left out of I4 (later increments I7–I12 extend this skeleton):
 *  - the "⏳生成中" busy guard that refuses a new reservation (§5-6),
 *  - placement systems (A)/(B) for the material-relative menu items (§5-9);
 *    I4 only uses (C) "at the cursor" for ✨,
 *  - the 4-stage text updates on the placeholder (§5-5) — `releaseIfSettled`
 *    here only flips the phase, it does not yet rewrite the object text.
 *
 * State is module-level (one seat per app, per spec) with a `reset` for tests.
 */

export type ReservationPhase = "idle" | "reserved" | "generating";

interface ReservationState {
  /** The `pending-…` id burned into the placeholder object name before it is
   * bound to a real job; `null` when idle. */
  pendingId: string | null;
  /** The real backend job id, once `bindToJob` has run; `null` until then. */
  jobId: string | null;
  phase: ReservationPhase;
  /** Where native actually placed the placeholder (may differ from the
   * requested cursor position on a `layer_max+1` fallback, §5-4). */
  placedLayer: number | null;
  placedFrame: number | null;
}

function idleState(): ReservationState {
  return { pendingId: null, jobId: null, phase: "idle", placedLayer: null, placedFrame: null };
}

let state: ReservationState = idleState();

/** Test-only: return the seat to its initial idle state (and clear the published
 * form values / chain-dirty flag / keyframe count below, so every test starts
 * from a clean module state). */
export function resetProvisionalReservation(): void {
  state = idleState();
  publishedFormValues = null;
  chainDirty = false;
  publishedKeyframeCount = 0;
  publishedSourceSlots = { referenceVideo: null, sourceAudio: null };
  retakeReservationPendingId = null;
  // Y3: clear the shared inserted-jobs store too (and notify, so any live
  // subscriber re-reads a clean state), keeping tests isolated per-module.
  insertedJobIds.clear();
  notifyInserted();
}

// --- Y3: W2 shared "inserted" store (⬇ insertProvisionalResult) --------------
//
// W2 (AppShell.handleRoute's `insertProvisionalResult`) runs the SAME
// replace-insert the panel's per-card 🎞 button does, but its ✅ "inserted"
// feedback lives in AppShell — it can't reach a JobCard's LOCAL `insertState`.
// This module-level store lets W2 PUBLISH "this jobId was inserted (at
// filePath)" and every JobCard SUBSCRIBE, so the same job's card(s) flip to ✅
// in sync with the W2 insert.
//
// Deliberately ASYMMETRIC (Y3 非対称案): the manual per-card 🎞 does NOT publish
// here — it stays instance-local exactly as before, so its current behavior is
// unchanged. ONLY W2 shares. The filePath is carried so a ✅→🎞 re-insert reuses
// the already-downloaded file (avoids "Could not open destination file"). Unlike
// the `publishFormValues` mirrors above (publish + synchronous pull only), this
// store has a real listener set so a cross-component ✅ actually propagates.
// Volatile module singleton (no persistence): a reload starts empty, exactly
// like the reservation seat.

const insertedJobIds = new Map<string, string | null>();
const insertedListeners = new Set<() => void>();

function notifyInserted(): void {
  for (const listener of insertedListeners) listener();
}

/** W2: record that `jobId`'s result was inserted, carrying `filePath` (so a
 * ✅→🎞 re-insert can reuse the downloaded file), and notify every subscriber. */
export function markJobInserted(jobId: string, filePath: string | null = null): void {
  insertedJobIds.set(jobId, filePath);
  notifyInserted();
}

/** Clear the shared inserted flag for `jobId` (the ✅→idle reset). Notifies only
 * when an entry was actually removed. */
export function clearJobInserted(jobId: string): void {
  if (insertedJobIds.delete(jobId)) notifyInserted();
}

/** Whether W2 has marked `jobId` inserted. */
export function isJobInserted(jobId: string): boolean {
  return insertedJobIds.has(jobId);
}

/** The filePath W2 recorded for `jobId` (null when marked without one, or when
 * `jobId` is not marked). */
export function getInsertedFilePath(jobId: string): string | null {
  return insertedJobIds.get(jobId) ?? null;
}

/** Subscribe to inserted-store changes (for `useSyncExternalStore`); returns the
 * unsubscribe. This is the NEW listener mechanism the publish/pull mirrors lack. */
export function subscribeInserted(listener: () => void): () => void {
  insertedListeners.add(listener);
  return () => {
    insertedListeners.delete(listener);
  };
}

// --- Chain-discard confirm source (I8 §3-4 ※) --------------------------------
//
// #1/#6 remount the Chain form, discarding any in-progress chain build. Before
// that remount `AppShell.handleRoute` must know whether the CURRENTLY-mounted
// Chain form has diverged from its defaults, so it can raise the "破棄して
// 読み込みますか？" confirm only when there is real work to lose. The Chain
// screen is always mounted (all tabs mount at once), so `useChainForm` PUBLISHES
// its live `isDirty` here whenever it changes and `AppShell` READS it — the same
// lightweight module-level shared-reference pattern `publishedFormValues` uses.

let chainDirty = false;

/** Publish the Chain form's current dirty state for the #1/#6 confirm gate. */
export function publishChainDirty(dirty: boolean): void {
  chainDirty = dirty;
}

/** Whether the live Chain form has diverged from its defaults (I8 §3-4 ※). */
export function getChainDirty(): boolean {
  return chainDirty;
}

// --- ✨ reservation length source (D3) --------------------------------------
//
// The ✨ (textToVideoHere) right-click no longer remounts Create (owner
// decision: keep the panel's form state), so it can no longer read the
// length/fps off a freshly-initialized form. Instead the live Create form
// PUBLISHES its current DURATION (numFrames), generation FPS and the current
// width/height here whenever they change, and `AppShell`'s ✨/📷 reservation
// reads them so the placeholder length matches what the user will actually
// generate. Width/height are what W8's DURATION engine resolves the resolution's
// comfort ceiling from for the two keep-panel-state cursor items (#9/#10). When
// nothing has been published yet (✨ fired before Create ever mounted) the
// caller falls back to the config defaults. Module-level, one-per-app, like the
// reservation seat.

export interface PublishedFormValues {
  numFrames: number;
  frameRate: number;
  /** W8: the live form's current generation width/height — the resolution the
   * #9/#10 comfort-ceiling DURATION is resolved for (`resolveSpillFreeFrames`). */
  width: number;
  height: number;
}

let publishedFormValues: PublishedFormValues | null = null;

/** Publish the Create form's current length/fps for the ✨ reservation to read. */
export function publishFormValues(values: PublishedFormValues): void {
  publishedFormValues = values;
}

/** The last-published Create form length/fps, or null before Create publishes. */
export function getPublishedFormValues(): PublishedFormValues | null {
  return publishedFormValues;
}

// --- #5 live keyframe count source (I11 §3-4 #5 branch) ----------------------
//
// #5 (addImageKeyframe) does NOT remount Create; it appends to the live keyframe
// panel over the live-command channel. AppShell must decide, at append time,
// whether this image BECOMES the 1st keyframe (a new generation origin →
// placement A) or is a 2nd+ append (no provisional at all — §3-4 #5 分岐 /
// §5-9). The static route can't know the runtime count, so the always-mounted
// Create screen PUBLISHES its current `keyframes.items.length` here whenever it
// changes and `AppShell.handleRoute` READS it to branch — the same lightweight
// module-level shared-reference pattern `publishedFormValues` uses. Defaults to
// 0 (empty KEYFRAMES) before Create ever publishes.

let publishedKeyframeCount = 0;

/** Publish the Create keyframe panel's current card count for the #5 branch. */
export function publishKeyframeCount(count: number): void {
  publishedKeyframeCount = count;
}

/** The last-published keyframe count (0 before Create publishes / when empty). */
export function getPublishedKeyframeCount(): number {
  return publishedKeyframeCount;
}

// --- W5 反対スロット保持: live Create source-slot mirror --------------------
//
// #2/#3/#7 remount Create (owner-approved full form reset), but the OPPOSITE
// source slot should survive that remount. The always-mounted Create screen
// PUBLISHES its two source slots' ready material here whenever they change, and
// `AppShell.handleRoute` READS it synchronously at route time to decide the
// remount's `carryOver` — the same lightweight module-level shared-reference
// pattern `publishedFormValues` uses. A slot is non-null ONLY while its upload
// is `ready` (uploading/idle/error publish `null`, so a half-finished slot is
// never carried — the safe side). The uploaded ids are still valid server-side,
// so carrying them re-attaches the material with no re-upload.

export interface PublishedSourceSlots {
  /** The IC-LoRA reference video slot + its two adapter strengths (each
   * `null` when the user hasn't opted the strength in), or `null` unless the
   * reference video upload is currently `ready`. */
  referenceVideo: {
    id: string;
    fileName: string;
    filePath: string;
    conditioningAttentionStrength: number | null;
    referenceVideoStrength: number | null;
  } | null;
  /** The A2V source-audio slot, or `null` unless the audio upload is currently
   * `ready`. */
  sourceAudio: {
    id: string;
    fileName: string;
    filePath: string;
  } | null;
}

let publishedSourceSlots: PublishedSourceSlots = { referenceVideo: null, sourceAudio: null };

/** Publish Create's current two source slots for the W5 carry-over to read. */
export function publishSourceSlots(slots: PublishedSourceSlots): void {
  publishedSourceSlots = slots;
}

/** The last-published Create source slots (both `null` before Create publishes
 * / when neither slot is ready). */
export function getPublishedSourceSlots(): PublishedSourceSlots {
  return publishedSourceSlots;
}

// --- Retake の❌が自分の席かを見分けるための publish（§2-3） ------------------
//
// 予約席はモジュール単一で「持ち主」を持たない。Retake パネルの❌（片付け）が
// 無条件に `rollbackReservedPlacement` を呼ぶと、「Retake の右クリック → Create
// 側で別の右クリック（席がそちらへ **移動** する）→ Edit に戻って❌」で、他タブの
// ⏳リボンを消してしまう。そこで系統D（retakeRange）の予約が成立したときだけ
// その `pending-…` id をここへ publish し、❌は
// `getReservationState().pendingId` と一致するときだけ rollback する。
//
// 一致判定が効くのは、席の移動（`reservePlacement` の 用途b 分岐）が **必ず新しい
// pendingId を発行する**ため（同ファイル内の move 分岐を参照）。席が動いた時点で
// publish 値は古い id のまま取り残され、二度と一致しない = 誤爆しない。
// 他の publish ミラーと同じ、リスナー無しのモジュール単一値。

let retakeReservationPendingId: string | null = null;

/** 系統D（Retake）の予約が取れた `pending-…` id を記録する。`null` で取り消し。 */
export function publishRetakeReservation(pendingId: string | null): void {
  retakeReservationPendingId = pendingId;
}

/** 最後に publish された Retake 由来の `pending-…` id（未発行なら `null`）。 */
export function getRetakeReservationPendingId(): string | null {
  return retakeReservationPendingId;
}

/** Current phase of the single reservation seat. */
export function getReservationPhase(): ReservationPhase {
  return state.phase;
}

/** Read-only snapshot of the seat (for tests / diagnostics). */
export function getReservationState(): Readonly<ReservationState> {
  return state;
}

/** The `pending-…` reservation-id prefix. Also used by `reconcileFromTimeline`
 * to tell a Generate-前 ⏳生成予約 (object name `NzLTX23#pending-…`) apart from a
 * job-bound ⏳生成中 (`NzLTX23#<jobId>`) after a project reload (§2). */
const PENDING_ID_PREFIX = "pending-";

/** Shared empty set for `reconcileFromTimeline`'s omitted-`settledJobIds` case,
 * so the default path allocates nothing and behaves exactly as before (X2(b)). */
const EMPTY_SETTLED: ReadonlySet<string> = new Set<string>();

/** A fresh `pending-…` reservation id. Uses `crypto.randomUUID` when
 * available (not guaranteed in every WebView2 context), else a Math.random
 * fallback — either way it only needs to be unique enough to name one object. */
function newPendingId(): string {
  const c = typeof crypto !== "undefined" ? crypto : undefined;
  const rand = c && typeof c.randomUUID === "function" ? c.randomUUID() : Math.random().toString(36).slice(2);
  return `${PENDING_ID_PREFIX}${rand}`;
}

// --- placement-aware reservation (§5-9) --------------------------------------

/** The selected material's frame range, for placement `"A"`/`"B"` (§5-9). */
export interface PlacementMaterial {
  layer: number;
  frameStart: number;
  frameEnd: number;
}

/** The right-click cursor position, for placement `"C"` (§5-9). */
export interface PlacementCursor {
  layer: number;
  frame: number;
}

export interface ReservePlacementOpts {
  /** Which placement系統 (§5-9) native uses to resolve the position. `"D"` (W0
   * Retake) and `"E"` (素材（末尾）v2 の末尾合わせ) are WebUI-only系統 that
   * `placementParams` maps onto `"B"` — native has neither and rejects them
   * (`bridge_core.cpp`'s `ParsePlacementFields`). */
  placement: "A" | "B" | "C" | "D" | "E";
  /** Required for placement `"A"`/`"B"`/`"D"`/`"E"` — the frame range the
   * position is derived from: the selected material's own range for `"A"`/`"B"`,
   * the user's SELECTED FRAME RANGE for `"D"`, and the TAIL-ALIGNED start the
   * caller has already computed (`timeline/tailAlign.ts`) for `"E"`. */
  material?: PlacementMaterial | undefined;
  /** Required for placement `"C"` — the edit-cursor position. */
  cursor?: PlacementCursor | undefined;
  numFrames: number;
  genFps: number;
  displayText: string;
  /** 4-stage label prefix (spec §5-5). The reservation insert/move is always
   * stage 1, so callers pass the ASCII "AI video will be placed here" prefix
   * here (I13 owner decision); omitted -> native's default "⏳生成中："
   * (never used on this path). */
  textPrefix?: string | undefined;
}

export interface ReservePlacementResult {
  pendingId: string;
  placedLayer: number;
  placedFrame: number;
  usedFallback: boolean;
  /** True when this call MOVED an existing (still-unbound) reservation rather
   * than inserting a fresh one (§5-6 "予約席は1つ", 用途b). */
  moved: boolean;
}

/** Build the placement-specific bridge params (frame range for `"A"`/`"B"`/
 * `"D"`/`"E"`, cursor for `"C"`) native needs to resolve the position (§5-9).
 *
 * W0 (2026-08-09) 系統"D"の写像: native knows only `"A"`/`"B"`/`"C"` — its
 * `ParsePlacementFields` rejects anything else — and `"D"` (Retake: 選択範囲の
 * 開始フレームで `layer_max+1`) has a formula IDENTICAL to `"B"`
 * (`ResolveProvisionalPlacement`'s `kSameStartFront`: `layer = layer_max + 1`,
 * `frame = material_frame_start`; re-verified against the current source on
 * 2026-08-09). So `"D"` is mapped to `"B"` HERE, at the single point where the
 * WebUI's placement系統 meets the wire, and the caller distinguishes the two by
 * what it puts in `material`: the object's own range for `"B"`, the user's
 * selected frame range for `"D"`. No native / `bridge/types.ts` / mockBridge
 * change is needed.
 *
 * 素材（末尾）v2 (2026-08-15) 系統"E"の写像: identical treatment, identical
 * reason. 末尾合わせ (「出力の末尾の帯が素材の頭に重なる位置」) has the same
 * `layer_max+1` + `material_frame_start` wire形式 as `"B"`; what differs is only
 * the NUMBER the caller puts in `material.frameStart`, which
 * `timeline/tailAlign.ts`'s `tailAlignedStartFrame` has already shifted. So this
 *改修 needs no C++ rebuild either.
 *
 * The return type is `Pick<ParamsOf<"timeline.insertProvisional">, …>`, whose
 * `placement` is the literal union `"A" | "B" | "C"` — so leaking a `"D"`/`"E"`
 * past this function is a COMPILE error, which is the safety net an `if`-chain
 * (unlike an exhaustive `switch`) would not otherwise provide. */
function placementParams(opts: ReservePlacementOpts): Pick<
  ParamsOf<"timeline.insertProvisional">,
  "placement" | "materialLayer" | "materialFrameStart" | "materialFrameEnd" | "cursorLayer" | "cursorFrame"
> {
  if (opts.placement === "C") {
    const cursor = opts.cursor ?? { layer: 0, frame: 0 };
    return { placement: "C", cursorLayer: cursor.layer, cursorFrame: cursor.frame };
  }
  const material = opts.material ?? { layer: 0, frameStart: 0, frameEnd: 0 };
  return {
    placement: opts.placement === "D" || opts.placement === "E" ? "B" : opts.placement,
    materialLayer: material.layer,
    materialFrameStart: material.frameStart,
    materialFrameEnd: material.frameEnd,
  };
}

/**
 * Generation-origin entry point (§5-9): reserve a placeholder at the position
 * the given placement系統 resolves — (A) 素材の直後 / (B) 素材と同開始で
 * `layer_max+1` / (C) 右クリック位置 / (D) 選択範囲と同開始で `layer_max+1`
 * (W0) / (E) 末尾合わせ (素材（末尾）v2) — the last two both mapped onto (B) by
 * `placementParams`. When a still-unbound reservation already
 * exists (`phase === "reserved"`), the single seat is MOVED via
 * `timeline.updateProvisionalReservation` (§5-3 用途b / §5-6 "予約席は1つ")
 * instead of adding a second one; otherwise a fresh `timeline.insertProvisional`
 * is issued.
 *
 * The `phase === "generating"` busy-guard (§5-6) is enforced by the CALLER
 * (`AppShell.handleRoute`) before this runs — a generation-origin right-click
 * consults `getReservationPhase()` and shows the "予約不可" note without ever
 * reaching here while a job is in flight.
 */
export async function reservePlacement(
  bridge: NativeBridge,
  opts: ReservePlacementOpts,
): Promise<ReservePlacementResult> {
  const pendingId = newPendingId();
  const place = placementParams(opts);
  // Stage-1 label prefix (spec §5-5): forwarded only when the caller supplies it,
  // so an omitted prefix leaves native on its default.
  const prefix = opts.textPrefix !== undefined ? { textPrefix: opts.textPrefix } : {};

  if (state.phase === "reserved" && state.pendingId) {
    // 用途b: move the existing (unbound) reservation to the new placement.
    const res = await bridge.request("timeline.updateProvisionalReservation", {
      oldJobId: state.pendingId,
      newJobId: pendingId,
      displayText: opts.displayText,
      ...prefix,
      numFrames: opts.numFrames,
      genFps: opts.genFps,
      ...place,
    });
    // I6 Join: the seat keeps its identity across a move, but its NAME changes —
    // so carry the v2v source-location memo (recorded under the OLD pending id by
    // #1 extend-video's right-click) over to the new one. Without this the memo is
    // orphaned the first time the seat moves and the joined 🎞 insert degrades to
    // its position-less fallback even though the source is perfectly well known.
    // Done AFTER the RPC resolved, so a rejected move leaves the memo where the
    // still-current reservation can find it. No-op when nothing was recorded.
    rekeySourceLocation(state.pendingId, pendingId);
    state = {
      pendingId,
      jobId: null,
      phase: "reserved",
      placedLayer: res.placedLayer,
      placedFrame: res.placedFrame,
    };
    return {
      pendingId,
      placedLayer: res.placedLayer,
      placedFrame: res.placedFrame,
      usedFallback: res.usedFallback,
      moved: true,
    };
  }

  // Fresh insert; native resolves the length from `numFrames`+`genFps` and the
  // position from the placement fields.
  const res = await bridge.request("timeline.insertProvisional", {
    jobId: pendingId,
    displayText: opts.displayText,
    ...prefix,
    numFrames: opts.numFrames,
    genFps: opts.genFps,
    ...place,
  });
  const placedLayer = res.placedLayer ?? res.layer;
  const placedFrame = res.placedFrame ?? res.frame;
  const usedFallback = res.usedFallback ?? false;
  state = { pendingId, jobId: null, phase: "reserved", placedLayer, placedFrame };
  return { pendingId, placedLayer, placedFrame, usedFallback, moved: false };
}

export interface ReserveAtCursorOpts {
  cursorLayer: number;
  cursorFrame: number;
  numFrames: number;
  genFps: number;
  displayText: string;
  /** Stage-1 label prefix (spec §5-5); see `ReservePlacementOpts.textPrefix`. */
  textPrefix?: string | undefined;
}

export type ReserveAtCursorResult = ReservePlacementResult;

/**
 * ✨ (#9 `textToVideoHere`) convenience wrapper: the placement-`"C"` special
 * case of `reservePlacement`, kept for the cursor-only call sites/tests. Other
 * generation-origin routes call `reservePlacement` directly with their own
 * placement系統 (§5-9).
 */
export async function reserveAtCursor(
  bridge: NativeBridge,
  opts: ReserveAtCursorOpts,
): Promise<ReserveAtCursorResult> {
  return reservePlacement(bridge, {
    placement: "C",
    cursor: { layer: opts.cursorLayer, frame: opts.cursorFrame },
    numFrames: opts.numFrames,
    genFps: opts.genFps,
    displayText: opts.displayText,
    textPrefix: opts.textPrefix,
  });
}

/**
 * X2(a) 送信失敗時のロールバック: drop a still-unbound reservation seat AND its
 * `NzLTX23#pending-…` placeholder when the Generate submit rejected synchronously
 * (a 422/busy that never reached `onSubmitted`, so the seat never bound to a real
 * job). Without this the `reserved` seat and its placeholder linger forever, and
 * `reconcileFromTimeline` keeps resurrecting them — the permanent "予約不可" block
 * the owner hit on real hardware.
 *
 * Deliberately NEVER touches a `generating` (bound, real-job) seat: returns
 * `false` and issues no bridge call unless we are exactly `reserved` with a
 * pending id. The pending id is SNAPSHOTTED before the seat is reset to idle, so
 * the seat is released even if the delete RPC rejects; the RPC failure is
 * swallowed (the next `reconcileFromTimeline` reclaims a stranded placeholder).
 */
export async function rollbackReservedPlacement(bridge: NativeBridge): Promise<boolean> {
  if (state.phase !== "reserved" || !state.pendingId) return false;
  // Snap the id before freeing the seat, so the delete targets the right
  // placeholder even though the seat is idle from here on.
  const pendingId = state.pendingId;
  state = idleState();
  try {
    await bridge.request("timeline.deleteProvisionalByJob", { jobId: pendingId });
  } catch {
    // Swallow: a failed placeholder cleanup is reclaimed by the next reconcile
    // (a lingering pending- orphan is re-adopted as `reserved`, never a block).
  }
  return true;
}

export interface BindToJobOpts {
  jobId: string;
  numFrames: number;
  genFps: number;
  displayText: string;
  /** Stage-2 label prefix (spec §5-5 段階2), passed as
   * `updateProvisionalReservation.textPrefix`. Callers pass
   * `strings.provisional.generatingPrefix` ("Generating: ") so the burned text
   * reads "Generating: 〔prompt head〕…". Omitted -> native's default
   * `kGeneratingPrefix` ("⏳生成中："); production always supplies this now so
   * that default is effectively unused (its removal is deferred to an I14 native
   * cleanup — kept for now to avoid touching native). */
  textPrefix?: string | undefined;
}

/**
 * Generate-time hand-off (§5-3 用途a): re-stamp the reserved `pending-…`
 * placeholder with the real backend `jobId` and the confirmed length, KEEPING
 * it at the position it was reserved at (placement `"C"` with the recorded
 * `placedLayer`/`placedFrame` — only the id and length change, not the spot).
 *
 * Only acts when a reservation is actually waiting (`phase === "reserved"`) —
 * a plain panel-origin Generate (not via ✨) leaves the seat untouched and
 * returns `false` without any bridge call. This is what keeps a non-✨ Generate
 * from mis-firing on the reservation flow.
 */
export async function bindToJob(bridge: NativeBridge, opts: BindToJobOpts): Promise<boolean> {
  if (state.phase !== "reserved" || !state.pendingId) return false;

  // Stage-2 prefix (§5-5 段階2): forwarded only when the caller supplies it, so
  // an omitted prefix leaves native on its default (kGeneratingPrefix).
  const prefix = opts.textPrefix !== undefined ? { textPrefix: opts.textPrefix } : {};

  const res = await bridge.request("timeline.updateProvisionalReservation", {
    oldJobId: state.pendingId,
    newJobId: opts.jobId,
    displayText: opts.displayText,
    ...prefix,
    numFrames: opts.numFrames,
    genFps: opts.genFps,
    placement: "C",
    cursorLayer: state.placedLayer ?? 0,
    cursorFrame: state.placedFrame ?? 0,
  });
  state = {
    pendingId: state.pendingId,
    jobId: opts.jobId,
    phase: "generating",
    placedLayer: res.placedLayer,
    placedFrame: res.placedFrame,
  };
  return true;
}

/**
 * Free the seat when the tracked job reaches a terminal state (§5-5 / §5-6):
 * the ⏳ placeholder is no longer a reservation once the job is done/failed.
 * No-op unless `jobId` matches the currently tracked (bound) job, so a settling
 * panel-origin job never releases an unrelated reservation.
 *
 * Text-blind: this ONLY flips the phase back to `idle`. The job-poll settle
 * handler uses `settleProvisionalText` instead (I12), which rewrites the
 * placeholder to its ✅/❌ terminal text (§5-5) AND frees the seat in one step;
 * this bare release remains for the reconcile-convergence path and unit tests.
 */
export function releaseIfSettled(jobId: string): boolean {
  if (state.jobId !== null && state.jobId === jobId) {
    state = idleState();
    return true;
  }
  return false;
}

/** The self-delimiting `[#<id>]` marker native's `BuildProvisionalTextAlias`
 * appends to the visible text — the PRIMARY key `FindObjectByJob` matches on. The
 * terminal ✅/❌ text (§5-5) goes through `timeline.updateProvisionalText`, which
 * OVERWRITES the whole text item, so we re-append this marker to keep the
 * placeholder re-discoverable by job id (the `NzLTX23#<id>` object_name double-tag
 * also survives independently, but the text marker is the primary match). */
function withJobMarker(text: string, jobId: string): string {
  return `${text} [#${jobId}]`;
}

/**
 * Terminal-state text update + seat release (§5-5 stage 3/4, §5-6). When `jobId`
 * is the currently tracked (bound, phase `generating`) job, rewrite its
 * placeholder to the ✅/❌ `text` the caller built from the current UI language
 * and free the single reservation seat. The seat is freed FIRST (synchronously,
 * before the RPC's `await`) using the position captured at bind time, so it is
 * released even if the text RPC rejects — the ⏳ is no longer a live reservation
 * the moment the job settles. The `updateProvisionalText` call is best-effort:
 * any failure is swallowed (§5-5 "掲示板更新の失敗で他を壊さない").
 *
 * No-op — returns `false`, issues NO bridge call — unless `jobId` matches the
 * bound job, so a settling panel-origin job never rewrites (or releases) an
 * unrelated reservation. This is the §5-5/§5-6 replacement for a bare
 * `releaseIfSettled` in the job-poll settle handler.
 */
export async function settleProvisionalText(
  bridge: NativeBridge,
  jobId: string,
  text: string,
): Promise<boolean> {
  if (state.jobId === null || state.jobId !== jobId) return false;
  const layer = state.placedLayer ?? 0;
  const frame = state.placedFrame ?? 0;
  // Free the seat up front so a rejected RPC can never strand it "generating".
  state = idleState();
  try {
    await bridge.request("timeline.updateProvisionalText", {
      jobId,
      layer,
      frame,
      text: withJobMarker(text, jobId),
    });
  } catch {
    // Swallow: a failed bulletin-board update must not disturb settle handling.
  }
  return true;
}

/**
 * Reload/startup re-sync (I7 work order §2, spec §5-6/§5-7): rebuild the single
 * reservation seat from whatever `NzLTX23#…`-tagged placeholders survived a
 * project reload.
 *
 * `timeline.scanProvisionals` returns every NzLTX23-tagged object as an orphan
 * candidate (`{ jobId, layer, frame }`), where `jobId` is the token after
 * `NzLTX23#`. It is text-blind — it cannot read the placeholder's 4-stage body
 * (§5-5) — so the state is rebuilt from the id token alone:
 *  - `pending-…`  -> a Generate-前 ⏳生成予約  -> restore as `reserved`.
 *  - anything else -> a job-bound marker (⏳生成中の可能性) -> restore as
 *    `generating`.
 *
 * A job-bound marker may actually be an already-terminal ✅/❌ marker left by a
 * prior session (indistinguishable while text-blind), so restoring it as
 * `generating` is deliberately CONSERVATIVE: it holds the seat busy until
 * JobsContext's 2s poll observes that job settled and calls
 * `releaseIfSettled(jobId)`, which frees the seat back to idle. That existing
 * mechanism is what lets a stale marker converge harmlessly to idle with no
 * extra bookkeeping; a genuinely-running ⏳生成中 stays busy exactly as long as
 * the poll still reports it running. When (unusually) both a job-bound and a
 * `pending-` marker are present, the job-bound one wins — it is the only one
 * that could be a live generation the busy-guard must protect.
 *
 * Never throws: any RPC failure (or a host without the method) resolves the
 * seat to `idle`, so app startup is never blocked.
 */
export async function reconcileFromTimeline(
  bridge: NativeBridge,
  settledJobIds?: ReadonlySet<string>,
): Promise<ReservationPhase> {
  let orphans: Array<{ jobId: string; layer: number; frame: number }>;
  try {
    const res = await bridge.request("timeline.scanProvisionals", {});
    orphans = res.orphans;
  } catch {
    state = idleState();
    return "idle";
  }

  // X2(b) 台帳認識: a job-bound `NzLTX23#<jobId>` tag whose job is already
  // terminal (completed/failed/cancelled per the live ledger) is a leftover
  // marker, not a live generation — `useJobsPoll`'s settle callback never fires
  // for a job that was already terminal on first observation, so nothing would
  // otherwise free the seat and this reconcile would resurrect it to
  // `generating` on every run (the permanent post-422 block). Excluding it from
  // the job-bound candidates lets a genuine `pending-` reservation (or idle)
  // win; the tag itself is left in place for the 🎞 insert's marker lookup.
  // Omitting `settledJobIds` (existing call sites) keeps the original behavior.
  const settled = settledJobIds ?? EMPTY_SETTLED;
  const jobBound = orphans.find((o) => !o.jobId.startsWith(PENDING_ID_PREFIX) && !settled.has(o.jobId));
  if (jobBound) {
    state = {
      pendingId: null,
      jobId: jobBound.jobId,
      phase: "generating",
      placedLayer: jobBound.layer,
      placedFrame: jobBound.frame,
    };
    return "generating";
  }

  const reserved = orphans.find((o) => o.jobId.startsWith(PENDING_ID_PREFIX));
  if (reserved) {
    state = {
      pendingId: reserved.jobId,
      jobId: null,
      phase: "reserved",
      placedLayer: reserved.layer,
      placedFrame: reserved.frame,
    };
    return "reserved";
  }

  state = idleState();
  return "idle";
}
