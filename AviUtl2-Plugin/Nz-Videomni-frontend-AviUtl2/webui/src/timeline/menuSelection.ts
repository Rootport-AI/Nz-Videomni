import type { NativeBridge, ResultOf } from "../bridge";
import type { MaterialKind, MenuRequiredKind, MenuRoute } from "./menuRouting";
import { RETAKE_WINDOW_MIN_PX } from "./retakeWindow";
import { selectionRangeFrames } from "./selectionRange";
import { END_SOURCE_MIN_FRAMES } from "./tailAlign";
import { decideSourceTrim } from "./sourceTrim";
import type { en } from "../i18n/strings";

/**
 * Two-stage selection resolution for menu-invoked commands (contract v5,
 * task 3).
 *
 * The `timeline.menuInvoked` event carries a selection snapshot captured when
 * the menu fired, but native (`plugin.cpp`) may push selected items with
 * `filePath: null` — it deliberately omits the (comparatively expensive)
 * per-object file-path lookup there to avoid duplicating code in the menu
 * callback. Before any cutout/upload/generation that actually needs the file on
 * disk, the WebUI therefore re-queries `timeline.getSelection` (which does fill
 * `filePath`) and uses that authoritative snapshot instead.
 *
 * The predicate is pure and unit-tested on its own; the resolver takes an
 * injected bridge so the fetch path is exercised against the mock bridge.
 */

/** The `timeline.getSelection` snapshot (also the `menuInvoked` payload's
 * `selection`). */
export type TimelineSelection = ResultOf<"timeline.getSelection">;

/**
 * Pure: does this snapshot need an authoritative re-query for a command that
 * operates on selected objects? True when at least one selected object is
 * missing its `filePath` (native pushed `null`). A snapshot with no selected
 * objects returns false — there is no object file path to complete (the command
 * either doesn't need one or there is nothing to act on); the caller decides how
 * to handle an empty selection separately.
 */
export function selectionHasMissingFilePath(selection: TimelineSelection): boolean {
  return selection.selected.some((item) => item.filePath === null);
}

export interface ResolveMenuSelectionParams {
  /** The snapshot carried by the `menuInvoked` event. */
  snapshot: TimelineSelection;
  /** From the route (`menuRouting.ts`): whether the command acts on selected
   * objects. When false, no file-path completion is attempted and `snapshot`
   * is returned as-is. */
  needsSelection: boolean;
}

export interface ResolveMenuSelectionResult {
  /** The selection to act on: the authoritative re-queried snapshot when a
   * fetch was performed, else the original `snapshot`. */
  selection: TimelineSelection;
  /** True when `timeline.getSelection` was called to complete missing file
   * paths (i.e. the returned `selection` is the fresh authoritative one). */
  refreshed: boolean;
}

/**
 * Return a selection with file paths guaranteed present (as far as native can
 * provide) for object-operating commands: if `needsSelection` and the pushed
 * snapshot has any `filePath: null`, re-query `timeline.getSelection` and
 * return that; otherwise return the pushed snapshot untouched (no bridge call).
 *
 * Only transport-level bridge failures reject — a successful `getSelection`
 * that still reports a null file path (e.g. an object genuinely without a
 * backing file) is returned as-is; the caller inspects it before cutout.
 */
export async function resolveMenuSelection(
  nativeBridge: NativeBridge,
  params: ResolveMenuSelectionParams,
): Promise<ResolveMenuSelectionResult> {
  const { snapshot, needsSelection } = params;
  if (!needsSelection || !selectionHasMissingFilePath(snapshot)) {
    return { selection: snapshot, refreshed: false };
  }
  const authoritative = await nativeBridge.request("timeline.getSelection", {});
  return { selection: authoritative, refreshed: true };
}

// ---------------------------------------------------------------------------
// I6: type classification, mismatch / multi-select / length guards
// (Docs/RIGHTCLICK_REDESIGN_SPEC.md §4-2 / §4-3 / §4-5). All pure.
// ---------------------------------------------------------------------------

/** One selected timeline object from `timeline.getSelection` / `menuInvoked`. */
export type SelectionItem = TimelineSelection["selected"][number];

/** The four supported kinds, or `"unknown"` when `effectName` matches none of
 * them (§4-3 "種別が判定不能な場合"). */
export type SelectionKind = MaterialKind | "unknown";

/**
 * The exact `effectName` strings native fills for the four object kinds. These
 * are the AviUtl2 Japanese effect names `GetSelectionEditProc` copies verbatim
 * into the selection (`native/src/bridge.cpp`'s `kEffectVideoFileJp` /
 * `kEffectImageFileJp` / `kEffectAudioFileJp` / `kEffectTextJp`, held there as
 * UTF-8 byte escapes because that file is ASCII-only). The WebUI receives them
 * already decoded, so it compares against the literal Japanese, matching the
 * four names the spec enumerates in §4-1 (動画ファイル/画像ファイル/音声ファイル/
 * テキスト). Kept as named constants so a native rename fails loudly here.
 */
export const EFFECT_NAME_VIDEO = "動画ファイル";
export const EFFECT_NAME_IMAGE = "画像ファイル";
export const EFFECT_NAME_AUDIO = "音声ファイル";
export const EFFECT_NAME_TEXT = "テキスト";

/**
 * Pure: classify a selected object by its `effectName` (§4-1). Returns
 * `"unknown"` for any name that is not one of the four supported effects
 * (including the empty string native leaves for objects it could not resolve),
 * which the caller surfaces as the "対応していない種類" note rather than a
 * template mismatch (§4-3).
 */
export function classifySelectionKind(item: SelectionItem): SelectionKind {
  switch (item.effectName) {
    case EFFECT_NAME_VIDEO:
      return "video";
    case EFFECT_NAME_IMAGE:
      return "image";
    case EFFECT_NAME_AUDIO:
      return "audio";
    case EFFECT_NAME_TEXT:
      return "text";
    default:
      return "unknown";
  }
}

// --- Length-guard thresholds (§4-5), pinned to the backend's real limits -----

/**
 * The backend's fixed generation frame rate: `config.yaml` `generation.frame_
 * rate: 24.0` (Nz-Videomni/config.yaml, also engine api_types). Source
 * video frame counts are evaluated after resample to this rate, so a real-time
 * duration (`mediaDurationSec`) maps to frames as `durationSec * GEN_FPS`.
 */
export const GEN_FPS = 24;

/**
 * #1 `extendVideo` (V2V continuation) minimum source duration.
 *
 * A V2V source video, after resample to `GEN_FPS`, must have at least
 * `context_frames` frames or the server rejects it up front with
 * `422 SOURCE_VIDEO_TOO_SHORT` before any GPU work
 * (Nz-Videomni/api/errors.py:76-85 `source_video_too_short`). The Chain
 * screen the right-click lands on defaults `context_frames` to
 * `v2v_context_frames_default = 73` (Nz-Videomni/config.py:236, also
 * config.yaml). We therefore guard at the *default* (73 frames), not the
 * absolute floor (`v2v_context_frames_min = 25`, config.py:237): a source
 * shorter than 73 frames hits the 422 at the prefilled default, and the spec's
 * §4-5 intent is to prevent that known rejection before Generate. 73 / 24 ≈
 * 3.04s.
 */
export const V2V_MIN_DURATION_SEC = 73 / GEN_FPS;

/**
 * #2 `referenceVideo` (IC-LoRA) minimum reference duration.
 *
 * TODO(要バックエンド照合): the backend does NOT validate reference-video length.
 * `api/generate.py` only 404s a missing `reference_video_id`
 * (Nz-Videomni/api/generate.py:57-58); there is no up-front frame-count
 * check, so a too-short reference crashes deep inside the pipeline (spec §4-5:
 * "サーバー側は入力の長さを検査しません"). No confirmed backend constant exists.
 * Conservative placeholder: reuse the documented `v2v_context_frames_min`
 * (25 frames, config.py:237 / config.yaml:176) as a ~1s UX floor for the
 * similarly conditioning-video role. Revisit if/when the backend adds an
 * explicit reference-length check. 25 / 24 ≈ 1.04s.
 *
 * What this floor is compared against changed on 2026-08-01, when #2's upload
 * gained the same ribbon-range trim #1 has (§1-6 拡張): it is now the
 * `decideSourceTrim` window when a trim fires, and the full backing file's
 * `mediaDurationSec` when it does not — i.e. the length that will actually be
 * conditioned on, in both cases. (W4's earlier rule of always measuring the
 * timeline SPAN was the right stand-in only while #2 uploaded whole files.)
 */
export const ICLORA_MIN_DURATION_SEC = 25 / GEN_FPS;

/**
 * 素材（末尾）`endWithThis` minimum source duration (窓内モード, 2026-08-17).
 *
 * v1 guarded at 73 frames, because the anchor was a SLIDER prefilled at
 * `end_context_frames_default = 72` and a shorter video would 422 at that
 * default. v2 derived the anchor from the material's own length and kept a
 * usefulness floor of 25 frames. The window-internal mode removed both: the
 * anchor is the fixed `END_SOURCE_CONTEXT_FRAMES` (8) whatever the material is,
 * so the only floor left is the technical one —
 * {@link END_SOURCE_MIN_FRAMES} = 9 frames (8 + the causal VAE's 1-frame
 * primer), the shortest video the server can actually read.
 *
 * The floor applies to VIDEO material only — an image has no duration to
 * measure, and the still-video conversion produces exactly the frames the
 * backend needs — which is why the (d) guard below is scoped to `kind ===
 * "video"`.
 */
export const END_SOURCE_MIN_DURATION_SEC = END_SOURCE_MIN_FRAMES / GEN_FPS;

/**
 * The slack the (d) length guard allows an end-source VIDEO, in seconds —
 * exactly ONE generation frame.
 *
 * Why it exists: the measured duration this guard compares is a CONTAINER
 * duration (`get_media_info`'s `total_time`, or a ribbon window derived from
 * it), which routinely under-reports a video by a fraction of a frame. Without
 * slack, a genuine 9-frame material recorded at 24fps can measure
 * 0.375s − ε and be refused at the right click — destructively, since the
 * panel never opens — even though the form (which measures the SERVER's own
 * frame count) would have accepted it. The right-click guard's job is to refuse
 * only what cannot possibly work, so it is biased the other way from the form's
 * gate: one frame of tolerance here, and the form stays the real arbiter.
 */
export const END_SOURCE_DURATION_MARGIN_SEC = 1 / GEN_FPS;

/**
 * §1-17 Retake の右クリックガードが「どの生成 fps でも足りない」を判定するときに
 * 使う、生成フレームレートの上限。フォームの frame_rate 入力そのものの `max`
 * （`modes/edit/RetakePanel.tsx` / `modes/edit/OutpaintingPanel.tsx` の 60）と
 * 同じ値で、**保守側**に効く: fps が高いほど同じフレーム数の窓は短い実時間で
 * 済むので、上限 fps で測った長さを下回る範囲だけが「絶対に足りない」。
 */
export const RETAKE_MAX_GEN_FPS = 60;

/** The note a failed guard asks the shared note area to show (§6). A pure data
 * descriptor — localization happens in `formatMenuGuardNote` so this stays
 * free of the i18n dictionary (existing 流儀: pure logic returns codes, strings
 * format). `noteKind` discriminates the union; the payload carries only the
 * data a message template needs. */
export type MenuGuardNote =
  /** §4-2: two or more objects selected (single-selection only in α). */
  | { noteKind: "multipleSelection" }
  /** §4-3: `effectName` matched none of the four supported kinds. */
  | { noteKind: "unsupportedType" }
  /** §4-3: a supported kind, but not the one this action requires. */
  | { noteKind: "typeMismatch"; selected: MaterialKind; required: MaterialKind }
  /** §4-3, 素材（末尾）(2026-08-15): a supported kind, but not one of the SEVERAL
   * this action accepts (`requiredKind` is an array — e.g. `endWithThis` takes a
   * video or an image). A distinct variant rather than a payload tweak on the
   * one above because the copy differs grammatically: the message has to
   * enumerate the accepted kinds ("a video or image" / 「動画か画像」), which the
   * single-kind template cannot express. Raised ONLY when the action accepts two
   * or more kinds; a one-element set still uses `typeMismatch` so no existing
   * message changes. */
  | { noteKind: "typeMismatchAny"; selected: MaterialKind; required: readonly MaterialKind[] }
  /** §4-3 near-neighbor special case: a video was selected for #7
   * "audio-to-video from an audio object"; steer the user to #3. */
  | { noteKind: "useVideoAudioInstead" }
  /** §4-3 near-neighbor, 台帳§1-16 (2026-08-10): the same mistake made on the
   * LONG-form audio item (`audioToLongA2v`); steer the user to its own video
   * sibling `videoAudioToLongA2v` rather than to the single-shot #3. A distinct
   * variant (not a payload on the one above) because the two point at two
   * different menu entries, and the copy must name the right one. */
  | { noteKind: "useVideoAudioLongInstead" }
  /** §4-5: source/reference video shorter than the action's length floor. */
  | { noteKind: "videoTooShort"; requiredSeconds: number }
  /** W0 §4 範囲系: the action's input IS the user's selected frame range
   * (Retake), but nothing is selected (`timeline.getSelection`'s
   * `hasRange: false`). */
  | { noteKind: "rangeNotSelected" }
  /** W0 §4 範囲系: a range IS selected but is shorter than the minimum window
   * the retake pipeline can regenerate. RESERVED BY W0, RAISED BY W3: the
   * threshold is fps-dependent (frames→the pipeline's window), so W3 owns the
   * judgement and only the variant + copy are pinned here. This union is the
   * single point W2 and W3 both extend, so fixing its shape up front is what
   * keeps the two streams from colliding on it. */
  | { noteKind: "rangeTooShort"; requiredFrames: number };

/** The unified guard result: a discriminated union so a successful guard
 * carries nothing and a failed one carries exactly the note to display. */
export type MenuGuardResult =
  | { ok: true }
  | { ok: false; note: MenuGuardNote };

/**
 * §4-3 近傍アクション対応表 (台帳§1-16 で一般化): the actions that REQUIRE an
 * audio object, each mapped to the note that steers a mis-selected VIDEO to the
 * sibling item which extracts that video's sound.
 *
 * Before §1-16 this was a single `route.action === "audioToVideo"` literal
 * inline in the guard. Now that the audio-required set has two members — #7
 * `audioToVideo` (→ #3, single shot) and `audioToLongA2v` (→
 * `videoAudioToLongA2v`, the clip chain) — the pairing lives in one table so
 * adding a third audio item cannot silently fall back to the bare
 * `typeMismatch` note. Actions absent from this table get that generic note,
 * which stays the right answer for every other kind mismatch.
 */
const VIDEO_FOR_AUDIO_ACTION_NOTE: Readonly<Record<string, MenuGuardNote>> = {
  audioToVideo: { noteKind: "useVideoAudioInstead" },
  audioToLongA2v: { noteKind: "useVideoAudioLongInstead" },
};

/**
 * Pure: normalize a route's `requiredKind` into the SET of kinds that pass
 * (素材（末尾）2026-08-15). `null` → `[]` (no kind is being required — the
 * layer-menu exemption, which `guardMenuSelection` has already short-circuited
 * before it ever calls this), a single kind → a one-element array, an array →
 * itself. Exists so the kind check reads the same for one accepted kind and for
 * several, instead of branching on the field's shape inline.
 */
export function requiredKindsOf(required: MenuRequiredKind): readonly MaterialKind[] {
  if (required === null) return [];
  // `typeof === "string"` (not `Array.isArray`) because `MaterialKind` IS a
  // string union: it narrows both arms exactly, with no cast on either side.
  if (typeof required === "string") return [required];
  return required;
}

/**
 * Pure integrated guard for a routed object-menu command (§4). Runs, in order:
 *  (a) multiple-selection (§4-2), (b) undeterminable kind (§4-3), (c) required-
 *  kind mismatch with the near-neighbor special case (§4-3), (d) the
 *  #1/#2 length guard (§4-5), and (e) W0's range guard for `retakeRange`.
 *  Returns `{ ok: true }` when the action may
 *  proceed, else `{ ok: false, note }` with the note to surface — and NO
 *  destructive side effect, matching the "合わないときは破壊的な動作をしない"
 *  policy.
 *
 * Layer-menu commands (#9/#10, `requiredKind: null`) generate at a position and
 * are exempt from every selection check (§3-6), so they always pass here.
 */
export function guardMenuSelection(route: MenuRoute, selection: TimelineSelection): MenuGuardResult {
  // Layer-menu items operate on a position, not a selection — exempt (§3-6).
  if (route.requiredKind === null) return { ok: true };

  const selected = selection.selected;

  // (a) Multiple selection — single-selection only in α (§4-2).
  if (selected.length >= 2) return { ok: false, note: { noteKind: "multipleSelection" } };

  // (b) Undeterminable kind (§4-3). An empty object-menu selection is also
  // unclassifiable, so it folds into the same "not supported" note.
  const item = selected[0];
  const kind = item ? classifySelectionKind(item) : "unknown";
  if (kind === "unknown") return { ok: false, note: { noteKind: "unsupportedType" } };

  // (c) Required-kind mismatch (§4-3). 素材（末尾）(2026-08-15): an action may now
  // accept a SET of kinds, so the check is "is the selection in the set?" — for
  // the single-kind entries that is byte-for-byte the old equality test.
  const requiredKinds = requiredKindsOf(route.requiredKind);
  if (!requiredKinds.includes(kind)) {
    // Near-neighbor special case: a VIDEO selected for an audio-required action.
    // Both audio items have a video sibling that does the right thing with a
    // video's sound, so instead of the bare mismatch note we name that sibling
    // (台帳§1-16 generalized this from the single `audioToVideo` literal).
    if (kind === "video") {
      const near = VIDEO_FOR_AUDIO_ACTION_NOTE[route.action];
      if (near) return { ok: false, note: near };
    }
    // One accepted kind keeps the original single-kind copy; two or more need
    // the enumerating variant (see `typeMismatchAny`'s doc).
    const [onlyKind] = requiredKinds;
    if (requiredKinds.length === 1 && onlyKind !== undefined) {
      return { ok: false, note: { noteKind: "typeMismatch", selected: kind, required: onlyKind } };
    }
    return { ok: false, note: { noteKind: "typeMismatchAny", selected: kind, required: requiredKinds } };
  }

  // (d) Length guard for #1 / #2 (§4-5). Both actions measure THE MATERIAL THAT
  // WILL ACTUALLY BE UPLOADED — and since 2026-08-01 that is the SAME quantity,
  // computed the same way, for both:
  //  - #1 `extendVideo` (V2V, §1-6) trims its source to the ribbon's range when
  //    `decideSourceTrim` says so (`modes/chained/ChainedScreen.tsx`);
  //  - #2 `referenceVideo` (IC-LoRA) now does exactly the same on its own upload
  //    path (`modes/single/SingleScreen.tsx`), which is what closed the "the
  //    ribbon shows frames 31..120 but the reference conditions on 1..90" bug.
  // So the guard asks `decideSourceTrim` once and measures the trim window when
  // one applies, the FULL file (`mediaDurationSec`) otherwise. Every conservative
  // skip (no playback position, non-neutral speed, multiple sections, an
  // unresolvable span) keeps the full-file judgement, which is exactly right: an
  // untrimmed upload really is the whole file.
  //
  // This REPLACES #2's earlier W4 rule of measuring the timeline SPAN
  // unconditionally. The span was a stand-in for "the length that will actually
  // be conditioned on" back when #2 always uploaded whole; now that the upload
  // really is cut, the trim window is that length directly, and — importantly —
  // when no trim fires the material genuinely IS the whole file, so measuring the
  // span there would under-report and false-block.
  //
  // An UNKNOWN measurement (`mediaDurationSec === 0`, i.e. a failed
  // `get_media_info`) means we have no measurement to assert "too short" on —
  // blocking on unknown would false-reject a valid video whenever the probe fails,
  // and deriveGenerationParams likewise falls back gracefully on missing media
  // info rather than hard-blocking. So we PASS on unknown and let the server be
  // the final arbiter (a genuine 422 then shows as the ❌ provisional reason,
  // §5-5). We only fire on a positive, below-threshold measurement.
  //
  // 素材（末尾）(2026-08-15): the floor is applied to VIDEO material only. Every
  // action that had a non-null floor before this change required a video
  // outright, so the added `kind === "video"` condition cannot change any of
  // their behaviour (verified against `lengthFloorForAction`'s three cases:
  // extendVideo / referenceVideo / referenceVideoChain, all `requiredKind:
  // "video"`). It matters for `endWithThis`, the first action that accepts
  // either kind: an IMAGE has no duration to compare — `mediaDurationSec` is 0
  // (or meaningless) for a still — so measuring it would either be a no-op or,
  // worse, a false "too short" on a valid still image.
  //
  // v2 (2026-08-15): the comparison additionally allows this action's own
  // measurement slack (`lengthMarginForAction`, non-zero only for
  // `endWithThis`) — see {@link END_SOURCE_DURATION_MARGIN_SEC} for why a
  // container duration needs it. Every pre-existing action gets a margin of 0,
  // i.e. the byte-identical comparison it had before.
  const minSeconds = lengthFloorForAction(route.action);
  if (minSeconds !== null && item && kind === "video") {
    const decision = decideSourceTrim(item, selection);
    const measured = decision.trim ? decision.durationSec : item.mediaDurationSec;
    const threshold = minSeconds - lengthMarginForAction(route.action);
    if (measured !== undefined && measured > 0 && measured < threshold) {
      return { ok: false, note: { noteKind: "videoTooShort", requiredSeconds: minSeconds } };
    }
  }

  // (e) W0 範囲系ガード: `retakeRange` regenerates the user's SELECTED FRAME
  // RANGE, so the range — not the object — is the real input. With nothing
  // selected there is no window to retake, and every downstream step (the
  // placement系統 "D" head-alignment, W3's window math) would silently fall back
  // to frame 0. So refuse up front,同じく破壊的動作なしで. Keyed on the action
  // name, matching the (c) near-neighbor special case's inline style rather than
  // introducing a second metadata table.
  //
  // This wakes `timeline.getSelection`'s `hasRange` — filled by native from
  // `EDIT_SECTION::info->select_range_start/end` (`native/src/bridge.cpp`) since
  // contract v5, but never read by production code until now.
  //
  // The 最小窓未満 check (`rangeTooShort`) is raised just below (W3).
  if (route.action === "retakeRange") {
    if (!selection.hasRange) return { ok: false, note: { noteKind: "rangeNotSelected" } };
    const tooShort = retakeRangeTooShort(selection);
    if (tooShort !== null) {
      return { ok: false, note: { noteKind: "rangeTooShort", requiredFrames: tooShort } };
    }
  }

  return { ok: true };
}

/**
 * W3 §1-17: is the selected range too short for ANY retake window, no matter
 * which generation frame rate the user later picks? Returns the number of
 * PROJECT frames the user would have to select (the number the note quotes),
 * or `null` when the range passes.
 *
 * ## Why the threshold is deliberately loose
 *
 * The real gate is `useRetakeForm`'s, which knows the actual generation fps and
 * runs `resolveRetakeWindow` against the real material. This guard fires at
 * RIGHT-CLICK time, before any of that exists, and its only job is to refuse
 * the cases that cannot possibly work — because refusing here is destructive to
 * the user's flow (the panel never opens, nothing is placed) while letting a
 * borderline range through merely means the panel opens and explains itself.
 *
 * So the bound is taken at the MOST generous generation frame rate the form
 * allows (60 fps, the `frame_rate` input's own max): the shortest window,
 * `RETAKE_WINDOW_MIN_PX` frames, lasts `73/60 ≈ 1.217 s` there, and at any
 * lower generation fps it lasts longer. A range shorter than that is short at
 * every setting.
 *
 * The project's own fps comes from the snapshot (`rate`/`scale`) — the guard
 * has always had it. An unresolvable rate/scale means we cannot convert frames
 * to seconds, so we pass (the same "measure nothing, block nothing" bias the
 * (d) length guard above uses for an unknown `mediaDurationSec`).
 */
export function retakeRangeTooShort(selection: TimelineSelection): number | null {
  const { rate, scale } = selection;
  if (!(rate > 0) || !(scale > 0)) return null;
  const projectFps = rate / scale;
  const rangeFrames = selectionRangeFrames(selection);
  if (rangeFrames <= 0) return null; // `hasRange` already handled the empty case
  const minSec = RETAKE_WINDOW_MIN_PX / RETAKE_MAX_GEN_FPS;
  if (rangeFrames / projectFps >= minSec) return null;
  // The note asks the user for PROJECT frames, since that is what they select.
  return Math.ceil(minSec * projectFps);
}

/** The §4-5 length floor (seconds) for an action, or `null` when the action has
 * no length guard (only #1 `extendVideo`, the two IC-LoRA items —
 * `referenceVideo` and 台帳§1-15 W4's `referenceVideoChain` — and 素材（末尾）'s
 * `endWithThis` do). Pure. NOTE the floor is only ever applied to VIDEO
 * material; see the (d) block in `guardMenuSelection`. */
export function lengthFloorForAction(action: string): number | null {
  switch (action) {
    case "extendVideo":
      return V2V_MIN_DURATION_SEC;
    // 素材（末尾）v2 (2026-08-15): 25 frames ≈ 1s — the owner-picked usefulness
    // floor, NOT a geometry one (the band is derived from the material now, so
    // no preset can out-run it). See END_SOURCE_MIN_DURATION_SEC's own doc.
    case "endWithThis":
      return END_SOURCE_MIN_DURATION_SEC;
    case "referenceVideo":
    // 台帳§1-15 W4 (2026-08-11): the Chain-targeted IC-LoRA item conditions on
    // the reference exactly the way #2 does, so it needs the same 25-frame
    // floor — the difference is only where the material lands.
    case "referenceVideoChain":
      return ICLORA_MIN_DURATION_SEC;
    default:
      return null;
  }
}

/** The measurement slack (seconds) the §4-5 length guard allows an action —
 * `0` for everything except 素材（末尾）'s `endWithThis`, which gets one
 * generation frame ({@link END_SOURCE_DURATION_MARGIN_SEC}). Pure. Kept as its
 * own tiny table rather than folded into {@link lengthFloorForAction}'s return
 * value so that the number the failure NOTE quotes stays the honest floor, not
 * a floor-minus-epsilon the user would then have to reverse-engineer. */
export function lengthMarginForAction(action: string): number {
  return action === "endWithThis" ? END_SOURCE_DURATION_MARGIN_SEC : 0;
}

/** The `notes` slice of a language dictionary (`en.notes` / `ja.notes`) — the
 * shape `formatMenuGuardNote` needs to render a guard note. */
export type NotesStrings = (typeof en)["notes"];

/**
 * Pure: render a `MenuGuardNote` to its localized message using the supplied
 * `notes` dictionary slice (§4-2/§4-3/§4-5). Kept out of `guardMenuSelection`
 * so the guard stays free of the i18n layer; the display-name resolution for
 * the mismatch template (§4-3 〔種別〕) also happens here, via `notes.kindName`.
 */
export function formatMenuGuardNote(note: MenuGuardNote, notes: NotesStrings): string {
  switch (note.noteKind) {
    case "multipleSelection":
      return notes.multipleSelection;
    case "unsupportedType":
      return notes.unsupportedType;
    case "typeMismatch":
      return notes.typeMismatch(notes.kindName(note.selected), notes.kindName(note.required));
    case "typeMismatchAny":
      return notes.typeMismatchAny(
        notes.kindName(note.selected),
        note.required.map((kind) => notes.kindName(kind)),
      );
    case "useVideoAudioInstead":
      return notes.useVideoAudioInstead;
    case "useVideoAudioLongInstead":
      return notes.useVideoAudioLongInstead;
    case "videoTooShort":
      return notes.videoTooShort(note.requiredSeconds);
    case "rangeNotSelected":
      return notes.rangeNotSelected;
    case "rangeTooShort":
      return notes.rangeTooShort(note.requiredFrames);
  }
}
