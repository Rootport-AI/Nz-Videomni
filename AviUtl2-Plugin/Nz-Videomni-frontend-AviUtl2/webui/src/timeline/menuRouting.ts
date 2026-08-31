/**
 * Timeline context-menu action routing (contract v5, right-click redesign I2).
 *
 * Native pushes a `timeline.menuInvoked` event carrying a stable ASCII
 * `action` identifier (see `plugin.cpp`'s `EmitMenuInvoked`) whenever the user
 * invokes one of the plugin's timeline context-menu commands. This module is
 * the pure mapping from that identifier to *what the WebUI should do*: which
 * top-level mode to open (`single`/`chained`/`inventory`), an `intent` string the
 * mode can branch on to pre-select its sub-flow, and whether the command
 * operates on a selected timeline object (`needsSelection`).
 *
 * It is deliberately side-effect-free (no bridge, no React) so the whole table
 * can be exhaustively unit-tested; the `useMenuRouter` hook applies it.
 *
 * IMPORTANT — identifier source of truth: the identifiers below must match the
 * strings `plugin.cpp` emits. The shipping plugin emits the twenty concrete
 * object/layer identifiers of the right-click redesign
 * (`extendVideo`, `referenceVideo`, `referenceVideoChain`, `videoAudioToVideo`,
 * `videoAudioToLongA2v`,
 * `imageToVideo`, `addImageKeyframe`, `imageToClipChain`, `endWithThis`,
 * `audioToVideo`,
 * `audioToLongA2v`, `appendText`, `insertProvisionalResult`, `outpaintVideo`,
 * `retakeRange` for the object menu;
 * `textToVideoHere`, `imageFromCurrentFrame`, `addCurrentFrameAsKeyframe`,
 * `currentFrameToClipChain`, `insertLatestResultHere` for the layer menu)
 * directly from its context-menu items (`kObjectMenuItems`/`kLayerMenuItems`),
 * so this table's job is routing exactly those. An action with no table entry
 * routes to `null` (the hook no-ops), so an unknown/new native identifier can
 * never throw.
 *
 * SCOPE: I2 defined `targetMode` / `intent` / `needsSelection`. I6 adds the two
 * pure metadata fields the type/mismatch/length guards and the provisional-flow
 * placement need: `requiredKind` (§3-4/§3-6 "必須種別", §4) and `placement`
 * (§5-9 配置系統). Both are static, side-effect-free table data; the guards that
 * consume them live in `menuSelection.ts` and the AppShell wiring is I7.
 */

/** The top-level WebUI modes an action can target. `"edit"` joined on
 * 2026-08-09 (W0 共通スパイン) when the Edit tab — promoted to a real mode the
 * same day — became a right-click destination for the two Edit-系 commands
 * (`outpaintVideo`/`retakeRange`). Kept a SUBSET of `AppShell`'s `AppMode`, so
 * `GenerationPrefill.targetMode` (typed `AppMode`) accepts every value here. */
export type MenuTargetMode = "single" | "chained" | "edit" | "inventory";

/** The four selectable timeline object kinds the type check distinguishes
 * (Docs/RIGHTCLICK_REDESIGN_SPEC.md §4-1). */
export type MaterialKind = "video" | "image" | "audio" | "text";

/** The object kind an action requires to run correctly (§3-4/§3-6 "必須種別"),
 * or `null` for the two layer-menu commands (#9/#10) which generate at a
 * position and are therefore exempt from the type-mismatch check (§3-6).
 *
 * 素材（末尾）(2026-08-15) widened this to accept an ARRAY: `["video","image"]`
 * means "any ONE of this set passes" — the action genuinely works on either
 * kind, and the guard refuses everything outside the set. This is a DIFFERENT
 * meaning from `null`: `null` EXEMPTS the layer-menu items from the type check
 * altogether (there is no object to check), whereas an array still checks —
 * it just accepts more than one answer. Single-value entries are unchanged and
 * keep the exact behaviour they had. */
export type MenuRequiredKind = MaterialKind | readonly MaterialKind[] | null;

/** The provisional-object placement系統 an action's generated result uses
 * (§5-9). Native computes the concrete layer/frame from the selection's frame
 * range; this only names which system applies:
 *  - `"A"` 素材オブジェクトの直後 (after the material's tail; native falls back to
 *    the same frame on `layer_max+1` if that slot is occupied — §5-4).
 *  - `"B"` 素材と同じ開始フレームで `layer_max+1` (head-aligned, overlaid on a
 *    fresh frontmost layer).
 *  - `"C"` 右クリック位置 (the `EDIT_INFO` cursor position; the layer-menu items).
 *  - `"D"` 選択範囲の開始フレームで `layer_max+1` (W0, 2026-08-09). The Retake
 *    input is the SELECTED FRAME RANGE, not the object's own span, so the
 *    provisional is head-aligned with that range. Native has NO `"D"`: its
 *    formula is IDENTICAL to `"B"` (`bridge_core.cpp`'s
 *    `ResolveProvisionalPlacement`, `kSameStartFront` → `layer = layer_max + 1`,
 *    `frame = material_frame_start`, re-verified against the current source on
 *    2026-08-09), so `provisionalReservation.ts`'s `placementParams` MAPS `"D"`
 *    onto `"B"` and the caller simply fills `material` from the selection range
 *    instead of the object. `bridge/types.ts` types the wire field as the
 *    literal union `"A" | "B" | "C"`, so a missed mapping is a COMPILE error,
 *    not a runtime one. It is a distinct系統 here (rather than plain `"B"`)
 *    because the位置の意図 genuinely differs — `"B"` means "頭を素材に揃える",
 *    `"D"` means "頭を選択範囲に揃える".
 *  - `"E"` 素材の**手前**に、末尾が素材に重なる位置 (素材（末尾）v2,
 *    2026-08-15). 「これで終わる動画を作る」の出力は末尾の帯が素材そのものなので、
 *    正しい位置は「素材開始 − (出力長 − 帯長)」——素材の手前へ伸ばす配置になる。
 *    Native has NO `"E"`: like `"D"`, its wire形式 is IDENTICAL to `"B"`
 *    (`layer = layer_max + 1`, `frame = material_frame_start`), so
 *    `provisionalReservation.ts`'s `placementParams` MAPS `"E"` onto `"B"` and
 *    the CALLER puts the already-shifted frame into `material.frameStart`
 *    (`timeline/tailAlign.ts`'s `tailAlignedStartFrame`). Exactly the same
 *    discipline 系統D uses, and the same reason: `bridge/types.ts` types the
 *    wire field as the literal union `"A" | "B" | "C"`, so a missed mapping is a
 *    COMPILE error. It is a distinct系統 here (rather than plain `"B"`) because
 *    the位置の意図 genuinely differs — `"B"`/`"D"` align a HEAD, `"E"` aligns a
 *    TAIL — and because only `"E"` needs the caller to compute a shift at all.
 *  - `null` has three distinct meanings, all "this route places nothing by
 *    itself":
 *    (1) 実行時分岐 — #5 `addImageKeyframe` / #11 `addCurrentFrameAsKeyframe`:
 *        whether a provisional is placed (system A / C) or left untouched
 *        depends on the *live keyframe count at append time* (1枚目になる場合
 *        のみ配置 — §3-4 #5 branch / §5-9). The static route cannot know that
 *        runtime count, so AppShell's live-append channel resolves it.
 *    (2) 仮オブジェクトを伴わない早期リターン — W2 `insertProvisionalResult` /
 *        W3 `insertLatestResultHere`, which insert finished media directly.
 *    (3) 配置は後続ストリームが決める — W0's `outpaintVideo`: it flows through
 *        the NORMAL route path (guard → busy-guard → prefill → Step 8) but
 *        Step 8 places nothing, because Outpainting's output geometry is W2's
 *        design decision (`Docs/OUTPAINTING_DESIGN_NOTES.md`). This is the only
 *        `null` route that reaches Step 8 at all. */
export type MenuPlacement = "A" | "B" | "C" | "D" | "E" | null;

/** The routing decision for one action (without the action key itself). */
export interface MenuRouteInfo {
  /** Which top-level screen the action should open. */
  targetMode: MenuTargetMode;
  /** Stable sub-flow identifier the target mode can branch on to pre-select
   * its variant (e.g. `"image-to-video"` vs `"text-to-video"`). Intentionally
   * decoupled from the native action name so the WebUI's internal intents can
   * be renamed/merged without a native contract change. */
  intent: string;
  /** True when the action operates on the currently selected timeline
   * object(s) (object-menu commands); false for layer/cursor commands that
   * generate at a position rather than from an existing object. Drives the
   * two-stage filePath completion (see `menuSelection.ts`). */
  needsSelection: boolean;
  /** The object kind this action requires (§3-4/§3-6 "必須種別"); `null` for the
   * layer-menu commands (#9/#10), which are exempt from the type-mismatch check.
   * Consumed by `guardMenuSelection` (`menuSelection.ts`). */
  requiredKind: MenuRequiredKind;
  /** Which provisional-object placement系統 the generated result uses (§5-9).
   * `null` only for #5 (`addImageKeyframe`), whose placement branches on the
   * live keyframe count at append time (see `MenuPlacement`). */
  placement: MenuPlacement;
}

/** A resolved route: `action` plus its `MenuRouteInfo`. */
export interface MenuRoute extends MenuRouteInfo {
  action: string;
}

/**
 * The action -> route table (Docs/RIGHTCLICK_REDESIGN_SPEC.md §3-4 / §3-6,
 * mapping in §7-1).
 *
 * Design judgments (per spec):
 *  - Object commands whose output is a *new clip that extends from / chains off
 *    the selected object* — `extendVideo` (v2v continuation) and
 *    `imageToClipChain` (long i2v via Clip Chain) — route to **chained**, whose
 *    `source_video`/multi-clip machinery is exactly the home for that.
 *  - `referenceVideo` (IC-LoRA) routes to **single**: it is the single-shot
 *    IC-LoRA generation, the one the Create form's reference slot serves.
 *    (`/generate/chain` had no `reference_video_id` at all when this table was
 *    written — see Docs/TIMELINE_ALPHA_REQUIREMENTS.md §2, §7-3-A / SPEC §3-4
 *    #2 — which is why this was the ONLY IC-LoRA route for a long time.)
 *  - `referenceVideoChain` (台帳§1-15 W4, 2026-08-11) is that item's long-form
 *    sibling and routes to **chained**: once `/generate/chain` grew its own
 *    `reference_video_id`, one reference could condition a whole clip chain,
 *    so the same material now has two destinations and the user picks which by
 *    picking the menu item. Same `requiredKind`/`placement` as `referenceVideo`
 *    — only the target mode and the intent differ, exactly like the §1-16
 *    long-a2v pair below.
 *  - `videoAudioToVideo` (#3, a2v from a video's sound) and `audioToVideo`
 *    (#7, a2v from an audio object) both route to **single**: single-shot A2V
 *    lives on the Create form (the standalone-A2V move of 2026-07-15). NOTE:
 *    `audioToVideo` used to route to **chained** in the骨組み table; that Chain
 *    audio path is fully de-wired, so the redesign corrects it to single
 *    (SPEC §7-1 note). Their §1-16 long-form siblings `videoAudioToLongA2v` /
 *    `audioToLongA2v` (2026-08-10) DO route to **chained**: that Chain audio
 *    path is wired again — as ONE track spread over the whole clip list by the
 *    server, which is a different feature from the骨組み's per-clip audio, and
 *    the only place a multi-clip A2V can live.
 *  - Object/layer commands whose output is a *fresh single generation* —
 *    `imageToVideo` (i2v), `addImageKeyframe` (live keyframe append),
 *    `textToVideoHere` (t2v), `imageFromCurrentFrame` (frame-grab i2v) — route
 *    to **single**, the single-shot generation form.
 *  - `appendText` (#8) is special: it appends the text object's body to the
 *    shared prompt with NO screen transition or remount. `targetMode` is set to
 *    `single` only as a formal/default value — `AppShell.handleRoute` is
 *    expected to early-return on this action in a later increment and never act
 *    on the `targetMode` (SPEC §3-4 #8 / §5-9 system B). `needsSelection` stays
 *    true because it still reads the selected text object's body.
 */
export const MENU_ROUTING_TABLE: Readonly<Record<string, MenuRouteInfo>> = {
  // --- Object-menu commands (operate on the selected object) ---------------
  /** #1 v2v: Chain画面. Continue the selected video with a generated tail.
   * Result extends off the material's tail (placement A). */
  extendVideo: {
    targetMode: "chained", intent: "extend-video", needsSelection: true,
    requiredKind: "video", placement: "A",
  },
  /** #2 IC-LoRA: Create画面. Use the selected video as a conditioning
   * reference. `/generate`-only, hence create (not chain). Result overlays the
   * material head-aligned (placement B). */
  referenceVideo: {
    targetMode: "single", intent: "reference-video", needsSelection: true,
    requiredKind: "video", placement: "B",
  },
  /** 台帳§1-15 W4 (2026-08-11): Chain画面. The long-form sibling of #2 — the
   * SAME selected video, attached to the CHAIN's reference slot
   * (`ChainReferencePanel`) so one IC-LoRA reference conditions the whole clip
   * chain instead of a single generation. Requires a video and overlays the
   * material head-aligned (placement B), matching #2. */
  referenceVideoChain: {
    targetMode: "chained", intent: "reference-video-chain", needsSelection: true,
    requiredKind: "video", placement: "B",
  },
  /** #3 a2v-from-video: Create画面. Extract the selected video's sound to a
   * wav and run A2V from it. Result head-aligned with the source (placement B). */
  videoAudioToVideo: {
    targetMode: "single", intent: "video-audio-to-video", needsSelection: true,
    requiredKind: "video", placement: "B",
  },
  /** §1-16 長尺A2V (2026-08-10): Chain画面. The long-form sibling of #3 — the
   * SAME mix-path extraction of the selected video's sound over its ribbon
   * range, but the resulting wav is attached to the whole clip chain
   * (`ChainAudioPanel`), not to a single generation. Requires a video, and the
   * result overlays the material head-aligned (placement B), matching #3. */
  videoAudioToLongA2v: {
    targetMode: "chained", intent: "video-audio-to-long-a2v", needsSelection: true,
    requiredKind: "video", placement: "B",
  },
  /** #4 i2v: Create画面. Generate a video from the selected image (frame-0
   * keyframe). Result starts after the image (placement A). */
  imageToVideo: {
    targetMode: "single", intent: "image-to-video", needsSelection: true,
    requiredKind: "image", placement: "A",
  },
  /** #5 keyframe append: Create画面 (live append, no remount). Add the
   * selected image to the end of the current keyframe list. Placement branches
   * on the live keyframe count at append time (`null`; §3-4 #5 branch / §5-9):
   * system A only when this append makes the image the 1st keyframe, else no
   * provisional at all. */
  addImageKeyframe: {
    targetMode: "single", intent: "add-image-keyframe", needsSelection: true,
    requiredKind: "image", placement: null,
  },
  /** #6 i2v Clip Chain: Chain画面. Set the selected image as the chain's
   * opening keyframe for a long-form generation. Result starts after the image
   * (placement A). */
  imageToClipChain: {
    targetMode: "chained", intent: "image-to-clip-chain", needsSelection: true,
    requiredKind: "image", placement: "A",
  },
  /** 素材（末尾）(2026-08-15): Chain画面. Generate a video that ENDS with the
   * selected material — the mirror image of #1 `extendVideo`, which generates
   * what comes AFTER it. Two differences from #1, both deliberate:
   *  - `requiredKind` is the SET `["video", "image"]`, not a single kind: the
   *    end slot takes either (a still image is turned into a still video by the
   *    backend and goes down the identical freeze path), so refusing an image
   *    here would refuse a supported input.
   *  - `placement: "E"` (末尾合わせ, v2 2026-08-15). The物理的に正しい系統は
   *    「素材の**手前**に、末尾の帯が素材に重なる位置」で、v1 はそれを native に
   *    系統が無いという理由で `"B"`（頭揃え）に妥協していた。v2 では
   *    `provisionalReservation.ts` が `"E"` を `"B"` へ写像し、呼び出し側
   *    (`AppShell` Step 8) が `timeline/tailAlign.ts` でシフト済みの開始フレーム
   *    を `material.frameStart` に入れる——系統Dと同じ手口なので **C++ 変更は
   *    やはり不要**のまま、意図どおりの位置に置ける。 */
  endWithThis: {
    targetMode: "chained", intent: "end-with-this", needsSelection: true,
    requiredKind: ["video", "image"], placement: "E",
  },
  /** #7 a2v-from-audio: Create画面. Upload the selected audio file directly
   * and enable A2V (corrected from the骨組み's chain route; SPEC §7-1). Result
   * head-aligned with the audio (placement B). */
  audioToVideo: {
    targetMode: "single", intent: "audio-to-video", needsSelection: true,
    requiredKind: "audio", placement: "B",
  },
  /** §1-16 長尺A2V (2026-08-10): Chain画面. The long-form sibling of #7. Unlike
   * #7 — which uploads the audio object's BACKING FILE whole — this one extracts
   * the object's own ribbon range through `timeline.extractAudio` in SOLO mode
   * (only the object's layer kept audible), so a trimmed audio object
   * contributes exactly the seconds the ribbon shows. Requires an audio object;
   * head-aligned placement (B), matching #7. */
  audioToLongA2v: {
    targetMode: "chained", intent: "audio-to-long-a2v", needsSelection: true,
    requiredKind: "audio", placement: "B",
  },
  /** #8 append text to the shared prompt. No transition/remount — handled by an
   * early-return in AppShell.handleRoute; `targetMode` is a formal default
   * only (see table doc above). Still a generation origin: a provisional is
   * placed head-aligned with the text object (placement B; §3-4 #8 / §5-9). */
  appendText: {
    targetMode: "single", intent: "append-text", needsSelection: true,
    requiredKind: "text", placement: "B",
  },
  /** W2 ⬇ quick-insert: right-click a Nz-Videomni provisional object and, when its
   * job is already complete, run the SAME replace-insert the panel's 🎞 button
   * does. Handled entirely by an early-return in `AppShell.handleRoute` keyed on
   * the object's `NzVideomni#<jobId>` name — no mode switch, no reservation, no
   * type check (`requiredKind: null`, so `guardMenuSelection` always passes) and
   * no provisional placement (`placement: null`). `needsSelection: false` skips
   * the file-path re-query: the flow reads only the object's name, which the
   * `menuInvoked` snapshot already carries. `targetMode`/`intent` are formal
   * defaults never acted on (the early-return fires first). */
  insertProvisionalResult: {
    targetMode: "single", intent: "insert-provisional-result", needsSelection: false,
    requiredKind: null, placement: null,
  },
  /** W0 Outpainting (2026-08-09): Edit画面 Outpainting サブタブ. Extend the
   * selected video's CANVAS outward (top/bottom/left/right) — a spatial, not a
   * temporal, extension, hence a different flow from #1 `extendVideo`. Requires
   * a video. `placement: null` (MenuPlacement doc, meaning 3): the route runs
   * the normal path but reserves nothing, because where an outpainted result
   * belongs on the timeline is W2's design decision
   * (`Docs/OUTPAINTING_DESIGN_NOTES.md`). NOTE this makes it the first
   * null-placement route that reaches AppShell's §5-6 busy-guard — deliberate:
   * α runs one job at a time, so a route into a form that cannot generate yet
   * is better refused with the standard note than opened. */
  outpaintVideo: {
    targetMode: "edit", intent: "outpaint", needsSelection: true,
    requiredKind: "video", placement: null,
  },
  /** W0 Retake (2026-08-09): Edit画面 Retake サブタブ. Regenerate ONLY the
   * user's selected frame range of the selected video, leaving the rest as it
   * is. Requires a video AND a frame range — `guardMenuSelection` refuses it
   * when `timeline.getSelection` reports `hasRange: false` (§4 範囲系ガード).
   * Placement `"D"`: head-aligned with the SELECTED RANGE (not the object), TS-
   * mapped onto native's `"B"` — see `MenuPlacement`. */
  retakeRange: {
    targetMode: "edit", intent: "retake", needsSelection: true,
    requiredKind: "video", placement: "D",
  },

  // --- Layer-menu commands (operate at a cursor/layer position) ------------
  /** #9 t2v: Create画面. Insert a provisional object at the cursor and open a
   * fresh Create form. Exempt from the type check (`requiredKind: null`);
   * placed at the right-click position (placement C). */
  textToVideoHere: {
    targetMode: "single", intent: "text-to-video", needsSelection: false,
    requiredKind: null, placement: "C",
  },
  /** #10 i2v from the current frame: Create画面. Grab the current frame as the
   * first keyframe. Exempt from the type check (`requiredKind: null`); placed at
   * the right-click position (placement C). */
  imageFromCurrentFrame: {
    targetMode: "single", intent: "image-from-frame", needsSelection: false,
    requiredKind: null, placement: "C",
  },
  /** #11 add-current-frame-keyframe (W5): Create画面 (live append, no remount).
   * Capture the current frame and append it as a keyframe at the END of the
   * keyframe list. Position-based (`requiredKind: null`). Placement branches on
   * the live keyframe count at append time exactly like #5 `addImageKeyframe`
   * (`null`; §5-9): a provisional at the cursor (system C) only when this append
   * makes the capture the 1st keyframe, else no provisional at all. */
  addCurrentFrameAsKeyframe: {
    targetMode: "single", intent: "add-current-frame-keyframe", needsSelection: false,
    requiredKind: null, placement: null,
  },
  /** #12 current-frame-to-clip-chain (W5): Chain画面. Capture the current frame
   * and set it as the chain's opening keyframe (Clip 1's head) for a long-form
   * i2v generation. Position-based (`requiredKind: null`); placed at the
   * right-click position (placement C). */
  currentFrameToClipChain: {
    targetMode: "chained", intent: "current-frame-to-clip-chain", needsSelection: false,
    requiredKind: null, placement: "C",
  },
  /** W3 ⬇ quick-insert here: insert the most-recent completed generation result
   * at the right-click position as a PLAIN insert (`downloadAndInsertVideo`'s
   * `plainInsertAt`), so it never goes through the replace-RPC and any
   * provisional at the cursor is left untouched. Position-based
   * (`requiredKind: null`, exempt from the type check). Handled by an
   * early-return in `AppShell.handleRoute`; `placement: null` because the flow
   * inserts the finished media directly rather than reserving a provisional, so
   * `targetMode`/`intent` are formal defaults never acted on. */
  insertLatestResultHere: {
    targetMode: "single", intent: "insert-latest-result", needsSelection: false,
    requiredKind: null, placement: null,
  },
};

/**
 * Resolve a native menu action to its route, or `null` when the action has no
 * table entry (unknown / not-yet-supported identifier). Pure.
 */
export function routeMenuAction(action: string): MenuRoute | null {
  const info = MENU_ROUTING_TABLE[action];
  if (!info) return null;
  return { action, ...info };
}

/** All action identifiers the table knows about (handy for tests/UI menus). */
export function knownMenuActions(): string[] {
  return Object.keys(MENU_ROUTING_TABLE);
}

/** Reverse index of {@link MENU_ROUTING_TABLE}, keyed on `intent`. Every intent
 * string appears on exactly ONE row, so the reverse direction is a function —
 * built once at module load rather than re-scanned per lookup. */
const TARGET_MODE_BY_INTENT: ReadonlyMap<string, MenuTargetMode> = new Map(
  Object.values(MENU_ROUTING_TABLE).map((info) => [info.intent, info.targetMode]),
);

/**
 * Which screen a routed `intent` belongs to, or `null` for an intent the table
 * has no row for. The inverse of {@link routeMenuAction}'s `intent` output, for
 * the code paths downstream of routing that only ever see the intent string
 * (`timeline/prefillSeed.ts`).
 *
 * Its one caller uses this to decide whether the SINGLE screen's comfort
 * ceiling applies to a prefill's DURATION seed, so note the shape of the
 * answer: `"single"` comes back for 11 rows, but only #2/#3/#4/#7 carry a
 * DURATION policy at all (`prefillSeed.DURATION_POLICY_BY_INTENT` — the rest
 * are `untouched` and never reach a ceiling). Retake routes to `"edit"` and
 * `end-with-this` to `"chained"`, so neither can collide with
 * `END_SOURCE_SEED_MAX_FRAMES` or the Retake window rules. Pure. */
export function targetModeForIntent(intent: string): MenuTargetMode | null {
  return TARGET_MODE_BY_INTENT.get(intent) ?? null;
}
