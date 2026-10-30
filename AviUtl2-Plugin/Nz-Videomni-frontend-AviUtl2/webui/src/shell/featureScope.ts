import type { AccelerationSettings } from "./accelerationSettings";
import type { AppMode } from "./AppShell";

/**
 * §3-135: the ONE place a server-declared FEATURE name turns into a closed bit
 * of UI.
 *
 * The server names FEATURES (`chain`, `outpaint`, `prune_vaed`, …) because that
 * is what a request field is; the WebUI has TABS, PANELS, SUB-TABS and SETTINGS
 * ROWS. This module is the translation, and it is the ONLY file in the WebUI
 * that is allowed to know a server feature name — every screen below it
 * receives finished booleans and never reasons about engines itself (the
 * shell's standing rule for feature scope).
 *
 * Before §3-135 that knowledge was spread across four exported mappers in
 * `shell/useBaseModels.ts` plus two hand-written `unsupportedFeatures.includes`
 * calls inside `AppShell`, so "which UI does feature X close?" could only be
 * answered by grepping. It is now one table, {@link FEATURE_UI}, and every
 * mapper below is a derivation from it. The mappers keep their old signatures
 * and return shapes on purpose: §3-135's WP3 was a pure equivalence refactor,
 * and their tests moved across unchanged as the proof.
 *
 * What this module deliberately does NOT decide: how the closing LOOKS. Greyed
 * vs. hidden, the wording of the note, whether a sub-tab selection falls back to
 * its sibling — all of that stays in the screen that renders it. This table only
 * says WHICH part closes.
 */

/** The "real UI part" a feature name closes. The ids are defined here and
 * nowhere else: a screen never names one, it is handed the boolean. */
export type UiTarget =
  | "mode.chained"
  | "mode.edit"
  | "chainPanel.v2v"
  | "chainPanel.a2v"
  | "chainPanel.endSource"
  | "chainPanel.reference"
  | "chainPanel.startImage"
  | "chainSection.batchI2vLong"
  | "editSubTab.retake"
  | "editSubTab.outpainting"
  | "editSubTab.inpainting"
  | "editSubTab.alphaGen"
  | "createSection.batchA2v"
  | "createSection.referenceVideo"
  | "createSection.a2v"
  | "createSection.keyframeStrength"
  | "promptBar.loraAudioStrength"
  | "shell.nag"
  | "settingsRow.vae"
  | "settingsRow.keepResidentEmbeddings"
  | "settingsRow.vram16"
  | "settingsRow.fusedGgufDequantKernel";

interface FeatureUiEntry {
  /** The UI parts to close while this feature is unavailable. A disjunction —
   * if ANY ONE of the features that list a target is unsupported, that target
   * closes. */
  readonly disables: readonly UiTarget[];
  /** Cleanup: having hidden a row, write this saved acceleration field back to
   * the SERVER default (the defaults themselves live in
   * `ACCELERATION_DEFAULTS` — only the field NAME belongs here). Hiding a row
   * is not enough on its own, because the choice PERSISTS: a value picked on a
   * base model that supports it would otherwise keep riding along on every
   * request after a switch and 422 every job. Where the server IGNORES the
   * field instead (`fused_gguf_dequant_kernel` on MiniMax H3), the write-back
   * stays all the same: the rule "a hidden row returns to its default" then
   * has no exception, and an OFF the user could not see there is not carried
   * back to LTX. */
  readonly resets?: keyof AccelerationSettings;
}

/**
 * The declaration table — feature name → UI (§3-135).
 *
 * Read it as "while the loaded engine cannot do X, these parts of the app are
 * closed". The reasoning behind the individual rows:
 *
 *  - **chain** is the Chained tab's whole endpoint, so it settles that tab on
 *    its own — and it takes the Batch A2V section with it, since every row that
 *    section queues is a `POST /generate/chain`.
 *  - **a2v** closes the Chain screen's audio panel and, again, Batch A2V: every
 *    batch row carries an audio source, so either limitation takes the panel
 *    down (that "or" is now the table's disjunction rather than a hand-written
 *    `||`).
 *  - **v2v / end_source / reference_video** are exactly one Chain panel each,
 *    because each panel IS one request field: `source_video`, `end_source`,
 *    `reference_video_id` (§3-102, LTX 2.5 Chained's first stage — once an
 *    engine can chain, the tab is live and the panels it still cannot feed have
 *    to grey on their own).
 *  - **chain_start_image** (§1-106, MiniMax H3's text-only chain) is the
 *    request field `clips[0].conditioning_images`: it closes the image half of
 *    the Chain screen's SOURCE card, and the Batch i2v-long section with it —
 *    every row of that section puts its image on clip 0.
 *  - **retake / outpaint / inpaint** are the three Edit sub-tabs. Note the
 *    SERVER's features are `outpaint`/`inpaint` (that is what a request field
 *    and a 422 say) while the sub-tabs are `outpainting`/`inpainting`; this
 *    table is exactly where that translation lives, and nothing downstream has
 *    to know either name. Which engines publish these names is the server's
 *    fact, not this table's: the rows are a translation, not a list of
 *    today's limitations, and a name nobody publishes closes nothing (see
 *    `disabledUiTargets`).
 *  - **alpha_gen** is the Alpha Gen Edit sub-tab; an engine without the
 *    alpha-gen IC-LoRA (it exists for LTX 2.5 only) publishes it.
 *  - **Create's own sections** have targets of their own, because an engine
 *    that can generate at all but not in LTX's every way has to close parts
 *    of Single:
 *     - `reference_video` also takes Create's reference-video (IC-LoRA)
 *       accordion — the same request field (`reference_video_id`) as Chain's
 *       panel. Create HIDES it (IC-LoRA is LTX's own mechanism).
 *     - `chain` also takes Create's A2V card: an attached audio turns a Create
 *       submit into `POST /generate/chain`, so a base model that cannot chain
 *       cannot run it either. Greyed — unless the engine publishes
 *       `constraints.source_audio` (`a2vRouteFor` then routes the card to `/generate`).
 *     - `nag` closes the shared Negative Prompt accordion. Greyed rather than
 *       hidden, and the shell stops sending its fields at the same time
 *       (`nagSettings.effectiveNag`) — hiding alone would strand a NAG left
 *       enabled on another base model, 422-ing every job with no control left
 *       to turn it off.
 *     - `keyframe_strength` HIDES each keyframe card's strength row, and the
 *       shell stops sending `strength` at the same time
 *       (`keyframeUtils.omitStrength`): the engine has no keyframe strength,
 *       and omitted is what the job record then shows (not a stand-in 1.0).
 *    LTX publishes none of these names, so none of this reaches it.
 *  - **lora_audio_strength** (台帳 §1-104) is scoped like `keyframe_strength`:
 *    published by MiniMax H3, whose LoRAs have one strength only. It HIDES
 *    each LoRA chip's audio toggle (and its 🎥 / audio value), and the shell
 *    stops sending `audio_strength` at the same time
 *    (`loraTags.omitLoraAudioStrength`).
 *  - **prune_vaed / keep_resident_embeddings** are Settings rows rather than
 *    generation surfaces, and both carry a `resets` because their choice is
 *    persisted in `localStorage`. Their SCOPES point in opposite directions:
 *    an engine that publishes `prune_vaed` refuses the pruned decoder, while
 *    `keep_resident_embeddings` names a component only LTX 2.5 HAS, so it is
 *    the engines without it that publish it. The table does not care — either way the row
 *    closes and the stored value goes back to the server default.
 *  - **vram16_mode** (台帳 §1-101) is a third Settings row with a `resets`,
 *    scoped like `keep_resident_embeddings`: it exists on MiniMax H3 only, so
 *    LTX 2.3 / 2.5 publish the name. Its default is ON, so the write-back
 *    lands on `true` — which is what an LTX request then omits.
 *  - **fused_gguf_dequant_kernel** (台帳 §1-100) is a fourth Settings row with
 *    a `resets`, published by MiniMax H3, which has no GGUF weights for the
 *    kernel to unpack. Unlike the rows above, the server IGNORES the field
 *    there rather than 422-ing it; the row hides and writes back (to ON) all
 *    the same — see {@link FeatureUiEntry}'s `resets`.
 *
 * Note what is NOT here. `single` and `inventory` have no row and never will:
 * Single IS the baseline any engine must serve, and Inventory only browses
 * finished files — it issues no generation at all, so no engine limitation can
 * reach it. `mode.edit` has no feature of its own either; see
 * {@link CONTAINER_TARGETS}.
 *
 * Exported for `featureScope.test.ts`'s table-health check (the name set and
 * "every row closes something"), which is the one assertion that cannot be made
 * through the derived mappers.
 */
export const FEATURE_UI = {
  chain: { disables: ["mode.chained", "createSection.batchA2v", "createSection.a2v"] },
  retake: { disables: ["editSubTab.retake"] },
  outpaint: { disables: ["editSubTab.outpainting"] },
  inpaint: { disables: ["editSubTab.inpainting"] },
  alpha_gen: { disables: ["editSubTab.alphaGen"] },
  v2v: { disables: ["chainPanel.v2v"] },
  a2v: { disables: ["chainPanel.a2v", "createSection.batchA2v"] },
  end_source: { disables: ["chainPanel.endSource"] },
  chain_start_image: { disables: ["chainPanel.startImage", "chainSection.batchI2vLong"] },
  reference_video: { disables: ["chainPanel.reference", "createSection.referenceVideo"] },
  nag: { disables: ["shell.nag"] },
  keyframe_strength: { disables: ["createSection.keyframeStrength"] },
  lora_audio_strength: { disables: ["promptBar.loraAudioStrength"] },
  prune_vaed: { disables: ["settingsRow.vae"], resets: "vaeMode" },
  keep_resident_embeddings: {
    disables: ["settingsRow.keepResidentEmbeddings"],
    resets: "keepResidentEmbeddings",
  },
  vram16_mode: { disables: ["settingsRow.vram16"], resets: "vram16Mode" },
  fused_gguf_dequant_kernel: {
    disables: ["settingsRow.fusedGgufDequantKernel"],
    resets: "fusedGgufDequantKernel",
  },
} as const satisfies Record<string, FeatureUiEntry>;

// No union type over the feature NAMES (no `UiRelevantFeature` or the like):
// every boundary below keeps `readonly string[]`. A build of this WebUI is
// older than the server it talks to more often than the reverse, so an unknown
// name is the ordinary case, not an error — and a feature with no UI target at
// all (`two_stage_hq`, `loras`, `sage_attention`, …) is simply absent
// from the table rather than listed with an empty `disables`. (`loras` itself
// has no UI target; its per-LoRA audio strength does — `lora_audio_strength`.)

/** A `Map`, not the plain object: looking `FEATURE_UI` up with an arbitrary
 * server string would also answer for `"constructor"`, `"__proto__"` and the
 * rest of `Object.prototype`, and an unknown name must map to nothing. */
const FEATURE_UI_BY_NAME: ReadonlyMap<string, FeatureUiEntry> = new Map(Object.entries(FEATURE_UI));

/** Targets no feature names directly — they close only once everything they
 * CONTAIN is closed.
 *
 * The Edit tab is the only one today: it survives as long as ONE of its
 * sub-tabs is runnable — a base model that could outpaint but not retake
 * would still have a use for the tab. Every real sub-tab belongs in its
 * `whenAll` (a mock closes nothing, but a real sub-tab keeps the tab alive on
 * its own).
 * Chained needs no entry here because `chain` names it directly.
 *
 * Invariant: SINGLE-PASS evaluation — a `whenAll` must never name another
 * container target, or the answer would depend on the order of this array.
 * Should a nested case ever arise, revisit the design then rather than
 * quietly ordering the list. */
const CONTAINER_TARGETS: ReadonlyArray<{ target: UiTarget; whenAll: readonly UiTarget[] }> = [
  {
    target: "mode.edit",
    whenAll: [
      "editSubTab.retake",
      "editSubTab.outpainting",
      "editSubTab.inpainting",
      "editSubTab.alphaGen",
    ],
  },
];

/**
 * Every UI part the loaded base model's `unsupported_features` closes.
 *
 * Unknown names are ignored rather than treated as suspicious — see the note
 * under {@link FEATURE_UI}. That is what makes a backend older than §3-98 P5
 * that publishes no list at all answer with an empty set, with no
 * special-casing anywhere.
 */
export function disabledUiTargets(unsupportedFeatures: readonly string[]): ReadonlySet<UiTarget> {
  const disabled = new Set<UiTarget>();
  for (const name of unsupportedFeatures) {
    const entry = FEATURE_UI_BY_NAME.get(name);
    if (!entry) continue;
    for (const target of entry.disables) disabled.add(target);
  }
  for (const { target, whenAll } of CONTAINER_TARGETS) {
    if (whenAll.every((t) => disabled.has(t))) disabled.add(target);
  }
  return disabled;
}

/** The saved acceleration fields that must go back to their server default
 * because the rows that own them are now hidden — see {@link FeatureUiEntry}'s
 * `resets`. Deduplicated, so two features pointing at one field write it back
 * once. `[]` for the ordinary case. */
export function accelerationResetsFor(
  unsupportedFeatures: readonly string[],
): readonly (keyof AccelerationSettings)[] {
  const fields = new Set<keyof AccelerationSettings>();
  for (const name of unsupportedFeatures) {
    const resets = FEATURE_UI_BY_NAME.get(name)?.resets;
    if (resets) fields.add(resets);
  }
  return [...fields];
}

/** The mode tabs, in the order {@link disabledModesFor} reports them. Single and
 * Inventory are absent for the reason {@link FEATURE_UI} gives. */
const MODE_TARGETS: ReadonlyArray<readonly [AppMode, UiTarget]> = [
  ["chained", "mode.chained"],
  ["edit", "mode.edit"],
];

/**
 * The mode tabs that `unsupportedFeatures` makes unreachable — the shell greys
 * these and bounces out of one if it is the current mode.
 *
 * Pure, exported and tested directly, like every mapper here: it must answer
 * `[]` for the ordinary case (LTX 2.3 / an older backend) without any
 * special-casing.
 */
export function disabledModesFor(unsupportedFeatures: readonly string[]): AppMode[] {
  const disabled = disabledUiTargets(unsupportedFeatures);
  return MODE_TARGETS.filter(([, target]) => disabled.has(target)).map(([mode]) => mode);
}

/**
 * Whether the Batch A2V panel (`modes/batch/BatchSection.tsx`) is unusable.
 *
 * Not a mode — it is a `<details>` section on the Create screen — so it cannot
 * ride {@link disabledModesFor}, but `chain` and `a2v` both list it in
 * {@link FEATURE_UI} and the table's disjunction is what used to be a
 * hand-written `||` here.
 */
export function batchA2vDisabledFor(unsupportedFeatures: readonly string[]): boolean {
  return disabledUiTargets(unsupportedFeatures).has("createSection.batchA2v");
}

/**
 * Whether the Batch i2v-long section (`modes/batch-i2v-long/`, on the Chain
 * screen) is unusable — the {@link batchA2vDisabledFor} twin. Its rows each
 * put an image on clip 0, so `chain_start_image` closes it.
 */
export function batchI2vLongDisabledFor(unsupportedFeatures: readonly string[]): boolean {
  return disabledUiTargets(unsupportedFeatures).has("chainSection.batchI2vLong");
}

/** The Chain-screen material panels an engine's feature scope can take
 * down individually, keyed the way `ChainedScreen`'s props are. */
export interface ChainPanelsDisabled {
  /** `SourceInputPanel` — the V2V source video (`source_video`). */
  v2v: boolean;
  /** `ChainAudioPanel` — the A2V track (`source_audio`). */
  a2v: boolean;
  /** `ChainEndSourcePanel` — 素材（末尾） (`end_source`). */
  endSource: boolean;
  /** `ChainReferencePanel` — the IC-LoRA reference (`reference_video_id`). */
  reference: boolean;
  /** §1-106: `SourceInputPanel`'s image half — clip 0's opening image
   * (`clips[0].conditioning_images`). */
  startImage: boolean;
}

/**
 * Which of {@link ChainPanelsDisabled}'s panels `unsupportedFeatures` makes
 * unusable — all `false` for the ordinary case.
 *
 * Deliberately blind to `chain`: a base model that cannot chain at all loses the
 * whole tab through {@link disabledModesFor}, so the table gives `chain` no
 * `chainPanel.*` target and this function cannot duplicate a decision already
 * made one level up.
 */
export function chainPanelsDisabledFor(unsupportedFeatures: readonly string[]): ChainPanelsDisabled {
  const disabled = disabledUiTargets(unsupportedFeatures);
  return {
    v2v: disabled.has("chainPanel.v2v"),
    a2v: disabled.has("chainPanel.a2v"),
    endSource: disabled.has("chainPanel.endSource"),
    reference: disabled.has("chainPanel.reference"),
    startImage: disabled.has("chainPanel.startImage"),
  };
}

/** The Edit-screen sub-tabs an engine's feature scope can take down
 * individually, keyed the way `EditScreen`'s `subTabsDisabled` prop is. */
export interface EditSubTabsDisabled {
  /** `RetakePanel` — 撮り直し (`retake`). */
  retake: boolean;
  /** `OutpaintingPanel` — 画角拡張 (`outpaint`). */
  outpainting: boolean;
  /** `InpaintingPanel` — マスクによる部分再生成 (`inpaint`). */
  inpainting: boolean;
  /** `AlphaGenPanel` — 透過用マットの作成 (`alpha_gen`). */
  alphaGen: boolean;
}

/**
 * Which of {@link EditSubTabsDisabled}'s sub-tabs `unsupportedFeatures` makes
 * unusable — all `false` for the ordinary case.
 *
 * All of them `true` at once is exactly the condition
 * {@link CONTAINER_TARGETS} closes `mode.edit` on: the screen stays mounted
 * (every mode screen does) but the shell greys its tab and bounces out of it,
 * so the user never sees this answer. Answering honestly anyway keeps this a
 * plain reading of the table rather than a special case.
 */
export function editSubTabsDisabledFor(unsupportedFeatures: readonly string[]): EditSubTabsDisabled {
  const disabled = disabledUiTargets(unsupportedFeatures);
  return {
    retake: disabled.has("editSubTab.retake"),
    outpainting: disabled.has("editSubTab.outpainting"),
    inpainting: disabled.has("editSubTab.inpainting"),
    alphaGen: disabled.has("editSubTab.alphaGen"),
  };
}

/** The Settings-panel rows an engine's feature scope HIDES outright (not greys:
 * an engine that publishes one of these names refuses the field with a 422 or
 * ignores it, rather than degrading, so there is nothing to offer). Keyed the
 * way `SettingsPanel`'s props are. */
export interface SettingsRowsHidden {
  /** The VAE decoder row — PrunaVAED (`prune_vaed`). */
  vae: boolean;
  /** LTX 2.5's Embeddings processor 常駐 row (`keep_resident_embeddings`). */
  keepResidentEmbeddings: boolean;
  /** MiniMax H3's VRAM 16GB mode row (`vram16_mode`). */
  vram16: boolean;
  /** The Fused GGUF Dequantization Kernel row and its ON-only note
   * (`fused_gguf_dequant_kernel`). */
  fusedGgufDequantKernel: boolean;
}

/**
 * Which of {@link SettingsRowsHidden}'s rows `unsupportedFeatures` hides — four
 * `false`s for the ordinary case.
 *
 * Read off the published `unsupported_features`, never off a base-model id:
 * which engine lacks which part is the server's fact to state. Hiding is only
 * half the job — see {@link accelerationResetsFor} for the write-back the
 * persisted choice makes necessary.
 */
export function settingsRowsHiddenFor(unsupportedFeatures: readonly string[]): SettingsRowsHidden {
  const disabled = disabledUiTargets(unsupportedFeatures);
  return {
    vae: disabled.has("settingsRow.vae"),
    keepResidentEmbeddings: disabled.has("settingsRow.keepResidentEmbeddings"),
    vram16: disabled.has("settingsRow.vram16"),
    fusedGgufDequantKernel: disabled.has("settingsRow.fusedGgufDequantKernel"),
  };
}

/** The parts of Create (Single) an engine's feature scope can close. Keyed
 * the way `SingleScreen`'s `sectionsUnavailable` prop is.
 * How each one closes is the screen's business (see {@link FEATURE_UI}'s
 * notes): the reference video and the keyframe strength are hidden, the rest
 * grey. */
export interface SingleSectionsDisabled {
  /** The reference-video (IC-LoRA) accordion (`reference_video`). */
  referenceVideo: boolean;
  /** The A2V (source audio) accordion: closed only when neither route is
   * open (see {@link a2vRouteFor}). */
  a2v: boolean;
  /** Each keyframe card's strength (`keyframe_strength`). */
  keyframeStrength: boolean;
}

/** Where Create's A2V card submits.
 * - `"generate"`: the engine takes the source audio on `/generate` itself
 *   (`singleA2vAvailable` — the server publishes `constraints.source_audio`;
 *   MiniMax H3).
 * - `"chain"`: otherwise, through `/generate/chain` as before (LTX), unless
 *   the feature table closes `createSection.a2v` (no `chain`).
 * - `null`: neither — the card is closed.
 *
 * Read off what the server publishes, never off an engine id. */
export function a2vRouteFor(
  unsupportedFeatures: readonly string[],
  singleA2vAvailable: boolean,
): "generate" | "chain" | null {
  if (singleA2vAvailable) return "generate";
  if (!disabledUiTargets(unsupportedFeatures).has("createSection.a2v")) return "chain";
  return null;
}

/** Which of {@link SingleSectionsDisabled}'s parts `unsupportedFeatures`
 * closes — all `false` for the ordinary case (both LTX engines). The A2V
 * card closes only when {@link a2vRouteFor} finds no route. */
export function singleSectionsDisabledFor(
  unsupportedFeatures: readonly string[],
  singleA2vAvailable = false,
): SingleSectionsDisabled {
  const disabled = disabledUiTargets(unsupportedFeatures);
  return {
    referenceVideo: disabled.has("createSection.referenceVideo"),
    a2v: a2vRouteFor(unsupportedFeatures, singleA2vAvailable) === null,
    keyframeStrength: disabled.has("createSection.keyframeStrength"),
  };
}

/** Whether the shared Negative Prompt (NAG/VSF) accordion is closed
 * (`nag`). The shell greys it AND stops its fields from riding on any request
 * (`nagSettings.effectiveNag`), so a NAG left on from another base model can
 * neither 422 a job nor be stuck on screen. `false` for the ordinary case. */
export function nagDisabledFor(unsupportedFeatures: readonly string[]): boolean {
  return disabledUiTargets(unsupportedFeatures).has("shell.nag");
}

/** Whether each LoRA chip's audio toggle (🎥 / 🔊 / audio value) is hidden
 * (`lora_audio_strength`). The shell hides it AND stops sending
 * `audio_strength` on Single's request (`loraTags.omitLoraAudioStrength`).
 * `false` for the ordinary case (both LTX engines). */
export function loraAudioStrengthHiddenFor(unsupportedFeatures: readonly string[]): boolean {
  return disabledUiTargets(unsupportedFeatures).has("promptBar.loraAudioStrength");
}
