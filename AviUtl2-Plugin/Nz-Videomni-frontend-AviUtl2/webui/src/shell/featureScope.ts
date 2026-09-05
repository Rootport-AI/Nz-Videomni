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
  | "editSubTab.retake"
  | "editSubTab.outpainting"
  | "createSection.batchA2v"
  | "settingsRow.vae"
  | "settingsRow.keepResidentEmbeddings";

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
   * request after a switch and 422 every job. */
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
 *  - **retake / outpaint** are the two Edit sub-tabs. Note the SERVER's feature
 *    is `outpaint` (that is what a request field and a 422 say) while the
 *    sub-tab is `outpainting`; this table is exactly where that translation
 *    lives, and nothing downstream has to know either name.
 *  - **prune_vaed / keep_resident_embeddings** are Settings rows rather than
 *    generation surfaces, and both carry a `resets` because their choice is
 *    persisted in `localStorage`. Their SCOPES point in opposite directions:
 *    an engine that publishes `prune_vaed` refuses the pruned decoder, while
 *    `keep_resident_embeddings` names a component only LTX 2.5 HAS, so it is
 *    LTX 2.3 that publishes it. The table does not care — either way the row
 *    closes and the stored value goes back to the server default.
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
  chain: { disables: ["mode.chained", "createSection.batchA2v"] },
  retake: { disables: ["editSubTab.retake"] },
  outpaint: { disables: ["editSubTab.outpainting"] },
  v2v: { disables: ["chainPanel.v2v"] },
  a2v: { disables: ["chainPanel.a2v", "createSection.batchA2v"] },
  end_source: { disables: ["chainPanel.endSource"] },
  reference_video: { disables: ["chainPanel.reference"] },
  prune_vaed: { disables: ["settingsRow.vae"], resets: "vaeMode" },
  keep_resident_embeddings: {
    disables: ["settingsRow.keepResidentEmbeddings"],
    resets: "keepResidentEmbeddings",
  },
} as const satisfies Record<string, FeatureUiEntry>;

// No union type over the feature NAMES (no `UiRelevantFeature` or the like):
// every boundary below keeps `readonly string[]`. A build of this WebUI is
// older than the server it talks to more often than the reverse, so an unknown
// name is the ordinary case, not an error — and a feature with no UI target at
// all (`two_stage_hq`, `nag`, `loras`, `sage_attention`, …) is simply absent
// from the table rather than listed with an empty `disables`.

/** A `Map`, not the plain object: looking `FEATURE_UI` up with an arbitrary
 * server string would also answer for `"constructor"`, `"__proto__"` and the
 * rest of `Object.prototype`, and an unknown name must map to nothing. */
const FEATURE_UI_BY_NAME: ReadonlyMap<string, FeatureUiEntry> = new Map(Object.entries(FEATURE_UI));

/** Targets no feature names directly — they close only once everything they
 * CONTAIN is closed.
 *
 * The Edit tab is the only one today: it hosts Retake and Outpainting
 * (Inpainting is still a disabled mock with no panel behind it), so it survives
 * as long as ONE of those two is runnable — a base model that could outpaint but
 * not retake would still have a use for the tab. Chained needs no entry here
 * because `chain` names it directly.
 *
 * Invariant: SINGLE-PASS evaluation — a `whenAll` must never name another
 * container target, or the answer would depend on the order of this array.
 * Should a nested case ever arise, revisit the design then rather than
 * quietly ordering the list. */
const CONTAINER_TARGETS: ReadonlyArray<{ target: UiTarget; whenAll: readonly UiTarget[] }> = [
  { target: "mode.edit", whenAll: ["editSubTab.retake", "editSubTab.outpainting"] },
];

/**
 * Every UI part the loaded base model's `unsupported_features` closes.
 *
 * Unknown names are ignored rather than treated as suspicious — see the note
 * under {@link FEATURE_UI}. That is what makes the ordinary case (LTX 2.3, or
 * any backend older than §3-98 P5 that publishes no list at all) answer with an
 * empty set and no special-casing anywhere.
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

/** The four Chain-screen material panels an engine's feature scope can take
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
}

/**
 * Which of {@link ChainPanelsDisabled}'s panels `unsupportedFeatures` makes
 * unusable — four `false`s for the ordinary case.
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
  };
}

/** The two Edit-screen sub-tabs an engine's feature scope can take down
 * individually, keyed the way `EditScreen`'s `subTabsDisabled` prop is.
 * (Inpainting is not here: it is a mock sub-tab with no panel behind it, so it
 * is disabled on EVERY base model and needs no feature name.) */
export interface EditSubTabsDisabled {
  /** `RetakePanel` — 撮り直し (`retake`). */
  retake: boolean;
  /** `OutpaintingPanel` — 画角拡張 (`outpaint`). */
  outpainting: boolean;
}

/**
 * Which of {@link EditSubTabsDisabled}'s sub-tabs `unsupportedFeatures` makes
 * unusable — two `false`s for the ordinary case.
 *
 * Both `true` at once is possible here, but the caller never sees it: that is
 * exactly the condition {@link CONTAINER_TARGETS} closes `mode.edit` on, so the
 * screen this feeds is not mounted at all. Answering honestly anyway keeps this
 * a plain reading of the table rather than a special case.
 */
export function editSubTabsDisabledFor(unsupportedFeatures: readonly string[]): EditSubTabsDisabled {
  const disabled = disabledUiTargets(unsupportedFeatures);
  return {
    retake: disabled.has("editSubTab.retake"),
    outpainting: disabled.has("editSubTab.outpainting"),
  };
}

/** The Settings-panel rows an engine's feature scope HIDES outright (not greys:
 * an engine that publishes one of these names answers the field with a 422
 * rather than degrading, so there is nothing to offer). Keyed the way
 * `SettingsPanel`'s props are. */
export interface SettingsRowsHidden {
  /** The VAE decoder row — PrunaVAED (`prune_vaed`). */
  vae: boolean;
  /** LTX 2.5's Embeddings processor 常駐 row (`keep_resident_embeddings`). */
  keepResidentEmbeddings: boolean;
}

/**
 * Which of {@link SettingsRowsHidden}'s rows `unsupportedFeatures` hides — two
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
  };
}
