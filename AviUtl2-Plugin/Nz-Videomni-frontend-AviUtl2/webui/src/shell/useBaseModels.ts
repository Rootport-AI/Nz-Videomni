import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { apiClient as defaultApiClient, BackendApiError } from "../api/client";
import type { ApiClient } from "../api/client";
import type { BaseModelBlock } from "../api/types";
import type { AppMode } from "./AppShell";

/**
 * Backs the header's BASE MODEL dropdown (multi-engine groundwork §3-97 P7;
 * `Docs/MULTI_ENGINE_DESIGN.md` §6). A base model is the unit the user picks —
 * "LTX 2.3", "LTX 2.5", a future "Wan 2.x" — as opposed to the four
 * per-category weight files behind it, which stay in Settings' Models panel
 * (`shell/useModels.ts`).
 *
 * Two rules from the design shape everything here:
 *
 *  - **§6.1 switching IS loading.** There is no "apply" button: picking an
 *    entry immediately runs `POST /pipeline/load` with that `base_model`, the
 *    same work Settings' "Load selected models" does. The point is to make the
 *    state "the dropdown says LTX 2.5 but LTX 2.3 is what's loaded"
 *    unrepresentable.
 *  - **§6.2 guard 4, the corollary: on ANY failure the displayed selection
 *    must return to what it was.** Leaving the failed choice on screen would
 *    recreate exactly the mismatch the rule above exists to prevent. That is
 *    why the displayed value is `pending ?? active` and `pending` is cleared
 *    in a `finally` — a rejected switch cannot leave it stuck.
 *
 * The WebUI never chooses an inference engine. The server derives that from
 * the weight file's own GGUF metadata (§2.1); this hook only ever sends an id.
 *
 * `apiClient` defaults to the app-wide singleton but accepts an override so
 * tests can bind a purpose-configured mock bridge.
 */

/** One selectable row of the dropdown, flattened from {@link BaseModelBlock}. */
export interface BaseModelOption {
  id: string;
  displayName: string;
  /** Every category's default file is on disk. */
  installed: boolean;
  /** At least one is — see {@link BaseModelBlock} for why the two differ. */
  present: boolean;
  missingCategories: string[];
  /** Feature names this base model's engine cannot run — see
   * {@link BaseModelBlock.unsupported_features}. `[]` on a backend that does
   * not publish the key, which is the same thing as "no restrictions". */
  unsupportedFeatures: string[];
}

/** Which mode tab each unsupported FEATURE takes down with it.
 *
 * The server names FEATURES (`chain`, `outpaint`, …) because that is what a
 * request field is; the shell only has TABS. This table is the translation,
 * and it deliberately lives next to the hook that receives the names rather
 * than inside `ModeTabs` — the tabs render what they are told, they do not
 * reason about engines.
 *
 * A mode is disabled when EVERY thing it can do is unsupported, not when any
 * one is:
 *
 *  - **chained** is the chain endpoint and nothing else, so `chain` alone
 *    settles it;
 *  - **edit** hosts Retake and Outpainting (Inpainting is still a disabled
 *    mock), so it survives as long as ONE of those two is runnable — a base
 *    model that could outpaint but not retake would still have a use for the
 *    tab;
 *  - **single** and **inventory** are never listed. Single IS the baseline any
 *    engine must serve, and Inventory only browses finished files — it issues
 *    no generation at all, so no engine limitation can reach it.
 */
const MODE_REQUIREMENTS: ReadonlyArray<{ mode: AppMode; needsAnyOf: readonly string[] }> = [
  { mode: "chained", needsAnyOf: ["chain"] },
  { mode: "edit", needsAnyOf: ["retake", "outpaint"] },
];

/**
 * The mode tabs that `unsupportedFeatures` makes unreachable. Pure, exported
 * and tested directly: it is the one place a feature name turns into a greyed
 * tab, and it must answer `[]` for the ordinary case (LTX 2.3 / an older
 * backend) without any special-casing.
 *
 * Unknown names are ignored rather than treated as suspicious — a build of
 * this WebUI is older than the server it talks to more often than the reverse.
 */
export function disabledModesFor(unsupportedFeatures: readonly string[]): AppMode[] {
  const unsupported = new Set(unsupportedFeatures);
  return MODE_REQUIREMENTS.filter(
    ({ needsAnyOf }) => needsAnyOf.every((feature) => unsupported.has(feature)),
  ).map(({ mode }) => mode);
}

/**
 * Whether the Batch A2V panel (`modes/batch/BatchSection.tsx`) is unusable.
 *
 * Not a mode — it is a `<details>` section on the Create screen — so it cannot
 * ride {@link disabledModesFor}, but the reasoning is the same shape: every row
 * it queues is a `POST /generate/chain` carrying an audio source, so either
 * limitation takes the whole panel down.
 */
export function batchA2vDisabledFor(unsupportedFeatures: readonly string[]): boolean {
  return unsupportedFeatures.includes("chain") || unsupportedFeatures.includes("a2v");
}

/** The four Chain-screen material panels an engine's feature scope can take
 * down individually, keyed the way `ChainedScreen`'s props are.
 *
 * §3-102 (LTX 2.5 Chained, first stage): once an engine can chain, `chain`
 * alone no longer settles the Chained tab — the tab is live, but the panels
 * that attach material the engine still cannot use have to grey on their own.
 * Each is exactly one feature name, because each panel IS one request field:
 * `source_video`, `source_audio`, `end_source`, `reference_video_id`.
 */
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
 * unusable. Pure, exported and tested directly for the same reason
 * {@link disabledModesFor} and {@link batchA2vDisabledFor} are: this is the one
 * place a server-side feature name turns into a greyed Chain panel, and it must
 * answer four `false`s for the ordinary case (LTX 2.3 / an older backend)
 * without any special-casing.
 *
 * Deliberately does NOT consult `chain`: a base model that cannot chain at all
 * loses the whole tab through {@link disabledModesFor}, so folding that in here
 * would only duplicate a decision already made one level up.
 */
export function chainPanelsDisabledFor(unsupportedFeatures: readonly string[]): ChainPanelsDisabled {
  const unsupported = new Set(unsupportedFeatures);
  return {
    v2v: unsupported.has("v2v"),
    a2v: unsupported.has("a2v"),
    endSource: unsupported.has("end_source"),
    reference: unsupported.has("reference_video"),
  };
}

/** The two Edit-screen sub-tabs an engine's feature scope can take down
 * individually, keyed the way `EditScreen`'s `subTabsDisabled` prop is.
 * (Inpainting is not here: it is a mock sub-tab with no panel behind it, so it
 * is disabled on EVERY base model and needs no feature name.)
 *
 * The Retake+End source 開通 makes this necessary. Until then LTX 2.5 declared
 * BOTH `retake` and `outpaint`, so `disabledModesFor` greyed the whole Edit tab
 * and no sub-tab was ever reachable to grey — the tab-level rule made this one
 * redundant by accident, not by design. The moment ONE of the two opens, the
 * Edit tab goes live and the sub-tab that is still out of scope has to grey on
 * its own, exactly as {@link chainPanelsDisabledFor}'s panels do.
 */
export interface EditSubTabsDisabled {
  /** `RetakePanel` — 撮り直し (`retake`). */
  retake: boolean;
  /** `OutpaintingPanel` — 画角拡張 (`outpaint`). */
  outpainting: boolean;
}

/**
 * Which of {@link EditSubTabsDisabled}'s sub-tabs `unsupportedFeatures` makes
 * unusable. Pure, exported and tested directly for the same reason
 * {@link disabledModesFor}, {@link batchA2vDisabledFor} and
 * {@link chainPanelsDisabledFor} are: this is the one place a server-side
 * feature name turns into a greyed Edit sub-tab, and it must answer two
 * `false`s for the ordinary case (LTX 2.3 / an older backend) without any
 * special-casing.
 *
 * Note the names differ on purpose: the SERVER's feature is `outpaint` (that is
 * what a request field and a 422 say), while the sub-tab is `outpainting`. The
 * translation is exactly what this function is for — nothing downstream should
 * have to know either name.
 *
 * Both `true` at once is possible here, but the caller never sees it: a base
 * model that can run NEITHER loses the whole Edit tab through
 * {@link disabledModesFor} (`needsAnyOf: ["retake", "outpaint"]`), so the screen
 * this feeds is not mounted at all. Answering honestly anyway keeps the
 * function a plain table rather than a special case.
 */
export function editSubTabsDisabledFor(unsupportedFeatures: readonly string[]): EditSubTabsDisabled {
  const unsupported = new Set(unsupportedFeatures);
  return {
    retake: unsupported.has("retake"),
    outpainting: unsupported.has("outpaint"),
  };
}

/** What a switch attempt settled on. Returned by
 * {@link UseBaseModelsResult.switchBaseModel} rather than pushed into hook
 * state on purpose: the caller (`AppShell`) raises exactly one toast per
 * user action that way, with no effect-driven state-change watcher that could
 * fire twice for one switch or once for a stale one. */
export type BaseModelSwitchOutcome =
  /** The server rebuilt the worker on the requested base model. */
  | { kind: "switched"; id: string }
  /** Guard 2: nothing on disk, so no request was made at all. */
  | { kind: "not-installed"; id: string; displayName: string }
  /** Guard 1: 409 `JOB_BUSY` — a generation job holds the single-job queue. */
  | { kind: "busy" }
  /** Guard 3: 409 `PIPELINE_LOADING` — a load is already in flight. */
  | { kind: "loading" }
  /** 422: the server refused this base model for a SPECIFIC, already-worded
   * reason (an incompatible weight file, a missing default file, …). `reason`
   * is the envelope's `detail` verbatim — the backend writes the sentence the
   * user should read (e.g. the "LTX 2.5 エンジンは次段階…" notice), and
   * paraphrasing it here would only make it staler than the server. */
  | { kind: "rejected"; reason: string }
  /** Anything else: 404 for an id the server does not know, a transport
   * failure, an unexpected status. */
  | { kind: "failed"; message: string };

export interface UseBaseModelsDeps {
  apiClient?: ApiClient;
}

export interface UseBaseModelsResult {
  /** Server-declared order, straight from `base_models[]`; never re-sorted. */
  options: BaseModelOption[];
  /** The id the dropdown should display: the in-flight target while a switch
   * runs, otherwise the server's `active_base_model`. `""` until the first
   * `GET /models` lands (or on an older backend that declares none), which is
   * the caller's cue to show its placeholder. */
  current: string;
  /** A switch is in flight — disable the dropdown (a second pick would only
   * earn a 409 `PIPELINE_LOADING` from the server). */
  switching: boolean;
  /** Feature names the LOADED base model's engine cannot run (§3-98 P5).
   *
   * Read off `active`, never off `current`: while a switch is in flight the
   * pipeline is still the OLD base model, so greying the new one's limitations
   * early would disable controls that still work — and, if the switch then
   * fails, leave them disabled for a base model that never loaded. */
  unsupportedFeatures: string[];
  /** The mode tabs {@link unsupportedFeatures} makes unreachable — the shell
   * greys these and bounces out of one if it is the current mode. */
  disabledModes: AppMode[];
  /** Fire-and-await: see {@link BaseModelSwitchOutcome}. Never throws. */
  switchBaseModel: (id: string) => Promise<BaseModelSwitchOutcome>;
  /** Re-reads `GET /models`. Runs on mount and after a successful switch. */
  refresh: () => Promise<void>;
}

function toOption(block: BaseModelBlock): BaseModelOption {
  return {
    id: block.id,
    displayName: block.display_name,
    installed: block.installed,
    present: block.present,
    missingCategories: block.missing_categories,
    // `?? []` is the whole backward-compatibility story: a backend older than
    // §3-98 P5 omits the key, and "omitted" means "no restrictions".
    unsupportedFeatures: block.unsupported_features ?? [],
  };
}

/** Maps a failed `POST /pipeline/load` onto the outcome union. Keyed on the
 * error CODE for the two 409s (they are distinct guards with distinct copy)
 * and on the STATUS for 422 (several codes — `MODEL_INCOMPATIBLE`,
 * `MODEL_FILE_MISSING` — share the "server already worded it" treatment). */
function classifySwitchFailure(err: unknown): BaseModelSwitchOutcome {
  if (err instanceof BackendApiError) {
    if (err.code === "JOB_BUSY") return { kind: "busy" };
    if (err.code === "PIPELINE_LOADING") return { kind: "loading" };
    if (err.httpStatus === 422) return { kind: "rejected", reason: err.detail ?? err.message };
    return { kind: "failed", message: err.message };
  }
  return { kind: "failed", message: err instanceof Error ? err.message : String(err) };
}

export function useBaseModels(deps: UseBaseModelsDeps = {}): UseBaseModelsResult {
  const client = deps.apiClient ?? defaultApiClient;
  const [options, setOptions] = useState<BaseModelOption[]>([]);
  const [active, setActive] = useState("");
  const [pending, setPending] = useState<string | null>(null);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const refresh = useCallback(async () => {
    let models;
    try {
      models = await client.getModels();
    } catch {
      // A failed listing must not blank a dropdown that is already populated:
      // the header would lose its label over one dropped poll. Keep whatever
      // we last knew — the server-status badge is what reports connectivity.
      return;
    }
    if (!mountedRef.current) return;
    const blocks = models.base_models ?? [];
    setOptions(blocks.map(toOption));
    setActive(models.active_base_model ?? blocks.find((b) => b.active)?.id ?? "");
  }, [client]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const switchBaseModel = useCallback(
    async (id: string): Promise<BaseModelSwitchOutcome> => {
      const option = options.find((o) => o.id === id);
      // Guard 2 (§6.2): with not one weight file on disk there is nothing for
      // the server to load, so spend no request on it — and, just as
      // important, never move the displayed selection. Only `present: false`
      // short-circuits: a PARTIAL install does go to the server, which fails
      // loud naming the category that is missing.
      if (option && !option.present) {
        return { kind: "not-installed", id, displayName: option.displayName };
      }

      setPending(id);
      try {
        // Empty selection + a base model: the server resolves that base's own
        // categories from its descriptor defaults / remembered selection
        // (§6.3). The WebUI has no business naming files for a base model it
        // has not listed yet.
        await client.loadPipeline({}, id);
        if (mountedRef.current) setActive(id);
        await refresh();
        return { kind: "switched", id };
      } catch (err) {
        return classifySwitchFailure(err);
      } finally {
        // Guard 4: unconditional, so every failure path above reverts the
        // display to `active` — the base model still actually loaded.
        if (mountedRef.current) setPending(null);
      }
    },
    [client, options, refresh],
  );

  const unsupportedFeatures = useMemo(
    () => options.find((o) => o.id === active)?.unsupportedFeatures ?? [],
    [options, active],
  );
  const disabledModes = useMemo(
    () => disabledModesFor(unsupportedFeatures),
    [unsupportedFeatures],
  );

  return {
    options,
    current: pending ?? active,
    switching: pending !== null,
    unsupportedFeatures,
    disabledModes,
    switchBaseModel,
    refresh,
  };
}
