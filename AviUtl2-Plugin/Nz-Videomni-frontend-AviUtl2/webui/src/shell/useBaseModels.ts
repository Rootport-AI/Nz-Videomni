import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { apiClient as defaultApiClient, BackendApiError } from "../api/client";
import type { ApiClient } from "../api/client";
import type { BaseModelBlock } from "../api/types";
import type { AppMode } from "./AppShell";
import { disabledModesFor } from "./featureScope";
import type { TrackPipelineLoad } from "../jobs/JobsContext";

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
  /** Inference-engine lineage (`BaseModelBlock.engine_family` — `"ltx"`,
   * `"ltx25"`, …), the key into the served `AppLimits.comfort_budgets` table
   * (2026-08-31). `""` on a backend older than §3-97 that does not publish the
   * key, which `shell/comfortTable.ts` reads as "engine unknown" and answers
   * with its compatibility shim. */
  engineFamily: string;
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

/** The batch file that installs a base model's weights. The server never
 * downloads anything itself (`Docs/MULTI_ENGINE_DESIGN.md` §6.2) — putting
 * files on disk is `install-<id>.bat`'s job, one bat per descriptor id
 * (台帳 §3-111). Derived from the id alone, so a base model added
 * server-side needs no change here. LTX 2.3 has no bat of its own — it ships
 * inside `setup.bat` — so this name is reachable for it only if someone
 * deletes the weights `setup.bat` already placed; that residual mismatch is
 * an accepted corner (オーナー裁定), not a bug to design around. */
export function baseModelInstaller(id: string): string {
  return `install-${id}.bat`;
}

export interface UseBaseModelsDeps {
  apiClient?: ApiClient;
  /** `JobsContext.trackPipelineLoad`. Wrapping the switch in it is what makes
   * the header badge light up the instant the user picks an entry, instead of
   * on whichever `GET /status` poll happens to land next. Optional so tests
   * (and any caller with no provider above it) can omit it. */
  trackLoad?: TrackPipelineLoad;
}

export interface UseBaseModelsResult {
  /** Server-declared order, straight from `base_models[]`; never re-sorted. */
  options: BaseModelOption[];
  /** The id the dropdown should display: the in-flight target while a switch
   * runs, otherwise the server's `active_base_model`. `""` until the first
   * `GET /models` lands (or on an older backend that declares none), which is
   * the caller's cue to show its placeholder. */
  current: string;
  /** Feature names the LOADED base model's engine cannot run (§3-98 P5).
   *
   * Read off `active`, never off `current`: while a switch is in flight the
   * pipeline is still the OLD base model, so greying the new one's limitations
   * early would disable controls that still work — and, if the switch then
   * fails, leave them disabled for a base model that never loaded. */
  unsupportedFeatures: string[];
  /** The LOADED base model's engine family (2026-08-31) — the key into the
   * served `AppLimits.comfort_budgets` table, threaded down to Create/Chained
   * so the comfort marker follows the engine.
   *
   * Read off `active`, never off `current`, for the SAME reason
   * {@link unsupportedFeatures} is: while a switch is in flight the pipeline is
   * still running the OLD base model, so moving the marker to the new engine's
   * line early would advertise a ceiling that does not apply yet — and, if the
   * switch then fails, leave it advertising one for a base model that never
   * loaded.
   *
   * `""` until the first `GET /models` lands (and on a pre-§3-97 backend),
   * which `shell/comfortTable.ts` reads as "engine unknown" → the
   * compatibility shim, i.e. no startup flicker. */
  activeEngineFamily: string;
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
    // `engine_family` is NON-optional on the wire (`api/types.ts`); the `?? ""`
    // is defence against a backend older than §3-97 that predates the field
    // entirely, and `""` is exactly the "engine unknown" value
    // `shell/comfortTable.ts` handles with its compatibility shim.
    engineFamily: block.engine_family ?? "",
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
  const track = deps.trackLoad;
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
      // Tracked from here, so the flag's window is exactly `pending`'s window.
      // The wrapped body is total — every path returns a
      // `BaseModelSwitchOutcome` — so `trackPipelineLoad` never sees a
      // rejection to re-raise.
      const attempt = (async (): Promise<BaseModelSwitchOutcome> => {
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
      })();
      return track ? track(attempt) : attempt;
    },
    [client, options, refresh, track],
  );

  const unsupportedFeatures = useMemo(
    () => options.find((o) => o.id === active)?.unsupportedFeatures ?? [],
    [options, active],
  );
  const activeEngineFamily = useMemo(
    () => options.find((o) => o.id === active)?.engineFamily ?? "",
    [options, active],
  );
  const disabledModes = useMemo(
    () => disabledModesFor(unsupportedFeatures),
    [unsupportedFeatures],
  );

  return {
    options,
    current: pending ?? active,
    unsupportedFeatures,
    activeEngineFamily,
    disabledModes,
    switchBaseModel,
    refresh,
  };
}
