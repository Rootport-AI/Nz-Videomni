import { useCallback, useEffect, useRef, useState } from "react";
import { apiClient as defaultApiClient, BackendApiError } from "../api/client";
import type { ApiClient } from "../api/client";
import type { ModelCategory, ModelsResponse } from "../api/types";
import { MODEL_DEFAULT_NAME } from "../api/types";

/** The categories this WebUI knows how to render, and the FALLBACK order
 * (Docs/API_REFERENCE.md §3.5, `services/model_registry.py`'s `CATEGORIES`).
 *
 * Since the multi-engine groundwork (§3-97 P3a) the category set is no longer
 * a code constant server-side: each base model declares its own in
 * `scripts/manifests/<base>.json`'s `categories`, and `GET /models` returns
 * the blocks in that declaration order. {@link UseModelsResult.categoryOrder}
 * is therefore what the panel renders by; this constant survives as the typed
 * shape of {@link ModelSelection} (a `Record<ModelCategory, string>` needs the
 * union spelled out somewhere) and as the order to fall back to when the
 * server sends nothing usable. */
export const MODEL_CATEGORIES: readonly ModelCategory[] = ["transformer", "text_encoder", "video_vae", "audio"];

/** Reads the display order out of a `GET /models` response: the ACTIVE base
 * model's own `categories` keys, which the server emits in the descriptor's
 * declaration order (unlike the top-level `categories` block, which is keyed
 * by the fixed server-side literal). Falls back to the top-level block's keys,
 * then to {@link MODEL_CATEGORIES}. Anything the WebUI has no renderer for is
 * dropped, and known categories the server omitted are NOT re-added — a base
 * model that genuinely has three categories must show three dropdowns. */
function resolveCategoryOrder(models: ModelsResponse | null): readonly ModelCategory[] {
  const known = new Set<string>(MODEL_CATEGORIES);
  const activeBlock = models?.base_models?.find((b) => b.id === models.active_base_model) ?? undefined;
  const keys = activeBlock ? Object.keys(activeBlock.categories) : Object.keys(models?.categories ?? {});
  const order = keys.filter((k): k is ModelCategory => known.has(k));
  return order.length > 0 ? order : MODEL_CATEGORIES;
}

export type ModelSelection = Record<ModelCategory, string>;

function allDefaultSelection(): ModelSelection {
  return {
    transformer: MODEL_DEFAULT_NAME,
    text_encoder: MODEL_DEFAULT_NAME,
    video_vae: MODEL_DEFAULT_NAME,
    audio: MODEL_DEFAULT_NAME,
  };
}

export type ModelsListState =
  | { status: "loading" }
  | { status: "ready"; models: ModelsResponse }
  | { status: "error"; message: string };

export type ModelsLoadState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "done"; models: ModelSelection }
  /** 409 JOB_BUSY — a generation job is running, so the worker cannot be
   * rebuilt right now (mirrors `useGeneration`'s `phase: "busy"`). Kept
   * distinct from the generic error branch so `ModelsPanel` can show the
   * dedicated "cannot switch while a job is running" copy. */
  | { status: "busy" }
  | { status: "error"; code: string; message: string };

export interface UseModelsDeps {
  apiClient?: ApiClient;
}

export interface UseModelsResult {
  list: ModelsListState;
  /** Order to render the per-category dropdowns in, driven by the server's
   * response (see {@link MODEL_CATEGORIES}). Equals `MODEL_CATEGORIES` until
   * the first fetch lands, and on any backend that declares nothing usable. */
  categoryOrder: readonly ModelCategory[];
  /** Re-fetches `GET /models`. The server rescans its model directories on
   * every call, so calling this is the WebUI's only "detect a newly
   * downloaded file" action — no separate rescan endpoint exists. On success
   * the per-category selection is reset to the response's `active` values,
   * so the dropdowns always reflect what is actually loaded. */
  refresh: () => Promise<void>;
  selection: ModelSelection;
  setSelection: (category: ModelCategory, name: string) => void;
  load: ModelsLoadState;
  /** `POST /pipeline/load` with the current `selection` for all four
   * categories. On success, refetches `GET /models` so `active` (and thus
   * the dropdowns) reflect the swap. Fire-and-forget: watch `load` for the
   * outcome. */
  loadPipeline: () => void;
}

/** Backs the Settings panel's "Models" section (model management S1):
 * `GET /models` on mount/refresh, per-category selection state, and
 * `POST /pipeline/load` to apply it (Docs/API_REFERENCE.md §3.3/§3.5).
 * Mirrors `useLoras`'s list-fetch shape plus `useGeneration`'s busy/error
 * action shape. `apiClient` defaults to the app-wide singleton but accepts
 * an override so tests (and `SettingsPanel`'s `nativeBridge` test hook) can
 * bind a purpose-configured mock bridge instead. */
export function useModels(deps: UseModelsDeps = {}): UseModelsResult {
  const client = deps.apiClient ?? defaultApiClient;
  const [list, setList] = useState<ModelsListState>({ status: "loading" });
  const [selection, setSelectionState] = useState<ModelSelection>(allDefaultSelection);
  const [load, setLoad] = useState<ModelsLoadState>({ status: "idle" });
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const refresh = useCallback(async () => {
    try {
      const models = await client.getModels();
      if (!mountedRef.current) return;
      setList({ status: "ready", models });
      setSelectionState({
        transformer: models.categories.transformer?.active ?? MODEL_DEFAULT_NAME,
        text_encoder: models.categories.text_encoder?.active ?? MODEL_DEFAULT_NAME,
        video_vae: models.categories.video_vae?.active ?? MODEL_DEFAULT_NAME,
        audio: models.categories.audio?.active ?? MODEL_DEFAULT_NAME,
      });
    } catch (err) {
      if (!mountedRef.current) return;
      setList({ status: "error", message: err instanceof Error ? err.message : String(err) });
    }
  }, [client]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const setSelection = useCallback((category: ModelCategory, name: string) => {
    setSelectionState((prev) => ({ ...prev, [category]: name }));
  }, []);

  const loadPipeline = useCallback(() => {
    setLoad({ status: "loading" });
    void (async () => {
      try {
        const resp = await client.loadPipeline(selection);
        if (!mountedRef.current) return;
        setLoad({ status: "done", models: (resp.models as ModelSelection | undefined) ?? selection });
        await refresh();
      } catch (err) {
        if (!mountedRef.current) return;
        if (err instanceof BackendApiError && err.code === "JOB_BUSY") {
          setLoad({ status: "busy" });
          return;
        }
        const { code, message } =
          err instanceof BackendApiError
            ? { code: err.code, message: err.message }
            : { code: "UNKNOWN", message: err instanceof Error ? err.message : String(err) };
        setLoad({ status: "error", code, message });
      }
    })();
  }, [client, selection, refresh]);

  return {
    list,
    categoryOrder: resolveCategoryOrder(list.status === "ready" ? list.models : null),
    refresh,
    selection,
    setSelection,
    load,
    loadPipeline,
  };
}
