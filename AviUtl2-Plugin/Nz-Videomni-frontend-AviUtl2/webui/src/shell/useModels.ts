import { useCallback, useEffect, useRef, useState } from "react";
import { apiClient as defaultApiClient, BackendApiError } from "../api/client";
import type { ApiClient } from "../api/client";
import type { ModelCategory, ModelsResponse } from "../api/types";
import { MODEL_DEFAULT_NAME } from "../api/types";

/** Fixed category order (Docs/API_REFERENCE.md §3.5,
 * `services/model_registry.py:69-97` `CATEGORIES`) — drives the ModelsPanel's
 * dropdown order and the shape of the per-category selection state below. */
export const MODEL_CATEGORIES: readonly ModelCategory[] = ["transformer", "text_encoder", "video_vae", "audio"];

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

  return { list, refresh, selection, setSelection, load, loadPipeline };
}
