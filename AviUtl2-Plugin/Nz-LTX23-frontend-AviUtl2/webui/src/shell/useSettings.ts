import { useCallback, useEffect, useRef, useState } from "react";
import { bridge as defaultBridge, BridgeError } from "../bridge";
import type { NativeBridge } from "../bridge";

export type SettingsLoadState =
  | { status: "loading" }
  | { status: "ready"; baseUrl: string }
  | { status: "error"; message: string };

export type SettingsSaveState =
  | { status: "idle" }
  | { status: "saving" }
  | { status: "error"; code: string; message: string };

export interface UseSettingsResult {
  load: SettingsLoadState;
  save: SettingsSaveState;
  /** The editable text field's current value — seeded from `load` once it
   * resolves, then owned by the caller's `<input>` until `submit()`. */
  baseUrlInput: string;
  setBaseUrlInput: (value: string) => void;
  /** Calls `settings.set`; on success, `load`/`baseUrlInput` are refreshed
   * from the (possibly normalized) resulting value and `save` returns to
   * `idle`. On failure, `save` becomes `{status:"error"}` — the caller
   * (`SettingsPanel`) decides how to render `code`/`message` (e.g.
   * `BAD_REQUEST` gets the translated inline copy, anything else the raw
   * bridge message) — and the promise rejects so the caller can skip
   * whatever it would have done on success (e.g. re-checking `/status`). */
  submit: () => Promise<void>;
}

/** Backs the M7b connection-settings panel: `settings.get` on mount,
 * `settings.set` on save (bridge contract v4 — see `bridge/types.ts`).
 * `nativeBridge` defaults to the app-wide singleton but accepts an override
 * so tests can bind a purpose-configured mock bridge. */
export function useSettings(nativeBridge: NativeBridge = defaultBridge): UseSettingsResult {
  const [load, setLoad] = useState<SettingsLoadState>({ status: "loading" });
  const [save, setSave] = useState<SettingsSaveState>({ status: "idle" });
  const [baseUrlInput, setBaseUrlInput] = useState("");
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  useEffect(() => {
    setLoad({ status: "loading" });
    nativeBridge
      .request("settings.get", {})
      .then((result) => {
        if (!mountedRef.current) return;
        setLoad({ status: "ready", baseUrl: result.baseUrl });
        setBaseUrlInput(result.baseUrl);
      })
      .catch((err) => {
        if (!mountedRef.current) return;
        setLoad({ status: "error", message: err instanceof Error ? err.message : String(err) });
      });
  }, [nativeBridge]);

  const submit = useCallback(async () => {
    setSave({ status: "saving" });
    try {
      const result = await nativeBridge.request("settings.set", { baseUrl: baseUrlInput });
      if (!mountedRef.current) return;
      setLoad({ status: "ready", baseUrl: result.baseUrl });
      setBaseUrlInput(result.baseUrl);
      setSave({ status: "idle" });
    } catch (err) {
      if (!mountedRef.current) return;
      const { code, message } =
        err instanceof BridgeError
          ? { code: err.code, message: err.message }
          : { code: "UNKNOWN", message: err instanceof Error ? err.message : String(err) };
      setSave({ status: "error", code, message });
      throw err;
    }
  }, [nativeBridge, baseUrlInput]);

  return { load, save, baseUrlInput, setBaseUrlInput, submit };
}
