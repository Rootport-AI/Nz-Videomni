import { useState } from "react";
import { useStrings } from "../i18n/LanguageContext";
import type { ApiClient } from "../api/client";
import { useDangerZone } from "./useDangerZone";
import "./SettingsPanel.css";

export interface DangerZonePanelProps {
  /** Test-only override; production call sites omit it (`useDangerZone`
   * falls back to the app-wide `apiClient` singleton, same convention as
   * `ModelsPanel`'s own `apiClient` prop). */
  apiClient?: ApiClient;
}

/** Which of the two destructive actions currently has its inline
 * "are you sure?" confirmation exposed. Only one at a time — picking the
 * other action's trigger button while one is pending swaps to it rather than
 * stacking both confirmations. */
type PendingAction = "unload" | "purge" | null;

/** N4: the Settings panel's "danger zone" section — an explicit pipeline
 * unload (`POST /pipeline/unload`) and a bulk purge of every terminal
 * (completed/failed/cancelled) job (`GET /jobs` + `DELETE /jobs/{id}` per
 * entry). Modeled on `ModelsPanel`'s layout and inline-state conventions
 * (no toast/JobsContext — see `useDangerZone.ts`'s doc comment for why).
 * Both actions require a second, explicit click before they fire: the
 * trigger button is replaced in place by a confirmation line plus
 * "confirm"/"cancel" buttons — no modal/dialog library, matching this app's
 * dependency-light approach (`SettingsPanel.tsx` itself is the only
 * dialog-shaped precedent, and it's a plain overlay div). */
export function DangerZonePanel({ apiClient }: DangerZonePanelProps) {
  const strings = useStrings();
  const dangerZone = useDangerZone({ ...(apiClient !== undefined ? { apiClient } : {}) });
  const [pending, setPending] = useState<PendingAction>(null);

  const unloadBusy = dangerZone.unload.status === "loading";
  const purgeBusy = dangerZone.purge.status === "loading";
  const busy = unloadBusy || purgeBusy;

  let unloadMessage: string | null = null;
  let unloadIsError = false;
  if (dangerZone.unload.status === "done") {
    unloadMessage = strings.settings.unloadDone;
  } else if (dangerZone.unload.status === "busy") {
    unloadMessage = strings.settings.unloadBusy;
    unloadIsError = true;
  } else if (dangerZone.unload.status === "error") {
    unloadMessage = strings.settings.unloadError;
    unloadIsError = true;
  }

  let purgeMessage: string | null = null;
  let purgeIsError = false;
  if (dangerZone.purge.status === "done") {
    const { deleted, attempted } = dangerZone.purge;
    if (attempted === 0) {
      purgeMessage = strings.settings.purgeNone;
    } else if (deleted === attempted) {
      purgeMessage = strings.settings.purgeDone(deleted);
    } else {
      // attempted > 0 && deleted < attempted: some (or all) deletes failed
      // after a terminal job was found — must not read as "nothing to purge".
      purgeMessage = strings.settings.purgePartial(deleted, attempted);
      purgeIsError = true;
    }
  } else if (dangerZone.purge.status === "error") {
    purgeMessage = strings.settings.purgeError;
    purgeIsError = true;
  }

  return (
    <div className="models-section danger-section">
      <h3>{strings.settings.dangerZoneSectionTitle}</h3>
      <p className="field-hint">{strings.settings.dangerZoneHint}</p>

      <div className="settings-panel-actions models-actions">
        {pending === "unload" ? (
          <div className="danger-confirm-row">
            <span className="field-hint">{strings.settings.unloadConfirm}</span>
            <button type="button" className="secondary-button" disabled={busy} onClick={() => setPending(null)}>
              {strings.settings.dangerCancelButton}
            </button>
            <button
              type="button"
              className="btn-danger"
              disabled={busy}
              onClick={() => {
                setPending(null);
                dangerZone.unloadPipeline();
              }}
            >
              {strings.settings.unloadConfirmButton}
            </button>
          </div>
        ) : (
          <button type="button" className="btn-danger" disabled={busy} onClick={() => setPending("unload")}>
            {strings.settings.unloadButton}
          </button>
        )}
      </div>
      {unloadMessage && <p className={`field-hint${unloadIsError ? " field-hint-error" : ""}`}>{unloadMessage}</p>}

      <div className="settings-panel-actions models-actions">
        {pending === "purge" ? (
          <div className="danger-confirm-row">
            <span className="field-hint">{strings.settings.purgeConfirm}</span>
            <button type="button" className="secondary-button" disabled={busy} onClick={() => setPending(null)}>
              {strings.settings.dangerCancelButton}
            </button>
            <button
              type="button"
              className="btn-danger"
              disabled={busy}
              onClick={() => {
                setPending(null);
                dangerZone.purgeTerminalJobs();
              }}
            >
              {strings.settings.purgeConfirmButton}
            </button>
          </div>
        ) : (
          <button type="button" className="btn-danger" disabled={busy} onClick={() => setPending("purge")}>
            {strings.settings.purgeButton}
          </button>
        )}
      </div>
      {purgeMessage && <p className={`field-hint${purgeIsError ? " field-hint-error" : ""}`}>{purgeMessage}</p>}
    </div>
  );
}
