import { useStrings } from "../../i18n/LanguageContext";
import type { ServerStatusState } from "./useServerStatus";

export function StatusHeader({ state, onRetry }: { state: ServerStatusState; onRetry: () => void }) {
  const strings = useStrings();
  switch (state.kind) {
    case "checking":
      return <span className="badge badge-checking">{strings.serverStatus.checking}</span>;
    case "bridge-unavailable":
      return <span className="badge badge-disconnected">{strings.serverStatus.bridgeUnavailable}</span>;
    case "offline":
      return (
        <span className="status-badge-group">
          <span className="badge badge-disconnected">{strings.serverStatus.offline}</span>
          <button type="button" className="link-button" onClick={onRetry}>
            {strings.serverStatus.retry}
          </button>
        </span>
      );
    case "online":
      return <span className="badge badge-connected">{strings.serverStatus.online}</span>;
    case "loading-models":
      // Same busy styling as a running job — both mean "the server is working,
      // don't submit" — but its own wording, because the remedy differs: a job
      // finishes on its own schedule, a model load is something the user just
      // started and is waiting out (Docs/MULTI_ENGINE_DESIGN.md §6.5).
      return <span className="badge badge-busy">{strings.serverStatus.loadingModels}</span>;
    case "busy":
      return <span className="badge badge-busy">{strings.serverStatus.busy}</span>;
    case "error":
      return (
        <span className="status-badge-group">
          <span className="badge badge-disconnected">{strings.serverStatus.error}</span>
          <button type="button" className="link-button" onClick={onRetry}>
            {strings.serverStatus.retry}
          </button>
        </span>
      );
    default:
      return null;
  }
}
