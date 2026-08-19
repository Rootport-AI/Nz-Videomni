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
