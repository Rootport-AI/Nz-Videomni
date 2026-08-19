import { useStrings } from "../i18n/LanguageContext";
import { useToasts } from "./ToastContext";

export interface ToastsProps {
  /** Task brief §4: clicking a toast should jump to the job it refers to
   * (expand the rail + highlight the card). */
  onSelectJob?: (jobId: string) => void;
}

export function Toasts({ onSelectJob }: ToastsProps) {
  const strings = useStrings();
  const { toasts, dismiss } = useToasts();
  if (toasts.length === 0) return null;

  return (
    <div className="toast-stack" role="status" aria-live="polite">
      {toasts.map((toast) => (
        <div key={toast.id} className={`toast toast-${toast.kind}`}>
          <button
            type="button"
            className="toast-message"
            onClick={() => {
              if (toast.jobId) onSelectJob?.(toast.jobId);
              dismiss(toast.id);
            }}
          >
            {toast.message}
          </button>
          <button
            type="button"
            className="toast-dismiss"
            aria-label={strings.common.dismissNotification}
            onClick={() => dismiss(toast.id)}
          >
            ×
          </button>
        </div>
      ))}
    </div>
  );
}
