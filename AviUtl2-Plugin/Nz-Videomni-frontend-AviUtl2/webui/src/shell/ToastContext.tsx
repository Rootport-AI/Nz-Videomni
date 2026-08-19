import { createContext, useCallback, useContext, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";

export interface ToastItem {
  id: string;
  /** M6 adds "warning" for the Chain screen's "Style LoRA tags are ignored"
   * notice (task brief §4) — informational, not an error, so it gets its
   * own (amber) styling in `Toasts.tsx`/`AppShell.css` rather than
   * overloading "error". */
  kind: "success" | "error" | "warning";
  message: string;
  /** When present, clicking the toast should select/highlight this job in
   * the job rail (task brief §4: "クリックで該当ジョブカードへ"). */
  jobId?: string;
}

export interface ToastsContextValue {
  toasts: ToastItem[];
  push: (toast: Omit<ToastItem, "id">) => void;
  dismiss: (id: string) => void;
}

const ToastsContext = createContext<ToastsContextValue | null>(null);

/** Auto-dismiss delay (task brief §4: "数秒で自動消滅"). */
const AUTO_DISMISS_MS = 6_000;

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<ToastItem[]>([]);
  const counterRef = useRef(0);
  const timersRef = useRef(new Map<string, ReturnType<typeof setTimeout>>());

  const dismiss = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
    const timer = timersRef.current.get(id);
    if (timer !== undefined) {
      clearTimeout(timer);
      timersRef.current.delete(id);
    }
  }, []);

  const push = useCallback(
    (toast: Omit<ToastItem, "id">) => {
      counterRef.current += 1;
      const id = `toast-${counterRef.current}`;
      setToasts((prev) => [...prev, { ...toast, id }]);
      const timer = setTimeout(() => dismiss(id), AUTO_DISMISS_MS);
      timersRef.current.set(id, timer);
    },
    [dismiss],
  );

  const value = useMemo<ToastsContextValue>(() => ({ toasts, push, dismiss }), [toasts, push, dismiss]);

  return <ToastsContext.Provider value={value}>{children}</ToastsContext.Provider>;
}

export function useToasts(): ToastsContextValue {
  const ctx = useContext(ToastsContext);
  if (!ctx) throw new Error("useToasts must be used within a ToastProvider");
  return ctx;
}
