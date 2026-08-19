import { createContext, useCallback, useContext, useMemo, useState } from "react";
import type { ReactNode } from "react";
import { DEFAULT_LANG, DICTIONARIES, LANG_STORAGE_KEY } from "./strings";
import type { Lang, Strings } from "./strings";

export type { Lang, Strings };

/** Reads the persisted language choice (task brief: "選択はlocalStorageに保存
 * （キー nzltx23.lang、既定 "en"）"). Wrapped in a `try` since `localStorage`
 * can throw in some restricted embeddings (e.g. a WebView2 host with storage
 * disabled) — falling back to the default is preferable to a crash. */
function readStoredLang(): Lang {
  try {
    const stored = window.localStorage.getItem(LANG_STORAGE_KEY);
    if (stored === "en" || stored === "ja") return stored;
  } catch {
    // Ignore — localStorage unavailable.
  }
  return DEFAULT_LANG;
}

function writeStoredLang(lang: Lang): void {
  try {
    window.localStorage.setItem(LANG_STORAGE_KEY, lang);
  } catch {
    // Ignore — localStorage unavailable (e.g. private mode).
  }
}

export interface LanguageContextValue {
  lang: Lang;
  setLang: (lang: Lang) => void;
}

const LanguageContext = createContext<LanguageContextValue | null>(null);

/** Wraps the whole app (see `shell/AppShell.tsx`) so every component can read
 * the active language via `useLanguage()`/`useStrings()` without prop
 * drilling. Must be mounted above any component that calls either hook —
 * `JobsProvider`/`ToastProvider` included, since `JobsContext.tsx` pushes
 * translated toast copy. */
export function LanguageProvider({ children }: { children: ReactNode }) {
  const [lang, setLangState] = useState<Lang>(() => readStoredLang());

  const setLang = useCallback((next: Lang) => {
    setLangState(next);
    writeStoredLang(next);
  }, []);

  const value = useMemo<LanguageContextValue>(() => ({ lang, setLang }), [lang, setLang]);

  return <LanguageContext.Provider value={value}>{children}</LanguageContext.Provider>;
}

export function useLanguage(): LanguageContextValue {
  const ctx = useContext(LanguageContext);
  if (!ctx) throw new Error("useLanguage must be used within a LanguageProvider");
  return ctx;
}

/** The active dictionary for the current language. Every component that
 * previously did `import { strings } from "../i18n/strings"` should instead
 * call this hook so switching languages re-renders it automatically (M7b). */
export function useStrings(): Strings {
  const { lang } = useLanguage();
  return DICTIONARIES[lang];
}
