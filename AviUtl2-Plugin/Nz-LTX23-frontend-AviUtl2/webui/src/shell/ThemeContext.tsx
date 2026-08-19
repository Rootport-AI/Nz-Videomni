import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";

export type Theme = "dark" | "light";

/** `localStorage` key the active theme is persisted under (N8 owner decision:
 * dark既定・手動light・localStorage単独 — no native/settings.json
 * involvement). Mirrors `i18n/strings.ts`'s `LANG_STORAGE_KEY` naming. */
export const THEME_STORAGE_KEY = "nzltx23.theme";

export const DEFAULT_THEME: Theme = "dark";

/** Reads the persisted theme choice. Wrapped in a `try` since `localStorage`
 * can throw in some restricted embeddings (e.g. a WebView2 host with storage
 * disabled) — falling back to the default is preferable to a crash. Mirrors
 * `i18n/LanguageContext.tsx`'s `readStoredLang`. Exported so `main.tsx` can
 * read it synchronously before the first paint (FOUC avoidance, see
 * `applyThemeToDocument` below). */
export function readStoredTheme(): Theme {
  try {
    const stored = window.localStorage.getItem(THEME_STORAGE_KEY);
    if (stored === "dark" || stored === "light") return stored;
  } catch {
    // Ignore — localStorage unavailable.
  }
  return DEFAULT_THEME;
}

function writeStoredTheme(theme: Theme): void {
  try {
    window.localStorage.setItem(THEME_STORAGE_KEY, theme);
  } catch {
    // Ignore — localStorage unavailable (e.g. private mode).
  }
}

/** Stamps `<html data-theme="...">`, which `index.css`'s
 * `:root[data-theme="light"]` selector reacts to (dark has no attribute
 * value to match, so it's the implicit default). Exported so `main.tsx` can
 * call it synchronously before `createRoot(...).render(...)` — applying the
 * persisted theme before the first paint avoids a one-frame flash of the
 * dark default when the stored preference is "light". */
export function applyThemeToDocument(theme: Theme): void {
  document.documentElement.setAttribute("data-theme", theme);
}

export interface ThemeContextValue {
  theme: Theme;
  setTheme: (theme: Theme) => void;
}

const ThemeContext = createContext<ThemeContextValue | null>(null);

/** Wraps the whole app (see `shell/AppShell.tsx`) so every component can read
 * the active theme via `useTheme()` without prop drilling — placed as the
 * outermost provider there (even above `LanguageProvider`), since it's the
 * most fundamental of the two. Structurally identical to
 * `i18n/LanguageContext.tsx`'s `LanguageProvider`. */
export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setThemeState] = useState<Theme>(() => readStoredTheme());

  // Keeps `<html data-theme>` in sync with state. Redundant with
  // `main.tsx`'s synchronous pre-render apply on first mount, but that's
  // harmless — this is what actually reacts to later `setTheme` calls.
  useEffect(() => {
    applyThemeToDocument(theme);
  }, [theme]);

  const setTheme = useCallback((next: Theme) => {
    setThemeState(next);
    writeStoredTheme(next);
  }, []);

  const value = useMemo<ThemeContextValue>(() => ({ theme, setTheme }), [theme, setTheme]);

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useTheme(): ThemeContextValue {
  const ctx = useContext(ThemeContext);
  if (!ctx) throw new Error("useTheme must be used within a ThemeProvider");
  return ctx;
}
