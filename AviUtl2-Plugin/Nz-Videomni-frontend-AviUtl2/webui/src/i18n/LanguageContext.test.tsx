import { renderHook, act } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import type { ReactNode } from "react";
import { LanguageProvider, useLanguage, useStrings } from "./LanguageContext";
import { LANG_STORAGE_KEY, en, ja } from "./strings";

function wrapper({ children }: { children: ReactNode }) {
  return <LanguageProvider>{children}</LanguageProvider>;
}

describe("LanguageContext", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  afterEach(() => {
    window.localStorage.clear();
  });

  it("defaults to English when nothing is stored", () => {
    const { result } = renderHook(() => ({ lang: useLanguage(), strings: useStrings() }), { wrapper });
    expect(result.current.lang.lang).toBe("en");
    expect(result.current.strings).toBe(en);
  });

  it("restores a previously persisted language on mount", () => {
    window.localStorage.setItem(LANG_STORAGE_KEY, "ja");
    const { result } = renderHook(() => ({ lang: useLanguage(), strings: useStrings() }), { wrapper });
    expect(result.current.lang.lang).toBe("ja");
    expect(result.current.strings).toBe(ja);
  });

  it("ignores an unrecognized stored value and falls back to English", () => {
    window.localStorage.setItem(LANG_STORAGE_KEY, "fr");
    const { result } = renderHook(() => useLanguage(), { wrapper });
    expect(result.current.lang).toBe("en");
  });

  it("setLang switches the active dictionary and persists the choice", () => {
    const { result } = renderHook(() => ({ ctx: useLanguage(), strings: useStrings() }), { wrapper });
    expect(result.current.strings).toBe(en);

    act(() => {
      result.current.ctx.setLang("ja");
    });

    expect(result.current.ctx.lang).toBe("ja");
    expect(result.current.strings).toBe(ja);
    expect(window.localStorage.getItem(LANG_STORAGE_KEY)).toBe("ja");
  });

  it("useLanguage/useStrings throw when used outside a LanguageProvider", () => {
    expect(() => renderHook(() => useLanguage())).toThrow(/LanguageProvider/);
    expect(() => renderHook(() => useStrings())).toThrow(/LanguageProvider/);
  });
});
