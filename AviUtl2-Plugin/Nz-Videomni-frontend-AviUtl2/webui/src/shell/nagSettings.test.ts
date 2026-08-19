import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  DEFAULT_NEGATIVE_PROMPT,
  NAG_ALPHA_DEFAULT,
  NAG_OFF,
  NAG_SCALE_DEFAULT,
  NAG_TAU_DEFAULT,
  NAG_TEXT_STORAGE_KEY,
  NEG_METHOD_DEFAULT,
  VSF_SCALE_DEFAULT,
  isNagNegativeEmpty,
  nagRequestFields,
  readStoredNagText,
  writeStoredNagText,
  type NagSettings,
} from "./nagSettings";

function makeNag(overrides: Partial<NagSettings> = {}): NagSettings {
  return {
    text: DEFAULT_NEGATIVE_PROMPT,
    enabled: false,
    scale: NAG_SCALE_DEFAULT,
    tau: NAG_TAU_DEFAULT,
    alpha: NAG_ALPHA_DEFAULT,
    method: NEG_METHOD_DEFAULT,
    vsfScale: VSF_SCALE_DEFAULT,
    ...overrides,
  };
}

describe("nagSettings", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  afterEach(() => {
    window.localStorage.clear();
    vi.restoreAllMocks();
  });

  describe("nagRequestFields", () => {
    it("returns {} for undefined", () => {
      expect(nagRequestFields(undefined)).toEqual({});
    });

    it("returns {} for the NAG_OFF sentinel", () => {
      expect(nagRequestFields(NAG_OFF)).toEqual({});
    });

    it("returns {} when enabled is false", () => {
      expect(nagRequestFields(makeNag({ enabled: false, text: "blurry" }))).toEqual({});
    });

    it("returns exactly the 7 keys, with the given values, when enabled and method is 'nag'", () => {
      const nag = makeNag({ enabled: true, text: "blurry, watermark", scale: 12, tau: 3, alpha: 0.5, method: "nag", vsfScale: 2 });
      expect(nagRequestFields(nag)).toEqual({
        negative_prompt: "blurry, watermark",
        nag_enabled: true,
        nag_scale: 12,
        nag_tau: 3,
        nag_alpha: 0.5,
        neg_method: "nag",
        vsf_scale: 2,
      });
    });

    it("returns exactly the 7 keys, with the given values, when enabled and method is 'vsf'", () => {
      const nag = makeNag({ enabled: true, text: "blurry, watermark", scale: 12, tau: 3, alpha: 0.5, method: "vsf", vsfScale: 4 });
      expect(nagRequestFields(nag)).toEqual({
        negative_prompt: "blurry, watermark",
        nag_enabled: true,
        nag_scale: 12,
        nag_tau: 3,
        nag_alpha: 0.5,
        neg_method: "vsf",
        vsf_scale: 4,
      });
    });
  });

  describe("isNagNegativeEmpty", () => {
    it("is false when disabled, even with empty text", () => {
      expect(isNagNegativeEmpty(makeNag({ enabled: false, text: "" }))).toBe(false);
    });

    it("is true when enabled and text is empty", () => {
      expect(isNagNegativeEmpty(makeNag({ enabled: true, text: "" }))).toBe(true);
    });

    it("is true when enabled and text is whitespace-only", () => {
      expect(isNagNegativeEmpty(makeNag({ enabled: true, text: "   " }))).toBe(true);
    });

    it("is false when enabled and text is non-empty", () => {
      expect(isNagNegativeEmpty(makeNag({ enabled: true, text: "blurry" }))).toBe(false);
    });
  });

  describe("readStoredNagText", () => {
    it("returns the default when nothing is stored", () => {
      expect(readStoredNagText()).toBe(DEFAULT_NEGATIVE_PROMPT);
    });

    it("returns the stored value when present", () => {
      window.localStorage.setItem(NAG_TEXT_STORAGE_KEY, "worst quality, jpeg artifacts");
      expect(readStoredNagText()).toBe("worst quality, jpeg artifacts");
    });

    it("falls back to the default when the stored value is whitespace-only", () => {
      window.localStorage.setItem(NAG_TEXT_STORAGE_KEY, "   ");
      expect(readStoredNagText()).toBe(DEFAULT_NEGATIVE_PROMPT);
    });

    it("falls back to the default when localStorage throws", () => {
      const getItemSpy = vi.spyOn(window.localStorage.__proto__, "getItem").mockImplementation(() => {
        throw new Error("storage disabled");
      });

      expect(readStoredNagText()).toBe(DEFAULT_NEGATIVE_PROMPT);

      getItemSpy.mockRestore();
    });
  });

  describe("writeStoredNagText", () => {
    it("persists the text under NAG_TEXT_STORAGE_KEY", () => {
      writeStoredNagText("worst quality");
      expect(window.localStorage.getItem(NAG_TEXT_STORAGE_KEY)).toBe("worst quality");
    });

    it("swallows a localStorage write failure instead of throwing", () => {
      const setItemSpy = vi.spyOn(window.localStorage.__proto__, "setItem").mockImplementation(() => {
        throw new Error("storage disabled");
      });

      expect(() => writeStoredNagText("worst quality")).not.toThrow();

      setItemSpy.mockRestore();
    });
  });
});
