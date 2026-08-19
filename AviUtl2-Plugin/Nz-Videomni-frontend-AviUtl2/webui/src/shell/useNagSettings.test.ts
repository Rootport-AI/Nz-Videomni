import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
  DEFAULT_NEGATIVE_PROMPT,
  NAG_ALPHA_DEFAULT,
  NAG_SCALE_DEFAULT,
  NAG_TAU_DEFAULT,
  NEG_METHOD_DEFAULT,
  VSF_SCALE_DEFAULT,
} from "./nagSettings";
import { useNagSettings } from "./useNagSettings";

describe("useNagSettings", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });
  afterEach(() => {
    window.localStorage.clear();
  });

  it("starts with method=\"nag\", vsfScale at its default, and enabled OFF", () => {
    const { result } = renderHook(() => useNagSettings());

    expect(result.current.nag.method).toBe(NEG_METHOD_DEFAULT);
    expect(result.current.nag.method).toBe("nag");
    expect(result.current.nag.vsfScale).toBe(VSF_SCALE_DEFAULT);
    expect(result.current.nag.enabled).toBe(false);
    // Also the sibling fields, for completeness against a fresh mount with
    // no leftover localStorage text (owner decision §UX 5: text is the ONLY
    // field that persists).
    expect(result.current.nag.text).toBe(DEFAULT_NEGATIVE_PROMPT);
    expect(result.current.nag.scale).toBe(NAG_SCALE_DEFAULT);
    expect(result.current.nag.tau).toBe(NAG_TAU_DEFAULT);
    expect(result.current.nag.alpha).toBe(NAG_ALPHA_DEFAULT);
  });

  it("setMethod(\"vsf\") flips nag.method and leaves every other field untouched", () => {
    const { result } = renderHook(() => useNagSettings());

    act(() => {
      result.current.setMethod("vsf");
    });

    expect(result.current.nag.method).toBe("vsf");
    expect(result.current.nag.vsfScale).toBe(VSF_SCALE_DEFAULT);
    expect(result.current.nag.enabled).toBe(false);
  });

  it("resetParams() restores scale/tau/alpha/vsfScale to their defaults but leaves method/text/enabled alone", () => {
    const { result } = renderHook(() => useNagSettings());

    act(() => {
      result.current.setMethod("vsf");
      result.current.setEnabled(true);
      result.current.setText("no cats");
      result.current.setScale(NAG_SCALE_DEFAULT + 1);
      result.current.setTau(NAG_TAU_DEFAULT + 1);
      result.current.setAlpha(NAG_ALPHA_DEFAULT + 0.1);
      result.current.setVsfScale(VSF_SCALE_DEFAULT + 1);
    });

    act(() => {
      result.current.resetParams();
    });

    expect(result.current.nag.scale).toBe(NAG_SCALE_DEFAULT);
    expect(result.current.nag.tau).toBe(NAG_TAU_DEFAULT);
    expect(result.current.nag.alpha).toBe(NAG_ALPHA_DEFAULT);
    expect(result.current.nag.vsfScale).toBe(VSF_SCALE_DEFAULT);
    // owner decision §UX 3: resetParams() never touches method/text/enabled.
    expect(result.current.nag.method).toBe("vsf");
    expect(result.current.nag.text).toBe("no cats");
    expect(result.current.nag.enabled).toBe(true);
  });
});
