import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { PromptTokensRequest, PromptTokensResponse } from "../api/types";
import { PROMPT_TOKEN_DEBOUNCE_MS, usePromptTokenCounts } from "./usePromptTokenCounts";

const reply = (n: number): PromptTokensResponse => ({ prompt: n, negative_prompt: null, clips: null });

describe("usePromptTokenCounts", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it("sends once after the debounce and returns the counts", async () => {
    const countPromptTokens = vi.fn().mockResolvedValue(reply(7));
    const { result, rerender } = renderHook(
      ({ text }) => usePromptTokenCounts({ prompt: text }, "m", { countPromptTokens }),
      { initialProps: { text: "a" } },
    );
    rerender({ text: "ab" });
    rerender({ text: "abc" });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(PROMPT_TOKEN_DEBOUNCE_MS - 1);
    });
    expect(countPromptTokens).not.toHaveBeenCalled();
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1);
    });
    expect(countPromptTokens).toHaveBeenCalledTimes(1);
    expect(countPromptTokens).toHaveBeenCalledWith({ prompt: "abc" });
    expect(result.current).toEqual(reply(7));
  });

  it("drops an out-of-order (superseded) response", async () => {
    const resolvers: Array<(r: PromptTokensResponse) => void> = [];
    const countPromptTokens = vi.fn(
      (_req: PromptTokensRequest) => new Promise<PromptTokensResponse>((resolve) => resolvers.push(resolve)),
    );
    const { result, rerender } = renderHook(
      ({ text }) => usePromptTokenCounts({ prompt: text }, "m", { countPromptTokens }),
      { initialProps: { text: "a" } },
    );
    await act(async () => {
      await vi.advanceTimersByTimeAsync(PROMPT_TOKEN_DEBOUNCE_MS);
    });
    rerender({ text: "ab" });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(PROMPT_TOKEN_DEBOUNCE_MS);
    });
    expect(resolvers).toHaveLength(2);
    await act(async () => {
      resolvers[1]!(reply(2));
    });
    await act(async () => {
      resolvers[0]!(reply(1));
    });
    expect(result.current).toEqual(reply(2));
  });

  it("keeps the previous value while debouncing and clears it on failure", async () => {
    const countPromptTokens = vi.fn().mockResolvedValueOnce(reply(3)).mockRejectedValueOnce(new Error("x"));
    const { result, rerender } = renderHook(
      ({ text }) => usePromptTokenCounts({ prompt: text }, "m", { countPromptTokens }),
      { initialProps: { text: "a" } },
    );
    await act(async () => {
      await vi.advanceTimersByTimeAsync(PROMPT_TOKEN_DEBOUNCE_MS);
    });
    expect(result.current).toEqual(reply(3));
    rerender({ text: "ab" });
    expect(result.current).toEqual(reply(3));
    await act(async () => {
      await vi.advanceTimersByTimeAsync(PROMPT_TOKEN_DEBOUNCE_MS);
    });
    expect(result.current).toBeNull();
  });

  it("counts again when only the model key changes", async () => {
    const countPromptTokens = vi.fn().mockResolvedValue(reply(1));
    const { rerender } = renderHook(
      ({ key }) => usePromptTokenCounts({ prompt: "same" }, key, { countPromptTokens }),
      { initialProps: { key: "LTX23|false" } },
    );
    await act(async () => {
      await vi.advanceTimersByTimeAsync(PROMPT_TOKEN_DEBOUNCE_MS);
    });
    expect(countPromptTokens).toHaveBeenCalledTimes(1);
    rerender({ key: "LTX25|false" });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(PROMPT_TOKEN_DEBOUNCE_MS);
    });
    expect(countPromptTokens).toHaveBeenCalledTimes(2);
  });

  it("sends nothing while disabled (request null)", async () => {
    const countPromptTokens = vi.fn().mockResolvedValue(reply(1));
    renderHook(() => usePromptTokenCounts(null, "m", { countPromptTokens }));
    await act(async () => {
      await vi.advanceTimersByTimeAsync(PROMPT_TOKEN_DEBOUNCE_MS * 4);
    });
    expect(countPromptTokens).not.toHaveBeenCalled();
  });
});
