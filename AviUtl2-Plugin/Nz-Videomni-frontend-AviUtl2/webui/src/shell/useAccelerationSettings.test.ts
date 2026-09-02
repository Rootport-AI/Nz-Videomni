import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
  ACCELERATION_STORAGE_KEY,
  ATTENTION_BACKEND_DEFAULT,
  BLOCK_SWAP_PREFETCH_SERVER_DEFAULT,
  KEEP_RESIDENT_SERVER_DEFAULT,
  KEEP_RESIDENT_EMBEDDINGS_SERVER_DEFAULT,
  FUSED_GGUF_DEQUANT_KERNEL_SERVER_DEFAULT,
  VAE_MODE_DEFAULT,
} from "./accelerationSettings";
import { useAccelerationSettings } from "./useAccelerationSettings";

describe("useAccelerationSettings", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  afterEach(() => {
    window.localStorage.clear();
  });

  it("seeds from the backend defaults when nothing is stored", () => {
    const { result } = renderHook(() => useAccelerationSettings());
    expect(result.current.acceleration).toEqual({
      attentionBackend: ATTENTION_BACKEND_DEFAULT,
      blockSwapPrefetch: BLOCK_SWAP_PREFETCH_SERVER_DEFAULT,
      keepResident: KEEP_RESIDENT_SERVER_DEFAULT,
      fusedGgufDequantKernel: FUSED_GGUF_DEQUANT_KERNEL_SERVER_DEFAULT,
      vaeMode: VAE_MODE_DEFAULT,
      keepResidentEmbeddings: KEEP_RESIDENT_EMBEDDINGS_SERVER_DEFAULT,
    });
  });

  it("restores the persisted attention backend on mount (unlike NAG, this one survives a reload)", () => {
    // Old bare-string form (pre-2026-08-01) — still accepted.
    window.localStorage.setItem(ACCELERATION_STORAGE_KEY, "sage");
    const { result } = renderHook(() => useAccelerationSettings());
    expect(result.current.acceleration.attentionBackend).toBe("sage");
  });

  it("restores a persisted blockSwapPrefetch=true from the JSON form", () => {
    window.localStorage.setItem(
      ACCELERATION_STORAGE_KEY,
      JSON.stringify({ attentionBackend: "sdpa", blockSwapPrefetch: true }),
    );
    const { result } = renderHook(() => useAccelerationSettings());
    expect(result.current.acceleration.blockSwapPrefetch).toBe(true);
    expect(result.current.acceleration.attentionBackend).toBe("sdpa");
  });

  it("setAttentionBackend updates state and writes through to localStorage", async () => {
    const { result } = renderHook(() => useAccelerationSettings());

    act(() => result.current.setAttentionBackend("sage"));
    expect(result.current.acceleration.attentionBackend).toBe("sage");
    await waitFor(() => {
      expect(JSON.parse(window.localStorage.getItem(ACCELERATION_STORAGE_KEY)!)).toEqual({
        attentionBackend: "sage",
        blockSwapPrefetch: BLOCK_SWAP_PREFETCH_SERVER_DEFAULT,
        keepResident: KEEP_RESIDENT_SERVER_DEFAULT,
        fusedGgufDequantKernel: FUSED_GGUF_DEQUANT_KERNEL_SERVER_DEFAULT,
        vaeMode: VAE_MODE_DEFAULT,
        keepResidentEmbeddings: KEEP_RESIDENT_EMBEDDINGS_SERVER_DEFAULT,
      });
    });

    act(() => result.current.setAttentionBackend("sdpa"));
    expect(result.current.acceleration.attentionBackend).toBe("sdpa");
    await waitFor(() => {
      expect(JSON.parse(window.localStorage.getItem(ACCELERATION_STORAGE_KEY)!)).toEqual({
        attentionBackend: "sdpa",
        blockSwapPrefetch: BLOCK_SWAP_PREFETCH_SERVER_DEFAULT,
        keepResident: KEEP_RESIDENT_SERVER_DEFAULT,
        fusedGgufDequantKernel: FUSED_GGUF_DEQUANT_KERNEL_SERVER_DEFAULT,
        vaeMode: VAE_MODE_DEFAULT,
        keepResidentEmbeddings: KEEP_RESIDENT_EMBEDDINGS_SERVER_DEFAULT,
      });
    });
  });

  it("setBlockSwapPrefetch updates state, writes through, and does not disturb attentionBackend", async () => {
    const { result } = renderHook(() => useAccelerationSettings());

    act(() => result.current.setAttentionBackend("sage"));
    act(() => result.current.setBlockSwapPrefetch(true));

    expect(result.current.acceleration.blockSwapPrefetch).toBe(true);
    expect(result.current.acceleration.attentionBackend).toBe("sage");
    await waitFor(() => {
      expect(JSON.parse(window.localStorage.getItem(ACCELERATION_STORAGE_KEY)!)).toEqual({
        attentionBackend: "sage",
        blockSwapPrefetch: true,
        keepResident: KEEP_RESIDENT_SERVER_DEFAULT,
        fusedGgufDequantKernel: FUSED_GGUF_DEQUANT_KERNEL_SERVER_DEFAULT,
        vaeMode: VAE_MODE_DEFAULT,
        keepResidentEmbeddings: KEEP_RESIDENT_EMBEDDINGS_SERVER_DEFAULT,
      });
    });

    act(() => result.current.setBlockSwapPrefetch(false));
    expect(result.current.acceleration.blockSwapPrefetch).toBe(false);
    await waitFor(() => {
      expect(JSON.parse(window.localStorage.getItem(ACCELERATION_STORAGE_KEY)!)).toEqual({
        attentionBackend: "sage",
        blockSwapPrefetch: false,
        keepResident: KEEP_RESIDENT_SERVER_DEFAULT,
        fusedGgufDequantKernel: FUSED_GGUF_DEQUANT_KERNEL_SERVER_DEFAULT,
        vaeMode: VAE_MODE_DEFAULT,
        keepResidentEmbeddings: KEEP_RESIDENT_EMBEDDINGS_SERVER_DEFAULT,
      });
    });
  });

  it("restores a persisted blockSwapPrefetch across remount", async () => {
    const { result, unmount } = renderHook(() => useAccelerationSettings());
    act(() => result.current.setBlockSwapPrefetch(true));
    await waitFor(() => {
      expect(JSON.parse(window.localStorage.getItem(ACCELERATION_STORAGE_KEY)!).blockSwapPrefetch).toBe(true);
    });
    unmount();

    const { result: result2 } = renderHook(() => useAccelerationSettings());
    expect(result2.current.acceleration.blockSwapPrefetch).toBe(true);
  });

  it("restores a persisted keepResident=true from the JSON form", () => {
    window.localStorage.setItem(
      ACCELERATION_STORAGE_KEY,
      JSON.stringify({ attentionBackend: "sdpa", blockSwapPrefetch: true, keepResident: true }),
    );
    const { result } = renderHook(() => useAccelerationSettings());
    expect(result.current.acceleration.keepResident).toBe(true);
    expect(result.current.acceleration.attentionBackend).toBe("sdpa");
  });

  it("setKeepResident updates state, writes through, and does not disturb the other two", async () => {
    const { result } = renderHook(() => useAccelerationSettings());

    act(() => result.current.setAttentionBackend("sage"));
    act(() => result.current.setKeepResident(true));

    expect(result.current.acceleration.keepResident).toBe(true);
    expect(result.current.acceleration.attentionBackend).toBe("sage");
    await waitFor(() => {
      expect(JSON.parse(window.localStorage.getItem(ACCELERATION_STORAGE_KEY)!)).toEqual({
        attentionBackend: "sage",
        blockSwapPrefetch: true,
        keepResident: true,
        fusedGgufDequantKernel: FUSED_GGUF_DEQUANT_KERNEL_SERVER_DEFAULT,
        vaeMode: VAE_MODE_DEFAULT,
        keepResidentEmbeddings: KEEP_RESIDENT_EMBEDDINGS_SERVER_DEFAULT,
      });
    });

    act(() => result.current.setKeepResident(false));
    expect(result.current.acceleration.keepResident).toBe(false);
    await waitFor(() => {
      expect(JSON.parse(window.localStorage.getItem(ACCELERATION_STORAGE_KEY)!)).toEqual({
        attentionBackend: "sage",
        blockSwapPrefetch: true,
        keepResident: false,
        fusedGgufDequantKernel: FUSED_GGUF_DEQUANT_KERNEL_SERVER_DEFAULT,
        vaeMode: VAE_MODE_DEFAULT,
        keepResidentEmbeddings: KEEP_RESIDENT_EMBEDDINGS_SERVER_DEFAULT,
      });
    });
  });

  // §1-10 (2026-08-03): what the hook EXPOSES is the effective object, what it
  // STORES is the raw choice — the split that lets the panel grey the toggle
  // out without losing the setting.
  it("exposes keepResident folded to false while prefetch is off, but still persists the stored true", async () => {
    const { result } = renderHook(() => useAccelerationSettings());

    act(() => result.current.setKeepResident(true));
    act(() => result.current.setBlockSwapPrefetch(false));

    expect(result.current.acceleration.keepResident).toBe(false);
    await waitFor(() => {
      expect(JSON.parse(window.localStorage.getItem(ACCELERATION_STORAGE_KEY)!).keepResident).toBe(true);
    });

    // Prefetch back on: the stored choice becomes effective again by itself.
    act(() => result.current.setBlockSwapPrefetch(true));
    expect(result.current.acceleration.keepResident).toBe(true);
  });

  it("folds keepResident to false when /status reports prefetch unavailable, and back when it does not", () => {
    window.localStorage.setItem(
      ACCELERATION_STORAGE_KEY,
      JSON.stringify({ attentionBackend: "sdpa", blockSwapPrefetch: true, keepResident: true }),
    );
    const unavailable = renderHook(() => useAccelerationSettings(false));
    expect(unavailable.result.current.acceleration.keepResident).toBe(false);
    // Only keep-resident folds — the prefetch choice itself is untouched.
    expect(unavailable.result.current.acceleration.blockSwapPrefetch).toBe(true);

    // `null` is UNKNOWN, not unavailable (the default when no status is given).
    expect(renderHook(() => useAccelerationSettings(null)).result.current.acceleration.keepResident).toBe(true);
    expect(renderHook(() => useAccelerationSettings(true)).result.current.acceleration.keepResident).toBe(true);
  });

  it("restores a persisted keepResident across remount", async () => {
    const { result, unmount } = renderHook(() => useAccelerationSettings());
    act(() => result.current.setKeepResident(true));
    await waitFor(() => {
      expect(JSON.parse(window.localStorage.getItem(ACCELERATION_STORAGE_KEY)!).keepResident).toBe(true);
    });
    unmount();

    const { result: result2 } = renderHook(() => useAccelerationSettings());
    expect(result2.current.acceleration.keepResident).toBe(true);
  });

  it("leaves vaeMode at its default while nothing touches it", () => {
    const { result } = renderHook(() => useAccelerationSettings());
    act(() => result.current.setAttentionBackend("sage"));
    act(() => result.current.setBlockSwapPrefetch(true));
    act(() => result.current.setKeepResident(true));
    act(() => result.current.setFusedGgufDequantKernel(true));
    expect(result.current.acceleration.vaeMode).toBe(VAE_MODE_DEFAULT);
  });

  // §1-11 (2026-08-04): the fourth persisted field. Unlike keep-resident it is
  // folded by nothing — what is stored is what every reader sees.
  it("restores a persisted fusedGgufDequantKernel=true from the JSON form", () => {
    window.localStorage.setItem(
      ACCELERATION_STORAGE_KEY,
      JSON.stringify({ attentionBackend: "sdpa", blockSwapPrefetch: true, fusedGgufDequantKernel: true }),
    );
    const { result } = renderHook(() => useAccelerationSettings());
    expect(result.current.acceleration.fusedGgufDequantKernel).toBe(true);
    expect(result.current.acceleration.attentionBackend).toBe("sdpa");
  });

  it("setFusedGgufDequantKernel updates state, writes through, and does not disturb the other three", async () => {
    const { result } = renderHook(() => useAccelerationSettings());

    act(() => result.current.setAttentionBackend("sage"));
    act(() => result.current.setKeepResident(true));
    act(() => result.current.setFusedGgufDequantKernel(true));

    expect(result.current.acceleration.fusedGgufDequantKernel).toBe(true);
    expect(result.current.acceleration.attentionBackend).toBe("sage");
    expect(result.current.acceleration.keepResident).toBe(true);
    await waitFor(() => {
      expect(JSON.parse(window.localStorage.getItem(ACCELERATION_STORAGE_KEY)!)).toEqual({
        attentionBackend: "sage",
        blockSwapPrefetch: true,
        keepResident: true,
        fusedGgufDequantKernel: true,
        vaeMode: VAE_MODE_DEFAULT,
        keepResidentEmbeddings: KEEP_RESIDENT_EMBEDDINGS_SERVER_DEFAULT,
      });
    });

    act(() => result.current.setFusedGgufDequantKernel(false));
    expect(result.current.acceleration.fusedGgufDequantKernel).toBe(false);
    await waitFor(() => {
      expect(JSON.parse(window.localStorage.getItem(ACCELERATION_STORAGE_KEY)!).fusedGgufDequantKernel).toBe(false);
    });
  });

  it("is NOT folded by prefetch being off — it is an independent toggle (unlike keep-resident)", () => {
    window.localStorage.setItem(
      ACCELERATION_STORAGE_KEY,
      JSON.stringify({
        attentionBackend: "sdpa",
        blockSwapPrefetch: false,
        keepResident: true,
        fusedGgufDequantKernel: true,
      }),
    );
    // Prefetch off locally AND reported unavailable by /status: keep-resident
    // folds to false, the fused kernel does not budge.
    const { result } = renderHook(() => useAccelerationSettings(false));
    expect(result.current.acceleration.keepResident).toBe(false);
    expect(result.current.acceleration.fusedGgufDequantKernel).toBe(true);
  });

  it("restores a persisted fusedGgufDequantKernel across remount", async () => {
    const { result, unmount } = renderHook(() => useAccelerationSettings());
    act(() => result.current.setFusedGgufDequantKernel(true));
    await waitFor(() => {
      expect(JSON.parse(window.localStorage.getItem(ACCELERATION_STORAGE_KEY)!).fusedGgufDequantKernel).toBe(true);
    });
    unmount();

    const { result: result2 } = renderHook(() => useAccelerationSettings());
    expect(result2.current.acceleration.fusedGgufDequantKernel).toBe(true);
  });

  // §52 (2026-08-05): the fifth persisted field — PrunaVAED. Like the fused
  // kernel and unlike keep-resident it is folded by nothing.
  it("restores a persisted vaeMode from the JSON form", () => {
    window.localStorage.setItem(
      ACCELERATION_STORAGE_KEY,
      JSON.stringify({ attentionBackend: "sdpa", blockSwapPrefetch: true, vaeMode: "prune_vaed" }),
    );
    const { result } = renderHook(() => useAccelerationSettings());
    expect(result.current.acceleration.vaeMode).toBe("prune_vaed");
    expect(result.current.acceleration.attentionBackend).toBe("sdpa");
  });

  it("setVaeMode updates state, writes through, and does not disturb the other five", async () => {
    const { result } = renderHook(() => useAccelerationSettings());

    act(() => result.current.setAttentionBackend("sage"));
    act(() => result.current.setKeepResident(true));
    act(() => result.current.setVaeMode("prune_vaed"));

    expect(result.current.acceleration.vaeMode).toBe("prune_vaed");
    expect(result.current.acceleration.attentionBackend).toBe("sage");
    expect(result.current.acceleration.keepResident).toBe(true);
    expect(result.current.acceleration.fusedGgufDequantKernel).toBe(FUSED_GGUF_DEQUANT_KERNEL_SERVER_DEFAULT);
    await waitFor(() => {
      expect(JSON.parse(window.localStorage.getItem(ACCELERATION_STORAGE_KEY)!)).toEqual({
        attentionBackend: "sage",
        blockSwapPrefetch: true,
        keepResident: true,
        fusedGgufDequantKernel: FUSED_GGUF_DEQUANT_KERNEL_SERVER_DEFAULT,
        vaeMode: "prune_vaed",
        keepResidentEmbeddings: KEEP_RESIDENT_EMBEDDINGS_SERVER_DEFAULT,
      });
    });

    act(() => result.current.setVaeMode("default"));
    expect(result.current.acceleration.vaeMode).toBe("default");
    await waitFor(() => {
      expect(JSON.parse(window.localStorage.getItem(ACCELERATION_STORAGE_KEY)!).vaeMode).toBe("default");
    });
  });

  // 台帳 §3-114 (2026-09-03): the sixth persisted field. Like the fused kernel
  // and the VAE row, and UNLIKE the keep-resident field it is named after, it is
  // folded by nothing — what is stored is what every reader sees.
  it("restores a persisted keepResidentEmbeddings=true from the JSON form", () => {
    window.localStorage.setItem(
      ACCELERATION_STORAGE_KEY,
      JSON.stringify({ attentionBackend: "sdpa", blockSwapPrefetch: true, keepResidentEmbeddings: true }),
    );
    const { result } = renderHook(() => useAccelerationSettings());
    expect(result.current.acceleration.keepResidentEmbeddings).toBe(true);
    expect(result.current.acceleration.attentionBackend).toBe("sdpa");
  });

  it("setKeepResidentEmbeddings updates state, writes through, and does not disturb the other five", async () => {
    const { result } = renderHook(() => useAccelerationSettings());

    act(() => result.current.setAttentionBackend("sage"));
    act(() => result.current.setKeepResident(true));
    act(() => result.current.setKeepResidentEmbeddings(true));

    expect(result.current.acceleration.keepResidentEmbeddings).toBe(true);
    expect(result.current.acceleration.attentionBackend).toBe("sage");
    expect(result.current.acceleration.keepResident).toBe(true);
    expect(result.current.acceleration.vaeMode).toBe(VAE_MODE_DEFAULT);
    await waitFor(() => {
      expect(JSON.parse(window.localStorage.getItem(ACCELERATION_STORAGE_KEY)!)).toEqual({
        attentionBackend: "sage",
        blockSwapPrefetch: true,
        keepResident: true,
        fusedGgufDequantKernel: FUSED_GGUF_DEQUANT_KERNEL_SERVER_DEFAULT,
        vaeMode: VAE_MODE_DEFAULT,
        keepResidentEmbeddings: true,
      });
    });

    act(() => result.current.setKeepResidentEmbeddings(false));
    expect(result.current.acceleration.keepResidentEmbeddings).toBe(false);
    await waitFor(() => {
      expect(JSON.parse(window.localStorage.getItem(ACCELERATION_STORAGE_KEY)!).keepResidentEmbeddings).toBe(false);
    });
  });

  it("keepResidentEmbeddings is NOT folded by prefetch being off — an independent toggle", () => {
    // The distinction that matters, since the two rows share a name:
    // `keepResident` is folded down while prefetch is off (§1-10), and this one
    // is a switch over a different object on a different engine, with no such
    // pairing to honour.
    window.localStorage.setItem(
      ACCELERATION_STORAGE_KEY,
      JSON.stringify({
        attentionBackend: "sdpa",
        blockSwapPrefetch: false,
        keepResident: true,
        keepResidentEmbeddings: true,
      }),
    );
    const { result } = renderHook(() => useAccelerationSettings(false));
    expect(result.current.acceleration.keepResident).toBe(false);
    expect(result.current.acceleration.keepResidentEmbeddings).toBe(true);
  });

  it("vaeMode is NOT folded by prefetch being off — an independent toggle", () => {
    window.localStorage.setItem(
      ACCELERATION_STORAGE_KEY,
      JSON.stringify({
        attentionBackend: "sdpa",
        blockSwapPrefetch: false,
        keepResident: true,
        vaeMode: "prune_vaed",
      }),
    );
    const { result } = renderHook(() => useAccelerationSettings(false));
    expect(result.current.acceleration.keepResident).toBe(false);
    expect(result.current.acceleration.vaeMode).toBe("prune_vaed");
  });

  it("restores a persisted vaeMode across remount", async () => {
    const { result, unmount } = renderHook(() => useAccelerationSettings());
    act(() => result.current.setVaeMode("prune_vaed"));
    await waitFor(() => {
      expect(JSON.parse(window.localStorage.getItem(ACCELERATION_STORAGE_KEY)!).vaeMode).toBe("prune_vaed");
    });
    unmount();

    const { result: result2 } = renderHook(() => useAccelerationSettings());
    expect(result2.current.acceleration.vaeMode).toBe("prune_vaed");
  });
});
