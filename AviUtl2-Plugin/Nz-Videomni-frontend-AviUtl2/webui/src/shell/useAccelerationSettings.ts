import { useCallback, useEffect, useMemo, useState } from "react";
import { effectiveAcceleration, readStoredAcceleration, writeStoredAcceleration } from "./accelerationSettings";
import type { AccelerationSettings, AttentionBackend, VaeMode } from "./accelerationSettings";

export interface UseAccelerationSettingsResult {
  /** The EFFECTIVE settings (§1-10, 2026-08-03): the stored choices with
   * `keepResident` folded down by `effectiveAcceleration` while block-swap
   * prefetch is off or unavailable. Every reader — the Settings panel's
   * rendering as much as Create/Chain/Batch's requests — sees the same object,
   * which is what keeps "what the toggle shows" and "what the request carries"
   * from drifting apart. The stored choice itself is untouched and comes back
   * as soon as prefetch does. */
  acceleration: AccelerationSettings;
  /** Write-through: the choice is mirrored to `localStorage` (via the
   * `useEffect` below, not synchronously). Unlike NAG's session-local
   * parameters, this one PERSISTS across reloads — it is a machine-level
   * capability choice ("this PC has SageAttention and I want to use it"), not
   * a per-generation creative parameter. */
  setAttentionBackend: (value: AttentionBackend) => void;
  /** Write-through to `localStorage`, same persistence rationale as
   * {@link setAttentionBackend} — block-swap prefetch is also a "this PC's
   * characteristics" choice rather than a per-generation one. */
  setBlockSwapPrefetch: (value: boolean) => void;
  /** Write-through to `localStorage`, same persistence rationale as
   * {@link setAttentionBackend} — keeping the model skeleton resident is a
   * "this PC has the RAM for it" choice, so it should survive a reload just
   * like the other two. */
  setKeepResident: (value: boolean) => void;
  /** Write-through to `localStorage`, same persistence rationale as
   * {@link setAttentionBackend} — whether the fused GGUF dequantization kernel
   * works on this machine (Triton availability) is a machine-level fact, so it
   * should survive a reload like the other three. */
  setFusedGgufDequantKernel: (value: boolean) => void;
  /** Write-through to `localStorage`, same persistence rationale as
   * {@link setAttentionBackend} — which decoder the user is willing to trade a
   * little quality to is a standing preference, not a per-generation one, so
   * it should survive a reload like the other four (backend §52,
   * 2026-08-05). */
  setVaeMode: (value: VaeMode) => void;
}

/**
 * Acceleration settings, owned by `AppShell` and called exactly ONCE (D1) so
 * the Settings panel and every submission path read the identical
 * `AccelerationSettings` object — the same "AppShell useState + props, no
 * Context" arrangement `useNagSettings` documents, and for the same reason: a
 * per-screen copy would silently reset on every right-click remount's `key`
 * bump.
 *
 * One `useState<AccelerationSettings>` with a lazy initializer, plus
 * functional updates in every setter (so none ever captures a stale value).
 * All FIVE fields — `attentionBackend` / `blockSwapPrefetch` / `keepResident` /
 * `fusedGgufDequantKernel` / `vaeMode` — restore from `localStorage` via
 * {@link readStoredAcceleration}. `vaeMode` joined them on 2026-08-05 (backend
 * §52) when it stopped being a mock; before that it seeded at its default
 * every mount and had no setter, exactly the state `fusedGgufDequantKernel`
 * left on 2026-08-04.
 *
 * The `localStorage` write itself lives in a `useEffect` keyed off the
 * persisted fields, NOT inside the setters' functional updaters — writing to
 * `localStorage` is a side effect, and running it inside a `setState`
 * updater function couples a render-phase-adjacent callback to an I/O call
 * that React may invoke more than once per commit (e.g. StrictMode's
 * double-invoke of updaters). Effects are the correct place for
 * synchronizing with a system outside React; this mirrors how
 * `PrefillPolicyContext.tsx` persists its own two axes.
 *
 * There is deliberately no capability hook alongside this one: whether the
 * server has SageAttention / block-swap prefetch installed already arrives on
 * the `GET /status` `AppShell` polls on a timer (`useServerStatus`), and
 * `sageAvailability` / `blockSwapPrefetchAvailability`
 * (`shell/accelerationSettings.ts`) read it out of that existing state.
 * `blockSwapPrefetchAvailable` is that same three-valued flag, handed in by
 * `AppShell` (which polls `/status`) so the effective-value fold below can
 * apply the capability half of §1-10's rule. It defaults to `null` = unknown,
 * which never disables anything — a caller that has no status at hand keeps
 * exactly the pre-§1-10 behaviour.
 */
export function useAccelerationSettings(
  blockSwapPrefetchAvailable: boolean | null = null,
): UseAccelerationSettingsResult {
  const [acceleration, setAcceleration] = useState<AccelerationSettings>(() => {
    const stored = readStoredAcceleration();
    return {
      attentionBackend: stored.attentionBackend,
      blockSwapPrefetch: stored.blockSwapPrefetch,
      keepResident: stored.keepResident,
      fusedGgufDequantKernel: stored.fusedGgufDequantKernel,
      vaeMode: stored.vaeMode,
    };
  });

  // Write-through happens here, not in the setters below — see the doc
  // comment above. Re-runs whenever any persisted field changes; a
  // StrictMode double-invoke just writes the same value twice (harmless).
  useEffect(() => {
    writeStoredAcceleration({
      attentionBackend: acceleration.attentionBackend,
      blockSwapPrefetch: acceleration.blockSwapPrefetch,
      keepResident: acceleration.keepResident,
      fusedGgufDequantKernel: acceleration.fusedGgufDequantKernel,
      vaeMode: acceleration.vaeMode,
    });
  }, [
    acceleration.attentionBackend,
    acceleration.blockSwapPrefetch,
    acceleration.keepResident,
    acceleration.fusedGgufDequantKernel,
    acceleration.vaeMode,
  ]);

  const setAttentionBackend = useCallback((value: AttentionBackend) => {
    setAcceleration((prev) => ({ ...prev, attentionBackend: value }));
  }, []);

  const setBlockSwapPrefetch = useCallback((value: boolean) => {
    setAcceleration((prev) => ({ ...prev, blockSwapPrefetch: value }));
  }, []);

  const setKeepResident = useCallback((value: boolean) => {
    setAcceleration((prev) => ({ ...prev, keepResident: value }));
  }, []);

  const setFusedGgufDequantKernel = useCallback((value: boolean) => {
    setAcceleration((prev) => ({ ...prev, fusedGgufDequantKernel: value }));
  }, []);

  const setVaeMode = useCallback((value: VaeMode) => {
    setAcceleration((prev) => ({ ...prev, vaeMode: value }));
  }, []);

  // §1-10: readers get the EFFECTIVE object, the state above stays the stored
  // one (so `localStorage` and the panel's own restore-on-return both keep
  // working). Memoized on the two inputs, and `effectiveAcceleration` returns
  // its argument unchanged in the common case, so the identity every consumer
  // depends on is as stable as the raw state was.
  const effective = useMemo(
    () => effectiveAcceleration(acceleration, blockSwapPrefetchAvailable),
    [acceleration, blockSwapPrefetchAvailable],
  );

  return {
    acceleration: effective,
    setAttentionBackend,
    setBlockSwapPrefetch,
    setKeepResident,
    setFusedGgufDequantKernel,
    setVaeMode,
  };
}
