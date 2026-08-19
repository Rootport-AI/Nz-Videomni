import { useMemo } from "react";
import type { AppConfig } from "../api/types";
import { resolveControlLoraNames, resolveDepthLoraNames, resolveReferenceDownscaleFactors } from "../lora/controlLoras";
import type { LorasState } from "../modes/inventory/useLoras";

/**
 * Memoizes `resolveControlLoraNames(config, lorasState)` (`lora/controlLoras.ts`)
 * against STABLE dependencies only — extracted out of `AppShell` (M-1,
 * post-implementation adversarial review, 2026-07-17) specifically so this
 * stability is independently unit-testable (`useControlLoraNames.test.ts`).
 *
 * `lorasState` itself is NOT referentially stable: `useLoras` returns
 * `{ ...state, refresh }` (`modes/inventory/useLoras.ts`), a brand-new object
 * literal on every render regardless of whether anything it holds actually
 * changed. Depending on `lorasState` directly here would recompute (and
 * return a brand-new `Set`) on EVERY render, which would in turn re-run
 * `AppShell`'s auto-migration effect (which depends on this hook's result) on
 * every keystroke — harmless today only because that effect is idempotent;
 * a future edit adding a non-idempotent step there would turn this into an
 * infinite loop. Depends instead on the pieces that are actually stable:
 * `lorasState.status` (a plain string) and the `loras` ARRAY itself once
 * ready — that array comes from `useLoras`'s own internal `useState` and
 * keeps its identity until the next successful `refresh()`, unlike the
 * wrapper object `useLoras` rebuilds around it every render.
 */
export function useControlLoraNames(config: AppConfig, lorasState: LorasState): ReadonlySet<string> {
  const lorasReady = lorasState.status === "ready" ? lorasState.loras : null;
  return useMemo(
    () => resolveControlLoraNames(config, lorasState),
    // eslint-disable-next-line react-hooks/exhaustive-deps -- `lorasState` is
    // read inside the factory but deliberately excluded from the deps array
    // (see the doc comment above); `lorasState.status`/`lorasReady` already
    // cover every way its CONTENT can actually change.
    [config, lorasState.status, lorasReady],
  );
}

/**
 * §1-15: the same memoization treatment for `resolveDepthLoraNames`
 * (`lora/controlLoras.ts`) — the depth-preprocess control adapters a MULTI-CLIP
 * chain cannot use. Handed to `ChainedScreen` -> `useChainForm`, whose
 * `depthChainUnsupported` gate pre-empts the server's 422.
 *
 * Depends on the same STABLE pieces as {@link useControlLoraNames} (see its doc
 * comment for why `lorasState` itself must never be a dependency); `config`
 * isn't one at all here, since `/config` carries no preprocess information.
 */
export function useDepthLoraNames(lorasState: LorasState): ReadonlySet<string> {
  const lorasReady = lorasState.status === "ready" ? lorasState.loras : null;
  return useMemo(
    () => resolveDepthLoraNames(lorasState),
    // eslint-disable-next-line react-hooks/exhaustive-deps -- same deliberate
    // exclusion of the unstable `lorasState` wrapper object as above.
    [lorasState.status, lorasReady],
  );
}

/**
 * §1-15, plan F5/F6: the same memoization treatment for
 * {@link resolveReferenceDownscaleFactors} — control-LoRA name ->
 * `reference_downscale_factor`, handed to `ChainedScreen` for its stage-1
 * comfort-budget warning banner (`shell/tokenBudget.ts`'s `chainStage1Tokens`).
 *
 * Depends on the same STABLE pieces as {@link useDepthLoraNames} (see
 * {@link useControlLoraNames}'s doc comment for why `lorasState` itself must
 * never be a dependency).
 */
export function useReferenceDownscaleFactors(lorasState: LorasState): ReadonlyMap<string, number> {
  const lorasReady = lorasState.status === "ready" ? lorasState.loras : null;
  return useMemo(
    () => resolveReferenceDownscaleFactors(lorasState),
    // eslint-disable-next-line react-hooks/exhaustive-deps -- same deliberate
    // exclusion of the unstable `lorasState` wrapper object as above.
    [lorasState.status, lorasReady],
  );
}
