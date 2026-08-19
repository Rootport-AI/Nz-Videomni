import type { AppConfig, LoraSpec } from "../api/types";
import type { LorasState } from "../modes/inventory/useLoras";

/**
 * IC-LoRA UI redesign (2026-07-17, 第5波): the control (IC-LoRA) selection is
 * a state-owned single choice — a name plus a strength — rendered as a
 * "none + selection-preserving dropdown + weight slider" panel, no longer a
 * `<lora:name:strength>` prompt tag (that channel stays reserved for STYLE
 * LoRAs — see `lora/loraTags.ts`'s own doc comment). This module is the pure,
 * framework-agnostic layer backing that panel: which names are selectable,
 * and how the selection merges into the `loras[]` array the backend actually
 * receives. `shell/AppShell.tsx` owns the actual `controlLora` state (see its
 * doc comment for why); `modes/single/useGenerationForm.ts` consumes it via
 * injected options.
 */

/** The panel's current selection: `null` = "none" (the dropdown's first
 * option). A single control LoRA at a time — the backend rejects more than
 * one via `lora_preprocess_conflict` (422), and it reads
 * `reference_downscale_factor`/preprocess config off the FIRST `loras[]`
 * entry, so "single, always-first" is a structural requirement, not just a
 * UI simplification. */
export interface ControlLoraSelection {
  name: string;
  strength: number;
}

/**
 * The control (IC-)LoRA the Outpainting panel pins. Registered server-side
 * under this exact name (`config.yaml`'s `ic_loras`) and reported by
 * `GET /loras` with `kind: "control"`. Lives here (not in
 * `modes/edit/useOutpaintForm.ts`, where it originated) so both this module's
 * {@link UI_HIDDEN_CONTROL_LORA_NAMES} and the Edit form can reference the
 * same literal without either importing the other's runtime module — `lora/`
 * only takes TYPE-ONLY imports from `modes/`, and `useOutpaintForm.ts` pulls
 * in React/bridge machinery a value-import from here would otherwise be
 * forced to drag along.
 */
export const OUTPAINT_LORA_NAME = "in-outpainting";

/**
 * Control-LoRA names that exist server-side and are real, selectable
 * `loras[]` entries — but must NOT appear in the Create/Chain control-LoRA
 * DROPDOWNS. `in-outpainting` is the only member: it has its own dedicated
 * place, the Edit tab's Outpainting panel, which pins it automatically and
 * always sends it — picking it from Create/Chain's generic dropdown would
 * produce a request that never sets `outpaint`, which the server silently
 * accepts and turns into a normal (non-outpaint) generation with a control
 * adapter that does nothing useful there. Depth-preprocess adapters get the
 * opposite treatment (stay in the dropdown, blocked at generate time instead
 * — see `strings.ts`'s `chainReferenceControlLoraDepthBlocked` and
 * `resolveDepthLoraNames`'s own doc comment): they have no dedicated place of
 * their own, so removing them from the list would leave no way to reach them
 * at all. `in-outpainting` does, so it is the one name excluded here instead.
 *
 * Deliberately NOT used by {@link resolveControlLoraNames} itself — see that
 * function's doc comment for why the raw (unfiltered) set must stay raw.
 */
export const UI_HIDDEN_CONTROL_LORA_NAMES: ReadonlySet<string> = new Set([OUTPAINT_LORA_NAME]);

/**
 * Narrows a control-LoRA name set down to the ones a dropdown should actually
 * OFFER, by removing {@link UI_HIDDEN_CONTROL_LORA_NAMES}. Call this ONLY at
 * the two dropdown render sites (`modes/single/GenerationForm.tsx`'s
 * `ReferenceVideoSection`, `modes/chained/ChainReferencePanel.tsx`) — every
 * other consumer of `resolveControlLoraNames`'s output (the reference-required
 * gate, `AppShell`'s hand-typed-tag auto-migration, the chip-suppression
 * check) needs the FULL set, because a hand-typed `<lora:in-outpainting:1.0>`
 * tag on Create must still be caught and blocked, not silently treated as an
 * unknown name.
 */
export function selectableControlLoraNames(names: ReadonlySet<string>): ReadonlySet<string> {
  if (![...names].some((name) => UI_HIDDEN_CONTROL_LORA_NAMES.has(name))) return names;
  return new Set([...names].filter((name) => !UI_HIDDEN_CONTROL_LORA_NAMES.has(name)));
}

/** Resolves the full set of control-LoRA names the server knows about. Prefers
 * `GET /loras`' own `kind === "control"` entries (the authoritative,
 * live-scanned source — covers anything in `config.model.lora_dir`, not just
 * what `/config` happened to enumerate); falls back to `config.model.ic_loras`'
 * keys only while `lorasState` hasn't resolved yet (`"loading"`) or failed
 * (`"error"`) — matching the pre-existing N2 behavior so a `/loras` outage
 * doesn't blank the dropdown outright, just narrows it to whatever `/config`
 * already knew about.
 *
 * Despite the name, this is NOT "the dropdown's options" — it deliberately
 * stays unfiltered (includes `in-outpainting`). `controlLoraNeedsReference`'s
 * gate, `AppShell`'s hand-typed-tag auto-migration, and the chip-suppression
 * check all key off this exact set and need `in-outpainting` present to keep
 * blocking/catching it correctly. Only the two dropdown render sites should
 * narrow it further, via {@link selectableControlLoraNames}. */
export function resolveControlLoraNames(config: AppConfig, lorasState: LorasState): ReadonlySet<string> {
  if (lorasState.status === "ready") {
    return new Set(lorasState.loras.filter((lora) => lora.kind === "control").map((lora) => lora.name));
  }
  return new Set(Object.keys(config.model?.ic_loras ?? {}));
}

/** §1-15 (chain reference video): the subset of the control LoRAs whose frame
 * PREPROCESS is `"depth"` — the one adapter family a MULTI-CLIP chain cannot
 * run (the depth estimator is a whole-video sequential pass with a global
 * normalization, so a chain-length reference would need tens of GB; the server
 * 422s the combination). The Chain form uses this to pre-empt that 422 with a
 * plain-language block reason.
 *
 * Deliberately narrow, and deliberately SILENT on missing data: the
 * `preprocess` field is new on `GET /loras` (`LoraEntry.preprocess`), so an
 * older server returns entries without it — those simply aren't in the set, and
 * the gate never fires. Same "only while `/loras` actually resolved" rule as
 * {@link resolveControlLoraNames}, minus its `config.model.ic_loras` fallback:
 * `/config` carries no preprocess information at all, so there is nothing to
 * fall back TO. */
export function resolveDepthLoraNames(lorasState: LorasState): ReadonlySet<string> {
  if (lorasState.status !== "ready") return new Set();
  return new Set(
    lorasState.loras.filter((lora) => lora.kind === "control" && lora.preprocess === "depth").map((lora) => lora.name),
  );
}

/** §1-15 (clip-wise IC-LoRA reference), plan F5/F6: control-LoRA name ->
 * `reference_downscale_factor`, for the Chain screen's stage-1 comfort-budget
 * warning (`shell/tokenBudget.ts`'s `chainStage1Tokens`' `refScale`
 * parameter). Only entries with a real POSITIVE number are included — a
 * `null`/absent/non-positive factor (an older server, an unreadable header, a
 * style LoRA) simply isn't in the map, and the caller falls back to `2` (the
 * union-control adapters' own factor), matching `reference_video_cond.py`'s
 * own default.
 *
 * Same "only while `/loras` actually resolved, no `/config` fallback" rule as
 * {@link resolveDepthLoraNames} — `/config` carries no downscale-factor
 * information to fall back to. */
export function resolveReferenceDownscaleFactors(lorasState: LorasState): ReadonlyMap<string, number> {
  if (lorasState.status !== "ready") return new Map();
  const result = new Map<string, number>();
  for (const lora of lorasState.loras) {
    if (lora.kind === "control" && typeof lora.reference_downscale_factor === "number" && lora.reference_downscale_factor > 0) {
      result.set(lora.name, lora.reference_downscale_factor);
    }
  }
  return result;
}

/** Assembles the final `loras[]` sent to the backend from the panel's
 * `controlLora` selection plus the prompt's own STYLE `<lora:...>` tags —
 * pure mirror of Gradio's own merge (`gradio_ui/handlers.py`'s
 * `_combine_generate_loras`): the control LoRA goes first (the engine reads
 * `reference_downscale_factor`/preprocess config off `loras[0]`, so its
 * position is a hard requirement, not cosmetic), the prompt's tags follow in
 * their existing order, and any name collision is resolved by keeping the
 * FIRST occurrence's position but the LAST occurrence's strength (a
 * `Map`-based dedupe, not a filter) — so a style tag that happens to share a
 * name with the selected control LoRA doesn't produce two `loras[]` entries
 * for the same name, which the backend would also reject. */
export function combineLoras(controlLora: ControlLoraSelection | null, promptLoras: readonly LoraSpec[]): LoraSpec[] {
  const ordered: LoraSpec[] = controlLora
    ? [{ name: controlLora.name, strength: controlLora.strength }, ...promptLoras]
    : [...promptLoras];

  const indexByName = new Map<string, number>();
  const result: LoraSpec[] = [];
  for (const spec of ordered) {
    const existingIndex = indexByName.get(spec.name);
    if (existingIndex === undefined) {
      indexByName.set(spec.name, result.length);
      result.push(spec);
    } else {
      result[existingIndex] = spec;
    }
  }
  return result;
}
