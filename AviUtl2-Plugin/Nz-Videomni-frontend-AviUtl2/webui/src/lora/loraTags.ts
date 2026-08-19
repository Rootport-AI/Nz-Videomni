import type { LoraSpec } from "../api/types";

export type { LoraSpec };

/**
 * Prompt `<lora:name:strength>` tag parsing/formatting (Docs/API_REFERENCE.md
 * §5.3 `LoraSpec`: "既存UIはプロンプト内 `<lora:名前:強さ>` トークンをパース
 * して `loras` 配列に変換…この記法はフロント側の慣習であり、API自体は
 * `loras` 配列で受ける").
 *
 * The prompt string is the *single source of truth* for STYLE LoRAs —
 * `LoraChips` (`lora/LoraChips.tsx`) and the Library's style-card click
 * (`modes/inventory/InventoryScreen.tsx`) never hold separate chip state; they
 * read tags out of the current prompt and write a new prompt string back.
 * `useGenerationForm.toGenerateRequest` (`modes/single/useGenerationForm.ts`)
 * calls `parseLoraPrompt` at submission time to split the prompt into the
 * body the backend actually renders (`strippedPrompt`) and the `loras[]`
 * array (Docs/API_REFERENCE.md §5.1) — the API itself never sees tag syntax.
 *
 * IC-LoRA UI redesign (2026-07-17, 第5波): the CONTROL LoRA (the one
 * conditioning off a reference video) is deliberately NOT part of this tag
 * channel any more — it's a state-owned single selection (`lora/controlLoras.ts`'s
 * `ControlLoraSelection`, owned by `shell/AppShell.tsx`) merged into `loras[]`
 * at submission time via `combineLoras`. A hand-typed control tag is treated
 * as a one-shot input to migrate INTO that state (see `AppShell`'s
 * auto-migration effect and {@link stripLoraTagsByName} below), not a
 * standing alternate representation of it.
 */

export interface ParsedLoraTag extends LoraSpec {
  /** Exact substring matched, e.g. `"<lora:Foo:1.5>"` — identifies this
   * specific occurrence for replace/remove (two tags can share the same
   * `name`+`strength` and thus the same `raw` text; `index` disambiguates). */
  raw: string;
  /** Index into the prompt string passed to `parseLoraPrompt` where `raw`
   * starts. */
  index: number;
  /** False when `name` contains `/`, `\`, or `..` (Docs/API_REFERENCE.md
   * §5.3: "`/`・`\`・`..` を含む名前は拒否"). An invalid tag is left
   * untouched in the prompt body: it's never stripped, never becomes a
   * `loras[]` entry, and never renders as a chip. */
  valid: boolean;
}

export interface ParsedLoraPrompt {
  /** `prompt` with every *valid* tag removed and the whitespace it leaves
   * behind collapsed. Strictly `=== prompt` (same string) when no valid tag
   * was found, so round-tripping a tag-free prompt through this function
   * never perturbs it (byte-for-byte, not just "equal"). */
  strippedPrompt: string;
  /** Valid tags only, ready to hand to `GenerateRequest.loras`. */
  loras: LoraSpec[];
  /** Every tag found, valid or not — `LoraChips` renders only the valid
   * ones (`tags.filter(t => t.valid)`). */
  tags: ParsedLoraTag[];
}

export const LORA_STRENGTH_MIN = 0.05;
export const LORA_STRENGTH_MAX = 2.0;
export const LORA_STRENGTH_STEP = 0.1;
export const LORA_STRENGTH_DEFAULT = 1.0;

/** Audio-side strength range (Docs/API_REFERENCE.md §5.3 `LoraSpec.audio_strength`).
 * Unlike {@link LORA_STRENGTH_MIN}, `0` IS a valid floor here — it means
 * "apply no audio-side weight at all" (skip the audio-side LoRA delta),
 * distinct from the field being omitted (which means "follow the video-side
 * `strength`"). `clampLoraStrength`'s 0.05 floor must never be reused for
 * this axis. */
export const LORA_AUDIO_STRENGTH_MIN = 0.0;
export const LORA_AUDIO_STRENGTH_MAX = 2.0;
export const LORA_AUDIO_STRENGTH_DEFAULT = 1.0;

/** Matches `<lora:name>`, `<lora:name:strength>`, or
 * `<lora:name:strength:audioStrength>`, case-insensitively. `name` stops at
 * the first `:` or `>`; `strength` (2nd group) also stops at the next `:` or
 * `>` now that a 3rd group can follow it; `audioStrength` (3rd group), when
 * present, is whatever text follows up to `>`. Both strength segments are
 * validated/clamped in `normalizeStrength`/`normalizeAudioStrength`, not by
 * the regex — an unparseable 3rd group (e.g. `<lora:A:1.0:junk>`) still
 * matches the tag as a whole, it just yields no `audio_strength` (see
 * {@link normalizeAudioStrength}). */
const LORA_TAG_RE = /<lora:([^:>]+)(?::([^:>]*))?(?::([^>]*))?>/gi;

/** Docs/API_REFERENCE.md §5.3: names containing a path separator or `..`
 * are rejected — the WebUI never lets a tag reach the backend as a path. */
const INVALID_NAME_RE = /[\\/]|\.\./;

export function clampLoraStrength(value: number): number {
  return Math.min(LORA_STRENGTH_MAX, Math.max(LORA_STRENGTH_MIN, value));
}

/** Clamps an audio-side strength to `[0.0, 2.0]` — NOT `clampLoraStrength`'s
 * 0.05 floor, since `0` is a valid, distinct audio-side value (mute). */
export function clampLoraAudioStrength(value: number): number {
  return Math.min(LORA_AUDIO_STRENGTH_MAX, Math.max(LORA_AUDIO_STRENGTH_MIN, value));
}

export function isValidLoraName(name: string): boolean {
  return name.length > 0 && !INVALID_NAME_RE.test(name);
}

/** Formats a strength value back into tag text: one decimal place for the
 * common case (default 1.0, ±0.1 chip steps), two only when that would lose
 * precision — reachable only at the 0.05 clamp floor. */
export function formatLoraStrength(value: number): string {
  const rounded = Math.round(value * 100) / 100;
  const oneDecimal = Math.round(rounded * 10) / 10;
  return Math.abs(rounded - oneDecimal) < 1e-9 ? rounded.toFixed(1) : rounded.toFixed(2);
}

function normalizeStrength(raw: string | undefined): number {
  if (raw === undefined || raw.trim() === "") return LORA_STRENGTH_DEFAULT;
  const parsed = Number(raw);
  return clampLoraStrength(Number.isFinite(parsed) ? parsed : LORA_STRENGTH_DEFAULT);
}

/** Normalizes the tag's optional 3rd (audio-strength) segment. Unlike
 * {@link normalizeStrength}, an absent/blank/unparseable segment yields
 * `undefined` — NOT a default value — so the resulting `LoraSpec` simply
 * omits `audio_strength` (follows the video-side strength) rather than
 * pinning it to some fallback number. A present-but-unparseable segment
 * (e.g. `<lora:A:1.0:junk>`) is treated the same as absent: the tag as a
 * whole still matches and strips, it just carries no audio strength. */
function normalizeAudioStrength(raw: string | undefined): number | undefined {
  if (raw === undefined || raw.trim() === "") return undefined;
  const parsed = Number(raw);
  if (!Number.isFinite(parsed)) return undefined;
  return clampLoraAudioStrength(parsed);
}

/** Collapses runs of spaces/tabs left behind by a removed tag, and trims the
 * ends — but never touches newlines beyond trimming surrounding
 * spaces/tabs, since the prompt textarea allows multi-line input. */
function collapseWhitespace(value: string): string {
  return value
    .replace(/[ \t]+/g, " ")
    .replace(/[ \t]*\n[ \t]*/g, "\n")
    .trim();
}

/** Parses every `<lora:...>` occurrence out of `prompt`. See
 * `ParsedLoraPrompt` for the three views this returns. */
export function parseLoraPrompt(prompt: string): ParsedLoraPrompt {
  const loras: LoraSpec[] = [];
  const tags: ParsedLoraTag[] = [];
  let anyStripped = false;

  const strippedRaw = prompt.replace(
    LORA_TAG_RE,
    // BLOCKER fix: the regex now has 3 capture groups (name/strength/audio),
    // so `offset` — the position `full` starts at, which `tag.index` (and
    // therefore every chip ±/× button's edit target) is built from — is now
    // the REPLACE CALLBACK'S 5TH ARGUMENT, not its 4th. Getting this wrong
    // silently shifts every `tag.index` by one capture group's worth of
    // params, which makes `setLoraTagStrength`/`removeLoraTag` edit the wrong
    // tag (see `loraTags.test.ts`'s multi-tag `tag.index` coverage).
    (full: string, rawName: string, rawStrength: string | undefined, rawAudio: string | undefined, offset: number) => {
      const name = rawName.trim();
      const strength = normalizeStrength(rawStrength);
      const audioStrength = normalizeAudioStrength(rawAudio);
      const valid = isValidLoraName(name);
      tags.push({
        name,
        strength,
        ...(audioStrength !== undefined ? { audio_strength: audioStrength } : {}),
        raw: full,
        index: offset,
        valid,
      });
      if (!valid) return full;
      anyStripped = true;
      loras.push({ name, strength, ...(audioStrength !== undefined ? { audio_strength: audioStrength } : {}) });
      return "";
    },
  );

  return {
    strippedPrompt: anyStripped ? collapseWhitespace(strippedRaw) : prompt,
    loras,
    tags,
  };
}

/** Formats a tag's text: `<lora:name:strength>` when `audioStrength` is
 * `undefined` (video-only — the tag's original 2-argument shape), or
 * `<lora:name:strength:audioStrength>` when it's a number. The video-side
 * strength is ALWAYS written out explicitly whenever an audio strength is
 * present — this never produces `<lora:name::audio>` (an empty 2nd
 * position), which Gradio's own tag parser does not accept (asymmetric by
 * design, Docs/API_REFERENCE.md §5.3). The single formatting choke point for
 * every tag-writing call site (`appendLoraTag`/`setLoraTagStrength`/
 * `setLoraTagAudioStrength`), so none of them can accidentally drop a 3rd
 * argument a caller already set. */
export function formatLoraTag(name: string, strength: number, audioStrength?: number): string {
  const strengthText = formatLoraStrength(strength);
  if (audioStrength === undefined) return `<lora:${name}:${strengthText}>`;
  return `<lora:${name}:${strengthText}:${formatLoraStrength(audioStrength)}>`;
}

/** Appends `<lora:name:1.0:1.0>` to `prompt` (Library style-card click)
 * unless a valid tag for that exact name is already present. The 3-argument
 * default (owner decision: "カードクリック挿入は3引数") surfaces the
 * audio-strength slot from the very first insertion so users notice the
 * feature exists; audio `1.0` is numerically identical to the video-follow
 * default, so this doesn't change the generated output on its own. Returns
 * `prompt` unchanged (same string) when it's a no-op, so callers can test
 * `result === prompt` to decide whether to show an "added" toast. */
export function appendLoraTag(prompt: string, name: string): string {
  const { loras } = parseLoraPrompt(prompt);
  if (loras.some((l) => l.name === name)) return prompt;
  const tag = formatLoraTag(name, LORA_STRENGTH_DEFAULT, LORA_AUDIO_STRENGTH_DEFAULT);
  if (prompt.length === 0) return tag;
  const separator = /\s$/.test(prompt) ? "" : " ";
  return `${prompt}${separator}${tag}`;
}

/** Rewrites one specific tag occurrence (matched by its exact
 * `index`/`raw`, as returned in `ParsedLoraPrompt.tags`) to a new strength —
 * a chip's −/+ buttons. Routed through {@link formatLoraTag} with the tag's
 * OWN `audio_strength` passed straight through, so adjusting the video-side
 * strength never silently drops an existing 3rd argument. */
export function setLoraTagStrength(prompt: string, tag: ParsedLoraTag, newStrength: number): string {
  const strength = clampLoraStrength(newStrength);
  const replacement = formatLoraTag(tag.name, strength, tag.audio_strength);
  return prompt.slice(0, tag.index) + replacement + prompt.slice(tag.index + tag.raw.length);
}

/** Rewrites one specific tag occurrence's audio-side strength — the mute
 * toggle's underlying primitive (`LoraChips.tsx`). `value === undefined`
 * removes the 3rd argument entirely (audio strength reverts to following the
 * video-side `strength`); any number is clamped to
 * `[LORA_AUDIO_STRENGTH_MIN, LORA_AUDIO_STRENGTH_MAX]` and written explicitly
 * (see {@link formatLoraTag}'s "never `::`" guarantee). The video-side
 * strength is left exactly as-is. */
export function setLoraTagAudioStrength(prompt: string, tag: ParsedLoraTag, value: number | undefined): string {
  const audioStrength = value === undefined ? undefined : clampLoraAudioStrength(value);
  const replacement = formatLoraTag(tag.name, tag.strength, audioStrength);
  return prompt.slice(0, tag.index) + replacement + prompt.slice(tag.index + tag.raw.length);
}

/** Removes one specific tag occurrence entirely (a chip's × button), then
 * collapses the whitespace left behind. */
export function removeLoraTag(prompt: string, tag: ParsedLoraTag): string {
  const removed = prompt.slice(0, tag.index) + prompt.slice(tag.index + tag.raw.length);
  return collapseWhitespace(removed);
}

/** Removes every *valid* `<lora:name:strength>` tag whose `name` is in
 * `names`, collapsing the whitespace each removal leaves behind — the IC-LoRA
 * UI redesign's auto-migration effect (`shell/AppShell.tsx`) calls this with
 * the control-LoRA name set right after moving a hand-typed control tag into
 * the panel's own state, so the tag doesn't linger in the prompt body
 * alongside its new home. Strictly `=== prompt` (same string, no-op) when
 * nothing matched, mirroring {@link parseLoraPrompt}'s own "untouched when
 * there's nothing to strip" contract — callers rely on this to skip a
 * redundant `setPrompt` call. */
export function stripLoraTagsByName(prompt: string, names: ReadonlySet<string>): string {
  if (names.size === 0) return prompt;
  let anyStripped = false;
  const stripped = prompt.replace(LORA_TAG_RE, (full: string, rawName: string) => {
    const name = rawName.trim();
    if (!isValidLoraName(name) || !names.has(name)) return full;
    anyStripped = true;
    return "";
  });
  return anyStripped ? collapseWhitespace(stripped) : prompt;
}
