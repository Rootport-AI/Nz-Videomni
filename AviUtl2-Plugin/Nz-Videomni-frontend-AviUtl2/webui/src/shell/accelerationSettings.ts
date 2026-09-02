/**
 * Acceleration (生成高速化) settings — the Settings-panel section that picks
 * which attention implementation the backend runs a job with (2026-07-31,
 * backend §43), whether it prefetches block-swap transfers (2026-08-01,
 * backend §44), whether it keeps the models' CPU-side skeleton resident
 * between jobs (2026-08-02, backend §48), and whether GGUF dequantization runs
 * as one fused Triton kernel (2026-08-04, backend §51), which VAE decoder
 * reconstructs the video (2026-08-05, backend §52 — PrunaVAED), and whether
 * LTX 2.5's embeddings processor stays resident between jobs (2026-09-03,
 * 台帳 §3-114). Shared by
 * Create/Chain/Batch via a
 * single `AccelerationSettings` object (D1: `AppShell` useState + props, no
 * Context — exactly the arrangement `shell/nagSettings.ts` documents and every
 * other cross-screen setting here follows; see
 * `shell/useAccelerationSettings.ts`). This module is the pure, UI-free layer:
 * the types, the constants, the `localStorage` read/write for the chosen
 * backend, and the additive request-field contract.
 *
 * Six items exist in the UI, and ALL SIX are implemented — the fifth
 * (`vaeMode`) was the last mock, which backend §52 turned real on 2026-08-05,
 * and the sixth (`keepResidentEmbeddings`) was born real on 2026-09-03:
 *  - `attentionBackend` — REAL. `sdpa` (PyTorch's own scaled-dot-product
 *    attention, the server default) vs. `sage` (SageAttention 2.2.0, ~1.2-1.6x
 *    faster with no extra VRAM). Sent as `attention_backend`.
 *  - `blockSwapPrefetch` — REAL (2026-08-01, backend §44). Async block-swap
 *    prefetch: hides the block-swap CPU<->GPU transfer behind compute on a
 *    separate CUDA stream. Unlike `attentionBackend`, the output is bit-for-
 *    bit identical on/off (only the transfer method changes), so no "this
 *    changes your frames" warning applies. Sent as `block_swap_prefetch`.
 *  - `keepResident` — REAL (2026-08-02, backend §48). Keeps the models'
 *    CPU-side skeleton (the `StateDictRegistry`) alive between jobs, so the
 *    second and later generations skip the re-materialization work (measured
 *    ~70s -> ~10s of preprocessing). Costs roughly 20GB of resident main
 *    memory, hence the "64GB or more recommended" note. Like
 *    `blockSwapPrefetch` (and unlike `attentionBackend`) the output is
 *    bit-for-bit identical on/off. Sent as `keep_resident`; the SERVER
 *    default is `false`, so the key rides along only while it is ON — and,
 *    since §1-10 (2026-08-03), only while block-swap prefetch is effectively
 *    on as well, since the pair is what makes it pay off (see
 *    {@link keepResidentEffective}).
 *  - `fusedGgufDequantKernel` — REAL (2026-08-04, backend §51, ledger §1-11).
 *    Runs the GGUF K-quant (Q4_K/Q5_K/Q6_K) dequantization as ONE fused Triton
 *    kernel instead of the pure-PyTorch multi-tensor path (measured ~21.8s of
 *    GPU time per job). Bit-for-bit identical output, same as
 *    `blockSwapPrefetch`/`keepResident` — the backend silently falls back to
 *    the old implementation on any failure and records
 *    `fused_gguf_dequant_kernel_used: "on->off"`. Sent as
 *    `fused_gguf_dequant_kernel`; the SERVER default is `true`, so the key
 *    rides along only while it is OFF. Independent of every other row here —
 *    no availability gate (the server does not publish one, same discipline as
 *    `keep_resident`) and no dependency on prefetch. This row REPLACED the old
 *    `fusedGgufDequantGemm` mock, which was removed outright (owner ruling
 *    2026-08-04; the underlying no-go was `Docs/PENDING_TASKS_CLOSED.md`
 *    §3-61 / old §3-49).
 *  - `vaeMode` — REAL (2026-08-05, backend §52). `"prune_vaed"` swaps the
 *    video VAE's decoder for PrunaVAED, a pruned decoder (~690MB vs. ~814MB)
 *    that reconstructs the video faster. UNLIKE the three bit-identical
 *    toggles above, and LIKE `sage`, this one changes the pixels: it is a
 *    different decoder, so output quality may be slightly reduced — that is
 *    the note the panel shows while it is selected. Sent as `vae_mode`; the
 *    server default is `"default"`, so the key rides along only while
 *    PrunaVAED is chosen. Independent of every other row (no `/status`
 *    capability flag — the backend degrades per job to `vae_mode_used:
 *    "on->off"` if the pruned weights are missing). `vaeMode` is unrelated to
 *    the server's existing `vae_tiling` VRAM option.
 *  - `keepResidentEmbeddings` — REAL (2026-09-03, 台帳 §3-114). Keeps LTX 2.5's
 *    EMBEDDINGS PROCESSOR (the component that shapes the text encoder's output
 *    before it reaches the transformer) resident between jobs, so its GGUF is
 *    not re-read every single time. Bit-identical output — the same part is
 *    reused rather than rebuilt — at a cost of roughly 5GB of resident main
 *    memory. Sent as `keep_resident_embeddings`; the SERVER default is `false`,
 *    so the key rides along only while it is ON, exactly like `keep_resident`.
 *    It is NOT gated on prefetch the way `keep_resident` is (a different engine
 *    and a different mechanism — there is no pairing to honour), and it has no
 *    `/status` capability flag for the same reason `keep_resident` has none:
 *    whether it is worth turning on is a question about this machine's RAM,
 *    which the server cannot answer.
 *
 *    THE DIRECTION OF ITS SCOPE IS THE OPPOSITE OF EVERY ROW ABOVE, and that is
 *    the one thing worth remembering about it: the embeddings processor exists
 *    only on LTX 2.5, so it is LTX 2.3 that publishes `keep_resident_embeddings`
 *    in `unsupported_features` and 422s a `true`. `AppShell` hides the row (and
 *    writes a leftover `true` back to `false`) on any engine that publishes the
 *    name — the same treatment `prune_vaed` gets, just pointing the other way.
 */
import type { StatusResponse } from "../api/types";

/** Which attention implementation the backend should use for a job. Mirrors
 * the backend's `api/models.py` `attention_backend` literal as of 2026-07-31
 * — if the backend's set of values ever changes, update both sides together. */
export type AttentionBackend = "sdpa" | "sage";

/** Which VAE decode path a job runs with. Mirrors the backend's
 * `api/models.py` `vae_mode` literal as of 2026-08-05 (backend §52) — if the
 * backend's set of values ever changes, update both sides together. The API
 * VALUE stays `"prune_vaed"` even though the display name settled on
 * "PrunaVAED": it is an external contract. */
export type VaeMode = "default" | "prune_vaed";

/** The Acceleration section's full state — one object shared by
 * Create/Chain/Batch, owned by `AppShell`. */
export interface AccelerationSettings {
  attentionBackend: AttentionBackend;
  /** REAL (2026-08-01): hides the block-swap CPU<->GPU transfer behind
   * compute on a separate stream. Unlike `sage`, the output is BIT-IDENTICAL
   * (only the transfer method changes), so the same seed reproduces the same
   * frames exactly — no "sage-style" note is warranted. Sent as
   * `block_swap_prefetch`. */
  blockSwapPrefetch: boolean;
  /** REAL (2026-08-02): keeps the models' CPU-side skeleton resident between
   * jobs so the second and later generations skip re-materialization
   * (~70s -> ~10s of preprocessing) at the cost of ~20GB of resident main
   * memory. Bit-identical output, same as {@link blockSwapPrefetch}. Sent as
   * `keep_resident` — but only while ON, since the server defaults to
   * `false`. */
  keepResident: boolean;
  /** REAL (2026-08-04, backend §51): runs GGUF K-quant dequantization as one
   * fused Triton kernel. Bit-identical output, same as {@link keepResident};
   * an INDEPENDENT toggle — it is gated on nothing (no `/status` capability
   * flag exists for it, and it has no relationship to prefetch). Sent as
   * `fused_gguf_dequant_kernel` — but only while OFF, since the server
   * defaults to `true`. */
  fusedGgufDequantKernel: boolean;
  /** REAL (2026-08-05, backend §52): which VAE decoder reconstructs the video.
   * `"prune_vaed"` picks PrunaVAED, the pruned decoder. Unlike the three
   * bit-identical toggles above (and like `sage`) it CHANGES the output —
   * quality may be slightly reduced — so the panel shows a note while it is
   * selected. Sent as `vae_mode`, and only while it is off the server default
   * (`"default"`). */
  vaeMode: VaeMode;
  /** REAL (2026-09-03, 台帳 §3-114): keeps LTX 2.5's embeddings processor
   * resident between jobs. Bit-identical output, same as {@link keepResident},
   * and INDEPENDENT of it — the two are separate switches over two different
   * objects, so their RAM costs add rather than overlap. Sent as
   * `keep_resident_embeddings` — but only while ON, since the server defaults
   * to `false`. The one row here whose SUPPORT points the other way: LTX 2.3
   * has no embeddings processor, so 2.3 is the engine that refuses it. */
  keepResidentEmbeddings: boolean;
}

/** Mirrors the backend's `api/models.py` `Field(...)` defaults as of
 * 2026-07-31 (`"sdpa"` / `False` / `"default"`) — if the backend's defaults
 * ever change, update both sides together. `ATTENTION_BACKEND_DEFAULT` is
 * load-bearing beyond seeding state: it is the value
 * {@link accelerationRequestFields} treats as "send nothing at all". */
export const ATTENTION_BACKEND_DEFAULT: AttentionBackend = "sdpa";
/** Server default for `block_swap_prefetch` as of backend §44 (2026-08-01).
 * S4 (backend, post real-device gate) flipped the SERVER's own default to
 * `true` — the bit-identity + VRAM gate (G1-G7) passed and the owner
 * confirmed "gate green -> default on". This constant is flipped to `true`
 * in the SAME change — it is what {@link accelerationRequestFields} treats
 * as "send nothing at all", so leaving it `false` after the server flips
 * would make every request explicitly send `block_swap_prefetch: false` to
 * a server that already defaults to `true` (turning every caller off by
 * accident). */
export const BLOCK_SWAP_PREFETCH_SERVER_DEFAULT = true;
/** Server default for `keep_resident` as of backend §48 (2026-08-02):
 * `false`. Deliberately NOT flipped on the way `BLOCK_SWAP_PREFETCH_SERVER_DEFAULT`
 * was — keeping the skeleton resident costs ~20GB of main memory, so it is
 * opt-in on a machine the owner judges to have the RAM for it (the panel's
 * note says "64GB or more recommended"). Being `false` also flips the
 * direction of {@link accelerationRequestFields} relative to block-swap
 * prefetch: `keep_resident` rides along only when the user turns it ON. */
export const KEEP_RESIDENT_SERVER_DEFAULT = false;
/** Server default for `fused_gguf_dequant_kernel` as of backend §51
 * (2026-08-04): `true`. The backend flipped its own default to `true` after
 * real-device gates G1-G8 passed (bit-identical output, ~17.5% faster) and the
 * owner approved — exactly the `BLOCK_SWAP_PREFETCH_SERVER_DEFAULT` story, and
 * this constant is flipped in the SAME change for the same reason: it is what
 * {@link accelerationRequestFields} treats as "send nothing at all", so leaving
 * it `false` would make every request explicitly send
 * `fused_gguf_dequant_kernel: false` and turn every caller off by accident.
 * Direction is therefore the OPPOSITE of `KEEP_RESIDENT_SERVER_DEFAULT`: the
 * key rides along only when the user turns it OFF. */
export const FUSED_GGUF_DEQUANT_KERNEL_SERVER_DEFAULT = true;
/** Server default for `vae_mode` as of backend §52 (2026-08-05): `"default"`
 * (the unmodified decoder). Deliberately NOT flipped the way
 * `BLOCK_SWAP_PREFETCH_SERVER_DEFAULT` and
 * `FUSED_GGUF_DEQUANT_KERNEL_SERVER_DEFAULT` were, and for the reason that
 * separates this row from those two: PrunaVAED is a DIFFERENT decoder, so the
 * output is not bit-identical and quality may be slightly reduced. A speed
 * knob that silently changes everyone's pixels is opt-in. Direction is
 * therefore the same as `KEEP_RESIDENT_SERVER_DEFAULT`: the key rides along
 * only when the user turns it ON. */
export const VAE_MODE_DEFAULT: VaeMode = "default";
/** Server default for `keep_resident_embeddings` as of 台帳 §3-114
 * (2026-09-03): `false`. Not flipped, and for `KEEP_RESIDENT_SERVER_DEFAULT`'s
 * exact reason — keeping the embeddings processor resident costs roughly 5GB of
 * main memory, so it is opt-in on a machine the owner judges to have the RAM
 * for it. Direction therefore matches `keep_resident`: the key rides along only
 * when the user turns it ON — which is also what keeps an LTX 2.3 request out
 * of the 422, since the engine that refuses this field refuses a `true` and
 * ignores an absent key (see
 * {@link AccelerationSettings.keepResidentEmbeddings}). */
export const KEEP_RESIDENT_EMBEDDINGS_SERVER_DEFAULT = false;

/** `localStorage` key every Acceleration choice is persisted under — all FIVE
 * as of 2026-08-05, when `vaeMode` stopped being a mock. The key is named for the section
 * rather than the field so a further persisted knob doesn't force a
 * storage-key rename on existing installs — as of 2026-08-05 that promise has
 * been cashed four times. Stored as JSON as of 2026-08-01 (was the BARE backend
 * string before {@link StoredAcceleration} gained a second field) —
 * {@link readStoredAcceleration} still accepts the old bare form, matching
 * this comment's original migration promise. */
export const ACCELERATION_STORAGE_KEY = "nzvideomni.acceleration";

/** The frozen "everything at its backend default" sentinel every form hook
 * defaults `acceleration` to when its caller omits it — mirrors
 * `nagSettings.ts`'s `NAG_OFF` (including the `Object.freeze`, which catches
 * an accidental in-place mutation of this shared reference immediately). Feeds
 * {@link accelerationRequestFields} `{}`, so an omitted `acceleration` keeps
 * every request byte-identical to before this feature existed. */
export const ACCELERATION_DEFAULTS: Readonly<AccelerationSettings> = Object.freeze({
  attentionBackend: ATTENTION_BACKEND_DEFAULT,
  blockSwapPrefetch: BLOCK_SWAP_PREFETCH_SERVER_DEFAULT,
  keepResident: KEEP_RESIDENT_SERVER_DEFAULT,
  fusedGgufDequantKernel: FUSED_GGUF_DEQUANT_KERNEL_SERVER_DEFAULT,
  vaeMode: VAE_MODE_DEFAULT,
  keepResidentEmbeddings: KEEP_RESIDENT_EMBEDDINGS_SERVER_DEFAULT,
});

/** The persisted subset of {@link AccelerationSettings} — as of 2026-08-05
 * (backend §52) that is the WHOLE of it: `vaeMode` was the last unpersisted
 * field, and it only was because it had no UI to change it. §3-114's
 * `keepResidentEmbeddings` (2026-09-03) arrived with its UI already attached,
 * so it has been persisted from its first day. */
export interface StoredAcceleration {
  attentionBackend: AttentionBackend;
  blockSwapPrefetch: boolean;
  keepResident: boolean;
  fusedGgufDequantKernel: boolean;
  vaeMode: VaeMode;
  keepResidentEmbeddings: boolean;
}

const STORED_DEFAULTS: StoredAcceleration = {
  attentionBackend: ATTENTION_BACKEND_DEFAULT,
  blockSwapPrefetch: BLOCK_SWAP_PREFETCH_SERVER_DEFAULT,
  keepResident: KEEP_RESIDENT_SERVER_DEFAULT,
  fusedGgufDequantKernel: FUSED_GGUF_DEQUANT_KERNEL_SERVER_DEFAULT,
  vaeMode: VAE_MODE_DEFAULT,
  keepResidentEmbeddings: KEEP_RESIDENT_EMBEDDINGS_SERVER_DEFAULT,
};

/** Reads the persisted acceleration choices. Wrapped in a `try` since
 * `localStorage` can throw in some restricted embeddings (e.g. a WebView2
 * host with storage disabled) — falling back to the defaults is preferable to
 * a crash. Accepts THREE shapes, oldest first:
 *  1. missing / unreadable → defaults
 *  2. the pre-2026-08-01 BARE backend string (`"sdpa"` / `"sage"`, written
 *     before this feature existed) — read as that backend with
 *     `blockSwapPrefetch`/`keepResident` at their defaults (the
 *     backward-compat path `ACCELERATION_STORAGE_KEY`'s doc comment promises)
 *  3. the current JSON object `{"attentionBackend":...,
 *     "blockSwapPrefetch":...,"keepResident":...,
 *     "fusedGgufDequantKernel":...,"vaeMode":...,
 *     "keepResidentEmbeddings":...}`
 * Any unrecognized/malformed value inside falls back to its own default —
 * same defensive posture as `ThemeContext.tsx`'s `readStoredTheme`. The
 * per-field fallback is what lets a JSON blob written by a pre-§48 build
 * (which has no `keepResident` key at all) — or a pre-§51 one (no
 * `fusedGgufDequantKernel`), or a pre-§52 one (no `vaeMode`), or a pre-§3-114
 * one (no `keepResidentEmbeddings`) — read back
 * cleanly. EVERY field added to
 * {@link StoredAcceleration} needs its own line here: without one, the
 * whole-object write below would persist an `undefined` that reads back as
 * the default forever. */
export function readStoredAcceleration(): StoredAcceleration {
  let raw: string | null = null;
  try {
    raw = window.localStorage.getItem(ACCELERATION_STORAGE_KEY);
  } catch {
    return { ...STORED_DEFAULTS };
  }
  if (raw === null) return { ...STORED_DEFAULTS };
  if (raw === "sdpa" || raw === "sage") {
    return {
      attentionBackend: raw,
      blockSwapPrefetch: BLOCK_SWAP_PREFETCH_SERVER_DEFAULT,
      keepResident: KEEP_RESIDENT_SERVER_DEFAULT,
      fusedGgufDequantKernel: FUSED_GGUF_DEQUANT_KERNEL_SERVER_DEFAULT,
      vaeMode: VAE_MODE_DEFAULT,
      keepResidentEmbeddings: KEEP_RESIDENT_EMBEDDINGS_SERVER_DEFAULT,
    };
  }
  try {
    const parsed: unknown = JSON.parse(raw);
    if (typeof parsed !== "object" || parsed === null) return { ...STORED_DEFAULTS };
    const rec = parsed as Record<string, unknown>;
    const backend = rec.attentionBackend;
    const vaeMode = rec.vaeMode;
    return {
      attentionBackend: backend === "sdpa" || backend === "sage" ? backend : ATTENTION_BACKEND_DEFAULT,
      blockSwapPrefetch:
        typeof rec.blockSwapPrefetch === "boolean" ? rec.blockSwapPrefetch : BLOCK_SWAP_PREFETCH_SERVER_DEFAULT,
      keepResident: typeof rec.keepResident === "boolean" ? rec.keepResident : KEEP_RESIDENT_SERVER_DEFAULT,
      fusedGgufDequantKernel:
        typeof rec.fusedGgufDequantKernel === "boolean"
          ? rec.fusedGgufDequantKernel
          : FUSED_GGUF_DEQUANT_KERNEL_SERVER_DEFAULT,
      vaeMode: vaeMode === "default" || vaeMode === "prune_vaed" ? vaeMode : VAE_MODE_DEFAULT,
      keepResidentEmbeddings:
        typeof rec.keepResidentEmbeddings === "boolean"
          ? rec.keepResidentEmbeddings
          : KEEP_RESIDENT_EMBEDDINGS_SERVER_DEFAULT,
    };
  } catch {
    return { ...STORED_DEFAULTS };
  }
}

/** Persists the acceleration choices as JSON. Wrapped in a `try` since
 * `localStorage` can throw in some restricted embeddings (e.g. private mode)
 * — a failed write is silently swallowed, matching `ThemeContext.tsx`'s
 * `writeStoredTheme`. */
export function writeStoredAcceleration(value: StoredAcceleration): void {
  try {
    window.localStorage.setItem(ACCELERATION_STORAGE_KEY, JSON.stringify(value));
  } catch {
    // Ignore — localStorage unavailable (e.g. private mode).
  }
}

/** Whether the server reports SageAttention as installed, read off `GET
 * /status`'s top-level `acceleration` block (backend §43).
 *
 * Three-valued ON PURPOSE: `null` means "unknown" — no status yet (still
 * checking / offline), or a backend older than §43 that has no `acceleration`
 * block at all. Callers must only disable the sage button on an explicit
 * `false`, never on `null`: the server falls back to sdpa on its own when sage
 * is asked for and unavailable (the job still completes and records
 * `attention_used: "sage->sdpa"`), so greying the button out is a courtesy,
 * not a safety gate — and applying it to "unknown" would silently hide a
 * working feature from anyone whose /status poll hasn't landed yet. */
export function sageAvailability(status: StatusResponse | null | undefined): boolean | null {
  const available = status?.acceleration?.sage_available;
  return typeof available === "boolean" ? available : null;
}

/** Whether the server is running with block-swap prefetch available, read off
 * `GET /status`'s top-level `acceleration` block (backend §44). Same
 * three-valued shape and same "disable on explicit `false` only, never on
 * `null`" rule as {@link sageAvailability} — see that function's own doc
 * comment for the full rationale. The backend's own availability predicate is
 * the single source of truth this mirrors (backend §8.5). */
export function blockSwapPrefetchAvailability(status: StatusResponse | null | undefined): boolean | null {
  const available = status?.acceleration?.block_swap_prefetch_available;
  return typeof available === "boolean" ? available : null;
}

/** Whether block-swap prefetch is EFFECTIVELY running: the local toggle is on
 * AND the server has not explicitly reported it unavailable. `prefetchAvailable`
 * is {@link blockSwapPrefetchAvailability}'s three-valued result, so `null`
 * (unknown / offline / a pre-§44 backend) does NOT count as unavailable — the
 * same "act on an explicit `false` only" rule the buttons themselves follow. */
export function blockSwapPrefetchEffective(
  acceleration: AccelerationSettings,
  prefetchAvailable: boolean | null,
): boolean {
  return acceleration.blockSwapPrefetch && prefetchAvailable !== false;
}

/**
 * Whether keep-resident is EFFECTIVELY on (§1-10, 2026-08-03). Keeping the
 * skeleton resident only pays off while block-swap prefetch is running — the
 * backend's own guard (`engine/worker.py`, G-C) already turns the "prefetch
 * off + keep-resident on" combination back off and records
 * `keep_resident_used: "on->off"`. This is the ONE rule both the Settings
 * panel's rendering and {@link accelerationRequestFields} read, so what the
 * user sees and what the request carries can never drift apart.
 *
 * The STORED choice is deliberately left untouched (see
 * {@link effectiveAcceleration}) — turning prefetch back on restores it.
 */
export function keepResidentEffective(
  acceleration: AccelerationSettings,
  prefetchAvailable: boolean | null,
): boolean {
  return acceleration.keepResident && blockSwapPrefetchEffective(acceleration, prefetchAvailable);
}

/** The settings object with `keepResident` folded down to its effective value
 * (§1-10) — what every READER (Create/Chain/Batch, the Settings panel) gets
 * handed by `useAccelerationSettings`, while the raw stored choice stays in
 * that hook's own state and in `localStorage`. Applied there rather than at
 * each submission path because only `AppShell` holds the `/status` capability
 * flag; {@link accelerationRequestFields} then needs no extra argument and its
 * three call sites stay untouched. Returns the SAME reference when nothing
 * changes, so a React consumer taking `acceleration` as a dependency doesn't
 * see a new object every render. */
export function effectiveAcceleration(
  acceleration: AccelerationSettings,
  prefetchAvailable: boolean | null,
): AccelerationSettings {
  const keepResident = keepResidentEffective(acceleration, prefetchAvailable);
  if (keepResident === acceleration.keepResident) return acceleration;
  return { ...acceleration, keepResident };
}

/** The additive acceleration fields as sent on a `/generate` or
 * `/generate/chain` request — see {@link accelerationRequestFields}. */
export interface AccelerationRequestFields {
  attention_backend?: AttentionBackend;
  block_swap_prefetch?: boolean;
  keep_resident?: boolean;
  fused_gguf_dequant_kernel?: boolean;
  vae_mode?: VaeMode;
  keep_resident_embeddings?: boolean;
}

/**
 * The six acceleration settings as they would EFFECTIVELY run, expressed in
 * the SERVER's own vocabulary (the request field names) — every key always
 * present, so it can be matched key-by-key against a served
 * `AppLimits.comfort_budgets` row's `requires` map (2026-08-31,
 * `shell/comfortTable.ts`'s `resolveComfortRow`). This is deliberately NOT
 * {@link accelerationRequestFields}: that one omits every field still sitting
 * on the server default (so a request stays byte-identical to before the
 * feature existed), which is the opposite of what a matcher needs.
 *
 * `acceleration` MUST be the EFFECTIVE settings object — the one
 * {@link effectiveAcceleration} already ran (what `useAccelerationSettings`
 * hands to every reader) — not the raw stored choice. Reading the raw object
 * here would misjudge `keep_resident` while block-swap prefetch is
 * unavailable/off (the pair the backend itself folds back to off — see
 * {@link keepResidentEffective}).
 *
 * `blockSwapPrefetch`'s own SERVER availability is deliberately NOT checked a
 * second time here for exactly that reason: it is already folded into
 * `acceleration.keepResident` before this function ever sees the object.
 *
 * `sageAvailable` gets the SAME three-valued treatment {@link sageAvailability}
 * documents: `null` ("unknown" — no `/status` yet, or a pre-§43 backend) counts
 * as effectively on, matching the sage button's own "disable on an explicit
 * `false` only" rule and avoiding a startup flicker where the comfort marker
 * would otherwise show smart, then jump back to the fallback the instant
 * `/status` lands. Only an explicit `false` demotes `attention_backend` to
 * `"sdpa"` here.
 *
 * ⚠ Returns a FRESH object every call, so never place it in a React
 * dependency array — callers build it once inside the function that consumes
 * it (R-9).
 */
export function effectiveAccelerationFields(
  acceleration: AccelerationSettings,
  sageAvailable: boolean | null,
): Required<AccelerationRequestFields> {
  return {
    attention_backend: acceleration.attentionBackend === "sage" && sageAvailable !== false ? "sage" : "sdpa",
    block_swap_prefetch: acceleration.blockSwapPrefetch,
    keep_resident: acceleration.keepResident,
    fused_gguf_dequant_kernel: acceleration.fusedGgufDequantKernel,
    vae_mode: acceleration.vaeMode,
    keep_resident_embeddings: acceleration.keepResidentEmbeddings,
  };
}

/**
 * Builds the fields to spread into a generate/chain request body — the single
 * place this feature's additive contract lives (mirrors
 * `nagSettings.nagRequestFields`).
 *
 * Returns `{}` unless the user has actively moved OFF the server's own
 * default for a given field, so every request the WebUI has ever sent stays
 * byte-identical to before this feature existed. That is not merely tidiness:
 * the backend's request-shape tests assert complete key-set equality on
 * several paths, and an unconditionally-sent field would break them (and
 * would make every stored job record differ from its pre-2026-07-31
 * equivalent for no reason).
 *
 * No mock fields are left: `vaeMode` was the last one and became REAL on
 * 2026-08-05 (backend §52), added here and nowhere else — exactly what
 * `fusedGgufDequantKernel` did on 2026-08-04.
 */
export function accelerationRequestFields(
  acceleration: AccelerationSettings | undefined,
): AccelerationRequestFields {
  if (!acceleration) return {};
  const out: AccelerationRequestFields = {};
  if (acceleration.attentionBackend !== ATTENTION_BACKEND_DEFAULT) {
    out.attention_backend = acceleration.attentionBackend;
  }
  if (acceleration.blockSwapPrefetch !== BLOCK_SWAP_PREFETCH_SERVER_DEFAULT) {
    out.block_swap_prefetch = acceleration.blockSwapPrefetch;
  }
  // Same "only when moved off the server default" rule, but note the
  // direction is the OPPOSITE of `block_swap_prefetch` above: the server
  // defaults keep-resident to `false`, so this key appears only while the
  // user has turned it ON — and, since §1-10, only while block-swap prefetch
  // is on too (`keepResidentEffective`). `null` is passed for the capability
  // flag because this layer has no `/status` access: the capability half of
  // the rule is already folded in by `effectiveAcceleration` before the
  // settings object ever reaches a submission path.
  const keepResident = keepResidentEffective(acceleration, null);
  if (keepResident !== KEEP_RESIDENT_SERVER_DEFAULT) {
    out.keep_resident = keepResident;
  }
  // §1-11 (2026-08-04): same "only when moved off the server default"
  // rule, but the direction is the same as `block_swap_prefetch` and the
  // OPPOSITE of `keep_resident` above — the server default is `true` (flipped
  // after the real-device gates), so this key only ever appears as `false`.
  // INDEPENDENT: there is no capability flag and no dependency on any other
  // row, so the raw setting is read straight off the object.
  if (acceleration.fusedGgufDequantKernel !== FUSED_GGUF_DEQUANT_KERNEL_SERVER_DEFAULT) {
    out.fused_gguf_dequant_kernel = acceleration.fusedGgufDequantKernel;
  }
  // §52 (2026-08-05): same "only when moved off the server default" rule.
  // Direction matches `keep_resident` (the server default is `"default"`, so
  // the key only ever appears as `"prune_vaed"`) and, like the fused kernel,
  // it is INDEPENDENT — no capability flag and no dependency on any other row,
  // so the raw setting is read straight off the object. Emitted LAST, keeping
  // declaration order stable for the backend's frozen key-set tests.
  if (acceleration.vaeMode !== VAE_MODE_DEFAULT) {
    out.vae_mode = acceleration.vaeMode;
  }
  // §3-114 (2026-09-03): same "only when moved off the server default" rule
  // again. Direction matches `keep_resident` (the server default is `false`, so
  // the key only ever appears as `true`) and, like the fused kernel and the VAE
  // row, it is INDEPENDENT — no capability flag and, unlike `keep_resident`, no
  // dependency on prefetch either, so the raw setting is read straight off the
  // object. Emitted LAST, keeping declaration order stable for the backend's
  // frozen key-set tests, exactly as `vae_mode` was before it.
  //
  // The SCOPE half of this field's rule is not enforced here and must not be:
  // an engine without an embeddings processor 422s a `true`, and it is
  // `AppShell` that hides the row and writes the stored choice back to `false`
  // when `GET /models` publishes the name. By the time a settings object
  // reaches this function the answer is already in it.
  if (acceleration.keepResidentEmbeddings !== KEEP_RESIDENT_EMBEDDINGS_SERVER_DEFAULT) {
    out.keep_resident_embeddings = acceleration.keepResidentEmbeddings;
  }
  return out;
}
