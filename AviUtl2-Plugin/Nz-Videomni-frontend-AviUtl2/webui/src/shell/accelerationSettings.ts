/**
 * NOTE (§3-164, 2026-09-24): this store is the Settings panel's home for
 * per-job generation settings, so besides acceleration it also carries the
 * panel's Output setting (`embedMp4Metadata`).
 *
 * Acceleration (生成高速化) settings — the Settings-panel section that picks
 * which attention implementation the backend runs a job with (2026-07-31,
 * backend §43), whether it prefetches block-swap transfers (2026-08-01,
 * backend §44), whether it keeps the models' CPU-side skeleton resident
 * between jobs (2026-08-02, backend §48), and whether GGUF dequantization runs
 * as one fused Triton kernel (2026-08-04, backend §51), which VAE decoder
 * reconstructs the video (2026-08-05, backend §52 — PrunaVAED), whether
 * LTX 2.5's embeddings processor stays resident between jobs (2026-09-03,
 * 台帳 §3-114), and whether MiniMax H3 runs its denoise in VRAM 16GB mode
 * (2026-10-18, 台帳 §1-101). Shared by
 * Create/Chain/Batch via a
 * single `AccelerationSettings` object (D1: `AppShell` useState + props, no
 * Context — exactly the arrangement `shell/nagSettings.ts` documents and every
 * other cross-screen setting here follows; see
 * `shell/useAccelerationSettings.ts`). This module is the pure, UI-free layer:
 * the types, the constants, the `localStorage` read/write for the chosen
 * backend, and the additive request-field contract.
 *
 * Seven items exist in the UI, and ALL SEVEN are implemented — the fifth
 * (`vaeMode`) was the last mock, which backend §52 turned real on 2026-08-05,
 * the sixth (`keepResidentEmbeddings`) was born real on 2026-09-03, and so was
 * the seventh (`vram16Mode`, 2026-10-18):
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
 *    ~70s -> ~10s of preprocessing). The resident main-memory cost differs
 *    per engine (the panel's note, `strings.settings.accelKeepResidentNote`,
 *    states it), hence the "64GB or more recommended" note. Like
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
 *    only on LTX 2.5, so it is the engines without it that publish
 *    `keep_resident_embeddings` in `unsupported_features` and 422 a `true`. `AppShell` hides the row (and
 *    writes a leftover `true` back to `false`) on any engine that publishes the
 *    name — the same treatment `prune_vaed` gets, just pointing the other way.
 *  - `vram16Mode` — REAL (2026-10-18, 台帳 §1-101). "VRAM 16GB mode": MiniMax
 *    H3 only. Splits the denoiser's activations into pieces so a job fits in
 *    16GB of VRAM; precision is unchanged. The target is bit-identical output:
 *    `sdpa` is; with `sage`, 2 of the 50 blocks differ slightly (the kernel
 *    depends on the heads per call), accepted by the owner's visual check
 *    (ruling 53, backend VERIFICATION_LOG §169.3).
 *    Sent as `vram16_mode`; the H3 server default is `true`, so the key rides
 *    along only while it is OFF — the `fusedGgufDequantKernel` direction. The
 *    scope points the same way as `keepResidentEmbeddings`: the engines that do
 *    NOT have it (LTX 2.3 / 2.5) publish `vram16_mode` in
 *    `unsupported_features` and 422 a `true`, and `AppShell` hides the row and
 *    writes a leftover choice back to the default on any engine that publishes
 *    the name. Since §1-106 the chain request carries it too (H3 chains).
 *  - `piH3` — REAL (2026-11-23, backend ruling 107 / MINIMAX_H3_DESIGN §6.7).
 *    MiniMax H3's distilled few-step sampling + DWD decode: `"off"` (the
 *    server default) / `"4steps"` / `"2steps"`. It changes the output (a
 *    different sampler and decoder), so — like PrunaVAED — it is opt-in and
 *    the key rides along only while it is not `"off"`. While on, the server
 *    fills the step count and both shifts with the choice's values (served as
 *    `GET /models` `constraints.pi_h3`), so the submission paths drop those
 *    three values (`samplingRequestFields`). The scope points the
 *    `vram16Mode` way: LTX 2.3 / 2.5 publish `pi_h3` in
 *    `unsupported_features`, and `AppShell` hides the row and writes a
 *    leftover choice back to `"off"`.
 *  - `solVae` — REAL (2026-11-24, Sol-H3-VAE). MiniMax H3's lightweight
 *    distilled video decoder: only the decode is swapped (the step count and
 *    the shifts are untouched), and the pixels differ slightly from the
 *    standard decoder's. Sent as `sol_vae`; the server default is `false`, so
 *    the key rides along only while it is ON (the `keep_resident` direction).
 *    It shares ONE Settings row ("Fast VAE") with `piH3`: the row's four
 *    buttons are written by the single {@link fastVaeSettings} rule, so the
 *    two are never on together (the server 422s that pair), and
 *    {@link readStoredAcceleration} folds a stored pair back the same way.
 *    The fields stay separate because the existing "pi-H3 is on" checks (the
 *    slider lock, the dropped sampling values, the red note) must not fire on
 *    Sol. Scope: LTX 2.3 / 2.5 publish `sol_vae` in `unsupported_features`,
 *    and `AppShell` hides the row and writes a leftover `true` back to `false`.
 *  - `keepReferenceLatents` — REAL (台帳 §1-297). MiniMax H3's reference
 *    encodings (pictures, videos, audio) are kept in main memory between jobs,
 *    so the next generation with the same references and the following clips of
 *    a Chained job skip their encoding; the output does not change. Sent as
 *    `keep_reference_latents`; the server default is `false`, so the key rides
 *    along only while it is ON (the `keep_resident` direction). Independent of
 *    every other row. Scope: LTX 2.3 / 2.5 publish `keep_reference_latents` in
 *    `unsupported_features`, and `AppShell` hides the row and writes a leftover
 *    `true` back to `false`.
 *  - `keepTextConditioning` — REAL (台帳 §1-298, widened by §1-299). The text
 *    encoder output (the conditioning) is cached in main memory, so a generation
 *    whose prompt and media are the same as last time skips the text encoder;
 *    the output does not change. Sent as `keep_text_conditioning`; the server
 *    default is `false`, so the key rides along only while it is ON. Independent
 *    of every other row, `keepReferenceLatents` included. Scope: MiniMax H3,
 *    LTX 2.3 and LTX 2.5 all take it. Only a server that still refuses it on
 *    LTX publishes `keep_text_conditioning` in `unsupported_features`; then
 *    `AppShell` hides the row and writes a leftover `true` back to `false`.
 *  - `sparkH3` / `sparkH3TopkPercent` / `sparkH3FullSteps` — REAL (裁定 113,
 *    MINIMAX_H3_DESIGN §6.11). Spark-H3: MiniMax H3's sparse attention, layered
 *    on top of sdpa / Sage. It changes the output (part of the attention is
 *    approximated), so it is opt-in: the server default is `false`, and the
 *    three keys `spark_h3` / `spark_h3_topk_ratio` / `spark_h3_full_steps`
 *    ride along only while it is ON (all three or none). The ratio is stored as
 *    a whole percent (1-50) and sent as `percent / 100`. `/status` publishes
 *    `acceleration.spark_h3_available` (three-valued, the Sage shape —
 *    {@link sparkH3Availability}). Scope: LTX 2.3 / 2.5 publish `spark_h3` in
 *    `unsupported_features`, and `AppShell` hides the row and writes a
 *    leftover `true` back to `false` (the two numbers stay stored). The two
 *    generate-button notes it adds are {@link sparkH3StepsTooFew} and
 *    {@link sparkH3VideoTooShort}.
 *  - `teCombinedRead` — REAL (台帳 §1-303). When a job has more than one prompt
 *    (the clips of a Chained job, a negative prompt), MiniMax H3's text encoder
 *    processes them in one load; the output does not change. Sent as
 *    `te_combined_read`; the server default is `false`, so the key rides along
 *    only while it is ON (the `keep_resident` direction). Independent of every
 *    other row. Scope: LTX 2.3 / 2.5 publish `te_combined_read` in
 *    `unsupported_features`, and `AppShell` hides the row and writes a leftover
 *    `true` back to `false`.
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

/** Which pi-H3 choice a MiniMax H3 job runs with. Mirrors the backend's
 * `api/models.py` `pi_h3` literal as of 2026-11-23 (ruling 107) — if the
 * backend's set of values ever changes, update both sides together. */
export type PiH3Mode = "off" | "4steps" | "2steps";

/** The Settings panel's "Fast VAE" row: one choice over the two stored fields
 * `piH3` and `solVae` (2026-11-24). `"sol"` is Sol-H3-VAE; the other three are
 * the pi-H3 choices verbatim. Not an API value — {@link fastVaeSettings} maps
 * it onto the two fields, {@link fastVaeChoice} maps them back. */
export type FastVaeChoice = PiH3Mode | "sol";

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
  /** REAL (2026-10-18, 台帳 §1-101): "VRAM 16GB mode" — MiniMax H3 only.
   * Splits the denoiser's activations so the job fits in 16GB of VRAM without
   * changing precision. Sent as `vram16_mode` — but only while OFF, since the
   * H3 server defaults to `true`. LTX 2.3 / 2.5 publish the name in
   * `unsupported_features`, so the row is hidden there. */
  vram16Mode: boolean;
  /** REAL (2026-11-23, ruling 107): pi-H3 — MiniMax H3 only. `"4steps"` /
   * `"2steps"` run the distilled few-step sampler and the DWD decode; the step
   * count and both shifts then come from the server (`constraints.pi_h3`), so
   * the sliders are locked and those three values are not sent. Sent as
   * `pi_h3` — but only while not `"off"` (the server default). LTX 2.3 / 2.5
   * publish the name in `unsupported_features`, so the row is hidden there. */
  piH3: PiH3Mode;
  /** REAL (2026-11-24): Sol-H3-VAE — MiniMax H3 only. Swaps the video decoder
   * for the lightweight distilled one (the pixels differ slightly). Sent as
   * `sol_vae` — but only while ON, since the server defaults to `false`. Never
   * on together with a {@link piH3} choice other than `"off"`: both are written
   * by {@link fastVaeSettings} only. LTX 2.3 / 2.5 publish the name in
   * `unsupported_features`, so the row is hidden there. */
  solVae: boolean;
  /** REAL (台帳 §1-297): "Keep reference latents resident" — MiniMax H3 only.
   * Keeps the encoded reference pictures / videos / audio in main memory so the
   * next generation with the same references (and the following clips of a
   * Chained job) skip their encoding; the output does not change. Sent as
   * `keep_reference_latents` — but only while ON, since the server defaults to
   * `false`. LTX 2.3 / 2.5 publish the name in `unsupported_features`, so the
   * row is hidden there. */
  keepReferenceLatents: boolean;
  /** REAL (台帳 §1-298): "Text encoder conditioning cache" — MiniMax H3, LTX 2.3, LTX 2.5 (§1-299).
   * Caches the text encoder's conditioning in main memory so a generation with
   * the same prompt and media as last time skips the text encoder; the output
   * does not change. Sent as `keep_text_conditioning` — but only while ON, since
   * the server defaults to `false`. A server that still refuses it on LTX
   * publishes the name in `unsupported_features`; the row is hidden there. */
  keepTextConditioning: boolean;
  /** REAL (裁定 113, MINIMAX_H3_DESIGN §6.11): Spark-H3 — MiniMax H3 only.
   * Sent as `spark_h3: true` (with the two values below) — but only while ON,
   * since the server defaults to `false`. LTX 2.3 / 2.5 publish the name in
   * `unsupported_features`, so the row is hidden there. */
  sparkH3: boolean;
  /** Spark-H3's "full-attention ratio" in whole percent (1-50). Sent as
   * `spark_h3_topk_ratio: sparkH3TopkPercent / 100`, only while
   * {@link sparkH3} is on. Kept while the row shows "off". */
  sparkH3TopkPercent: number;
  /** Spark-H3's "full-attention steps" (0-10): the first steps of each clip
   * that run without Spark-H3. Sent as `spark_h3_full_steps`, only while
   * {@link sparkH3} is on. Kept while the row shows "off". */
  sparkH3FullSteps: number;
  /** REAL (台帳 §1-303): "Text encoder combined read" — MiniMax H3 only. When a
   * job has more than one prompt (the clips of a Chained job, a negative
   * prompt), the text encoder processes them in one load; the output does not
   * change. Sent as `te_combined_read` — but only while ON, since the server
   * defaults to `false`. LTX 2.3 / 2.5 publish the name in
   * `unsupported_features`, so the row is hidden there. */
  teCombinedRead: boolean;
  /** Output setting (§3-164, 2026-09-24), not an acceleration knob — it lives
   * in this object because this is the Settings panel's per-job store (see the
   * note at the top of this file). Whether the finished output.mp4 /
   * joined.mp4 gets the same JSON as metadata.json written into its `comment`
   * tag. Sent as `embed_mp4_metadata` — but only while OFF, since the server
   * defaults to `true`. It is deliberately NOT part of
   * {@link AccelerationRequestFields}: the comfort-budget matcher
   * ({@link effectiveAccelerationFields}) has nothing to match it against. */
  embedMp4Metadata: boolean;
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
/** Server default for `embed_mp4_metadata` as of §3-164 (2026-09-24): `true`
 * (the finished mp4 carries its generation conditions, like A1111's
 * `enable_pnginfo`). Direction matches `fused_gguf_dequant_kernel`: the key
 * rides along only when the user turns it OFF. */
export const EMBED_MP4_METADATA_SERVER_DEFAULT = true;
/** Server default for `vram16_mode` as of 台帳 §1-101 (2026-10-18): `true` on
 * MiniMax H3 (`api/models.py`'s `VRAM16_MODE_DEFAULT_H3`; the server writes it
 * back when the key is absent). Direction matches
 * `fused_gguf_dequant_kernel`: the key rides along only when the user turns it
 * OFF. On LTX the row is hidden and the stored value is reset to this default,
 * so an absent key is all an LTX request ever carries. */
export const VRAM16_MODE_SERVER_DEFAULT = true;
/** Server default for `pi_h3` as of ruling 107 (2026-11-23): `"off"`.
 * Direction matches `vae_mode`: the key rides along only while the user picks
 * a choice. On LTX the row is hidden and the stored value is reset to this
 * default, so an absent key is all an LTX request ever carries. */
export const PI_H3_DEFAULT: PiH3Mode = "off";
/** Server default for `sol_vae` as of 2026-11-24: `false`. Direction matches
 * `keep_resident`: the key rides along only while the user turns it ON. On LTX
 * the row is hidden and the stored value is reset to this default. */
export const SOL_VAE_SERVER_DEFAULT = false;
/** Server default for `keep_reference_latents` as of 台帳 §1-297: `false`.
 * Direction matches `keep_resident`: the key rides along only while the user
 * turns it ON. On LTX the row is hidden and the stored value is reset to this
 * default. */
export const KEEP_REFERENCE_LATENTS_SERVER_DEFAULT = false;
/** Server default for `keep_text_conditioning` as of 台帳 §1-298: `false`. Same
 * direction as `keep_reference_latents`: the key rides along only while the
 * user turns it ON. MiniMax H3, LTX 2.3 and LTX 2.5 all take it (台帳 §1-299),
 * so the row is shown on LTX too; only a server that still refuses it there
 * publishes the name in `unsupported_features`, and then the row is hidden and
 * the stored value is reset to this default. */
export const KEEP_TEXT_CONDITIONING_SERVER_DEFAULT = false;
/** Server default for `spark_h3` (裁定 113): `false`. Direction matches
 * `keep_resident`: the keys ride along only while the user turns it ON. On LTX
 * the row is hidden and the stored value is reset to this default. */
export const SPARK_H3_SERVER_DEFAULT = false;
/** Server default for `te_combined_read` as of 台帳 §1-303: `false`. Direction
 * matches `keep_reference_latents`: the key rides along only while the user
 * turns it ON. On LTX the row is hidden and the stored value is reset to this
 * default. */
export const TE_COMBINED_READ_SERVER_DEFAULT = false;
/** The "full-attention ratio" slider's range and default, in whole percent
 * (裁定 113: 1-50%, 1% steps, default 20%). Fixed values the panel owns, like
 * the other Settings choices. */
export const SPARK_H3_TOPK_PERCENT_MIN = 1;
export const SPARK_H3_TOPK_PERCENT_MAX = 50;
export const SPARK_H3_TOPK_PERCENT_DEFAULT = 20;
/** Where the ratio slider's two markers sit (the recommended 10-20% range). */
export const SPARK_H3_TOPK_PERCENT_MARKERS: readonly number[] = [10, 20];
/** The "full-attention steps" slider's range and default (裁定 113: 0-10,
 * default 2). */
export const SPARK_H3_FULL_STEPS_MIN = 0;
export const SPARK_H3_FULL_STEPS_MAX = 10;
export const SPARK_H3_FULL_STEPS_DEFAULT = 2;
/** The sequence length below which the engine runs full attention instead of
 * Spark-H3 — the same gate as the author's ComfyUI version (MINIMAX_H3_DESIGN
 * §6.11). {@link sparkH3VideoTooShort} compares the panel's estimate with it. */
export const SPARK_H3_MIN_TOKENS = 12288;

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
  vram16Mode: VRAM16_MODE_SERVER_DEFAULT,
  piH3: PI_H3_DEFAULT,
  solVae: SOL_VAE_SERVER_DEFAULT,
  keepReferenceLatents: KEEP_REFERENCE_LATENTS_SERVER_DEFAULT,
  keepTextConditioning: KEEP_TEXT_CONDITIONING_SERVER_DEFAULT,
  sparkH3: SPARK_H3_SERVER_DEFAULT,
  sparkH3TopkPercent: SPARK_H3_TOPK_PERCENT_DEFAULT,
  sparkH3FullSteps: SPARK_H3_FULL_STEPS_DEFAULT,
  teCombinedRead: TE_COMBINED_READ_SERVER_DEFAULT,
  embedMp4Metadata: EMBED_MP4_METADATA_SERVER_DEFAULT,
});

/** The pressed button of the "Fast VAE" row: the pi-H3 choice while it is not
 * `"off"`, else `"sol"` while Sol-H3-VAE is on, else `"off"`. */
export function fastVaeChoice(acceleration: Pick<AccelerationSettings, "piH3" | "solVae">): FastVaeChoice {
  if (acceleration.piH3 !== "off") return acceleration.piH3;
  if (acceleration.solVae) return "sol";
  return "off";
}

/** The two stored fields one "Fast VAE" button writes — the ONLY way the row
 * changes them, so a pi-H3 choice and Sol are never on together: `"sol"` is
 * pi-H3 off + Sol on; any other choice is that pi-H3 choice + Sol off. */
export function fastVaeSettings(choice: FastVaeChoice): Pick<AccelerationSettings, "piH3" | "solVae"> {
  if (choice === "sol") return { piH3: "off", solVae: true };
  return { piH3: choice, solVae: false };
}

/**
 * The settings object with the named fields forced back to their SERVER default
 * (§3-135) — the "cleanup" half of `shell/featureScope.ts`'s table, which names
 * the FIELDS while {@link ACCELERATION_DEFAULTS} holds the VALUES. `AppShell`
 * calls it through `useAccelerationSettings`'s `resetToServerDefaults` whenever
 * the loaded base model hides a Settings row: hiding is not enough on its own,
 * because the choice is persisted and would keep riding along on every request
 * to an engine that 422s it.
 *
 * Returns the SAME REFERENCE when there is nothing to move — React's `setState`
 * bails out on an identical value, and the cleanup effect that calls this is
 * allowed to run on any render (it re-fires whenever the memoized field list is
 * rebuilt, which a `/models` refresh does). That bail-out is what keeps a
 * no-op call from looping.
 *
 * The "nothing to move" test reads the STORED value, not the effective one
 * ({@link effectiveAcceleration}). The two are equivalent today — the only field
 * `effectiveAcceleration` rewrites is `keepResident`, and no table row resets
 * that — but they would DIVERGE the moment a `resets: "keepResident"` row is
 * added: the effective value can already read `false` (prefetch off) while the
 * stored choice is still `true`, so an effective-basis test would skip a
 * write-back that is still needed. Stored is the basis that stays correct.
 *
 * All fields are enumerated explicitly rather than spread over a computed
 * key, deliberately and for {@link effectiveAccelerationFields}'s reason: a
 * new field added to {@link AccelerationSettings} must fail `typecheck` here
 * instead of quietly slipping through a `{ ...acceleration, [f]: … }`.
 */
export function withServerDefaults(
  acceleration: AccelerationSettings,
  fields: readonly (keyof AccelerationSettings)[],
): AccelerationSettings {
  if (fields.every((f) => acceleration[f] === ACCELERATION_DEFAULTS[f])) return acceleration;
  const at = (f: keyof AccelerationSettings) => fields.includes(f);
  return {
    attentionBackend: at("attentionBackend") ? ACCELERATION_DEFAULTS.attentionBackend : acceleration.attentionBackend,
    blockSwapPrefetch: at("blockSwapPrefetch")
      ? ACCELERATION_DEFAULTS.blockSwapPrefetch
      : acceleration.blockSwapPrefetch,
    keepResident: at("keepResident") ? ACCELERATION_DEFAULTS.keepResident : acceleration.keepResident,
    fusedGgufDequantKernel: at("fusedGgufDequantKernel")
      ? ACCELERATION_DEFAULTS.fusedGgufDequantKernel
      : acceleration.fusedGgufDequantKernel,
    vaeMode: at("vaeMode") ? ACCELERATION_DEFAULTS.vaeMode : acceleration.vaeMode,
    keepResidentEmbeddings: at("keepResidentEmbeddings")
      ? ACCELERATION_DEFAULTS.keepResidentEmbeddings
      : acceleration.keepResidentEmbeddings,
    vram16Mode: at("vram16Mode") ? ACCELERATION_DEFAULTS.vram16Mode : acceleration.vram16Mode,
    piH3: at("piH3") ? ACCELERATION_DEFAULTS.piH3 : acceleration.piH3,
    solVae: at("solVae") ? ACCELERATION_DEFAULTS.solVae : acceleration.solVae,
    keepReferenceLatents: at("keepReferenceLatents")
      ? ACCELERATION_DEFAULTS.keepReferenceLatents
      : acceleration.keepReferenceLatents,
    keepTextConditioning: at("keepTextConditioning")
      ? ACCELERATION_DEFAULTS.keepTextConditioning
      : acceleration.keepTextConditioning,
    sparkH3: at("sparkH3") ? ACCELERATION_DEFAULTS.sparkH3 : acceleration.sparkH3,
    sparkH3TopkPercent: at("sparkH3TopkPercent")
      ? ACCELERATION_DEFAULTS.sparkH3TopkPercent
      : acceleration.sparkH3TopkPercent,
    sparkH3FullSteps: at("sparkH3FullSteps") ? ACCELERATION_DEFAULTS.sparkH3FullSteps : acceleration.sparkH3FullSteps,
    teCombinedRead: at("teCombinedRead") ? ACCELERATION_DEFAULTS.teCombinedRead : acceleration.teCombinedRead,
    // No feature-scope row resets the Output setting, so it is carried over as is.
    embedMp4Metadata: acceleration.embedMp4Metadata,
  };
}

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
  /** 台帳 §1-101 (2026-10-18): persisted from its first day. */
  vram16Mode: boolean;
  /** Ruling 107 (2026-11-23): persisted from its first day, like `vaeMode`. */
  piH3: PiH3Mode;
  /** 2026-11-24: Sol-H3-VAE, persisted from its first day. */
  solVae: boolean;
  /** 台帳 §1-297: keep reference latents resident, persisted from its first day. */
  keepReferenceLatents: boolean;
  /** 台帳 §1-298: text encoder conditioning cache, persisted from its first day. */
  keepTextConditioning: boolean;
  /** 裁定 113: Spark-H3 and its two slider values, persisted from their first day. */
  sparkH3: boolean;
  sparkH3TopkPercent: number;
  sparkH3FullSteps: number;
  /** 台帳 §1-303: text encoder combined read, persisted from its first day. */
  teCombinedRead: boolean;
  /** §3-164 (2026-09-24): the Output row, persisted from its first day. */
  embedMp4Metadata: boolean;
}

const STORED_DEFAULTS: StoredAcceleration = {
  attentionBackend: ATTENTION_BACKEND_DEFAULT,
  blockSwapPrefetch: BLOCK_SWAP_PREFETCH_SERVER_DEFAULT,
  keepResident: KEEP_RESIDENT_SERVER_DEFAULT,
  fusedGgufDequantKernel: FUSED_GGUF_DEQUANT_KERNEL_SERVER_DEFAULT,
  vaeMode: VAE_MODE_DEFAULT,
  keepResidentEmbeddings: KEEP_RESIDENT_EMBEDDINGS_SERVER_DEFAULT,
  vram16Mode: VRAM16_MODE_SERVER_DEFAULT,
  piH3: PI_H3_DEFAULT,
  solVae: SOL_VAE_SERVER_DEFAULT,
  keepReferenceLatents: KEEP_REFERENCE_LATENTS_SERVER_DEFAULT,
  keepTextConditioning: KEEP_TEXT_CONDITIONING_SERVER_DEFAULT,
  sparkH3: SPARK_H3_SERVER_DEFAULT,
  sparkH3TopkPercent: SPARK_H3_TOPK_PERCENT_DEFAULT,
  sparkH3FullSteps: SPARK_H3_FULL_STEPS_DEFAULT,
  teCombinedRead: TE_COMBINED_READ_SERVER_DEFAULT,
  embedMp4Metadata: EMBED_MP4_METADATA_SERVER_DEFAULT,
};

/** A stored whole number inside `[min, max]`, else `fallback` (a missing key,
 * a fraction, or a value out of range all read back as the default). */
function storedInt(value: unknown, min: number, max: number, fallback: number): number {
  return typeof value === "number" && Number.isInteger(value) && value >= min && value <= max ? value : fallback;
}

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
 *     "keepResidentEmbeddings":...,"vram16Mode":...,"piH3":...,
 *     "solVae":...,"keepReferenceLatents":...,"keepTextConditioning":...,
 *     "sparkH3":...,"sparkH3TopkPercent":...,"sparkH3FullSteps":...,
 *     "teCombinedRead":...,"embedMp4Metadata":...}`
 * Any unrecognized/malformed value inside falls back to its own default —
 * same defensive posture as `ThemeContext.tsx`'s `readStoredTheme`. The
 * per-field fallback is what lets a JSON blob written by a pre-§48 build
 * (which has no `keepResident` key at all) — or a pre-§51 one (no
 * `fusedGgufDequantKernel`), or a pre-§52 one (no `vaeMode`), or a pre-§3-114
 * one (no `keepResidentEmbeddings`), or a pre-§3-164 one (no
 * `embedMp4Metadata`), or a pre-§1-101 one (no `vram16Mode`), or a pre-ruling-107
 * one (no `piH3`), or a pre-Sol one (no `solVae`), or a pre-§1-297 one (no
 * `keepReferenceLatents`), or a pre-§1-298 one (no `keepTextConditioning`), or a
 * pre-Spark-H3 one (no `sparkH3` / `sparkH3TopkPercent` / `sparkH3FullSteps`), or a
 * pre-§1-303 one (no `teCombinedRead`) — read back
 * cleanly. The two Spark-H3 numbers also fall back when they are not whole
 * numbers inside their slider's range. `solVae` is also normalized against `piH3` (a pi-H3 choice other
 * than off wins and Sol reads back off). EVERY field added to
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
      vram16Mode: VRAM16_MODE_SERVER_DEFAULT,
      piH3: PI_H3_DEFAULT,
      solVae: SOL_VAE_SERVER_DEFAULT,
      keepReferenceLatents: KEEP_REFERENCE_LATENTS_SERVER_DEFAULT,
      keepTextConditioning: KEEP_TEXT_CONDITIONING_SERVER_DEFAULT,
      sparkH3: SPARK_H3_SERVER_DEFAULT,
      sparkH3TopkPercent: SPARK_H3_TOPK_PERCENT_DEFAULT,
      sparkH3FullSteps: SPARK_H3_FULL_STEPS_DEFAULT,
      teCombinedRead: TE_COMBINED_READ_SERVER_DEFAULT,
      embedMp4Metadata: EMBED_MP4_METADATA_SERVER_DEFAULT,
    };
  }
  try {
    const parsed: unknown = JSON.parse(raw);
    if (typeof parsed !== "object" || parsed === null) return { ...STORED_DEFAULTS };
    const rec = parsed as Record<string, unknown>;
    const backend = rec.attentionBackend;
    const vaeMode = rec.vaeMode;
    const piH3: PiH3Mode = rec.piH3 === "off" || rec.piH3 === "4steps" || rec.piH3 === "2steps" ? rec.piH3 : PI_H3_DEFAULT;
    // The "Fast VAE" row never stores a pi-H3 choice and Sol together
    // (`fastVaeSettings`); a stored pair anyway keeps the pi-H3 choice and
    // turns Sol off, so the request never carries the pair the server 422s.
    let solVae = typeof rec.solVae === "boolean" ? rec.solVae : SOL_VAE_SERVER_DEFAULT;
    if (piH3 !== "off" && solVae) solVae = false;
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
      vram16Mode: typeof rec.vram16Mode === "boolean" ? rec.vram16Mode : VRAM16_MODE_SERVER_DEFAULT,
      piH3,
      solVae,
      keepReferenceLatents:
        typeof rec.keepReferenceLatents === "boolean"
          ? rec.keepReferenceLatents
          : KEEP_REFERENCE_LATENTS_SERVER_DEFAULT,
      keepTextConditioning:
        typeof rec.keepTextConditioning === "boolean"
          ? rec.keepTextConditioning
          : KEEP_TEXT_CONDITIONING_SERVER_DEFAULT,
      sparkH3: typeof rec.sparkH3 === "boolean" ? rec.sparkH3 : SPARK_H3_SERVER_DEFAULT,
      sparkH3TopkPercent: storedInt(
        rec.sparkH3TopkPercent,
        SPARK_H3_TOPK_PERCENT_MIN,
        SPARK_H3_TOPK_PERCENT_MAX,
        SPARK_H3_TOPK_PERCENT_DEFAULT,
      ),
      sparkH3FullSteps: storedInt(
        rec.sparkH3FullSteps,
        SPARK_H3_FULL_STEPS_MIN,
        SPARK_H3_FULL_STEPS_MAX,
        SPARK_H3_FULL_STEPS_DEFAULT,
      ),
      teCombinedRead: typeof rec.teCombinedRead === "boolean" ? rec.teCombinedRead : TE_COMBINED_READ_SERVER_DEFAULT,
      embedMp4Metadata:
        typeof rec.embedMp4Metadata === "boolean" ? rec.embedMp4Metadata : EMBED_MP4_METADATA_SERVER_DEFAULT,
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

/** Whether the server reports Spark-H3 as usable, read off `GET /status`'s
 * `acceleration.spark_h3_available` (裁定 113). Same three-valued shape and
 * same "disable on explicit `false` only, never on `null`" rule as
 * {@link sageAvailability}: the server runs the job without Spark-H3 on its own
 * when it cannot use it (recording `spark_h3_used: "on->off"`), so greying the
 * button out is a courtesy, not a safety gate. */
export function sparkH3Availability(status: StatusResponse | null | undefined): boolean | null {
  const available = status?.acceleration?.spark_h3_available;
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
  /** 台帳 §1-101: VRAM 16GB mode — `/generate` and, since §1-106,
   * `/generate/chain` (MiniMax H3 chains). */
  vram16_mode?: boolean;
  /** Ruling 107 (2026-11-23): pi-H3 — `/generate` and `/generate/chain`
   * (MiniMax H3 only). Never sent as `"off"`. */
  pi_h3?: PiH3Mode;
  /** 2026-11-24: Sol-H3-VAE — `/generate` and `/generate/chain` (MiniMax H3
   * only). Never sent as `false`. */
  sol_vae?: boolean;
  /** 台帳 §1-297: keep reference latents resident — `/generate` and
   * `/generate/chain` (MiniMax H3 only). Never sent as `false`. */
  keep_reference_latents?: boolean;
  /** 台帳 §1-298: text encoder conditioning cache — `/generate` and
   * `/generate/chain` (MiniMax H3, LTX 2.3, LTX 2.5), and `/generate/alpha`.
   * Never sent as `false`. */
  keep_text_conditioning?: boolean;
  /** 裁定 113: Spark-H3 — `/generate` and `/generate/chain` (MiniMax H3
   * only). Never sent as `false`; the two keys below ride along with it. */
  spark_h3?: boolean;
  /** 裁定 113: the full-attention ratio, 0.01-0.5 (`sparkH3TopkPercent / 100`). */
  spark_h3_topk_ratio?: number;
  /** 裁定 113: the full-attention steps, 0-10. */
  spark_h3_full_steps?: number;
  /** 台帳 §1-303: text encoder combined read — `/generate` and
   * `/generate/chain` (MiniMax H3 only). Never sent as `false`. */
  te_combined_read?: boolean;
}

/**
 * The seven acceleration settings as they would EFFECTIVELY run, expressed in
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
): Required<Omit<AccelerationRequestFields, "spark_h3_topk_ratio" | "spark_h3_full_steps">> {
  return {
    attention_backend: acceleration.attentionBackend === "sage" && sageAvailable !== false ? "sage" : "sdpa",
    block_swap_prefetch: acceleration.blockSwapPrefetch,
    keep_resident: acceleration.keepResident,
    fused_gguf_dequant_kernel: acceleration.fusedGgufDequantKernel,
    vae_mode: acceleration.vaeMode,
    keep_resident_embeddings: acceleration.keepResidentEmbeddings,
    vram16_mode: acceleration.vram16Mode,
    pi_h3: acceleration.piH3,
    // No served row's `requires` names `sol_vae` (Sol does not move the
    // comfort line), so this key takes no part in the match.
    sol_vae: acceleration.solVae,
    // Likewise no served row's `requires` names it (a memory cache; the
    // output does not change), so it takes no part in the match.
    keep_reference_latents: acceleration.keepReferenceLatents,
    // Likewise no served row's `requires` names it.
    keep_text_conditioning: acceleration.keepTextConditioning,
    // Likewise no served row's `requires` names it (Spark-H3 adds only
    // 10-130MB of VRAM, so it does not move the comfort line — 裁定 113); it is
    // here to satisfy the type and takes no part in the match. Its two number
    // keys are left out of the return type: a `requires` value is a string or
    // a boolean, never a number.
    spark_h3: acceleration.sparkH3,
    // Likewise no served row's `requires` names it (the output does not
    // change), so it takes no part in the match.
    te_combined_read: acceleration.teCombinedRead,
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
): AccelerationRequestFields & { embed_mp4_metadata?: false } {
  if (!acceleration) return {};
  const out: AccelerationRequestFields & { embed_mp4_metadata?: false } = {};
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
  // 台帳 §1-101 (2026-10-18): VRAM 16GB mode. The H3 server default is `true`,
  // so the key only ever appears as `false` (the `fused_gguf_dequant_kernel`
  // direction). INDEPENDENT — no capability flag, no dependency on any other
  // row. The scope half is `AppShell`'s, exactly as for
  // `keep_resident_embeddings` above: LTX publishes the name, the row is hidden
  // and the stored choice is written back to the default (`true`), so an LTX
  // request never carries the key. (LTX's reject table fires on `true` only.)
  if (acceleration.vram16Mode !== VRAM16_MODE_SERVER_DEFAULT) {
    out.vram16_mode = acceleration.vram16Mode;
  }
  // Ruling 107 (2026-11-23): pi-H3. The server default is `"off"`, so the key
  // only ever appears as `"4steps"` / `"2steps"` (the `vae_mode` direction).
  // INDEPENDENT — no capability flag, no dependency on any other row. The
  // scope half is `AppShell`'s, as for `vram16_mode` above: LTX publishes the
  // name, the row is hidden and the stored choice is written back to `"off"`.
  // While on, the three sampling values are dropped by `samplingRequestFields`
  // (the server fills them with the choice's), not here.
  if (acceleration.piH3 !== PI_H3_DEFAULT) {
    out.pi_h3 = acceleration.piH3;
  }
  // 2026-11-24: Sol-H3-VAE. The server default is `false`, so the key only
  // ever appears as `true` (the `keep_resident` direction). Never together
  // with `pi_h3` above: the "Fast VAE" row writes both through
  // `fastVaeSettings` and the stored pair is folded on read. The scope half
  // is `AppShell`'s, as for `pi_h3`.
  if (acceleration.solVae !== SOL_VAE_SERVER_DEFAULT) {
    out.sol_vae = acceleration.solVae;
  }
  // 台帳 §1-297: keep reference latents resident. The server default is
  // `false`, so the key only ever appears as `true` (the `keep_resident`
  // direction). INDEPENDENT — no capability flag, no dependency on any other
  // row. The scope half is `AppShell`'s, as for `sol_vae`: LTX publishes the
  // name, the row is hidden and the stored choice is written back to `false`.
  if (acceleration.keepReferenceLatents !== KEEP_REFERENCE_LATENTS_SERVER_DEFAULT) {
    out.keep_reference_latents = acceleration.keepReferenceLatents;
  }
  // 台帳 §1-298: text encoder conditioning cache. Same direction and same scope
  // half as `keep_reference_latents`: the key only ever appears as `true`, and
  // `AppShell` hides the row and writes the stored choice back to `false` on a
  // server that still names the feature in `unsupported_features`.
  // INDEPENDENT of `keep_reference_latents`.
  if (acceleration.keepTextConditioning !== KEEP_TEXT_CONDITIONING_SERVER_DEFAULT) {
    out.keep_text_conditioning = acceleration.keepTextConditioning;
  }
  // 裁定 113: Spark-H3. The server default is `false`, so the three keys appear
  // only while it is ON, and then all three together; while OFF not one key is
  // added, so the body stays exactly what it was before Spark-H3 existed. The
  // ratio goes out as a fraction (`percent / 100`), unrounded beyond that. The
  // scope half is `AppShell`'s, as for `sol_vae`.
  if (acceleration.sparkH3) {
    out.spark_h3 = true;
    out.spark_h3_topk_ratio = acceleration.sparkH3TopkPercent / 100;
    out.spark_h3_full_steps = acceleration.sparkH3FullSteps;
  }
  // 台帳 §1-303: text encoder combined read. Same direction and same scope half
  // as `keep_reference_latents`: the key only ever appears as `true`, and
  // `AppShell` hides the row and writes the stored choice back to `false` on
  // LTX. INDEPENDENT of every other row.
  if (acceleration.teCombinedRead !== TE_COMBINED_READ_SERVER_DEFAULT) {
    out.te_combined_read = acceleration.teCombinedRead;
  }
  // §3-164 (2026-09-24): the Output row. The server default is `true`, so the
  // key only ever appears as `false`. Emitted last, for the same key-order
  // reason as the rows above.
  if (acceleration.embedMp4Metadata !== EMBED_MP4_METADATA_SERVER_DEFAULT) {
    out.embed_mp4_metadata = false;
  }
  return out;
}

/**
 * Spark-H3's first generate-button note (裁定 113 ③, MINIMAX_H3_DESIGN §6.11):
 * Spark-H3 is on and the screen's total step count is at or below
 * "full-attention steps", so every step runs full attention and Spark-H3 never
 * acts. Equal counts too: then not one step is left for it.
 *
 * The total is what the screen shows: `fixedSampling` (the steps the Fast VAE
 * row's pi-H3 choice fixes, `samplingValues.ts`'s `piH3FixedSampling`) when it
 * is set, else the sampling sliders' `sampling`. `null` for both (an engine
 * with no step slider) = no note. "Full-attention steps" 0 never shows it:
 * the step count is at least 1 (the served `num_inference_steps.min`, and
 * pi-H3's 4 / 2), so the comparison alone already answers false.
 *
 * Scope is not checked here: on an engine that hides the row (LTX), `AppShell`
 * has already written `sparkH3` back to `false`. Whether this environment can
 * run Spark-H3 (`spark_h3_available`) is deliberately not part of the rule.
 * Informational only — never a reason to disable Generate.
 */
export function sparkH3StepsTooFew(
  acceleration: AccelerationSettings | undefined,
  sampling: { numInferenceSteps: number } | null,
  fixedSampling: { numInferenceSteps: number } | null,
): boolean {
  if (!acceleration?.sparkH3) return false;
  const totalSteps = (fixedSampling ?? sampling)?.numInferenceSteps;
  return totalSteps !== undefined && totalSteps <= acceleration.sparkH3FullSteps;
}

/**
 * Spark-H3's second generate-button note (裁定 113 ③, MINIMAX_H3_DESIGN §6.11):
 * Spark-H3 is on and at least one of `rows` — the sequence estimate of the job,
 * or of each clip of a Chained job (`shell/h3SequenceRows.ts`'s
 * `h3SequenceRows`) — is shorter than {@link SPARK_H3_MIN_TOKENS}, below which
 * the engine runs full attention. A `null` entry (a size or frame count the
 * estimate cannot read) counts as no answer and never shows the note.
 *
 * The estimate leaves out the prompt's own tokens, so right at the line the
 * note can show while the engine still runs Spark-H3 (MINIMAX_H3_DESIGN §6.11).
 * Same scope and availability remarks as {@link sparkH3StepsTooFew}.
 */
export function sparkH3VideoTooShort(
  acceleration: AccelerationSettings | undefined,
  rows: readonly ({ total: number } | null)[],
): boolean {
  if (!acceleration?.sparkH3) return false;
  return rows.some((row) => row !== null && row.total < SPARK_H3_MIN_TOKENS);
}
