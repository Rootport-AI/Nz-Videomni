/**
 * REST payload types for the LTX23 backend, transcribed from
 * Docs/API_REFERENCE.md. Kept separate from `bridge/types.ts` (the WebUI<->
 * native transport contract) — these describe the *body* that travels inside
 * a `backend.request` call, opaque to the bridge itself.
 *
 * Deliberately excluded from `GenerateRequest`/`GenerateChainRequest`
 * (Docs/API_REFERENCE.md §7, "UIに出してはいけない機能"): `guidance_scale`,
 * `num_inference_steps`, `pipeline`. The server fills its own defaults
 * (distilled / 8 steps / CFG 1.0) when these are omitted, and the WebUI must
 * never expose controls for them. `negative_prompt` was in this list too
 * until 2026-07-28, when NAG (Normalized Attention Guidance, backend commit
 * 2ae497b) made a negative prompt actually take effect without CFG — it is
 * now allowed, but ONLY alongside `nag_enabled`/`nag_scale`/`nag_tau`/
 * `nag_alpha`/`neg_method`/`vsf_scale` (see `shell/nagSettings.ts`'s
 * `nagRequestFields`, the single place that builds this set of 7); never
 * send `negative_prompt` on its own.
 *
 * Acceleration (2026-07-31, backend §43) follows the same additive rule:
 * `attention_backend` is built exclusively by `shell/accelerationSettings.ts`'s
 * `accelerationRequestFields` and is omitted entirely while the server default
 * (`"sdpa"`) is selected. Every sibling in that block follows the same rule and
 * every one of them is REAL as of 2026-08-05, when `vae_mode` — the last mock —
 * went live with backend §52. (`fused_gguf_dequant_gemm` used to sit beside
 * them; it was removed from both sides on 2026-08-04 and replaced by the REAL
 * `fused_gguf_dequant_kernel`.)
 */

export interface CropOutput {
  width: number;
  height: number;
}

/** One I2V conditioning keyframe (Docs/API_REFERENCE.md §5.1 `ConditioningImage`,
 * `api/models.py:40-45`). `image_id` comes from `POST /upload/image`'s
 * response. `frame_idx==0` is the leading frame (latent-replace); anything
 * greater is snapped by the server to the 8n+1 grid
 * (`(frame_idx-1)//8*8+1`) and clamped to range — the WebUI may submit the
 * user's natural value and only needs to *preview* the snap client-side (see
 * `modes/single/keyframeUtils.ts`). `crf` is intentionally omitted: never
 * exposed by this WebUI (M4 scope). */
export interface ConditioningImage {
  image_id: string;
  frame_idx: number;
  strength: number;
}

/** One LoRA reference (Docs/API_REFERENCE.md §5.3 `LoraSpec`,
 * `api/models.py:47-67`). `name` must be a server-registered LoRA name (see
 * `GET /loras`) — never a filesystem path; `/`, `\`, and `..` are rejected.
 * `strength` is clamped 0.05-2.0 server-side, and identically client-side
 * before it's ever sent (`lora/loraTags.ts`, parsed from the shared
 * prompt's `<lora:name:strength>` tags — this array is never editable
 * directly by the user). */
export interface LoraSpec {
  name: string;
  strength: number;
  /** 音声側の重み（0.0〜2.0、省略時は`strength`の映像側の重みに追従する）。
   * `0`は音声側の重みを一切適用しない（音声側のLoRA差分をスキップする）こと
   * を意味し、明示的に有効な値として送る必要がある — `undefined`/`null`と
   * `0`は区別される（`api/models.py`の`audio_strength: float | None`、
   * `ge=0.0, le=2.0`）。内部表現もsnake_caseのまま統一している（変換層を作
   * らない） — `lora/loraTags.ts`のプロンプトタグ第3位置引数
   * `<lora:name:映像強度:音声強度>`からパースされる。音声側の重み差分を持
   * たないLoRAでは無効（no-op）。 */
  audio_strength?: number;
}

/**
 * Outpainting（動画キャンバス拡張）の指定 —— 元動画の上下左右に何ピクセル
 * 描き足すか（設計方針書 `Docs/OUTPAINTING_DESIGN_NOTES.md` §1-13）。
 *
 * The canvas-extension spec for `POST /generate` (Edit タブ / Outpainting).
 * All four pads are pixel counts >= 0, and at least one of them must be
 * positive. The server cross-checks them against the reference video itself
 * (ffprobe) and 422s on any mismatch, so the WebUI must send pads that satisfy:
 *
 *  - `width  - pad_left - pad_right  === 元動画の幅`
 *  - `height - pad_top  - pad_bottom === 元動画の高さ`
 *  - `width`/`height` are the EXTENDED canvas and must be multiples of 128
 *    (設計書 §3-7 — the existing reference-video rule, inherited as-is)
 *  - the kept region (i.e. the source video's own size) is at least 256px on
 *    both axes
 *
 * Server-side companions: `outpaint` requires `reference_video_id` plus exactly
 * one control-kind LoRA (`"in-outpainting"`) in `loras`, and is mutually
 * exclusive with both `conditioning_images` and `crop_output`.
 *
 * The 詰め方向 (alignment) the panel offers is NOT part of this contract — it is
 * a front-end convenience that folds down into these four numbers.
 */
export interface OutpaintSpec {
  pad_left: number;
  pad_right: number;
  pad_top: number;
  pad_bottom: number;
  /**
   * マスクブラー（ラプラシアン・ピラミッド合成の膨張段数）。stage 1 は半分の
   * 解像度、stage 2 は元の解像度で適用される。`api/models.py` の `OutpaintSpec`
   * が正本で、どちらも `0..15`、既定は 5 / 2（公式ワークフローの node 5266 /
   * 5226 と同じ値）。省略すればサーバー側の既定値がそのまま使われる。
   *
   * 実寸のなじみ幅（ピクセル）は `段数 × キャンバス長辺 ÷ 64` —— マスクを長辺
   * 64px まで縮めてから膨張させ、元解像度へ戻す実装だから。既定の 5 段は
   * 1920px 幅なら約 150px にあたる（`modes/edit/outpaintGeometry.featherWidthPx`）。
   */
  blend_dilation_stage1?: number;
  blend_dilation_stage2?: number;
  /** 元動画の音声をそのまま残すか。既定 `true`（サーバー側の既定値と同じ）。 */
  freeze_source_audio?: boolean;
}

/** Fields the WebUI is allowed to submit for a `/generate` call (M2 T2V scope,
 * extended by M4 for I2V, M5 for style LoRAs). Omitting `conditioning_images`
 * (or passing `[]`) is T2V; 1-5 entries is I2V (Docs/API_REFERENCE.md §5.1). */
export interface GenerateRequest {
  prompt: string;
  width: number;
  height: number;
  crop_output?: CropOutput | null;
  num_frames: number;
  frame_rate: number;
  seed: number;
  conditioning_images?: ConditioningImage[];
  loras?: LoraSpec[];
  /** IC-LoRA (A3 "reference video") conditioning source: the `video_id`
   * returned by `POST /upload/video` for the selected reference clip
   * (Docs/API_REFERENCE.md §2, §7-3-A). Omitted for a plain T2V/I2V request;
   * the right-click `referenceVideo` flow sets it from the reference video
   * the user uploads in Create's IC-LoRA block. `GenerateChainRequest` grew
   * an analogous field 2026-07-11 (owner decision, alpha scope: clips=1
   * only) — see there. */
  reference_video_id?: string | null;
  /** 0.0-1.0; requires `loras` server-side (mirrors
   * `GenerateChainRequest.conditioning_attention_strength`). `null`/`undefined`
   * omits the field — `0` is a valid, distinct value that must still be sent. */
  conditioning_attention_strength?: number | null;
  /** 0.0-1.0; requires `loras` server-side (mirrors
   * `GenerateChainRequest.reference_video_strength`). Same `!= null` (not
   * truthy) omission rule as {@link GenerateRequest.conditioning_attention_strength}. */
  reference_video_strength?: number | null;
  /** NAG (2026-07-28, backend commit 2ae497b): additive contract, built
   * exclusively by `shell/nagSettings.ts`'s `nagRequestFields` and spread in
   * — omitted entirely when NAG is off, never sent individually. See that
   * module for the field meanings/defaults. */
  negative_prompt?: string;
  nag_enabled?: boolean;
  nag_scale?: number;
  nag_tau?: number;
  nag_alpha?: number;
  neg_method?: "nag" | "vsf";
  vsf_scale?: number;
  /** Acceleration (2026-07-31, backend §43): which attention implementation
   * the job runs with. Built exclusively by
   * `shell/accelerationSettings.ts`'s `accelerationRequestFields` and spread
   * in — omitted entirely while the server default (`"sdpa"`) is selected. */
  attention_backend?: "sdpa" | "sage";
  /** Acceleration (2026-08-01, backend §44): async block-swap prefetch. Built
   * exclusively by `shell/accelerationSettings.ts`'s `accelerationRequestFields`
   * and spread in — omitted entirely while the server default (`false`) is
   * selected. Unlike `attention_backend`, the output is bit-identical on/off. */
  block_swap_prefetch?: boolean;
  /** Acceleration (2026-08-02, backend §48): keep the models' CPU-side
   * skeleton resident between jobs. Built exclusively by
   * `shell/accelerationSettings.ts`'s `accelerationRequestFields` and spread
   * in — omitted entirely while the server default (`false`) is selected, so
   * unlike `block_swap_prefetch` this key only ever appears as `true`. The
   * output is bit-identical on/off; the cost is ~20GB of resident RAM. */
  keep_resident?: boolean;
  /** Acceleration (2026-08-04, backend §51): run the GGUF K-quant
   * dequantization as one fused Triton kernel. Built exclusively by
   * `shell/accelerationSettings.ts`'s `accelerationRequestFields` and spread
   * in — omitted entirely while the server default (`false`) is selected, so
   * like `keep_resident` this key only ever appears as `true`. The output is
   * bit-identical on/off; the backend falls back silently on any failure and
   * records `fused_gguf_dequant_kernel_used` in the job metadata. */
  fused_gguf_dequant_kernel?: boolean;
  /** Acceleration (2026-08-05, backend §52): which VAE decoder reconstructs
   * the video. `"prune_vaed"` picks PrunaVAED, the pruned decoder. Built
   * exclusively by `shell/accelerationSettings.ts`'s `accelerationRequestFields`
   * and spread in — omitted entirely while the server default (`"default"`) is
   * selected, so like `keep_resident` this key only ever appears as
   * `"prune_vaed"`. UNLIKE the three toggles above the output is NOT
   * bit-identical (it is a different decoder); the backend degrades per job to
   * the usual decoder and records `vae_mode_used` when the pruned weights are
   * missing. Unrelated to the server's existing `vae_tiling` VRAM option. */
  vae_mode?: "default" | "prune_vaed";
  /** Outpainting (2026-08-09, 台帳 §1-13): the canvas-extension spec. Built
   * exclusively by `modes/edit/useOutpaintForm.ts` and omitted entirely by
   * every other caller, so an ordinary Create/Chain request stays byte-identical
   * to before. See {@link OutpaintSpec} for the constraints the WebUI must
   * satisfy before sending one. */
  outpaint?: OutpaintSpec | null;
}

export interface GenerateAcceptedResponse {
  job_id: string;
  status: "queued";
  created_at: string;
}

/** One clip in a `/generate/chain` request (Docs/API_REFERENCE.md §5.2
 * `ChainClip`, `api/models.py:204-216`). `prompt` overrides the chain-wide
 * common prompt for this clip only when present (omitted/empty means "use
 * the shared prompt"). `conditioning_images` is only ever populated for clip
 * 0 (M6 scope: "clip 0のみ画像を持てる") — the WebUI never attaches it to any
 * other clip. */
export interface ChainClip {
  prompt?: string;
  num_frames: number;
  conditioning_images?: ConditioningImage[];
}

/** V2V continuation source (Docs/API_REFERENCE.md §5.2 `SourceVideoSpec`).
 * `context_frames` must be 8n+1 in `[v2v_context_frames_min,
 * v2v_context_frames_max]` and strictly less than `clips[0].num_frames`
 * (`api/models.py:219-259`). */
export interface SourceVideoSpec {
  video_id: string;
  context_frames: number;
}

/**
 * §1-17 Retake（選択範囲の撮り直し）の指定。`GenerateChainRequest.retake` に
 * 載せると、`clips[0].num_frames` の長さの窓を素材の `window_start_sec` から
 * 切り出し、その**両端を凍結したまま中身だけ**を作り直す。
 *
 * 3 フィールドしか無いのは意図的で、糊代（`head_px`/`tail_px`）と
 * `stage2_window` は**送らない** —— サーバ既定（頭25/尾24、`"standard"`）に
 * 自動追随させるため。糊代は実測で較正された値であり、UI に出す価値のある
 * つまみではない（RangeBand が「触れない領域」として見せるだけ）。
 * `stage2_window="high_resolution"` との併用はサーバが 422 で弾く。
 * `stage2_window="full_length"` との併用も同様にサーバが 422 で弾く（§1-19）。
 */
export interface RetakeSpec {
  /** `POST /upload/video` の `video_id`。 */
  video_id: string;
  /**
   * 窓の開始秒。**アップロードしたファイルの時間軸**で測る —— §1-6 のトリム付き
   * アップロードなら「切り出し後ファイルの先頭からの秒」であって、元ファイルの
   * 秒ではない。この変換（素材秒 − `decideSourceTrim` の `startSec`）は
   * `modes/edit/useRetakeForm.ts` の `buildRequest` が 1 箇所で行う。
   */
  window_start_sec: number;
  /** 音声も作り直すか。`false` なら元の波形をそのまま乗せ直す。 */
  regenerate_audio: boolean;
}

/**
 * 素材（末尾）の指定。`GenerateChainRequest.end_source` に載せると、チェーン
 * 全体の**最後尾**が指定素材そのもので終わる動画を生成する（Retake と同じ
 * ハード凍結機構をチェーンの末尾へ適用する）。`video_id` / `image_id` は
 * ちょうど一方だけを送る（画像はバックエンドが静止動画化してから、動画と
 * 完全に同一の経路を通す）。
 *
 * サーバには**3 つのモード**があり、`clips` の本数だけで決まる
 * （`Docs/API_REFERENCE.md` §5.2）。**UI が使うのは窓内モードと逆順モードの
 * 2 つ**（旧方式は API 上は生き残っているが到達不能な死蔵経路）。
 *
 * - **窓内モード（クリップ 1 本）**: 凍結フレームは**クリップ自身の末尾**に置か
 *   れ、stage-1 の 1 つのデノイズ窓の中で素材へ到達する。**出力の長さは変わら
 *   ない**（`chainUtils.computeOutputFrames` の結果がそのまま配信尺）。
 * - **逆順モード（クリップ 2 本以上、2026-08-18・第2段階）**: 凍結フレームは
 *   タイムライン**最終クリップ自身の末尾**に置かれる点は窓内モードと同じ（何も
 *   継ぎ足さない）が、Stage-1 はタイムラインと**逆順**（末尾クリップから先頭
 *   クリップへ）に生成し、各セグメントの尾が次に生成する（タイムライン上は
 *   前の）セグメントの頭へキャリーされる。**出力の長さは窓内モードと同じ式**
 *   （`total_px == clips_total_px`）。`overlap_frames`（K_v）は 1 潜在まで許容
 *   （窓内モードは 2 以上を要求——`useChainForm` の `endSourceNeedsOverlap`）。
 * - **旧方式（内部区画方式）・非推奨・死蔵**: 帯をクリップ列の後ろへ継ぎ足す方式
 *   で、出力尺は「クリップ合計＋帯」になる。素材へクロスフェードで接続する既知
 *   の問題があり、UI からは到達できない（テスト／切り戻し専用の内部フラグ経由
 *   でのみ再現可能）。
 *
 * - **`context_frames` は 8 の倍数**（[8, 136]、既定 72）。冒頭 `SourceVideoSpec`
 *   の 8n+1 とは**非対称**で、これは因果 VAE の末尾グリッドがそうなっている
 *   ため（冒頭の +1 はキーフレーム、末尾側にはそれが無い）。UI は常に 8 を送る
 *   （`timeline/tailAlign.ts` の `END_SOURCE_CONTEXT_FRAMES`）。
 * - 素材からは `context_frames + 1` フレームが切り出され、その 1 枚目は因果 VAE
 *   のプライマとして消費される。つまり出力末尾に現れるのは素材の 2 フレーム目
 *   以降で、素材には最低 9 フレーム必要（`END_SOURCE_MIN_FRAMES`）。
 * - `retake` / `source_audio` / `reference_video_id` とは**排他**。
 *   `source_video`（冒頭素材）とは**クリップ1本のときのみ併用可**（2本以上は422 —
 *   `useChainForm` の `endSourceWithSourceVideoMultiClip`）。`clips[0].conditioning_images`
 *   とは併用可。
 */
export interface EndSourceSpec {
  /** `POST /upload/video` の `video_id`。`image_id` とはどちらか一方のみ。 */
  video_id?: string;
  /** `POST /upload/image` の `image_id`。`video_id` とはどちらか一方のみ。 */
  image_id?: string;
  /** 末尾を素材そのものにするフレーム数（8 の倍数、[8, 136]）＝錨の長さ。
   *
   * 窓内モード（2026-08-17）では**クリップの内側**の末尾フレームが凍結される
   * ので、この値は出力尺を伸ばさない。UI の設定項目ではなく、**常に 8 固定**を
   * 明示送信する（`timeline/tailAlign.ts` の `END_SOURCE_CONTEXT_FRAMES`。
   * 長い錨ほど素材へのクロスフェードが起きやすいという実機比較の結論）。 */
  context_frames: number;
  /** 錨の固定強度（[0, 1]、既定 1.0、バッチ1・2026-08-18）。1.0 ＝ハード凍結
   * （既定・従来と完全同値）で、下げると **Stage-1 だけ** マスク値が緩む
   * （`1.0 - strength`）。**Stage-2 は常にハード凍結**なので、`strength` の値に
   * かかわらず最終フレームは常に素材どおりになる——緩めているのは Stage-1 の
   * 途中経過（素材への「なじみ方」）だけ。`overlap_strength`（生成物同士の継ぎ
   * 目のブレンド）とは別物。UI は既定値でも常に明示送信する（`chainUtils.ts`
   * の `DEFAULT_END_SOURCE_STRENGTH`）。 */
  strength: number;
}

/** A2V source audio (Docs/API_REFERENCE.md §5.2). Mutually exclusive with
 * `source_video` — the WebUI enforces this by construction via the Chain
 * screen's sub-mode selector rather than validating it after the fact. */
export interface SourceAudioSpec {
  audio_id: string;
}

/** Fields the WebUI is allowed to submit for a `/generate/chain` call (M6
 * scope: clip chaining / V2V continuation / A2V; extended for style/control
 * IC-LoRAs and chunked upsampling). The backend started accepting `loras` on
 * chain requests 2026-07-03 and fully unblocked it (reference-video CONTROL
 * IC-LoRA too) 2026-07-11 — `<lora:name:strength>` tags found in the shared
 * prompt are parsed into the `loras` array and sent (the tag syntax itself is
 * never sent to the API; see `modes/chained/chainUtils.ts`). */
export interface GenerateChainRequest {
  prompt: string;
  width: number;
  height: number;
  crop_output?: CropOutput | null;
  frame_rate: number;
  seed: number;
  overlap_frames: number;
  overlap_strength: number;
  clips: ChainClip[];
  source_video?: SourceVideoSpec | null;
  source_audio?: SourceAudioSpec | null;
  /** Style/character or control IC-LoRAs applied uniformly to every clip and
   * every stage of the chain (Docs/API_REFERENCE.md §5.2, `api/models.py`
   * around line 344). Same `LoraSpec` type/validation as
   * `GenerateRequest.loras`; there are no per-clip strengths in v1. Parsed
   * from the shared prompt's `<lora:name:strength>` tags, same as
   * `GenerateRequest.loras` (see `modes/chained/chainUtils.ts`). */
  loras?: LoraSpec[];
  /** Opt into temporal-chunk upsampling (halo overlap + CPU offload) instead
   * of one whole-timeline GPU upsample pass, trading time for a flat VRAM
   * ceiling so long/high-res chains fit in 16GB (Docs/API_REFERENCE.md §5.2,
   * `api/models.py` around line 361-367). Server default is `false` (the old
   * one-pass path, unchanged behavior) — the WebUI must always send this
   * field explicitly (WebUI default `true`); omitting it silently falls back
   * to the one-pass upsample path permanently. */
  chunked_upsample?: boolean;
  /** Stage-2 window preset (`api/models.py` `GenerateChainRequest.stage2_window`,
   * owner decision 2026-08-09). `"standard"` is the server default and the
   * frozen 22/18 tile geometry; `"high_resolution"` switches to 19/12, which
   * costs ~14% fewer attention tokens per stage-2 window and so keeps high
   * resolutions inside the comfortable budget
   * (`shell/tokenBudget.ts`'s `CHAIN_COMFORT_TOKEN_BUDGET`) instead of spilling.
   *
   * Unlike `chunked_upsample` above, this is OMITTED at its default — the
   * ordinary optional-field treatment every other field on this interface gets,
   * so a chain that doesn't opt in sends a request byte-identical to before this
   * field existed. Named for the geometry, not a duration: the advance is a
   * latent-frame count, so its wall-clock length depends on `frame_rate` (the UI
   * renders the seconds).
   *
   * `"full_length"` (§1-19) is a THIRD preset — one stage-2 window covering the
   * whole clip in a single tile (61 video-latent frames = 481 pixel frames, no
   * seams). It requires exactly 1 clip plus `source_audio` (the server 422s
   * otherwise), so it is only ever sent by the Single and Batch A2V builders
   * (`modes/batch/buildA2vChainPayload.ts`) as a fixed literal — it is not a
   * choice the interactive Chain screen offers, which is why
   * `shell/tokenBudget.ts`'s `Stage2Window` union deliberately stays at just
   * `"standard"`/`"high_resolution"`. */
  stage2_window?: "standard" | "high_resolution" | "full_length";
  /** Reference-video CONTROL IC-LoRA conditioning source, mirroring
   * `GenerateRequest.reference_video_id`: the `video_id` from
   * `POST /upload/video`. Requires at least one entry in `loras`; mutually
   * exclusive with `source_video` (Docs/API_REFERENCE.md §5.2, `api/models.py`).
   *
   * §1-15 (2026-08-11): the old "the chain must be exactly 1 clip" limit is
   * GONE. ONE long reference video is sent for the WHOLE chain and the server
   * slices it per clip by FRAME NUMBER (`chain_math.video_segment_windows`),
   * injecting each clip's own window into its stage-1 pass. A reference shorter
   * than the chain is not an error: the clips it does not reach are generated
   * without one. The upload is capped at `MAX_CHAIN_TOTAL_FRAMES` (11544)
   * frames, which is exactly what a maximal chain can consume. */
  reference_video_id?: string | null;
  /** 0.0-1.0; requires `loras` (Docs/API_REFERENCE.md §5.2, mirrors
   * `GenerateRequest.conditioning_attention_strength`). */
  conditioning_attention_strength?: number | null;
  /** 0.0-1.0; requires `loras` (Docs/API_REFERENCE.md §5.2, mirrors
   * `GenerateRequest.reference_video_strength`). */
  reference_video_strength?: number | null;
  /** NAG (2026-07-28, backend commit 2ae497b): additive contract, built
   * exclusively by `shell/nagSettings.ts`'s `nagRequestFields` and spread in
   * — omitted entirely when NAG is off, never sent individually. See that
   * module for the field meanings/defaults. */
  negative_prompt?: string;
  nag_enabled?: boolean;
  nag_scale?: number;
  nag_tau?: number;
  nag_alpha?: number;
  neg_method?: "nag" | "vsf";
  vsf_scale?: number;
  /** Acceleration (2026-07-31, backend §43): mirrors
   * `GenerateRequest.attention_backend` — same single builder
   * (`shell/accelerationSettings.ts`'s `accelerationRequestFields`), same
   * "omitted while `sdpa`" rule. */
  attention_backend?: "sdpa" | "sage";
  /** Acceleration (2026-08-01, backend §44): mirrors
   * `GenerateRequest.block_swap_prefetch` — same single builder
   * (`shell/accelerationSettings.ts`'s `accelerationRequestFields`), same
   * "omitted while server default" rule. */
  block_swap_prefetch?: boolean;
  /** Acceleration (2026-08-02, backend §48): mirrors
   * `GenerateRequest.keep_resident` — same single builder
   * (`shell/accelerationSettings.ts`'s `accelerationRequestFields`), same
   * "omitted while server default (`false`)" rule. */
  keep_resident?: boolean;
  /** Acceleration (2026-08-04, backend §51): mirrors
   * `GenerateRequest.fused_gguf_dequant_kernel` — same single builder
   * (`shell/accelerationSettings.ts`'s `accelerationRequestFields`), same
   * "omitted while server default (`false`)" rule. */
  fused_gguf_dequant_kernel?: boolean;
  /** Acceleration (2026-08-05, backend §52): mirrors
   * `GenerateRequest.vae_mode` — same single builder
   * (`shell/accelerationSettings.ts`'s `accelerationRequestFields`), same
   * "omitted while server default (`"default"`)" rule. */
  vae_mode?: "default" | "prune_vaed";
  /** §1-17 Retake（選択範囲の撮り直し）。指定すると `clips` はちょうど 1 本
   * （その `num_frames` が窓の長さ＝窓長の単一ソース）でなければならず、
   * `source_video`/`source_audio`/`reference_video_id` とは排他。
   * 省略時はこれまでと 1 バイトも変わらない普通の chain リクエスト。 */
  retake?: RetakeSpec | null;
  /** 素材（末尾）。指定するとチェーンの最後尾が素材そのもので終わる。
   * `retake`/`source_audio`/`reference_video_id` とは排他、`source_video` とは
   * クリップ 1 本のときのみ併用可（2 本以上は 422）。省略時はこれまでと
   * 1 バイトも変わらない（additive 契約）。詳細は {@link EndSourceSpec}。 */
  end_source?: EndSourceSpec | null;
}

/** `POST /generate/chain`'s 202 response (Docs/API_REFERENCE.md §3.13) —
 * same shape as `GenerateAcceptedResponse` plus `num_clips`. */
export interface GenerateChainAcceptedResponse {
  job_id: string;
  status: "queued";
  created_at: string;
  num_clips: number;
}

/** `POST /jobs/{id}/join`'s optional body (Docs/API_REFERENCE.md §3.17). The
 * WebUI always sends `{}` (server defaults: `audio_smoothing=true`,
 * `handle_crossfade_ms=300`) — no UI control exposes these knobs in M6. */
export interface JoinRequestBody {
  audio_smoothing?: boolean;
  handle_crossfade_ms?: number;
  /** V2V tail-keep: keep only this many trailing seconds of the uploaded
   * source before concatenating the continuation (server default 5.0). `0`
   * explicitly joins the source at full length. Clamped `>= 0` server-side
   * (`JoinRequest.source_tail_seconds`, `api/models.py`). */
  source_tail_seconds?: number;
}

/** `POST /jobs/{id}/join`'s response (Docs/API_REFERENCE.md §3.17,
 * `api/models.py:527-546`). Only the fields the WebUI actually reads are
 * declared; the real response has more (omitted here, exactly like
 * `AppConfig` above). */
export interface JoinResponse {
  job_id: string;
  joined_path: string;
  join_mode: string;
  source_normalized: boolean;
  /** Seconds of the source dropped from its head by the tail-keep trim (=
   * source full duration - kept duration), i.e. the offset at which the
   * continuation begins in the joined timeline. `0.0` when no trim ran
   * (`source_tail_seconds=0`, or the source was already at/under that length).
   * (`JoinResponse.trimmed_source_seconds`, `api/models.py`). */
  trimmed_source_seconds: number;
  /** The source's measured fps (ffprobe). `null` when it could not be probed
   * (`JoinResponse.source_fps`, `api/models.py`). */
  source_fps: number | null;
}

export type JobStatus = "queued" | "running" | "completed" | "failed" | "cancelled";

export interface JobResult {
  video_url: string;
  duration_seconds: number;
  resolution: string;
  file_size_bytes: number;
  generation_time_seconds: number;
  seed_used: number;
  output_path: string;
  metadata_path: string;
}

export interface JobResponse {
  job_id: string;
  status: JobStatus;
  progress: number | null;
  current_step: number | null;
  total_steps: number | null;
  stage: string | null;
  clip: number | null;
  clip_count: number | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  error: string | null;
  /** True when the job is a chain continuation of an uploaded source video
   * (it carried a `source_video` spec) — the only jobs `POST /jobs/{id}/join`
   * accepts, and the gate the WebUI uses to show the Join controls
   * (`jobs/joinUtils.ts`). The server derives this from the job's chain
   * request; it is NOT recoverable from {@link JobResponse.request}, which is
   * the whitelisted per-clip `GenerateRequest` and never carries
   * `source_video` (`JobResponse.is_v2v`, `services/job_store.py`). */
  is_v2v: boolean;
  /** True when a `joined.mp4` currently exists next to the job output (a join
   * has run and not been deleted); `false` otherwise (`JobResponse.joined`,
   * file-existence based, `services/job_store.py`). */
  joined: boolean;
  /** Echoed per-clip `GenerateRequest` (`to_clip_request` server-side): the
   * fields of {@link GenerateRequest} plus the server-only ones
   * (negative_prompt/num_inference_steps/guidance_scale/pipeline) the WebUI
   * never renders. Chain-only fields (source_video/source_audio/clips/...) are
   * NOT present here — use {@link JobResponse.is_v2v} to detect a V2V job. */
  request: Record<string, unknown>;
  result: JobResult | null;
}

export interface DeleteJobResponse {
  job_id: string;
  /** Present when the job had already finished: record + output dir removed. */
  deleted?: boolean;
  /** Present when the job was still in flight: best-effort cancel requested,
   * it will settle on `cancelled` after the current step finishes. */
  cancel_requested?: boolean;
  status?: JobStatus;
}

export interface GpuInfo {
  available: boolean;
  name: string | null;
  vram_total_mb: number;
  vram_used_mb: number;
  vram_free_mb: number;
}

export interface QueueInfo {
  mode: string;
  pending: number;
  running: number;
  completed: number;
  failed: number;
}

export interface StatusResponse {
  server: string;
  version: string;
  host: string;
  port: number;
  pipeline_loaded: boolean;
  pipeline_type: string | null;
  gpu: GpuInfo;
  queue: QueueInfo;
  /** Optional (defensive): the pipeline's own state machine, one of
   * `"unloaded" | "loading" | "ready" | "running" | "error"`
   * (`services/pipeline_manager.py`'s `STATE_*` constants, surfaced by
   * `GET /status` since the multi-engine groundwork §3-97 P6). Typed as a bare
   * `string` on purpose — an unknown future value must not become a type
   * error, and every reader compares against a literal it knows.
   *
   * Why it exists: `pipeline_loaded` is `false` BOTH while nothing is loaded
   * and while a load is in flight, so before this field the WebUI could not
   * tell "idle" from "busy loading a base model" and showed the plain
   * "server online" badge for the several minutes a swap takes
   * (`Docs/MULTI_ENGINE_DESIGN.md` §5.5(b)/§6.5). `modes/single/useServerStatus.ts`
   * is the single reader; absent (older backend) simply means the old
   * behaviour, never an error. */
  state?: string;
  /** Optional (defensive): the id of the base model the pipeline is currently
   * on (`"LTX23"`, …), retained across an unload. Same additive/optional rule
   * as {@link StatusResponse.state}. `shell/useBaseModels.ts` drives the
   * header dropdown off `GET /models` instead (which carries the selectable
   * list too), so this is currently diagnostic/forward-compat only. */
  base_model?: string | null;
  /** Optional (defensive): the acceleration capability block (backend §43,
   * 2026-07-31). Absent from every backend older than that commit, so every
   * reader must treat it — and each field inside it — as optional and never
   * assume presence; `shell/accelerationSettings.ts`'s `sageAvailability` is
   * the single reader, and it maps "missing" to `null` ("unknown"), NOT to
   * `false`. Same optional-section shape as `AppConfig.model` below. Note
   * `StatusResponse` is itself only the part of `/status` this WebUI reads
   * (the real response also carries e.g. `vram_optimization`). */
  acceleration?: {
    /** Every backend this build knows sends `["sdpa", "sage"]`; the WebUI
     * renders its own fixed two buttons rather than driving them off this
     * list, so it is here for diagnostics/forward-compat only. */
    attention_backends?: string[];
    sage_available?: boolean;
    /** Optional (backend §44, 2026-08-01): whether the server is running with
     * block-swap prefetch available (same predicate as the real per-job
     * gate). Absent from every backend older than §44; treated as "unknown"
     * by `shell/accelerationSettings.ts`'s `blockSwapPrefetchAvailability`,
     * same as a missing `sage_available`. */
    block_swap_prefetch_available?: boolean;
  };
}

export interface GenerationPreset {
  width: number;
  height: number;
  crop_output: CropOutput | null;
  num_frames: number;
}

export interface GenerationDefaults {
  width: number;
  height: number;
  crop_output: CropOutput | null;
  num_frames: number;
  frame_rate: number;
  seed: number;
}

export interface UploadLimits {
  max_image_size_mb: number;
  allowed_image_extensions: string[];
  max_video_size_mb: number;
  allowed_video_extensions: string[];
  max_audio_size_mb: number;
  allowed_audio_extensions: string[];
}

export interface AppLimits {
  max_width: number;
  max_height: number;
  max_num_frames: number;
  max_conditioning_images: number;
  conditioning_frame_idx_multiple: number;
  conditioning_keyframe_grid_offset: number;
  phase1_max_concurrent_jobs: number;
  spill_free_frames: Record<string, number>;
  v2v_context_frames_default: number;
  v2v_context_frames_min: number;
  v2v_context_frames_max: number;
  /** §1-17 Retake の窓長の下限・上限（生成 fps 上の画素フレーム数、いずれも
   * 8n+1）。169 は「stage-2 のタイル1枚に収まる最大の画素フレーム数」という
   * 幾何そのもので、73 は自由中間潜在が成立する最小窓
   * （`outputs/retake_spike/T1_RESULTS.md` C1 の実測）。サーバの
   * `LimitsConfig` に公開されている 2 定数だけを写しており、糊代（頭25/尾24）は
   * UI 非公開なのでここには出さない。
   *
   * `timeline/retakeWindow.ts` の `RETAKE_WINDOW_MIN_PX`/`MAX_PX` はこの 2 つが
   * 読めないとき（古いサーバ・`GET /config` 失敗）のフォールバックであり、
   * 実際の判定は常にここの値が優先される。 */
  retake_window_min_frames: number;
  retake_window_max_frames: number;
  /** 素材（末尾）の `end_source.context_frames` の既定値・下限・上限
   * （いずれも**8 の倍数**。既定 72／下限 8／上限 136）。冒頭の
   * `v2v_context_frames_*` が 8n+1 なのと非対称なのは、因果 VAE の末尾グリッド
   * がそうなっているため（{@link EndSourceSpec} 参照）。
   *
   * **UI はこの 3 つを読まない**。窓内モード（2026-08-17）の錨は素材にもサーバ
   * 既定にも依らない定数で、`timeline/tailAlign.ts` の
   * `END_SOURCE_CONTEXT_FRAMES = 8`（実機比較で決めた値）を常に明示送信する。
   * 既定 72 を種にする箇所も、上限 136 と突き合わせる箇所も存在しない。
   * フィールド自体はサーバ契約として残す。 */
  end_context_frames_default: number;
  end_context_frames_min: number;
  end_context_frames_max: number;
  /** stage-2 の 1 回分が快適に収まる注意トークン数の上限（トークン =
   * （幅//32）×（高さ//32）× 潜在フレーム数）。Chained 画面の解像度スライダーに
   * 引く目安線と、速度低下の注意書きの基準になる。サーバはこの値で何も判定
   * しない純粋な助言値で、VRAM 容量に応じて `config.yaml` で上下できる。
   *
   * **オプショナル**（この鍵を持たない古いサーバがあり得るため）。読む側は必ず
   * `shell/tokenBudget.ts` の `resolveChainComfortBudget` を通すこと——欠落・0・
   * 負・NaN のときは同ファイルのミラー定数 40,000 へ落ちる。生の値を直接
   * 参照してはならない。 */
  chain_comfort_token_budget?: number;
  /** Create（単発 `/generate`）1 発が快適に収まる注意トークン数の上限（トークン
   * = （幅//32）×（高さ//32）× 潜在フレーム数）。`chain_comfort_token_budget`
   * とは別軸——あちらは Chained のstage-2 1 窓、こちらは単発 1 パス全体。混同
   * しないこと。サーバはこの値で何も判定しない純粋な助言値。
   *
   * **表を持たない古いサーバ向けの互換値**（2026-08-31）。今のサーバは
   * {@link AppLimits.comfort_budgets} でエンジン系統×条件ごとの予算を配信し、
   * こちらのスカラ鍵は互換のために残っているだけ——`comfort_budgets` が無い
   * サーバに当たったとき、`shell/comfortTable.ts` の互換シムが
   * 「5つの高速化トグル（sage・block_swap_prefetch・keep_resident・
   * fused_gguf_dequant_kernel・vae_mode=prune_vaed）が全て on」のときだけ
   * この値を単発予算として採用する（＝2026-08-31 以前と完全に同じ挙動）。
   *
   * **オプショナル**（この鍵を持たない古いサーバ対応）。読む側は必ず
   * `shell/comfortTable.ts` の `resolveSingleComfortBudget` を通すこと
   * ——欠落・0・負・NaN のときは同ファイルのミラー定数 44,880 へ落ちる。生の
   * 値を直接参照してはならない。 */
  single_comfort_token_budget?: number;
  /** 快適上限マーカーの配信テーブル（2026-08-31）。エンジン系統 id
   * （`BaseModelBlock.engine_family`。`"ltx"`／`"ltx25"`）→ そのエンジンの
   * プロファイル。将来のモデル追加・高速化トグル追加に**行の追加だけ**で
   * 対応するための骨格で、フロントは条件をハードコードしない。
   *
   * **読む側は必ず `shell/comfortTable.ts` の `resolveComfortRow` を通すこと。**
   * 生の `rows` を自前で走査してはならない——実効の高速化設定への写像
   * （sage の3値・keepResident の畳み込み）と、不正値の正規化（予算が正の
   * 有限数でなければミラー定数／`spatial_factor` < 1 → 32／
   * `temporal_factor` < 1 → 8）は全てそこに1箇所だけある。
   *
   * **一致する行が無い（＝`resolveComfortRow` が `null`）のは異常ではなく
   * 正常な状態**で、呼び手はレガシーの {@link AppLimits.spill_free_frames}
   * へ落ちる。とくに LTX 2.3 の既定構成は**意図的に**行を持たない：素の VAE
   * デコーダのせいで快適境界がトークン数に対して単調でなく（境界がデコードの
   * チャンク数増分 7→8／4→5／2→3 と一致する）、1本のトークン線では表せない
   * ため、`spill_free_frames` の実測5点が正である。「`ltx` に requires 空の
   * 行を足せば全構成で賢くなる」というのは誤り。
   *
   * **オプショナル**（この鍵を持たない古いサーバ対応）。欠落時は
   * `resolveComfortRow` の互換シムが働く——
   * {@link AppLimits.single_comfort_token_budget} の doc を参照。 */
  comfort_budgets?: Record<string, EngineComfortProfile>;
}

/** {@link EngineComfortProfile} の1行。上から順に、`requires` の**全鍵**が
 * 実効の高速化設定に一致した**最初の行**が採用される（`shell/comfortTable.ts`
 * の `resolveComfortRow`）。 */
export interface ComfortRow {
  /** サーバ語彙（リクエストのフィールド名）で書かれた適用条件。空 `{}` は
   * 「無条件」＝どの高速化設定でもこの行が当たる。フロントが知らない鍵を
   * 要求する行は**不一致**として扱われる（安全側）。 */
  requires: Record<string, string | boolean>;
  /** Create（単発 `/generate`）1 発分の快適トークン予算。 */
  single_budget: number;
  /** Chained の stage-2 1 窓分の快適トークン予算。 */
  chain_budget: number;
}

/** 1つのエンジン系統ぶんの快適予算プロファイル（{@link AppLimits.comfort_budgets}
 * の値）。係数は将来モデル用にサーバが持つ——現行の LTX 系はいずれも 32/8。 */
export interface EngineComfortProfile {
  /** 動画 VAE の空間圧縮率：`spatial_factor` × `spatial_factor` ピクセルが
   * 潜在1マス。現行は 32。 */
  spatial_factor: number;
  /** 潜在フレーム1つあたりの画素フレーム数（`8n+1` グリッドの `n` 係数）。
   * 現行は 8。 */
  temporal_factor: number;
  /** 上から順に評価される条件行。空配列は「このエンジンに賢い線は無い」＝
   * 常にレガシー表へ落ちる、という正当な状態。 */
  rows: ComfortRow[];
}

/** One entry of `AppConfig.model.ic_loras` (Docs/API_REFERENCE.md §3.2) — a
 * control IC-LoRA the server has registered, surfaced in Create's IC-LoRA
 * dropdown (N2, `modes/single/GenerationForm.tsx`). The real backend may send
 * either a bare path string (legacy) or this richer object; the WebUI only
 * ever reads the enclosing map's KEYS (the selectable name), never `path`/
 * `preprocess`, so both shapes are accepted by `AppConfig.model.ic_loras`'s
 * `string | IcLoraEntry` union below. */
export interface IcLoraEntry {
  path: string;
  preprocess: "none" | "canny" | "dwpose" | "depth";
}

/** Subset of `AppConfig.model_dump()` (`GET /config`) the WebUI reads.
 * The real response has additional sections (server/model/vram/output) that
 * the WebUI never needs and are omitted here. */
export interface AppConfig {
  generation_presets: Record<string, GenerationPreset>;
  generation_defaults: GenerationDefaults;
  limits: AppLimits;
  upload: UploadLimits;
  /** Optional (defensive): the real backend's `model` section carries more
   * than this — `ic_loras` is the only piece the WebUI reads (Create's
   * control-LoRA dropdown, N2). Absent from `FALLBACK_APP_CONFIG` and
   * possibly from older backends, so every reader must treat this as
   * optional (`config.model?.ic_loras ?? {}`), never assume presence. */
  model?: {
    ic_loras?: Record<string, string | IcLoraEntry>;
  };
  /** Optional (defensive): the real backend's `server` section carries more
   * than this — `api_key` is the only piece the WebUI reads (N13's
   * set/unset status badge, `shell/apiKeyStatus.ts`). Absent from
   * `FALLBACK_APP_CONFIG` and possibly from older backends, so every reader
   * must treat this as optional, never assume presence. The raw value must
   * never be rendered — only its presence/length (see `redactConfig.ts` for
   * the display-time redaction, and `apiKeyStatus.ts` for the presence
   * check). */
  server?: {
    api_key?: string | null;
  };
}

export interface BackendErrorDetail {
  loc: unknown[];
  msg: string;
  type: string;
}

export interface BackendErrorEnvelope {
  error: {
    code: string;
    message: string;
    job_id?: string;
    detail?: BackendErrorDetail[] | string;
  };
}

/** The model categories this WebUI knows how to render (Docs/API_REFERENCE.md
 * §3.3/§3.5, `services/model_registry.py`'s `CATEGORIES`). Since the
 * multi-engine groundwork (§3-97 P3a) the category SET is declared per base
 * model in its descriptor (`scripts/manifests/<base>.json`'s `categories`) and
 * the DISPLAY ORDER comes from the server response — see
 * `shell/useModels.ts`'s `categoryOrder`. This union is kept as the typed
 * shape of the per-category selection state. `transformer` is the
 * video-generation GGUF, `text_encoder` the Gemma GGUF, `video_vae`/`audio`
 * the safetensors VAE components (`audio` also carries the vocoder — see
 * `services/model_registry.py`'s module docstring). */
export type ModelCategory = "transformer" | "text_encoder" | "video_vae" | "audio";

/** The reserved NAME of the injected per-category default entry
 * (`services/model_registry.py:43` `DEFAULT_NAME`). Always present, always
 * selectable, never removed by a rescan. */
export const MODEL_DEFAULT_NAME = "default";

/** One row of a `GET /models` category listing (Docs/API_REFERENCE.md §3.5,
 * `services/model_registry.py:102-119` `ModelEntryInfo.as_dict`). `path` is a
 * display-only, project-relative path (or bare filename) — never an
 * absolute path outside the project; `name` is the only identifier ever sent
 * back in `PipelineLoadRequest.models`. `exists: false` means the entry is
 * registered (config or a previous scan) but its file is currently missing
 * on disk — still selectable (a load against it fails with
 * `MODEL_FILE_MISSING`), so the WebUI keeps it in the dropdown with a
 * "missing" annotation rather than hiding it. */
export interface ModelEntry {
  name: string;
  path: string;
  is_default: boolean;
  exists: boolean;
  source: "config" | "scan";
}

/** One category block of `GET /models`'s response (Docs/API_REFERENCE.md
 * §3.5). `active` is the NAME used by the last successful `/pipeline/load`
 * for this category (`"default"` until an explicit selection succeeds) —
 * retained even while the worker is unloaded (`pipeline_loaded` in
 * `StatusResponse` is a separate concern). `entries` is default-first, then
 * the rest sorted by name (server-side order; the WebUI must not re-sort). */
export interface ModelCategoryBlock {
  default: string;
  active: string;
  entries: ModelEntry[];
}

/** One entry of `GET /models`'s `base_models[]` — a declared BASE MODEL (the
 * unit the user picks in the header dropdown), with its own three-layer model
 * listing and its install state (`api/models_registry.py`'s `list_models`,
 * multi-engine groundwork §3-97 P3a; `Docs/MULTI_ENGINE_DESIGN.md` §5.1).
 *
 * `installed` vs `present` is the distinction the dropdown is built on:
 *  - `present: false` — not a single default file is on disk. Selecting it
 *    must NOT hit the API at all; the WebUI answers with the "run
 *    `install-<id>.bat`" guidance (`Docs/MULTI_ENGINE_DESIGN.md` §6.2 guard 2).
 *  - `present: true, installed: false` — a PARTIAL install (some categories
 *    still listed in `missing_categories`). Selecting it does go to the
 *    server, which fails loud with a specific reason.
 *  - `installed: true` — every category's default file is there.
 *
 * `categories` mirrors the top-level {@link ModelsResponse.categories} block
 * for THIS base model. Only the active base model carries real `active` names;
 * every other one reports `"default"` throughout. The DISPLAY ORDER is
 * {@link BaseModelBlock.category_order}, never this object's key order — see
 * that field. */
export interface BaseModelBlock {
  /** Descriptor id — the value `PipelineLoadRequest.base_model` takes, and the
   * `<id>` in the `install-<id>.bat` guidance (e.g. `"LTX23"`, `"LTX25"`). */
  id: string;
  /** Human-facing label (e.g. `"LTX 2.3"`). The dropdown renders THIS, never
   * a hard-coded literal — adding a base model is a server-side descriptor
   * change only. */
  display_name: string;
  /** Inference-engine lineage (`"ltx"`, …). The server picks the engine from
   * the weight file's own GGUF metadata, never from anything the WebUI sends
   * (`Docs/MULTI_ENGINE_DESIGN.md` §2.1).
   *
   * Since 2026-08-31 this is no longer purely diagnostic: it is the key into
   * {@link AppLimits.comfort_budgets}. NON-optional on the wire — a reader
   * that still writes `?? ""` is defending against a backend older than
   * §3-97, not against a normal response, and `""` is exactly what
   * `shell/comfortTable.ts`'s `resolveComfortRow` treats as "engine unknown"
   * (→ the compatibility shim, i.e. today's behaviour). */
  engine_family: string;
  active: boolean;
  installed: boolean;
  present: boolean;
  missing_categories: string[];
  /** The order to render this base model's category dropdowns in: the
   * descriptor's own declaration order (`scripts/manifests/<base>.json`),
   * published as an ARRAY on purpose.
   *
   * `categories` below carries the same order in its keys, but a JSON
   * OBJECT's key order does not reliably survive the trip to this WebUI: the
   * response reaches us through the AviUtl2 plugin's WebView2 message channel,
   * and on 2026-08-20 the Settings dropdowns rendered alphabetised
   * (audio / text_encoder / transformer / video_vae) from a response the
   * server had emitted in descriptor order. An array's element order has no
   * such ambiguity, so this — not `Object.keys(categories)` — is what
   * `shell/useModels.ts` renders by.
   *
   * Optional purely defensively: a backend older than this fix omits it, and
   * `resolveCategoryOrder` falls back rather than assuming presence. */
  category_order?: string[];
  /** Feature names this base model's ENGINE cannot run (§3-98 Phase 5,
   * `services/engines/ltx25/adapter.py`'s `UNSUPPORTED_FEATURES`). `[]` for a
   * base model with no restrictions — LTX 2.3 sends exactly that, which is why
   * nothing about the 2.3 response changed when this key was added.
   *
   * The WebUI greys out the tabs and panels it RECOGNISES here and ignores the
   * rest; a name it has never heard of is not an error, just a feature this
   * build has no control for. Enforcement is always the server's: `POST
   * /generate` answers 422 `FEATURE_UNSUPPORTED` for anything that arrives
   * anyway (a page left open across a base-model switch, a script).
   *
   * Optional purely defensively: a backend older than §3-98 P5 omits it, and
   * "omitted" must read as "no restrictions", never as "everything is off".
   *
   * Known names at the time of writing: `chain`, `retake`, `end_source`,
   * `v2v`, `a2v`, `two_stage_hq`, `outpaint`, `loras`, `reference_video`,
   * `nag`, `prune_vaed`, `sage_attention`. That list is a history, not a
   * contract: a name leaves it as soon as an engine gains the feature
   * (`chain`/`v2v`/`a2v` on 2026-08-23, `loras`/`reference_video` on
   * 2026-08-24, `keep_resident` on 2026-08-25 -- LTX 2.5 opened the
   * text-encoder form of it -- and `sage_attention` on 2026-08-25 as well,
   * once LTX 2.5 started running LTX 2.3's sage attention kernels). Read the
   * live array, never this comment; the canonical list is the backend spec's
   * 6.10(b), 6 names for LTX 2.5. */
  unsupported_features?: string[];
  categories: Record<string, ModelCategoryBlock>;
}

/** `GET /models`'s response (Docs/API_REFERENCE.md §3.5,
 * `api/models_registry.py`'s `list_models`). Rescans the model directories on
 * every call, so a newly downloaded file appears without a server restart —
 * refetching this is the WebUI's only "detect new models" action.
 *
 * `categories` is UNCHANGED and always describes the ACTIVE base model,
 * exactly as it always described the only one; the base-model layer was added
 * beside it (§3-97 P3a), never on top of it. Both new fields are optional here
 * purely defensively — a backend older than the multi-engine groundwork omits
 * them, and every reader must fall back rather than assume presence. */
export interface ModelsResponse {
  categories: Record<ModelCategory, ModelCategoryBlock>;
  /** Id of the base model `categories` describes. */
  active_base_model?: string;
  base_models?: BaseModelBlock[];
}

/** Optional body for `POST /pipeline/load` (Docs/API_REFERENCE.md §3.3,
 * `api/pipeline.py:24-32` `LoadPipelineRequest`). Omitting `models` (or the
 * whole body) keeps the legacy behavior byte-identical — the current active
 * selection reloads. A partial block only swaps the categories it names;
 * `MODEL_DEFAULT_NAME` means "use the config default", never a
 * client-supplied path (`/`, `\`, `..` are rejected server-side). */
export interface PipelineLoadRequest {
  models?: Partial<Record<ModelCategory, string>>;
  /** Multi-engine (§3-97 P6): switch the pipeline to this BASE MODEL id. Omit
   * to stay on the current one. Valid on its own — `{"base_model": "LTX25"}`
   * with no `models` block is the "just switch base models" call the header
   * dropdown makes, and the server resolves every category from that base's
   * descriptor defaults / remembered selection. Unknown id → 404
   * `MODEL_NOT_FOUND`. */
  base_model?: string;
}

/** `POST /pipeline/load`'s response (Docs/API_REFERENCE.md §3.3,
 * `api/pipeline.py:39-93`). `models` (the resulting active selection per
 * category) is present only when the request carried a non-empty `models`
 * block — the legacy bodyless path's response omits it, matching the
 * byte-identical golden-snapshot guarantee. */
export interface PipelineLoadResponse {
  pipeline_loaded: boolean;
  pipeline_type: string | null;
  state: string;
  models?: Record<ModelCategory, string>;
  /** Multi-engine (§3-97 P6): the base model the pipeline ended up on.
   * Present on exactly the same condition as `models` — the legacy bodyless
   * path's response still carries neither, preserving its byte-identical
   * golden snapshot. */
  base_model?: string;
}

/** `POST /pipeline/unload`'s response (N4 "danger zone": explicit engine
 * teardown, freeing VRAM without loading a replacement). Minimal shape —
 * mirrors the subset of `PipelineLoadResponse` the WebUI actually reads. */
export interface PipelineUnloadResponse {
  pipeline_loaded: boolean;
  state: string;
}

/** One row from `GET /loras` (Docs/API_REFERENCE.md §3.6,
 * `services/lora_registry.py:68-76`). Deliberately excludes any filesystem
 * path — `name` is the only identifier the WebUI ever sees or sends back.
 * `kind` drives the Library screen's Style/Control tab split: `"style"`
 * needs no reference video and can be applied via a prompt tag; `"control"`
 * requires one (`reference_video_id`) and isn't wireable until M6. */
export interface LoraEntry {
  name: string;
  kind: "style" | "control";
  has_thumbnail: boolean;
  exists: boolean;
  source: string;
  /** §1-15 (chain reference video): the control adapter's frame PREPROCESS
   * kind (`"depth"`/`"canny"`/`"pose"`/… — `config.ic_loras[name].preprocess`
   * on the server, exposed through `services/lora_registry.py`'s `to_dict()`).
   *
   * OPTIONAL on purpose: the field is being added to `GET /loras` in parallel
   * with this feature, and an older server simply omits it. Every consumer must
   * treat "absent" as "unknown" and stay silent rather than guess — the Chain
   * form's `depthChainUnsupported` gate (which pre-empts the server's 422 for
   * depth-preprocess adapters on a multi-clip chain) does not fire at all
   * without it. */
  preprocess?: string | null;
  /** §1-15 (clip-wise IC-LoRA reference): the adapter's `reference_downscale_factor`
   * off its safetensors header (`services/lora_registry.py`'s `to_dict()`) —
   * union-control adapters are `2`, deblur is `1`. `null`/absent when the
   * header carries no such key (style LoRAs; a control entry whose header is
   * missing/unreadable) or the value is unparsable. Feeds
   * `shell/tokenBudget.ts`'s `chainStage1Tokens`' `refScale` so the Chain
   * screen's stage-1 comfort-budget warning (F5) can size itself BEFORE a job
   * is ever submitted, rather than only after a 422/OOM.
   *
   * OPTIONAL for the same reason as {@link preprocess} (the server that ships
   * alongside this feature always includes it, but an older server, and every
   * pre-existing test fixture that builds a `LoraEntry` by hand, omits it) —
   * every consumer must treat "absent" the same as "unknown"/`null`. */
  reference_downscale_factor?: number | null;
}

export interface LorasListResponse {
  loras: LoraEntry[];
}

/** `POST /loras/reload`'s response (Docs/API_REFERENCE.md §3.7) — an
 * explicit rescan of `config.model.lora_dir`, for the Library screen's
 * "Reload" button. */
export interface LorasReloadResponse {
  total: number;
  styles: number;
  controls: number;
}
