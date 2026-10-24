# R1 — Existing LoRA mechanism (LTX 2.3 / LTX 2.5) inventory

Root: S:\OriginalApps\12_Nz-LTX23-AviUtl2\Nz-Videomni (B = backend). Frontend: B\AviUtl2-Plugin\Nz-Videomni-frontend-AviUtl2\webui\src (F). Read-only; facts only.

## 1. Model directory layout and discovery

- On disk: `models/LTX23/StyleLoRA/` (10 user .safetensors + `put_LoRA_safetensors_here.txt`), `models/LTX23/IC-LoRA/{deblur,in-outpainting,pixel-spatial-upscaler,union-control}/`, `models/LTX25/IC-LoRA/alpha-gen/`. `models/MiniMaxH3/` has only LICENSE, TextEncoder, VAE, Weights (no LoRA dir).
- **There is ONE engine-agnostic LoRA registry** (`services/lora_registry.py`, `LoraRegistry`), built once in `api/context.py:83` and shared by every engine. It has no notion of engine family / base model.
- Two sources, config authoritative (`services/lora_registry.py:215-252`):
  1. `config.model.ic_loras` (name -> path string, or `IcLoraEntry{path, preprocess: none|canny|dwpose|depth}`) — `config.py:58-70`, `config.py:138`. Live `config.yaml:50-68` registers the LTX23 IC-LoRAs plus `alpha-gen` (path under `models/LTX25/IC-LoRA/...`, line 65). Template: `config.yaml.example:48-77`.
  2. Flat (non-recursive) scan of **one** directory `config.model.lora_dir`, default `"./models/LTX23/StyleLoRA"` (`config.py:150`; not overridden in `config.yaml`). `_scan_lora_dir` `lora_registry.py:254-269`: `*.safetensors` only, dotfiles skipped, sorted.
- Name = filename stem; stem collision -> `<parentdir>__<stem>`; second collision -> warning + ignored (`:238-245`). Same resolved path as a config entry -> skipped (`:236`).
- Per-entry metadata from the safetensors header (pure Python, `_read_safetensors_header` `:128-146`, 100 MB header cap `:72`):
  - `kind`: `"control"` if `preprocess != "none"` or header `__metadata__` has `reference_downscale_factor`; else `"style"` (`:288`, `:309-314`).
  - `layout` via `_detect_layout` (`:154-199`): DoRA -> unsupported; any `.lora_A.weight` -> `ab`; `.lora_down.weight` with dotted module path -> `kohya`; underscore-joined kohya names / LoHa (`hada_`) / LoKr (`lokr_`) / no LoRA keys -> `unsupported:<reason>`.
  - `has_thumbnail`: sibling `<stem>.png` (`:285`).
  - Scan entries with unreadable headers are silently dropped (`:295-299`); config entries kept (`:300-306`).
- Hard-coded directory names: only the `lora_dir` default (LTX23/StyleLoRA, `config.py:150`), IC-LoRA paths in config.yaml, and `engine25/worker.py:1763-1767` (`_LORA_SEARCH_ROOT = ROOT/"models"/"LTX23"` for the selftest `--lora NAME` CLI; comment: "Both engines share ONE adapter library ... the 2.3 adapters are what 2.5 runs"). So LTX 2.5 consumes the LTX23 StyleLoRA folder; there is no `models/LTX25/StyleLoRA`.
- Base-model descriptors (`scripts/manifests/*.json` -> `services/base_models.py`) do NOT declare LoRA dirs or a "has Style LoRA" flag; `services/base_models.py`, `services/model_registry.py`, `api/models_registry.py` contain no LoRA logic (model_registry only cites lora_registry as precedent, `services/model_registry.py:3,32`). `models/INSTALLED_PATHS.txt` lists no LoRA paths.
- "Has LoRA" per family is expressed only negatively: the engine adapter's `REJECT_TABLE` / `UNSUPPORTED_FEATURES` (see §5). `GET /models` publishes per base model `unsupported_features` (`api/models_registry.py:135-137`, from `services/engines/__init__.py:166-173`). No positive `loras` / `features` field in `GET /models`.
- Frontend `featureScope.ts`: `FEATURE_UI` (`F/shell/featureScope.ts:144`) has **no `loras` row**; comment at `:172` says `loras` has no UI target. An engine publishing `loras` closes nothing in the operation panel.

## 2. API contract

- Listing (`api/loras.py`):
  - `GET /api/v1/loras` -> `{"loras": [row...]}`, rescans each call (`:24-28`). Row = `LoraEntryInfo.as_dict()` (`services/lora_registry.py:110-125`): `name, kind, has_thumbnail, exists, source ("config"|"scan"), preprocess, reference_downscale_factor`. No path, no family field.
  - `POST /api/v1/loras/reload` -> `{total, styles, controls}` (`:31-40`).
  - `GET /api/v1/loras/{name}/thumbnail` -> PNG or 404 `LORA_THUMBNAIL_NOT_FOUND` (`:43-63`).
  - Router `api/router.py:12,25`. `GET /config` dumps the whole config incl. `model.ic_loras` / `lora_dir` (`api/status.py:58-61`); frontend and Gradio read `config.model.ic_loras` for control names.
- Request: `LoraSpec` (`api/models.py:196-223`): `name: str (1..200)`, `strength: float = 1.0 (gt 0, le 2.0)`, `audio_strength: float|None (ge 0, le 2.0; None = follow strength)`; validator rejects `/`, `\`, `..`.
  - `GenerateRequest.loras: list[LoraSpec] = []` (`api/models.py:612`); `GenerateChainRequest.loras` (`:1288`, same strength for every clip/stage). **No count limit** (no max_length); multiple LoRAs allowed.
  - Cross-field: `reference_video_id`, `conditioning_attention_strength`, `reference_video_strength` require non-empty `loras` (`api/models.py:675-689`; chain `:1662-1686`).
- Endpoint order (`api/generate.py:68-140`): (1) `engines.reject_unsupported(active_family, request)` FIRST (`:79`), "without touching ... the lora registry"; (2) `engines.validate_for_family` (`:91`); then (3) per-spec `lora_registry.resolve()` + `info()` (`:128-140`) -> `LORA_REQUIRES_REFERENCE`, `REFERENCE_REQUIRES_CONTROL_LORA`, `LORA_PREPROCESS_CONFLICT`. Chain mirror: `api/generate_chain.py:144-183` (+ `LORA_DEPTH_CHAIN_UNSUPPORTED`).
- `resolve()` (`lora_registry.py:361-404`): path-like / unknown / empty registry / missing file -> 404 `LORA_NOT_FOUND`; unsupported layout -> 422 `LORA_FORMAT_UNSUPPORTED`. Strength passed through untouched (no alpha/rank multiplier; docstring `:28-34`). Returns `ResolvedLora(path, strength, preprocess, audio_strength)` (`:59-68`).
- Error factories (`api/errors.py`): `:224` LORA_NOT_FOUND 404, `:270` LORA_REQUIRES_REFERENCE, `:285` REFERENCE_REQUIRES_CONTROL_LORA, `:301` LORA_FORMAT_UNSUPPORTED 422, `:318` LORA_DEPTH_CHAIN_UNSUPPORTED, `:340` LORA_THUMBNAIL_NOT_FOUND, `:351` LORA_PREPROCESS_CONFLICT, `:646-667` FEATURE_UNSUPPORTED 422 (bilingual message).
- Alpha Gen pins `alpha-gen` @ 1.0 server-side (`api/generate_alpha.py:69-70,164-165,249`).
- Spec docs: `Videomni_Backend_Specification.md:739-742` (request fields), `:759-761` (validation), `:915-920` (LoraSpec table), `:1425-1431` (error codes), `:891` (H3 unsupported_features incl. `loras`), `:601-616` (installer IC-LoRA §5.1b), `:524` (layout). Frontend `AviUtl2-Plugin/Nz-Videomni-frontend-AviUtl2/Docs/API_REFERENCE.md` §3.6 / §5.3 (cited from F code). Others: `Docs/STYLE_LORA_WORKORDER.md`, `Docs/LORA_AUDIO_STRENGTH_WORKORDER.md`, `Docs/IC_LORA_PHASE_*`.
- Metadata: `metadata.json` `ic_lora` block only when `req.loras` non-empty (`services/pipeline_manager.py:1930-1945`; chain `:1732-1788`; row builder `_lora_metadata_entry` `:121`). Console line `Job %s base=%s loras=%s` (`:942-945`, `_loras_for_log` `:98`).
- `api/prompt_tokens.py:9`: the token counter gets the prompt with `<lora:>` tags already stripped by the panel.

## 3. Frontends

### webui (operation panel)
- Style LoRAs ride as **prompt tags** `<lora:name[:strength[:audioStrength]]>`; the prompt string is the single source of truth (`F/lora/loraTags.ts:5-28`; regex `:82`; limits `:58-71`: strength 0.05..2.0 step 0.1 default 1.0; audio 0..2.0). `parseLoraPrompt` (`:143`) strips tags -> `loras[]`.
- Chips under the prompt: `F/lora/LoraChips.tsx:38` (+/- strength, remove, audio mute toggle 0<->1), mounted in `F/shell/PromptBar.tsx:57`.
- Inventory ("Library") tab: `F/modes/inventory/InventoryScreen.tsx` — `useLoras` fetches `GET /loras` (`F/modes/inventory/useLoras.ts:25-53`); shows only `kind === "style"` (`InventoryScreen.tsx:71`); click appends a tag via `appendLoraTag` (`:62`); Reload -> `POST /loras/reload` (`:49-51`). Card/thumbnail: `F/modes/inventory/LoraCard.tsx:23-25`, URL helper `F/api/client.ts:238`; client calls `:216-217`. Types: `LoraSpec` `F/api/types.ts:63-75`, `LoraEntry` `:1352+` (no family field).
- Empty-state text hard-codes the folder: `F/i18n/strings.ts:1300` (EN "Drop files into models/LTX23/StyleLoRA") and `:3076` (JA).
- Control LoRA = separate single selection (`F/lora/controlLoras.ts`; `UI_HIDDEN_CONTROL_LORA_NAMES` hides alpha-gen), merged with prompt tags by `combineLoras` (`F/modes/single/useGenerationForm.ts:1159-1166`); request gets `...(loras.length > 0 ? { loras } : {})` (`:1272`, A2V path `:1321`). Chain: `F/modes/chained/useChainForm.ts`. Batch builders also carry loras (`F/modes/batch/*`, `F/modes/batch-i2v-long/*`).
- **No per-family gating of style LoRAs**: Inventory, chips and request building do not consult `unsupported_features` for `loras`. (`reference_video` row hides the IC-LoRA reference accordion, `featureScope.ts:101-103`.) On H3 a `<lora:>` tag reaches the server and gets 422.

### Gradio (`gradio_ui/`)
- Style LoRA tab: gallery of GET /loras style entries (`gradio_ui/adapters.py:202-243`, `ui.py:1102-1115`, events `ui.py:2709-2723`); selection appends `<lora:name:1.0:1.0>` to the shared prompt (`ui.py:1105`).
- Tag parser `_LORA_TOKEN_RE` + limits (`gradio_ui/handlers.py:362-391`), names checked against GET /loras (case-insensitive); batch per-row parsing (`gradio_ui/batch.py:296-358`, `ui.py:1621-1644`).
- IC-LoRA adapter dropdown from `/config model.ic_loras` (`gradio_ui/adapters.py:1-56`).
- `gradio_ui/feature_scope.py` has no LoRA entry; `hidden_controls(unsupported)` at `:77`.

### MCP (`mcp_server/`)
- Tool `list_loras` -> GET /loras (`mcp_server/tools/system.py:233-244`).
- `LoraArg{name, strength=1.0, audio_strength=None}` (`mcp_server/params.py:38-52`); `submit_generate(loras=...)` / `submit_chain(loras=...)` dump into payload (`mcp_server/tools/generate.py:166,566-567,652`); outpaint auto-appends `in-outpainting` (`:85-87,603-607`).
- Server instructions: H3 422s `loras` with FEATURE_UNSUPPORTED (`mcp_server/server.py:50-52`); LTX 2.5 supports loras (`:75-77`).

## 4. Worker / engine side (LTX)

- Shared loader `engine/gguf/ic_lora_common.py`, used by the 2.3 GGUF path, the 2.3 quantized-safetensors path and LTX 2.5.
  - **Runtime low-rank path, not merged**: A/B attached as non-persistent buffers on each target `nn.Linear` (`attach_ic_loras` `:277-380`), spec list attr `_ic_lora_specs` (`:44`); stored weights never written.
  - Delta: `(B.float()*strength) @ A.float()` in fp32, cast once (`:21-29`). GGUF forward adds in place on the transient dequant tensor (`engine/gguf/quant_service.py:498-530`); float-weight branch and sft path out of place (`engine/sft_quant/quant_service.py:274-292`, casts to `x.dtype` to avoid fp8 rounding).
  - Buffers keep native dtype (e.g. bf16) (`ic_lora_common.py:350-353`); move with `block.to(device)` (block swap).
  - Key renaming: wheel `LTXV_LORA_COMFY_RENAMING_MAP` = strip `diffusion_model.` prefix only (`.venv-engine/Lib/site-packages/ltx_core/loader/sd_ops.py:117-119`; vendor copy `vendor/LTX-2/packages/ltx-core/src/ltx_core/loader/sd_ops.py:135`). Stripped prefix must equal the dotted `named_modules()` name inside `LTXModel` (`_find_target_model` `:252-274`). No diffusers->native or kohya-underscore remapping.
  - Layouts (`load_ic_lora_pairs` `:156-249`): A/B (alpha assumed baked) wins per file; else kohya (`.lora_down/.lora_up` + `.alpha`, scale = alpha/rank folded into B out of place, `:233-247`). Missing partner key -> RuntimeError.
  - Fused vs split QKV: no handling (no qkv logic); LTX has separate Linears, 1:1 prefix match assumed.
  - Unmatched keys: a prefix with no `nn.Linear` counterpart is **silently skipped** (`:315-319`); if 0 prefixes resolve for a file -> WARNING only, job proceeds (`:368-374`). Shape mismatch -> RuntimeError (`:324-335`). Non-`nn.Linear` modules skipped by isinstance (`:316`).
  - Audio/video axis for `audio_strength`: `classify_lora_axis` keyed on LTX module names (`:62-127`); effective strength 0 -> no buffers.
  - Several LoRAs on one Linear accumulate as successive specs (`:288-290`). `detach_ic_loras` (`:383+`) restores the no-LoRA state.
- 2.3 wiring: `engine/gguf/quant_service.py:648-656`, `engine/sft_quant/quant_service.py:442-446` (attach if ic_loras else detach).
- 2.5 wiring: `engine25/gguf_transformer.py:705-783` (`set_loras` / `begin_job` / `_apply_loras`, memoised `load_ic_lora_pairs`, attach between X0Model wrap and placement).

## 5. Request -> worker, and H3's current handling

- Orchestrator: `services/pipeline_manager.py:686-695` (single), `:1346-1356` (chain) re-resolve `LoraSpec` -> `ResolvedLora` and pass `lora_paths=` to the backend (`:963`, `:1526`).
- LTX payload entries via `_lora_payload_entry` -> `{"path", "strength"[, "audio_strength"]}` (`services/engines/ltx/adapter.py:571-586`); single generate always sends `"loras"` (`:2265,2295`), chain only when non-empty (`:2630-2631`); `_resolve_reference_preprocess` (`:549-568`). LTX 2.5 imports the same helper (`services/engines/ltx25/adapter.py:59,878,909,1358-1359`).
- Worker parse: 2.3 `engine/worker.py:72-79` (protocol), `:928-935` (`msg.get("loras", [])` -> `(path, strength, audio_strength)`), chain `:1258-1265`. 2.5 `engine25/worker.py:53-62`, `_ic_loras` `:407-427`, passed every job incl. `[]` (`:922-1022`).
- LTX adapters list `loras` in `HONOURED_FIELDS` (`ltx/adapter.py:198`; `ltx25/adapter.py:255,440`).
- **H3 adapter** (`services/engines/h3/adapter.py`):
  - `REJECT_TABLE` first row `("loras", "loras", lambda r: bool(r.loras))` (`:200-212`); `reject_unsupported` (`:269-273`) -> 422 `FEATURE_UNSUPPORTED`, detail "MiniMax H3 は loras に対応していません（リクエストの loras が既定値ではありません）。" (`_refusal_detail` `:260-266`).
  - `GOVERNED_FIELDS`: `conditioning_attention_strength`, `reference_video_strength` governed by `loras` (`:254-255`). `UNSUPPORTED_FEATURES` derived from REJECT_TABLE, so includes `loras` (`:257+`).
  - `generate(..., lora_paths=...)` keeps the shared signature, documented as never filled; re-runs `reject_unsupported` as backstop (`:419-440`). Payload has no `loras` key (`:460+`). `ResolvedLora` imported for typing only (`:70`). Chain refused wholesale (`reject_chain`).
- H3 worker: `engine_h3/worker.py:13-14` — "no IC-LoRA / reference video / chain / NAG / VSF branches"; nothing in `engine_h3/` reads `loras`.
- H3 DiT facts relevant to reuse: stacked fused `qkv_proj` (`engine_h3/dit.py:32,229,240`), `fc1` = 2*ffn (`:256`); quantized layers replaced by `H3Int8Linear(nn.Module)`, not an `nn.Linear` subclass (`engine_h3/quant_linear.py:59`, replacement `:210-231`). `attach_ic_loras` skips non-`nn.Linear` modules (`ic_lora_common.py:316`).
- H3 rulings: `Docs/MINIMAX_H3_DESIGN.md:46` 裁定10 (LoRA deferred; Turbo LoRA to be done together with Style LoRA), `:58` 裁定30 (LTX-specific IC-LoRA hidden on H3), `:418`.

## 6. Install / distribution

- `scripts/manifests/10-ltx23.json`: IC-LoRA downloads (`:60-68`, `:114-122`, `:154-180`); `migrate` maps legacy `loras` -> `LTX23/StyleLoRA` (`:196`). Style LoRAs are never downloaded.
- `scripts/manifests/20-ltx25.json:115-124`: alpha-gen IC-LoRA -> `LTX25/IC-LoRA`.
- `scripts/manifests/40-minimaxh3.json`: no LoRA entries.
- `scripts/install_ltx.ps1:784-800`: config.yaml follow-up rewrites `ic_loras` (and a user-set `lora_dir`) paths during migration; `:1966` warns not to delete models/ (user LoRAs). `scripts/setup.ps1:308-310` lists `models\LTX23\StyleLoRA` as user data; `scripts/install_model.ps1:262` similar.

## 7. Tests (paths)

Backend (B/tests): `test_lora_registry.py`, `test_loras_endpoint.py`, `test_ic_lora_api.py`, `test_ic_lora_forward.py`, `test_ic_lora_engine_conditioning.py`, `test_chain_lora.py`, `test_engine_h3_adapter.py` (`_LORAS` `:45`, overrides `:120-141`, governed `:189`, HTTP refusal parametrize incl. `loras` `:734-746`), `test_engine_dispatch.py`, `test_ltx_adapter_audit.py`, `test_ltx25_adapter.py`, `test_ltx25_api_guard.py`, `test_ltx_runner_payload.py`, `test_sft_quant_linear_forward.py`, `test_pipeline_gguf_install_fail_loud.py`, `test_generate_alpha_api.py`, `test_gradio_lora_parse.py`, `test_gradio_style_tab.py`, `test_gradio_handlers.py`, `test_gradio_batch_runner.py`, `test_gradio_batch_fullstack.py`, `test_gradio_models.py`, `test_mcp_tools_generate.py`, `test_mcp_tools_system.py`, `test_mcp_registration.py`, `test_logging.py`, `test_model_registry.py`. (`test_h3_engine.py` has no LoRA.)
Frontend (F): `lora/loraTags.test.ts`, `lora/LoraChips.test.tsx`, `lora/controlLoras.test.ts`, `modes/inventory/InventoryScreen.test.tsx`, `modes/inventory/useLoras.test.ts`, `shell/featureScope.test.ts` (`:208` name set incl. `loras`), `shell/AppShell.minimaxH3.test.tsx`, `shell/AppShell.featureScope.test.tsx`, `shell/AppShell.controlLora.test.tsx`, `shell/useControlLoraNames.test.ts`, plus mode tests (`modes/single/*`, `modes/chained/useChainForm.test.ts`, batch builders).

## Open questions / not found

- No per-family / per-base-model LoRA filtering anywhere (registry, GET /loras row, frontend). A `models/MiniMaxH3/LoRA/` would need a second scan root or family-aware listing; the single `lora_dir` cannot express it. Whether GET /loras becomes family-scoped is undecided.
- No positive "has Style LoRA" capability in GET /models; `FEATURE_UI` has no `loras` row, so the panel never hides Inventory / LoRA chips on H3 today.
- The existing loader has no fused-QKV, diffusers-name or kohya-underscore key conversion; H3's stacked `qkv_proj` and `H3Int8Linear` (not `nn.Linear`) are both outside what `attach_ic_loras` handles. The key naming of H3 LoRA files (e.g. a MiniMax Turbo LoRA) was not investigated (no H3 LoRA file on disk).
- `_detect_layout` / kind logic is header-only and family-agnostic (any A/B file = `style`); `classify_lora_axis` is LTX-module-name specific.
- Not read in detail: `Docs/STYLE_LORA_WORKORDER.md`, frontend `Docs/API_REFERENCE.md` §3.6/§5.3.
