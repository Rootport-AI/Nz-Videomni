"""MiniMax H3 text encoder: Qwen3-VL-32B (text + vision), the text side cut after decoder layer 50.

Ported in part from DiffSynth-Studio (Apache-2.0), commit 4e34969:
  diffsynth/models/minimax_h3_text_encoder.py            (the config values, the T2V and
                                                         FL2VA presentations, the token type ids)
  diffsynth/utils/state_dict_converters/minimax_h3_text_encoder.py  (the Comfy-Org name mapping)
Copyright (c) the DiffSynth-Studio authors. Licensed under the Apache License,
Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0). Changes: built on the
meta device; the 350 quantized Linears run through ``H3Int8Linear``; the
weights are read straight from the file onto the GPU while the forward runs
(below); the vision tower is read only when a key frame is given to it; each
key frame given to the vision tower is the cover-and-crop canvas image
(``engine_h3.image_cond``), not a stretched one.

I2V presentation (Docs/MINIMAX_H3_DESIGN.md §3)
-----------------------------------------------
Per key frame given to the vision tower (the pipeline gives it the first and
the last frame, in that order): ``"<Picture i>: "`` (text, tag 1, i = 1, 2 ...)
+ ``<|vision_start|> <|image_pad|> x N <|vision_end|>`` (tag 0, video); then
the prompt (tag 1), each part tokenized on its own (no chat template, no
specials). N comes from the Qwen3-VL image
processor's grid (``prod(grid_thw) // merge_size**2``; ``(H/32) x (W/32)`` for a
canvas of 65 536 .. 16 777 216 pixels). ``mm_token_type_ids`` marks the
``<|image_pad|>`` positions with 1 (transformers 5 refuses a multimodal forward
without it); the DeepStack injection and the 3-axis vision positions happen
inside Hugging Face's ``Qwen3VLModel.forward``, which is used unmodified.

Direct read (Docs/MINIMAX_H3_DESIGN.md §4)
------------------------------------------
The encoder runs ONE forward per job (one prompt, no CFG), so no layer is
worth keeping in main memory. ``load`` only builds the model on meta (and the
tokenizer); ``encode`` reads the embedding table (and, for I2V, the vision
tower) onto the GPU, then runs the
unmodified Hugging Face ``forward`` with a pre-hook on every decoder layer that
reads that layer's tensors from the file onto the GPU right before its forward,
and a hook that drops them back to meta right after. The layer loop itself is
HF's (the vision path's DeepStack injection and 3-axis positions live inside
it), so nothing of it is rewritten here. The tensors keep the file's dtypes and
the forward is the same code on the same device, so reading the weights this
way does not change the hidden state (VERIFICATION_LOG §166, 2a-G2). The CPU
side never holds more than one tensor of a layer, of the embedding table or of
the vision tower at a time.

What H3 reads from the encoder (R2 D, R2 (3) item 10):

* the hidden state AFTER decoder layer 50, BEFORE the final RMSNorm -- the
  model is built with 50 layers and ``norm = Identity`` (DiffSynth's method;
  the Comfy-Org file is already cut to 50 layers and has no ``model.norm``);
* the prompt is tokenized as is: no chat template, no system prompt,
  ``add_special_tokens=False``;
* ``tokenizer_config.json``'s extra special tokens (``<d>`` ... ``<|caption_end|>``)
  must be single tokens 151669..151675. Read from ``tokenizer.json`` alone they
  would split into ordinary BPE pieces and the dialogue markup of the official
  prompts (``<d>[English] ...</d>``) would silently mean something else. ``load``
  checks the ids and, when the tokenizer did not add them, adds them in order.

The modality tag of every T2V text token is 1 (text); the vision-block tokens
of an I2V presentation are 0 (video), as both references tag them.
"""

from __future__ import annotations

import gc
import logging
import time
from collections.abc import Sequence

import torch
import torch.nn as nn

import h3_format
from engine_h3.constants import MODALITY_TEXT, MODALITY_VIDEO
from engine_h3.quant_linear import swap_in_int8_linears
from engine_h3.sft_load import drop_to_meta, load_mapped_to, move_module_tree

logger = logging.getLogger(__name__)

NUM_RETAINED_LAYERS = 50

#: tokenizer_config.json additional_special_tokens that tokenizer.json lacks,
#: in id order (ComfyUI pins the same numbers; R2 D).
EXTRA_SPECIAL_TOKENS = (
    "<d>", "</d>", "<|cutoff|>", "<|lyrics_start|>", "<|lyrics_end|>", "<|caption_start|>", "<|caption_end|>",
)
EXTRA_SPECIAL_FIRST_ID = 151669

#: Qwen3-VL-32B text config, verbatim from DiffSynth minimax_h3_text_encoder.py:24-49
#: (identical to the official FL2VA/text_encoder/config.json text_config except
#: num_hidden_layers, which is cut to 50).
TEXT_CONFIG = {
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "dtype": "bfloat16",
    "eos_token_id": 151645,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 5120,
    "initializer_range": 0.02,
    "intermediate_size": 25600,
    "max_position_embeddings": 262144,
    "model_type": "qwen3_vl_text",
    "num_attention_heads": 64,
    "num_hidden_layers": NUM_RETAINED_LAYERS,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-06,
    "rope_scaling": {
        "mrope_interleaved": True,
        "mrope_section": [24, 20, 20],
        "rope_type": "default",
    },
    "rope_theta": 5000000,
    "use_cache": True,
    "vocab_size": 151936,
}

#: What the built config must say after transformers 5 has translated
#: ``rope_scaling`` / ``rope_theta`` into ``rope_parameters`` (R1 2-6).
EXPECTED_ROPE = {"rope_type": "default", "rope_theta": 5000000.0, "mrope_section": [24, 20, 20]}

#: Qwen3-VL-32B vision config, verbatim from DiffSynth minimax_h3_text_encoder.py:53-68
#: (identical to the official FL2VA/text_encoder/config.json). Its tensors are
#: BF16 in the Comfy-Org TE file (351 tensors, 1.11 GiB, none quantized).
VISION_CONFIG = {
    "deepstack_visual_indexes": [8, 16, 24],
    "depth": 27,
    "hidden_act": "gelu_pytorch_tanh",
    "hidden_size": 1152,
    "in_channels": 3,
    "initializer_range": 0.02,
    "intermediate_size": 4304,
    "model_type": "qwen3_vl",
    "num_heads": 16,
    "num_position_embeddings": 2304,
    "out_hidden_size": 5120,
    "patch_size": 16,
    "spatial_merge_size": 2,
    "temporal_patch_size": 2,
}
IMAGE_TOKEN_ID = 151655      # <|image_pad|>
VIDEO_TOKEN_ID = 151656      # <|video_pad|>
VISION_START_ID = 151652     # <|vision_start|>
VISION_END_ID = 151653       # <|vision_end|>
#: The FL2VA label in front of each key frame's vision block, numbered from 1 in
#: the order the key frames are given (first frame then last frame).
PICTURE_LABEL = "<Picture {}>: "
#: ``Qwen3VLVisionRotaryEmbedding``'s default theta (its inverse frequencies are a
#: non-persistent buffer, recomputed here because no file carries them).
VISION_ROPE_THETA = 10000.0


def build_text_config():
    """The ``Qwen3VLTextConfig`` H3 uses (50 layers, SDPA attention)."""
    from transformers import Qwen3VLTextConfig

    config = Qwen3VLTextConfig(**TEXT_CONFIG)
    try:
        config._attn_implementation = "sdpa"
    except Exception:  # noqa: BLE001 -- older/newer setter shapes; verified below
        pass
    return config


def build_config():
    """The ``Qwen3VLConfig`` H3 uses: the text side cut to 50 layers, SDPA on both sides."""
    from transformers import Qwen3VLConfig

    config = Qwen3VLConfig(
        architectures=["Qwen3VLForConditionalGeneration"],
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
        vision_start_token_id=VISION_START_ID,
        vision_end_token_id=VISION_END_ID,
        tie_word_embeddings=False,
        text_config=dict(TEXT_CONFIG),
        vision_config=dict(VISION_CONFIG),
    )
    for cfg in (config, config.text_config, config.vision_config):
        try:
            cfg._attn_implementation = "sdpa"
        except Exception:  # noqa: BLE001 -- setter shapes differ between versions; checked by check_rope
            pass
    return config


def rope_summary(config) -> dict:
    """The rope settings as the built config holds them (for the log and the checks)."""
    rp = dict(getattr(config, "rope_parameters", None) or {})
    return {
        "rope_type": rp.get("rope_type"),
        "rope_theta": float(rp["rope_theta"]) if "rope_theta" in rp else None,
        "mrope_section": list(rp.get("mrope_section") or []),
        "mrope_interleaved": rp.get("mrope_interleaved"),
        "head_dim": int(getattr(config, "head_dim", 0) or 0),
        "num_attention_heads": int(config.num_attention_heads),
        "num_key_value_heads": int(config.num_key_value_heads),
        "num_hidden_layers": int(config.num_hidden_layers),
        "attn_implementation": getattr(config, "_attn_implementation", None),
    }


def check_rope(config) -> dict:
    """Refuse a config whose rope settings did not survive the rope_scaling translation."""
    got = rope_summary(config)
    for key, want in EXPECTED_ROPE.items():
        if got[key] != want:
            raise RuntimeError(
                f"H3 TE config: rope {key}={got[key]!r}, expected {want!r} "
                f"(transformers' rope_scaling -> rope_parameters translation changed?)"
            )
    if got["head_dim"] != 128:
        raise RuntimeError(f"H3 TE config: head_dim={got['head_dim']}, expected 128")
    if got["num_hidden_layers"] != NUM_RETAINED_LAYERS:
        raise RuntimeError(f"H3 TE config: num_hidden_layers={got['num_hidden_layers']}, expected 50")
    return got


def te_key_map(name: str) -> str | None:
    """Comfy-Org TE tensor name -> ``Qwen3VLModel`` state-dict key (None = never read).

    DiffSynth's Comfy-Org converter (state_dict_converters/minimax_h3_text_encoder.py:30-45)
    minus its ``model.`` wrapper prefix: ``model.layers.N.*`` -> ``language_model.layers.N.*``
    (N < 50), ``model.embed_tokens.*`` -> ``language_model.embed_tokens.*``, ``visual.*``
    as is (read for I2V only). ``lm_head``, ``model.norm`` and any layer >= 50 are dropped.
    """
    if name.startswith("model.layers."):
        idx = int(name[len("model.layers."):].split(".", 1)[0])
        if idx >= NUM_RETAINED_LAYERS:
            return None
        return "language_model." + name[len("model."):]
    if name.startswith("model.embed_tokens."):
        return "language_model." + name[len("model."):]
    if name.startswith("visual."):
        return name
    return None


def te_quant_names(quant: dict) -> dict:
    """``h3_format`` TE quant keys (``model.layers.N.x``) -> module names in the built model
    (``language_model.layers.N.x``; the vision tower has no quantized layer)."""
    out = {}
    for layer, spec in quant.items():
        mapped = te_key_map(layer + ".weight")
        if mapped is not None:
            out[mapped[: -len(".weight")]] = spec
    return out


def ensure_extra_special_tokens(tokenizer) -> dict:
    """Make ``<d>`` ... ``<|caption_end|>`` single tokens 151669..151675; return what was found.

    Adds them (in order, as special tokens) only when NONE is present and the
    vocabulary ends right before 151669 -- the one state in which appending
    gives exactly ComfyUI's numbers. Any other state is refused: guessing would
    shift every id after the first wrong one.
    """
    want = {tok: EXTRA_SPECIAL_FIRST_ID + i for i, tok in enumerate(EXTRA_SPECIAL_TOKENS)}
    vocab = tokenizer.get_vocab()
    present = {tok: vocab[tok] for tok in EXTRA_SPECIAL_TOKENS if tok in vocab}
    added = False
    if not present and len(tokenizer) == EXTRA_SPECIAL_FIRST_ID:
        tokenizer.add_tokens(list(EXTRA_SPECIAL_TOKENS), special_tokens=True)
        vocab = tokenizer.get_vocab()
        present = {tok: vocab[tok] for tok in EXTRA_SPECIAL_TOKENS if tok in vocab}
        added = True
    if present != want:
        raise RuntimeError(
            f"H3 tokenizer: extra special tokens are {present}, expected {want} "
            f"(len(tokenizer)={len(tokenizer)}). tokenizer_config.json from the official "
            "FL2VA/text_encoder folder is required."
        )
    # The tokens must survive tokenization as single ids, not just exist.
    probe = tokenizer("<d>[English] hi</d>", add_special_tokens=False)["input_ids"]
    if probe[0] != want["<d>"] or probe[-1] != want["</d>"]:
        raise RuntimeError(f"H3 tokenizer: '<d>…</d>' tokenized as {probe}, not as single special tokens")
    return {"ids": present, "added_here": added}


def te_read_plan(
    tensor_names, num_layers: int = NUM_RETAINED_LAYERS,
) -> tuple[dict[str, str], list[dict[str, str]], dict[str, str]]:
    """File tensor names -> ``(embedding, per-layer, vision tower)`` mappings for the direct read.

    Each mapping is file name -> the key relative to the module it is read
    into (``language_model.embed_tokens`` / ``language_model.layers[i]`` /
    ``visual``), through :func:`te_key_map`. ``.comfy_quant`` markers are
    header-only and never read.
    """
    embed: dict[str, str] = {}
    layers: list[dict[str, str]] = [{} for _ in range(num_layers)]
    visual: dict[str, str] = {}
    for name in tensor_names:
        if name.endswith(".comfy_quant"):
            continue
        mapped = te_key_map(name)
        if mapped is None:
            continue
        if mapped.startswith("language_model.embed_tokens."):
            embed[name] = mapped[len("language_model.embed_tokens."):]
        elif mapped.startswith("language_model.layers."):
            idx_s, rest = mapped[len("language_model.layers."):].split(".", 1)
            layers[int(idx_s)][name] = rest
        elif mapped.startswith("visual."):
            visual[name] = mapped[len("visual."):]
        else:  # te_key_map maps nothing else; a new branch there must be planned here
            raise RuntimeError(f"H3 TE: no read plan for {name!r} -> {mapped!r}")
    return embed, layers, visual


def presentation(tokenizer, prompt: str, image_tokens: Sequence[int] = ()) -> tuple[list[int], list[int]]:
    """The token ids and per-token modality tags the encoder reads (Docs/MINIMAX_H3_DESIGN.md §3).

    ``image_tokens``: the vision-token count of each image, in order. T2V
    (empty): the prompt alone, all tag 1. I2V: per image, the
    ``"<Picture i>: "`` label (tag 1, i from 1) and its vision block of pads
    (tag 0); then the prompt (tag 1), each part tokenized on its own --
    tokenizing the joined string would glue the label's trailing space to the
    prompt's first word.
    """
    ids: list[int] = []
    tags: list[int] = []
    for i, count in enumerate(image_tokens):
        if int(count) < 1:
            raise ValueError(f"H3 TE: an image needs at least one vision token ({count})")
        label = list(tokenizer(PICTURE_LABEL.format(i + 1), add_special_tokens=False)["input_ids"])
        block = [VISION_START_ID] + [IMAGE_TOKEN_ID] * int(count) + [VISION_END_ID]
        ids += label + block
        tags += [MODALITY_TEXT] * len(label) + [MODALITY_VIDEO] * len(block)
    prompt_ids = list(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    if not prompt_ids:
        raise ValueError("プロンプトが空です（H3 は文章の条件づけが必須です）")
    ids += prompt_ids
    tags += [MODALITY_TEXT] * len(prompt_ids)
    return ids, tags


def vision_inv_freq(dim: int) -> torch.Tensor:
    """``Qwen3VLVisionRotaryEmbedding``'s inverse frequencies (its own ``__init__`` formula), float32 on the CPU."""
    return 1.0 / (VISION_ROPE_THETA ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))


def mm_token_type_ids(ids: list[int]) -> list[int]:
    """Qwen-internal token types: 1 on ``<|image_pad|>``, 0 elsewhere (DiffSynth TE 88-92,
    transformers ``ProcessorMixin.create_mm_token_type_ids``)."""
    return [1 if i == IMAGE_TOKEN_ID else 0 for i in ids]


def install_layer_read_hooks(layers, plans, path: str, header, device, stats: dict) -> list:
    """Hooks on each (meta) decoder layer: read it from ``path`` onto ``device`` right
    before its forward, drop it back to meta right after; returns the hook handles.

    The model's own ``forward`` (and its layer loop) is left untouched -- the
    hooks fire from ``Module.__call__``. ``stats`` accumulates ``bytes``,
    ``read_seconds`` and ``layers`` (how many layers were read).
    """
    handles = []
    for layer, plan in zip(layers, plans):

        def pre(module, args, kwargs, _plan=plan):
            t0 = time.perf_counter()
            stats["bytes"] += load_mapped_to(module, path, header, _plan, device)
            stats["read_seconds"] += time.perf_counter() - t0
            stats["layers"] += 1

        def post(module, args, kwargs, output):
            drop_to_meta(module)

        handles.append(layer.register_forward_pre_hook(pre, with_kwargs=True))
        handles.append(layer.register_forward_hook(post, with_kwargs=True))
    return handles


class H3TextEncoder:
    """Build (meta) -> encode (direct read) -> release, once per job (the pipeline's ``encode`` stage)."""

    def __init__(self, te_path: str, tokenizer_dir: str, *, device) -> None:
        self.te_path = str(te_path)
        self.tokenizer_dir = str(tokenizer_dir)
        self.device = torch.device(device)
        self._tokenizer = None
        self._model: nn.Module | None = None
        self._header = None
        self._embed_plan: dict[str, str] = {}
        self._layer_plans: list[dict[str, str]] = []
        self._visual_plan: dict[str, str] = {}
        #: What the last ``encode`` read (bytes / seconds / vision tokens), for the job's phases.
        self.read_report: dict = {}
        #: The token ids of the last ``encode`` (the gate compares them with the reference).
        self.last_ids: list[int] = []

    # -- load: tokenizer and the meta skeleton, no weights ------------------------

    def load(self) -> dict:
        from transformers import AutoTokenizer
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLModel,
            Qwen3VLTextRotaryEmbedding,
        )

        started = time.perf_counter()
        # Tokenizer first: a wrong folder is reported before anything else.
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_dir)
        special = ensure_extra_special_tokens(tokenizer)

        header, te_cfg = h3_format.inspect_h3_te(self.te_path)
        config = build_config()
        rope = check_rope(config.text_config)
        logger.info("H3 TE config: %s", rope)

        with torch.device("meta"):
            model = Qwen3VLModel(config)
        lm = model.language_model
        lm.norm = nn.Identity()  # the hidden state BEFORE the final norm (R2 D)
        model.to(torch.bfloat16)  # meta: free; assign=True keeps the file dtypes anyway
        model.requires_grad_(False)  # assign=True carries this flag over to the read tensors
        model.eval()
        # The rotary inverse frequencies are NON-persistent buffers: no file
        # carries them, so they would stay on meta. Recompute them on the CPU
        # from the checked config, in float32 (the rotary forwards cast to
        # float, but a bf16-rounded inv_freq would already be wrong).
        rotary = lm.rotary_emb
        if rotary.rope_type != "default":
            raise RuntimeError(f"H3 TE: rope_type {rotary.rope_type!r} is not 'default'")
        inv_freq, scaling = Qwen3VLTextRotaryEmbedding.compute_default_rope_parameters(
            config.text_config, device="cpu",
        )
        rotary.inv_freq = inv_freq.to(torch.float32)
        rotary.original_inv_freq = inv_freq.to(torch.float32).clone()
        rotary.attention_scaling = scaling
        vrot = model.visual.rotary_pos_emb
        vrot.inv_freq = vision_inv_freq(vrot.dim)

        quant = te_quant_names(te_cfg.quant)
        with torch.device("meta"):
            swapped = swap_in_int8_linears(model, quant, bias_dtype_of=lambda name: torch.bfloat16)

        embed_plan, layer_plans, visual_plan = te_read_plan(header.tensors, len(lm.layers))
        empty = [i for i, plan in enumerate(layer_plans) if not plan]
        if not embed_plan or empty:
            raise RuntimeError(
                f"H3 TE: {self.te_path} has no tensors for "
                + ("the embedding table" if not embed_plan else f"decoder layer(s) {empty[:10]}")
            )
        self._tokenizer = tokenizer
        self._model = model
        self._header = header
        self._embed_plan, self._layer_plans, self._visual_plan = embed_plan, layer_plans, visual_plan
        info = {
            "num_layers_file": te_cfg.num_layers,
            "num_layers_built": len(lm.layers),
            "int8_linears": swapped,
            "tensors_planned": len(embed_plan) + sum(len(p) for p in layer_plans),
            "vision_tensors_in_file": len(visual_plan),
            "rope": rope,
            "extra_special_tokens": special,
            "seconds": round(time.perf_counter() - started, 2),
        }
        logger.info("H3 TE built (weights are read during encode): %s", info)
        return info

    # -- encode ----------------------------------------------------------------

    def tokenize(self, prompt: str) -> list[int]:
        """Prompt -> ids exactly as DiffSynth / ComfyUI do it: no template, no specials."""
        if self._tokenizer is None:
            raise RuntimeError("H3TextEncoder.tokenize before load")
        return list(self._tokenizer(prompt, add_special_tokens=False)["input_ids"])

    def vision_inputs(self, images) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """Canvas images (PIL RGB) -> ``(pixel_values f32, image_grid_thw [K, 3] long,
        vision tokens per image)``.

        The Qwen3-VL image processor of the tokenizer folder (the same
        ``preprocessor_config.json`` as the official ``FL2VA/processor``),
        called once with all the images.
        """
        from transformers import AutoProcessor

        images = list(images)
        processor = AutoProcessor.from_pretrained(self.tokenizer_dir)
        image_processor = processor.image_processor
        vision = image_processor(images=images, return_tensors="pt")
        grid = vision["image_grid_thw"]
        merge = int(image_processor.merge_size) ** 2
        tokens = [int(grid[i].prod().item()) // merge for i in range(len(images))]
        return vision["pixel_values"], grid, tokens

    def encode(self, prompt: str, images=()) -> tuple[torch.Tensor, torch.Tensor]:
        """``([L, 5120] bf16 on device, [L] long modality tags)``.

        ``images`` (I2V): the canvas images (``engine_h3.image_cond.load_canvas_image``)
        the vision tower reads, in order; their vision blocks go in front of the
        prompt. None (T2V, or only middle key frames): tags are all text (1).
        """
        if self._model is None:
            raise RuntimeError("H3TextEncoder.encode before load")
        model = self._model
        lm = model.language_model
        images = list(images)
        vision = bool(images)  # whether the vision tower is read
        vision_kwargs: dict = {}
        image_tokens: list[int] = []
        if vision:
            if not self._visual_plan:
                raise RuntimeError(f"H3 TE: {self.te_path} has no vision tower (visual.*); I2V needs it")
            pixel_values, grid, image_tokens = self.vision_inputs(images)
            vision_kwargs = {"pixel_values": pixel_values, "image_grid_thw": grid}
        ids, tags_list = presentation(self._tokenizer, prompt, image_tokens)   # refuses an empty prompt
        stats = {"bytes": 0, "read_seconds": 0.0, "layers": 0}
        started = time.perf_counter()
        handles: list = []
        try:
            # The embedding table (1.5 GB), the vision tower (1.1 GB, I2V only)
            # and the rotary buffers go to the GPU for this forward only; the
            # decoder layers come in one by one through the hooks.
            t0 = time.perf_counter()
            stats["bytes"] += load_mapped_to(
                lm.embed_tokens, self.te_path, self._header, self._embed_plan, self.device,
            )
            if vision:
                stats["bytes"] += load_mapped_to(
                    model.visual, self.te_path, self._header, self._visual_plan, self.device,
                )
                move_module_tree(model.visual.rotary_pos_emb, self.device, skip=set())
            stats["read_seconds"] += time.perf_counter() - t0
            move_module_tree(lm.rotary_emb, self.device, skip=set())
            handles = install_layer_read_hooks(
                lm.layers, self._layer_plans, self.te_path, self._header, self.device, stats,
            )
            input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
            kwargs = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids), "use_cache": False}
            if vision:
                model.rope_deltas = None  # never carry a previous call's M-RoPE offsets
                kwargs["mm_token_type_ids"] = torch.tensor(
                    [mm_token_type_ids(ids)], dtype=torch.long, device=self.device,
                )
                kwargs["pixel_values"] = vision_kwargs["pixel_values"].to(self.device, torch.bfloat16)
                kwargs["image_grid_thw"] = vision_kwargs["image_grid_thw"].to(self.device, torch.long)
            with torch.no_grad():
                out = model(**kwargs)
            hidden = out.last_hidden_state[0]
            del out
            if hidden.dtype != torch.bfloat16:
                hidden = hidden.to(torch.bfloat16)
        finally:
            for handle in handles:
                handle.remove()
            # Whatever the forward reached (all of it, or a layer that failed):
            # no weight outlives the call.
            drop_to_meta(lm.embed_tokens)
            for layer in lm.layers:
                drop_to_meta(layer)
            if vision:
                drop_to_meta(model.visual)
                # The rotary buffer is computed, not read: put it back on the CPU so
                # the tower can be read again (drop_to_meta took it to meta too).
                model.visual.rotary_pos_emb.inv_freq = vision_inv_freq(model.visual.rotary_pos_emb.dim)
        if stats["layers"] != len(lm.layers):
            raise RuntimeError(f"H3 TE: {stats['layers']} of {len(lm.layers)} decoder layers ran")
        if int(hidden.shape[0]) != len(ids):
            raise RuntimeError(f"H3 TE: {hidden.shape[0]} hidden rows for {len(ids)} tokens")
        self.read_report = {
            "read_gib": round(stats["bytes"] / 2**30, 2),
            "read_seconds": round(stats["read_seconds"], 2),
            "layers_read": stats["layers"],
            "encode_seconds": round(time.perf_counter() - started, 2),
            "tokens": len(ids),
            "vision_tokens": int(sum(image_tokens)),
            "vision_tokens_per_image": [int(n) for n in image_tokens],
        }
        logger.info("H3 TE encode (direct read): %s", self.read_report)
        self.last_ids = ids
        tags = torch.tensor(tags_list, dtype=torch.long, device=self.device)
        return hidden, tags

    # -- release ---------------------------------------------------------------

    def release(self) -> None:
        """Drop the skeleton and the tokenizer (``encode`` already dropped every weight)."""
        self._model = None
        self._tokenizer = None
        self._header = None
        self._embed_plan, self._layer_plans, self._visual_plan = {}, [], {}
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
