"""MiniMax H3 text encoder: Qwen3-VL-32B's text tower, cut after decoder layer 50.

Ported in part from DiffSynth-Studio (Apache-2.0), commit 4e34969:
  diffsynth/models/minimax_h3_text_encoder.py            (the config values, the T2V presentation)
  diffsynth/utils/state_dict_converters/minimax_h3_text_encoder.py  (the Comfy-Org name mapping)
Copyright (c) the DiffSynth-Studio authors. Licensed under the Apache License,
Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0). Changes: only the
text tower (``Qwen3VLTextModel``) is built -- the vision tower is not needed for
T2V and comes with I2V in the second stage; built on the meta device and filled
by streaming; the 350 quantized Linears run through ``H3Int8Linear``; the
decoder layers are streamed CPU->GPU by ``GemmaLayerOffloadService``.

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

The modality tag of every T2V text token is 1 (text); vision-block tokens (I2V,
second stage) would be 0.
"""

from __future__ import annotations

import gc
import logging
import time

import torch
import torch.nn as nn

import h3_format
from engine.gemma.layer_offload_service import GemmaLayerOffloadService
from engine_h3.constants import MODALITY_TEXT
from engine_h3.quant_linear import swap_in_int8_linears
from engine_h3.sft_load import load_streaming, move_module_tree

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


def build_text_config():
    """The ``Qwen3VLTextConfig`` H3 uses (50 layers, SDPA attention)."""
    from transformers import Qwen3VLTextConfig

    config = Qwen3VLTextConfig(**TEXT_CONFIG)
    try:
        config._attn_implementation = "sdpa"
    except Exception:  # noqa: BLE001 -- older/newer setter shapes; verified below
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
    """Comfy-Org TE tensor name -> ``Qwen3VLTextModel`` state-dict key (None = not read).

    ``model.layers.N.*`` -> ``layers.N.*`` (N < 50), ``model.embed_tokens.*`` ->
    ``embed_tokens.*``. ``visual.*`` (the vision tower, I2V only), ``lm_head``,
    ``model.norm`` and any layer >= 50 are not read in this stage.
    """
    if name.startswith("model.layers."):
        idx = int(name[len("model.layers."):].split(".", 1)[0])
        if idx >= NUM_RETAINED_LAYERS:
            return None
        return name[len("model."):]
    if name.startswith("model.embed_tokens."):
        return name[len("model."):]
    return None


def te_quant_names(quant: dict) -> dict:
    """``h3_format`` TE quant keys (``model.layers.N.x``) -> module names in the built model."""
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


class H3TextEncoder:
    """Load -> encode -> release, once per job (stage 1 of the pipeline)."""

    def __init__(self, te_path: str, tokenizer_dir: str, *, layers_on_gpu: int, device) -> None:
        self.te_path = str(te_path)
        self.tokenizer_dir = str(tokenizer_dir)
        self.layers_on_gpu = int(layers_on_gpu)
        self.device = torch.device(device)
        self._tokenizer = None
        self._model: nn.Module | None = None
        self._service: GemmaLayerOffloadService | None = None

    # -- load ------------------------------------------------------------------

    def load(self) -> dict:
        from transformers import AutoTokenizer
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLTextModel,
            Qwen3VLTextRotaryEmbedding,
        )

        started = time.perf_counter()
        # Tokenizer first: a wrong folder is reported before 27 GB are read.
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_dir)
        special = ensure_extra_special_tokens(tokenizer)

        header, te_cfg = h3_format.inspect_h3_te(self.te_path)
        config = build_text_config()
        rope = check_rope(config)
        logger.info("H3 TE config: %s", rope)

        with torch.device("meta"):
            model = Qwen3VLTextModel(config)
        model.norm = nn.Identity()  # the hidden state BEFORE the final norm (R2 D)
        model.to(torch.bfloat16)  # meta: free; assign=True below keeps the file dtypes anyway
        # The rotary inverse frequencies are a NON-persistent buffer: no file
        # carries them, so they would stay on meta. Recompute them on the CPU
        # from the checked config, in float32 (the rotary forward casts to
        # float, but a bf16-rounded inv_freq would already be wrong).
        rotary = model.rotary_emb
        if rotary.rope_type != "default":
            raise RuntimeError(f"H3 TE: rope_type {rotary.rope_type!r} is not 'default'")
        inv_freq, scaling = Qwen3VLTextRotaryEmbedding.compute_default_rope_parameters(config, device="cpu")
        rotary.inv_freq = inv_freq.to(torch.float32)
        rotary.original_inv_freq = inv_freq.to(torch.float32).clone()
        rotary.attention_scaling = scaling

        quant = te_quant_names(te_cfg.quant)
        with torch.device("meta"):
            swapped = swap_in_int8_linears(model, quant, bias_dtype_of=lambda name: torch.bfloat16)
        report = load_streaming(model, self.te_path, header, key_map=te_key_map)
        model.requires_grad_(False)
        model.eval()

        layers = list(model.layers)
        # layers_on_gpu is the streaming window; 0 means ONE layer at a time
        # (GemmaLayerOffloadService treats 0 as "off", which would put all
        # 24 GB on the GPU, so it is never handed a 0).
        window = max(1, self.layers_on_gpu)
        if window >= len(layers):
            move_module_tree(model, self.device, skip=set())
            service = None
        else:
            skip = {id(m) for layer in layers for m in layer.modules()}
            move_module_tree(model, self.device, skip=skip)
            service = GemmaLayerOffloadService(layers_on_gpu=window, compute_device=self.device)
            service.install(model)
        self._tokenizer = tokenizer
        self._model = model
        self._service = service
        info = {
            "num_layers_file": te_cfg.num_layers,
            "num_layers_built": len(layers),
            "int8_linears": swapped,
            "loaded_tensors": report.loaded,
            "loaded_gib": round(report.bytes / 2**30, 2),
            "read_seconds": report.seconds,
            "layers_on_gpu_window": window if service is not None else len(layers),
            "rope": rope,
            "extra_special_tokens": special,
            "seconds": round(time.perf_counter() - started, 2),
        }
        logger.info("H3 TE loaded: %s", info)
        return info

    # -- encode ----------------------------------------------------------------

    def tokenize(self, prompt: str) -> list[int]:
        """Prompt -> ids exactly as DiffSynth / ComfyUI do it: no template, no specials."""
        if self._tokenizer is None:
            raise RuntimeError("H3TextEncoder.tokenize before load")
        return list(self._tokenizer(prompt, add_special_tokens=False)["input_ids"])

    def encode(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        """``([L, 5120] bf16 on device, [L] long modality tags)``; T2V tags are all text (1)."""
        if self._model is None:
            raise RuntimeError("H3TextEncoder.encode before load")
        ids = self.tokenize(prompt)
        if not ids:
            raise ValueError("プロンプトが空です（H3 は文章の条件づけが必須です）")
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            out = self._model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
            )
        hidden = out.last_hidden_state[0]
        if hidden.dtype != torch.bfloat16:
            hidden = hidden.to(torch.bfloat16)
        tags = torch.full((len(ids),), MODALITY_TEXT, dtype=torch.long, device=self.device)
        return hidden, tags

    # -- release ---------------------------------------------------------------

    def release(self) -> None:
        """Drop the model (GPU and CPU). Never calls the service's ``uninstall``:
        that moves every layer back onto the GPU, i.e. 24 GB of int8 at once."""
        self._service = None
        self._model = None
        self._tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
