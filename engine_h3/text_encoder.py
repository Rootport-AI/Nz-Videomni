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

The reads go past the OS file cache through one reused staging buffer
(``engine_h3.unbuffered_read``, 台帳 1-303): ``load`` makes the reader,
``release`` lets it go, and when it cannot be used here ``load`` records why
(``unbuffered_read: "on->off"``) and the buffered read (a fresh CPU buffer per
tensor) is used. Either way the same bytes reach the same device.

Combined read (te_combined_read, 台帳 1-303 方向 B)
--------------------------------------------------
``encode_many`` takes two or more prompts (a chain's clips, a negative prompt)
through ONE read of the weights, and runs its own layer loop instead of HF's
``forward``: each decoder layer is read once, every prompt's hidden state goes
through it, one prompt at a time, and the layer is dropped. The loop copies,
call for call, what ``Qwen3VLModel.forward`` / ``Qwen3VLTextModel.forward`` do
for one prompt (the embeddings, one vision call per prompt, the scatter, the
positions, the layer calls, DeepStack after the first layers, no final norm),
so each prompt's output is ``encode``'s for it bit for bit (VERIFICATION_LOG
§241). The waiting prompts' states stay in main memory between their turns.

Prompts whose reference part (the rows in front of the prompt: the labels and
the vision blocks) is the same share it (:func:`share_groups`): the reference
part goes through each layer once, its key / value rows are kept, and each
prompt's own rows go through with them in front (``blocked_causal_attention``
with an offset). Only where the measurement gave the same bits (VERIFICATION_LOG
§243, Docs/H3_TE_READ_NOTES.md §13): sequences of at most
:data:`SHARE_MAX_TOKENS` tokens whose text rows fall in attention blocks of at
least :data:`SHARE_MIN_ROWS` rows, and only prompts with a part the vision
tower reads. ``encode`` (one prompt) and its hooks are unchanged.

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
import hashlib
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

import h3_format
from engine_h3.constants import MODALITY_TEXT, MODALITY_VIDEO, REFERENCE_KINDS
from engine_h3.quant_linear import swap_in_int8_linears
from engine_h3.sft_load import drop_to_meta, load_mapped_to, move_module_tree
from engine_h3.unbuffered_read import UnbufferedReadUnavailable, UnbufferedReader

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
#: Ref2VA labels (台帳 §1-107): plain text, numbered by the reference card (slot).
VIDEO_LABEL = "<Video {}>: "
AUDIO_LABEL = "<Audio {}>: "
#: ``Qwen3VLVisionRotaryEmbedding``'s default theta (its inverse frequencies are a
#: non-persistent buffer, recomputed here because no file carries them).
VISION_ROPE_THETA = 10000.0


#: The name the text side's attention function is registered under (``AttentionInterface``).
H3_TEXT_ATTENTION = "nz_h3_blocked_math"
#: Query rows per block in ``blocked_causal_attention`` (the size measured in VERIFICATION_LOG §181). Not a setting.
TEXT_ATTENTION_QUERY_BLOCK = 512
#: A last block shorter than this is folded into the block before it. Blocks of 1, 2, 4, 8,
#: 12 and 16 rows did not match stock SDPA bit for bit (probably a different cuBLAS kernel
#: for small M); every height in 17-23 and 64-575 did, on dummy (1,64,N,128)/(1,8,N,128)
#: inputs at N = 3,123 / 4,100 / 6,148 (VERIFICATION_LOG §182). 64 keeps a margin. Not a
#: setting.
TEXT_ATTENTION_MIN_TAIL = 64

#: The combined read's share (台帳 1-303 方向 B): only a sequence of at most this many tokens
#: shares its reference part. Up to 9,216 (512 x 18) the attention's rows came out the same
#: bits whatever the sequence was cut to; from 9,217 on they did not (VERIFICATION_LOG §243,
#: Docs/H3_TE_READ_NOTES.md §13.3 (a), §13.5). Not a setting.
SHARE_MAX_TOKENS = 9216
#: ... and only when every attention block its text rows fall in has at least this many rows:
#: blocks of 16 rows or fewer gave other bits, 17 or more the same (the same records, §13.3 (b)).
#: Not a setting.
SHARE_MIN_ROWS = 17


def text_attention_blocks(n: int, block: int = TEXT_ATTENTION_QUERY_BLOCK,
                          min_tail: int = TEXT_ATTENTION_MIN_TAIL) -> list[tuple[int, int]]:
    """The ``(start, end)`` query blocks :func:`blocked_causal_attention` cuts ``n`` rows into,
    in row order: ``block`` rows each, a last block shorter than ``min_tail`` folded into the one
    before it. The one rule of the attention and of the share decision."""
    blocks = []
    s = 0
    while s < n:
        e = min(s + block, n)
        if n - e < min_tail:         # a short tail: make this the last block (at most block + min_tail - 1 rows)
            e = n
        blocks.append((s, e))
        s = e
    return blocks


def blocked_causal_attention(query, key, value, *, scale, block=TEXT_ATTENTION_QUERY_BLOCK,
                             min_tail=TEXT_ATTENTION_MIN_TAIL):
    """Causal GQA attention, query rows in blocks of ``block`` against the full key / value.

    The query rows may be the LAST ``n_q`` of the ``n_kv`` rows (the combined read's
    share: a prompt's own rows with the shared reference part's key / value in front):
    ``offset = n_kv - n_q``, every coordinate below is an absolute row of the ``n_kv``
    rows, the blocks are the ``n_kv`` rows' (:func:`text_attention_blocks`), a block
    that ends at or before the offset is skipped and the others start at
    ``max(start, offset)``. Offset 0 (``n_q == n_kv``, every call but the share's) is
    the blocks, the slices, the masks and the order of before.

    The mask slice keeps the rows' absolute positions (``is_causal=True`` on a
    block would align the triangle top-left, wrong for every block but the
    first). The math kernel is forced: it gives the numbers stock SDPA gave
    (which fell back to math anyway), and no future kernel choice can change
    them silently. At most a (block + min_tail - 1, N) score table exists at a
    time, not (N, N).

    The blocks are computed tallest first (a merged tail is up to block +
    min_tail - 1 rows): the allocator is then expected to carve its largest
    segment first and serve the 512-row blocks from it (台帳 §1-275; measured in
    VERIFICATION_LOG §197). The rows' values do not depend on the order.
    """
    n_q, n_kv = query.shape[2], key.shape[2]
    offset = n_kv - n_q
    if offset < 0:
        raise RuntimeError(f"H3 TE attention: {n_q} query rows for {n_kv} key rows")
    kv_idx = torch.arange(n_kv, device=query.device)
    # (s, e) in row order, absolute rows; the boundaries are the n_kv rows', cut at the offset
    blocks = [(max(s, offset), e) for s, e in text_attention_blocks(n_kv, block, min_tail) if e > offset]
    # Tallest block first (stable: ties keep row order). Each block's rows depend only on its own
    # q rows and the full k / v, so the order cannot change a single bit; it only decides which
    # block's temporaries the allocator carves its segments for first (台帳 §1-275, §197).
    order = sorted(range(len(blocks)), key=lambda i: blocks[i][1] - blocks[i][0], reverse=True)
    outs = [None] * len(blocks)
    with sdpa_kernel([SDPBackend.MATH]):
        for i in order:
            s, e = blocks[i]
            mask = torch.arange(s, e, device=query.device)[:, None] >= kv_idx[None, :]
            outs[i] = F.scaled_dot_product_attention(
                query[:, :, s - offset:e - offset], key, value, attn_mask=mask, is_causal=False, enable_gqa=True,
                scale=scale,
            )
    return torch.cat(outs, dim=2)


# --------------------------------------------------------------------------- #
# the combined read's share decision (pure: numbers only) and its run-time check
# --------------------------------------------------------------------------- #

def shared_text_heights(n: int, p: int) -> list[int]:
    """The heights of the attention blocks a prompt's own rows ``p .. n`` are computed in when
    its reference part (rows ``0 .. p``) is shared: the ``n`` rows' blocks cut at ``p``."""
    return [e - max(s, p) for s, e in text_attention_blocks(n) if e > p]


def share_facts(n: int, p: int) -> list[str]:
    """What keeps an item of ``n`` tokens, ``p`` of them its reference part, from sharing it
    (empty: it can): the facts as numbers and the condition, nothing more."""
    heights = shared_text_heights(n, p)
    if not heights:
        raise ValueError(f"H3 TE share: no text rows ({n} tokens, reference part {p})")
    facts = []
    if n > SHARE_MAX_TOKENS:
        facts.append(f"tokens {n} > {SHARE_MAX_TOKENS}")
    low = min(heights)
    if low < SHARE_MIN_ROWS:
        facts.append(f"rows {low} < {SHARE_MIN_ROWS}")
    return facts


def share_groups(rows) -> tuple[list[list[int]], dict[int, str]]:
    """The combined read's share decision, per item.

    ``rows``: per item ``(n, p, prefix)``; ``n`` its tokens, ``p`` its reference part's
    (the tokens in front of the prompt), ``prefix`` anything hashable that is equal
    exactly when two items' reference parts are (their ids, tags and vision inputs), or
    None for an item without a part the vision tower reads (it never shares).

    Returns ``(groups, reasons)``: ``groups`` the lists of items (ascending) that share
    their reference part, each of two or more, in the order of their first item;
    ``reasons`` item -> what was found for every other item: ``"no_vision_tokens"``,
    ``"no_same_references"``, :func:`share_facts` joined with ``"; "``, or
    ``"others_with_same_references_not_shared"`` (it could, no other could).
    """
    by_prefix: dict = {}
    reasons: dict[int, str] = {}
    for k, (_n, _p, prefix) in enumerate(rows):
        if prefix is None:
            reasons[k] = "no_vision_tokens"
        else:
            by_prefix.setdefault(prefix, []).append(k)
    groups: list[list[int]] = []
    for members in by_prefix.values():
        if len(members) < 2:
            reasons[members[0]] = "no_same_references"
            continue
        able = []
        for k in members:
            facts = share_facts(int(rows[k][0]), int(rows[k][1]))
            if facts:
                reasons[k] = "; ".join(facts)
            else:
                able.append(k)
        if len(able) >= 2:
            groups.append(able)
        else:
            for k in able:
                reasons[k] = "others_with_same_references_not_shared"
    return groups, dict(sorted(reasons.items()))


def check_shared_prefix(states, p: int) -> None:
    """The run-time check of a share group (each state: ``h`` the embeddings, ``cos`` / ``sin``,
    ``tpos`` the text position ids, ``vpm`` the vision rows): the reference part (rows
    ``0 .. p``) of every member is the first member's, no member has text position ids, and
    no vision row lies in a member's own rows. A mismatch is a bug, never a fallback: it raises."""
    first = states[0]
    for name, label in (("h", "embeddings"), ("cos", "cos"), ("sin", "sin")):
        if not all(torch.equal(st[name][:, :p], first[name][:, :p]) for st in states[1:]):
            raise RuntimeError(f"H3 TE share: the reference part's {label} differ within a group")
    if any(st["tpos"] is not None for st in states):
        raise RuntimeError("H3 TE share: a member has text position ids")
    if any(st["vpm"] is not None and bool(st["vpm"][:, p:].any()) for st in states):
        raise RuntimeError("H3 TE share: image rows in a member's own rows")


class _KeepKeyValue:
    """``past_key_values`` for a shared reference part: keeps the key / value the layer's
    attention hands to ``update`` (after the rotary) and hands them back as they are."""

    def __init__(self) -> None:
        self.k = self.v = None

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        self.k, self.v = key_states, value_states
        return key_states, value_states


def _like_layout(t: torch.Tensor, n_rows: int) -> torch.Tensor:
    """An empty ``(B, H, n_rows, D)`` tensor whose memory order is ``t``'s (the attention's key /
    value are a transposed ``(B, N, H, D)`` store; the joined ones keep that order)."""
    perm = sorted(range(t.ndim), key=lambda d: t.stride(d), reverse=True)
    shape = list(t.shape)
    shape[2] = n_rows
    base = torch.empty([shape[d] for d in perm], dtype=t.dtype, device=t.device)
    inv = [perm.index(d) for d in range(t.ndim)]
    return base.permute(*inv)


class _JoinKeyValue:
    """``past_key_values`` for a member's own rows: the kept reference part's key / value
    (rows ``0 .. p``) in front of the member's new ones, copied into one store of the new
    ones' memory order (not ``torch.cat``)."""

    def __init__(self, k0, v0, p: int, n: int) -> None:
        self.k0, self.v0, self.p, self.n = k0, v0, int(p), int(n)

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        if self.p + key_states.shape[2] != self.n or self.k0.shape[2] != self.p:
            raise RuntimeError(f"H3 TE share: {self.k0.shape[2]} + {key_states.shape[2]} rows for {self.n}")
        out = []
        for old, new in ((self.k0, key_states), (self.v0, value_states)):
            buf = _like_layout(new, self.n)
            buf[:, :, :self.p] = old
            buf[:, :, self.p:] = new
            out.append(buf)
        return out[0], out[1]


def _vision_key(pixel_values: torch.Tensor, grid: torch.Tensor) -> str:
    """The sha256 of an item's vision inputs (pixel values and grid, CPU)."""
    h = hashlib.sha256()
    h.update(pixel_values.detach().contiguous().numpy().tobytes())
    h.update(grid.detach().to(torch.long).contiguous().numpy().tobytes())
    return h.hexdigest()


def h3_text_attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
    """The text side's attention function (Hugging Face's contract): q (B, 64, N, 128),
    k / v (B, 8, N, 128) -> ``(out.transpose(1, 2).contiguous(), None)``.

    The name has no entry in the mask-function table, so Hugging Face drops a
    2-D mask instead of building one and ``attention_mask`` arrives as None
    (only a 4-D mask would pass through). Causality is applied here; the
    product's batch of 1 with an all-ones mask loses nothing.
    """
    if attention_mask is not None:
        raise RuntimeError(f"H3 TE attention: unexpected attention_mask {tuple(attention_mask.shape)}")
    scale = scaling if scaling is not None else module.scaling
    out = blocked_causal_attention(query, key, value, scale=scale)
    return out.transpose(1, 2).contiguous(), None


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
    """The ``Qwen3VLConfig`` H3 uses: the text side cut to 50 layers with ``h3_text_attention``,
    the vision tower on SDPA."""
    from transformers import AttentionInterface, Qwen3VLConfig

    # Registered before any model is built: an unknown name stops the build (ValueError).
    # Registering again only replaces the same dict entry.
    AttentionInterface.register(H3_TEXT_ATTENTION, h3_text_attention)
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
    # The text side only: stock SDPA falls back to math there and builds the whole
    # 64 x N x N fp32 score table. The vision tower stays on SDPA (per-image cu_seqlens,
    # linear in the image count). text_config has no sub-configs, so this stays put.
    config.text_config._attn_implementation = H3_TEXT_ATTENTION
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


@dataclass(frozen=True)
class RefPresentation:
    """One Ref2VA reference as the text encoder presents it (台帳 §1-107).

    ``kind``: ``"picture"`` / ``"video"`` / ``"audio"``; ``slot``: the card
    number (the n of ``<Picture n>`` etc., from 1). ``vision_tokens``: one count
    for a picture, one per pair for a video, none for audio. ``labels``: a
    video's per-pair ``"<x.x seconds>"`` (same length as ``vision_tokens``),
    empty otherwise.
    """

    kind: str
    slot: int
    vision_tokens: tuple[int, ...] = ()
    labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class RefMedia:
    """One Ref2VA reference as :meth:`H3TextEncoder.encode` takes it.

    ``image`` (picture): a PIL RGB image, or the ``[1, 3, 1, h, w]`` float32
    tensor of ``engine_h3.reference.load_reference_image`` (``uint8 / 255``;
    turned back into the image exactly). ``pairs`` (video): the list
    ``engine_h3.reference.qwen_pairs`` returns, ``[(uint8 [2, h, w, 3], label)]``.
    Audio carries neither.
    """

    kind: str
    slot: int
    image: object = None
    pairs: tuple = ()


def _vision_block(count: int) -> list[int]:
    if int(count) < 1:
        raise ValueError(f"H3 TE: an image needs at least one vision token ({count})")
    return [VISION_START_ID] + [IMAGE_TOKEN_ID] * int(count) + [VISION_END_ID]


def presentation(
    tokenizer, prompt: str, image_tokens: Sequence[int] = (), *,
    ref_items: Sequence[RefPresentation] | None = None,
) -> tuple[list[int], list[int]]:
    """The token ids and per-token modality tags the encoder reads (Docs/MINIMAX_H3_DESIGN.md §3).

    ``image_tokens``: the vision-token count of each image, in order. T2V
    (empty): the prompt alone, all tag 1. I2V: per image, the
    ``"<Picture i>: "`` label (tag 1, i from 1) and its vision block of pads
    (tag 0); then the prompt (tag 1), each part tokenized on its own --
    tokenizing the joined string would glue the label's trailing space to the
    prompt's first word.

    ``ref_items`` (Ref2VA, 台帳 §1-107): None or empty takes the path above
    unchanged. Otherwise, per item in the given order, each part tokenized on
    its own, no separator: picture ``"<Picture {slot}>: "`` (tag 1) + vision
    block (tag 0); audio ``"<Audio {slot}>: "`` (tag 1) alone; video
    ``"<Video {slot}>: "`` (tag 1), then per pair its ``"<x.x seconds>"`` label
    (tag 1) + vision block (tag 0); then the prompt (tag 1). With references
    the key frames do not reach the text encoder (ComfyUI's AddGuide):
    ``image_tokens`` is not read (:meth:`H3TextEncoder.encode` refuses both at
    once).
    """
    if ref_items:
        return _ref_presentation(tokenizer, prompt, ref_items)
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


def _ref_presentation(tokenizer, prompt: str, ref_items: Sequence[RefPresentation]) -> tuple[list[int], list[int]]:
    """:func:`presentation` with references (its docstring)."""
    ids: list[int] = []
    tags: list[int] = []

    def text(s: str) -> None:
        part = list(tokenizer(s, add_special_tokens=False)["input_ids"])
        ids.extend(part)
        tags.extend([MODALITY_TEXT] * len(part))

    def block(count: int) -> None:
        part = _vision_block(count)
        ids.extend(part)
        tags.extend([MODALITY_VIDEO] * len(part))

    for item in ref_items:
        if item.kind == "picture":
            if len(item.vision_tokens) != 1 or item.labels:
                raise ValueError(f"H3 TE: a picture reference has one vision block and no label ({item})")
            text(PICTURE_LABEL.format(int(item.slot)))
            block(item.vision_tokens[0])
        elif item.kind == "audio":
            if item.vision_tokens or item.labels:
                raise ValueError(f"H3 TE: an audio reference has no vision block ({item})")
            text(AUDIO_LABEL.format(int(item.slot)))
        elif item.kind == "video":
            if not item.vision_tokens or len(item.labels) != len(item.vision_tokens):
                raise ValueError(f"H3 TE: a video reference needs one label per pair ({item})")
            text(VIDEO_LABEL.format(int(item.slot)))
            for label, count in zip(item.labels, item.vision_tokens):
                text(label)
                block(count)
        else:
            raise ValueError(f"H3 TE: unknown reference kind {item.kind!r} (expected one of {REFERENCE_KINDS})")
    prompt_ids = list(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    if not prompt_ids:
        raise ValueError("プロンプトが空です（H3 は文章の条件づけが必須です）")
    ids += prompt_ids
    tags += [MODALITY_TEXT] * len(prompt_ids)
    return ids, tags


def _picture_image(ref: RefMedia):
    """A picture reference's image as PIL RGB (a ``[1, 3, 1, h, w]`` ``uint8 / 255`` tensor is turned back exactly)."""
    from PIL import Image

    img = ref.image
    if isinstance(img, torch.Tensor):
        if img.ndim != 5 or img.shape[0] != 1 or img.shape[1] != 3 or img.shape[2] != 1:
            raise ValueError(f"H3 TE: picture reference {ref.slot}: [1, 3, 1, h, w] expected, got {tuple(img.shape)}")
        arr = (img.detach().to("cpu", torch.float32)[0, :, 0] * 255.0).round().clamp(0, 255)
        img = Image.fromarray(arr.to(torch.uint8).permute(1, 2, 0).contiguous().numpy(), "RGB")
    if not isinstance(img, Image.Image):
        raise ValueError(f"H3 TE: picture reference {ref.slot} has no image")
    return img.convert("RGB")


def vision_inv_freq(dim: int) -> torch.Tensor:
    """``Qwen3VLVisionRotaryEmbedding``'s inverse frequencies (its own ``__init__`` formula), float32 on the CPU."""
    return 1.0 / (VISION_ROPE_THETA ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))


def mm_token_type_ids(ids: list[int]) -> list[int]:
    """Qwen-internal token types: 1 on ``<|image_pad|>``, 0 elsewhere (DiffSynth TE 88-92,
    transformers ``ProcessorMixin.create_mm_token_type_ids``)."""
    return [1 if i == IMAGE_TOKEN_ID else 0 for i in ids]


def install_layer_read_hooks(layers, plans, path: str, header, device, stats: dict, *, reader=None) -> list:
    """Hooks on each (meta) decoder layer: read it from ``path`` onto ``device`` right
    before its forward, drop it back to meta right after; returns the hook handles.

    The model's own ``forward`` (and its layer loop) is left untouched -- the
    hooks fire from ``Module.__call__``. ``stats`` accumulates ``bytes``,
    ``read_seconds`` and ``layers`` (how many layers were read). ``reader``
    (``engine_h3.unbuffered_read.UnbufferedReader``, or None for the buffered
    read) is handed to ``load_mapped_to``.
    """
    handles = []
    for layer, plan in zip(layers, plans):

        def pre(module, args, kwargs, _plan=plan):
            t0 = time.perf_counter()
            stats["bytes"] += load_mapped_to(module, path, header, _plan, device, reader=reader)
            stats["read_seconds"] += time.perf_counter() - t0
            stats["layers"] += 1

        def post(module, args, kwargs, output):
            drop_to_meta(module)

        handles.append(layer.register_forward_pre_hook(pre, with_kwargs=True))
        handles.append(layer.register_forward_hook(post, with_kwargs=True))
    return handles


def install_drop_after_forward_hooks(modules) -> list:
    """Hooks that drop each module's weights to meta right after its forward returns
    (the embedding table and the vision tower: their VRAM is back before the 50
    layers run); returns the hook handles.

    The vision tower runs once per forward for image input (``pixel_values``)
    only. Video input (``pixel_values_videos``) would call it a second time on
    meta weights and fail; this TE passes video pairs as images (see ``encode``).
    """
    def post(module, args, output):
        drop_to_meta(module)
    return [m.register_forward_hook(post) for m in modules]


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
        #: The unbuffered reader of the stage (``load`` makes it, ``release`` lets it go);
        #: None = the buffered read (it could not be used here).
        self._reader: UnbufferedReader | None = None
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
        # The unbuffered read (台帳 1-303): decided here, once per stage; when it cannot
        # be used, the buffered read is used and what failed is recorded.
        self._close_reader()
        read_mode: dict = {"unbuffered_read": "on"}
        try:
            self._reader = UnbufferedReader(self.te_path, self.device)
        except UnbufferedReadUnavailable as exc:
            read_mode = {"unbuffered_read": "on->off", "unbuffered_read_reason": str(exc)}
            logger.warning("H3 TE: unbuffered read not used, reading through the OS cache: %s", exc)
        info = {
            "num_layers_file": te_cfg.num_layers,
            "num_layers_built": len(lm.layers),
            "int8_linears": swapped,
            "tensors_planned": len(embed_plan) + sum(len(p) for p in layer_plans),
            "vision_tensors_in_file": len(visual_plan),
            "rope": rope,
            "extra_special_tokens": special,
            **read_mode,
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

    def ref_vision_inputs(
        self, refs: Sequence[RefMedia],
    ) -> tuple[list[RefPresentation], torch.Tensor | None, torch.Tensor | None, list[dict]]:
        """References -> ``(ref_items, pixel_values f32 | None, image_grid_thw [K, 3] | None, per-reference report)``.

        Pictures go through :meth:`vision_inputs` together (one call, the same
        path as the key frames); each video pair through
        ``engine_h3.reference.pair_patches`` (one grid row ``[1, gh, gw]`` per
        pair). The rows of ``pixel_values`` / ``image_grid_thw`` follow the
        order of the vision blocks in the token ids (Hugging Face's
        ``get_rope_index`` takes one grid row per run of image pads, in order).
        None when no reference has a vision block (audio only).
        """
        from PIL import Image

        from engine_h3.reference import pair_patches

        pictures = []
        for ref in refs:
            if ref.kind not in REFERENCE_KINDS:
                raise ValueError(f"H3 TE: unknown reference kind {ref.kind!r} (expected one of {REFERENCE_KINDS})")
            if ref.kind == "picture":
                pictures.append(_picture_image(ref))
        picture_chunks: list[tuple[torch.Tensor, torch.Tensor, int]] = []
        if pictures:
            pv, grid, tokens = self.vision_inputs(pictures)
            start = 0
            for i, count in enumerate(tokens):
                rows = int(grid[i].prod().item())
                picture_chunks.append((pv[start:start + rows], grid[i:i + 1], count))
                start += rows
        patch = int(VISION_CONFIG["patch_size"])
        merge = int(VISION_CONFIG["spatial_merge_size"])
        items: list[RefPresentation] = []
        pixel_parts: list[torch.Tensor] = []
        grid_parts: list[torch.Tensor] = []
        report: list[dict] = []
        next_picture = 0
        for ref in refs:
            slot = int(ref.slot)
            if ref.kind == "picture":
                pv, grid, count = picture_chunks[next_picture]
                next_picture += 1
                pixel_parts.append(pv)
                grid_parts.append(grid)
                items.append(RefPresentation("picture", slot, (int(count),)))
                report.append({"kind": "picture", "slot": slot, "vision_tokens": int(count)})
            elif ref.kind == "video":
                if not ref.pairs:
                    raise ValueError(f"H3 TE: video reference {slot} has no frame pairs")
                counts, labels = [], []
                for pair, label in ref.pairs:
                    patches, grid = pair_patches(pair, patch=patch, merge=merge)
                    pixel_parts.append(patches)
                    grid_parts.append(grid)
                    counts.append(int(grid.prod().item()) // merge**2)
                    labels.append(str(label))
                items.append(RefPresentation("video", slot, tuple(counts), tuple(labels)))
                report.append({"kind": "video", "slot": slot, "pairs": len(counts), "vision_tokens": int(sum(counts))})
            else:
                items.append(RefPresentation("audio", slot))
                report.append({"kind": "audio", "slot": slot, "vision_tokens": 0})
        if not pixel_parts:
            return items, None, None, report
        return items, torch.cat(pixel_parts, dim=0), torch.cat(grid_parts, dim=0), report

    def encode(
        self, prompt: str, images=(), *, refs: Sequence[RefMedia] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``([L, 5120] bf16 on device, [L] long modality tags)``.

        ``images`` (I2V): the canvas images (``engine_h3.image_cond.load_canvas_image``)
        the vision tower reads, in order; their vision blocks go in front of the
        prompt. None (T2V, or only middle key frames): tags are all text (1).

        ``refs`` (Ref2VA, 台帳 §1-107): the references in presentation order
        (:class:`RefMedia`); None or empty is the path above, unchanged. With
        references ``images`` must be empty (refused here). Audio-only
        references do not read the vision tower. ``read_report["references"]``
        (only when ``refs`` is given) lists each reference's vision tokens.
        """
        images = list(images)
        refs = list(refs or ())
        if refs and images:
            raise ValueError("H3 TE: key-frame images and references cannot both be given to the text encoder")
        if self._model is None:
            raise RuntimeError("H3TextEncoder.encode before load")
        model = self._model
        lm = model.language_model
        ref_items: list[RefPresentation] | None = None
        ref_report: list[dict] = []
        vision_kwargs: dict = {}
        image_tokens: list[int] = []
        if refs:
            ref_items, pixel_values, grid, ref_report = self.ref_vision_inputs(refs)
            if pixel_values is not None:
                vision_kwargs = {"pixel_values": pixel_values, "image_grid_thw": grid}
        elif images:
            pixel_values, grid, image_tokens = self.vision_inputs(images)
            vision_kwargs = {"pixel_values": pixel_values, "image_grid_thw": grid}
        vision = bool(vision_kwargs)  # whether the vision tower is read
        if vision and not self._visual_plan:
            raise RuntimeError(f"H3 TE: {self.te_path} has no vision tower (visual.*); images need it")
        ids, tags_list = presentation(self._tokenizer, prompt, image_tokens, ref_items=ref_items)   # refuses an empty prompt
        stats = {"bytes": 0, "read_seconds": 0.0, "layers": 0}
        started = time.perf_counter()
        handles: list = []
        try:
            # The embedding table (1.5 GB), the vision tower (1.1 GB, I2V only)
            # and the rotary buffers go to the GPU here; each stays only until its own
            # forward returns (install_drop_after_forward_hooks); the decoder layers
            # come in one by one through the hooks.
            t0 = time.perf_counter()
            stats["bytes"] += load_mapped_to(
                lm.embed_tokens, self.te_path, self._header, self._embed_plan, self.device, reader=self._reader,
            )
            if vision:
                stats["bytes"] += load_mapped_to(
                    model.visual, self.te_path, self._header, self._visual_plan, self.device, reader=self._reader,
                )
                move_module_tree(model.visual.rotary_pos_emb, self.device, skip=set())
            stats["read_seconds"] += time.perf_counter() - t0
            move_module_tree(lm.rotary_emb, self.device, skip=set())
            handles = install_drop_after_forward_hooks([lm.embed_tokens] + ([model.visual] if vision else []))
            handles += install_layer_read_hooks(
                lm.layers, self._layer_plans, self.te_path, self._header, self.device, stats, reader=self._reader,
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
        if refs:
            self.read_report["references"] = ref_report
        logger.info("H3 TE encode (direct read): %s", self.read_report)
        self.last_ids = ids
        tags = torch.tensor(tags_list, dtype=torch.long, device=self.device)
        return hidden, tags

    # -- encode_many: the combined read (te_combined_read, 台帳 1-303 方向 B) ----

    def encode_many(self, items) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], list[dict], dict]:
        """Two or more prompts through ONE read of the weights (module docstring).

        ``items``: ``(prompt, images, refs)`` per prompt, as :meth:`encode` takes them
        (``images`` the vision tower's canvases, ``refs`` the :class:`RefMedia`; None /
        empty = none; both at once is refused). One prompt is :meth:`encode`'s (refused here).

        Returns ``(outs, counts, report)``, per item in the given order: ``outs[k] =
        (hidden [L, 5120] bf16, tags [L] long)``, both on the CPU, :meth:`encode`'s
        values bit for bit; ``counts[k]`` :meth:`encode`'s read report numbers for it
        (``tokens``, ``vision_tokens``, ``vision_tokens_per_image``, and ``references``
        with references). ``report``: ``items`` (how many), ``read_gib`` /
        ``read_seconds`` (the read, all items together), ``layers_read``,
        ``encode_seconds``, ``shared`` (per share group: ``{"reference_rows": P,
        "items": [k, ...]}``) and ``not_shared`` (``[{"item": k, "reason": str}]``, the
        facts :func:`share_groups` found). ``read_report`` / ``last_ids`` are not touched.

        Whatever fails, the embedding table, the vision tower and every layer are back on
        meta when this returns or raises, and no state of the call is kept.
        """
        items = [(str(prompt), list(images or ()), list(refs or ())) for prompt, images, refs in items]
        if len(items) < 2:
            raise ValueError(f"H3 TE: encode_many takes two or more prompts ({len(items)}); one goes through encode")
        if self._model is None:
            raise RuntimeError("H3TextEncoder.encode_many before load")
        model = self._model
        lm = model.language_model
        dev = self.device
        host = torch.device("cpu")
        cuda = dev.type == "cuda"
        started = time.perf_counter()

        def to_dev(t):
            if t is None:
                return None
            t = t.to(dev)
            if cuda:
                torch.cuda.synchronize(dev)   # each copy is waited for (no pinned memory)
            return t

        def to_host(t):
            return None if t is None else t.to(host)

        # 1. The inputs, on the CPU (encode's steps); the vision inputs kept once per equal key.
        entries: list[dict] = []
        pixels: dict[str, tuple] = {}
        for prompt, images, refs in items:
            if refs and images:
                raise ValueError("H3 TE: key-frame images and references cannot both be given to the text encoder")
            ref_items = None
            ref_report: list[dict] = []
            image_tokens: list[int] = []
            pv = grid = None
            if refs:
                ref_items, pv, grid, ref_report = self.ref_vision_inputs(refs)
            elif images:
                pv, grid, image_tokens = self.vision_inputs(images)
            if pv is not None and not self._visual_plan:
                raise RuntimeError(f"H3 TE: {self.te_path} has no vision tower (visual.*); images need it")
            ids, tags = presentation(self._tokenizer, prompt, image_tokens, ref_items=ref_items)   # refuses an empty prompt
            prompt_ids = self.tokenize(prompt)
            p = len(ids) - len(prompt_ids)
            if p < 0 or ids[p:] != prompt_ids:
                raise RuntimeError("H3 TE: the prompt's tokens are not the presentation's last part")
            key = None
            if pv is not None:
                key = _vision_key(pv, grid)
                pixels.setdefault(key, (pv, grid))
            del pv, grid
            counts = {
                "tokens": len(ids),
                "vision_tokens": int(sum(image_tokens)),
                "vision_tokens_per_image": [int(n) for n in image_tokens],
            }
            if refs:
                counts["references"] = ref_report
            entries.append({"ids": ids, "tags": tags, "n": len(ids), "p": p, "key": key, "counts": counts})

        # 2. The share decision (numbers only).
        groups, reasons = share_groups([
            (e["n"], e["p"], (tuple(e["ids"][:e["p"]]), tuple(e["tags"][:e["p"]]), e["key"])
             if e["key"] is not None and e["p"] > 0 else None)
            for e in entries
        ])
        vision = bool(pixels)
        stats = {"bytes": 0, "read_seconds": 0.0, "layers": 0}
        states: list = [None] * len(entries)     # per item not shared: its state, in main memory
        shared: list = []                        # per group: the reference part's state and the members'
        deepstack_by_key: dict[str, list] = {}   # the DeepStack features, once per vision key (main memory)
        vision_out: dict = {}
        try:
            with torch.no_grad():
                # 3. The embedding table and (any vision input) the vision tower, read at the same time.
                t0 = time.perf_counter()
                stats["bytes"] += load_mapped_to(
                    lm.embed_tokens, self.te_path, self._header, self._embed_plan, dev, reader=self._reader,
                )
                if vision:
                    stats["bytes"] += load_mapped_to(
                        model.visual, self.te_path, self._header, self._visual_plan, dev, reader=self._reader,
                    )
                    move_module_tree(model.visual.rotary_pos_emb, dev, skip=set())
                stats["read_seconds"] += time.perf_counter() - t0
                move_module_tree(lm.rotary_emb, dev, skip=set())
                from transformers.masking_utils import create_causal_mask

                last_use = {e["key"]: k for k, e in enumerate(entries) if e["key"] is not None}
                # 3-4. Per item: the embedding, the vision call (once per key), the scatter, the
                # positions and the rotary (Qwen3VLModel.forward / Qwen3VLTextModel.forward).
                for k, e in enumerate(entries):
                    input_ids = torch.tensor([e["ids"]], dtype=torch.long, device=dev)
                    attention_mask = torch.ones_like(input_ids)
                    inputs_embeds = lm.embed_tokens(input_ids)
                    vpm = position_ids = None
                    key = e["key"]
                    if key is not None:
                        if key not in vision_out:
                            pv, grid = pixels.pop(key)
                            model.rope_deltas = None   # never carry a previous call's M-RoPE offsets
                            grid_dev = grid.to(dev, torch.long)
                            out = model.get_image_features(
                                pv.to(dev, torch.bfloat16), grid_dev, return_dict=True, use_cache=False,
                            )
                            vision_out[key] = (out.pooler_output, grid_dev)
                            deepstack_by_key[key] = [to_host(d) for d in out.deepstack_features]
                            del out, pv, grid
                        pooled, grid_dev = vision_out[key]
                        image_embeds = torch.cat(pooled, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
                        image_mask, _ = model.get_placeholder_mask(
                            input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds,
                        )
                        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
                        vpm = image_mask[..., 0]
                        del image_embeds, image_mask
                        model.rope_deltas = None
                        mm = torch.tensor([mm_token_type_ids(e["ids"])], dtype=torch.long, device=dev)
                        position_ids = model.compute_3d_position_ids(
                            input_ids=input_ids, image_grid_thw=grid_dev, video_grid_thw=None,
                            inputs_embeds=inputs_embeds, attention_mask=attention_mask, past_key_values=None,
                            mm_token_type_ids=mm,
                        )
                        del pooled, grid_dev, mm
                        if last_use[key] == k:
                            del vision_out[key]
                    if position_ids is None:
                        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
                        position_ids = position_ids.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
                    elif position_ids.ndim == 2:
                        position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)
                    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
                        tpos = position_ids[0]
                        position_ids = position_ids[1:]
                    else:
                        tpos = None
                    mask = create_causal_mask(
                        config=lm.config, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                        past_key_values=None, position_ids=tpos,
                    )
                    if mask is not None:
                        raise RuntimeError(f"H3 TE: create_causal_mask gave {type(mask).__name__}, None expected")
                    cos, sin = lm.rotary_emb(inputs_embeds, position_ids)
                    states[k] = {
                        "h": to_host(inputs_embeds), "cos": to_host(cos), "sin": to_host(sin),
                        "tpos": to_host(tpos), "vpm": to_host(vpm), "ds": key,
                    }
                    del input_ids, attention_mask, inputs_embeds, position_ids, tpos, vpm, cos, sin, mask
                # The table and the tower go before the layers run.
                drop_to_meta(lm.embed_tokens)
                if vision:
                    drop_to_meta(model.visual)
                    model.visual.rotary_pos_emb.inv_freq = vision_inv_freq(model.visual.rotary_pos_emb.dim)
                vision_out.clear()
                pixels.clear()

                # 5. The share groups: checked at run time, then split into the reference part and
                # each member's own rows (the positions taken from the member's full length).
                for members in groups:
                    p = entries[members[0]]["p"]
                    check_shared_prefix([states[k] for k in members], p)
                    f0 = states[members[0]]
                    group = {
                        "p": p, "h": f0["h"][:, :p].clone(), "cos": f0["cos"][:, :p].clone(),
                        "sin": f0["sin"][:, :p].clone(), "ds": f0["ds"],
                        "vpm": None if f0["vpm"] is None else f0["vpm"][:, :p].clone(),
                        "members": [],
                    }
                    for k in sorted(members, key=lambda m: entries[m]["n"], reverse=True):
                        st = states[k]
                        group["members"].append((k, {
                            "h": st["h"][:, p:].clone(), "cos": st["cos"][:, p:].clone(),
                            "sin": st["sin"][:, p:].clone(),
                        }))
                        states[k] = None
                    del f0
                    shared.append(group)

                # 6. The 50 layers: each read once, every unit through it (heaviest first), dropped.
                def weight(n):
                    return max(e - s for s, e in text_attention_blocks(n)) * n

                units = [("item", k, weight(entries[k]["n"])) for k in range(len(entries)) if states[k] is not None]
                units += [("group", g, weight(grp["p"])) for g, grp in enumerate(shared)]
                order = [(kind, idx) for kind, idx, _w in sorted(units, key=lambda u: u[2], reverse=True)]
                deepstack_layers = max((len(v) for v in deepstack_by_key.values()), default=0)
                for li, layer in enumerate(lm.layers):
                    t0 = time.perf_counter()
                    stats["bytes"] += load_mapped_to(
                        layer, self.te_path, self._header, self._layer_plans[li], dev, reader=self._reader,
                    )
                    stats["read_seconds"] += time.perf_counter() - t0
                    stats["layers"] += 1
                    for kind, idx in order:
                        if kind == "item":
                            st = states[idx]
                            cos, sin = to_dev(st["cos"]), to_dev(st["sin"])
                            h = layer(
                                to_dev(st["h"]), attention_mask=None, position_ids=to_dev(st["tpos"]),
                                past_key_values=None, position_embeddings=(cos, sin),
                            )
                            del cos, sin
                            if st["ds"] is not None and li < len(deepstack_by_key[st["ds"]]):
                                h = lm._deepstack_process(
                                    h, to_dev(st["vpm"]), to_dev(deepstack_by_key[st["ds"]][li]),
                                )
                            st["h"] = to_host(h)
                            del h
                            continue
                        grp = shared[idx]
                        keep = _KeepKeyValue()
                        cos, sin = to_dev(grp["cos"]), to_dev(grp["sin"])
                        out_ref = layer(
                            to_dev(grp["h"]), attention_mask=None, position_ids=None, past_key_values=keep,
                            position_embeddings=(cos, sin),
                        )
                        del cos, sin
                        for _k, m in grp["members"]:
                            join = _JoinKeyValue(keep.k, keep.v, grp["p"], grp["p"] + int(m["h"].shape[1]))
                            cos, sin = to_dev(m["cos"]), to_dev(m["sin"])
                            out = layer(
                                to_dev(m["h"]), attention_mask=None, position_ids=None, past_key_values=join,
                                position_embeddings=(cos, sin),
                            )
                            m["h"] = to_host(out)   # no DeepStack on a member's own rows (no vision row there)
                            del out, join, cos, sin
                        del keep
                        if grp["ds"] is not None and li < len(deepstack_by_key[grp["ds"]]):
                            out_ref = lm._deepstack_process(
                                out_ref, to_dev(grp["vpm"]), to_dev(deepstack_by_key[grp["ds"]][li]),
                            )
                        grp["h"] = to_host(out_ref)
                        del out_ref
                    drop_to_meta(layer)
                    if li + 1 == deepstack_layers:
                        deepstack_by_key.clear()   # DeepStack is added after the first layers only
                        for st in states:
                            if st is not None:
                                st["ds"] = st["vpm"] = None
                        for grp in shared:
                            grp["ds"] = grp["vpm"] = None
                if stats["layers"] != len(lm.layers):
                    raise RuntimeError(f"H3 TE: {stats['layers']} of {len(lm.layers)} decoder layers ran")

                # 7. Per item: the last hidden state (no final norm), checked, on the CPU.
                hidden_of: dict[int, torch.Tensor] = {}
                for k, st in enumerate(states):
                    if st is not None:
                        hidden_of[k] = st["h"][0]
                for grp in shared:
                    for k, m in grp["members"]:
                        hidden_of[k] = torch.cat([grp["h"][0], m["h"][0]], dim=0)
                outs = []
                for k, e in enumerate(entries):
                    hidden = hidden_of.pop(k)
                    if hidden.dtype != torch.bfloat16:
                        hidden = hidden.to(torch.bfloat16)
                    if int(hidden.shape[0]) != e["n"]:
                        raise RuntimeError(f"H3 TE: {hidden.shape[0]} hidden rows for {e['n']} tokens")
                    outs.append((hidden, torch.tensor(e["tags"], dtype=torch.long)))
        finally:
            # Whatever was reached: no weight and no state of the call outlives it.
            drop_to_meta(lm.embed_tokens)
            for layer in lm.layers:
                drop_to_meta(layer)
            if vision:
                drop_to_meta(model.visual)
                # The rotary buffer is computed, not read: back on the CPU (as encode).
                model.visual.rotary_pos_emb.inv_freq = vision_inv_freq(model.visual.rotary_pos_emb.dim)
            states = shared = None
            deepstack_by_key.clear()
            vision_out.clear()
            pixels.clear()
        report = {
            "items": len(entries),
            "read_gib": round(stats["bytes"] / 2**30, 2),
            "read_seconds": round(stats["read_seconds"], 2),
            "layers_read": stats["layers"],
            "encode_seconds": round(time.perf_counter() - started, 2),
            "shared": [{"reference_rows": int(entries[m[0]]["p"]), "items": list(m)} for m in groups],
            "not_shared": [{"item": k, "reason": r} for k, r in reasons.items()],
        }
        logger.info("H3 TE encode_many (direct read): %s", report)
        return outs, [e["counts"] for e in entries], report

    # -- release ---------------------------------------------------------------

    def _close_reader(self) -> None:
        reader, self._reader = self._reader, None
        if reader is not None:
            reader.close()

    def release(self) -> None:
        """Drop the skeleton, the tokenizer and the unbuffered reader's staging buffer
        (``encode`` already dropped every weight)."""
        self._close_reader()
        self._model = None
        self._tokenizer = None
        self._header = None
        self._embed_plan, self._layer_plans, self._visual_plan = {}, [], {}
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
