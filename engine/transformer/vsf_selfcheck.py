"""Standalone numerical self-check for engine.transformer.vsf_service.

Run with the ENGINE venv (needs the real ltx_core wheel, not just torch):

    .venv-engine\\Scripts\\python.exe -m engine.transformer.vsf_selfcheck

Same conventions as nag_selfcheck (whose helpers this module reuses): NOT a
pytest module, every check either PASSes or FAILs loudly, exit code 0 only
when all of them pass.

What is checked, and why:
  1. The concatenated single-attention core matches a hand-written
     ``softmax(q[K+;K-]^T/sqrt(d)) · [V+; -alpha·V-]`` reference (an explicit
     softmax/matmul pair, NOT a second call into the wheel's attention), plus
     the per-head gate and to_out tail — on fp32 and bf16, at batch 1 and at
     batch 2 where the batch-1 negative context has to be expanded.
  2. The encode-time slice: ``encode_negative(..., slice_to_real_tokens=True)``
     keeps exactly the tokenizer's real-token count and drops the connector's
     learned register tail, while the default (False) leaves NAG's historical
     encoding untouched. The forward half of the same check shows that a
     sliced negative context genuinely changes the attention output — i.e.
     that the registers would otherwise be sign-flipped into the result.
  3. All three AdaLN modes match independently-written reference formulas,
     driven through a REAL ``BasicAVTransformerBlock.forward``
     (cross_attention_adaln=True, apply_gated_attention=True — the production
     GGUF configuration) so the ``apply_cross_attention_adaln`` monkeypatch is
     proven to actually run: "modulated"/"v_scale" cannot produce a result at
     all unless the window's wrapper stashed the coefficients.
  4. OFF is structurally inert: install() with nothing requested patches zero
     modules and leaves no instance-level ``forward`` at all, and a window
     opened for None / NAG params / VSF-with-adaln_mode="raw" leaves the
     wheel's ``apply_cross_attention_adaln`` global identical (``is``) during
     AND after the window.
  5. Fail-loud: unexpected call shapes, unknown adaln_mode (rejected once, at
     VsfParams construction), a missing stash, a shift_kv whose sequence axis
     is not 1, install for a non-VSF params object, and install-before-encode.
  6. The worker's method-resolution helpers (``engine.worker._resolve_nag``/
     ``_neg_label``): a ``nag`` block with no ``method`` key resolves to
     NagParams (forward-compat default), an explicit ``method="vsf"`` block
     resolves to VsfParams with its own knobs propagated, an unknown method
     raises RuntimeError, and ``_neg_label`` maps None/NagParams/VsfParams to
     "off"/"nag"/"vsf".
"""

from __future__ import annotations

import math
import sys
import traceback
from typing import Callable

import torch

from engine.transformer.nag_selfcheck import (
    _FakeTransformer,
    _build_test_block,
    _manual_gate_and_out,
)
from engine.transformer.nag_service import NagParams, NagState, encode_negative
from engine.transformer.vsf_service import (
    VsfParams,
    VsfService,
    adaln_stash_window,
)

_RESULTS: list[tuple[str, bool, str]] = []


def _record(name: str, passed: bool, detail: str = "") -> None:
    status = "PASS" if passed else "FAIL"
    suffix = f" - {detail}" if detail else ""
    print(f"[{status}] {name}{suffix}")
    _RESULTS.append((name, passed, detail))


def _run_check(name: str, fn: Callable[[], None]) -> None:
    try:
        fn()
        _record(name, True)
    except Exception as exc:  # noqa: BLE001 — a broken check must FAIL loudly, never skip
        _record(name, False, f"{type(exc).__name__}: {exc}")
        traceback.print_exc()


def _expect_runtime_error(label: str, fn: Callable[[], object]) -> None:
    try:
        fn()
    except RuntimeError:
        return
    raise AssertionError(f"{label}: expected RuntimeError, none raised")


# --------------------------------------------------------------------------- #
# Shared references (hand-written; never call back into vsf_service)           #
# --------------------------------------------------------------------------- #


def _reference_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int
) -> torch.Tensor:
    """Explicit softmax(QK^T/sqrt(d))V in fp32 — deliberately NOT the wheel's
    attention_function, so check 1 compares against the formula rather than
    against another call into the same code path."""
    b, _, inner = q.shape
    dim_head = inner // heads
    q_h = q.reshape(b, -1, heads, dim_head).transpose(1, 2).float()
    k_h = k.reshape(b, -1, heads, dim_head).transpose(1, 2).float()
    v_h = v.reshape(b, -1, heads, dim_head).transpose(1, 2).float()
    weights = torch.softmax(
        torch.matmul(q_h, k_h.transpose(-1, -2)) / math.sqrt(dim_head), dim=-1
    )
    out = torch.matmul(weights, v_h)
    return out.transpose(1, 2).reshape(b, -1, heads * dim_head)


def _reference_vsf(
    attn: torch.nn.Module,
    x: torch.Tensor,
    context: torch.Tensor,
    neg_k_input: torch.Tensor,
    neg_v_input: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Full hand-written VSF forward built from attn's own submodules:
    concatenate, sign-flip+scale the negative values, one attention, gate,
    to_out. ``neg_k_input``/``neg_v_input`` differ only under the AdaLN modes.
    """
    q = attn.q_norm(attn.to_q(x))
    k = torch.cat(
        [attn.k_norm(attn.to_k(context)), attn.k_norm(attn.to_k(neg_k_input))], dim=1
    )
    v = torch.cat([attn.to_v(context), attn.to_v(neg_v_input) * (-scale)], dim=1)
    out = _reference_attention(q, k, v, attn.heads).to(q.dtype)
    return _manual_gate_and_out(attn, x, out)


def _install_vsf(block: torch.nn.Module, params: VsfParams, neg_ctx: torch.Tensor) -> NagState:
    state = NagState()
    state.set_params(params)
    state.set_contexts(neg_ctx, neg_ctx.clone())
    service = VsfService(state_provider=lambda: state)
    n = service.install(_FakeTransformer([block]))
    if n != 2:
        raise AssertionError(f"VsfService.install patched {n} modules, expected 2")
    return state


# --------------------------------------------------------------------------- #
# Check 1: concatenated single-attention core vs the written-out formula       #
# --------------------------------------------------------------------------- #


def check_concat_attention_matches_formula() -> None:
    for dtype, atol in ((torch.float32, 1e-5), (torch.bfloat16, 8e-2)):
        for batch in (1, 2):
            block = _build_test_block()
            if dtype is not torch.float32:
                block = block.to(dtype)
            attn2 = block.attn2

            torch.manual_seed(1234 + batch)
            tokens, neg_tokens = 6, 3
            x = torch.randn(batch, tokens, 16, dtype=dtype)
            pos_ctx = torch.randn(batch, 9, 16, dtype=dtype)
            # Encoded once per job at batch 1 — the patched forward expands it
            # when the positive side is batched (batch=2 below exercises that).
            neg_ctx = torch.randn(1, neg_tokens, 16, dtype=dtype)

            params = VsfParams(negative_prompt="blurry", scale=1.7, adaln_mode="raw")
            _install_vsf(block, params, neg_ctx)

            got = attn2.forward(x, context=pos_ctx)

            neg_expanded = neg_ctx.expand(batch, -1, -1)
            want = _reference_vsf(
                attn2, x, pos_ctx, neg_expanded, neg_expanded, params.scale
            )
            if not torch.allclose(got.float(), want.float(), rtol=1e-3, atol=atol):
                max_diff = (got.float() - want.float()).abs().max().item()
                raise AssertionError(
                    f"dtype={dtype} batch={batch}: patched output != "
                    f"hand-written VSF formula, max_abs_diff={max_diff} "
                    f"(atol={atol})"
                )

            # Sanity: the sign flip must actually matter. Same call with the
            # negative values NOT negated has to differ, otherwise the check
            # above would pass on a no-op implementation.
            neutral = _reference_vsf(
                attn2, x, pos_ctx, neg_expanded, neg_expanded, -params.scale
            )
            if torch.allclose(got.float(), neutral.float(), rtol=1e-3, atol=atol):
                raise AssertionError(
                    f"dtype={dtype} batch={batch}: sign-flipped and "
                    "non-flipped references are indistinguishable — the check "
                    "is not sensitive to the feature it tests"
                )


# --------------------------------------------------------------------------- #
# Check 2: encode-time slice to real tokens                                    #
# --------------------------------------------------------------------------- #


class _FakeTokenizer:
    """tokenize_with_weights stub: ``n_real`` weight-1 tokens, then zeros —
    the same (token_id, attention_weight) pair shape the real Gemma tokenizer
    returns (ltx_core/text_encoders/gemma/tokenizer.py:28)."""

    def __init__(self, n_real: int, total: int) -> None:
        self._n_real = n_real
        self._total = total

    def tokenize_with_weights(self, text: str) -> dict[str, list[tuple[int, int]]]:
        return {
            "gemma": [
                (100 + i, 1 if i < self._n_real else 0) for i in range(self._total)
            ]
        }


class _FakeTextEncoder:
    def __init__(self, tokenizer: object | None) -> None:
        self.tokenizer = tokenizer


def _with_fake_encode_text(video_ctx: torch.Tensor, audio_ctx: torch.Tensor):
    """Swap ltx_core's encode_text for one returning fixed tensors. Returns a
    (module, original) pair so the caller can restore it."""
    from ltx_core.text_encoders.gemma.encoders import base_encoder as _be

    original = _be.encode_text
    _be.encode_text = lambda _te, _prompts: [(video_ctx, audio_ctx)]
    return _be, original


def check_encode_time_slice() -> None:
    total, n_real, dim = 12, 4, 16
    torch.manual_seed(555)
    # Real prompt tokens are packed at the FRONT; the tail is the connector's
    # learned register embeddings, given an unmistakable sentinel value here.
    video_ctx = torch.randn(1, total, dim)
    video_ctx[:, n_real:, :] = 999.0
    audio_ctx = video_ctx.clone() * 0.5

    module, original = _with_fake_encode_text(video_ctx, audio_ctx)
    try:
        encoder = _FakeTextEncoder(_FakeTokenizer(n_real, total))

        # Default (NAG's historical behaviour): nothing is dropped.
        v_full, a_full = encode_negative(encoder, "blurry, low quality")
        if v_full.shape[1] != total or a_full.shape[1] != total:
            raise AssertionError(
                "slice_to_real_tokens=False changed the encoding length "
                f"({v_full.shape[1]}/{a_full.shape[1]} != {total})"
            )

        # VSF: exactly the real tokens, no registers.
        v_cut, a_cut = encode_negative(
            encoder, "blurry, low quality", slice_to_real_tokens=True
        )
        if v_cut.shape[1] != n_real or a_cut.shape[1] != n_real:
            raise AssertionError(
                f"expected {n_real} tokens after the slice, got "
                f"{v_cut.shape[1]}/{a_cut.shape[1]}"
            )
        if not torch.equal(v_cut, video_ctx[:, :n_real, :]):
            raise AssertionError("sliced video context is not the leading N tokens")
        if not torch.equal(a_cut, audio_ctx[:, :n_real, :]):
            raise AssertionError("sliced audio context is not the leading N tokens")
        if (v_cut == 999.0).any() or (a_cut == 999.0).any():
            raise AssertionError("register sentinel survived the slice")
    finally:
        module.encode_text = original

    # Forward half: the slice must change the result — i.e. the registers
    # really would be concatenated (and sign-flipped) without it.
    block = _build_test_block()
    attn2 = block.attn2
    torch.manual_seed(556)
    x = torch.randn(1, 5, dim)
    pos_ctx = torch.randn(1, 9, dim)
    params = VsfParams(negative_prompt="blurry", scale=1.5, adaln_mode="raw")
    _install_vsf(block, params, v_cut)

    got = attn2.forward(x, context=pos_ctx)
    want_sliced = _reference_vsf(attn2, x, pos_ctx, v_cut, v_cut, params.scale)
    if not torch.allclose(got, want_sliced, rtol=1e-4, atol=1e-5):
        raise AssertionError("patched forward does not match the sliced reference")

    want_unsliced = _reference_vsf(
        attn2, x, pos_ctx, video_ctx, video_ctx, params.scale
    )
    if torch.allclose(got, want_unsliced, rtol=1e-3, atol=1e-3):
        raise AssertionError(
            "sliced and unsliced negative contexts give the same output — the "
            "check cannot detect register contamination"
        )


# --------------------------------------------------------------------------- #
# Check 3: the three AdaLN modes, driven through a real block.forward          #
# --------------------------------------------------------------------------- #


def _build_adaln_block():
    """_build_test_block, but with every parameter initialised.

    BasicAVTransformerBlock allocates its scale_shift_table / prompt_scale_
    shift_table with torch.empty (uninitialised memory, potentially NaN). NAG's
    selfcheck calls attn2.forward directly and never touches them; this check
    drives the whole block.forward, so they must hold real numbers.
    """
    block = _build_test_block()
    torch.manual_seed(2024)
    with torch.no_grad():
        for param in block.parameters():
            param.normal_(0.0, 0.1)
    return block


def _video_args(x: torch.Tensor, context: torch.Tensor, dim: int):
    """Minimal video-only TransformerArgs for BasicAVTransformerBlock.forward.

    positional_embeddings=None keeps attn1 on its no-RoPE path and
    context_mask=None matches what the production text cross-attention always
    passes (the VSF forward fails loud on anything else). timesteps carries
    9 AdaLN parameter groups (6 base + 3 cross-attention), prompt_timestep the
    2 (shift_kv, scale_kv) groups.
    """
    from ltx_core.model.transformer.transformer_args import TransformerArgs

    batch = x.shape[0]
    torch.manual_seed(4096)
    return TransformerArgs(
        x=x,
        context=context,
        context_mask=None,
        timesteps=torch.randn(batch, 1, 9 * dim) * 0.1,
        embedded_timestep=torch.zeros(batch, 1, dim),
        positional_embeddings=None,
        cross_positional_embeddings=None,
        cross_scale_shift_timestep=None,
        cross_gate_timestep=None,
        enabled=True,
        prompt_timestep=torch.randn(batch, 1, 2 * dim) * 0.1,
    )


def check_adaln_modes_through_block_forward() -> None:
    from ltx_core.model.transformer import transformer as _tmod

    dim, batch, tokens, neg_tokens = 16, 2, 5, 3
    outputs: dict[str, torch.Tensor] = {}

    for mode in ("raw", "modulated", "v_scale"):
        block = _build_adaln_block()
        attn2 = block.attn2
        torch.manual_seed(77)
        x = torch.randn(batch, tokens, dim)
        pos_ctx = torch.randn(batch, 9, dim)
        neg_ctx = torch.randn(1, neg_tokens, dim)
        args = _video_args(x, pos_ctx, dim)

        params = VsfParams(negative_prompt="blurry", scale=1.5, adaln_mode=mode)
        _install_vsf(block, params, neg_ctx)

        # Record what the block actually hands the patched forward: the AdaLN
        # path builds both the query input and the MODULATED positive context
        # internally, so the reference has to be computed from the real values,
        # not from reconstructed ones.
        seen: dict[str, torch.Tensor] = {}
        vsf_forward = attn2.forward

        def recording_forward(x_in, context=None, **kwargs):
            seen["x"] = x_in
            seen["context"] = context
            out = vsf_forward(x_in, context=context, **kwargs)
            seen["out"] = out
            return out

        attn2.forward = recording_forward  # type: ignore[method-assign]

        original_global = _tmod.apply_cross_attention_adaln
        with adaln_stash_window(_FakeTransformer([block]), params):
            patched_global = _tmod.apply_cross_attention_adaln
            if mode == "raw":
                if patched_global is not original_global:
                    raise AssertionError(
                        'adaln_mode="raw" opened a real window (the wheel '
                        "global was replaced) — it must be a no-op"
                    )
            elif patched_global is original_global:
                raise AssertionError(
                    f"adaln_mode={mode!r}: the window did not replace "
                    "apply_cross_attention_adaln"
                )
            with torch.no_grad():
                block(args, None)
        if _tmod.apply_cross_attention_adaln is not original_global:
            raise AssertionError(f"adaln_mode={mode!r}: the window did not restore")

        if "out" not in seen:
            raise AssertionError(
                f"adaln_mode={mode!r}: the patched attn2 forward was never "
                "called by block.forward"
            )

        # Independent reproduction of transformer.py:394-397's coefficients.
        shift_kv, scale_kv = (
            block.prompt_scale_shift_table[None, None].to(
                device=x.device, dtype=x.dtype
            )
            + args.prompt_timestep.reshape(batch, args.prompt_timestep.shape[1], 2, -1)
        ).unbind(dim=2)

        neg_expanded = neg_ctx.expand(batch, -1, -1)
        if mode == "raw":
            neg_k = neg_v = neg_expanded
        elif mode == "modulated":
            neg_k = neg_v = neg_expanded * (1 + scale_kv) + shift_kv
        else:  # v_scale
            neg_k = neg_expanded
            neg_v = neg_expanded * (1 + scale_kv)

        want = _reference_vsf(
            attn2, seen["x"], seen["context"], neg_k, neg_v, params.scale
        )
        if not torch.allclose(seen["out"], want, rtol=1e-4, atol=1e-5):
            max_diff = (seen["out"] - want).abs().max().item()
            raise AssertionError(
                f"adaln_mode={mode!r}: attn2 output != reference, "
                f"max_abs_diff={max_diff}"
            )
        outputs[mode] = seen["out"].clone()

        # A stale stash would silently feed the next step the wrong
        # coefficients; the forward pops it and the window sweeps the rest.
        if hasattr(attn2, "_vsf_kv_mod"):
            raise AssertionError(
                f"adaln_mode={mode!r}: a stash survived the window"
            )

    for a, b in (("raw", "modulated"), ("raw", "v_scale"), ("modulated", "v_scale")):
        if torch.allclose(outputs[a], outputs[b], rtol=1e-4, atol=1e-5):
            raise AssertionError(
                f"adaln modes {a!r} and {b!r} produced identical output — the "
                "modes are not actually distinct"
            )


# --------------------------------------------------------------------------- #
# Check 4: OFF is structurally inert                                           #
# --------------------------------------------------------------------------- #


def check_off_is_inert() -> None:
    from ltx_core.model.transformer import transformer as _tmod

    block = _build_test_block()
    state = NagState()  # requested is False: set_params was never called
    service = VsfService(state_provider=lambda: state)

    n = service.install(_FakeTransformer([block]))
    if n != 0:
        raise AssertionError(f"install() with VSF not requested returned {n}, expected 0")
    # Strongest available form of "forward was not patched": the patch works by
    # writing an INSTANCE attribute that shadows the class method, so its
    # absence from __dict__ proves the class method is still what runs.
    for name in ("attn2", "audio_attn2"):
        module = getattr(block, name)
        if "forward" in module.__dict__:
            raise AssertionError(f"install() with VSF off shadowed {name}.forward")

    original_global = _tmod.apply_cross_attention_adaln
    no_op_params = (
        None,
        NagParams(negative_prompt="x", scale=11.0, tau=2.5, alpha=0.25),
        VsfParams(negative_prompt="x", scale=1.5, adaln_mode="raw"),
    )
    for params in no_op_params:
        with adaln_stash_window(_FakeTransformer([block]), params):
            if _tmod.apply_cross_attention_adaln is not original_global:
                raise AssertionError(
                    f"adaln_stash_window({type(params).__name__}) replaced the "
                    "wheel global — it must be a no-op window"
                )
        if _tmod.apply_cross_attention_adaln is not original_global:
            raise AssertionError(
                f"adaln_stash_window({type(params).__name__}) left the wheel "
                "global changed after exit"
            )


# --------------------------------------------------------------------------- #
# Check 5: fail-loud                                                           #
# --------------------------------------------------------------------------- #


def check_fail_loud() -> None:
    dim = 16
    block = _build_test_block()
    attn2 = block.attn2
    torch.manual_seed(31337)
    x = torch.randn(1, 4, dim)
    pos_ctx = torch.randn(1, 7, dim)
    neg_ctx = torch.randn(1, 3, dim)

    # --- call shapes this patch was never designed for ---
    _install_vsf(block, VsfParams("blurry", 1.5, "raw"), neg_ctx)
    _expect_runtime_error(
        "pe set", lambda: attn2.forward(x, context=pos_ctx, pe=torch.zeros(1))
    )
    _expect_runtime_error("context=None", lambda: attn2.forward(x, context=None))
    _expect_runtime_error(
        "mask set", lambda: attn2.forward(x, context=pos_ctx, mask=torch.zeros(1))
    )

    # --- unknown adaln_mode: rejected once, at VsfParams construction ---
    # (install/forward/window no longer validate it separately — see
    # VsfParams.__post_init__ — so there is nothing left to construct a
    # "bogus" instance to feed them; the RuntimeError fires before one exists.)
    _expect_runtime_error(
        "VsfParams(unknown adaln_mode)",
        lambda: VsfParams(negative_prompt="blurry", scale=1.5, adaln_mode="modulate"),
    )

    # --- adaln_mode != "raw" with no window open -> no stash -> fail ---
    mod_block = _build_test_block()
    mod_attn2 = mod_block.attn2
    _install_vsf(mod_block, VsfParams("blurry", 1.5, "modulated"), neg_ctx)
    _expect_runtime_error(
        "modulated without an open window",
        lambda: mod_attn2.forward(x, context=pos_ctx),
    )

    # --- shift_kv whose sequence axis is not 1 (per-token prompt timesteps) ---
    mod_attn2._vsf_kv_mod = (torch.zeros(1, 2, dim), torch.zeros(1, 2, dim))
    _expect_runtime_error(
        "shift_kv.shape[1] != 1", lambda: mod_attn2.forward(x, context=pos_ctx)
    )
    if hasattr(mod_attn2, "_vsf_kv_mod"):
        delattr(mod_attn2, "_vsf_kv_mod")

    # --- wrong params type / requested-but-not-encoded ---
    nag_state = NagState()
    nag_state.set_params(NagParams("blurry", 11.0, 2.5, 0.25))
    nag_state.set_contexts(neg_ctx, neg_ctx.clone())
    _expect_runtime_error(
        "install(NagParams)",
        lambda: VsfService(lambda: nag_state).install(
            _FakeTransformer([_build_test_block()])
        ),
    )

    not_ready = NagState()
    not_ready.set_params(VsfParams("blurry", 1.5, "raw"))  # no set_contexts
    _expect_runtime_error(
        "install(requested but not encoded)",
        lambda: VsfService(lambda: not_ready).install(
            _FakeTransformer([_build_test_block()])
        ),
    )

    # --- encode-time slice guards ---
    module, original = _with_fake_encode_text(
        torch.randn(1, 8, dim), torch.randn(1, 8, dim)
    )
    try:
        _expect_runtime_error(
            "encode_negative(slice) without a tokenizer",
            lambda: encode_negative(
                _FakeTextEncoder(None), "blurry", slice_to_real_tokens=True
            ),
        )
        _expect_runtime_error(
            "encode_negative(slice) with N > seq_len",
            lambda: encode_negative(
                _FakeTextEncoder(_FakeTokenizer(9, 9)),
                "blurry",
                slice_to_real_tokens=True,
            ),
        )
    finally:
        module.encode_text = original


# --------------------------------------------------------------------------- #
# Check 6: engine.worker's method-resolution helpers                          #
# --------------------------------------------------------------------------- #


def check_worker_resolve() -> None:
    """``engine.worker._resolve_nag``/``_neg_label`` pick between NagParams and
    VsfParams from a job's raw ``nag`` dict — deliberately checked here rather
    than only implicitly through the two services above, since a mistake in
    this resolution (e.g. dropping a VSF knob, or defaulting method wrong)
    would silently run the wrong algorithm without either service's own
    checks ever seeing it.

    Imported lazily (inside this function, not at module scope) because
    engine.worker has real import-time side effects (chdir to the project
    root, torch/ltx_core/ltx_pipelines imports, a couple of stdout log
    lines) that the rest of this selfcheck module does not need and should
    not pay for just to run checks 1-5.
    """
    from engine.worker import _neg_label, _resolve_nag

    # --- nag block absent -> None ---
    if _resolve_nag({}) is not None:
        raise AssertionError("_resolve_nag({}) did not return None")
    if _resolve_nag({"nag": None}) is not None:
        raise AssertionError('_resolve_nag({"nag": None}) did not return None')

    # --- method absent -> defaults to NAG, all 4 values propagated ---
    nag_params = _resolve_nag(
        {"nag": {"negative_prompt": "x", "scale": 9.0, "tau": 3.0, "alpha": 0.5}}
    )
    if not isinstance(nag_params, NagParams):
        raise AssertionError(f"method-absent nag block resolved to {type(nag_params).__name__}, expected NagParams")
    if (nag_params.negative_prompt, nag_params.scale, nag_params.tau, nag_params.alpha) != (
        "x",
        9.0,
        3.0,
        0.5,
    ):
        raise AssertionError(f"NagParams fields did not propagate: {nag_params!r}")

    # --- method="vsf" -> VsfParams, its own knobs propagated (non-default
    # values, so a copy-paste of the API defaults would not accidentally pass) ---
    vsf_params = _resolve_nag(
        {
            "nag": {
                "negative_prompt": "x",
                "method": "vsf",
                "vsf_scale": 7.5,
                "vsf_adaln": "v_scale",
            }
        }
    )
    if not isinstance(vsf_params, VsfParams):
        raise AssertionError(f'method="vsf" nag block resolved to {type(vsf_params).__name__}, expected VsfParams')
    if (vsf_params.scale, vsf_params.adaln_mode) != (7.5, "v_scale"):
        raise AssertionError(f"VsfParams fields did not propagate: {vsf_params!r}")

    # --- unknown method -> RuntimeError ---
    _expect_runtime_error(
        "_resolve_nag(unknown method)",
        lambda: _resolve_nag({"nag": {"negative_prompt": "x", "method": "bogus"}}),
    )

    # --- _neg_label: off / nag / vsf ---
    if _neg_label(None) != "off":
        raise AssertionError('_neg_label(None) != "off"')
    if _neg_label(NagParams("x", 11.0, 2.5, 0.25)) != "nag":
        raise AssertionError('_neg_label(NagParams(...)) != "nag"')
    if _neg_label(VsfParams("x", 1.5, "raw")) != "vsf":
        raise AssertionError('_neg_label(VsfParams(...)) != "vsf"')


def main() -> int:
    # Windows consoles are frequently cp932/cp1252, not UTF-8 — a stray
    # non-ASCII character in an exception message must never crash the
    # reporter itself (that would defeat "fail loudly, never skip").
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    torch.manual_seed(0)
    checks: list[tuple[str, Callable[[], None]]] = [
        ("concatenated attention matches the written-out VSF formula (fp32+bf16, B=1/B=2)", check_concat_attention_matches_formula),
        ("encode-time slice keeps real tokens only and changes the forward", check_encode_time_slice),
        ("AdaLN raw/modulated/v_scale match references via real block.forward", check_adaln_modes_through_block_forward),
        ("VSF OFF: zero patches + wheel global identity preserved", check_off_is_inert),
        ("fail-loud: shapes, unknown mode, missing stash, wrong params", check_fail_loud),
        ("engine.worker._resolve_nag/_neg_label: method resolution and labels", check_worker_resolve),
    ]

    for name, fn in checks:
        _run_check(name, fn)

    n_pass = sum(1 for _, ok, _ in _RESULTS if ok)
    n_total = len(_RESULTS)
    print(f"\n{n_pass}/{n_total} checks passed")
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    raise SystemExit(main())
