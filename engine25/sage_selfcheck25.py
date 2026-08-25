"""Standalone self-check for SageAttention on the LTX 2.5 engine.

Run with the 2.5 ENGINE venv (needs the official ltx_core 1.2.0 wheel, a CUDA
GPU, and the sageattention wheel -- none of which the app venv has):

    .venv-engine-ltx25\\Scripts\\python.exe -m engine25.sage_selfcheck25

Same conventions as ``engine.transformer.sage_selfcheck``: NOT a pytest module,
every check either PASSes or FAILs loudly (nothing is skipped), exit code 0 only
when all of them pass.

WHAT THIS FILE IS, AND WHAT IT DELIBERATELY IS NOT
    The kernel wrapper is SHARED with LTX 2.3 -- one
    ``sage_attention_service.py``, imported by both engines, because 2.5's
    attention contract is 2.3's to the letter. So the three checks that judge
    the WRAPPER (numerical parity, the fallback matrix, the 288-module install
    count) are 2.3's, imported and re-run here rather than copied: a second copy
    could only drift, and a drifted copy of a passing test is worse than no test.
    What they are re-run AGAINST is 2.5's own block, via the module-attribute
    swap in :func:`_ltx25_helpers` -- ``BasicAVTransformerBlock`` lost its
    ``idx`` argument and moved its attention selection into a
    ``TransformerOpsConfig``, so the block builder is the one piece that cannot
    be shared.

    The three checks BELOW are the ones with no 2.3 counterpart, because they
    are about the thing the two engines do differently: 2.3 builds a brand new
    transformer per job, 2.5 reuses ONE model shell forever.

      1. THE MASKED SLOT IS NEVER TOUCHED. 2.5 routes a masked call to a
         SEPARATE attribute (``masked_attention_function``), so installing sage
         must leave all 288 of those identical -- ``is``-identical, not merely
         equal. This is what makes "an IC-LoRA attention-strength job on 2.5
         reaches SDPA structurally, and logs no masked-fallback line" a fact
         about the code rather than an observation about one run.

      2. INSTALL IS IDEMPOTENT. The shell registry hands back the same
         ``LTXModel`` on every build and ``dispose()`` does not touch plain
         Python attributes, so a wrapper survives into the next job. Three
         properties: the surviving wrapper is DETECTED (proved with the
         selfcheck-only fail-loud flag, because "it logged an ERROR" is a weaker
         assertion than "it raised"); install -> uninstall -> install wraps
         exactly 288 each time and restores the shared original in between; and
         a second install with NO uninstall self-repairs to exactly one layer
         rather than nesting.

      3. WRAPPERS SURVIVE ``dispose()``, AND CAN BE PEELED OFF A DISPOSED
         SHELL. This is the mechanism behind (2) and the reason ``uninstall``
         may not touch a tensor: after ``dispose()`` every parameter is on
         ``device="meta"``, where any operator dispatch raises. The check
         disposes a wrapped transformer and then peels it.

A TRAP WORTH KNOWING, since it is the one place 2.5 exercises the mask branch:
    ``engine25.gguf_transformer``'s selftest dummy passes
    ``context_mask=torch.ones(...)`` (gguf_transformer.py:1129), so on THAT path
    ``attn2`` takes ``Attention.forward``'s masked route -- through
    ``masked_attention_function``, which sage does not own, so it is not a sage
    fallback at all. A real 2.5 job never gets there (reference25's IC-LoRA
    attention-strength wrapper is the only mask producer and it goes to the same
    unowned slot). Do not read a masked-fallback line from that selftest as
    evidence about production, and do not read its ABSENCE as a failure.

Why no end-to-end generation here: that is the real-device gate's job, and it
needs the 22B model. This file only has to prove the swap itself is correct and
safe on this engine's reused shell.
"""

from __future__ import annotations

import contextlib
import sys
from typing import Callable, Iterator

import torch

from engine.transformer import sage_attention_service as sage_mod
from engine.transformer import sage_selfcheck as base
from engine.transformer.sage_attention_service import SageAttentionService, SageState

#: 48 blocks x 6 attention modules, the production LTX 2.5 transformer's count
#: and the same number 2.3 has -- which is not a coincidence: it is the same
#: dual-stream block layout.
_EXPECTED_MODULES = 288


def _build_block(video_d_head: int, audio_d_head: int, heads: int = 2, dim: int = 32):
    """2.5's ``BasicAVTransformerBlock``, in the production configuration.

    Two differences from 2.3's builder, and only two:

    * no ``idx`` -- 1.2.0 dropped the per-block index from the constructor;
    * the attention backend is chosen through a ``TransformerOpsConfig`` rather
      than an ``attention_function=`` argument, and BOTH slots are pinned to
      PYTORCH. The masked one matters as much as the unmasked one here: check 1
      asserts that installing sage leaves it untouched, and an AUTOMATIC slot
      would make that assertion true for an uninteresting reason on a machine
      where AUTOMATIC happened to resolve to the same shared instance.

    ``ops.attention_ops`` is ONE frozen ``AttentionOps`` shared by all six
    ``Attention`` modules in the block, so all six start out holding the SAME
    callable object. That is not incidental -- it is what check 2 asserts
    ``uninstall`` restores.

    The block can stay on CPU even when the tensors handed to it live on the
    GPU: an attention callable holds no tensors.
    """
    from ltx_core.model.transformer.attention import AttentionFunction, MaskedAttentionFunction
    from ltx_core.model.transformer.transformer import (
        BasicAVTransformerBlock,
        TransformerConfig,
        TransformerOpsConfig,
    )

    torch.manual_seed(42)
    block = BasicAVTransformerBlock(
        video=TransformerConfig(
            dim=dim,
            heads=heads,
            d_head=video_d_head,
            context_dim=dim,
            apply_gated_attention=True,
            cross_attention_adaln=True,
        ),
        audio=TransformerConfig(
            dim=dim,
            heads=heads,
            d_head=audio_d_head,
            context_dim=dim,
            apply_gated_attention=True,
            cross_attention_adaln=True,
        ),
        ops=TransformerOpsConfig.from_functions(
            attention=AttentionFunction.PYTORCH,
            masked_attention=MaskedAttentionFunction.PYTORCH,
        ),
    )
    block.eval()
    return block


class _Spy(base._Spy):
    """2.3's counting spy, with 2.5's DEFAULT callable inside it.

    2.3 wraps ``PytorchAttention()``; here the inner callable is whatever
    ``AttentionFunction.AUTOMATIC`` resolves to on this machine, because that is
    what a real 2.5 build puts in the slot -- so "the fallback result is
    bit-identical to sdpa" is checked against the callable production would
    actually have used, not against a stand-in that merely resembles it. (On an
    RTX 40xx box the two are the same class with the same priority list; on a
    machine where AUTOMATIC picked something else this would be the difference
    between a real check and a tautology.)

    ``base._Spy.__init__`` is deliberately NOT called: its whole body is the
    line this class replaces.
    """

    def __init__(self) -> None:  # noqa: D107 -- see the class docstring
        from ltx_core.model.transformer.attention import AttentionFunction

        self._inner = AttentionFunction.AUTOMATIC.to_callable()
        self.calls = 0
        self.last_args: tuple = ()


@contextlib.contextmanager
def _ltx25_helpers() -> Iterator[None]:
    """Point 2.3's imported checks at 2.5's block builder and spy, then restore.

    The three checks re-run below resolve ``_build_block`` and ``_Spy`` from
    THEIR OWN module globals, so rebinding those two names in
    ``engine.transformer.sage_selfcheck`` is what makes an imported check run
    against this engine. Rebinding rather than reimplementing is the point: the
    assertions stay 2.3's, byte for byte, and only the fixture changes.
    """
    saved_block, saved_spy = base._build_block, base._Spy
    base._build_block = _build_block  # type: ignore[assignment]
    base._Spy = _Spy  # type: ignore[assignment]
    try:
        yield
    finally:
        base._build_block = saved_block  # type: ignore[assignment]
        base._Spy = saved_spy  # type: ignore[assignment]


def _transformer_48() -> torch.nn.Module:
    """A 48-block stand-in: 288 attention modules, the production count."""
    return base._FakeTransformer(
        [_build_block(video_d_head=128, audio_d_head=64) for _ in range(48)]
    )


def _sage_on() -> SageState:
    state = SageState()
    state.set_backend("sage")
    return state


def _attention_functions(transformer: torch.nn.Module) -> list:
    return [module.attention_function for _name, module in sage_mod._attention_modules(transformer)]


def _masked_attention_functions(transformer: torch.nn.Module) -> list:
    return [
        module.masked_attention_function
        for _name, module in sage_mod._attention_modules(transformer)
    ]


# --------------------------------------------------------------------------- #
# Check 1: the masked slot is never touched (2.5 only)                         #
# --------------------------------------------------------------------------- #


def check_masked_slot_untouched() -> None:
    transformer = _transformer_48()
    before_masked = _masked_attention_functions(transformer)
    before_unmasked = _attention_functions(transformer)
    if len(before_masked) != _EXPECTED_MODULES:
        raise AssertionError(
            f"the fixture has {len(before_masked)} attention modules, expected {_EXPECTED_MODULES}"
        )

    install_state = _sage_on()
    count = SageAttentionService(lambda: install_state).install(transformer)
    if count != _EXPECTED_MODULES:
        raise AssertionError(f"install() wrapped {count} modules, expected {_EXPECTED_MODULES}")

    after_masked = _masked_attention_functions(transformer)
    for index, (was, now) in enumerate(zip(before_masked, after_masked)):
        if now is not was:
            raise AssertionError(
                f"module {index}: masked_attention_function was REPLACED by install(). "
                "2.5 routes masked calls to that separate slot, and sage must not own it."
            )

    # ...and the check is not vacuous: the slot sage DOES own changed on all 288.
    after_unmasked = _attention_functions(transformer)
    unchanged = [i for i, (was, now) in enumerate(zip(before_unmasked, after_unmasked)) if now is was]
    if unchanged:
        raise AssertionError(
            f"{len(unchanged)} module(s) kept their original attention_function, so "
            "'the masked slot did not change' would be true for the wrong reason"
        )

    # The masked slot is also still CALLABLE with the 5-argument masked protocol
    # -- i.e. what Attention.forward does with a non-None mask still works, and
    # still goes nowhere near sage.
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    block = _build_block(video_d_head=128, audio_d_head=64)
    attn = block.attn1
    q, k, v = base._qkv(1, 64, attn.heads, 128, torch.bfloat16, device)
    mask = torch.zeros(1, attn.heads, 64, 64, dtype=torch.bfloat16, device=device)
    state = _sage_on()
    SageAttentionService(lambda: state).install(base._FakeTransformer([block]))
    out = attn.masked_attention_function(q, k, v, attn.heads, mask)
    if out.shape != q.shape:
        raise AssertionError(f"masked_attention_function returned {out.shape}, expected {q.shape}")
    if state.masked_calls != 0:
        raise AssertionError(
            f"the masked call was counted as a sage fallback (masked_calls="
            f"{state.masked_calls}); on 2.5 it must not reach the wrapper at all"
        )


# --------------------------------------------------------------------------- #
# Check 2: install is idempotent on a reused shell (2.5 only)                  #
# --------------------------------------------------------------------------- #


def check_install_idempotent() -> None:
    state = _sage_on()
    service = SageAttentionService(lambda: state)

    transformer = _transformer_48()
    originals = _attention_functions(transformer)
    # The block builder shares ONE AttentionOps across its six modules, so a
    # block's six originals are one object. Stated as an assertion because the
    # restore check below is only meaningful if the identity is real.
    if len({id(fn) for fn in originals}) != 48:
        raise AssertionError(
            "expected exactly one shared attention callable per block (48 distinct "
            f"objects across {_EXPECTED_MODULES} modules), got "
            f"{len({id(fn) for fn in originals})}"
        )

    # -- (a) the surviving wrapper is DETECTED --------------------------------
    first = service.install(transformer)
    if first != _EXPECTED_MODULES:
        raise AssertionError(f"first install wrapped {first}, expected {_EXPECTED_MODULES}")
    try:
        service.install(transformer, fail_on_surviving_wrapper=True)
    except RuntimeError as exc:
        if "already a sage wrapper" not in str(exc):
            raise AssertionError(f"the fail-loud flag raised the wrong error: {exc!r}") from exc
    else:
        raise AssertionError(
            "install(fail_on_surviving_wrapper=True) did not raise on an already-wrapped "
            "transformer -- the detection that the product path relies on is not firing"
        )

    # -- (b) the PRODUCT path self-repairs instead of raising, and to ONE layer -
    with base._LogCapture() as logs:
        second = service.install(transformer)
    if second != _EXPECTED_MODULES:
        raise AssertionError(f"self-repairing install wrapped {second}, expected {_EXPECTED_MODULES}")
    stripped_logs = [m for m in logs.messages if "surviving wrapper" in m]
    if len(stripped_logs) != 1:
        raise AssertionError(
            f"expected exactly 1 surviving-wrapper ERROR line, got {len(stripped_logs)}: "
            f"{stripped_logs}"
        )
    if f"stripped {_EXPECTED_MODULES} surviving" not in stripped_logs[0]:
        raise AssertionError(f"the ERROR line does not name 288 wrappers: {stripped_logs[0]!r}")
    for index, fn in enumerate(_attention_functions(transformer)):
        if not isinstance(fn, sage_mod._SageAttentionFunction):
            raise AssertionError(f"module {index} is not wrapped after the self-repairing install")
        if isinstance(fn._fallback, sage_mod._SageAttentionFunction):
            raise AssertionError(
                f"module {index}: the wrapper is NESTED -- self-repair stripped nothing"
            )

    # -- (c) install -> uninstall -> install, 288 every time -------------------
    removed = service.uninstall(transformer)
    if removed != _EXPECTED_MODULES:
        raise AssertionError(f"uninstall() peeled {removed}, expected {_EXPECTED_MODULES}")
    for index, (was, now) in enumerate(zip(originals, _attention_functions(transformer))):
        if now is not was:
            raise AssertionError(
                f"module {index}: uninstall() restored a DIFFERENT object than the one "
                "install() found -- the shared AttentionOps callable was not put back"
            )
    third = service.install(transformer)
    if third != _EXPECTED_MODULES:
        raise AssertionError(f"third install wrapped {third}, expected {_EXPECTED_MODULES}")

    # A clean strip-and-reinstall must be SILENT on the ERROR channel: that
    # silence is the judging criterion for the real-device idempotency gate,
    # since a self-repaired install is otherwise indistinguishable from a clean
    # one (same count, same numbers, same speed).
    service.uninstall(transformer)
    with base._LogCapture() as clean_logs:
        service.install(transformer)
    surviving_lines = [m for m in clean_logs.messages if "surviving wrapper" in m]
    if surviving_lines:
        raise AssertionError(
            f"a strip-then-install pair still logged a surviving-wrapper ERROR: {surviving_lines}"
        )

    # And uninstall on an unwrapped transformer is a no-op, not an error: the
    # build path calls it unconditionally, including on the very first build.
    fresh = _transformer_48()
    if service.uninstall(fresh) != 0:
        raise AssertionError("uninstall() on an unwrapped transformer reported peeling something")


# --------------------------------------------------------------------------- #
# Check 3: wrappers survive dispose(), and peel off a disposed shell (2.5 only)#
# --------------------------------------------------------------------------- #


def check_wrappers_survive_dispose() -> None:
    from ltx_core.model.disposable import Disposable

    state = _sage_on()
    service = SageAttentionService(lambda: state)
    transformer = _transformer_48()
    if service.install(transformer) != _EXPECTED_MODULES:
        raise AssertionError("the fixture did not wrap 288 modules")

    # ``Disposable`` is a mixin; the fixture is a plain nn.Module, so the method
    # is called unbound. Its only precondition is ``isinstance(self, nn.Module)``.
    Disposable.dispose(transformer)  # type: ignore[arg-type]

    metas = [p for p in transformer.parameters() if p.device.type == "meta"]
    total = list(transformer.parameters())
    if not total or len(metas) != len(total):
        raise AssertionError(
            f"dispose() left {len(total) - len(metas)} of {len(total)} parameters off meta; "
            "the rest of this check would not be testing anything"
        )

    survivors = [
        fn for fn in _attention_functions(transformer)
        if isinstance(fn, sage_mod._SageAttentionFunction)
    ]
    if len(survivors) != _EXPECTED_MODULES:
        raise AssertionError(
            f"{len(survivors)} of {_EXPECTED_MODULES} wrappers survived dispose(). If this "
            "ever becomes 0, the strip-before-install on the build path is dead code -- and "
            "the reason it exists has changed."
        )

    # THE POINT: peeling has to work HERE, on the meta'd shell, because that is
    # exactly the state the next build finds it in. Anything in uninstall() that
    # dispatched an operator would raise NotImplementedError on this line.
    removed = service.uninstall(transformer)
    if removed != _EXPECTED_MODULES:
        raise AssertionError(
            f"uninstall() peeled {removed} wrappers off a disposed shell, expected "
            f"{_EXPECTED_MODULES}"
        )
    if any(
        isinstance(fn, sage_mod._SageAttentionFunction) for fn in _attention_functions(transformer)
    ):
        raise AssertionError("a wrapper survived uninstall() on the disposed shell")


def main() -> int:
    # Windows consoles are frequently cp932/cp1252 -- a stray non-ASCII character
    # in an exception message must never crash the reporter itself.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    torch.manual_seed(0)
    base._RESULTS.clear()

    shared: list[tuple[str, Callable[[], None]]] = [
        (
            "[shared with 2.3] numerical parity vs the wheel's own attention (d_head 64+128, bf16+fp16)",
            base.check_numerical_parity,
        ),
        (
            "[shared with 2.3] fallback matrix (mask / dtype / device / stale / kernel error+latch / static skip)",
            base.check_fallback_matrix,
        ),
        (
            "[shared with 2.3] install wraps exactly 288 modules on 48 blocks, and 0 when OFF",
            base.check_install_count_288,
        ),
    ]
    ltx25_only: list[tuple[str, Callable[[], None]]] = [
        ("[2.5] install leaves all 288 masked_attention_function slots untouched", check_masked_slot_untouched),
        ("[2.5] install is idempotent on a reused shell (detect / self-repair / restore)", check_install_idempotent),
        ("[2.5] wrappers survive dispose(), and uninstall works on the meta'd shell", check_wrappers_survive_dispose),
    ]

    with _ltx25_helpers():
        for name, fn in shared:
            base._run_check(name, fn)
    # Outside the swap on purpose: the 2.5-only checks call this module's own
    # ``_build_block`` directly, so a leftover rebinding would hide a mistake
    # here rather than a difference between the engines.
    for name, fn in ltx25_only:
        base._run_check(name, fn)

    n_pass = sum(1 for _, ok, _ in base._RESULTS if ok)
    n_total = len(base._RESULTS)
    print(f"\n{n_pass}/{n_total} checks passed")
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    raise SystemExit(main())
