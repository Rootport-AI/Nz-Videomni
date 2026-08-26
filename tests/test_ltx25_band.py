"""The LTX 2.5 frozen band: its TAIL half, its strengths, and its position.

§3-102 C1 (retake / end source, engine side, still INACTIVE) widened
``engine25.chain25._band_conditionings`` from "freeze the head" to "freeze the
head and/or the tail", added ``_freeze_strengths`` and moved the keyframe-marker
decision out to the caller. Nothing in the shipped chain asks for a tail yet, so
the arithmetic evidence that the widening is inert is the pair of digests in
gate G1(a)/(d); what is checkable here, without a GPU, is everything ELSE that
could go wrong later:

1. ``_freeze_strengths`` is the shared resolver read from the other end -- and
   is NOT on the path an ordinary carry seam takes. That second half is the
   point: the ordinary head band is handed ``float(spec.overlap_strength)``
   verbatim, and routing it through a complement-of-a-complement instead would
   move ``0.3`` to ``0.30000000000000004`` and change an existing job's output
   with no test failing at the default ``0.5`` (which round-trips exactly).
2. The head and tail masks are DISJOINT even in the shortest window the app can
   ask for, so two separate items cannot fight over a latent frame.
3. The band items are FIRST in every conditioning list built from them. The band
   arithmetic is elementwise over the whole token axis, so an item that appends
   tokens (a ``frame_idx > 0`` keyframe image) must not run before it. C1
   deliberately did not encode that as a type (no ``VideoBandMask`` subclass);
   construction order carries it, and this file is what makes a future
   reordering an error instead of a wrong freeze.
4. The keyframe-marker clear, now that the caller decides, still resolves to
   exactly what it resolved to before at both of the two existing call sites.

CPU-only and model-free: ``_band_conditionings`` builds mask tensors and
conditioning items and nothing else, and ``_freeze_strengths`` is a pure
function over floats.

Run with ``.venv-engine-ltx25`` and ``--noconftest`` (the app conftest builds a
FastAPI app that venv does not have).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("ltx_core")

import torch  # noqa: E402

from chain_math import freeze_mask_values  # noqa: E402
from engine25 import chain25  # noqa: E402
from engine25.chain25 import (  # noqa: E402
    AudioBandMask,
    ClearKeyframesMask,
    _band_conditionings,
    _freeze_strengths,
)
from engine25.ltxcore_compat import VideoConditionByMask  # noqa: E402

#: The SHORTEST window retake can be asked for (73 pixel frames) is the case
#: where a head band and a tail band come closest to each other: 10 video latent
#: frames, 4 frozen at the head, 3 at the tail. Three free frames in the middle
#: is not much, but it is not zero, and "not zero" is the whole claim.
F_TOTAL = 10
N_HEAD = 4
N_TAIL = 3


def _latents(frames: int = F_TOTAL) -> tuple[torch.Tensor, torch.Tensor]:
    """A (video, audio) latent pair with DISTINCT values everywhere.

    ``arange`` rather than ``zeros``: a mask bug that copies the wrong frames is
    invisible against an all-zero latent.
    """
    video = torch.arange(2 * frames * 2 * 3, dtype=torch.float32).reshape(1, 2, frames, 2, 3)
    audio = torch.arange(4 * frames * 8, dtype=torch.float32).reshape(1, 4, frames, 8)
    return video, audio


def _video_bands(items: list) -> list[VideoConditionByMask]:
    return [it for it in items if isinstance(it, VideoConditionByMask)]


def _audio_bands(items: list) -> list[AudioBandMask]:
    return [it for it in items if isinstance(it, AudioBandMask)]


# ---------------------------------------------------------------------------
# 1. _freeze_strengths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "args",
    [
        (0.5,),
        (0.0,),
        (1.0,),
        (0.3,),
        (1.0 - 0.7,),
        # retake: head and tail split (the two-sided freeze).
        (0.5, 0.0),
        # long A2V: video and audio split.
        (0.5, None, 0.0),
        # end source: the TAIL splits by modality -- video honours the user's
        # strength, the material's audio is always hard-frozen.
        (0.5, 1.0 - 0.7, None, 0.0),
    ],
)
def test_freeze_strengths_is_the_shared_resolver_read_from_the_other_end(args: tuple) -> None:
    """Every arm equals ``1 - chain_math.freeze_mask_values(...)``, elementwise.

    2.3 writes denoise-mask VALUES; 2.5's band items take the COMPLEMENT. The
    two engines therefore agree on the same four numbers read from opposite
    ends, and this asserts the reading -- not a second table of numbers, which
    is exactly what the shared resolver exists to prevent.
    """
    got = _freeze_strengths(*args)
    expected = tuple(1.0 - v for v in freeze_mask_values(*args))
    assert got == expected
    assert len(got) == 4


def test_freeze_strengths_end_source_shape() -> None:
    """The end source's pair: video tail at the user's strength, audio tail hard."""
    v_head, v_tail, a_head, a_tail = _freeze_strengths(
        1.0 - 0.5, tail_mask_value=1.0 - 0.7, audio_tail_mask_value=0.0
    )
    assert (v_head, a_head) == (0.5, 0.5)
    assert v_tail == pytest.approx(0.7)
    assert a_tail == 1.0


def test_a_double_complement_is_not_the_identity() -> None:
    """WHY ``_freeze_strengths`` is off limits for the ordinary carry seam.

    ``1 - (1 - 0.3)`` is ``0.30000000000000004``. At the default
    ``overlap_strength`` of 0.5 the round trip IS exact, so a gate that only
    ever runs the default cannot see the difference -- which is precisely how
    this would have shipped as a silent change to every non-default job.
    """
    assert _freeze_strengths(1.0 - 0.3)[0] != 0.3
    assert _freeze_strengths(1.0 - 0.3)[0] == pytest.approx(0.3)
    assert _freeze_strengths(1.0 - 0.5)[0] == 0.5  # the default hides it


# ---------------------------------------------------------------------------
# 2. the head band is handed its strength verbatim
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strength", [0.3, 0.7, 0.0, 1.0, 0.5])
def test_head_band_strength_reaches_the_item_unchanged(strength: float) -> None:
    """What ``run_chain`` passes as ``strength`` is what the item gets.

    The 0.3 arm is the regression net for the double-complement trap above: it
    fails the moment anyone routes ``float(spec.overlap_strength)`` through
    ``_freeze_strengths`` on its way here.
    """
    video, audio = _latents()
    band_v, band_a = _band_conditionings(
        video_latent=video, video_frames_frozen=N_HEAD,
        audio_latent=audio, audio_frames_frozen=N_HEAD,
        strength=strength, clear_keyframes=False,
    )
    assert _video_bands(band_v)[0].strength == strength
    assert _audio_bands(band_a)[0].strength == strength
    assert repr(_video_bands(band_v)[0].strength) == repr(strength)


def test_the_two_routes_to_a_strength_disagree_at_0_3() -> None:
    """The direct value and the resolver's value are NOT interchangeable.

    Stated as a test rather than only as a comment: if a future refactor makes
    them equal (say by rounding), the reason this file separates them is gone
    and the assertion should be revisited deliberately.
    """
    video, _ = _latents()
    band_v, _ = _band_conditionings(
        video_latent=video, video_frames_frozen=N_HEAD,
        audio_latent=None, audio_frames_frozen=0,
        strength=0.3, clear_keyframes=False,
    )
    assert _video_bands(band_v)[0].strength != _freeze_strengths(1.0 - 0.3)[0]


# ---------------------------------------------------------------------------
# 3. head and tail masks
# ---------------------------------------------------------------------------


def test_head_and_tail_are_separate_items_with_disjoint_masks() -> None:
    """4 at the head, 3 at the tail, 10 in total: two items, no shared frame."""
    video, audio = _latents()
    band_v, band_a = _band_conditionings(
        video_latent=video, video_frames_frozen=N_HEAD,
        audio_latent=audio, audio_frames_frozen=N_HEAD,
        strength=0.5, clear_keyframes=False,
        video_tail_frozen=N_TAIL, audio_tail_frozen=N_TAIL,
        tail_strength=1.0,
    )

    head_v, tail_v = _video_bands(band_v)
    assert head_v is not tail_v
    assert head_v.mask.shape == (1, F_TOTAL, 2, 3)
    assert tail_v.mask.shape == head_v.mask.shape
    assert torch.equal(head_v.mask[:, :N_HEAD], torch.ones_like(head_v.mask[:, :N_HEAD]))
    assert head_v.mask[:, N_HEAD:].sum() == 0
    assert torch.equal(tail_v.mask[:, -N_TAIL:], torch.ones_like(tail_v.mask[:, -N_TAIL:]))
    assert tail_v.mask[:, :-N_TAIL].sum() == 0
    assert (head_v.mask * tail_v.mask).sum() == 0

    head_a, tail_a = _audio_bands(band_a)
    assert head_a is not tail_a
    assert head_a.mask.shape == (1, F_TOTAL)
    assert head_a.mask[:, :N_HEAD].sum() == N_HEAD
    assert head_a.mask[:, N_HEAD:].sum() == 0
    assert tail_a.mask[:, -N_TAIL:].sum() == N_TAIL
    assert tail_a.mask[:, :-N_TAIL].sum() == 0
    assert (head_a.mask * tail_a.mask).sum() == 0

    # Both halves condition on the SAME tensor; only the masks differ. Handing
    # the tail a slice instead would break the item's own shape assertion.
    assert tail_v.latent is video
    assert tail_a.latent is audio


def test_the_tail_may_be_frozen_without_a_head() -> None:
    """The end source's shape: nothing carried in, the material pinned at the end."""
    video, audio = _latents()
    band_v, band_a = _band_conditionings(
        video_latent=video, video_frames_frozen=0,
        audio_latent=audio, audio_frames_frozen=0,
        strength=0.5, clear_keyframes=False,
        video_tail_frozen=N_TAIL, audio_tail_frozen=N_TAIL,
        tail_strength=0.7, audio_tail_strength=1.0,
    )
    (tail_v,) = _video_bands(band_v)
    (tail_a,) = _audio_bands(band_a)
    assert tail_v.strength == 0.7
    assert tail_a.strength == 1.0
    assert tail_v.mask[:, :-N_TAIL].sum() == 0


def test_tail_strengths_default_the_way_the_shared_resolver_does() -> None:
    """Omitted overrides fall back exactly as ``freeze_mask_values`` does."""
    video, audio = _latents()
    band_v, band_a = _band_conditionings(
        video_latent=video, video_frames_frozen=N_HEAD,
        audio_latent=audio, audio_frames_frozen=N_HEAD,
        strength=0.5, clear_keyframes=False,
        video_tail_frozen=N_TAIL, audio_tail_frozen=N_TAIL,
    )
    assert [it.strength for it in _video_bands(band_v)] == [0.5, 0.5]
    assert [it.strength for it in _audio_bands(band_a)] == [0.5, 0.5]

    band_v, band_a = _band_conditionings(
        video_latent=video, video_frames_frozen=N_HEAD,
        audio_latent=audio, audio_frames_frozen=N_HEAD,
        strength=0.5, clear_keyframes=False,
        video_tail_frozen=N_TAIL, audio_tail_frozen=N_TAIL,
        audio_strength=0.0, tail_strength=1.0,
    )
    assert [it.strength for it in _video_bands(band_v)] == [0.5, 1.0]
    # audio head follows its own override; the audio tail follows the TAIL one,
    # which is the resolver's precedence.
    assert [it.strength for it in _audio_bands(band_a)] == [0.0, 1.0]


def test_overlapping_head_and_tail_is_refused() -> None:
    """Two bands that share a latent frame would silently fight over it."""
    video, audio = _latents(frames=6)
    with pytest.raises(AssertionError):
        _band_conditionings(
            video_latent=video, video_frames_frozen=4,
            audio_latent=None, audio_frames_frozen=0,
            strength=0.5, clear_keyframes=False,
            video_tail_frozen=3,
        )
    with pytest.raises(AssertionError):
        _band_conditionings(
            video_latent=None, video_frames_frozen=0,
            audio_latent=audio, audio_frames_frozen=4,
            strength=0.5, clear_keyframes=False,
            audio_tail_frozen=3,
        )


def test_a_zero_width_tail_adds_nothing() -> None:
    """The shipped chain's call: tails at 0, and the lists are what they were.

    One video item and one audio item, no tail, no marker clear -- i.e. the
    widening is inert until something asks for it, which is what makes the
    unchanged digests in gate G1(a)/(d) mean what they say.
    """
    video, audio = _latents()
    band_v, band_a = _band_conditionings(
        video_latent=video, video_frames_frozen=N_HEAD,
        audio_latent=audio, audio_frames_frozen=N_HEAD,
        strength=0.5, clear_keyframes=False,
    )
    assert len(band_v) == 1 and isinstance(band_v[0], VideoConditionByMask)
    assert len(band_a) == 1 and isinstance(band_a[0], AudioBandMask)

    band_v, band_a = _band_conditionings(
        video_latent=None, video_frames_frozen=0,
        audio_latent=None, audio_frames_frozen=0,
        strength=0.5, clear_keyframes=False,
    )
    assert band_v == [] and band_a == []


# ---------------------------------------------------------------------------
# 4. position: the band items come first
# ---------------------------------------------------------------------------


def _chain25_source() -> str:
    return Path(chain25.__file__).read_text(encoding="utf-8")


def test_every_video_conditioning_list_starts_with_the_band() -> None:
    """``conds_v = band_v + ...`` -- at every site, with nothing before it.

    This is the M-6 substitute for a ``VideoBandMask`` subclass: the ordering
    requirement (band items before anything that APPENDS tokens) is carried by
    construction order, so the construction order is what gets pinned.
    """
    assignments = re.findall(r"^\s*conds_v\s*=\s*(.+)$", _chain25_source(), re.M)
    assert assignments, "no `conds_v = ...` assignment found -- has the name changed?"
    assert all(rhs.startswith("band_v") for rhs in assignments), assignments


def test_conditioning_lists_handed_to_a_modality_are_band_led() -> None:
    """Every ``conditionings=`` argument is a band-led list, video and audio alike.

    The audio side passes ``band_a`` directly (there is no audio item that
    appends tokens today); the video side passes ``conds_v``, whose composition
    the test above pins. A bare ``band_v`` is band-led by definition and is
    allowed too. Anything else appearing here is a list this file has never
    checked the order of.
    """
    names = set(re.findall(r"conditionings=([A-Za-z_][A-Za-z_0-9]*)", _chain25_source()))
    assert names <= {"conds_v", "band_v", "band_a"}, names


# ---------------------------------------------------------------------------
# 5. the keyframe-marker clear, now decided by the caller
# ---------------------------------------------------------------------------


def test_the_marker_clear_can_now_stand_alone() -> None:
    """C-3's enabler: a cleared marker with NO frozen head.

    Before C1 the clear could only be reached from inside the "a head is frozen"
    branch, which tied the marker's correctness (a question about POSITION on
    the timeline) to a question about freezing. Reaching it alone is the whole
    reason the parameter moved out.
    """
    band_v, band_a = _band_conditionings(
        video_latent=None, video_frames_frozen=0,
        audio_latent=None, audio_frames_frozen=0,
        strength=0.5, clear_keyframes=True,
    )
    assert len(band_v) == 1 and isinstance(band_v[0], ClearKeyframesMask)
    assert band_a == []


def test_the_marker_clear_precedes_the_band_items() -> None:
    video, _ = _latents()
    band_v, _ = _band_conditionings(
        video_latent=video, video_frames_frozen=N_HEAD,
        audio_latent=None, audio_frames_frozen=0,
        strength=0.5, clear_keyframes=True,
        video_tail_frozen=N_TAIL,
    )
    assert [type(it) for it in band_v] == [
        ClearKeyframesMask, VideoConditionByMask, VideoConditionByMask
    ]


#: The two existing call sites, as (label, clear_flag, frozen_head_frames), and
#: whether the marker was cleared BEFORE C1 -- when the clear lived inside
#: ``video_latent is not None and video_frames_frozen > 0``.
_CALL_SITES = [
    # stage 1
    ("s1_clip0_plain", chain25.CLEAR_KEYFRAMES_ON_CARRY, 0, False),
    ("s1_clip0_v2v", chain25.CLEAR_KEYFRAMES_ON_V2V_HEAD, 4, False),
    ("s1_carry", chain25.CLEAR_KEYFRAMES_ON_CARRY, 2, True),
    # stage 2
    ("s2_tile0_plain", chain25.CLEAR_KEYFRAMES_ON_CARRY, 0, False),
    ("s2_tile0_v2v", chain25.CLEAR_KEYFRAMES_ON_V2V_HEAD, 4, False),
    ("s2_tile_join", chain25.CLEAR_KEYFRAMES_ON_CARRY, 6, True),
]


@pytest.mark.parametrize(("label", "flag", "frozen", "expected"), _CALL_SITES)
def test_existing_call_sites_keep_todays_effective_value(
    label: str, flag: bool, frozen: int, expected: bool
) -> None:
    """Each of the six situations the two call sites can be in, driven directly.

    The arguments are the ones ``run_chain`` builds (the constants are read from
    the module, not restated), and ``expected`` is what the pre-C1 code did.
    """
    video, audio = _latents()
    band_v, _ = _band_conditionings(
        video_latent=video if frozen else None, video_frames_frozen=frozen,
        audio_latent=audio if frozen else None, audio_frames_frozen=frozen,
        strength=0.5,
        # exactly the expression both call sites now spell out
        clear_keyframes=flag and frozen > 0,
    )
    assert any(isinstance(it, ClearKeyframesMask) for it in band_v) is expected, label


def test_the_two_call_sites_are_the_expressions_c3_left_behind() -> None:
    """The caller-side expressions are pinned, so the table above cannot go stale.

    C1's contract was "the effective value does not move", and both call sites
    spelled out ``<flag> and fkv > 0``. C3 CHANGED THE STAGE-1 ONE ON PURPOSE --
    the marker's correctness is a question about POSITION on the timeline, and a
    reverse schedule is where position and "is a head frozen" come apart -- so
    the head-freeze conjunct is gone from stage 1 and ``seg_clear_kf`` carries
    the whole predicate. This assertion is what made that a deliberate edit of a
    test rather than an unnoticed drift, and it goes on being that for the next
    change.

    STAGE 2 IS DELIBERATELY UNCHANGED, and the two halves are asserted together
    so that stays visible: tile 0 IS timeline position 0 (no schedule reorders
    tiles), and every tile with ``i > 0`` freezes ``kt_v > 0`` leading latents,
    so there the conjunct is not a coincidence -- it is the same fact said twice.
    """
    found = re.findall(r"^\s*clear_keyframes=(.+),$", _chain25_source(), re.M)
    assert found == ["seg_clear_kf", "tile_clear_kf and fkv > 0"], found


def test_the_stage1_predicate_is_the_carry_test_gate_m8_chose() -> None:
    """The predicate is pinned as a SOURCE EXPRESSION, because the thing it
    decides is invisible in any output a unit test can build and because it has
    now changed twice.

    C3 was planned around the TIMELINE INDEX (``clear_kf and i > 0``): the
    adversarial review argued the first-frame marker needs a fresh causal
    encode AND position 0, and made position decisive. Gate M8 was built to
    test exactly that, because only the end source's ``reverse`` mode can tell
    the two predicates apart -- and it came back AGAINST the design at both
    seeds (junction PSNR 30.236 vs 32.465 dB, 31.143 vs 32.631 dB; the line was
    'no worse than 0.5 dB'). The owner ruled on those numbers, so the shipped
    predicate is the CARRY test: clear the marker exactly when latent 0 is the
    previous segment's tail.

    A build that went back to ``i > 0`` would clear the marker on a reverse
    schedule's free-headed segments, which is the measured-worse arm; one that
    used ``clear_kf`` alone would clear it on the segment that opens the video.
    Both are silent picture faults, so the expression itself is pinned."""
    found = re.findall(r"^\s*seg_clear_kf = (.+)$", _chain25_source(), re.M)
    # One initialisation, plus the two i == 0 branches' constant overrides.
    assert found == [
        "clear_kf and layout.seg_head_source[i] is not None",
        "CLEAR_KEYFRAMES_ON_RETAKE_HEAD",
        "CLEAR_KEYFRAMES_ON_V2V_HEAD",
    ], found


def test_the_carry_predicate_is_the_pre_c3_value_on_every_forward_schedule() -> None:
    """The other half of the ruling, and the reason G3(g)'s digests held: on a
    FORWARD schedule ``seg_head_source[i] is not None`` and the pre-C3
    ``fkv > 0`` are the same set, element for element -- a carried head is
    exactly a frozen head. So the predicate change is a change of VOCABULARY
    there and of BEHAVIOUR only in ``reverse``, where every head is free and
    the table is all-None.

    Driven through the shared ``chain_math``, not restated: this is a claim
    about the layout the app and both engines share."""
    from chain_math import compute_chain_layout, resolve_stage2_window

    v_tile, v_adv = resolve_stage2_window(None)
    checked = 0
    for n_clips in (1, 2, 3, 5):
        for frames in (25, 49, 121):
            for kv in (1, 2, 3):
                layout = compute_chain_layout(
                    [frames] * n_clips, 24.0, kv=kv, v_tile=v_tile, v_adv=v_adv
                )
                n_seg = len(layout.seg_frames)
                # ``fkv`` on a plain forward chain is 0 at i == 0 and kv after.
                pre_c3 = [(kv if i > 0 else 0) > 0 for i in range(n_seg)]
                shipped = [layout.seg_head_source[i] is not None for i in range(n_seg)]
                assert shipped == pre_c3, (n_clips, frames, kv, shipped, pre_c3)
                checked += 1
    assert checked >= 30

    # ...and in ``reverse`` the table is all-None, i.e. NO segment clears --
    # which is the arm M8 measured as the better one.
    rev = compute_chain_layout(
        [169, 169], 24.0, kv=1, v_tile=v_tile, v_adv=v_adv, end_context_px=8
    )
    assert rev.end_source_mode == "reverse"
    assert all(h is None for h in rev.seg_head_source), rev.seg_head_source


def test_a_marker_clear_beside_a_tail_band_builds_both_in_order() -> None:
    """The builder's capability, NOT a shipped path -- and the difference is
    stated because it changed under this theme.

    C1 separated the marker clear from the head-freeze branch so §3-102 C3's
    reverse schedule could clear the marker on a free-headed segment. Gate M8
    then measured that such a segment should KEEP its marker, and the owner
    ruled accordingly, so no caller asks for this combination today: the
    shipped predicate clears only on a carried head, which always brings a head
    band with it.

    The test stays because the SEPARATION stays -- the two questions are
    independent whatever the predicate is -- and because a builder that quietly
    re-coupled them would be found here rather than by the next feature that
    needs them apart. ``clear_keyframes=True`` is passed directly for that."""
    video, audio = _latents()
    band_v, _ = _band_conditionings(
        video_latent=video, video_frames_frozen=0,
        audio_latent=audio, audio_frames_frozen=0,
        video_tail_frozen=2, audio_tail_frozen=2,
        strength=0.5,
        clear_keyframes=True,
    )
    # The marker clear AND the tail band, and the marker FIRST: every item here
    # is elementwise over the whole token axis, so order is carried by
    # construction and this is where that is checked.
    assert isinstance(band_v[0], ClearKeyframesMask)
    assert len(band_v) == 2
