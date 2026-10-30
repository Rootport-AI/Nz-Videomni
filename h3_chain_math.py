"""MiniMax H3 chain (clip concatenation) geometry -- torch-free.

The H3 counterpart of ``chain_math.py``: the frame / latent bookkeeping of a
chain whose clips carry the previous clip's last latents over as their fixed
head (Docs/MINIMAX_H3_DESIGN.md 裁定 84; measured in VERIFICATION_LOG §183).
Lives at the repository root, next to ``chain_math.py`` / ``h3_format.py``, because the app
process (torch-less ``.venv``) needs it for the request check and the mock,
and the server side does not import the ``engine_h3`` package (a boundary).
The engine (``engine_h3.chain_h3``) lays out its clips from the same function.

Clip i has ``F_i = 17 a_i + 5`` frames, ``T'_i = 5 a_i + 2`` video latent
frames and ``A_i = round(F_i * 5/3)`` audio latents (40 per second at 24 fps).
The overlap is fixed at 22 frames = ``k`` = 7 video latent frames (裁定 84-3).
Clip i >= 1 starts at frame ``S_i`` of the timeline (``S_1 = F_0 - 22``,
``S_i = S_{i-1} + F_{i-1} - 22``) and its new content starts at
``J_i = S_i + 22``. Its fixed head is the previous clip's last ``k`` video
latents and the CONCATENATED audio's last ``ka_i = A_acc - round(S_i * 5/3)``
audio latents (``A_acc`` = the concatenated audio's actual length so far).
Concatenation: video ``cat(prev, cur[:, :, k:])``, audio
``cat(prev, cur[:, :, ka_i:])``.

The audio of a seam lands on a latent boundary only when ``S_i * 5/3`` is an
integer (``S_i`` a multiple of 3); otherwise it is placed up to 1/3 latent
off, and the concatenated audio can end one latent longer or shorter than
``round(F_total * 5/3)`` (``audio_latent_excess`` in -1..+1; the mp4 writer
cuts or keeps the audio at the video's length). Both are accepted and
recorded (裁定 84-6); ``audio_drift_flags`` tells the UI which seams drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

#: The overlap in pixel frames (裁定 84-3: fixed; the choice is 台帳 §4-44).
H3_CHAIN_OVERLAP_FRAMES = 22
#: The fixed rows' strength (裁定 84-1: fixed at 1.0, not a setting).
H3_CHAIN_OVERLAP_STRENGTH = 1.0
#: The clip count a chain accepts (the same range as the LTX chain).
H3_CHAIN_MIN_CLIPS = 2
H3_CHAIN_MAX_CLIPS = 24
#: The timeline's frame ceiling: the LTX chain's ``MAX_CHAIN_TOTAL_PIXEL_FRAMES``
#: (api/models.py; 24 * 481 = 11,544), held here because this module is
#: imported where ``api`` is not (agreement checked by test).
H3_CHAIN_MAX_TOTAL_FRAMES = 24 * 481

#: The H3 frame grid (17n+5) and the engine's shortest clip (engine_h3: 22).
_FRAMES_STEP = 17
_FRAMES_OFFSET = 5
_MIN_FIRST_FRAMES = 22
#: Video latent frames per grid step, and the audio latents per frame (40 / 24).
_LATENTS_PER_STEP = 5
_AUDIO_PER_FRAME = Fraction(5, 3)


def _latent_frames(frames: int) -> int:
    """``17a+5`` pixel frames -> ``5a+2`` video latent frames (the caller checked the grid)."""
    return (frames - _FRAMES_OFFSET) // _FRAMES_STEP * _LATENTS_PER_STEP + 2


def _audio_latents(frames: int) -> int:
    """``round(F * 5/3)``, exact (a third never ties). Equals engine_h3.constants.audio_latents."""
    return round(Fraction(frames) * _AUDIO_PER_FRAME)


#: The overlap in video latent frames: k = 5m + 2 for an overlap of 17m + 5 frames.
H3_CHAIN_OVERLAP_LATENTS = _latent_frames(H3_CHAIN_OVERLAP_FRAMES)
#: Clip i >= 1 needs at least one new latent frame past its fixed head:
#: T'_i - k >= 1, i.e. F_i >= overlap + 17 (39).
H3_CHAIN_MIN_NEXT_FRAMES = H3_CHAIN_OVERLAP_FRAMES + _FRAMES_STEP


@dataclass(frozen=True)
class H3ChainLayout:
    """The resolved geometry of one H3 chain. Per-seam lists have one entry per
    clip i >= 1 (index 0 = the seam into clip 1); per-clip lists one per clip."""

    clip_frames: list[int]              # F_i
    clip_latent_frames: list[int]       # T'_i
    clip_audio_latents: list[int]       # A_i
    seam_start_frames: list[int]        # S_i: clip i's first frame on the timeline
    seam_new_frames: list[int]          # J_i = S_i + overlap: clip i's first new frame
    audio_overlap_latents: list[int]    # ka_i: the audio head clip i keeps
    total_frames: int                   # F_total
    total_latent_frames: int            # T'_total
    total_audio_latents: int            # the concatenated audio's actual length
    audio_latents_expected: int         # round(F_total * 5/3)
    audio_latent_excess: int            # actual - expected, in -1..+1
    audio_place_error_latents: list[float]   # |round(S_i*5/3) - S_i*5/3| per seam
    audio_head_shift_latents: list[float]    # round(S_i*5/3) - S_i*5/3 per seam (signed)
    audio_drift_flags: list[bool]       # per clip: its seam's audio is off a latent boundary (clip 0: False)
    overlap_frames: int = H3_CHAIN_OVERLAP_FRAMES
    overlap_latents: int = H3_CHAIN_OVERLAP_LATENTS
    overlap_strength: float = H3_CHAIN_OVERLAP_STRENGTH
    # The LTX metadata writer's junction keys, same meaning: 0-based last frame
    # before the new content (the boundary is J / J+1), and the +-1 spread.
    segment_seam_junctions: list[int] = field(default_factory=list)
    all_junctions: list[int] = field(default_factory=list)

    @property
    def n_clips(self) -> int:
        return len(self.clip_frames)

    def to_dict(self) -> dict:
        return {
            # The keys services/pipeline_manager.py reads from an LTX layout, same meaning.
            "total_px": self.total_frames,
            "segment_seam_junctions": list(self.segment_seam_junctions),
            "tile_seam_junctions": [],
            "all_junctions": list(self.all_junctions),
            "n_tiles": 0,
            "overlap_frames": self.overlap_frames,
            "overlap_strength": self.overlap_strength,
            # H3's own.
            "clip_frames": list(self.clip_frames),
            "clip_latent_frames": list(self.clip_latent_frames),
            "clip_audio_latents": list(self.clip_audio_latents),
            "overlap_latents": self.overlap_latents,
            "seam_start_frames": list(self.seam_start_frames),
            "seam_new_frames": list(self.seam_new_frames),
            "audio_overlap_latents": list(self.audio_overlap_latents),
            "total_latent_frames": self.total_latent_frames,
            "total_audio_latents": self.total_audio_latents,
            "audio_latents_expected": self.audio_latents_expected,
            "audio_latent_excess": self.audio_latent_excess,
            "audio_place_error_latents": list(self.audio_place_error_latents),
            "audio_head_shift_latents": list(self.audio_head_shift_latents),
            "audio_drift_flags": list(self.audio_drift_flags),
        }


def _check_clip_frames(clip_frames) -> list[int]:
    frames = list(clip_frames)
    n = len(frames)
    if not H3_CHAIN_MIN_CLIPS <= n <= H3_CHAIN_MAX_CLIPS:
        raise ValueError(f"chain needs {H3_CHAIN_MIN_CLIPS}..{H3_CHAIN_MAX_CLIPS} clips (got {n})")
    out: list[int] = []
    for i, f in enumerate(frames):
        if isinstance(f, bool) or not isinstance(f, int):
            raise ValueError(f"clips[{i}]: num_frames must be an integer ({f!r})")
        if f < _FRAMES_OFFSET or (f - _FRAMES_OFFSET) % _FRAMES_STEP:
            raise ValueError(f"clips[{i}]: num_frames must be 17n+5")
        least = _MIN_FIRST_FRAMES if i == 0 else H3_CHAIN_MIN_NEXT_FRAMES
        if f < least:
            if i == 0:
                raise ValueError(f"clips[0]: num_frames must be at least {least}")
            raise ValueError(
                f"clips[{i}]: num_frames must be at least {least} "
                f"(overlap {H3_CHAIN_OVERLAP_FRAMES} + one grid step)"
            )
        out.append(f)
    return out


def _spread(junctions: list[int], total: int) -> list[int]:
    """The junctions with their +-1 neighbours, inside the timeline (chain_math's rule)."""
    out: set[int] = set()
    for j in junctions:
        out.update((j - 1, j, j + 1))
    return sorted(x for x in out if 0 <= x < total - 1)


def compute_h3_chain_layout(clip_frames) -> H3ChainLayout:
    """The chain's geometry for ``clip_frames`` (one 17n+5 count per clip, 2..24 clips).

    Raises ValueError (English, the 422 message; a clip's is prefixed with
    ``clips[i]: ``) for a count off the grid, ``clips[0]`` shorter than 22
    frames, a later clip shorter than 39 frames (it would have no new latent
    frame past its fixed head), a clip count outside 2..24 or a timeline
    longer than 11,544 frames.
    """
    frames = _check_clip_frames(clip_frames)
    overlap, k = H3_CHAIN_OVERLAP_FRAMES, H3_CHAIN_OVERLAP_LATENTS
    total = sum(frames) - overlap * (len(frames) - 1)
    if total > H3_CHAIN_MAX_TOTAL_FRAMES:
        raise ValueError(
            f"total frames {total} exceed the cap {H3_CHAIN_MAX_TOTAL_FRAMES} (reduce clip count or lengths)"
        )
    t_lat = [_latent_frames(f) for f in frames]
    a_lat = [_audio_latents(f) for f in frames]
    starts: list[int] = []
    news: list[int] = []
    ka: list[int] = []
    shift: list[float] = []
    a_acc = a_lat[0]
    start_frame = 0
    for i in range(1, len(frames)):
        start_frame += frames[i - 1] - overlap
        exact = Fraction(start_frame) * _AUDIO_PER_FRAME
        place = round(exact)
        ka_i = a_acc - place
        if not 0 <= ka_i <= a_lat[i]:
            # No 17n+5 input reaches this (the head is about 22 * 5/3 latents).
            raise AssertionError(f"audio overlap ka={ka_i} out of [0, {a_lat[i]}] at S={start_frame}")
        starts.append(start_frame)
        news.append(start_frame + overlap)
        ka.append(ka_i)
        shift.append(float(place - exact))
        a_acc += a_lat[i] - ka_i
    t_total = t_lat[0] + sum(t - k for t in t_lat[1:])
    if t_total != _latent_frames(total):
        raise AssertionError(f"T'_total {t_total} != latent_frames({total})")
    expected = _audio_latents(total)
    junctions = [j - 1 for j in news]
    return H3ChainLayout(
        clip_frames=frames,
        clip_latent_frames=t_lat,
        clip_audio_latents=a_lat,
        seam_start_frames=starts,
        seam_new_frames=news,
        audio_overlap_latents=ka,
        total_frames=total,
        total_latent_frames=t_total,
        total_audio_latents=a_acc,
        audio_latents_expected=expected,
        audio_latent_excess=a_acc - expected,
        audio_place_error_latents=[abs(x) for x in shift],
        audio_head_shift_latents=shift,
        audio_drift_flags=[False] + [x != 0.0 for x in shift],
        segment_seam_junctions=junctions,
        all_junctions=_spread(junctions, total),
    )
