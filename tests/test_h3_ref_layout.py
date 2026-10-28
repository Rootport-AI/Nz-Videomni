"""The Ref2VA reference rows in the packed sequence (台帳 §1-107, 弾 R1).

Runs in ``.venv-engine-h3`` through the engine runner, like test_h3_engine.py::

    .venv-engine-h3/Scripts/python.exe scripts/run_h3_pytest.py tests/test_h3_ref_layout.py

What is pinned:
  * without references, ``H3SeqLayout.make``, ``build_coords``, ``keyframe_t``
    and ``embed_rows`` give the numbers of the functions as they were before
    the reference rows (frozen copies below), bit for bit;
  * with a picture, a video (vt = 7) and an audio (rt = 5) reference: the
    layout ranges, the cursor walk, the shifted origin of the cond / audio /
    video rows and the audio references' w ends, against hand-built values;
  * ``embed_rows`` sends picture / video references through
    ``video_patch_proj`` and audio references through ``audio_patch_proj``.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("comfy_kitchen")

import h3_format  # noqa: E402
import numpy as np  # noqa: E402

# Tiny widths, the same as test_h3_engine.py.
HID, HEADS, HD, FFN, TEXT, LAT, AUD = 64, 1, 128, 48, 40, 24, 32


def _tiny_cfg():
    return h3_format.H3DiTConfig(
        prefix="", hidden_size=HID, num_layers=1, token_refiner_layers=1,
        num_heads=HEADS, head_dim=HD, ffn_hidden=FFN, text_dim=TEXT, latents_dim=LAT,
        audio_latents_dim=AUD, pruned=True, adaln_grid=1025, adaln_rank=8,
        rope_inv_freq_len=16, quant={}, unknown=(),
    )


# --------------------------------------------------------------------------- #
# frozen copies of the functions before the reference rows (git 9269a38)
# --------------------------------------------------------------------------- #

def _make_before(*, text_rows, cond_rows, audio_rows, video_rows):
    a = text_rows
    b = a + cond_rows
    c = b + audio_rows
    d = c + video_rows
    return dict(text=slice(0, a), cond=slice(a, b), audio=slice(b, c), video=slice(c, d), total=d)


def _keyframe_t_before(frame_idx, latent_frames, text_len):
    num_frames = (latent_frames - 2) // 5 * 17 + 5
    L = float(text_len)
    if frame_idx == 0:
        return L
    if frame_idx == num_frames - 1:
        spans = np.ones(latent_frames, dtype=np.float64) * (5.0 / 3.0)
        for k, per in enumerate((1, 4, 4, 4, 4)):
            spans[k::5] *= per
        return (L + float(spans.sum())) - (5.0 / 3.0)
    return L + (5.0 / 3.0) * frame_idx


def _build_coords_before(*, text_len, latent_frames, grid_h, grid_w, audio_latents, cond_frame_idx=()):
    from engine_h3.rope import _frame_grid

    def video_t_grid(n, origin):
        spans = torch.tensor([(5.0 / 3.0) * (1, 4, 4, 4, 4)[k % 5] for k in range(n)], dtype=torch.float64)
        return origin + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])

    cond_t = [_keyframe_t_before(n, latent_frames, text_len) for n in cond_frame_idx]
    latent_h, latent_w = grid_h * 2, grid_w * 2
    frame_rows = grid_h * grid_w
    cond_rows = len(cond_t) * frame_rows
    audio_rows = 2 * audio_latents
    video_rows = latent_frames * frame_rows
    total = text_len + cond_rows + audio_rows + video_rows
    g = torch.zeros(total, 3, dtype=torch.float64)
    L = float(text_len)
    g[:text_len, 0] = torch.arange(text_len, dtype=torch.float64)
    frame, w_grid = _frame_grid(latent_h, latent_w)
    start = text_len
    for k, t in enumerate(cond_t):
        cond = g[start + k * frame_rows:start + (k + 1) * frame_rows]
        cond[:, 0] = t
        cond[:, 1:] = frame
    start += cond_rows
    if audio_rows:
        audio = g[start:start + audio_rows]
        audio[:, 0] = (L + torch.arange(audio_latents, dtype=torch.float64)).repeat(2)
        audio[:, 2] = torch.cat(
            [
                torch.full((audio_latents,), float(w_grid[0]), dtype=torch.float64),
                torch.full((audio_latents,), float(w_grid[-1]), dtype=torch.float64),
            ]
        )
    start += audio_rows
    video = g[start:start + video_rows].view(latent_frames, frame_rows, 3)
    video[:, :, 0] = video_t_grid(latent_frames, L)[:, None]
    video[:, :, 1:] = frame[None]
    return g


def _embed_rows_before(dit, text_refined, audio_rows, video_rows, cond_rows=None):
    from engine_h3.dit import _linear_f32

    parts = [text_refined.to(torch.bfloat16)]
    if cond_rows is not None and cond_rows.shape[0] > 0:
        parts.append(_linear_f32(cond_rows, dit.video_patch_proj).to(torch.bfloat16))
    parts.append(_linear_f32(audio_rows, dit.audio_patch_proj).to(torch.bfloat16))
    parts.append(_linear_f32(video_rows, dit.video_patch_proj).to(torch.bfloat16))
    return torch.cat(parts, dim=0)


def _tiny_dit():
    from engine_h3.dit import H3DiT

    torch.manual_seed(0)
    dit = H3DiT(_tiny_cfg())
    for proj in (dit.video_patch_proj, dit.audio_patch_proj):
        torch.nn.init.normal_(proj.weight, std=0.1)
        torch.nn.init.normal_(proj.bias, std=0.1)
    return dit


# --------------------------------------------------------------------------- #
# (a) no references: the same numbers as before
# --------------------------------------------------------------------------- #

def test_make_without_references_is_the_function_of_before():
    from engine_h3.dit import H3SeqLayout

    for text, cond, audio, video in [(5, 0, 4, 6), (3, 0, 0, 2), (7, 8, 10, 12), (0, 4, 2, 4)]:
        got = H3SeqLayout.make(text_rows=text, cond_rows=cond, audio_rows=audio, video_rows=video)
        want = _make_before(text_rows=text, cond_rows=cond, audio_rows=audio, video_rows=video)
        for name, value in want.items():
            assert getattr(got, name) == value, name
        b = text + cond
        assert got.ref_video == slice(b, b) and got.ref_audio == slice(b, b)


def test_keyframe_t_without_origin_is_the_function_of_before():
    from engine_h3.rope import keyframe_t

    for n in (1, 2, 5):
        t_lat, num_frames = 5 * n + 2, 17 * n + 5
        for L in (0, 1, 7, 300, 4097):
            for f in range(num_frames):
                want = _keyframe_t_before(f, t_lat, L)
                assert keyframe_t(f, t_lat, L) == want
                assert keyframe_t(f, t_lat, L, origin=None) == want
                # origin == text_len is the same expression
                assert keyframe_t(f, t_lat, L, origin=float(L)) == want


@pytest.mark.parametrize("gh, gw", [(1, 2), (2, 3), (12, 20)])
@pytest.mark.parametrize("frames", [[], [0], [21], [0, 11, 21]])
@pytest.mark.parametrize("audio_latents", [0, 4])
def test_build_coords_without_references_is_the_function_of_before(gh, gw, frames, audio_latents):
    from engine_h3.rope import build_coords

    kw = dict(text_len=10, latent_frames=7, grid_h=gh, grid_w=gw, audio_latents=audio_latents, cond_frame_idx=frames)
    want = _build_coords_before(**kw)
    for got in (build_coords(**kw), build_coords(**kw, refs=())):
        assert got.dtype == want.dtype and torch.equal(got, want)


def test_embed_rows_without_references_is_the_function_of_before():
    dit = _tiny_dit()
    g = torch.Generator().manual_seed(1)
    text = torch.randn(5, HID, generator=g)
    audio = torch.randn(6, AUD, generator=g)
    video = torch.randn(8, LAT * 4, generator=g)
    for cond in (None, torch.zeros(0, LAT * 4), torch.randn(4, LAT * 4, generator=g)):
        want = _embed_rows_before(dit, text, audio, video, cond)
        for got in (
            dit.embed_rows(text, audio, video, cond),
            dit.embed_rows(text, audio, video, cond, ref_video_rows=None, ref_audio_rows=None),
            dit.embed_rows(
                text, audio, video, cond,
                ref_video_rows=torch.zeros(0, LAT * 4), ref_audio_rows=torch.zeros(0, AUD),
            ),
        ):
            assert got.dtype == want.dtype and torch.equal(got, want)


# --------------------------------------------------------------------------- #
# (b) with references: the cursor walk and the layout, by hand
# --------------------------------------------------------------------------- #

def _axis(latent_dim, sqrt_area):
    ratio = latent_dim / sqrt_area
    left = (1.0 - ratio) * 0.5
    return np.linspace(left, left + ratio, latent_dim // 2, endpoint=False) * 32


def _grid_rows(lh, lw):
    s = np.sqrt(lh * lw)
    h, w = _axis(lh, s), _axis(lw, s)
    return [(float(a), float(b)) for a in h for b in w], w


def test_build_coords_with_a_picture_a_video_and_an_audio():
    from engine_h3.dit import H3SeqLayout
    from engine_h3.rope import RefShape, build_coords, keyframe_t

    L, T, gh, gw, A = 10, 7, 1, 2, 3            # target: latent 2 x 4, F = 22
    pic = RefShape.picture(2, 2)                # 1 row
    vid = RefShape.video(7, 4, 2)               # 7 x (2 x 1) = 14 rows
    aud = RefShape.audio(5)                     # 10 rows
    assert (pic.rows, vid.rows, aud.rows) == (1, 14, 10)
    frames = [0, 21]
    # the audio is passed first: it still goes to the ref_audio block, after the video block
    g = build_coords(text_len=L, latent_frames=T, grid_h=gh, grid_w=gw, audio_latents=A,
                     cond_frame_idx=frames, refs=(aud, pic, vid))

    layout = H3SeqLayout.make(text_rows=L, cond_rows=2 * gh * gw, audio_rows=2 * A, video_rows=T * gh * gw,
                              ref_video_rows=pic.rows + vid.rows, ref_audio_rows=aud.rows)
    assert layout.text == slice(0, 10) and layout.cond == slice(10, 14)
    assert layout.ref_video == slice(14, 29) and layout.ref_audio == slice(29, 39)
    assert layout.audio == slice(39, 45) and layout.video == slice(45, 59) and layout.total == 59
    assert g.shape == (layout.total, 3) and g.dtype == torch.float64

    # the cursor, by hand: L, +1 (picture), +sum of the 7 video spans, +5 (audio)
    spans = [(5.0 / 3.0) * p for p in (1, 4, 4, 4, 4, 1, 4)]
    vid_t = [11.0]
    for s in spans[:-1]:
        vid_t.append(vid_t[-1] + s)
    vid_span = 0.0
    for s in spans:
        vid_span += s
    cursor = 11.0 + vid_span
    aud_origin = cursor
    cursor += 5.0
    assert abs(cursor - (16.0 + 110.0 / 3.0)) < 1e-12

    target_rows, target_w = _grid_rows(2 * gh, 2 * gw)

    # text
    assert g[:L, 0].tolist() == [float(k) for k in range(L)] and not bool(g[:L, 1:].any())
    # cond: keyframe_t with the origin moved to the final cursor
    cond = g[layout.cond]
    assert cond[:2, 0].tolist() == [cursor, cursor]
    last = cond[2:, 0].tolist()
    assert last == [keyframe_t(21, T, L, origin=cursor)] * 2
    sp = np.ones(T, dtype=np.float64) * (5.0 / 3.0)
    for k, per in enumerate((1, 4, 4, 4, 4)):
        sp[k::5] *= per
    assert last[0] == (cursor + float(sp.sum())) - 5.0 / 3.0
    assert [tuple(r) for r in cond[:, 1:].tolist()] == target_rows * 2
    # picture: t = L, its own 2 x 2 latent grid (one row at the centre)
    rv = g[layout.ref_video]
    assert rv[0].tolist() == [10.0, 0.0, 0.0]
    # video: 7 latents of its own 4 x 2 latent grid (2 rows each), t from 11
    own_rows, _ = _grid_rows(4, 2)
    vrows = rv[1:].view(7, 2, 3)
    for k in range(7):
        assert vrows[k, :, 0].tolist() == pytest.approx([vid_t[k]] * 2, abs=1e-12)
        assert [tuple(r) for r in vrows[k, :, 1:].tolist()] == own_rows
    # audio reference: t = aud_origin + k on 2 channels, h = 0, w = the TARGET grid's ends
    ra = g[layout.ref_audio]
    assert ra[:, 0].tolist() == pytest.approx([aud_origin + k for k in range(5)] * 2, abs=1e-12)
    assert not bool(ra[:, 1].any())
    assert ra[:, 2].tolist() == [float(target_w[0])] * 5 + [float(target_w[-1])] * 5
    # target audio and video: from the final cursor
    au = g[layout.audio]
    assert au[:, 0].tolist() == [cursor + k for k in range(A)] * 2
    assert au[:, 2].tolist() == [float(target_w[0])] * A + [float(target_w[-1])] * A
    vv = g[layout.video].view(T, gh * gw, 3)
    t_want = [cursor]
    for s in spans[:-1]:
        t_want.append(t_want[-1] + s)
    assert vv[:, 0, 0].tolist() == pytest.approx(t_want, abs=1e-12)
    assert float(vv[0, 0, 0]) == cursor
    # the rows after the references are the no-reference coordinates shifted by (cursor - L) in t
    plain = build_coords(text_len=L, latent_frames=T, grid_h=gh, grid_w=gw, audio_latents=A, cond_frame_idx=frames)
    tail = g[layout.audio.start:]
    plain_tail = plain[L + 4:]
    assert torch.equal(tail[:, 1:], plain_tail[:, 1:])
    assert torch.allclose(tail[:, 0] - plain_tail[:, 0], torch.full_like(tail[:, 0], cursor - L), atol=1e-12)


def test_build_coords_reference_order_within_a_block():
    """Pictures and videos keep the order given (the pipeline sorts them);
    the cursor advances in that order."""
    from engine_h3.rope import RefShape, build_coords

    kw = dict(text_len=4, latent_frames=2, grid_h=1, grid_w=1, audio_latents=0)
    a = build_coords(**kw, refs=(RefShape.picture(2, 2), RefShape.video(2, 2, 2)))
    b = build_coords(**kw, refs=(RefShape.video(2, 2, 2), RefShape.picture(2, 2)))
    assert a[4:7, 0].tolist() == [4.0, 5.0, 5.0 + 5.0 / 3.0]
    assert b[4:7, 0].tolist() == pytest.approx([4.0, 4.0 + 5.0 / 3.0, 4.0 + 25.0 / 3.0], abs=1e-12)
    # both end at the same cursor: 4 + 1 + 25/3
    assert float(a[-1, 0]) == pytest.approx(float(b[-1, 0]), abs=1e-12)


def test_build_coords_refuses_bad_references():
    from engine_h3.rope import RefShape, build_coords

    kw = dict(text_len=4, latent_frames=2, grid_h=1, grid_w=1, audio_latents=0)
    for bad in (
        RefShape.audio(0), RefShape.picture(3, 2), RefShape.picture(0, 2), RefShape.video(0, 2, 2),
        RefShape(kind="image", lh=2, lw=2, vt=1), RefShape(kind="picture", lh=2, lw=2, vt=2),
        ("picture", 2, 2),
    ):
        with pytest.raises(ValueError):
            build_coords(**kw, refs=(bad,))


# --------------------------------------------------------------------------- #
# (c) embed_rows: row count and projection of the reference rows
# --------------------------------------------------------------------------- #

def test_embed_rows_reference_rows_go_through_the_right_projection():
    from engine_h3.dit import H3SeqLayout, _linear_f32

    dit = _tiny_dit()
    g = torch.Generator().manual_seed(2)
    text = torch.randn(5, HID, generator=g)
    cond = torch.randn(2, LAT * 4, generator=g)
    ref_v = torch.randn(3, LAT * 4, generator=g)
    ref_a = torch.randn(4, AUD, generator=g)
    audio = torch.randn(6, AUD, generator=g)
    video = torch.randn(8, LAT * 4, generator=g)
    layout = H3SeqLayout.make(text_rows=5, cond_rows=2, audio_rows=6, video_rows=8,
                              ref_video_rows=3, ref_audio_rows=4)

    tok = dit.embed_rows(text, audio, video, cond, ref_video_rows=ref_v, ref_audio_rows=ref_a)
    assert tok.shape == (layout.total, HID) and tok.dtype == torch.bfloat16
    assert torch.equal(tok[layout.ref_video], _linear_f32(ref_v, dit.video_patch_proj).to(torch.bfloat16))
    assert torch.equal(tok[layout.ref_audio], _linear_f32(ref_a, dit.audio_patch_proj).to(torch.bfloat16))
    # the other rows are where they were, relative to their own blocks
    want = _embed_rows_before(dit, text, audio, video, cond)
    assert torch.equal(tok[:layout.cond.stop], want[:7])
    assert torch.equal(tok[layout.audio.start:], want[7:])

    # zero the video projection: only the cond / ref_video / video rows go to 0
    with torch.no_grad():
        dit.video_patch_proj.weight.zero_()
        dit.video_patch_proj.bias.zero_()
    tok = dit.embed_rows(text, audio, video, cond, ref_video_rows=ref_v, ref_audio_rows=ref_a)
    assert not bool(tok[layout.ref_video].any())
    assert bool(tok[layout.ref_audio].abs().sum(dim=1).gt(0).all())
    # and the audio projection
    dit = _tiny_dit()
    with torch.no_grad():
        dit.audio_patch_proj.weight.zero_()
        dit.audio_patch_proj.bias.zero_()
    tok = dit.embed_rows(text, audio, video, cond, ref_video_rows=ref_v, ref_audio_rows=ref_a)
    assert not bool(tok[layout.ref_audio].any())
    assert bool(tok[layout.ref_video].abs().sum(dim=1).gt(0).all())
