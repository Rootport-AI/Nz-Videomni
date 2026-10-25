"""engine_h3.denoise_mask: the per-row continuous mask's pieces (台帳 §1-103, gate G1 ②④⑤⑥).

Runs in ``.venv-engine-h3`` through the engine-venv runner::

    .venv-engine-h3/Scripts/python.exe scripts/run_h3_pytest.py tests/test_h3_denoise_mask.py

Each piece is checked row by row against its definition (Docs/MINIMAX_H3_DESIGN.md
§3.1), with the expected values computed here independently (in float64, and
for the state velocity through the EXPANDED form the module does not use).
The whole denoise loop with a mask is checked in tests/test_h3_engine.py (G1 ①③④).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from engine_h3 import denoise_mask as DM  # noqa: E402


def _gen(seed):
    return torch.Generator().manual_seed(seed)


def _audio(a=6, seed=1):
    return torch.randn(2, 32, a, generator=_gen(seed))


def _video(t=3, lh=4, lw=6, seed=2):
    return torch.randn(1, 24, t, lh, lw, generator=_gen(seed))


# --------------------------------------------------------------------------- #
# building: all ones -> None; shapes and row order
# --------------------------------------------------------------------------- #

def test_all_ones_is_none_and_none_is_the_identity():
    clean_a, clean_v = _audio(), _video()
    assert DM.audio_mask(clean_a, torch.ones(6)) is None
    assert DM.audio_mask(clean_a, [1.0] * 6) is None
    assert DM.video_mask(clean_v, torch.ones(3), grid=6, noise=torch.zeros_like(clean_v)) is None
    x, v = _audio(seed=3), _audio(seed=4)
    # The very same objects: the no-mask path adds no tensor operation.
    assert DM.dit_input(x, None) is x
    assert DM.state_velocity(v, x, None, 0.7) is v
    assert DM.finish(x, None) is x


def test_audio_mask_rows_are_channel_0_then_channel_1():
    from engine_h3.dit import audio_to_rows

    clean = _audio(a=5)
    values = torch.tensor([0.0, 0.25, 0.5, 1.0, 0.75])
    mask = DM.audio_mask(clean, values)
    assert mask.m.shape == (1, 1, 5) and mask.m.dtype == torch.float32
    assert mask.rows.tolist() == values.tolist() * 2
    # The same order as the packed audio rows: broadcast m to the state and pack it.
    assert torch.equal(audio_to_rows(mask.m.expand(2, 32, 5))[:, 0], mask.rows)
    assert mask.inject is mask.clean and torch.equal(mask.clean, clean)
    assert mask.zero.tolist() == [[[True, False, False, False, False]]]


def test_video_mask_rows_are_t_then_h_then_w():
    """G1 ⑥: one value per latent frame, repeated over the frame's grid rows in t -> h -> w order."""
    from engine_h3.dit import patchify_video

    clean = _video(t=3, lh=4, lw=6)
    gh, gw = 2, 3
    values = torch.tensor([0.0, 0.5, 1.0])
    mask = DM.video_mask(clean, values, grid=gh * gw, noise=_video(seed=9))
    assert mask.m.shape == (1, 1, 3, 1, 1)
    assert mask.rows.tolist() == [0.0] * 6 + [0.5] * 6 + [1.0] * 6
    # patchify_video is the sequence's own order: m broadcast to the state, packed.
    packed = patchify_video(mask.m.expand(1, 24, 3, 4, 6).contiguous())
    assert packed.shape[0] == 3 * gh * gw
    assert torch.equal(packed[:, 0], mask.rows) and torch.equal(packed.amin(1), packed.amax(1))


def test_video_inject_is_0999_clean_plus_0001_noise():
    clean, noise = _video(), _video(seed=7)
    mask = DM.video_mask(clean, torch.tensor([0.0, 1.0, 0.3]), grid=6, noise=noise)
    want = clean.double() * 0.999 + noise.double() * 0.001
    assert torch.allclose(mask.inject.double(), want, atol=1e-6)
    assert not torch.equal(mask.inject, mask.clean)
    with pytest.raises(ValueError):
        DM.video_mask(clean, torch.zeros(3), grid=6, noise=noise[:, :, :2])


@pytest.mark.parametrize("bad", [[0.0] * 5, [0.0] * 7, [-0.1] + [1.0] * 5, [1.1] + [1.0] * 5,
                                 [float("nan")] + [1.0] * 5])
def test_audio_mask_refuses_bad_values(bad):
    with pytest.raises(ValueError):
        DM.audio_mask(_audio(), torch.tensor(bad))


# --------------------------------------------------------------------------- #
# per-row definitions (G1 ②)
# --------------------------------------------------------------------------- #

def test_dit_input_keeps_m1_rows_and_injects_m0_rows():
    clean = _audio()
    values = torch.tensor([0.0, 1.0, 0.5, 0.0, 1.0, 0.25])
    mask = DM.audio_mask(clean, values)
    x = _audio(seed=5)
    got = DM.dit_input(x, mask)
    for k, m in enumerate(values.tolist()):
        if m == 1.0:
            assert torch.equal(got[:, :, k], x[:, :, k])
        elif m == 0.0:
            assert torch.equal(got[:, :, k], clean[:, :, k])
        else:
            want = m * x[:, :, k].double() + (1 - m) * clean[:, :, k].double()
            assert torch.allclose(got[:, :, k].double(), want, atol=1e-6)
    video_clean, noise = _video(), _video(seed=8)
    vmask = DM.video_mask(video_clean, torch.tensor([0.0, 1.0, 0.5]), grid=6, noise=noise)
    xv = _video(seed=6)
    gv = DM.dit_input(xv, vmask)
    assert torch.equal(gv[:, :, 0], vmask.inject[:, :, 0])      # m = 0: the injected value, not clean
    assert torch.equal(gv[:, :, 1], xv[:, :, 1])


def _expanded_velocity(v, x, m, clean, inject, sigma):
    """The expanded form (plan §2.1), float64: m^2 v + (1-m^2)(clean-x)/s + m(1-m)(inject-clean)/s."""
    v, x, m, clean, inject = (t.double() for t in (v, x, m, clean, inject))
    return m * m * v + (1 - m * m) * (clean - x) / sigma + m * (1 - m) * (inject - clean) / sigma


def _direct_velocity(v, x, m, clean, inject, sigma):
    """The definition, float64: (x0_b - x) / s."""
    v, x, m, clean, inject = (t.double() for t in (v, x, m, clean, inject))
    x_in = m * x + (1 - m) * inject
    x0_hat = x_in + sigma * (m * v)
    x0_b = m * x0_hat + (1 - m) * clean
    return (x0_b - x) / sigma


@pytest.mark.parametrize("sigma", [1.0, 0.73, 0.05])
def test_state_velocity_matches_the_definition_audio_and_video(sigma):
    """G1 ⑤: audio (inject = clean) and video (inject != clean), mixed values per row."""
    clean_a = _audio()
    am = DM.audio_mask(clean_a, torch.tensor([0.0, 0.2, 0.5, 0.8, 1.0, 0.0]))
    x, v = _audio(seed=11), _audio(seed=12)
    got = DM.state_velocity(v, x, am, sigma)
    assert got.dtype == torch.float32
    for want in (_direct_velocity(v, x, am.m, am.clean, am.inject, sigma),
                 _expanded_velocity(v, x, am.m, am.clean, am.inject, sigma)):
        assert torch.allclose(got.double(), want, rtol=1e-5, atol=1e-5)
    # m = 1: the DiT's own velocity, up to the rounding of (x + s v - x) / s (bit identity
    # is the None path's job: an all-ones mask is never built).
    assert torch.allclose(got[:, :, 4], v[:, :, 4], rtol=1e-4, atol=1e-4 / sigma)

    clean_v = _video()
    vm = DM.video_mask(clean_v, torch.tensor([0.0, 0.5, 1.0]), grid=6, noise=_video(seed=13))
    xv, vv = _video(seed=14), _video(seed=15)
    got_v = DM.state_velocity(vv, xv, vm, sigma)
    for want in (_direct_velocity(vv, xv, vm.m, vm.clean, vm.inject, sigma),
                 _expanded_velocity(vv, xv, vm.m, vm.clean, vm.inject, sigma)):
        assert torch.allclose(got_v.double(), want, rtol=1e-5, atol=1e-5)
    # The (inject - clean) term is what makes the video differ from the audio form.
    without = _expanded_velocity(vv, xv, vm.m, vm.clean, vm.clean, sigma)
    assert not torch.allclose(got_v[:, :, 1].double(), without[:, :, 1], rtol=1e-7, atol=1e-7)


def test_m0_rows_follow_return_to_timestep_and_finish_exactly():
    """m = 0: the update is ``clean + (s'/s)(x - clean)``; finish() makes the row clean bit for bit."""
    from engine_h3.scheduler import euler_step, sigma_schedule

    clean = _audio(a=4)
    mask = DM.audio_mask(clean, torch.tensor([0.0, 1.0, 0.0, 0.5]))
    x = torch.randn(2, 32, 4, generator=_gen(21))
    sig = sigma_schedule(4, 3.0)
    for i in range(4):
        s, s1 = float(sig[i]), float(sig[i + 1])
        v = torch.randn(2, 32, 4, generator=_gen(30 + i))
        nxt = euler_step(x, DM.state_velocity(v, x, mask, s), s, s1)
        want = clean.double() + (s1 / s) * (x.double() - clean.double())
        assert torch.allclose(nxt[:, :, 0].double(), want[:, :, 0], atol=1e-5)
        x = nxt
    out = DM.finish(x, mask)
    assert torch.equal(out[:, :, 0], clean[:, :, 0]) and torch.equal(out[:, :, 2], clean[:, :, 2])
    assert torch.equal(out[:, :, 1], x[:, :, 1]) and torch.equal(out[:, :, 3], x[:, :, 3])


# --------------------------------------------------------------------------- #
# per-row times (G1 ②④)
# --------------------------------------------------------------------------- #

def test_masked_row_times_per_row_and_caps():
    rows = torch.tensor([0.0, 0.5, 1.0, 0.0005, 0.25])
    sigma, t_stream = 0.8, 1.0 - 0.8
    audio = DM.masked_row_times(t_stream, sigma, rows, DM.AUDIO_TIME_CAP)
    video = DM.masked_row_times(t_stream, sigma, rows, DM.VIDEO_TIME_CAP)
    f32 = lambda x: float(torch.tensor(x, dtype=torch.float32))  # noqa: E731
    assert audio.dtype == torch.float32 and audio.shape == rows.shape
    assert float(audio[0]) == 1.0 and float(video[0]) == f32(0.999)          # m = 0: the caps
    assert float(audio[1]) == f32(1 - 0.5 * sigma) and float(video[1]) == f32(1 - 0.5 * sigma)
    assert float(audio[2]) == f32(t_stream) and float(video[2]) == f32(t_stream)   # m = 1: the stream's t
    assert float(audio[3]) == f32(1 - float(torch.tensor(0.0005)) * sigma)    # 0.9996: below the audio cap
    assert float(video[3]) == f32(0.999)                                      # ... capped for video
    assert torch.allclose(audio[4].double(), torch.tensor(1 - 0.25 * sigma, dtype=torch.float64), atol=1e-7)


def test_time_rows_unique_set_with_masks():
    """G1 ④: the numbered set holds audio m=0 -> 1.0, video m=0 -> 0.999, m=0.5 -> 1 - 0.5 sigma."""
    from engine_h3.dit import H3SeqLayout
    from engine_h3.pipeline_h3 import time_rows

    a_lat, t_lat, grid = 3, 2, 4
    layout = H3SeqLayout.make(text_rows=2, cond_rows=grid, audio_rows=2 * a_lat, video_rows=t_lat * grid)
    sv, sa = 0.9, 0.6
    t_v, t_a = 1.0 - sv, 1.0 - sa
    am = DM.audio_mask(_audio(a=a_lat), torch.tensor([0.0, 0.5, 1.0]))
    vm = DM.video_mask(_video(t=t_lat, lh=4, lw=4), torch.tensor([0.0, 0.5]), grid=grid,
                       noise=_video(t=t_lat, lh=4, lw=4, seed=3))
    t_unique, idx = time_rows(
        layout, t_v, t_a,
        audio_t=DM.masked_row_times(t_a, sa, am.rows, DM.AUDIO_TIME_CAP),
        video_t=DM.masked_row_times(t_v, sv, vm.rows, DM.VIDEO_TIME_CAP),
    )
    t = t_unique[idx]
    f32 = lambda x: float(torch.tensor(x, dtype=torch.float32))  # noqa: E731
    audio = t[layout.audio].tolist()
    assert audio == [1.0, f32(1 - 0.5 * sa), f32(t_a)] * 2
    video = t[layout.video].tolist()
    assert video == [f32(0.999)] * grid + [f32(1 - 0.5 * sv)] * grid
    assert t[layout.text].tolist() == [f32(t_v)] * 2                 # text keeps the video t
    assert t[layout.cond].tolist() == [f32(0.999)] * grid           # cond still max(t_v, 0.999)
    assert sorted(set(t_unique.tolist())) == sorted({1.0, f32(1 - 0.5 * sa), f32(t_a), f32(0.999),
                                                     f32(1 - 0.5 * sv), f32(t_v)})
    assert int(t_unique.numel()) == 6                                # M > 3
