"""End source (end_source on POST /generate/chain) — API + mock e2e (no GPU).

The mirror of test_v2v_chain.py at the other end of the timeline: the last
``context_frames`` pixel frames of the chain come from an uploaded video or
still, hard-frozen the way a retake freezes its glue bands. The mock runner
mirrors the engine geometry via chain_math
(``compute_chain_layout(end_context_px=...)``), so the whole additive contract
(schema, endpoint 404/422, the ``context_frames + 1`` cut, the metadata
end_source block) is pinned without weights.

Two things this file guards that are easy to get wrong:

  * OUTPUT LENGTH IS THE CLIPS' OWN TOTAL, whichever mode the clip count picks:
      - ONE clip -> ``"in_window"``: the band is the clip's OWN tail, so the
        delivered mp4 is exactly ``num_frames`` frames, of which the last
        ``context_frames`` are the material.
      - TWO OR MORE clips -> ``"reverse"``: the band is the LAST clip's own
        tail and the clips are generated last-to-first towards it, so the mp4
        holds ``clips_total_px`` frames — the same length those clips would
        deliver with no end source at all.
    (V2V still trims its own frozen head off the FRONT, so a chain with both is
    ``total_px - trim_px``; that pair is single-clip only now.)
  * THE +1 PRIMER. The app always cuts/synthesises ``context_frames + 1``
    frames because the causal video VAE spends the first one on its lone
    keyframe latent. Mode-independent.

The retired ``internal_segment`` geometry delivered ``clips_total_px +
context_frames``, so THE FRAME COUNT ALONE DISCRIMINATES the two designs — the
multi-clip tests below assert it for exactly that reason.

Two v1 REJECTIONS stay gone (the band may span as many stage-2 tiles as it
needs, and it cannot collide with a clip-0 keyframe), and their INVERTED tests
are kept below so the old rules cannot be reintroduced by accident. A third —
"the band must fit inside the final clip" — is back in ``reverse`` mode for a
different reason, and has a boundary test of its own.

Frozen-API discipline: a request omitting end_source is byte-shape identical to
before (regression test below).
"""

from __future__ import annotations

import json

from PIL import Image

import chain_math
from services import video_io


BASE = {
    "prompt": "a lantern-lit alley in the rain, cinematic",
    "width": 384,
    "height": 256,
    "frame_rate": 24.0,
    "num_inference_steps": 8,
    "guidance_scale": 1.0,
    "seed": 123,
    "pipeline": "distilled",
    "overlap_frames": 2,
    "overlap_strength": 0.5,
}


def _run_chain(client, clips, **overrides):
    payload = {**BASE, "clips": clips, **overrides}
    return client.post("/api/v1/generate/chain", json=payload)


def _make_source_mp4(path, n_frames, fps, size=(96, 64)):
    frames = [Image.new("RGB", size, (i * 4 % 256, 90, 160)) for i in range(n_frames)]
    video_io.encode_frames_to_mp4(frames, path, frame_rate=fps)
    return path


def _upload_video(client, mp4_path) -> str:
    data = mp4_path.read_bytes()
    r = client.post("/api/v1/upload/video", files={"file": ("end.mp4", data, "video/mp4")})
    assert r.status_code == 200, r.text
    return r.json()["video_id"]


def _upload_image(client, png_bytes) -> str:
    r = client.post("/api/v1/upload/image", files={"file": ("end.png", png_bytes, "image/png")})
    assert r.status_code == 200, r.text
    return r.json()["image_id"]


def _completed(client, r):
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    job = client.get(f"/api/v1/jobs/{job_id}").json()
    assert job["status"] == "completed", job
    return job_id


def _metadata(client, job_id) -> dict:
    ctx = client.app_context
    path = ctx.config.output_dir / job_id / "metadata.json"
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- (a) regression


def test_chain_without_end_source_regression(client):
    """A chain omitting end_source: the field defaults to None and neither the
    top-level nor the chain metadata grows an end_source block — the pre-feature
    metadata key set is unchanged."""
    r = _run_chain(client, [{"num_frames": 25}, {"num_frames": 25}])
    job_id = _completed(client, r)
    meta = _metadata(client, job_id)
    assert meta["request"]["end_source"] is None
    assert "end_source" not in meta
    assert "end_source" not in meta["chain"]


# ---------------------------------------------------------------- (b) mock e2e


def test_end_source_video_mock_e2e(client, tmp_path):
    """Video end source on ONE clip ("in_window" mode): 202 -> completed, the
    delivered mp4 is the clip itself with its last frames frozen onto the
    material, and the metadata end_source block carries geometry + runtime +
    provenance — but never a fabricated freeze_proof."""
    src = _make_source_mp4(tmp_path / "src24.mp4", n_frames=40, fps=24.0)
    vid = _upload_video(client, src)

    r = _run_chain(
        client, [{"num_frames": 49}],
        end_source={"video_id": vid, "context_frames": 24},
    )
    job_id = _completed(client, r)

    layout = chain_math.compute_chain_layout([49], 24.0, kv=2, end_context_px=24)
    ctx = client.app_context
    out = ctx.config.output_dir / job_id / "output.mp4"
    # THE headline property of the in-window mode: a 49-frame clip plus a
    # 24-frame band is a 49-frame mp4 — the band is the clip's own tail, so the
    # requested length is the delivered length.
    assert video_io.frame_count(out) == layout.total_px == 49

    meta = _metadata(client, job_id)
    es = meta["end_source"]
    # geometry (from chain_math)
    assert es["mode"] == "in_window"
    assert es["end_context_px"] == 24
    assert es["n_end_v"] == layout.n_end_v == 3
    assert es["end_source_junction_px"] == layout.end_source_junction_px == 24
    assert es["cut_frames"] == 25  # context_frames + 1 primer
    # the identity this mode exists to guarantee: the clips ARE the timeline
    assert es["clips_total_px"] == 49 == layout.total_px
    # ...and no band segment was appended at all
    assert es["end_segment_latent"] == layout.end_segment_latent == 0
    assert es["end_segment_px"] == layout.end_segment_px == 0
    # the stage-2 freeze plan reaches exactly the end of the band
    assert layout.end_tile_bands[-1][1] == layout.n_end_v
    assert es["end_tile_bands"] == [list(b) for b in layout.end_tile_bands]
    # runtime (mock) + provenance (app-side)
    assert es["kind"] == "video"
    assert es["decoded_frames_px"] == layout.total_px
    assert es["end_source_video_id"] == vid
    assert es["resampled"] is False
    assert es["written_frames"] == 25
    # The mock has no latents, so it must NOT claim a passing freeze proof.
    assert "freeze_proof" not in es
    # the additive request field round-trips
    assert meta["request"]["end_source"]["video_id"] == vid
    assert meta["request"]["end_source"]["context_frames"] == 24
    # the app-cut material really is context_frames + 1 frames long
    cut = ctx.config.output_dir / job_id / "_end_source.mp4"
    assert video_io.frame_count(cut) == 25


def test_in_window_mock_e2e_at_the_experiment_geometry(client, tmp_path):
    """The exact shape the real-hardware experiment submits (one 169-frame clip,
    24-frame band), end to end through the mock. Guards the three statements a
    finished job is machine-checked on: the output is the requested length, the
    metadata says which mode ran, and no band segment was appended."""
    src = _make_source_mp4(tmp_path / "src.mp4", n_frames=60, fps=24.0)
    vid = _upload_video(client, src)
    r = _run_chain(
        client, [{"num_frames": 169}],
        end_source={"video_id": vid, "context_frames": 24},
    )
    job_id = _completed(client, r)

    layout = chain_math.compute_chain_layout([169], 24.0, kv=2, end_context_px=24)
    ctx = client.app_context
    out = ctx.config.output_dir / job_id / "output.mp4"
    assert video_io.frame_count(out) == 169 == layout.total_px

    es = _metadata(client, job_id)["end_source"]
    assert es["mode"] == "in_window"
    assert es["end_segment_latent"] == 0
    assert es["end_segment_px"] == 0
    assert es["clips_total_px"] == 169
    assert es["end_source_junction_px"] == 144      # 169 - 24 - 1
    assert es["end_tile_bands"] == [[3, 3]]         # one stage-2 tile
    # The mock still has no latents, so it must not claim a freeze proof.
    assert "freeze_proof" not in es
    assert video_io.frame_count(ctx.config.output_dir / job_id / "_end_source.mp4") == 25


def test_end_source_video_resample_is_recorded(client, tmp_path):
    src = _make_source_mp4(tmp_path / "src30.mp4", n_frames=40, fps=30.0)
    vid = _upload_video(client, src)
    r = _run_chain(
        client, [{"num_frames": 49}],
        end_source={"video_id": vid, "context_frames": 24},
    )
    job_id = _completed(client, r)
    es = _metadata(client, job_id)["end_source"]
    assert es["resampled"] is True
    assert abs(es["source_fps"] - 30.0) < 0.5


def test_end_source_image_mock_e2e(client, png_bytes):
    """Image end source: the still is looped into a silent 9-frame mp4 (8 + the
    primer) so the engine sees the same kind of file a video end source produces."""
    image_id = _upload_image(client, png_bytes)
    r = _run_chain(
        client, [{"num_frames": 49}],
        end_source={"image_id": image_id, "context_frames": 8},
    )
    job_id = _completed(client, r)

    layout = chain_math.compute_chain_layout([49], 24.0, kv=2, end_context_px=8)
    ctx = client.app_context
    out = ctx.config.output_dir / job_id / "output.mp4"
    assert video_io.frame_count(out) == layout.total_px == 49

    meta = _metadata(client, job_id)
    es = meta["end_source"]
    assert es["kind"] == "image"
    assert es["mode"] == "in_window"
    assert es["end_context_px"] == 8
    assert es["n_end_v"] == 1
    assert es["cut_frames"] == 9
    assert es["end_source_image_id"] == image_id
    assert es["written_frames"] == 9
    assert "freeze_proof" not in es
    assert video_io.frame_count(ctx.config.output_dir / job_id / "_end_source.mp4") == 9


def test_end_source_default_context_frames_is_72(client, tmp_path):
    """context_frames omitted -> the published default (72). On one clip the band
    is that clip's last 72 frames, so a 97-frame request delivers 97 frames: 25
    newly generated, then the material."""
    src = _make_source_mp4(tmp_path / "long.mp4", n_frames=80, fps=24.0)
    vid = _upload_video(client, src)
    r = _run_chain(client, [{"num_frames": 97}], end_source={"video_id": vid})
    job_id = _completed(client, r)
    meta = _metadata(client, job_id)
    assert meta["request"]["end_source"]["context_frames"] == 72
    assert meta["end_source"]["mode"] == "in_window"
    assert meta["end_source"]["end_context_px"] == 72
    assert meta["end_source"]["cut_frames"] == 73
    assert meta["end_source"]["clips_total_px"] == 97
    ctx = client.app_context
    out = ctx.config.output_dir / job_id / "output.mp4"
    assert video_io.frame_count(out) == 97


def test_end_source_allows_a_single_clip(client, tmp_path):
    """"A five-second video that ENDS with this" is a single clip + end source —
    the >= 2 clip floor is lifted for it, exactly as it is for a start source.
    It is also the shape that selects the in-window mode, so the accepted job
    must actually report that mode rather than quietly running the old one."""
    src = _make_source_mp4(tmp_path / "src.mp4", n_frames=40, fps=24.0)
    vid = _upload_video(client, src)
    r = _run_chain(
        client, [{"num_frames": 49}],
        end_source={"video_id": vid, "context_frames": 24},
    )
    assert r.status_code == 202, r.text
    assert r.json()["num_clips"] == 1
    job_id = _completed(client, r)
    es = _metadata(client, job_id)["end_source"]
    assert es["mode"] == "in_window"
    assert es["end_segment_latent"] == 0


def test_end_source_with_source_video_is_allowed(client, tmp_path):
    """Start source + end source = interpolation, the headline use case."""
    start = _make_source_mp4(tmp_path / "start.mp4", n_frames=40, fps=24.0)
    end = _make_source_mp4(tmp_path / "end.mp4", n_frames=40, fps=24.0)
    start_id = _upload_video(client, start)
    end_id = _upload_video(client, end)

    r = _run_chain(
        client, [{"num_frames": 73}],
        source_video={"video_id": start_id, "context_frames": 25},
        end_source={"video_id": end_id, "context_frames": 24},
    )
    job_id = _completed(client, r)

    layout = chain_math.compute_chain_layout(
        [73], 24.0, kv=2, source_context_px=25, end_context_px=24
    )
    ctx = client.app_context
    out = ctx.config.output_dir / job_id / "output.mp4"
    # One clip -> in-window mode, so the 73-frame window holds the frozen head,
    # the free middle AND the frozen band. V2V still trims its own head off the
    # FRONT and the end source adds no trim of its own, so 73 - 25 == 48 frames
    # are delivered: 24 newly generated, then the 24-frame band.
    assert video_io.frame_count(out) == layout.new_frames_px == 48

    meta = _metadata(client, job_id)
    assert meta["v2v"]["source_video_id"] == start_id
    assert meta["end_source"]["mode"] == "in_window"
    assert meta["end_source"]["end_source_video_id"] == end_id
    assert meta["end_source"]["end_source_junction_px"] == layout.end_source_junction_px


def test_end_source_with_clip0_keyframe_is_allowed(client, tmp_path, png_bytes):
    """A clip-0 keyframe OUTSIDE the frozen band is fine (unlike a retake, which
    refuses conditioning images outright)."""
    src = _make_source_mp4(tmp_path / "src.mp4", n_frames=40, fps=24.0)
    vid = _upload_video(client, src)
    image_id = _upload_image(client, png_bytes)
    clips = [{
        "num_frames": 49,
        "conditioning_images": [{"image_id": image_id, "frame_idx": 0, "strength": 0.9}],
    }]
    r = _run_chain(client, clips, end_source={"video_id": vid, "context_frames": 24})
    assert r.status_code == 202, r.text


# -------------------------------------------------------------------- (c) 404


def test_end_source_unknown_video_id_404(client):
    r = _run_chain(
        client, [{"num_frames": 49}],
        end_source={"video_id": "no-such-video", "context_frames": 24},
    )
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "END_SOURCE_NOT_FOUND"


def test_end_source_unknown_image_id_404(client):
    """The id resolves against the IMAGE store when image_id was sent — a
    video_id-shaped lookup would 404 on the wrong store (or worse, succeed)."""
    r = _run_chain(
        client, [{"num_frames": 49}],
        end_source={"image_id": "no-such-image", "context_frames": 8},
    )
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "END_SOURCE_NOT_FOUND"


# -------------------------------------------------------------- (d) too short


def test_end_source_too_short_422(client, tmp_path):
    """24 requested frames + 1 primer = 25 needed; a 10-frame upload cannot."""
    src = _make_source_mp4(tmp_path / "short.mp4", n_frames=10, fps=24.0)
    vid = _upload_video(client, src)
    r = _run_chain(
        client, [{"num_frames": 49}],
        end_source={"video_id": vid, "context_frames": 24},
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "END_SOURCE_TOO_SHORT"


def test_end_source_exactly_context_frames_long_is_too_short_422(client, tmp_path):
    """The +1 primer is load-bearing: a source of exactly context_frames frames
    is one frame short, and that must fail up front rather than in the engine."""
    src = _make_source_mp4(tmp_path / "exact.mp4", n_frames=24, fps=24.0)
    vid = _upload_video(client, src)
    r = _run_chain(
        client, [{"num_frames": 49}],
        end_source={"video_id": vid, "context_frames": 24},
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "END_SOURCE_TOO_SHORT"


# ------------------------------------------------------------- (e) exclusivity


def test_end_source_conflicts_with_retake_422(client):
    r = _run_chain(
        client, [{"num_frames": 73}],
        retake={"video_id": "rt", "window_start_sec": 0.0},
        end_source={"video_id": "es", "context_frames": 24},
    )
    assert r.status_code == 422
    assert "mutually exclusive" in r.text


def test_end_source_conflicts_with_source_audio_422(client):
    r = _run_chain(
        client, [{"num_frames": 49}],
        source_audio={"audio_id": "aud"},
        end_source={"video_id": "es", "context_frames": 24},
    )
    assert r.status_code == 422
    assert "mutually exclusive" in r.text


def test_end_source_conflicts_with_reference_video_422(client):
    r = _run_chain(
        client, [{"num_frames": 49}],
        reference_video_id="ref",
        loras=[{"name": "pixel-spatial-upscaler-x2", "strength": 1.0}],
        end_source={"video_id": "es", "context_frames": 24},
    )
    assert r.status_code == 422
    assert "mutually exclusive" in r.text


def test_end_source_keyframe_near_the_end_of_clip0_is_allowed(client, tmp_path, png_bytes):
    """INVERTED v1 test. Frame 33 of a 49-frame clip fell inside v1's frozen band
    (which was carved out of the clip: frames 25..48) and was a 422. The band is
    now a separate segment appended after the clip, so NO clip frame can reach
    it and the combination is unconditionally legal — the collision check is
    gone, not merely relaxed."""
    src = _make_source_mp4(tmp_path / "src.mp4", n_frames=40, fps=24.0)
    vid = _upload_video(client, src)
    image_id = _upload_image(client, png_bytes)
    clips = [{
        "num_frames": 49,
        "conditioning_images": [{"image_id": image_id, "frame_idx": 33, "strength": 0.9}],
    }]
    r = _run_chain(client, clips, end_source={"video_id": vid, "context_frames": 24})
    job_id = _completed(client, r)
    meta = _metadata(client, job_id)
    # The keyframe is 16 frames clear of the band, which starts at frame 49.
    assert meta["end_source"]["clips_total_px"] == 49
    assert meta["request"]["clips"][0]["conditioning_images"][0]["frame_idx"] == 33


# ----------------------------------------------------------------- (f) schema


def test_end_source_requires_exactly_one_id_both_422(client):
    r = _run_chain(
        client, [{"num_frames": 49}],
        end_source={"video_id": "v", "image_id": "i", "context_frames": 24},
    )
    assert r.status_code == 422
    assert "exactly one of video_id / image_id" in r.text


def test_end_source_requires_exactly_one_id_neither_422(client):
    r = _run_chain(client, [{"num_frames": 49}], end_source={"context_frames": 24})
    assert r.status_code == 422
    assert "exactly one of video_id / image_id" in r.text


def test_end_source_context_frames_not_multiple_of_8_422(client):
    r = _run_chain(
        client, [{"num_frames": 49}],
        end_source={"video_id": "v", "context_frames": 71},
    )
    assert r.status_code == 422
    assert "multiple of 8" in r.text


def test_end_source_context_frames_below_min_422(client):
    r = _run_chain(
        client, [{"num_frames": 49}],
        end_source={"video_id": "v", "context_frames": 4},
    )
    assert r.status_code == 422
    assert ">= 8" in r.text


def test_end_source_context_frames_above_max_422(client):
    # 144 is a valid multiple of 8 but exceeds the published ceiling of 136.
    r = _run_chain(
        client, [{"num_frames": 481}],
        end_source={"video_id": "v", "context_frames": 144},
    )
    assert r.status_code == 422
    assert "136" in r.text


def test_last_clip_too_short_for_the_band_is_rejected_at_the_boundary(client, tmp_path):
    """``reverse`` mode puts the band back INSIDE the last clip, so that clip has
    to hold the band AND the のり代 the clip before it takes from its head. With
    K_v=2 and a 24-frame band (3 latents) the last clip needs 6 latents == 41
    pixel frames: 33 is refused, 41 is accepted and delivers the clips' own total.

    (This is the one v1 rejection the internal-segment mode had removed and this
    mode brings back — for a different reason. v1 refused because the band had to
    fit in the final clip's stage-2 tile; this refuses because the latents the
    reverse carry hands BACKWARDS would otherwise be frozen material rather than
    newly generated content.)"""
    src = _make_source_mp4(tmp_path / "src.mp4", n_frames=40, fps=24.0)
    vid = _upload_video(client, src)
    end_source = {"video_id": vid, "context_frames": 24}

    r = _run_chain(client, [{"num_frames": 33}, {"num_frames": 33}], end_source=end_source)
    assert r.status_code == 422
    assert "41 pixel frames" in r.text
    assert "nothing to carry backwards" in r.text

    r = _run_chain(client, [{"num_frames": 41}, {"num_frames": 41}], end_source=end_source)
    job_id = _completed(client, r)
    meta = _metadata(client, job_id)
    layout = chain_math.compute_chain_layout([41, 41], 24.0, kv=2, end_context_px=24)
    assert meta["end_source"]["clips_total_px"] == layout.total_px
    ctx = client.app_context
    out = ctx.config.output_dir / job_id / "output.mp4"
    assert video_io.frame_count(out) == layout.total_px
    assert meta["chain"]["num_clips"] == 2
    assert meta["chain"]["clip_num_frames"] == [41, 41]


def test_start_and_end_source_together_need_a_single_clip(client, tmp_path):
    """A start source and an end source interpolate BETWEEN two materials, which
    stays accepted on ONE clip. On a chain it would leave clip 0 frozen at both
    ends (head from the source video, tail from the reverse のり代) — refused
    rather than shipped untested."""
    src = _make_source_mp4(tmp_path / "src.mp4", n_frames=120, fps=24.0)
    start_id = _upload_video(client, src)
    end_id = _upload_video(client, src)
    end_source = {"video_id": end_id, "context_frames": 24}
    source_video = {"video_id": start_id, "context_frames": 25}

    r = _run_chain(
        client, [{"num_frames": 169}, {"num_frames": 169}],
        source_video=source_video, end_source=end_source,
    )
    assert r.status_code == 422
    assert "cannot be combined" in r.text

    # The same pair on ONE clip is the interpolation case and is unchanged.
    r = _run_chain(
        client, [{"num_frames": 169}],
        source_video=source_video, end_source=end_source,
    )
    job_id = _completed(client, r)
    assert _metadata(client, job_id)["end_source"]["mode"] == "in_window"


# ------------------------------------------------- (g) stage-2 window ceiling


def test_end_source_above_the_old_high_resolution_ceiling_is_allowed(client, tmp_path):
    """INVERTED v1 test. The "high_resolution" window advances by 12 latents
    instead of 18, and v1 derived a 8*(12-1) = 88 ceiling from that and refused
    96. A band may now span as many stage-2 tiles as it needs, so the per-window
    ceiling is gone: 96 is accepted."""
    src = _make_source_mp4(tmp_path / "src.mp4", n_frames=120, fps=24.0)
    vid = _upload_video(client, src)
    r = _run_chain(
        client, [{"num_frames": 145}],
        stage2_window="high_resolution",
        end_source={"video_id": vid, "context_frames": 96},
    )
    job_id = _completed(client, r)
    meta = _metadata(client, job_id)
    assert meta["end_source"]["end_context_px"] == 96
    assert meta["end_source"]["clips_total_px"] == 145
    bands = meta["end_source"]["end_tile_bands"]
    assert meta["end_source"]["n_end_v"] == 12
    assert bands[-1][1] == 12


def test_end_source_at_the_published_ceiling_under_high_resolution_202(client, tmp_path):
    """136 is now an OPERATIONAL cap that applies to every stage-2 window alike,
    so the narrowest published window must accept it too (v1 capped this preset
    at 88). This is the geometry the H12 real-run gate then judges for quality.

    TWO clips deliberately, so this doubles as the ``reverse`` mode's end-to-end
    pin at the ceiling. 153-frame clips are the shortest that can take a
    136-frame band at K_v=2 (20 latents against 2 + 17), and they leave the last
    stage-2 tile shorter than the band, which is what makes the straddle
    assertions below bite."""
    src = _make_source_mp4(tmp_path / "src.mp4", n_frames=160, fps=24.0)
    vid = _upload_video(client, src)
    r = _run_chain(
        client, [{"num_frames": 153}, {"num_frames": 153}],
        stage2_window="high_resolution",
        end_source={"video_id": vid, "context_frames": 136},
    )
    job_id = _completed(client, r)
    meta = _metadata(client, job_id)
    assert meta["end_source"]["mode"] == "reverse"
    assert meta["end_source"]["end_context_px"] == 136
    assert meta["end_source"]["n_end_v"] == 17
    # The band is the LAST CLIP'S OWN TAIL, so the delivered length is the clips'
    # own total — it does NOT grow by the band.
    assert meta["end_source"]["clips_total_px"] == 297
    ctx = client.app_context
    assert video_io.frame_count(ctx.config.output_dir / job_id / "output.mp4") == 297
    # ...and this is a genuine STRADDLING band: two tiles each carry part of it,
    # which is the whole reason the per-window ceiling could be dropped. (The
    # per-tile counts SUM TO MORE than n_end_v -- stage-2 tiles overlap, so a
    # band latent near a tile seam is written into both tiles. The band's own
    # extent is the last tile's offset.)
    bands = meta["end_source"]["end_tile_bands"]
    assert sum(1 for t, _off in bands if t > 0) >= 2
    assert bands[-1][1] == 17
    assert max(t for t, _off in bands) < 17


def test_published_ceiling_is_operational_not_geometric():
    """The published ceiling is a plain constant now. It must NOT be re-derived
    from any stage-2 window's geometry — that function is gone precisely so no
    caller can reintroduce the coupling."""
    from config import LimitsConfig

    assert LimitsConfig().end_context_frames_max == 136
    assert not hasattr(chain_math, "stage2_max_end_context_px")


# ------------------------------------------------ (g2) the total-timeline cap


def test_total_timeline_cap_is_charged_on_the_clips_only(client, tmp_path, monkeypatch):
    """MAX_CHAIN_TOTAL_PIXEL_FRAMES bounds what the USER asked to generate, not
    what gets delivered. ``clips_total_px`` is the ONE definition of that
    subtraction, and it is what the cap is charged on — which is why it keeps
    holding after the mode change made the two numbers equal in every reachable
    mode: the property, not the coincidence, is what is pinned.

    The real cap is unreachable by construction (11544 is the naive sum, and
    every overlap makes the assembled timeline shorter), so it is lowered here to
    put the boundary where a small chain can reach it: with the cap AT the clip
    total, a request whose delivered length is well past it must still be
    accepted — that is exactly the comparison v1 got wrong."""
    import api.models as models

    src = _make_source_mp4(tmp_path / "src.mp4", n_frames=40, fps=24.0)
    vid = _upload_video(client, src)
    end_source = {"video_id": vid, "context_frames": 24}

    # clips_total_px == 49, delivered == 73.
    monkeypatch.setattr(models, "MAX_CHAIN_TOTAL_PIXEL_FRAMES", 49)
    r = _run_chain(client, [{"num_frames": 49}], end_source=end_source)
    assert r.status_code == 202, r.text

    # One frame lower and the CLIPS themselves are over: still a 422, and the
    # message names the clip total (49), never the delivered 73.
    monkeypatch.setattr(models, "MAX_CHAIN_TOTAL_PIXEL_FRAMES", 48)
    r = _run_chain(client, [{"num_frames": 49}], end_source=end_source)
    assert r.status_code == 422
    assert "49 pixel frames exceeds the cap 48" in r.text


def test_maximum_chain_plus_a_full_band_is_accepted(client, tmp_path):
    """The real cap, exercised at the real maximum: 24 clips of 481 frames plus a
    136-frame band. Schema validation only (no job) — rendering 11473 frames
    through the mock would take minutes and prove nothing extra."""
    from api.models import GenerateChainRequest

    req = GenerateChainRequest(
        prompt="x",
        clips=[{"num_frames": 481} for _ in range(24)],
        overlap_frames=2,
        end_source={"video_id": "v", "context_frames": 136},
    )
    layout = chain_math.compute_chain_layout(
        [c.num_frames for c in req.clips], req.frame_rate, kv=2, end_context_px=136
    )
    from api.models import MAX_CHAIN_TOTAL_PIXEL_FRAMES

    assert layout.clips_total_px <= MAX_CHAIN_TOTAL_PIXEL_FRAMES
    # ...and in ``reverse`` mode the band is inside those clips, so the DELIVERED
    # length is under the cap too (the retired geometry's was 136 frames longer).
    assert layout.total_px == layout.clips_total_px


# --------------------------------------- (g3) one source of truth for the band


def test_reverse_mode_agrees_across_layout_mock_and_metadata(client, tmp_path):
    """The multi-clip band is described in exactly ONE place (chain_math). Pinned
    across all three layers a client can observe it through: the layout object,
    the job's metadata.json, and the mock runner's output length. If the engine or
    the mock ever re-derives the arithmetic, this test is what breaks.

    ALSO THE NEW/OLD DISCRIMINATOR. The delivered length is the CLIPS' OWN TOTAL;
    the retired ``internal_segment`` geometry delivered that plus the 72-frame
    band, so the frame count alone says which one ran."""
    src = _make_source_mp4(tmp_path / "src.mp4", n_frames=120, fps=24.0)
    vid = _upload_video(client, src)
    r = _run_chain(
        client, [{"num_frames": 169}, {"num_frames": 169}],
        end_source={"video_id": vid, "context_frames": 72},
    )
    job_id = _completed(client, r)

    layout = chain_math.compute_chain_layout([169, 169], 24.0, kv=2, end_context_px=72)
    plain = chain_math.compute_chain_layout([169, 169], 24.0, kv=2)
    meta = _metadata(client, job_id)
    es = meta["end_source"]

    # layer 1: the layout itself — nothing is appended, so the geometry is the
    # end-source-less chain's own, and the schedule runs backwards.
    assert layout.end_source_mode == "reverse"
    assert layout.seg_frames == layout.clip_frames == [169, 169]
    assert layout.end_segment_latent == layout.end_segment_px == 0
    assert layout.total_px == plain.total_px == layout.clips_total_px
    assert layout.seg_generation_order == [1, 0]
    assert layout.seg_head_source == [None, None]
    assert layout.seg_tail_source == [1, None]
    # layer 2: the metadata (ChainLayout.to_dict, carried through the runner)
    assert es["mode"] == "reverse"
    assert es["generation_order"] == [1, 0]
    assert es["end_segment_latent"] == 0 and es["end_segment_px"] == 0
    assert es["clips_total_px"] == layout.total_px
    assert meta["chain"]["num_clips"] == 2
    # layer 3: the delivered mp4's own length — the clips' total, NOT + 72.
    ctx = client.app_context
    out = ctx.config.output_dir / job_id / "output.mp4"
    assert video_io.frame_count(out) == layout.total_px
    assert video_io.frame_count(out) != layout.total_px + 72


def test_reverse_mode_accepts_overlap_one(client, tmp_path):
    """K_v == 1 is the ``reverse`` mode's INTENDED のり代 (one shared latent per
    seam) and is accepted, while the same request on ONE clip still gets the
    ``kv >= 2`` rejection. The mode's exemption and the in-window rule are pinned
    together so neither can drift into the other."""
    src = _make_source_mp4(tmp_path / "src.mp4", n_frames=40, fps=24.0)
    vid = _upload_video(client, src)
    end_source = {"video_id": vid, "context_frames": 8}

    r = _run_chain(
        client, [{"num_frames": 169}] * 3, overlap_frames=1, end_source=end_source,
    )
    job_id = _completed(client, r)
    meta = _metadata(client, job_id)
    layout = chain_math.compute_chain_layout([169] * 3, 24.0, kv=1, end_context_px=8)
    assert meta["end_source"]["mode"] == "reverse"
    assert meta["end_source"]["generation_order"] == [2, 1, 0]
    assert layout.total_px == 505
    ctx = client.app_context
    out = ctx.config.output_dir / job_id / "output.mp4"
    assert video_io.frame_count(out) == 505

    r = _run_chain(
        client, [{"num_frames": 169}], overlap_frames=1, end_source=end_source,
    )
    assert r.status_code == 422
    assert ">= 2" in r.text


# ----------------------------------------------------------- (i) strength (batch 1)


def test_end_source_strength_defaults_to_1_0(client, tmp_path):
    """strength omitted -> the published default (1.0), echoed in the request
    AND in the mock's runtime end_source block (contract passthrough, not an
    observed effect -- the mock has no latents to soften)."""
    src = _make_source_mp4(tmp_path / "src.mp4", n_frames=40, fps=24.0)
    vid = _upload_video(client, src)
    r = _run_chain(
        client, [{"num_frames": 49}],
        end_source={"video_id": vid, "context_frames": 24},
    )
    job_id = _completed(client, r)
    meta = _metadata(client, job_id)
    assert meta["request"]["end_source"]["strength"] == 1.0
    assert meta["end_source"]["strength"] == 1.0


def test_end_source_strength_explicit_value_round_trips(client, tmp_path):
    """An explicit strength rides the same block as context_frames / video_id and
    round-trips unchanged through both the request echo and the mock's runtime
    end_source block."""
    src = _make_source_mp4(tmp_path / "src.mp4", n_frames=40, fps=24.0)
    vid = _upload_video(client, src)
    r = _run_chain(
        client, [{"num_frames": 49}],
        end_source={"video_id": vid, "context_frames": 24, "strength": 0.4},
    )
    job_id = _completed(client, r)
    meta = _metadata(client, job_id)
    assert meta["request"]["end_source"]["strength"] == 0.4
    assert meta["end_source"]["strength"] == 0.4


def test_end_source_strength_out_of_range_422(client):
    r = _run_chain(
        client, [{"num_frames": 49}],
        end_source={"video_id": "v", "context_frames": 24, "strength": 1.5},
    )
    assert r.status_code == 422


# --------------------------------------------------------- (h) the cut contract


def test_end_source_video_is_cut_from_the_front_with_the_primer(client, tmp_path, monkeypatch):
    """The app cuts the material's FIRST context_frames + 1 frames — start 0.0
    (an over-long upload keeps its head, mirroring the start source) and the
    extra primer frame the causal VAE consumes."""
    src = _make_source_mp4(tmp_path / "src.mp4", n_frames=60, fps=24.0)
    vid = _upload_video(client, src)

    calls: list[tuple] = []
    real_cut = video_io.cut_window_mp4

    def _spy(src_path, out_path, window_start_sec, num_frames, fps):
        calls.append((src_path, out_path, window_start_sec, num_frames, fps))
        return real_cut(src_path, out_path, window_start_sec, num_frames, fps)

    monkeypatch.setattr(video_io, "cut_window_mp4", _spy)

    r = _run_chain(
        client, [{"num_frames": 49}],
        end_source={"video_id": vid, "context_frames": 24},
    )
    job_id = _completed(client, r)

    assert len(calls) == 1
    src_path, out_path, start_sec, num_frames, fps = calls[0]
    assert start_sec == 0.0
    assert num_frames == 25  # context_frames + 1
    assert fps == 24.0
    assert out_path.name == "_end_source.mp4"
    assert out_path.parent == client.app_context.config.output_dir / job_id
