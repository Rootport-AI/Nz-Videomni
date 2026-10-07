"""The LTX 2.5 AlphaGen one-stage driver (台帳 §1-83).

``engine25/alphagen25.py`` drives the official blocks directly so that stage 1
runs at the FULL canvas size and its latent is decoded as it is. What the
real-GPU gate measures is the matte; what is checkable HERE, without a GPU and
without building a single model, is the shape of the procedure:

1. stage 1 is called exactly ONCE, at the full width/height, announced as
   ``STAGE_1_DENOISE`` / ``21_stage1_denoise``;
2. the reference is read at the FULL size with factor 1
   (``_encode_reference_conditionings(full_resolution=True)``), and a
   reference that yields no frames is an error, not the "generate without one"
   fallback;
3. the encode is ``crf=0`` and ``audio=None``; the upsampler and the audio
   decoder are never touched;
4. a missing reference is an :class:`AlphaGenError`;
5. the result is a plain :class:`GenerationResult`;
6. the shared helper's DEFAULTS still read the reference at half size and still
   fall back to "no reference" -- the in/outpainting behaviour is unchanged;
7. the module imports without building anything.

Run with ``.venv-engine-ltx25`` and ``--noconftest`` (the app conftest builds a
FastAPI app that venv does not have)::

    .venv-engine-ltx25/Scripts/python.exe \\
        Docs/Outputs-archive/start-end-bridge-2026-09-07/implA_engine_runner/run_ltx25_pytest.py \\
        tests/test_ltx25_alphagen.py
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("ltx_core")

import torch  # noqa: E402

from engine25 import alphagen25, outpaint25  # noqa: E402
from engine25.alphagen25 import AlphaGenError, run_alpha_gen  # noqa: E402
from engine25.pipeline25 import STAGE_1_DENOISE, GenerationResult  # noqa: E402

CPU = torch.device("cpu")
W, H, F = 384, 256, 9


class _Vram:
    def __init__(self) -> None:
        self.phases: dict[str, dict] = {}
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1

    def record(self, name: str, seconds: float) -> None:
        self.phases[name] = {"seconds": seconds}


class _Stage:
    def __init__(self, log: list) -> None:
        self.log = log
        self.calls: list[dict] = []
        self.announced: list[tuple] = []
        self.progress = None

    def set_loras(self, entries) -> None:
        self.log.append(("set_loras", list(entries)))

    def begin_job(self) -> None:
        self.log.append(("begin_job",))

    def announce(self, stage_name=None, *, vram_phase=None, **_kw) -> None:
        self.announced.append((stage_name, vram_phase))

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        latent = torch.zeros(1, 4, 2, 2, 2)
        return SimpleNamespace(latent=latent), SimpleNamespace(latent=torch.zeros(1, 2))


class _Forbidden:
    """A block the one-stage driver must never touch."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, *_a, **_k):
        raise AssertionError(f"{self.name} must not be called by the one-stage driver")

    def __getattr__(self, attr):
        raise AssertionError(f"{self.name}.{attr} must not be touched by the one-stage driver")


class _VideoDecoder:
    checkpoint_path = "vae.safetensors"
    diffvae_optimization = None

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def __call__(self, latent, tiling, generator):
        self.calls.append((latent, tiling, generator))
        return iter([torch.zeros(F, H, W, 3)])


def _image_conditioner(fn):
    return fn("the-video-encoder")


class _Encoder:
    def __init__(self, log: list) -> None:
        self.log = log
        self.progress = None

    def __call__(self, prompts):
        self.log.append(("prompt", list(prompts)))
        return [SimpleNamespace(video_encoding=torch.zeros(1, 1), audio_encoding=torch.zeros(1, 1))]


class _Pipe:
    """Everything ``run_alpha_gen`` reads off an ``Ltx25Pipeline``."""

    dtype = torch.bfloat16

    def __init__(self) -> None:
        self.log: list = []
        self.stage = _Stage(self.log)
        self.prompt_encoder = _Encoder(self.log)
        self.video_decoder = _VideoDecoder()
        self.pipeline = SimpleNamespace(
            image_conditioner=_image_conditioner,
            video_decoder=self.video_decoder,
            upsampler=_Forbidden("upsampler"),
            audio_decoder=_Forbidden("audio_decoder"),
        )
        self.device = CPU
        self.vram = _Vram()
        self.decode_progress_calls: list[int] = []

    def _with_decode_progress(self, video, chunks):
        self.decode_progress_calls.append(chunks)
        return video


@pytest.fixture()
def rig(monkeypatch, tmp_path):
    pipe = _Pipe()
    seen: dict = {"encode": [], "load": [], "cond": []}

    def fake_encode_video(**kwargs):
        frames = list(kwargs["video"])
        seen["encode"].append({**kwargs, "frames": frames})
        Path(kwargs["output_path"]).write_bytes(b"\x00" * 64)

    def fake_load(path, *, height, width, frame_cap, device):
        seen["load"].append({"path": path, "height": height, "width": width, "frame_cap": frame_cap})
        if seen.get("empty"):
            return None
        return torch.zeros(1, 3, frame_cap, height, width)

    def fake_cond(pixels, *, scale, **kwargs):
        seen["cond"].append({"scale": scale, "shape": tuple(pixels.shape), **kwargs})
        return ["reference-conditioning"]

    monkeypatch.setattr(alphagen25, "encode_video", fake_encode_video)
    monkeypatch.setattr(alphagen25, "resolve_reference_downscale_factor", lambda loras, ref: 1)
    monkeypatch.setattr(alphagen25, "ensure_tiling_config", lambda *a, **k: "TILING")
    monkeypatch.setattr(alphagen25, "tiling_scale_factors_for_vae", lambda *a, **k: None)
    monkeypatch.setattr(alphagen25, "get_video_chunks_number", lambda frames, tiling: 3)
    monkeypatch.setattr(alphagen25, "cleanup_memory", lambda: None)
    monkeypatch.setattr(outpaint25, "cleanup_memory", lambda: None)
    monkeypatch.setattr(outpaint25, "load_reference_pixels_cpu", fake_load)
    monkeypatch.setattr(outpaint25, "reference_conditioning_from_pixels", fake_cond)
    return pipe, seen, tmp_path


def _run(pipe, tmp_path, **over):
    kwargs = dict(
        prompt=" ",
        width=W,
        height=H,
        num_frames=F,
        frame_rate=24.0,
        seed=7,
        output_path=str(tmp_path / "output.mp4"),
        ic_loras=[("alpha-gen.safetensors", 1.0)],
        ic_reference=(str(tmp_path / "alpha_reference.mp4"), 1.0),
        ic_attention_strength=1.0,
    )
    kwargs.update(over)
    return run_alpha_gen(pipe, **kwargs)


# ── 1. one stage, full size ─────────────────────────────────────────────────
def test_stage_one_runs_once_at_the_full_size(rig):
    pipe, _seen, tmp_path = rig
    _run(pipe, tmp_path)
    assert len(pipe.stage.calls) == 1
    call = pipe.stage.calls[0]
    assert (call["width"], call["height"], call["frames"]) == (W, H, F)
    assert call["fps"] == 24.0
    assert call["video"].conditionings == ["reference-conditioning"]
    assert call["video"].initial_latent is None
    assert call["audio"].conditionings == [] and call["audio"].initial_latent is None
    # The distilled schedule, float32, and the ancestral stage-1 sampler.
    assert call["sigmas"].dtype == torch.float32
    assert "stepper" in call and "loop" in call


def test_stage_one_is_announced_as_stage_1_denoise(rig):
    pipe, _seen, tmp_path = rig
    _run(pipe, tmp_path)
    assert pipe.stage.announced == [(STAGE_1_DENOISE, "21_stage1_denoise")]


def test_set_loras_comes_before_begin_job(rig):
    pipe, _seen, tmp_path = rig
    _run(pipe, tmp_path)
    names = [entry[0] for entry in pipe.log]
    assert names[:3] == ["set_loras", "begin_job", "prompt"]
    assert pipe.log[0][1] == [("alpha-gen.safetensors", 1.0)]
    assert pipe.log[2][1] == [" "]


# ── 2. the reference at full size, factor 1 ─────────────────────────────────
def test_the_reference_is_read_at_the_full_size_with_factor_1(rig):
    pipe, seen, tmp_path = rig
    _run(pipe, tmp_path)
    assert len(seen["load"]) == 1
    assert (seen["load"][0]["width"], seen["load"][0]["height"]) == (W, H)
    assert seen["load"][0]["frame_cap"] == F
    assert seen["cond"][0]["scale"] == 1
    assert seen["cond"][0]["video_encoder"] == "the-video-encoder"
    assert seen["cond"][0]["tiling_config"] == "TILING"


def test_a_reference_that_yields_no_frames_is_an_error(rig):
    pipe, seen, tmp_path = rig
    seen["empty"] = True
    with pytest.raises(ValueError, match="yielded no frames"):
        _run(pipe, tmp_path)
    assert pipe.stage.calls == []
    assert seen["encode"] == []


def test_a_missing_reference_is_an_alpha_gen_error(rig):
    pipe, seen, tmp_path = rig
    with pytest.raises(AlphaGenError, match="requires a reference video"):
        _run(pipe, tmp_path, ic_reference=None)
    assert pipe.stage.calls == [] and seen["load"] == []


# ── 3. the encode ───────────────────────────────────────────────────────────
def test_the_encode_is_crf_0_and_video_only(rig):
    pipe, seen, tmp_path = rig
    _run(pipe, tmp_path)
    assert len(seen["encode"]) == 1
    enc = seen["encode"][0]
    assert enc["crf"] == 0
    assert enc["audio"] is None
    assert enc["fps"] == 24
    assert enc["video_chunks_number"] == 3
    assert enc["output_path"] == str(tmp_path / "output.mp4")
    # The decoder was fed the stage-1 latent through the progress wrapper.
    assert len(pipe.video_decoder.calls) == 1
    assert pipe.video_decoder.calls[0][1] == "TILING"
    assert pipe.decode_progress_calls == [3]


def test_the_phases_record_one_stage(rig):
    pipe, _seen, tmp_path = rig
    result = _run(pipe, tmp_path)
    assert "30_decode_encode" in result.phases and "40_job_end" in result.phases
    assert not any(name.startswith("22_") for name in result.phases)
    assert pipe.vram.resets == 1


# ── 5. the result ───────────────────────────────────────────────────────────
def test_the_result_is_a_plain_generation_result(rig):
    pipe, _seen, tmp_path = rig
    result = _run(pipe, tmp_path)
    assert type(result) is GenerationResult
    assert result.output_path == str(tmp_path / "output.mp4")
    assert (result.width, result.height, result.num_frames) == (W, H, F)
    assert result.seed == 7 and result.encode_fps == 24 and result.frame_rate == 24.0
    assert result.num_images == 0
    assert result.size_bytes == 64
    assert result.video_chunks == 3
    assert result.tiling == repr("TILING")


def test_an_empty_output_is_an_error(rig, monkeypatch):
    pipe, _seen, tmp_path = rig
    monkeypatch.setattr(alphagen25, "encode_video", lambda **k: list(k["video"]))
    with pytest.raises(AlphaGenError, match="no/empty output"):
        _run(pipe, tmp_path)


def test_a_geometry_off_the_grid_is_rejected(rig):
    pipe, _seen, tmp_path = rig
    with pytest.raises(Exception, match="multiple of"):
        _run(pipe, tmp_path, width=W + 32)
    assert pipe.stage.calls == []


# ── 6. the shared helper's defaults are unchanged ───────────────────────────
def _encode_ref(**over):
    kwargs = dict(
        image_conditioner=_image_conditioner,
        ic_reference=("ref.mp4", 1.0),
        reference_factor=1,
        height=H,
        width=W,
        num_frames=F,
        tiling_config="TILING",
        ic_attention_strength=1.0,
        device=CPU,
        vram=_Vram(),
    )
    kwargs.update(over)
    return outpaint25._encode_reference_conditionings(**kwargs)


def test_the_helper_still_reads_the_reference_at_half_size_by_default(rig):
    _pipe, seen, _tmp = rig
    conds, frames = _encode_ref()
    assert conds == ["reference-conditioning"] and frames == F
    assert (seen["load"][0]["width"], seen["load"][0]["height"]) == (W // 2, H // 2)


def test_the_helper_still_falls_back_to_no_reference_by_default(rig):
    _pipe, seen, _tmp = rig
    seen["empty"] = True
    assert _encode_ref() == ([], 0)
    assert seen["cond"] == []


def test_the_helper_with_no_reference_is_a_no_op_even_when_frames_are_required(rig):
    _pipe, seen, _tmp = rig
    assert _encode_ref(ic_reference=None, require_frames=True) == ([], 0)
    assert seen["load"] == []


# ── 7. import ───────────────────────────────────────────────────────────────
def test_the_module_imports_without_building_anything() -> None:
    import importlib

    module = importlib.import_module("engine25.alphagen25")
    assert callable(module.run_alpha_gen)
    assert issubclass(module.AlphaGenError, RuntimeError)
    assert module.ALPHA_GEN_CRF == 0
