"""Spark-H3's glue in the MiniMax H3 engine (``engine_h3/spark_h3.py``), stage 0 of the plan
"Spark-H3 を製品に組み込む計画".

Runs in ``.venv-engine-h3`` through the engine-venv runner, without a GPU and without the
``spark_h3_kitchen`` wheel (the kernel is faked)::

    CUDA_VISIBLE_DEVICES=-1 .venv-engine-h3/Scripts/python.exe scripts/run_h3_pytest.py tests/test_h3_spark_engine.py

What is pinned:
  * importing the glue needs neither the wheel nor the copied modules; importing the copied
    modules leaves ``sys.path`` alone and creates no top-level ``h3_sparse_attention`` /
    ``comfy`` module;
  * the layout (video rows first, the inverse, the grid, the refusals);
  * ``for_evaluation``: the dense steps, the ``MIN_TOKENS`` boundary (12287 / 12288);
  * the settings of ComfyUI's ``_RunState.create`` (node defaults, ``ablation_mode="full"``,
    ``direct_output`` True) and the ratio reaching the kernel unrounded;
  * the call through the copied ``comfy_kitchen_spark_attention`` (the planner and the kernel
    faked), in both output layouts;
  * ``probe``: each missing piece gives a reason, never an exception; cached;
  * ``close``.
"""

from __future__ import annotations

import math
import subprocess
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from engine_h3 import spark_h3 as SP  # noqa: E402
from engine_h3.dit import H3SeqLayout  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
_REAL_IMPORT_VENDORED = SP._import_vendored
_REAL_IMPORT_KERNEL = SP._import_kernel


def _layout(*, text=5, cond=0, audio=4, grid=(2, 3, 4)):
    return H3SeqLayout.make(text_rows=text, cond_rows=cond, audio_rows=audio, video_rows=math.prod(grid))


def _clip(layout=None, grid=(2, 3, 4), *, steps=4, ratio=0.2, full=1, coords=None):
    layout = _layout(grid=grid) if layout is None else layout
    return SP.SparkClip(layout, grid, steps=steps, topk_ratio=ratio, full_steps=full, device="cpu", coords=coords)


def _long_layout(total: int, grid=(1, 64, 64)):
    """A layout of exactly ``total`` rows whose video (``grid``) is the suffix."""
    video = math.prod(grid)
    return H3SeqLayout.make(text_rows=total - video - 8, cond_rows=0, audio_rows=8, video_rows=video)


# --------------------------------------------------------------------------- #
# imports
# --------------------------------------------------------------------------- #

def _run(code: str) -> str:
    out = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(ROOT)], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip().splitlines()[-1]


def test_importing_the_glue_needs_neither_the_wheel_nor_the_copied_modules():
    code = ("import sys; sys.path.insert(0, sys.argv[1]); "
            "sys.modules['spark_h3_kitchen'] = None; "          # any import of the wheel would fail
            "import engine_h3.spark_h3; "
            "print(sorted(m for m in sys.modules if m.startswith('engine_h3.third_party')))")
    assert _run(code) == "[]"


def test_copied_modules_load_inside_the_package_without_touching_sys_path():
    code = ("import sys; sys.path.insert(0, sys.argv[1]); before = list(sys.path); "
            "from engine_h3 import spark_h3 as SP; SP._import_vendored(); "
            "top = sorted(m for m in sys.modules if m.split('.')[0] in "
            "('h3_sparse_attention', 'comfy', 'comfy_extras', 'comfyui_backend', 'comfyui_reblock_plan', "
            "'comfyui_nodes', 'spark_h3_kitchen')); "
            "pkg = sum(1 for m in sys.modules if m.startswith('engine_h3.third_party.spark_h3.')); "
            "print(sys.path == before, top, pkg >= len(SP.VENDORED_MODULES))")
    assert _run(code) == "True [] True"


# --------------------------------------------------------------------------- #
# the layout
# --------------------------------------------------------------------------- #

def test_layout_puts_the_video_rows_first_and_inverts():
    layout = _layout()                       # text 0:5, audio 5:9, video 9:33
    coords = torch.arange(33 * 3, dtype=torch.float64).view(33, 3)
    clip = _clip(layout, coords=coords)
    want = list(range(9, 33)) + list(range(0, 9))
    assert clip.permutation.tolist() == want and clip.permutation.dtype == torch.long
    assert torch.equal(clip.inverse[clip.permutation], torch.arange(33))
    assert torch.equal(clip.permutation[clip.inverse], torch.arange(33))
    lay = clip.layout
    assert (lay.grid, lay.video_tokens, lay.sequence_length, lay.video_start) == ((2, 3, 4), 24, 33, 9)
    assert lay.permutation is clip.permutation and lay.inverse_permutation is clip.inverse
    assert torch.equal(lay.video_positions, coords[9:33])
    assert _clip(layout).layout.video_positions is None
    # with references and key frames before the video, still [video | everything before it]
    ref = H3SeqLayout.make(text_rows=3, cond_rows=2, audio_rows=2, video_rows=24, ref_video_rows=4, ref_audio_rows=2)
    assert _clip(ref).permutation.tolist() == list(range(13, 37)) + list(range(0, 13))


def test_layout_refusals():
    not_suffix = H3SeqLayout(text=slice(0, 5), cond=slice(5, 5), audio=slice(29, 33), video=slice(5, 29), total=33)
    with pytest.raises(RuntimeError, match="映像の行が系列の末尾にありません（映像 5:29、系列 33 行）"):
        _clip(not_suffix)
    with pytest.raises(RuntimeError, match=r"格子 \[2, 3, 5\] の積が映像の行数 24 と合いません"):
        SP.SparkClip(_layout(), (2, 3, 5), steps=4, topk_ratio=0.2, full_steps=1, device="cpu")
    with pytest.raises(RuntimeError, match="格子"):
        SP.SparkClip(_layout(), (24,), steps=4, topk_ratio=0.2, full_steps=1, device="cpu")


# --------------------------------------------------------------------------- #
# for_evaluation
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("steps, full, sparse", [(4, 1, [1, 2, 3]), (4, 2, [2, 3]), (4, 0, [0, 1, 2, 3]),
                                                 (20, 4, list(range(4, 20))), (4, 4, []), (2, 10, [])])
def test_for_evaluation_dense_steps(steps, full, sparse):
    clip = _clip(_long_layout(SP.MIN_TOKENS), (1, 64, 64), steps=steps, full=full)
    got = []
    for i in range(steps):
        r = clip.for_evaluation(i)
        assert r is None or r is clip
        if r is not None:
            assert clip.controller.evaluation_index == i
            got.append(i)
    assert got == sparse and clip.sparse_evaluations == len(sparse)
    rec = clip.record()
    assert rec["reason"] == (None if sparse else SP.FALLBACK_FEW_STEPS)


def test_for_evaluation_min_tokens_boundary():
    assert SP.MIN_TOKENS == 12288
    short = _clip(_long_layout(12287), (1, 64, 64), steps=4, full=1)
    assert short.sequence_tokens == 12287
    assert [short.for_evaluation(i) for i in range(4)] == [None] * 4
    assert short.controller.evaluation_index == -1
    assert short.record()["reason"] == SP.FALLBACK_SHORT_SEQUENCE
    long = _clip(_long_layout(12288), (1, 64, 64), steps=4, full=1)
    assert long.sequence_tokens == 12288
    assert [long.for_evaluation(i) is long for i in range(4)] == [False, True, True, True]
    assert long.record()["reason"] is None
    # both gates: the steps are named first
    assert _clip(_long_layout(12287), (1, 64, 64), steps=2, full=2).reason() == SP.FALLBACK_FEW_STEPS


# --------------------------------------------------------------------------- #
# the settings
# --------------------------------------------------------------------------- #

def test_settings_are_the_nodes():
    """``comfyui_nodes.py`` ``_RunState.create`` with the node's defaults (ablation_mode "full",
    reblock_layout "q_reuse_k") and ``H3_SPARK_DIRECT_OUTPUT`` at its default "1"."""
    clip = _clip(ratio=0.07, full=3)
    cfg = clip.controller.config
    assert cfg.topk_ratio == 0.07 and type(cfg.topk_ratio) is float
    assert (cfg.topk_mode, cfg.topk_blocks, cfg.video_tail_mode, cfg.global_anchor_dtype, cfg.tail_granularity,
            cfg.landmark_tree_v2_midpoint_direction_mode, cfg.landmark_tree_v2_layout_reuse) == (
        "topk_ratio", 228, "dense", "float32", "query", "legacy", "q_from_k")
    defaults = type(cfg)()
    for name in ("tau", "force_local_blocks", "landmark_tree_v2_initial_order", "landmark_tree_v2_children",
                 "landmark_tree_v2_fanout_mode", "landmark_tree_v2_landmark_mode", "landmark_tree_v2_distance",
                 "landmark_tree_v2_group_size", "rope_sol_key_ridge_epsilon"):
        assert getattr(cfg, name) == getattr(defaults, name), name
    assert cfg.tau == 1.0
    c = clip.controller
    assert (c.reblock_enabled, c.reweight_enabled, c.direct_output, c.reblock_reuse_layers,
            c.reblock_reuse_start_layer) == (True, True, True, 1, 1)
    assert (clip.topk_ratio, clip.full_steps) == (0.07, 3)
    rec = clip.record()
    assert (rec["topk_ratio"], rec["full_steps"], rec["steps"], rec["min_tokens"]) == (0.07, 3, 4, 12288)


def test_direct_output_ignores_the_environment(monkeypatch):
    monkeypatch.setenv("H3_SPARK_DIRECT_OUTPUT", "0")
    assert _clip().controller.direct_output is True


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_ratio_range_is_the_authors(bad):
    with pytest.raises(ValueError, match="topk_ratio must lie in"):
        _clip(ratio=bad)


# --------------------------------------------------------------------------- #
# the call (the planner and the kernel faked)
# --------------------------------------------------------------------------- #

@pytest.fixture
def fake_kernel(monkeypatch):
    """A fake ``spark_h3_kitchen.backends.cuda.spark_attn`` (records its arguments; returns v in
    the original row order when asked to write it there, else in the video-first order) and
    the copied backend's planner replaced by one returning no permutation."""
    calls = {"kernel": [], "plan": []}

    def spark_attn(q, k, v, **kw):
        calls["kernel"].append({"q": q, "k": k, "v": v, **kw})
        out = v.clone()
        if kw["video_start"]:
            out = out.index_select(1, calls["inverse"])
        return out

    cuda = types.ModuleType("spark_h3_kitchen.backends.cuda")
    cuda.spark_attn = spark_attn
    backends = types.ModuleType("spark_h3_kitchen.backends")
    backends.cuda = cuda
    pkg = types.ModuleType("spark_h3_kitchen")
    pkg.backends = backends
    for name, mod in (("spark_h3_kitchen", pkg), ("spark_h3_kitchen.backends", backends),
                      ("spark_h3_kitchen.backends.cuda", cuda)):
        monkeypatch.setitem(sys.modules, name, mod)
    cb = SP._backend()

    def plan(controller, q, k, layout, layer, **kw):
        calls["plan"].append({"layer": layer, "evaluation_index": controller.evaluation_index})
        return None, None, None

    monkeypatch.setattr(cb, "_comfy_reblock_permutations", plan)
    return calls


def _qkv(s, heads, seed=0):
    g = torch.Generator().manual_seed(seed)
    qkv = torch.randn(s, 3, heads, 128, generator=g).to(torch.bfloat16)
    return qkv.unbind(dim=1)                 # views, as H3Attention.forward has them


@pytest.mark.parametrize("direct", [True, False])
@pytest.mark.parametrize("heads", [56, 8])
def test_attention_through_the_copied_backend(fake_kernel, direct, heads):
    grid = (2, 4, 16)                        # 128 video rows = 2 blocks of 64
    layout = H3SeqLayout.make(text_rows=70, cond_rows=0, audio_rows=10, video_rows=128)
    clip = _clip(layout, grid, ratio=0.07, full=1)
    clip.controller.direct_output = direct
    fake_kernel["inverse"] = clip.inverse
    q, k, v = _qkv(layout.total, heads)
    assert not v.is_contiguous()
    clip.controller.evaluation_index = 2     # what for_evaluation(2) sets on a long sequence
    out = clip.attention(q, k, v, 17)
    assert out.shape == (208, heads, 128) and torch.equal(out, v)
    (call,) = fake_kernel["kernel"]
    perm = clip.permutation
    for name, t in (("q", q), ("k", k), ("v", v)):
        assert call[name].shape == (1, 208, heads, 128) and call[name].is_contiguous()
        assert torch.equal(call[name][0], t.index_select(0, perm))
    assert call["topk_ratio"] == 0.07 and call["video_tokens"] == 128 and call["tau"] == 1.0
    assert call["tail_granularity"] == "query" and call["anchor_dtype"] == torch.float32
    assert call["sink_blocks"] == [2, 4] and call["sink_q"] == [2, 4]     # ceil(208 / 64) = 4
    assert call["reweight"] is True and call["force_local_blocks"] is None
    assert call["video_start"] == (80 if direct else 0)
    assert call["query_permutation"] is None and call["key_permutation"] is None
    assert fake_kernel["plan"] == [{"layer": 17, "evaluation_index": 2}]
    assert clip.controller.output_in_original_layout is direct
    rec = clip.record()
    assert rec["kernel_calls"] == 1 and rec["backend"] == "comfy-kitchen-spark-global"


def test_kernel_errors_are_not_caught(fake_kernel):
    clip = _clip(H3SeqLayout.make(text_rows=70, cond_rows=0, audio_rows=10, video_rows=128), (2, 4, 16))
    q, k, v = _qkv(208, 8)
    with pytest.raises(RuntimeError, match="BF16"):
        clip.attention(q.float(), k.float(), v.float(), 0)


# --------------------------------------------------------------------------- #
# close
# --------------------------------------------------------------------------- #

def test_close_resets_and_lets_go(fake_kernel):
    clip = _clip(H3SeqLayout.make(text_rows=70, cond_rows=0, audio_rows=10, video_rows=128), (2, 4, 16), full=0)
    fake_kernel["inverse"] = clip.inverse
    controller = clip.controller
    clip.for_evaluation(0)
    clip.attention(*_qkv(208, 8), 0)
    before = clip.record()                   # taken before the close
    assert before["kernel_calls"] == 1 and before["backend"] == "comfy-kitchen-spark-global"
    clip.close()
    assert controller.evaluation_index == -1 and not controller.counts and controller.backend is None
    assert clip.controller is None and clip.layout is None and clip.permutation is None and clip.inverse is None
    assert not hasattr(clip, "_counts") and not hasattr(clip, "_backend_used")
    clip.close()                             # twice is fine
    with pytest.raises(RuntimeError, match="閉じたあと"):
        clip.for_evaluation(1)
    with pytest.raises(RuntimeError, match="閉じたあとのクリップの記録は取れません"):
        clip.record()


def test_close_lets_go_even_when_the_reset_raises(monkeypatch):
    clip = _clip()
    controller = clip.controller

    def reset():
        raise RuntimeError("reset failed")

    monkeypatch.setattr(controller, "reset", reset)
    with pytest.raises(RuntimeError, match="reset failed"):
        clip.close()
    assert clip.controller is None and clip.layout is None and clip.permutation is None and clip.inverse is None
    clip.close()                             # closed: nothing more to do


# --------------------------------------------------------------------------- #
# probe
# --------------------------------------------------------------------------- #

GOOD = "0.2.37+spark.h3.sm89.cu130.1"


@pytest.fixture
def probe_env(monkeypatch):
    """A process where everything is present (sm89 wheel, an 8.9 GPU, the kernel, the copied
    modules); each test takes one piece away."""
    monkeypatch.setattr(SP, "_PROBE_RESULT", SP._UNSET)
    monkeypatch.setattr(SP, "kitchen_version", lambda: GOOD)
    monkeypatch.setattr(SP.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(SP.torch.cuda, "get_device_capability", lambda *a: (8, 9))
    native = types.SimpleNamespace(spark_attn=lambda *a, **k: None)
    kernel = types.SimpleNamespace(spark_attn=lambda *a, **k: None, _C=native)
    monkeypatch.setattr(SP, "_import_kernel", lambda: kernel)
    monkeypatch.setattr(SP, "_import_vendored", lambda: None)
    return {"monkeypatch": monkeypatch, "kernel": kernel, "native": native}


def test_probe_all_present(probe_env, capsys):
    assert SP.probe() is None
    assert capsys.readouterr().err == ""


def test_probe_with_the_real_copied_modules(probe_env):
    probe_env["monkeypatch"].setattr(SP, "_import_vendored", _REAL_IMPORT_VENDORED)
    assert SP.probe() is None


def _boom(exc):
    def f(*a, **k):
        raise exc
    return f


@pytest.mark.parametrize("change, reason", [
    (lambda mp, env: mp.setattr(SP, "kitchen_version", lambda: None),
     "Spark-H3 の部品（spark-h3-kitchen）が入っていません"),
    (lambda mp, env: mp.setattr(SP, "kitchen_version", lambda: "0.2.39+spark.h3.sm89.cu130.1"),
     "Spark-H3 の部品の版が 0.2.39+spark.h3.sm89.cu130.1 です（使うのは 0.2.37）"),
    (lambda mp, env: mp.setattr(SP, "kitchen_version", lambda: "0.2.37"),
     "Spark-H3 の部品の版 0.2.37 に GPU の系統の札がありません"),
    (lambda mp, env: mp.setattr(SP.torch.cuda, "is_available", lambda: False),
     "CUDA が使えません"),
    (lambda mp, env: mp.setattr(SP.torch.cuda, "get_device_capability", lambda *a: (7, 5)),
     "GPU の系統が 7.5 です（Spark-H3 の部品があるのは 8.6・8.9・12.0）"),
    (lambda mp, env: mp.setattr(SP.torch.cuda, "get_device_capability", lambda *a: (8, 6)),
     "Spark-H3 の部品は sm89 向けで、GPU の系統は 8.6（sm86）です"),
    (lambda mp, env: mp.setattr(SP, "_import_kernel", _boom(ImportError("DLL load failed"))),
     "Spark-H3 の部品を読み込めません（ImportError('DLL load failed')）"),
    (lambda mp, env: mp.setattr(SP, "_import_kernel", _boom(OSError("WinError 126"))),
     "Spark-H3 の部品を読み込めません（OSError('WinError 126')）"),
    (lambda mp, env: mp.setattr(SP, "_import_kernel", _boom(SystemError("abi"))),
     "Spark-H3 の部品を読み込めません（SystemError('abi')）"),
    (lambda mp, env: mp.delattr(env["kernel"], "spark_attn"),
     "Spark-H3 の部品にカーネル（spark_attn）がありません"),
    (lambda mp, env: mp.delattr(env["native"], "spark_attn"),
     "Spark-H3 の部品にカーネル（spark_attn）がありません"),
    (lambda mp, env: mp.setattr(SP, "_import_vendored", _boom(ImportError("No module named 'triton'"))),
     "写した Spark-H3 のモジュールを読み込めません（ImportError(\"No module named 'triton'\")）"),
    (lambda mp, env: mp.setattr(SP, "_probe_reason", _boom(RuntimeError("x"))),
     "Spark-H3 を使えるかを確かめられません（RuntimeError('x')）"),
])
def test_probe_reasons(probe_env, capsys, change, reason):
    change(probe_env["monkeypatch"], probe_env)
    assert SP.probe() == reason
    assert reason in capsys.readouterr().err


def test_probe_with_the_wheel_absent_from_sys_modules(probe_env):
    """The real ``_import_kernel`` when the wheel cannot be imported."""
    mp = probe_env["monkeypatch"]
    mp.setattr(SP, "_import_kernel", _REAL_IMPORT_KERNEL)
    mp.setitem(sys.modules, "spark_h3_kitchen", None)
    got = SP.probe()
    assert got is not None and got.startswith("Spark-H3 の部品を読み込めません（ModuleNotFoundError(")


@pytest.mark.parametrize("cap, tag", [((8, 6), "sm86"), ((8, 9), "sm89"), ((12, 0), "sm120")])
def test_probe_each_generation(probe_env, cap, tag):
    mp = probe_env["monkeypatch"]
    mp.setattr(SP, "kitchen_version", lambda: f"0.2.37+spark.h3.{tag}.cu130.1")
    mp.setattr(SP.torch.cuda, "get_device_capability", lambda *a: cap)
    assert SP.probe() is None


def test_probe_is_cached(probe_env):
    mp = probe_env["monkeypatch"]
    assert SP.probe() is None
    mp.setattr(SP, "_probe_reason", _boom(AssertionError("probed twice")))
    assert SP.probe() is None
    mp.setattr(SP, "_PROBE_RESULT", SP._UNSET)
    mp.setattr(SP, "_probe_reason", lambda: "no")
    assert SP.probe() == "no" and SP.probe() == "no"


def test_kitchen_version_reads_the_metadata(monkeypatch):
    seen = []

    def version(name):
        seen.append(name)
        raise SP.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(SP.metadata, "version", version)
    assert SP.kitchen_version() is None and seen == ["spark-h3-kitchen"]
    monkeypatch.setattr(SP.metadata, "version", lambda name: GOOD)
    assert SP.kitchen_version() == GOOD
