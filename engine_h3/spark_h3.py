r"""Spark-H3: the sparse attention of a ``spark_h3`` job's DiT evaluations (the product's glue).

The computation is Spark-H3's ComfyUI backend, zechengtang/Spark-H3 tag
``comfyui-v0.2.0`` (Apache-2.0), kept unchanged in
``engine_h3/third_party/spark_h3/`` (``comfyui_backend.py``,
``comfyui_reblock_plan.py``, ``h3_sparse_attention/``; origin, license and the
SHA-256 of every file in its README.md), on the kernel of the prebuilt
``spark_h3_kitchen`` wheel the installer puts into ``.venv-engine-h3``. Nothing
of it is rewritten here.

Written here (read from ``comfyui_nodes.py`` of the same release, which imports
ComfyUI and is not copied; the experiment's ``outputs/h3-spark-spike-2026-11-26/
scripts/spk_hooks.py`` is the same code run without changing the product):

* the settings of one clip's controller: ``_RunState.create``
  (``comfyui_nodes.py:402-501``) with the node's defaults, ``ablation_mode
  "full"``, and ``direct_output = True`` (the node reads it from
  ``H3_SPARK_DIRECT_OUTPUT``, default ``"1"``; this module reads no environment
  variable). See :data:`CONFIG` / :data:`CONTROLLER`.
* the layout: the video rows first, then every row before them (the product's
  video rows are the sequence's suffix), the grid ``(T', H/32, W/32)``; the
  same result as ``_make_target_spark_layout`` (``comfyui_nodes.py:220-253``)
  on the product's rows (the experiment's check f).
* the call: the path of ComfyUI's KJNodes Low VRAM attention
  (``comfyui_nodes.py:872-924``), which receives the q, k, v made outside:
  ``index_select`` to the video-first order (this also makes them
  contiguous), ``comfy_kitchen_spark_attention``, the inverse order when the
  kernel did not write the original order.
* the gates: an evaluation is dense (None from :meth:`SparkClip.for_evaluation`)
  while ``i < min(steps, full_steps)`` (``warmup_mode="warmup_steps"``) and when
  the sequence has fewer than :data:`MIN_TOKENS` rows (the node's ``min_tokens``
  default; the product counts every row of its sequence, the negative prompt's
  rows of VSF included). A dense evaluation is the job's own sdpa / sage.
* :func:`probe`: whether this process can run it (the
  ``engine.transformer.sage_attention_service.probe_sage`` idiom).

``spark_h3_kitchen`` and the copied modules are imported only by :func:`probe`
and by :class:`SparkClip`, never when this module is imported. A kernel error
during a job is not caught here (the job fails, as the node's ``strict``
default).
"""

from __future__ import annotations

import importlib
import math
import re
import sys
from importlib import metadata

import torch

#: The node's ``min_tokens`` default: a sequence with fewer rows is computed densely.
MIN_TOKENS = 12288

#: The wheel's distribution name and the version used (the one the release's
#: installer defaults to; the RTX 30 / 50 series ZIPs also carry 0.2.39).
KITCHEN_DISTRIBUTION = "spark-h3-kitchen"
KITCHEN_VERSION = "0.2.37"

#: GPU generations Spark-H3 publishes a wheel for (the release's ``install.py``):
#: ``torch.cuda.get_device_capability()`` -> the tag in the wheel's local version.
ARCHITECTURES = {(8, 6): "sm86", (8, 9): "sm89", (12, 0): "sm120"}

#: ``ComfySparkConfig`` of one clip; ``topk_ratio`` is the job's, unrounded.
#: ``_RunState.create``: ``topk_mode="topk_ratio"``, ``topk_blocks=228``,
#: ``video_tail_mode="dense"``, ``global_anchor_dtype="float32"``,
#: ``tail_granularity="query"``, ``midpoint_direction_mode="legacy"``,
#: ``reblock_layout="q_reuse_k"`` -> ``landmark_tree_v2_layout_reuse="q_from_k"``;
#: the rest of the config is the dataclass's defaults (``tau=1.0`` among them).
CONFIG = {
    "topk_mode": "topk_ratio",
    "topk_blocks": 228,
    "video_tail_mode": "dense",
    "global_anchor_dtype": "float32",
    "tail_granularity": "query",
    "landmark_tree_v2_midpoint_direction_mode": "legacy",
    "landmark_tree_v2_layout_reuse": "q_from_k",
}

#: The controller's attributes ``_RunState.create`` sets for ``ablation_mode="full"``.
CONTROLLER = {
    "reblock_enabled": True,
    "reweight_enabled": True,
    "direct_output": True,
    "reblock_reuse_layers": 1,
    "reblock_reuse_start_layer": 1,
}

#: Why a clip ran no sparse evaluation (:meth:`SparkClip.record` ``reason``).
FALLBACK_FEW_STEPS = "few_steps"
FALLBACK_SHORT_SEQUENCE = "short_sequence"

_PACKAGE = "engine_h3.third_party.spark_h3"

#: The copied modules, by their names inside the package. ``comfyui_reblock_plan``
#: imports the ``h3_sparse_attention`` ones lazily and, when that relative import
#: fails, adds its own folder to ``sys.path`` and imports them top-level; loading
#: them here first (in :func:`probe`) keeps a job on the package names.
VENDORED_MODULES = (
    "comfyui_backend",
    "comfyui_reblock_plan",
    "h3_sparse_attention.landmark_direction",
    "h3_sparse_attention.landmark_tree_v2",
    "h3_sparse_attention.mahalanobis_kmeans",
    "h3_sparse_attention.landmark_projection",
    "h3_sparse_attention.landmark_v2_fused_node",
    "h3_sparse_attention.reblock_hierarchy",
)

_UNSET = object()
_PROBE_RESULT: object = _UNSET


def kitchen_version() -> str | None:
    """The installed ``spark-h3-kitchen`` version (from its metadata; nothing is imported)."""
    try:
        return metadata.version(KITCHEN_DISTRIBUTION)
    except metadata.PackageNotFoundError:
        return None


def _import_kernel():
    from spark_h3_kitchen.backends import cuda  # noqa: PLC0415

    return cuda


def _import_vendored() -> None:
    for name in VENDORED_MODULES:
        importlib.import_module(f"{_PACKAGE}.{name}")


def _backend():
    return importlib.import_module(f"{_PACKAGE}.comfyui_backend")


def _probe_reason() -> str | None:
    installed = kitchen_version()
    if installed is None:
        return f"Spark-H3 の部品（{KITCHEN_DISTRIBUTION}）が入っていません"
    implementation, _, local = installed.partition("+")
    if implementation != KITCHEN_VERSION:
        return f"Spark-H3 の部品の版が {installed} です（使うのは {KITCHEN_VERSION}）"
    tag = re.match(r"spark\.h3\.(sm\d+)\.", local)
    if tag is None:
        return f"Spark-H3 の部品の版 {installed} に GPU の系統の札がありません"
    if not torch.cuda.is_available():
        return "CUDA が使えません"
    capability = tuple(int(c) for c in torch.cuda.get_device_capability())
    gpu = ARCHITECTURES.get(capability)
    if gpu is None:
        return (f"GPU の系統が {capability[0]}.{capability[1]} です"
                f"（Spark-H3 の部品があるのは 8.6・8.9・12.0）")
    if gpu != tag.group(1):
        return (f"Spark-H3 の部品は {tag.group(1)} 向けで、"
                f"GPU の系統は {capability[0]}.{capability[1]}（{gpu}）です")
    try:
        cuda = _import_kernel()
    except BaseException as exc:  # noqa: BLE001 - see probe()
        return f"Spark-H3 の部品を読み込めません（{exc!r}）"
    native = getattr(getattr(cuda, "_C", None), "spark_attn", None)
    if not callable(getattr(cuda, "spark_attn", None)) or not callable(native):
        return "Spark-H3 の部品にカーネル（spark_attn）がありません"
    try:
        _import_vendored()
    except BaseException as exc:  # noqa: BLE001 - see probe()
        return f"写した Spark-H3 のモジュールを読み込めません（{exc!r}）"
    return None


def probe() -> str | None:
    """Why Spark-H3 cannot run in THIS process (None when it can). Cached; never raises.

    Checked in order: the wheel's version (:data:`KITCHEN_VERSION`) and the GPU
    tag of its local version, CUDA, the GPU's capability against that tag
    (:data:`ARCHITECTURES`, exact), the kernel's import and ``spark_attn``, the
    copied modules' import (:data:`VENDORED_MODULES`). The guard is
    ``BaseException``, as ``probe_sage``: a broken wheel can surface as
    ImportError, OSError, SystemError or a bare RuntimeError out of the DLL
    loader, and none of them may take the worker down. The reason states what
    was found; it also goes to STDERR (the worker's log) once.
    """
    global _PROBE_RESULT
    if _PROBE_RESULT is not _UNSET:
        return _PROBE_RESULT  # type: ignore[return-value]
    try:
        reason = _probe_reason()
    except BaseException as exc:  # noqa: BLE001 - see docstring
        reason = f"Spark-H3 を使えるかを確かめられません（{exc!r}）"
    if reason is not None:
        print(f"[spark-h3] Spark-H3 unavailable, 'spark_h3' requests will run without it: {reason}",
              file=sys.stderr, flush=True)
    _PROBE_RESULT = reason
    return reason


class SparkClip:
    """Spark-H3's state for the denoising of ONE clip: made after the clip's layout
    and coordinates exist; in the denoising's ``finally`` its :meth:`record` is taken
    and then it is closed by :meth:`close` (before the decode, so nothing of it stays
    on the device).

    ``layout``: the clip's ``engine_h3.dit.H3SeqLayout`` (the video rows must be the
    sequence's suffix); ``grid``: ``(T', H/32, W/32)``, whose product must be the
    video rows; ``steps``: the clip's denoising steps (one DiT evaluation each);
    ``topk_ratio`` / ``full_steps``: the job's values, as given (the ratio's range
    is ``ComfySparkConfig``'s: a ValueError out of it); ``device``: where the
    permutations live (the DiT's device); ``coords``: the clip's rotary
    coordinates [S, 3] (kept on the CPU, for the layout's ``video_positions``,
    which neither the kernel nor the planner reads) or None.
    """

    def __init__(self, layout, grid, *, steps: int, topk_ratio: float, full_steps: int,
                 device, coords=None) -> None:
        cb = _backend()
        v0, v1, total = int(layout.video.start), int(layout.video.stop), int(layout.total)
        if v1 != total:
            raise RuntimeError(
                f"Spark-H3: 映像の行が系列の末尾にありません（映像 {v0}:{v1}、系列 {total} 行）")
        video_tokens = v1 - v0
        grid = tuple(int(g) for g in grid)
        if len(grid) != 3 or math.prod(grid) != video_tokens:
            raise RuntimeError(
                f"Spark-H3: 格子 {list(grid)} の積が映像の行数 {video_tokens} と合いません")
        self.steps = int(steps)
        self.full_steps = int(full_steps)
        self.topk_ratio = topk_ratio
        self.sequence_tokens = total
        self.video_tokens = video_tokens
        self.grid = grid
        self.sparse_evaluations = 0
        controller = cb.ComfySparkController(cb.ComfySparkConfig(topk_ratio=topk_ratio, **CONFIG))
        for name, value in CONTROLLER.items():
            setattr(controller, name, value)
        self.controller = controller
        video = torch.arange(v0, total, device=device, dtype=torch.long)
        before = torch.arange(0, v0, device=device, dtype=torch.long)
        self.permutation = torch.cat((video, before))
        self.inverse = torch.empty_like(self.permutation)
        self.inverse[self.permutation] = torch.arange(total, device=device, dtype=torch.long)
        self.layout = cb.ComfyPackedLayout(
            permutation=self.permutation, inverse_permutation=self.inverse, grid=grid,
            video_tokens=video_tokens, sequence_length=total,
            video_positions=None if coords is None else coords[v0:v1], video_start=v0,
        )
        self._cb = cb

    @property
    def dense_steps(self) -> int:
        return min(self.steps, self.full_steps)

    def reason(self) -> str | None:
        """Why this clip runs no sparse evaluation, from its settings and size alone
        (:data:`FALLBACK_FEW_STEPS` before :data:`FALLBACK_SHORT_SEQUENCE`), else None."""
        if self.steps <= self.full_steps:
            return FALLBACK_FEW_STEPS
        if self.sequence_tokens < MIN_TOKENS:
            return FALLBACK_SHORT_SEQUENCE
        return None

    def for_evaluation(self, i: int) -> "SparkClip | None":
        """What evaluation ``i`` (0-based, one per step) attends with: None = dense
        (the job's sdpa / sage), else this object with the controller set to ``i``."""
        if self.controller is None:
            raise RuntimeError("Spark-H3: 閉じたあとのクリップの状態が使われました")
        i = int(i)
        if i < self.dense_steps or self.sequence_tokens < MIN_TOKENS:
            return None
        self.controller.evaluation_index = i
        self.sparse_evaluations += 1
        return self

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer: int) -> torch.Tensor:
        """q, k, v [S, H, 128] in the product's row order (H = all heads, or one head
        group of VRAM 16GB mode) -> [S, H, 128], the same order."""
        q4, k4, v4 = (t.unsqueeze(0).index_select(1, self.permutation).contiguous() for t in (q, k, v))
        out = self._cb.comfy_kitchen_spark_attention(self.controller, q4, k4, v4, self.layout, int(layer))
        if not self.controller.output_in_original_layout:
            out = out.index_select(1, self.inverse)
        return out[0]

    def record(self) -> dict:
        """The clip's facts; taken before :meth:`close` (the kernel's counts are the controller's)."""
        if self.controller is None:
            raise RuntimeError("Spark-H3: 閉じたあとのクリップの記録は取れません")
        counts = dict(self.controller.counts)
        backend = self.controller.backend
        return {
            "topk_ratio": self.topk_ratio,
            "full_steps": self.full_steps,
            "steps": self.steps,
            "min_tokens": MIN_TOKENS,
            "sequence_tokens": self.sequence_tokens,
            "video_tokens": self.video_tokens,
            "grid": list(self.grid),
            "sparse_evaluations": self.sparse_evaluations,
            "kernel_calls": int(counts.get("comfy_kitchen_calls", 0)),
            "backend": backend,
            "kitchen_version": kitchen_version(),
            "reason": None if self.sparse_evaluations else self.reason(),
        }

    def close(self) -> None:
        """``controller.reset()`` (the node's cleanup) and the permutations / layout
        let go. Safe to call twice."""
        controller = self.controller
        if controller is None:
            return
        self.controller = None
        try:
            controller.reset()
        finally:
            # let go of them even when the reset raises (the caller logs it)
            self.layout = None
            self.permutation = self.inverse = None
