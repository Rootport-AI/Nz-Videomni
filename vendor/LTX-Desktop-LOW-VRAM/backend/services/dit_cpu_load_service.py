"""DiT (transformer) CPU-resident build — removes the load-time GPU spike.

Background / 背景:
    block-swap は denoise 中こそ 8 ブロックだけ GPU に置くが、transformer ロードの
    一瞬だけ凍結 ``ModelLedger.transformer()`` が全 48 ブロックを GPU に materialize
    してから CPU へ退避する（max_alloc ~16.9GB の一過性スパイク）。本サービスは
    transformer を最初から CPU RAM 上に構築し、**非ブロック部分だけ** GPU へ移すことで
    このスパイクを消す。denoise 時の CPU<->GPU ブロック・ストリーミングは
    ``BlockSwapService`` がそのまま担い、演算は従来どおり GPU・出力はバイト同一。

Design / 設計:
    - CPU ビルド強制：凍結 ``ModelLedger.transformer()`` が読む ``ledger.device`` を
      一瞬 CPU にすり替える。``_target_device()`` も trailing ``.to(self.device)`` も
      CPU に解決され、GPU への全モデル materialization（=スパイク）が起きない。
    - ブロック同定は ``BlockSwapService._get_blocks()`` を**単一ソース**として再利用し、
      ストリーミング・フックが見るブロック集合と食い違わないようにする。
    - 非ブロックの leaf は per-module ``recurse=False`` 走査で移動する。これは
      ``LTXModel`` のモジュール直付け param（``scale_shift_table`` /
      ``audio_scale_shift_table``）も確実に拾い、かつブロック部分木を巻き込まない。
    - GGML buffer は ``tensor.to`` 経由で移動する＝``GGMLQuantizedTensor.to()`` の
      override がメタ（``_ggml_type`` / ``_float_shape``）と subclass を保持する。
"""

import logging

import torch


class DitCpuLoadService:
    """transformer を GPU スパイクなしに CPU 構築し、非ブロック部分だけ GPU へ移す。

    denoise 時の block-swap ストリーミングは BlockSwapService がそのまま担う。
    """

    def __init__(self, compute_device, block_swap_service):
        self.device = compute_device
        self.block_swap = block_swap_service

    def build_cpu_resident(self, ledger, build_fn):
        # Force CPU build: temporarily point ledger.device at CPU so the frozen
        # ModelLedger.transformer() (_target_device() AND trailing .to(self.device))
        # both resolve to CPU -> NO full-model GPU materialization (no spike).
        saved = ledger.device
        try:
            ledger.device = torch.device("cpu")
            t = build_fn()
        finally:
            ledger.device = saved
        blocks = self.block_swap._get_blocks(t)
        if not blocks or self.block_swap.blocks_on_gpu >= len(blocks):
            # Guard: if swapping is effectively disabled, blocks must be on GPU
            # (no streaming hook would be installed) -> move everything to GPU.
            self._move_all_to_gpu(t)
        else:
            self._move_non_block_tensors_to_gpu(t, blocks)
        return t

    def _move_all_to_gpu(self, transformer):
        transformer.to(self.device)

    def _move_non_block_tensors_to_gpu(self, transformer, blocks):
        device = self.device
        block_descendant_ids = set()
        for b in blocks:
            for sub in b.modules():
                block_descendant_ids.add(id(sub))
        moved = 0
        for module in transformer.modules():
            if id(module) in block_descendant_ids:
                continue
            for name, p in list(module._parameters.items()):
                if p is not None and p.device.type == "cpu":
                    p.data = p.data.to(device)
                    moved += 1
            for name, buf in list(module._buffers.items()):
                if buf is not None and getattr(buf, "device", None) is not None and buf.device.type == "cpu":
                    module._buffers[name] = buf.to(device)
                    moved += 1
        logging.getLogger(__name__).info(
            "dit_cpu_load: moved %d non-block leaf tensors to %s; %d blocks remain on CPU",
            moved, device, len(blocks),
        )
