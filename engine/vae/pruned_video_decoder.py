"""PrunaVAED: the pruned LTX-2.3 video VAE **decoder** (§3-50).

PrunaVAED (Pruna AI) is a channel-pruned + distilled drop-in replacement for the
video VAE decoder only — the encoder is byte-identical to the stock one and is
never touched here. Two things make it impossible to express with the wheel's
own vocabulary, and this module exists for exactly those two:

1. **Two projection resnets** (512->384 and 384->256) sit between the stock
   blocks. ``_make_decoder_block``'s ``res_x_y`` derives its output width by
   INTEGER DIVISION (``in_channels // multiplier``, video_vae.py:505), and
   512->384 needs 4/3. So the block list cannot be written as ``decoder_blocks``
   at all — the ``up_blocks`` ModuleList has to be built explicitly.
2. **``norm3`` is a different operator** in the implementation the weights were
   trained against. See ``ChannelLayerNorm3d`` below; this is the one place in
   the whole feature where getting it wrong is SILENT.

Everything else is inherited: ``forward`` / ``tiled_decode`` / ``_prepare_tiles``
are unchanged, and the tile geometry does not depend on channel widths at all
(``video_downscale_factors`` is a hardcoded constant, video_vae.py:601-605), so
a pruned job uses exactly the same tiling config as a default job.

The wheel is not modified in any way (the ltx-core wheel inside ``.venv-engine``
is reinstallable and must stay pristine); this module only composes the wheel's
public building blocks.

Authority for every number here: ``Docs/PRUNAVAED_WORKORDER.md`` §4.1 (the flat
key layout) and §4.2 (this module's design).
"""

from __future__ import annotations

from typing import Any, Final

import torch
from torch import nn

from ltx_core.model.video_vae.enums import NormLayerType, PaddingModeType
from ltx_core.model.video_vae.resnet import ResnetBlock3D
from ltx_core.model.video_vae.video_vae import VideoDecoder, _make_decoder_block

# 使い捨ての「幅合わせ専用」ブロック列。1024 // 16 = 64 で、枝刈りデコーダの
# 最終幅と一致する。``VideoDecoder.__init__`` はブロック列を (1) up_blocks の
# 構築と (2) ループ後の feature_channels（= conv_norm_out と conv_out の入力幅）
# にしか使わない（video_vae.py:620-668）。up_blocks は直後に丸ごと差し替えるので、
# この骨組みが必要なのは **conv_out を 64 入力で作らせるためだけ** である。
# Configurator は ``torch.device("meta")`` の下で呼ばれる
# （single_gpu_model_builder.py:60-62）ため、捨てる畳み込み1個の確保コストはゼロ。
_WIDTH_ONLY_SKELETON: Final[list[tuple[str, dict[str, Any]]]] = [
    ("compress_space", {"multiplier": 16}),
]

# 射影 resnet を表す擬似ブロック名。wheel の ``_make_decoder_block`` は
# この名前を知らない（渡せば ValueError）ので、下のループが自前で分岐する。
_PROJECTION: Final = "__projection__"

# 実行順に並べた枝刈りデコーダの up_blocks（§4.1 の平坦キー表と1対1に対応する。
# flat idx はこのタプルの添字そのもの）。第3要素はブロック通過後の出力チャンネル
# 数で、**検算値**である——wheel 側の導出（multiplier による整数除算）が変わった
# ことを黙って受け入れないために、構築のたびに突き合わせる。
_PRUNED_UP_BLOCKS: Final[tuple[tuple[str, dict[str, Any], int], ...]] = (
    ("res_x", {"num_layers": 2}, 1024),          # 0
    ("compress_all", {"multiplier": 2}, 512),    # 1  conv [4096,1024,3,3,3]
    ("res_x", {"num_layers": 2}, 512),           # 2
    (_PROJECTION, {}, 384),                      # 3  512->384（新設）
    ("compress_all", {"multiplier": 1}, 384),    # 4  conv [3072,384,3,3,3]
    ("res_x", {"num_layers": 4}, 384),           # 5
    (_PROJECTION, {}, 256),                      # 6  384->256（新設）
    ("compress_time", {"multiplier": 2}, 128),   # 7  conv [256,256,3,3,3]
    ("res_x", {"num_layers": 6}, 128),           # 8
    ("compress_space", {"multiplier": 2}, 64),   # 9  conv [256,128,3,3,3]
    ("res_x", {"num_layers": 4}, 64),            # 10
)

# 上の構成から積算した学習パラメータ数（``per_channel_statistics`` の2バッファ
# 計256要素は別勘定）。上流モデルカードの報告値と1個の違いも無く一致する数で、
# 「構造が仕様どおりに組まれた」ことの決定的な指紋になる（§2.3）。
EXPECTED_PARAMETER_COUNT: Final = 345_006_256

# 学習パラメータ・バッファを合わせた state_dict のキー本数（§2.3）。
EXPECTED_STATE_DICT_KEYS: Final = 102

# 変換後ファイルの ``__metadata__["config"]["vae"]`` に対する突き合わせ表。
# 正本はこのコードであり、ファイル側の config は「同じ値を持つが権威ではない
# 参照情報」として扱う（§4.2）。食い違えば黙って動かず例外で落とす。
_REQUIRED_CONFIG: Final[dict[str, Any]] = {
    "_class_name": "PrunaVAEDDecoder",
    "latent_channels": 128,
    "patch_size": 4,
    "norm_layer": "pixel_norm",
    "causal_decoder": False,
    "timestep_conditioning": False,
    "decoder_base_channels": 128,
}


class ChannelLayerNorm3d(nn.LayerNorm):
    """diffusers の ``norm3`` と同じ演算。**枝刈りデコーダ最大の落とし穴**。

    上流の重みは diffusers の ``LTX2VideoResnetBlock3d`` で学習されており、
    そこでの ``norm3`` は ``nn.LayerNorm(in_channels)`` を channel-last で
    掛けたもの＝**各 (バッチ, フレーム, 縦, 横) 位置ごとにチャンネル方向だけ**
    で平均と分散を取る。一方 ltx-core の ``ResnetBlock3D`` は
    ``nn.GroupNorm(num_groups=1)`` を使っており（resnet.py:91-97）、これは
    ``nn.LayerNorm([C, F, H, W])`` 相当＝チャンネルと時空間を**まとめて**
    正規化する別物である（wheel 側のコメントの「GroupNorm(1) は LayerNorm と
    等価」という主張は、チャンネルのみの LayerNorm に対しては誤り）。

    たちが悪いのは**両者の学習パラメータがどちらも形状 [C] の weight / bias**
    だという点で、取り違えても ``load_state_dict(strict=False)`` の
    "Uninitialized parameters" 警告に一切掛からない。発現するのは映像が壊れる
    形だけである（§2.5）。無改変デコーダは ``in_channels == out_channels`` の
    resnet しか持たず ``norm3`` が ``nn.Identity()`` になるため、この経路を
    踏むのは射影 resnet を持ち込む本モジュールが初めてである。

    ``nn.Module`` ではなく **``nn.LayerNorm`` を直接継承する**。こうすると学習
    パラメータが自分自身の weight / bias になり、state_dict のキーが GroupNorm
    版と同じ ``norm3.weight`` / ``norm3.bias`` のまま（1段深くならない）で済む
    ——配布ファイル側の改名も階層追加も不要になる（§4.2）。
    """

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__(num_channels, eps=eps, elementwise_affine=True, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, F, H, W) -> (B, F, H, W, C) で正規化し、元の並びへ戻す。
        return super().forward(x.movedim(1, -1)).movedim(-1, 1)


def _make_projection_resnet(
    in_channels: int,
    out_channels: int,
    convolution_dimensions: int,
    norm_layer: NormLayerType,
    norm_num_groups: int,
    spatial_padding_mode: PaddingModeType,
) -> ResnetBlock3D:
    """入出力の幅が違う resnet を1つ作り、``norm3`` だけ差し替える。

    **属性差し替えを選ぶ理由**: ``VideoDecoder.forward`` のブロック分岐は
    ``isinstance(up_block, ResnetBlock3D)`` である（video_vae.py:734）。派生
    クラスを作っても ``isinstance`` は通るが、属性差し替えなら**分岐の判定
    方法に一切依存しない**ぶん安全である。分岐の順序は ``UNetMidBlock3D`` →
    ``ResnetBlock3D`` → else（＝``DepthToSpaceUpsample``）で、射影 resnet は
    ``causal`` と ``generator`` を受け取る2番目の経路に正しく乗る。

    ``eps`` は上流 config の ``resnet_norm_eps=1e-06`` に一致させる（wheel の
    ``res_x_y`` 経路も 1e-6 なので既定と同じ）。``norm1`` / ``norm2`` は
    ``PixelNorm``＝パラメータ無しで、diffusers の ``PerChannelRMSNorm``
    （eps=1e-8）と同一式である（§2.5 の付随確認表）。
    """
    block = ResnetBlock3D(
        dims=convolution_dimensions,
        in_channels=in_channels,
        out_channels=out_channels,
        eps=1e-6,
        groups=norm_num_groups,
        norm_layer=norm_layer,
        inject_noise=False,
        timestep_conditioning=False,
        spatial_padding_mode=spatial_padding_mode,
    )
    block.norm3 = ChannelLayerNorm3d(in_channels, eps=1e-6)
    return block


class PrunedVideoDecoder(VideoDecoder):
    """``VideoDecoder`` の派生。``up_blocks`` だけを枝刈り構成へ差し替える。

    ``decoder_blocks`` に依存する学習パラメータは ``up_blocks`` と ``conv_out``
    の2つに限られる（§2.4-E で ``VideoDecoder.__init__`` を通読して確定）ので、
    基底クラスに幅合わせ用の骨組みを渡して ``conv_out`` を 64 入力で作らせ、
    ``up_blocks`` はこちらで組み直す、という最小の介入で済む。
    """

    def __init__(
        self,
        convolution_dimensions: int = 3,
        in_channels: int = 128,
        out_channels: int = 3,
        patch_size: int = 4,
        norm_layer: NormLayerType = NormLayerType.PIXEL_NORM,
        causal: bool = False,
        timestep_conditioning: bool = False,
        decoder_spatial_padding_mode: PaddingModeType = PaddingModeType.ZEROS,
        base_channels: int = 128,
    ) -> None:
        if timestep_conditioning:
            # 有効にすると基底クラスが last_time_embedder / scale_shift_table を
            # 足してしまい、配布ファイルに存在しないパラメータが増える。
            raise ValueError(
                "PrunedVideoDecoder does not support timestep_conditioning "
                "(the PrunaVAED checkpoint carries no timestep parameters)"
            )
        super().__init__(
            convolution_dimensions=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            decoder_blocks=_WIDTH_ONLY_SKELETON,
            patch_size=patch_size,
            norm_layer=norm_layer,
            causal=causal,
            timestep_conditioning=timestep_conditioning,
            decoder_spatial_padding_mode=decoder_spatial_padding_mode,
            base_channels=base_channels,
        )

        feature_channels = base_channels * 8
        blocks: list[nn.Module] = []
        for flat_idx, (block_name, block_config, expected_out) in enumerate(_PRUNED_UP_BLOCKS):
            if block_name == _PROJECTION:
                block: nn.Module = _make_projection_resnet(
                    in_channels=feature_channels,
                    out_channels=expected_out,
                    convolution_dimensions=convolution_dimensions,
                    norm_layer=norm_layer,
                    norm_num_groups=self._norm_num_groups,
                    spatial_padding_mode=decoder_spatial_padding_mode,
                )
                out_channels_here = expected_out
            else:
                # 素の段は wheel の工場をそのまま通す。無改変デコーダと同じ
                # 構築経路を共有することが、ブロックの中身が食い違わない保証。
                block, out_channels_here = _make_decoder_block(
                    block_name=block_name,
                    block_config=block_config,
                    in_channels=feature_channels,
                    convolution_dimensions=convolution_dimensions,
                    norm_layer=norm_layer,
                    timestep_conditioning=timestep_conditioning,
                    norm_num_groups=self._norm_num_groups,
                    spatial_padding_mode=decoder_spatial_padding_mode,
                )
            if out_channels_here != expected_out:
                raise ValueError(
                    f"up_blocks.{flat_idx} ({block_name}) produced "
                    f"{out_channels_here} channels, expected {expected_out} — "
                    "the wheel's width derivation no longer matches §4.1"
                )
            blocks.append(block)
            feature_channels = out_channels_here

        self.up_blocks = nn.ModuleList(blocks)

        # 骨組みが意図どおり効いたことをここで固定する（黙って形が変わる事故の
        # 防止）。``assert`` ではなく ``raise`` を使う: assert は ``python -O``
        # で消えるため、最適化起動された環境で検証が丸ごと無効化される。
        if tuple(self.conv_in.conv.weight.shape) != (1024, 128, 3, 3, 3):
            raise ValueError(
                f"conv_in shape mismatch: {tuple(self.conv_in.conv.weight.shape)}"
            )
        if tuple(self.conv_out.conv.weight.shape) != (48, 64, 3, 3, 3):
            raise ValueError(
                f"conv_out shape mismatch: {tuple(self.conv_out.conv.weight.shape)}"
            )
        if len(self.up_blocks) != len(_PRUNED_UP_BLOCKS):
            raise ValueError(
                f"up_blocks has {len(self.up_blocks)} entries, expected "
                f"{len(_PRUNED_UP_BLOCKS)}"
            )


class PrunedVideoDecoderConfigurator:
    """``ModelConfigurator`` プロトコルの唯一のメソッドを実装する固定仕様版。

    構成の正本は**このモジュールのコード**であり、ファイルの
    ``__metadata__["config"]`` は権威ではない。それでも config を読むのは、
    ltx-core のローダーが起動時に必ず ``json.loads(f.metadata()["config"])``
    を呼ぶ（sft_loader.py:58-60、ガード無し）ため、埋め込みが**必須**だから
    である（§2.4-C）。

    そこで本 Configurator は config を「素性の確認」にだけ使う: 主要キーが固定値
    と食い違ったら例外で落とす。こうすると (i) ローダーの必須要求を満たし、
    (ii) 正本はコード側の1箇所に留まり、(iii) 想定外のファイルを差されたときに
    静かに動くのではなく確実に落ちる。
    """

    @classmethod
    def from_config(cls, config: dict) -> PrunedVideoDecoder:
        vae = (config or {}).get("vae", {})
        if not isinstance(vae, dict):
            raise ValueError(
                f"PrunaVAED config: 'vae' must be an object, got {type(vae).__name__}"
            )

        mismatched = {
            key: vae.get(key, "<absent>")
            for key, want in _REQUIRED_CONFIG.items()
            if vae.get(key, "<absent>") != want
        }
        if mismatched:
            raise ValueError(
                "PrunaVAED config mismatch (this file is not a PrunaVAED decoder, "
                f"or its metadata is stale): got {mismatched}, expected "
                f"{ {k: _REQUIRED_CONFIG[k] for k in mismatched} }"
            )
        if "decoder_blocks" in vae:
            # 含めてはならない（§4.2）。ltx-core の語彙では射影 resnet を表現
            # できないので、含まれていれば標準の VideoDecoderConfigurator に
            # 食わされたとき「読めてしまい」、射影 resnet を欠いた別物が**静か
            # に**構築される。省いてあれば、そのとき確実に例外で落ちる。
            raise ValueError(
                "PrunaVAED config must NOT carry 'decoder_blocks': the pruned "
                "layout cannot be expressed in ltx-core's block vocabulary, and "
                "its presence would let the STOCK configurator build a silently "
                "wrong decoder from this same file"
            )

        return PrunedVideoDecoder(
            convolution_dimensions=3,
            in_channels=128,
            out_channels=3,
            patch_size=4,
            norm_layer=NormLayerType.PIXEL_NORM,
            causal=False,
            timestep_conditioning=False,
            decoder_spatial_padding_mode=PaddingModeType.ZEROS,
            base_channels=128,
        )
