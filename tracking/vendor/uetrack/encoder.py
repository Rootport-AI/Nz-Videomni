# Vendored from https://github.com/kangben258/UETrack @ fd13b0ea
# Source file: lib/models/uetrack/encoder.py
#
# NZ modifications:
#   * `Encoder_Teacher` / `build_encoder_teacher` are not carried, and neither is
#     the `fastitpn_teacher` module they wrap. The teacher exists only to distil
#     the student during training; at inference it is dead weight (it is also the
#     bulk of the 1.18 GB official checkpoint -- see scripts/trim_uetrack_checkpoint.py).
#   * `pretrained=is_main_process()` -> `pretrained=False`. That flag makes the
#     factory load ENCODER.PRETRAIN_TYPE, an ImageNet backbone used as a TRAINING
#     starting point; this port loads a finished tracker checkpoint over the top
#     of the freshly built module, so reading the backbone first would be pure
#     waste (and PRETRAIN_TYPE is '' in the Base config anyway). Dropping it also
#     drops the `lib.utils.misc` import, which pulls in torch.distributed.

from torch import nn

from . import fastitpn as fastitpn_module


class EncoderBase(nn.Module):

    def __init__(self, encoder: nn.Module, train_encoder: bool, open_layers: list, num_channels: int):
        super().__init__()
        open_blocks = open_layers[2:]
        open_items = open_layers[0:2]
        for name, parameter in encoder.named_parameters():

            if not train_encoder:
                freeze = True
                for open_block in open_blocks:
                    if open_block in name:
                        freeze = False
                if name in open_items:
                    freeze = False
                if freeze == True:
                    parameter.requires_grad_(False)  # here should allow users to specify which layers to freeze !

        self.body = encoder
        self.num_channels = num_channels

    def forward(self, template_list, search_list, template_anno_list, text_src, task_index):
        xs,feature_list = self.body(template_list, search_list, template_anno_list, text_src, task_index)
        return xs,feature_list


class Encoder(EncoderBase):
    """ViT encoder."""
    def __init__(self, name: str,
                 train_encoder: bool,
                 search_size: int,
                 template_size: int,
                 open_layers: list,
                 cfg=None):
        if "fastitpn" in name.lower():
            encoder = getattr(fastitpn_module, name)(
                pretrained=False,  # NZ: was is_main_process() -- see module header
                search_size=search_size,
                template_size=template_size,
                drop_rate=0.0,
                drop_path_rate=0.1,
                attn_drop_rate=0.0,
                init_values=0.1,
                drop_block_rate=None,
                use_mean_pooling=True,
                grad_ckpt=False,
                cls_token=cfg.MODEL.ENCODER.CLASS_TOKEN,
                pos_type=cfg.MODEL.ENCODER.POS_TYPE,
                token_type_indicate=cfg.MODEL.ENCODER.TOKEN_TYPE_INDICATE,
                pretrain_type=cfg.MODEL.ENCODER.PRETRAIN_TYPE,
                patchembed_init=cfg.MODEL.ENCODER.PATCHEMBED_INIT,
                num_experts=cfg.MODEL.ENCODER.NUM_EXPERT,
                moe_layer=cfg.MODEL.ENCODER.MOE_LAYER,
                distill_layer=cfg.TRAIN.DISTILL_LAYER_S,
            )
            num_channels = 384
        else:
            raise ValueError()
        super().__init__(encoder, train_encoder, open_layers, num_channels)


def build_encoder(cfg):
    train_encoder = (cfg.TRAIN.ENCODER_MULTIPLIER > 0) and (cfg.TRAIN.FREEZE_ENCODER == False)
    encoder = Encoder(cfg.MODEL.ENCODER.TYPE, train_encoder,
                      cfg.DATA.SEARCH.SIZE,
                      cfg.DATA.TEMPLATE.SIZE,
                      cfg.TRAIN.ENCODER_OPEN, cfg)
    return encoder
