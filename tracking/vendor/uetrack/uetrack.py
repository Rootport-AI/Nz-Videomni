# Vendored from https://github.com/kangben258/UETrack @ fd13b0ea
# Source file: lib/models/uetrack/uetrack.py
#
# NZ modifications:
#   * Only the `UETrack` module and `build_uetrack_inference` are carried.
#     `SUTRACK` (the teacher), `ADAPTIVE_NET` (the distillation gate) and
#     `build_uetrack` (the training builder, which torch.loads a teacher
#     checkpoint from a path that only exists on the authors' machines) are
#     training-side and are not.
#   * `interface_text_proj` is built only when a text encoder exists. Upstream
#     dereferences `self.text_encoder.textencoder_dim` unconditionally, which
#     raises AttributeError the moment MULTI_MODAL_LANGUAGE is False -- so
#     upstream's own "language off" switch is unreachable as written. That
#     projection is the seam between CLIP and the tracker; with CLIP gone it has
#     nothing to project, and its two tensors are the only non-CLIP names that
#     turn up as missing keys when the official checkpoint is loaded.
#   * `forward_textencoder_inference` is not carried (CLIP-only).

import math

from torch import nn

from .decoder import build_decoder
from .encoder import build_encoder
from .task_decoder import build_task_decoder
from .box_ops import box_xyxy_to_cxcywh


class UETrack(nn.Module):
    """ This is the base class for UETrack """
    def __init__(self, text_encoder, encoder, decoder, task_decoder,
                 num_frames=1, num_template=1,
                 decoder_type="CENTER", task_feature_type="average",adjust_layers=None):
        """ Initializes the model.
        """
        super().__init__()
        self.encoder = encoder
        self.text_encoder = text_encoder
        self.decoder_type = decoder_type

        self.class_token = False if (encoder.body.cls_token is None) else True
        self.task_feature_type = task_feature_type

        self.num_patch_x = self.encoder.body.num_patches_search
        self.num_patch_z = self.encoder.body.num_patches_template
        self.fx_sz = int(math.sqrt(self.num_patch_x))
        self.fz_sz = int(math.sqrt(self.num_patch_z))

        self.task_decoder = task_decoder
        self.decoder = decoder

        self.num_frames = num_frames
        self.num_template = num_template

        # NZ: guarded (was unconditional). See module header.
        if text_encoder is not None:
            from .timm_compat import trunc_normal_
            self.interface_text_proj = nn.Linear(self.text_encoder.textencoder_dim, self.encoder.num_channels)
            trunc_normal_(self.interface_text_proj.weight, std=.02)
            nn.init.constant_(self.interface_text_proj.bias, 0)
        else:
            self.interface_text_proj = None
        self.adjust_layers = adjust_layers

    def forward_encoder(self, template_list, search_list, template_anno_list, text_src, task_index):
        # Forward the encoder
        xz,feature_list = self.encoder(template_list, search_list, template_anno_list, text_src, task_index)
        if self.adjust_layers is not None:
            feature_list_aligned = []
            for adjust_layer, feat in zip(self.adjust_layers, feature_list):
                feature_list_aligned.append(adjust_layer(feat))
        else:
            feature_list_aligned = feature_list
        return xz,feature_list,feature_list_aligned

    def forward_decoder(self, feature, gt_score_map=None):

        feature = feature[0]
        if self.class_token:
            feature = feature[:,1:self.num_patch_x * self.num_frames+1]
        else:
            feature = feature[:,0:self.num_patch_x * self.num_frames] # (B, HW, C)

        bs, HW, C = feature.size()
        if self.decoder_type in ['CORNER', 'CENTER']:
            feature = feature.permute((0, 2, 1)).contiguous()
            feature = feature.view(bs, C, self.fx_sz, self.fx_sz)
        if self.decoder_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.decoder(feature, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.decoder_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.decoder(feature, gt_score_map)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        elif self.decoder_type == "MLP":
            # run the mlp head
            score_map, bbox, offset_map = self.decoder(feature, gt_score_map)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError

    def forward_task_decoder(self, feature):
        feature = feature[0]
        if self.task_feature_type == 'class':
            feature = feature[:, 0:1]
        elif self.task_feature_type == 'text':
            feature = feature[:, -1:]
        elif self.task_feature_type == 'average':
            feature = feature.mean(1).unsqueeze(1)
        else:
            raise NotImplementedError('task_feature_type must be choosen from class, text, and average')
        feature = self.task_decoder(feature)
        return feature


def build_uetrack_inference(cfg):
    encoder = build_encoder(cfg)
    if cfg.DATA.MULTI_MODAL_LANGUAGE:
        # NZ: unreachable in this port -- MULTI_MODAL_LANGUAGE is pinned False by
        # tracking/uetrack_runtime.py, and the CLIP text encoder is not vendored.
        raise NotImplementedError(
            "MULTI_MODAL_LANGUAGE is not supported by this port (the CLIP text "
            "encoder is not vendored). See tracking/VENDOR_NOTICE.md.")
    text_encoder = None
    decoder = build_decoder(cfg, encoder)
    task_decoder = build_task_decoder(cfg, encoder)
    model = UETrack(
        text_encoder,
        encoder,
        decoder,
        task_decoder,
        num_frames = cfg.DATA.SEARCH.NUMBER,
        num_template = cfg.DATA.TEMPLATE.NUMBER,
        decoder_type=cfg.MODEL.DECODER.TYPE,
        task_feature_type=cfg.MODEL.TASK_DECODER.FEATURE_TYPE
    )

    return model
