"""
Basic ViPT model.
"""
import math
import os
from typing import List
from timm.models.layers import to_2tuple
import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones
from lib.models.layers.head import build_box_head
from lib.models.vipt.vit_prompt import vit_base_patch16_224_prompt
from lib.models.vipt.vit_ce_prompt import vit_base_patch16_224_ce_prompt
from lib.utils.box_ops import box_xyxy_to_cxcywh
from timm.models.vision_transformer import trunc_normal_


class ViPTrack(nn.Module):
    """ This is the base class for ViPTrack """

    def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER", template_preprocess=None, search_preprocess=None):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

        self.template_preprocess = template_preprocess
        self.search_preprocess = search_preprocess
        if self.search_preprocess == "learn":
            self.search_alpha = torch.nn.Parameter(torch.zeros(1, 3, 256, 256))
            trunc_normal_(self.search_alpha, std=.02)

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                template_anno=None,
                past_search_anno=None,
                template_mask=None,
                ):
        B, C, W, H = template.shape
        if type(self.template_preprocess) is nn.Identity:
            assert H == W
            image_size = H
            indice = torch.arange(0, W).view(-1, 1)
            coord_x = indice.repeat((image_size, 1)).view(image_size, image_size).float().to(template.device)
            coord_y = indice.repeat((1, image_size)).view(image_size, image_size).float().to(template.device)
            x1, y1, w, h = (template_anno.view(B, 4, 1, 1) * image_size).unbind(1)
            x2, y2 = x1 + w, y1 + h
            template_alpha = (x2 > coord_x) & (coord_x > x1) & (y2 > coord_y) & (coord_y > y1)
            template_alpha = template_alpha.float().view(B, 1, H, W).cuda()
            template = torch.concat((template, template_alpha, template_alpha, template_alpha), dim=1)
        elif self.template_preprocess == "image_vipt":
            assert H == W
            image_size = H
            indice = torch.arange(0, W).view(-1, 1)
            coord_x = indice.repeat((image_size, 1)).view(image_size, image_size).float().to(template.device)
            coord_y = indice.repeat((1, image_size)).view(image_size, image_size).float().to(template.device)
            x1, y1, w, h = (template_anno.view(B, 4, 1, 1) * image_size).unbind(1)
            x2, y2 = x1 + w, y1 + h
            template_alpha = (x2 > coord_x) & (coord_x > x1) & (y2 > coord_y) & (coord_y > y1)
            template_alpha = template_alpha.float().view(B, 1, H, W).cuda() * template
            template = torch.concat((template, template_alpha), dim=1)
        if self.search_preprocess == "learn":
            search = torch.concat((search, self.search_alpha.repeat((B, 1, 1, 1))), dim=1)
        else:
            search_alpha = torch.zeros((B, 1, H * 2, W * 2)).cuda()
            search = torch.concat((search, search_alpha, search_alpha, search_alpha), dim=1)
        x, aux_dict = self.backbone(z=template, x=search,
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn, )

        # Forward head
        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]
        out = self.forward_head(feat_last, None)

        out.update(aux_dict)
        out['backbone_feat'] = x
        return out

    def forward_head(self, cat_feature, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        enc_opt = cat_feature[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError


def build_viptrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')  # use pretrained OSTrack as initialization
    if cfg.MODEL.PRETRAIN_FILE and ('OSTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_prompt':
        backbone = vit_base_patch16_224_prompt(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                               search_size=to_2tuple(cfg.DATA.SEARCH.SIZE),
                                               template_size=to_2tuple(cfg.DATA.TEMPLATE.SIZE),
                                               new_patch_size=cfg.MODEL.BACKBONE.STRIDE,
                                               prompt_type=cfg.TRAIN.PROMPT.TYPE
                                               )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce_prompt':
        backbone = vit_base_patch16_224_ce_prompt(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                           ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                           search_size=to_2tuple(cfg.DATA.SEARCH.SIZE),
                                           template_size=to_2tuple(cfg.DATA.TEMPLATE.SIZE),
                                           new_patch_size=cfg.MODEL.BACKBONE.STRIDE,
                                           prompt_type=cfg.TRAIN.PROMPT.TYPE
                                           )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    else:
        raise NotImplementedError
    """For prompt no need, because we have OSTrack as initialization"""
    # backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)

    if cfg.MODEL.PROCESS.TEMPLATE == "draw_vipt":
        template_preprocess = nn.Identity()
    elif cfg.MODEL.PROCESS.TEMPLATE == "image_vipt":
        template_preprocess = "image_vipt"
    model = ViPTrack(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
        template_preprocess=template_preprocess,
        search_preprocess=getattr(cfg.MODEL.PROCESS, "SEARCH", "")
    )

    if 'OSTrack' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)
        #print(f"missing_keys: {missing_keys}")
        #print(f"unexpected_keys: {unexpected_keys}")

    return model
