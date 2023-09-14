import torch
from torch import nn
from lib.models.layers.head import build_box_head
from lib.utils.box_ops import box_xyxy_to_cxcywh
import argparse
import importlib

class EfficientTrack(nn.Module):
    def __init__(self, box_head, num_heads=4, depth=3, embed_dim=128, head_type="CENTER", mode="eval", type="AF"):
        super().__init__()
        if type.endswith("LN"):
            from lib.models.efficientvit.efficientvitLN import EfficientViT
            self.back = EfficientViT(template_size=128, search_size=256, patch_size=16, in_chans=3,
                                     embed_dim=embed_dim, depth=depth, num_heads=num_heads, stages=type[:-2])
        else:
            from lib.models.efficientvit.efficientvit import EfficientViT
            self.back = EfficientViT(template_size=128, search_size=256, patch_size=16, in_chans=3,
                                         embed_dim=embed_dim, depth=depth, num_heads=num_heads, stages=type)
        self.box_head = box_head
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER" or head_type == "CORNER_LITE":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

    def forward(self, z, x):
        features = self.back(z, x)
        out = self.forward_head(features[-1], None)
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
            # x = self.box_head(opt_feat, gt_score_map)
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


import math


def build_efficienttrack(cfg, mode='eval'):
    embed_dim = cfg.MODEL.BACKBONE.CHANNELS
    depth = cfg.MODEL.BACKBONE.DEPTH
    # depth = 4
    num_heads=cfg.MODEL.BACKBONE.HEADS
    head_type = cfg.MODEL.HEAD.TYPE
    box_head = build_box_head(cfg, embed_dim)
    model = EfficientTrack(box_head, num_heads, depth, embed_dim, head_type, mode=mode, type=cfg.MODEL.BACKBONE.TYPE)
    if cfg.MODEL.PRETRAIN_FILE and mode != 'eval':
        ckpt = torch.load(cfg.MODEL.PRETRAIN_FILE)['model']#pth用model,tar用net
        pe = ckpt['pos_embed'][:, 1:, :]
        b_pe, hw_pe, c_pe = pe.shape
        side_pe = int(math.sqrt(hw_pe))
        pe_2D = pe.reshape([b_pe, side_pe, side_pe, c_pe]).permute([0, 3, 1, 2])  # b,c,h,w
        side_num_patches_search = 16
        side_num_patches_template = 8
        pe_s_2D = nn.functional.interpolate(pe_2D, [side_num_patches_search, side_num_patches_search],
                                            align_corners=True, mode='bicubic')
        pe_s = torch.flatten(pe_s_2D.permute([0, 2, 3, 1]), 1, 2)
        pe_t_2D = nn.functional.interpolate(pe_2D, [side_num_patches_template, side_num_patches_template],
                                            align_corners=True, mode='bicubic')
        pe_t = torch.flatten(pe_t_2D.permute([0, 2, 3, 1]), 1, 2)
        ckpt['pos_embed_z'] = pe_t
        ckpt['pos_embed_x'] = pe_s
        missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=False)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a train scripts in train_settings.')
    parser.add_argument('--script', type=str, default='efficienttrack', help='Name of the train script.')
    parser.add_argument('--config', type=str, default='experiments/efficienttrack/AFLN.yaml', help="Name of the config file.")
    args = parser.parse_args()

    config_module = importlib.import_module("lib.config.%s.config" % args.script)
    cfg = config_module.cfg
    config_module.update_config_from_file(args.config)
    model = build_efficienttrack(cfg, mode='training').to(torch.device(0))
    template = torch.randn((1, 3, 128, 128)).to(torch.device(0))
    search = torch.randn((1, 3, 256, 256)).to(torch.device(0))
    # for _ in range(100):
    #     __ = model(template, search)
    #
    # import time
    # tic = time.time()
    # for _ in range(1000):
    #     __ = model(template, search)
    # print(time.time() - tic)
    a = 1
    from thop import profile
    from thop.utils import clever_format
    macs1, params1 = profile(model, inputs=(template, search),
                             custom_ops=None, verbose=False)
    macs, params = clever_format([macs1, params1], "%.3f")
    print('overall macs is ', macs)
    print('overall params is ', params)