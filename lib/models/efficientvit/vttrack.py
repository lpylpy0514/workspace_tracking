import torch
from torch import nn
from timm.models.vision_transformer import Block
from lib.models.layers.head import build_box_head
from lib.utils.box_ops import box_xyxy_to_cxcywh
import importlib
import argparse


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


def b16(n, activation):
    return torch.nn.Sequential(
        Conv2d_BN(3, n // 8, 3, 2, 1),
        activation(),
        Conv2d_BN(n // 8, n // 4, 3, 2, 1),
        activation(),
        Conv2d_BN(n // 4, n // 2, 3, 2, 1),
        activation(),
        Conv2d_BN(n // 2, n, 3, 2, 1))


class LevitPatchEmbedding(nn.Module):
    def __init__(self, embed_dim, activation):
        super().__init__()
        self.net = b16(embed_dim, activation)

    def forward(self, x):
        x = self.net(x).flatten(2).transpose(1, 2)
        return x


class VtTrack(nn.Module):
    def __init__(self,
                 # box_head,
                 # head_type='CENTER',
                 template_size=128,
                 search_size=256,
                 patch_size=16,
                 in_chans=3,
                 stages='AF',
                 embed_dim=128,
                 key_dim=16,
                 depth=6,
                 num_heads=4,
                 distillation=False):
        super().__init__()
        self.patch_embed = torch.nn.Sequential(
                                    Conv2d_BN(3, embed_dim // 8, 3, 2, 1),
                                    torch.nn.Hardswish(),
                                    Conv2d_BN(embed_dim // 8, embed_dim // 4, 3, 2, 1),
                                    torch.nn.Hardswish(),
                                    Conv2d_BN(embed_dim // 4, embed_dim // 2, 3, 2, 1),
                                    torch.nn.Hardswish(),
                                    Conv2d_BN(embed_dim // 2, embed_dim, 3, 2, 1))
        self.pos_embed_z = nn.Parameter(torch.zeros(1, (template_size // patch_size) ** 2, embed_dim))
        self.pos_embed_x = nn.Parameter(torch.zeros(1, (search_size // patch_size) ** 2, embed_dim))
        # self.box_head = box_head
        # self.head_type = head_type
        # if head_type == "CORNER" or head_type == "CENTER":
        #     self.feat_sz_s = int(box_head.feat_sz)
        #     self.feat_len_s = int(box_head.feat_sz ** 2)

        self.blocks = []
        for i in range(depth):
            self.blocks.append(Block(embed_dim, num_heads, qkv_bias=True))
        self.blocks = torch.nn.Sequential(*self.blocks)

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, z, x):
        z = self.patch_embed(z).flatten(2).transpose(1, 2)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)

        z += self.pos_embed_z
        x += self.pos_embed_x

        x = torch.cat((z, x), dim=1)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
        x = self.norm(x)
        # out = self.forward_head(x, None)
        return [x]

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


def build_vttrack(cfg, depth=3):
    embed_dim = cfg.MODEL.BACKBONE.CHANNELS
    num_heads = cfg.MODEL.BACKBONE.HEADS
    patch_embedding = LevitPatchEmbedding(embed_dim, nn.Hardswish)
    box_head = build_box_head(cfg, embed_dim)
    model = VtTrack(box_head, num_heads=num_heads, embed_dim=embed_dim, depth=depth)
    return model
