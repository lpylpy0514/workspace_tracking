# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
import math
import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block


class VisionTransformer(nn.Module):
    def __init__(self, search_size: int, template_size: int, patch_size: int, depth: int, num_heads: int, embed_dim: int, mlp_ratio=4):
        super().__init__()
        self.search_size = search_size
        self.template_size = template_size
        self.num_patches_search = (search_size // patch_size) * (search_size // patch_size)
        self.num_patches_template = (template_size // patch_size) * (template_size // patch_size)

        #  MAE part
        # self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.patch_embed = nn.Conv2d(in_channels=3, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)
        # num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches_search + self.num_patches_template + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=nn.LayerNorm)
            for i in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, images_list):
        class_embedding = self.cls_token
        B = images_list[0].shape[0]
        class_embedding = class_embedding.expand(B, -1, -1)
        xz = class_embedding + self.pos_embed[:, 0:1, :].to(images_list[0].dtype)
        for i in range(len(images_list)):
            x = images_list[i]
            x = self.patch_embed(x)
            x = x.reshape(x.shape[0], x.shape[1], -1)
            x = x.permute(0, 2, 1)
            if i == 0:
                x = x + self.pos_embed[:, 1:self.num_patches_search + 1, :].to(x.dtype)
                xz = torch.cat([xz, x], dim=1)
            else:
                x = x + self.pos_embed[:, self.num_patches_search + 1:, :].to(x.dtype)
                xz = torch.cat([xz, x], dim=1)
        feature = []
        for i, blk in enumerate(self.blocks):
            xz = blk(xz)
            feature.append(xz[:, 1:, :])

        return feature

    def load_weight(self, pretrained):
        if pretrained is None:
            return
        else:
            # ckpt = torch.load("/home/lpy/OSTrack/pretrained_models/mae_pretrain_vit_large.pth")['model']
            ckpt = torch.load(pretrained)['model']
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
            ckpt['pos_embed'] = torch.cat([ckpt['pos_embed'][:, 0:1, :], pe_s, pe_t], dim=1)
            ckpt['patch_embed.weight'] = ckpt['patch_embed.proj.weight']
            ckpt['patch_embed.bias'] = ckpt['patch_embed.proj.bias']

            missing_keys, unexpected_keys = self.load_state_dict(ckpt, strict=False)
            print("missing keys: ", missing_keys)
            print("unexpected keys: ", unexpected_keys)


def mae_vit_b(ckpt):
    model = VisionTransformer(search_size=256, template_size=128, patch_size=16, depth=12, num_heads=12, embed_dim=768)
    model.load_weight(ckpt)
    return model


def mae_vit_l(ckpt):
    model = VisionTransformer(search_size=256, template_size=128, patch_size=16, depth=24, num_heads=16, embed_dim=1024)
    model.load_weight(ckpt)
    return model


def mae_vit_h(ckpt):
    model = VisionTransformer(search_size=256, template_size=128, patch_size=16, depth=36, num_heads=20, embed_dim=1280)
    model.load_weight(ckpt)
    return model


if __name__ == '__main__':
    model = VisionTransformer(search_size=256, template_size=128, patch_size=16, depth=24, num_heads=16, embed_dim=1024)
    pretrained = "/home/lpy/OSTrack/pretrained_models/mae_pretrain_vit_large.pth"
    model.load_weight(pretrained)
    search_img = torch.rand([1, 3, 256, 256])
    template_img = torch.rand([1, 3, 128, 128])
    res = model([search_img, template_img])
