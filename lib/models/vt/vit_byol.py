""" Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale' - https://arxiv.org/abs/2010.11929

The official jax code is released and available at https://github.com/google-research/vision_transformer

Status/TODO:
* Models updated to be compatible with official impl. Args added to support backward compat for old PyTorch weights.
* Weights ported from official jax impl for 384x384 base and small models, 16x16 and 32x32 patches.
* Trained (supervised on ImageNet-1k) my custom 'small' patch model to 77.9, 'base' to 79.4 top-1 with this code.
* Hopefully find time and GPUs for SSL or unsupervised pretraining on OpenImages w/ ImageNet fine-tune in future.

Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert

Hacked together by / Copyright 2020 Ross Wightman
"""
import os
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
from functools import partial
import numpy as np

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

from lib.utils.pos_embed import get_sinusoid_encoding_table, get_2d_sincos_pos_embed



def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    # patch models
    # mae ViT-B/16-224 pre-trained model
    'vit_byol_base_patch16_256_v_b_16_256_500k_e200_bs64': _cfg(
        url='https://azsusw3.blob.core.windows.net/v-zhaojie/projects/playground/playground/checkpoints/train/byol/v_b_16_256_500k_e200_bs64/BYOL_ep0200.pth.tar?sv=2020-10-02&st=2022-04-28T01%3A03%3A38Z&se=2025-04-29T01%3A03%3A00Z&sr=c&sp=racwl&sig=juG7219CQ5wQFVIE5dWpNcaXilZQ9h7bBW42KwKI6oQ%3D'
    ),
    'vit_byol_base_patch16_256_v_b_16_256_500k_e200_bs64_drop': _cfg(
        url='https://azsusw3.blob.core.windows.net/v-zhaojie/projects/playground/playground/checkpoints/train/byol/v_b_16_256_500k_e200_bs64_drop/BYOL_ep0200.pth.tar?sv=2020-10-02&st=2022-04-28T01%3A03%3A38Z&se=2025-04-29T01%3A03%3A00Z&sr=c&sp=racwl&sig=juG7219CQ5wQFVIE5dWpNcaXilZQ9h7bBW42KwKI6oQ%3D'
    ),
}


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # B, C, H, W = x.shape
        # # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, num_img=2, search_size=384, template_size=192,
                 patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.embed_dim_list = [embed_dim]
        self.num_img = num_img

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=search_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=search_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        # num_patches = self.patch_embed.num_patches
        self.num_patches_search = (search_size // patch_size) * (search_size // patch_size)
        self.num_patches_template = (template_size // patch_size) * (template_size // patch_size)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # 1D pos
        num_patches = self.num_patches_search * 1 + self.num_patches_template * (self.num_img-1)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # # 2D pos
        # num_patches = self.num_patches_search
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches * 2 + 1, embed_dim))
        # pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(num_patches ** .5),
        #                                     cls_token=True)
        # pos_embed_1 = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(num_patches ** .5),
        #                                     cls_token=False)
        # pos_embed_all = np.concatenate([pos_embed, pos_embed_1])
        # self.pos_embed.data.copy_(torch.from_numpy(pos_embed_all).float().unsqueeze(0))

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        #self.repr = nn.Linear(embed_dim, representation_size)
        #self.repr_act = nn.Tanh()

        # Classifier head
        # self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # trunc_normal_(self.pos_embed, std=.02)
        # trunc_normal_(self.pos_embed_search, std=.02)
        # trunc_normal_(self.pos_embed_template, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
        pos_embed = get_sinusoid_encoding_table(num_patches, self.pos_embed.shape[-1], cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, images_list):

        B = images_list[0].shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        xz = cls_tokens + self.pos_embed[:, 0:1, :]
        for i in range(len(images_list)):
            x = images_list[i]
            x = self.patch_embed(x)
            if i == 0:
                x = x + self.pos_embed[:, 1:self.num_patches_search + 1, :]
                xz = torch.cat((xz, x), dim=1)
            else:
                x = x + self.pos_embed[:, self.num_patches_search + 1:, :]
                xz = torch.cat((xz, x), dim=1)
        xz = self.pos_drop(xz)


        # B = images_list[0].shape[0]
        # cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        #
        # x = self.patch_embed(x)
        # z = self.patch_embed(z)
        #
        #
        # xz = torch.cat((cls_tokens, x, z), dim=1)
        # xz = xz + self.pos_embed
        # xz = self.pos_drop(xz)



        for blk in self.blocks:   #vit use the their implemented attention,so the batchszie is the first.
            xz = blk(xz)

        xz = self.norm(xz) # torch.Size([16, 721, 768]) B,N,C
        return xz
        # return xz[:, 0]

    def forward(self, images_list):
        xz = self.forward_features(images_list)
        # x = self.head(x)
        out=[xz]
        return out


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict

@register_model
def vit_byol_xs_patch16(num_img=2,
                       pretrained=False, pretrain_type='scratch',
                       search_size=256, template_size=256, **kwargs):
    patch_size = 16
    model = VisionTransformer(
        num_img=num_img,
        search_size=search_size, template_size=template_size,
        patch_size=patch_size, num_classes=0,
        embed_dim=768, depth=4, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    cfg_type = 'vit_byol_xs_patch16_256_' + pretrain_type
    if pretrain_type == 'scratch':
        pretrained = False
        return model
    model.default_cfg = default_cfgs[cfg_type]
    if pretrained:
        load_pretrained(model, pretrain_type)
    return model

@register_model
def vit_byol_base_patch16(num_img=2,
                         pretrained=False, pretrain_type='scratch',
                         search_size=256, template_size=256, **kwargs):
    patch_size = 16
    model = VisionTransformer(
        num_img=num_img,
        search_size=search_size, template_size=template_size,
        patch_size=patch_size, num_classes=0,
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    cfg_type = 'vit_byol_base_patch16_256_' + pretrain_type
    if pretrain_type == 'scratch':
        pretrained = False
        return model
    model.default_cfg = default_cfgs[cfg_type]
    if pretrained:
        load_pretrained(model, pretrain_type)
    return model

@register_model
def vit_byol_large_patch16(num_img=2,
                          pretrained=False, pretrain_type='byol',
                              search_size=256, template_size=256, **kwargs):
    patch_size = 16
    model = VisionTransformer(
        num_img=num_img,
        search_size=search_size, template_size=template_size,
        patch_size=patch_size, num_classes=0,
        embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    cfg_type = 'vit_byol_large_patch16_256'
    if pretrain_type == 'scratch':
        pretrained = False
        return model
    model.default_cfg = default_cfgs[cfg_type]
    if pretrained:
        load_pretrained(model, pretrain_type)
    return model

@register_model
def vit_byol_huge_patch14(num_img=2,
                         pretrained=False, pretrain_type='byol',
                        search_size=256, template_size=256, **kwargs):
    patch_size = 14
    model = VisionTransformer(
        num_img=num_img,
        search_size=search_size, template_size=template_size,
        patch_size=patch_size, num_classes=0,
        embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    cfg_type = 'vit_byol_huge_patch16_256'
    if pretrain_type == 'scratch':
        pretrained = False
        return model
    model.default_cfg = default_cfgs[cfg_type]
    if pretrained:
        load_pretrained(model, pretrain_type)
    return model

def load_pretrained(model, pretrain_type='default', cfg=None, filter_fn=None, strict=True):
    if cfg is None:
        cfg = getattr(model, 'default_cfg')
    if cfg is None or 'url' not in cfg or not cfg['url']:
        print("Pretrained model URL is invalid, using random initialization.")
        return

    # model_dir = os.path.join('./pretrained', pretrain_type)
    model_dir = os.path.join(os.path.dirname(__file__),'../../..','pretrained/byol', pretrain_type)
    state_dict = model_zoo.load_url(cfg['url'], progress=False, map_location='cpu', model_dir=model_dir)
    if pretrain_type == 'mae':
        state_dict = state_dict['model']
    else:
        state_dict = state_dict['net']

    if filter_fn is not None:
        state_dict = filter_fn(state_dict)

    state_dict_to_load = {}
    del_list = ['net.pos_embed', 'net.ins_token']
    # del_list = ['net.ins_token']
    # del_list = ['mask_token']
    for key in state_dict.keys():
        if (key[0:3] == 'net') and (key not in del_list) and ('blocks_token_only' not in key):
            state_dict_to_load[key[4:]] = state_dict[key]

    state_dict_to_load['pos_embed'] = model.state_dict()['pos_embed']
    # state_dict_to_load['norm.weight'] = state_dict['net.blocks_token_only.0.norm1.weight']
    # state_dict_to_load['norm.bias'] = state_dict['net.blocks_token_only.0.norm1.bias']


    model.load_state_dict(state_dict_to_load, strict=strict)

