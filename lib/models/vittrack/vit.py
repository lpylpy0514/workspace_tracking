# 2023.10.25
# Liu Pengyu
# vision transformer backbone
import torch
from torch import nn
from timm.models.vision_transformer import Block
from timm.models.layers import trunc_normal_
from lib.utils.pos_embed import get_sinusoid_encoding_table
import argparse


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        bn = torch.nn.BatchNorm2d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

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


class PatchEmbedding(nn.Module):
    def __init__(self, embed_dim, activation, img_size=224, patch_size=16, mode="standard"):
        super().__init__()
        assert img_size % patch_size == 0
        if mode == "standard":
            self.proj = torch.nn.Conv2d(in_channels=3, out_channels=embed_dim, kernel_size=(patch_size, patch_size),
                                       stride=(patch_size, patch_size))
        elif mode == "overlap":
            self.proj = torch.nn.Sequential(Conv2d_BN(3, embed_dim // 8, 3, 2, 1), activation,
                                               Conv2d_BN(embed_dim // 8, embed_dim // 4, 3, 2, 1,), activation,
                                               Conv2d_BN(embed_dim // 4, embed_dim // 2, 3, 2, 1,), activation,
                                               Conv2d_BN(embed_dim // 2, embed_dim, 3, 2, 1,))
        elif mode == "conv2x2":
            self.proj = torch.nn.Sequential(Conv2d_BN(3, embed_dim // 8, 2, 2, 0), activation,
                                               Conv2d_BN(embed_dim // 8, embed_dim // 4, 2, 2, 0,), activation,
                                               Conv2d_BN(embed_dim // 4, embed_dim // 2, 2, 2, 0,), activation,
                                               Conv2d_BN(embed_dim // 2, embed_dim, 2, 2, 0,))
        else:
            raise NotImplementedError

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, template_size, search_size, patch_embedding, patch_size, num_heads=12,
                 mlp_ratio=4, depth=12, embed_dim=768, distillation=False):
        super().__init__()
        self.patch_embed = patch_embedding
        # self.pos_embed_z = nn.Parameter(torch.zeros(1, (template_size // patch_size) ** 2, embed_dim))
        # self.pos_embed_x = nn.Parameter(torch.zeros(1, (search_size // patch_size) ** 2, embed_dim))
        num_patches = (search_size // patch_size) ** 2 + (template_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.embed_dim = embed_dim
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True)for i in range(depth)])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.norm = nn.LayerNorm(embed_dim)
        self.distillation = distillation

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

    def forward(self, z, x):
        B = x.shape[0]
        z = self.patch_embed(z)
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(B, -1, -1)
        xz = cls_token
        xz = torch.cat((xz, x), dim=1)
        xz = torch.cat((xz, z), dim=1)
        xz = xz + self.pos_embed
        res = []
        for i, blk in enumerate(self.blocks):
            xz = blk(xz)
            res.append(xz)
        xz = self.norm(xz)
        res[-1] = xz
        return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a train scripts in train_settings.')
    parser.add_argument('--device', type=str, default='cpu', help='Device of speed test.')
    args = parser.parse_args()

    embed_dim = 256
    pemode = "standard"
    device = args.device
    patch_embed = PatchEmbedding(embed_dim=embed_dim, activation=nn.Hardswish(), img_size=256, patch_size=16, mode=pemode)
    model = VisionTransformer(template_size=128, search_size=256, patch_embedding=patch_embed, patch_size=16,
                              depth=12, embed_dim=embed_dim, num_heads=embed_dim // 64, mlp_ratio=4)

    bss = [1, 2, 4, 8, 16]
    for bs in bss:
        print("batch size is" + str(bs))
        template = torch.randn((bs, 3, 128, 128))
        search = torch.randn((bs, 3, 256, 256))
        model = model.to(device)
        template = template.to(device)
        search = search.to(device)
        # speed test
        import time
        for _ in range(50):
            __ = model(template, search)
        tic = time.time()
        for _ in range(100):
            __ = model(template, search)
        print(bs * 100/(time.time() - tic))
    # param & macs test
    from thop import profile
    from thop.utils import clever_format
    macs1, params1 = profile(model, inputs=(template, search),
                             custom_ops=None, verbose=False)
    macs, params = clever_format([macs1, params1], "%.3f")
    print('overall macs is ', macs)
    print('overall params is ', params)
