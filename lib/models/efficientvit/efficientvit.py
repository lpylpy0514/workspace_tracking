# light vit tracker
import torch
from torch import nn
from timm.models.vision_transformer import trunc_normal_
from thop import profile
from thop.utils import clever_format
import itertools


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


class Conv1d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv1d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm1d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv1d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        self.add_module('l', torch.nn.Linear(a, b, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            torch.nn.init.constant_(self.l.bias, 0)

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps)**0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = torch.nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)


# class FFN(torch.nn.Module):
#     def __init__(self, ed, h):
#         super().__init__()
#         in_features = ed
#         hidden_features = h
#         out_features = ed
#         self.pw1 = torch.nn.Conv1d(in_features, hidden_features, 1)
#         self.bn1 = torch.nn.BatchNorm1d(hidden_features)
#         self.act = torch.nn.ReLU()
#         self.pw2 = torch.nn.Conv1d(hidden_features, out_features, 1)
#         self.bn2 = torch.nn.BatchNorm1d(out_features)
#
#     def forward(self, x):
#         x = self.bn2(self.pw2(self.act(self.bn1(self.pw1(x)))))
#         return x


class FFN(torch.nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.pw1 = Conv1d_BN(embed_dim, hidden_dim, 1)
        self.act = torch.nn.ReLU()
        self.pw2 = Conv1d_BN(hidden_dim, embed_dim, 1)

    def forward(self, x):
        x = self.pw2(self.act(self.pw1(x)))
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, return_attention=False):
        x = x.transpose(1, 2)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attention:
            return x, attn
        x = x.transpose(1, 2)
        return x


class GroupAttention(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads=8, attn_ratio=4):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.d = int(attn_ratio * key_dim)
        self.attn_ratio = attn_ratio
        self.qkvs = torch.nn.Conv1d(dim, num_heads * (self.key_dim * 2 + self.d), kernel_size=1, groups=num_heads)
        self.proj = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Conv1d(self.d * num_heads, dim, 1), torch.nn.BatchNorm1d(dim))

    def forward(self, x):
        B, C, N = x.shape
        x = self.qkvs(x)
        x = x.reshape(B, self.num_heads, self.key_dim * 2 + self.d, N)
        q, k, v = x.split([self.key_dim, self.key_dim, self.d], dim=2)
        attn = ((q.transpose(-2, -1) @ k) * self.scale)
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1))
        x = x.reshape(B, self.d * self.num_heads, N)
        x = self.proj(x)
        return x


class EfficientViTBlock(torch.nn.Module):
    def __init__(self, type, embed_dim, kd, num_heads=8, attn_ratio=4):
        super().__init__()
        self.type = type
        if self.type == 'FGAF':
            self.ffn0 = Residual(FFN(embed_dim, int(embed_dim * 2)))
            self.mixer = Residual(GroupAttention(embed_dim, kd, num_heads, attn_ratio=attn_ratio))
            self.ffn1 = Residual(FFN(embed_dim, int(embed_dim * 2)))
        elif self.type == "GAF":
            self.ffn0 = torch.nn.Identity()
            self.mixer = Residual(GroupAttention(embed_dim, kd, num_heads, attn_ratio=attn_ratio))
            self.ffn1 = Residual(FFN(embed_dim, int(embed_dim * 4)))
        elif self.type == 'FAF':
            self.ffn0 = Residual(FFN(embed_dim, int(embed_dim * 2)))
            self.mixer = Residual(Attention(embed_dim, num_heads=num_heads))
            self.ffn1 = Residual(FFN(embed_dim, int(embed_dim * 2)))
        elif self.type == 'AF':
            self.ffn0 = torch.nn.Identity()
            self.mixer = Residual(Attention(embed_dim, num_heads=num_heads))
            self.ffn1 = Residual(FFN(embed_dim, int(embed_dim * 4)))
        else:
            raise NotImplementedError

    def forward(self, x):
        return self.ffn1(self.mixer(self.ffn0(x)))


class EfficientViT(torch.nn.Module):
    def __init__(self, template_size=128,
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

        # Patch embedding
        self.patch_embed = torch.nn.Sequential(Conv2d_BN(in_chans, embed_dim // 8, 3, 2, 1), torch.nn.Hardswish(),
                                               Conv2d_BN(embed_dim // 8, embed_dim // 4, 3, 2, 1,), torch.nn.Hardswish(),
                                               Conv2d_BN(embed_dim // 4, embed_dim // 2, 3, 2, 1,), torch.nn.Hardswish(),
                                               Conv2d_BN(embed_dim // 2, embed_dim, 3, 2, 1,))

        self.distillation = distillation
        self.blocks = []
        attn_ratio = embed_dim // (num_heads * key_dim)
        for i in range(depth):
            self.blocks.append(EfficientViTBlock(stages, embed_dim, key_dim, num_heads, attn_ratio))
        self.blocks = torch.nn.Sequential(*self.blocks)
        self.norm = torch.nn.LayerNorm(embed_dim)
        self.pos_embed_z = torch.nn.Parameter(torch.zeros(1, (template_size // patch_size) ** 2, embed_dim))
        self.pos_embed_x = torch.nn.Parameter(torch.zeros(1, (search_size // patch_size) ** 2, embed_dim))

    @torch.jit.ignore
    def no_weight_decay(self):
        return {x for x in self.state_dict().keys() if 'attention_biases' in x}

    def forward(self, z, x):
        z = self.patch_embed(z).flatten(2).transpose(1, 2) + self.pos_embed_z
        x = self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos_embed_x
        x = torch.concat((z, x), dim=1).permute(0, 2, 1)
        res_list = []
        if self.distillation:
            for block in self.blocks:
                x = block(x)
                res_list.append(self.norm(x.permute(0, 2, 1)))
            return res_list
        else:
            for block in self.blocks:
                x = block(x)
            return [self.norm(x.permute(0, 2, 1))]


#  replace conv+BN to conv
def replace_batchnorm(net):
    for child_name, child in net.named_children():
        if hasattr(child, 'fuse'):
            setattr(net, child_name, child.fuse())
        else:
            replace_batchnorm(child)


if __name__ == '__main__':
    model = EfficientViT(depth=3, stages="FGAF")
    search = torch.randn((1, 3, 256, 256))
    template = torch.randn((1, 3, 128, 128))
    res = model(template, search)
    # model = model.to(torch.device(0))
    # template = template.to(torch.device(0))
    # search = search.to(torch.device(0))

    # for _ in range(100):
    #     __ = model(template, search)
    #
    # import time
    # tic = time.time()
    # for _ in range(1000):
    #     __ = model(template, search)
    # print(time.time() - tic)
    macs1, params1 = profile(model, inputs=(template, search),
                             custom_ops=None, verbose=False)
    macs, params = clever_format([macs1, params1], "%.3f")
    print('overall macs is ', macs)
    print('overall params is ', params)
    model = GroupAttention(128, 16, 4, 2)
    total_params = 0

    # 遍历网络模型的所有参数
    for param in model.parameters():
        # 获取参数的数量并添加到总数中
        total_params += param.numel()
    print(total_params)

