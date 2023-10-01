from lib.models.efficientvit.efficientvit import EfficientViTBlock, Conv2d_BN
from functools import partial
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from timm.models.vision_transformer import Block
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple
from lib.utils.pos_embed import get_sinusoid_encoding_table


class MTR(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, num_img=2, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 mask_type="RANDOM"):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.mask_type = mask_type

        self.patch_embed = torch.nn.Sequential(Conv2d_BN(in_chans, embed_dim // 8, 3, 2, 1), torch.nn.Hardswish(),
                                               Conv2d_BN(embed_dim // 8, embed_dim // 4, 3, 2, 1,), torch.nn.Hardswish(),
                                               Conv2d_BN(embed_dim // 4, embed_dim // 2, 3, 2, 1,), torch.nn.Hardswish(),
                                               Conv2d_BN(embed_dim // 2, embed_dim, 3, 2, 1,))
        key_dim = 16 #  11111111111111111111111111
        stages = "FGAF"
        attn_ratio = embed_dim // (num_heads * key_dim)
        for i in range(depth):
            self.blocks.append(EfficientViTBlock(stages, embed_dim, key_dim, num_heads, attn_ratio))
        self.blocks = torch.nn.Sequential(*self.blocks)
        self.norm = torch.nn.LayerNorm(embed_dim)
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches * num_img + 1, embed_dim), requires_grad=True)

        # self.pos_embed_z = torch.nn.Parameter(torch.zeros(1, (img_size // patch_size) ** 2, embed_dim))
        # self.pos_embed_x = torch.nn.Parameter(torch.zeros(1, (img_size // patch_size) ** 2, embed_dim))

        # self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)

        # self.num_img = num_img
        #
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches * num_img + 1, embed_dim),
        #                               requires_grad=False)  # fixed sin-cos embedding
        #
        # self.blocks = nn.ModuleList([
        #     Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
        #     for i in range(depth)])
        # self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches * num_img + 1, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        # pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
        #                                     cls_token=True)
        pos_embed = get_sinusoid_encoding_table(self.patch_embed.num_patches * self.num_img, self.pos_embed.shape[-1],
                                                cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
        #                                             int(self.patch_embed.num_patches ** .5), cls_token=True)
        decoder_pos_embed = get_sinusoid_encoding_table(self.patch_embed.num_patches * self.num_img,
                                                        self.decoder_pos_embed.shape[-1], cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def reconstruction(self, xs, images_list):
        for i in range(len(images_list)):
            z = images_list[i]
            z = self.patchify(z)
            if i == 0:
                zs = z
            else:
                zs = torch.cat((zs, z), dim=1)
        target = zs
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            # mean = target.mean()
            # var = target.var()
            xs_norm = (xs * ((var + 1.e-6) ** .5)) + mean

        num_patches = self.patch_embed.num_patches
        images_cons_list = []
        for i in range(self.num_img):
            x = xs[:, num_patches * i:num_patches * (i + 1), :]
            x = self.unpatchify(x)
            images_cons_list.append(x)

        num_patches = self.patch_embed.num_patches
        images_cons_norm_list = []
        if self.norm_pix_loss:
            for i in range(self.num_img):
                x_norm = xs_norm[:, num_patches * i:num_patches * (i + 1), :]
                x_norm = self.unpatchify(x_norm)
                images_cons_norm_list.append(x_norm)
        return images_cons_list, images_cons_norm_list, target
        # if self.norm_pix_loss:
        #     mean = target.mean(dim=-1, keepdim=True)
        #     var = target.var(dim=-1, keepdim=True)
        #     target = (target - mean) / (var + 1.e-6)**.5

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def space_random_masking(self, x, mask_ratio, num_frames):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int((L // num_frames) * (1 - mask_ratio)) * num_frames

        noise = torch.rand(N, L // num_frames, device=x.device)  # noise in [0, 1]
        noise = noise.unsqueeze(1).repeat(1, num_frames, 1).flatten(1)

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, images_list, mask_ratio):
        # embed patches
        for i in range(len(images_list)):
            x = images_list[i]
            x = self.patch_embed(x)
            if i == 0:
                xz = x
            else:
                xz = torch.cat((xz, x), dim=1)

        # add pos embed w/o cls token
        xz = xz + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        if self.mask_type == "RANDOM":
            xz, mask, ids_restore = self.random_masking(xz, mask_ratio)
        elif self.mask_type == "SPACE_RANDOM":
            xz, mask, ids_restore = self.space_random_masking(xz, mask_ratio, len(images_list))
        else:
            raise ValueError('Illegal mask_type')

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(xz.shape[0], -1, -1)
        xz = torch.cat((cls_tokens, xz), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            xz = blk(xz)
        xz = self.norm(xz)

        return xz, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, images_list, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """

        # target = self.patchify(images_list)

        for i in range(len(images_list)):
            x = images_list[i]
            x = self.patchify(x)
            if i == 0:
                xz = x
            else:
                xz = torch.cat((xz, x), dim=1)
        target = xz
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, images_list, mask_ratio=0.75):
        assert len(images_list) == self.num_img
        latent, mask, ids_restore = self.forward_encoder(images_list, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(images_list, pred, mask)
        return loss, pred, mask