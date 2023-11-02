import torch
from torch import nn
from lib.models.vittrack.head import build_box_head
from lib.models.vittrack.vit import PatchEmbedding, VisionTransformer
from lib.utils.box_ops import box_xyxy_to_cxcywh
import importlib
import argparse


class VitTrack(nn.Module):
    def __init__(self, backbone, box_head, embed_dim, bottleneck=None, head_type="CENTER", simple_head=False):
        super().__init__()
        self.backbone = backbone
        if bottleneck is None:
            self.bottleneck = nn.Linear(backbone.embed_dim, embed_dim)  # the bottleneck layer
        else:
            self.bottleneck = bottleneck
        self.box_head = box_head
        self.head_type = head_type
        self.simple_head = simple_head
        if "CORNER" in head_type or "CENTER" in head_type:
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

    def forward(self, z, x):
        feat = self.backbone(z, x)
        feat = feat[-1].permute(1, 0, 2)  # (N, B, C)
        cls_token = feat[0:1, :, :]  # (1, B, C)
        search_feature = feat[1:self.feat_len_s+1]
        out = self.forward_head(cls_token, search_feature)
        return out

    def forward_head(self, cls_token, search_feature):
        # corner head
        if "CORNER" in self.head_type:
            feature = search_feature.transpose(0, 1)  # (B, C, N)
            if self.simple_head is True:
                feature = feature.transpose(2, 1)  # (B, N, C)
                bs, C, HW = feature.size()
                feature = feature.view(bs, C, self.feat_sz_s, self.feat_sz_s)
                Nq = 1
            else:
                dec_opt = cls_token.permute(1, 2, 0)  # (B, C, N)
                att = torch.matmul(feature, dec_opt)  # (B, HW, N)
                feature = (feature.unsqueeze(-1) * att.unsqueeze(-2)).permute((0, 3, 2, 1)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW)
                bs, Nq, C, HW = feature.size()
                feature = feature.view(-1, C, self.feat_sz_s, self.feat_sz_s)
            outputs_coord = box_xyxy_to_cxcywh(self.box_head(feature))
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new}
            return out, outputs_coord_new
        elif "CENTER" in self.head_type:
            opt = (search_feature.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
            bs, Nq, C, HW = opt.size()
            opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, None)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError


def build_vittrack(cfg):
    embed_dim = cfg.MODEL.BACKBONE.CHANNELS
    num_heads = cfg.MODEL.BACKBONE.HEADS
    depth = cfg.MODEL.BACKBONE.DEPTH
    patchEmbedMode = cfg.MODEL.BACKBONE.PEMODE
    template_size = cfg.DATA.TEMPLATE.SIZE
    search_size = cfg.DATA.SEARCH.SIZE
    patch_embed = PatchEmbedding(embed_dim=embed_dim, activation=nn.Hardswish(), img_size=256, patch_size=16,
                                 mode=patchEmbedMode)
    backbone = VisionTransformer(template_size=template_size, search_size=search_size, patch_embedding=patch_embed,
                                patch_size=16, depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4)
    box_head = build_box_head(cfg)
    head_type = cfg.MODEL.HEAD.TYPE
    simple_head = cfg.MODEL.SIMPLE_HEAD
    if "CENTER" in head_type:
        bottleneck = torch.nn.Identity()
    else:
        bottleneck = None
    model = VitTrack(backbone, box_head, embed_dim, head_type=head_type, simple_head=simple_head, bottleneck=bottleneck)
    if cfg.MODEL.PRETRAIN_FILE and model.training is True:
        if cfg.MODEL.PRETRAIN_FILE.endswith('pth'):
            ckpt = torch.load(cfg.MODEL.PRETRAIN_FILE)['model']  # pth(MAE) model
        elif cfg.MODEL.PRETRAIN_FILE.endswith('tar'):
            ckpt = torch.load(cfg.MODEL.PRETRAIN_FILE)['net']  # tar(MTR) net
        del ckpt["pos_embed"]
        # do not load pos_embed data
        missing_keys, unexpected_keys = model.backbone.load_state_dict(ckpt, strict=False)
        print("missing keys: ", missing_keys)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='build vit track.')
    parser.add_argument('--script', type=str, default='vittrack', help='Name of the train script.')
    parser.add_argument('--config', type=str, default='experiments/vittrack/debug.yaml', help="Name of the config file.")
    parser.add_argument('--device', type=str, default='cuda:0', help='Device of speed test.')
    args = parser.parse_args()

    device = args.device
    config_module = importlib.import_module("lib.config.%s.config" % args.script)
    cfg = config_module.cfg
    config_module.update_config_from_file(args.config)
    model = build_vittrack(cfg)

    template = torch.randn((1, 3, 128, 128))
    search = torch.randn((1, 3, 256, 256))
    model = model.to(device)
    template = template.to(device)
    search = search.to(device)

    for _ in range(50):
        __ = model(template, search)

    import time
    tic = time.time()
    for _ in range(100):
        __ = model(template, search)
    print(100/(time.time() - tic))
    a = 1
    from thop import profile
    from thop.utils import clever_format
    macs1, params1 = profile(model, inputs=(template, search),
                             custom_ops=None, verbose=False)
    macs, params = clever_format([macs1, params1], "%.3f")
    print('overall macs is ', macs)
    print('overall params is ', params)
