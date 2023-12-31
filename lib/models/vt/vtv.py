"""
Basic STARK Model (Spatial-only).
"""
import torch
import math
from torch import nn

from lib.utils.misc import NestedTensor

from .backbone import build_backbone
from .head import build_box_head
from .neck import build_neck
from lib.utils.box_ops import box_xyxy_to_cxcywh
from lib.models.vt.vt import VT


class VTV(VT):
    """ This is the base class for Transformer Tracking """
    def __init__(self, backbone, box_head, hidden_dim, num_queries,
                 bottleneck=None, aux_loss=False, head_type="CORNER",
                 neck_type='LINEAR'):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_queries: number of object queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__(backbone, box_head, hidden_dim,
                         num_queries, bottleneck, aux_loss,
                         head_type, neck_type)

    def forward(self, images_list=None, xz=None, mode="backbone", run_box_head=True, run_cls_head=False):
        if mode == "backbone":
            return self.forward_backbone(images_list)
        elif mode == "transformer": # this is head not transformer
            return self.forward_transformer(xz, run_box_head=run_box_head, run_cls_head=run_cls_head)
        else:
            raise ValueError

    def forward_backbone(self, images_list):
        """The input type is Tensors:
               - x: batched images, of shape [batch_size x 3 x Hs x Ws]
               - z: a binary mask of shape [batch_size x Ht x Wt]
        """
        # Forward the backbone
        xz = self.backbone(images_list)  # features & masks, position embedding for the search
        # Adjust the shapes
        return xz

    def forward_transformer(self, xz, run_box_head=True, run_cls_head=False):
        #chenxin
        # self.adjust(xz)
        if self.aux_loss:
            raise ValueError("Deep supervision is not supported.")
        # Forward the transformer encoder and decoder
        if self.neck_type == 'FPN':
            xz_mem = self.bottleneck(xz)
        else:
            xz_mem = xz[-1].permute(1, 0, 2)
            xz_mem = self.bottleneck(xz_mem)
        output_embed = xz_mem[0:1,:,:].unsqueeze(-2)
        x_mem = xz_mem[1:1+self.num_patch_x]
        # Forward the corner head
        out, outputs_coord = self.forward_box_head(output_embed, x_mem)
        return out, outputs_coord, output_embed

    def forward_box_head(self, hs, memory):
        """
        hs: output embeddings (1, B, N, C)
        memory: encoder embeddings (HW1+HW2, B, C)"""
        if "CORNER" in self.head_type:
            # adjust shape
            enc_opt = memory[-self.feat_len_s:].transpose(0, 1)  # encoder output for the search region (B, HW, C)
            dec_opt = hs.squeeze(0).transpose(1, 2)  # (B, C, N)
            att = torch.matmul(enc_opt, dec_opt)  # (B, HW, N)
            opt = (enc_opt.unsqueeze(-1) * att.unsqueeze(-2)).permute((0, 3, 2, 1)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW)
            # opt = enc_opt.unsqueeze(1).permute(0, 1, 3, 2).contiguous()
            bs, Nq, C, HW = opt.size()
            opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)
            # run the corner head
            outputs_coord = box_xyxy_to_cxcywh(self.box_head(opt_feat))
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new}
            return out, outputs_coord_new
        elif self.head_type == "MLP":
            # Forward the class and box head
            outputs_coord = self.box_head(hs).sigmoid()
            out = {'pred_boxes': outputs_coord[-1]}
            if self.aux_loss:
                out['aux_outputs'] = self._set_aux_loss(outputs_coord)
            return out, outputs_coord


    @torch.jit.unused
    def _set_aux_loss(self, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_boxes': b}
                for b in outputs_coord[:-1]]


def build_vtv(cfg):
    backbone = build_backbone(cfg)  # backbone and positional encoding are built together
    box_head = build_box_head(cfg)
    bottleneck = build_neck(cfg, backbone.num_channels, backbone.body.num_patches_search, backbone.body.embed_dim_list)
    model = VTV(
        backbone,
        box_head,
        bottleneck = bottleneck,
        hidden_dim=cfg.MODEL.HIDDEN_DIM,
        num_queries=cfg.MODEL.NUM_OBJECT_QUERIES,
        aux_loss=cfg.TRAIN.DEEP_SUPERVISION,
        head_type=cfg.MODEL.HEAD_TYPE,
        neck_type=cfg.MODEL.NECK.TYPE
    )

    return model
