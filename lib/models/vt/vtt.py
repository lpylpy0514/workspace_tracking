"""
Basic STARK Model (Spatial-only).
"""
import torch
import math
from torch import nn

from lib.utils.misc import NestedTensor

from .backbone import build_backbone
from .head import build_box_head, MLP
from lib.utils.box_ops import box_xyxy_to_cxcywh
from lib.models.vt.vt import VT


class VTT(VT):
    """ This is the base class for Transformer Tracking """
    def __init__(self, backbone, box_head, hidden_dim, num_queries,
                 aux_loss=False, head_type="CORNER", iou_head=None):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_queries: number of object queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__(backbone, box_head, hidden_dim, num_queries,
                         aux_loss=aux_loss, head_type=head_type)
        self.iou_head = iou_head

    def forward(self, images_list=None, xz=None, mode="backbone", run_box_head=False, run_iou_head=False):
        if mode == "backbone":
            return self.forward_backbone(images_list)
        elif mode == "transformer": # this is head not transformer
            return self.forward_transformer(xz, run_box_head=run_box_head, run_iou_head=run_iou_head)
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

    def forward_transformer(self, xz, run_box_head=False, run_iou_head=False):
        #chenxin
        # self.adjust(xz)
        if self.aux_loss:
            raise ValueError("Deep supervision is not supported.")
        # Forward the transformer encoder and decoder
        xz_mem = xz[-1].permute(1,0,2)
        xz_mem = self.bottleneck(xz_mem)
        output_embed = xz_mem[0:1,:,:].unsqueeze(-2)
        x_mem = xz_mem[1:1+self.num_patch_x]
        # Forward the corner head
        out, outputs_coord = self.forward_head(output_embed, x_mem, run_box_head=run_box_head, run_iou_head=run_iou_head)
        return out, outputs_coord, output_embed

    def forward_head(self, hs, memory, run_box_head=False, run_iou_head=False):
        """
        hs: output embeddings (1, B, N, C)
        memory: encoder embeddings (HW1+HW2, B, C)"""
        out_dict = {}
        if run_iou_head:
            # forward the iou head
            out_dict.update({'pred_iou': self.iou_head(hs).sigmoid()[-1]})
        if run_box_head:
            # forward the box prediction head
            out_dict_box, outputs_coord = self.forward_box_head(hs, memory)
            # merge results
            out_dict.update(out_dict_box)
            return out_dict, outputs_coord
        else:
            return out_dict, None


def build_vtt(cfg):
    backbone = build_backbone(cfg)  # backbone and positional encoding are built together
    box_head = build_box_head(cfg)
    iou_head = MLP(cfg.MODEL.HIDDEN_DIM, cfg.MODEL.HIDDEN_DIM, 1, cfg.MODEL.NLAYER_HEAD)
    model = VTT(
        backbone,
        box_head,
        hidden_dim=cfg.MODEL.HIDDEN_DIM,
        num_queries=cfg.MODEL.NUM_OBJECT_QUERIES,
        aux_loss=cfg.TRAIN.DEEP_SUPERVISION,
        head_type=cfg.MODEL.HEAD_TYPE,
        iou_head=iou_head
    )

    return model
