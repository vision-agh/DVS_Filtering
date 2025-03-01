import math
import torch
from torch import nn

from omegaconf import DictConfig
from vit_pytorch.max_vit import MaxViT


class MaxVitModel(nn.Module):
    def __init__(self,
                    cfg,
                    num_classes=101):
        super().__init__()

        self.model = MaxViT(
                num_classes = num_classes,
                dim_conv_stem = cfg.dim_conv_stem,               # dimension of the convolutional stem, would default to dimension of first layer if not specified
                dim = cfg.dim,                         # dimension of first layer, doubles every layer
                dim_head = cfg.dim_head,                    # dimension of attention heads, kept at 32 in paper
                depth = (1, 1, 1, 1),             # number of MaxViT blocks per stage, which consists of MBConv, block-like attention, grid-like attention
                window_size = cfg.window_size,                  # window size for block and grids
                mbconv_expansion_rate = cfg.mbconv_expansion_rate,        # expansion rate of MBConv
                mbconv_shrinkage_rate = cfg.mbconv_shrinkage_rate,     # shrinkage rate of squeeze-excitation in MBConv
                dropout = cfg.dropout,                     # dropout
                channels = cfg.channels,                      # number of input channels
)
    def forward(self, x: torch.Tensor):
        x = self.model(x)
        return x

