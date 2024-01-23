""" Implementation of spatiotemporal cnn model for dense prediction of
    brain dynamism.
"""

import torch
from torch import nn, Tensor
from torch.nn.functional import interpolate
from typing import Tuple


class ProjectionUnit(nn.Module):
    def __init__(self, 
                 in_ch=1, 
                 embed_dim=16,
                 enable_bias=True,
                 enable_activation=True) -> None:
        super().__init__()
        self.proj = nn.Conv3d(in_ch, embed_dim, kernel_size=1, bias=enable_bias)
        self.act = nn.ReLU6() if enable_activation else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.act(x)
        return x
