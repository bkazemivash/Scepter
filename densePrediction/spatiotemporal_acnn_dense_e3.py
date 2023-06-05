""" Implementation of spatiotemporal base cnn model for dense prediction of
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
        self.act = nn.GELU() if enable_activation else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.act(x)
        return x


class FullyPreactivatedResidualUnit(nn.Module):
    def __init__(self,
                 in_ch=1,
                 embed_dim=16,
                 blk_bias=True, 
                 blk_p=0.) -> None:
        
        super().__init__()
        self.norm1 = nn.BatchNorm3d(embed_dim)
        self.act = nn.Sigmoid()
        self.stage1 = nn.Conv3d(in_ch, embed_dim, 1, bias=blk_bias, groups=in_ch)
        self.norm2 = nn.BatchNorm3d(embed_dim)
        self.stage2 = nn.Conv3d(in_ch, embed_dim, 1, bias=blk_bias, groups=in_ch)
        self.drop = nn.Dropout3d(blk_p)

    def forward(self, x):
        x = x + self.stage1(self.act(self.norm1(x)))
        x = x + self.stage2(self.act(self.norm2(x)))
        x = self.drop(x)
        return x
    

class ScepterVoxelwiseEncoder(nn.Module):
    def __init__(self,
                 depth=7,
                 in_ch=1,
                 head_dim=53,
                 is_active=True,
                 encoder_type='space_time_general_view',
                 use_bias=True, 
                 blk_p=0.,
                 p=0.) -> None:
        super().__init__()      
        self.encoder = encoder_type
        self.depth = depth
        self.entry = ProjectionUnit(in_ch, head_dim, use_bias, is_active,)
        self.enc_blocks = nn.ModuleList(
                [
                    FullyPreactivatedResidualUnit(
                        in_ch=head_dim,
                        embed_dim=head_dim,
                        blk_bias=use_bias,
                        blk_p=blk_p
                    )
                    for _ in range(depth)
                ]
            )
        
        self.drop = nn.Dropout3d(p)
        self.head = ProjectionUnit(in_ch=head_dim, 
                             embed_dim=head_dim, 
                             enable_activation=True,
                             enable_bias=False)

    
    def forward(self, x):
        x = self.entry(x)
        for block in self.enc_blocks:
                x = block(x)
        x = self.drop(x)
        x = self.head(x).permute(0,2,3,4,1)
        return x
