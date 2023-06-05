""" Implementation of spatiotemporal atrous cnn model for dense prediction of
    brain dynamism.
"""

import torch
from torch import nn, Tensor
from torch.nn.functional import interpolate
from typing import Tuple


class StemUnit(nn.Module):
    def __init__(self, 
                 in_ch=1, 
                 embed_dim=16,
                 enable_bias=False,
                 is_bottleneck=False) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=1, bias=enable_bias)
        self.act = nn.GELU() if is_bottleneck else nn.ReLU()

    def forward(self, x):
        x = self.proj(x)
        x = self.act(x)
        return x
    

class GlobalView(nn.Module):
    def __init__(self,
                 embed_dim=16,
                 ) -> None:
        super().__init__()

        self.norm = nn.BatchNorm2d(embed_dim)
        self.pool = nn.AvgPool2d(2)
        self.act = nn.GELU()

    def forward(self, x):
        _, _, t, v = x.shape
        x = self.pool(self.act(self.norm(x)))
        x = interpolate(x, size=(t,v), mode='bicubic', align_corners=True,)
        return x
    

class AtrousFullyPreactivatedResidualUnit(nn.Module):
    def __init__(self,
                 in_ch=1,
                 embed_dim=16,
                 atrous_unit_kernel=3,
                 atrous_ratio=1,
                 atrous_bias=True, 
                 atrous_p=0.) -> None:
        
        super().__init__()
        self.norm1 = nn.BatchNorm2d(embed_dim)
        self.act = nn.GELU()
        self.stage1 = nn.Conv2d(in_ch, embed_dim, atrous_unit_kernel, bias=atrous_bias, padding=atrous_ratio, dilation=atrous_ratio)
        self.norm2 = nn.BatchNorm2d(embed_dim)
        self.stage2 = nn.Conv2d(in_ch, embed_dim, atrous_unit_kernel, bias=atrous_bias, padding=atrous_ratio, dilation=atrous_ratio)
        self.drop = nn.Dropout2d(atrous_p)

    def forward(self, x):
        x = x + self.stage1(self.act(self.norm1(x)))
        x = x + self.stage2(self.act(self.norm2(x)))
        x = self.drop(x)
        return x


class FullyPreactivatedBaseUnit(nn.Module):
    def __init__(self,
                 in_ch=1,
                 embed_dim=16,
                 unit_kernel=3,
                 unit_bias=True, 
                 unit_p=0.) -> None:        
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dim)
        self.act = nn.GELU()
        self.stage = nn.Conv2d(in_ch, embed_dim, unit_kernel, bias=unit_bias, padding=1,)
        self.drop = nn.Dropout2d(unit_p)

    def forward(self, x):
        x = self.act(self.norm(x))
        x = x + self.stage(x)
        x = self.drop(x)
        return x


class ScepterAtrousPyramidEncoder(nn.Module):
    def __init__(self,
                 depth=3,
                 in_ch=1,
                 dim=16,
                 head_dim=53,
                 encoder_type='space_time_general_view',
                 atrous_bias=False, 
                 blk_p=0.,
                 p=0.) -> None:
        super().__init__()      
        self.encoder = encoder_type
        self.depth = depth
        self.entry = StemUnit(in_ch, dim, False,)
        if encoder_type == 'space_time_general_view':
            self.enc_blocks = nn.ModuleList(
                [
                    FullyPreactivatedBaseUnit(
                        in_ch=dim,
                        embed_dim=dim,
                        unit_kernel=3,
                        unit_bias=atrous_bias,
                        unit_p=blk_p
                    )
                    for _ in range(depth)
                ]
            )
        else:
            self.enc_blocks = nn.ModuleList(
                [
                    AtrousFullyPreactivatedResidualUnit(
                        in_ch=dim,
                        embed_dim=dim,
                        atrous_unit_kernel=3,
                        atrous_ratio=2**(i+1),
                        atrous_bias=atrous_bias,
                        atrous_p=blk_p,
                    )
                    for i in range(depth)
                ]
            )
            self.enc_blocks.append(GlobalView(embed_dim=dim))            
        
        self.drop = nn.Dropout2d(p)
        self.head = StemUnit(in_ch=dim if encoder_type == 'space_time_general_view' else dim*(depth+1), 
                             embed_dim=head_dim, 
                             is_bottleneck=True,
                             enable_bias=False)

    
    def forward(self, x):
        x = self.entry(x)
        if self.encoder == 'space_time_general_view':
            for block in self.enc_blocks:
                x = block(x)
        else:
            assert self.depth < 5, 'Depth size of the model must be in range of [1..4]'
            x = [block(x) for block in self.enc_blocks]
            x = torch.cat(x, 1)

        x = self.drop(x)
        x = self.head(x).permute(0,2,3,1)
        return x
