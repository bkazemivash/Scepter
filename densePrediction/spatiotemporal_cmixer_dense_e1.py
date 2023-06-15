""" Implementation of spatiotemporal ConvMixer model for dense prediction of
    brain dynamism.
"""

import torch
import torch.nn as nn
from typing import Tuple
from tools.utils import get_num_patches
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    """Split volume into patches.

    Args:
        img_size (Tuple[int, ...]): Size of the image (3D volume)
        patch_size (int): Size of the patch
        in_chans (int, optional): Number of channels. Defaults to 1.
        embed_dim (int, optional): Size of embedding. Defaults to 768.
    """        
    def __init__(self, img_size: Tuple[int, ...], patch_size: int, in_chans=1, embed_dim=768,) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = get_num_patches(img_size, patch_size)
        self.split = nn.Sequential(
                        nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size,),
                        nn.GELU(),
                        nn.BatchNorm3d(embed_dim),
                        )

    def forward(self, x):
        return self.split(x)
 

class EncoderBlock(nn.Module):
    def __init__(self, dim, kernel_size) -> None:
        super().__init__()
        self.cnv = nn.Conv3d(dim, dim, kernel_size, groups=dim, padding="same")
        self.act = nn.GELU()
        self.bn = nn.BatchNorm3d(dim)
    
    def forward(self, x):
        x = x + self.bn(self.act(self.cnv(x)))
        return x
    

class ScepterConvMixer(nn.Module):
    def __init__(self, img_size: Tuple[int, ...], dim: int,  embed_dim: int, patch_size: int, depth: int, 
                 kernel_size=5, head_dim=(53,62,52), encoding='mixture') -> None:
        self.mixer = nn.Sequential(
        nn.Conv3d(dim, dim, kernel_size=patch_size, stride=patch_size,),
        nn.GELU(),
        nn.BatchNorm3d(dim),
        *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv3d(dim, dim, kernel_size, groups=dim, padding="same"),
                    nn.GELU(),
                    nn.BatchNorm3d(dim)
                )),
                nn.Conv3d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm3d(dim)
        ) for i in range(depth)],
    )


    def forward(self, x):
        return x