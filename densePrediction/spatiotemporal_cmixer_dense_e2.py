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
        separable_ratio (int, optional): Separable kernels for input channels. Defaults to 1.
    """        
    def __init__(self, img_size: Tuple[int, ...], patch_size: int, in_chans=1, 
                 embed_dim=768, separable_ratio=1) -> None:  
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = get_num_patches(img_size, patch_size)
        self.split = nn.Sequential(
                        nn.BatchNorm3d(embed_dim),
                        nn.GELU(),
                        nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, groups=separable_ratio),
                        )

    def forward(self, x):
        return self.split(x)


class EncoderBlock(nn.Module):
    """Implementation of Encoder block

    Args:
        dim (int): Dimension of embedding.
    """        
    def __init__(self, dim, kernel_size, enc_p=0.,) -> None:
        super().__init__()
        self.depthwise_enc = nn.Sequential(
                                    nn.BatchNorm3d(dim),                                
                                    nn.GELU(),
                                    nn.Conv3d(dim, dim, kernel_size, groups=dim, padding="same"),
                                    nn.BatchNorm3d(dim),                                
                                    nn.GELU(),
                                    nn.Conv3d(dim, dim, kernel_size, groups=dim, padding="same"),                                    
                                    )
        self.pointwise_enc = nn.Sequential(
                                    nn.Conv3d(dim, dim, kernel_size=1),
                                    nn.BatchNorm3d(dim),                                                                                                                                                      
                                    nn.GELU(),
                                    )
        self.pos_drop = nn.Dropout(enc_p)

    def forward(self, x):
        x = x + self.depthwise_enc(x)
        x = self.pointwise_enc(x)
        x = self.pos_drop(x)
        return x
    

class ScepterConvMixer(nn.Module):
    def __init__(self, patch_size=3, time_dim=25, kernel_size=3, depth=7, 
                 encoder_type='space_time_mixture', head_dim=(53,63,52), p=0.,) -> None:
        super().__init__()
        self.patches = PatchEmbed(img_size=head_dim, 
                                      patch_size=patch_size, 
                                      in_chans=time_dim, 
                                      embed_dim=time_dim,
                                      separable_ratio=time_dim)
        self.enc_blocks = nn.ModuleList(
            [
                EncoderBlock(
                    dim=time_dim,
                    kernel_size=kernel_size,
                    enc_p=p)
                for _ in range(depth)
            ]
        )


    def forward(self, x):
        x = self.patches(x)
        for block in self.enc_blocks:
            x = block(x)
        x = F.interpolate(x, size=tuple(self.patches.img_size)).permute(0,2,3,4,1)

        return x