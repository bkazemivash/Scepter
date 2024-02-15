"""
Implementation of spatiotemporal Vision Transfor for dense prediction task with 
space_time and sequential_encoders scenarios for encoding both space and time dimension.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple
from tools.utils import configure_patch_embedding, tuple_prod

class PatchEmbed(nn.Module):
    """Split volume into patches.

    Args:
        img_size (Tuple[int, ...]): Size of the image (3D volume)
        patch_size (int): Size of the patch
        down_ratio (float): Ratio of down sampling
        in_chans (int, optional): Number of channels. Defaults to 1.
        embed_dim (int, optional): Size of embedding. Defaults to 768.
    """        
    def __init__(self, img_size: Tuple[int, ...], patch_size: int, down_ratio: float =1., 
                 in_chans: int = 1, embed_dim: int =768,) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.up_head, self.n_patches, self.sample_shape = configure_patch_embedding(img_size, patch_size, down_ratio)
        self.proj = nn.Conv3d(
                in_chans, 
                embed_dim, 
                kernel_size=patch_size, 
                stride=patch_size,)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Attention(nn.Module):
    """Multi-Head Attention

    Args:
        dim (int): The input/output dimension of token feature.
        n_heads (int, optional): Number of attention heads. Defaults to 12.
        qkv_bias (bool, optional): Enable bias for query, key, and value projections. Defaults to True.
        attn_p (float, optional): Drop out ratio for attention module. Defaults to 0.
        proj_p (float, optional): Drop out ratio for output. Defaults to 0.
    """            
    def __init__(self, dim: int, n_heads: int = 12, qkv_bias: bool = True, attn_p: float = 0., 
                 proj_p: float = 0.) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3 , bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)
    
    def forward(self, x):
        n_samples, n_tokens, dim = x.shape
        assert dim ==  self.dim, ValueError('In/Out dimension mismatch!')
        qkv = self.qkv(x)
        qkv = qkv.reshape(
            n_samples, n_tokens, 3, self.n_heads, self.head_dim
            ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv
        dot_product = (query @ key.transpose(-2, -1)) * self.scale
        attn = dot_product.softmax(dim=-1)
        attn = self.attn_drop(attn)
        weighted_avg = (attn @ value).transpose(1, 2).flatten(2)
        x = self.proj(weighted_avg)
        x = self.proj_drop(x)
        return x
        

class MLP(nn.Module):
    """Implementation of Multilayer perceptron

    Args:
        in_feature (int): Number of in_feature
        hidden_feature (int): Number of hidden_feature
        out_feature (int): Number of out_feature
        p (float, optional): Drop out ratio. Defaults to 0.
    """        
    def __init__(self, in_feature: int , hidden_feature: int , out_feature: int, p: float = 0.) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_feature, hidden_feature)
        self.act = nn.RReLU(lower=0.1, upper=0.7)
        self.fc2 = nn.Linear(hidden_feature, out_feature)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class EncoderBlock(nn.Module):
    """Implementation of Encoder block

    Args:
        dim (int): Dimension of embedding.
        n_heads (int): Number of heads.
        mlp_ratio (float): Determines ratio of emb dim for MLP hidden layer.Defaults to 4.0
        qkv_bias (bool, optional): Enable bias. Defaults to False.
        attn_p (float, optional): Drop out ratio for attn block. Defaults to 0.
        p (float, optional): Drop out ratio for projection. Defaults to 0.
    """        
    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 4.0, qkv_bias: bool = True, 
                 p: float = 0., attn_p: float = 0.,) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
                dim, 
                n_heads=n_heads,
                qkv_bias=qkv_bias,
                attn_p=attn_p, 
                proj_p=p,)
        
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(
                in_feature=dim, 
                hidden_feature=hidden_features,
                out_feature=dim,)


    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PositionalEncoding(nn.Module):
    """Implementation of position embedding for tuning temporal variations
        between differnet timepoints.

    Args:
        embed_dim (int): Dimension of embedding.
        max_len (int, optional): Length of sequence. Defaults to 10.
        dropout (float, optional): Drop out ratio for Position Encoding block. Defaults to 0.1.
    """    
    def __init__(self, embed_dim: int, max_len: int = 10,):
        super().__init__()
        self.ac = nn.Tanhshrink()
        _position = torch.arange(max_len).unsqueeze(1)
        _div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float) * (-math.log(100_000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(_position * _div_term)
        pe[:, 1::2] = torch.cos(_position * _div_term)
        
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe
        x = self.ac(x)
        return x


class ScepterVisionTransformer(nn.Module):
    """Implementation of Vision Transformer 

    Args:
        img_size (Tuple[int, ...]): Size of the image with channel (5D tensor)
        patch_size (int, optional): Size of the patch. Defaults to 16.
        in_chans (int, optional): Number of channels. Defaults to 3.
        embed_dim (int, optional): Dimension of embedding. Defaults to 768.
        down_sample_ratio (float, optional): Ratio of down sampling for input image.
        depth (int, optional): Number of Transformer blocks. Defaults to 12.
        n_heads (int, optional): Number of attention heads. Defaults to 12.
        mlp_ratio (float, optional): Drop out ratio of MLP. Defaults to 4.
        qkv_bias (bool, optional): Enable bias. Defaults to True.
        p (float, optional): Drop out ratio of ViT. Defaults to 0.
        attn_p (float, optional): Dropo ut ratio of attention heads. Defaults to 0.
        attn_type (str, optional): Spatiotemporal encoding strategy. Defaults to 'space_time'
        n_timepoints (int, optional): Number of timepoints. Defaults to 490.
    """        
    def __init__(self, img_size: Tuple, patch_size: int = 7, in_chans: int = 1, embed_dim: int = 768, 
                 down_sample_ratio: float = 1., depth: int = 2, n_heads: int = 12, mlp_ratio: float = 4., 
                 qkv_bias: bool = True, p: float = 0., attn_p: float = 0., attn_type: str = 'space_time', 
                 n_timepoints: int = 490) -> None:
        super().__init__()
        self.attention_type = attn_type
        self.down_sampling_ratio = down_sample_ratio
        self.time_dim = n_timepoints
        self.patch_embed = PatchEmbed(
                img_size=img_size, 
                patch_size=patch_size,
                down_ratio=down_sample_ratio, 
                in_chans=in_chans, 
                embed_dim=embed_dim,)

        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches * self.time_dim, embed_dim))
        self.pos_drop = nn.Dropout(p)
        self.spatial_enc_blocks = nn.ModuleList(
            [
                EncoderBlock(
                    dim=embed_dim, 
                    n_heads=n_heads, 
                    mlp_ratio=mlp_ratio, 
                    qkv_bias=qkv_bias, 
                    p=p, 
                    attn_p=attn_p,)
                for _ in range(depth)
            ]
        )

        if attn_type in ['sequential_encoders']:
            self.spatial_pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches, embed_dim))
            self.pos_embed_temporal = nn.Parameter(torch.zeros(1, self.time_dim, embed_dim))                        
            self.temporal_encoder = EncoderBlock(
                                        dim=embed_dim, 
                                        n_heads=n_heads, 
                                        mlp_ratio=mlp_ratio, 
                                        qkv_bias=qkv_bias, 
                                        p=p, 
                                        attn_p=attn_p,)
            
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.variation_token = PositionalEncoding(tuple_prod(self.patch_embed.img_size), max_len=self.time_dim,)

    def forward(self, x):
        if self.down_sampling_ratio != 1.0:
            x = x.squeeze()
            x = F.interpolate(x, size=self.patch_embed.sample_shape, mode='nearest')
            x = x.unsqueeze(1)
        b, c, t, i, j, z = x.shape
        x = x.permute(0,2,1,3,4,5).reshape(b * t, c, i, j, z)            
        x = self.patch_embed(x)
        n_samples, n_patch, embbeding_dim = x.shape   

        if self.attention_type == 'space_time':
            n_samples //= self.time_dim 
            n_patch *= self.time_dim
            x = torch.reshape(x, (n_samples, n_patch, embbeding_dim))            
        else:
            n_samples //= self.time_dim 
            x = x.reshape(n_samples, self.time_dim, n_patch, embbeding_dim).permute(0,2,1,3)
            n_samples *= n_patch
            x = torch.reshape(x, (n_samples, self.time_dim, embbeding_dim))

            x = x + self.pos_embed_temporal
            x = self.pos_drop(x)
            x = self.temporal_encoder(x)

            n_samples //= n_patch 
            x = x.reshape(n_samples, n_patch, self.time_dim, embbeding_dim).permute(0,2,1,3)
            n_samples *= self.time_dim
            x = torch.reshape(x, (n_samples, n_patch, embbeding_dim))             

        x = x + self.spatial_pos_embed
        x = self.pos_drop(x)        
        for block in self.spatial_enc_blocks:
            x = block(x)

        if self.attention_type == 'space_time':
            n_patch //= self.time_dim
        else:
            n_samples //= self.time_dim

        x = self.norm(x)        
        x = x.reshape(n_samples, self.time_dim, n_patch, -1).flatten(2)
        x = F.interpolate(x, torch.prod(torch.tensor(self.patch_embed.img_size)), mode='nearest')
        x = self.variation_token(x)   
        x = x.reshape(n_samples, 1, self.time_dim, *self.patch_embed.img_size).permute(0,1,3,4,5,2)

        return x
