"""
Implementation of spatiotemporal Vision Transfor (ViT) classifier with 
space_time, encoder_factorization, joint_attention_factorization, and
parallel_attention_factorization.
"""


import torch
import torch.nn as nn
from typing import Tuple, Union
from tools.utils import get_num_patches

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
    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0., proj_p=0.) -> None:
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
    def __init__(self, in_feature, hidden_feature, out_feature, p=0.) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_feature, hidden_feature)
        self.act = nn.GELU()
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
    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, p=0., attn_p=0.,) -> None:
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


class ScepterVisionTransformer(nn.Module):
    """Implementation of Vision Transformer 

    Args:
        img_size (Tuple[int, ...]): Size of the image with channel (5D tensor)
        patch_size (int, optional): Size of the patch. Defaults to 16.
        in_chans (int, optional): Number of channels. Defaults to 3.
        n_classes (int, optional): Number of classes. Defaults to 1000.
        embed_dim (int, optional): Dimension of embedding. Defaults to 768.
        depth (int, optional): Number of Transformer blocks. Defaults to 12.
        n_heads (int, optional): Number of attention heads. Defaults to 12.
        mlp_ratio (float, optional): Drop out ratio of MLP. Defaults to 4.
        qkv_bias (bool, optional): Enable bias. Defaults to True.
        p (float, optional): Drop out ratio of ViT. Defaults to 0.
        attn_p (float, optional): Dropo ut ratio of attention heads. Defaults to 0.
        attn_type (str, optional): Spatiotemporal encoding strategy. Defaults to 'space_time'
        n_timepoints (int, optional): Number of timepoints. Defaults to 490.
    """        
    def __init__(self, img_size, patch_size=7, in_chans=1, n_classes=1000, embed_dim=768, 
                 depth=2, n_heads=12, mlp_ratio=4., qkv_bias=True, p=0., attn_p=0.,
                 attn_type='space_time', n_timepoints=490) -> None:
        super().__init__()
        self.attention_type = attn_type
        self.time_dim = n_timepoints
        self.patch_embed = PatchEmbed(
                img_size=img_size, 
                patch_size=patch_size, 
                in_chans=in_chans, 
                embed_dim=embed_dim,)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim))
        if attn_type == 'space_time':
            self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.n_patches * self.time_dim, embed_dim))
        self.pos_drop = nn.Dropout(p)
        self.enc_blocks = nn.ModuleList(
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
        if attn_type in ['joint_encoders', 'sequential_encoders', 'parallel_encoders']:
            self.cls_token_temporal = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed_temporal = nn.Parameter(torch.zeros(1, 1 + self.time_dim, embed_dim))            
            self.temporal_encoder = EncoderBlock(
                                        dim=embed_dim, 
                                        n_heads=n_heads, 
                                        mlp_ratio=mlp_ratio, 
                                        qkv_bias=qkv_bias, 
                                        p=p, 
                                        attn_p=attn_p,)

        if attn_type == 'parallel_encoders':
            self.cls_token_fusion = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed_fusion = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.n_patches + self.time_dim, embed_dim))            
            self.fusion_encoder = EncoderBlock(
                                        dim=embed_dim, 
                                        n_heads=n_heads, 
                                        mlp_ratio=mlp_ratio, 
                                        qkv_bias=qkv_bias, 
                                        p=p, 
                                        attn_p=attn_p,)
            
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        b, c, t, i, j, z = x.shape
        x = x.permute(0,2,1,3,4,5).reshape(b * t, c, i, j, z)            
        x = self.patch_embed(x)
        if self.attention_type == 'parallel_encoders':
            y = x.clone()
        n_samples, n_patch, embbeding_dim = x.shape        
        if self.attention_type == 'space_time':
            n_samples //= self.time_dim 
            n_patch *= self.time_dim
            x = torch.reshape(x, (n_samples, n_patch, embbeding_dim))

        cls_token = self.cls_token.expand(n_samples, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)        
        for block in self.enc_blocks:
            x = block(x)
        
        if self.attention_type == 'joint_encoders':
            x = x[:, 0]
            n_samples //= self.time_dim
            n_patch = self.time_dim
            x = torch.reshape(x, (n_samples, n_patch, embbeding_dim))
            cls_token_temporal = self.cls_token_temporal.expand(n_samples, -1, -1)
            x = torch.cat((cls_token_temporal, x), dim=1)
            x = x + self.pos_embed_temporal
            x = self.pos_drop(x)        
            x = self.temporal_encoder(x)
            
        if self.attention_type == 'sequential_encoders':
            x = x[:,1:]
            n_samples //= self.time_dim 
            x = x.reshape(n_samples, self.time_dim, n_patch, embbeding_dim).permute(0,2,1,3)
            n_samples *= n_patch
            x = torch.reshape(x, (n_samples, self.time_dim, embbeding_dim))
            cls_token_temporal = self.cls_token_temporal.expand(n_samples, -1, -1)
            x = torch.cat((cls_token_temporal, x), dim=1)
            x = x + self.pos_embed_temporal
            x = self.pos_drop(x)
            x = self.temporal_encoder(x)
            x = x[:, 0]
            n_samples //= n_patch
            x = torch.reshape(x, (n_samples, n_patch, embbeding_dim)).sum(dim=1, keepdim=True)

        if self.attention_type == 'parallel_encoders':
            x = x[:,0]
            n_samples //= self.time_dim
            x = torch.reshape(x, (n_samples, self.time_dim, embbeding_dim))

            n_samples, n_patch, embbeding_dim = y.shape  
            n_samples //= self.time_dim 
            y = y.reshape(n_samples, self.time_dim, n_patch, embbeding_dim).permute(0,2,1,3)
            n_samples *= n_patch
            y = torch.reshape(y, (n_samples, self.time_dim, embbeding_dim))
            cls_token_temporal = self.cls_token_temporal.expand(n_samples, -1, -1)
            y = torch.cat((cls_token_temporal, y), dim=1)
            y = y + self.pos_embed_temporal
            y = self.pos_drop(y)
            y = self.temporal_encoder(y)
            y = y[:,0]
            n_samples //= n_patch
            y = torch.reshape(y, (n_samples, n_patch, embbeding_dim))

            x = torch.cat((x, y), dim=1)
            token_fusion = self.cls_token_fusion.expand(n_samples, -1, -1)
            x = torch.cat((token_fusion, x), dim=1)
            x = x + self.pos_embed_fusion
            x = self.pos_drop(x)
            x = self.fusion_encoder(x)

        x = self.norm(x)
        cls_token_final = x[:, 0]
        x = self.head(cls_token_final)
        return x

