"""Implementation of spatiotemporal ViT for brain disease classification"""


import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    """Split volume into patches.

    Args:
        vol_size (int): Size of the volume (Max dim if not square)
        patch_size (int): Size of the patch
        in_chans (int, optional): Number of channels. Defaults to 1.
        embed_dim (int, optional): Size of embedding. Defaults to 768.
    """        
    def __init__(self, img_size, patch_size, in_chans=1, embed_dim=768) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 3
        self.proj = nn.Conv3d(
                in_chans, 
                embed_dim, 
                kernel_size=patch_size, 
                stride=patch_size,)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class Attention(nn.Module):
    """Multi-Head Attention

    Args:
        dim (int): The input/output dimension of token feature.
        n_heads (int, optional): Number of attention heads. Defaults to 12.
        qkv_bias (bool, optional): Enable bias for query, key, and value projections. Defaults to True.
        attn_p (float, optional): Drop out ratio for attention module. Defaults to 0..
        proj_p (float, optional): Drop out ratio for output. Defaults to 0..
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
    """Implementation of Transformer block

    Args:
        dim (int): Dimension of embedding.
        n_heads (int): Number of heads.
        mlp_ratio (float): Determines ratio of emb dim for MLP hidden layer.Defaults to 4.0
        qkv_bias (bool, optional): Enable bias. Defaults to False.
        attn_p (float, optional): Drop out ratio for attn block. Defaults to 0.
        p (float, optional): Drop out ratio for projection. Defaults to 0.
    """        
    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, p=0., attn_p=0.) -> None:
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
    
class VisionTransformer(nn.Module):
    """Implementation of Vision Transformer 

    Args:
        img_size (int): Size of the image (Max dim if not square) Defaults to 384.
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
    """        
    def __init__(self, img_size=384, patch_size=16, in_chans=3, n_classes=1000, embed_dim=768, 
                 depth=12, n_heads=12, mlp_ratio=4., qkv_bias=True, p=0., attn_p=0., num_timepoints=8, 
                 attention_type='divided_space_time') -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(
                img_size=img_size, 
                patch_size=patch_size, 
                in_chans=in_chans, 
                embed_dim=embed_dim,)
        self.cls_token_space = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed_space = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim))
        self.pos_drop = nn.Dropout(p)

        self.spatial_encoder = nn.ModuleList(
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
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        n_samples = x.shape[0]
        x = self.patch_embed(x)
        cls_token_space = self.cls_token_space.expand(n_samples, -1, -1)
        x = torch.cat((cls_token_space, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        cls_token_final = x[:, 0]
        x = self.head(cls_token_final)
        return x
