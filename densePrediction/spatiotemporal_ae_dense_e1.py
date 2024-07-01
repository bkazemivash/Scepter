import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple


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

class AttentionMechanism(nn.Module): 
    """Multi-Head Attention

    Args:
        dim (int): The input/output dimension of token feature.
        n_heads (int, optional): Number of attention heads. Defaults to 6.
    """    
    def __init__(self, dim: int, n_heads: int = 4, p_ratio: float = 0.) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)   
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(
                in_feature=dim, 
                hidden_feature=dim,
                out_feature=dim,
                p=p_ratio,)


    def forward(self, x):
        bs,f,i,j,z = x.shape
        x = x.view(bs, f, -1).swapaxes(1, 2)        
        x = x + self.attn(*[self.norm1(x)]*3)[0]
        x = x + self.mlp(self.norm2(x))
        x = x.swapaxes(2, 1).view(bs,f,i,j,z)
        
        return x
        
class DoubleConv(nn.Module):
    """Basic convolution block.

    Args:
        in_ch (int): Input channel size.
        out_ch (int): Output channel size.
        mid_ch (Union[int, None], optional): Middle channel size. Defaults to None.
        residual (bool, optional): True if this is a residual block. Defaults to False.
    """
    def __init__(self, in_ch: int, out_ch: int, mid_ch: Union[int, None] = None, 
                 residual: bool = False):
        super().__init__()
        self.residual = residual
        if not mid_ch:
            mid_ch = out_ch
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(mid_ch, mid_ch),
            nn.GELU(),
            nn.Conv3d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(out_ch, out_ch),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    """Downscaling block with maxpooling and double conv layers. 

    Args:
        in_ch (int): Input channel size.
        out_ch (int): Output channel size.
        emb_dim (int, optional): Embedding dimenstion of iteration index. Defaults to 256.
    """
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int = 64):
        super().__init__()
        self.ds = nn.MaxPool3d(2)
        self.double_stage = nn.Sequential(
            DoubleConv(in_ch, in_ch, residual=True),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x):
        x = self.ds(x)
        return self.double_stage(x)


class Up(nn.Module):
    """Upscaling block with iteration embedding and double conv layers. 

    Args:
        in_ch (int): Input channel size.
        out_ch (int): Output channel size.
        emb_dim (int, optional): Embedding dimenstion of iteration index. Defaults to 256.
    """
    def __init__(self, in_ch, out_ch,):
        super().__init__()

        self.conv = nn.Sequential(
            DoubleConv(in_ch, in_ch, residual=True),
            DoubleConv(in_ch, out_ch, in_ch // 2),
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2)
        x = self.conv(x)
        return x

class SpatiotemporalAutoEncoder(nn.Module):
    def __init__(self, in_ch, o_ch, ks = 3, embed_dim = 3456, n_head = 4, 
                 depth = 4, sequence_type: str = 'LSTM', p: float = 0.) -> None:
        super().__init__()
        
        self.enc_type = sequence_type
        self.down_stage = nn.ModuleList(
            [
                layer for i in range(depth) for layer in (
                    DoubleConv(in_ch, o_ch) if i == 0 else nn.Identity(),
                    Down(o_ch * 2**i, o_ch * 2**(i+1)),
                    AttentionMechanism(o_ch * 2**(i+1), n_head, p_ratio=p) if i == depth - 1 else nn.Identity(),
                )
            ]
        )
        
        self.temporal_enc = nn.LSTM(embed_dim, embed_dim, 2, batch_first=True) \
            if sequence_type == 'LSTM' else nn.GRU(embed_dim, embed_dim, 2, batch_first=True)

        self.Up_stage = nn.ModuleList(
            [
                layer for i in range(depth) for layer in (
                    Up(o_ch * 2**(depth-i), o_ch * 2**(depth-1-i)), 
                    DoubleConv(o_ch * 2**(depth-1-i), o_ch * 2**(depth-i-1)),
                    Up(o_ch, in_ch) if i == depth - 1 else nn.Identity(),
                )
            ]
        )

    def forward(self, x):
        b, c, t, i, j, z = x.shape
        x = x.permute(0,2,1,3,4,5).reshape(b * t, c, i, j, z) 
        for layer in self.down_stage:
            x = layer(x)
        head_shape = x.shape
        x = x.flatten(1).reshape(b, t, -1)     
        x = self.temporal_enc(x)[0]
        x = x.reshape(head_shape)
        for layer in self.Up_stage:
            x = layer(x)
        x = F.interpolate(x, size=(i, j, z))
        x = x.reshape(b, t, c, i, j, z).permute(0,2,3,4,5,1)        
        return x