"""
Implementation of spatiotemporal diffusion model for dense prediction of 
dynamic brain networks with the modified version of UNet as the backbone 
of the our model.
"""


import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple, Union


class DoubleConv(nn.Module):
    """Basic block for UNet with residual option.

    Args:
        in_ch (int): Input channel size.
        o_ch (int): Output channel size.
        md_ch (Union[int, None], optional): Middle channel size in not None. Defaults to None.
        residual (bool, optional): If the block is residual block or not. Defaults to False.
    """    
    def __init__(self, in_ch: int, o_ch: int, md_ch: Union[int, None] = None, residual: bool = False) -> None:
        super().__init__()
        self.residual = residual
        if not md_ch:
            md_ch = o_ch
        
        self.stage = nn.Sequential(
            nn.Conv2d(in_ch, md_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, md_ch),
            nn.GELU(),
            nn.Conv2d(md_ch, o_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, o_ch),
        )
    
    def forward(self, x: torch.Tensor):
        if self.residual:
            return F.gelu(x + self.stage(x))        
        return self.stage(x)


class DownFlow(nn.Module):
    def __init__(self, in_ch: int, o_ch: int, embed_dim: int = 256,) -> None:
        super().__init__()
        self.stage = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, in_ch, residual=True),
            DoubleConv(in_ch, o_ch,),
        )

        self.embedding = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, o_ch,),
        )

    def forward(self, x: torch.Tensor, t: int):
        x = self.stage(x)
        return x + self.embedding(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])


class UpFlow(nn.Module):
    def __init__(self, in_ch: int, o_ch: int, embed_dim: int = 256,) -> None:
        super().__init__()
        self.up_scale = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.stage = nn.Sequential(
            DoubleConv(in_ch, in_ch, residual=True),
            DoubleConv(in_ch, o_ch, o_ch // 2),
        )
        self.embedding = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, o_ch,),
        )

    def forward(self, x: torch.Tensor, skip_con: torch.Tensor, t: int):
        x = self.up_scale(x)
        x = torch.cat([skip_con, x], dim=1)
        x = self.stage(x)
        return x + self.embedding(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])


class MLP(nn.Module):
    """Implementation of Multilayer perceptron.

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
    def __init__(self, dim: int, head_size: int,) -> None:
        super().__init__()
        self.dim = dim
        self.head_size = head_size
        self.attn = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(dim, dim, dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x.swapaxis(2,1).view(-1, self.dim, self.head_size, self.head_size)


class UNet(nn.Module):
    def __init__(self, c_in: int, c_out: int, iter: int = 256,) -> None:
        super().__init__()
        self.iter_dim = iter
        self.inc = DoubleConv(c_in, 64)
        self.down1 = DownFlow(64, 128)
        self.attn1 = AttentionMechanism(128, 32)
        self.down2 = DownFlow(128, 256)
        self.attn2 = AttentionMechanism(256, 16)
        self.down3 = DownFlow(256, 256)
        self.attn3 = AttentionMechanism(256, 8)

        self.btlk1 = DoubleConv(256, 512)
        self.btlk2 = DoubleConv(512, 512)
        self.btlk3 = DoubleConv(512, 256)         

        self.up1 = UpFlow(512, 128)
        self.attn4 = AttentionMechanism(128, 16)
        self.up2 = UpFlow(256, 64)
        self.attn5 = AttentionMechanism(64, 32)
        self.up3 = UpFlow(128, 64)
        self.attn6 = AttentionMechanism(64, 64)
        self.head = nn.Conv2d(64, c_out, kernel_size=1)                

    
    def position_embedding(self, t: int, embed_dim: int, ) -> torch.Tensor:
        _div_term =  1.0 / (10_000 ** (torch.arange(0, embed_dim, 2, dtype=torch.float) / embed_dim))
        pos_enc_sin = torch.sin(t.repeat(1, embed_dim // 2) * _div_term)
        pos_enc_cos = torch.cos(t.repeat(1, embed_dim // 2) * _div_term)
        return torch.cat([pos_enc_sin, pos_enc_cos], dim=-1)

    def forward(self, x:torch.Tensor, t: int) -> torch.Tensor:
        t = torch.unsqueeze(-1).type(torch.float)
        t =self.position_embedding(t, self.iter_dim)

        res1 = self.inc(x)
        res2 = self.down1(res1, t)
        res2 = self.attn1(res2)
        res3 = self.down2(res2, t)
        res3 = self.attn2(res3)
        res4 = self.down3(res3, t)
        res4 = self.attn3(res4)

        res4 = self.btlk1(res4)
        res4 = self.btlk2(res4)
        res4 = self.btlk3(res4)

        x = self.up1(res4, res3, t)
        x = self.attn4(x)
        x = self.up2(x, res2, t)
        x = self.attn5(x)
        x = self.up3(x, res1, t)
        x = self.attn6(x)
        x = self.head(x)
        return x


class DiffusionModel(nn.Module):
    def __init__(self, noise_step: int = 1000, beta_begin: float = 1e-4, beta_end: float = 0.02, 
                 img_size: int = 64, ) -> None:
        super().__init__()
        self.noise_step = noise_step
        self.beta_begin = beta_begin
        self.beta_end = beta_end
        self.img_size = img_size

        self.beta = self.initialize_noise_schedule()    
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def initialize_noise_schedule(self,) -> torch.Tensor:
        return torch.linspace(self.beta_begin, self.beta_end, self.noise_step)
    
    def noisy_image(self, x: torch.Tensor, t: int,) -> torch.Tensor:
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1. - self.alpha_hat[t])[:, None, None, None]
        epsilon =torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon
    
    def sample_timesteps(self, n: int, ) -> torch.Tensor:
        return torch.randint(low = 1, high = self.noise_step, size=(n,))
    
    
    
