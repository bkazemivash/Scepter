"""
Implementation of spatiotemporal diffusion model for dense prediction of 
dynamic brain networks along with implementation of conditional UNet as the backbone.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union


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
    def __init__(self, dim: int, n_heads: int = 4,) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)   
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(
                in_feature=dim, 
                hidden_feature=dim,
                out_feature=dim,)


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
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_ch, in_ch, residual=True),
            DoubleConv(in_ch, out_ch),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim,out_ch),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        _, _, i, j, z = x.shape
        emb = self.emb_layer(t)[:, :, None, None, None].repeat(1, 1, i, j, z)
        return x + emb


class Up(nn.Module):
    """Upscaling block with iteration embedding and double conv layers. 

    Args:
        in_ch (int): Input channel size.
        out_ch (int): Output channel size.
        emb_dim (int, optional): Embedding dimenstion of iteration index. Defaults to 256.
    """
    def __init__(self, in_ch, out_ch, emb_dim=64):
        super().__init__()

        self.conv = nn.Sequential(
            DoubleConv(in_ch, in_ch, residual=True),
            DoubleConv(in_ch, out_ch, in_ch // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_ch),
        )

    def forward(self, x, skip_x, t):
        x = F.interpolate(x, size=skip_x.shape[2:])
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        _, _, i, j, z = x.shape
        emb = self.emb_layer(t)[:, :, None, None, None].repeat(1, 1, i, j, z)
        return x + emb


class ConditionalUNet(nn.Module):
    def __init__(self, in_ch=10, out_ch=10, emb_dim=64):
        super().__init__()
        self.it_emb_dim = emb_dim
        self.inc = DoubleConv(in_ch, 16)
        self.down1 = Down(16, 32)
        self.sa1 = AttentionMechanism(32, 4)
        self.down2 = Down(32, 64)
        self.sa2 = AttentionMechanism(64, 8)
        self.down3 = Down(64, 64)
        self.sa3 = AttentionMechanism(64, 8)

        self.bot1 = DoubleConv(64, 128)
        self.bot2 = DoubleConv(128, 128, residual=True)
        self.bot3 = DoubleConv(128, 64)

        self.up1 = Up(128, 32)
        self.sa4 = AttentionMechanism(32, 4)
        self.up2 = Up(64, 16)
        self.sa5 = AttentionMechanism(16, 4)
        self.up3 = Up(32, 10)
        self.sa6 = AttentionMechanism(10, 1)
        self.head = nn.Conv3d(10, out_ch, kernel_size=1)  
          
    @torch.no_grad
    def pos_encoding(self, t, channels):
        dev = next(self.parameters()).device
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2, device=dev).float() / channels))
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.it_emb_dim)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)

        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)

        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        x = self.head(x)

        return x


class DiffusionModel(nn.Module):
    def __init__(self, backbone_arch: str = 'C-UNet', noise_step: int = 1000, beta_begin: float = 1e-4, 
                 beta_end: float = 0.02, img_size: tuple = (10, 33, 43, 32), ) -> None:
        super().__init__()
        self.noise_step = noise_step
        self.img_size = img_size

        self.beta = torch.linspace(beta_begin, beta_end, noise_step)    
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.backbone = ConditionalUNet(in_ch=img_size[0], out_ch=img_size[0]) if backbone_arch == 'C-UNet' else None
    
    @torch.no_grad
    def noisy_image(self, x: torch.Tensor, t: torch.tensor,) -> torch.Tensor:
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1. - self.alpha_hat[t])[:, None, None, None]
        epsilon =torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon
    
    @torch.no_grad    
    def sample_timesteps(self, n: int, ) -> torch.Tensor:
        return torch.randint(low = 1, high = self.noise_step, size=(n,))
    
    @torch.no_grad  
    def sample(self, condition: torch.Tensor, n):
        self.backbon.eval()
        x= torch.rand((n,) + self.img_size)
        for i in reversed(range(1, self.noise_steps)):
            t = (torch.ones(n) * i).long()
            predicted_noise = self.backbone(x,t)
            alpha = self.alpha[t][:, None, None, None]
            alpha_hat = self.alpha_hat[t][:, None, None,None]
            beta = self.beta[t][:, None, None, None]
        if i> 1:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)
        x = 1. / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        self.backbon.train()

        return x
    
    def forward(self, x: torch.Tensor):
        assert self.backbone != None, ValueError('This backbone architecutre is not supported yet!')
        x = F.interpolate(x, self.img_size)
        t = self.sample_timesteps(x.shape[0])
        x, noise = self.noisy_image(x, t)
        x = self.backbone(x, t)
        return x, noise