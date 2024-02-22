"""
Implementation of spatiotemporal diffusion model for dense prediction of 
dynamic brain networks.
"""


import torch
from torch import nn, Tensor
from torch.nn.functional import interpolate
from typing import Tuple


class DiffusionModel(nn.Module):
    def __init__(self, base_mdl: nn.Module, noise_step: int = 1000, beta_begin: float = 1e-4, beta_end: float = 0.02, 
                 img_size: int = 64, ) -> None:
        super().__init__()
        self.noise_step = noise_step
        self.img_size = img_size
        self.backbone = base_mdl

        self.beta = torch.linspace(beta_begin, beta_end, noise_step)    
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
    
    def noisy_image(self, x: torch.Tensor, t: int,) -> torch.Tensor:
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1. - self.alpha_hat[t])[:, None, None, None]
        epsilon =torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon
    
    def sample_timesteps(self, n: int, ) -> torch.Tensor:
        return torch.randint(low = 1, high = self.noise_step, size=(n,))