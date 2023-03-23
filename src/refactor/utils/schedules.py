import math
from math import exp, log

import torch


def beta_linear_log_snr(t: torch.Tensor) -> torch.Tensor:
    return -torch.log(exp(1e-4 + 10 * (t**2)))


def alpha_cosine_log_snr(t: torch.Tensor, s: float = 0.008) -> torch.Tensor:
    # not sure if this accounts for beta being clipped to 0.999 in discrete version
    return -log((torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** -2) - 1, eps=1e-5)


def log_snr_to_alpha_sigma(log_snr) -> torch.Tensor:
    return torch.sqrt(torch.sigmoid(log_snr)), torch.sqrt(torch.sigmoid(-log_snr))


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps, beta_end=0.005) -> torch.Tensor:
    beta_start = 0.0001
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps) -> torch.Tensor:
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps) -> torch.Tensor:
    beta_start = 0.001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
