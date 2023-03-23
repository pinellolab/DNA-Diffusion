from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from dnadiffusion.utils.utils import extract


def q_sample(
    x_start: torch.Tensor,
    t: torch.Tensor,
    sqrt_alphas_cumprod: torch.Tensor,
    sqrt_one_minus_alphas_cumprod: torch.Tensor,
    noise: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape).to(device)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape).to(device)

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


def p_losses(
    denoise_model: nn.Module,
    x_start: torch.Tensor,
    t: torch.Tensor,
    classes: torch.Tensor,
    sqrt_alphas_cumprod_in: torch.Tensor,
    sqrt_one_minus_alphas_cumprod_in: torch.Tensor,
    noise: Optional[torch.Tensor] = None,
    loss_type: str = "l1",
    p_uncond: float = 0.1,
    device: Optional[torch.device] = None,
):
    if noise is None:
        noise = torch.randn_like(x_start)
    x_noisy = q_sample(
        x_start=x_start,
        t=t,
        sqrt_alphas_cumprod=sqrt_alphas_cumprod_in,
        sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod_in,
        noise=noise,
        device=device,
    )  # this is the auto generated noise given t and Noise

    context_mask = torch.bernoulli(torch.zeros(classes.shape[0]) + (1 - p_uncond)).to(device)

    # mask for unconditinal guidance
    classes = classes * context_mask
    # nn.Embedding needs type to be long, multiplying with mask changes type
    classes = classes.type(torch.long)
    t = t.to(device)
    predicted_noise = denoise_model(x_noisy, t, classes)

    if loss_type == "l1":
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == "l2":
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss
