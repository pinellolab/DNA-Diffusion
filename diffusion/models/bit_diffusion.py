from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.special import expm1
from tqdm import tqdm

from diffusion.models.bits import bits_to_decimal, decimal_to_bits
from diffusion.models.utils import (alpha_cosine_log_snr, beta_linear_log_snr,
                                    default, log, log_snr_to_alpha_sigma,
                                    right_pad_dims_to)


class BitDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps=1000,
        sampling_fn=None,
        noise_schedule="cosine",
        time_difference=0.0,
        bit_scale=1.0,
        # prob we train an unconditionally
        p_uncond=0.1,
        cond_weight=0.0,
    ):
        super().__init__()
        self.model = model
        self.channels = self.model.channels

        self.image_size = image_size

        if noise_schedule == "linear":
            self.log_snr = beta_linear_log_snr
        elif noise_schedule == "cosine":
            self.log_snr = alpha_cosine_log_snr
        else:
            raise ValueError(f"invalid noise schedule {noise_schedule}")

        self.bit_scale = bit_scale

        self.timesteps = timesteps
        self.sampling_fn = sampling_fn

        # proposed in the paper, summed to time_next
        # as a way to fix a deficiency in self-conditioning and lower FID when the number of sampling timesteps is < 400

        self.time_difference = time_difference
        self.p_uncond = p_uncond
        self.w = cond_weight

    @property
    def device(self):
        return next(self.model.parameters()).device

    def get_sampling_timesteps(self, batch, *, device):
        times = torch.linspace(1.0, 0.0, self.timesteps + 1, device=device)
        times = repeat(times, "t -> b t", b=batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim=0)
        times = times.unbind(dim=-1)
        return times

    @torch.no_grad()
    def ddpm_sample(self, shape, time_difference=None, classes=None):
        batch, device = shape[0], self.device

        time_difference = default(time_difference, self.time_difference)

        time_pairs = self.get_sampling_timesteps(batch, device=device)
        img = torch.randn(shape, device=device)

        x_start = None

        for time, time_next in tqdm(
            time_pairs, desc="sampling loop time step", total=self.timesteps
        ):

            # add the time delay

            time_next = (time_next - self.time_difference).clamp(min=0.0)

            noise_cond = self.log_snr(time)

            # get predicted x0

            x_start = self.model(img, noise_cond, classes, x_start)

            # clip x0

            x_start.clamp_(-self.bit_scale, self.bit_scale)

            # get log(snr)

            log_snr = self.log_snr(time)
            log_snr_next = self.log_snr(time_next)
            log_snr, log_snr_next = map(
                partial(right_pad_dims_to, img), (log_snr, log_snr_next)
            )

            # get alpha sigma of time and next time

            alpha, sigma = log_snr_to_alpha_sigma(log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)

            # derive posterior mean and variance

            c = -expm1(log_snr - log_snr_next)

            mean = alpha_next * (img * (1 - c) / alpha + c * x_start)
            variance = (sigma_next ** 2) * c
            log_variance = log(variance)

            # get noise

            noise = torch.where(
                rearrange(time_next > 0, "b -> b 1 1 1"),
                torch.randn_like(img),
                torch.zeros_like(img),
            )

            img = mean + (0.5 * log_variance).exp() * noise

        return bits_to_decimal(img)

    @torch.no_grad()
    def ddim_sample(self, shape, classes, time_difference=None):
        batch, device = shape[0], self.device

        time_difference = default(time_difference, self.time_difference)

        time_pairs = self.get_sampling_timesteps(batch, device=device)
        img = torch.randn(shape, device=device)

        x_start = None

        for times, times_next in tqdm(time_pairs, desc="sampling loop time step"):

            # get times and noise levels

            log_snr = self.log_snr(times)
            log_snr_next = self.log_snr(times_next)

            padded_log_snr, padded_log_snr_next = map(
                partial(right_pad_dims_to, img), (log_snr, log_snr_next)
            )

            alpha, sigma = log_snr_to_alpha_sigma(padded_log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(padded_log_snr_next)

            # add the time delay

            times_next = (times_next - time_difference).clamp(min=0.0)

            # predict x0

            x_start = self.model(img, log_snr, classes, x_start)

            # clip x0

            x_start.clamp_(-self.bit_scale, self.bit_scale)

            # get predicted noise

            pred_noise = (img - alpha * x_start) / sigma.clamp(min=1e-8)

            # calculate x next

            img = x_start * alpha_next + pred_noise * sigma_next

        return bits_to_decimal(img)

    @torch.no_grad()
    def class_conditioned_sample(self, shape, classes, time_difference=None):
        batch, device = shape[0], self.device

        time_pairs = self.get_sampling_timesteps(batch, device=device)
        z_t = torch.randn(shape, device=device)

        context_mask = torch.zeros(batch)
        context_mask = context_mask[:, None]
        context_mask = context_mask.repeat(2, 1)
        # mask so sample unconditionally
        context_mask[batch:] = 1

        classes = classes.repeat(2, 1)
        for step in tqdm(range(self.timesteps), desc="sampling loop time step"):
            t_now = 1 - step / self.timesteps
            t_next = max(1 - (step + 1) / self.timesteps, 0)

            times = torch.full((batch,), t_now)
            times_next = torch.full((batch,), t_next)

            # double batch so you can sample conditonally and unconditionally and weight
            z_t_doubled = z_t.repeat(2, 1, 1, 1)
            times_t = times.repeat(2)

            # get alpha sigma of time and next time

            x_pred = self.model(z_t_doubled, times_t, classes, None, context_mask)
            x_pred.clamp_(-self.bit_scale, self.bit_scale)

            eps1 = (1 + self.w) * x_pred[:batch]
            eps2 = self.w * x_pred[:batch]
            x_t = eps1 - eps2

            gamma = self.log_snr(times)
            gamma_next = self.log_snr(times_next)
            gamma, gamma_next = map(
                partial(right_pad_dims_to, z_t), (gamma, gamma_next)
            )

            alpha = gamma / gamma_next
            sigma = torch.sqrt(1 - alpha)

            z = torch.randn(shape)

            eps = 1 / torch.sqrt(1 - gamma) * (z_t - torch.sqrt(gamma) * x_t)

            z_t = (
                1
                / torch.sqrt(alpha)
                * (z_t - ((1 - alpha) / torch.sqrt(1 - gamma)) * eps)
                + sigma * z
            )

        return bits_to_decimal(z_t)

    @torch.no_grad()
    def sample(self, batch_size=16, classes=None):
        image_size, channels = self.image_size, self.channels
        if self.sampling_fn == "ddim":
            sample_fn = self.ddim_sample
        elif self.sampling_fn == "ddpm":
            sample_fn = self.ddpm_sample
        else:
            sample_fn = self.class_conditioned_sample

        return sample_fn(
            (batch_size, channels, 4, image_size), classes=classes
        )  # Lucas

    def forward(self, img, class_enc, *args, **kwargs):
        batch, c, h, w, device, img_size, = (
            *img.shape,
            img.device,
            self.image_size,
        )

        times = torch.zeros((batch,), device=device).float().uniform_(0, 0.999)
        # convert image to bit representation

        img = decimal_to_bits(img) * self.bit_scale

        noise = torch.randn_like(img)

        noise_level = self.log_snr(times)
        padded_noise_level = right_pad_dims_to(img, noise_level)
        alpha, sigma = log_snr_to_alpha_sigma(padded_noise_level)

        noised_img = alpha * img + sigma * noise

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        self_cond = None
        # TODO: Does it make sense to self condition with a class?
        # if random() < 0.5:
        #     with torch.no_grad():
        #         self_cond = self.model(noised_img, noise_level, torch.zeros_like(class_enc)).detach_()

        # sample 0 or 1 with prob self.p_uncond
        context_mask = torch.bernoulli(torch.zeros(class_enc.shape[0]) + self.p_uncond)
        # bs x 1
        context_mask = context_mask[:, None]

        pred = self.model(noised_img, noise_level, class_enc, self_cond, context_mask)

        return F.mse_loss(pred, img)
