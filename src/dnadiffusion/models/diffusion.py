from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

from dnadiffusion.utils.utils import default, extract, linear_beta_schedule


class Diffusion(nn.Module):
    def __init__(
        self,
        model,
        timesteps,
    ):
        super().__init__()
        self.model = model
        self.timesteps = timesteps

        # Diffusion params
        betas = linear_beta_schedule(timesteps, beta_end=0.2)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Store as buffers
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )

    @property
    def device(self):
        return self.betas.device

    @torch.no_grad()
    def sample(self, classes, shape, cond_weight):
        return self.p_sample_loop(
            classes=classes,
            image_size=shape,
            cond_weight=cond_weight,
        )

    @torch.no_grad()
    def sample_cross(self, classes, shape, cond_weight):
        return self.p_sample_loop(
            classes=classes,
            image_size=shape,
            cond_weight=cond_weight,
            get_cross_map=True,
        )

    @torch.no_grad()
    def p_sample_loop(self, classes, image_size, cond_weight, get_cross_map=False):
        b = image_size[0]
        device = self.device

        img = torch.randn(image_size, device=device)
        imgs = []
        cross_images_final = []

        if classes is not None:
            n_sample = classes.shape[0]
            context_mask = torch.ones_like(classes).to(device)
            # make 0 index unconditional
            # double the batch
            classes = classes.repeat(2)
            context_mask = context_mask.repeat(2)
            context_mask[n_sample:] = 0.0
            sampling_fn = partial(
                self.p_sample_guided,
                classes=classes,
                cond_weight=cond_weight,
                context_mask=context_mask,
            )

        else:
            sampling_fn = partial(self.p_sample)

        for i in reversed(range(0, self.timesteps)):
            img, cross_matrix = sampling_fn(x=img, t=torch.full((b,), i, device=device, dtype=torch.long), t_index=i)
            imgs.append(img.cpu().numpy())
            cross_images_final.append(cross_matrix.cpu().numpy())

        if get_cross_map:
            return imgs, cross_images_final
        else:
            return imgs

    @torch.no_grad()
    def p_sample(self, x, t, t_index):
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (x - betas_t * self.model(x, time=t) / sqrt_one_minus_alphas_cumprod_t)

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_guided(self, x, classes, t, t_index, context_mask, cond_weight):
        # adapted from: https://openreview.net/pdf?id=qw8AKxfYbI
        batch_size = x.shape[0]
        device = self.device
        # double to do guidance with
        t_double = t.repeat(2).to(device)
        x_double = x.repeat(2, 1, 1, 1).to(device)
        betas_t = extract(self.betas, t_double, x_double.shape, device)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t_double, x_double.shape, device)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t_double, x_double.shape, device)

        # classifier free sampling interpolates between guided and non guided using `cond_weight`
        classes_masked = classes * context_mask
        classes_masked = classes_masked.type(torch.long)
        # model = self.accelerator.unwrap_model(self.model)
        self.model.output_attention = True
        preds, cross_map_full = self.model(x_double, time=t_double, classes=classes_masked)
        self.model.output_attention = False
        cross_map = cross_map_full[:batch_size]
        eps1 = (1 + cond_weight) * preds[:batch_size]
        eps2 = cond_weight * preds[batch_size:]
        x_t = eps1 - eps2

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t[:batch_size] * (
            x - betas_t[:batch_size] * x_t / sqrt_one_minus_alphas_cumprod_t[:batch_size]
        )

        if t_index == 0:
            return model_mean, cross_map
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape, device)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise, cross_map

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, torch.randn_like(x_start))
        device = self.device

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape, device)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape, device)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, x_start, t, classes, noise=None, loss_type="huber", p_uncond=0.1):
        device = self.device
        noise = default(noise, torch.randn_like(x_start))

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        context_mask = torch.bernoulli(torch.zeros(classes.shape[0]) + (1 - p_uncond)).to(device)

        # Mask for unconditional guidance
        classes = classes * context_mask
        # nn.Embedding needs type to be long, multiplying with mask changes type
        classes = classes.type(torch.long)
        predicted_noise = self.model(x_noisy, t, classes)

        if loss_type == "l1":
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == "l2":
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()

        return loss

    def forward(self, x, classes):
        device = self.device
        x = x.type(torch.float32)
        classes = classes.type(torch.long)
        b = x.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=device).long()

        return self.p_losses(x, t, classes)
