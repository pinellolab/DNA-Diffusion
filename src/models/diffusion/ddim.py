import torch
import torch.nn as nn

import pytorch_lightning as pl

from models.diffusion.diffusion import DiffusionModel


class DDIM(DiffusionModel):
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps=1000,
        use_ddim=False,
        noise_schedule="cosine",
        time_difference=0.0,
        bit_scale=1.0,
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
        self.use_ddim = use_ddim

        # proposed in the paper, summed to time_next
        # as a way to fix a deficiency in self-conditioning and lower FID when the number of sampling timesteps is < 400

        self.time_difference = time_difference

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

    # TODO: Need to add class conditioned weight
    @torch.no_grad()
    def sample(self, batch_size=16, classes=None):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.ddpm_sample if not self.use_ddim else self.ddim_sample
        return sample_fn((batch_size, 8, 4, image_size), classes=classes)  # Lucas

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
        # #TODO: Does it make sense to self condition with a class?
        # if random() < 0.5:
        #     with torch.no_grad():
        #         self_cond = self.model(noised_img, noise_level, class_enc).detach_()

        pred = self.model(
            noised_img, noise_level, class_enc, self_cond
        )  # BACK TO NOISE_LEVEL

        # return F.mse_loss(pred, img) # LUCAS
        return F.smooth_l1_loss(pred, img)  # LUCAS ADDED
