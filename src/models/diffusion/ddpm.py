import tqdm
import torch
from torch import nn
from torch.nn.functional import F

from models.diffusion.diffusion import DiffusionModel

from utils.schedules import (
    beta_linear_log_snr,
    alpha_cosine_log_snr,
    linear_beta_schedule,
)
from utils.misc import extract


class DDPM(DiffusionModel):
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps=1000,
        noise_schedule="cosine",
        bit_scale=1,
        unet_config: dict,
        is_conditional: bool,
        logdir: str,
        img_size: int,
        optimizer_config: dict,
        lr_scheduler_config: dict = None,
        criterion: nn.Module,
        use_ema: bool = True,
        ema_decay: float = 0.9999,
        lr_warmup=0,
    ):
        super().__init__(
            unet_config,
            is_conditional,
            logdir,
            img_size,
            optimizer_config,
            lr_scheduler_config,
            criterion,
            use_ema,
            ema_decay,
            lr_warmup,
        )

        self.image_size = image_size
        self.bit_scale = bit_scale
        self.timesteps = timesteps

        if noise_schedule == "linear":
            self.log_snr = beta_linear_log_snr
            self.betas = linear_beta_schedule(timesteps=timesteps, beta_end=0.05)
        elif noise_schedule == "cosine":
            # Refer to Section 3.2 of https://arxiv.org/abs/2102.09672 for details
            self.log_snr = alpha_cosine_log_snr
            self.betas = cosine_beta_schedule(timesteps=timesteps, s=0.0001)
        else:
            raise ValueError(f"invalid noise schedule {noise_schedule}")

        # Define Beta Schedule
        self.set_noise_schedule(self.betas)

    def set_noise_schedule(self, betas: torch.Tensor) -> None:
        # define alphas
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise=None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    @torch.no_grad()
    def p_sample(self, x, t, t_index):
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )

        # print (x.shape, 'x_shape')
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * self.model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, shape):
        device = next(self.model.parameters()).device

        b = shape[0]
        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=device)
        imgs = []

        for i in tqdm(
            reversed(range(0, self.timesteps)),
            desc="sampling loop time step",
            total=self.timesteps,
        ):
            img = self.p_sample(
                img, torch.full((b,), i, device=device, dtype=torch.long), i
            )
            imgs.append(img.cpu().numpy())
        return imgs

    @torch.no_grad()
    def sample(self, batch, channels=3, nucleotides=4):
        return self.p_sample_loop(
            shape=(batch.shape[0], channels, nucleotides, self.image_size)
        )

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        if noise is None:
            noise = torch.randn_like(batch)
        x_noisy = self.q_sample(x_start=batch, t=self.timesteps, noise=noise)

        # calculating generic loss function, we'll add it to the class constructor once we have the code
        # we should log more metrics at train and validation e.g. l1, l2 and other suggestions
        predicted_noise = self.model(x_noisy, self.timesteps)
        loss = self.criterion(predicted_noise, noise)
        self.log("train", loss, batch_size=batch.shape[0])

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        if self.is_conditional:
            return None
        else:
            return self.inference_step(batch, batch_idx, "validation")

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        if self.is_conditional:
            return None
        else:
            return self.inference_step(batch, batch_idx, "test")

    def inference_step(self, batch: torch.Tensor, batch_idx: int, phase="validation"):
        predictions = self.sample(batch)

        loss = self.criterion(predictions, batch)

        self.log("val_loss", loss) if phase == "validation" else self.log(
            "test_loss", loss
        )

        return predictions
