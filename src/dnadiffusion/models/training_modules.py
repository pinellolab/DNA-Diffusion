import copy

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import Adam

from dnadiffusion.models.diffusion import p_losses
from dnadiffusion.models.unet import Unet
from dnadiffusion.utils.ema import EMA
from dnadiffusion.utils.scheduler import linear_beta_schedule


class UnetDiffusion(pl.LightningModule):
    def __init__(
        self,
        model: Unet,
        lr: float = 1e-3,
        timesteps: int = 50,
        beta=0.995,
        optimizer: torch.optim.Optimizer = Adam,
    ) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.timesteps = timesteps

        self.betas = linear_beta_schedule(timesteps=self.timesteps, beta_end=0.2)
        # define alphas
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.ema = EMA(beta)
        self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)

    def forward(self, x: torch.Tensor, time: torch.Tensor, classes: torch.Tensor):
        return self.model(x, time, classes)

    def training_step(self, batch, batch_idx: int) -> int:
        x, y = batch
        x = x.type(torch.float32)
        y = y.type(torch.long)
        batch_size = x.shape[0]
        t = torch.randint(0, self.timesteps, (batch_size,)).long()
        loss = p_losses(
            self.model,
            x,
            t,
            y,
            loss_type="huber",
            sqrt_alphas_cumprod_in=self.sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod_in=self.sqrt_one_minus_alphas_cumprod,
            device=self.device,
        )
        self.log("loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.lr)
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)

        self.ema.step_ema(self.ema_model, self.model)

    def configure_callbacks(self):
        pass
