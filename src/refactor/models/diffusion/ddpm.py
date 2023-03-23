from functools import partial

import torch
import torch.nn.functional as F
import tqdm
from models.diffusion.diffusion import DiffusionModel
from torch import nn
from utils.misc import extract, extract_data_from_batch, mean_flat
from utils.schedules import (
    alpha_cosine_log_snr,
    beta_linear_log_snr,
    linear_beta_schedule,
)


class DDPM(DiffusionModel):
    def __init__(
        self,
        *,
        image_size,
        timesteps=50,
        noise_schedule="cosine",
        time_difference=0.0,
        unet: nn.Module,
        is_conditional: bool,
        p_uncond: float = 0.1,
        use_fp16: bool,
        logdir: str,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
        criterion: nn.Module,
        use_ema: bool = True,
        ema_decay: float = 0.9999,
        lr_warmup=0,
        use_p2_weigthing: bool = False,
        p2_gamma: float = 0.5,
        p2_k: float = 1,
    ):
        super().__init__(
            unet,
            is_conditional,
            use_fp16,
            logdir,
            optimizer,
            lr_scheduler,
            criterion,
            use_ema,
            ema_decay,
            lr_warmup,
        )
        print("saludos del matei")
        print("\n")
        self.image_size = image_size

        if noise_schedule == "linear":
            self.log_snr = beta_linear_log_snr
        elif noise_schedule == "cosine":
            self.log_snr = alpha_cosine_log_snr
        else:
            raise ValueError(f"invalid noise schedule {noise_schedule}")

        self.timesteps = timesteps
        self.p_uncond = p_uncond

        # self.betas = cosine_beta_schedule(timesteps=timesteps,  s=0.0001)
        self.set_noise_schedule(self.betas, self.timesteps)

        # proposed in the paper, summed to time_next
        # as a way to fix a deficiency in self-conditioning and lower FID when the number of sampling timesteps is < 400

        self.time_difference = time_difference

    def set_noise_schedule(self, betas, timesteps):
        # define beta schedule
        self.betas = linear_beta_schedule(timesteps=timesteps, beta_end=0.05)

        # define alphas
        alphas = 1.0 - self.betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)

        # sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        """
        Forward pass with noise.
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    @torch.no_grad()
    def p_sample(self, x, t, t_index):
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        # print (x.shape, 'x_shape')
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
    def p_sample_guided(self, x, classes, t, t_index, context_mask, cond_weight=0.0):
        # adapted from: https://openreview.net/pdf?id=qw8AKxfYbI
        # print (classes[0])
        batch_size = x.shape[0]
        # double to do guidance with
        t_double = t.repeat(2)
        x_double = x.repeat(2, 1, 1, 1)
        betas_t = extract(self.betas, t_double, x_double.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t_double, x_double.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t_double, x_double.shape)

        # classifier free sampling interpolates between guided and non guided using `cond_weight`
        classes_masked = classes * context_mask
        classes_masked = classes_masked.type(torch.long)
        # print ('class masked', classes_masked)
        preds = self.model(x_double, time=t_double, classes=classes_masked)
        eps1 = (1 + cond_weight) * preds[:batch_size]
        eps2 = cond_weight * preds[batch_size:]
        x_t = eps1 - eps2

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t[:batch_size] * (
            x - betas_t[:batch_size] * x_t / sqrt_one_minus_alphas_cumprod_t[:batch_size]
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    # Algorithm 2 but save all images:
    @torch.no_grad()
    def p_sample_loop(self, classes, shape, cond_weight):
        device = next(self.model.parameters()).device

        b = shape[0]
        # start from pure noise (for each example in the batch)
        image = torch.randn(shape, device=device)
        images = []

        if classes is not None:
            n_sample = classes.shape[0]
            context_mask = torch.ones_like(classes).to(device)
            # make 0 index unconditional
            # double the batch
            classes = classes.repeat(2)
            context_mask = context_mask.repeat(2)
            context_mask[n_sample:] = 0.0  # makes second half of batch context free
            sampling_fn = partial(
                self.p_sample_guided,
                classes=classes,
                cond_weight=cond_weight,
                context_mask=context_mask,
            )
        else:
            sampling_fn = partial(self.p_sample)

        for i in tqdm(
            reversed(range(0, self.timesteps)),
            desc="sampling loop time step",
            total=self.timesteps,
        ):
            image = sampling_fn(
                self.model,
                x=image,
                t=torch.full((b,), i, device=device, dtype=torch.long),
                t_index=i,
            )
            images.append(image.cpu().numpy())
        return images

    @torch.no_grad()
    def sample(self, image_size, classes=None, batch_size=16, channels=3, cond_weight=0):
        return self.p_sample_loop(
            self.model,
            classes=classes,
            shape=(batch_size, channels, 4, image_size),
            cond_weight=cond_weight,
        )

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        x_start, condition = extract_data_from_batch(batch)

        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=self.timesteps, noise=noise)

        # calculating generic loss function, we'll add it to the class constructor once we have the code
        # we should log more metrics at train and validation e.g. l1, l2 and other suggestions
        if self.use_fp16:
            with torch.cuda.amp.autocast():
                if self.is_conditional:
                    predicted_noise = self.model(x_noisy, self.timesteps, condition)
                else:
                    predicted_noise = self.model(x_noisy, self.timesteps)
        else:
            if self.is_conditional:
                predicted_noise = self.model(x_noisy, self.timesteps, condition)
            else:
                predicted_noise = self.model(x_noisy, self.timesteps)

        loss = self.criterion(predicted_noise, noise)
        self.log("train", loss, batch_size=batch.shape[0])

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        return self.inference_step(batch, batch_idx, "validation")

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        return self.inference_step(batch, batch_idx, "test")

    def inference_step(self, batch: torch.Tensor, batch_idx: int, phase="validation", noise=None):
        x_start, condition = extract_data_from_batch(batch)
        device = x_start.device
        batch_size = batch.shape[0]

        t = torch.randint(0, self.timesteps, (batch_size,), device=device).long()  # sampling a t to generate t and t+1

        if noise is None:
            noise = torch.randn_like(x_start)  #  gauss noise
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)  # this is the auto generated noise given t and Noise

        context_mask = torch.bernoulli(torch.zeros(classes.shape[0]) + (1 - self.p_uncond)).to(device)

        # mask for unconditinal guidance
        classes = classes * context_mask
        classes = classes.type(torch.long)

        predictions = self.model(x_noisy, t, condition)

        loss = self.criterion(predictions, batch)

        self.log("validation_loss", loss) if phase == "validation" else self.log("test_loss", loss)

        """
            Log multiple losses at validation/test time according to internal discussions.
        """

        return predictions

    def p2_weighting(self, x_t, ts, target, prediction):
        """
        From Perception Prioritized Training of Diffusion Models: https://arxiv.org/abs/2204.00227.
        """
        weight = (1 / (self.p2_k + self.snr) ** self.p2_gamma, ts, x_t.shape)
        loss_batch = mean_flat(weight * (target - prediction) ** 2)
        loss = torch.mean(loss_batch)
        return loss
