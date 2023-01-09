import torch
import torch.nn as nn
import pytorch_lightning as pl

from utils.ema import EMA


class DiffusionModel(pl.LightningModule):
    def __init__(
        self,
        unet_config: dict,
        timesteps: int,
        is_conditional: bool,
        use_fp16: bool,
        logdir: str,
        image_size: int,
        optimizer_config: dict,
        lr_scheduler_config: dict,
        criterion: nn.Module,
        use_ema: bool = True,
        ema_decay: float = 0.9999,
        lr_warmup=0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["criterion"])

        # create Unet
        # attempt using hydra.utils.instantiate to instantiate both unet, lr scheduler and optimizer
        self.model = instantiate_from_config(unet_config)
        self.timesteps = timesteps

        # training parameters
        self.use_ema = use_ema
        if self.use_ema:
            self.eps_model_ema = EMA(self.model, decay=ema_decay)
        self.is_conditional = is_conditional
        self.use_fp16 = use_fp16
        self.image_size = image_size
        self.lr_scheduler_config = lr_scheduler_config
        self.optimizer_config = optimizer_config
        self.lr_warmup = lr_warmup
        self.criterion = criterion
        self.logdir = logdir

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        loss = 0
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        preds = self.inference_step(batch)
        return preds

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        preds = self.inference_step(batch)
        return preds

    def inference_step(self, batch: torch.Tensor):
        return

    def sample(
        self,
        n_sample: int,  # number of samples
        condition=None,
        timesteps=None,
        *args,
        **kwargs
    ) -> torch.Tensor:
        return

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        if self.trainer.global_step < self.lr_warmup:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.lr_warmup)
            for pg in optimizer.param_groups:
                pg["learning_rate"] = lr_scale * self.optimizer_config.params.lr

        optimizer.step(closure=optimizer_closure)

    def on_before_zero_grad(self, *args, **kwargs) -> None:
        if self.use_ema:
            self.eps_model_ema.update(self.model)

    def configure_optimizers(self):
        optimizer = instantiate_from_config(
            self.optimizer_config, params=self.parameters()
        )
        if self.lr_scheduler_config is not None:
            scheduler = instantiate_from_config(
                self.lr_scheduler_config, optimizer=optimizer
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return optimizer
