from typing import Any

import lightning as L
import torch
from lightning.pytorch.utilities import rank_zero_only

# Check this link
# https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/common/callbacks/ema.py


class EMA(L.Callback):
    def init(
        self,
        beta: float = 0.995,
    ) -> None:
        self.beta = beta
        self.step = 0
        self.step_start_ema = 100

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        self.step_ema(pl_module.ema_model, pl_module.model)

    def update_model_average(
        self,
        ma_model: L.LightningModule,
        current_model: L.LightningModule,
    ) -> None:
        for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
        ):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
        if old is None:
            return new
        old = old
        return old * self.beta + (1 - self.beta) * new

    def step_ema(
        self,
        ema_model: L.LightningModule,
        model: L.LightningModule,
    ) -> None:
        if self.step < self.step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())
