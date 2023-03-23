import torch
from torch import nn


class EMA:
    # https://github.com/dome272/Diffusion-Models-pytorch/blob/main/modules.py
    def __init__(self, beta: float = 0.995) -> None:
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model: nn.Module, current_model: nn.Module) -> None:
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
        if old is None:
            return new
        device = new.device
        old = old.to(device)
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model: nn.Module, model: nn.Module, step_start_ema: int = 100) -> None:
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())
