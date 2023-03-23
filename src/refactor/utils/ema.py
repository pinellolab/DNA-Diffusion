from copy import deepcopy

from torch import nn


class EMA:
    def __init__(self, model: nn.Module, beta: float):
        super().__init__()
        self.beta = beta
        self.step = 0
        self.ema_model = deepcopy(model).eval().requires_grad_(False)

    def update_model_average(self, current_model):
        for current_params, ema_params in zip(current_model.parameters(), self.ema_model.parameters()):
            old_weight, up_weight = ema_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(self.ema_model, model)
            self.step += 1
            return
        self.update_model_average(self.ema_model, model)
        self.step += 1

    def reset_parameters(self, model):
        self.ema_model.load_state_dict(model.state_dict())
