class EMA:
    def __init__(self, model, beta):
        super().__init__()
        self.ma_model = model
        self.beta = beta
        self.step = 0

    def update_model_average(self, current_model):
        for current_params, ma_params in zip(
            current_model.parameters(), self.ma_model.parameters()
        ):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(self.model, model)
            self.step += 1
            return
        self.update_model_average(self.ma_model, model)
        self.step += 1

    def reset_parameters(self, model):
        self.ma_model.load_state_dict(model.state_dict())
