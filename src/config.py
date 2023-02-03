### file to include dataclass definition
from dataclasses import dataclass

### needs overhaul with new folder structure
### ignore for now

@dataclass
class DiffusionParams:
    model: str
    beta_end: float
    schedule: str
    timestep: int
    is_conditional: bool
    criterion: str #utils.metrics.MetricName
    use_ema: bool
    lr_warmup: int

@dataclass
class UNetParams:
    model: str
    dim: int
    init_dim: int
    dim_mults: int
    channels: int
    resnet_block_groups: int
    learned_sinusoidal_dim: int
    num_classes: int
    self_conditioned: bool

@dataclass
class DNADiffusionConfig:
    unetparams: UNetParams
    diffusionparams: DiffusionParams
    strategy: str
    dataset: str
    seed: int
    batch_size: int
    min_epochs: int
    max_epochs: int
    gradient_clip_val: float
    accumulate_grad_batches: int
    log_every_n_steps: int
    check_val_every_n_epoch: int #for debug purposes
    save_last: bool
    precision: int
    optimizer: str
    lr_scheduler: str
    logger: str 
    ckpt: str # path to ckpt
    callbacks: str
    accelerator: str
    devices: str

