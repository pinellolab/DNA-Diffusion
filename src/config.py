### file to include dataclass definition
from dataclasses import dataclass

from hydra.core.config_store import ConfigStore

### needs overhaul with new folder structure
### ignore for now
"""
@dataclass
class DNADiffusionConfig:
    defaults:
      - _self_
      - optimizer: adam
      - lr_scheduler: MultiStepLR
      - unet: unet_conditional

    _target_: str = "__main__.trgt"  # dotpath describing location of callable
    timesteps: 200
    use_fp16: True
    criterion: torch.nn.MSELoss #utils.metrics.MetricName
    use_ema: True
    ema_decay: float = 0.999
    lr_warmup: 5000
    image_size: 200
"""
