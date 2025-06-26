from typing import Optional

from transformers import PretrainedConfig, PreTrainedModel

from dnadiffusion.models.unet import UNet


class UNetConfig(PretrainedConfig):
    model_type = "unet"

    def __init__(
        self,
        dim: int = 200,
        dim_mults: tuple = (1, 2, 4),
        init_dim: Optional[int] = None,
        resnet_block_groups: int = 4,
        num_classes: int = 10,
        channels: int = 1,
        **kwargs,
    ):
        self.dim = dim
        self.dim_mults = dim_mults
        self.init_dim = init_dim
        self.resnet_block_groups = resnet_block_groups
        self.num_classes = num_classes
        self.channels = channels

        super().__init__(**kwargs)


class PretrainedUNet(PreTrainedModel):
    config_class = UNetConfig
    base_model_prefix = "model"

    def __init__(self, config: UNetConfig):
        super().__init__(config)

        # Initialize UNet directly as the model
        self.model = UNet(
            dim=config.dim,
            init_dim=config.init_dim,
            dim_mults=config.dim_mults,
            channels=config.channels,
            resnet_block_groups=config.resnet_block_groups,
            num_classes=config.num_classes,
        )

    def forward(self, x, time, classes):
        return self.model(x, time, classes)
