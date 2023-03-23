from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore
from hydra_zen import MISSING, instantiate, make_config
from omegaconf import DictConfig

from dnadiffusion.configs import LightningTrainer, sample

Config = make_config(
    hydra_defaults=[
        "_self_",
        {"data": "LoadingData"},
        {"model": "Unet"},
    ],
    data=MISSING,
    model=MISSING,
    trainer=LightningTrainer,
    sample=sample,
    # Constants
    data_dir="dna_diffusion/data",
    random_seed=42,
    ckpt_path=None,
)

cs = ConfigStore.instance()
cs.store(name="config", node=Config)


def train(config):
    data = instantiate(config.data)
    sample = instantiate(config.sample, data_module=data)
    model = instantiate(config.model)
    trainer = instantiate(config.trainer)

    # Adding custom callbacks
    trainer.callbacks.append(sample)

    trainer.fit(model, data)

    return model


@hydra.main(config_path=None, config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    return train(cfg)


if __name__ == "__main__":
    main()
