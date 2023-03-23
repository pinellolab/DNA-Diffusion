import logging
import os
import sys
from dataclasses import dataclass

import hydra
import pyrootutils
import pytorch_lightning as pl
import wandb
from hydra.core.config_store import ConfigStore
from hydra.utils import get_original_cwd, instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

root = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


@dataclass
class DNADiffusionConfig:
    data: str = "vanilla_sequences"
    model: str = "dnadiffusion"
    logger: str = "wandb"
    trainer: str = "ddp"
    callbacks: str = "default"
    paths: str = "default"
    seed: int = 42
    train: bool = True
    test: bool = False
    # ckpt_path: None


cs = ConfigStore.instance()
cs.store(name="dnadiffusion_config", node=DNADiffusionConfig)


@hydra.main(version_base="1.3", config_path="configs", config_name="main")
def main(cfg: DNADiffusionConfig):
    # print(HydraConfig.get().job.name)

    # run = wandb.init(
    #    name=parser.logdir,
    #    save_dir=parser.logdir,
    #    project=cfg.logger.wandb.project,
    #    config=cfg,
    # )

    # Placeholder for what loss or metric values we plan to track with wandb
    # wandb.log({"loss": cfg.model.criterion})
    print(f"Current working directory : {os.getcwd()}")
    print(f"Orig working directory    : {get_original_cwd()}")

    pl.seed_everything(cfg.seed)
    # Check if this works
    model = instantiate(cfg.model)
    train_dl = instantiate(cfg.data)
    print(train_dl)
    return
    val_dl = instantiate(cfg.data)
    if cfg.ckpt_path:
        model.load_from_checkpoint(cfg.ckpt_path)

    model_checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        monitor="val_loss",
        mode="min",
        save_top_k=10,
        save_last=True,
    )
    lr_monitor_callback = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        callbacks=[model_checkpoint_callback, lr_monitor_callback],
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        logger=cfg.logger.wandb,
    )
    trainer.fit(model, train_dl, val_dl)


if __name__ == "__main__":
    main()
