import hydra
import pytorch_lightning as pl
import logging
import wandb

from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

logger = logging.getLogger()


@hydra.main(config_path="configs", config_name="train")
def train(cfg: DictConfig):
    # Keeping track of current config settings in logger
    logger.info(f"Training with config:\n{OmegaConf.to_yaml(cfg)}")
    run = wandb.init(project=cfg.logger.wandb.project, config=cfg)

    # Placeholder for what loss or metric values we plan to track with wandb
    wandb.log({"loss": loss})

    pl.seed_everything(cfg.seed)

    # Check if this works
    model = instantiate(cfg.model)
    train_dl = instantiate(cfg.dataset.train_dl)
    val_dl = instantiate(cfg.dataset.val_dl)

    if cfg.ckpt:
        model.load_from_checkpoint(cfg.ckpt)

    trainer = pl.Trainer(
        callbacks=cfg.callbacks,
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        logger=cfg.logger.wandb
    )

    trainer.fit(model, train_dl, val_dl)


if __name__ == "__main__":
    train()
