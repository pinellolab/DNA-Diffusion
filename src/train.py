import hydra
import pytorch_lightning as pl
import logging
import wandb

from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from pytorch_lightning.loggers import WandbLogger

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="template_config")
def train(cfg: DictConfig):
    # Keeping track of current config settings in logger
    logger.info(f"Training with config:\n{OmegaConf.to_yaml(cfg)}")
    run = wandb.init(project="", config=cfg, group="train")
    pl_logger = WandbLogger()  # include in config?
    local_logger = instantiate(cfg.logger)

    pl.seed_everything(cfg.seed)

    # Check if this works
    scheduler = instantiate(cfg.scheduler)
    model = instantiate(cfg.model, variance_schedule=scheduler)
    train_dl = instantiate(cfg.train)
    val_dl = instantiate(cfg.val)

    if cfg.ckpt:
        ckpt_path = cfg.ckpt
        model.load_from_checkpoint(ckpt_path)

    trainer = pl.Trainer(
        callbacks=cfg.callbacks,
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        logger=pl_logger
    )

    trainer.fit(model, train_dl, val_dl)


if __name__ == "__main__":
    train()
