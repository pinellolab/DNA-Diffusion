import hydra
import pytorch_lightning as pl
import logging
import wandb

from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from hydra.core.config_store import ConfigStore
from utils.misc import get_parser
from config import DNADiffusionConfig

#_configstore = ConfigStore.instance()
#_configstore.store(name="dnadiffusion_config", node=DNADiffusionConfig)

#logger = logging.getLogger()

@hydra.main(config_path="configs", config_name="main")
def main(cfg):
   #run = wandb.init(
    #    name=parser.logdir,
    #    save_dir=parser.logdir,
    #    project=cfg.logger.wandb.project,
    #    config=cfg,
    #)

    # Placeholder for what loss or metric values we plan to track with wandb
    #wandb.log({"loss": cfg.model.criterion})

    pl.seed_everything(cfg.seed)
    # Check if this works
    model = instantiate(cfg.model)
    return
    train_dl = instantiate(cfg.data.train_dl)
    val_dl = instantiate(cfg.dataset.val_dl)

    if cfg.ckpt:
        model.load_from_checkpoint(cfg.ckpt)

    trainer = pl.Trainer(
        callbacks=cfg.callbacks,
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        logger=cfg.logger.wandb,
    )

    trainer.fit(model, train_dl, val_dl)


if __name__ == "__main__":
    main()