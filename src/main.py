import hydra
import pytorch_lightning as pl
import logging
import wandb

from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from hydra.core.config_store import ConfigStore
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
#from config import DNADiffusionConfig

#_configstore = ConfigStore.instance()
#_configstore.store(name="dnadiffusion_config", node=DNADiffusionConfig)


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
    train_dl = instantiate(cfg.data.train_dl)
    val_dl = instantiate(cfg.data.val_dl)
    if cfg.ckpt_path:
        model.load_from_checkpoint(cfg.ckpt_path)

    model_checkpoint_callback = ModelCheckpoint(dirpath='checkpoints', monitor='val_loss', mode='min', save_top_k=10, save_last=True)
    lr_monitor_callback = LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(
        callbacks=[model_checkpoint_callback, lr_monitor_callback],
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        logger=cfg.logger.wandb,
    )
    return
    trainer.fit(model, train_dl, val_dl)


if __name__ == "__main__":
    main()