import os

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.distributed.checkpoint.state_dict import get_state_dict
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from dnadiffusion.data.dataloader import get_dataloader
from dnadiffusion.utils.sample_util import create_sample
from dnadiffusion.utils.train_util import distributed_setup, init_wandb, train_step, val_step


def train(
    distributed: bool,
    precision: str,
    num_workers: int,
    pin_memory: bool,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data: tuple,
    batch_size: int,
    sample_batch_size: int,
    log_step: int,
    num_epochs: int,
    min_epochs: int,
    patience: int,
    sample_epoch: int,
    number_of_samples: int,
    use_wandb: bool,
) -> None:
    if distributed:
        rank, device, local_batch_size = distributed_setup(batch_size)
        model = DDP(model, device_ids=[rank])
        device = torch.device(f"cuda:{rank}")
        rank_0 = rank == 0
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        rank_0 = True
        local_batch_size = batch_size

    # Data
    train_data, val_data, cell_num_list, numeric_to_tag_dict = data

    train_dl, train_sampler = get_dataloader(train_data, local_batch_size, num_workers, distributed, pin_memory)
    val_dl, _ = get_dataloader(val_data, local_batch_size, num_workers, distributed, pin_memory)

    # Metrics
    if rank_0 == 0 and use_wandb:
        init_wandb()

    global_step = 0

    model.train()

    # Early stopping
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None
    checkpoint_files = []

    for epoch in tqdm(range(num_epochs), disable=not rank_0):
        if distributed:
            train_sampler.set_epoch(epoch)

        for x, y in train_dl:
            loss = train_step(x, y, model, optimizer, device, precision)
            global_step += 1
            if rank_0 == 0 and global_step % log_step == 0 and use_wandb:
                wandb.log({"loss": loss, "epoch": epoch}, step=global_step)

        val_losses = []
        for x, y in val_dl:
            val_loss = val_step(x, y, model, device, precision)
            val_losses.append(val_loss)

        avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else float("inf")
        # print(f"Epoch: {epoch}, Train Loss: {loss}, Val Loss: {avg_val_loss}")

        if rank_0 == 0 and use_wandb:
            wandb.log({"loss": loss, "val_loss": avg_val_loss, "epoch": epoch}, step=global_step)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            if distributed:
                best_model_state, best_optimizer_state = get_state_dict(model, optimizer)
            else:
                best_model_state = model.state_dict()
                best_optimizer_state = optimizer.state_dict()

            if rank_0:
                checkpoint_dict = {
                    "model": best_model_state,
                    "optimizer": best_optimizer_state,
                    "epoch": epoch,
                    "global_step": global_step,
                    "val_loss": best_val_loss,
                }
                checkpoint_file = f"checkpoints/model_epoch{epoch}_step{global_step}_valloss_{best_val_loss:2f}.pt"
                torch.save(
                    checkpoint_dict,
                    checkpoint_file,
                )
                checkpoint_files.append(checkpoint_file)
                if len(checkpoint_files) > 2:
                    os.remove(checkpoint_files.pop(0))
        else:
            patience_counter += 1

        if epoch >= min_epochs and patience_counter >= patience:
            print(
                f"Early stopping at epoch {epoch}, Best val loss: {best_val_loss} achieved at epoch {epoch - patience_counter}"
            )
            break

        if rank_0 == 0 and (epoch + 1) % sample_epoch == 0:
            # for i in data["cell_types"]:
            for i in cell_num_list:
                create_sample(
                    model,
                    cell_types=cell_num_list,
                    sample_bs=sample_batch_size,
                    conditional_numeric_to_tag=numeric_to_tag_dict,
                    number_of_samples=number_of_samples,
                    group_number=i,
                    cond_weight_to_metric=1,
                    save_timesteps=False,
                    save_dataframe=True,
                    generate_attention_maps=False,
                )


@hydra.main(config_path="configs", config_name="train_debug", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    train_setup = {**cfg.training}
    model = hydra.utils.instantiate(cfg.model)
    data = hydra.utils.instantiate(cfg.data)
    optimizer = hydra.utils.instantiate(cfg.optimizer, model.parameters())
    diffusion = hydra.utils.instantiate(cfg.diffusion, model=model)

    train(
        **train_setup,
        model=diffusion,
        optimizer=optimizer,
        data=data,
    )


if __name__ == "__main__":
    main()
