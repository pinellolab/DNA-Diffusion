from typing import Any

import torch
import torch.distributed as dist
import wandb


def init_wandb() -> Any:
    wandb_id = wandb.util.generate_id()
    wandb.init(project="dnadiffusion", id=wandb_id)


def distributed_setup(batch_size: int) -> tuple:
    assert dist.is_available() and torch.cuda.is_available(), "Distributed training is not available."

    dist.init_process_group("nccl")

    assert batch_size % dist.get_world_size() == 0, "Batch size should be divisible by the number of processes."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    local_batch_size = batch_size // dist.get_world_size()

    return rank, device, local_batch_size


def train_step(
    x: torch.Tensor,
    y: torch.Tensor,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: Any,
    precision: str | None = None,
):
    if precision == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    x = x.to(device, dtype=torch.float32)
    y = y.to(device)
    with torch.autocast(device_type=device, dtype=dtype):
        loss = model(x, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.mean().item()


def val_step(
    x: torch.Tensor,
    y: torch.Tensor,
    model: torch.nn.Module,
    device: Any,
    precision: str | None = None,
):
    if precision == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    x = x.to(device, dtype=dtype)
    y = y.to(device)
    with torch.no_grad():
        loss = model(x, y)
    return loss.mean().item()
