import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file

from dnadiffusion.utils.sample_util import create_sample


def sample(
    data: dict,
    model: nn.Module,
    checkpoint_path: str,
    sample_batch_size: int,
    number_of_samples: int,
    guidance_scale: float,
) -> None:
    print("Loading checkpoint")
    if checkpoint_path.endswith(".safetensors"):
        checkpoint_dict = (
            load_file(checkpoint_path) if torch.cuda.is_available() else load_file(checkpoint_path, device="cpu")
        )
    else:
        checkpoint_dict = (
            torch.load(checkpoint_path, map_location="cuda")
            if torch.cuda.is_available()
            else torch.load(checkpoint_path, map_location="cpu")
        )
        model.load_state_dict(checkpoint_dict["model"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Model sent to {device}")

    numeric_to_tag_dict = data[-1]
    cell_num_list = data[-2]

    print(f"Found cell types: {[numeric_to_tag_dict[i] for i in cell_num_list]}")

    for cell_type in cell_num_list:
        print(f"Generating {number_of_samples} samples for cell {numeric_to_tag_dict[cell_type]}")
        create_sample(
            model,
            cell_type=cell_type,
            sample_bs=sample_batch_size,
            conditional_numeric_to_tag=numeric_to_tag_dict,
            number_of_samples=number_of_samples,
            cond_weight_to_metric=guidance_scale,
            save_timesteps=False,
            save_dataframe=True,
            generate_attention_maps=False,
        )


@hydra.main(config_path="configs", config_name="sample", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    model = hydra.utils.instantiate(cfg.model)
    data = hydra.utils.instantiate(cfg.data)
    diffusion = hydra.utils.instantiate(cfg.diffusion, model=model)

    sample(
        data=data,
        model=diffusion,
        checkpoint_path=cfg.sampling.checkpoint_path,
        sample_batch_size=cfg.sampling.sample_batch_size,
        number_of_samples=cfg.sampling.number_of_samples,
        guidance_scale=cfg.sampling.guidance_scale,
    )


if __name__ == "__main__":
    main()
