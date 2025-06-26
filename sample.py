import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from dnadiffusion.utils.sample_util import create_sample


def sample(
    data: dict,
    model: nn.Module,
    checkpoint_path: str,
    sample_batch_size: int,
    number_of_samples: int,
    guidance_scale: float,
) -> None:
    print(data)
    """numeric_to_tag_dict, cell_num_list, cell_list = (
        data["numeric_to_tag"],
        data["cell_types"],
        list(data["tag_to_numeric"].keys()),
    )
    """

    numeric_to_tag_dict = data[-1]
    cell_num_list = data[-2]

    # Load checkpoint
    print("Loading checkpoint")
    checkpoint_dict = (
        torch.load(checkpoint_path) if torch.cuda.is_available() else torch.load(checkpoint_path, map_location="cpu")
    )
    # Load unet state dict
    model.model.load_state_dict(checkpoint_dict["model"])

    # Send model to device
    print("Sending model to device")
    model = model.to("cuda") if torch.cuda.is_available() else model

    for i in cell_num_list:
        print(f"Generating {number_of_samples} samples for cell {numeric_to_tag_dict[i]}")
        create_sample(
            model,
            cell_types=cell_num_list,
            sample_bs=sample_batch_size,
            conditional_numeric_to_tag=numeric_to_tag_dict,
            number_of_samples=number_of_samples,
            group_number=i,
            cond_weight_to_metric=guidance_scale,
            save_timesteps=False,
            save_dataframe=True,
            generate_attention_maps=False,
        )


@hydra.main(config_path="configs", config_name="sample", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    sampling_setup = {**cfg.sampling}
    model = hydra.utils.instantiate(cfg.model)
    data = hydra.utils.instantiate(cfg.data)
    diffusion = hydra.utils.instantiate(cfg.diffusion, model=model)

    sample(
        data=data,
        model=diffusion,
        **sampling_setup,
    )


if __name__ == "__main__":
    main()
