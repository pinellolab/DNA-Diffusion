import torch

from dnadiffusion.data.dataloader import load_data
from dnadiffusion.models.diffusion import Diffusion
from dnadiffusion.models.unet import UNet
from dnadiffusion.utils.sample_util import create_sample


def sample(model_path: str, num_samples: int = 1000):
    # Instantiating data and model

    print("Loading data")
    encode_data, _ = load_data(
        data_path="dnadiffusion/data/K562_hESCT0_HepG2_GM12878_12k_sequences_per_group.txt",
        saved_data_path="dnadiffusion/data/encode_data.pkl",
        subset_list=[
            "GM12878_ENCLB441ZZZ",
            "hESCT0_ENCLB449ZZZ",
            "K562_ENCLB843GMH",
            "HepG2_ENCLB029COU",
        ],
        limit_total_sequences=0,
        num_sampling_to_compare_cells=1000,
        load_saved_data=True,
        batch_size=240,
    )

    print("Instantiating unet")
    unet = UNet(
        dim=200,
        channels=1,
        dim_mults=(1, 2, 4),
        resnet_block_groups=4,
    )

    print("Instantiating diffusion class")
    diffusion = Diffusion(
        unet,
        timesteps=50,
    )

    # Load checkpoint
    print("Loading checkpoint")
    checkpoint_dict = torch.load(model_path)
    diffusion.load_state_dict(checkpoint_dict["model"])

    # Generating cell specific samples
    cell_num_list = encode_data["tag_to_numeric"].values()

    for i in cell_num_list:
        print(
            f"Generating {num_samples} samples for cell {encode_data['numeric_to_tag'][i]}"
        )
        create_sample(
            diffusion,
            conditional_numeric_to_tag=encode_data["numeric_to_tag"],
            cell_types=encode_data["cell_types"],
            num_sampling_to_compare_cells=int(num_samples / 10),
            specific_group=True,
            group_number=i,
            cond_weight_to_metric=1,
            save_timestep_dataframe=True,
        )


if __name__ == "__main__":
    sample()
