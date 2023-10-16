from accelerate import Accelerator

from dnadiffusion.data.dataloader import load_data
from dnadiffusion.models.diffusion import Diffusion
from dnadiffusion.models.unet import UNet
from dnadiffusion.utils.train_util import TrainLoop


def train():
    accelerator = Accelerator(split_batches=True, log_with=["wandb"], mixed_precision="bf16")

    data = load_data(
        data_path="src/dnadiffusion/data/K562_hESCT0_HepG2_GM12878_12k_sequences_per_group.txt",
        saved_data_path="src/dnadiffusion/data/encode_data.pkl",
        subset_list=[
            "GM12878_ENCLB441ZZZ",
            "hESCT0_ENCLB449ZZZ",
            "K562_ENCLB843GMH",
            "HepG2_ENCLB029COU",
        ],
        limit_total_sequences=0,
        num_sampling_to_compare_cells=1000,
        load_saved_data=True,
    )

    unet = UNet(
        dim=200,
        channels=1,
        dim_mults=(1, 2, 4),
        resnet_block_groups=4,
    )

    diffusion = Diffusion(
        unet,
        timesteps=50,
    )

    TrainLoop(
        data=data,
        model=diffusion,
        accelerator=accelerator,
        epochs=10000,
        loss_show_epoch=10,
        sample_epoch=500,
        save_epoch=500,
        model_name="model_48k_sequences_per_group_K562_hESCT0_HepG2_GM12878_12k",
        image_size=200,
        num_sampling_to_compare_cells=1000,
        batch_size=960,
    ).train_loop()


if __name__ == "__main__":
    train()
