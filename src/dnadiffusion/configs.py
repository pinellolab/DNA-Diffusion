import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from hydra.core.config_store import ConfigStore
from hydra_zen import MISSING, builds, make_custom_builds_fn
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger

from dnadiffusion.callbacks.sampling import Sample
from dnadiffusion.data.dataloader import LoadingDataModule
from dnadiffusion.models.training_modules import UnetDiffusion
from dnadiffusion.models.unet import Unet as UnetBase

# Custom Builds Function
sbuilds = make_custom_builds_fn(populate_full_signature=True)
pbuilds = make_custom_builds_fn(zen_partial=True, populate_full_signature=True)

# Transforms config if we need to add more
# transforms = builds(T.Compose, [builds(T.ToTensor)])
transforms = builds(T.ToTensor)

# Loading data
LoadingData = builds(
    LoadingDataModule,
    input_csv="./dnadiffusion/data/K562_hESCT0_HepG2_GM12878_12k_sequences_per_group.txt",
    subset_components=[
        "GM12878_ENCLB441ZZZ",
        "hESCT0_ENCLB449ZZZ",
        "K562_ENCLB843GMH",
        "HepG2_ENCLB029COU",
    ],
    load_saved_data=False,
    change_component_index=True,
    number_of_sequences_to_motif_creation=1000,
    transform=transforms,
    populate_full_signature=True,
)

# Diffusion Model
Unet = builds(
    UnetBase,
    dim=200,
    init_dim=None,
    dim_mults=(1, 2, 4),
    channels=1,
    resnet_block_groups=4,
    learned_sinusoidal_dim=10,
    num_classes=10,
    output_attention=False,
)

# Optimizers
Adam = pbuilds(torch.optim.Adam, lr=1e-3)

# Lightning Module
UnetConfig = builds(
    UnetDiffusion,
    model=Unet,
    lr=1e-3,
    timesteps=50,
    beta=0.995,
    optimizer=Adam,
)

# Callbacks
sample = builds(
    Sample,
    data_module=MISSING,
    image_size=200,
    num_sampling_to_compare_cells=1000,
)

wandb = builds(
    WandbLogger,
    project="dnadiffusion",
    notes="lightning",
)

checkpoint = builds(
    ModelCheckpoint,
    dirpath="dnadiffusion/checkpoints/",
    every_n_epochs=500,
)

# Lightning Trainer
LightningTrainer = builds(
    pl.Trainer,
    accelerator="cuda",
    strategy="ddp_find_unused_parameters_true",
    num_nodes=1,
    devices=8,
    max_epochs=10000,
    logger=wandb,
    callbacks=[checkpoint],
)

# Registering the builds
cs = ConfigStore.instance()

cs.store(group="data", name="LoadingData", node=LoadingData)
cs.store(group="model", name="Unet", node=UnetConfig)
