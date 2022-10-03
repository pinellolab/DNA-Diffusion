import numpy as np
import torch

from diffusion.models.bit_diffusion import BitDiffusion
from diffusion.models.unet import Unet

model = Unet(
    dim=200, channels=1, dim_mults=(1, 2, 4), resnet_block_groups=1, class_embed_dim=200, num_classes=16
)

bit_diffusion = BitDiffusion(
    model,
    image_size=200,
    timesteps=100,
)


sampled = torch.from_numpy(np.random.randint(0, 16, size=(4)))
random_classes = torch.zeros((4, 16))
random_classes = random_classes.scatter_(1, sampled.unsqueeze(dim=1), 1).float()

sampled_images = bit_diffusion.sample(batch_size=4, classes=random_classes)
print(sampled_images.shape)
