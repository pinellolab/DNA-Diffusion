import os
from functools import partial
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import nn

from dnadiffusion.utils.utils import extract


def sampling_to_metric(
    cell_types: list,
    conditional_numeric_to_tag: dict,
    additional_variables: dict,
    number_of_samples: int = 20,
    specific_group: bool = False,
    group_number: Optional[list] = None,
    cond_weight_to_metric: int = 0,
):
    # Sampling regions using the trained  model
    nucleotides = ["A", "C", "G", "T"]
    final_sequences = []
    # for n_a in tqdm(range(number_of_samples)): # generating number_of_samples *10 sequences
    for n_a in range(number_of_samples):  # generating number_of_samples *10 sequences
        print(n_a)
        sample_bs = 10
        if specific_group:
            sampled = torch.from_numpy(np.array([group_number] * sample_bs))
        else:
            sampled = torch.from_numpy(np.random.choice(cell_types, sample_bs))

        random_classes = sampled.float()  # .cuda() to accelerate
        if additional_variables:
            random_classes = random_classes.to(additional_variables["device"])

        sampled_images = sample(
            classes=random_classes,
            batch_size=sample_bs,
            channels=1,
            cond_weight=cond_weight_to_metric,
            **additional_variables,
        )
        # sampled_images = sampled_images
        for n_b, x in enumerate(sampled_images[-1]):
            seq_final = f">seq_test_{n_a}_{n_b}\n" + "".join(
                [nucleotides[s] for s in np.argmax(x.reshape(4, 200), axis=0)]
            )
            final_sequences.append(seq_final)

    if group_number:
        current_cell = conditional_numeric_to_tag[group_number]
        save_motifs_syn = open(f"synthetic_motifs_{current_cell}.fasta", "w")
        save_motifs_syn.write("\n".join(final_sequences))
        save_motifs_syn.close()
        os.system(
            f"gimme scan synthetic_motifs_{current_cell}.fasta -p   JASPAR2020_vertebrates -g hg38 > syn_results_motifs_{current_cell}.bed"
        )
        df_results_syn = pd.read_csv(f"syn_results_motifs_{current_cell}.bed", sep="\t", skiprows=5, header=None)
    else:
        save_motifs_syn = open("synthetic_motifs.fasta", "w")
        save_motifs_syn.write("\n".join(final_sequences))
        save_motifs_syn.close()
        os.system("gimme scan synthetic_motifs.fasta -p   JASPAR2020_vertebrates -g hg38 > syn_results_motifs.bed")
        df_results_syn = pd.read_csv("syn_results_motifs.bed", sep="\t", skiprows=5, header=None)

    """df_results_syn["motifs"] = df_results_syn[8].apply(
        lambda x: x.split('motif_name "')[1].split('"')[0]
    )
    """
    df_results_syn["motifs"] = (
        df_results_syn[8].dropna().apply(lambda x: x.split(" ")[1].strip('"')).reset_index(drop=True)
    )
    df_results_syn[0] = df_results_syn[0].apply(lambda x: "_".join(x.split("_")[:-1]))
    df_motifs_count_syn = df_results_syn[[0, "motifs"]].drop_duplicates().groupby("motifs").count()

    return df_motifs_count_syn


@torch.no_grad()
def p_sample(
    model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    t_index: int,
    betas: torch.Tensor,
    sqrt_one_minus_alphas_cumprod: torch.Tensor,
    sqrt_recip_alphas: torch.Tensor,
    posterior_variance: torch.Tensor,
):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, time=t) / sqrt_one_minus_alphas_cumprod_t)

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def p_sample_guided(
    model: nn.Module,
    x: torch.Tensor,
    classes: torch.Tensor,
    t: torch.Tensor,
    t_index: int,
    context_mask: torch.Tensor,
    betas: torch.Tensor,
    sqrt_one_minus_alphas_cumprod: torch.Tensor,
    sqrt_recip_alphas: torch.Tensor,
    posterior_variance: torch.Tensor,
    device: str,
    cond_weight: float = 0.0,
):
    # adapted from: https://openreview.net/pdf?id=qw8AKxfYbI
    batch_size = x.shape[0]
    # double to do guidance with
    t_double = t.repeat(2).to(device)
    x_double = x.repeat(2, 1, 1, 1).to(device)
    betas = betas.to(device)
    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device)
    betas_t = extract(betas, t_double, x_double.shape, device=device)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t_double, x_double.shape, device=device)
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t_double, x_double.shape, device=device)

    # classifier free sampling interpolates between guided and non guided using `cond_weight`
    classes_masked = classes * context_mask
    classes_masked = classes_masked.type(torch.long)
    # if accelerator:
    # model = accelerator.unwrap_model(model)
    model.output_attention = True
    preds, cross_map_full = model(x_double, time=t_double, classes=classes_masked)  # I added cross_map
    model.output_attention = False
    cross_map = cross_map_full[:batch_size]
    eps1 = (1 + cond_weight) * preds[:batch_size]
    eps2 = cond_weight * preds[batch_size:]
    x_t = eps1 - eps2

    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t[:batch_size] * (
        x - betas_t[:batch_size] * x_t / sqrt_one_minus_alphas_cumprod_t[:batch_size]
    )

    if t_index == 0:
        return model_mean, cross_map
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape, device=device)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise, cross_map


# Algorithm 2 but save all images:
@torch.no_grad()
def p_sample_loop(
    model: nn.Module,
    classes: torch.Tensor,
    shape: tuple,
    cond_weight: int,
    timesteps: int,
    device: str,
    betas: torch.Tensor,
    sqrt_one_minus_alphas_cumprod: torch.Tensor,
    sqrt_recip_alphas: torch.Tensor,
    posterior_variance: torch.Tensor,
    get_cross_map: bool = False,
):  # to accelerate add timesteps
    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []
    cross_images_final = []

    if classes is not None:
        n_sample = classes.shape[0]
        context_mask = torch.ones_like(classes).to(device)
        # make 0 index unconditional
        # double the batch
        classes = classes.repeat(2)
        context_mask = context_mask.repeat(2)
        context_mask[n_sample:] = 0.0  # makes second half of batch context free
        sampling_fn = partial(
            p_sample_guided,
            classes=classes,
            cond_weight=cond_weight,
            context_mask=context_mask,
            betas=betas,
            device=device,
            sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
            sqrt_recip_alphas=sqrt_recip_alphas,
            posterior_variance=posterior_variance,
        )  # to accelerate betas
    else:
        sampling_fn = partial(p_sample)

    # for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
    for i in reversed(range(0, timesteps)):
        img, cross_matrix = sampling_fn(
            model,
            x=img,
            t=torch.full((b,), i, device=device, dtype=torch.long),
            t_index=i,
        )
        imgs.append(img.cpu().numpy())
        cross_images_final.append(cross_matrix.cpu().numpy())
    if get_cross_map:
        return imgs, cross_images_final
    else:
        return imgs


@torch.no_grad()
def sample(
    model: nn.Module,
    image_size: int,
    classes: torch.Tensor,
    timesteps: int,
    device: str,
    betas: torch.Tensor,
    sqrt_one_minus_alphas_cumprod: torch.Tensor,
    sqrt_recip_alphas: torch.Tensor,
    posterior_variance: torch.Tensor,
    batch_size: int = 16,
    channels: int = 3,
    cond_weight: int = 0,
    get_cross_map: bool = False,
):  # to accelerate add timesteps, device , betas
    return p_sample_loop(
        model,
        classes=classes,
        shape=(batch_size, channels, 4, image_size),
        cond_weight=cond_weight,
        get_cross_map=get_cross_map,
        timesteps=timesteps,
        device=device,
        betas=betas,
        sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
        sqrt_recip_alphas=sqrt_recip_alphas,
        posterior_variance=posterior_variance,
    )
