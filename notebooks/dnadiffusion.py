import copy
import itertools
import math
import os
import pickle
import random
from functools import partial
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from accelerate import Accelerator, DistributedDataParallelKwargs
from einops import rearrange
from memory_efficient_attention_pytorch import Attention as EfficientAttention
from scipy.special import rel_entr
from torch import einsum, nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# Helper Modules
def exists(x):
    return x is not None


def cycle(dl):
    while True:
        yield from dl


def l2norm(t):
    return F.normalize(t, dim=-1)


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def extract(a, t, x_shape, device=None):
    batch_size = t.shape[0]
    if device:
        a = a.to(device)
        t = t.to(device)

    out = a.gather(-1, t)
    result = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    if device:
        result.to(device)
    return result


# Utils
class EMA:  # https://github.com/dome272/Diffusion-Models-pytorch/blob/main/modules.py
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
        ):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        device = new.device
        old = old.to(device)
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


def one_hot_encode(seq, alphabet, max_seq_len):
    """One-hot encode a sequence."""
    seq_len = len(seq)
    seq_array = np.zeros((max_seq_len, len(alphabet)))
    for i in range(seq_len):
        seq_array[i, alphabet.index(seq[i])] = 1
    return seq_array


def encode(seq, alphabet):
    """Encode a sequence."""
    seq_len = len(seq)
    seq_array = np.zeros(len(alphabet))
    for i in range(seq_len):
        seq_array[alphabet.index(seq[i])] = 1

    return seq_array


# Metrics
def convert_to_seq(x, nucleotides):
    return "".join(
        [nucleotides[s] for s in np.argmax(x.reshape(4, 200), axis=0)]
    )


def create_sample(
    diffusion_model,
    cell_types: list,
    conditional_numeric_to_tag: dict,
    number_of_samples: int = 20,
    specific_group: bool = False,
    group_number: list | None = None,
    cond_weight_to_metric: int = 0,
    save_timestep_dataframe: bool = False,
):
    nucleotides = ["A", "C", "G", "T"]
    final_sequences = []
    final_df = []
    for n_a in range(number_of_samples):
        print(n_a)
        sample_bs = 10
        if specific_group:
            sampled = torch.from_numpy(np.array([group_number] * sample_bs))
        else:
            sampled = torch.from_numpy(np.random.choice(cell_types, sample_bs))

        classes = sampled.float().to(diffusion_model.device)
        sampled_images = diffusion_model.sample(
            classes, (sample_bs, 1, 4, 200), cond_weight_to_metric
        )

        if save_timestep_dataframe:
            seqs_to_df = {}
            for en, step in enumerate(sampled_images):
                seqs_to_df[en] = [convert_to_seq(x, nucleotides) for x in step]
            final_df.append(pd.DataFrame(seqs_to_df))

        else:
            for n_b, x in enumerate(sampled_images[-1]):
                seq_final = f">seq_test_{n_a}_{n_b}\n" + "".join(
                    [
                        nucleotides[s]
                        for s in np.argmax(x.reshape(4, 200), axis=0)
                    ]
                )
                final_sequences.append(seq_final)

    if save_timestep_dataframe:
        pd.concat(final_df, ignore_index=True).to_csv(
            f"final_{conditional_numeric_to_tag[group_number]}.txt",
            header=True,
            sep="\t",
            index=False,
        )
        return

    else:
        save_motifs_syn = open("synthetic_motifs.fasta", "w")
        save_motifs_syn.write("\n".join(final_sequences))
        save_motifs_syn.close()
        os.system(
            "gimme scan synthetic_motifs.fasta -p JASPAR2020_vertebrates -g hg38 > syn_results_motifs.bed"
        )
        df_results_syn = pd.read_csv(
            "syn_results_motifs.bed", sep="\t", skiprows=5, header=None
        )

    df_results_syn["motifs"] = df_results_syn[8].apply(
        lambda x: x.split('motif_name "')[1].split('"')[0]
    )
    df_results_syn[0] = df_results_syn[0].apply(
        lambda x: "_".join(x.split("_")[:-1])
    )
    df_motifs_count_syn = (
        df_results_syn[[0, "motifs"]]
        .drop_duplicates()
        .groupby("motifs")
        .count()
    )
    return df_motifs_count_syn


def compare_motif_list(df_motifs_a, df_motifs_b):
    # Using KL divergence to compare motifs lists distribution
    set_all_mot = set(
        df_motifs_a.index.values.tolist() + df_motifs_b.index.values.tolist()
    )
    create_new_matrix = []
    for x in set_all_mot:
        list_in = []
        list_in.append(x)  # adding the name
        if x in df_motifs_a.index:
            list_in.append(df_motifs_a.loc[x][0])
        else:
            list_in.append(1)

        if x in df_motifs_b.index:
            list_in.append(df_motifs_b.loc[x][0])
        else:
            list_in.append(1)

        create_new_matrix.append(list_in)

    df_motifs = pd.DataFrame(
        create_new_matrix, columns=["motif", "motif_a", "motif_b"]
    )

    df_motifs["Diffusion_seqs"] = (
        df_motifs["motif_a"] / df_motifs["motif_a"].sum()
    )
    df_motifs["Training_seqs"] = (
        df_motifs["motif_b"] / df_motifs["motif_b"].sum()
    )
    kl_pq = rel_entr(
        df_motifs["Diffusion_seqs"].values, df_motifs["Training_seqs"].values
    )
    return np.sum(kl_pq)


def kl_comparison_between_dataset(first_dic, second_dict):
    final_comp_kl = []
    for k, v in first_dic.items():
        comp_array = []
        for k_second in second_dict.keys():
            kl_out = compare_motif_list(v, second_dict[k_second])
            comp_array.append(kl_out)
        final_comp_kl.append(comp_array)
    return final_comp_kl


def generate_heatmap(df_heat, x_label, y_label, cell_components):
    plt.clf()
    plt.rcdefaults()
    plt.rcParams["figure.figsize"] = (10, 10)
    df_plot = pd.DataFrame(df_heat)
    df_plot.columns = [x.split("_")[0] for x in cell_components]
    df_plot.index = df_plot.columns
    sns.heatmap(df_plot, cmap="Blues_r", annot=True, lw=0.1, vmax=1, vmin=0)
    plt.title(
        f"Kl divergence \n {x_label} sequences x  {y_label} sequences \n MOTIFS probabilities"
    )
    plt.xlabel(f"{x_label} Sequences  \n(motifs dist)")
    plt.ylabel(f"{y_label} \n (motifs dist)")
    plt.grid(False)
    plt.savefig(f"./graphs/{x_label}_{y_label}_kl_heatmap.png")
    # wandb.log({f"Kl divergence \n {x_label} sequences x  {y_label} sequences \n MOTIFS probabilities": plt})


def generate_similarity_metric(nucleotides):
    """Capture the syn_motifs.fasta and compare with the  dataset motifs"""
    seqs_file = open("synthetic_motifs.fasta").readlines()
    seqs_to_hotencoder = [
        one_hot_encode(s.replace("\n", ""), nucleotides, 200).T
        for s in seqs_file
        if ">" not in s
    ]

    return seqs_to_hotencoder


def get_best_match(db, x_seq):  # transforming in a function
    return (db * x_seq).sum(1).sum(1).max()


def calculate_mean_similarity(database, input_query_seqs, seq_len=200):
    final_base_max_match = np.mean(
        [get_best_match(database, x) for x in tqdm(input_query_seqs)]
    )
    return final_base_max_match / seq_len


def generate_similarity_using_train(X_train_in, nucleotides):
    convert_X_train = X_train_in.copy()
    convert_X_train[convert_X_train == -1] = 0
    generated_seqs_to_similarity = generate_similarity_metric(nucleotides)
    return calculate_mean_similarity(
        convert_X_train, generated_seqs_to_similarity
    )


# Linear Beta Schedule
def linear_beta_schedule(timesteps, beta_end=0.005):
    beta_start = 0.0001

    return torch.linspace(beta_start, beta_end, timesteps)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim) * -embeddings)

        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResBlock(nn.Module):

    """
    Iniialize a residual block with two convolutions followed by batchnorm layers
    """

    def __init__(self, in_size: int, hidden_size: int, out_size: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_size, hidden_size, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_size, out_size, 3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(hidden_size)
        self.batchnorm2 = nn.BatchNorm2d(out_size)

    def convblock(self, x):
        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = F.relu(self.batchnorm2(self.conv2(x)))
        return x

    """
    Combine output with the original input
    """

    def forward(self, x):
        return x + self.convblock(x)


class ConvBlock_2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 4, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        return x


class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super().__init__()
        """
        generic one layer FC NN for embedding things
        """
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
    )


def Downsample(dim, dim_out=None):
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


# positional embeds
class LearnedSinusoidalPosEmb(nn.Module):
    """following @crowsonkb 's lead with learned sinusoidal pos emb"""

    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


# building block modules
class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = (
            nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        )

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


class ResnetBlockClassConditioned(ResnetBlock):
    def __init__(
        self,
        dim,
        dim_out,
        *,
        num_classes,
        class_embed_dim,
        time_emb_dim=None,
        groups=8,
    ):
        super().__init__(
            dim=dim + class_embed_dim,
            dim_out=dim_out,
            time_emb_dim=time_emb_dim,
            groups=groups,
        )
        self.class_mlp = EmbedFC(num_classes, class_embed_dim)

    def forward(self, x, time_emb=None, c=None):
        emb_c = self.class_mlp(c)
        emb_c = emb_c.view(*emb_c.shape, 1, 1)
        emb_c = emb_c.expand(-1, -1, x.shape[-2], x.shape[-1])
        x = torch.cat([x, emb_c], axis=1)

        return super().forward(x, time_emb)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1), LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = (
            rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads)
            for t in qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(
            out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w
        )
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, scale=10):
        super().__init__()
        self.scale = scale
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = (
            rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads)
            for t in qkv
        )

        q, k = map(l2norm, (q, k))

        sim = einsum("b h d i, b h d j -> b h i j", q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class CrossAttention_lucas(nn.Module):
    def __init__(self, dim, heads=1, dim_head=32, scale=10):
        super().__init__()
        self.scale = scale
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x, y):
        b, c, h, w = x.shape
        b_y, c_y, h_y, w_y = y.shape

        qkv_x = self.to_qkv(x).chunk(3, dim=1)
        qkv_y = self.to_qkv(y).chunk(3, dim=1)

        q_x, k_x, v_x = (
            rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads)
            for t in qkv_x
        )

        q_y, k_y, v_y = (
            rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads)
            for t in qkv_y
        )

        q, k = map(l2norm, (q_x, k_y))

        sim = einsum("b h d i, b h d j -> b h i j", q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = einsum("b h i j, b h d j -> b h i d", attn, v_y)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


# Unet Model
class Unet_lucas(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        dim_mults=(1, 2, 4),
        channels=1,
        resnet_block_groups=8,
        learned_sinusoidal_dim=18,
        num_classes=10,
        class_embed_dim=3,
        output_attention=False,
    ):
        super().__init__()

        # determine dimensions

        channels = 1
        self.channels = channels
        # if you want to do self conditioning uncomment this
        input_channels = channels
        self.output_attention = output_attention

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, (7, 7), padding=3)
        dims = [init_dim, *(dim * m for m in dim_mults)]

        # in_out = list(zip(dims[:-1], dims[1:]))
        in_out = itertools.pairwise(dims)
        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        time_dim = dim * 4

        sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
        fourier_dim = learned_sinusoidal_dim + 1

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Downsample(dim_in, dim_out)
                        if not is_last
                        else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(
                            dim_out + dim_in, dim_out, time_emb_dim=time_dim
                        ),
                        block_klass(
                            dim_out + dim_in, dim_out, time_emb_dim=time_dim
                        ),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, 1, 1)
        self.cross_attn = EfficientAttention(
            dim=200,
            dim_head=64,
            heads=1,
            memory_efficient=True,
            q_bucket_size=1024,
            k_bucket_size=2048,
        )
        self.norm_to_cross = nn.LayerNorm(dim * 4)

    def forward(self, x, time, classes, x_self_cond=None):
        x = self.init_conv(x)
        r = x.clone()

        t_start = self.time_mlp(time)
        t_mid = t_start.clone()
        t_end = t_start.clone()
        t_cross = t_start.clone()

        if classes is not None:
            t_start += self.label_emb(classes)
            t_mid += self.label_emb(classes)
            t_end += self.label_emb(classes)
            t_cross += self.label_emb(classes)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t_start)
            h.append(x)

            x = block2(x, t_start)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t_mid)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_mid)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t_mid)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t_mid)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t_end)
        x = self.final_conv(x)
        x_reshaped = x.reshape(-1, 4, 200)
        t_cross_reshaped = t_cross.reshape(-1, 4, 200)

        crossattention_out = self.cross_attn(
            self.norm_to_cross(x_reshaped.reshape(-1, 800)).reshape(-1, 4, 200),
            context=t_cross_reshaped,
        )  # (-1,1, 4, 200)
        crossattention_out = x.view(-1, 1, 4, 200)
        x = x + crossattention_out
        if self.output_attention:
            return x, crossattention_out
        return x


# Loading data and Motifs
def motifs_from_fasta(fasta: str):
    print("Computing Motifs....")
    os.system(
        f"gimme scan {fasta} -p  JASPAR2020_vertebrates -g hg38 > train_results_motifs.bed"
    )
    df_results_seq_guime = pd.read_csv(
        "train_results_motifs.bed", sep="\t", skiprows=5, header=None
    )
    df_results_seq_guime["motifs"] = df_results_seq_guime[8].apply(
        lambda x: x.split('motif_name "')[1].split('"')[0]
    )

    df_results_seq_guime[0] = df_results_seq_guime[0].apply(
        lambda x: "_".join(x.split("_")[:-1])
    )
    df_results_seq_guime_count_out = (
        df_results_seq_guime[[0, "motifs"]]
        .drop_duplicates()
        .groupby("motifs")
        .count()
    )
    plt.rcParams["figure.figsize"] = (30, 2)
    df_results_seq_guime_count_out.sort_values(0, ascending=False).head(50)[
        0
    ].plot.bar()
    plt.title("Top 50 MOTIFS on component 0 ")
    plt.show()
    return df_results_seq_guime_count_out


def save_fasta(
    df: pd.DataFrame,
    name: str,
    num_sequences: int,
    seq_to_subset_comp: bool = False,
) -> str:
    fasta_path = f"{name}.fasta"
    save_fasta_file = open(fasta_path, "w")
    num_to_sample = df.shape[0]

    # Subsetting sequences
    if num_sequences and seq_to_subset_comp:
        num_to_sample = num_sequences

    # Sampling sequences
    print(f"Sampling {num_to_sample} sequences")
    write_fasta_component = "\n".join(
        df[["dhs_id", "sequence", "TAG"]]
        .head(num_to_sample)
        .apply(lambda x: f">{x[0]}_TAG_{x[2]}\n{x[1]}", axis=1)
        .values.tolist()
    )
    save_fasta_file.write(write_fasta_component)
    save_fasta_file.close()

    return fasta_path


def generate_motifs_and_fastas(
    df: pd.DataFrame,
    name: str,
    num_sequences: int,
    subset_list: list | None = None,
) -> dict[str, Any]:
    print("Generating Motifs and Fastas...", name)
    print("---" * 10)

    # Saving fasta
    if subset_list:
        fasta_path = save_fasta(
            df,
            f"{name}_{'_'.join([str(c) for c in subset_list])}",
            num_sequences,
        )
    else:
        fasta_path = save_fasta(df, name, num_sequences)

    # Computing motifs
    motifs = motifs_from_fasta(fasta_path)

    # Generating subset specific motifs
    final_subset_motifs = {}
    for comp, v_comp in df.groupby("TAG"):
        print(comp)
        c_fasta = save_fasta(
            v_comp, f"{name}_{comp}", num_sequences, seq_to_subset_comp=True
        )
        final_subset_motifs[comp] = motifs_from_fasta(c_fasta)

    return {
        "fasta_path": fasta_path,
        "motifs": motifs,
        "final_subset_motifs": final_subset_motifs,
        "df": df,
    }


def preprocess_data(
    input_csv: str,
    subset_list: list | None = None,
    limit_total_sequences: int | None = None,
    number_of_sequences_to_motif_creation: int = 1000,
    save_output: bool = True,
):
    # Reading the csv file
    df = pd.read_csv(input_csv, sep="\t")

    # Subsetting the dataframe
    if subset_list:
        print(" or ".join([f"TAG == {c}" for c in subset_list]))
        df = df.query(" or ".join([f'TAG == "{c}" ' for c in subset_list]))
        print("Subseting...")

    # Limiting the total number of sequences
    if limit_total_sequences:
        print(f"Limiting total sequences to {limit_total_sequences}")
        df = df.sample(limit_total_sequences)

    # Creating train/test/shuffle groups
    df_test = df[df["chr"] == "chr1"].reset_index(drop=True)
    df_train_shuffled = df[df["chr"] == "chr2"].reset_index(drop=True)
    df_train = df[(df["chr"] != "chr1") & (df["chr"] != "chr2")].reset_index(
        drop=True
    )

    df_train_shuffled["sequence"] = df_train_shuffled["sequence"].apply(
        lambda x: "".join(random.sample(list(x), len(x)))
    )

    # Getting motif information from the sequences
    train = generate_motifs_and_fastas(
        df_train, "train", number_of_sequences_to_motif_creation, subset_list
    )
    test = generate_motifs_and_fastas(
        df_test, "test", number_of_sequences_to_motif_creation, subset_list
    )
    train_shuffled = generate_motifs_and_fastas(
        df_train_shuffled,
        "train_shuffled",
        number_of_sequences_to_motif_creation,
        subset_list,
    )

    combined_dict = {
        "train": train,
        "test": test,
        "train_shuffled": train_shuffled,
    }

    # Writing to pickle
    if save_output:
        # Saving all train, test, train_shuffled dictionaries to pickle
        with open("dna_diffusion/data/encode_data.pkl", "wb") as f:
            pickle.dump(combined_dict, f)

    return combined_dict


# Sequence Dataloader
class SequenceDataset(Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, seqs, c, transform=None):
        "Initialization"
        self.seqs = seqs
        self.c = c
        self.transform = transform

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.seqs)

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        image = self.seqs[index]

        x = self.transform(image)

        y = self.c[index]

        return x, y


def load_data(
    data_path: str = "K562_hESCT0_HepG2_GM12878_12k_sequences_per_group.txt",
    saved_data_path: str = "encode_data_dict.npy",
    subset_list: list = [
        "GM12878_ENCLB441ZZZ",
        "hESCT0_ENCLB449ZZZ",
        "K562_ENCLB843GMH",
        "HepG2_ENCLB029COU",
    ],
    limit_total_sequences: int = 0,
    num_sampling_to_compare_cells: int = 1000,
    load_saved_data: bool = False,
    batch_size: int = 240,
):
    # Preprocessing data
    if load_saved_data:
        with open(saved_data_path, "rb") as f:
            encode_data = pickle.load(f)

    else:
        encode_data = preprocess_data(
            input_csv=data_path,
            subset_list=subset_list,
            limit_total_sequences=limit_total_sequences,
            number_of_sequences_to_motif_creation=num_sampling_to_compare_cells,
        )

    # Splitting enocde data into train/test/shuffle
    train_motifs = encode_data["train"]["motifs"]
    train_motifs_cell_specific = encode_data["train"]["final_subset_motifs"]

    test_motifs = encode_data["test"]["motifs"]
    test_motifs_cell_specific = encode_data["test"]["final_subset_motifs"]

    shuffle_motifs = encode_data["train_shuffled"]["motifs"]
    shuffle_motifs_cell_specific = encode_data["train_shuffled"][
        "final_subset_motifs"
    ]

    # Creating sequence dataset
    df = encode_data["train"]["df"]
    nucleotides = ["A", "C", "G", "T"]
    x_train_seq = np.array(
        [
            one_hot_encode(x, nucleotides, 200)
            for x in df["sequence"]
            if "N" not in x
        ]
    )
    X_train = np.array([x.T.tolist() for x in x_train_seq])
    X_train[X_train == 0] = -1

    # Creating labels
    tag_to_numeric = {x: n for n, x in enumerate(df["TAG"].unique(), 1)}
    numeric_to_tag = dict(enumerate(df["TAG"].unique(), 1))
    cell_types = list(numeric_to_tag.keys())
    x_train_cell_type = torch.tensor([tag_to_numeric[x] for x in df["TAG"]])

    # Wrapping data into dataloader
    tf = T.Compose([T.ToTensor()])
    seq_dataset = SequenceDataset(
        seqs=X_train, c=x_train_cell_type, transform=tf
    )
    train_dl = DataLoader(
        seq_dataset, batch_size, shuffle=True, num_workers=96, pin_memory=True
    )

    # Collecting variables into a dict
    encode_data_dict = {
        "train_motifs": train_motifs,
        "train_motifs_cell_specific": train_motifs_cell_specific,
        "test_motifs": test_motifs,
        "test_motifs_cell_specific": test_motifs_cell_specific,
        "shuffle_motifs": shuffle_motifs,
        "shuffle_motifs_cell_specific": shuffle_motifs_cell_specific,
        "tag_to_numeric": tag_to_numeric,
        "numeric_to_tag": numeric_to_tag,
        "cell_types": cell_types,
    }

    return encode_data_dict, train_dl


class Diffusion(nn.Module):
    def __init__(
        self,
        model,
        timesteps,
    ):
        super().__init__()
        self.model = model
        self.timesteps = timesteps

        # Diffusion params
        betas = linear_beta_schedule(timesteps, beta_end=0.2)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Store as buffers
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )

    @property
    def device(self):
        return self.betas.device

    @torch.no_grad()
    def sample(self, classes, shape, cond_weight):
        return self.p_sample_loop(
            classes=classes,
            image_size=shape,
            cond_weight=cond_weight,
        )

    @torch.no_grad()
    def p_sample_loop(
        self, classes, image_size, cond_weight, get_cross_map=False
    ):
        b = image_size[0]
        device = self.device

        img = torch.randn(image_size, device=device)
        imgs = []
        cross_images_final = []

        if classes is not None:
            n_sample = classes.shape[0]
            context_mask = torch.ones_like(classes).to(device)
            # make 0 index unconditional
            # double the batch
            classes = classes.repeat(2)
            context_mask = context_mask.repeat(2)
            context_mask[n_sample:] = 0.0
            sampling_fn = partial(
                self.p_sample_guided,
                classes=classes,
                cond_weight=cond_weight,
                context_mask=context_mask,
            )

        else:
            sampling_fn = partial(self.p_sample)

        for i in reversed(range(0, self.timesteps)):
            img, cross_matrix = sampling_fn(
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
    def p_sample(self, x, t, t_index):
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
            x
            - betas_t * self.model(x, time=t) / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_guided(
        self, x, classes, t, t_index, context_mask, cond_weight
    ):
        # adapted from: https://openreview.net/pdf?id=qw8AKxfYbI
        batch_size = x.shape[0]
        device = self.device
        # double to do guidance with
        t_double = t.repeat(2).to(device)
        x_double = x.repeat(2, 1, 1, 1).to(device)
        betas_t = extract(self.betas, t_double, x_double.shape, device)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t_double, x_double.shape, device
        )
        sqrt_recip_alphas_t = extract(
            self.sqrt_recip_alphas, t_double, x_double.shape, device
        )

        # classifier free sampling interpolates between guided and non guided using `cond_weight`
        classes_masked = classes * context_mask
        classes_masked = classes_masked.type(torch.long)
        # model = self.accelerator.unwrap_model(self.model)
        self.model.output_attention = True
        preds, cross_map_full = self.model(
            x_double, time=t_double, classes=classes_masked
        )
        self.model.output_attention = False
        cross_map = cross_map_full[:batch_size]
        eps1 = (1 + cond_weight) * preds[:batch_size]
        eps2 = cond_weight * preds[batch_size:]
        x_t = eps1 - eps2

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t[:batch_size] * (
            x
            - betas_t[:batch_size]
            * x_t
            / sqrt_one_minus_alphas_cumprod_t[:batch_size]
        )

        if t_index == 0:
            return model_mean, cross_map
        else:
            posterior_variance_t = extract(
                self.posterior_variance, t, x.shape, device
            )
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return (
                model_mean + torch.sqrt(posterior_variance_t) * noise,
                cross_map,
            )

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, torch.randn_like(x_start))
        device = self.device

        sqrt_alphas_cumprod_t = extract(
            self.sqrt_alphas_cumprod, t, x_start.shape, device
        )
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape, device
        )

        return (
            sqrt_alphas_cumprod_t * x_start
            + sqrt_one_minus_alphas_cumprod_t * noise
        )

    def p_losses(
        self, x_start, t, classes, noise=None, loss_type="huber", p_uncond=0.1
    ):
        device = self.device
        noise = default(noise, torch.randn_like(x_start))

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        context_mask = torch.bernoulli(
            torch.zeros(classes.shape[0]) + (1 - p_uncond)
        ).to(device)

        # Mask for unconditional guidance
        classes = classes * context_mask
        # nn.Embedding needs type to be long, multiplying with mask changes type
        classes = classes.type(torch.long)
        predicted_noise = self.model(x_noisy, t, classes)

        if loss_type == "l1":
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == "l2":
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()

        return loss

    def forward(self, x, classes):
        device = self.device
        x = x.type(torch.float32)
        classes = classes.type(torch.long)
        b = x.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=device).long()

        return self.p_losses(x, t, classes)


class Trainer:
    def __init__(
        self,
        data_path: str = "dnadiffusion/data/K562_hESCT0_HepG2_GM12878_12k_sequences_per_group.txt",
        limit_total_sequences: int = 0,
        model_name: str = "model_48k_sequences_per_group_K562_hESCT0_HepG2_GM12878_12k.pt",
        load_saved_data: bool = True,
        save_model_by_epoch: bool = False,
        save_and_sample_every: int = 10,
        epochs: int = 10000,
        epochs_loss_show: int = 10,
        num_sampling_to_compare_cells: int = 1000,
        timesteps: int = 50,
        batch_size: int = 240,
        channels: int = 1,
        image_size: int = 200,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.load_saved_model = load_saved_data
        self.save_model_by_epoch = save_model_by_epoch
        self.save_and_sample_every = save_and_sample_every
        self.epochs = epochs
        self.epochs_loss_show = epochs_loss_show
        self.num_sampling_to_compare_cells = num_sampling_to_compare_cells
        self.timesteps = timesteps
        self.batch_size = batch_size
        self.channels = channels
        self.image_size = image_size

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            kwargs_handlers=[ddp_kwargs],
            split_batches=True,
            log_with=["wandb"],
            mixed_precision="bf16",
        )
        self.device = self.accelerator.device

        # Loading data
        self.encode_data, self.train_dl = load_data(
            data_path=data_path,
            saved_data_path="dnadiffusion/data/encode_data.pkl",
            subset_list=[
                "GM12878_ENCLB441ZZZ",
                "hESCT0_ENCLB449ZZZ",
                "K562_ENCLB843GMH",
                "HepG2_ENCLB029COU",
            ],
            limit_total_sequences=limit_total_sequences,
            num_sampling_to_compare_cells=num_sampling_to_compare_cells,
            load_saved_data=load_saved_data,
            batch_size=batch_size,
        )

        # Preparing model/optimizer/EMA/dataloader
        model = Unet_lucas(
            dim=200, channels=1, dim_mults=(1, 2, 4), resnet_block_groups=4
        )

        # Creating diffusion_model
        self.diffusion_model = Diffusion(
            model,
            timesteps=self.timesteps,
        )

        self.optimizer = Adam(self.diffusion_model.parameters(), lr=1e-4)

        if self.accelerator.is_main_process:
            self.ema = EMA(0.995)
            self.ema_model = (
                copy.deepcopy(self.diffusion_model).eval().requires_grad_(False)
            )

        self.train_kl, self.test_kl, self.shuffle_kl = 1, 1, 1
        self.seq_similarity = 0.38
        (
            self.diffusion_model,
            self.optimizer,
            self.train_dl,
        ) = self.accelerator.prepare(
            self.diffusion_model, self.optimizer, self.train_dl
        )

    def train(self):
        """if self.accelerator.is_main_process:
        self.accelerator.init_trackers(
            "dnadiffusion",
            init_kwargs={
                "wandb": {
                    "notes" : "testing wandb accelerate script"
                }
            }
        )
        """

        for epoch in tqdm(range(self.start_epoch, self.epochs)):
            self.diffusion_model.train()

            total_loss = 0.0
            for step, batch in enumerate(self.train_dl):
                x, y = batch
                with self.accelerator.autocast():
                    loss = self.diffusion_model(x, y)
                    total_loss += loss.item()

                self.optimizer.zero_grad()
                self.accelerator.backward(loss)
                self.accelerator.wait_for_everyone()
                self.optimizer.step()

                self.accelerator.wait_for_everyone()
                if self.accelerator.is_main_process:
                    self.ema.step_ema(
                        self.ema_model,
                        self.accelerator.unwrap_model(self.diffusion_model),
                    )

            # if (epoch % self.epochs_loss_show) == 0:
            if epoch % self.epochs_loss_show == 0:
                if self.accelerator.is_main_process:
                    """self.accelerator.log({"train": self.train_kl,
                    "test": self.test_kl,
                    "shuffle": self.shuffle_kl,
                    "loss" : loss.item(),
                    "seq_similarity": self.seq_similarity}, step=epoch)
                    """
                    print(f" Epoch {epoch} Loss:", loss.item())

            # if epoch != 0 and epoch % self.save_and_sample_every == 0 and self.accelerator.is_main_process:
            if (
                epoch != 0
                and epoch % self.save_and_sample_every == 0
                and self.accelerator.is_main_process
            ):
                self.diffusion_model.eval()

                print("saving")
                synt_df = create_sample(
                    self.accelerator.unwrap_model(self.diffusion_model),
                    conditional_numeric_to_tag=self.encode_data[
                        "numeric_to_tag"
                    ],
                    cell_types=self.encode_data["cell_types"],
                    number_of_samples=int(
                        self.num_sampling_to_compare_cells / 10
                    ),
                )
                self.train_kl = compare_motif_list(
                    synt_df, self.encode_data["train_motifs"]
                )
                self.test_kl = compare_motif_list(
                    synt_df, self.encode_data["test_motifs"]
                )
                self.shuffle_kl = compare_motif_list(
                    synt_df, self.encode_data["shuffle_motifs"]
                )
                print("Similarity", self.seq_similarity, "Similarity")
                print("KL_TRAIN", self.train_kl, "KL")
                print("KL_TEST", self.test_kl, "KL")
                print("KL_SHUFFLE", self.shuffle_kl, "KL")

            if (
                epoch != 0
                and epoch % 500 == 0
                and self.accelerator.is_main_process
            ):
                model_path = (
                    f"dnadiffusion/checkpoints/epoch_{epoch}_" + self.model_name
                )
                self.save(epoch, model_path)

    # Saving model
    def save(self, epoch, results_path):
        checkpoint_dict = {
            "model": self.accelerator.get_state_dict(self.diffusion_model),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
            "ema_model": self.accelerator.get_state_dict(self.ema_model),
            "train_kl": self.train_kl,
            "test_kl": self.test_kl,
            "shuffle_kl": self.shuffle_kl,
            "seq_similarity": self.seq_similarity,
        }
        torch.save(checkpoint_dict, results_path)


if __name__ == "__main__":
    # temp_dir = tempfile.mkdtemp()
    # os.environ["XDG_CACHE_HOME"] = temp_dir

    trainer = Trainer()
    trainer.train()
