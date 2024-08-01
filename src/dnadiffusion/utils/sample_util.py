import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from dnadiffusion.utils.utils import convert_to_seq


def create_sample(
    model: torch.nn.Module,
    cell_types: list,
    sample_bs: int,
    conditional_numeric_to_tag: dict,
    number_of_samples: int = 1000,
    group_number: list | None = None,
    cond_weight_to_metric: int = 0,
    save_timesteps: bool = False,
    save_dataframe: bool = False,
    generate_attention_maps: bool = False,
):
    nucleotides = ["A", "C", "G", "T"]
    final_sequences = []
    num_batches = number_of_samples // sample_bs
    for n_a in tqdm(range(num_batches)):
        if group_number:
            sampled = torch.from_numpy(np.array([group_number] * sample_bs))
        else:
            sampled = torch.from_numpy(np.random.choice(cell_types, sample_bs))

        classes = sampled.float().to(model.device)

        if generate_attention_maps:
            sampled_images, cross_att_values = model.sample_cross(
                classes, (sample_bs, 1, 4, 200), cond_weight_to_metric
            )
            # save cross attention maps in a numpy array
            np.save(f"cross_att_values_{conditional_numeric_to_tag[group_number]}.npy", cross_att_values)

        else:
            sampled_images = model.sample(classes, (sample_bs, 1, 4, 200), cond_weight_to_metric)

        if save_timesteps:
            seqs_to_df = {}
            for en, step in enumerate(sampled_images):
                seqs_to_df[en] = [convert_to_seq(x, nucleotides) for x in step]
            final_sequences.append(pd.DataFrame(seqs_to_df))

        if save_dataframe:
            # Only using the last timestep
            for en, step in enumerate(sampled_images[-1]):
                final_sequences.append(convert_to_seq(step, nucleotides))
        else:
            for n_b, x in enumerate(sampled_images[-1]):
                seq_final = f">seq_test_{n_a}_{n_b}\n" + "".join(
                    [nucleotides[s] for s in np.argmax(x.reshape(4, 200), axis=0)]
                )
                final_sequences.append(seq_final)

    if save_timesteps:
        # Saving dataframe containing sequences for each timestep
        pd.concat(final_sequences, ignore_index=True).to_csv(
            f"final_{conditional_numeric_to_tag[group_number]}.txt",
            header=True,
            sep="\t",
            index=False,
        )
        return

    if save_dataframe:
        # Saving list of sequences to txt file
        with open(f"final_{conditional_numeric_to_tag[group_number]}.txt", "w") as f:
            f.write("\n".join(final_sequences))
        return

    df_motifs_count_syn = extract_motifs(final_sequences)
    return df_motifs_count_syn


def extract_motifs(sequence_list: list):
    """Extract motifs from a list of sequences"""
    motifs = open("synthetic_motifs.fasta", "w")
    motifs.write("\n".join(sequence_list))
    motifs.close()
    os.system("gimme scan synthetic_motifs.fasta -p JASPAR2020_vertebrates -g hg38 -n 20> syn_results_motifs.bed")
    df_results_syn = pd.read_csv("syn_results_motifs.bed", sep="\t", skiprows=5, header=None)

    df_results_syn["motifs"] = df_results_syn[8].apply(lambda x: x.split('motif_name "')[1].split('"')[0])
    df_results_syn[0] = df_results_syn[0].apply(lambda x: "_".join(x.split("_")[:-1]))
    df_motifs_count_syn = df_results_syn[[0, "motifs"]].groupby("motifs").count()
    return df_motifs_count_syn


def convert_sample_to_fasta(sample_path: list):
    """Convert cell specific samples to a fasta format"""
    sequences = []
    samples = pd.read_csv(sample_path, sep="\t", header=None)
    # Extract each line of the dataframe into a list
    samples_list = samples[0].tolist()
    # Convert into a fasta format
    for i, seq in enumerate(samples_list):
        sequences.append(f">sequence_{i}\n" + seq)
    return sequences
