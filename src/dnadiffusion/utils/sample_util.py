import os
from typing import Optional

import numpy as np
import pandas as pd
import torch

from dnadiffusion.utils.utils import convert_to_seq


def create_sample(
    diffusion_model,
    cell_types: list,
    conditional_numeric_to_tag: dict,
    number_of_samples: int = 20,
    specific_group: bool = False,
    group_number: Optional[list] = None,
    cond_weight_to_metric: int = 0,
    save_timesteps: bool = False,
    save_dataframe: bool = False,
    generate_attention_maps: bool = False,
):
    print("sample_util")
    nucleotides = ["A", "C", "G", "T"]
    final_sequences = []
    cross_maps = []
    for n_a in range(number_of_samples):
        print(n_a)
        sample_bs = 10
        if specific_group:
            sampled = torch.from_numpy(np.array([group_number] * sample_bs))
        else:
            sampled = torch.from_numpy(np.random.choice(cell_types, sample_bs))

        classes = sampled.float().to(diffusion_model.device)

        if generate_attention_maps:
            sampled_images, cross_att_values = diffusion_model.sample_cross(
                classes, (sample_bs, 1, 4, 200), cond_weight_to_metric
            )
            # save cross attention maps in a numpy array
            np.save(f"cross_att_values_{conditional_numeric_to_tag[group_number]}.npy", cross_att_values)

        else:
            sampled_images = diffusion_model.sample(classes, (sample_bs, 1, 4, 200), cond_weight_to_metric)

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

    save_motifs_syn = open("synthetic_motifs.fasta", "w")
    save_motifs_syn.write("\n".join(final_sequences))
    save_motifs_syn.close()
    os.system("gimme scan synthetic_motifs.fasta -p JASPAR2020_vertebrates -g hg38 > syn_results_motifs.bed")
    df_results_syn = pd.read_csv("syn_results_motifs.bed", sep="\t", skiprows=5, header=None)

    df_results_syn["motifs"] = df_results_syn[8].apply(lambda x: x.split('motif_name "')[1].split('"')[0])
    df_results_syn[0] = df_results_syn[0].apply(lambda x: "_".join(x.split("_")[:-1]))
    df_motifs_count_syn = df_results_syn[[0, "motifs"]].drop_duplicates().groupby("motifs").count()
    return df_motifs_count_syn
