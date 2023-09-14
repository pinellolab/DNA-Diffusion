import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.special import rel_entr
from tqdm import tqdm

from dnadiffusion.utils.sample_util import convert_sample_to_fasta, create_sample, extract_motifs
from dnadiffusion.utils.utils import one_hot_encode


def compare_motif_list(df_motifs_a: pd.DataFrame, df_motifs_b: pd.DataFrame):
    # Using KL divergence to compare motifs lists distribution
    set_all_mot = set(df_motifs_a.index.values.tolist() + df_motifs_b.index.values.tolist())
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

    df_motifs = pd.DataFrame(create_new_matrix, columns=["motif", "motif_a", "motif_b"])

    df_motifs["Diffusion_seqs"] = df_motifs["motif_a"] / df_motifs["motif_a"].sum()
    df_motifs["Training_seqs"] = df_motifs["motif_b"] / df_motifs["motif_b"].sum()
    kl_pq = rel_entr(df_motifs["Diffusion_seqs"].values, df_motifs["Training_seqs"].values)
    return np.sum(kl_pq)


def kl_comparison_between_dataset(first_dict: dict, second_dict: dict):
    final_comp_kl = []
    for _, v in first_dict.items():
        comp_array = []
        for k_second in second_dict.keys():
            kl_out = compare_motif_list(v, second_dict[k_second])
            comp_array.append(kl_out)
        final_comp_kl.append(comp_array)
    return final_comp_kl


def kl_heatmap(
    cell_list: list,
    target_cells_dict: dict,
):
    final_comp_kl = []
    for cell in cell_list:
        comparison_array = []
        file_name = next(f for f in os.listdir(os.getcwd()) if cell in f)
        sequences = convert_sample_to_fasta(file_name)
        motifs = extract_motifs(sequences)
        for c in cell_list:
            kl_out = compare_motif_list(motifs, target_cells_dict[f"{c}"])
            comparison_array.append(kl_out)
        final_comp_kl.append(comparison_array)
    return final_comp_kl


def generate_heatmap(df_heat: pd.DataFrame, x_label: str, y_label: str, cell_list: list):
    plt.clf()
    plt.rcdefaults()
    plt.rcParams["figure.figsize"] = (10, 10)
    df_plot = pd.DataFrame(df_heat)
    df_plot.columns = [x.split("_")[0] for x in cell_list]
    df_plot.index = df_plot.columns
    sns.heatmap(df_plot, cmap="Blues_r", annot=True, lw=0.1, vmax=1, vmin=0)
    plt.title(f"Kl divergence \n {x_label} sequences x  {y_label} sequences \n MOTIFS probabilities")
    plt.xlabel(f"{x_label} Sequences  \n(motifs dist)")
    plt.ylabel(f"{y_label} \n (motifs dist)")
    plt.grid(False)
    plt.savefig(f"./{x_label}_{y_label}_kl_heatmap.png")


def generate_similarity_metric():
    """Capture the syn_motifs.fasta and compare with the  dataset motifs"""
    nucleotides = ["A", "C", "G", "T"]
    seqs_file = open("synthetic_motifs.fasta").readlines()
    seqs_to_hotencoder = [one_hot_encode(s.replace("\n", ""), nucleotides, 200).T for s in seqs_file if ">" not in s]

    return seqs_to_hotencoder


def get_best_match(db, x_seq):  # transforming in a function
    return (db * x_seq).sum(1).sum(1).max()


def calculate_mean_similarity(database, input_query_seqs, seq_len=200):
    final_base_max_match = np.mean([get_best_match(database, x) for x in tqdm(input_query_seqs)])
    return final_base_max_match / seq_len


def generate_similarity_using_train(X_train_in):
    convert_X_train = X_train_in.copy()
    convert_X_train[convert_X_train == -1] = 0
    generated_seqs_to_similarity = generate_similarity_metric()
    return calculate_mean_similarity(convert_X_train, generated_seqs_to_similarity)
