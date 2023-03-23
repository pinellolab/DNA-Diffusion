import os
from typing import Callable, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.special import rel_entr
from tqdm.auto import tqdm


def motif_scoring_KL_divergence(original: pd.Series, generated: pd.Series) -> torch.Tensor:
    """
    This function encapsulates the logic of evaluating the KL divergence metric
    between two sequences.
    Returns
    -------
    kl_divergence: Float
      The KL divergence between the input and output (generated)
      sequences' distribution
    """

    kl_pq = rel_entr(original, generated)
    return np.sum(kl_pq)


def compare_motif_list(
    df_motifs_a: pd.DataFrame,
    df_motifs_b: pd.DataFrame,
    motif_scoring_metric: Callable = motif_scoring_KL_divergence,
    plot_motif_probs: bool = False,
) -> torch.Tensor:
    """
    This function encapsulates the logic of evaluating the difference between the distribution
    of frequencies between generated (diffusion/df_motifs_a) and the input (training/df_motifs_b) for an arbitrary metric ("motif_scoring_metric")

    Please note that some metrics, like KL_divergence, are not metrics in official sense. Reason
    for that is that they dont satisfy certain properties, such as in KL case, the simmetry property.
    Hence it makes a big difference what are the positions of input.
    """
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
    if plot_motif_probs:
        plt.rcParams["figure.figsize"] = (3, 3)
        sns.regplot(x="Diffusion_seqs", y="Training_seqs", data=df_motifs)
        plt.xlabel("Diffusion Seqs")
        plt.ylabel("Training Seqs")
        plt.title("Motifs Probs")
        plt.show()

    return motif_scoring_metric(df_motifs["Diffusion_seqs"].values, df_motifs["Training_seqs"].values)


def sampling_to_metric(
    model,
    cell_types,
    image_size,
    nucleotides,
    number_of_samples=20,
    specific_group=False,
    group_number=None,
    cond_weight_to_metric=0,
):
    """
    Might need to add to the DDPM class since if we can't call the sample() method outside PyTorch Lightning.

    This function encapsulates the logic of sampling from the trained model in order to generate counts of the motifs.
    The reasoning is that we are interested only in calculating the evaluation metric
    for the count of occurances and not the nucleic acids themselves.
    """
    final_sequences = []
    for n_a in tqdm(range(number_of_samples)):
        sample_bs = 10
        if specific_group:
            sampled = torch.from_numpy(np.array([group_number] * sample_bs))
            print("specific")
        else:
            sampled = torch.from_numpy(np.random.choice(cell_types, sample_bs))

        random_classes = sampled.float().cuda()
        sampled_images = model.sample(
            classes=random_classes,
            image_size=image_size,
            batch_size=sample_bs,
            channels=1,
            cond_weight=cond_weight_to_metric,
        )
        for n_b, x in enumerate(sampled_images[-1]):
            seq_final = f">seq_test_{n_a}_{n_b}\n" + "".join(
                [nucleotides[s] for s in np.argmax(x.reshape(4, 200), axis=0)]
            )
            final_sequences.append(seq_final)

    save_motifs_syn = open("synthetic_motifs.fasta", "w")

    save_motifs_syn.write("\n".join(final_sequences))
    save_motifs_syn.close()

    # Scan for motifs
    os.system("gimme scan synthetic_motifs.fasta -p   JASPAR2020_vertebrates -g hg38 > syn_results_motifs.bed")
    df_results_syn = pd.read_csv("syn_results_motifs.bed", sep="\t", skiprows=5, header=None)
    df_results_syn["motifs"] = df_results_syn[8].apply(lambda x: x.split('motif_name "')[1].split('"')[0])
    df_results_syn[0] = df_results_syn[0].apply(lambda x: "_".join(x.split("_")[:-1]))
    df_motifs_count_syn = df_results_syn[[0, "motifs"]].drop_duplicates().groupby("motifs").count()
    plt.rcParams["figure.figsize"] = (30, 2)
    df_motifs_count_syn.sort_values(0, ascending=False).head(50)[0].plot.bar()
    plt.show()

    return df_motifs_count_syn


def metric_comparison_between_components(
    original_data: Dict,
    generated_data: Dict,
    x_label_plot: str,
    y_label_plot: str,
    cell_components,
) -> None:
    """
    This functions takes as inputs dictionaries, which contain as keys different components (cell types)
    and as values the distribution of occurances of different motifs. These two dictionaries represent two different datasets, i.e.
    generated dataset and the input (train) dataset.

    The goal is to then plot a the main evaluation metric (KL or otherwise) across all different types of cell types
    in a heatmap fashion.
    """
    ENUMARATED_CELL_NAME = """7 Trophoblasts
        5 CD8_cells
        15 CD34_cells
        9 Fetal_heart
        12 Fetal_muscle
        14 HMVEC(vascular)
        3 hESC(Embryionic)
        8 Fetal(Neural)
        13 Intestine
        2 Skin(stromalA)
        4 Fibroblast(stromalB)
        6 Renal(Cancer)
        16 Esophageal(Cancer)
        11 Fetal_Lung
        10 Fetal_kidney
        1 Tissue_Invariant""".split(
        "\n"
    )
    CELL_NAMES = {int(x.split(" ")[0]): x.split(" ")[1] for x in ENUMARATED_CELL_NAME}

    final_comparison_all_components = []
    for components_1, motif_occurance_frequency in original_data.items():
        comparisons_single_component = []
        for components_2 in generated_data.keys():
            compared_motifs_occurances = compare_motif_list(motif_occurance_frequency, generated_data[components_2])
            comparisons_single_component.append(compared_motifs_occurances)

        final_comparison_all_components.append(comparisons_single_component)

    plt.rcParams["figure.figsize"] = (10, 10)
    df_plot = pd.DataFrame(final_comparison_all_components)
    df_plot.columns = [CELL_NAMES[x] for x in cell_components]
    df_plot.index = df_plot.columns
    sns.heatmap(df_plot, cmap="Blues_r", annot=True, lw=0.1, vmax=1, vmin=0)
    plt.title(f"Kl divergence \n {x_label_plot} sequences x  {y_label_plot} sequences \n MOTIFS probabilities")
    plt.xlabel(f"{x_label_plot} Sequences  \n(motifs dist)")
    plt.ylabel(f"{y_label_plot} \n (motifs dist)")
