import pandas as pd
from typing import Any, Dict, List
import os


def subset_by_experiment(df: pd.DataFrame, subset_components: List[str]) -> pd.DataFrame:
    df_generate = df
    if subset_components is not None:
        query = " or ".join([f'TAG == "{c}" ' for c in subset_components])
        df_generate = df_generate.query(query).copy()
        print("Subsetting...")

    return df_generate


def read_master_dataset(input_csv: str, limit_total_sequences=0, change_comp_index=False) -> pd.DataFrame:
    df = pd.read_csv(input_csv, sep="\t")
    if change_comp_index:
        df["component"] = df["component"] + 1

    if limit_total_sequences > 0:
        print(f"Limiting total sequences {limit_total_sequences}")
        df = df.sample(limit_total_sequences)

    return df


def motifs_from_fasta(fasta: str):
    print("Computing Motifs....")
    os.system(f"gimme scan {fasta} -p  JASPAR2020_vertebrates -g hg38 > train_results_motifs.bed")

    df_results_seq_guime = pd.read_csv("train_results_motifs.bed", sep="\t", skiprows=5, header=None)
    df_results_seq_guime["motifs"] = df_results_seq_guime[8].apply(lambda x: x.split('motif_name "')[1].split('"')[0])

    df_results_seq_guime[0] = df_results_seq_guime[0].apply(lambda x: "_".join(x.split("_")[:-1]))
    df_results_seq_guime_count_out = df_results_seq_guime[[0, "motifs"]].drop_duplicates().groupby("motifs").count()
    return df_results_seq_guime_count_out


def get_motif(
    df_train: pd.DataFrame,
    df_shuffled: pd.DataFrame,
    df_test: pd.DataFrame,
    subset_components: List[str],
    number_of_sequences_to_motif_creation: int,
) -> None:
    train = generate_motifs_and_fastas(
        df_train,
        "train",
        subset_components=subset_components,
        number_of_sequences_to_motif_creation=number_of_sequences_to_motif_creation,
    )
    test = generate_motifs_and_fastas(
        df_test,
        "test",
        subset_components=subset_components,
        number_of_sequences_to_motif_creation=number_of_sequences_to_motif_creation,
    )
    train_shuffle = generate_motifs_and_fastas(
        df_shuffled,
        "val",
        subset_components=subset_components,
        number_of_sequences_to_motif_creation=number_of_sequences_to_motif_creation,
    )
    return train, test, train_shuffle


def generate_motifs_and_fastas(
    df: pd.DataFrame, name: str, subset_components: List[str], number_of_sequences_to_motif_creation
) -> Dict[str, Any]:
    """return fasta anem , and dict with components motifs"""
    print("Generating Fasta and Motis:", name)
    print("---" * 10)
    name_fasta = f"{name}_{'_'.join([str(c) for c in subset_components])}"
    fasta_saved = save_fasta(df, name_fasta, number_of_sequences_to_motif_creation)
    print("FASTA SAVED", fasta_saved)
    print("Generating Motifs (all seqs)")
    motif_all_components = motifs_from_fasta(fasta_saved)
    print("Generating Motifs per component")
    train_comp_motifs_dict = generate_motifs_components(df, number_of_sequences_to_motif_creation)

    return {
        "fasta_name": fasta_saved,
        "motifs": motif_all_components,
        "motifs_per_components_dict": train_comp_motifs_dict,
        "dataset": df,
    }


def save_fasta(
    df: pd.DataFrame, name_fasta: str, to_seq_groups_comparison: bool = False, number_of_sequences_to_motif_creation=1
) -> str:
    fasta_final_name = name_fasta + ".fasta"
    save_fasta_file = open(fasta_final_name, "w")
    number_to_sample = df.shape[0]

    if to_seq_groups_comparison and number_of_sequences_to_motif_creation:
        number_to_sample = number_of_sequences_to_motif_creation

    print(number_to_sample, "#seq used")
    write_fasta_component = "\n".join(
        df[["dhs_id", "sequence", "TAG"]]
        .head(number_to_sample)
        .apply(lambda x: f">{x[0]}_TAG_{x[2]}\n{x[1]}", axis=1)
        .values.tolist()
    )
    save_fasta_file.write(write_fasta_component)
    save_fasta_file.close()
    return fasta_final_name


def generate_motifs_components(df: pd.DataFrame, number_of_sequences_to_motif_creation) -> dict:
    final_comp_values = {}
    for comp, v_comp in df.groupby("TAG"):
        print(comp)
        print("number of sequences used to generate the motifs")
        name_c_fasta = save_fasta(
            v_comp,
            "temp_component",
            to_seq_groups_comparison=True,
            number_of_sequences_to_motif_creation=number_of_sequences_to_motif_creation,
        )
        final_comp_values[comp] = motifs_from_fasta(name_c_fasta)
        return final_comp_values
