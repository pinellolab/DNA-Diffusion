import os
import random
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torchvision.transforms as T
from torch.utils.data import Dataset


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
        df_results_seq_guime[[0, "motifs"]].drop_duplicates().groupby("motifs").count()
    )
    plt.rcParams["figure.figsize"] = (30, 2)
    df_results_seq_guime_count_out.sort_values(0, ascending=False).head(50)[
        0
    ].plot.bar()
    plt.title("Top 50 MOTIFS on component 0 ")
    plt.show()
    return df_results_seq_guime_count_out


class LoadingData:
    def __init__(
        self,
        input_csv: str,
        subset_components: list,
        sample_number: int = 0,
        change_component_index: bool = True,
        limit_total_sequences: Optional[int] = None,
        number_of_sequences_to_motif_creation: Optional[int] = None,
    ) -> None:
        self.csv = input_csv
        self.limit_total_sequences = limit_total_sequences
        self.sample_number = sample_number
        self.subset_components = subset_components
        self.change_comp_index = change_component_index
        self.data = self.read_csv()
        self.df_generate = self.experiment()
        (
            self.df_train_in,
            self.df_test_in,
            self.df_train_shuffled_in,
        ) = self.create_train_groups()
        self.number_of_sequences_to_motif_creation = (
            number_of_sequences_to_motif_creation
        )
        self.train = None
        self.test = None
        self.train_shuffle = None
        self.get_motif()

    def read_csv(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv, sep="\t")
        if self.change_comp_index:
            df["component"] = df["component"] + 1

        if self.limit_total_sequences:
            print(f"Limiting total sequences {self.limit_total_sequences}")
            df = df.sample(self.limit_total_sequences)

        # change this in simon original table
        df.columns = [c.replace("seqname", "chr") for c in df.columns.values]
        return df

    def experiment(self) -> pd.DataFrame:
        df_generate = self.data.copy()
        if self.subset_components is not None and type(self.subset_components) == list:
            print(" or ".join([f"TAG == {c}" for c in self.subset_components]))
            df_generate = df_generate.query(
                " or ".join([f'TAG == "{c}" ' for c in self.subset_components])
            ).copy()
            print("Subseting...")

        return df_generate

    def create_train_groups(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # solve it inside the simon dataloader
        df_sampled = self.df_generate.query('chr != "chr1" ')
        df_train = df_sampled.query('chr != "chr2" ')
        df_test = self.df_generate.query('chr == "chr1" ')
        df_train_shuffled = df_sampled.query('chr == "chr2" ')

        df_train_shuffled["sequence"] = df_train_shuffled["sequence"].apply(
            lambda x: "".join(random.sample(list(x), len(x)))
        )
        return df_train, df_test, df_train_shuffled

    def get_motif(self) -> None:
        self.train = self.generate_motifs_and_fastas(self.df_train_in, "train")
        self.test = self.generate_motifs_and_fastas(self.df_test_in, "test")
        self.train_shuffle = self.generate_motifs_and_fastas(
            self.df_train_shuffled_in, "train_shuffle"
        )

    def generate_motifs_and_fastas(self, df: pd.DataFrame, name: str) -> Dict[str, Any]:
        """return fasta anem , and dict with components motifs"""
        print("Generating Fasta and Motis:", name)
        print("---" * 10)
        fasta_saved = self.save_fasta(
            df, f"{name}_{'_'.join([str(c) for c in self.subset_components])}"
        )
        print("Generating Motifs (all seqs)")
        motif_all_components = motifs_from_fasta(fasta_saved)
        print("Generating Motifs per component")
        train_comp_motifs_dict = self.generate_motifs_components(df)

        return {
            "fasta_name": fasta_saved,
            "motifs": motif_all_components,
            "motifs_per_components_dict": train_comp_motifs_dict,
            "dataset": df,
        }

    def save_fasta(
        self, df: pd.DataFrame, name_fasta: str, to_seq_groups_comparison: bool = False
    ) -> str:
        fasta_final_name = name_fasta + ".fasta"
        save_fasta_file = open(fasta_final_name, "w")
        number_to_sample = df.shape[0]

        if to_seq_groups_comparison and self.number_of_sequences_to_motif_creation:
            number_to_sample = self.number_of_sequences_to_motif_creation

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

    def generate_motifs_components(self, df: pd.DataFrame) -> dict:
        final_comp_values = {}
        for comp, v_comp in df.groupby("TAG"):
            print(comp)
            print("number of sequences used to generate the motifs")
            name_c_fasta = self.save_fasta(
                v_comp, "temp_component", to_seq_groups_comparison=True
            )
            final_comp_values[comp] = motifs_from_fasta(name_c_fasta)
        return final_comp_values


class SequenceDataset(Dataset):
    def __init__(
        self,
        seqs: str,
        c: str,
        transform: Optional[T.Compose] = T.Compose([T.ToTensor()]),
    ):
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

        if self.transform:
            x = self.transform(image)
        else:
            x = image

        y = self.c[index]

        return x, y
