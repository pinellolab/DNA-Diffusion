import os
import pickle
import random
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from pytorch_lightning.utilities import rank_zero_only
from torch.utils.data import DataLoader, Dataset

from dnadiffusion.utils.utils import one_hot_encode


def motifs_from_fasta(fasta: str):
    print("Computing Motifs....")
    os.system(f"gimme scan {fasta} -p  JASPAR2020_vertebrates -g hg38 > train_results_motifs.bed")
    df_results_seq_guime = pd.read_csv("train_results_motifs.bed", sep="\t", skiprows=5, header=None)
    df_results_seq_guime["motifs"] = df_results_seq_guime[8].apply(lambda x: x.split('motif_name "')[1].split('"')[0])

    df_results_seq_guime[0] = df_results_seq_guime[0].apply(lambda x: "_".join(x.split("_")[:-1]))
    df_results_seq_guime_count_out = df_results_seq_guime[[0, "motifs"]].drop_duplicates().groupby("motifs").count()
    plt.rcParams["figure.figsize"] = (30, 2)
    df_results_seq_guime_count_out.sort_values(0, ascending=False).head(50)[0].plot.bar()
    plt.title("Top 50 MOTIFS on component 0 ")
    plt.show()
    return df_results_seq_guime_count_out


@rank_zero_only
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
        self.number_of_sequences_to_motif_creation = number_of_sequences_to_motif_creation

    def __call__(self) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        df = self.read_csv(self.csv)
        subset_df = self.experiment(df)
        df_train, df_test, df_train_shuffled = self.create_train_groups(subset_df)
        train, test, train_shuffle = self.get_motif(df_train, df_test, df_train_shuffled)
        return train, test, train_shuffle

    def read_csv(self, input_csv: str) -> pd.DataFrame:
        df = pd.read_csv(input_csv, sep="\t")
        if self.change_comp_index:
            df["component"] = df["component"] + 1

        if self.limit_total_sequences:
            print(f"Limiting total sequences {self.limit_total_sequences}")
            df = df.sample(self.limit_total_sequences)

        return df

    def experiment(self, df: pd.DataFrame) -> pd.DataFrame:
        df_generate = df
        if self.subset_components is not None and type(self.subset_components) == list:
            print(" or ".join([f"TAG == {c}" for c in self.subset_components]))
            df_generate = df_generate.query(" or ".join([f'TAG == "{c}" ' for c in self.subset_components])).copy()
            print("Subseting...")

        return df_generate

    def create_train_groups(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df_test = df[df["chr"] == "chr1"].reset_index(drop=True)
        df_train_shuffled = df[df["chr"] == "chr2"].reset_index(drop=True)
        df_train = df_train = df[(df["chr"] != "chr1") & (df["chr"] != "chr2")].reset_index(drop=True)

        df_train_shuffled["sequence"] = df_train_shuffled["sequence"].apply(
            lambda x: "".join(random.sample(list(x), len(x)))
        )
        return df_train, df_test, df_train_shuffled

    def get_motif(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        df_train_shuffled: pd.DataFrame,
    ) -> None:
        train = self.generate_motifs_and_fastas(df_train, "train")
        test = self.generate_motifs_and_fastas(df_test, "test")
        train_shuffle = self.generate_motifs_and_fastas(df_train_shuffled, "train_shuffle")
        return train, test, train_shuffle

    def generate_motifs_and_fastas(self, df: pd.DataFrame, name: str) -> Dict[str, Any]:
        """return fasta anem , and dict with components motifs"""
        print("Generating Fasta and Motis:", name)
        print("---" * 10)
        fasta_saved = self.save_fasta(df, f"{name}_{'_'.join([str(c) for c in self.subset_components])}")
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

    def save_fasta(self, df: pd.DataFrame, name_fasta: str, to_seq_groups_comparison: bool = False) -> str:
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
            name_c_fasta = self.save_fasta(v_comp, "temp_component", to_seq_groups_comparison=True)
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


class LoadingDataModule(pl.LightningDataModule):
    def __init__(
        self,
        input_csv: str,
        subset_components: list,
        load_saved_data: bool = False,
        sample_number: int = 0,
        change_component_index: bool = True,
        limit_total_sequences: Optional[int] = None,
        number_of_sequences_to_motif_creation: Optional[int] = None,
        transform: Optional[T.Compose] = None,
        batch_size: int = 30,
        shuffle: bool = True,
        num_workers: int = 48,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()
        self.input_csv = input_csv
        self.subset_components = subset_components
        self.load_saved_data = load_saved_data
        self.sample_number = sample_number
        self.change_component_index = change_component_index
        self.limit_total_sequences = limit_total_sequences
        self.number_of_sequences_to_motif_creation = number_of_sequences_to_motif_creation
        self.transform = transform
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def prepare_data(self) -> None:
        if not self.load_saved_data:
            print("Loading data")
            encode_data = LoadingData(
                self.input_csv,
                self.subset_components,
                self.sample_number,
                self.change_component_index,
                self.limit_total_sequences,
                self.number_of_sequences_to_motif_creation,
            )
            train, test, train_shuffle = encode_data()
            combined_dict = {
                "train": train,
                "test": test,
                "train_shuffle": train_shuffle,
            }
            with open("dnadiffusion/data/encode_data.pkl", "wb") as f:
                pickle.dump(combined_dict, f)

    def setup(self, stage: Optional[str] = None) -> None:
        with open("dnadiffusion/data/encode_data.pkl", "rb") as f:
            encode_data = pickle.load(f)
        train = encode_data["train"]
        test = encode_data["test"]
        train_shuffle = encode_data["train_shuffle"]

        # Getting motif related data from encode_data
        self.train_motifs = train["motifs"]
        self.test_motifs = test["motifs"]
        self.shuffle_motifs = train_shuffle["motifs"]

        self.train_motifs_per_components_dict = train["motifs_per_components_dict"]
        self.test_motifs_per_components_dict = test["motifs_per_components_dict"]
        self.shuffle_motifs_per_components_dict = train_shuffle["motifs_per_components_dict"]

        # Sequence related data
        df = train["dataset"]
        self.cell_components = df.sort_values("TAG")["TAG"].unique().tolist()
        self.tag_to_numeric = {x: n + 1 for n, x in enumerate(df.TAG.unique())}
        self.numeric_to_tag = {n + 1: x for n, x in enumerate(df.TAG.unique())}
        self.cell_types = sorted(self.numeric_to_tag.keys())
        self.x_train_cell_type = torch.from_numpy(df["TAG"].apply(lambda x: self.tag_to_numeric[x]).to_numpy())
        nucleotides = ["A", "C", "G", "T"]
        X_train = np.array([one_hot_encode(x, nucleotides, 200) for x in (df["sequence"]) if "N" not in x])
        X_train = np.array([x.T.tolist() for x in X_train])
        X_train[X_train == 0] = -1
        self.X_train = X_train

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            SequenceDataset(
                self.X_train,
                self.x_train_cell_type,
                self.transform,
            ),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
