import pickle
import random
from typing import Any, Dict, List, Optional, Tuple
import os
from pathlib import Path

import pandas as pd
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import rank_zero_only
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

import torchvision.transforms as T

from refactor.utils.data import get_motif, read_master_dataset, subset_by_experiment

DEFAULT_BASE_PATH = Path('./src/refactor')
DEFAULT_DATA_DIR_PATH = DEFAULT_BASE_PATH / Path("data")
DEFAULT_DATA_ENCODE_FILENAME = Path("encode_data.pkl")
DEFAULT_DATA_ENCODE_PATH = DEFAULT_DATA_DIR_PATH / DEFAULT_DATA_ENCODE_FILENAME
DEFAULT_SEQUENCES_PER_GROUP_FILENAME = Path("K562_hESCT0_HepG2_GM12878_12k_sequences_per_group.txt")
DEFAULT_SEQUENCES_PER_GROUP_PATH = DEFAULT_DATA_DIR_PATH / DEFAULT_SEQUENCES_PER_GROUP_FILENAME

DEFAULT_SUBSET_COMPONENTS=[
    "GM12878_ENCLB441ZZZ",
    "hESCT0_ENCLB449ZZZ",
    "K562_ENCLB843GMH",
    "HepG2_ENCLB029COU",
],


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


class SequenceDatasetBase(Dataset):
    """
    Base class for sequence datasets.

    Args:
        df (pd.DataFrame): Dataframe containing the all sequence metadata
        sequence_length (int): Length of the sequence
        sequence_encoding (str): Encoding scheme for the sequence ("polar", "onehot", "ordinal")
        sequence_transform (callable): Transformation for the sequence
        cell_type_transform (callable): Transformation for the cell type
    """

    def __init__(
        self,
        df,
        sequence_length: int = 200,
        sequence_encoding: str = "polar",
        sequence_transform=None,
        cell_type_transform=None,
    ) -> None:
        super().__init__()
        self.data = df
        self.sequence_length = sequence_length
        self.sequence_encoding = sequence_encoding
        self.sequence_transform = sequence_transform
        self.cell_type_transform = cell_type_transform
        self.alphabet = ["A", "C", "T", "G"]
        self.check_data_validity()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        # Iterating through DNA sequences from dataset and one-hot encoding all nucleotides
        current_seq = self.data["sequence"][index]
        if "N" not in current_seq:
            X_seq = self.encode_sequence(current_seq, encoding=self.sequence_encoding)

            # Reading cell component at current index
            X_cell_type = self.data["TAG"][index]

            if self.sequence_transform is not None:
                X_seq = self.sequence_transform(X_seq)
            if self.cell_type_transform is not None:
                X_cell_type = self.cell_type_transform(X_cell_type)

            return X_seq, X_cell_type

    def check_data_validity(self) -> None:
        """
        Checks if the data is valid.
        """
        if not set("".join(self.data["sequence"])).issubset(set(self.alphabet)):
            raise ValueError(f"Sequence contains invalid characters.")

        uniq_raw_seq_len = self.data["sequence"].str.len().unique()
        if len(uniq_raw_seq_len) != 1 or uniq_raw_seq_len[0] != self.sequence_length:
            raise ValueError(f"The sequence length does not match the data.")

    def encode_sequence(self, seq, encoding):
        """
        Encodes a sequence using the given encoding scheme ("polar", "onehot", "ordinal").
        """
        if encoding == "polar":
            seq = self.one_hot_encode(seq).T
            seq[seq == 0] = -1
        elif encoding == "onehot":
            seq = self.one_hot_encode(seq).T
        elif encoding == "ordinal":
            seq = np.array([self.alphabet.index(n) for n in seq])
        else:
            raise ValueError(f"Unknown encoding scheme: {encoding}")
        return seq

    # Function for one hot encoding each line of the sequence dataset
    def one_hot_encode(self, seq) -> np.ndarray:
        """
        One-hot encoding a sequence
        """
        seq_len = len(seq)
        seq_array = np.zeros((self.sequence_length, len(self.alphabet)))
        for i in range(seq_len):
            seq_array[i, self.alphabet.index(seq[i])] = 1
        return seq_array


class SequenceDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning data module for sequence datasets.

    Args:
        data_path (str): Path to the data
        sequence_length (int): Length of the sequence
        sequence_encoding (str): Encoding scheme for the sequence ("polar", "onehot", "ordinal")
        sequence_transform (callable): Transformation for the sequence
        cell_type_transform (callable): Transformation for the cell type
        batch_size (int): Batch size
        num_workers (int): Number of workers
    """

    df_train = None
    df_validation = None
    df_test = None

    train_chr: List[str] = None
    val_chr: List[str] = ['chr1']
    test_chr: List[str] = ['chr2']

    def __init__(
        self,
        data_dir: str,
        sequence_length: int = 200,
        sequence_encoding: str = "polar",
        sequence_transform=None,
        cell_type_transform=None,
        batch_size=None,
        num_workers: int = 0,
        load_saved_data: bool = False, 
        train_chr: List[str] = None, 
        val_chr: List[str] = None,
        test_chr: List[str] = None,
        subset_components: List[str] = DEFAULT_SUBSET_COMPONENTS,
        number_of_sequences_to_motif_creation: int = 1000,
    ) -> None:
        
        super().__init__()
        self.save_hyperparameters(logger=False)
        # self.df_train, self.df_validation, self.df_test = Optional[Dataset] = None
        self.number_of_sequences_to_motif_creation = number_of_sequences_to_motif_creation
        self.sequence_length = sequence_length
        self.sequence_encoding = sequence_encoding
        self.sequence_transform = sequence_transform
        self.cell_type_transform = cell_type_transform
        self.data_dir = data_dir
        self.data_path = DEFAULT_BASE_PATH / Path(self.data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.load_saved_data = load_saved_data
        self.subset_components = subset_components

        self.train_chr = train_chr
        
        if val_chr: 
            self.val_chr = val_chr

        if test_chr: 
            self.test_chr = test_chr


    def prepare_data__(self) -> None:
        print("Preparing data...")
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
            with open(DEFAULT_DATA_ENCODE_PATH, "wb") as f:
                pickle.dump(combined_dict, f)
        print("Preparing data OK!")

    def prepare_data(self) -> None:
        if self.load_saved_data: 
            return 
        
        print("Preparing data...")
        data_path = self.data_path / DEFAULT_SEQUENCES_PER_GROUP_FILENAME
        df = read_master_dataset(data_path)
        if len(self.subset_components) < 4:
            df = subset_by_experiment(df, subset_components=self.subset_components)
        
        if not self.df_train and not self.df_validation and not self.df_test:
            self.df_train, self.df_validation, self.df_test = self.create_train_groups(df)
        
        self.df_train, self.df_validation, self.df_test = get_motif(
            self.df_train, 
            self.df_validation, 
            self.df_test,
            self.subset_components,
            self.number_of_sequences_to_motif_creation
        )

        combined_dict = {
            "train": self.df_train,
            "test": self.df_test,
            "train_shuffle": self.df_validation,
        }
        with open("dnadiffusion/data/encode_data.pkl", "wb") as f:
            pickle.dump(combined_dict, f)

        print("Preparing data DONE!")

    def setup(self, stage: str):
        print("Setup...")
        data_path = self.data_path / DEFAULT_SEQUENCES_PER_GROUP_FILENAME
        # TODO: incorporate some extra information after the split (experiement -> split -> motif -> train/test assignment)
        # WARNING: have to be able to call loading_data on the main process of accelerate/fabric bc of gimme_motifs caching dependecies
        # Creating sequence datasets unless they exist already
        # df = pd.read_csv(data_path, sep="\t")
        df = read_master_dataset(data_path)
        if len(self.subset_components < 4):
            df = subset_by_experiment(df, subset_components=self.subset_components)
        
        if not self.df_train and not self.df_validation and not self.df_test:
            self.df_train, self.df_validation, self.df_test = self.create_train_groups(df)
        
        self.df_train, self.df_validation, self.df_test = get_motif(
            self.df_train, 
            self.df_validation, 
            self.df_test,
            self.subset_components,
            self.number_of_sequences_to_motif_creation
        )

        combined_dict = {
            "train": self.df_train,
            "test": self.df_test,
            "train_shuffle": self.df_validation,
        }
        with open("dnadiffusion/data/encode_data.pkl", "wb") as f:
            pickle.dump(combined_dict, f)

        ##### DELETE ABOVE #####

        with open("dnadiffusion/data/encode_data.pkl", "rb") as f:
            encode_data = pickle.load(f)
        self.df_train = encode_data["train"]
        self.df_test = encode_data["test"]
        self.df_validation = encode_data["shuffle"]

        self.train_data = self.create_sequence_dataset(self.df_train)
        self.validation_data = self.create_sequence_dataset(self.df_validation)
        self.test_data = self.create_sequence_dataset(self.df_test)

        # Creating sequence dataloaders
        self.train_dl = self.create_dataloader(self.train_data, self.batch_size, self.num_workers)
        self.validation_dl = self.create_dataloader(self.validation_data, self.batch_size, self.num_workers)
        self.test_dl = self.create_dataloader(self.test_data, self.batch_size, self.num_workers)
        print("Setup OK!")

    def create_train_groups(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if self.train_chr is None: 
            val_test_chr = self.val_chr + self.test_chr
            df_train = df[~df.chr.isin(val_test_chr)].reset_index(drop=True)
        else: 
            df_train = df[df.chr.isin(self.train_chr)].reset_index(drop=True)

        df_validation = df[df.chr.isin(self.val_chr)].reset_index(drop=True)
        df_test = df[df.chr.isin(self.test_chr)].reset_index(drop=True)

        df_validation["sequence"] = df_validation["sequence"].apply(
            lambda x: "".join(random.sample(list(x), len(x)))
        )
        return df_train, df_validation, df_test

    def create_sequence_dataset(self, df):
        return SequenceDatasetBase(
            df=df,
            sequence_length=self.sequence_length,
            sequence_encoding=self.sequence_encoding,
            sequence_transform=self.sequence_transform,
            cell_type_transform=self.cell_type_transform,
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            # pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            # pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            # pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    _ = SequenceDataModule()


