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
from refactor.utils.misc import one_hot_encode

DEFAULT_BASE_PATH = Path('.')
DEFAULT_DATA_DIR_PATH = DEFAULT_BASE_PATH / Path("data")
DEFAULT_DATA_ENCODE_FILENAME = "encode_data.pkl"
DEFAULT_DATA_ENCODE_PATH = DEFAULT_DATA_DIR_PATH / DEFAULT_DATA_ENCODE_FILENAME
DEFAULT_SEQUENCES_PER_GROUP_FILENAME = "K562_hESCT0_HepG2_GM12878_12k_sequences_per_group.txt"
DEFAULT_SEQUENCES_PER_GROUP_PATH = DEFAULT_DATA_DIR_PATH / DEFAULT_SEQUENCES_PER_GROUP_FILENAME

DEFAULT_SUBSET_COMPONENTS = [
    "GM12878_ENCLB441ZZZ",
    "hESCT0_ENCLB449ZZZ",
    "K562_ENCLB843GMH",
    "HepG2_ENCLB029COU",
]


class SequenceDataset(Dataset):
    def __init__(
        self,
        seqs: str,
        c: str,
        sequence_transform: Optional[T.Compose] = T.Compose([T.ToTensor()]),
        cell_type_transform: Optional[T.Compose] = T.Compose([T.ToTensor()])
    ):
        "Initialization"
        self.seqs = seqs
        self.c = c
        self.sequence_transform = sequence_transform
        self.cell_type_transform = cell_type_transform

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.seqs)

    def __getitem__(self, index):
        "Generates one sample of data"
        x = self.seqs[index]
        if self.sequence_transform:
            x = self.transform(x)

        y = self.c[index]
        if self.cell_type_transform: 
            y = self.cell_type_transform(y)

        return x, y


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

    encode_data = None
    train_dataset: Dataset = None
    val_dataset: Dataset = None
    test_dataset: Dataset = None

    datasets_per_stage: Dict[str, Dataset] = dict()
    motifs_per_stage: Dict[str, Any] = dict()
    motifs_per_components_dict_per_stage: Dict[str, Dict[str, Any]] = dict()

    def __init__(
        self,
        data_dir: str,
        encoded_filename: str=DEFAULT_DATA_ENCODE_FILENAME,
        sequences_per_group_filename: str=DEFAULT_SEQUENCES_PER_GROUP_FILENAME,
        sequence_length: int = 200,
        sequence_encoding: str = "polar",
        sequence_transform=None,
        cell_type_transform=None,
        batch_size=None,
        num_workers: int = 0,
        load_saved_data: bool = True, 
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
        self.data_dir = data_dir  # 'data'
        self.data_path = Path(self.data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.load_saved_data = load_saved_data
        self.subset_components = subset_components
        
        self.encoded_filename = encoded_filename
        self.sequences_per_group_filename = sequences_per_group_filename

        self.train_chr = train_chr
        
        if val_chr: 
            self.val_chr = val_chr

        if test_chr: 
            self.test_chr = test_chr

    def prepare_data(self) -> None:
        if self.load_saved_data: 
            return 
        
        print("Preparing data...")
        data_path = self.data_path / self.sequences_per_group_filename
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
            "validation": self.df_validation,
            "test": self.df_test,
        }

        for stage, data in combined_dict.items(): 
            stage_data_path = self.data_path / f"{stage}_{self.encoded_filename}" # src/refactor/data/encode_data.pkl
            with open(stage_data_path, "wb") as f: 
                pickle.dump(data, f)

        print("Preparing data DONE!")

    def setup(self, stage: str):
        # TODO: incorporate some extra information after the split (experiement -> split -> motif -> train/test assignment)
        # WARNING: have to be able to call loading_data on the main process of accelerate/fabric bc of gimme_motifs caching dependecies
        # Creating sequence datasets unless they exist already

        print(f"Loading {stage}...")
        stage_data_path = self.data_path / f"{stage}_{self.encoded_filename}"
        with open(stage_data_path, "rb") as f: 
            encode_data = pickle.load(f)

        self.motifs_per_stage[stage] = encode_data['motifs']
        self.motifs_per_components_dict_per_stage[stage] = encode_data["motifs_per_components_dict"]
        self.datasets_per_stage[stage] =  self.create_sequence_dataset(encode_data)
        print(f"Loading {stage} DONE!")


    def create_sequence_dataset(self, data):
        df = data["dataset"]
        self.cell_components = df.sort_values("TAG")["TAG"].unique().tolist()
        self.tag_to_numeric = {x: n + 1 for n, x in enumerate(df.TAG.unique())}
        self.numeric_to_tag = {n + 1: x for n, x in enumerate(df.TAG.unique())}

        self.cell_types = sorted(self.numeric_to_tag.keys())
        X_cell_types = torch.from_numpy(df["TAG"].apply(lambda x: self.tag_to_numeric[x]).to_numpy())
        
        nucleotides = ["A", "C", "G", "T"]
        X_sequences = np.array([one_hot_encode(x, nucleotides, 200) for x in (df["sequence"]) if "N" not in x])
        X_sequences = np.array([x.T.tolist() for x in X_sequences])
        X_sequences[X_sequences == 0] = -1
        
        return SequenceDataset(
            X_sequences, 
            X_cell_types,
            sequence_transform=self.sequence_transform,
            cell_type_transform=self.cell_type_transform,
        )

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

    def train_dataloader(self):
        train_dataset = self.datasets_per_stage['train']
        return DataLoader(
            dataset=train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            # pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        val_dataset = self.datasets_per_stage['val']
        return DataLoader(
            dataset=val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            # pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        test_dataset = self.datasets_per_stage['test']
        return DataLoader(
            dataset=test_dataset,
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


