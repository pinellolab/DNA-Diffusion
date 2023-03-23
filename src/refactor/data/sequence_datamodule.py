from typing import Any, Dict, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

# from torchvision.transforms import transforms


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

    def create_sequence_dataset(self, df):
        return SequenceDatasetBase(
            df=df,
            sequence_length=self.sequence_length,
            sequence_encoding=self.sequence_encoding,
            sequence_transform=self.sequence_transform,
            cell_type_transform=self.cell_type_transform,
        )


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

    def __init__(
        self,
        data_dir: str,
        sequence_length: int = 200,
        sequence_encoding: str = "polar",
        sequence_transform=None,
        cell_type_transform=None,
        batch_size=None,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        # self.df_train, self.df_validation, self.df_test = Optional[Dataset] = None
        self.sequence_length = sequence_length
        self.sequence_encoding = sequence_encoding
        self.sequence_transform = sequence_transform
        self.cell_type_transform = cell_type_transform
        self.data_dir = data_dir

    def setup(self, stage: str):
        # TODO: incorporate some extra information after the split (experiement -> split -> motif -> train/test assignment)
        # WARNING: have to be able to call loading_data on the main process of accelerate/fabric bc of gimme_motifs caching dependecies
        # Creating sequence datasets unless they exist already
        self.df = pd.read_csv(data_path, sep="\t")
        if not self.data_train and not self.data_test and not self.data_test:
            df_test = self.df[self.df.chr == "chr1"].reset_index(drop=True)
            df_validation = self.df[self.df.chr == "chr2"].reset_index(drop=True)
            df_train = self.df[~self.df.chr.isin(["chr1", "chr2"])].reset_index(drop=True)

        train_data = self.create_sequence_dataset(self.df_train)
        validation_data = self.create_sequence_dataset(self.df_validation)
        test_data = self.create_sequence_dataset(self.df_test)

        # Creating sequence dataloaders
        self.train_dl = self.create_dataloader(self.train_data, self.batch_size, self.num_workers)
        self.validation_dl = self.create_dataloader(self.validation_data, self.batch_size, self.num_workers)
        self.test_dl = self.create_dataloader(self.test_data, self.batch_size, self.num_workers)

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
