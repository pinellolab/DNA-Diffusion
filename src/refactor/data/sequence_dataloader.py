import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset


class SequenceDatasetBase(Dataset):
    def __init__(
        self,
        data_path,
        sequence_length: int = 200,
        sequence_encoding: str = "polar",
        sequence_transform=None,
        cell_type_transform=None,
    ) -> None:
        super().__init__()
        self.data = pd.read_csv(data_path, sep="\t")
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
        current_seq = self.data["raw_sequence"][index]
        if "N" not in current_seq:
            X_seq = self.encode_sequence(current_seq, encoding=self.sequence_encoding)

            # Reading cell component at current index
            X_cell_type = self.data["component"][index]

            if self.sequence_transform is not None:
                X_seq = self.sequence_transform(X_seq)
            if self.cell_type_transform is not None:
                X_cell_type = self.cell_type_transform(X_cell_type)

            return X_seq, X_cell_type

    def check_data_validity(self) -> None:
        """
        Checks if the data is valid.
        """
        if not set("".join(self.data["raw_sequence"])).issubset(set(self.alphabet)):
            raise ValueError(f"Sequence contains invalid characters.")

        uniq_raw_seq_len = self.data["raw_sequence"].str.len().unique()
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


class SequenceDatasetTrain(SequenceDatasetBase):
    def __init__(self, data_path="", **kwargs) -> None:
        super().__init__(data_path=data_path, **kwargs)


class SequenceDatasetValidation(SequenceDatasetBase):
    def __init__(self, data_path="", **kwargs) -> None:
        super().__init__(data_path=data_path, **kwargs)


class SequenceDatasetTest(SequenceDatasetBase):
    def __init__(self, data_path="", **kwargs) -> None:
        super().__init__(data_path=data_path, **kwargs)


class SequenceDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_path=None,
        val_path=None,
        test_path=None,
        sequence_length: int = 200,
        sequence_encoding: str = "polar",
        sequence_transform=None,
        cell_type_transform=None,
        batch_size=None,
        num_workers: int = 1,
    ) -> None:
        super().__init__()
        self.datasets = {}
        self.train_dataloader, self.val_dataloader, self.test_dataloader = (
            None,
            None,
            None,
        )

        if train_path:
            self.datasets["train"] = train_path
            self.train_dataloader = self._train_dataloader

        if val_path:
            self.datasets["validation"] = val_path
            self.val_dataloader = self._val_dataloader

        if test_path:
            self.datasets["test"] = test_path
            self.test_dataloader = self._test_dataloader

        self.sequence_length = sequence_length
        self.sequence_encoding = sequence_encoding
        self.sequence_transform = sequence_transform
        self.cell_type_transform = cell_type_transform
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        return

    def setup(self):
        if "train" in self.datasets:
            self.train_data = SequenceDatasetTrain(
                data_path=self.datasets["train"],
                sequence_length=self.sequence_length,
                sequence_encoding=self.sequence_encoding,
                sequence_transform=self.sequence_transform,
                cell_type_transform=self.cell_type_transform,
            )
        if "validation" in self.datasets:
            self.val_data = SequenceDatasetValidation(
                data_path=self.datasets["validation"],
                sequence_length=self.sequence_length,
                sequence_encoding=self.sequence_encoding,
                sequence_transform=self.sequence_transform,
                cell_type_transform=self.cell_type_transform,
            )
        if "test" in self.datasets:
            self.test_data = SequenceDatasetTest(
                data_path=self.datasets["test"],
                sequence_length=self.sequence_length,
                sequence_encoding=self.sequence_encoding,
                sequence_transform=self.sequence_transform,
                cell_type_transform=self.cell_type_transform,
            )

    def _train_dataloader(self):
        return DataLoader(
            self.train_data,
            self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def _val_dataloader(self):
        return DataLoader(
            self.val_data,
            self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def _test_dataloader(self):
        return DataLoader(
            self.test_data,
            self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
