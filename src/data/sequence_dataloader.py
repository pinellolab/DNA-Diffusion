import os

import pandas as pd
import numpy as np

import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader


class Dataloading():
    """
    Class for loading and preprocessing data. 
    Initial version only contains train/val/test split, but will be updated with more functionality.

    Args:
        data_path (str): Path to the data.
    """
    def __init__(self,
                 data_path: str) -> None:
        super().__init__()
        self.df = pd.read_csv(data_path, sep="\t") 

    def create_splits(self):
        """
        Creates train/val/test splits.

        Args:
            df (pd.DataFrame): Dataframe to be split

        Returns:
            df_train (pd.DataFrame): Training dataframe
            df_validation (pd.DataFrame): Validation dataframe
            df_test (pd.DataFrame): Test dataframe
        """
        df_test = self.df[self.df.chr == 'chr1'].reset_index(drop=True)
        df_validation = self.df[self.df.chr == 'chr2'].reset_index(drop=True)
        df_train = self.df[~self.df.chr.isin(['chr1', 'chr2'])].reset_index(drop=True)

        return df_train, df_validation, df_test


class DataPreprocessing():
    """
    A data preprocessing class to extract motif data train/validation/test sets

    1. Generate motifs and fasta files
    2. Generate motifs per cell type
    3. Create dictionary of motifs per cell type


    """
    def __init__(self, 
                 df_train, 
                 df_val, 
                 df_test) -> None:
        super().__init__()
        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test
        self.train = None
        self.val = None
        self.test = None
        self.get_motif()
    
    def get_motif(self):
        '''
        Fetch motifs and generate fasta for train, val, and test sets
        '''
        self.train = self.generate_motifs_and_fasta(self.df_train, 'train')
        self.val = self.generate_motifs_and_fasta(self.df_val, 'val')
        self.test = self.generate_motifs_and_fasta(self.df_test, 'val')

    def save_fasta(self, df, name_fasta: str):
        '''
        Saves fasta file for a given dataframe
        '''
        fasta_filename = name_fasta + '.fasta'
        save_fasta = open(fasta_filename, 'w')
        write_fasta = '\n'.join([f'>{row.dhs_id}_{row.TAG}\n{row.sequence}' for _, row in df.iterrows()])
        save_fasta.write(write_fasta)
        save_fasta.close()
        return fasta_filename

    def motifs_from_fasta(self, fasta):
        print('Computing motifs from fasta')
        os.system(f'gimme scan {fasta} -p JASPAR2020_vertebrates -g hg38 > train_results_motifs.bed')
        df_seq_motifs = pd.read_csv('train_results_motifs.bed', sep='\t',skiprows=5, header=None)
        df_seq_motifs['motifs'] = df_seq_motifs[8].apply(lambda x: x.split('motif_name "')[1].split('"')[0])
        df_seq_motifs[0] = df_seq_motifs[0].apply(lambda x : '_'.join(  x.split('_')[-2:]))
        df_seq_motifs_out = df_seq_motifs[['motifs', 0]].drop_duplicates().groupby('motifs').count()
        return df_seq_motifs_out

    def generate_motifs_cells(self, df):
        """
        Generating a dictionary with motif components.
        """
        final_comp_values = {}
        for cell, df_subset in df.groupby('TAG'):
            print(cell)
            name_c_fasta = self.save_fasta(df_subset, 'temp_cell')
            final_comp_values[cell] = self.motifs_from_fasta(name_c_fasta)
        return final_comp_values

    def generate_motifs_and_fasta(self, df, name):
        """
        Generate a dictionary containing:
        1. Fasta saved.
        2. Motifs.
        3. Motifs per component.
        4. Dataset.

        Args:
            df (pd.DataFrame): Subsetted dataframe created from the master dataset
            name (str): Name for generated fasta file
        """
        print(f'Generating fasta and motifs: {name}')
        print('---' * 10)
        fasta_saved = self.save_fasta(df, name)
        print('Generating motifs')
        all_motifs = self.motifs_from_fasta(fasta_saved)
        print('Generating motifs per cell type')
        cell_motifs_dict = {cell: self.generate_motifs_cells(df) for cell, df in df.groupby('TAG')} 
        print('Creating dictionary')
        return {'fasta': fasta_saved, 'motifs': all_motifs, 'motifs_per_cell_type': cell_motifs_dict, 'dataset': df}

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
        cell_type_transform=None
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
        self.datasets = dict()
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
