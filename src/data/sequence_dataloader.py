import pandas as pd
import numpy as np

import torch
import torchvision.transforms as T
import torch.nn.functional as F
import pytorch_lightning as pl 
from torch.utils.data import Dataset, DataLoader 

class SequenceDatasetBase(Dataset):
    def __init__(self, data_path, sequence_length=200, transform=None):
        super().__init__()
        self.data = pd.read_csv(data_path, sep="\t")
        self.sequence_length = sequence_length
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # Iterating through DNA sequences from dataset and one-hot encoding all nucleotides
        current_seq = self.data["raw_sequence"][index]
        if 'N' not in current_seq: 
            X_seq = np.array(self.one_hot_encode(current_seq, ['A','C','T','G'], self.sequence_length))
            X_seq = X_seq.T
            X_seq[X_seq == 0] = -1
            
            # Reading cell component at current index
            X_cell_type = self.data["component"][index]
            
            if self.transform:
                X_seq = self.transform(X_seq)
                X_cell_type = self.transform(X_cell_type)

            return X_seq, X_cell_type

    # Function for one hot encoding each line of the sequence dataset
    def one_hot_encode(self, seq, alphabet, sequence_length):
        """
        One-hot encoding a sequence 
        """
        seq_len = len(seq)
        seq_array = np.zeros((sequence_length, len(alphabet)))
        for i in range(seq_len):
            seq_array[i, alphabet.index(seq[i])] = 1 
        return seq_array


class SequenceDatasetTrain(SequenceDatasetBase):
    def __init__(self, data_path="", **kwargs):
        super().__init__(data_path=data_path, **kwargs)

class SequenceDatasetValidation(SequenceDatasetBase):
    def __init__(self, data_path="", **kwargs):
        super().__init__(data_path=data_path, **kwargs)

class SequenceDatasetTest(SequenceDatasetBase):
    def __init__(self, data_path="", **kwargs):
        super().__init__(data_path=data_path, **kwargs)


class SequenceDataModule(pl.LightningDataModule):
    def __init__(self, train_path=None, val_path=None, test_path=None, transform=None, batch_size=None, num_workers=None):
        super().__init__()
        self.datasets = dict()
        self.train_dataloader, self.val_dataloader, self.test_dataloader = None, None, None

        if train_path:
            self.datasets["train"] = train_path
            self.train_dataloader = self._train_dataloader

        if val_path:
            self.datasets["validation"] = val_path
            self.val_dataloader = self._val_dataloader

        if test_path:
            self.datasets["test"] = test_path
            self.test_dataloader = self._test_dataloader

        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self):
       self.train_data = SequenceDatasetTrain(self.datasets["train"], transform = self.transform)
       self.val_data = SequenceDatasetValidation(self.datasets["validation"], transform = self.transform)
       self.test_data = SequenceDatasetTest(self.datasets["test"], transform = self.transform)

    def _train_dataloader(self):
        return DataLoader(self.train_data,
                          self.batch_size, 
                          shuffle=True, 
                          num_workers=self.num_workers, 
                          pin_memory=True)

    def _val_dataloader(self):
        return DataLoader(self.val_data,
                          self.batch_size, 
                          shuffle=True,
                          num_workers=self.num_workers,
                          pin_memory=True)

    def _test_dataloader(self):
        return DataLoader(self.test_data,
                          self.batch_size, 
                          shuffle=True, 
                          num_workers=self.num_workers, 
                          pin_memory=True)

