import pandas as pd
import numpy as np

import torch
import torchvision.transforms as T
import torch.nn.functional as F
import pytorch_lightning as pl 
from torch.utils.data import Dataset, DataLoader 

class SequenceDataset(Dataset):
    def __init__(self, data_path, transform):
        super().__init__()
        self.data = pd.read_csv(data_path, sep="\t")
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # Iterating through DNA sequences from dataset and one-hot encoding all nucleotides
        current_seq = self.data["raw_sequence"][index]
        if 'N' not in current_seq: 
            X_seq = np.array(self.one_hot_encode(current_seq, ['A','C','T','G'], 200))
            X_seq = X_seq.T
            X_seq[X_seq == 0] = -1
            
            # Reading cell component at current index
            X_cell_type = self.data["component"][index]
            
            if self.transform:
                X_seq = self.transform(X_seq)
                X_cell_type = self.transform(X_cell_type)

            return X_seq, X_cell_type

    # Function for one hot encoding each line of the sequence dataset
    def one_hot_encode(self, seq, alphabet, max_seq_len):
        """
        One-hot encoding a sequence 
        """
        seq_len = len(seq)
        seq_array = np.zeros((max_seq_len, len(alphabet)))
        for i in range(seq_len):
            seq_array[i, alphabet.index(seq[i])] = 1 
        return seq_array
     

class SequenceDataModule(pl.LightningDataModule):
    def __init__(self, train_path, val_path, test_path, transform, batch_size=32, num_workers=3):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self):
       #transform = T.Compose([T.ToTensor()])
       self.train_data = SequenceDataset(self.train_path, transform = self.transform)
       self.val_data = SequenceDataset(self.val_path, transform = self.transform)
       self.test_data = SequenceDataset(self.test_path, transform = self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_data,
                          self.batch_size, 
                          shuffle=True, 
                          num_workers=self.num_workers, 
                          pin_memory=True)

    def val_dataloder(self):
        return DataLoader(self.val_data,
                          self.batch_size, 
                          shuffle=True,
                          num_workers=self.num_workers,
                          pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_data,
                          self.batch_size, 
                          shuffle=True, 
                          num_workers=self.num_workers, 
                          pin_memory=True)

