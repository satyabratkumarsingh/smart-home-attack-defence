import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from utils.file_utils import find_all_file_names

class CSVDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.features = []
        self.labels = []
        for file in find_all_file_names():
            file_df = pd.read_csv(file)
            file_features = file_df.iloc[:, :-1].values
            file_label = np.where(file_df.iloc[:, -1] == 'BenignTraffic', 0, 1)
            self.features.extend(file_features)
            self.labels.extend(file_label)

    def __len__(self):
        return len(self.features)
    
    def collate_fn(batch):
    # Pad sequences to the same length within each batch
        padded_batch = pad_sequence(batch, batch_first=True, padding_value=0)
        return padded_batch
    
    def __getitem__(self, idx):
        features = self.features[idx]
        labels = self.labels[idx]
        if self.transform:
            features = self.transform(features)
            labels = self.transform(labels)
        return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)
