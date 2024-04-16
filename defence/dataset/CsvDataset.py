import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class CsvDataset(Dataset):
    def __init__(self, csv_file, transform = None):
        self.data = pd.read_csv(csv_file)
        self.labels = pd.get_dummies(self.data['label']).as_matrix()
        self.transform = transform

    def __len__(self):
        len(self.labels)

    @staticmethod
    def collate_fn(batch):
        attack_features, benign_features = zip(*batch)
        return torch.stack(attack_features), torch.stack(benign_features)

    def __getitem__(self, idx):
        benign_idx = idx % len(self.labels)
        benign_feature = torch.tensor(self.benign_features[benign_idx], dtype=torch.float32)
        benign_label = self.benign_labels.iloc[benign_idx]

        attack_idx = idx % len(self.attack_features)
        attack_feature = torch.tensor(self.attack_features[attack_idx], dtype=torch.float32)
        attack_label = self.attack_labels.iloc[attack_idx]

        return benign_feature, benign_label, attack_feature, attack_label

