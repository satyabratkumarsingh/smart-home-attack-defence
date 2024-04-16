import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class CICIDataset(Dataset):
    def __init__(self, csv_file, attack_features, attack_labels, benign_features, benign_labels):
       
        self.attack_features = attack_features
        self.attack_labels = attack_labels

        self.benign_features = benign_features
        self.benign_labels = benign_labels

        self.length_dataset = min(len(self.benign_labels), len(self.attack_labels))  


    def __len__(self):
        return self.length_dataset

    @staticmethod
    def collate_fn(batch):
        attack_features, benign_features = zip(*batch)
        return torch.stack(attack_features), torch.stack(benign_features)

    def __getitem__(self, idx):
        benign_idx = idx % len(self.benign_features)
        benign_feature = torch.tensor(self.benign_features[benign_idx], dtype=torch.float32)
        benign_label = self.benign_labels.iloc[benign_idx]

        attack_idx = idx % len(self.attack_features)
        attack_feature = torch.tensor(self.attack_features[attack_idx], dtype=torch.float32)
        attack_label = self.attack_labels.iloc[attack_idx]

        return benign_feature, benign_label, attack_feature, attack_label

