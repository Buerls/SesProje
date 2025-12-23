import torch
from torch.utils.data import Dataset
import pandas as pd


class AudioDataset(Dataset):
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
        self.label_map = {'tr': 0, 'ko': 1, 'sv': 2}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        file_path = row['path']
        label_str = row['label']

        spec = torch.load(file_path, weights_only=True).float()

        label = self.label_map[label_str]
        return spec, torch.tensor(label, dtype=torch.long)
