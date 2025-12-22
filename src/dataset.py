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

        # 1. Hazır Tensörü Yükle
        # weights_only=True ekleyerek o kırmızı uyarıları susturuyoruz.
        spec = torch.load(file_path, weights_only=True).float()

        # 2. Boyut Ayarı (TCN İÇİN DEĞİŞİKLİK)
        # Eski kodda: spec = spec.unsqueeze(0) yapıyorduk.
        # Yeni (TCN): Bunu SİLİYORUZ. TCN düz (128, 157) ister.
        # Eğer spec boyutu zaten (128, 157) ise hiçbir şey yapma.

        label = self.label_map[label_str]
        return spec, torch.tensor(label, dtype=torch.long)