import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    """
    TCN'in en önemli parçası: Geleceği görmemesi için
    sağ taraftaki (gelecek zaman) padding'i kırpar.
    """

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    TCN'in temel yapı taşı (Residual Block).
    İki adet Dilated Conv1D katmanından oluşur.
    """

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()

        # 1. Katman
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)  # Çıktıyı kırpıp boyutu eşitler
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # 2. Katman
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)

        # Residual bağlantı için (Giriş ve çıkış kanalları farklıysa 1x1 Conv ile eşitle)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class LanguageTCN(nn.Module):
    def __init__(self, num_inputs=128, num_channels=[64, 128, 256, 512], kernel_size=7, dropout=0.2, num_classes=3):
        """
        num_inputs: Mel-Spectrogram frekans sayısı (senin verinde 128)
        num_channels: Her katmandaki kanal sayısı (Filtre sayısı)
        """
        super(LanguageTCN, self).__init__()
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2 ** i  # 1, 2, 4, 8 diye artar
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            # Padding hesaplaması: (Kernel-1) * Dilation
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

        # Sınıflandırma Katmanı
        self.linear = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x):
        # Gelen veri şekli: (Batch, 1, 128, 157) -> Eski dataset yapısı
        # TCN için gereken: (Batch, 128, 157) -> (Batch, Kanal, Zaman)

        if x.dim() == 4:
            x = x.squeeze(1)  # O fazladan '1' boyutunu atıyoruz.

        y = self.network(x)

        # Global Average Pooling (Zaman eksenindeki tüm özellikleri ortala)
        # y shape: (Batch, Channels, Time) -> (Batch, Channels)
        y = y.mean(dim=2)

        return self.linear(y)