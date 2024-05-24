import ujson
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch
import torch.nn as nn
from scipy.fftpack import fft


class IMUDataset_fft(Dataset):
    """ Load sentence pair (sequential or random order) from corpus """
    def __init__(self, data, labels, isNormalization=True):
        super().__init__()
        self.data = data
        self.labels = labels
        self.feature_len = 6
        if isNormalization:
            self.instance_norm = nn.InstanceNorm1d(self.feature_len)
            self.data = torch.tensor(self.data.transpose((0,2,1)))
            self.data = self.instance_norm(self.data)
            self.data = self.data.numpy().transpose((0,2,1))
        self.data = self.data.transpose((0,2,1))

    def __getitem__(self, index):
        instance = self.data[index]
        N = instance.shape[1]
        fft_y = fft(instance, axis=1)
        abs_y = np.abs(fft_y)
        normalization_y = abs_y/N
        return torch.from_numpy(instance).float(), torch.from_numpy(normalization_y).float(), torch.from_numpy(np.array(self.labels[index])).long()

    def __len__(self):
        return len(self.data)


