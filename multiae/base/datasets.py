"""
Class for loading data into Pytorch float tensor

From: https://gitlab.com/acasamitjana/latentmodels_ad

"""
import numpy as np
import pandas as pd
import torch

from torch.functional import Tensor
from torchvision import transforms
from torch.utils.data import Dataset

class MVDataset(Dataset):
    def __init__(self, data, labels, return_index=False, transform=None):
        self.data = data
        self.labels = labels
        self.return_index = return_index
        self.transform = transform

        # TODO: assumes the same N? assert somewhere
        self.N = len(self.data[0])
        self.data = [
            torch.from_numpy(d).float() if isinstance(d, np.ndarray) else d
            for d in self.data
        ]
        self.shape = [
            np.shape(d) for d in self.data
        ]

        if labels is not None:
            self.labels = torch.from_numpy(self.labels).long()

    def __getitem__(self, index):
        x = [d[index] for d in self.data]
        if self.transform:
            x = self.transform(x)

        if self.return_index:
            if self.labels is not None:
                return x, self.labels[index], index
            return x, index

        if self.labels is not None:
            return x, self.labels[index]
        return x

    def __len__(self):
        return self.N
