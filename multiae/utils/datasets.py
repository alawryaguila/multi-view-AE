"""
Class for loading data into Pytorch float tensor

From: https://gitlab.com/acasamitjana/latentmodels_ad

"""
import torch
from torch.functional import Tensor
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import pandas as pd


class MyDataset(Dataset):
    def __init__(self, data, indices=False, transform=None):
        
        self.data = data
        if isinstance(data, (list, tuple)):
            self.data = [
                torch.from_numpy(d).float() if isinstance(d, np.ndarray) else d
                for d in self.data
            ]
            self.N = len(self.data[0])
            self.shape = np.shape(self.data[0])
        elif isinstance(data, np.ndarray):
            self.data = torch.from_numpy(self.data).float()
            self.N = len(self.data)
            self.shape = np.shape(self.data)

        self.transform = transform
        self.indices = indices

    def __getitem__(self, index):
        if isinstance(self.data, (list, tuple)):
            x = [d[index] for d in self.data]
        else:
            x = self.data[index]
        if self.transform:
            x = self.transform(x)

        if self.indices:
            return x, index
        return x

    def __len__(self):
        return self.N


class MyDataset_labels(Dataset):
    def __init__(self, data, labels, indices=False, transform=None):
        self.data = data
        self.labels = labels
        if isinstance(data, (list, tuple)):
            self.data = [
                torch.from_numpy(d).float() if isinstance(d, np.ndarray) else d
                for d in self.data
            ]
            self.N = len(self.data[0])
            self.shape = np.shape(self.data[0])
        elif isinstance(data, np.ndarray):
            self.data = torch.from_numpy(self.data).float()
            self.N = len(self.data)
            self.shape = np.shape(self.data)

        self.labels = torch.from_numpy(self.labels).long()

        self.transform = transform
        self.indices = indices

    def __getitem__(self, index):
        if isinstance(self.data, list):
            x = [d[index] for d in self.data]
        else:
            x = self.data[index]

        if self.transform:
            x = self.transform(x)
        t = self.labels[index]
        if self.indices:
            return x, t, index
        return x, t

    def __len__(self):
        return self.N
