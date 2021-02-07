'''
Class for loading data into Pytorch float tensor

From: https://gitlab.com/acasamitjana/latentmodels_ad

'''
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class MyDataset(Dataset):
    def __init__(self, data, indices = False, transform=None):
        if isinstance(data,list):
            self.data = [torch.from_numpy(d).float() for d in data ]
            self.N = len(self.data[0])
            self.shape = np.shape(self.data[0])
        else:
            self.data = torch.from_numpy(data).float()
            self.N = len(self.data)
            self.shape = np.shape(self.data)

        self.transform = transform
        self.indices = indices

    def __getitem__(self, index):
        if isinstance(self.data,list):
            x = [d[index] for d in self.data]
        else:
            x = self.data[index]

        if self.transform:
            x = self.transform(x)

        if self.indices:
            return x, index
        else:
            return x

    def __len__(self):
        return self.N

