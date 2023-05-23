"""
Class for loading data into Pytorch float tensor

From: https://gitlab.com/acasamitjana/latentmodels_ad

"""
import numpy as np
import torch
from torch.utils.data import Dataset
from os.path import join

class MVDataset(Dataset):
    """PyTorch Dataset for storing and accessing multi-view data.

    Args:
        data (list): Input data. list of torch.Tensors.
        labels (np.array): Dataset labels. 
        return_index (bool): Whether to return batch index labels.
        transform (torchvision.transforms): Torchvision transformation to apply to the data. Default is None.
    """
    def __init__(self, data, n_views, labels=None, return_index=False, transform=None):

        self.data = data
        self.labels = labels
        self.n_views = n_views
        self.return_index = return_index
        self.transform = transform

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

class IndexMVDataset(Dataset):

    def __init__(self, data, n_views, labels=None, filename="", data_dir=''):
        self.N = len(data[0]) 
        self.n_views = n_views

        self.filename = filename
        self.data_dir = data_dir
        
        self.data = data[0]
        self.labels = labels

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            list (): returns list of numpy arrays
        """
        x = []
        idx = self.data[index]
        for i in range(self.n_views):
            x_i = np.load(join(self.data_dir, self.filename.format(i, idx)))
            x_i = torch.from_numpy(x_i).float()
            x.append(x_i)
        return x

    def __len__(self):
        return self.N
    

