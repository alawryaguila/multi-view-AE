from torchvision import transforms
from torch.utils.data import DataLoader
from .datasets import MyDataset, MyDataset_labels
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import random
import torch

class MultiviewDataModule(pl.LightningDataModule):
    def __init__(self, *data, labels=None, batch_size=None, val=False):
        self.data = data
        if batch_size is None:
            self.batch_size = (
                data[0].shape[0]
                if (type(data) == list or type(data) == tuple)
                else data.shape[0]
            )
        else:
            self.batch_size = batch_size

        self.val = val
        self.labels = labels

    def split(self, *data, labels=None, split=0.9):
        random.seed(42)
        if isinstance(data, (list, tuple)) and isinstance(data[0], (np.ndarray, torch.Tensor)):
            return self.data_split(data, labels, split)
        return self.list_split(data, labels, split)

    def data_split(self, data, labels, split):
        N = data[0].shape[0]
        idx_1 = list(random.sample(range(0, N), int(N * split)))
        idx_2 = np.setdiff1d(list(range(N)), idx_1)
        data_1 = []
        data_2 = []
        for data_ in data:
            data_1_ = data_[idx_1, :]
            data_2_ = data_[idx_2, :]
            data_1.append(data_1_)
            data_2.append(data_2_)
        if labels is not None:
            labels_1, labels_2 = self.labels_split(labels, idx_1, idx_2)
            return [data_1, data_2, labels_1, labels_2]
        return [data_1, data_2, None, None]

    def list_split(self, idx_list, labels, split):
        N = len(idx_list)
        idx_1 = list(random.sample(range(0, N), int(N * split)))
        idx_2 = np.setdiff1d(list(range(N)), idx_1)
        list_1 = idx_list[idx_1, :]
        list_2 = idx_list[idx_2, :]
        if labels is not None:
            labels_1, labels_2 = self.labels_split(labels, idx_1, idx_2)
            return [list_1, list_2, labels_1, labels_2]
        return [list_1, list_2, None, None]

    def labels_split(self, labels, idx_1, idx_2):
        labels_1 = labels[idx_1]
        labels_2 = labels[idx_2]

        return [labels_1, labels_2]

    def process_labels(self, labels):
        if isinstance(labels, pd.core.series.Series):
            return self.labels.values.reshape(-1)
        elif isinstance(labels, np.ndarray):
            return self.labels.reshape(-1)
        elif isinstance(labels, list):
            return np.array(self.labels)

    def setup(self, stage):
        if self.labels is not None:
            self.labels = self.process_labels(self.labels)
        if self.val:
            train_data, val_data, train_labels, val_labels = self.split(
                *self.data, labels=self.labels
            )
            self.train_dataset = self.dataset(*train_data, labels=train_labels)
            self.val_dataset = self.dataset(*val_data, labels=val_labels)
        else:
            self.train_dataset = self.dataset(*self.data, labels=self.labels)
            self.val_dataset = None

    @staticmethod
    def dataset(*data, labels=None):
        data = (
            data[0] if len(data) == 1 else data
        )  # hacky work around to single data view
        if labels is not None:
            return MyDataset_labels(data, labels)
        return MyDataset(data)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False)

    def val_dataloader(self):
        if self.val:
            return DataLoader(
                self.val_dataset, batch_size=self.batch_size, shuffle=False
            )
        else:
            return None
