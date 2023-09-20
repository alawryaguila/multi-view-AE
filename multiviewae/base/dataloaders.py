import random
import numpy as np
import pytorch_lightning as pl
import hydra
from torch.utils.data import DataLoader

class MultiviewDataModule(pl.LightningDataModule):
    """LightningDataModule for multi-view data.

    Args:
        n_views (int): Number of views in the data.
        batch_size (int): Batch size.
        is_validate (bool): Whether to use a validation set.
        train_size (float): Proportion of batch to use for training between 0 and 1. Remainder of batch is used for validation.
        data (list): Input data. list of torch.Tensors.
        labels (np.array): Dataset labels. 
    """
    def __init__(
            self,
            n_views,
            batch_size,
            is_validate,
            train_size,
            dataset,
            data,
            labels
        ):

        super().__init__()
        self.n_views = n_views
        self.batch_size = batch_size
        self.is_validate = is_validate
        self.train_size = train_size
        self.data = data  
        self.labels = labels 
        self.dataset = dataset
        if not isinstance(self.batch_size, int):
            self.batch_size = self.data[0].shape[0]

    def setup(self, stage):
        if self.is_validate:
            train_data, test_data, train_labels, test_labels = self.train_test_split()
            self.train_dataset = hydra.utils.instantiate(self.dataset, data=train_data, labels=train_labels, n_views=self.n_views)
            self.test_dataset = hydra.utils.instantiate(self.dataset, data=test_data, labels=test_labels, n_views=self.n_views)
        else:
            self.train_dataset = hydra.utils.instantiate(self.dataset, data=self.data, labels=self.labels, n_views=self.n_views) 
            self.test_dataset = None
        del self.data

    def train_test_split(self):

        N = self.data[0].shape[0]
        train_idx = list(random.sample(range(N), int(N * self.train_size)))
        test_idx = np.setdiff1d(list(range(N)), train_idx)

        train_data = []
        test_data = []
        for dt in self.data:
            train_data.append(dt[train_idx, :])
            test_data.append(dt[test_idx, :])

        train_labels = None
        test_labels = None
        if self.labels is not None:
            train_labels = self.labels[train_idx]
            test_labels = self.labels[test_idx]

        return train_data, test_data, train_labels, test_labels

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0) #use default num_workers for now, problem in windows! 

    def val_dataloader(self):
        if self.is_validate:
            return DataLoader(
                self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        return None


class IndexDataModule(MultiviewDataModule):
    """LightningDataModule for multi-view data.

    Args:
        n_views (int): Number of views in the data.
        batch_size (int): Batch size.
        is_validate (bool): Whether to use a validation set.
        train_size (float): Proportion of batch to use for training between 0 and 1. Remainder of batch is used for validation.
        data (list): Input data. list of identifiers to load data from.
        labels (np.array): Dataset labels. 
    """

    def __init__(
            self,
            n_views, 
            batch_size,
            is_validate,
            train_size,
            dataset,
            data,
            labels,
        ):
        data_ = data[0]
        if not isinstance(batch_size, int):
            batch_size = len(data_)

        super().__init__(n_views=n_views, batch_size=batch_size, is_validate=is_validate, train_size=train_size, 
                         dataset=dataset, data=data_, labels=labels)

    def train_test_split(self):

        N = len(self.data)
        train_idx = list(random.sample(range(N), int(N * self.train_size)))
        test_idx = list(set(list(range(N))) -  set(train_idx))
        data = self.data

        train_data = [data[i] for i in train_idx]
        test_data = [data[i] for i in test_idx]

        train_labels = None
        test_labels = None
        if self.labels is not None:
            train_labels = self.labels[train_idx]
            test_labels = self.labels[test_idx]

        return [train_data], [test_data], train_labels, test_labels
