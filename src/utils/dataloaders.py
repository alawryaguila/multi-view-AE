
from torchvision import transforms
from torch.utils.data import  DataLoader
from .datasets import MyDataset
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import random 

class MultiviewDataModule(pl.LightningDataModule):
    def __init__(self, data, batch_size=None, val=False):
        self.data = data
        self.batch_size = batch_size
        self.val = val

    def data_split(self, split=0.9):
        random.seed(42)
        idx_1 = list(random.sample(range(0, self.data[0].shape[0]), int(self.data[0].shape[0]*split)))
        idx_2 = np.setdiff1d(list(range(self.data[0].shape[0])),idx_1)
        data_1 = []
        data_2 = []
        for data_ in self.data:
            data_1_ = data_[idx_1,:]
            data_2_ = data_[idx_2,:]
            data_1.append(data_1_)
            data_2.append(data_2_)
        return [data_1, data_2]

    def setup(self, stage):
        if self.val:
            train_data, val_data = self.data_split()
            self.train_dataset = self.dataset(train_data) 
            self.val_dataset = self.dataset(val_data)
        else:
            self.train_dataset = self.dataset(self.data)
            self.val_dataset = None      

    @staticmethod
    def dataset(data):
        return MyDataset(data)
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False)

    def val_dataloader(self):
        if self.val:
            return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        else:
            return None