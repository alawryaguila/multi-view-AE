import random
import hydra
import torch
from torch.utils.data import Dataset
from os.path import join
import torch
import pytorch_lightning as pl
import imageio
from torch.utils.data import DataLoader
from torchvision import transforms
torch.manual_seed(42)


class MMNISTDataModule(pl.LightningDataModule):

    def __init__(
            self,
            batch_size,
            n_views,
            is_validate,
            train_size,
            dataset,
            data,
            labels,
        ):

        super().__init__()
        self.batch_size = batch_size
        self.is_validate = is_validate
        self.train_size = train_size
        self.n_views = n_views
        self.data = data[0] #data here is actually just a list of file names, got converted to list even though already list so just take first element
        self.labels = labels
        self.dataset = dataset
        if not isinstance(self.batch_size, int):
            self.batch_size = len(data[0])

    def train_test_split(self):

        N = len(self.data)
        train_idx = list(random.sample(range(N), int(N * self.train_size)))
        test_idx = list(set(list(range(N))) -  set(train_idx))
        data = self.data
        labels = self.labels

        train_data = [data[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        test_data = [data[i] for i in test_idx]
        test_labels = [labels[i] for i in test_idx]

        return train_data, test_data, train_labels, test_labels

    def setup(self, stage):

        if self.is_validate:
            train_data, test_data, train_labels, test_labels = self.train_test_split()
            self.train_dataset = hydra.utils.instantiate(self.dataset, data=[train_data], labels=train_labels, n_views=self.n_views) 
            self.test_dataset = hydra.utils.instantiate(self.dataset, data=[test_data], labels=test_labels, n_views=self.n_views)
        else:
            self.train_dataset = hydra.utils.instantiate(self.dataset, data=[self.data], labels=self.labels, n_views=self.n_views) 
            self.test_dataset = None


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8, drop_last=True)  

    def val_dataloader(self):
        if self.is_validate:
            return DataLoader(
                self.test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8, drop_last=True)
        return None


class MVDataset(Dataset):
    def __init__(self, 
                data, 
                n_views,
                is_path_ds,
                labels=None,
                data_dir='',
                views=[0, 1]
                ):
        self.N = len(data[0])
        self.views = views
        self.data_dir = data_dir
        self.data = data[0]
        self.labels = labels
        self.transform = transforms.Compose([transforms.ToTensor()]) 
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            list (): returns list of images
        """
        x = []
        for i in self.views:
            idx = self.data[index]
            x_i = imageio.imread(join(self.data_dir, 'm{0}/{1}'.format(i, idx)))
            x_i = self.transform(x_i)
            x.append(x_i)
        return x

    def __len__(self):
        return self.N
