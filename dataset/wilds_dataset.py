import copy
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

class GeneralWilds_Batched_Dataset(Dataset):
    """
    Batched dataset for Amazon, Camelyon and IwildCam. Allows for getting a batch of data given
    a specific domain index.
    """
    def __init__(self, train_data, nth_domain, batch_size=16, domain_idx=0):
        
        domains = train_data.metadata_array[:, domain_idx]
        train_data._input_array = [train_data.dataset._input_array[i] for i in train_data.indices]
        
        
        self.num_envs = len(domains.unique())
        self.eval = train_data.eval
        self.collate = train_data.collate
        self.metadata_fields = train_data.metadata_fields
        self.data_dir = train_data.data_dir
        self.transform = train_data.transform
        self.batch_size = batch_size

        self.data = train_data._input_array
        self.targets = train_data.y_array
        self.domains = train_data.metadata_array[:, domain_idx]

        
        flags = self.domains==nth_domain
        self.data = [self.data[i] for i,_ in enumerate(self.data) if flags[i]]
        self.targets = self.targets[self.domains==nth_domain]
        self.domains = self.domains[self.domains==nth_domain]
        
    def get_input(self, idx):
        """Returns x for a given idx."""

        img_path = f'{self.data_dir}/{self.data[idx]}'
        img = Image.open(img_path).convert('RGB')
        return img

    def __getitem__(self, idx):
        return self.transform(self.get_input(idx)), self.targets[idx]

    def __len__(self):
        return len(self.targets)
