import os
import glob
import numpy as np
import torch
import lightning as L
import torch_geometric
import numba

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from torch.utils.data import DataLoader
from class_dict import ncaltech_dict

class NCaltech101(L.LightningDataModule):
    def __init__(self, cfg_dataset, cfg_model):
        super().__init__()

        self.cfg_dataset = cfg_dataset
        self.cfg_model = cfg_model

        # Dataset directory and name.
        self.data_name = cfg_dataset.name
        self.path = cfg_dataset.path

        # Time window, original dimension.
        self.time_window = cfg_dataset.general.time_window
        self.dim = cfg_dataset.general.dim

        # Number of classes and class dictionary.
        self.num_classes = cfg_dataset.general.num_classes
        self.class_dict = ncaltech_dict
    
    def setup(self, stage=None):
        split_train, split_val, split_test = self.cfg_dataset.general.train_val_test_split

        all_files = glob.glob(os.path.join(self.path, '*.dat'))

        data_files_train = np.random.choice(all_files, int(split_train * len(all_files)), replace=False)
        data_files_test_val = np.setdiff1d(all_files, data_files_train)
        data_files_test = np.random.choice(data_files_test_val, int(split_test/(split_test+split_val) * len(data_files_test_val)), replace=False)
        data_files_val = np.setdiff1d(data_files_test_val, data_files_test)

        self.train_data = DS(data_files_train, augmentation=True, cfg=self.cfg_dataset)
        self.test_data = DS(data_files_test, augmentation=False, cfg=self.cfg_dataset)
        self.val_data = DS(data_files_val, augmentation=False, cfg=self.cfg_dataset)

        print(self.train_data)

    def train_dataloader(self):
        return DataLoader(self.train_data, 
                            batch_size=self.batch_size, 
                            num_workers=self.num_workers, 
                            shuffle=True, 
                            collate_fn=self.collate_fn, 
                            persistent_workers=False)

    def val_dataloader(self):
        return DataLoader(self.val_data, 
                            batch_size=self.batch_size, 
                            num_workers=self.num_workers, 
                            shuffle=False, 
                            collate_fn=self.collate_fn, 
                            persistent_workers=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, 
                            batch_size=self.batch_size, 
                            num_workers=self.num_workers, 
                            shuffle=False, 
                            collate_fn=self.collate_fn, 
                            persistent_workers=False)
    
    @staticmethod
    def collate_fn(data_list):
        batch = torch_geometric.data.Batch.from_data_list(data_list)
        if hasattr(data_list[0], 'bbox'):
            batch_bbox = sum([[i] * len(data.y) for i, data in enumerate(data_list)], [])
            batch.batch_bbox = torch.tensor(batch_bbox, dtype=torch.long)
        return batch
