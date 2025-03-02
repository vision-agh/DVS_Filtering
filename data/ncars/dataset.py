import os
import glob
import numpy as np
import lightning as L
import torch_geometric
import cv2
import numba
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset

from .class_dict import ncars_dict
from data.utils.load_data import load_cd_events
from data.utils.augmentation import RandomHFlip, RandomCrop, RandomTranslate, Crop
from data.utils.representations import generate_event_frame, generate_event_voxel, generate_event_spikes, generate_event_graph

#########################################################################
############################## DATA MODULE ##############################
#########################################################################

class NCars(L.LightningDataModule):
    def __init__(self, cfg_dataset, cfg_model):
        super().__init__()

        self.cfg_dataset = cfg_dataset
        self.cfg_model = cfg_model

        # Dataset directory and name.
        self.data_name = cfg_dataset.name
        self.path = cfg_dataset.path

        # Number of classes and class dictionary.
        self.num_classes = cfg_dataset.general.num_classes
        self.class_dict = ncars_dict
    
    def setup(self, stage=None):
        data_files_train = glob.glob(os.path.join(self.path, 'n-cars_train', '*', '*.dat'))
        data_files_test = glob.glob(os.path.join(self.path, 'n-cars_test', '*', '*.dat'))

        data_files_train = np.array(data_files_train)
        np.random.shuffle(data_files_train)

        split_idx = int(len(data_files_train) * 0.9)
        data_files_train, data_files_val = data_files_train[:split_idx], data_files_train[split_idx:]
        
        data_files_train = data_files_train.tolist()
        data_files_val = data_files_val.tolist()

        if self.cfg_dataset.train.all_noisy:
            noise_list = ['0.1', '0.01', '0.5', '0.05', '0.25', '0.75', '1', '1.5', '2', '2.5', '3', '4', '5']
            new_data_files_val = []
            for file in data_files_val:
                for nl in noise_list:
                    new_path = file.replace('NCARS_dat', f'NCARS_filtered40000_noise/{nl}')
                    new_data_files_val.append(new_path)

            data_files_val = new_data_files_val

        self.train_data = DS(data_files_train, 
                            augmentation=True, 
                            cfg=self.cfg_dataset)

        self.test_data = DS(data_files_test, 
                            augmentation=False, 
                            cfg=self.cfg_dataset)

        self.val_data = DS(data_files_val, 
                            augmentation=False,
                            cfg=self.cfg_dataset)

    def train_dataloader(self):
        return DataLoader(self.train_data, 
                            batch_size=self.cfg_dataset.train.batch_size, 
                            num_workers=self.cfg_dataset.train.num_workers, 
                            shuffle=True, 
                            collate_fn=self.collate_fn, 
                            persistent_workers=False)

    def val_dataloader(self):
        return DataLoader(self.val_data, 
                            batch_size=self.cfg_dataset.train.batch_size, 
                            num_workers=self.cfg_dataset.train.num_workers, 
                            shuffle=False, 
                            collate_fn=self.collate_fn, 
                            persistent_workers=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, 
                            batch_size=self.cfg_dataset.train.batch_size, 
                            num_workers=self.cfg_dataset.train.num_workers, 
                            shuffle=False, 
                            collate_fn=self.collate_fn, 
                            persistent_workers=False)
    
    @staticmethod
    def collate_fn(data_list):
        batch_img = [img for img, _ in data_list]
        batch_img = torch.stack(batch_img, dim=0)
        batch_label = [label for _, label in data_list]
        batch_label = torch.tensor(batch_label)
        return {"x": batch_img, 
                "y": batch_label}

#########################################################################
################################ DATASET ################################
#########################################################################

class DS(Dataset):
    def __init__(self, 
                    files, 
                    augmentation, 
                    cfg):

        self.files = files
        self.augmentation = augmentation
        self.cfg = cfg

        self.random_h_flip = RandomHFlip(cfg)
        self.random_crop = RandomCrop(cfg)
        self.random_translate = RandomTranslate(cfg)
        self.crop = Crop(cfg)

    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, index: int):
        data_file = self.files[index]

        #changed start
        if self.augmentation and self.cfg.train.all_noisy:
            list_aug = ['0.1', '0.01', '0.5', '0.05', '0.25', '0.75', '1', '1.5', '2', '2.5', '3', '4', '5']
            aug = np.random.choice(list_aug)
            data_file = data_file.replace('NCARS_dat', f'NCARS_filtered40000_noise/{aug}')
        # changed end

        data = load_cd_events(data_file)
        data = torch.tensor(data, dtype=torch.float32)
        
        data = data.clone()
        data[:, 2] = torch.where(data[:, 2] == 2, torch.tensor(0, dtype=torch.float32), data[:, 2])
        data[:, 2] = torch.where(data[:, 2] == 3, torch.tensor(1, dtype=torch.float32), data[:, 2])

        class_name = data_file.split('/')[-2]
        class_id = ncars_dict[class_name]

        if self.augmentation:
            data = self.random_h_flip(data)
            data = self.random_crop(data)
            data = self.random_translate(data)
            data = self.crop(data)

        representation = self.generate_representation(data)
        return representation, class_id

    def generate_representation(self, events):
        rep_type = self.cfg.representation.type
        if rep_type == 'event_frame':
            return generate_event_frame(events, self.cfg)
        elif rep_type == 'event_voxel':
            return generate_event_voxel(events, self.cfg)
        elif rep_type == 'event_spikes':
            return generate_event_spikes(events, self.cfg)
        elif rep_type == 'event_graph'
            return generate_event_graph(events, self.cfg)
        else:
            raise ValueError(f"Representation type {rep_type} not supported.")