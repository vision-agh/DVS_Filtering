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
from .class_dict import ncaltech_dict
from data.utils.load_data import load_cd_events

from data.utils.augmentation import RandomHFlip, RandomCrop, RandomZoom, RandomTranslate, Crop


#########################################################################
############################## DATA MODULE ##############################
#########################################################################

class NCaltech101(L.LightningDataModule):
    def __init__(self, cfg_dataset, cfg_model):
        super().__init__()

        self.cfg_dataset = cfg_dataset
        self.cfg_model = cfg_model

        # Dataset directory and name.
        self.data_name = cfg_dataset.name
        self.path = cfg_dataset.path

        # Number of classes and class dictionary.
        self.num_classes = cfg_dataset.general.num_classes
        self.class_dict = ncaltech_dict
    
    def setup(self, stage=None):
        data_files_train = glob.glob(os.path.join(self.path, 'training', '*.dat'))
        data_files_test = glob.glob(os.path.join(self.path, 'testing', '*.dat'))
        data_files_val = glob.glob(os.path.join(self.path, 'validation', '*.dat'))

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
        # self.random_zoom = RandomZoom(cfg)
        self.random_translate = RandomTranslate(cfg)
        self.crop = Crop(cfg)

    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, index: int):
        data_file = self.files[index]
        data = load_cd_events(data_file)
        data = torch.tensor(data, dtype=torch.float32)

        data = self.cut_events(data)

        if self.augmentation:
            data = self.random_h_flip(data)
            data = self.random_crop(data)
            # data = self.random_zoom(data)
            data = self.random_translate(data)
            data = self.crop(data)

        representation = self.generate_representation(data)
        return representation, 0
    
    def cut_events(self, events):
        time_window = self.cfg.general.time_window

        num_events = len(events[:, 0])
        t = events[:, 3][num_events//2]
        idx1 = torch.clip(torch.searchsorted(events[:, 3].contiguous(), t + time_window//2), 0, num_events)
        idx0 = torch.clip(torch.searchsorted(events[:, 3].contiguous(), t - time_window//2), 0, num_events)

        events = events[idx0:idx1]
        return events

    def generate_representation(self, events):
        rep_type = self.cfg.representation.type
        if rep_type == 'event_frame':
            return generate_event_frame(events, self.cfg)
        elif rep_type == 'event_voxel':
            return generate_event_voxel(events, self.cfg)
        else:
            raise ValueError(f"Representation type {rep_type} not supported.")


#########################################################################
############################# REPRESENTATION ############################
#########################################################################

def generate_event_frame(events, cfg):
    cfg = cfg.representation.event_frame
    width, height = cfg.dim

    frame = torch.zeros(2, height, width, dtype=torch.float32)

    indices = torch.stack([ events[:, 2].long(), 
                            events[:, 1].long(), 
                            events[:, 0].long()], dim=0)

    values = torch.ones_like(events[:, 0], dtype=torch.float32)
    frame.index_put_(tuple(indices), values, accumulate=True)
    return frame

def generate_event_voxel(events, cfg):
    cfg = cfg.representation.event_frame
    width, height = cfg.dim

    frame = torch.zeros(2, height, width, dtype=torch.float32)

    indices = torch.stack([ events[:, 2].long(), 
                            events[:, 1].long(), 
                            events[:, 0].long()], dim=0)

    values = torch.ones_like(events[:, 0], dtype=torch.float32)
    frame.index_put_(tuple(indices), values, accumulate=True)
    return frame