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

from .class_dict import nimagenet_cls
from data.utils.load_data import load_cd_events
from data.utils.augmentation import RandomHFlip, RandomCrop, RandomTranslate, Crop
from data.utils.representations import generate_event_frame, generate_event_voxel, generate_event_spikes, generate_event_graph

#########################################################################
############################## DATA MODULE ##############################
#########################################################################

class NImageNet(L.LightningDataModule):
    def __init__(self, cfg_dataset, cfg_model):
        super().__init__()

        self.cfg_dataset = cfg_dataset
        self.cfg_model = cfg_model

        # Dataset directory and name.
        self.data_name = cfg_dataset.name
        self.path = cfg_dataset.path

        # Number of classes and class dictionary.
        self.num_classes = cfg_dataset.general.num_classes
        self.class_dict = nimagenet_cls
    
    def setup(self, stage=None):
        data_files_train = glob.glob(os.path.join(self.path, 'Train', '*', '*.dat'))
        data_files_test = glob.glob(os.path.join(self.path, 'Validate', '*', '*.dat'))
        data_files_val = glob.glob(os.path.join(self.path, 'Validate', '*', '*.dat'))

        if self.cfg_dataset.train.all_noisy:
            data_files_val = glob.glob('/net/storage/pr3/plgrid/plgg_dvs_phd/N-miniImageNet_filtered40000_noise/*/Validate/*/*.dat')

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
                            collate_fn=self.collate_fn_graph if self.cfg_dataset.representation.type == 'event_graph' else self.collate_fn_dense, 
                            persistent_workers=False)

    def val_dataloader(self):
        return DataLoader(self.val_data, 
                            batch_size=self.cfg_dataset.train.batch_size, 
                            num_workers=self.cfg_dataset.train.num_workers, 
                            shuffle=False, 
                            collate_fn=self.collate_fn_graph if self.cfg_dataset.representation.type == 'event_graph' else self.collate_fn_dense, 
                            persistent_workers=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, 
                            batch_size=self.cfg_dataset.train.batch_size, 
                            num_workers=self.cfg_dataset.train.num_workers, 
                            shuffle=False, 
                            collate_fn=self.collate_fn_graph if self.cfg_dataset.representation.type == 'event_graph' else self.collate_fn_dense, 
                            persistent_workers=False)
    
    @staticmethod
    def collate_fn_dense(data_list):
        batch_img = [img for img, _ in data_list]
        batch_img = torch.stack(batch_img, dim=0)
        batch_label = [label for _, label in data_list]
        batch_label = torch.tensor(batch_label)
        return {"x": batch_img, 
                "y": batch_label}

    @staticmethod
    def collate_fn_graph(data_list):
        list_graph = [graph for graph, _ in data_list]
        list_label = [label for _, label in data_list]
        batch_graph = torch_geometric.data.Batch.from_data_list(list_graph)
        batch_label = torch.tensor(list_label)
        return {"x": batch_graph,
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
        data_file = self.augment_noise(data_file)
        class_id = self.get_class_id(data_file)

        data = load_cd_events(data_file)
        data = torch.tensor(data, dtype=torch.float32)
        data = self.change_polarity(data)
        data = self.cut_events(data)

        if self.augmentation:
            data = self.random_h_flip(data)
            data = self.random_crop(data)
            data = self.random_translate(data)
            data = self.crop(data)

        representation = self.generate_representation(data)
        return representation, class_id

    def augment_noise(self, data_file):
        if self.augmentation and self.cfg.train.all_noisy:
            list_aug = ['0.1', '0.01', '0.5', '0.05', '0.25', '0.75', '1', '1.5', '2', '2.5', '3', '4', '5']
            aug = np.random.choice(list_aug)
            data_file = data_file.replace('N-miniImageNet_dat', f'N-miniImageNet_filtered40000_noise/{aug}')
        return data_file

    def cut_events(self, events):
        time_window = self.cfg.general.time_window

        num_events = len(events[:, 0])
        if num_events == 0:
            return events
        t = events[:, 3][num_events//2]
        idx1 = torch.clip(torch.searchsorted(events[:, 3].contiguous(), t + time_window//2), 0, num_events)
        idx0 = torch.clip(torch.searchsorted(events[:, 3].contiguous(), t - time_window//2), 0, num_events)

        events = events[idx0:idx1]
        return events

    def change_polarity(self, data):
        data = data.clone()
        data[:, 2] = torch.where(data[:, 2] == 2, torch.tensor(0, dtype=torch.float32), data[:, 2])
        data[:, 2] = torch.where(data[:, 2] == 3, torch.tensor(1, dtype=torch.float32), data[:, 2])
        return data

    def get_class_id(self, data_file):
        class_name = data_file.split('/')[-2]
        class_id = nimagenet_cls[class_name]
        return class_id

    def generate_representation(self, events):
        rep_type = self.cfg.representation.type
        if rep_type == 'event_frame':
            return generate_event_frame(events, self.cfg)
        elif rep_type == 'event_voxel':
            return generate_event_voxel(events, self.cfg)
        elif rep_type == 'event_spikes':
            return generate_event_spikes(events, self.cfg)
        elif rep_type == 'event_graph':
            return generate_event_graph(events, self.cfg)
        else:
            raise ValueError(f"Representation type {rep_type} not supported.")