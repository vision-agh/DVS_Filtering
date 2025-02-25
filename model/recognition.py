import torch
import lightning as L

from torchmetrics.functional.classification import accuracy
from torchmetrics import Accuracy
from torchmetrics.classification import ConfusionMatrix

from typing import Dict, Tuple
from torch.nn.functional import softmax

from model.cnn.resnet import ResNetModel

import wandb
import numpy as np
import matplotlib.pyplot as plt


class LNRecognition(L.LightningModule):
    def __init__(self, cfg_dataset, cfg_model):
        super().__init__()
        self.lr = cfg_dataset.train.lr
        self.weight_decay = cfg_dataset.train.weight_decay

        self.batch_size = cfg_dataset.train.batch_size
        self.num_classes = cfg_dataset.general.num_classes

        self.model = ResNetModel(2, 'resnet18', num_classes=self.num_classes)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)

        if self.num_classes > 3:
            self.accuracy_top_3 = Accuracy(task="multiclass", num_classes=self.num_classes, top_k=3).to(self.device)

        self.save_hyperparameters()

        self.val_pred = None
        self.train_pred = None
        self.pred = []
        self.target = []

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=self.lr,
            div_factor=25,
            final_div_factor=10000 / 25,
            total_steps=150000,
            pct_start=0.005,
            cycle_momentum=False,
            anneal_strategy='linear')
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1,
            "strict": True,
            "name": 'learning_rate',
        }

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}

    def forward(self, data):
        x = self.model(data)
        return x

    def training_step(self, batch, batch_idx):
        outputs = self.forward(data=batch['x'])
        loss = self.criterion(outputs, target=batch['y'])

        y_prediction = torch.argmax(outputs, dim=-1)
        acc = accuracy(preds=y_prediction, target=batch['y'], task="multiclass", num_classes=self.num_classes)

        self.log('train_loss', loss, on_epoch=True, logger=True, batch_size=self.batch_size)
        self.log('train_acc', acc, on_epoch=True, logger=True, batch_size=self.batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(data=batch['x'])
        
        loss = self.criterion(outputs, target=batch['y'])
        y_prediction = torch.argmax(outputs, dim=-1)

        acc = accuracy(preds=y_prediction, target=batch['y'], task="multiclass", num_classes=self.num_classes)
        self.log('val_loss', loss, on_epoch=True, logger=True, batch_size=self.batch_size)
        self.log('val_acc', acc, on_epoch=True, logger=True, batch_size=self.batch_size)

        if self.num_classes > 3:
            pred = softmax(outputs, dim=-1)
            top_3 = self.accuracy_top_3(preds=pred, target=batch['y'])
            self.log('val_acc_top_3', top_3, on_epoch=True, logger=True, batch_size=self.batch_size)
    
    def test_step(self, batch, batch_idx):
        outputs = self.forward(data=batch['x'])

        loss = self.criterion(outputs, target=batch['y'])
        y_prediction = torch.argmax(outputs, dim=-1)
        
        acc = accuracy(preds=y_prediction, target=batch['y'], task="multiclass", num_classes=self.num_classes)

        self.log('test_loss', loss, on_epoch=True, logger=True, batch_size=self.batch_size)
        self.log('test_acc', acc, on_epoch=True, logger=True, batch_size=self.batch_size)

        if self.num_classes > 3:
            pred = softmax(outputs, dim=-1)
            top_3 = self.accuracy_top_3(preds=pred, target=batch['y'])
            self.log('test_acc_top_3', top_3, on_epoch=True, logger=True, batch_size=self.batch_size)