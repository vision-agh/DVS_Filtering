import lightning as L
import argparse
import multiprocessing as mp
import wandb

from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from omegaconf import OmegaConf
from data.ncaltech101.dataset import NCaltech101
from model.recognition import LNRecognition

def main(args):
    cfg_dataset = OmegaConf.load(args.config_data)

    if 'cnn' in args.config_model:
        cfg_dataset.representation.type = 'event_frame'
    elif 'vit' in args.config_model:
        cfg_dataset.representation.type = 'event_voxel'

    cfg_model = OmegaConf.load(args.config_model)

    dm = NCaltech101(cfg_dataset, cfg_model)
    dm.setup()

    model = LNRecognition(cfg_dataset, cfg_model)

    lr_monitor = LearningRateMonitor(logging_interval='step')

    wandb_logger = WandbLogger(project=f'dvs_filtering', name=f'{cfg_dataset.name}_{cfg_dataset.representation.type}')
    wandb_logger.watch(model)

    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='best_model_filtered',
        monitor='val_acc',
        mode='max',
        save_top_k=1
    )

    trainer = L.Trainer(max_epochs=100, 
                        log_every_n_steps=1, 
                        gradient_clip_val=1.0,
                        logger=wandb_logger,
                        callbacks=[lr_monitor, checkpoint_callback],
                        deterministic=True,
                        devices=1)

    trainer.fit(model, dm)
    trainer.test(ckpt_path="best")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-cd', '--config_data', type=str, default='configs/dataset/ncaltech.yaml')
    parser.add_argument('-cm', '--config_model', type=str, default='configs/model/cnn.yaml')

    args = parser.parse_args()
    main(args)