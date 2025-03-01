import lightning as L
import argparse
import multiprocessing as mp
import wandb

from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from omegaconf import OmegaConf
from data.ncaltech101.dataset import NCaltech101
from data.ncars.dataset import NCars
# from data.dvs_lip.dataset import DVSLip

from model.recognition import LNRecognition

def main(args):
    L.seed_everything(42, workers=True)

    cfg_dataset = OmegaConf.load(args.config_data)
    cfg_dataset.train.all_noisy = args.all_noisy

    if args.dataset is not None:
        cfg_dataset.path = args.dataset

    if 'cnn' in args.model:
        model_type = 'cnn'
        cfg_dataset.representation.type = 'event_frame'
    elif 'vit' in args.model:
        model_type = 'vit'
        cfg_dataset.representation.type = 'event_voxel'
    elif 'snn' in args.model:
        model_type = 'snn'
        cfg_dataset.representation.type = 'event_spikes'

    cfg_model = cfg_dataset.model[model_type]
    
    print(cfg_dataset)
    print(cfg_model)

    if 'ncaltech' in args.config_data:
        dm = NCaltech101(cfg_dataset, cfg_model)
    
    elif 'ncars' in args.config_data:
        dm = NCars(cfg_dataset, cfg_model)

    dm.setup()

    model = LNRecognition(cfg_dataset, cfg_model)

    lr_monitor = LearningRateMonitor(logging_interval='step')

    folder_name = cfg_dataset.path.split('/')[-1]
    sub_folder_name = cfg_dataset.path.split('/')[-2]

    wandb_logger = WandbLogger(project=f'dvs_filtering', 
                                group=f'{cfg_dataset.name}',
                                name=f'{cfg_dataset.name}_{cfg_dataset.representation.type}_{folder_name}_all_noise={args.all_noisy}')
    wandb_logger.watch(model)

    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename=f'{model_type}_{sub_folder_name}_{folder_name}_all_noise={args.all_noisy}',
        monitor='val_acc',
        mode='max',
        save_top_k=1
    )

    trainer = L.Trainer(max_epochs=200, 
                        log_every_n_steps=1, 
                        gradient_clip_val=1.0,
                        logger=wandb_logger,
                        callbacks=[lr_monitor, checkpoint_callback],
                        deterministic=True,
                        devices=1)

    trainer.fit(model, dm)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-cd', '--config_data', type=str, default='configs/dataset/ncaltech.yaml')
    parser.add_argument('-m', '--model', type=str, default='cnn')
    parser.add_argument('-all', '--all_noisy', type=bool, default=False)
    parser.add_argument('-d', '--dataset', type=str)

    args = parser.parse_args()
    main(args)